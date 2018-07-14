/*
Work in progress: patch by Marco (AUG,19th 2012)
> oni fixed
> pcl added: mostly to include rgb treatment while grabbing from PCD files obtained by pcl_openni_grab_frame -noend 
> sync issue fixed
> volume_size issue fixed
> world.pcd write exception on windows fixed on new trunk version

+ minor changes
*/

/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/

#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <hash_map>

#include <XnLog.h>
#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>

#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include <pcl/gpu/kinfu_large_scale/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "openni_capture.h"
#include "color_handler.h"
#include "evaluation.h"

#include <pcl/common/angles.h>

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#ifdef HAVE_OPENCV  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
typedef pcl::ScopeTime ScopeTimeT;

#include "../src/internal.h"
#include <pcl/gpu/kinfu_large_scale/screenshot_manager.h>

#include <AHCPlaneFitter.hpp> //peac代码: http://www.merl.com/demos/point-plane-slam
#include "zcAhcUtility.h"
//#include <pcl/kdtree/kdtree_flann.h>

#include "testAhc.h"

pcl::console::TicToc tt0, tt1, tt2, tt3; //一些备用计时器

const string pt_picked_str = "MousePickedPoint";
const string pt_dbg_init_str = "InitDbgPoint";
const string plane_slice_str = "plane_slice_str";

//zc: 离散图像序列处理, 避免 oni 跳帧
int fid_;
bool png_source_;
string pngDir_;
vector<string> pngFnames_;
int pngSid_ = -1, 
	pngEid_ = -1; //命令行参数, 设定起始终止帧序号, 调试用 2016-11-28 21:40:30
int pngPauseId_ = -1; //-eval 时, 走到某帧之前, 暂停, 等待人工调试
//bool isPngPause_ = false;

bool isRealPng_, isZchiPng_;

int everyXframes_ = 1; //采样间隔, 默认=1, 即连续采样 @2017-11-21 10:06:36

bool isReadOn_;   //之前一直没用, 现在启用, 做空格暂停控制键 2016-3-26 15:46:37
int png_fps_ = 1000;
bool hasRtCsv_; //是否（在 pngDir_）存在 {R, t} csv 描述文件
bool csv_rt_hint_; //是否用 csv {R, t} 做初值？（不一定用）
bool show_gdepth_; //show-generated-depth. 是否显示当前时刻 (模型, 视角) 对应的深度图
bool debug_level1_; //是否download并显示一些中间调试窗口？运行时效率相关
bool debug_level2_; //if true, imwrite 中间结果到文件, 且 debug_level1_=true (包含关系)
bool imud_; //是否 csv_rt_hint_ 存的是 IMUD-fusion 所需的 Rt, 其特点: 只有 rmat 有用, t=(0,0,0)

//已知立方体三邻边长度, 命令行参数 -cusz //2017-1-2 11:57:19
vector<float> cuSideLenVec_;
bool isUseCube_; //若 cuSideLenVec_.size() >0, 则 true
Eigen::Affine3f camPoseUseCu_; //用立方体定位方式得到的相机姿态
bool isLastFoundCrnr_; //上一帧有没有看到顶角
//bool isCuInitialized_ = false; //若第一次定位到立方体: 1, 设定 cu2gPose_

void dbgAhcPeac_app( const RGBDImage *rgbdObj, PlaneFitter *pf){
	cv::Mat dbgSegMat(rgbdObj->height(), rgbdObj->width(), CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac_app";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac_app

//ahc/peac 窗口鼠标双击选点 //2017-2-20 18:04:55
void peacMouseCallBack(int event, int x, int y, int flags, void *param) {
	//if (event == cv::EVENT_LBUTTONDOWN) {
	//    single_click();
	//}
	cv::Point *pt = (cv::Point*)param;
	if (event == cv::EVENT_LBUTTONDBLCLK) {
		printf("@peacMouseCallBack: event, (x, y), flags:-- %d, (%d, %d), %d\n", event, x, y, flags);

		pt->x = x;
		pt->y = y;
	}
}//peacMouseCallBack

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
{																\
	printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
	return;													\
}

#define CHECK_RC_ERR(rc, what, errors)			\
{												\
	if (rc == XN_STATUS_NO_NODE_PRESENT)		\
{											\
	XnChar strError[1024];					\
	errors.ToString(strError, 1024);		\
	printf("%s\n", strError);				\
}											\
	CHECK_RC(rc, what)							\
}

using namespace std;
using namespace stdext;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace pcl
{
	namespace gpu
	{
		void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
		void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<string> getPcdFilesInDir(const string& directory)
{
	namespace fs = boost::filesystem;
	fs::path dir(directory);

	std::cout << "path: " << directory << std::endl;
	if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
		PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");

	vector<string> result;
	fs::directory_iterator pos(dir);
	fs::directory_iterator end;           

	for(; pos != end ; ++pos)
		if (fs::is_regular_file(pos->status()) )
			if (fs::extension(*pos) == ".pcd")
			{
#if BOOST_FILESYSTEM_VERSION == 3
				result.push_back (pos->path ().string ());
#else
				result.push_back (pos->path ());
#endif
				cout << "added: " << result.back() << endl;
			}

			return result;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
	enum { EACH = 33 };
	SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
	~SampledScopeTime()
	{
		static int i_ = 0;
		time_ms_ += getTime ();    
		if (i_ % EACH == 0 && i_)
		{
			cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << endl;
			time_ms_ = 0;        
		}
		++i_;
	}
private:    
	int& time_ms_;    
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
	setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
	Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
	Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
	Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
	viewer.camera_.pos[0] = pos_vector[0];
	viewer.camera_.pos[1] = pos_vector[1];
	viewer.camera_.pos[2] = pos_vector[2];
	viewer.camera_.focal[0] = look_at_vector[0];
	viewer.camera_.focal[1] = look_at_vector[1];
	viewer.camera_.focal[2] = look_at_vector[2];
	viewer.camera_.view[0] = up_vector[0];
	viewer.camera_.view[1] = up_vector[1];
	viewer.camera_.view[2] = up_vector[2];
	viewer.updateCamera ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
	getViewerPose (visualization::PCLVisualizer& viewer)
{
	Eigen::Affine3f pose = viewer.getViewerPose();
	Eigen::Matrix3f rotation = pose.linear();

	Matrix3f axis_reorder;  
	axis_reorder << 0,  0,  1,
		-1,  0,  0,
		0, -1,  0;

	rotation = rotation * axis_reorder;
	pose.linear() = rotation;
	return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
	writeCloudFile (int format, const CloudT& cloud);

template<typename CloudT> void
	writeCloudFile ( int file_index, int format, const CloudT& cloud );

void writeTransformation( int file_index, const Eigen::Matrix4f& trans );

void writeRawTSDF( int file_index, pcl::TSDFVolume<float, short> & tsdf );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
	writePoligonMeshFile (int format, const pcl::PolygonMesh& mesh, int file_index);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{    
	typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

	pcl::copyPointCloud (points, *merged_ptr);      
	for (size_t i = 0; i < colors.size (); ++i)
		merged_ptr->points[i].rgba = colors.points[i].rgba;

	return merged_ptr;
}

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<PointT>& normals, const PointCloud<RGB>& colors)
{    
	typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

	pcl::copyPointCloud (points, *merged_ptr);      
	for (size_t i = 0; i < colors.size (); ++i) {
		merged_ptr->points[i].normal_x = normals.points[i].x;
		merged_ptr->points[i].normal_y = normals.points[i].y;
		merged_ptr->points[i].normal_z = normals.points[i].z;
		merged_ptr->points[i].rgba = colors.points[i].rgba;
	}

	return merged_ptr;
}

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<PointT>& normals)
{    
	typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

	pcl::copyPointCloud (points, *merged_ptr);      
	for (size_t i = 0; i < normals.size (); ++i) {
		merged_ptr->points[i].normal_x = normals.points[i].x;
		merged_ptr->points[i].normal_y = normals.points[i].y;
		merged_ptr->points[i].normal_z = normals.points[i].z;
	}

	return merged_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{ 
	if (triangles.empty())
		return boost::shared_ptr<pcl::PolygonMesh>();

	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width  = (int)triangles.size();
	cloud.height = 1;
	triangles.download(cloud.points);

	boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
	pcl::toROSMsg(cloud, mesh_ptr->cloud);  

	mesh_ptr->polygons.resize (triangles.size() / 3);
	for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
	{
		pcl::Vertices v;
		v.vertices.push_back(i*3+0);
		v.vertices.push_back(i*3+2);
		v.vertices.push_back(i*3+1);              
		mesh_ptr->polygons[i] = v;
	}    
	return mesh_ptr;
}

boost::shared_ptr<pcl::PolygonMesh> convertToMeshCompact(const DeviceArray<PointXYZ>& triangles)
{ 
  if (triangles.empty () )
  {
    return boost::shared_ptr<pcl::PolygonMesh>();
  }

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size ();
  cloud.height = 1;
  triangles.download (cloud.points);

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr ( new pcl::PolygonMesh () ); 

  std::unordered_map< pcl::PointXYZ, int, PointXYZHasher, PointXYZEqualer > map;
  pcl::PointCloud<pcl::PointXYZ> compactcloud;
  for ( size_t i = 0; i < cloud.size(); i++ ) {
	  if ( map.find( cloud.points[ i ] ) == map.end() ) {
		  map.insert( std::pair< pcl::PointXYZ, int >( cloud.points[ i ], compactcloud.size() ) );
		  compactcloud.push_back( cloud.points[ i ] );
	  }
  }

  PCL_INFO( "[convertTrianglesToMeshCompact] Reduce mesh vertices from %d to %d\n", cloud.size(), compactcloud.size() );
  
  pcl::toROSMsg (compactcloud, mesh_ptr->cloud);
      
  //mesh_ptr->polygons.resize (triangles.size () / 3);
  for (size_t i = 0; i < triangles.size () / 3; ++i)
  {
    pcl::Vertices v;
    //v.vertices.push_back (i*3+0);
    //v.vertices.push_back (i*3+2);
    //v.vertices.push_back (i*3+1);              
	v.vertices.push_back( map.find( cloud.points[ i*3+0 ] )->second );
	v.vertices.push_back( map.find( cloud.points[ i*3+2 ] )->second );
	v.vertices.push_back( map.find( cloud.points[ i*3+1 ] )->second );
    //mesh_ptr->polygons[i] = v;
	if ( v.vertices[ 0 ] != v.vertices[ 1 ] && v.vertices[ 1 ] != v.vertices[ 2 ] && v.vertices[ 2 ] != v.vertices[ 0 ] ) {
		mesh_ptr->polygons.push_back( v );
	}
  }    
  return (mesh_ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct RGBDGraph {
	struct RGBDGraphEdge {
		int i_;
		int frame_i_;
		int j_;
		int frame_j_;
		RGBDGraphEdge( int i, int fi, int j, int fj ) : i_(i), j_(j), frame_i_(fi), frame_j_(fj) {}
		RGBDGraphEdge() {}
	};
	vector< RGBDGraphEdge > edges_;
	int index_;
	Eigen::Matrix4f head_inv_;
	Eigen::Matrix4f head_mat_;
	int head_frame_;
	Eigen::Matrix4f tail_inv_;
	Eigen::Matrix4f tail_mat_;
	int tail_frame_;

	void loadFromFile( string filename ) {
		index_ = 0;
		edges_.clear();
		int id1, frame1, id2, frame2;
		FILE * f = fopen( filename.c_str(), "r" );
		if ( f != NULL ) {
			char buffer[1024];
			while ( fgets( buffer, 1024, f ) != NULL ) {
				if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
					sscanf( buffer, "%d:%d %d:%d", &id1, &frame1, &id2, &frame2 );
					edges_.push_back( RGBDGraphEdge( id1, frame1, id2, frame2 ) );
				}
			}
			fclose ( f );
		}
	}

	void saveToFile( string filename ) {
		std::ofstream file( filename.c_str() );
		if ( file.is_open() ) {
			for ( unsigned int i = 0; i < edges_.size(); i++ ) {
				RGBDGraphEdge & edge = edges_[ i ];
				file << edge.i_ << ":" << edge.frame_i_ << " " << edge.j_ << ":" << edge.frame_j_ << endl;
			}
			file.close();
		}
	}

	bool ended() {
		return ( index_ >= ( int )edges_.size() );
	}

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct BBoxFrame {
	int id_;
	int bounds_[ 6 ];
};

struct BBox {
	vector< BBoxFrame > frames_;
	void loadFromFile( string filename ) {
		frames_.clear();
		FILE * f = fopen( filename.c_str(), "r" );
		if ( f != NULL ) {
			char buffer[1024];
			int id;
			BBoxFrame frame;
			while ( fgets( buffer, 1024, f ) != NULL ) {
				if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
					sscanf( buffer, "%d %d %d %d %d %d %d", 
						&frame.id_, 
						&frame.bounds_[ 0 ], 
						&frame.bounds_[ 1 ], 
						&frame.bounds_[ 2 ], 
						&frame.bounds_[ 3 ], 
						&frame.bounds_[ 4 ], 
						&frame.bounds_[ 5 ] 
					);
					frames_.push_back( frame );
				}
			}
			fclose ( f );
		}
	}
};

struct RGBDTrajectory {
	vector< pcl::gpu::FramedTransformation > data_;
	vector< Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > cov_;
	int index_;
	Eigen::Matrix4f head_inv_;
	void loadFromFile( string filename ) {
		data_.clear();
		index_ = 0;
		int id1, id2, frame;
		Matrix4f trans;
		FILE * f = fopen( filename.c_str(), "r" );
		if ( f != NULL ) {
			char buffer[1024];
			while ( fgets( buffer, 1024, f ) != NULL ) {
				if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
					sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%f %f %f %f", &trans(0,0), &trans(0,1), &trans(0,2), &trans(0,3) );
					fgets( buffer, 1024, f );
					sscanf( buffer, "%f %f %f %f", &trans(1,0), &trans(1,1), &trans(1,2), &trans(1,3) );
					fgets( buffer, 1024, f );
					sscanf( buffer, "%f %f %f %f", &trans(2,0), &trans(2,1), &trans(2,2), &trans(2,3) );
					fgets( buffer, 1024, f );
					sscanf( buffer, "%f %f %f %f", &trans(3,0), &trans(3,1), &trans(3,2), &trans(3,3) );
					data_.push_back( FramedTransformation( id1, id2, frame, trans ) );
				}
			}
			fclose ( f );
			head_inv_ = data_[ 0 ].transformation_.inverse();
		}
	}

	void saveToFile( string filename ) {
		std::ofstream file( filename.c_str() );
		if ( file.is_open() ) {
			for ( unsigned int i = 0; i < data_.size(); i++ ) {
				file << data_[ i ].id1_ << "\t" << data_[ i ].id2_ << "\t" << data_[ i ].frame_ << std::endl;
				file << data_[ i ].transformation_ << std::endl;
			}
			file.close();
		}
	}

	void saveCovToFile( string filename ) {
		std::ofstream file( filename.c_str() );
		if ( file.is_open() ) {
			for ( unsigned int i = 0; i < data_.size(); i++ ) {
				file << data_[ i ].id1_ << "\t" << data_[ i ].id2_ << "\t" << data_[ i ].frame_ << std::endl;
				file << cov_[ i ] << std::endl;
			}
			file.close();
		}
	}

	bool ended() {
		return ( index_ >= ( int )data_.size() );
	}

	void clear() {
		data_.clear();
		cov_.clear();
	}
};

struct CameraParam {
public:
	bool isFileLoaded; //zc
	double fx_, fy_, cx_, cy_, ICP_trunc_, integration_trunc_;

	CameraParam() : fx_( 525.0 ), fy_( 525.0 ), cx_( 319.5 ), cy_( 239.5 ), ICP_trunc_( 2.5 ), integration_trunc_( 2.5 )  {
		isFileLoaded = false;
	}

	void loadFromFile( std::string filename ) {
		FILE * f = fopen( filename.c_str(), "r" );
		if ( f != NULL ) {
			isFileLoaded = true;

			char buffer[1024];
			while ( fgets( buffer, 1024, f ) != NULL ) {
				if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
					sscanf( buffer, "%lf", &fx_);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%lf", &fy_);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%lf", &cx_);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%lf", &cy_);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%lf", &ICP_trunc_);
					fgets( buffer, 1024, f );
					sscanf( buffer, "%lf", &integration_trunc_);
				}
			}
			fclose ( f );
			PCL_WARN( "Camera model set to (fx, fy, cx, cy, icp_trunc, int_trunc):\n\t%.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", fx_, fy_, cx_, cy_, ICP_trunc_, integration_trunc_ );
		}
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
	CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
	{
		cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

		cloud_viewer_.setBackgroundColor (0, 0, 0.15);
		cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
		cloud_viewer_.addCoordinateSystem (1.0);
		cloud_viewer_.initCameraParameters ();
		cloud_viewer_.setPosition (0, 500);
		cloud_viewer_.setSize (640, 480);
		cloud_viewer_.camera_.clip[0] = 0.01;
		cloud_viewer_.camera_.clip[1] = 10.01;
	}

	void
		show (const KinfuTracker& kinfu)
	{
		kinfu.getLastFrameCloud (cloud_device_);

		int c;
		cloud_device_.download (cloud_ptr_->points, c);
		cloud_ptr_->width = cloud_device_.cols ();
		cloud_ptr_->height = cloud_device_.rows ();
		cloud_ptr_->is_dense = false;

		cloud_viewer_.removeAllPointClouds ();
		cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
		cloud_viewer_.spinOnce ();
	}

	void
		setViewerPose (const Eigen::Affine3f& viewer_pose) {
			::setViewerPose (cloud_viewer_, viewer_pose);
	}

	PointCloud<PointXYZ>::Ptr cloud_ptr_;
	DeviceArray2D<PointXYZ> cloud_device_;
	visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
	ImageView() : paint_image_ (false), accumulate_views_ (false)
	{
		viewerScene_.setWindowTitle ("View3D from ray tracing");
		viewerScene_.setPosition (0, 0);
		viewerDepth_.setWindowTitle ("Kinect Depth stream");
		viewerDepth_.setPosition (640, 0);
		//viewerColor_.setWindowTitle ("Kinect RGB stream");
		viewerTraj_.setWindowTitle("Trajectory");
		viewerTraj_.setPosition(640, 500);

		viewerVMap_.setWindowTitle( "VMap" );
		viewerVMap_.setPosition( 1280, 0 );
		viewerNMap_.setWindowTitle( "NMap" );
		viewerNMap_.setPosition( 1280, 500 );
	}

	void
		showScene (KinfuTracker& kinfu, const PtrStepSz<const pcl::gpu::PixelRGB>& rgb24, bool registration, int frame_id, Eigen::Affine3f* pose_ptr = 0, string basedir = "image/")
	{
		if (pose_ptr)
		{
			raycaster_ptr_->run ( kinfu.volume (), *pose_ptr, kinfu.getCyclicalBufferStructure () ); //says in cmake it does not know it
			raycaster_ptr_->generateSceneView(view_device_);
		}
		else
		{
			kinfu.getImage (view_device_);
		}

		if (paint_image_ && registration && !pose_ptr)
		{
			colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
			paint3DView (colors_device_, view_device_);
		}

		int cols;
		view_device_.download (view_host_, cols);
		viewerScene_.showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());    

		if ( frame_id != -1 ) {
			char filename[ 1024 ];
			sprintf( filename, "%skinfu/%06d.png", basedir.c_str(), frame_id );

			cv::Mat m( 480, 640, CV_8UC3, (void*)&view_host_[0] );
			cv::imwrite( filename, m );
		}

		// new add for vmap and nmap
		//raycaster_ptr_->run(kinfu.volume(), kinfu.getCameraPose(), kinfu.getCyclicalBufferStructure ());
		Eigen::Affine3f cam_pose_curr_fake = kinfu.getCameraPose();
		cam_pose_curr_fake.translation() -= kinfu.volume000_gcoo_;
		raycaster_ptr_->run(kinfu.volume(), cam_pose_curr_fake, kinfu.getCyclicalBufferStructure ());

		raycaster_ptr_->generateDepthImage(generated_depth_);
		int c;
		vector<unsigned short> data;
		generated_depth_.download(data, c);
		viewerVMap_.showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);

		if ( frame_id != -1 ) {
			char filename[ 1024 ];
			sprintf( filename, "%svmap/%06d.png", basedir.c_str(), frame_id );

			//cv::Mat m( 480, 640, CV_8UC3, (void*)&view_host_[0] );
			//cv::imwrite( filename, m );
			cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
			//cv::imwrite( filename, m );
		}

		raycaster_ptr_->generateNormalImage(view_device_);
		view_device_.download (view_host_, cols);
		viewerNMap_.showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows (), "short_image");    

		if ( frame_id != -1 ) {
			char filename[ 1024 ];
			sprintf( filename, "%snmap/%06d.png", basedir.c_str(), frame_id );

			cv::Mat m( 480, 640, CV_8UC3, (void*)&view_host_[0] );
			cv::Mat mm;
			cv::cvtColor( m, mm, CV_RGB2BGR );
			//cv::imwrite( filename, mm );
		}

		//viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);
#ifdef HAVE_OPENCV
		if (accumulate_views_)
		{
			views_.push_back (cv::Mat ());
			cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
			//cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
		}
#endif
	}

	void
		showDepth (const PtrStepSz<const unsigned short>& depth) 
	{ 
		viewerDepth_.showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true); 
	}

	void
		showTraj( const cv::Mat & traj, int frame_id, string basedir ) {
			viewerTraj_.showRGBImage ( (unsigned char *) traj.data, traj.cols, traj.rows);//, "short_image" );
			if ( frame_id != -1 && basedir.compare( "image/" ) == 0 ) {
				char filename[ 1024 ];
				sprintf( filename, "%straj/%06d.png", basedir.c_str(), frame_id );
				cv::imwrite( filename, traj );
			}
	}

	void
		showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose)
	{            
		raycaster_ptr_->run(kinfu.volume(), pose, kinfu.getCyclicalBufferStructure ());
		raycaster_ptr_->generateDepthImage(generated_depth_);    

		int c;
		vector<unsigned short> data;
		generated_depth_.download(data, c);

		viewerDepth_.showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
	}

	void
		toggleImagePaint()
	{
		paint_image_ = !paint_image_;
		cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
	}

	bool paint_image_;
	bool accumulate_views_;

	visualization::ImageViewer viewerScene_;
	visualization::ImageViewer viewerDepth_;
	visualization::ImageViewer viewerTraj_;
	//visualization::ImageViewer viewerColor_;

	visualization::ImageViewer viewerVMap_;
	visualization::ImageViewer viewerNMap_;

	KinfuTracker::View view_device_;
	KinfuTracker::View colors_device_;
	vector<pcl::gpu::PixelRGB> view_host_;

	RayCaster::Ptr raycaster_ptr_;

	KinfuTracker::DepthMap generated_depth_;

#ifdef HAVE_OPENCV
	vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
	enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

	SceneCloudView() : extraction_mode_ (GPU_Connected6), compute_normals_ (true), valid_combined_ (false), cube_added_(false), cloud_viewer_ ("Scene Cloud Viewer")
	{
		cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
		//normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
		normals_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
		combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
		point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

		cloud_viewer_.setBackgroundColor (0, 0, 0);
		cloud_viewer_.addCoordinateSystem (1.0);
		cloud_viewer_.initCameraParameters ();
		cloud_viewer_.setPosition (0, 500);
		cloud_viewer_.setSize (640, 480);
		cloud_viewer_.camera_.clip[0] = 0.01;
		cloud_viewer_.camera_.clip[1] = 10.01;

		cloud_viewer_.addText ("H: print help", 2, 15, 20, 34, 135, 246);         
	}

	void
		show (KinfuTracker& kinfu, bool integrate_colors)
	{
		viewer_pose_ = kinfu.getCameraPose();

		ScopeTimeT time ("PointCloud Extraction");
		cout << "\nGetting cloud... " << flush;

		valid_combined_ = false;

		if (extraction_mode_ != GPU_Connected6)     // So use CPU
		{
			kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
		}
		else
		{
			DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_, kinfu.getCyclicalBufferStructure() );             

			if (integrate_colors)
			{
				kinfu.colorVolume().fetchColors(extracted, point_colors_device_, kinfu.getCyclicalBufferStructure());
				point_colors_device_.download(point_colors_ptr_->points);
				point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
				point_colors_ptr_->height = 1;
			}
			else
				point_colors_ptr_->points.clear();

			// do the in-space normal extraction
			extracted.download (cloud_ptr_->points);
			cloud_ptr_->width = (int)cloud_ptr_->points.size ();
			cloud_ptr_->height = 1;
			if ( compute_normals_ ) {
				kinfu.volume().fetchNormalsInSpace( extracted, kinfu.getCyclicalBufferStructure() );
				extracted.download( normals_ptr_->points );
			}

			/*
			if (compute_normals_)
			{
			kinfu.volume().fetchNormals (extracted, normals_device_);
			pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
			combined_device_.download (combined_ptr_->points);
			combined_ptr_->width = (int)combined_ptr_->points.size ();
			combined_ptr_->height = 1;

			valid_combined_ = true;
			}
			else
			{
			extracted.download (cloud_ptr_->points);
			cloud_ptr_->width = (int)cloud_ptr_->points.size ();
			cloud_ptr_->height = 1;
			}

			if (integrate_colors)
			{
			kinfu.colorVolume().fetchColors(extracted, point_colors_device_, kinfu.getCyclicalBufferStructure());
			point_colors_device_.download(point_colors_ptr_->points);
			point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
			point_colors_ptr_->height = 1;
			}
			else
			point_colors_ptr_->points.clear();
			*/
		}
		size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
		cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

		cloud_viewer_.removeAllPointClouds ();    
		visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
		cloud_viewer_.addPointCloud<PointXYZ> (cloud_ptr_, rgb);

		///*
		cloud_viewer_.removeAllPointClouds ();    
		if (valid_combined_)
		{
		visualization::PointCloudColorHandlerRGBHack<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
		cloud_viewer_.addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
		cloud_viewer_.addPointCloudNormals<PointNormal>(combined_ptr_, 50);
		}
		else
		{
		visualization::PointCloudColorHandlerRGBHack<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
		cloud_viewer_.addPointCloud<PointXYZ> (cloud_ptr_, rgb);
		}    
		//*/
	}

	void
		toggleCube(const Eigen::Vector3f& size)
	{
		if (cube_added_)
			cloud_viewer_.removeShape("cube");
		else
			cloud_viewer_.addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

		cube_added_ = !cube_added_;
	}

	void
		toggleExctractionMode ()
	{
		extraction_mode_ = (extraction_mode_ + 1) % 3;

		switch (extraction_mode_)
		{
		case 0: cout << "Cloud extraction mode: GPU, Connected-6" << endl; break;
		case 1: cout << "Cloud extraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
		case 2: cout << "Cloud extraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
		}
		;
	}

	void
		toggleNormals ()
	{
		compute_normals_ = !compute_normals_;
		cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
	}

	void
		clearClouds (bool print_message = false)
	{
		cloud_viewer_.removeAllPointClouds ();
		cloud_ptr_->points.clear ();
		normals_ptr_->points.clear ();    
		if (print_message)
			cout << "Clouds/Meshes were cleared" << endl;
	}

	void
		showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
	{
		ScopeTimeT time ("Mesh Extraction");
		cout << "\nGetting mesh... " << flush;

		if (!marching_cubes_)
			marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

		DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);    
		mesh_ptr_ = convertToMeshCompact(triangles_device);

		cloud_viewer_.removeAllPointClouds ();
		if (mesh_ptr_)
			cloud_viewer_.addPolygonMesh(*mesh_ptr_);	

		cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
	}

	int extraction_mode_;
	bool compute_normals_;
	bool valid_combined_;
	bool cube_added_;

	Eigen::Affine3f viewer_pose_;

	visualization::PCLVisualizer cloud_viewer_;

	PointCloud<PointXYZ>::Ptr cloud_ptr_;
	//PointCloud<Normal>::Ptr normals_ptr_;
	PointCloud<PointXYZ>::Ptr normals_ptr_;

	DeviceArray<PointXYZ> cloud_buffer_device_;
	DeviceArray<Normal> normals_device_;

	PointCloud<PointNormal>::Ptr combined_ptr_;
	DeviceArray<PointNormal> combined_device_;  

	DeviceArray<RGB> point_colors_device_; 
	PointCloud<RGB>::Ptr point_colors_ptr_;

	MarchingCubes::Ptr marching_cubes_;
	DeviceArray<PointXYZ> triangles_buffer_device_;

	boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuLSApp
{
	enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };

	KinFuLSApp(pcl::Grabber& source, float vsz, float shiftDistance, int snapshotRate, bool useDevice, int fragmentRate, int fragmentStart, float trunc_dist, float sx, float sy, float sz) 
		: exit_ (false), scan_ (false), scan_mesh_(false), file_index_( 0 ), transformation_( Eigen::Matrix4f::Identity() ), scan_volume_ (false), independent_camera_ (false),
		slice2d_(false), print_nbr_(false), //zc
		registration_ (false), integrate_colors_ (false), pcd_source_ (false), focal_length_(-1.f), capture_ (source), time_ms_(0), record_script_ (false), play_script_ (false), recording_ (false), use_device_ (useDevice), traj_(cv::Mat::zeros( 480, 640, CV_8UC3 )), traj_buffer_( 480, 640, CV_8UC3, cv::Scalar( 255, 255, 255 )),
		use_rgbdslam_ (false), record_log_ (false), fragment_rate_ (fragmentRate), fragment_start_ (fragmentStart), use_schedule_ (false), use_graph_registration_ (false), frame_id_ (0), use_bbox_ ( false ), seek_start_( -1 ), kinfu_image_ (false), traj_token_ (0), use_mask_ (false), use_omask_(false), use_tmask_(false), 
		kintinuous_( false ), rgbd_odometry_( false ), slac_( false ), bdr_odometry_( false )
		,cu_odometry_(false), s2s_odometry_(false)
		, kdtree_odometry_(false)
	{    
		//Init Kinfu Tracker
		Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);    

		PCL_WARN ("--- CURRENT SETTINGS ---\n");
		PCL_INFO ("Volume size is set to %.2f meters\n", vsz);
		PCL_INFO ("Volume will shift when the camera target point is farther than %.2f meters from the volume center\n", shiftDistance);
		PCL_INFO ("The target point is located at [0, 0, %.2f] in camera coordinates\n", 0.6*vsz);
		PCL_WARN ("------------------------\n");

		// warning message if shifting distance is abnormally big compared to volume size
		if(shiftDistance > 2.5 * vsz)
			PCL_WARN ("WARNING Shifting distance (%.2f) is very large compared to the volume size (%.2f).\nYou can modify it using --shifting_distance.\n", shiftDistance, vsz);

		//volume_size (2) *= 2;
		kinfu_ = new pcl::gpu::KinfuTracker(volume_size, shiftDistance);

		Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
		Eigen::Vector3f t;
		if ( vsz < 2.0f ) {
			t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 + 0.3);
		} else {
			t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 + 0.3);
		}
		t( 0 ) += sx;
		t( 1 ) += sy;
		t( 2 ) += sz;

		Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);
		//cout<<"KinFuLSApp-ctor\n"
		//	<<"R, t\n"<<R<<endl<<t<<endl
		//	<<"pose:\n"<<pose.matrix()<<endl
		//	<<"pose.linear:\n"<<pose.linear()<<endl
		//	<<"pose.rotation:\n"<<pose.rotation()<<endl
		//	;
		//zc: TMP-TESTS

		transformation_inverse_ = pose.matrix().inverse();

		kinfu_->setInitialCameraPose (pose);
		kinfu_->volume().setTsdfTruncDist (0.030f / 3.0f * volume_size(0)/*meters*/);
		//zc: 定死 3cm    @2017-3-17 00:25:42
		//kinfu_->volume().setTsdfTruncDist (0.030f /*meters*/);
		kinfu_->setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
		kinfu_->setDepthTruncationForICP(2.5f/*meters*/);
		//kinfu_->setDepthTruncationForIntegrate(2.5f/*meters*/);
		kinfu_->setDepthTruncationForIntegrate( trunc_dist );
		kinfu_->setCameraMovementThreshold(0.001f);

		//Init KinFuLSApp            
		tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
		image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols ()) );

		scene_cloud_view_.cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
		image_view_.viewerScene_.registerKeyboardCallback (keyboard_callback, (void*)this);
		image_view_.viewerDepth_.registerKeyboardCallback (keyboard_callback, (void*)this);
		image_view_.viewerTraj_.registerKeyboardCallback (keyboard_callback, (void*)this);
		image_view_.viewerVMap_.registerKeyboardCallback (keyboard_callback, (void*)this);
		image_view_.viewerNMap_.registerKeyboardCallback (keyboard_callback, (void*)this);

		scene_cloud_view_.toggleCube(volume_size);

		//zc: https://stackoverflow.com/questions/27101646/how-to-pick-two-points-from-viewer-in-pcl
		//scene_cloud_view_.cloud_viewer_->registerPointPickingCallback(pointPickingCallback, (void*)scene_cloud_view_.cloud_viewer_.get()); //当 pointPickingCallback 是 static member 时
		scene_cloud_view_.cloud_viewer_.registerPointPickingCallback(&KinFuLSApp::pointPickingCallback, *this, (void*)&scene_cloud_view_.cloud_viewer_); //当 pointPickingCallback 是 static member 时


		frame_counter_ = 0;
		enable_texture_extraction_ = false;

		//~ float fx, fy, cx, cy;
		//~ boost::shared_ptr<openni_wrapper::OpenNIDevice> d = ((pcl::OpenNIGrabber)source).getDevice ();
		//~ kinfu_->getDepthIntrinsics (fx, fy, cx, cy);

		float height = 480.0f;
		float width = 640.0f;
		screenshot_manager_.setCameraIntrinsics (pcl::device::FOCAL_LENGTH, height, width);
		snapshot_rate_ = snapshotRate;
	}

	~KinFuLSApp()
	{
		if (evaluation_ptr_)
			evaluation_ptr_->saveAllPoses(*kinfu_);
	}

	//zc: 给 SceneCloudView-cloud_viewer_ 增加选点回调函数; 放在这里是因为 KinFuApp 可获取到 volume_size //2017-2-13 06:12:09
	//static void //一种可行
	void //非 static 也有能成功回调的接口
	pointPickingCallback(const pcl::visualization::PointPickingEvent& event, void* cookie){
			if (event.getPointIndex () == -1) 
				return;

			PointXYZ pt_picked;
			event.getPoint(pt_picked.x, pt_picked.y, pt_picked.z);
			cout << pt_picked << endl;
			Vector3f cell_size = this->kinfu_->volume().getVoxelSize();

			pcl::visualization::PCLVisualizer *v_visualizer = (pcl::visualization::PCLVisualizer*)(cookie);

			v_visualizer->removeShape(pt_picked_str);

			v_visualizer->addSphere(pt_picked, cell_size.x()/2, 1,0,1, pt_picked_str);

			vector<int> &vxlDbg = this->kinfu_->vxlDbg_;

			vxlDbg[0] = int(pt_picked.x / cell_size.x());
			vxlDbg[1] = int(pt_picked.y / cell_size.y());
			vxlDbg[2] = int(pt_picked.z / cell_size.z());
			cout << "this->kinfu_->vxlDbg_[0/1/2]: "
				<< vxlDbg[0] << ","
				<< vxlDbg[1] << ","
				<< vxlDbg[2] << "," << endl;

			//调试面重绘: @2018-2-1 15:48:12
			v_visualizer->removeShape(plane_slice_str);

			const int fix_axis = vxlDbg[3];
			pcl::ModelCoefficients plane_coeff;
			plane_coeff.values.resize(4, 0); //默认填零
			plane_coeff.values[fix_axis] = 1; //垂直于某坐标轴, 则填1
			plane_coeff.values[3] = -pt_picked.data[fix_axis];

			v_visualizer->addPlane(plane_coeff, plane_slice_str);

			printf("pt_picked.xyz: (%f, %f, %f), plane_coeff: (%f, %f, %f, %f)\n", pt_picked.x, pt_picked.y, pt_picked.z, 
				plane_coeff.values[0], plane_coeff.values[1], plane_coeff.values[2], plane_coeff.values[3]);

	}//pointPickingCallback

	void
		initCurrentFrameView ()
	{
		current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
		current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
		current_frame_cloud_view_->setViewerPose (kinfu_->getCameraPose ());
	}

	void
		initRegistration ()
	{        
		registration_ = capture_.providesCallback<pcl::ONIGrabber::sig_cb_openni_image_depth_image> ();
		cout << "Registration mode: " << (registration_ ? "On" : "Off (not supported by source)") << endl;
	}

	void
		toggleKinfuImage()
	{
		kinfu_image_ = true;
		cout << "Output images to image folder." << endl;
	}

	void
		toggleRGBDOdometry()
	{
		rgbd_odometry_ = true;
		cout << "Using RGBDOdometry." << endl;
	}

	void
		toggleKintinuous()
	{
		kintinuous_ = true;
		cout << "Using Kintinuous." << endl;
	}

	void toggleBdrOdometry()
	{
		bdr_odometry_ = true;
	}

	//zc
	void toggleCuOdometry(){ cu_odometry_ = true; }
	void toggleS2sOdometry(){ s2s_odometry_ = true; }

	void toggleKdtreeOdometry()
	{
		kdtree_odometry_ = true;
	}

	void toggleBdrAmplifier( float amp )
	{
		kinfu_->amplifier_ = amp;
		cout << "Amplifier is : " << amp << endl;
	}

	void 
		toggleColorIntegration()
	{
		if(registration_)
		{
			const int max_color_integration_weight = 2;
			kinfu_->initColorIntegration(max_color_integration_weight);
			integrate_colors_ = true;      
		}    
		cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
	}

	void toggleRecording()
	{
		if ( use_device_ && registration_ ) {
			recording_ = true;
		}
		cout << "Recording ONI: " << (recording_ ? "On" : "Off ( requires registration mode )") << endl;
	}

	void
		toggleScriptRecord()
	{
		record_script_ = true;
		cout << "Script record: " << ( record_script_ ? "On" : "Off ( requires triggerd mode )" ) << endl;
	}

	void
		toggleLogRecord( std::string record_log_file )
	{
		record_log_ = true;
		record_log_file_ = record_log_file;
		cout << "Log record: " << ( record_log_ ? "On" : "Off" ) << endl;
	}

	void
		toggleCameraParam( std::string camera_file )
	{
		camera_.loadFromFile( camera_file );

		//kinfu_->setDepthIntrinsics( 582.62448167737955f, 582.69103270988637f, 313.04475870804731f, 238.44389626620386f );
		kinfu_->setDepthIntrinsics( camera_.fx_, camera_.fy_, camera_.cx_, camera_.cy_ );
		kinfu_->setDepthTruncationForICP( camera_.ICP_trunc_ );
		kinfu_->setDepthTruncationForIntegrate( camera_.integration_trunc_ );
	}

	void
		toggleScriptPlay( string script_file )
	{
		FILE * f = fopen( script_file.c_str(), "r" );
		if ( f != NULL ) {
			char buffer[1024];
			while ( fgets( buffer, 1024, f ) != NULL ) {
				if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
					script_frames_.push( ScriptAction( buffer[ 0 ], atoi( buffer + 2 ) ) );
				}
			}
			play_script_ = true;
			cout << "Script contains " << script_frames_.size() << " shifting actions." << endl;
			fclose ( f );
		}
		cout << "Script play: " << ( play_script_ ? "On" : "Off ( requires triggerd mode )" ) << endl;
	}

	void
		toggleRGBDSlam( string log_file )
	{
		use_rgbdslam_ = !use_device_;
		rgbd_traj_.loadFromFile( log_file );

		/*
		// draw on traj buffer
		Eigen::Vector4f sensor_origin( 0, 0, 0, 1 );
		Eigen::Vector4f last_sensor_origin;
		Eigen::Matrix4f init_inverse = rgbd_traj_.data_[ 0 ].transformation_.inverse();
		for (unsigned int i = 1; i < rgbd_traj_.data_.size(); ++i) {
		Eigen::Matrix4f world2base = rgbd_traj_.data_[ i ].transformation_;
		Eigen::Affine3f world2base_aff( init_inverse * world2base );

		last_sensor_origin = sensor_origin;
		sensor_origin = Eigen::Vector4f(world2base_aff.translation()(0), world2base_aff.translation()(1), world2base_aff.translation()(2), 1.0f);

		cv::Point2f p,q; //TODO: Use sub-pixel-accuracy
		p.x = 320.0 + sensor_origin(0) * 100.0;
		p.y = 240.0 - sensor_origin(2) * 100.0;
		q.x = 320.0 + last_sensor_origin(0) * 100.0;
		q.y = 240.0 - last_sensor_origin(2) * 100.0;
		cv::line(traj_buffer_, p, q, cv::Scalar( 255, 0, 0 ), 1, 16);
		}
		*/

		cout << "RGBD slam contains " << rgbd_traj_.data_.size() << " key frames." << endl;
		cout << "Use rgbd slam: " << ( use_rgbdslam_ ? "On" : "Off ( requires triggerd mode )" ) << endl;
	}

	void
		toggleBBox( string bbox_file )
	{
		use_bbox_ = use_rgbdslam_;
		bbox_.loadFromFile( bbox_file );
		cout << "BBox contains " << bbox_.frames_.size() << " key frames." << endl;
		cout << "Use rgbd bbox schedule: " << ( use_bbox_ ? "On" : "Off ( requires use rgbdslam mode )" ) << endl;
	}

	void
		toggleSLAC( int num )
	{
		slac_ = true;
		kinfu_->initOnlineSLAC( num );
		cout << "SLAC On." << endl;
		//kinfu_->initSLAC( num );
		//cout << "SLAC is on (0x40 flag is enabled in schedule)." << endl;
	}

	void
		toggleMask( string mask )
	{
		use_mask_ = true;
		sscanf( mask.c_str(), "%d,%d,%d,%d", &mask_[ 0 ], &mask_[ 1 ], &mask_[ 2 ], &mask_[ 3 ] );
		PCL_INFO( "Use mask: [%d,%d] - [%d,%d]\n", mask_[ 0 ], mask_[ 2 ], mask_[ 1 ], mask_[ 3 ] );
	}

	void
		toggleRGBDGraphRegistration( string graph_file )
	{
		rgbd_graph_.loadFromFile( graph_file );
		use_graph_registration_ = ( use_rgbdslam_ && fragment_rate_ > 0 );
		cout << "Use rgbd graph registration: " << ( use_graph_registration_ ? "On" : "Off ( requires use rgbdslam mode )" ) << endl;
	}

	void
		toggleSchedule( string schedule_file )
	{
		schedule_traj_.loadFromFile( schedule_file );
		use_schedule_ = true;
		//use_schedule_ = use_graph_registration_;

		if ( use_graph_registration_ ) {
			next_pointers_.resize( rgbd_traj_.data_.size() );
			for ( int i = 0; i < ( int )rgbd_graph_.edges_.size(); i++ ) {
				RGBDGraph::RGBDGraphEdge & edge = rgbd_graph_.edges_[ i ];
				next_pointers_[ edge.frame_i_ - 1 ].push_back( edge.frame_j_ - 1 );
			}
		}

		cout << "Use schedule: " << ( use_schedule_ ? "On" : "Off ( requires use graph registration mode )" ) << endl;
	}

	void
		writeScriptFile()
	{
		time_t rawtime;
		struct tm *timeinfo;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		char strFileName[ 1024 ];
		sprintf(strFileName, "%04d%02d%02d-%02d%02d%02d.script",
			timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

		printf("Creating script file %s\n", strFileName);
		FILE * f = fopen( strFileName, "w" );

		if ( f != NULL ) {
			while ( script_frames_.empty() == false ) {
				fprintf( f, "%c %d\n", script_frames_.front().action_, script_frames_.front().frame_ );
				script_frames_.pop();
			}
		}

		fclose( f );
	}

	void
		writeLogFile()
	{
		if ( record_log_file_.length() == 0 ) {
			time_t rawtime;
			struct tm *timeinfo;
			time(&rawtime);
			timeinfo = localtime(&rawtime);
			char strFileName[ 1024 ];
			sprintf(strFileName, "%04d%02d%02d-%02d%02d%02d.log",
				timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

			printf("Creating log file %s\n", strFileName);

			kinfu_traj_.saveToFile( string( strFileName ) );

			if ( kinfu_traj_.data_.size() == kinfu_traj_.cov_.size() ) {
				sprintf(strFileName, "%04d%02d%02d-%02d%02d%02d.cov",
					timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

				printf("Creating cov file %s\n", strFileName);

				kinfu_traj_.saveCovToFile( string( strFileName ) );
			}
		} else {
			kinfu_traj_.saveToFile( record_log_file_ );
		}
	}

	void
		toggleIndependentCamera()
	{
		independent_camera_ = !independent_camera_;
		cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
	}

	void
		toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
	{
		eval_folder_ = eval_folder;
		cout<<"eval_folder_: "<<eval_folder_<<endl;
		evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
		if (!match_file.empty())
			evaluation_ptr_->setMatchFile(match_file);

		if(!camera_.isFileLoaded){ //改成仅在没用到 --camera 时才用 eval 写死的内参
			kinfu_->setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
			image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols (),
				evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
		}
	}

	void drawTrajectory()
	{
		double resolution = 100;

		traj_ = traj_buffer_.clone();
		Eigen::Vector4d sensor_origin( 0, 0, 0, 1 );
		Eigen::Vector4d last_sensor_origin;
		Eigen::Affine3f init_pose = kinfu_->getCameraPose(0);
		Eigen::Matrix4f init_inverse = init_pose.matrix().inverse();
		Eigen::Vector4d init_origin = Eigen::Vector4d(init_pose.translation()(0), init_pose.translation()(1), init_pose.translation()(2), 0.0);
		for (unsigned int i = 1; i < kinfu_->getNumberOfPoses(); ++i) {
			Eigen::Affine3f pose = kinfu_->getCameraPose(i);
			//		cout << i << " debug" << endl;
			//cout << pose.matrix() << endl;

			Eigen::Affine3f pose_1( init_inverse * pose.matrix() );

			//if ( use_graph_registration_ ) {
			//	  pose_1.matrix() = rgbd_graph_.head_mat_ * pose_1.matrix();
			//}

			last_sensor_origin = sensor_origin;
			//sensor_origin = Eigen::Vector4d(pose.translation()(0), pose.translation()(1), pose.translation()(2), 1.0) - init_origin;
			sensor_origin = Eigen::Vector4d(pose_1.translation()(0), pose_1.translation()(1), pose_1.translation()(2), 1.0);

			if ( i <= traj_token_ )
				continue;

			cv::Point2f p,q; //TODO: Use sub-pixel-accuracy
			p.x = 320.0 + sensor_origin(0) * resolution;
			p.y = 440.0 - sensor_origin(2) * resolution;
			q.x = 320.0 + last_sensor_origin(0) * resolution;
			q.y = 440.0 - last_sensor_origin(2) * resolution;
			cv::line(traj_, p, q, cv::Scalar( 0, 0, 0 ), 1, 16);

			if ( i == kinfu_->getNumberOfPoses() - 1 ) {
				//Eigen::Vector4d ref = pose.matrix().cast<double>() * Eigen::Vector4d( 0.0, 0.0, 1.5, 1.0 ) - init_origin;
				Eigen::Vector4d ref = pose_1.matrix().cast<double>() * Eigen::Vector4d( 0.0, 0.0, 0.5, 1.0 );
				q.x = 320.0 + ref(0) * resolution;
				q.y = 440.0 - ref(2) * resolution;
				cv::line(traj_, p, q, cv::Scalar( 0, 0, 255 ), 1, 16);

				Eigen::Vector4d cams[ 4 ] = { 
					pose_1.matrix().cast<double>() * Eigen::Vector4d( -0.05, 0.05, 0, 1.0 ),
					pose_1.matrix().cast<double>() * Eigen::Vector4d( 0.05, 0.05, 0, 1.0 ),
					pose_1.matrix().cast<double>() * Eigen::Vector4d( 0.05, -0.05, 0, 1.0 ),
					pose_1.matrix().cast<double>() * Eigen::Vector4d( -0.05, -0.05, 0, 1.0 )
				};

				for ( int k = 0; k < 4; k++ ) {
					p.x = 320.0 + cams[ k ](0) * resolution;
					p.y = 440.0 - cams[ k ](2) * resolution;
					q.x = 320.0 + cams[ (k + 1)%4 ](0) * resolution;
					q.y = 440.0 - cams[ (k + 1)%4 ](2) * resolution;
					cv::line(traj_, p, q, cv::Scalar( 0, 0, 255 ), 1, 16);
				}

				float3 org = kinfu_->getCyclicalBufferStructure ()->origin_metric;
				Eigen::Vector4d camsa[ 4 ] = { 
					Eigen::Vector4d( 1.5, 1.5 + org.y, 0.3 + org.z, 1.0 ),
					Eigen::Vector4d( 1.5, 1.5 + org.y, 3.3 + org.z, 1.0 ),
					Eigen::Vector4d( -1.5, -1.5 + org.y, 3.3 + org.z, 1.0 ),
					Eigen::Vector4d( -1.5, -1.5 + org.y, 0.3 + org.z, 1.0 ),
				};

				for ( int k = 0; k < 4; k++ ) {
					p.x = 320.0 + camsa[ k ](0) * resolution;
					p.y = 440.0 - camsa[ k ](2) * resolution;
					q.x = 320.0 + camsa[ (k + 1)%4 ](0) * resolution;
					q.y = 440.0 - camsa[ (k + 1)%4 ](2) * resolution;
					cv::line(traj_, p, q, cv::Scalar( 0, 255, 0 ), 1, 16);
				}
			}
		}
	}

	void processGraphSchedule_SetHead( int head_frame_ )
	{
		framed_transformation_.flag_ = framed_transformation_.ResetFlag;
		rgbd_graph_.head_frame_ = head_frame_;
		rgbd_graph_.head_mat_ = rgbd_traj_.data_[ head_frame_ - 1 ].transformation_;
		rgbd_graph_.head_inv_ = rgbd_graph_.head_mat_.inverse();
	}

	void processGraphSchedule_SetTail( int tail_frame )
	{
		// note that tail matrix is the relevant matrix to current world
		framed_transformation_.flag_ = framed_transformation_.IgnoreIntegrationFlag;
		rgbd_graph_.tail_frame_ = tail_frame;
		rgbd_graph_.tail_mat_ = kinfu_->getCameraPose().matrix();
		rgbd_graph_.tail_inv_ = rgbd_graph_.tail_mat_.inverse();
	}

	void processGraphSchedule()
	{
		if ( use_schedule_ ) {
			if ( schedule_traj_.ended() ) {
				exit_ = true;
				return;
			}
			FramedTransformation & ft = schedule_traj_.data_[ schedule_traj_.index_ ];
			framed_transformation_.type_ = ( FramedTransformation::RegistrationType )ft.id1_;
			framed_transformation_.flag_ = ft.id2_;
			framed_transformation_.transformation_ = ft.transformation_;
			framed_transformation_.frame_ = ft.frame_;

			if ( ft.id1_ == 2 ) {
				traj_token_ = kinfu_->getNumberOfPoses();
				cout << traj_token_ << endl;
			}

			if ( frame_id_ + 1 != ft.frame_ ) {
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( ft.frame_ );
			}
			schedule_traj_.index_++;
		} else if ( use_graph_registration_ ) {
			if ( rgbd_graph_.ended() ) {
				return;
			}
			framed_transformation_.flag_ &= ~framed_transformation_.ResetFlag;
			RGBDGraph::RGBDGraphEdge & edge = rgbd_graph_.edges_[ rgbd_graph_.index_ ];
			// we need to check frame_id_
			// when frame_id_ == 0, we need to initialize, seek to (edge.frame_j_ - rate_ + 1)
			// when frame_id_ in [edge.frame_j_ - rate_ + 1, edge.frame_j_ - 1], it is initialization
			// when frame_id_ == (edge.frame_j_), initlize finished, seek to (edge.frame_i_ - rate_ + 1)
			// when frame_id_ in [edge.frame_i_ - rate_ + 1, edge.frame_i_ - 1], it is processing
			// when frame_id_ == (edge.frame_i_), processing finished.
			//// move on to the next index_
			//// if frame_j_ remains, seek to (newedge.frame_i_ - rate_ + 1) --- note that sometimes seek is not needed
			//// otherwise, reset, seek to (newedge.frame_j_ - rate_ + 1)
			int js = edge.frame_j_ - fragment_rate_ + 1;
			int je = edge.frame_j_;
			int is = edge.frame_i_ - fragment_rate_ + 1;
			int ie = edge.frame_i_;
			if ( frame_id_ == 0 ) {
				processGraphSchedule_SetHead( js );
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( js );
			} else if ( frame_id_ == je ) {
				processGraphSchedule_SetTail( je );
				if ( frame_id_ + 1 != ie ) {
					( ( ONIGrabber * ) &capture_ )->seekDepthFrame( ie );
				}
				//if ( frame_id_ + 1 != is ) {
				//( ( ONIGrabber * ) &capture_ )->seekDepthFrame( is );
				//}
			} else if ( frame_id_ == ie ) {
				rgbd_graph_.index_++;
				if ( !rgbd_graph_.ended() ) {
					RGBDGraph::RGBDGraphEdge & newedge = rgbd_graph_.edges_[ rgbd_graph_.index_ ];
					int njs = newedge.frame_j_ - fragment_rate_ + 1;
					int nje = newedge.frame_j_;
					int nis = newedge.frame_i_ - fragment_rate_ + 1;
					int nie = newedge.frame_i_;
					if ( newedge.frame_j_ == edge.frame_j_ ) {
						if ( frame_id_ + 1 != nie ) {
							( ( ONIGrabber * ) &capture_ )->seekDepthFrame( nie );
						}
						//if ( frame_id_ + 1 != nis ) {
						//( ( ONIGrabber * ) &capture_ )->seekDepthFrame( nis );
						//}
					} else {
						// reset
						processGraphSchedule_SetHead( njs );
						( ( ONIGrabber * ) &capture_ )->seekDepthFrame( njs );
					}
				} else {
					exit_ = true;
				}
			}
		} else if ( use_bbox_ ) {
			if ( frame_id_ > 0 && frame_id_ < ( int )bbox_.frames_.size() ) {	// when there is a next
				int i = frame_id_;
				for ( i = frame_id_; i < ( int )bbox_.frames_.size(); i++ ) {
					if ( kinfu_->intersect( bbox_.frames_[ i ].bounds_ ) ) {
						break;
					}
				}
				if ( i == ( int )bbox_.frames_.size() ) {
					exit_ = true;
				} else {
					if ( frame_id_ != i ) {
						( ( ONIGrabber * ) &capture_ )->seekDepthFrame( i + 1 );
					}
				}
			}
		}
	}

	void processFramedTransformation()
	{
		if ( use_schedule_ ) {
			// follow the rules set in schedule file
		} else if ( use_graph_registration_ ) {
			if ( frame_id_ > 0 && frame_id_ <= ( int )rgbd_traj_.data_.size() ) {
				if ( framed_transformation_.flag_ & framed_transformation_.ResetFlag ) {
					framed_transformation_.transformation_ = rgbd_traj_.data_[ frame_id_ - 1 ].transformation_;
					framed_transformation_.type_ = framed_transformation_.Kinfu;
				} else {
					framed_transformation_.transformation_ = kinfu_->getCameraPose( 0 ) * rgbd_graph_.head_inv_ * rgbd_traj_.data_[ frame_id_ - 1 ].transformation_;
					framed_transformation_.type_ = framed_transformation_.InitializeOnly;
				}
			}
		} else if ( use_rgbdslam_ ) {
			if ( frame_id_ > 0 && frame_id_ <= ( int )rgbd_traj_.data_.size() ) {
				framed_transformation_.transformation_ = kinfu_->getCameraPose( 0 ) * rgbd_traj_.head_inv_ * rgbd_traj_.data_[ frame_id_ - 1 ].transformation_;
				framed_transformation_.type_ = framed_transformation_.DirectApply;
			}
			if ( fragment_rate_ > 0 ) {
				if ( frame_id_ > 0 && frame_id_ % ( fragment_rate_ * 2 ) == fragment_start_ + 1 ) {
					framed_transformation_.flag_ = framed_transformation_.ResetFlag;
				} else if ( frame_id_ > 0 && frame_id_ % ( fragment_rate_ * 2 ) == fragment_start_ ) {
					framed_transformation_.flag_ = FramedTransformation::SavePointCloudFlag;
				} else {
					framed_transformation_.flag_ = 0;
				}
			}
		} else if ( fragment_rate_ > 0 ) {
			if ( frame_id_ > 0 && frame_id_ % ( fragment_rate_ * 2 ) == fragment_start_ + 1 ) {
				framed_transformation_.flag_ = framed_transformation_.ResetFlag;
			} else if ( frame_id_ > 0 && frame_id_ % ( fragment_rate_ * 2 ) == fragment_start_ ) {
				framed_transformation_.flag_ = FramedTransformation::SavePointCloudFlag;
			} else {
				framed_transformation_.flag_ = 0;
			}
		}

		/*
		// check rgbdslam data
		FramedTransformation temp_frame;
		FramedTransformation * frame_ptr = &temp_frame;
		if ( use_rgbdslam_ ) {
		if ( frame_id_ > 0 && frame_id_ <= ( int )rgbd_traj_.data_.size() && rgbd_traj_.data_[ frame_id_ - 1 ].frame_ == frame_id_ ) {
		frame_ptr = &rgbd_traj_.data_[ frame_id_ - 1 ];
		} else if ( rgbd_traj_.index_ < rgbd_traj_.data_.size() && rgbd_traj_.data_[ rgbd_traj_.index_ ].frame_ == frame_counter_ ) {
		frame_ptr = &rgbd_traj_.data_[ rgbd_traj_.index_ ];
		rgbd_traj_.index_ ++;
		}
		}

		if ( frame_id_ > 0 ) {
		frame_ptr->frame_ = frame_id_;
		} else {
		frame_ptr->frame_ = frame_counter_;
		}

		if ( use_graph_registration_ ) {
		if ( frame_ptr->frame_ == fragment_start_ + 1 ) {
		frame_ptr->flag_ |= frame_ptr->ResetFlag;
		} else if ( rgbd_graph_.index_ < ( int )rgbd_graph_.edges_.size() ) {
		RGBDGraph::RGBDGraphEdge & edge = rgbd_graph_.edges_[ rgbd_graph_.index_ ];
		if ( frame_ptr->frame_ > fragment_start_ + fragment_rate_ ) {				// fun part, when map is built
		if ( frame_ptr->frame_ + fragment_rate_ < edge.frame_i_ ) {
		// just ignore it
		frame_ptr->flag_ |= ( frame_ptr->IgnoreRegistrationFlag | frame_ptr->IgnoreIntegrationFlag );
		} else if ( frame_ptr->frame_ <= edge.frame_i_ ) {
		// registration from rgbd pose
		//frame_ptr->flag_ |= frame_ptr->IgnoreIntegrationFlag;
		frame_ptr->type_ = frame_ptr->InitializeOnly;

		if ( frame_ptr->frame_ == edge.frame_i_ ) {
		// shift index
		for ( rgbd_graph_.index_++; rgbd_graph_.index_ < ( int )rgbd_graph_.edges_.size(); rgbd_graph_.index_++ ) {
		if ( rgbd_graph_.edges_[ rgbd_graph_.index_ ].frame_j_ == edge.frame_j_ ) {
		break;
		}
		}
		}
		}
		}
		} else {
		// just ignore it
		frame_ptr->flag_ |= ( frame_ptr->IgnoreRegistrationFlag | frame_ptr->IgnoreIntegrationFlag );
		}
		} else {
		if ( fragment_rate_ > 0 ) {
		if ( frame_ptr->frame_ % ( fragment_rate_ * 2 ) == fragment_start_ + 1 ) {
		frame_ptr->flag_ |= frame_ptr->ResetFlag;
		}
		}
		}
		*/
	}

	void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const pcl::gpu::PixelRGB>& rgb24, bool has_data)
	{        
		bool has_image = false;

		if ( has_data ) {
			frame_counter_++;
		}

		if ( record_script_ ) {
			if ( kinfu_->shiftNextTime() ) {
				script_frames_.push( ScriptAction( 'g', frame_counter_ ) );
			}
		}
		if ( play_script_ ) {
			if ( script_frames_.empty() == false && frame_counter_ == script_frames_.front().frame_ && 'g' == script_frames_.front().action_ ) {
				script_frames_.pop();
				kinfu_->forceShift();
			}
		}

		if ( kinfu_->shiftNextTime() ) {
			scene_cloud_view_.show( *kinfu_, integrate_colors_ );
			if(scene_cloud_view_.point_colors_ptr_->points.empty()) // no colors
			{
				if (scene_cloud_view_.compute_normals_)
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_));
				else
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, scene_cloud_view_.cloud_ptr_);
				// if (scene_cloud_view_.valid_combined_)
				//writeCloudFile (file_index_, KinFuApp::PCD_BIN, scene_cloud_view_.combined_ptr_);
				// else
				//writeCloudFile (file_index_, KinFuApp::PCD_BIN, scene_cloud_view_.cloud_ptr_);
			}
			else
			{        
				if (scene_cloud_view_.compute_normals_) {
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGBNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_, *scene_cloud_view_.point_colors_ptr_));
				}
				else
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGB>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.point_colors_ptr_));
				// if (scene_cloud_view_.valid_combined_)
				//writeCloudFile (file_index_, KinFuApp::PCD_BIN, merge<PointXYZRGBNormal>(*scene_cloud_view_.combined_ptr_, *scene_cloud_view_.point_colors_ptr_));
				// else
				//writeCloudFile (file_index_, KinFuApp::PCD_BIN, merge<PointXYZRGB>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.point_colors_ptr_));
			}

			Eigen::Affine3f aff = kinfu_->getCameraPose();
			//cout << aff.matrix() << endl;

			//cout << "Update transformation matrix from:" << endl;
			//cout << transformation_ << endl;
			transformation_ = Eigen::Matrix4f::Identity();
			transformation_(0,3) = kinfu_->getCyclicalBufferStructure()->origin_metric.x;
			transformation_(1,3) = kinfu_->getCyclicalBufferStructure()->origin_metric.y;
			transformation_(2,3) = kinfu_->getCyclicalBufferStructure()->origin_metric.z;

			transformation_ = kinfu_->getInitTrans() * transformation_;

			writeTransformation( file_index_, transformation_ );
			//transformation_ = transformation_ * aff.matrix() * transformation_inverse_;
			cout << "Update transformation matrix to:" << endl;
			cout << transformation_ << endl;

			file_index_++;

			if ( has_data == false ) {
				kinfu_->clearForceShift();
			}
		}

		if ( play_script_ ) {
			if ( script_frames_.empty() == false && frame_counter_ == script_frames_.front().frame_ && 'q' == script_frames_.front().action_ ) {
				script_frames_.pop();
				exit_ = true;
				return;
			}
		}

		if (has_data)
		{
			//ScopeTimeT time("if-has_data");
			tt3.tic();

			//if ( frame_id_ == 601 ) {
			//	char * buf;
			//	buf = new char[ 512 * 512 * 512 * sizeof( int ) ];
			//	std::ifstream ifile( "test.bin", std::ifstream::binary );
			//	ifile.read( buf, 512 * 512 * 512 * sizeof( int ) );
			//	ifile.close();
			//	kinfu_->volume().data().upload( buf, kinfu_->volume().data().cols() * sizeof(int), kinfu_->volume().data().rows(), kinfu_->volume().data().cols() );
			//	delete []buf;
			//}

			if(kinfu_->dbgKf_ >= 1)
				printf("depth_device_. c/r/step: %d, %d, %d\n", depth_device_.cols(), depth_device_.rows(), depth_device_.step()); //640, 480, 1536

			depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);

			if(kinfu_->dbgKf_ >= 1){
				printf("depth_device_-after. c/r/step: %d, %d, %d\n", depth_device_.cols(), depth_device_.rows(), depth_device_.step()); //640, 480, 1536
				printf("execute.rgb24. c/r, data: (%d, %d), %d\n", rgb24.cols, rgb24.rows, rgb24.data);
				printf("colors_device_. c/r/step: %d, %d, %d\n", image_view_.colors_device_.cols(), image_view_.colors_device_.rows(), image_view_.colors_device_.step());
			}

			if (integrate_colors_)
				image_view_.colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);

			{
				SampledScopeTime fps(time_ms_);

				processFramedTransformation();

				if ( rgbd_odometry_ ) {

					// normalize color of grayImage1_
					cv::Scalar scale0 = cv::mean( grayImage0_ );
					cv::Scalar scale1 = cv::mean( grayImage1_ );
					grayImage1_.convertTo( grayImage1_, -1, scale0.val[ 0 ] / scale1.val[ 0 ] );
					//cout << scale0.val[ 0 ] / scale1.val[ 0 ] << endl;

					vector<int> iterCounts(4);
					iterCounts[0] = 7;
					iterCounts[1] = 7;
					iterCounts[2] = 7;
					iterCounts[3] = 10;

					vector<float> minGradMagnitudes(4);
					minGradMagnitudes[0] = 12;
					minGradMagnitudes[1] = 5;
					minGradMagnitudes[2] = 3;
					minGradMagnitudes[3] = 1;
					const float minDepth = 0.f; //in meters
					const float maxDepth = 7.5f; //in meters
					const float maxDepthDiff = 0.07f; //in meters

					//float vals[] = {525., 0., 3.1950000000000000e+02,
					//	0., 525., 2.3950000000000000e+02,
					//	0., 0., 1.};
					float vals[] = { camera_.fx_, 0., camera_.cx_,
						0., camera_.fy_, camera_.cy_,
						0., 0., 1.
					};

					const cv::Mat cameraMatrix = cv::Mat(3,3,CV_32FC1,vals);
					const cv::Mat distCoeff(1,5,CV_32FC1,cv::Scalar(0));

					//run kinfu algorithm
					if (integrate_colors_) {
						has_image = (*kinfu_).rgbdodometry(
							grayImage0_, depthFlt0_, cv::Mat(),
							grayImage1_, depthFlt1_, cv::Mat(),
							cameraMatrix, minDepth, maxDepth, maxDepthDiff,
							iterCounts, minGradMagnitudes,
							depth_device_, &image_view_.colors_device_, &framed_transformation_
							);
					} else {
						has_image = (*kinfu_).rgbdodometry(
							grayImage0_, depthFlt0_, cv::Mat(),
							grayImage1_, depthFlt1_, cv::Mat(),
							cameraMatrix, minDepth, maxDepth, maxDepthDiff,
							iterCounts, minGradMagnitudes,
							depth_device_
							);
					}
				} else if ( kintinuous_ ) {
					cv::Scalar scale0 = cv::mean( grayImage0_ );
					cv::Scalar scale1 = cv::mean( grayImage1_ );
					grayImage1_.convertTo( grayImage1_, -1, scale0.val[ 0 ] / scale1.val[ 0 ] );

					vector<int> iterCounts(3);
					iterCounts[0] = 10;
					iterCounts[1] = 5;
					iterCounts[2] = 4;

					vector<float> minGradMagnitudes(3);
					minGradMagnitudes[0] = 9;
					minGradMagnitudes[1] = 3;
					minGradMagnitudes[2] = 1;

					const float minDepth = 0.f; //in meters
					const float maxDepth = 7.5f; //in meters
					const float maxDepthDiff = 0.07f; //in meters

					//float vals[] = {525., 0., 3.1950000000000000e+02,
					//	0., 525., 2.3950000000000000e+02,
					//	0., 0., 1.};
					float vals[] = { camera_.fx_, 0., camera_.cx_,
						0., camera_.fy_, camera_.cy_,
						0., 0., 1.
					};

					const cv::Mat cameraMatrix = cv::Mat(3,3,CV_32FC1,vals);
					const cv::Mat distCoeff(1,5,CV_32FC1,cv::Scalar(0));

					//run kinfu algorithm
					if (integrate_colors_) {
						has_image = (*kinfu_)(
							grayImage0_, depthFlt0_, cv::Mat(),
							grayImage1_, depthFlt1_, cv::Mat(),
							cameraMatrix, minDepth, maxDepth, maxDepthDiff,
							iterCounts, minGradMagnitudes,
							depth_device_, &image_view_.colors_device_, &framed_transformation_
							);
					} else {
						has_image = (*kinfu_)(
							grayImage0_, depthFlt0_, cv::Mat(),
							grayImage1_, depthFlt1_, cv::Mat(),
							cameraMatrix, minDepth, maxDepth, maxDepthDiff,
							iterCounts, minGradMagnitudes,
							depth_device_
							);
					}
					if ( has_image == false && kinfu_->getGlobalTime() > 1 ) {
						ten_has_data_fail_then_we_call_it_a_day_ = 100;
					}
				} else if ( slac_ ) {
					depth_device_raw_.upload (depth_raw_.data, depth_raw_.step, depth_raw_.rows, depth_raw_.cols);
					has_image = kinfu_->slac(depth_device_raw_, depth_device_, &image_view_.colors_device_);
				} else if ( bdr_odometry_ ) {
					tt0.tic(); //40~60ms
					has_image = kinfu_->bdrodometry( depth_device_, &image_view_.colors_device_ );
					printf("kinfu_->bdrodometry: "); tt0.toc_print();
					if ( has_image == false && kinfu_->getGlobalTime() > 1 ) {
						ten_has_data_fail_then_we_call_it_a_day_ = 100;
					}
				} 
				else if (cu_odometry_){ //长方体定位
					//planeFitter_.run 写在 kinfu.cpp 出错, 原因未知, 放弃 @2017-4-2 13:58:23
					//kinfu_->dbgAhcPeac(depth_device_, &image_view_.colors_device_); //×
					//kinfu_->dbgAhcPeac2(depCloud); //×
					//kinfu_->dbgAhcPeac3(depCloud, &planeFitter_); //×
					//kinfu_->dbgAhcPeac4(&rgbdObj, &planeFitter_); //×
					//::dbgAhcPeac_app(&rgbdObj, &planeFitter_); //√
					//kinfu_->dbgAhcPeac5(&rgbdObj, &planeFitter_); //×
					//kinfu_->dbgAhcPeac6(&rgbdObj, &planeFitter_, dbgAhcPeac_app); //√
					//dbgAhcPeac_kinfu(&rgbdObj, &planeFitter_); //×
					//dbgAhcPeac_testAhc(&rgbdObj, &planeFitter_); //×

					//zc: cuOdo 放到最前面, 这样当用 cam2g 求解 kinfu_->cube_g_.cuVerts8_ 时, 用的是已求出的当前帧(i)姿态, 而非(i-1) 姿态
					//2017-4-22 21:36:41
					tt0.tic(); //40~60ms
					//|-> 目前版本, ~100~130ms, 是 f2mod+f2mkr+e2c (后两个只在 level_index==0 才做) @2017-10-7 16:43:04
					has_image = kinfu_->cuOdometry(depth_device_, &image_view_.colors_device_);
					printf("kinfu_->cuOdometry: "); tt0.toc_print();

					if(!kinfu_->isCuInitialized_){ //若全局 cu 没有初始化、定位
						ScopeTimeT time("NOT kinfu_->isCuInitialized_");

						//1, 平面分割,拟合:
						PlaneFitter planeFitter_;

						cv::Mat dm_raw(depth_device_.rows(), depth_device_.cols(), CV_16UC1); //milli-m, raw
						int c;
						depth_device_.download(dm_raw.data, depth_device_.colsBytes());

						float fx = camera_.fx_, fy = camera_.fy_,
							cx = camera_.cx_, cy = camera_.cy_;
						pcl::device::Intr intr(fx, fy, cx, cy, camera_.integration_trunc_);
						CloudType::Ptr depCloud = cvMat2PointCloud(dm_raw, intr);
						RGBDImage rgbdObj(*depCloud);

						cv::Mat dbgSegMat(depCloud->height, depCloud->width, CV_8UC3); //分割结果可视化
						vector<vector<int>> idxss;

						planeFitter_.minSupport = 000;
						planeFitter_.run(&rgbdObj, &idxss, &dbgSegMat);
						annotateLabelMat(planeFitter_.membershipImg, &dbgSegMat);
						const char *winNameAhc = "ahc-dbg@kinfLS_app";
						//imshow(winNameAhc, dbgSegMat);
						cv::namedWindow(winNameAhc); //因为要 setMouseCallback, 所以必须先有窗口
						cv::Point pxSelect(-1,-1); //目前自动定位立方体方案的初始化过程不完善, 改用手动/鼠标选定第一个顶角方式 //2017-2-20 17:42:45
						cv::setMouseCallback(winNameAhc, peacMouseCallBack, &pxSelect);


						vector<PlaneSeg> plvec; //存放各个平面参数
						plvec = zcRefinePlsegParam(rgbdObj, idxss); //决定是否 refine 平面参数
						vector<vector<double>> cubeCandiPoses; //内vec必有size=12=(t3+R9), 是cube在相机坐标系的姿态, 且规定row-major; 外vec表示多个候选顶角的姿态描述
						//2, 正交三邻面查找:
						zcFindOrtho3tup(plvec, planeFitter_.membershipImg, fx, fy, cx, cy, cubeCandiPoses, dbgSegMat);

						imshow(winNameAhc, dbgSegMat); //改放这里 @2017-4-19 11:32:30
						if(!this->kinfu_->term_1_) //若仅 term_1_, 则不 waitKey
							cv::waitKey(0);

						size_t crnrCnt = cubeCandiPoses.size();
						bool isFoundCrnr = (crnrCnt != 0);
						size_t crnrIdx = 0; //鼠标选中的下标

						//遍历找到的顶角, 看鼠标选中了哪一个: //2017-2-20 17:45:36
						double minPxDist = 1e5; //鼠标双击坐标与【候选顶角】像素坐标距离，的平方, 用于标记更近的【候选点序号】
						for(size_t i=0; i<crnrCnt; i++){
							Vector3d pti(cubeCandiPoses[i].data()); //候选顶角3D坐标
							cv::Point pxi = getPxFrom3d(pti, fx, fy, cx, cy);
							double dist = cv::norm(pxSelect - pxi);
							if(dist < minPxDist){
								minPxDist = dist;
								crnrIdx = i;
							}
						}

						//3, 仅当找到四顶点, 才算定位到立方体
						tt0.tic();
						vector<double> cu4pts; //meters, 因为↓传入的 cuSideLenVec_ 已是量纲米
						bool isFoundCu4pts = false;
						if(isFoundCrnr)
							isFoundCu4pts = getCu4Pts(cubeCandiPoses[crnrIdx], cuSideLenVec_, dm_raw, planeFitter_.membershipImg, fx, fy, cx, cy, cu4pts);
						//找到 crnr 未必 isFoundCu4pts, 因为可能显示不全
						printf("getCu4Pts: "); tt0.toc_print();

						if(isFoundCu4pts){
							//调试观察:
							for(size_t i=0; i<4; i++){
								Vector3d pti(cu4pts.data() + i*3);
								cv::Point pxi = getPxFrom3d(pti, fx, fy, cx, cy);
								cv::circle(dbgSegMat, pxi, 2, 255, 1); //蓝小圆圈
								cv::circle(dbgSegMat, pxi, 12, cv::Scalar(0,255,0), 2); //绿大圆圈,
							}

							kinfu_->cubeCandiPoses_ = cubeCandiPoses;
							kinfu_->crnrIdx_ = crnrIdx;
							//kinfu_->isFirstFoundCube_ = true;

							//8顶点, 固化编号: 0,123,456,7
							//各面顶点分组: 0142/3675, 0253/1476, 0361/2574;
							//最初4个:
							vector<Vector3d> cuVerts8_c;
							for(size_t i=0; i<4; i++){
								Vector3d pti(cu4pts.data() + i*3);
								cuVerts8_c.push_back(pti);
							}
							//再3个:
							//for(size_t i=1; i<4; i++)
							cuVerts8_c.push_back(cuVerts8_c[1] + cuVerts8_c[2] - cuVerts8_c[0]);
							cuVerts8_c.push_back(cuVerts8_c[2] + cuVerts8_c[3] - cuVerts8_c[0]);
							cuVerts8_c.push_back(cuVerts8_c[3] + cuVerts8_c[1] - cuVerts8_c[0]);
							//最后最远一个:
							//cuVerts8_c.push_back(cuVerts8_c[4] + cuVerts8_c[5] + cuVerts8_c[6] - 2 * cuVerts8_c[0]); //-2倍, 非3倍 //×!!!!
							cuVerts8_c.push_back(cuVerts8_c[1] + cuVerts8_c[2] + cuVerts8_c[3] - 2 * cuVerts8_c[0]);

							Eigen::Affine3f cam2g = kinfu_->getCameraPose();
							//转到世界坐标系, 填充 cube_g_:
							//8顶点:
							for(size_t i=0; i<8; i++){
								Vector3d cuVerti_g = cam2g.cast<double>() * cuVerts8_c[i];
								kinfu_->cube_g_.cuVerts8_.push_back(cuVerti_g);
							}

							//6个面: //各面顶点分组: 0142/3675, 0253/1476, 0361/2574;
							int faceVertIdxs[] = {0,1,4,2, 3,6,7,5, 0,2,5,3, 1,4,7,6, 0,3,6,1, 2,5,7,4};
							for(size_t i=0; i<6; i++){
								//Facet facet_i;
								//facet_i.vertIds_.insert(facet_i.vertIds_.end(), faceVertIdxs+4*i, faceVertIdxs+4*(i+1));
								//cube_g_.facetVec_.push_back(facet_i);
								vector<int> vertIds(faceVertIdxs+4*i, faceVertIdxs+4*(i+1) );
								kinfu_->cube_g_.addFacet(vertIds);
							}
							//12棱边:
							int lineVertIds[] = {0,1, 1,4, 4,2, 2,0, 
													3,6, 6,7, 7,5, 5,3, 
													0,3, 1,6, 4,7, 2,5};
							pcl::PointCloud<PointXYZ>::Ptr edgeCloud(new pcl::PointCloud<PointXYZ>);
							edgeCloud->reserve(3000);

							for(size_t i=0; i<12; i++){
								vector<int> edgeIds(lineVertIds+2*i, lineVertIds+2*(i+1) ); 
								kinfu_->cube_g_.addEdgeId(edgeIds);

								Vector3d begPt = kinfu_->cube_g_.cuVerts8_[lineVertIds[2*i] ];
								Vector3d endPt = kinfu_->cube_g_.cuVerts8_[lineVertIds[2*i+1] ];
								Vector3d edgeVec = endPt - begPt;
								Vector3d edgeVecNormed = edgeVec.normalized();
								double edgeVecNorm = edgeVec.norm();
								Vector3d pos = begPt;

								const float STEP = 0.003; //1MM
								float len = 0;
								while(len < edgeVecNorm){
									PointXYZ pt;
									pt.x = pos[0];
									pt.y = pos[1];
									pt.z = pos[2];

									edgeCloud->push_back(pt);

									len += STEP;
									pos += len * edgeVecNormed;
								}
							}
							PCL_WARN("edgeCloud->size:= %d\n", edgeCloud->size());
							//kinfu_->cuEdgeTree_.setInputCloud(edgeCloud); //放到 kinfu.cpp 内
							kinfu_->cuContCloud_ = edgeCloud;
							kinfu_->isCuInitialized_ = true;
							kinfu_->volume().create_init_cu_volume();

							//zc: 求解用于分割基座、扫描物体的平面参数, @2017-8-13 17:18:53
							//暂定朝上 (cam-coo Y负方向)
							//注意: 此平面求解转到 g-coo
							Vector3d pt0(cu4pts.data());
							for(size_t i=1; i<4; i++){
								Vector3d pti(cu4pts.data() + i*3);

								//假设相机水平, 斜俯视, 123 号只有一个点在 0号点下方
								if(pt0.y() < pti.y()){ //此时还在相机坐标系下
									//计算 ABCD 务必在全局坐标系下
									Affine3d cam2gd = cam2g.cast<double>();
									Vector3d pt0g = cam2gd * pt0;
									Vector3d pti_g = cam2gd * pti;

									Vector3d nvec_g = pt0g - pti_g; //0-i 反着做差, 量纲=米
									nvec_g.normalize();

									double D = -(nvec_g.dot(pt0g) + kinfu_->plFiltShiftMM_ * MM2M);
									kinfu_->planeFiltParam_ << nvec_g.x(), nvec_g.y(), nvec_g.z(), D;
									//调试观察:
									cout << "kinfu_->planeFiltParam_: " << kinfu_->planeFiltParam_ << endl;
									cout << "(pt0-pti).dot(pt0): " << (pt0-pti).dot(pt0) << endl;
									cv::Point px0 = getPxFrom3d(pt0, fx, fy, cx, cy);
									cv::Point pxnvec = getPxFrom3d(pt0+cam2gd.rotation().transpose()*nvec_g*5e-2, fx, fy, cx, cy);
									cv::line(dbgSegMat, px0, pxnvec, 255, 2);

									break;
								}
							}

						}//if-(isFoundCu4pts)

						imshow(winNameAhc, dbgSegMat);

					}//if-(!isCuInitialized_)

				}//elif-(cu_odometry_)
				else if (s2s_odometry_){
					kinfu_->volume().create_init_s2s_volume();

					tt0.tic();
					has_image = kinfu_->s2sOdometry(depth_device_, &image_view_.colors_device_);
					printf("kinfu_->s2sOdometry: "); tt0.toc_print();

				}

				else if ( kdtree_odometry_ ) {
					has_image = kinfu_->kdtreeodometry( depth_device_, &image_view_.colors_device_ );
				} else {
					//run kinfu algorithm
					if (integrate_colors_)
						has_image = (*kinfu_) (depth_device_, &image_view_.colors_device_, &framed_transformation_);
					else
						has_image = (*kinfu_) (depth_device_, NULL, &framed_transformation_);

					if ( has_image == false && kinfu_->getGlobalTime() > 1 ) {
						ten_has_data_fail_then_we_call_it_a_day_ = 100;
					}
				}
			}

			if(kinfu_->dbgKf_ >= 1){
			tt2.tic();
			image_view_.showDepth (depth_);
			printf("image_view_.showDepth: "); tt2.toc_print(); //5~30ms, 不稳定, why?
			tt2.tic();
			// update traj_
			drawTrajectory();
			printf("drawTrajectory: "); tt2.toc_print(); //0~1ms, 绘制其实很快
			tt2.tic();
			image_view_.showTraj (traj_, kinfu_image_ ? frame_id_ : -1, kinfu_image_dir_);
			//image_view_.showGeneratedDepth(kinfu_, kinfu_->getCameraPose());
			printf("image_view_.showTraj: "); tt2.toc_print(); //3~70ms, why?
			}

			if ( record_log_ ) {
				if ( use_schedule_ ) {
					if ( framed_transformation_.flag_ & framed_transformation_.SaveAbsoluteMatrix ) {
						kinfu_traj_.data_.push_back( FramedTransformation( 
							kinfu_traj_.data_.size(),
							frame_id_ - 1,
							frame_id_,
							kinfu_->getCameraPose().matrix()
							) );
						kinfu_traj_.cov_.push_back( kinfu_->getCoVarianceMatrix() );
					}
					if ( framed_transformation_.flag_ & framed_transformation_.ResetFlag ) {
						schedule_matrices_.clear();
					}
					if ( framed_transformation_.flag_ & framed_transformation_.PushMatrixHashFlag ) {
						pair< int, FramedTransformation > map_data;
						schedule_matrices_.insert( pair< int, Matrix4f >( frame_id_ - 1, kinfu_->getCameraPose().matrix().inverse() ) );
						//PCL_INFO( "Frame #%d : insert matrix into hash map\n", frame_id_ );
					} else if ( framed_transformation_.flag_ & framed_transformation_.IgnoreIntegrationFlag ) {
						if ( framed_transformation_.flag_ & framed_transformation_.SaveAbsoluteMatrix ) {
						} else {
							int i = frame_id_ - 1;
							vector< int > & prev = next_pointers_[ i ];
							hash_map< int, Matrix4f >::const_iterator it;
							for ( int k = 0; k < prev.size(); k++ ) {
								it = schedule_matrices_.find( prev[ k ] );
								if ( it != schedule_matrices_.end() ) {
									// found!
									kinfu_traj_.data_.push_back( FramedTransformation( 
										kinfu_traj_.data_.size(),
										prev[ k ] + 1,
										frame_id_,
										it->second * kinfu_->getCameraPose().matrix()
										) );
									kinfu_traj_.cov_.push_back( kinfu_->getCoVarianceMatrix() );
									//PCL_INFO( "Frame #%d : find edge base %d\n", frame_id_, prev[ k ] + 1 );
								} else {
									// not found! write down the absolute transformation
									kinfu_traj_.data_.push_back( FramedTransformation( 
										kinfu_traj_.data_.size(),
										file_index_,
										frame_id_,
										kinfu_->getCameraPose().matrix()
										) );
									kinfu_traj_.cov_.push_back( kinfu_->getCoVarianceMatrix() );
									break;
									//PCL_INFO( "Frame #%d : find edge base %d\n", frame_id_, prev[ k ] + 1 );
								}
							}
						}
					}
				} else if ( use_graph_registration_ ) {
					if ( framed_transformation_.flag_ & framed_transformation_.IgnoreIntegrationFlag ) {
						kinfu_traj_.data_.push_back( FramedTransformation( kinfu_traj_.data_.size(), rgbd_graph_.tail_frame_, frame_id_, rgbd_graph_.tail_inv_ * kinfu_->getCameraPose().matrix() ) );
						//cout << rgbd_graph_.tail_inv_ << endl;
						//cout << kinfu_->getCameraPose().matrix() << endl;
						//cout << rgbd_graph_.tail_inv_ * kinfu_->getCameraPose().matrix() << endl;
					}
				} else {
					if ( kinfu_->getGlobalTime() > 0 ) {
						// global_time_ == 0 only when lost and reset, in this case, we lose one frame
						kinfu_traj_.data_.push_back( FramedTransformation( kinfu_traj_.data_.size(), kinfu_->getGlobalTime() - 1, frame_id_, kinfu_->getCameraPose().matrix() ) );
					}
				}
			}
			printf("if-has_data: "); tt3.toc_print();
		}//if-(has_data)

		/*
		if ( frame_id_ == 960 ) {
		scene_cloud_view_.show( *kinfu_, integrate_colors_ );
		if(scene_cloud_view_.point_colors_ptr_->points.empty()) // no colors
		{
		if (scene_cloud_view_.compute_normals_)
		writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_));
		else
		writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, scene_cloud_view_.cloud_ptr_);
		}
		else
		{        
		if (scene_cloud_view_.compute_normals_) {
		writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGBNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_, *scene_cloud_view_.point_colors_ptr_));
		}
		else
		writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGB>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.point_colors_ptr_));
		}

		Eigen::Matrix4f xx = kinfu_->getCameraPose().matrix();
		xx( 0, 3 ) -= kinfu_->getCyclicalBufferStructure()->origin_metric.x;
		xx( 1, 3 ) -= kinfu_->getCyclicalBufferStructure()->origin_metric.y;
		xx( 2, 3 ) -= kinfu_->getCyclicalBufferStructure()->origin_metric.z;
		cout << xx << endl;

		}
		*/

		if ( ( use_schedule_ || fragment_rate_ > 0 ) && record_log_ && ( framed_transformation_.flag_ & framed_transformation_.SavePointCloudFlag ) && ten_has_data_fail_then_we_call_it_a_day_ == 0 ) {

			scene_cloud_view_.show( *kinfu_, integrate_colors_ );
			if(scene_cloud_view_.point_colors_ptr_->points.empty()) // no colors
			{
				if (scene_cloud_view_.compute_normals_)
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_));
				else
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, scene_cloud_view_.cloud_ptr_);
			}
			else
			{        
				if (scene_cloud_view_.compute_normals_) {
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGBNormal>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.normals_ptr_, *scene_cloud_view_.point_colors_ptr_));
				}
				else
					writeCloudFile (file_index_, KinFuLSApp::PCD_BIN, merge<PointXYZRGB>(*scene_cloud_view_.cloud_ptr_, *scene_cloud_view_.point_colors_ptr_));
			}

			// enable when you need mesh output instead of pcd output when using --fragment
			//scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
			//writeMesh( KinFuLSApp::MESH_PLY, file_index_ );

			/*s
			if ( framed_transformation_.flag_ & framed_transformation_.SaveAbsoluteMatrix ) {
			} else {
			char filename[ 1024 ];
			memset( filename, 0, 1024 );

			sprintf( filename, "cloud_bin_fragment_%d.log", file_index_ );

			kinfu_traj_.saveToFile( filename );
			kinfu_traj_.clear();
			}
			*/

			/*
			//std::vector< int > raw_data;
			//int col;
			//kinfu_->volume().data().download( raw_data, col );
			cout << "Downloading TSDF volume from device ... " << flush;
			kinfu_->volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
			tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_->volume().getSize ());

			int cnt = 0;
			for ( int i = 0; i < ( int )tsdf_volume_.size(); i++ ) {
			if ( tsdf_volume_.volume().at( i ) != 0.0f )
			cnt++;
			}
			cout << "valid voxel number is " << cnt << endl;

			writeRawTSDF( file_index_, tsdf_volume_ );
			*/

			file_index_++;
		}

		//tt3.tic();
		if (scan_)
		{
			scan_ = false;
			scene_cloud_view_.show (*kinfu_, integrate_colors_);

			if (scan_volume_)
			{                
				cout << "Downloading TSDF volume from device ... " << flush;
				kinfu_->volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
				tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_->volume().getSize ());
				cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;

				cout << "Converting volume to TSDF cloud ... " << flush;
				tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
				cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;        

				//zc: @2018-1-26 18:54:49
				//cv::Mat sliceDbgMat32f;
				//tsdf_volume_.slice2D(kinfu_->vxlDbg_[0], kinfu_->vxlDbg_[1], kinfu_->vxlDbg_[2],
				//	1, sliceDbgMat32f);

				//cv::Mat sliceDbgMatCmap;
				//cv::applyColorMap(sliceDbgMat32f, sliceDbgMatCmap, cv::COLORMAP_RAINBOW);
				//cv::imshow("sliceDbgMatCmap", sliceDbgMatCmap);
			}
			else
				cout << "[!] tsdf volume download is disabled" << endl << endl;
		}
		//printf("if-scan_: "); tt3.toc_print(); //0~1ms

		if(scan_volume_ && slice2d_){
			//zc: @2018-1-26 18:54:49
			cout << "Downloading TSDF volume from device ... " << flush;
			kinfu_->volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
			tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_->volume().getSize ());
			cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;

			const int fix_axis = kinfu_->vxlDbg_[3];
			cv::Mat sliceGlobal32f,
					sliceLocal32f;
			sliceGlobal32f = tsdf_volume_.slice2D(kinfu_->vxlDbg_[0], kinfu_->vxlDbg_[1], kinfu_->vxlDbg_[2],
				fix_axis, sliceLocal32f, true);

			cv::Mat sliceGlobal8u,
					sliceLocal8u;
			sliceLocal32f.convertTo(sliceLocal8u, CV_8U, UCHAR_MAX*0.5, UCHAR_MAX*0.5); //数值映射: [-1,1]->[0,255], 因为 applyColorMap-COLORMAP_RAINBOW 对 float 居然成灰白色
			sliceGlobal32f.convertTo(sliceGlobal8u, CV_8U, UCHAR_MAX*0.5, UCHAR_MAX*0.5);

			cv::Mat sliceGlobalCmap,
					sliceLocalCmap;
			//cv::applyColorMap(sliceDbgMat32f, sliceDbgMatCmap, cv::COLORMAP_RAINBOW);
			cv::applyColorMap(sliceLocal8u, sliceLocalCmap, cv::COLORMAP_RAINBOW);
			cv::applyColorMap(sliceGlobal8u, sliceGlobalCmap, cv::COLORMAP_RAINBOW);

			char *sliceLocalCmap_win_str = "sliceLocalCmap";
			//cv::namedWindow(sliceLocalCmap_win_str);
			//cv::moveWindow(sliceLocalCmap_win_str, 1150, 650);
			cv::imshow(sliceLocalCmap_win_str, sliceLocalCmap);

			char *sliceGlobal8u_win_str = "sliceGlobal8u";
			cv::namedWindow(sliceGlobal8u_win_str);
			cv::moveWindow(sliceGlobal8u_win_str, 1150, 650);
			cv::imshow(sliceGlobal8u_win_str, sliceGlobal8u);

			if(print_nbr_){
				int x2d, y2d;
				if(0 == fix_axis){
					x2d = kinfu_->vxlDbg_[2]; //z做img的X轴
					y2d = kinfu_->vxlDbg_[1];
				}
				else if(1 == fix_axis){
					x2d = kinfu_->vxlDbg_[0]; //x做img的X轴
					y2d = kinfu_->vxlDbg_[2];
				}
				else if(2 == fix_axis){
					x2d = kinfu_->vxlDbg_[0];
					y2d = kinfu_->vxlDbg_[1];
				}

				cv::Rect nbrRoi(x2d-2, y2d-2, 5,5);
				cout << sliceGlobal32f(nbrRoi) << endl;
			}
		}

		//tt3.tic();
		if (scan_mesh_)
		{
			scan_mesh_ = false;
			scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
		}
		//printf("if-scan_mesh_: "); tt3.toc_print(); //0~2ms

		//tt3.tic();
		if (has_image)
		{
			Eigen::Affine3f viewer_pose = getViewerPose(scene_cloud_view_.cloud_viewer_);
			image_view_.showScene (*kinfu_, rgb24, registration_, kinfu_image_ ? frame_id_ : -1, independent_camera_ ? &viewer_pose : 0, kinfu_image_dir_);
		}
		//printf("if-has_image: "); tt3.toc_print(); //30ms, why? 因为: raycast改慢了 @2017-10-8 09:48:54

		if (current_frame_cloud_view_)
			current_frame_cloud_view_->show (*kinfu_);

		if (!independent_camera_)
		{
			//setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose());

			//s2s 实现把 T移到原点, 相应各种viewer 要调整, 避免显示错乱
			Eigen::Affine3f cam_pose_curr_fake = kinfu_->getCameraPose();
			cam_pose_curr_fake.translation() -= kinfu_->volume000_gcoo_;
			setViewerPose (scene_cloud_view_.cloud_viewer_, cam_pose_curr_fake);
		}

		if (enable_texture_extraction_) {
			if ( (frame_counter_  % snapshot_rate_) == 0 )   // Should be defined as a parameter. Done.
			{
				screenshot_manager_.saveImage (kinfu_->getCameraPose(), rgb24);
			}
		}
	}

	void source_cb1(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
	{        
		{
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			if (exit_ || !lock)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];     
		}
		data_ready_cond_.notify_one();
	}

	void source_cb1_trigger(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
	{        
		{
			boost::mutex::scoped_lock lock(data_ready_mutex_);
			if (exit_ || !lock)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];  

			int depth_frame_id = depth_wrapper->getDepthMetaData().FrameID();
			frame_id_ = depth_frame_id;
		}
		data_ready_cond_.notify_one();
	}

	void source_cb2(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
	{
		{
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);

			if (exit_ || !lock)
			{
				return;
			}

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];      

			rgb24_.cols = image_wrapper->getWidth();
			rgb24_.rows = image_wrapper->getHeight();
			rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

			source_image_data_.resize(rgb24_.cols * rgb24_.rows);
			image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
			rgb24_.data = &source_image_data_[0];    

			if ( recording_ ) {
				xn_depth_.CopyFrom( depth_wrapper->getDepthMetaData() );
				xn_image_.CopyFrom( image_wrapper->getMetaData() );
			}
		}
		data_ready_cond_.notify_one();
	}

	void source_cb2_trigger(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
	{
		{
			boost::mutex::scoped_lock lock(data_ready_mutex_);		// in trigger mode, must wait until lock is required

			if (exit_)
			{
				return;
			}

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);

			if ( slac_ ) {
				depth_raw_.cols = depth_wrapper->getWidth();
				depth_raw_.rows = depth_wrapper->getHeight();
				depth_raw_.step = depth_raw_.cols * depth_raw_.elemSize();
				source_depth_data_raw_.resize( depth_raw_.cols * depth_raw_.rows );
				depth_wrapper->fillDepthImageRaw(depth_raw_.cols, depth_raw_.rows, &source_depth_data_raw_[0]);
				depth_raw_.data = &source_depth_data_raw_[ 0 ];
				kinfu_->deformDepthImage( source_depth_data_ );
			}

			if ( use_mask_ ) {
				unsigned short * depth_buffer = &source_depth_data_[0];
				for (unsigned yIdx = 0; yIdx < depth_.rows; ++yIdx)
				{
					for (unsigned xIdx = 0; xIdx < depth_.cols; ++xIdx, ++depth_buffer)
					{
						if ( xIdx < mask_[ 0 ] || xIdx > mask_[ 1 ] || yIdx < mask_[ 2 ] || yIdx > mask_[ 3 ] ) {
							*depth_buffer = 0;
						}
					}
				}
			}

			depth_.data = &source_depth_data_[0];      

			rgb24_.cols = image_wrapper->getWidth();
			rgb24_.rows = image_wrapper->getHeight();
			rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

			source_image_data_.resize(rgb24_.cols * rgb24_.rows);
			image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
			rgb24_.data = &source_image_data_[0];    

			int image_frame_id = image_wrapper->getMetaData().FrameID();
			int depth_frame_id = depth_wrapper->getDepthMetaData().FrameID();
			if ( image_frame_id != depth_frame_id ) {
				frame_id_ = depth_frame_id;
				PCL_WARN( "Triggered frame number asynchronized : depth %d, image %d\n", depth_frame_id, image_frame_id );
			} else {
				frame_id_ = depth_frame_id;
				//PCL_INFO( "Triggered frame : depth %d, image %d\n", depth_frame_id, image_frame_id );
			}
			//cout << "[" << boost::this_thread::get_id() << "] : " << "Process Depth " << depth_wrapper->getDepthMetaData().FrameID() << ", " << depth_wrapper->getDepthMetaData().Timestamp() 
			// << " Image" << image_wrapper->getMetaData().FrameID() << ", " << image_wrapper->getMetaData().Timestamp() << endl;

			if ( rgbd_odometry_ || kintinuous_ ) {
				depthFlt0_.copyTo( depthFlt1_ );
				grayImage0_.copyTo( grayImage1_ );

				cv::Mat depth_mat( depth_wrapper->getHeight(), depth_wrapper->getWidth(), CV_16UC1 );
				cv::Mat image_mat( image_wrapper->getHeight(), image_wrapper->getWidth(), CV_8UC3 );
				depth_wrapper->fillDepthImageRaw( depth_wrapper->getWidth(), depth_wrapper->getHeight(), ( unsigned short * )depth_mat.data );
				image_wrapper->fillRGB( image_wrapper->getWidth(), image_wrapper->getHeight(), ( unsigned char* )image_mat.data );

				cv::cvtColor( image_mat, grayImage0_, CV_RGB2GRAY );
				depth_mat.convertTo( depthFlt0_, CV_32FC1, 1./1000 );
			}

			if ( recording_ ) {
				xn_depth_.CopyFrom( depth_wrapper->getDepthMetaData() );
				xn_image_.CopyFrom( image_wrapper->getMetaData() );
			}
		}
		data_ready_cond_.notify_one();
	}

	void startRecording() {
		pcl::OpenNIGrabber * current_grabber = ( pcl::OpenNIGrabber * )( &capture_ );
		openni_wrapper::OpenNIDevice & device = * current_grabber->getDevice();
		xn::Context & context = device.getContext();
		cout << "Synchronization mode : " << ( device.isSynchronized() ? "On" : "Off" ) << endl;

		xn::EnumerationErrors errors;
		XnStatus rc;
		rc = device.getContext().CreateAnyProductionTree( XN_NODE_TYPE_RECORDER, NULL, xn_recorder_, &errors );
		CHECK_RC_ERR(rc, "Create recorder", errors);

		time_t rawtime;
		struct tm *timeinfo;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		XnChar strFileName[XN_FILE_MAX_PATH];
		sprintf(strFileName, "%04d%02d%02d-%02d%02d%02d.oni",
			timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
		xn_recorder_.SetDestination(XN_RECORD_MEDIUM_FILE, strFileName);
		printf("Creating recording file %s\n", strFileName);

		//XnUInt64 nprop;
		//device.getDepthGenerator().GetIntProperty( "InputFormat", nprop );
		//cout << nprop << endl;
		//device.getDepthGenerator().GetIntProperty( "OutputFormat", nprop );
		//cout << nprop << endl;
		//device.getImageGenerator().GetIntProperty( "InputFormat", nprop );
		//cout << nprop << endl;
		//device.getImageGenerator().GetIntProperty( "OutputFormat", nprop );
		//cout << nprop << endl;

		// Create mock nodes based on the depth generator, to save depth
		rc = context.CreateMockNodeBasedOn( device.getDepthGenerator(), NULL, xn_mock_depth_ );
		CHECK_RC(rc, "Create depth node");
		rc = xn_recorder_.AddNodeToRecording( xn_mock_depth_, XN_CODEC_16Z_EMB_TABLES );
		CHECK_RC(rc, "Add depth node");
		xn_mock_depth_.SetData( xn_depth_ );

		// Create mock nodes based on the image generator, to save image
		rc = context.CreateMockNodeBasedOn( device.getImageGenerator(), NULL, xn_mock_image_ );
		CHECK_RC(rc, "Create image node");
		rc = xn_recorder_.AddNodeToRecording( xn_mock_image_, XN_CODEC_JPEG );
		CHECK_RC(rc, "Add image node");
		xn_mock_image_.SetData( xn_image_ );
	}

	void stopRecording() {
		xn_recorder_.Release();
	}

	void source_cb3(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr & DC3)
	{
		{
			//std::cout << "Giving colors1\n";
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			std::cout << lock << std::endl;
			if (exit_ || !lock)
				return;
			//std::cout << "Giving colors2\n";
			int width  = DC3->width;
			int height = DC3->height;
			depth_.cols = width;
			depth_.rows = height;
			depth_.step = depth_.cols * depth_.elemSize();
			source_depth_data_.resize(depth_.cols * depth_.rows);   

			rgb24_.cols = width;
			rgb24_.rows = height;
			rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 
			source_image_data_.resize(rgb24_.cols * rgb24_.rows);

			unsigned char *rgb    = (unsigned char *)  &source_image_data_[0];
			unsigned short *depth = (unsigned short *) &source_depth_data_[0];  

			//std::cout << "Giving colors3\n";
			for (int i=0; i<width*height; i++) {
				PointXYZRGBA pt = DC3->at(i);
				rgb[3*i +0] = pt.r;
				rgb[3*i +1] = pt.g;
				rgb[3*i +2] = pt.b;
				depth[i]    = pt.z/0.001;
			}
			//std::cout << "Giving colors4\n";
			rgb24_.data = &source_image_data_[0];   
			depth_.data = &source_depth_data_[0];      
		}	
		data_ready_cond_.notify_one();
	}

	void
		startMainLoop (bool triggered_capture)
	{   
		using namespace openni_wrapper;
		typedef boost::shared_ptr<DepthImage> DepthImagePtr;
		typedef boost::shared_ptr<Image>      ImagePtr;

		boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1 = boost::bind (&KinFuLSApp::source_cb2, this, _1, _2, _3);
		boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1t = boost::bind (&KinFuLSApp::source_cb2_trigger, this, _1, _2, _3);
		boost::function<void (const DepthImagePtr&)> func2 = boost::bind (&KinFuLSApp::source_cb1, this, _1);
		boost::function<void (const DepthImagePtr&)> func2t = boost::bind (&KinFuLSApp::source_cb1_trigger, this, _1);
		boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&) > func3 = boost::bind (&KinFuLSApp::source_cb3, this, _1);

		bool need_colors = integrate_colors_ || registration_;

		if ( pcd_source_ && !capture_.providesCallback<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)>() ) {
			std::cout << "grabber doesn't provide pcl::PointCloud<pcl::PointXYZRGBA> callback !\n";
		}

		if(this->evaluation_ptr_ == NULL){
//#if 0   //为了使用 -eval, 暂时屏蔽此 while 逻辑, 改写代码
		boost::signals2::connection c = 
			pcd_source_? capture_.registerCallback (func3) : need_colors ? ( triggered_capture ? capture_.registerCallback (func1t) : capture_.registerCallback (func1) ) : ( triggered_capture ? capture_.registerCallback (func2t) : capture_.registerCallback (func2) );

		{
			ten_has_data_fail_then_we_call_it_a_day_ = 0;

			boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

			capture_.start ();
			if ( recording_ ) {
				startRecording();
			}

			if ( seek_start_ != -1 ) {
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( seek_start_ );
			}

			while (!exit_ && !scene_cloud_view_.cloud_viewer_.wasStopped () && !image_view_.viewerScene_.wasStopped () && !this->kinfu_->isFinished ())
			{ 
				bool has_data;
				if (triggered_capture) {
					processGraphSchedule();
					if ( exit_ ) {
						break;
					}
					//cout << " Triggering, now frame_id_ = " << frame_id_ << endl;
					( ( ONIGrabber * ) &capture_ )->trigger(); // Triggers new frame
				}
				has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(300));
				if ( has_data ) {
					ten_has_data_fail_then_we_call_it_a_day_ = 0;
				} else {
					ten_has_data_fail_then_we_call_it_a_day_++;
				}

				try { 
					tt1.tic();
					this->execute (depth_, rgb24_, has_data); 
					printf("KinFuLSApp.execute: "); tt1.toc_print();

					//cout << frame_id_ << " : " << framed_transformation_.frame_ << endl;
					//boost::this_thread::sleep (boost::posix_time::millisec (300));

					if ( recording_ && has_data ) {
						xn_mock_depth_.SetData( xn_depth_, frame_counter_, frame_counter_ * 30000 + 1 );
						xn_mock_image_.SetData( xn_image_, frame_counter_, frame_counter_ * 30000 );
						xn_recorder_.Record();
					}
				}
				catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
				catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

				/*
				if ( frame_id_ == 5 ) {
				cout << "seek depth to " << frame_id_ + 5 << endl;
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( frame_id_ + 5 );
				boost::this_thread::sleep( boost::posix_time::millisec( 10000 ) );
				} else if ( frame_id_ == 15 ) {
				cout << "seek depth to " << frame_id_ + 5 << endl;
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( frame_id_ + 5 );
				} else if ( frame_id_ == 25 ) {
				cout << "seek depth to " << frame_id_ + 5 << endl;
				( ( ONIGrabber * ) &capture_ )->seekDepthFrame( frame_id_ + 5 );
				}
				*/

				scene_cloud_view_.cloud_viewer_.spinOnce (3);
				//~ cout << "In main loop" << endl;          

				if ( ten_has_data_fail_then_we_call_it_a_day_ >= 20 ) {
					//scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
					//writeMesh( KinFuLSApp::MESH_PLY, -1 );

					exit_ = true;
				}
			} 
			exit_ = true;
			boost::this_thread::sleep (boost::posix_time::millisec (300));

			if (!triggered_capture) {
				capture_.stop (); // Stop stream
				if ( recording_ ) {
					stopRecording();
				}
			}

			if ( record_script_ ) {
				script_frames_.push( ScriptAction( 'q', frame_counter_ ) );
				writeScriptFile ();
			}

			if ( record_log_ ) {
				writeLogFile ();
			}

			cout << "Total " << frame_counter_ << " frames processed." << endl;
		}
		c.disconnect();
		}
		else{ //this->evaluation_ptr_ != NULL
			PCL_WARN("evaluation_ptr_--(fx,fy,cx,cy):= (%.1f,%.1f,%.1f,%.1f)\n", evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
			//↑--代码已改: 当 --camera 时, 并不用 eval 类内写死的 525,525,319.5,239.5 (又在 longRange.param)	@2017-3-30 21:36:58

			int key = -1; //键盘控制
			int currentIndex = 0; //eval 用
			while (evaluation_ptr_->grab (currentIndex, depth_)) { 
			//while (evaluation_ptr_->grab (currentIndex, depth_, rgb24_)) { //加上 rgb, 试图 -r -ic
				if(currentIndex < pngSid_){
					currentIndex ++;
					continue;
				}
				if(currentIndex > pngEid_)
					break;

				printf("---------------currentIndex:= %d\n", currentIndex);

				frame_id_ = currentIndex;

				//--mask xy窗口包围盒分割, 拷贝自上面 @2017-10-10 16:52:53
				if ( use_mask_ ) {
					//unsigned short * depth_buffer = &source_depth_data_[0];
					unsigned short * depth_buffer = const_cast<unsigned short *>(depth_.data); //const-ptr, 这里 dirty hack
					
					for (unsigned yIdx = 0; yIdx < depth_.rows; ++yIdx){
						for (unsigned xIdx = 0; xIdx < depth_.cols; ++xIdx, ++depth_buffer){
							if ( xIdx < mask_[ 0 ] || xIdx > mask_[ 1 ] || yIdx < mask_[ 2 ] || yIdx > mask_[ 3 ] ) {
								*depth_buffer = 0;
							}
						}
					}
				}

				//逐像素的 mask Mat, 非 bbox @2018-4-2 00:15:23
				if(use_omask_){
					char omFnBuf[333];
					sprintf(omFnBuf, "%s/omask/omask_%06d.png", eval_folder_.c_str(), currentIndex);
					cout<<"eval_folder_@use_omask_: "<<eval_folder_<<endl
						<<"omFnBuf: "<<omFnBuf<<endl;

					cv::Mat omMat = cv::imread(omFnBuf, -1);
					//cv::cvtColor(omMat, omMat, CV_RGB2GRAY);

					cv::Mat totalMsk = omMat.clone(); //可能要 +tmask, 看情况 cmd 参数

					if(use_tmask_){
						char tmFnBuf[333];
						sprintf(tmFnBuf, "%s/tmask/tmask_%06d.png", eval_folder_.c_str(), currentIndex);
						cv::Mat tmMat = cv::imread(tmFnBuf, -1);
						totalMsk.setTo(UCHAR_MAX, tmMat == UCHAR_MAX);
					}

					unsigned short * depth_buffer = const_cast<unsigned short *>(depth_.data); //const-ptr, 这里 dirty hack

					for (unsigned yIdx = 0; yIdx < depth_.rows; ++yIdx){
						for (unsigned xIdx = 0; xIdx < depth_.cols; ++xIdx, ++depth_buffer){
							if(totalMsk.at<uchar>(yIdx, xIdx) != UCHAR_MAX)
								//depth_(yIdx, xIdx) = 0;
								*depth_buffer = 0;
						}
					}
				}

				tt1.tic();
				//printf("rgb24_. c/r, data: (%d, %d), %d\n", rgb24_.cols, rgb24_.rows, rgb24_.data);
				this->execute(depth_, rgb24_, true); 
				printf("KinFuLSApp.execute: "); tt1.toc_print();
				scene_cloud_view_.cloud_viewer_.spinOnce (3); 
				//currentIndex += 1; 
				currentIndex += everyXframes_; 

				const string dummy_win0_str = "dummy-win0";
				cv::namedWindow(dummy_win0_str);
				cv::moveWindow(dummy_win0_str, 550, 750);
				//cv::Mat dum_win0_mat = cv::Mat::zeros(333, 333, CV_8UC1);
				cv::Mat dum_win0_mat(333, 333, CV_8UC3, 255);
				cv::imshow(dummy_win0_str, dum_win0_mat);
				//if(currentIndex >= pngPauseId_)
				//	cv::waitKey(0);
					//isPngPause_ = true; //zc: DEPRECATED~

				key = cv::waitKey(currentIndex != pngPauseId_ && png_fps_ > 0 && isReadOn_ ? int(1e3 / png_fps_) : 0);
				//int key = cv::waitKey(110);
				//printf("key:= %d\n", key);

				if(key == 27) //Esc
					break;
				else if(key == ' '){ //这里 waitKey 不灵敏导致切换有问题?
					isReadOn_ = !isReadOn_;
					printf("isReadOn_ = !isReadOn_;\n");
				}
				else if(key == 's'){ //
					isReadOn_ = false;
					printf("isReadOn_ = false;\n");
				}
				else if(key == 't'){
					//@2017-5-8 19:56:28
					kinfu_->genSynData_ = true;
				}
				else if(key == 'd'){
					//开启2D切片观察 tsdf 调试窗口 @2018-1-26 18:56:53
					slice2d_ = !slice2d_;
					printf("slice2d_ = !slice2d_; --> %s\n", slice2d_ ? "TTT" : "FFF");
				}
				else if(key == 'n'){
					print_nbr_ = !print_nbr_;
					printf("print_nbr_ = !print_nbr_; --> %s\n", print_nbr_ ? "TTT" : "FFF");
				}
				else if(key >= '1' && key <= '7'){
					PCL_WARN(">>>>>>>>>PRESSING: %d\n", key - '0');
					kinfu_->setTerm123(key - '0');
				}

				if(currentIndex >= evaluation_ptr_->getStreamSize() - 5){
// 					isReadOn_ = false; //zc: 暂改成: 存 mesh, 不暂停, 走完自动退出 @2018-3-23 22:47:46
					this->scan_mesh_ = true;
					this->writeMesh (7, -1);

					printf("isReadOn_ = false;\n~~~~~~~~~最后几帧\n");
				}
				//isFirstFoundCube_ = false; //这里总设置 false, 没毛病

			}//while-eval

			printf("evaluation_ptr_->grab DONE...\n");

			//cv::waitKey(0); //等手动存 mesh 之类, 不行
			//再次 esc 才退出程序:
			imshow("dummy-win", cv::Mat::zeros(333, 333, CV_8UC1));
			while(1){
				scene_cloud_view_.cloud_viewer_.spinOnce (3); 
				key = cv::waitKey(3);
				if(key == 27){
					printf("exiting-startMainLoop......\n");
					break;
				}
			}
		}//if-(!this->evaluation_ptr_)
	}//startMainLoop

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		writeCloud (int format) const
	{      
		const SceneCloudView& view = scene_cloud_view_;

		// Points to export are either in cloud_ptr_ or combined_ptr_.
		// If none have points, we have nothing to export.
		if (view.cloud_ptr_->points.empty () && view.combined_ptr_->points.empty ())
		{
			cout << "Not writing cloud: Cloud is empty" << endl;
		}
		else
		{
			if(view.point_colors_ptr_->points.empty()) // no colors
			{
				if (view.valid_combined_)
					writeCloudFile (format, view.combined_ptr_);
				else
					writeCloudFile (format, view.cloud_ptr_);
			}
			else
			{        
				if (view.valid_combined_)
					writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
				else
					writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		writeMesh(int format, int file_index) const
	{
		if (scene_cloud_view_.mesh_ptr_) 
			writePoligonMeshFile(format, *scene_cloud_view_.mesh_ptr_, file_index);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		printHelp ()
	{
		cout << endl;
		cout << "KinFu app hotkeys" << endl;
		cout << "=================" << endl;
		cout << "    H    : print this help" << endl;
		cout << "   Esc   : exit" << endl;
		cout << "    G    : immediately shift" << endl;
		cout << "    T    : take cloud" << endl;
		cout << "    A    : take mesh" << endl;
		cout << "    M    : toggle cloud exctraction mode" << endl;
		cout << "    N    : toggle normals exctraction" << endl;
		cout << "    I    : toggle independent camera mode" << endl;
		cout << "    B    : toggle volume bounds" << endl;
		cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
		cout << "    C    : clear clouds" << endl;    
		cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
		cout << "    7,8  : save mesh to PLY, VTK" << endl;
		cout << "   X, V  : TSDF volume utility" << endl;
		//cout << "   L, l  : On the next shift, KinFu will extract the whole current cube, extract the world and stop" << endl;
		//cout << "   S, s  : On the next shift, KinFu will extract the world and stop" << endl;
		cout << endl;
	}  

	bool exit_;
	bool scan_;
	bool scan_mesh_;
	bool scan_volume_;
	bool slice2d_; //zc: 2D切片, 调试观察 tsdf @2018-1-26 18:58:35
	bool print_nbr_; //输出-vxlDbg调试点/面5x5邻域2D矩阵 (float)
	//bool save_and_shift_;
	int file_index_;
	Eigen::Matrix4f transformation_;
	Eigen::Matrix4f transformation_inverse_;

	struct ScriptAction {
		char action_;
		int frame_;
		ScriptAction( char a, int f ) : action_(a), frame_(f) {}
	};

	queue< ScriptAction > script_frames_;
	bool record_script_;
	bool play_script_;
	bool record_log_;
	string record_log_file_;

	bool use_rgbdslam_;
	RGBDTrajectory rgbd_traj_;
	RGBDTrajectory kinfu_traj_;

	bool use_bbox_;
	BBox bbox_;

	bool use_graph_registration_;
	RGBDGraph rgbd_graph_;
	FramedTransformation framed_transformation_;

	bool use_schedule_;
	RGBDTrajectory schedule_traj_;
	vector< vector< int > > next_pointers_;
	hash_map< int, Matrix4f > schedule_matrices_;

	int fragment_rate_;
	int fragment_start_;
	int seek_start_;

	bool use_device_;
	bool recording_;

	bool independent_camera_;
	int frame_counter_;
	int frame_id_;
	bool enable_texture_extraction_;
	pcl::gpu::ScreenshotManager screenshot_manager_;
	int snapshot_rate_;

	bool kinfu_image_;
	std::string kinfu_image_dir_;

	xn::MockDepthGenerator xn_mock_depth_;
	xn::MockImageGenerator xn_mock_image_;
	xn::DepthMetaData xn_depth_;
	xn::ImageMetaData xn_image_;
	xn::Recorder xn_recorder_;

	bool registration_;
	bool integrate_colors_;
	bool pcd_source_;
	float focal_length_;

	pcl::Grabber& capture_;
	KinfuTracker *kinfu_;

	SceneCloudView scene_cloud_view_;
	ImageView image_view_;
	boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

	KinfuTracker::DepthMap depth_device_;
	KinfuTracker::DepthMap depth_device_raw_;

	pcl::TSDFVolume<float, short> tsdf_volume_;
	pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

	Evaluation::Ptr evaluation_ptr_;

	boost::mutex data_ready_mutex_;
	boost::condition_variable data_ready_cond_;

	std::vector<pcl::gpu::PixelRGB> source_image_data_;
	std::vector<unsigned short> source_depth_data_;
	PtrStepSz<const unsigned short> depth_;
	std::vector<unsigned short> source_depth_data_raw_;
	PtrStepSz<const unsigned short> depth_raw_;
	PtrStepSz<const pcl::gpu::PixelRGB> rgb24_;  
	cv::Mat traj_;
	cv::Mat traj_buffer_;
	int traj_token_;

	int time_ms_;

	bool use_mask_;
	int mask_[ 4 ];

	//zc: object-mask, for "sdf2sdf" eval; 相对路径ptn: "./omask/omask_%06d.png" @2018-4-2 00:10:25
	bool use_omask_;
	bool use_tmask_; //table-mask @2018-7-5 14:53:00

	string eval_folder_; //改类内

	bool rgbd_odometry_;
	bool kintinuous_;
	cv::Mat grayImage0_, grayImage1_, depthFlt0_, depthFlt1_;

	int ten_has_data_fail_then_we_call_it_a_day_;

	bool slac_;
	bool bdr_odometry_;
	bool kdtree_odometry_;

	//zc: cuboid 长方体定位
	bool cu_odometry_;
	bool s2s_odometry_; //sdf2sdf impl on GPU

	CameraParam camera_;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	static void
		keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
	{
		KinFuLSApp* app = reinterpret_cast<KinFuLSApp*> (cookie);

		int key = e.getKeyCode ();

		if (e.keyUp ())    
			switch (key)
		{
			case 27: app->exit_ = true; break;
			case (int)'g': case (int)'G': app->kinfu_->forceShift(); break;
			case (int)'t': case (int)'T': app->scan_ = true; break;
			case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
			case (int)'h': case (int)'H': app->printHelp (); break;
			case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExctractionMode (); break;
			case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;      
			case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
			case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
			case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_->volume().getSize()); break;
			case (int)'l': case (int)'L': app->kinfu_->performLastScan (); break;
			case (int)'s': case (int)'S': app->kinfu_->extractAndMeshWorld (); break;
			case (int)'7': case (int)'8': app->writeMesh (key - (int)'0', -1); break;  
			case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;      
			case '*': app->image_view_.toggleImagePaint (); break;

			case (int)'x': case (int)'X':
				app->scan_volume_ = !app->scan_volume_;
				cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
				break;
			case (int)'v': case (int)'V':
				cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
				app->tsdf_volume_.save ("tsdf_volume.dat", true);
				cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
				cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
				pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
				cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;
				break;

			default:
				break;
		}    
	}

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
	writeCloudFile (int format, const CloudPtr& cloud_prt)
{
	if (format == KinFuLSApp::PCD_BIN)
	{
		cout << "Saving point cloud to 'cloud_bin.pcd' (binary)... " << flush;
		pcl::io::savePCDFile ("cloud_bin.pcd", *cloud_prt, true);
	}
	else
		if (format == KinFuLSApp::PCD_ASCII)
		{
			cout << "Saving point cloud to 'cloud.pcd' (ASCII)... " << flush;
			pcl::io::savePCDFile ("cloud.pcd", *cloud_prt, false);
		}
		else   /* if (format == KinFuLSApp::PLY) */
		{
			cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
			pcl::io::savePLYFileASCII ("cloud.ply", *cloud_prt);

		}
		cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
	writeCloudFile ( int file_index, int format, const CloudPtr& cloud_prt )
{
	char filename[ 1024 ];
	memset( filename, 0, 1024 );

	if (format == KinFuLSApp::PCD_BIN)
	{
		sprintf( filename, "cloud_bin_%d.pcd", file_index );
		cout << "Saving point cloud to '" << filename << "' (binary)... " << flush;
		pcl::io::savePCDFile (filename, *cloud_prt, true);
	}
	else
		if (format == KinFuLSApp::PCD_ASCII)
		{
			sprintf( filename, "cloud_%d.pcd", file_index );
			cout << "Saving point cloud to '" << filename << "' (ASCII)... " << flush;
			pcl::io::savePCDFile (filename, *cloud_prt, false);
		}
		else   /* if (format == KinFuApp::PLY) */
		{
			sprintf( filename, "cloud_%d.ply", file_index );
			cout << "Saving point cloud to '" << filename << "' (ASCII)... " << flush;
			pcl::io::savePLYFileASCII (filename, *cloud_prt);

		}
		cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeTransformation( int file_index, const Eigen::Matrix4f & trans )
{
	char filename[ 1024 ];
	memset( filename, 0, 1024 );

	sprintf( filename, "cloud_bin_%d.log", file_index );

	ofstream file( filename );
	if ( file.is_open() ) {
		file << trans << endl;
		file.close();
	}
}

void writeRawTSDF( int file_index, pcl::TSDFVolume<float, short> & tsdf )
{
	char filename[ 1024 ];
	memset( filename, 0, 1024 );

	sprintf( filename, "cloud_bin_tsdf_%d.tsdf", file_index );

	tsdf.save( filename );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
	writePoligonMeshFile (int format, const pcl::PolygonMesh& mesh, int file_index = -1)
{
	if (format == KinFuLSApp::MESH_PLY)
	{
		char filename[ 1024 ];
		memset( filename, 0, 1024 );
		if ( file_index == -1 ) {
			sprintf( filename, "mesh.ply" );
		} else {
			sprintf( filename, "mesh_%d.ply", file_index );
		}

		cout << "Saving mesh to " << filename << "... " << flush;
		//pcl::io::savePLYFile( filename, mesh );		
		pcl::io::savePLYFileBinary( filename, mesh );		
	}
	else /* if (format == KinFuLSApp::MESH_VTK) */
	{
		cout << "Saving mesh to to 'mesh.vtk'... " << flush;
		pcl::io::saveVTKFile("mesh.vtk", mesh);    
	}  
	cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
	print_cli_help ()
{
	cout << "\nKinFu parameters:" << endl;
	cout << "    --help, -h                          : print this message" << endl;  
	cout << "    --verbose                           : print driver information" << endl;
	cout << "    --registration, -r                  : try to enable registration (source needs to support this)" << endl;
	cout << "    --current-cloud, -cc                : show current frame cloud" << endl;
	cout << "    --save-views, -sv                   : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;  
	cout << "    --registration, -r                  : enable registration mode" << endl; 
	cout << "    --integrate-colors, -ic             : enable color integration mode (allows to get cloud with colors)" << endl;
	cout << "    --extract-textures, -et             : extract RGB PNG images to KinFuSnapshots folder." << endl;
	cout << "    --volume_size <in_meters>, -vs      : define integration volume size" << endl;
	cout << "    --shifting_distance <in_meters>, -sd : define shifting threshold (distance target-point / cube center)" << endl;
	cout << "    --snapshot_rate <X_frames>, -sr     : Extract RGB textures every <X_frames>. Default: 45  " << endl;
	cout << "    --integration_trunc <in_meters>     : truncation distance for integration" << endl;
	cout << "    --record                            : record the stream to .oni file" << endl;
	cout << "    --record_script                     : record playback script file" << endl;
	cout << "    --play_script <script file>         : playback script file" << endl;
	cout << "    --use_rgbdslam <log file>           : use rgbdslam estimation" << endl;
	cout << "    --fragment <X_frames>               : fragments the stream every <X_frames>" << endl;
	cout << "    --fragment_start <X_frames>         : fragments start from <X_frames>" << endl;
	cout << "    --record_log <log_file>             : record transformation log file" << endl;
	cout << "    --graph_registration <graph file>   : register the fragments in the file" << endl;
	cout << "    --schedule <schedule file>          : schedule Kinfu processing from the file" << endl;
	cout << "    --seek_start <X_frames>              : start from X_frames" << endl;
	cout << "    --kinfu_image                       : record kinfu images to image folder" << endl;
	cout << "    --world                             : turn on world.pcd extraction" << endl;
	cout << "    --bbox <bbox file>                  : turn on bbox, used with --rgbdslam" << endl;
	cout << "    --mask <x1,x2,y1,y2>                : trunc the depth image with a window" << endl;
	cout << "    --camera <param_file>               : launch parameters from the file" << endl;
	cout << "    --slac <slac_num>                   : enable slac (0x40 flag in schedule)" << endl;
	cout << "    --kintinuous                        : turn on kintinuous" << endl;	
	cout << "    --rgbd_odometry                     : turn on rgbd odometry (overwrite kintinuous)" << endl;
	cout << "    --bdr_odometry                      : turn on new boundary odometry" << endl;
	cout << "    --kdtree_odometry                   : turn on kdtree odometry (experimental use)" << endl;
	cout << "    --bdr_amplifier <amp>               : default : 4" << endl;	
	cout << "    --shift_x <in_meters>               : initial shift along x axis" << endl;
	cout << "    --shift_y <in_meters>               : initial shift along y axis" << endl;
	cout << "    --shift_z <in_meters>               : initial shift along z axis" << endl;
	cout << endl << "";
	cout << "Valid depth data sources:" << endl; 
	cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
	cout << endl << "";
	cout << " For RGBD benchmark (Requires OpenCV):" << endl; 
	cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl << endl;

	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
	main (int argc, char* argv[])
{  
	if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
		return print_cli_help ();

	int device = 0;
	pc::parse_argument (argc, argv, "-gpu", device);
	pcl::gpu::setDevice (device);
	pcl::gpu::printShortCudaDeviceInfo (device);

	//  if (checkIfPreFermiGPU(device))
	//    return cout << endl << "Kinfu is supported only for Fermi and Kepler arhitectures. It is not even compiled for pre-Fermi by default. Exiting..." << endl, 1;

	boost::shared_ptr<pcl::Grabber> capture;
	bool triggered_capture = false;
	bool pcd_input = false;
	bool use_device = false;

	if (pc::find_switch (argc, argv, "--verbose")) {
		xnLogInitSystem();
		xnLogSetConsoleOutput(TRUE);
		xnLogSetMaskMinSeverity(XN_LOG_MASK_ALL, XN_LOG_VERBOSE);
	}

	std::string eval_folder, match_file, openni_device, oni_file, pcd_dir, script_file, log_file, graph_file, schedule_file, bbox_file, camera_file;
	try
	{    
		if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
		{
			capture.reset (new pcl::OpenNIGrabber (openni_device));
			use_device = true;
		}
		else if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
		{
			triggered_capture = true;
			bool repeat = false; // Only run ONI file once
			capture.reset (new pcl::ONIGrabber (oni_file, repeat, !triggered_capture));
		}
		else if (pc::parse_argument (argc, argv, "-pcd", pcd_dir) > 0)
		{
			float fps_pcd = 15.0f;
			pc::parse_argument (argc, argv, "-pcd_fps", fps_pcd);

			vector<string> pcd_files = getPcdFilesInDir(pcd_dir);    
			// Sort the read files by name
			sort (pcd_files.begin (), pcd_files.end ());
			capture.reset (new pcl::PCDGrabber<pcl::PointXYZRGBA> (pcd_files, fps_pcd, false));
			triggered_capture = true;
			pcd_input = true;
		}
		else if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
		{
			//init data source latter
			pc::parse_argument (argc, argv, "-match_file", match_file);

			//zc:
			pc::parse_argument(argc, argv, "-fps", png_fps_);
			//↓--TUM 默认放大 5(说是5000) 存储便于观察: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
			//但是我一些旧数据是原始 ushort, 故加载旧数据时, 用 -dscale 1
			//pc::parse_argument(argc, argv, "-dscale", )
		}
		else
		{
			capture.reset( new pcl::OpenNIGrabber() );
			use_device = true;

			//capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224932.oni", true, true) );
			//capture.reset( new pcl::ONIGrabber("d:/onis/reg20111229-180846.oni, true, true) );    
			//capture.reset( new pcl::ONIGrabber("/media/Main/onis/20111013-224932.oni", true, true) );
			//capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224551.oni", true, true) );
			//capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224719.oni", true, true) );    
		}
	}
	catch (const pcl::PCLException& /*e*/) { return cout << "Can't open depth source" << endl, -1; }

	float volume_size = pcl::device::VOLUME_SIZE;
	pc::parse_argument (argc, argv, "--volume_size", volume_size);
	pc::parse_argument (argc, argv, "-vs", volume_size);

	float shift_distance = pcl::device::DISTANCE_THRESHOLD;
	pc::parse_argument (argc, argv, "--shifting_distance", shift_distance);
	pc::parse_argument (argc, argv, "-sd", shift_distance);

	int snapshot_rate = pcl::device::SNAPSHOT_RATE; // defined in internal.h
	pc::parse_argument (argc, argv, "--snapshot_rate", snapshot_rate);
	pc::parse_argument (argc, argv, "-sr", snapshot_rate);

	int fragment_rate = 0;
	pc::parse_argument (argc, argv, "--fragment", fragment_rate);

	int fragment_start = 0;
	pc::parse_argument (argc, argv, "--fragment_start", fragment_start);

	float trunc_dist = 2.5f;
	pc::parse_argument ( argc, argv, "--integration_trunc", trunc_dist );

	float shift_x = 0.0f;
	float shift_y = 0.0f;
	float shift_z = 0.0f;
	pc::parse_argument ( argc, argv, "--shift_x", shift_x );
	pc::parse_argument ( argc, argv, "--shift_y", shift_y );
	pc::parse_argument ( argc, argv, "--shift_z", shift_z );

	KinFuLSApp app (*capture, volume_size, shift_distance, snapshot_rate, use_device, fragment_rate, fragment_start, trunc_dist, shift_x, shift_y, shift_z);

	//zc:   @2017-3-23 15:45:57
	double tsdf_trunc_dist = 0.03;
	pc::parse_argument(argc, argv, "-trunc_dist", tsdf_trunc_dist);
	app.kinfu_->volume().setTsdfTruncDist(tsdf_trunc_dist);

	//关于: 对后端 fusion 策略改进
	app.kinfu_->tsdf_version_ = 2.0; //2:= tsdf.kf.orig 原融合算法
	pc::parse_argument(argc, argv, "-tsdfVer", app.kinfu_->tsdf_version_);

	//对应策略 v4-pcTsdf23Cont
	app.kinfu_->tsdfErodeRad_ = 3;
	pc::parse_argument(argc, argv, "-tsdfErodeRad", app.kinfu_->tsdfErodeRad_);

	//zc: <int,int,int>的 vec3, 表示体素坐标, 用于调试 @tsdf23_v8 //2017-2-13 13:55:44
	//app.kinfu_->vxlDbg_ = vector<int>(3, 0);
	app.kinfu_->vxlDbg_ = vector<int>(4, 0); //slice2D 之后, 用 vxlDbg_[3] 存 "012", 表示 fix_axis
	pc::parse_x_arguments(argc, argv, "-vxlDbg", app.kinfu_->vxlDbg_);
	{//初始就绘制小圆球, 做调试观察用 //2017-2-17 10:10:05
        PCL_WARN("vxlDbg_: [%d,%d,%d], slice2D-fix_axis: %d\n", app.kinfu_->vxlDbg_[0], app.kinfu_->vxlDbg_[1], app.kinfu_->vxlDbg_[2], app.kinfu_->vxlDbg_[3]);

		PointXYZ vxl2pt;
		vector<int> &vxlDbg = app.kinfu_->vxlDbg_;
		Vector3f cellsz = app.kinfu_->volume().getVoxelSize();
		vxl2pt.x = (vxlDbg[0] + 0.5f) * cellsz.x();
		vxl2pt.y = (vxlDbg[1] + 0.5f) * cellsz.y();
		vxl2pt.z = (vxlDbg[2] + 0.5f) * cellsz.z();

		app.scene_cloud_view_.cloud_viewer_.addSphere(vxl2pt, cellsz.x()/2, 0,1,0, pt_dbg_init_str); //绿色
		const int fix_axis = vxlDbg[3];
		//再加 四个环绕调试点	@2018-3-3 16:54:52
		int d0[] = {-1, 0, +1, 0},
			d1[] = {0, +1, 0, -1};
		const string pt_nbr_str_stub = "NbrDbgPoint";
		for(int k=01; k<5; k++){ //邻接距离, k>1 则非直接相邻
			for(int i=0; i<4; i++){	 //四邻域
				PointXYZ vxl2pt_nbr = vxl2pt; //待绘制邻接点
				if(0 == fix_axis){
					//vxl2pt_nbr.x = vxl2pt.x;
					vxl2pt_nbr.z += k*d0[i] * cellsz.z(); //固定 X轴时, Z轴做切片面的横轴 (d0)
					vxl2pt_nbr.y += k*d1[i] * cellsz.y();
				}
				if(1 == fix_axis){
					vxl2pt_nbr.x += k*d0[i] * cellsz.x();
					vxl2pt_nbr.z += k*d1[i] * cellsz.z();
				}
				if(2 == fix_axis){
					vxl2pt_nbr.x += k*d0[i] * cellsz.x();
					vxl2pt_nbr.y += k*d1[i] * cellsz.y();
				}

				app.scene_cloud_view_.cloud_viewer_.
					addSphere(vxl2pt_nbr, cellsz.x()/2, 1,0,0, pt_nbr_str_stub + char('0'+(k-1)*4+i)); //红色邻接点
			}
		}

		pcl::ModelCoefficients plane_coeff;
		plane_coeff.values.resize(4, 0); //默认填零
		plane_coeff.values[fix_axis] = 1; //垂直于某坐标轴, 则填1
		//plane_coeff.values[3] = vxlDbg[fix_axis]; //错！
		//plane_coeff.values[3] = -(vxlDbg[fix_axis] + 0.5f ) * cellsz[fix_axis]; //注意负号 //对, 只是长了点
		plane_coeff.values[3] = -vxl2pt.data[fix_axis];
		app.scene_cloud_view_.cloud_viewer_.addPlane(plane_coeff, plane_slice_str);

		printf("vxl2pt.xyz: (%f, %f, %f), plane_coeff: (%f, %f, %f, %f)\n", vxl2pt.x, vxl2pt.y, vxl2pt.z, 
			plane_coeff.values[0], plane_coeff.values[1], plane_coeff.values[2], plane_coeff.values[3]);

	}

	//vxlDbg_ 对应的 px 像素调试点, 外部传参, 暂无法自动求解, 因为 t 时刻求的是 t-1 的 px @2018-3-2 20:43:41
	app.kinfu_->pxDbg_ = vector<int>(2, 0);
	pc::parse_x_arguments(argc, argv, "-pxDbg", app.kinfu_->pxDbg_);

	//控制 kinfu.cpp 中的一些调试窗口开关 //2017-3-3 01:09:38
	//app.kinfu_->dbgKf_ = pc::find_switch(argc, argv, "-dbgKf");
	//bool 改 int: @2017-11-30 10:41:48
	app.kinfu_->dbgKf_ = 0;
	pc::parse_argument(argc, argv, "-dbgKf", app.kinfu_->dbgKf_);

	if ( pc::parse_argument( argc, argv, "--camera", camera_file ) > 0 ) {
		app.toggleCameraParam( camera_file );
	}

	//int seek_start = 0;
	int seek_start = -1;
	if ( pc::parse_argument (argc, argv, "--seek_start", seek_start) ) {
		app.seek_start_ = seek_start;
	}

	if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0){
		app.toggleEvaluationMode(eval_folder, match_file);

		//zc: 放在 -eval 检查之后	@2017-4-19 14:29:49
		if (pc::parse_argument (argc, argv, "-sid", pngSid_) > 0 //start-id
			|| pc::parse_argument (argc, argv, "-eid", pngEid_) > 0) //end-id, 必须与 sid 成对
		{
			if(pngSid_ < 0) pngSid_ = 0;
			if(pngEid_ < 0) pngEid_ = app.evaluation_ptr_->getStreamSize() - 1; //默认范围: [0, size-1]
			printf("--SELECT png id range: [%d, %d]\n", pngSid_, pngEid_);
		}
		else{
			pngSid_ = 0;
			pngEid_ = app.evaluation_ptr_->getStreamSize() - 1;
		}

		pc::parse_argument(argc, argv, "-everyX", everyXframes_);
		pc::parse_argument(argc, argv, "-pauseId", pngPauseId_);

	}

	if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
		app.initCurrentFrameView ();

	if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
		app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  

	if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))  {
		if (pcd_input) {
			app.pcd_source_   = true;
			app.registration_ = true; // since pcd provides registered rgbd
		} else if(pc::find_switch(argc, argv, "-eval")){ //zc: -eval 时, 也直接 reg=true
			app.registration_ = true;
		} else {
			app.initRegistration();
		}
	}

	if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))      
		app.toggleColorIntegration();

	if (pc::find_switch (argc, argv, "--record") )
		app.toggleRecording();

	if (pc::find_switch (argc, argv, "--extract-textures") || pc::find_switch (argc, argv, "-et"))      
		app.enable_texture_extraction_ = true;

	if (triggered_capture) {
		if (pc::find_switch (argc, argv, "--record_script"))
			app.toggleScriptRecord();
		else if (pc::parse_argument (argc, argv, "--play_script", script_file) > 0)
			app.toggleScriptPlay( script_file );

		if (pc::parse_argument (argc, argv, "--use_rgbdslam", log_file) > 0)
			app.toggleRGBDSlam( log_file );

		if (pc::parse_argument (argc, argv, "--graph_registration", graph_file) > 0)
			app.toggleRGBDGraphRegistration( graph_file );

		if (pc::parse_argument (argc, argv, "--schedule", schedule_file) > 0)
			app.toggleSchedule( schedule_file );

		if (pc::parse_argument (argc, argv, "--bbox", bbox_file) > 0)
			app.toggleBBox( bbox_file );

		int slac_num = 0;
		if ( pc::parse_argument ( argc, argv, "--slac", slac_num ) > 0 )
			app.toggleSLAC( slac_num );
	}

	if ( pc::find_switch (argc, argv, "--record_log") ) {
		std::string record_log_file;
		pc::parse_argument( argc, argv, "--record_log", record_log_file );
		app.toggleLogRecord( record_log_file );
	}

	if ( pc::find_switch ( argc, argv, "--world" ) )
		app.kinfu_->toggleExtractWorld();

	if ( pc::find_switch ( argc, argv, "--kinfu_image" ) ) {
		app.toggleKinfuImage();
		if ( oni_file.find( "input.oni" ) != string::npos ) {
			app.kinfu_image_dir_ = oni_file.substr( 0, oni_file.length() - 9 );
		} else {
			app.kinfu_image_dir_ = "image/";
		}
		boost::filesystem::create_directories( app.kinfu_image_dir_ + "kinfu/" );
		boost::filesystem::create_directories( app.kinfu_image_dir_ + "vmap/" );
		boost::filesystem::create_directories( app.kinfu_image_dir_ + "nmap/" );
	}

	if ( pc::find_switch( argc, argv, "--rgbd_odometry" ) )
		app.toggleRGBDOdometry();

	if ( pc::find_switch( argc, argv, "--kintinuous" ) )
		app.toggleKintinuous();

	if ( pc::find_switch( argc, argv, "--bdr_odometry" ) ) {
		app.toggleBdrOdometry();
		float amp = 4.0;
		pc::parse_argument( argc, argv, "--bdr_amplifier", amp );
		app.toggleBdrAmplifier( amp );
	}

	//zc: 长方体三边尺寸做参数, 毫米尺度
	pc::parse_x_arguments(argc, argv, "-cusz", cuSideLenVec_);
	if(cuSideLenVec_.size() == 3){//仅仅 >0 还不够, 必须==3
		PCL_WARN("cube-side-lengths: [%.2f, %.2f, %.2f]\n", cuSideLenVec_[0], cuSideLenVec_[1], cuSideLenVec_[2]);
		
		//转成米:
		for(size_t i=0; i<cuSideLenVec_.size(); i++)
			cuSideLenVec_[i] *= MM2M;

		app.kinfu_->cuSideLenVec_ = cuSideLenVec_;
		app.toggleCuOdometry();

		//zc: --bdr_amplifier 之前只在 -bdr 上生效, 现在 -cusz 也要用 @2017-7-19 10:32:22
		//float amp = 4.0;
		//pc::parse_argument( argc, argv, "--bdr_amplifier", amp );
		//app.toggleBdrAmplifier( amp );

		//zc: 改成 -cuAmp 控制, 双参数, 分别控制 f2mkr, e2c @2017-11-30 14:52:26
		app.kinfu_->w_f2mkr_ = 1.0f;
		app.kinfu_->e2c_weight_ = 4.0f;
		pc::parse_2x_arguments(argc, argv, "-cuAmp", 
			app.kinfu_->w_f2mkr_, app.kinfu_->e2c_weight_);
		PCL_WARN("app.kinfu_->w_f2mkr_, app.kinfu_->e2c_weight_: %f, %f\n", app.kinfu_->w_f2mkr_, app.kinfu_->e2c_weight_);

		//@2017-6-3 18:56:05
		app.kinfu_->with_nmap_ = pc::find_switch(argc, argv, "-nmap");
		app.kinfu_->term_123_ = pc::find_switch(argc, argv, "-123");
		app.kinfu_->term_12_ = pc::find_switch(argc, argv, "-12");
		app.kinfu_->term_23_ = pc::find_switch(argc, argv, "-23");
		app.kinfu_->term_13_ = pc::find_switch(argc, argv, "-13");
		app.kinfu_->term_1_ = pc::find_switch(argc, argv, "-1");
		app.kinfu_->term_2_ = pc::find_switch(argc, argv, "-2");
		app.kinfu_->term_3_ = pc::find_switch(argc, argv, "-3");

		app.kinfu_->e2c_dist_ = 0.05;
		pc::parse_argument(argc, argv, "-e2cDist", app.kinfu_->e2c_dist_);

		//对重建指定平面进行微调, 量纲毫米	@2017-8-14 02:44:48
		app.kinfu_->isPlFilt_ = false;
		app.kinfu_->plFiltShiftMM_ = 0;
		if(pc::parse_argument(argc, argv, "-plfilt", app.kinfu_->plFiltShiftMM_) > 0)
			app.kinfu_->isPlFilt_ = true;
		cout<<"-plfilt: " <<pc::parse_argument (argc, argv, "-plfilt", app.kinfu_->plFiltShiftMM_)<<endl;

	}

	//s2s CPU impl port to GPU @2018-5-24 17:30:18
	vector<float> s2s_params;
	pc::parse_x_arguments(argc, argv, "-s2s", s2s_params);
	if(s2s_params.size() == 3){
		//float delta = s2s_params[0],
		//    eta = s2s_params[1],
		//    beta = s2s_params[2],
		//    bilat_sigma_color = s2s_params[3], //depth diff in mm //暂未用到, 定死默认 (30mm, 4.5px)
		//    bilat_sigma_space = s2s_params[4]; //neighbor pixels
		PCL_WARN("sdf2sdf-odometry\n");

		float eta = s2s_params[0],
			beta = s2s_params[1];

		app.toggleS2sOdometry();
		app.kinfu_->s2s_eta_ = eta;
		app.kinfu_->s2s_beta_ = beta;
		app.kinfu_->s2s_f2m_ = bool(s2s_params[2]);
		cout<<"app.kinfu_->s2s_f2m_: " << app.kinfu_->s2s_f2m_<<endl;

		if(0){ //已验证: volume000_gcoo_ 不是配准好坏因素, 关闭(默认 000) 无妨 @2018-7-14 11:03:33
		Eigen::Affine3f init_pose_orig = app.kinfu_->getCameraPose(0);
		
		app.kinfu_->volume000_gcoo_ = -init_pose_orig.translation();

		Eigen::Vector3f t(0, 0, 0);
		init_pose_orig.translation() = t;
		app.kinfu_->setInitialCameraPose(init_pose_orig);
		}
	}

	app.kinfu_->incidAngleThresh_ = 75; //默认值, 按 pcc 的经验值  @2017-3-8 20:47:22
	pc::parse_argument(argc, argv, "-incidTh", app.kinfu_->incidAngleThresh_);

	if ( pc::find_switch( argc, argv, "--kdtree_odometry" ) ) {
		app.toggleKdtreeOdometry();
	}

	std::string mask;
	if ( pc::parse_argument( argc, argv, "--mask", mask ) > 0 ) {
		app.toggleMask( mask );
	}

	app.use_omask_ = pc::find_switch(argc, argv, "-omask");
	app.use_tmask_ = pc::find_switch(argc, argv, "-tm");

	// executing
	if (triggered_capture) std::cout << "Capture mode: triggered\n";
	else				     std::cout << "Capture mode: stream\n";

	// set verbosity level
	pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
	try { app.startMainLoop (triggered_capture); }  
	catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
	catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
	catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

	//~ #ifdef HAVE_OPENCV
	//~ for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
	//~ {
	//~ if (t == 0)
	//~ {
	//~ cout << "Saving depth map of first view." << endl;
	//~ cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
	//~ cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
	//~ }
	//~ char buf[4096];
	//~ sprintf (buf, "./%06d.png", (int)t);
	//~ cv::imwrite (buf, app.image_view_.views_[t]);
	//~ printf ("writing: %s\n", buf);
	//~ }
	//~ #endif
	std::cout << "pcl_kinfu_largeScale exiting\n";
	return 0;
}
