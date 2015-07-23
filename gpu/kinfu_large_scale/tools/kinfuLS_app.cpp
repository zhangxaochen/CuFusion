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
	double fx_, fy_, cx_, cy_, ICP_trunc_, integration_trunc_;

	CameraParam() : fx_( 525.0 ), fy_( 525.0 ), cx_( 319.5 ), cy_( 239.5 ), ICP_trunc_( 2.5 ), integration_trunc_( 2.5 )  {
	}

	void loadFromFile( std::string filename ) {
		FILE * f = fopen( filename.c_str(), "r" );
		if ( f != NULL ) {
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
		raycaster_ptr_->run(kinfu.volume(), kinfu.getCameraPose(), kinfu.getCyclicalBufferStructure ());
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
			viewerTraj_.showRGBImage ( (unsigned char *) traj.data, traj.cols, traj.rows, "short_image" );
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

		/*
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
		*/
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
		registration_ (false), integrate_colors_ (false), pcd_source_ (false), focal_length_(-1.f), capture_ (source), time_ms_(0), record_script_ (false), play_script_ (false), recording_ (false), use_device_ (useDevice), traj_(cv::Mat::zeros( 480, 640, CV_8UC3 )), traj_buffer_( 480, 640, CV_8UC3, cv::Scalar( 255, 255, 255 )),
		use_rgbdslam_ (false), record_log_ (false), fragment_rate_ (fragmentRate), fragment_start_ (fragmentStart), use_schedule_ (false), use_graph_registration_ (false), frame_id_ (0), use_bbox_ ( false ), seek_start_( -1 ), kinfu_image_ (false), traj_token_ (0), use_mask_ (false),
		kintinuous_( false ), rgbd_odometry_( false ), slac_( false ), bdr_odometry_( false )
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
		transformation_inverse_ = pose.matrix().inverse();

		kinfu_->setInitialCameraPose (pose);
		kinfu_->volume().setTsdfTruncDist (0.030f / 3.0f * volume_size(0)/*meters*/);
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
		evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
		if (!match_file.empty())
			evaluation_ptr_->setMatchFile(match_file);

		kinfu_->setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
		image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols (),
			evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
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
			//if ( frame_id_ == 601 ) {
			//	char * buf;
			//	buf = new char[ 512 * 512 * 512 * sizeof( int ) ];
			//	std::ifstream ifile( "test.bin", std::ifstream::binary );
			//	ifile.read( buf, 512 * 512 * 512 * sizeof( int ) );
			//	ifile.close();
			//	kinfu_->volume().data().upload( buf, kinfu_->volume().data().cols() * sizeof(int), kinfu_->volume().data().rows(), kinfu_->volume().data().cols() );
			//	delete []buf;
			//}

			depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
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
					has_image = kinfu_->bdrodometry( depth_device_, &image_view_.colors_device_ );
					if ( has_image == false && kinfu_->getGlobalTime() > 1 ) {
						ten_has_data_fail_then_we_call_it_a_day_ = 100;
					}
				} else if ( kdtree_odometry_ ) {
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

			image_view_.showDepth (depth_);

			// update traj_
			drawTrajectory();
			image_view_.showTraj (traj_, kinfu_image_ ? frame_id_ : -1, kinfu_image_dir_);
			//image_view_.showGeneratedDepth(kinfu_, kinfu_->getCameraPose());

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
		}

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
			}
			else
				cout << "[!] tsdf volume download is disabled" << endl << endl;
		}

		if (scan_mesh_)
		{
			scan_mesh_ = false;
			scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
		}

		if (has_image)
		{
			Eigen::Affine3f viewer_pose = getViewerPose(scene_cloud_view_.cloud_viewer_);
			image_view_.showScene (*kinfu_, rgb24, registration_, kinfu_image_ ? frame_id_ : -1, independent_camera_ ? &viewer_pose : 0, kinfu_image_dir_);
		}

		if (current_frame_cloud_view_)
			current_frame_cloud_view_->show (*kinfu_);

		if (!independent_camera_)
			setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose());

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
					this->execute (depth_, rgb24_, has_data); 

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

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		writeCloud (int format) const
	{      
		const SceneCloudView& view = scene_cloud_view_;

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

	bool rgbd_odometry_;
	bool kintinuous_;
	cv::Mat grayImage0_, grayImage1_, depthFlt0_, depthFlt1_;

	int ten_has_data_fail_then_we_call_it_a_day_;

	bool slac_;
	bool bdr_odometry_;
	bool kdtree_odometry_;

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

	if ( pc::parse_argument( argc, argv, "--camera", camera_file ) > 0 ) {
		app.toggleCameraParam( camera_file );
	}

	int seek_start = 0;
	if ( pc::parse_argument (argc, argv, "--seek_start", seek_start) ) {
		app.seek_start_ = seek_start;
	}

	if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
		app.toggleEvaluationMode(eval_folder, match_file);

	if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
		app.initCurrentFrameView ();

	if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
		app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  

	if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))  {
		if (pcd_input) {
			app.pcd_source_   = true;
			app.registration_ = true; // since pcd provides registered rgbd
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

	if ( pc::find_switch( argc, argv, "--kdtree_odometry" ) ) {
		app.toggleKdtreeOdometry();
	}

	std::string mask;
	if ( pc::parse_argument( argc, argv, "--mask", mask ) > 0 ) {
		app.toggleMask( mask );
	}

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
