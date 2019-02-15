/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2011, Willow Garage, Inc.
*
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
*/
//#include <sophus/se3.hpp> //with older eigen (from pcl 160), error C2144: syntax error : 'int' should be preceded by ';'

#include <iostream>
#include <algorithm>
#include "HashSparseMatrix.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/time.h>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>

//using namespace cv;

#include <limits>

//zc: @2017-3-22 17:45:39
#include <pcl/console/time.h> //zc: tictoc
//#include "contour_cue_impl.h"

//for g2o   //不能放 @kinfu.h; //拷贝自之前 pcl/gpu/.../kinfu.cpp; @2017-11-13 10:51:02
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

//#include <sophus/se3.hpp> //with older eigen (from pcl 160), error C2144: syntax error : 'int' should be preceded by ';'
#include "sophus_utils.h"

pcl::console::TicToc tt0, tt1, tt2, tt3; //一些备用计时器
static int callCnt_ = 0;

PlaneFitter planeFitter_;

pcl::KdTreeFLANN<pcl::PointXYZ> cuEdgeKdtree_;
bool cuTreeInited_ = false;

//图优化
g2o::SparseOptimizer g2oOptmizer_;

namespace zc{
	//copy from "contour_cue_impl.h" to avoid extra header file	@2019-1-27 17:48:04

	using namespace pcl::device;

	//@brief generate the contour correspondence candidates (a MaskMap repr.) using the "tangency property": "for all points along the contour generator, the normal is orthogonal to the view ray."
	//@param[in] camPos, the camera coords in global reference frame
	//@param[in] vmap, the vertex map in *GLOBAL FRAME*, to compute "view ray" with *camPos*
	//@param[in] nmap, the normal map in *GLOBAL FRAME*
	//@param[in] angleThresh, angle threshold of the "tangency property" in *degree*.
	//@param[out] outMask, contour mask
	void contourCorrespCandidate(const float3 &camPos, const MapArr &vmap, const MapArr &nmap, int angleThresh, MaskMap &outMask);

	//@brief Perform affine transform of vmap (part of @tranformMaps), also can be used on Nmaps.
	//@param[in] vmap_src source vertex map
	//@param[in] Rmat rotation mat
	//@param[in] tvec translation
	//@param[out] vmap_dst destination vertex map
	void transformVmap(const MapArr &vmap_src, const Mat33 &Rmat, const float3 &tvec, MapArr &vmap_dst);

	//@brief 构造 wmap: 原则: weight 正比于 cos/D(u), 最初用于 tsdf-v11.4  @2017-3-13 17:09:42
	//@param[in] nmapLocal 相机坐标系下, 这样就不用传相机坐标 t
	//@param[in] contMask, 本意是 wei 正比于 到边缘距离, 以便削弱边缘翻翘的权重, 这里用边缘蒙板近似
	PCL_EXPORTS void calcWmap(const MapArr &vmapLocal, const MapArr &nmapLocal, const zc::MaskMap &contMask, MapArr &wmap_out);

	//@brief v2, 修改原 calcWmap 参数, 之前 contMask 算作二分类权重 mask, 改成浮点型, 平滑过渡权重 (控制量 edgeDistMap)
	PCL_EXPORTS void calcWmap(const MapArr &vmapLocal, const MapArr &nmapLocal, const DeviceArray2D<float> &edgeDistMap, const float fxy, MapArr &wmap_out);

	//@brief 截取自 calcWmap, 只用 edge, 其他控制量不够稳定; 目的: 为了 tsdf-v18.4 @2018-3-9 22:43:18
	PCL_EXPORTS void edge2wmap(const MapArr &vmapLocal, const DeviceArray2D<float> &edgeDistMap, const float fxy, float maxEdgeDist, MapArr &wmap_out);

}//namespace zc

/**
 * @brief CameraPoseWriter writes all camera poses computed by
 * the KinfuTracker to a file on disk.
 * 拷贝自 @camera_pose.h
 */
class CameraPoseWriter //: public CameraPoseProcessor
{
  std::string output_filename_;
  std::ofstream out_stream_;
  public:
    /**
       * @param output_filename name of file to write
       */
    CameraPoseWriter (std::string output_filename) :
      output_filename_ (output_filename)
    {
      out_stream_.open (output_filename_.c_str () );
    }

    ~CameraPoseWriter ()
    {
      if (out_stream_.is_open ())
      {
        out_stream_.close ();
        std::cout << "wrote camera poses to file " << output_filename_ << std::endl;
      }
    }

    void
    processPose (const Eigen::Affine3f &pose)
    {
      if (out_stream_.good ())
      {
        // convert 3x4 affine transformation to quaternion and write to file
        Eigen::Quaternionf q (pose.rotation ());
        Eigen::Vector3f t (pose.translation ());
        // write translation , quaternion in a row
        out_stream_ << t[0] << "," << t[1] << "," << t[2]
                    << "," << q.w () << "," << q.x ()
                    << "," << q.y ()<< ","  << q.z () << std::endl;
      }
    }

};//CameraPoseWriter

//////////////////////////////
CloudType::Ptr cvMat2PointCloud(const cv::Mat &dmat, const pcl::device::Intr &intr){
	CV_Assert(dmat.type() == CV_16UC1);

	int imWidth = dmat.cols,
		imHeight = dmat.rows;

	//待返回的点云:
	CloudType::Ptr pCloud(new CloudType);
	pCloud->points.reserve(imWidth * imHeight); //注意: 非 resize, 后面必须用 push_back, 而非 "[]"

	float fx_inv = 1 / intr.fx,
		fy_inv = 1 / intr.fy;
	const float mm2m = 0.001;
	const float qnan = numeric_limits<float>::quiet_NaN ();

	ushort *pDat = (ushort*)dmat.data;
	for(int i = 0; i < imHeight; i++){
		for(int j = 0; j < imWidth; j++){
			PtType pt;
			ushort z = *pDat;
			if(0 == z){ //零深度值在 cloud 中应算作 nan! 否则语义错误!
				pt.x = pt.y = pt.z = qnan;
			}
			else{
				pt.z = z * mm2m; //转换到米(m)尺度
				pt.x = pt.z * (j - intr.cx) * fx_inv;
				pt.y = pt.z * (i - intr.cy) * fy_inv;
			}
			pCloud->push_back(pt);

			++pDat;
		}
	}

	pCloud->width = imWidth; //表示有序点云, 必须放在最后, 因为 push_back 会冲毁设定
	pCloud->height = imHeight;

	return pCloud;
}//cvMat2PointCloud

cv::Mat cloud2cvmat(const CloudType::Ptr pCloud, const pcl::device::Intr intr){
	const int ww = pCloud->width, 
			  hh = pCloud->height;
	printf("@cloud2cvmat, (w,h):= (%d, %d)\n", ww, hh);
	
	cv::Mat res(hh, ww, CV_16UC1);

	const float M2MM = 1e3;
	const float qnan = numeric_limits<float>::quiet_NaN ();

	for(int i = 0; i < hh; i++){
		for(int j = 0; j < ww; j++){
			PtType pt = pCloud->at(j, i);
			if(!_isnan(pt.x)){
				//int u = pt.x / pt.z * intr.fx + intr.cx; //不必
				//int v = pt.y / pt.z * intr.fy + intr.cy;
				res.at<ushort>(i, j) = pt.z * M2MM;
			}
			else{
				res.at<ushort>(i, j) = 0;
			}
		}
	}

	return res;
}//cloud2cvmat

void dbgAhcPeac_kinfu( const RGBDImage *rgbdObj, PlaneFitter *pf){
	cv::Mat dbgSegMat(rgbdObj->height(), rgbdObj->width(), CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac_kinfu";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac_kinfu

//////////////////////////////
// RGBDOdometry part
//////////////////////////////

inline static
	void computeC_RigidBodyMotion( double* C, double dIdx, double dIdy, const cv::Point3f& p3d, double fx, double fy )
{
	double invz  = 1. / p3d.z,
		v0 = dIdx * fx * invz,
		v1 = dIdy * fy * invz,
		v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

	C[0] = -p3d.z * v1 + p3d.y * v2;
	C[1] =  p3d.z * v0 - p3d.x * v2;
	C[2] = -p3d.y * v0 + p3d.x * v1;
	C[3] = v0;
	C[4] = v1;
	C[5] = v2;
}

inline static
	void computeProjectiveMatrix( const cv::Mat& ksi, cv::Mat& Rt )
{
	CV_Assert( ksi.size() == cv::Size(1,6) && ksi.type() == CV_64FC1 );

	// for infinitesimal transformation
	Rt = cv::Mat::eye(4, 4, CV_64FC1);

	cv::Mat R = Rt(cv::Rect(0,0,3,3));
	cv::Mat rvec = ksi.rowRange(0,3);

	Rodrigues( rvec, R );

	Rt.at<double>(0,3) = ksi.at<double>(3);
	Rt.at<double>(1,3) = ksi.at<double>(4);
	Rt.at<double>(2,3) = ksi.at<double>(5);
}

static
	void cvtDepth2Cloud( const cv::Mat& depth, cv::Mat& cloud, const cv::Mat& cameraMatrix )
{
	//CV_Assert( cameraMatrix.type() == CV_64FC1 );
	const double inv_fx = 1.f/cameraMatrix.at<double>(0,0);
	const double inv_fy = 1.f/cameraMatrix.at<double>(1,1);
	const double ox = cameraMatrix.at<double>(0,2);
	const double oy = cameraMatrix.at<double>(1,2);
	cloud.create( depth.size(), CV_32FC3 );
	for( int y = 0; y < cloud.rows; y++ )
	{
		cv::Point3f* cloud_ptr = reinterpret_cast<cv::Point3f*>(cloud.ptr(y));
		const float* depth_prt = reinterpret_cast<const float*>(depth.ptr(y));
		for( int x = 0; x < cloud.cols; x++ )
		{
			float z = depth_prt[x];
			cloud_ptr[x].x = (float)((x - ox) * z * inv_fx);
			cloud_ptr[x].y = (float)((y - oy) * z * inv_fy);
			cloud_ptr[x].z = z;
		}
	}
}

static inline
	void set2shorts( int& dst, int short_v1, int short_v2 )
{
	unsigned short* ptr = reinterpret_cast<unsigned short*>(&dst);
	ptr[0] = static_cast<unsigned short>(short_v1);
	ptr[1] = static_cast<unsigned short>(short_v2);
}

static inline
	void get2shorts( int src, int& short_v1, int& short_v2 )
{
	typedef union { int vint32; unsigned short vuint16[2]; } s32tou16;
	const unsigned short* ptr = (reinterpret_cast<s32tou16*>(&src))->vuint16;
	short_v1 = ptr[0];
	short_v2 = ptr[1];
}

static
	int computeCorresp( const cv::Mat& K, const cv::Mat& K_inv, const cv::Mat& Rt,
	const cv::Mat& depth0, const cv::Mat& depth1, const cv::Mat& texturedMask1, float maxDepthDiff,
	cv::Mat& corresps )
{
	CV_Assert( K.type() == CV_64FC1 );
	CV_Assert( K_inv.type() == CV_64FC1 );
	CV_Assert( Rt.type() == CV_64FC1 );

	corresps.create( depth1.size(), CV_32SC1 );

	cv::Mat R = Rt(cv::Rect(0,0,3,3)).clone();

	cv::Mat KRK_inv = K * R * K_inv;
	const double * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());

	cv::Mat Kt = Rt(cv::Rect(3,0,1,3)).clone();
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

	cv::Rect r(0, 0, depth1.cols, depth1.rows);

	corresps = cv::Scalar(-1);
	int correspCount = 0;
	for( int v1 = 0; v1 < depth1.rows; v1++ )
	{
		for( int u1 = 0; u1 < depth1.cols; u1++ )
		{
			float d1 = depth1.at<float>(v1,u1);
			if( !cvIsNaN(d1) && texturedMask1.at<uchar>(v1,u1) )
			{
				float transformed_d1 = (float)(d1 * (KRK_inv_ptr[6] * u1 + KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);
				int u0 = cvRound((d1 * (KRK_inv_ptr[0] * u1 + KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1);
				int v0 = cvRound((d1 * (KRK_inv_ptr[3] * u1 + KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1);

				if( r.contains(cv::Point(u0,v0)) )
				{
					float d0 = depth0.at<float>(v0,u0);
					if( !cvIsNaN(d0) && std::abs(transformed_d1 - d0) <= maxDepthDiff )
					{
						int c = corresps.at<int>(v0,u0);
						if( c != -1 )
						{
							int exist_u1, exist_v1;
							get2shorts( c, exist_u1, exist_v1);

							float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) * (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] * exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);

							if( transformed_d1 > exist_d1 )
								continue;
						}
						else
							correspCount++;

						set2shorts( corresps.at<int>(v0,u0), u1, v1 );
					}
				}
			}
		}
	}

	return correspCount;
}

static inline
	void preprocessDepth( cv::Mat depth0, cv::Mat depth1,
	const cv::Mat& validMask0, const cv::Mat& validMask1,
	float minDepth, float maxDepth )
{
	CV_DbgAssert( depth0.size() == depth1.size() );

	for( int y = 0; y < depth0.rows; y++ )
	{
		for( int x = 0; x < depth0.cols; x++ )
		{
			float& d0 = depth0.at<float>(y,x);
			if( !cvIsNaN(d0) && (d0 > maxDepth || d0 < minDepth || d0 <= 0 || (!validMask0.empty() && !validMask0.at<uchar>(y,x))) )
				d0 = std::numeric_limits<float>::quiet_NaN();

			float& d1 = depth1.at<float>(y,x);
			if( !cvIsNaN(d1) && (d1 > maxDepth || d1 < minDepth || d1 <= 0 || (!validMask1.empty() && !validMask1.at<uchar>(y,x))) )
				d1 = std::numeric_limits<float>::quiet_NaN();
		}
	}
}

static
	void buildPyramids( const cv::Mat& image0, const cv::Mat& image1,
	const cv::Mat& depth0, const cv::Mat& depth1,
	const cv::Mat& cameraMatrix, int sobelSize, double sobelScale,
	const vector<float>& minGradMagnitudes,
	vector<cv::Mat>& pyramidImage0, vector<cv::Mat>& pyramidDepth0,
	vector<cv::Mat>& pyramidImage1, vector<cv::Mat>& pyramidDepth1,
	vector<cv::Mat>& pyramid_dI_dx1, vector<cv::Mat>& pyramid_dI_dy1,
	vector<cv::Mat>& pyramidTexturedMask1, vector<cv::Mat>& pyramidCameraMatrix )
{
	const int pyramidMaxLevel = (int)minGradMagnitudes.size() - 1;

	buildPyramid( image0, pyramidImage0, pyramidMaxLevel );
	buildPyramid( image1, pyramidImage1, pyramidMaxLevel );

	pyramid_dI_dx1.resize( pyramidImage1.size() );
	pyramid_dI_dy1.resize( pyramidImage1.size() );
	pyramidTexturedMask1.resize( pyramidImage1.size() );

	pyramidCameraMatrix.reserve( pyramidImage1.size() );

	cv::Mat cameraMatrix_dbl;
	cameraMatrix.convertTo( cameraMatrix_dbl, CV_64FC1 );

	for( size_t i = 0; i < pyramidImage1.size(); i++ )
	{
		Sobel( pyramidImage1[i], pyramid_dI_dx1[i], CV_16S, 1, 0, sobelSize );
		Sobel( pyramidImage1[i], pyramid_dI_dy1[i], CV_16S, 0, 1, sobelSize );

		const cv::Mat& dx = pyramid_dI_dx1[i];
		const cv::Mat& dy = pyramid_dI_dy1[i];

		cv::Mat texturedMask( dx.size(), CV_8UC1, cv::Scalar(0) );
		const float minScalesGradMagnitude2 = (float)((minGradMagnitudes[i] * minGradMagnitudes[i]) / (sobelScale * sobelScale));
		for( int y = 0; y < dx.rows; y++ )
		{
			for( int x = 0; x < dx.cols; x++ )
			{
				float m2 = (float)(dx.at<short>(y,x)*dx.at<short>(y,x) + dy.at<short>(y,x)*dy.at<short>(y,x));
				if( m2 >= minScalesGradMagnitude2 )
					texturedMask.at<uchar>(y,x) = 255;
			}
		}
		pyramidTexturedMask1[i] = texturedMask;
		cv::Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
		levelCameraMatrix.at<double>(2,2) = 1.;
		pyramidCameraMatrix.push_back( levelCameraMatrix );
	}

	buildPyramid( depth0, pyramidDepth0, pyramidMaxLevel );
	buildPyramid( depth1, pyramidDepth1, pyramidMaxLevel );
}

static
	bool solveSystem( const cv::Mat& C, const cv::Mat& dI_dt, double detThreshold, cv::Mat& ksi, Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	cv::Mat A = C.t() * C;
	cv::Mat B = -C.t() * dI_dt;

	cv2eigen( A, AA );
	cv2eigen( B, bb );

	double det = cv::determinant(A);

	if( fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det) )
		return false;

	cv::solve( A, B, ksi, cv::DECOMP_CHOLESKY );
	return true;
}

typedef void (*ComputeCFuncPtr)( double* C, double dIdx, double dIdy, const cv::Point3f& p3d, double fx, double fy );

static
	bool computeKsi( int transformType,
	const cv::Mat& image0, const cv::Mat&  cloud0,
	const cv::Mat& image1, const cv::Mat& dI_dx1, const cv::Mat& dI_dy1,
	const cv::Mat& corresps, int correspsCount,
	double fx, double fy, double sobelScale, double determinantThreshold,
	cv::Mat& ksi,
	Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	int Cwidth = -1;
	ComputeCFuncPtr computeCFuncPtr = 0;
	computeCFuncPtr = computeC_RigidBodyMotion;
	Cwidth = 6;
	cv::Mat C( correspsCount, Cwidth, CV_64FC1 );
	cv::Mat dI_dt( correspsCount, 1, CV_64FC1 );

	double sigma = 0;
	int pointCount = 0;
	for( int v0 = 0; v0 < corresps.rows; v0++ )
	{
		for( int u0 = 0; u0 < corresps.cols; u0++ )
		{
			if( corresps.at<int>(v0,u0) != -1 )
			{
				int u1, v1;
				get2shorts( corresps.at<int>(v0,u0), u1, v1 );
				double diff = static_cast<double>(image1.at<uchar>(v1,u1)) -
					static_cast<double>(image0.at<uchar>(v0,u0));
				sigma += diff * diff;
				pointCount++;
			}
		}
	}
	sigma = std::sqrt(sigma/pointCount);

	pointCount = 0;
	for( int v0 = 0; v0 < corresps.rows; v0++ )
	{
		for( int u0 = 0; u0 < corresps.cols; u0++ )
		{
			if( corresps.at<int>(v0,u0) != -1 )
			{
				int u1, v1;
				get2shorts( corresps.at<int>(v0,u0), u1, v1 );

				double diff = static_cast<double>(image1.at<uchar>(v1,u1)) -
					static_cast<double>(image0.at<uchar>(v0,u0));
				double w = sigma + std::abs(diff);
				w = w > DBL_EPSILON ? 1./w : 1.;

				(*computeCFuncPtr)( (double*)C.ptr(pointCount),
					w * sobelScale * dI_dx1.at<short int>(v1,u1),
					w * sobelScale * dI_dy1.at<short int>(v1,u1),
					cloud0.at<cv::Point3f>(v0,u0), fx, fy);

				dI_dt.at<double>(pointCount) = w * diff;
				pointCount++;
			}
		}
	}

	cv::Mat sln;
	bool solutionExist = solveSystem( C, dI_dt, determinantThreshold, sln, AA, bb );

	/*
	cout << C << endl;
	cout << dI_dt << endl;
	cout << sln << endl;

	cout << endl;
	*/

	if( solutionExist )
	{
		ksi.create(6,1,CV_64FC1);
		ksi = cv::Scalar(0);

		cv::Mat subksi;
		subksi = ksi;
		sln.copyTo( subksi );
	}

	return solutionExist;
}

//////////////////////////////
// RGBDOdometry part end
//////////////////////////////

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
	namespace gpu
	{
		Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker (const Eigen::Vector3f &volume_size, const float shiftingDistance, int rows, int cols)
	: cyclical_( DISTANCE_THRESHOLD, pcl::device::VOLUME_SIZE, VOLUME_X), cyclical2_( DISTANCE_THRESHOLD, pcl::device::VOLUME_SIZE, VOLUME_X)
	, rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), perform_last_scan_ (false), finished_(false), force_shift_(false), max_integrate_distance_(0)
	, extract_world_( false ), use_slac_( false ), dist_thresh_( 0.03 ), amplifier_( 4.0f )
	, disable_icp_(false) //默认不 disable
{
	//const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
	const Vector3i volume_resolution (VOLUME_X, VOLUME_Y, VOLUME_Z);

	volume_size_ = volume_size(0);

	tsdf_volume_ = TsdfVolume::Ptr ( new TsdfVolume(volume_resolution) );
	tsdf_volume_->setSize (volume_size);

	shifting_distance_ = shiftingDistance;

	// set cyclical buffer values
	cyclical_.setDistanceThreshold (shifting_distance_);
	cyclical_.setVolumeSize (volume_size(0), volume_size(1), volume_size(2));

	cyclical2_.setDistanceThreshold (shifting_distance_);
	cyclical2_.setVolumeSize (volume_size(0), volume_size(1), volume_size(2));

	setDepthIntrinsics (525.f, 525.f); // default values, can be overwritten

	init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
	//init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
	init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 + 0.3);

	Eigen::Affine3f init_pose;
	init_pose.linear() = init_Rcam_;
	init_pose.translation() = init_tcam_;
	//grid_.Init( 12, volume_size_, init_pose.matrix() );
	Eigen::Matrix4f grid_init_pose;
	grid_init_pose.setIdentity();
	grid_init_pose( 0, 3 ) = 1.5;
	grid_init_pose( 1, 3 ) = 1.5;
	grid_init_pose( 2, 3 ) = -0.3;
	grid_.Init( 12, 3, grid_init_pose );

	//const int iters[] = {10, 5, 4};
	//const int iters[] = {7, 4, 3}; //icp3
	//const int iters[] = {3, 4, 3}; //icp4.4
	//const int iters[] = {2, 1, 1}; //icp4.4.2 -BAD
	const int iters[] = {1, 4, 3}; //icp4.4.3 -GOOD;
	std::copy (iters, iters + LEVELS, icp_iterations_);

	const float default_distThres = 0.10f; //meters
	const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
	const float default_tranc_dist = 0.03f / 3.0f * volume_size(0); //meters

	setIcpCorespFilteringParams (default_distThres, default_angleThres);
	tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

	allocateBufffers (rows, cols);

	rmats_.reserve (30000);
	tvecs_.reserve (30000);

	reset ();

	// initialize cyclical buffer
	//cyclical_.initBuffer(tsdf_volume_, color_volume_);

	//zc:
	planeFitter_.minSupport = 000;
	isFirstFoundCube_ = true;
	isFoundCu4pts_ = false;
	isCuInitialized_ = false;
	cuContCloudNormal_ = PointCloud<Normal>::Ptr(new PointCloud<Normal>());

	genSynData_ = false;
	sample_step_ = 1;
	//图优化控制量
	isLastFoundCrnr_ = false;

	//zc: 图优化配置预设
	// 使用Cholmod中的线性方程求解器
	g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
	g2o::BlockSolver_6_3 *block_solver = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg *algo = new g2o::OptimizationAlgorithmLevenberg(block_solver);
	g2oOptmizer_.setAlgorithm(algo);
	g2oOptmizer_.setVerbose(true);

	//s2s-init
	s2s_beta_ = 0.5f; //inc step
	s2s_eta_ = 0.01f; //10mm
	s2s_f2m_ = true;

	dbgFid_ = -1;
	dbgLvlIdx_ = -1;
	dbgIter_ = -1;
	dbgV2_18_ = -1;

	is_integrate_ = true;
}//KinfuTracker-ctor

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
	fx_ = fx;
	fy_ = fy;
	cx_ = (cx == -1) ? cols_/2-0.5f : cx;
	cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setInitialCameraPose (const Eigen::Affine3f& pose)
{
	printf("setInitialCameraPose>>>>>>>>>>>>>>>>>>>>>\n");
	//init_Rcam_ = pose.rotation (); //eigen334 .rotation 值错误, 改用 .linear
	init_Rcam_ = pose.linear ();
// 	cout<<"pose:\n"<<pose.matrix()<<endl
// 		<<"pose.linear:\n"<<pose.linear()<<endl
// 		<<"pose.rotation:\n"<<pose.rotation()<<endl
// 		<<"init_Rcam_:\n"<<init_Rcam_<<endl;
	init_tcam_ = pose.translation ();
	init_rev_ = pose.matrix().inverse();
	init_trans_ = Eigen::Matrix4f::Identity();
	reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
	max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setDepthTruncationForIntegrate (float max_integrate_distance)
{
	max_integrate_distance_ = max_integrate_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
	integration_metric_threshold_ = threshold;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
	distThres_  = distThreshold; //mm
	angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
	pcl::gpu::KinfuTracker::cols ()
{
	return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
	pcl::gpu::KinfuTracker::rows ()
{
	return (rows_);
}

void
	pcl::gpu::KinfuTracker::extractAndMeshWorld ()
{
	finished_ = true;
	int cloud_size = 0;
	cloud_size = cyclical_.getWorldModel ()->getWorld ()->points.size();

	if (cloud_size <= 0)
	{
		PCL_WARN ("World model currently has no points. Skipping save procedure.\n");
		return;
	}
	else
	{
		PCL_INFO ("Saving current world to world.pcd with %d points.\n", cloud_size);
		pcl::io::savePCDFile<pcl::PointXYZI> ("world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
		return;
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::reset ()
{
	if (global_time_) {
		PCL_WARN ("Reset\n");
		// update init_trans_ to transform current coordinate to global_time_0 coordinate
		init_trans_ = init_trans_ * getCameraPose().matrix() * init_rev_;
	}

	// dump current world to a pcd file
	/*
	if (global_time_)
	{
	PCL_INFO ("Saving current world to current_world.pcd\n");
	pcl::io::savePCDFile<pcl::PointXYZI> ("current_world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
	// clear world model
	cyclical_.getWorldModel ()->reset ();
	}
	*/

	global_time_ = 0;
	rmats_.clear ();
	tvecs_.clear ();

	rmats_.push_back (init_Rcam_);
	tvecs_.push_back (init_tcam_);
	cout<<"\tinit_Rcam_@reset():\n"<<init_Rcam_<<endl
		<<"\tinit_tcam_:\n"<<init_tcam_.transpose()<<endl;

	tsdf_volume_->reset ();

	if (color_volume_) // color integration mode is enabled
		color_volume_->reset ();    

	if (tsdf_volume2_)
		tsdf_volume2_->reset();

	// reset cyclical buffer as well
	cyclical_.resetBuffer (tsdf_volume_, color_volume_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::allocateBufffers (int rows, int cols)
{    
	depths_curr_.resize (LEVELS);
	vmaps_g_curr_.resize (LEVELS);
	nmaps_g_curr_.resize (LEVELS);

	vmaps_g_prev_.resize (LEVELS);
	nmaps_g_prev_.resize (LEVELS);

	vmaps_curr_.resize (LEVELS);
	nmaps_curr_.resize (LEVELS);

	//zc:
	vmaps_cu_g_prev_.resize(LEVELS); //立方体 v/nmap-vec 初始化, 但不必 .create
	nmaps_cu_g_prev_.resize(LEVELS);
	edgeDistMap_device_.create(rows, cols);
	dmapModel_.create(rows, cols);
	diffDmap_.create(rows, cols);

	vmap_g_model_.create(rows * 3, cols);
	nmap_g_model_.create(rows * 3, cols);

	vmap_g_model_vol2_.create(rows * 3, cols);
	nmap_g_model_vol2_.create(rows * 3, cols);

	coresps_.resize (LEVELS);

	for (int i = 0; i < LEVELS; ++i)
	{
		int pyr_rows = rows >> i;
		int pyr_cols = cols >> i;

		depths_curr_[i].create (pyr_rows, pyr_cols);

		vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
		nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

		vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
		nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

		vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
		nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

		coresps_[i].create (pyr_rows, pyr_cols);
	}  
	depthRawScaled_.create (rows, cols);
	depthRawScaled_prev_.create (rows, cols);

	// see estimate tranform for the magic numbers
	//gbuf_.create (27, 20*60);
	gbuf_.create( 27, 640*480 );
	sumbuf_.create (27);

	//zc:
// 	gbuf_f2mkr_.create( 27, 640*480 ); //没用
// 	sumbuf_f2mkr_.create (27);
	gbuf_s2s_.create( 27, 640*480 );
	sumbuf_s2s_.create (27);
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::operator() (const DepthMap& depth_raw, const View * pcolor, FramedTransformation * frame_ptr)
{
	//ScopeTime time( "Kinfu Tracker All" );
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->ResetFlag ) ) {
		reset();
		if ( frame_ptr->type_ == frame_ptr->DirectApply )
		{
			Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
			rmats_[0] = aff_rgbd.linear();
			tvecs_[0] = aff_rgbd.translation();
		}
	}

	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) ) {
	}
	else
	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr_[0]);
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		/*
		int c;
		vector<unsigned short> data;
		depths_curr_[ 0 ].download(data, c);
		char filename[ 1024 ];
		std::sprintf( filename, "bf/%06d.png", global_time_ + 1 );
		cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
		cv::imwrite( filename, m );
		*/

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
		pcl::device::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		/*
		// save nmaps_curr_[ 0 ];
		std::vector< float > temp;
		int ttemp;
		temp.resize( 640 * 480 * 3 );
		//nmaps_curr_[ 0 ].download( &( temp[ 0 ] ), nmaps_curr_[ 0 ].cols() * sizeof( float ) );
		nmaps_curr_[ 0 ].download( temp, ttemp );

		FILE * f = fopen( "nmap0.txt", "w" );
		for ( int i = 0; i < 480 * 640 * 3; i++ ) {
		if ( temp[ i ] >= 0 && temp[ i ] <= 1 )
		fprintf( f, "%.6f\n", temp[ i ] );
		else
		fprintf( f, "%.6f\n", 0 );
		}
		fclose( f );

		//vmaps_curr_[ 0 ].download( &( temp[ 0 ] ), vmaps_curr_[ 0 ].cols() * sizeof( float ) );
		vmaps_curr_[ 0 ].download( temp, ttemp );
		f = fopen( "vmap0.txt", "w" );
		for ( int i = 0; i < 480 * 640 * 3; i++ ) {
		if ( temp[ i ] >= 0 && temp[ i ] <= 1 )
		fprintf( f, "%.6f\n", temp[ i ] );
		else
		fprintf( f, "%.6f\n", 0 );
		}
		fclose( f );
		*/

		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		//device::integrateTsdfVolume(depths_curr_[ 0 ], intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		/*
		Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
		float3& device_tcam = device_cast<float3>(init_tcam);

		Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
		Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize ());

		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);    
		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		*/

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);


		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point


	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->InitializeOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	} else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->IncrementalOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ * getCameraPose().matrix() );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	}


	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		//device::sync();
		//raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
		//device::sync();
		//raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
		//device::sync();
		//raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
		//device::sync();
		//raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
		//device::sync();
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
		//device::sync();
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		//Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		Matrix3frm R_iden;
		R_iden = Eigen::Matrix3f::Identity ();
		Mat33&  rotation_id = device_cast<Mat33> (R_iden); /// Identity Rotation Matrix
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}
	/*

	Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
	Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

	//Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
	Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
	float3& device_tprev     = device_cast<float3> (tprev);

	Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
	Vector3f   tcurr = tprev;
	*/
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) )
	{
		rmats_.push_back (cam_rot_global_prev); 
		tvecs_.push_back (cam_trans_global_prev);
	}
	else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->DirectApply ) )
	{
		//Eigen::Affine3f aff_rgbd( getCameraPose( 0 ).matrix() * frame_ptr->transformation_ );
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_curr = aff_rgbd.linear();
		cam_trans_global_curr = aff_rgbd.translation();
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}
	else
	{
		//ScopeTime time(">>> icp-all");
		for (int level_index = LEVELS-1; level_index>=0; --level_index)
		{
			int iter_num = icp_iterations_[level_index];

			// current maps
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];   

			// previous maps
			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

			// We need to transform the maps from global to the local coordinates
			//Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
			Matrix3frm R_iden;
			R_iden = Eigen::Matrix3f::Identity ();
			Mat33&  rotation_id = device_cast<Mat33> (R_iden); /// Identity Rotation Matrix
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			MapArr& vmap_temp = vmap_g_prev;
			MapArr& nmap_temp = nmap_g_prev;
			device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 

			/*
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];

			//MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
			//MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];
			*/
			//CorespMap& coresp = coresps_[level_index];

			for (int iter = 0; iter < iter_num; ++iter)
			{
				/*
				Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
				float3& device_tcurr = device_cast<float3>(tcurr);
				*/
				//CONVERT TO DEVICE TYPES
				// CURRENT LOCAL TRANSFORM
				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				estimateCombined (device_cam_rot_local_curr, device_cam_trans_local_curr, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr (level_index), 
					vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());
				/*
				estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
				vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
				*/

				/*
				if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->DirectApply || ( frame_ptr->type_ == frame_ptr->InitializeOnly && level_index == LEVELS - 1 && iter == 0 ) ) ) {
				Eigen::Matrix<double, 6, 1> b_rgbd;
				Eigen::Matrix4f trans_rgbd = getCameraPose( 0 ).matrix() * frame_ptr->transformation_;		// <--- global should be like this
				Eigen::Affine3f aff_last;
				aff_last.linear() = cam_rot_global_curr;
				aff_last.translation() = cam_trans_global_curr;
				Eigen::Matrix4f trans_shift = trans_rgbd * aff_last.matrix().inverse();
				// hack
				b_rgbd( 0, 0 ) = - ( trans_shift( 1, 2 ) - trans_shift( 2, 1 ) ) / 2.0;		// alpha is beta in the paper
				b_rgbd( 1, 0 ) = - ( trans_shift( 2, 0 ) - trans_shift( 0, 2 ) ) / 2.0;		// beta is gamma in the paper
				b_rgbd( 2, 0 ) = - ( trans_shift( 0, 1 ) - trans_shift( 1, 0 ) ) / 2.0;		// gamma is alpha in the paper
				b_rgbd( 3, 0 ) = trans_shift( 0, 3 );
				b_rgbd( 4, 0 ) = trans_shift( 1, 3 );
				b_rgbd( 5, 0 ) = trans_shift( 2, 3 );

				A = 10000.0 * Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Identity();
				b = 10000.0 * b_rgbd;
				}
				*/

				//checking nullspace
				double det = A_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					if ( frame_ptr != NULL && frame_ptr->type_ == frame_ptr->InitializeOnly ) {
						Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
						cam_rot_global_curr = aff_rgbd.linear();
						cam_trans_global_curr = aff_rgbd.translation();
						break;
					} else {
						//cam_rot_global_curr = cam_rot_global_prev;
						//cam_trans_global_curr = cam_trans_global_prev;
						//break;
						//reset ();
						return (false);
					}
				}
				//float maxc = A.maxCoeff();

				Eigen::Matrix<float, 6, 1> result = A_.llt ().solve (b_).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

				/*
				PCL_INFO ("At ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
				cout << "Determinant : " << det << endl;
				cout << "Singular matrix :" << endl << A_ << endl;
				cout << "Corresponding b :" << endl << b_ << endl;
				cout << "result " << endl << result << endl;
				*/

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				/*
				if ( global_time_ == 960 ) {
				PCL_ERROR ("@%d frame.%d level.%d iteration, old matrices are\n", global_time_, level_index, iter);
				Eigen::Affine3f xx;
				xx.linear() = cam_rot_global_curr;
				xx.translation() = cam_trans_global_curr - Eigen::Vector3f( (getCyclicalBufferStructure ())->origin_metric.x, (getCyclicalBufferStructure ())->origin_metric.y, (getCyclicalBufferStructure ())->origin_metric.z );
				cout << "Matrix is : " << endl << xx.matrix() << endl;
				}
				*/

				//compose
				cam_trans_global_curr = cam_rot_incremental * cam_trans_global_curr + cam_trans_incremental;
				cam_rot_global_curr = cam_rot_incremental * cam_rot_global_curr;

				/*
				if ( global_time_ == 960 ) {
				PCL_ERROR ("@%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
				cout << "Determinant : " << det << endl;
				cout << "Singular matrix :" << endl << A_ << endl;
				cout << "Corresponding b :" << endl << b_ << endl;
				Eigen::Affine3f xx;
				xx.linear() = cam_rot_global_curr;
				xx.translation() = cam_trans_global_curr - Eigen::Vector3f( (getCyclicalBufferStructure ())->origin_metric.x, (getCyclicalBufferStructure ())->origin_metric.y, (getCyclicalBufferStructure ())->origin_metric.z );
				cout << "Matrix is : " << endl << xx.matrix() << endl;
				}
				*/
			}
		}

		//cout << "Singular matrix :" << endl << A_ << endl;
		//cout << "Corresponding b :" << endl << b_ << endl;

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->ExtractSLACMatrix ) ) {
		//ScopeTime time(">>> SLAC");

		//cout << A_ << endl;
		//cout << b_ << endl;

		if ( frame_ptr->frame_ > ( int )slac_trans_mats_.size() ) {
			slac_trans_mats_.push_back( getCameraPose().matrix() );
		}

		Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations  
		float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
		float3 device_cam_trans_local_curr; 
		device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
		device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
		device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

		Matrix3frm cam_rot_global_curr_t = cam_rot_global_curr.transpose();
		Mat33&  device_cam_rot_local_curr_t = device_cast<Mat33> (cam_rot_global_curr_t);

		//cout << A_ << endl;
		//estimateCombined (device_cam_rot_local_curr, device_cam_trans_local_curr, vmaps_curr_[ 0 ], nmaps_curr_[ 0 ], device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr(0), 
		//                  vmaps_g_prev_[ 0 ], nmaps_g_prev_[ 0 ], distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());
		//cout << A_ << endl;

		//cout << slac_A_.block< 10, 10 >( 0, 0 ) << endl;
		device::createVMap (intr(0), depth_raw, vmaps_curr_[ 0 ]);

		estimateCombinedEx (device_cam_rot_local_curr, device_cam_rot_local_curr_t, device_cam_trans_local_curr, vmaps_curr_[ 0 ], nmaps_curr_[ 0 ], device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr(0), 
			vmaps_g_prev_[ 0 ], nmaps_g_prev_[ 0 ], distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data (), 
			gbuf_slac_triangle_, gbuf_slac_block_, slac_A_.data(), slac_block_.data());

		if ( frame_ptr->frame_ == 1 ) {
			slac_full_mat_ = slac_base_mat_ * slac_num_;
		}
		int idx = frame_ptr->frame_ - 1;
		slac_full_mat_.block< 6591, 6591 >( 0, 0 ) += slac_A_;
		slac_full_mat_.block< 6591, 6 >( 0, 6591 + idx * 6 ) = slac_block_.block< 6591, 6 >( 0, 0 );
		slac_full_mat_.block< 6, 6 >( 6591 + idx * 6, 6591 + idx * 6 ) = A_;

		slac_full_b_.block< 6591, 1 >( 0, 0 ) += slac_block_.block< 6591, 1 >( 0, 6 );
		slac_full_b_.block< 6, 1 >( 6591 + idx * 6, 0 ) = b_;

		if ( frame_ptr->frame_ == slac_num_ ) {
			addRegularizationTerm();

			Eigen::LLT< Eigen::MatrixXf, Eigen::Upper > solver( slac_full_mat_ );
			Eigen::VectorXf result = - solver.solve( slac_full_b_ );

			/*
			std::ofstream file("slac_full.txt");
			if ( file.is_open() ) {
			file << slac_full_mat_ << endl;
			file.close();
			}
			std::ofstream file1("slac_full_b.txt");
			if ( file1.is_open() ) {
			file1 << slac_full_b_ << endl;
			file1.close();
			}
			*/
			std::ofstream file2("slac_full_result.txt");
			if ( file2.is_open() ) {
				file2 << result << endl;
				file2.close();
			}
		}

		/*
		if ( frame_ptr->frame_ == 1 ) {
		std::ofstream file("slac_base.txt");
		if ( file.is_open() ) {
		file << slac_base_mat_ << endl;
		file.close();
		}
		std::ofstream file1("slac_ctr.txt");
		if ( file1.is_open() ) {
		file1 << slac_init_ctr_ << endl;
		file1.close();
		}
		}
		*/
		/*
		cout << frame_ptr->id1_ << " " << frame_ptr->id2_ << " " << frame_ptr->frame_ << endl;
		cout << frame_ptr->transformation_ << endl;
		if ( frame_ptr->frame_ == 1 ) {
		std::ofstream file("slac_A.txt");
		if ( file.is_open() ) {
		file << slac_A_ << endl;
		file.close();
		}
		std::ofstream fileb("slab_block.txt");
		if ( fileb.is_open() ) {
		fileb << slac_block_ << endl;
		fileb.close();
		}
		}
		*/
		//cout << A_ << endl;
		//cout << slac_A_.block< 10, 10 >( 0, 0 ) << endl;

		/*
		cout << A_ << endl;
		cout << b_ << endl;
		cout << A_.llt ().solve (b_).cast<float>() << endl;
		*/
		//cout << A_ << endl << endl;
	}

	/*
	//check for shift
	bool has_shifted = cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_);
	force_shift_ = false;

	if(has_shifted)
	PCL_WARN ("SHIFTING\n");
	*/

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;
	//~ if(integrate)
	//~ std::cout << "\tCamera movement since previous frame was " << (rnorm + alpha * tnorm)/2 << " integrate is set to " << integrate << std::endl;
	//~ else
	//~ std::cout << "Camera movement since previous frame was " << (rnorm + alpha * tnorm)/2 << " integrate is set to " << integrate << std::endl;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	/*
	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);*/
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreIntegrationFlag ) ) {
		} else {
			//ScopeTime time( ">>> integrate" );
			//device::sync();
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
			//device::sync();
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
			//device::sync();
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
			//device::sync();
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
			//device::sync();
			integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
			//device::sync();
			//integrateTsdfVolume (depths_curr_[0], intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		}
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	//if(has_shifted && perform_last_scan_)
	//  extractAndMeshWorld ();

	++global_time_;
	return (true);
}//operator()

cv::Mat pcl::gpu::KinfuTracker::bdrodometry_interpmax( cv::Mat depth )
{
	float depth_inf = 10.0f;
	cv::Mat result = depth.clone();

	for ( int i = 0; i < 480; i++ ) {
		result.at< float >( i, 0 ) = depth_inf;
		result.at< float >( i, 639 ) = depth_inf;
		int marker = -1;
		for ( int j = 0; j < 640; j++ ) {
			if ( marker == -1 ) {
				if ( result.at< float >( i, j ) == 0.0f ) {
					marker = j - 1;
				}
			} else {
				if ( result.at< float >( i, j ) > 0.0f ) {
					float x = __max( result.at< float >( i, j ), result.at< float >( i, marker ) );
					for ( int k = marker + 1; k < j; k++ ) {
						result.at< float >( i, k ) = x;
					}
					marker = -1;
				}
			}
		}
	}
	return result;
}//bdrodometry_interpmax

cv::Mat pcl::gpu::KinfuTracker::interpmax_xy( cv::Mat depth )
{
	float depth_inf = 10.0f;
	cv::Mat result = depth.clone();
	int rr = depth.rows,
		cc = depth.cols;

	for ( int i = 0; i < rr; i++ ) {
		result.at< float >( i, 0 ) = depth_inf;
		result.at< float >( i, cc-1) = depth_inf;
		int marker = -1;
		for ( int j = 0; j < cc; j++ ) {
			if ( marker == -1 ) {
				if ( result.at< float >( i, j ) == 0.0f ) {
					marker = j - 1;
				}
			} else {
				if ( result.at< float >( i, j ) > 0.0f ) {
					float x = __max( result.at< float >( i, j ), result.at< float >( i, marker ) );
					for ( int k = marker + 1; k < j; k++ ) {
						result.at< float >( i, k ) = x;
					}
					marker = -1;
				}
			}
		}
	}//for-i

	cv::Mat interp_y_mat = depth.clone();

	for ( int j = 0; j < cc; j++ ) {
		interp_y_mat.at< float >( 0, j ) = depth_inf;
		interp_y_mat.at< float >( rr-1, j ) = depth_inf;
		int marker = -1;
		//for ( int j = 0; j < 640; j++ ) {
		for ( int i = 0; i < rr; i++ ) {
			if ( marker == -1 ) {
				if ( interp_y_mat.at< float >( i, j ) == 0.0f ) {
					marker = i - 1;
				}
			} else {
				if ( interp_y_mat.at< float >( i, j ) > 0.0f ) {
					float x = __max( interp_y_mat.at< float >( i, j ), interp_y_mat.at< float >( marker, j ) );
					for ( int k = marker + 1; k < i; k++ ) {
						interp_y_mat.at< float >( k, j ) = x;
					}
					marker = -1;
				}
			}
		}
	}//for-j

	//return result;
	//return interp_y_mat;
	return cv::max(result, interp_y_mat);
}//interpmax_xy

cv::Mat pcl::gpu::KinfuTracker::bdrodometry_getOcclusionBoundary( cv::Mat depth, float dist_threshold )
{
	cv::Mat mask( 480, 640, CV_8UC1 );
	mask.setTo( 0 );

	cv::Mat depth_max = bdrodometry_interpmax( depth );
	//cv::imshow("depth_max", depth_max); //zc

	int nbr[ 8 ][ 2 ] = {
		{ -1, -1 },
		{ -1, 0 },
		{ -1, 1 },
		{ 0, -1 },
		{ 0, 1 },
		{ 1, -1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	for ( int i = 30; i < 460; i++ ) {
		for ( int j = 20; j < 580; j++ ) {
			if ( depth.at< float >( i, j ) > 0.0f ) {
				for ( int k = 0; k < 8; k++ ) {
					if ( depth_max.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) > dist_threshold ) {
						mask.at< unsigned char >( i, j ) = 255;
						break;
					}
				}
			}
		}
	}
	return mask;
}

cv::Mat pcl::gpu::KinfuTracker::edge_from_dmap(cv::Mat depth, cv::Mat &edge_map_whole, float dist_threshold /*= 0.05f*/ ){
	cv::Mat mask( 480, 640, CV_8UC1 );
	mask.setTo( 0 );

	edge_map_whole = mask.clone();

	cv::Mat depth_max = bdrodometry_interpmax( depth );

	int nbr[ 8 ][ 2 ] = {
		{ -1, -1 },
		{ -1, 0 },
		{ -1, 1 },
		{ 0, -1 },
		{ 0, 1 },
		{ 1, -1 },
		{ 1, 0 },
		{ 1, 1 }
	};

#if 0	//1-for-loop
	for ( int i = 30; i < 460; i++ ) {
		for ( int j = 20; j < 580; j++ ) {
			if ( depth.at< float >( i, j ) > 0.0f ) { //当前 px 要有效
				for ( int k = 0; k < 8; k++ ) {
					if ( abs(depth.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) ) > dist_threshold ) { //注意 abs(depth...) 非 depth_max
						edge_map_whole.at< unsigned char >( i, j ) = 255;
						//不break
					}
					if ( depth_max.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) > dist_threshold ) {
						mask.at< unsigned char >( i, j ) = 255;
						break;
					}
				}
			}
		}
	}
#elif 1	//2-for-loops
	for ( int i = 30; i < 460; i++ ) {
		for ( int j = 20; j < 580; j++ ) {
			if ( depth.at< float >( i, j ) > 0.0f ) { //当前 px 要有效
				for ( int k = 0; k < 8; k++ ) {
					if ( depth_max.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) > dist_threshold ) {
						mask.at< unsigned char >( i, j ) = 255;
						break;
					}
				}
			}
		}
	}
	for ( int i = 1; i < 479; i++ ) {
		for ( int j = 1; j < 639; j++ ) {
			if ( depth.at< float >( i, j ) > 0.0f ) { //当前 px 要有效
				for ( int k = 0; k < 8; k++ ) {
					if ( abs(depth.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) ) > dist_threshold ) { //注意 abs(depth...) 非 depth_max
						edge_map_whole.at< unsigned char >( i, j ) = 255;
						break;
					}
				}
			}
		}
	}

#endif
	return mask;

}//edge_from_dmap

void pcl::gpu::KinfuTracker::edge_from_dmap2(cv::Mat depth, cv::Mat &edge_map_whole, float dist_threshold /*= 0.05f*/ ){
	//cv::Mat mask( 480, 640, CV_8UC1 );
	//mask.setTo( 0 );
	edge_map_whole = cv::Mat::zeros(480, 640, CV_8UC1);

	int nbr[ 8 ][ 2 ] = {
		{ -1, -1 },
		{ -1, 0 },
		{ -1, 1 },
		{ 0, -1 },
		{ 0, 1 },
		{ 1, -1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	for ( int i = 1; i < 479; i++ ) {
		for ( int j = 1; j < 639; j++ ) {
			if ( depth.at< float >( i, j ) > 0.0f ) { //当前 px 要有效
				for ( int k = 0; k < 8; k++ ) {
					if ( abs(depth.at< float >( i + nbr[ k ][ 0 ], j + nbr[ k ][ 1 ] ) - depth.at< float >( i, j ) ) > dist_threshold ) { //注意 abs(depth...) 非 depth_max
						edge_map_whole.at< unsigned char >( i, j ) = 255;
						break;
					}
				}
			}
		}
	}

}//edge_from_dmap2

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::bdrodometry(const DepthMap& depth_raw, const View * pcolor)
{
	//printf("---------------callCnt:= %d\n", callCnt_);
	callCnt_++;

	//ScopeTime time( "Kinfu Tracker All" );
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	cv::Mat md;
	cv::Mat mvmap;
	cv::Mat md_mask;
	cv::Mat mvmap_max;
	cv::Mat gx( 480, 640, CV_32FC1 );
	cv::Mat gy( 480, 640, CV_32FC1 );
	pcl::PointCloud< pcl::PointNormal >::Ptr vmappcd( new pcl::PointCloud< pcl::PointNormal >() );
	pcl::PointCloud< pcl::PointNormal >::Ptr vmappcdraw( new pcl::PointCloud< pcl::PointNormal >() );
	pcl::KdTreeFLANN< pcl::PointNormal > vmaptree;
	pcl::PointCloud< pcl::PointNormal >::Ptr maskedpts( new pcl::PointCloud< pcl::PointNormal >() );

	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all"); //release 下 ~12ms
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		int c;
		vector<unsigned short> data;
		depths_curr_[ 0 ].download(data, c);
		char filename[ 1024 ];
		std::sprintf( filename, "image/bf/%06d.png", global_time_ + 1 );
		cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
		//cv::imwrite( filename, m );
		cv::Mat m8u; //zc
		m.convertTo(m8u, CV_8UC1, UCHAR_MAX/3e3);
		cv::imshow( "m8u", m8u );
		
		m.convertTo( md, CV_32FC1, 1.0 / 1000.0, 0.0 );
		md_mask = bdrodometry_getOcclusionBoundary( md );

		cv::imshow( "temp", md_mask );

		for ( int i = 0; i < 480; i++ ) {
			for ( int j = 0; j < 640; j++ ) {
				if ( md_mask.at< unsigned char >( i, j ) > 0 && md.at< float >( i, j ) < max_icp_distance_ ) {
					pcl::PointNormal pt;
					pt.z = md.at< float >( i, j );
					pt.x = ( j - intr.cx ) * pt.z / intr.fx;
					pt.y = ( i - intr.cy ) * pt.z / intr.fy;
					maskedpts->points.push_back( pt );
				}
			}
		}

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
		pcl::device::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point


	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );

#if 0   //bdr.orig, 若指定 r3t3, 则此逻辑有错 @2018-11-14 01:54:12
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
#else   //从 kf.operator() 拷过来
		Matrix3frm R_iden;
		R_iden = Eigen::Matrix3f::Identity ();
		Mat33&  rotation_id = device_cast<Mat33> (R_iden); /// Identity Rotation Matrix
#endif
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		/*
		DeviceArray2D<PixelRGB> temp_view;
		DeviceArray2D<unsigned short> temp_depth;
		temp_depth.create( 480, 640 );
		int c;
		char filename[ 1024 ];
		vector<unsigned short> data;
		device::generateDepth(device_cam_rot_local_prev_inv, device_cam_trans_local_prev, vmaps_g_prev_[0], temp_depth);
		temp_depth.download(data, c);
		std::sprintf( filename, "image/vmap/%06d.png", global_time_ + 1 );
		cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
		//cv::imwrite( filename, m );

		vector<pcl::gpu::PixelRGB> ndata;
		temp_view.create( 480, 640 );
		device::generateNormal(device_cam_rot_local_prev_inv, device_cam_trans_local_prev, vmaps_g_prev_[0], nmaps_g_prev_[0], temp_view);
		temp_view.download( ndata, c );
		sprintf( filename, "image/nmap/%06d.png", global_time_ + 1 );
		cv::Mat m1( 480, 640, CV_8UC3, (void*)&ndata[0] );
		cv::Mat mm;
		cv::cvtColor( m1, mm, CV_RGB2BGR );
		//cv::imwrite( filename, mm );
		*/

		if ( global_time_ == 1 ) {
			bdr_temp_depth_.create( 480, 640 );
		}
		device::generateDepth(device_cam_rot_local_prev_inv, device_cam_trans_local_prev, vmaps_g_prev_[0], bdr_temp_depth_);
		int c;
		vector<unsigned short> data;
		bdr_temp_depth_.download(data, c);
		cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );


		m.convertTo( mvmap, CV_32FC1, 1.0 / 1000.0, 0.0 );

		mvmap_max = bdrodometry_interpmax( mvmap );
		cv::Sobel( mvmap_max, gx, -1, 1, 0, 7, 1.0 / 2048.0 );
		cv::Sobel( mvmap_max, gy, -1, 0, 1, 7, 1.0 / 2048.0 );
		//cv::Sobel( mvmap_max, gx, -1, 1, 0, 5, 1.0 / 128.0 );
		//cv::Sobel( mvmap_max, gy, -1, 0, 1, 5, 1.0 / 128.0 );

		//gradient to normal

		cv::Mat mask( 480, 640, CV_8UC1 );
		mask.setTo( 0 );
		cv::Mat mask1( 480, 640, CV_32FC1 );
		mask1.setTo( 0.0f );

		for ( int v = 0; v < 480; v++ ) {
			for ( int u = 0; u < 640; u++ ) {
				float d = mvmap.at< float >( v, u );
				if ( d > 0.0f ) {
					Eigen::Matrix2f xx;
					Eigen::Vector3f nn;
					float gxx = gx.at< float >( v, u );
					float gyy = gy.at< float >( v, u );
					mask1.at< float >( v, u ) = sqrtf( gxx * gxx + gyy * gyy ) * 100;
					xx( 0, 0 ) = ( d + ( u - intr.cx ) * gxx ) / intr.fx;
					xx( 0, 1 ) = ( v - intr.cy ) * gxx / intr.fy;
					xx( 1, 0 ) = ( u - intr.cx ) * gyy / intr.fx;
					xx( 1, 1 ) = ( d + ( v - intr.cy ) * gyy ) / intr.fy;
					//Eigen::Vector2f res = xx.llt().solve( Eigen::Vector2f( gxx, gyy ) );
					float det = xx( 0, 0 ) * xx( 1, 1 ) - xx( 0, 1 ) * xx( 1, 0 );

					if ( abs( det ) < 1e-6 ) {
						continue;
					}

					Eigen::Matrix2f yy;
					yy( 0, 0 ) = xx( 1, 1 );
					yy( 0, 1 ) = - xx( 0, 1 );
					yy( 1, 0 ) = - xx( 1, 0 );
					yy( 1, 1 ) = xx( 0, 0 );
					Eigen::Vector2f res = 1.0f / det * yy * Eigen::Vector2f( gxx, gyy );

					Eigen::Vector3f n( res( 0 ), res( 1 ), -1 );
					n.normalize();

					Eigen::Vector3f pp( ( u - intr.cx ) * d / intr.fx, ( v - intr.cy ) * d / intr.fy, d );
					pp.normalize();

					float cosangle = - pp.dot( n );
					if ( cosangle < cos( M_PI / 2.0 - M_PI / 12.0 ) ) {

						Eigen::Vector3f x;
						x = n.cross( pp );
						nn = pp.cross( x );
						nn.normalize();

						mask.at< unsigned char >( v, u ) = 255;
						pcl::PointNormal pt;
						pt.x = ( u - intr.cx ) * d / intr.fx;
						pt.y = ( v - intr.cy ) * d / intr.fy;
						pt.z = d;
						pt.normal_x = nn( 0 );
						pt.normal_y = nn( 1 );
						pt.normal_z = nn( 2 );
						
						if ( !pcl_isnan( pt.normal_x ) && !pcl_isnan( pt.normal_y ) && !pcl_isnan( pt.normal_z ) ) {
							vmappcdraw->points.push_back( pt );
						}
						/*
						if ( vmappcd.size() == 1 ) {
							cout << v << ", " << u << endl << endl;
							cout << gxx << endl << gyy << endl << endl;
							cout << xx << endl;
							cout << res << endl << endl;
							cout << n << endl << endl;
							cout << pp << endl << endl;
							cout << nn << endl << endl;
						}
						*/

						/*
						if ( global_time_ + 1 == 950 && vmappcdraw->size() == 1441 ) {
							cout << v << ", " << u << endl << endl;
							cout << gxx << endl << gyy << endl << endl;
							cout << xx << endl;
							cout << res << endl << endl;
							cout << n << endl << endl;
							cout << pp << endl << endl;
							cout << x << endl << endl;
							cout << nn << endl << endl;
						}
						*/

					}

					/*
					if ( global_time_ + 1 == 173 && v == 132 && u == 470 ) {
						cout << v << ", " << u << endl << endl;
						cout << gxx << endl << gyy << endl << endl;
						cout << xx << endl;
						cout << res << endl << endl;
						cout << n << endl << endl;
						cout << pp << endl << endl;
						cout << cosangle << endl << endl;
						//cout << nn << endl << endl;
					}
					*/
				}
			}
			//imshow( "temp", mask1 );
			//imshow( "tempmask", mask );
		}

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}
	{
		//ScopeTime time(">>> icp-all");
		Eigen::Affine3f aff;
		aff.linear() = cam_rot_global_curr;
		aff.translation() = cam_trans_global_curr;
		pcl::transformPointCloudWithNormals( *vmappcdraw, *vmappcd, aff );
		vmaptree.setInputCloud( vmappcd );
		float amplifier = amplifier_ / 4.0f / 4.0f;
		//cout << global_time_ + 1 << endl;

		for (int level_index = LEVELS-1; level_index>=0; --level_index)
		{
			int iter_num = icp_iterations_[level_index];

			// current maps
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];   

			// previous maps
			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

			// We need to transform the maps from global to the local coordinates

#if 0   //bdr.orig, 若指定 r3t3, 则此逻辑有错 @2018-11-14 01:54:12
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
#else   //从 kf.operator() 拷过来
			Matrix3frm R_iden;
			R_iden = Eigen::Matrix3f::Identity ();
			Mat33&  rotation_id = device_cast<Mat33> (R_iden); /// Identity Rotation Matrix
#endif

			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			MapArr& vmap_temp = vmap_g_prev;
			MapArr& nmap_temp = nmap_g_prev;
			device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 

			for (int iter = 0; iter < iter_num; ++iter)
			{
				//CONVERT TO DEVICE TYPES
				// CURRENT LOCAL TRANSFORM
				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				estimateCombined (device_cam_rot_local_curr, device_cam_trans_local_curr, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr (level_index), 
					vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());
				/*
				estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
				vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
				*/

				pcl::PointCloud<pcl::PointNormal>::Ptr transformed( new pcl::PointCloud<pcl::PointNormal>() );
				Eigen::Affine3f aff;
				aff.linear() = cam_rot_global_curr;
				aff.translation() = cam_trans_global_curr;
				pcl::transformPointCloud( *maskedpts, *transformed, aff );
				std::vector<int> pointIdxNKNSearch(1);
				std::vector<float> pointNKNSquaredDistance(1);
				const float kdtreeDistTh = e2c_dist_; //5cm	跟 cuOdo 公用同一个命令行参数 @2018-11-14 09:57:55

				float rr = 0.0f;
				int ll = 0;
				AA_.setZero();
				bb_.setZero();
				//printf("level_index, iter, transformed->size(): %d, %d, %d\n", level_index, iter, transformed->size());
				//zc: transformed 点云并不缩放, 而且不慢; 见: http://codepad.org/AkP2y8z7
				//tt0.tic();
				for ( int l = 0; l < transformed->size(); l++ ) {
					vmaptree.nearestKSearch( transformed->points[ l ], 1, pointIdxNKNSearch, pointNKNSquaredDistance );
					/*
					if ( l == 500 ) {
						cout << maskedpts->points[ l ] << endl;
						cout << transformed->points[ l ] << endl;
						cout << pointIdxNKNSearch[ 0 ] << endl;
						cout << pointNKNSquaredDistance[ 0 ] << endl;
						cout << vmappcd->points[ pointIdxNKNSearch[ 0 ] ] << endl;
					}
					*/
					//if ( pointNKNSquaredDistance[ 0 ] < 0.1 * 0.1 ) {
					if(pointNKNSquaredDistance[0] < kdtreeDistTh * kdtreeDistTh){

						Eigen::Vector3f nn( vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_x, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_y, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_z );
						Eigen::Vector3f qq( vmappcd->points[ pointIdxNKNSearch[ 0 ] ].x, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].y, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].z );
						Eigen::Vector3f pp( transformed->points[ l ].x, transformed->points[ l ].y, transformed->points[ l ].z );
						float r = nn.dot( qq - pp );
						Eigen::Vector3f pxn = pp.cross( nn );
						float xx[ 6 ] = { pxn( 0 ), pxn( 1 ), pxn( 2 ), nn( 0 ), nn( 1 ), nn( 2 ) };
						for ( int ii = 0; ii < 6; ii++ ) {
							for ( int jj = 0; jj < 6; jj++ ) {
								AA_( ii, jj ) += xx[ ii ] * xx[ jj ];
							}
							bb_( ii ) += xx[ ii ] * r;
						}
						ll ++; rr += r * r;

						/*
						if ( pcl_isnan(r) ) {
							PCL_ERROR( "warning\n" );
							cout << aff.matrix() << endl;
							cout << l << endl;
							cout << transformed->points[ l ] << endl;
							cout << pointIdxNKNSearch[ 0 ] << endl;
							cout << vmappcd->points[ pointIdxNKNSearch[ 0 ] ] << endl;
							PCL_ERROR( "debug done" );
						}
						*/
					}
				}
				//printf("bdr-vmaptree.nearestKSearch: "); tt0.toc_print();
				//zc: 这里nearestKSearch比我们程序中快 8倍, 没懂; 输出见: http://codepad.org/AkP2y8z7

				cout << "vmappcd_size: " << vmappcd->size() << " : transformed: " << transformed->size() 
					<< " -->ll: " << ll << "; e2c_dist_: " << e2c_dist_ << endl;
				//cout << rr << endl;

				/*调试输出, 用 py 3D 绘制观察
				from mpl_toolkits.mplot3d import axes3d
				import matplotlib.pyplot as plt
				from matplotlib import cm
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				xs, ys, zs = corresp_pts.T
				xxs, yys, zzs = edge_pts.T
				ax.scatter(xs, ys, zs, 'b')
				ax.scatter(xxs, yys, zzs, 'r')
				*/

				//printf("corresp_pts=np.array([\n");
				//for(size_t i = 0; i < vmappcd->size(); i++){
				//	Eigen::Vector3f qq( vmappcd->points[ i ].x, vmappcd->points[ i ].y, vmappcd->points[ i ].z );
				//	printf("[%f, %f, %f],\n", qq.x(), qq.y(), qq.z());
				//}
				//printf("])\n");

				//printf("edge_pts=np.array([\n");
				//for(size_t i = 0; i < transformed->size(); i++){
				//	Eigen::Vector3f pp( transformed->points[ i ].x, transformed->points[ i ].y, transformed->points[ i ].z );
				//	printf("[%f, %f, %f],\n", pp.x(), pp.y(), pp.z());
				//}
				//printf("])\n");

				/*
				if ( global_time_ + 1 == 950 )
				{
					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;
					cout << "Singular matrixx :" << endl << AA_ << endl;
					cout << "Corresponding bb :" << endl << bb_ << endl;

					//reset ();
					//return (false);
				}
				*/

				//checking nullspace
				/*
				double det = A_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					reset ();
					return (false);
				}
				*/

				double det = A_.determinant();
				double det1 = AA_.determinant();
				if ( fabs( det ) < 1e-15 && fabs( det1 ) < 1e-15 && ll < 10 ) {
					return false;
				}

				//Eigen::Matrix<float, 6, 1> result = A_.llt ().solve (b_).cast<float>();

				//if ( global_time_ + 1 == 173 && level_index == 0 ) {
// 				Eigen::Matrix<float, 6, 1> result = ( A_ /*+ amplifier * AA_*/ ).llt ().solve ( b_ /*+ amplifier * bb_*/ ).cast<float>();
// 				cout << "lvl, iter: " << level_index << ", " << iter << result.transpose() << endl
// 					<< "A_:\n" << A_ << "\nb_:\n" << b_.transpose() << endl
// 					;
				Eigen::Matrix<float, 6, 1> result = ( A_ + amplifier * AA_ ).llt ().solve ( b_ + amplifier * bb_ ).cast<float>();
				//}
				//Eigen::Matrix<float, 6, 1> result = AA_.llt ().solve (bb_).cast<float>();

				//cout << A_ << endl;
				//cout << AA_ << endl;
				//cout << b_ << endl;
				//cout << bb_ << endl;

				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

				/*
				PCL_INFO ("At ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
				cout << "Determinant : " << det << endl;
				cout << "Singular matrix :" << endl << A_ << endl;
				cout << "Corresponding b :" << endl << b_ << endl;
				cout << "result " << endl << result << endl;
				*/

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				//compose
				cam_trans_global_curr = cam_rot_incremental * cam_trans_global_curr + cam_trans_incremental;
				cam_rot_global_curr = cam_rot_incremental * cam_rot_global_curr;
			}

			amplifier *= 4.0f;
		}

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	/*
	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);*/
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		//ScopeTime time( ">>> integrate" );
		integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	++global_time_;
	return (true);
}//bdrodometry

bool
pcl::gpu::KinfuTracker::cuOdometry( const DepthMap &depth_raw, const View *pcolor /*= NULL*/){
	//printf("---------------callCnt:= %d\n", callCnt_);
	callCnt_++;
	//ScopeTime time( "Kinfu Tracker All" );
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

#if 0	//ahc 放到 kinfu_app 里了, 这里分割有错, 搞不定 @2017-4-3 20:42:51
	cv::Mat dm_raw(depth_raw.rows(), depth_raw.cols(), CV_16UC1); //milli-m, raw
	int c;
	depth_raw.download(dm_raw.data, depth_raw.colsBytes());
	tt0.tic();
	CloudType::Ptr depCloud = cvMat2PointCloud(dm_raw, intr);
	printf("cvMat2PointCloud "); tt0.toc_print();

	if(0){//ahc bug, 所以再取出来看看:
		cv::Mat dbgMm = cloud2cvmat(depCloud, intr);
		cv::Mat dbgMm8u;
		dbgMm.convertTo(dbgMm8u, CV_8UC1, UCHAR_MAX/3e3);
		imshow("dbgMm8u", dbgMm8u);
	}

	RGBDImage rgbdObj(*depCloud);
	cv::Mat dbgSegMat(depCloud->height, depCloud->width, CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	planeFitter_.run(&rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(planeFitter_.membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@cuOdometry";
	imshow(winNameAhc, dbgSegMat);
#endif

	cv::Mat m_filt; //milli-meters, bilateral-filt 之后
	cv::Mat m8u; //dbg-draw
    cv::Mat m8uc3; //rgb-dbg-draw
	cv::Mat md; //meters
	cv::Mat mvmap;
	cv::Mat md_mask; //occluding contour mask
	cv::Mat edge_whole_mask; //whole edge, 包含 md_mask
	cv::Mat mvmap_max; //inp-mat
	cv::Mat gx( 480, 640, CV_32FC1 );
	cv::Mat gy( 480, 640, CV_32FC1 );
	pcl::PointCloud< pcl::PointXYZ >::Ptr maskedpts( new pcl::PointCloud< pcl::PointXYZ >() );

	//zc:
	cv::Mat dcurrFiltHost(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
	DepthMap depthPlFilt;
	{
	ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all"); //release 下 ~12ms
	device::bilateralFilter (depth_raw, depths_curr_[0]);
	
	depths_curr_[0].download(dcurrFiltHost.data, depths_curr_[0].colsBytes()); //用滤波 dmap 替换 raw, 观察

	int c;
	vector<unsigned short> data;
	depths_curr_[ 0 ].download(data, c);
	char filename[ 1024 ];
	std::sprintf( filename, "image/bf/%06d.png", global_time_ + 1 );
	//cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
	m_filt = cv::Mat( 480, 640, CV_16UC1, (void *)&data[0] ).clone(); //一定要拷贝！！ 因为这里局部作用域
	
	//cout<<"mm:" << mm(cv::Rect(0,0,11,11)) <<endl;
	//cv::imwrite( filename, m );
	m_filt.convertTo(m8u, CV_8UC1, UCHAR_MAX/3e3);
	//cv::imshow( "m8u", m8u );

	m_filt.convertTo( md, CV_32FC1, 1.0 / 1000.0, 0.0 );
	md_mask = bdrodometry_getOcclusionBoundary( md );
	//md_mask = edge_from_dmap(md, edge_whole_mask, 0.01); 	//tsdf-v18.16

	cv::Mat md_interp_xy;
	if(dbgKf_ >= 1){
		tt3.tic();
		/*cv::Mat*/ md_interp_xy = interpmax_xy(md);
		printf("md_interp_xy "); tt3.toc_print();
		//cout << md_interp_xy(cv::Rect(260, 320, 10, 10)) << endl; //in meters
	}

	//cv::imshow( "md_mask", md_mask );
	if(!isTsdfVer(18))
		m8u.setTo(UCHAR_MAX, md_mask);
	else{
		edge_from_dmap2(md, edge_whole_mask, 0.01); 	//tsdf-v18.16
		//md_mask = edge_whole_mask; //alias for *maskedpts* //slows down edge2contour alignment a lot due to maskedpts growth

		if(dbgKf_ >= 1){
			m8u.setTo(UCHAR_MAX, edge_whole_mask);

			cv::Mat md_interp_xy8u;
			md_interp_xy.convertTo(md_interp_xy8u, CV_8UC1, UCHAR_MAX/3.);
			cv::imshow("md_interp_xy8u", md_interp_xy8u);
		}
	}
	if(dbgKf_ >= 1)
		cv::cvtColor(m8u, m8uc3, cv::COLOR_GRAY2BGR);

	//cv::imshow( "m8u", m8u );


	for ( int i = 0; i < 480; i++ ) {
		for ( int j = 0; j < 640; j++ ) {
			if ( md_mask.at< unsigned char >( i, j ) > 0 && md.at< float >( i, j ) < max_icp_distance_ ) {
				pcl::PointXYZ pt;
				pt.z = md.at< float >( i, j );
				pt.x = ( j - intr.cx ) * pt.z / intr.fx;
				pt.y = ( i - intr.cy ) * pt.z / intr.fy;
				maskedpts->points.push_back( pt );
			}
		}
	}
	if(dbgKf_ >= 1)
		printf("maskedpts->size(): %d\n", maskedpts->size());

	if (max_icp_distance_ > 0)
		device::truncateDepth(depths_curr_[0], max_icp_distance_);

	for (int i = 1; i < LEVELS; ++i)
		device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

	for (int i = 0; i < LEVELS; ++i)
	{
		device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
		//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
		//computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);

		int2 pxDbg;
		pxDbg.x = pxDbg_[0];
		pxDbg.y = pxDbg_[1];
		computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i], pxDbg);
	}
	pcl::device::sync ();

	}//">>> Bilateral, pyr-down-all, create-maps-all"

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		//↓--v21.6.0.1 参考 slac, 借用此 buf 给我的 ghost vol2 用 @2018-11-30 00:24:23
		//cyclical2_.resetBuffer ( tsdf_volume2_ );
		cyclical2_.resetBuffer_inner ( tsdf_volume_->volume2nd_ );
		printf("AFTER-cyclical2_.resetBuffer_inner\n"); //应在 create_init_cu_volume() 之后, 否则逻辑错误, buffer 空指针 @2018-12-4 11:29:47

		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

#if 0	//放弃, 因为 fid==0 时, 尚未初始化 planeFiltParam_
		if(isPlFilt_){
			//fid==0 时, 也要构造 depthPlFilt; 拷贝自后面
			Eigen::Affine3f cam2g = this->getCameraPose(); //移植到 kinfuLS 之后, 可能存在 shift 导致出错的风险, 暂不处理 @2017-4-5 11:29:23
			Eigen::Affine3d cam2gd = cam2g.cast<double>();

			Eigen::Vector4f planeFiltParam_c;
			Eigen::Vector3d nvec, nvec_c;
			nvec << planeFiltParam_.x(), planeFiltParam_.y(), planeFiltParam_.z(); //此时还是 g-coo
			nvec_c = cam2gd.rotation().transpose() * nvec;
			double D_c = planeFiltParam_.w() + nvec.dot(cam2gd.translation());
			planeFiltParam_c << nvec_c.x(), nvec_c.y(), nvec_c.z(), D_c;

			float4 &device_plparam = device_cast<float4>(planeFiltParam_c);
			printf("device_plparam.xyzw: %f, %f, %f, %f\n", device_plparam.x, device_plparam.y, device_plparam.z, device_plparam.w);

			device::planeFilter(depth_raw, intr, device_plparam, depthPlFilt);
		}//if-(isPlFilt_)

		//device::integrateTsdfVolume(depth_raw, 
		device::integrateTsdfVolume(isPlFilt_ ? depthPlFilt : depth_raw, 
			intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
#else
		if(!isPlFilt_){
		device::integrateTsdfVolume(depth_raw, 
			intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		}
#endif

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}//if (global_time_ == 0)

#if 0	//暂时放弃考虑 shift 问题 @2017-4-5 16:03:36
	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}
#endif

	Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
	Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

	Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
	Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
	float3& device_tprev     = device_cast<float3> (tprev);

	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	if(this->isCuInitialized_){
		//本段核心: zcRenderCubeDmap 生成synCuDmap_device_, 生成 vmaps_cu_g_prev_, nmap...; 其余是调试观察+杂项测试
		ScopeTime time("this->isCuInitialized_-zcRenderCubeDmap");

		Eigen::Affine3f cam2g = this->getCameraPose(); //移植到 kinfuLS 之后, 可能存在 shift 导致出错的风险, 暂不处理 @2017-4-5 11:29:23

		//test @2017-11-12 21:27:22
		//tt2.tic();
		//bool isTrihedronVisible = cube_g_.isTrihedronVisible(cam2g.cast<double>(), fx_, fy_, cx_, cy_, dbgKf_ >= 1);
		//printf("@@@@@@@@@@@@@@@@@@@isTrihedronVisible: %s; took: ", isTrihedronVisible ? "TRUE" : "FALSE");
		//tt2.toc_print(); //1ms, 效率可以

		//tt.tic(); //~160ms, 可能是 Cube.drawContour 耗时间? 并不是
		//虚拟立方体配准前, 【BA】-- BeforeAlignment
		Cube cubeCamBA(cube_g_, cam2g.cast<double>().inverse());
		//printf("cubeCamBA-ctor: "); tt.toc_print(); //几乎~0ms
		//tt.tic();
		//调试绘制配准结果 @耿老师需求    //2017-1-18 00:53:18
		//1, 线框图
		//cv::Mat /*dmat8uc1,*/ dmat8uc3;
		//dmatRaw_.convertTo(dmat8uc1, CV_8UC1, UCHAR_MAX/2e3);
		//cv::cvtColor(m8u, dmat8uc3, cv::COLOR_GRAY2BGR);
		//cubeCamBA.drawContour(dmat8uc3, fx_, fy_, cx_, cy_, 255); //蓝色轮廓图
		if(dbgKf_ >= 1)
			cubeCamBA.drawContour(m8uc3, fx_, fy_, cx_, cy_, 255); //蓝色轮廓图

		tt0.tic(); //仅仅 zcRenderCubeDmap: ~130ms; 改用水平射线法后, 4~13ms
		//int step_tmp = 1; //间隔采样, 对生成图像速度有提升, 但对 ICP 速度没有提升, 因为是 dst, 不是src
		cv::Mat synCuDmapBA = zcRenderCubeDmap(cubeCamBA, fx_, fy_, cx_, cy_, sample_step_); //cv16u
		//↑--icp2: sample_step_=2
		printf("zcRenderCubeDmap: "); tt0.toc_print();

#if 0	//genSynData_
		if(genSynData_){ //键盘 't' 控制
			//先确定 cu->g 矩阵 T_(g,cu), 其实该放在全局, 或说单例模式; 暂时不管
			//记得先前添加 8顶点顺序: 0,123,456,7
			Vector3f cuOrig = cube_g_.cuVerts8_[0].cast<float>();
			Vector3f cuXpt = cube_g_.cuVerts8_[1].cast<float>(),
					 cuYpt = cube_g_.cuVerts8_[2].cast<float>(),
					 cuZpt = cube_g_.cuVerts8_[3].cast<float>();

			//轴向两端点向量, 量纲米 (m)
			Vector3f cuXax = cuXpt - cuOrig,
					 cuYax = cuYpt - cuOrig,
					 cuZax = cuZpt - cuOrig;
			//归一化, 否则不能用
			//cuXax.normalize();
			//cuYax.normalize();
			//cuZax.normalize();

			Matrix3f cuRg;
			cuRg << cuXax.normalized(), cuYax.normalized(), cuZax.normalized();
			cout << "cuXax, cuYax, cuZax:\n"
				 << cuXax << endl
				 << cuYax << endl
				 << cuZax << endl
				 << "cuRg:\n" << cuRg << endl
				 << "det(cuRg): " << cuRg.determinant() << endl;

			Affine3f cuTg;
			cuTg.linear() = cuRg;
			cuTg.translation() = cuOrig;
			cout << "cuTg, cuTg.T:\n"
				 << cuTg.matrix() << endl
				 << cuTg.inverse().matrix() << endl;

			//cam->g, 再 g->cu
			Affine3f camTcu = cuTg.inverse() * cam2g;
			cout << "camTcu:\n" << camTcu.matrix() << endl;

			//7点 3D立方体坐标系(即新全局坐标系) 坐标: 不用计算值, 用传参"定义" 值 //cuSideLenVec_
			int totalPts = 7;
			vector<Vector3f> cuNewCoos;
			cuNewCoos.push_back(Vector3f(0,0,0)); //先添加原点, 0号

			//1,2,3号点:
			Matrix3f iden3f = Matrix3f::Identity();
			for(int i=0; i<3; i++){
				Vector3f axi = (i==0 ? cuXax : (i==1 ? cuYax : cuZax));
				//Vector3f t = cuSideLenVec_[i]
				//看看与哪条边长度相近:
				float minLenDiff = 10.f; //m, 默认初值 10m, 权当极大值
				size_t j_idx = 0;
				for(size_t j=0; j<3; j++){
					float sdLenDiff = abs(axi.norm() - cuSideLenVec_[j]);
					if(sdLenDiff < minLenDiff){
						minLenDiff = sdLenDiff;
						j_idx = j;
					}
				}
				cuNewCoos.push_back(iden3f.row(i) * cuSideLenVec_[j_idx]);
			}

			//4,5,6号点:
			cuNewCoos.push_back(cuNewCoos[1]+cuNewCoos[2]); //本应 -[0], 但因 [0]=000, 所以省略
			cuNewCoos.push_back(cuNewCoos[2]+cuNewCoos[3]);
			cuNewCoos.push_back(cuNewCoos[3]+cuNewCoos[1]);


			//存文件: Dmap, ({p3d}, {p2d})
			std::ofstream synCu3d_fout("synCu3d.txt");
			std::ofstream synCu2d_fout("synCu2d.txt");
			std::ofstream synCu2d_noisy_fout("synCu2d-noisy.txt"); //对像素坐标扰动,加高斯噪声
			std::ofstream camPoseGT_fout("camPoseGT.txt"); //这里采用 opencv-rodrigues 习惯, rvec 存轴角表示
			//先存 cam-post-GT: t.xyz+rvec.xyz
			//cv::Mat rmatCam2cu;
			//eigen2cv(camTcu.linear().matrix(), rmatCam2cu);

			AngleAxisf rvec(camTcu.linear());
			Vector3f tvec(camTcu.translation());
			camPoseGT_fout << tvec(0) * M2MM << ' ' << tvec(1) * M2MM << ' ' << tvec(2) * M2MM
						<< ' ' << rvec.axis()(0) * rvec.angle()
						<< ' ' << rvec.axis()(1) * rvec.angle()
						<< ' ' << rvec.axis()(2) * rvec.angle();

			cv::imwrite("synCu16u.png", synCuDmapBA);
			if(1){
				//自产自销, 本程序生成, 再回代本程序, 目的对比 ICP 精度
				char fnBuf[80] = {0};
				//无噪声的
				sprintf(fnBuf, "synCu16u-%04d.png", callCnt_);
				cv::imwrite(fnBuf, synCuDmapBA);
				//有噪声的
				sprintf(fnBuf, "synCu16u-noisy-%04d.png", callCnt_);
				cv::Mat noiseMat(synCuDmapBA.size(), synCuDmapBA.type());
				randn(noiseMat, 0, 1.5);

				cv::Mat synCuNoisy = synCuDmapBA + noiseMat;
				//确保无效区域, 不要乱加噪声
				synCuNoisy.setTo(0, synCuDmapBA == 0);

				cv::imwrite(fnBuf, synCuNoisy);

				std::ofstream camPoseGT_fout("camPoseGT-K.txt", ios::app); //这里采用 opencv-rodrigues 习惯, rvec 存轴角表示
				camPoseGT_fout << tvec(0) * M2MM << ' ' << tvec(1) * M2MM << ' ' << tvec(2) * M2MM
					<< ' ' << rvec.axis()(0) * rvec.angle()
					<< ' ' << rvec.axis()(1) * rvec.angle()
					<< ' ' << rvec.axis()(2) * rvec.angle()
					<< endl;
			}


			for(int i=0; i<totalPts; i++){
				//立方体坐标输出观察:
				cout << cuNewCoos[i].transpose() * M2MM << endl;
				synCu3d_fout << cuNewCoos[i].transpose() * M2MM << endl;
			}

			//求 2D 像素坐标
			//vector<cv::Point2f> pxs;
			cv::Mat_<cv::Point3f> pxsMat(totalPts, 1); //2D+1D:= (u,v, depth)
			for(int i=0; i<totalPts; i++){
				//cv::Point pxi = getPxFrom3d(cubeCamBA.cuVerts8_[i], fx_, fy_, cx_, cy_);
				//改用float 像素坐标, 相当于亚像素
				cv::Point2f pxi = getPx2fFrom3d(cubeCamBA.cuVerts8_[i], fx_, fy_, cx_, cy_);
				//pxs.push_back(pxi);
				pxsMat(i,0).x = pxi.x;
				pxsMat(i,0).y = pxi.y;

				float depthMM = cubeCamBA.cuVerts8_[i].z() * M2MM;
				pxsMat(i,0).z = depthMM;

				cout << pxi << endl;
				//逐元素, 防止带 "[..]":
				//synCu2d_fout << pxi.x << ' ' << pxi.y << ' ' << depthMM << endl;
				//再改用 int, 观察精度损失, 误差增大情况:
				synCu2d_fout << (int)pxi.x << ' ' << (int)pxi.y << ' ' << (int)depthMM << endl;
			}

			//2D像素位置加噪声
			//cv::randn(pxsMat, pxsMat, 0.5); //均值是自身旧数据 //error
			cv::Mat_<cv::Point3f> pxNoiseMat(pxsMat.rows, pxsMat.cols);
			cv::randn(pxNoiseMat, 0, 1.5);
			pxsMat += pxNoiseMat;
			for(int i=0; i<totalPts; i++){
				//synCu2d_noisy_fout << pxsMat(i,0).x << ' ' << pxsMat(i,0).y << ' ' << pxsMat(i,0).z << endl;
				//再改用 int, 观察精度损失, 误差增大情况:
				synCu2d_noisy_fout << (int)pxsMat(i,0).x << ' ' << (int)pxsMat(i,0).y << ' ' << (int)pxsMat(i,0).z << endl;
			}

			synCu3d_fout.close();
			synCu2d_fout.close();
			synCu2d_noisy_fout.close();
			camPoseGT_fout.close();

			genSynData_ = false; //重置false, 及每次按键仅当次生效
		}//if-genSynData_
#endif	//genSynData_

		if(dbgKf_ >= 1){
			cv::Mat synCu8u;
			synCuDmapBA.convertTo(synCu8u, CV_8U, UCHAR_MAX/2e3);

			cubeCamBA.drawContour(synCu8u, fx_, fy_, cx_, cy_, 255);
			imshow("synCu8u", synCu8u);

			if(0){
			int step_tmp2 = 2;
			cv::Mat synCuDmapBA_s2 = zcRenderCubeDmap(cubeCamBA, fx_, fy_, cx_, cy_, step_tmp2); //cv16u
			cv::Mat synCu8u_s2;
			synCuDmapBA_s2.convertTo(synCu8u_s2, CV_8U, UCHAR_MAX/2e3);

			cubeCamBA.drawContour(synCu8u_s2, fx_, fy_, cx_, cy_, 255);
			imshow("synCu8u_s2", synCu8u_s2);
			}

			if(dbgKf_ == 2){
				char fnBuf[80] = {0};
				sprintf(fnBuf, "synCu8u-%06d.png", callCnt_);
				cv::imwrite(fnBuf, synCu8u);
			}
		}

		synCuDmap_device_.upload(synCuDmapBA.data, synCuDmapBA.cols * synCuDmapBA.elemSize(), synCuDmapBA.rows, synCuDmapBA.cols);
		
		device::createVMap(intr(0), synCuDmap_device_, vmaps_cu_g_prev_[0]);

		tt3.tic();
#if 01
		computeNormalsEigen(vmaps_cu_g_prev_[0], nmaps_cu_g_prev_[0]);
#elif 1	//还得用 bdr 的nmap 求解方案:
		cv::Sobel(synCuDmapBA, gx, CV_32F, 1,0, 7, 1.0/2048);
		cv::Sobel(synCuDmapBA, gy, CV_32F, 0,1, 7, 1.0/2048);
		gx_device_.upload(gx.data, gx.cols * gx.elemSize(), gx.rows, gx.cols);
		gy_device_.upload(gy.data, gy.cols * gy.elemSize(), gy.rows, gy.cols);
		device::computeNormalsContourcue(intr, synCuDmap_device_, gx_device_, gy_device_, nmaps_cu_g_prev_[0]);
#endif
		printf("synCuDmap_device_-computeNormalsContourcue"); tt3.toc_print();

		device::tranformMaps(vmaps_cu_g_prev_[0], nmaps_cu_g_prev_[0], device_Rprev, device_tprev, vmaps_cu_g_prev_[0], nmaps_cu_g_prev_[0]);

		if(dbgKf_ >= 1){
			View nmap_cu_rgb; //这个 nmap 渲染是在相机坐标系下, 见 generateNormal 代码
			nmap_cu_rgb.create(gx.rows, gx.cols);
			device::generateNormal(device_Rprev_inv, device_tprev, vmaps_cu_g_prev_[0], nmaps_cu_g_prev_[0], nmap_cu_rgb);
			cv::Mat nmap_cu_rgb_host(gx.rows, gx.cols, CV_8UC3);
			nmap_cu_rgb.download(nmap_cu_rgb_host.data, nmap_cu_rgb.colsBytes());
			cv::cvtColor(nmap_cu_rgb_host, nmap_cu_rgb_host, CV_RGB2BGR);
			imshow("nmap_cu_rgb_host", nmap_cu_rgb_host);
		}


		for(int i = 1; i < LEVELS; ++i){
			//device::pyrDown //不必, resizeVMap 已经金字塔
			//device::createVMap(intr(i), )
			resizeVMap(vmaps_cu_g_prev_[i-1], vmaps_cu_g_prev_[i]);
			resizeNMap(nmaps_cu_g_prev_[i-1], nmaps_cu_g_prev_[i]);
		}

		//kdtree: 用 cuTreeInited_ 控制, 仅构造一次 tree
		if(!cuTreeInited_ && this->cuContCloud_->size() > 0){
			cuEdgeKdtree_.setInputCloud(this->cuContCloud_);

			//这里 kdtree 虽然没用 PointNormal, 但是也存 normal-vec, 使其与 pt-vec 序号对的上
			cv::Mat nmap_cu_g_prev_host(nmaps_cu_g_prev_[0].rows(), nmaps_cu_g_prev_[0].cols(), CV_32F);
			nmaps_cu_g_prev_[0].download(nmap_cu_g_prev_host.data, nmaps_cu_g_prev_[0].colsBytes());
			//PCL_INFO("nmaps_cu_g_prev_[0].download---------------OK\n");

			PointCloud<PointXYZ> cuEdgeCloud_cam_coo;
			pcl::transformPointCloud(*cuContCloud_, cuEdgeCloud_cam_coo, cam2g.inverse());
			for(int i = 0; i < cuEdgeCloud_cam_coo.size(); i++){
				PointXYZ pti = cuEdgeCloud_cam_coo.points[i];
				int u = pti.x / pti.z * fx_ + cx_,
					v = pti.y / pti.z * fy_ + cy_;
				
				Normal pti_norm;
				if(0 <= u && u < 640 && 0 <= v && v < 480){
					pti_norm.normal_x = nmap_cu_g_prev_host.at<float>(v, u);
					pti_norm.normal_y = nmap_cu_g_prev_host.at<float>(v + 480, u);
					pti_norm.normal_z = nmap_cu_g_prev_host.at<float>(v + 480 * 2, u);
				}
				else{ //u,v 超出图像范围
					const float qnan = numeric_limits<float>::quiet_NaN ();
					pti_norm.normal_x = pti_norm.normal_y = pti_norm.normal_z = qnan;
				}

				cuContCloudNormal_->push_back(pti_norm); //可能等于 nan, 之后使用一定要有效性检测
			}

			cuTreeInited_ = true;
		}
	}//if-(this->isCuInitialized_)

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// Ray casting //icp 之前先做了, 仿 bdrOdometry 规矩, 与原来 kinfu 不同 @2017-4-5 17:10:59
	{
	//ScopeTime time("ray-cast-all"); //11ms, 这么慢? @2018-3-26 00:58:34
	tt1.tic();
	raycast (intr, device_Rprev, device_tprev, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
	printf("raycast-orig "); tt1.toc_print(); //0ms
	//DepthMap depthRcast;
	//generateDepth(device_Rprev_inv, device_tprev, vmaps_g_prev_[0], depthRcast);
	//int c;
	//vector<unsigned short> depthRcast_host_data;
	//depthRcast.download(depthRcast_host_data, c);
	//cv::Mat depthRcast_host16u( 480, 640, CV_16UC1, (void *)&depthRcast_host_data[0] );
	//cv::Mat depthRcast_host8u;
	//depthRcast_host16u.convertTo(depthRcast_host8u, CV_8U, UCHAR_MAX/1500.);
	//cv::imshow("depthRcast@before_icp", depthRcast_host8u);
	//cv::Rect roi_tmp(340, 200, 10, 10);
	//cout<<depthRcast_host16u(roi_tmp)<<endl;

	for (int i = 1; i < LEVELS; ++i)
	{
		resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
		resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
	}
	//pcl::device::sync ();
	}//ray-cast-all


	Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
	Vector3f   tcurr = tprev;
	{
        tt1.tic();
	ScopeTime time(">>> icp-all");
	bool doLvlIterBreak = false;

	if(dbgKf_ > 1){ //icp-loop外, 调试
		//三项组合 (默认):
		if(term_123_){
			PCL_WARN(">>>>>>>>>using term_123_<<<<<<<<<\n");
		}
		//两项组合: 
		else if(term_12_){ //f2mod+f2mkr
			PCL_WARN(">>>>>>>>>using term_12_<<<<<<<<<\n");
		}
		else if(term_13_){
			PCL_WARN(">>>>>>>>>using term_13_<<<<<<<<<\n");
		}
		else if(term_23_){
			PCL_WARN(">>>>>>>>>using term_23_<<<<<<<<<\n");
		}
		//若仅用一项测试
		else if(term_1_){ //kf
			PCL_WARN(">>>>>>>>>using term_1_<<<<<<<<<\n");
		}
		else if(term_2_){ //f2cuboid
			PCL_WARN(">>>>>>>>>using term_2_<<<<<<<<<\n");
		}
		else if(term_3_){ //edge2contour
			PCL_WARN(">>>>>>>>>using term_3_<<<<<<<<<\n");
		}
	}

	double f2mod_total_time = 0
		,f2mkr_total_time = 0
		,e2c_total_time = 0
		,one_cycle_total_time = 0
		,misc1_total_time = 0
		,misc2_total_time = 0
		;

	for (int level_index = LEVELS-1; level_index>=0; --level_index){
		int iter_num = icp_iterations_[level_index];

		// current maps
		MapArr& vmap_curr = vmaps_curr_[level_index];
		MapArr& nmap_curr = nmaps_curr_[level_index];

		// previous maps
		MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
		MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

		for (int iter = 0; iter < iter_num; ++iter){
			Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
			float3& device_tcurr = device_cast<float3>(tcurr);

			Eigen::Matrix<float, 6, 1> result; //6*1 Rt 
			//Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
			//Eigen::Matrix<double, 6, 1> b;

			//配准目标1: frame2model
			tt0.tic();
			tt2.tic();
			estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
				vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());
			f2mod_total_time += tt2.toc();

			double det_f2mod = A_f2mod_.determinant();
			//printf("det_f2mod: %f\n", det_f2mod);

			if(with_nmap_ && this->isCuInitialized_){
//#if 01	//耿老师要求, 增加法向nmap 做配准惩罚项; 写成一个非线性式子没法解, 改成两步线性式子 @2017-6-1 12:35:21
				//cout << "estimateCombined @kf.orig: @A_f2mod_, b_f2mod_:\n" << A_f2mod_ << endl << b_f2mod_ << endl;
				//↓--其实已经是 A_f2mkr_ 了
				estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
					//vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());
					vmaps_cu_g_prev_[level_index], nmaps_cu_g_prev_[level_index], distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());

				double det_f2mod = A_f2mod_.determinant();

				if(fabs(det_f2mod) < 1e-15 ){
					printf("det_f2mod==0, align-failed, resetting....\n");
					return false;
				}

				result = A_f2mod_.llt().solve(b_f2mod_).cast<float>();

				float alpha1 = result (0);
				float beta1  = result (1);
				float gamma1 = result (2);

				//printf("level_index,iter:= (%d, %d)\n", level_index, iter);
				//printf("estimateCombined @kf.orig:a,b,r, tx, ty, tz::- %f, %f, %f, %f, %f, %f\n", alpha1, beta1, gamma1, result(3), result(4), result(5));

				Eigen::Matrix3f Rinc1 = (Eigen::Matrix3f)AngleAxisf (gamma1, Vector3f::UnitZ ()) * AngleAxisf (beta1, Vector3f::UnitY ()) * AngleAxisf (alpha1, Vector3f::UnitX ());
				Vector3f tinc1 = result.tail<3> ();

				//compose
				tcurr = Rinc1 * tcurr + tinc1;
				Rcurr = Rinc1 * Rcurr;

				device_Rcurr = device_cast<Mat33> (Rcurr);
				device_tcurr = device_cast<float3>(tcurr);

				//////////////////////////////////////////////////////////////////////////
				A_f2mod_.setZero();
				b_f2mod_.setZero();

				estimateCombined_nmap (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
					//vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());
					vmaps_cu_g_prev_[level_index], nmaps_cu_g_prev_[level_index], distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());

				//cout << "estimateCombined_nmap @A_f2mod_, b_f2mod_:\n" << A_f2mod_ << endl << b_f2mod_ << endl;
				//前面 6x6 矩阵填充只是为了接口统一, 实际这里只用前 3x3 矩阵, 用 66 反而导致 det(..)==0
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> A_f2mod_33 = A_f2mod_.block<3,3>(0,0);
				Eigen::Matrix<float, 3, 1> b_f2mod_31 = b_f2mod_.block<3,1>(0,0);

				det_f2mod = A_f2mod_33.determinant();
				if(fabs(det_f2mod) < 1e-15 ){
					printf("!!!estimateCombined_nmap::-- det_f2mod @A_f2mod_33==0, align-failed, resetting....\n");
					return false;
				}

#if 0   //之前思路是 argmin(SUM(|(R*ng~-ng)*(ng~-ng)|)), 放弃
				Eigen::Matrix<float, 3, 1> result31; //3*1 R, 不带 t
				result31 = A_f2mod_33.llt().solve(b_f2mod_31).cast<float>();

				alpha1 = result31 (0);
				beta1  = result31 (1);
				gamma1 = result31 (2);
				printf("estimateCombined_nmap:a,b,r::- %f, %f, %f\n", alpha1, beta1, gamma1);

				Rinc1 = (Eigen::Matrix3f)AngleAxisf (gamma1, Vector3f::UnitZ ()) * AngleAxisf (beta1, Vector3f::UnitY ()) * AngleAxisf (alpha1, Vector3f::UnitX ());
				//tinc1 = result.tail<3> (); //这里不更新 t

#elif 1 //改用 orthogonal-procrustes 思路
				JacobiSVD<Matrix3f> svd(A_f2mod_33, ComputeFullU | ComputeFullV);
				Matrix3f svdU = svd.matrixU();
				Matrix3f svdV = svd.matrixV();
				Rinc1 = svdU * svdV.transpose(); //
				CV_Assert(Rinc1.determinant() > 1e-6);

#endif
				//if(!(level_index == 0 && iter_num == icp_iterations_[level_index] - 1)){
				//if(level_index > 0 ){
				if(term_123_){
					//compose
					//tcurr = Rinc1 * tcurr + tinc1;
					Rcurr = Rinc1 * Rcurr;
				}

				continue; //跳到下次循环, 即实验独立于之前代码
//#endif
			}//if-(with_nmap_ && this->isCuInitialized_)

			if(!this->isCuInitialized_)
				//|| level_index > 0) //改设定, 仅当最底层金字塔才做 A_f2mkr_, A_e2c_, 试图提升效率; 结果: 60ms->40ms @2017-10-7 16:09:03
				//↑-不行, owl, fid<200 会飘
			{
			//if(1){
				if(level_index == 0 && iter == 0)
					PCL_WARN("this->isCuInitialized_--FALSE, using kf.orig\n");
				//checking nullspace
				//double det_f2mod = A_f2mod_.determinant();

				if (fabs (det_f2mod) < 1e-15 || pcl_isnan (det_f2mod))
				{
					if (pcl_isnan (det_f2mod)) cout << "qnan" << endl;
					printf("RESETTING... (level_index, iter):= (%d, %d); det_f2mod, %f\n", level_index, iter, det_f2mod);
					cout<<"A_f2mod_:"<<A_f2mod_<<endl;

					reset ();
					return (false);
				}

				result = A_f2mod_.llt().solve(b_f2mod_).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
			}
			else{ //if-(this->isCuInitialized_)
				//ScopeTime time("icp-A_f2mkr_-A_e2c_"); //不要在循环内

				tt2.tic();
				A_f2mkr_.setZero(); //因为 estimateCombined 是赋值, 不是 "+=" 所以其实不必; 此处仅做保险
				b_f2mkr_.setZero();

				if((term_123_ || term_12_ || term_23_ || term_2_)
					&& iter % sample_step_ == 0) //借用 sample_step_, 其本来是用于 synCuDmapBA 降采样的 @2019-1-25 10:24:25
				{
					//ScopeTime time("icp-A_f2mkr_");
					//配准目标2: frame2marker (3d cuboid fiducial marker)

					//↓--改放 if外面, 因为 if不总是执行
					//A_f2mkr_.setZero(); //因为 estimateCombined 是赋值, 不是 "+=" 所以其实不必; 此处仅做保险
					//b_f2mkr_.setZero();
					estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(level_index),
						//vmaps_cu_g_prev_[level_index], nmaps_cu_g_prev_[level_index], distThres_, angleThres_, gbuf_f2mkr_, sumbuf_f2mkr_, A_f2mkr_.data(), b_f2mkr_.data()); //没用
						vmaps_cu_g_prev_[level_index], nmaps_cu_g_prev_[level_index], distThres_, angleThres_, gbuf_, sumbuf_, A_f2mkr_.data(), b_f2mkr_.data());
					//vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_f2mkr_.data(), b_f2mkr_.data());
				}//if-term_123_ or xx2_ //ScopeTime "icp-A_f2mkr_"
				f2mkr_total_time += tt2.toc();

				tt2.tic();
				A_e2c_.setZero(); //因为 "+=", 所以必须!!!
				b_e2c_.setZero();

				int ll = 0; //匹配点计数器
				if((term_123_ || term_13_ || term_23_ || term_3_)
					//&& iter % sample_step_ == 0) //icp4: 借用 sample_step_, 其本来是用于 synCuDmapBA 降采样的 @2019-1-25 10:24:25
					//&& iter == 0) //icp4.2: 改用仅每次内循环开始时用 e2c, 结果long4 上BAD	@2019-1-25 17:00:53
					&& iter == iter_num - 1) //icp4.3: 改用每次内循环终止时用 e2c; 已验证: long4上 OK, 且增大权重好一点-cuAmp 4,4->4,24	@2019-1-25 17:05:56
				{
					//ScopeTime time("icp-A_e2c_");
					//配准目标3: edge2cont; 这里不受金字塔影响, 始终 640*480
					pcl::PointCloud<pcl::PointXYZ>::Ptr maskedpts_g(new pcl::PointCloud<pcl::PointXYZ>);
					Eigen::Affine3f cam2g;
					cam2g.linear() = Rcurr;
					cam2g.translation() = tcurr;
					pcl::transformPointCloud(*maskedpts, *maskedpts_g, cam2g);
					vector<int> pointIdxNKNSearch(1);
					vector<float> pointNKNSquaredDistance(1);
					const float kdtreeDistTh = e2c_dist_; //5cm

					//↓--改放 if外面, 因为 if不总是执行
					//A_e2c_.setZero(); //因为 "+=", 所以必须!!!
					//b_e2c_.setZero();

					//int ll = 0; //匹配点计数器 //移到 ScopeTime 外
					for(int k = 0; k < maskedpts_g->size(); k++){
						cuEdgeKdtree_.nearestKSearch(maskedpts_g->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance);


						if(pointNKNSquaredDistance[0] < kdtreeDistTh * kdtreeDistTh){
							int ptIdx = pointIdxNKNSearch[0];

							//与 bdr 不同, 这里 kdtree 没有 norm信息, 所以先去nmap 上确定法向:
							PointXYZ pt_d = cuContCloud_->points[ptIdx]; //dst
							Normal pt_d_n = cuContCloudNormal_->points[ptIdx];
							if(pcl_isnan(pt_d_n.normal_x)) //可能触发, 已验证
								continue;

							PointXYZ pt_s = maskedpts_g->points[k]; //src

							//用 eigen 是因为 pcl 没有所需的运算
							Eigen::Vector3f nn(pt_d_n.normal_x, pt_d_n.normal_y, pt_d_n.normal_z);
							Eigen::Vector3f qq(pt_d.x, pt_d.y, pt_d.z);
							Eigen::Vector3f pp(pt_s.x, pt_s.y, pt_s.z);

							float r = nn.dot(qq - pp);
							Eigen::Vector3f pxn = pp.cross(nn);
							float xx[6] = {pxn(0), pxn(1), pxn(2), nn(0), nn(1), nn(2)};

							//zc: dbg
							if(pcl_isnan(r)){
								printf("k, ptIdx: %d, %d\n", k, ptIdx);
								cout << "nn: " << nn << endl
									<< "qq: " << qq << endl
									<< "pp: " << pp << endl;
							}

							for(int ii = 0; ii < 6; ii++){
								for(int jj = 0; jj < 6; jj++){
									A_e2c_(ii, jj) += xx[ii] * xx[jj];
								}
								b_e2c_(ii) += xx[ii] * r;
							}
							ll++;
						}//dist < distTh
						else if(0){ //DBG: zcRenderCubeDmap 用sample_step_ 的后遗症 @2019-1-24 21:28:48
							PointXYZ pt_s = maskedpts_g->points[k]; //src
							printf("pt_s_%d:= (%f, %f, %f); pointNKNSquaredDistance[0]: %f, \n", k, pt_s.x, pt_s.y, pt_s.z, pointNKNSquaredDistance[0]);
						}
					}//for-each-edge-point
				}//ScopeTime "icp-A_e2c_"
				e2c_total_time += tt2.toc();

				tt2.tic(); //misc1...
				//double det_f2mod = A_f2mod_.determinant(),
				double det_f2mkr = A_f2mkr_.determinant(),
						det_e2c = A_e2c_.determinant();

				//↓--是"与", 非"或"; 只要有一个 det 有效, 就继续求解, 而不算失败
				if(fabs(det_f2mod) < 1e-15 
					&& fabs(det_f2mkr) < 1e-15 
					&& fabs(det_e2c) < 1e-15 && ll < 10){
						printf("RESETTING... (level_index, iter):= (%d, %d); det_f2mod, det_f2mkr, det_e2c, ll: %f, %f, %f, %d\n", level_index, iter, det_f2mod, det_f2mkr, det_e2c, ll);
						return false;
				}

				//const float amplifier_e2c = 10; //以下改借用 -bdr_amp :: amplifier_
				//三项组合 (默认):
				if(term_123_){
					result = (A_f2mod_ + w_f2mkr_ * A_f2mkr_ + e2c_weight_ * A_e2c_).llt()
						.solve(b_f2mod_ + w_f2mkr_ * b_f2mkr_ + e2c_weight_ * b_e2c_).cast<float>();
					if(dbgKf_ > 1 && 0 == level_index && iter == iter_num - 1)
						PCL_WARN("\tw_f2mkr_: %f, e2c_weight_: %f, ll: %d\n", w_f2mkr_, e2c_weight_, ll);
				}
				//两项组合: 
				else if(term_12_){ //f2mod+f2mkr
					result = (A_f2mod_ + w_f2mkr_ * A_f2mkr_)
						.llt().solve(b_f2mod_ + w_f2mkr_ * b_f2mkr_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_12_<<<<<<<<<\n");
				}
				else if(term_13_){
					result = (A_f2mod_ + e2c_weight_ * A_e2c_)
						.llt().solve(b_f2mod_ + e2c_weight_ * b_e2c_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_13_<<<<<<<<<\n");
				}
				else if(term_23_){
					result = (w_f2mkr_ * A_f2mkr_ + e2c_weight_ * A_e2c_)
						.llt().solve(w_f2mkr_ * b_f2mkr_ + e2c_weight_ * b_e2c_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_23_<<<<<<<<<\n");
				}
				//若仅用一项测试
				else if(term_1_){ //kf
					result = A_f2mod_.llt().solve(b_f2mod_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_1_<<<<<<<<<\n");
				}
				else if(term_2_){ //f2cuboid
					result = A_f2mkr_.llt().solve(b_f2mkr_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_2_<<<<<<<<<\n");
				}
				else if(term_3_){ //edge2contour
					result = A_e2c_.llt().solve(b_e2c_).cast<float>();
					//PCL_WARN(">>>>>>>>>using term_3_<<<<<<<<<\n");
				}

#if 0	//zc: dbg: 调试分别只用 f2mkr, e2c 做优化目标, 相机姿态求解结果
				result = A_f2mod_.llt ().solve (b_f2mod_).cast<float>();

				Eigen::Matrix<float, 6, 1> result_f2mkr = (A_f2mkr_).llt().solve(b_f2mkr_).cast<float>();; //6*1 Rt 
				Eigen::Matrix<float, 6, 1> result_e2c = (A_e2c_).llt().solve(b_e2c_).cast<float>();; //6*1 Rt 

				result = result_f2mkr; //测试只用 f2mkr 误差项
				result = result_e2c; //测试只用 e2c 误差项

				if(level_index == 0 && iter == 0){
					cout << "A_f2mod_, b_f2mod_:\n" << A_f2mod_ << endl << b_f2mod_ <<endl;
					cout << "A_f2mkr_, b_f2mkr_:\n" << A_f2mkr_ << endl << b_f2mkr_ <<endl;
					cout << "A_e2c_, b_e2c_:\n" << A_e2c_ << endl << b_e2c_ <<endl;

					cout << "\tresult:\n" << result << endl
						<< "\tresult_f2mkr:\n" << result_f2mkr <<endl
						<< "\tresult_e2c:\n" << result_e2c <<endl;
				}
#endif
				misc1_total_time += tt2.toc();
			}//if-isCuInitialized_

			tt2.tic();
			float alpha = result (0);
			float beta  = result (1);
			float gamma = result (2);

			Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
			Vector3f tinc = result.tail<3> ();

			//if(global_time_ == 144){ //dbg tsdf-v18 Kinect_Bunny_Handheld
			if(global_time_ == dbgFid_){
				printf("global_time_:= %d;... (level_index, iter):= (%d, %d); det_f2mod, %f\n", global_time_, level_index, iter, det_f2mod);
				cout<<"A_f2mod_:"<<A_f2mod_<<endl;
				cout<<"a,b,gamma: "<<alpha<<", "<<beta<<", "<<gamma<<"; "
					<<"tinc: "<<tinc.transpose()<<endl;

				//if(level_index == 0 && iter > 5)
				if(level_index == dbgLvlIdx_ && iter == dbgIter_){
					doLvlIterBreak = true;
					break;
				}
			}

			//compose
			tcurr = Rinc * tcurr + tinc;
			Rcurr = Rinc * Rcurr;

			misc2_total_time += tt2.toc();
			one_cycle_total_time += tt0.toc();
		}//iter
		if(doLvlIterBreak)
			break;
	}//level_index

	double icp3sum = f2mod_total_time + f2mkr_total_time + e2c_total_time;
	printf("f2mod_total_time, f2mkr_total_time, e2c_total_time: %f, %f, %f; icp3sum= %f\n", f2mod_total_time, f2mkr_total_time, e2c_total_time, icp3sum);
	printf("\tmisc1_2_total_time: %f, %f; one_cycle_total_time: %f\n", misc1_total_time, misc2_total_time, one_cycle_total_time);

	printf("ScopeTime-icp-all "); tt1.toc_print();
	}//ScopeTime-icp-all

	//save tranform
	rmats_.push_back (Rcurr);
	tvecs_.push_back (tcurr);

	if(this->isCuInitialized_){
		ScopeTime time("this->isCuInitialized_-planeFilter");

		//zc: 调试, 拷贝自上面 *cubeCamBA* 代码段 @2017-6-20 09:37:37
		Eigen::Affine3f cam2g = this->getCameraPose(); //移植到 kinfuLS 之后, 可能存在 shift 导致出错的风险, 暂不处理 @2017-4-5 11:29:23
		Eigen::Affine3d cam2gd = cam2g.cast<double>();
		Cube cubeCamAA(cube_g_, cam2gd.inverse());
		if(dbgKf_ >= 1){
			cubeCamAA.drawContour(m8uc3, fx_, fy_, cx_, cy_, cv::Scalar(0,0,255)); //红色轮廓图
			cv::imshow("m8uc3", m8uc3);

			if(dbgKf_ == 2){
				char fnBuf[80] = {0};
				sprintf(fnBuf, "m8uc3-%06d.png", callCnt_);
				cv::imwrite(fnBuf, m8uc3);
			}
		}

		Eigen::Vector4f planeFiltParam_c;
		Eigen::Vector3d nvec, nvec_c;
		nvec << planeFiltParam_.x(), planeFiltParam_.y(), planeFiltParam_.z(); //此时还是 g-coo
		nvec_c = cam2gd.rotation().transpose() * nvec;
		double D_c = planeFiltParam_.w() + nvec.dot(cam2gd.translation());
		planeFiltParam_c << nvec_c.x(), nvec_c.y(), nvec_c.z(), D_c;

		float4 &device_plparam = device_cast<float4>(planeFiltParam_c);
		//printf("device_plparam.xyzw: %f, %f, %f, %f\n", device_plparam.x, device_plparam.y, device_plparam.z, device_plparam.w); //√

		if(isPlFilt_){
		device::planeFilter(depth_raw, intr, device_plparam, depthPlFilt);
		cv::Mat depthPlFiltHost(depth_raw.rows(), depth_raw.cols(), CV_16UC1),
			depthPlFiltHost8u;
		depthPlFilt.download(depthPlFiltHost.data, depthPlFilt.colsBytes());
		depthPlFiltHost.convertTo(depthPlFiltHost8u, CV_8UC1, UCHAR_MAX/3e3);
		imshow("depthPlFiltHost8u", depthPlFiltHost8u);
		}

		//+++++++++++++++图优化相机轨迹, 后处理; 发生在所有配准之后, integrate 之前 @2017-11-13 00:53:03
		static CameraPoseWriter graphOptPoseWriter("zcpose-ahc-g2o.csv");

		static bool isGraphOptBegin = false;
		static Affine3fVec camPoseVec2opt;
		static Affine3fVec camDeltaVec; //这个是 icp 求得的增量们, 用作 edge.measurement

		Affine3f cam2g_i = this->getCameraPose(), //==cam2g
			cam2g_i_1 = this->getCameraPose(this->rmats_.size()-2);

		//Matrix3f dR = Rprev.transpose() * Rcurr; //i→(i-1)
		//Vector3f dt = Rprev.transpose() * (tcurr - tprev);
		//camPoseDelta_.linear() = dR;
		//camPoseDelta_.translation() = dt;
		Affine3f camPoseDelta = cam2g_i_1.inverse() * cam2g_i; //Ci->C(i-1); aff 直接 inverse 可以？

		//tt2.tic();
		bool isFoundCuCrnr = cube_g_.isTrihedronVisible(cam2g.cast<double>(), fx_, fy_, cx_, cy_, dbgKf_ >= 1); //这个 cam2g 是当前帧配准求解之后的
		//printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@isFoundCuCrnr: %s; took: ", isFoundCuCrnr ? "TRUE" : "FALSE"); tt2.toc_print(); //1ms, 效率可以

		if(!isFoundCuCrnr){ //当前帧没
			if(isLastFoundCrnr_){//上一帧有, 刚刚跟丢, 要初始化图优化构造
				if(dbgKf_ >= 1)
					printf("**************************************************P(T)-C(F); isGraphOptBegin=true\n");

				isGraphOptBegin = true; //标记启动

				g2oOptmizer_.clear(); //是否有副作用? 未验证 2017-1-5 02:09:55

				camPoseVec2opt.clear();
				camDeltaVec.clear();

				camPoseVec2opt.push_back(cam2g_i_1); //[0]-记得需要fix
			}

			//不管上一帧有无
			if(isGraphOptBegin){
				camPoseVec2opt.push_back(cam2g_i);
				camDeltaVec.push_back(camPoseDelta); //dT 这里才开始填充
			}
			else{//!isGraphOptBegin     //若没开始图优化; 普通ICP, 很可能是刚开始扫描的时段
				//graphOptPoseWriter.processPose(cam2g_i); //写文件
				//DO-NOTHING, 逻辑改为: curr=T,prevSaved=F 时, 前面的统一 save
			}

			isLastFoundCrnr_ = false; //用完
		}
		else{//if-isFoundCuCrnr //当前帧有
			//static bool 控制, 第一次进入此逻辑时, 会把之前的 poses 序列预先全部存储 (这部分不做图优化)
			//此逻辑也适合放在 *当前帧有, 上一帧没有* 段内, 暂不动
			static bool isPrevPosesSaved = false;
			if(dbgKf_ >= 1)
				printf("**************************************************isPrevPosesSaved:= %s\n", isPrevPosesSaved ? "TTT" : "FALSE");

			if(false == isPrevPosesSaved){ //不管上一帧有无 crnr, 因为此逻辑块仅执行一次
				for(size_t i=0; i<rmats_.size(); i++){
					Affine3f cam2g_i = this->getCameraPose(i); //local, 外面还有一个
					graphOptPoseWriter.processPose(cam2g_i);
				}

				isPrevPosesSaved = true;
			}

			if(isLastFoundCrnr_){ //上一帧也有
				if(dbgKf_ >= 1)
					printf("**************************************************P(T)-C(T)\n");

				//不push, 只更新 back: //改: DO-NOTHING
				if(camPoseVec2opt.size() > 0)
					camPoseVec2opt.back() = cam2g_i;
				else
					camPoseVec2opt.push_back(cam2g_i);

				graphOptPoseWriter.processPose(cam2g_i); //写文件
			}
			else{ //if-!isLastFoundCrnr_ //当前帧有, 上一帧没有
				if(dbgKf_ >= 1)
					printf("**************************************************P(F)-C(T); isGraphOptBegin:= %s\n", isGraphOptBegin ? "TTT" : "FALSE");

				if(!isGraphOptBegin){ //未启动 graphOpt, 原因举例比如: ①程序开始时没定位到立方体, 第一次定位时, 进入此段
					//graphOptPoseWriter.processPose(cam2g_i); //写文件
					//DO-NOTHING, 因为 isPrevPosesSaved 已经控制 save 过了
				}
				else{ //if-isGraphOptBegin //启动过 graphOpt, 说明无立方体空窗期结束, 触发当前段图优化+++++++++++++++
					isGraphOptBegin = false;

					//填充最后一个 vert, 记得需要fix
					camPoseVec2opt.push_back(cam2g_i);
					camDeltaVec.push_back(camPoseDelta);

					//add-verts
					size_t vertCnt = camPoseVec2opt.size();
					for(size_t i=0; i<vertCnt; i++){
						Affine3d &pose_i = camPoseVec2opt[i].cast<double>();

						g2o::VertexSE3 *vi = new g2o::VertexSE3;
						vi->setId(i);
						if(i == 0 || i == vertCnt - 1) //首尾帧做固定点
							vi->setFixed(true);
						else
							vi->setFixed(false); //中间帧, 显示设定 false, 不管 g2o 默认怎样

						//vi->setEstimate(pose_i.matrix()); //报错C2664: 'g2o::BaseVertex<D,T>::setEstimate' : cannot convert parameter 1 from 'Eigen::Matrix<_Scalar,_Rows,_Cols>' to 'const Eigen::Transform<_Scalar,_Dim,_Mode> &'
						vi->setEstimate(g2o::Isometry3D(pose_i.matrix()) );
						g2oOptmizer_.addVertex(vi);
					}

					//add-edges
					for(size_t i=1; i<vertCnt; i++){
						//Affine3f &pose_i_1_from = camPoseVec2opt[i-1];
						//Affine3f &pose_i_to = camPoseVec2opt[i];
						Affine3d &dPose = camDeltaVec[i-1].cast<double>();

						g2o::EdgeSE3 *edge = new g2o::EdgeSE3;
						//edge->setVertex(0, pose_i_1_from); //胡扯
						//edge->setVertex(1, pose_i_to);
						edge->setVertex(0, g2oOptmizer_.vertex(i-1));
						edge->setVertex(1, g2oOptmizer_.vertex(i));
						//edge->setMeasurement(dPose.matrix()); //报错 C2664: 'g2o::EdgeSE3::setMeasurement' : cannot convert parameter 1 from 'Eigen::Matrix<_Scalar,_Rows,_Cols>' to 'const g2o::Isometry3D &'
						edge->setMeasurement(g2o::Isometry3D(dPose.matrix()) ); 
						//edge->setInformation(g2o::Isometry3D::Identity()); //报错C2664: cannot convert parameter 1 from 'const Eigen::Transform<_Scalar,_Dim,_Mode>' to 'const Eigen::Matrix<_Scalar,_Rows,_Cols,_Options> &'
						typedef Eigen::Matrix<double, 6, 6> Matrix66;
						Matrix66 info = Matrix66::Identity();
						edge->setInformation(info);
						edge->setRobustKernel(new g2o::RobustKernelHuber);

						g2oOptmizer_.addEdge(edge);
					}

					static int g2oTimes = 0;

					pcl::console::TicToc tt;
					tt.tic();
					g2oOptmizer_.initializeOptimization();
					g2oOptmizer_.optimize(30);
					printf("g2oOptmizer_.optimize, times:= %d, TOOK: ", g2oTimes); tt.toc_print();

					char buf[4096];
					sprintf(buf, "kf-cu-g2o_%d.g2o", (int)g2oTimes);

					g2oOptmizer_.save(buf);
					g2oTimes++;

					for(size_t i=1; i<vertCnt; i++){ //i=1开始, [0] 已在 isLastFoundCrnr_ && isFoundCrnr 时写文件过了
						g2o::VertexSE3 *vi = dynamic_cast<g2o::VertexSE3*>(g2oOptmizer_.vertex(i));
						g2o::Affine3D pose_i;
						pose_i = vi->estimate();

						graphOptPoseWriter.processPose(pose_i.cast<float>()); //写文件
					}
				}//if-isGraphOptBegin
			}//if-isLastFoundCrnr_-else

			isLastFoundCrnr_ = true;

		}//if-!isFoundCrnr-else

	}//if-(this->isCuInitialized_)

	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
	float tnorm = (tcurr - tprev).norm();  
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;  
	//zc: 调试, 总是融合:
	integrate = true;
	integrate = is_integrate_;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration

	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);

	//zc: 调试观察: @2018-9-30 10:09:30
	//1, 配准后生成的 dmap 是否与 d-curr 吻合? 对!
	//2, rcFlag 什么样子

	//定点调试观察某体素:
	int3 vxlPos;
	vxlPos.x = vxlDbg_[0];
	vxlPos.y = vxlDbg_[1];
	vxlPos.z = vxlDbg_[2];

// 	tt1.tic();
// 	//DepthMap rcFlagMap_;
// 	rcFlagMap_.create(rows_, cols_);
// 	raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), 
// 		getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_, wmap_, rcFlagMap_, vxlPos);
// 	printf("raycast_rcFlagMap_: "); tt1.toc_print(); //0ms
// 	tt1.tic();
// 	cv::Mat rcFlag16s(480, 640, CV_16S);
// 	cv::Mat rcFlag8uc3(480, 640, CV_8UC3, cv::Scalar(0));
// 	rcFlagMap_.download(rcFlag16s.data, rcFlagMap_.colsBytes());
// #if 0 //rcFlagMap_ 确实存几个 flag 的情形
// 	//rcFlag16s.convertTo(rcFlag8u, CV_8U);
// 	//cv::imshow("rcFlag8u", rcFlag8u);
// 	rcFlag8uc3.setTo(cv::Scalar(255, 255, 255), rcFlag16s == 255);
// 	rcFlag8uc3.setTo(cv::Scalar(255, 0, 0), rcFlag16s == 127);	//蓝色, ->+区域
// 	rcFlag8uc3.setTo(cv::Scalar(0, 255, 0), rcFlag16s == 128);	//绿色, 0>-区域
// 	rcFlag8uc3.setTo(cv::Scalar(0, 0, 255), rcFlag16s == 129);	//红色, ->0区域
// 	rcFlag8uc3.setTo(cv::Scalar(255, 0, 255), rcFlag16s == 130);	//洋红色, 0>->0区域
// #elif 1 //rcFlagMap_ 改存正负深度值后
// 	cv::Mat rcFlag8u_posi(rcFlag16s.size(), CV_8U, cv::Scalar(0));
// 	cv::Mat rcFlag8u_nega(rcFlag16s.size(), CV_8U, cv::Scalar(0));
// 	cv::Mat tmp16s = rcFlag16s.clone();
// 	tmp16s.setTo(0, tmp16s < 0);
// 	tmp16s.convertTo(rcFlag8u_posi, CV_8U, UCHAR_MAX/1500.);
// 
// 	tmp16s = rcFlag16s.clone() * -1; //取反
// 	tmp16s.setTo(0, tmp16s < 0);
// 	tmp16s.convertTo(rcFlag8u_nega, CV_8U, UCHAR_MAX/1500.);
// 	
// 	cv::Mat tmp_channel3[3] = {cv::Mat::zeros(rcFlag16s.size(), CV_8U), rcFlag8u_posi, rcFlag8u_nega};
// 	cv::merge(tmp_channel3, 3, rcFlag8uc3);
// 
// 	char filename[ 1024 ];
// 	std::sprintf( filename, "image/rcFlag/%06d.png", global_time_ + 1 );
// 	cv::imwrite(filename, rcFlag8uc3);
// #endif
// 	cv::imshow("rcFlag8uc3", rcFlag8uc3);
// 	printf("rcFlag8uc3 "); tt1.toc_print(); //~21ms
// 
// 	tt1.tic();
// 	//DepthMap depthRcast_;
// 	generateDepth(device_Rcurr_inv, device_tcurr, vmap_g_model_, dmapModel_);
// 	vector<unsigned short> depthRcast_host_data;
// 	int c;
// 	dmapModel_.download(depthRcast_host_data, c);
// 	cv::Mat depthRcast_host16u( 480, 640, CV_16UC1, (void *)&depthRcast_host_data[0] );
// 	cv::Mat depthRcast_host8u;
// 	depthRcast_host16u.convertTo(depthRcast_host8u, CV_8U, UCHAR_MAX/1500.);
// 	cv::imshow("depthRcast@after_icp", depthRcast_host8u);
// 	printf("depthRcast_host8u "); tt1.toc_print();
	
	if (integrate)
	{
		ScopeTime time("if-integrate");

		//ScopeTime time("tsdf");
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
#pragma region NOT tv18
		if(isTsdfVer(2)){
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_);
			integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_, vxlPos);
		}
		else if(isTsdfVer(11)){
#define USE_BDR_NMAP_METHOD_ 0  //用 bdr 求解 nmap 方法, 否则, 用 PCA 求 nmap
			//tsdf-v11.9: 改用 PCA 求 nmap

			raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_);

			float3 t_tmp;
			t_tmp.x=t_tmp.y=t_tmp.z=0;
#if USE_BDR_NMAP_METHOD_
			//2.1, vmap_g_model_ 在 T(i) 视角的投射, 对应的 nmap
			//cpu 上 sobel 计算梯度
			//DepthMap dmapModel_, dmapModel_inp_;
			device::generateDepth(device_Rcurr_inv, device_tcurr, vmap_g_model_, dmapModel_); //model
			//zc::inpaintGpu(dmapModel, dmapModel_inp); //深度图修补 inpaint, 这里 src-dst 同变量; 懒得 //改了, 俩变量

			//移植时候改逻辑: 不要 inpaint depth-model,	@2017-4-9 15:34:41
			cv::Mat dmapModel_host(dmapModel_.rows(), dmapModel_.cols(), CV_16UC1);
			dmapModel_.download(dmapModel_host.data, dmapModel_.colsBytes());
			cv::Mat gradu, gradv;
			cv::Sobel(dmapModel_host, gradu, CV_32F, 1,0, 7, 1./1280);
			cv::Sobel(dmapModel_host, gradv, CV_32F, 0,1, 7, 1./1280);
			//MapArr gx_device_, gy_device_;
			gx_device_.upload(gradu.data, gradu.cols * gradu.elemSize(), gradu.rows, gradu.cols);
			gy_device_.upload(gradv.data, gradv.cols * gradv.elemSize(), gradv.rows, gradv.cols);
			//MapArr nmap_model_;
			device::computeNormalsContourcue(intr, dmapModel_, gx_device_, gy_device_, nmap_model_);
			//一定记得 nmap 转到世界坐标系
			zc::transformVmap(nmap_model_, device_Rcurr, t_tmp, nmap_g_model_); //借用 
#else
			//DO-NOTHING, 因为 raycast 已经得到了 nmap_g_model_
#endif

			//pcl::device::MaskMap largeIncidMask_model_; //大入射角mask, vmap_g_model_ 的
			zc::contourCorrespCandidate(device_tcurr, vmap_g_model_, nmap_g_model_, this->incidAngleThresh_, largeIncidMask_model_);

			//2.2, dmap-bilateral-filt 对应的 nmap
#if USE_BDR_NMAP_METHOD_
			cv::Mat dcurrFiltHost(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
			depths_curr_[0].download(dcurrFiltHost.data, depths_curr_[0].colsBytes()); //用滤波 dmap 替换 raw, 观察
			cv::Sobel(dcurrFiltHost,gradu,CV_32F,1,0,7,1.0/1280); //借用 gradu 变量
			cv::Sobel(dcurrFiltHost,gradv,CV_32F,0,1,7,1.0/1280);
			gx_device_.upload(gradu.data, gradu.cols * gradu.elemSize(), gradu.rows, gradu.cols);
			gy_device_.upload(gradv.data, gradv.cols * gradv.elemSize(), gradv.rows, gradv.cols);
			//MapArr nmap_filt_, nmap_filt_g_;
			device::computeNormalsContourcue(intr(0), depths_curr_[0], gx_device_, gy_device_, nmap_filt_);
#else
			nmap_filt_ = nmaps_curr_[0];
#endif
			zc::transformVmap(nmap_filt_, device_Rcurr, t_tmp, nmap_filt_g_); //对 curr 就用 local, 不用 global, 

#if 0	//上面 USE_BDR_NMAP_METHOD_=0 之后, nmap_filt_, nmap_g_model_ 都是 PCA 得到, 但是 incidMask 仍想用 bdr 得到
			//pcl::device::MaskMap largeIncidMask_curr_; //大入射角mask, dmap-curr 的
			zc::contourCorrespCandidate(t_tmp, vmaps_curr_[0], nmap_filt_, this->incidAngleThresh_, largeIncidMask_curr_); //不用 device_tcurr, 不用 nmap..._g

			//2.3 俩 mask 叠加
			//做 cv-close 操作, 消除微小黑洞
			cv::Mat largeIncidMsk_model_host(largeIncidMask_model_.rows(), largeIncidMask_model_.cols(), CV_8UC1);
			largeIncidMask_model_.download(largeIncidMsk_model_host.data, largeIncidMask_model_.colsBytes());

			cv::Mat largeIncidMask_curr_host(largeIncidMask_curr_.rows(), largeIncidMask_curr_.cols(), CV_8UC1);
			largeIncidMask_curr_.download(largeIncidMask_curr_host.data, largeIncidMask_curr_.colsBytes());

			int krnlRad = 3;
			cv::Mat krnlMat = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*krnlRad + 1, 2*krnlRad+1 ), cv::Point( krnlRad, krnlRad ) );

			cv::Mat dcurrFiltHostEroded;
			cv::erode(dcurrFiltHost, dcurrFiltHostEroded, krnlMat);
			cv::Mat largeIncidMask_total_host = largeIncidMask_curr_host + (dcurrFiltHostEroded == 0 & largeIncidMsk_model_host);

			krnlRad = 1;
			krnlMat = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*krnlRad + 1, 2*krnlRad+1 ), cv::Point( krnlRad, krnlRad ) );
			//cv::morphologyEx(largeIncidMask75_total_host, largeIncidMask75_total_host, cv::MORPH_CLOSE, krnlMat);
			//cv::morphologyEx(largeIncidMask_total_host, largeIncidMask_total_host, cv::MORPH_DILATE, krnlMat); //直接膨胀
			//↑--tsdf-v11.3 去掉 morphologyEx, 避免逻辑太 tricky

			//pcl::device::MaskMap largeIncidMask_total_;
			largeIncidMask_total_.upload(largeIncidMask_total_host.data, largeIncidMask_model_.colsBytes(), largeIncidMask_model_.rows(), largeIncidMask_model_.cols());
#elif 1	//bdr 获取 incidMask
			cv::Mat dcurrFiltHost(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
			depths_curr_[0].download(dcurrFiltHost.data, depths_curr_[0].colsBytes()); //用滤波 dmap 替换 raw, 观察
			cv::Mat gradu, gradv;
			cv::Sobel(dcurrFiltHost,gradu,CV_32F,1,0,7,1.0/1280); //借用 gradu 变量
			cv::Sobel(dcurrFiltHost,gradv,CV_32F,0,1,7,1.0/1280);
			gx_device_.upload(gradu.data, gradu.cols * gradu.elemSize(), gradu.rows, gradu.cols);
			gy_device_.upload(gradv.data, gradv.cols * gradv.elemSize(), gradv.rows, gradv.cols);
			//MapArr nmap_filt_, nmap_filt_g_;
			MapArr nmap_bdr;
			device::computeNormalsContourcue(intr(0), depths_curr_[0], gx_device_, gy_device_, nmap_bdr);
			zc::contourCorrespCandidate(t_tmp, vmaps_curr_[0], nmap_bdr, this->incidAngleThresh_, largeIncidMask_total_); //不用 device_tcurr, 不用 nmap..._g
#endif
			if(dbgKf_ >= 1){
			cv::Mat incidMskHost(largeIncidMask_total_.rows(), largeIncidMask_total_.cols(), CV_8UC1);
			largeIncidMask_total_.download(incidMskHost.data, incidMskHost.cols * incidMskHost.elemSize());
			imshow("incidMskHost", incidMskHost);
			}

			cv::Mat edgeDistMap;
			cv::distanceTransform(md_mask==0, edgeDistMap, CV_DIST_L1, 3);
			if(dbgKf_ >= 1){
				cv::Mat edgeMatShow;
				cv::normalize(edgeDistMap, edgeMatShow, 0, 1, cv::NORM_MINMAX);
				imshow("edgeMatShow", edgeMatShow);
			}
			//DeviceArray2D<float> edgeDistMap_device_;
			edgeDistMap_device_.upload(edgeDistMap.data, edgeDistMap.cols * edgeDistMap.elemSize(), edgeDistMap.rows, edgeDistMap.cols);

			//为 v11.4 生成, 但是放在外面, 以备其他版本需要 @2017-3-13 17:06:01
			//weight-map, 按: 1, 入射角cos; 2, D(u); 3, 到边缘距离 来加权
			//MapArr wmap_;
			//zc::calcWmap(vmaps_curr_[0], nmap_filt, thickContMsk, wmap);
			zc::calcWmap(vmaps_curr_[0], nmap_filt_, edgeDistMap_device_, fx_, wmap_); //for tsdf-v11.8
			if(dbgKf_ >= 1)
			{
				cv::Mat wmapHost(wmap_.rows(), wmap_.cols(), CV_32F);
				wmap_.download(wmapHost.data, wmap_.colsBytes());
				cv::Mat wmapHost8u;
				wmapHost.convertTo(wmapHost8u, CV_8UC1, UCHAR_MAX); //*255, 因为 wmap 本身 ~(0,1)
				imshow("wmapHost8u", wmapHost8u);
			}

			tt1.tic();
			integrateTsdfVolume_v11(isPlFilt_ ? depthPlFilt : depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(),
				tsdf_volume_->volume2nd_, tsdf_volume_->flagVolume_, tsdf_volume_->surfNormPrev_, tsdf_volume_->vrayPrevVolume_, 
				largeIncidMask_total_,
				//largeIncidMask_model_, //model 为啥放弃?:: 因为边缘也是不稳定区域, 希望与大入射角区域同等处理 @2017-12-3 19:35:27
				//答: tv11 里, 暂时 total 好一点 (其实是 nmap_bdr 求解得到); 后续 tv12 存疑
				/*nmap_total_g, */ 
				//nmap_g_model_, //到底用哪个? 不好弄
				nmap_filt_g_, 
				nmap_g_model_, //v11.6
				wmap_,
				depthRawScaled_, vxlPos);
			printf("integrateTsdfVolume_v11 "); tt1.toc_print(); //很快, 1ms

			//zc: 单用 kf-odo, 但是 tv11.10, 在 roma 数据上 f1110 漂移, 改成 tv2 就没事, 原因不明, @2017-4-14 17:25:09
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_, vxlPos);

		}//elif-isTsdfVer(11)
#pragma endregion 
		else if(isTsdfVer(12) || isTsdfVer(13) || isTsdfVer(14) || isTsdfVer(15) || isTsdfVer(16) || isTsdfVer(17) || isTsdfVer(18)){
			//v12 核心目的: 运动模糊在 fusion 阶段造成薄片结构损坏, 改进他 @2017-12-1 11:04:32
			//部分照搬tsdfVer11; 

			//1, fusion 前, 用新T & 旧tsdf, 投射半新 vmap_g_model_, nmap_g_model_
			//存疑: 直接用旧的行不行? 因为 dT 很小	//试试!!!
			double rcFlagMap_total_time = 0
				,rcOrig_vol2_total_time = 0
				,genDmap_total_time = 0
				,dist_trans_total_time = 0
				,distMap_upload_total_time = 0
				,nmap_filt_g_total_time = 0
				,integ18_total_time = 0
				;

			tt2.tic();
			tt3.tic();
			//raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_);
			//↓--rcast-through (折射+穿透) 策略提前到这里 @2018-11-29 16:59:34
			raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), 
				getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_, rcFlagMap_, vxlPos);
			rcFlagMap_total_time = tt3.toc();

			tt3.tic();
			//再对 vol-2nd, 即 ghost volume 做 rcast.orig, 为了在 cluster 时求稳定的 sdf_forward (之前用 dcurr 求, 但是 dcurr 是瞬时的, 本身当前时刻不可靠)	@2018-11-29 17:30:06
			//printf("DBG【【raycast_volume2nd_\n");
// 			raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, 
// // 				tsdf_volume_->data(), 
// // 				getCyclicalBufferStructure(), //错, 对应的是 cyclical1_, 进而 vol1
// 				tsdf_volume_->volume2nd_, 
// 				cyclical2_.getBuffer(),
// 				vmap_g_model_vol2_, nmap_g_model_vol2_);
			//device::generateDepth(device_Rcurr_inv, device_tcurr, vmap_g_model_vol2_, dmapModel_vol2_); //model
			//printf("DBG【【AFTER-raycast_volume2nd_\n");
			rcOrig_vol2_total_time = tt3.toc();

			if(dbgKf_ >= 1){
				//printf("isTsdfVer(12)-raycast: "); tt3.toc_print(); //0ms

				tt3.tic();
			}

			tt3.tic();
			device::generateDepth(device_Rcurr_inv, device_tcurr, vmap_g_model_, dmapModel_);
			genDmap_total_time = tt3.toc();

			if(dbgKf_ >= 1){
				//printf("isTsdfVer(12)-generateDepth: "); tt3.toc_print(); //cudaSyn 8~12ms; 去掉syn后 0ms

				tt3.tic();
			}
#pragma region tv12【DEPRECATED】
			//【DEPRECATED】	@2018-2-25 22:13:37
			if(isTsdfVer(12)){	//v12 单独一版, 主要观察、测试 depth-uncertainty
#if 0	//opencv CPU
				cv::Mat dmapModel_host(dmapModel_.rows(), dmapModel_.cols(), CV_16UC1);
				dmapModel_.download(dmapModel_host.data, dmapModel_.colsBytes());

				cv::Mat dmapRaw_host(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
				depth_raw.download(dmapRaw_host.data, dmapRaw_host.step);

				cv::Mat dmapModel_host_16s, dmapRaw_host_16s;
				dmapModel_host.convertTo(dmapModel_host_16s, CV_16S);
				dmapRaw_host.convertTo(dmapRaw_host_16s, CV_16S);
				cv::Mat diffDmap16s = dmapRaw_host_16s - dmapModel_host_16s;

				//cv::Mat invalidMask8u(dmapRaw_host.size(), CV_8U, cv::Scalar(255));
				//invalidMask8u.setTo(0, dmapModel_host_16s==0);
				//invalidMask8u.setTo(0, dmapRaw_host_16s==0);

#elif 1	//GPU
				diffDmaps(depth_raw, dmapModel_, diffDmap_);

				//测试: uncertainty 描述子 @2017-12-8 11:01:16
				//krnl 外先求 dR, dt:: i->(i-1)
				//Matrix3frm Rcurr_inv = Rcurr.inverse ();
				//Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
				//Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
				//float3& device_tcurr = device_cast<float3> (tcurr);

				Matrix3frm dR_i_i1 = Rprev_inv * Rcurr;
				Vector3f dT_i_i1 = Rprev_inv * (tcurr - tprev);
				Mat33& device_dR_i_i1 = device_cast<Mat33>(dR_i_i1);
				float3& device_dt_i_i1 = device_cast<float3>(dT_i_i1);

				DeviceArray2D<float> uncertaintyMap;
				test_depth_uncertainty(vmaps_curr_[0], nmaps_curr_[0], device_dR_i_i1, device_dt_i_i1, uncertaintyMap);

				//可视化
				cv::Mat uncertMap32f(uncertaintyMap.rows(), uncertaintyMap.cols(), CV_32F);
				uncertaintyMap.download(uncertMap32f.data, uncertMap32f.step);

				const float uncertThresh = 30; //e.g.:: 1. displacement 位移度量
				uncertMap32f.setTo(-uncertThresh, uncertMap32f < -uncertThresh);
				uncertMap32f.setTo(+uncertThresh, uncertMap32f > +uncertThresh);

				cv::Mat uncertMap8u;
				uncertMap32f.convertTo(uncertMap8u, CV_8U, 128./uncertThresh, 128.);
				cv::Mat uncertMapCmap;
				cv::applyColorMap(uncertMap8u, uncertMapCmap, cv::COLORMAP_RAINBOW);
				cv::imshow("uncertMapCmap", uncertMapCmap);
#endif

				if(dbgKf_ >= 1){
					//printf("isTsdfVer(12)-diffDmapCmap: "); tt3.toc_print(); //0ms

					cv::Mat diffDmap16s(diffDmap_.rows(), diffDmap_.cols(), CV_16S);
					diffDmap_.download(diffDmap16s.data, diffDmap16s.step);
					const short diffDepthThresh = //1000; //1m
						15; //1cm
					diffDmap16s.setTo(-diffDepthThresh, diffDmap16s < -diffDepthThresh);
					diffDmap16s.setTo(diffDepthThresh, diffDmap16s > diffDepthThresh);

					cv::Mat diffDmap8u;
					//假设: diff范围 ±diffDepthThresh (mm), 
					diffDmap16s.convertTo(diffDmap8u, CV_8U, 128./diffDepthThresh, 128.);
					cv::Mat diffDmapColormap;
					cv::applyColorMap(diffDmap8u, diffDmapColormap, cv::COLORMAP_RAINBOW);
					cv::imshow("diffDmapColormap", diffDmapColormap);

					if(dbgKf_ == 2){
						char fnBuf[80] = {0};
						sprintf(fnBuf, "diffDmapCmap-%06d.png", callCnt_);
						cv::imwrite(fnBuf, diffDmapColormap);
					}


					tt3.tic();
				}//if-(dbgKf_ >= 1)
			}//if-isTsdfVer(12)	【DEPRECATED】
#pragma endregion 

			tt3.tic();
			cv::Mat edgeDistMap;
			if(!isTsdfVer(18))
				cv::distanceTransform(md_mask==0, edgeDistMap, CV_DIST_L1, 3);
			else //if-(isTsdfVer(18))
				cv::distanceTransform(edge_whole_mask==0, edgeDistMap, CV_DIST_L1, 3); //tsdf v18.16
			dist_trans_total_time = tt3.toc();

			//printf("DBG【【edgeDistMap_device_.upload: edgeDistMap.rows_cols: %d, %d; elemSize: %u (%%d: %d);\n", 
			//	edgeDistMap.rows, edgeDistMap.cols, edgeDistMap.elemSize(), edgeDistMap.elemSize());

			tt3.tic();
			//DeviceArray2D<float> edgeDistMap_device_;
			edgeDistMap_device_.upload(edgeDistMap.data, edgeDistMap.cols * edgeDistMap.elemSize(), edgeDistMap.rows, edgeDistMap.cols);
			distMap_upload_total_time += tt3.toc();
			//printf("distMap_upload_total_time: %f\n", distMap_upload_total_time);
#if 0	//【DBG: 多次 upload 并不耗时间， 说明 <1ms, 但为什么统计第一次为 10ms? @2019-1-27 00:29:23
			tt3.tic();
			edgeDistMap_device_.upload(edgeDistMap.data, edgeDistMap.cols * edgeDistMap.elemSize(), edgeDistMap.rows, edgeDistMap.cols);
			distMap_upload_total_time += tt3.toc();
			tt3.tic();
			edgeDistMap_device_.upload(edgeDistMap.data, edgeDistMap.cols * edgeDistMap.elemSize(), edgeDistMap.rows, edgeDistMap.cols);
			distMap_upload_total_time += tt3.toc();
			tt3.tic();
			edgeDistMap_device_.upload(edgeDistMap.data, edgeDistMap.cols * edgeDistMap.elemSize(), edgeDistMap.rows, edgeDistMap.cols);
			distMap_upload_total_time += tt3.toc();
#endif
			//printf("DBG【【AFTER-edgeDistMap_device_.upload\n");
			if(dbgKf_ >= 1){
				printf("isTsdfVer(12)-distanceTransform: "); tt3.toc_print(); //4ms, 有没有 GPU上的 DistTrans 代码?

				cv::Mat edgeMatShow;
				cv::normalize(edgeDistMap, edgeMatShow, 0, 1, cv::NORM_MINMAX);
				imshow("edgeMatShow", edgeMatShow);

				tt3.tic();
			}

			float3 t_tmp;
			t_tmp.x=t_tmp.y=t_tmp.z=0;

			nmap_filt_ = nmaps_curr_[0];
			tt3.tic();
			zc::transformVmap(nmap_filt_, device_Rcurr, t_tmp, nmap_filt_g_); //对 curr 就用 local, 不用 global, 
			nmap_filt_g_total_time = tt3.toc();

#pragma region NOT tv18
			if(!isTsdfVer(18)){ //if-not-v18, 即, v18.x 放弃 largeIncidMask_model_, computeNormalsContourcue, largeIncidMask_total_ @2018-3-25 23:27:28

				zc::contourCorrespCandidate(device_tcurr, vmap_g_model_, nmap_g_model_, this->incidAngleThresh_, largeIncidMask_model_); //tv11里已放弃? @2017-12-3 19:31:47
				if(dbgKf_ >= 1){
					//printf("isTsdfVer(12)-contourCorrespCandidate(largeIncidMask_model_)+transformVmap: "); tt3.toc_print(); //2ms

					tt3.tic();
				}

				//bdr 获取 incidMask
				cv::Mat dcurrFiltHost(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
				depths_curr_[0].download(dcurrFiltHost.data, depths_curr_[0].colsBytes()); //用滤波 dmap 替换 raw, 观察
				cv::Mat gradu, gradv;
				cv::Sobel(dcurrFiltHost,gradu,CV_32F,1,0,7,1.0/1280); //借用 gradu 变量
				cv::Sobel(dcurrFiltHost,gradv,CV_32F,0,1,7,1.0/1280);
				gx_device_.upload(gradu.data, gradu.cols * gradu.elemSize(), gradu.rows, gradu.cols);
				gy_device_.upload(gradv.data, gradv.cols * gradv.elemSize(), gradv.rows, gradv.cols);
				//MapArr nmap_filt_, nmap_filt_g_;
				MapArr nmap_bdr;
				device::computeNormalsContourcue(intr(0), depths_curr_[0], gx_device_, gy_device_, nmap_bdr);
				if(dbgKf_ >= 1){
					//printf("isTsdfVer(12)-computeNormalsContourcue: "); tt3.toc_print(); //8ms

					tt3.tic();
				}

				zc::contourCorrespCandidate(t_tmp, vmaps_curr_[0], nmap_bdr, this->incidAngleThresh_, largeIncidMask_total_); //不用 device_tcurr, 不用 nmap..._g

				if(dbgKf_ >= 1){
					//printf("isTsdfVer(12)-contourCorrespCandidate(largeIncidMask_total_): "); tt3.toc_print(); //1ms

					cv::Mat incidMskHost(largeIncidMask_total_.rows(), largeIncidMask_total_.cols(), CV_8UC1);
					largeIncidMask_total_.download(incidMskHost.data, incidMskHost.cols * incidMskHost.elemSize());
					imshow("incidMskHost", incidMskHost);

					tt3.tic();
				}

				//weight-map, 按: 1, 入射角cos; 2, D(u); 3, 到边缘距离 来加权
				//MapArr wmap_;
				//zc::calcWmap(vmaps_curr_[0], nmap_filt, thickContMsk, wmap);
				zc::calcWmap(vmaps_curr_[0], nmap_filt_, edgeDistMap_device_, fx_, wmap_); //for tsdf-v11.8
				if(dbgKf_ >= 1)
				{
					//printf("isTsdfVer(12)-calcWmap: "); tt3.toc_print(); //0~1ms

					cv::Mat wmapHost(wmap_.rows(), wmap_.cols(), CV_32F);
					wmap_.download(wmapHost.data, wmap_.colsBytes());
					cv::Mat wmapHost8u;
					wmapHost.convertTo(wmapHost8u, CV_8UC1, UCHAR_MAX); //*255, 因为 wmap 本身 ~(0,1)
					imshow("wmapHost8u", wmapHost8u);

					tt3.tic();
				}
			}//if-(!isTsdfVer(18))

			if(isTsdfVer(12)){
				integrateTsdfVolume_v12
					(isPlFilt_ ? depthPlFilt : depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(),
					tsdf_volume_->volume2nd_, tsdf_volume_->flagVolume_, tsdf_volume_->surfNormPrev_, tsdf_volume_->vrayPrevVolume_, 
					largeIncidMask_total_,
					//largeIncidMask_model_, //model 为啥放弃?:: 因为边缘也是不稳定区域, 希望与大入射角区域同等处理 @2017-12-3 19:35:27
					//答: tv11 里, 暂时 total 好一点 (其实是 nmap_bdr 求解得到); 后续 tv12 存疑
					/*nmap_total_g, */ 
					//nmap_g_model_, //到底用哪个? 不好弄
					nmap_filt_g_, 
					nmap_g_model_, //v11.6
					wmap_,
					dmapModel_,
					diffDmap_,
					depthRawScaled_, vxlPos);
			}
			else if(isTsdfVer(13) || isTsdfVer(14) || isTsdfVer(15) || isTsdfVer(16) || isTsdfVer(17)){	//v13~v17
				integrateTsdfVolume_v13
					(isPlFilt_ ? depthPlFilt : depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(),
					tsdf_volume_->volume2nd_, tsdf_volume_->flagVolume_, tsdf_volume_->surfNormPrev_, tsdf_volume_->vrayPrevVolume_, 
					largeIncidMask_total_,
					//largeIncidMask_model_, //model 为啥放弃?:: 因为边缘也是不稳定区域, 希望与大入射角区域同等处理 @2017-12-3 19:35:27
					//答: tv11 里, 暂时 total 好一点 (其实是 nmap_bdr 求解得到); 后续 tv12 存疑
					/*nmap_total_g, */ 
					//nmap_g_model_, //到底用哪个? 不好弄
					nmap_filt_g_, 
					nmap_g_model_, //v11.6
					wmap_,
					dmapModel_,
					diffDmap_,
					depthRawScaled_, vxlPos);
			}
#pragma endregion 
			else if(isTsdfVer(18)){
				tt3.tic();
				//zc::edge2wmap(vmaps_curr_[0], edgeDistMap_device_, fx_, wmap_); //for tsdf-v18.4
				//float maxEdgeDist = tsdf_volume_->getTsdfTruncDist() * 6e3; //in mm
				float maxEdgeDist = 30; //in mm, 定死
				zc::edge2wmap(vmaps_curr_[0], edgeDistMap_device_, fx_, maxEdgeDist, wmap_); //增加 maxEdgeDist 控制量, 暂定 2~3 倍 tdist @2018-8-17 17:42:08

				//tt1.tic();
				//DepthMap rcFlagMap_;
				//rcFlagMap_.create(rows_, cols_); //放到 rcast 里
				//↓--放到前面
// 				raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), 
// 					getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_, rcFlagMap_, vxlPos);
// 					//getCyclicalBufferStructure(), vmap_g_model_, nmap_g_model_, wmap_, rcFlagMap_, vxlPos); //放弃在 rcast 中用 wmap_ 策略 @2018-11-29 16:52:01
				//printf("raycast_rcFlagMap_: "); tt1.toc_print(); //0ms

				if(dbgKf_ >= 1){
					tt1.tic();
					cv::Mat rcFlag16s(480, 640, CV_16S);
					cv::Mat rcFlag8uc3(480, 640, CV_8UC3, cv::Scalar(0));
					rcFlagMap_.download(rcFlag16s.data, rcFlagMap_.colsBytes());
#if 0 //rcFlagMap_ 确实存几个 flag 的情形
					//rcFlag16s.convertTo(rcFlag8u, CV_8U);
					//cv::imshow("rcFlag8u", rcFlag8u);
					rcFlag8uc3.setTo(cv::Scalar(255, 255, 255), rcFlag16s == 255);
					rcFlag8uc3.setTo(cv::Scalar(255, 0, 0), rcFlag16s == 127);	//蓝色, ->+区域
					rcFlag8uc3.setTo(cv::Scalar(0, 255, 0), rcFlag16s == 128);	//绿色, 0>-区域
					rcFlag8uc3.setTo(cv::Scalar(0, 0, 255), rcFlag16s == 129);	//红色, ->0区域
					rcFlag8uc3.setTo(cv::Scalar(255, 0, 255), rcFlag16s == 130);	//洋红色, 0>->0区域
#elif 1 //rcFlagMap_ 改存正负深度值后
					cv::Mat rcFlag8u_posi(rcFlag16s.size(), CV_8U, cv::Scalar(0));
					cv::Mat rcFlag8u_nega(rcFlag16s.size(), CV_8U, cv::Scalar(0));
					cv::Mat tmp16s = rcFlag16s.clone();
					tmp16s.setTo(0, tmp16s < 0);
					tmp16s.convertTo(rcFlag8u_posi, CV_8U, UCHAR_MAX/1500.);

					tmp16s = rcFlag16s.clone() * -1; //取反
					tmp16s.setTo(0, tmp16s < 0);
					tmp16s.convertTo(rcFlag8u_nega, CV_8U, UCHAR_MAX/1500.);

					cv::Mat tmp_channel3[3] = {cv::Mat::zeros(rcFlag16s.size(), CV_8U), rcFlag8u_posi, rcFlag8u_nega};
					cv::merge(tmp_channel3, 3, rcFlag8uc3);

					char filename[ 1024 ];
					std::sprintf( filename, "image/rcFlag/%06d.png", global_time_ + 1 );
					cv::imwrite(filename, rcFlag8uc3);
#endif
					cv::imshow("rcFlag8uc3", rcFlag8uc3);
					printf("rcFlag8uc3 "); tt1.toc_print(); //~21ms
				}

				if(dbgKf_ >= 1){
					tt1.tic();
					//DepthMap depthRcast_;
					generateDepth(device_Rcurr_inv, device_tcurr, vmap_g_model_, dmapModel_);
					vector<unsigned short> depthRcast_host_data;
					int c;
					dmapModel_.download(depthRcast_host_data, c);
					cv::Mat depthRcast_host16u( 480, 640, CV_16UC1, (void *)&depthRcast_host_data[0] );
					cv::Mat depthRcast_host8u;
					depthRcast_host16u.convertTo(depthRcast_host8u, CV_8U, UCHAR_MAX/1500.);
					cv::imshow("depthRcast@after_icp", depthRcast_host8u);
					printf("depthRcast_host8u "); tt1.toc_print();
				}

				if(dbgKf_ >= 1)
				{
					printf("isTsdfVer(18)-edge2wmap: "); tt3.toc_print(); //0~1ms

					cv::Mat wmapHost(wmap_.rows(), wmap_.cols(), CV_32F);
					wmap_.download(wmapHost.data, wmap_.colsBytes());
					cv::Mat wmapHost8u;
					wmapHost.convertTo(wmapHost8u, CV_8UC1, UCHAR_MAX); //*255, 因为 wmap 本身 ~(0,1)
					imshow("wmapHost8u", wmapHost8u);

					tt3.tic();
				}

				//printf("DBG【【BEFORE-integrateTsdfVolume_v18\n");

				if(dbgV2_18_ >0)
					integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_, vxlPos);
				else
					integrateTsdfVolume_v18
					(isPlFilt_ ? depthPlFilt : depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(),
					tsdf_volume_->volume2nd_, tsdf_volume_->flagVolume_, tsdf_volume_->surfNormPrev_, tsdf_volume_->vrayPrevVolume_, 
					largeIncidMask_total_,
					//largeIncidMask_model_, //model 为啥放弃?:: 因为边缘也是不稳定区域, 希望与大入射角区域同等处理 @2017-12-3 19:35:27
					//答: tv11 里, 暂时 total 好一点 (其实是 nmap_bdr 求解得到); 后续 tv12 存疑
					/*nmap_total_g, */ 
					//nmap_g_model_, //到底用哪个? 不好弄
					nmap_filt_g_, 
					nmap_g_model_, //v11.6
					wmap_,
					dmapModel_,
					diffDmap_,
					dmapModel_vol2_, //用作 sdf-forward/back 二分类, 之前用的第一形参 depth_raw (dcurr), 但是噪声大, 太不可靠 @2018-11-29 17:50:32
					rcFlagMap_,
					depthRawScaled_, vxlPos);
				integ18_total_time = tt3.toc();
				
				printf("rcFlagMap_total_time, rcOrig_vol2, genDmap, dist_trans, distMap_upload, nmap_filt_g, integ18: %f, %f, %f, %f, %f, %f, %f\n", 
					rcFlagMap_total_time, rcOrig_vol2_total_time, genDmap_total_time, dist_trans_total_time, distMap_upload_total_time, nmap_filt_g_total_time, integ18_total_time);
			}

			//if(dbgKf_ >= 1)
				printf("integrateTsdfVolume_v18 "); tt2.toc_print(); //很快, 5ms
		}//elif-isTsdfVer(12)~18
	}//if-(integrate)

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting //改调到最前面, 仿 bdrOdometry 规矩


	++global_time_;
	return true;
}//cuOdometry

bool pcl::gpu::KinfuTracker::s2sOdometry( const DepthMap &depth_raw, const View *pcolor /*= NULL*/){
	//printf("---------------callCnt:= %d\n", callCnt_);
	callCnt_++;
	ScopeTime time( "s2sOdometry All" );
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	float3& device_volume000_gcoo = device_cast<float3>(volume000_gcoo_);

	//scale-depth 放最外面 @2018-7-4 16:54:24
	DeviceArray2D<float> depthFiltScaled;
	DeviceArray2D<float> depthModelScaled;

	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all"); //release 下 ~2ms
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}

		scaleDepthDevice(depth_raw, intr, depthRawScaled_);
		scaleDepthDevice(depths_curr_[0], intr, depthFiltScaled);

		pcl::device::sync ();
	}//">>> Bilateral, pyr-down-all, create-maps-all"

	//定点调试观察某体素:
	int3 vxlPos;
	vxlPos.x = vxlDbg_[0];
	vxlPos.y = vxlDbg_[1];
	vxlPos.z = vxlDbg_[2];


	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		//device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		cout<<"~~~~~~~~~~~(((global_time_ == 0; init-tvec: "<<tvecs_[0].transpose()
			<<"; volume000_gcoo_: "<<volume000_gcoo_.transpose()<<endl;;
		integrateTsdfVolume_s2s(intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, device_volume000_gcoo, //s2s-v0.2
			tsdf_volume_->getTsdfTruncDist(), s2s_eta_, false, tsdf_volume_->data(), depthRawScaled_, vxlPos);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;

		depthRawScaled_.copyTo(depthRawScaled_prev_); //若后面 f2f, 这里要预先生成一次, 避免 fid=1 时无配准目标 @2018-11-26 18:02:14

		++global_time_;
		return (false);
	}//if-(global_time_ == 0)

	Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
	Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();
	//cout<<"Rprev: "<<Rprev<<endl;
	Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
	Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
	float3& device_tprev     = device_cast<float3> (tprev);

	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	Vector3f tprev_fake = tprev - volume000_gcoo_;
	float3& device_tprev_fake = device_cast<float3> (tprev_fake);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// Ray casting //icp 之前先做了, 仿 bdrOdometry 规矩, 与原来 kinfu 不同 @2017-4-5 17:10:59
	{
	ScopeTime time("ray-cast-all"); //11ms, 这么慢? @2018-3-26 00:58:34
	//raycast (intr, device_Rprev, device_tprev, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
	//s2s-v0.2:
// 	Vector3f tprev_fake = tprev - volume000_gcoo_; //放到外面
// 	float3& device_tprev_fake = device_cast<float3> (tprev_fake);

	raycast (intr, device_Rprev, device_tprev_fake, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
	//scaleDepthDevice(vmaps_g_prev_[0], device_tprev, intr, depthModelScaled);

// 	for (int i = 1; i < LEVELS; ++i)
// 	{
// 		resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
// 		resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
// 	}
	//pcl::device::sync ();
	}

	Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
	Vector3f   tcurr = tprev;
	{
	ScopeTime time(">>> icp-all");

	//for (int level_index = LEVELS-1; level_index>=0; --level_index){
	//	int iter_num = icp_iterations_[level_index];

	//	// current maps
	//	MapArr& vmap_curr = vmaps_curr_[level_index];
	//	MapArr& nmap_curr = nmaps_curr_[level_index];

	//	// previous maps
	//	MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
	//	MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

	//	for (int iter = 0; iter < iter_num; ++iter){
	//		Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
	//		float3& device_tcurr = device_cast<float3>(tcurr);

	//		Eigen::Matrix<float, 6, 1> result; //6*1 Rt //twist: t3+w3 @2018-5-28 22:04:48
	//	}
	//}

	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
	float avgErr_i1 = 1e3, 
		avgErr_curr = 0;

	const int max_iter =30;
	for (int iter = 0; iter < max_iter; ++iter){
		//cout<<">>>>>>>>>>iter: "<<iter<<endl;
		Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
		float3& device_tcurr = device_cast<float3>(tcurr);

		Matrix3frm Rcurr_inv = Rcurr.inverse ();
		Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);

		typedef Eigen::Matrix<float, 6, 1> Vector6f;

		Vector6f result; //6*1 Rt //twist: t3+w3 @2018-5-28 22:04:48

		Vector6f xi_curr;

		//cout<<"BEFORE-get_twist_from_rt\n";
#define XI_C2G 01 //1-c2g; 0-g2c    
		//↑-已测试, 这里 g2c 必须对应@estimate_combined.cu中: tmp = tsdf1 - tsdf2 +...;
		//c2g 则必须对应 t2-t1 @2018-7-9 15:18:03
#if XI_C2G
 		//Eigen::AngleAxisf aa(Rcurr);
 		//Vector3f rvec = aa.axis() * aa.angle();
		//xi_curr << tcurr, rvec; //wrong: head(3) ≠ real t-part
		//cout<<"Rcurr: "<<Rcurr<<endl;
		//cout<<"rvec:"<<rvec.transpose()<<", aa.ax, ang: "<<aa.axis()<<", "<<aa.angle()<<endl;

		//Sophus::SE3f se_rt(Rcurr, tcurr);
		//xi_curr = se_rt.log(); 

		xi_curr = get_twist_from_rt(Rcurr, tcurr);
#else	//g2c
		//Eigen::AngleAxisf aa(Rcurr_inv); //g2c
		//Vector3f rvec = aa.axis() * aa.angle();
		Vector3f tcurr_inv = - Rcurr_inv * tcurr;
		//xi_curr << tcurr_inv, rvec;

		//xi_curr = get_twist_from_rt(Rcurr_inv, tcurr_inv); //(R', -R'*t)
#endif
		//xi_curr << 0,0,0, 0,0,0;		//xi 不对应 (R,t)_curr, 而仅对应一次迭代中的增量
		//↑-已验证, 不用 000, 不是误差来源 @2018-7-14 01:31:05
		//cout<<"xi_curr: "<<xi_curr.transpose()<<endl;
		float6& device_xi_curr = device_cast<float6>(xi_curr);

		//estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
		//	vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_f2mod_.data (), b_f2mod_.data ());
		A_s2s_.setZero();
		b_s2s_.setZero();

		tt3.tic();
		initVrayPrevVolume(tsdf_volume_->volume2nd_);
		bool use_eta_trunc = true;
		integrateTsdfVolume_s2s(intr, device_volume_size, device_Rcurr_inv, device_tcurr,
			device_volume000_gcoo, tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, //配准时都用 eta截断, 目的是借用 beam 在边缘形成约束(抠住)
			tsdf_volume_->volume2nd_, 
			depthRawScaled_, 
			//depthFiltScaled, //配准改用 bilat-filt, 不用 raw @2018-7-6 19:58:36
			vxlPos); //这里 set vol-2

		int vxlValidCnt = -1;
		float sum_s2s_err = 0;

		if(s2s_f2m_){ //f2m
			scaleDepthDevice(vmaps_g_prev_[0], device_tprev, intr, depthModelScaled);
			//scaleDepthDevice(vmaps_g_prev_[0], device_tprev_fake, intr, depthModelScaled); //也没差别
			//initVrayPrevVolume(tsdf_volume_->volume3rd_proxy_); //之前【逐帧】都在 create_init_s2s_volume 初始化过, 暂不考虑在【逐循环】中reset
			integrateTsdfVolume_s2s(intr, device_volume_size, device_Rprev_inv, device_tprev, device_volume000_gcoo,
				tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, tsdf_volume_->volume3rd_proxy_, depthModelScaled, vxlPos);
		}
		else{ //f2f
			integrateTsdfVolume_s2s(intr, device_volume_size, device_Rprev_inv, device_tprev, device_volume000_gcoo,
				tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, tsdf_volume_->volume3rd_proxy_, depthRawScaled_prev_, vxlPos);
		}
		estimateCombined_s2s(depth_raw, intr, device_volume_size, 
			device_Rcurr_inv, device_tcurr, 
			//↓--主要用作 i, (i-1)对应点匹配(projective-association), 滤除 outlier, 因为仅 tsdf-p-c diff 根本无法判定实际 diff (sdf可解, 但是存的不是sdf, 无解) @2018-11-26 18:42:03
			device_Rprev_inv, device_tprev, vmaps_g_prev_[0], //prev量 每帧仅更新一次, 循环内不变动
			device_volume000_gcoo, device_xi_curr, 
			tsdf_volume_->getTsdfTruncDist(), 
			//tsdf_volume_->data(), 
			tsdf_volume_->volume3rd_proxy_, //改用 vol-3rd, 可切换 f2m/f2f
			tsdf_volume_->volume2nd_,
			s2s_eta_, gbuf_s2s_, sumbuf_s2s_, A_s2s_.data(), b_s2s_.data(), 
			depthRawScaled_, 
			//depthFiltScaled, //配准改用 bilat-filt, 不用 raw @2018-7-6 19:58:36
			vxlValidCnt, sum_s2s_err,
			vxlPos);
		//printf("estimateCombined_s2s, vxlValidCnt= 【【%d; sum_s2s_err: 【【%f; avg-err: 【【%f; \n"
		//	"tsdf_diff_sum_host_: 【【%f; tsdf_abs_diff_sum_host_: 【【%f; ", 
		//	vxlValidCnt, sum_s2s_err, sum_s2s_err/vxlValidCnt, tsdf_diff_sum_host_, tsdf_abs_diff_sum_host_); 
		//tt3.toc_print();

		avgErr_curr = sum_s2s_err / vxlValidCnt;
		if(avgErr_i1 - avgErr_curr < 1e-4){ //收敛终止条件
			if(dbgKf_ >= 1)
				printf("vxlValidCnt= 【【%d; sum_s2s_err: 【【%f; avgErr_i1, avgErr_curr: %f, %f, diff: %f; iter= %d\n", 
				vxlValidCnt,sum_s2s_err, 
				avgErr_i1, avgErr_curr, avgErr_i1 - avgErr_curr, iter);

#if 0	//f2m->f2f 切换条件放在这里不合适吧？ 改成: 若迭代到第x次不收敛 (sum_s2s_err > 5e4), 就【从头】f2f一遍
			//增加检测: 若 sum_s2s_err > 5w, 就重新用 f2f 再配一遍	@2018-11-25 22:05:17
			if(sum_s2s_err > 5e4){
				scaleDepthDevice(vmaps_g_prev_[0], device_tprev, intr, depthModelScaled);
				//scaleDepthDevice(vmaps_g_prev_[0], device_tprev_fake, intr, depthModelScaled); //也没差别
				initVrayPrevVolume(tsdf_volume_->volume3rd_proxy_);
				//integrateTsdfVolume_s2s(intr, device_volume_size, device_Rprev_inv, device_tprev, device_volume000_gcoo,
				//	tsdf_volume_->getTsdfTruncDist(), s2s_eta_, tsdf_volume_->volume3rd_proxy_, depthModelScaled, vxlPos);
				integrateTsdfVolume_s2s(intr, device_volume_size, device_Rprev_inv, device_tprev, device_volume000_gcoo,
					tsdf_volume_->getTsdfTruncDist(), s2s_eta_, tsdf_volume_->volume3rd_proxy_, depthRawScaled_prev_, vxlPos);

				A_s2s_.setZero();
				b_s2s_.setZero();

				estimateCombined_s2s(depth_raw, intr, device_volume_size, 
					device_Rcurr_inv, device_tcurr, 
					device_Rprev_inv, device_tprev, vmaps_g_prev_[0],
					device_volume000_gcoo, device_xi_curr, 
					tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->volume3rd_proxy_, tsdf_volume_->volume2nd_,
					s2s_eta_, gbuf_s2s_, sumbuf_s2s_, A_s2s_.data(), b_s2s_.data(), 
					depthRawScaled_, 
					//depthFiltScaled, //配准改用 bilat-filt, 不用 raw @2018-7-6 19:58:36
					vxlValidCnt, sum_s2s_err,
					vxlPos);
				
				avgErr_curr = sum_s2s_err / vxlValidCnt;

				printf("\t F2F:= vxlValidCnt= 【【%d; sum_s2s_err: 【【%f; avgErr_i1, avgErr_curr: %f, %f, diff: %f; iter= %d\n", 
					vxlValidCnt,sum_s2s_err, 
					avgErr_i1, avgErr_curr, avgErr_i1 - avgErr_curr, iter);
			}
			else //sum_s2s_err 够小才能结束迭代 @2018-11-26 04:07:05
				break;
#else
			break;
#endif
		}
		else{
			if(iter > max_iter - 3){
				printf("iter_gt_max_iter:: vxlValidCnt= 【【%d; sum_s2s_err: 【【%f; avgErr_i1, avgErr_curr: %f, %f, diff: %f; iter= %d\n", 
					vxlValidCnt,sum_s2s_err, 
					avgErr_i1, avgErr_curr, avgErr_i1 - avgErr_curr, iter);
			}
			avgErr_i1 = avgErr_curr;
		}

		double det_f2mod = A_s2s_.determinant();
		if ( fabs (det_f2mod) < 1e-15 || pcl_isnan (det_f2mod) ){
			printf("!!!estimateCombined_s2s::-- det_f2mod @A_s2s_==0, align-failed, resetting....\n");
			return false;
		}

		result = A_s2s_.llt().solve(b_s2s_).cast<float>(); //twist*, 二阶导, 牛顿法
		if(dbgKf_ == 2)
			cout<<"A_s2s_: "<<A_s2s_<<endl
				<<"A_s2s_(DOUBLE): "<<A_s2s_.cast<double>()<<endl
				<<"b_s2s_: "<<b_s2s_.transpose()<<endl
				<<"det_f2mod: "<<det_f2mod<<endl
				<<"det_f2mod(DOUBLE): "<<A_s2s_.cast<double>().determinant()<<endl
				<<"result: "<<result.transpose()<<endl
				<<"A'*b: "<<(A_s2s_.inverse()*b_s2s_).transpose()<<endl
				<<"A'*b(DOUBLE): "<<(A_s2s_.cast<double>().inverse()*b_s2s_.cast<double>()).transpose()<<endl
				;

		Vector6f xi_inc = s2s_beta_ * (result - xi_curr);
		//cout
		//	<<"xi-star: "<<result.transpose()<<endl
		//	<<"xi_inc: "<<xi_inc.transpose()<<endl
			;

		Matrix3frm Rinc; //此 inc 其实是 i~i+1 之间的, 所以是先乘而非后乘	@2018-7-10 20:45:01 
		Vector3f tinc;
		get_rt_from_twist(xi_inc, Rinc, tinc);
#if XI_C2G //看 e-combined.cu 情况

#if 0	//用李代数加法 //不如矩阵乘法, 会飘
		xi_curr = get_twist_from_rt(Rcurr, tcurr);
		xi_curr += xi_inc;
		get_rt_from_twist(xi_curr, Rcurr, tcurr);
#elif 1	//用李群, 矩阵乘法
		//tcurr = Rinc * tcurr + tinc; //左乘 Tinc, 不行
		//Rcurr = Rinc * Rcurr;
		tcurr = Rcurr * tinc + tcurr; //右乘 Tinc, 一样不行, 其实应该左乘, 跟kf论文一致, 因为此 Tinc≠T(k, k+1)
		Rcurr = Rcurr * Rinc;
#endif //李群 vs. 李代数

#else //g2c

#if 0	//用李代数加法
#elif 1	//用李群, 矩阵乘法
		tcurr_inv = Rinc * tcurr_inv + tinc;
		Rcurr_inv = Rinc * Rcurr_inv;

		Rcurr = Rcurr_inv.inverse();
		tcurr = - Rcurr * tcurr_inv;
#endif //李群 vs. 李代数

#endif //XI_C2G vs. g2c

//#if 0	//按 s2s 论文, 李代数加法求解, 但似乎仅在 xi≈0附近成立, 原理未证明 @2018-7-9 16:17:46
//		//float BETA = 0.5f;
//		//xi_curr += BETA * (result - xi_curr);
//		xi_curr += xi_inc;
//
//		//rvec = xi_curr.tail(3);
//		//float angle = rvec.norm();
//		//rvec.normalize();
//
//#elif 1 //尝试改用李群乘法
//		Matrix3frm Rinc;
//		Vector3f tinc;
//		get_rt_from_twist(xi_inc, Rinc, tinc);
//		//Affine3f aff_inc;
//		//aff_inc.linear() = Rinc;
//		//aff_inc.translation() = tinc;
//
//		Matrix3frm Rcurr_tmp; //不是真的 Rcurr, 可能是 inv, 随 XI_C2G而变
//		Vector3f tcurr_tmp;
//		//get_rt_from_twist(xi_curr, Rcurr_tmp, tcurr_tmp); //不用 xi 做桥
//#if XI_C2G
//		Rcurr_tmp = Rcurr;
//		tcurr_tmp = tcurr;
//#else
//		Rcurr_tmp = Rcurr_inv;
//		tcurr_tmp = tcurr_inv;
//#endif
//		cout<<"Rcurr_tmp, tcurr_tmp:\n"<<Rcurr_tmp<<endl;
//
//		tcurr_tmp = Rinc * tcurr_tmp + tinc;
//		Rcurr_tmp = Rinc * Rcurr_tmp;
//
//		xi_curr = get_twist_from_rt(Rcurr_tmp, tcurr_tmp);
//#endif
//		cout
//			//<<"Rcurr_tmp, tcurr_tmp:\n"<<Rcurr_tmp<<endl
//			//<<tcurr_tmp.transpose()<<endl
//			<<"xi-star: "<<result.transpose()<<endl
//			<<"xi_inc: "<<xi_inc.transpose()<<endl
//			<<"AFTER-xi_curr: "<<xi_curr.transpose()<<endl
//			;
//#if XI_C2G
//		//Rcurr = Eigen::AngleAxisf(angle, rvec).toRotationMatrix();
//		//tcurr << xi_curr; //6elems->3elems, OK
//		//tcurr = xi_curr.head(3);
//
//		//cout<<"BEFORE-get_rt_from_twist\n";
//		get_rt_from_twist(xi_curr, Rcurr, tcurr);
//#else
//		//Rcurr_inv = Eigen::AngleAxisf(angle, rvec).toRotationMatrix();
//		//tcurr_inv = xi_curr.head(3);
//
//		get_rt_from_twist(xi_curr, Rcurr_inv, tcurr_inv);
//		Rcurr = Rcurr_inv.inverse();
//		tcurr = - Rcurr * tcurr_inv;
//#endif

	}//for-iter
	}//ScopeTime-icp-all
	if(dbgKf_ == 2)
		cout<<"tcurr,Rcurr:"<<tcurr.transpose()<<endl
			<<Rcurr<<endl;

	rmats_.push_back (Rcurr);
	tvecs_.push_back (tcurr);

	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
	float tnorm = (tcurr - tprev).norm();  
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;  
	//zc: 调试, 总是融合:
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration

	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);
	if (integrate)
	{
		ScopeTime time("if-integrate");
		////定点调试观察某体素: //移到前面
		//int3 vxlPos;
		//vxlPos.x = vxlDbg_[0];
		//vxlPos.y = vxlDbg_[1];
		//vxlPos.z = vxlDbg_[2];

		//ScopeTime time("tsdf");
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		//if(isTsdfVer(2)){
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_);
			//integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure(), depthRawScaled_, vxlPos);
			
// 			if(!s2s_f2m_){ //若命令行参数 f2f, 而非 f2m, 每帧清空 global volume
// 				initVrayPrevVolume(tsdf_volume_->data());
// 			}
			//↑--注释掉, 尝试启用 tsdf_volume_->volume3rd_proxy_, 主 vol 仅用作 fuse	@2018-11-26 17:40:29

			bool use_eta_trunc = false; //实际 integ 时候, 仍是用 tdist, 而非 eta
			integrateTsdfVolume_s2s(intr, device_volume_size, device_Rcurr_inv, device_tcurr, device_volume000_gcoo,
				tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, tsdf_volume_->data(), depthRawScaled_, vxlPos);
		//}//if-(isTsdfVer(2))
	}//if-(integrate)

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting //改调到最前面, 仿 bdrOdometry 规矩

	depthRawScaled_.copyTo(depthRawScaled_prev_);

	++global_time_;
	return true;
}//s2sOdometry

bool pcl::gpu::KinfuTracker::integrateWithGtPoses( const DepthMap &depth_raw, const View *pcolor /*= NULL*/){
	//printf("---------------callCnt:= %d\n", callCnt_);
	callCnt_++;
	ScopeTime time( "integrateWithGtPoses All" );
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	float3& device_volume000_gcoo = device_cast<float3>(volume000_gcoo_);

	//scale-depth 放最外面 @2018-7-4 16:54:24
	//DeviceArray2D<float> depthFiltScaled; //-no_icp 所以不需要 bilateral-filt, pyramid, registration, etc.

	scaleDepthDevice(depth_raw, intr, depthRawScaled_);
	//scaleDepthDevice(depths_curr_[0], intr, depthFiltScaled);

	//定点调试观察某体素:
	int3 vxlPos;
	vxlPos.x = vxlDbg_[0];
	vxlPos.y = vxlDbg_[1];
	vxlPos.z = vxlDbg_[2];

	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());
		cout<<"~~~~~~~~~~~(((global_time_ == 0; init-tvec: "<<tvecs_[0].transpose()
			<<"; volume000_gcoo_: "<<volume000_gcoo_.transpose()<<endl;;

		bool use_eta_trunc = true;
		integrateTsdfVolume_s2s(intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, device_volume000_gcoo, //s2s-v0.2
			tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, tsdf_volume_->data(), depthRawScaled_, vxlPos);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		++global_time_;
		return (false);
	}//if-(global_time_ == 0)

	Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
	Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();
	//cout<<"Rprev: "<<Rprev<<endl;
	Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
	Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
	float3& device_tprev     = device_cast<float3> (tprev);

	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	// Ray casting //icp 之前先做了, 仿 bdrOdometry 规矩, 与原来 kinfu 不同 @2017-4-5 17:10:59
	{
		ScopeTime time("ray-cast-all"); //11ms, 这么慢? @2018-3-26 00:58:34
		//raycast (intr, device_Rprev, device_tprev, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
		//s2s-v0.2:
		Vector3f tprev_fake = tprev - volume000_gcoo_;
		float3& device_tprev_fake = device_cast<float3> (tprev_fake);

		raycast (intr, device_Rprev, device_tprev_fake, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), getCyclicalBufferStructure(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
	}

	//直接用 gt, 不配准
	Matrix3frm gtR_i = gtRmats_[global_time_]; //pose: c_i->g
	Vector3f gtT_i = gtTvecs_[global_time_];

	Matrix3frm gtR_0 = gtRmats_[0];
	Vector3f gtT_0 = gtTvecs_[0];

	//用gt 预加载的求解 i->g->0
	Matrix3frm gtR_i0 = gtR_0.transpose() * gtR_i;
	Vector3f gtT_i0 =  gtR_0.transpose() * (gtT_i - gtT_0);

	Matrix3frm R_0g = rmats_[0];
	Vector3f t_0g = tvecs_[0];

	//new: i->0->g
	Matrix3frm Rcurr = R_0g * gtR_i0;
	Vector3f tcurr = R_0g * gtT_i0 + t_0g;

	rmats_.push_back (Rcurr);
	tvecs_.push_back (tcurr);

	bool integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration

	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);
	if (integrate)
	{
		ScopeTime time("if-integrate");
// 		cout<<"depthRawScaled_:"<<depthRawScaled_.cols()<<","<<depthRawScaled_.rows()<<endl	//640*480
// 			<<"tsdf_volume_:"<<tsdf_volume_->data().cols()<<";"<<tsdf_volume_->data().rows()<<endl;	//64*4096

		bool use_eta_trunc = true;
		integrateTsdfVolume_s2s(intr, device_volume_size, device_Rcurr_inv, device_tcurr, device_volume000_gcoo,
			tsdf_volume_->getTsdfTruncDist(), s2s_eta_, use_eta_trunc, tsdf_volume_->data(), depthRawScaled_, vxlPos);
	}//if-(integrate)

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting //改调到最前面, 仿 bdrOdometry 规矩

	++global_time_;
	return true;

}//integrateWithGtPoses

void pcl::gpu::KinfuTracker::dbgAhcPeac( const DepthMap &depth_raw, const View *pcolor /*= NULL*/){
	//device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);
	pcl::device::Intr intr(529.22, 528.98, 313.77, 254.10, 5.0);

	cv::Mat dm_raw(depth_raw.rows(), depth_raw.cols(), CV_16UC1); //milli-m, raw
	int c;
	depth_raw.download(dm_raw.data, depth_raw.colsBytes());
	tt0.tic();
	CloudType::Ptr depCloud = cvMat2PointCloud(dm_raw, intr);
	
	RGBDImage rgbdObj(*depCloud);
	cv::Mat dbgSegMat(depCloud->height, depCloud->width, CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	PlaneFitter planeFitter_;
	planeFitter_.minSupport = 0;
	planeFitter_.run(&rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(planeFitter_.membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac

void pcl::gpu::KinfuTracker::dbgAhcPeac2( const CloudType::Ptr depCloud){
	RGBDImage rgbdObj(*depCloud);
	cv::Mat dbgSegMat(depCloud->height, depCloud->width, CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	PlaneFitter planeFitter_;
	planeFitter_.minSupport = 1000;
	planeFitter_.run(&rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(planeFitter_.membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac2";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac2

void pcl::gpu::KinfuTracker::dbgAhcPeac3( const CloudType::Ptr depCloud, PlaneFitter *pf){
	RGBDImage rgbdObj(*depCloud);
	cv::Mat dbgSegMat(depCloud->height, depCloud->width, CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(&rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac3";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac3

void pcl::gpu::KinfuTracker::dbgAhcPeac4( const RGBDImage *rgbdObj, PlaneFitter *pf){
	cv::Mat dbgSegMat(rgbdObj->height(), rgbdObj->width(), CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac4";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac4

void dbgAhcPeac5( const RGBDImage *rgbdObj, PlaneFitter *pf){
	cv::Mat dbgSegMat(rgbdObj->height(), rgbdObj->width(), CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac5";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac5

void pcl::gpu::KinfuTracker::dbgAhcPeac5( const RGBDImage *rgbdObj, PlaneFitter *pf){
	::dbgAhcPeac5( rgbdObj, pf);
}//dbgAhcPeac5


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::kdtreeodometry(const DepthMap& depth_raw, const View * pcolor)
{
	//ScopeTime time( "Kinfu Tracker All" );
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	pcl::PointCloud< pcl::PointNormal >::Ptr vmappcd( new pcl::PointCloud< pcl::PointNormal >() );
	pcl::KdTreeFLANN< pcl::PointNormal > vmaptree;

	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
		pcl::device::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point


	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		if ( global_time_ == 1 ) {
			kdtree_cloud_.create( 480, 640 );
			kdtree_normal_.create( 480, 640 );
			kdtree_curr_.resize( LEVELS );
			int rows = 480;
			int cols = 640;
			for ( int i = 0; i < LEVELS; i++ ) {
				kdtree_curr_[ i ].create( rows, cols );
				rows /= 2;
				cols /= 2;
			}
		}
		device::convert (vmaps_g_prev_[0], kdtree_cloud_);
		device::convert (nmaps_g_prev_[0], kdtree_normal_);
		vector< float4 > cloud;
		vector< float4 > normal;
		int c;
		kdtree_cloud_.download( cloud, c );
		kdtree_normal_.download( normal, c );
		vmappcd->points.clear();

		for ( int i = 0; i < ( int )cloud.size(); i++ ) {
			PointNormal pt;
			if ( !pcl_isnan( cloud[ i ].x ) && !pcl_isnan( cloud[ i ].y ) && !pcl_isnan( cloud[ i ].z ) 
				&& !pcl_isnan( normal[ i ].x ) && !pcl_isnan( normal[ i ].y ) && !pcl_isnan( normal[ i ].z ) ) {
				pt.x = cloud[ i ].x;
				pt.y = cloud[ i ].y;
				pt.z = cloud[ i ].z;
				pt.normal_x = normal[ i ].x;
				pt.normal_y = normal[ i ].y;
				pt.normal_z = normal[ i ].z;
				vmappcd->points.push_back( pt );
			}
		}

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}
	{
		//ScopeTime time(">>> icp-all");
		vmaptree.setInputCloud( vmappcd );
		//cout << global_time_ + 1 << endl;

		for (int level_index = LEVELS-1; level_index>=0; --level_index)
		{
			int iter_num = icp_iterations_[level_index];

			// current maps
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];   

			// previous maps
			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

			// We need to transform the maps from global to the local coordinates
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			MapArr& vmap_temp = vmap_g_prev;
			MapArr& nmap_temp = nmap_g_prev;
			device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 

			vector< float4 > cloud_curr;
			int c;
			device::convert (vmap_curr, kdtree_curr_[level_index]);
			kdtree_curr_[ level_index ].download( cloud_curr, c );

			for (int iter = 0; iter < iter_num; ++iter)
			{
				//CONVERT TO DEVICE TYPES
				// CURRENT LOCAL TRANSFORM
				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				estimateCombined (device_cam_rot_local_curr, device_cam_trans_local_curr, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr (level_index), 
					vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());

				Eigen::Affine3f aff;
				aff.linear() = cam_rot_global_curr;
				aff.translation() = cam_trans_global_curr;

				std::vector<int> pointIdxNKNSearch(1);
				std::vector<float> pointNKNSquaredDistance(1);
				float rr = 0.0f;
				int ll = 0;
				AA_.setZero();
				bb_.setZero();
				for ( int i = 0; i < cloud_curr.size(); i++ ) {
					if ( !pcl_isnan( cloud_curr[ i ].x ) && !pcl_isnan( cloud_curr[ i ].y ) && !pcl_isnan( cloud_curr[ i ].z ) ) {
						Eigen::Vector4f xx = aff.matrix() * Eigen::Vector4f( cloud_curr[ i ].x, cloud_curr[ i ].y, cloud_curr[ i ].z, 1.0f );
						PointNormal pt;
						pt.x = xx( 0 );
						pt.y = xx( 1 );
						pt.z = xx( 2 );
						vmaptree.nearestKSearch( pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance );

						//if ( ll == 0 ) {
						//	cout << pt << endl;
						//	cout << vmappcd->points[ pointIdxNKNSearch[ 0 ] ] << endl;
						//	cout << Eigen::Vector4f( cloud_curr[ i ].x, cloud_curr[ i ].y, cloud_curr[ i ].z, 1.0f ) << endl;
						//	cout << vmappcd->points[ 0 ] << endl;
						//	cout << i << endl;
						//}

						if ( pointNKNSquaredDistance[ 0 ] < distThres_ * distThres_ ) {
							Eigen::Vector3f nn( vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_x, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_y, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].normal_z );
							Eigen::Vector3f qq( vmappcd->points[ pointIdxNKNSearch[ 0 ] ].x, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].y, vmappcd->points[ pointIdxNKNSearch[ 0 ] ].z );
							Eigen::Vector3f pp( pt.x, pt.y, pt.z );
							float r = nn.dot( qq - pp );
							Eigen::Vector3f pxn = pp.cross( nn );
							float xx[ 6 ] = { pxn( 0 ), pxn( 1 ), pxn( 2 ), nn( 0 ), nn( 1 ), nn( 2 ) };
							for ( int ii = 0; ii < 6; ii++ ) {
								for ( int jj = 0; jj < 6; jj++ ) {
									AA_( ii, jj ) += xx[ ii ] * xx[ jj ];
								}
								bb_( ii ) += xx[ ii ] * r;
							}
							ll ++; rr += r * r;
						}

					}
				}

				//cout << level_index << " " << ll << "-" << distThres_ << endl;
				//cout << A_ << endl << b_ << endl;
				//cout << AA_ << endl << bb_ << endl;

				//Eigen::Matrix<float, 6, 1> result = ( A_ + amplifier * AA_ ).llt ().solve ( b_ + amplifier * bb_ ).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A_.llt ().solve ( b_ ).cast<float>();
				Eigen::Matrix<float, 6, 1> result = AA_.llt ().solve ( bb_ ).cast<float>();

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				//compose
				cam_trans_global_curr = cam_rot_incremental * cam_trans_global_curr + cam_trans_incremental;
				cam_rot_global_curr = cam_rot_incremental * cam_rot_global_curr;
			}
		}

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	/*
	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);*/
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		//ScopeTime time( ">>> integrate" );
		integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	++global_time_;
	return (true);
}//kdtreeodometry

bool pcl::gpu::KinfuTracker::slac(const DepthMap& depth_rawraw, const DepthMap& depth_raw, const View * pcolor)
{
	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);
	device::bilateralFilter (depth_raw, depths_curr_[0]);

	if (max_icp_distance_ > 0)
		device::truncateDepth(depths_curr_[0], max_icp_distance_);

	for (int i = 1; i < LEVELS; ++i)
		device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

	for (int i = 0; i < LEVELS; ++i)
	{
		device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
		computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
	}
	pcl::device::sync ();

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		cyclical2_.resetBuffer ( tsdf_volume2_ );

		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		//device::integrateTsdfVolume(depths_curr_[ 0 ], intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		if ( global_time_ % slac_num_ == 0 ) {
			base_pose_cur_ = getCameraPose().matrix() * grid_.init_pose_inv_;
			base_pose_cur_inv_ = base_pose_cur_.inverse();
		}
		Eigen::Affine3f cube_pose( base_pose_cur_inv_ * getCameraPose().matrix() );
		Matrix3frm temp = cube_pose.linear().inverse();
		Mat33 aa = device_cast< Mat33 >( temp );
		float3 bb;
		bb.x = cube_pose.translation()(0);
		bb.y = cube_pose.translation()(1);
		bb.z = cube_pose.translation()(2);
		float3 new_volume_size;
		new_volume_size.x = new_volume_size.y = new_volume_size.z = grid_.length_;
		device::integrateTsdfVolume(depth_rawraw, intr, new_volume_size, aa, bb, tsdf_volume2_->getTsdfTruncDist(), tsdf_volume2_->data(), cyclical2_.getBuffer(), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);


		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point


	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}

	{
		//ScopeTime time(">>> icp-all");
		for (int level_index = LEVELS-1; level_index>=0; --level_index)
		{
			int iter_num = icp_iterations_[level_index];

			// current maps
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];   

			// previous maps
			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

			// We need to transform the maps from global to the local coordinates
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			MapArr& vmap_temp = vmap_g_prev;
			MapArr& nmap_temp = nmap_g_prev;
			device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 

			for (int iter = 0; iter < iter_num; ++iter)
			{
				/*
				Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
				float3& device_tcurr = device_cast<float3>(tcurr);
				*/
				//CONVERT TO DEVICE TYPES
				// CURRENT LOCAL TRANSFORM
				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				estimateCombined (device_cam_rot_local_curr, device_cam_trans_local_curr, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr (level_index), 
					vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());

				//checking nullspace
				double det = A_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					reset ();
					return (false);
				}
				//float maxc = A.maxCoeff();

				Eigen::Matrix<float, 6, 1> result = A_.llt ().solve (b_).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				//compose
				cam_trans_global_curr = cam_rot_incremental * cam_trans_global_curr + cam_trans_incremental;
				cam_rot_global_curr = cam_rot_incremental * cam_rot_global_curr;

			}
		}

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  

	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;
	//~ if(integrate)
	//~ std::cout << "\tCamera movement since previous frame was " << (rnorm + alpha * tnorm)/2 << " integrate is set to " << integrate << std::endl;
	//~ else
	//~ std::cout << "Camera movement since previous frame was " << (rnorm + alpha * tnorm)/2 << " integrate is set to " << integrate << std::endl;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	/*
	Matrix3frm Rcurr_inv = Rcurr.inverse ();
	Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
	float3& device_tcurr = device_cast<float3> (tcurr);*/
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		//ScopeTime time( ">>> integrate" );
		integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		//integrateTsdfVolume (depths_curr_[0], intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);

		if ( global_time_ % slac_num_ == 0 ) {
			base_pose_cur_ = getCameraPose().matrix() * grid_.init_pose_inv_;
			base_pose_cur_inv_ = base_pose_cur_.inverse();
		}
		Eigen::Affine3f cube_pose( base_pose_cur_inv_ * getCameraPose().matrix() );
		Matrix3frm temp = cube_pose.linear().inverse();
		Mat33 aa = device_cast< Mat33 >( temp );
		float3 bb;
		bb.x = cube_pose.translation()(0);
		bb.y = cube_pose.translation()(1);
		bb.z = cube_pose.translation()(2);
		float3 new_volume_size;
		new_volume_size.x = new_volume_size.y = new_volume_size.z = grid_.length_;
		integrateTsdfVolume (depth_rawraw, intr, new_volume_size, aa, bb, tsdf_volume2_->getTsdfTruncDist (), tsdf_volume2_->data (), cyclical2_.getBuffer(), depthRawScaled_);
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}
	{
		if ( ( global_time_ + 1 ) % slac_num_ == 0 ) {

			slac_mutex_.lock();

			PointCloud< PointXYZ >::Ptr cloud_ptr( new PointCloud< PointXYZ > );
			PointCloud< PointXYZ >::Ptr normals_ptr( new PointCloud< PointXYZ > );

			DeviceArray<PointXYZ> extracted = tsdf_volume2_->fetchCloud (cloud_buffer_device_, cyclical2_.getBuffer() );             
			// do the in-space normal extraction
			extracted.download (cloud_ptr->points);
			cloud_ptr->width = (int)cloud_ptr->points.size ();
			cloud_ptr->height = 1;
			tsdf_volume2_->fetchNormalsInSpace( extracted, cyclical2_.getBuffer() );
			extracted.download( normals_ptr->points );

			SLACPointCloud::Ptr cloud( new SLACPointCloud( fragments_.size(), slac_resolution_, volume_size_ ) );
			cloud->Init( cloud_ptr, normals_ptr );
			fragments_dummy_.push_back( cloud_ptr );
			fragments_.push_back( cloud );

			base_pose_.push_back( base_pose_cur_ );

			//fragments_.push_back( merge< PointNormal>( *cloud_ptr_, *normals_ptr_ ) );
			// debug use
			char filename[ 1024 ];
			memset( filename, 0, 1024 );
			sprintf( filename, "cloud_bin_%d.pcd", fragments_.back()->index_ );
			pcl::io::savePCDFile( filename, *merge< PointNormal>( *cloud_ptr, *normals_ptr ), true );

			//char filename[ 1024 ];
			//memset( filename, 0, 1024 );
			//sprintf( filename, "cloud_bin_xyzn_%d.xyzn", fragments_.back()->index_ );
			//FILE * f = fopen( filename, "w" );
			//for ( int i = 0; i < cloud_ptr->points.size(); i++ ) {
			//	fprintf( f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", cloud_ptr->points[ i ].x, cloud_ptr->points[ i ].y, cloud_ptr->points[ i ].z, 
			//		normals_ptr->points[ i ].x, normals_ptr->points[ i ].y, normals_ptr->points[ i ].z );
			//}
			//fclose( f );

			// end of debug use

			//cout << ( fragments_.size() - 1 ) * slac_num_ << endl;
			//cout << getCameraPose( ( fragments_.size() - 1 ) * slac_num_ ).matrix() << endl;
			//base_pose_.push_back( getCameraPose( ( fragments_.size() - 1 ) * slac_num_ ).matrix() * grid_.init_pose_inv_ );
			cout << "New fragment with id : " << fragments_.back()->index_ << endl;
			cout << "\t" << fragments_.back()->points_.size() << " points" << endl;
			cout << "\tInit transformation : " << endl << base_pose_.back() << endl;

			//if ( fragments_.size() > 1 ) {
			// add correspondence
			for ( int idid = 0; idid < 3 && ( int )fragments_.size() - 2 - idid >= 0; idid++ ) {
				int id1 = fragments_.size() - 2 - idid;
				int id2 = fragments_.size() - 1;

				Correspondence::Ptr cor( new Correspondence( id1, id2 ) );
				corres_.push_back( cor );

				// trans
				//cor.trans_ = base_pose_[ id1 ].inverse() * base_pose_[ id2 ];
				//cout << cor.trans_ << endl;
				cor->trans_ = base_pose_[ id1 ].inverse() * base_pose_[ id2 ];

				// corres
				int old_id = -1;
				pcl::KdTreeFLANN< pcl::PointXYZ > tree;
				const int K = 1;
				std::vector< int > pointIdxNKNSearch(K);
				std::vector< float > pointNKNSquaredDistance(K);
				PointCloud< pcl::PointXYZ >::Ptr pcd0 = fragments_dummy_[ id1 ];
				PointCloud< pcl::PointXYZ >::Ptr pcd1 = fragments_dummy_[ id2 ];

				PointCloud< pcl::PointXYZ >::Ptr transformed( new pcl::PointCloud< pcl::PointXYZ > );
				pcl::transformPointCloud( *pcd1, *transformed, cor->trans_ );
				tree.setInputCloud( pcd0 );

				for ( int k = 0; k < ( int )transformed->size(); k++ ) {
					if ( tree.nearestKSearch( transformed->points[ k ], K, pointIdxNKNSearch, pointNKNSquaredDistance ) > 0 ) {
						if ( pointNKNSquaredDistance[ 0 ] < dist_thresh_ * dist_thresh_ &&
							fragments_[ id1 ]->IsValidPoint( pointIdxNKNSearch[ 0 ] ) &&
							fragments_[ id2 ]->IsValidPoint( k ) ) {
								cor->corres_.push_back( CorrespondencePair( pointIdxNKNSearch[ 0 ], k ) );
						}
					}
				}

				cout << "\tAdd correspondences : " << cor->corres_.size() << endl;

				// debug use
				//char filename[ 1024 ];
				//memset( filename, 0, 1024 );
				//sprintf( filename, "corres_%d_%d.txt", cor->idx0_, cor->idx1_ );
				//FILE * f = fopen( filename, "w" );
				//for ( int i = 0; i < cor->corres_.size(); i++ ) {
				//	fprintf( f, "%d %d\n", cor->corres_[ i ].first, cor->corres_[ i ].second );
				//}
				//fclose( f );
				// end of debug use
			}

			slac_mutex_.unlock();

			tsdf_volume2_->reset();

			if ( fragments_.size() == 4 ) {
				//FILE * f = fopen( "reg_output.log", "w" );
				//for ( int i = 0; i < 7; i++ ) {
				//	fprintf( f, "%d %d %d\n", corres_[ i ]->idx0_, corres_[ i ]->idx1_, 50000 );
				//	fprintf( f, "%.8f %.8f %.8f %.8f\n", corres_[ i ]->trans_( 0, 0 ), corres_[ i ]->trans_( 0, 1 ), corres_[ i ]->trans_( 0, 2 ), corres_[ i ]->trans_( 0, 3 ) );
				//	fprintf( f, "%.8f %.8f %.8f %.8f\n", corres_[ i ]->trans_( 1, 0 ), corres_[ i ]->trans_( 1, 1 ), corres_[ i ]->trans_( 1, 2 ), corres_[ i ]->trans_( 1, 3 ) );
				//	fprintf( f, "%.8f %.8f %.8f %.8f\n", corres_[ i ]->trans_( 2, 0 ), corres_[ i ]->trans_( 2, 1 ), corres_[ i ]->trans_( 2, 2 ), corres_[ i ]->trans_( 2, 3 ) );
				//	fprintf( f, "%.8f %.8f %.8f %.8f\n", corres_[ i ]->trans_( 3, 0 ), corres_[ i ]->trans_( 3, 1 ), corres_[ i ]->trans_( 3, 2 ), corres_[ i ]->trans_( 3, 3 ) );
				//}
				//fclose( f );
				boost::thread t( &KinfuTracker::OptimizeSLAC, this );
			}
		}
	}

	++global_time_;
	return (true);
}//slac

void pcl::gpu::KinfuTracker::OptimizeSLAC()
{
	while ( true ) {
		slac_mutex_.lock();
		cout << corres_[ 0 ]->corres_[ 0 ].first << " - " << corres_[ 0 ]->corres_[ 0 ].second << endl;

		std::vector< SLACPointCloud::Ptr > fragments;
		std::vector< Correspondence::Ptr > corres;
		fragments.resize( fragments_.size() );
		corres.resize( corres_.size() );
		for ( int i = 0; i < fragments_.size(); i++ ) {
			SLACPointCloud::Ptr pp( new SLACPointCloud );
			*pp = *( fragments_[ i ] );
			fragments[ i ] = pp;
		}
		for ( int i = 0; i < corres_.size(); i++ ) {
			corres[ i ] = corres_[ i ];
		}
		int num = fragments_.size();
		int resolution = 12;
		int nper = ( resolution + 1 ) * ( resolution + 1 ) * ( resolution + 1 ) * 3;
		int matrix_size = 6 * num + nper;
		float unit_length = grid_.unit_length_;
		std::vector< Eigen::Matrix4d > ipose;
		ipose.resize( base_pose_.size() );
		for ( int i = 0; i < base_pose_.size(); i++ ) {
			ipose[ i ] = base_pose_[ i ].cast< double >();
		}
		slac_mutex_.unlock();

		cout << corres[ 0 ]->corres_[ 0 ].first << " - " << corres[ 0 ]->corres_[ 0 ].second << endl;

		double default_weight = num;
		int max_iteration = 5;

		std::vector< Eigen::Matrix3d > pose_rot_t;
		pose_rot_t.resize( base_pose_.size() );
		std::vector< Eigen::Matrix4d > pose;
		pose.resize( ipose.size() );

		//boost::this_thread::sleep(boost::posix_time::milliseconds(3000));
		Eigen::VectorXd ictr( matrix_size );
		Eigen::VectorXd thisCtr( matrix_size );
		for ( int i = 0; i <= resolution; i++ ) {
			for ( int j = 0; j <= resolution; j++ ) {
				for ( int k = 0; k <= resolution; k++ ) {
					Eigen::Vector4d pos( i * unit_length, j * unit_length, k * unit_length, 1 );
					int idx = i + j * ( resolution + 1 ) + k * ( resolution + 1 ) * ( resolution + 1 );
					ictr( num * 6 + idx * 3 + 0 ) = pos( 0 );
					ictr( num * 6 + idx * 3 + 1 ) = pos( 1 );
					ictr( num * 6 + idx * 3 + 2 ) = pos( 2 );
				}
			}
		}
		thisCtr = ictr;
		Eigen::VectorXd expand_ctr( nper * num );

		for ( int i = 0; i < ( int )ipose.size(); i++ ) {
			pose[ i ] = ipose[ i ];
			pose_rot_t[ i ] = pose[ i ].block< 3, 3 >( 0, 0 ).transpose();
			fragments[ i ]->UpdatePose( pose[ i ].cast< float >() );
		}

		{
			//pcl::ScopeTime ttime( "Neat Optimization" );

			Eigen::SparseMatrix< double > baseJJ( matrix_size, matrix_size );
			TripletVector tri;
			HashSparseMatrix mat( 6 * num, 6 * num );
			for ( int i = 0; i <= resolution; i++ ) {
				for ( int j = 0; j <= resolution; j++ ) {
					for ( int k = 0; k <= resolution; k++ ) {
						int idx[ 2 ] = { grid_.GetIndex( i, j, k ) * 3, 0 };
						double val[ 2 ] = { 1, -1 };
						if ( i > 0 ) {
							idx[ 1 ] = grid_.GetIndex( i - 1, j, k ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
						if ( i < resolution ) {
							idx[ 1 ] = grid_.GetIndex( i + 1, j, k ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
						if ( j > 0 ) {
							idx[ 1 ] = grid_.GetIndex( i, j - 1, k ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
						if ( j < resolution ) {
							idx[ 1 ] = grid_.GetIndex( i, j + 1, k ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
						if ( k > 0 ) {
							idx[ 1 ] = grid_.GetIndex( i, j, k - 1 ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
						if ( k < resolution ) {
							idx[ 1 ] = grid_.GetIndex( i, j, k + 1 ) * 3;
							mat.AddHessian2( idx, val, tri );
						}
					}
				}
			}

			int base_anchor = grid_.GetIndex( resolution / 2, resolution / 2, 0 ) * 3;

			mat.Add( base_anchor + 0, base_anchor + 0, 1, tri );
			mat.Add( base_anchor + 1, base_anchor + 1, 1, tri );
			mat.Add( base_anchor + 2, base_anchor + 2, 1, tri );

			baseJJ.setFromTriplets( tri.begin(), tri.end() );

			baseJJ *= default_weight;

			for ( int itr = 0; itr < max_iteration; itr++ ) {
				Eigen::VectorXd thisJb( matrix_size );
				Eigen::MatrixXd thisJJ( baseJJ.triangularView< Eigen::Upper >() );
				Eigen::VectorXd result;

				thisJJ( 6 * num - 6, 6 * num - 6 ) += 1.0;
				thisJJ( 6 * num - 5, 6 * num - 5 ) += 1.0;
				thisJJ( 6 * num - 4, 6 * num - 4 ) += 1.0;
				thisJJ( 6 * num - 3, 6 * num - 3 ) += 1.0;
				thisJJ( 6 * num - 2, 6 * num - 2 ) += 1.0;
				thisJJ( 6 * num - 1, 6 * num - 1 ) += 1.0;

				thisJb.setZero();

				double thisscore = 0.0;
				int nprocessed = 0;

#pragma omp parallel for num_threads( 6 ) schedule( dynamic )
				for ( int l = 0; l < ( int )corres.size(); l++ ) {
					int i = corres[ l ]->idx0_;
					int j = corres[ l ]->idx1_;
					const int buck_size = 12 + 24 * 2;
					int idx[ buck_size ];
					double val[ buck_size ];
					double b;
					double score = 0.0;
					Eigen::MatrixXd tempJJ( matrix_size, matrix_size );
					tempJJ.setZero();
					Eigen::VectorXd tempJb( matrix_size );
					tempJb.setZero();

					for ( int k = 0; k < ( int )corres[ l ]->corres_.size(); k++ ) {
						int ii = corres[ l ]->corres_[ k ].first;
						int jj = corres[ l ]->corres_[ k ].second;
						SLACPoint & pi = fragments[ i ]->points_[ ii ];
						SLACPoint & pj = fragments[ j ]->points_[ jj ];
						Eigen::Vector3d ppi( pi.p_[ 0 ], pi.p_[ 1 ], pi.p_[ 2 ] );
						Eigen::Vector3d ppj( pj.p_[ 0 ], pj.p_[ 1 ], pj.p_[ 2 ] );
						Eigen::Vector3d npi( pi.n_[ 0 ], pi.n_[ 1 ], pi.n_[ 2 ] );
						Eigen::Vector3d npj( pj.n_[ 0 ], pj.n_[ 1 ], pj.n_[ 2 ] );
						Eigen::Vector3d diff = ppi - ppj;
						b = diff.dot( npi );
						score += b * b;

						//if ( score > 1 || _isnan( score ) ) {
						//	if ( fragments[ i ]->IsValidPoint( ii ) && fragments[ j ]->IsValidPoint( jj ) ) {
						//		cout << npj << endl;
						//		cout << fragments[ j ]->points_[ jj ].n_[ 0 ] << endl;
						//		cout << fragments[ j ]->points_[ jj ].n_[ 1 ] << endl;
						//		cout << fragments[ j ]->points_[ jj ].n_[ 2 ] << endl;
						//		cout << _isnan( fragments[ j ]->points_[ jj ].n_[ 0 ] ) << endl;

						//	}
						//	cout << l << endl;
						//	cout << i << "-" << ii << endl;
						//	cout << j << "-" << jj << endl;
						//	cout << npi << endl << npj << endl;
						//	cout << ppi << endl << ppj << endl << b << endl << score << endl << endl;
						//}

						idx[ 0 ] = i * 6;
						idx[ 1 ] = i * 6 + 1;
						idx[ 2 ] = i * 6 + 2;
						idx[ 3 ] = i * 6 + 3;
						idx[ 4 ] = i * 6 + 4;
						idx[ 5 ] = i * 6 + 5;
						idx[ 6 ] = j * 6;
						idx[ 7 ] = j * 6 + 1;
						idx[ 8 ] = j * 6 + 2;
						idx[ 9 ] = j * 6 + 3;
						idx[ 10 ] = j * 6 + 4;
						idx[ 11 ] = j * 6 + 5;

						Eigen::Vector3d temp = ppj.cross( npi );

						val[ 0 ] = temp( 0 );
						val[ 1 ] = temp( 1 );
						val[ 2 ] = temp( 2 );
						val[ 3 ] = npi( 0 );
						val[ 4 ] = npi( 1 );
						val[ 5 ] = npi( 2 );
						val[ 6 ] = -temp( 0 );
						val[ 7 ] = -temp( 1 );
						val[ 8 ] = -temp( 2 );
						val[ 9 ] = -npi( 0 );
						val[ 10 ] = -npi( 1 );
						val[ 11 ] = -npi( 2 );

						// next part of Jacobian
						// deal with control lattices
						Eigen::Vector3d dTi = pose_rot_t[ i ] * npi;
						Eigen::Vector3d dTj = - pose_rot_t[ j ] * npi;

						for ( int ll = 0; ll < 8; ll++ ) {
							for ( int xyz = 0; xyz < 3; xyz++ ) {
								idx[ 12 + ll * 3 + xyz ] = 6 * num + pi.idx_[ ll ] + xyz;
								val[ 12 + ll * 3 + xyz ] = pi.val_[ ll ] * dTi( xyz );
							}
						}

						for ( int ll = 0; ll < 8; ll++ ) {
							for ( int xyz = 0; xyz < 3; xyz++ ) {
								idx[ 12 + 24 + ll * 3 + xyz ] = 6 * num + pj.idx_[ ll ] + xyz;
								val[ 12 + 24 + ll * 3 + xyz ] = pj.val_[ ll ] * dTj( xyz );
							}
						}

						for ( int i = 0; i < buck_size; i++ ) {
							tempJJ( idx[ i ], idx[ i ] ) += val[ i ] * val[ i ];
							for ( int j = i + 1; j < buck_size; j++ ) {
								if ( idx[ i ] == idx[ j ] ) {
									tempJJ( idx[ i ], idx[ j ] ) += 2 * val[ i ] * val[ j ];
								} else if ( idx[ i ] < idx[ j ] ) {
									tempJJ( idx[ i ], idx[ j ] ) += val[ i ] * val[ j ];
								} else {
									tempJJ( idx[ j ], idx[ i ] ) += val[ i ] * val[ j ];
								}
							}
							tempJb( idx[ i ] ) += b * val[ i ];
						}
					}

#pragma omp critical
					{
						nprocessed++;
						thisscore += score;
						thisJJ += tempJJ;
						thisJb += tempJb;
					}
				}
				PCL_INFO( " ... Done.\n" );
				PCL_INFO( "Data error score is : %.2f\n", thisscore );
				//cout << thisJJ << endl;
				//cout << thisJb << endl;
				//cout << ipose[ 0 ] << endl;
				//cout << ipose[ 1 ] << endl;
				//cout << ipose[ 7 ] << endl;
				//if ( itr == 0 ) {
				//	Eigen::MatrixXf temp = thisJJ - Eigen::MatrixXf( baseJJ.triangularView< Eigen::Upper >() );
				//	cout << temp.block< 10, 10 >( 0, 0 ) << endl;
				//	cout << temp.block< 10, 10 >( 0, 2710 ) << endl;
				//	cout << temp.block< 10, 10 >( 2710, 2710 ) << endl;
				//	cout << temp.block< 10, 10 >( matrix_size - 11, matrix_size - 11 ) << endl;
				//	cout << thisJJ.block< 10, 10 >( 0, 0 ) << endl;
				//	cout << thisJJ.block< 10, 10 >( 0, 2710 ) << endl;
				//	cout << thisJJ.block< 10, 10 >( 2710, 2710 ) << endl;
				//	cout << thisJJ.block< 10, 10 >( matrix_size - 11, matrix_size - 11 ) << endl;
				//}

				Eigen::SparseMatrix< double > thisSparseJJ = thisJJ.sparseView();
				Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix< double >, Eigen::Upper > solver;
				//Eigen::ConjugateGradient< SparseMatrix, Eigen::Upper > solver;
				solver.analyzePattern( thisSparseJJ );
				solver.factorize( thisSparseJJ );

				Eigen::VectorXd tempCtr( thisCtr );

				// regularizer

				Eigen::VectorXd dataJb( thisJb );

				//for ( int tmp = 0; tmp < 10; tmp ++ ) {

				Eigen::VectorXd baseJb( matrix_size );
				baseJb.setZero();

				double regscore = 0.0;
				for ( int i = 0; i <= resolution; i++ ) {
					for ( int j = 0; j <= resolution; j++ ) {
						for ( int k = 0; k <= resolution; k++ ) {
							int idx = grid_.GetIndex( i, j, k ) * 3 + 6 * num;
							std::vector< int > idxx;
							if ( i > 0 ) {
								idxx.push_back( grid_.GetIndex( i - 1, j, k ) * 3 + 6 * num );
							}
							if ( i < resolution ) {
								idxx.push_back( grid_.GetIndex( i + 1, j, k ) * 3 + 6 * num );
							}
							if ( j > 0 ) {
								idxx.push_back( grid_.GetIndex( i, j - 1, k ) * 3 + 6 * num );
							}
							if ( j < resolution ) {
								idxx.push_back( grid_.GetIndex( i, j + 1, k ) * 3 + 6 * num );
							}
							if ( k > 0 ) {
								idxx.push_back( grid_.GetIndex( i, j, k - 1 ) * 3 + 6 * num );
							}
							if ( k < resolution ) {
								idxx.push_back( grid_.GetIndex( i, j, k + 1 ) * 3 + 6 * num );
							}

							Eigen::Matrix3d R;
							if ( i == resolution / 2 && j == resolution / 2 && k == 0 ) {
								// anchor point, Rotation matrix is always identity
								R = R.Identity();
							} else {
								R = GetRotationd( idx, idxx, ictr, tempCtr );
							}

							for ( int t = 0; t < ( int )idxx.size(); t++ ) {
								Eigen::Vector3d bx = Eigen::Vector3d( tempCtr( idx ) - tempCtr( idxx[ t ] ), tempCtr( idx + 1 ) - tempCtr( idxx[ t ] + 1 ), tempCtr( idx + 2 ) - tempCtr( idxx[ t ] + 2 ) )
									- R * Eigen::Vector3d( ictr( idx ) - ictr( idxx[ t ] ), ictr( idx + 1 ) - ictr( idxx[ t ] + 1 ), ictr( idx + 2 ) - ictr( idxx[ t ] + 2 ) );
								regscore += default_weight * bx.transpose() * bx;
								baseJb( idx ) += bx( 0 ) * default_weight;
								baseJb( idxx[ t ] ) -= bx( 0 ) * default_weight;
								baseJb( idx + 1 ) += bx( 1 ) * default_weight;
								baseJb( idxx[ t ] + 1 ) -= bx( 1 ) * default_weight;
								baseJb( idx + 2 ) += bx( 2 ) * default_weight;
								baseJb( idxx[ t ] + 2 ) -= bx( 2 ) * default_weight;
							}
						}
					}
				}
				thisJb = dataJb + baseJb;

				PCL_INFO( "Regularization error score is : %.2f\n", regscore );

				result = - solver.solve( thisJb );
				//cout << result << endl;

				for ( int l = 0; l < nper; l++ ) {
					tempCtr( num * 6 + l ) = thisCtr( num * 6 + l ) + result( num * 6 + l );
				}
				//}

				for ( int l = 0; l < nper; l++ ) {
					thisCtr( num * 6 + l ) += result( num * 6 + l );
				}

				for ( int l = 0; l < num; l++ ) {
					Eigen::Affine3d aff_mat;
					aff_mat.linear() = ( Eigen::Matrix3d ) Eigen::AngleAxisd( result( l * 6 + 2 ), Eigen::Vector3d::UnitZ() )
						* Eigen::AngleAxisd( result( l * 6 + 1 ), Eigen::Vector3d::UnitY() )
						* Eigen::AngleAxisd( result( l * 6 + 0 ), Eigen::Vector3d::UnitX() );
					aff_mat.translation() = Eigen::Vector3d( result( l * 6 + 3 ), result( l * 6 + 4 ), result( l * 6 + 5 ) );
					//cout << aff_mat.matrix() << endl << endl;
					// update
					pose[ l ] = aff_mat.matrix() * pose[ l ];
					pose_rot_t[ l ] = pose[ l ].block< 3, 3 >( 0, 0 ).transpose();
				}
				expand_ctr.setZero();
				for ( int l = 0; l < num; l++ ) {
					for ( int i = 0; i < nper / 3; i++ ) {
						Eigen::Vector4d pos = pose[ l ] * Eigen::Vector4d( thisCtr( num * 6 + i * 3 + 0 ), thisCtr( num * 6 + i * 3 + 1 ), thisCtr( num * 6 + i * 3 + 2 ), 1 );
						expand_ctr( l * nper + i * 3 + 0 ) = pos( 0 );
						expand_ctr( l * nper + i * 3 + 1 ) = pos( 1 );
						expand_ctr( l * nper + i * 3 + 2 ) = pos( 2 );
					}
				}
				for ( int l = 0; l < num; l++ ) {
					fragments[ l ]->UpdateAllPointPN( expand_ctr );
				}
			}
		}

		/*

		//Eigen::VectorXd ctr( nper_ * num_ );
		//Pose2Ctr( pose_, ctr );
		ExpandCtr( thisCtr, expand_ctr );
		SaveCtr( expand_ctr, ctr_filename_ );
		SavePoints( expand_ctr, sample_filename_ );
		*/

		slac_mutex_.lock();
		// write control grid;
		for ( int i = 0; i < nper / 3; i++ ) {
			grid_.ctr_[ i ]( 0 ) = thisCtr( num * 6 + i * 3 + 0 );
			grid_.ctr_[ i ]( 1 ) = thisCtr( num * 6 + i * 3 + 1 );
			grid_.ctr_[ i ]( 2 ) = thisCtr( num * 6 + i * 3 + 2 );
		}
		slac_mutex_.unlock();

		cout << "boosted " << fragments.size() << endl;
	}
}//OptimizeSLAC


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::operator() (
	const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
	const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
	const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
	const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
	const DepthMap& depth_raw, const View * pcolor, FramedTransformation * frame_ptr
	)
{
	//ScopeTime time( "Kinfu Tracker All" );
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->ResetFlag ) ) {
		reset();
		if ( frame_ptr->type_ == frame_ptr->DirectApply )
		{
			Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
			rmats_[0] = aff_rgbd.linear();
			tvecs_[0] = aff_rgbd.translation();
		}
	}

	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) ) {
	}
	else
	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr_[0]);
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
		pcl::device::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		//device::integrateTsdfVolume(depths_curr_[ 0 ], intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	const int sobelSize = 3;
	const double sobelScale = 1./8;

	cv::Mat depth0 = _depth0.clone(),
		depth1 = _depth1.clone();

	// check RGB-D input data
	CV_Assert( !image0.empty() );
	CV_Assert( image0.type() == CV_8UC1 );
	CV_Assert( depth0.type() == CV_32FC1 && depth0.size() == image0.size() );

	CV_Assert( image1.size() == image0.size() );
	CV_Assert( image1.type() == CV_8UC1 );
	CV_Assert( depth1.type() == CV_32FC1 && depth1.size() == image0.size() );

	// check masks
	CV_Assert( validMask0.empty() || (validMask0.type() == CV_8UC1 && validMask0.size() == image0.size()) );
	CV_Assert( validMask1.empty() || (validMask1.type() == CV_8UC1 && validMask1.size() == image0.size()) );

	// check camera params
	CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == cv::Size(3,3) );

	// other checks
	CV_Assert( iterCounts.empty() || minGradientMagnitudes.empty() ||
		minGradientMagnitudes.size() == iterCounts.size() );

	vector<int> defaultIterCounts;
	vector<float> defaultMinGradMagnitudes;
	vector<int> const* iterCountsPtr = &iterCounts;
	vector<float> const* minGradientMagnitudesPtr = &minGradientMagnitudes;

	preprocessDepth( depth0, depth1, validMask0, validMask1, minDepth, maxDepth );

	vector<cv::Mat> pyramidImage0, pyramidDepth0,
		pyramidImage1, pyramidDepth1, pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1,
		pyramidCameraMatrix;
	buildPyramids( image0, image1, depth0, depth1, cameraMatrix, sobelSize, sobelScale, *minGradientMagnitudesPtr,
		pyramidImage0, pyramidDepth0, pyramidImage1, pyramidDepth1,
		pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1, pyramidCameraMatrix );

	cv::Mat resultRt = cv::Mat::eye(4,4,CV_64FC1);
	cv::Mat currRt, ksi;

	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->InitializeOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	} else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->IncrementalOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ * getCameraPose().matrix() );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	}


	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) )
	{
		rmats_.push_back (cam_rot_global_prev); 
		tvecs_.push_back (cam_trans_global_prev);
	}
	else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->DirectApply ) )
	{
		//Eigen::Affine3f aff_rgbd( getCameraPose( 0 ).matrix() * frame_ptr->transformation_ );
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_curr = aff_rgbd.linear();
		cam_trans_global_curr = aff_rgbd.translation();
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}
	else
	{
		//ScopeTime time(">>> icp-all");

		Eigen::Matrix4f result_trans = Eigen::Matrix4f::Identity();
		Eigen::Affine3f mat_prev;
		mat_prev.linear() = cam_rot_global_prev;
		mat_prev.translation() = cam_trans_global_prev;

		for (int level_index = LEVELS-1; level_index>=0; --level_index)
		{
			int iter_num = icp_iterations_[level_index];

			// current maps
			MapArr& vmap_curr = vmaps_curr_[level_index];
			MapArr& nmap_curr = nmaps_curr_[level_index];   

			// previous maps
			MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
			MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

			// We need to transform the maps from global to the local coordinates
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			MapArr& vmap_temp = vmap_g_prev;
			MapArr& nmap_temp = nmap_g_prev;
			device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 

			const cv::Mat& levelCameraMatrix = pyramidCameraMatrix[ level_index ];

			const cv::Mat& levelImage0 = pyramidImage0[ level_index ];
			const cv::Mat& levelDepth0 = pyramidDepth0[ level_index ];
			cv::Mat levelCloud0;
			cvtDepth2Cloud( pyramidDepth0[ level_index ], levelCloud0, levelCameraMatrix );

			const cv::Mat& levelImage1 = pyramidImage1[ level_index ];
			const cv::Mat& levelDepth1 = pyramidDepth1[ level_index ];
			const cv::Mat& level_dI_dx1 = pyramid_dI_dx1[ level_index ];
			const cv::Mat& level_dI_dy1 = pyramid_dI_dy1[ level_index ];

			CV_Assert( level_dI_dx1.type() == CV_16S );
			CV_Assert( level_dI_dy1.type() == CV_16S );

			const double fx = levelCameraMatrix.at<double>(0,0);
			const double fy = levelCameraMatrix.at<double>(1,1);
			const double determinantThreshold = 1e-6;

			cv::Mat corresps( levelImage0.size(), levelImage0.type() );

			for (int iter = 0; iter < iter_num; ++iter)
			{
				// rgbd odometry part
				bool odo_good = true;

				int correspsCount = computeCorresp( levelCameraMatrix, levelCameraMatrix.inv(), resultRt.inv(cv::DECOMP_SVD),
					levelDepth0, levelDepth1, pyramidTexturedMask1[ level_index ], maxDepthDiff,
					corresps );
				if( correspsCount == 0 ) {
					odo_good = false;
				} else {
					odo_good = computeKsi( 0,
						levelImage0, levelCloud0,
						levelImage1, level_dI_dx1, level_dI_dy1,
						corresps, correspsCount,
						fx, fy, sobelScale, determinantThreshold,
						ksi, AA_, bb_ );
				}

				if ( odo_good == false ) {
					AA_.setZero();
					bb_.setZero();
				}

				//computeProjectiveMatrix( ksi, currRt );
				//resultRt = currRt * resultRt;

				//CONVERT TO DEVICE TYPES
				// CURRENT LOCAL TRANSFORM
				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				estimateCombinedPrevSpace (device_cam_rot_local_curr, device_cam_trans_local_curr, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, intr (level_index), 
					vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A_.data (), b_.data ());

				//checking nullspace
				//AAA_ = A_ + AA_ * ( 0.1 / 255.0 / 255.0 );
				//bbb_ = b_ + bb_ * ( 0.1 / 255.0 / 255.0 );
				//AAA_ = A_ + AA_ * ( 1.0 / 255.0 / 255.0 );
				//bbb_ = b_ + bb_ * ( 1.0 / 255.0 / 255.0 );
				AAA_ = A_ + AA_ * ( 25.0 / 255.0 / 255.0 );
				bbb_ = b_ + bb_ * ( 25.0 / 255.0 / 255.0 );
				//AAA_ = AA_;
				//bbb_ = bb_;

				double det = AAA_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					if ( frame_ptr != NULL && frame_ptr->type_ == frame_ptr->InitializeOnly ) {
						Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
						cam_rot_global_curr = aff_rgbd.linear();
						cam_trans_global_curr = aff_rgbd.translation();
						break;
					} else {
						reset ();
						return (false);
					}
				}

				//Eigen::Matrix<float, 6, 1> result = A_.llt ().solve (b_).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
				//Eigen::Matrix<float, 6, 1> result = AA_.llt ().solve (bb_).cast<float>();
				Eigen::Matrix< float, 6, 1 > result = AAA_.llt().solve( bbb_ ).cast< float >();

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				//cout << AA_ / ( 255.0 * 255.0 ) << endl;
				//cout << A_ << endl;
				//cout << endl;

				//compose
				//cam_trans_global_curr = cam_rot_incremental * cam_trans_global_curr + cam_trans_incremental;
				//cam_rot_global_curr = cam_rot_incremental * cam_rot_global_curr;
				Eigen::Affine3f mat_inc;
				mat_inc.linear() = cam_rot_incremental;
				mat_inc.translation() = cam_trans_incremental;

				result_trans = mat_inc * result_trans;
				Eigen::Matrix4d temp = result_trans.cast< double >();
				eigen2cv( temp, resultRt );

				Eigen::Affine3f mat_curr;
				mat_curr.matrix() = mat_prev.matrix() * result_trans;

				cam_rot_global_curr = mat_curr.linear();
				cam_trans_global_curr = mat_curr.translation();
			}
		}

		/*
		cout << global_time_ << endl;
		cout << cam_trans_global_curr << endl;
		cout << cam_rot_global_curr << endl;
		cout << resultRt << endl;
		cout << endl;
		*/

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreIntegrationFlag ) ) {
		} else {
			//ScopeTime time( ">>> integrate" );
			integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		}
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	++global_time_;
	return (true);
}//operator()


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::rgbdodometry(
	const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
	const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
	const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
	const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
	const DepthMap& depth_raw, const View * pcolor, FramedTransformation * frame_ptr
	)
{
	//ScopeTime time( "Kinfu Tracker All" );
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->ResetFlag ) ) {
		reset();
		if ( frame_ptr->type_ == frame_ptr->DirectApply )
		{
			Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
			rmats_[0] = aff_rgbd.linear();
			tvecs_[0] = aff_rgbd.translation();
		}
	}

	device::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) ) {
	}
	else
	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr_[0]);
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
		pcl::device::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		//device::integrateTsdfVolume(depths_curr_[ 0 ], intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	const int sobelSize = 3;
	const double sobelScale = 1./8;

	cv::Mat depth0 = _depth0.clone(),
		depth1 = _depth1.clone();

	// check RGB-D input data
	CV_Assert( !image0.empty() );
	CV_Assert( image0.type() == CV_8UC1 );
	CV_Assert( depth0.type() == CV_32FC1 && depth0.size() == image0.size() );

	CV_Assert( image1.size() == image0.size() );
	CV_Assert( image1.type() == CV_8UC1 );
	CV_Assert( depth1.type() == CV_32FC1 && depth1.size() == image0.size() );

	// check masks
	CV_Assert( validMask0.empty() || (validMask0.type() == CV_8UC1 && validMask0.size() == image0.size()) );
	CV_Assert( validMask1.empty() || (validMask1.type() == CV_8UC1 && validMask1.size() == image0.size()) );

	// check camera params
	CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == cv::Size(3,3) );

	// other checks
	CV_Assert( iterCounts.empty() || minGradientMagnitudes.empty() ||
		minGradientMagnitudes.size() == iterCounts.size() );

	vector<int> defaultIterCounts;
	vector<float> defaultMinGradMagnitudes;
	vector<int> const* iterCountsPtr = &iterCounts;
	vector<float> const* minGradientMagnitudesPtr = &minGradientMagnitudes;
	if( iterCounts.empty() || minGradientMagnitudes.empty() )
	{
		defaultIterCounts.resize(4);
		defaultIterCounts[0] = 7;
		defaultIterCounts[1] = 7;
		defaultIterCounts[2] = 7;
		defaultIterCounts[3] = 10;

		defaultMinGradMagnitudes.resize(4);
		defaultMinGradMagnitudes[0] = 12;
		defaultMinGradMagnitudes[1] = 5;
		defaultMinGradMagnitudes[2] = 3;
		defaultMinGradMagnitudes[3] = 1;

		iterCountsPtr = &defaultIterCounts;
		minGradientMagnitudesPtr = &defaultMinGradMagnitudes;
	}

	preprocessDepth( depth0, depth1, validMask0, validMask1, minDepth, maxDepth );

	vector<cv::Mat> pyramidImage0, pyramidDepth0,
		pyramidImage1, pyramidDepth1, pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1,
		pyramidCameraMatrix;
	buildPyramids( image0, image1, depth0, depth1, cameraMatrix, sobelSize, sobelScale, *minGradientMagnitudesPtr,
		pyramidImage0, pyramidDepth0, pyramidImage1, pyramidDepth1,
		pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1, pyramidCameraMatrix );

	cv::Mat resultRt = cv::Mat::eye(4,4,CV_64FC1);
	cv::Mat currRt, ksi;

	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->InitializeOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	} else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->IncrementalOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ * getCameraPose().matrix() );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	}


	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		device::tranformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::sync ();
	}

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) )
	{
		rmats_.push_back (cam_rot_global_prev); 
		tvecs_.push_back (cam_trans_global_prev);
	}
	else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->DirectApply ) )
	{
		//Eigen::Affine3f aff_rgbd( getCameraPose( 0 ).matrix() * frame_ptr->transformation_ );
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_curr = aff_rgbd.linear();
		cam_trans_global_curr = aff_rgbd.translation();
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}
	else
	{
		//ScopeTime time(">>> icp-all");

		Eigen::Matrix4f result_trans = Eigen::Matrix4f::Identity();
		Eigen::Affine3f mat_prev;
		mat_prev.linear() = cam_rot_global_prev;
		mat_prev.translation() = cam_trans_global_prev;

		for( int level_index = (int)iterCountsPtr->size() - 1; level_index >= 0; level_index-- )
		{
			// We need to transform the maps from global to the local coordinates
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			const cv::Mat& levelCameraMatrix = pyramidCameraMatrix[ level_index ];

			const cv::Mat& levelImage0 = pyramidImage0[ level_index ];
			const cv::Mat& levelDepth0 = pyramidDepth0[ level_index ];
			cv::Mat levelCloud0;
			cvtDepth2Cloud( pyramidDepth0[ level_index ], levelCloud0, levelCameraMatrix );

			const cv::Mat& levelImage1 = pyramidImage1[ level_index ];
			const cv::Mat& levelDepth1 = pyramidDepth1[ level_index ];
			const cv::Mat& level_dI_dx1 = pyramid_dI_dx1[ level_index ];
			const cv::Mat& level_dI_dy1 = pyramid_dI_dy1[ level_index ];

			CV_Assert( level_dI_dx1.type() == CV_16S );
			CV_Assert( level_dI_dy1.type() == CV_16S );

			const double fx = levelCameraMatrix.at<double>(0,0);
			const double fy = levelCameraMatrix.at<double>(1,1);
			const double determinantThreshold = 1e-6;

			cv::Mat corresps( levelImage0.size(), levelImage0.type() );

			for( int iter = 0; iter < (*iterCountsPtr)[ level_index ]; iter ++ ) {
				bool odo_good = true;

				int correspsCount = computeCorresp( levelCameraMatrix, levelCameraMatrix.inv(), resultRt.inv(cv::DECOMP_SVD),
					levelDepth0, levelDepth1, pyramidTexturedMask1[ level_index ], maxDepthDiff,
					corresps );
				if( correspsCount == 0 ) {
					odo_good = false;
				} else {
					odo_good = computeKsi( 0,
						levelImage0, levelCloud0,
						levelImage1, level_dI_dx1, level_dI_dy1,
						corresps, correspsCount,
						fx, fy, sobelScale, determinantThreshold,
						ksi, AA_, bb_ );
				}

				if ( odo_good == false ) {
					AA_.setZero();
					bb_.setZero();
				}

				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				AAA_ = AA_;
				bbb_ = bb_;

				double det = AAA_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					if ( frame_ptr != NULL && frame_ptr->type_ == frame_ptr->InitializeOnly ) {
						Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
						cam_rot_global_curr = aff_rgbd.linear();
						cam_trans_global_curr = aff_rgbd.translation();
						break;
					} else {
						reset ();
						return (false);
					}
				}

				Eigen::Matrix< float, 6, 1 > result = AAA_.llt().solve( bbb_ ).cast< float >();

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				Eigen::Affine3f mat_inc;
				mat_inc.linear() = cam_rot_incremental;
				mat_inc.translation() = cam_trans_incremental;

				result_trans = mat_inc * result_trans;
				Eigen::Matrix4d temp = result_trans.cast< double >();
				eigen2cv( temp, resultRt );

				Eigen::Affine3f mat_curr;
				mat_curr.matrix() = mat_prev.matrix() * result_trans;

				cam_rot_global_curr = mat_curr.linear();
				cam_trans_global_curr = mat_curr.translation();
			}
		}

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");

	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreIntegrationFlag ) ) {
		} else {
			//ScopeTime time( ">>> integrate" );
			integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		}
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	++global_time_;
	return (true);
}//rgbdodometry

bool
	pcl::gpu::KinfuTracker::intersect( int bounds[ 6 ] )
{
	pcl::gpu::tsdf_buffer* buffer = getCyclicalBufferStructure();
	// need more work here.
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
	pcl::gpu::KinfuTracker::getCameraPose (int time) const
{
	if (time > (int)rmats_.size () || time < 0)
		time = rmats_.size () - 1;

	Eigen::Affine3f aff;
	aff.linear () = rmats_[time];
	aff.translation () = tvecs_[time];
	return (aff);
}

Eigen::Matrix4f
	pcl::gpu::KinfuTracker::getInitTrans() const
{
	return init_trans_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
	pcl::gpu::KinfuTracker::getNumberOfPoses () const
{
	return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume& 
	pcl::gpu::KinfuTracker::volume() const 
{ 
	return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume& 
	pcl::gpu::KinfuTracker::volume()
{
	return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const ColorVolume& 
	pcl::gpu::KinfuTracker::colorVolume() const
{
	return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume& 
	pcl::gpu::KinfuTracker::colorVolume()
{
	return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::getImage (View& view) const
{
	//Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
	Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

	device::LightSource light;
	light.number = 1;
	light.pos[0] = device_cast<const float3>(light_source_pose);

	view.create (rows_, cols_);
	generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
	cloud.create (rows_, cols_);
	DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
	device::convert (vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
	pcl::gpu::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
	normals.create (rows_, cols_);
	DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
	device::convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
	pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{     
	color_volume_ = pcl::gpu::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );  
	cyclical_.initBuffer(tsdf_volume_, color_volume_);
}

void
	pcl::gpu::KinfuTracker::initSLAC( int slac_num )
{
	slac_num_ = slac_num;
	slac_resolution_ = 12;
	slac_lean_matrix_size_ = ( slac_resolution_ + 1 ) * ( slac_resolution_ + 1 ) * ( slac_resolution_ + 1 ) * 3;
	//slac_lean_matrix_size_gpu_2d_ = ( slac_lean_matrix_size_ * slac_lean_matrix_size_ - slac_lean_matrix_size_ ) / 2 + slac_lean_matrix_size_;
	slac_lean_matrix_size_gpu_2d_ = slac_lean_matrix_size_ * slac_lean_matrix_size_;
	gbuf_slac_triangle_.create( slac_lean_matrix_size_gpu_2d_ );
	gbuf_slac_block_.create( 7 * slac_lean_matrix_size_ );
	ctr_buf_.create( slac_lean_matrix_size_ );

	slac_full_matrix_size_ = slac_num_ * 6 + slac_lean_matrix_size_;

	initSLACMatrices();

	ctr_buf_.upload( slac_this_ctr_.data(), slac_lean_matrix_size_ );
}

void pcl::gpu::KinfuTracker::initOnlineSLAC( int num )
{
	use_slac_ = true;
	slac_num_ = num;
	slac_resolution_ = 12;

	tsdf_volume2_ = TsdfVolume::Ptr ( new TsdfVolume( Eigen::Vector3i( VOLUME_X, VOLUME_Y, VOLUME_Z ) ) );
	tsdf_volume2_->setSize( Eigen::Vector3f( grid_.length_, grid_.length_, grid_.length_ ) );
	tsdf_volume2_->setTsdfTruncDist( tsdf_volume_->getTsdfTruncDist() );
	tsdf_volume2_->reset();
}

void pcl::gpu::KinfuTracker::initSLACMatrices()
{
	slac_trans_mats_.clear();
	slac_base_mat_ = Eigen::MatrixXf( slac_full_matrix_size_, slac_full_matrix_size_ );
	slac_full_mat_ = Eigen::MatrixXf( slac_full_matrix_size_, slac_full_matrix_size_ );
	slac_this_ctr_ = Eigen::VectorXf( slac_lean_matrix_size_ );
	slac_full_b_ = Eigen::VectorXf( slac_full_matrix_size_ );
	slac_full_b_.setZero();

	for ( int i = 0; i <= slac_resolution_; i++ ) {
		for ( int j = 0; j <= slac_resolution_; j++ ) {
			for ( int k = 0; k <= slac_resolution_; k++ ) {
				Eigen::Vector4d pos( i * volume_size_ / slac_resolution_, j * volume_size_ / slac_resolution_, k * volume_size_ / slac_resolution_, 1 );
				int idx = getSLACIndex( i, j, k );
				slac_this_ctr_( idx + 0 ) = pos( 0 );
				slac_this_ctr_( idx + 1 ) = pos( 1 );
				slac_this_ctr_( idx + 2 ) = pos( 2 );
			}
		}
	}
	slac_init_ctr_ = slac_this_ctr_;

	for ( int i = 0; i <= slac_resolution_; i++ ) {
		for ( int j = 0; j <= slac_resolution_; j++ ) {
			for ( int k = 0; k <= slac_resolution_; k++ ) {
				int idx[ 2 ] = { getSLACIndex( i, j, k ), 0 };
				double val[ 2 ] = { 1, -1 };
				if ( i > 0 ) {
					idx[ 1 ] = getSLACIndex( i - 1, j, k );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 0 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 0 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 0 ] + 2 ) += -1;
				}
				if ( i < slac_resolution_ ) {
					idx[ 1 ] = getSLACIndex( i + 1, j, k );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 0 ], idx[ 1 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 1 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 1 ] + 2 ) += -1;
				}
				if ( j > 0 ) {
					idx[ 1 ] = getSLACIndex( i, j - 1, k );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 0 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 0 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 0 ] + 2 ) += -1;
				}
				if ( j < slac_resolution_ ) {
					idx[ 1 ] = getSLACIndex( i, j + 1, k );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 0 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 0 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 0 ] + 2 ) += -1;
				}
				if ( k > 0 ) {
					idx[ 1 ] = getSLACIndex( i, j, k - 1 );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 0 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 0 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 0 ] + 2 ) += -1;
				}
				if ( k < slac_resolution_ ) {
					idx[ 1 ] = getSLACIndex( i, j, k + 1 );
					slac_base_mat_( idx[ 0 ], idx[ 0 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 1 ] ) += 1;
					slac_base_mat_( idx[ 1 ], idx[ 0 ] ) += -1;
					slac_base_mat_( idx[ 0 ] + 1, idx[ 0 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 1 ] + 1 ) += 1;
					slac_base_mat_( idx[ 1 ] + 1, idx[ 0 ] + 1 ) += -1;
					slac_base_mat_( idx[ 0 ] + 2, idx[ 0 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 1 ] + 2 ) += 1;
					slac_base_mat_( idx[ 1 ] + 2, idx[ 0 ] + 2 ) += -1;
				}
			}
		}
	}

	int base_anchor = getSLACIndex( slac_resolution_ / 2, slac_resolution_ / 2, 0 );
	slac_base_mat_( base_anchor + 0, base_anchor + 0 ) += 1;
	slac_base_mat_( base_anchor + 1, base_anchor + 1 ) += 1;
	slac_base_mat_( base_anchor + 2, base_anchor + 2 ) += 1;
}

void pcl::gpu::KinfuTracker::addRegularizationTerm()
{
	double regscore = 0.0;
	for ( int i = 0; i <= slac_resolution_; i++ ) {
		for ( int j = 0; j <= slac_resolution_; j++ ) {
			for ( int k = 0; k <= slac_resolution_; k++ ) {
				int idx = getSLACIndex( i, j, k );
				std::vector< int > idxx;
				if ( i > 0 ) {
					idxx.push_back( getSLACIndex( i - 1, j, k ) );
				}
				if ( i < slac_resolution_) {
					idxx.push_back( getSLACIndex( i + 1, j, k ) );
				}
				if ( j > 0 ) {
					idxx.push_back( getSLACIndex( i, j - 1, k ) );
				}
				if ( j < slac_resolution_) {
					idxx.push_back( getSLACIndex( i, j + 1, k ) );
				}
				if ( k > 0 ) {
					idxx.push_back( getSLACIndex( i, j, k - 1 ) );
				}
				if ( k < slac_resolution_) {
					idxx.push_back( getSLACIndex( i, j, k + 1 ) );
				}

				Eigen::Matrix3f R;
				if ( i == slac_resolution_ / 2 && j == slac_resolution_ / 2 && k == 0 ) {
					// anchor point, Rotation matrix is always identity
					R = R.Identity();
				} else {
					R = GetRotation( idx, idxx, slac_init_ctr_, slac_this_ctr_ );
				}

				for ( int t = 0; t < ( int )idxx.size(); t++ ) {
					Eigen::Vector3f bx = Eigen::Vector3f( 
						slac_this_ctr_( idx ) - slac_this_ctr_( idxx[ t ] ), 
						slac_this_ctr_( idx + 1 ) - slac_this_ctr_( idxx[ t ] + 1 ), 
						slac_this_ctr_( idx + 2 ) - slac_this_ctr_( idxx[ t ] + 2 )
						)
						- R * Eigen::Vector3f( 
						slac_init_ctr_( idx ) - slac_init_ctr_( idxx[ t ] ), 
						slac_init_ctr_( idx + 1 ) - slac_init_ctr_( idxx[ t ] + 1 ),
						slac_init_ctr_( idx + 2 ) - slac_init_ctr_( idxx[ t ] + 2 )
						);
					regscore += slac_num_ * bx.transpose() * bx;
					slac_full_b_( idx ) += bx( 0 ) * slac_num_;
					slac_full_b_( idxx[ t ] ) -= bx( 0 ) * slac_num_;
					slac_full_b_( idx + 1 ) += bx( 1 ) * slac_num_;
					slac_full_b_( idxx[ t ] + 1 ) -= bx( 1 ) * slac_num_;
					slac_full_b_( idx + 2 ) += bx( 2 ) * slac_num_;
					slac_full_b_( idxx[ t ] + 2 ) -= bx( 2 ) * slac_num_;
				}
			}
		}
	}

}

Eigen::Matrix3f pcl::gpu::KinfuTracker::GetRotation( const int idx, const std::vector< int > & idxx, const Eigen::VectorXf & ictr, const Eigen::VectorXf & ctr )
{
	int n = ( int )idxx.size();
	Eigen::Matrix3f C = Eigen::Matrix3f::Zero();

	for ( int i = 0; i < n; i++ ) {
		Eigen::RowVector3f dif( ictr( idx ) - ictr( idxx[ i ] ), ictr( idx + 1 ) - ictr( idxx[ i ] + 1 ), ictr( idx + 2 ) - ictr( idxx[ i ] + 2 ) );
		Eigen::RowVector3f diff( ctr( idx ) - ctr( idxx[ i ] ), ctr( idx + 1 ) - ctr( idxx[ i ] + 1 ), ctr( idx + 2 ) - ctr( idxx[ i ] + 2 ) );
		C += dif.transpose() * diff;
	}

	Eigen::JacobiSVD< Eigen::Matrix3f > svd( C, Eigen::ComputeFullU | Eigen::ComputeFullV );
	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();
	Eigen::Matrix3f R = V * U.transpose();
	if ( R.determinant() < 0 ) {
		U( 0, 2 ) *= -1;
		U( 1, 2 ) *= -1;
		U( 2, 2 ) *= -1;
		R = V * U.transpose();
	}
	return R;
}

Eigen::Matrix3d pcl::gpu::KinfuTracker::GetRotationd( const int idx, const std::vector< int > & idxx, const Eigen::VectorXd & ictr, const Eigen::VectorXd & ctr )
{
	int n = ( int )idxx.size();
	Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

	for ( int i = 0; i < n; i++ ) {
		Eigen::RowVector3d dif( ictr( idx ) - ictr( idxx[ i ] ), ictr( idx + 1 ) - ictr( idxx[ i ] + 1 ), ictr( idx + 2 ) - ictr( idxx[ i ] + 2 ) );
		Eigen::RowVector3d diff( ctr( idx ) - ctr( idxx[ i ] ), ctr( idx + 1 ) - ctr( idxx[ i ] + 1 ), ctr( idx + 2 ) - ctr( idxx[ i ] + 2 ) );
		C += dif.transpose() * diff;
	}

	Eigen::JacobiSVD< Eigen::Matrix3d > svd( C, Eigen::ComputeFullU | Eigen::ComputeFullV );
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d R = V * U.transpose();
	if ( R.determinant() < 0 ) {
		U( 0, 2 ) *= -1;
		U( 1, 2 ) *= -1;
		U( 2, 2 ) *= -1;
		R = V * U.transpose();
	}
	return R;
}

/*
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool 
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{ 
bool res = (*this)(depth);

if (res && color_volume_)
{
const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
device::Intr intr(fx_, fy_, cx_, cy_);

Matrix3frm cam_rot_global_curr = rmats_.back();
Vector3f cam_trans_global_curr = tvecs_.back();
Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 
float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
float3 device_cam_trans_local_curr;
device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  

device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
colors, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());

//   Matrix3frm R_inv = rmats_.back().inverse();
//   Vector3f   t     = tvecs_.back();
//   
//   Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
//   float3& device_tcurr = device_cast<float3> (t);
//   
//   device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
//	colors, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
}

return res;
}
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
	namespace gpu
	{
		PCL_EXPORTS void 
			paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
		{
			device::paint3DView(rgb24, view, colors_weight);
		}

		PCL_EXPORTS void
			mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
		{
			const size_t size = min(cloud.size(), normals.size());
			output.create(size);

			const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
			const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
			const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
			device::mergePointNormal(c, n, o);           
		}

		Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
		{
			Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
			Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

			double rx = R(2, 1) - R(1, 2);
			double ry = R(0, 2) - R(2, 0);
			double rz = R(1, 0) - R(0, 1);

			double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
			double c = (R.trace() - 1) * 0.5;
			c = c > 1. ? 1. : c < -1. ? -1. : c;

			double theta = acos(c);

			if( s < 1e-5 )
			{
				double t;

				if( c > 0 )
					rx = ry = rz = 0;
				else
				{
					t = (R(0, 0) + 1)*0.5;
					rx = sqrt( std::max(t, 0.0) );
					t = (R(1, 1) + 1)*0.5;
					ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
					t = (R(2, 2) + 1)*0.5;
					rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

					if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
						rz = -rz;
					theta /= sqrt(rx*rx + ry*ry + rz*rz);
					rx *= theta;
					ry *= theta;
					rz *= theta;
				}
			}
			else
			{
				double vth = 1/(2*s);
				vth *= theta;
				rx *= vth; ry *= vth; rz *= vth;
			}
			return Eigen::Vector3d(rx, ry, rz).cast<float>();
		}
	}
}
