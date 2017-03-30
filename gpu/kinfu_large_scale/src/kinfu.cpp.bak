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

using namespace cv;

#include <limits>

//////////////////////////////
// RGBDOdometry part
//////////////////////////////

inline static
	void computeC_RigidBodyMotion( double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy )
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
	void computeProjectiveMatrix( const Mat& ksi, Mat& Rt )
{
	CV_Assert( ksi.size() == Size(1,6) && ksi.type() == CV_64FC1 );

	// for infinitesimal transformation
	Rt = Mat::eye(4, 4, CV_64FC1);

	Mat R = Rt(Rect(0,0,3,3));
	Mat rvec = ksi.rowRange(0,3);

	Rodrigues( rvec, R );

	Rt.at<double>(0,3) = ksi.at<double>(3);
	Rt.at<double>(1,3) = ksi.at<double>(4);
	Rt.at<double>(2,3) = ksi.at<double>(5);
}

static
	void cvtDepth2Cloud( const Mat& depth, Mat& cloud, const Mat& cameraMatrix )
{
	//CV_Assert( cameraMatrix.type() == CV_64FC1 );
	const double inv_fx = 1.f/cameraMatrix.at<double>(0,0);
	const double inv_fy = 1.f/cameraMatrix.at<double>(1,1);
	const double ox = cameraMatrix.at<double>(0,2);
	const double oy = cameraMatrix.at<double>(1,2);
	cloud.create( depth.size(), CV_32FC3 );
	for( int y = 0; y < cloud.rows; y++ )
	{
		Point3f* cloud_ptr = reinterpret_cast<Point3f*>(cloud.ptr(y));
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
	int computeCorresp( const Mat& K, const Mat& K_inv, const Mat& Rt,
	const Mat& depth0, const Mat& depth1, const Mat& texturedMask1, float maxDepthDiff,
	Mat& corresps )
{
	CV_Assert( K.type() == CV_64FC1 );
	CV_Assert( K_inv.type() == CV_64FC1 );
	CV_Assert( Rt.type() == CV_64FC1 );

	corresps.create( depth1.size(), CV_32SC1 );

	Mat R = Rt(Rect(0,0,3,3)).clone();

	Mat KRK_inv = K * R * K_inv;
	const double * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());

	Mat Kt = Rt(Rect(3,0,1,3)).clone();
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

	Rect r(0, 0, depth1.cols, depth1.rows);

	corresps = Scalar(-1);
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

				if( r.contains(Point(u0,v0)) )
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
	void preprocessDepth( Mat depth0, Mat depth1,
	const Mat& validMask0, const Mat& validMask1,
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
	void buildPyramids( const Mat& image0, const Mat& image1,
	const Mat& depth0, const Mat& depth1,
	const Mat& cameraMatrix, int sobelSize, double sobelScale,
	const vector<float>& minGradMagnitudes,
	vector<Mat>& pyramidImage0, vector<Mat>& pyramidDepth0,
	vector<Mat>& pyramidImage1, vector<Mat>& pyramidDepth1,
	vector<Mat>& pyramid_dI_dx1, vector<Mat>& pyramid_dI_dy1,
	vector<Mat>& pyramidTexturedMask1, vector<Mat>& pyramidCameraMatrix )
{
	const int pyramidMaxLevel = (int)minGradMagnitudes.size() - 1;

	buildPyramid( image0, pyramidImage0, pyramidMaxLevel );
	buildPyramid( image1, pyramidImage1, pyramidMaxLevel );

	pyramid_dI_dx1.resize( pyramidImage1.size() );
	pyramid_dI_dy1.resize( pyramidImage1.size() );
	pyramidTexturedMask1.resize( pyramidImage1.size() );

	pyramidCameraMatrix.reserve( pyramidImage1.size() );

	Mat cameraMatrix_dbl;
	cameraMatrix.convertTo( cameraMatrix_dbl, CV_64FC1 );

	for( size_t i = 0; i < pyramidImage1.size(); i++ )
	{
		Sobel( pyramidImage1[i], pyramid_dI_dx1[i], CV_16S, 1, 0, sobelSize );
		Sobel( pyramidImage1[i], pyramid_dI_dy1[i], CV_16S, 0, 1, sobelSize );

		const Mat& dx = pyramid_dI_dx1[i];
		const Mat& dy = pyramid_dI_dy1[i];

		Mat texturedMask( dx.size(), CV_8UC1, Scalar(0) );
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
		Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
		levelCameraMatrix.at<double>(2,2) = 1.;
		pyramidCameraMatrix.push_back( levelCameraMatrix );
	}

	buildPyramid( depth0, pyramidDepth0, pyramidMaxLevel );
	buildPyramid( depth1, pyramidDepth1, pyramidMaxLevel );
}

static
	bool solveSystem( const Mat& C, const Mat& dI_dt, double detThreshold, Mat& ksi, Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	Mat A = C.t() * C;
	Mat B = -C.t() * dI_dt;

	cv2eigen( A, AA );
	cv2eigen( B, bb );

	double det = cv::determinant(A);

	if( fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det) )
		return false;

	cv::solve( A, B, ksi, DECOMP_CHOLESKY );
	return true;
}

typedef void (*ComputeCFuncPtr)( double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy );

static
	bool computeKsi( int transformType,
	const Mat& image0, const Mat&  cloud0,
	const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
	const Mat& corresps, int correspsCount,
	double fx, double fy, double sobelScale, double determinantThreshold,
	Mat& ksi,
	Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	int Cwidth = -1;
	ComputeCFuncPtr computeCFuncPtr = 0;
	computeCFuncPtr = computeC_RigidBodyMotion;
	Cwidth = 6;
	Mat C( correspsCount, Cwidth, CV_64FC1 );
	Mat dI_dt( correspsCount, 1, CV_64FC1 );

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
					cloud0.at<Point3f>(v0,u0), fx, fy);

				dI_dt.at<double>(pointCount) = w * diff;
				pointCount++;
			}
		}
	}

	Mat sln;
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
		ksi = Scalar(0);

		Mat subksi;
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

	const int iters[] = {10, 5, 4};
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

}

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
	init_Rcam_ = pose.rotation ();
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
	// see estimate tranform for the magic numbers
	//gbuf_.create (27, 20*60);
	gbuf_.create( 27, 640*480 );
	sumbuf_.create (27);
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
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
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
}

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
}

cv::Mat pcl::gpu::KinfuTracker::bdrodometry_getOcclusionBoundary( cv::Mat depth, float dist_threshold )
{
	cv::Mat mask( 480, 640, CV_8UC1 );
	mask.setTo( 0 );

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::KinfuTracker::bdrodometry(const DepthMap& depth_raw, const View * pcolor)
{
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
		ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		device::bilateralFilter (depth_raw, depths_curr_[0]);

		int c;
		vector<unsigned short> data;
		depths_curr_[ 0 ].download(data, c);
		char filename[ 1024 ];
		std::sprintf( filename, "image/bf/%06d.png", global_time_ + 1 );
		cv::Mat m( 480, 640, CV_16UC1, (void *)&data[0] );
		//cv::imwrite( filename, m );
		
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
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
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

				float rr = 0.0f;
				int ll = 0;
				AA_.setZero();
				bb_.setZero();
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
					if ( pointNKNSquaredDistance[ 0 ] < 0.1 * 0.1 ) {
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
				//cout << vmappcd->size() << " : " << transformed->size() << " --> " << ll << endl;
				//cout << rr << endl;

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
}

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
}

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
}

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
}


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

	Mat depth0 = _depth0.clone(),
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
	CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == Size(3,3) );

	// other checks
	CV_Assert( iterCounts.empty() || minGradientMagnitudes.empty() ||
		minGradientMagnitudes.size() == iterCounts.size() );

	vector<int> defaultIterCounts;
	vector<float> defaultMinGradMagnitudes;
	vector<int> const* iterCountsPtr = &iterCounts;
	vector<float> const* minGradientMagnitudesPtr = &minGradientMagnitudes;

	preprocessDepth( depth0, depth1, validMask0, validMask1, minDepth, maxDepth );

	vector<Mat> pyramidImage0, pyramidDepth0,
		pyramidImage1, pyramidDepth1, pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1,
		pyramidCameraMatrix;
	buildPyramids( image0, image1, depth0, depth1, cameraMatrix, sobelSize, sobelScale, *minGradientMagnitudesPtr,
		pyramidImage0, pyramidDepth0, pyramidImage1, pyramidDepth1,
		pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1, pyramidCameraMatrix );

	Mat resultRt = Mat::eye(4,4,CV_64FC1);
	Mat currRt, ksi;

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

			const Mat& levelCameraMatrix = pyramidCameraMatrix[ level_index ];

			const Mat& levelImage0 = pyramidImage0[ level_index ];
			const Mat& levelDepth0 = pyramidDepth0[ level_index ];
			Mat levelCloud0;
			cvtDepth2Cloud( pyramidDepth0[ level_index ], levelCloud0, levelCameraMatrix );

			const Mat& levelImage1 = pyramidImage1[ level_index ];
			const Mat& levelDepth1 = pyramidDepth1[ level_index ];
			const Mat& level_dI_dx1 = pyramid_dI_dx1[ level_index ];
			const Mat& level_dI_dy1 = pyramid_dI_dy1[ level_index ];

			CV_Assert( level_dI_dx1.type() == CV_16S );
			CV_Assert( level_dI_dy1.type() == CV_16S );

			const double fx = levelCameraMatrix.at<double>(0,0);
			const double fy = levelCameraMatrix.at<double>(1,1);
			const double determinantThreshold = 1e-6;

			Mat corresps( levelImage0.size(), levelImage0.type() );

			for (int iter = 0; iter < iter_num; ++iter)
			{
				// rgbd odometry part
				bool odo_good = true;

				int correspsCount = computeCorresp( levelCameraMatrix, levelCameraMatrix.inv(), resultRt.inv(DECOMP_SVD),
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
}


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

	Mat depth0 = _depth0.clone(),
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
	CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == Size(3,3) );

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

	vector<Mat> pyramidImage0, pyramidDepth0,
		pyramidImage1, pyramidDepth1, pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1,
		pyramidCameraMatrix;
	buildPyramids( image0, image1, depth0, depth1, cameraMatrix, sobelSize, sobelScale, *minGradientMagnitudesPtr,
		pyramidImage0, pyramidDepth0, pyramidImage1, pyramidDepth1,
		pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1, pyramidCameraMatrix );

	Mat resultRt = Mat::eye(4,4,CV_64FC1);
	Mat currRt, ksi;

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

			const Mat& levelCameraMatrix = pyramidCameraMatrix[ level_index ];

			const Mat& levelImage0 = pyramidImage0[ level_index ];
			const Mat& levelDepth0 = pyramidDepth0[ level_index ];
			Mat levelCloud0;
			cvtDepth2Cloud( pyramidDepth0[ level_index ], levelCloud0, levelCameraMatrix );

			const Mat& levelImage1 = pyramidImage1[ level_index ];
			const Mat& levelDepth1 = pyramidDepth1[ level_index ];
			const Mat& level_dI_dx1 = pyramid_dI_dx1[ level_index ];
			const Mat& level_dI_dy1 = pyramid_dI_dy1[ level_index ];

			CV_Assert( level_dI_dx1.type() == CV_16S );
			CV_Assert( level_dI_dy1.type() == CV_16S );

			const double fx = levelCameraMatrix.at<double>(0,0);
			const double fy = levelCameraMatrix.at<double>(1,1);
			const double determinantThreshold = 1e-6;

			Mat corresps( levelImage0.size(), levelImage0.type() );

			for( int iter = 0; iter < (*iterCountsPtr)[ level_index ]; iter ++ ) {
				bool odo_good = true;

				int correspsCount = computeCorresp( levelCameraMatrix, levelCameraMatrix.inv(), resultRt.inv(DECOMP_SVD),
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
}

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
