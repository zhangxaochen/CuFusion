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

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  //~ #include <opencv2/gpu/gpu.hpp>
  //~ #include <pcl/gpu/utils/timers_opencv.hpp>
#endif

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
	: cyclical_( DISTANCE_THRESHOLD, pcl::device::VOLUME_SIZE, VOLUME_X), rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), perform_last_scan_ (false), finished_(false), force_shift_(false), max_integrate_distance_(0)
	, extract_world_( false ), use_slac_( false )
{
  //const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution (VOLUME_X, VOLUME_Y, VOLUME_Z);

  volume_size_ = volume_size(0);

  tsdf_volume_ = TsdfVolume::Ptr ( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize (volume_size);
  
  shifting_distance_ = shiftingDistance;

  // set cyclical buffer values
  cyclical_.setDistanceThreshold (shifting_distance_);
  cyclical_.setVolumeSize (volume_size_, volume_size_, volume_size_);

  setDepthIntrinsics (525.f, 525.f); // default values, can be overwritten
  
  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

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
    //depth_raw.copyTo(depths_curr[0]);
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
			  reset ();
			  return (false);
			}
        }
        //float maxc = A.maxCoeff();

        Eigen::Matrix<float, 6, 1> result = A_.llt ().solve (b_).cast<float>();
        //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

		/*
          PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
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
      integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
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
  use_slac_ = true;
  slac_num_ = slac_num;
  slac_resolution_ = 8;
  slac_lean_matrix_size_ = 6 + ( slac_resolution_ + 1 ) * ( slac_resolution_ + 1 ) * ( slac_resolution_ + 1 ) * 3;
  slac_lean_matrix_size_gpu_2d_ = ( slac_lean_matrix_size_ * slac_lean_matrix_size_ - slac_lean_matrix_size_ ) / 2 + slac_lean_matrix_size_ * 2;

  // TODO: clear the memory difficulty.
  // TODO: investigate the possible speed up of Kinfu
  // TODO: 10x30 (blocks) x 1024 (threads) possible?

  //gbuf_slac_.create( slac_lean_matrix_size_gpu_2d_, 20 * 60 );
  sumbuf_slac_.create( slac_lean_matrix_size_gpu_2d_ );
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
