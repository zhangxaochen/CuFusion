/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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

#ifndef PCL_CYCLICAL_BUFFER_IMPL_HPP_
#define PCL_CYCLICAL_BUFFER_IMPL_HPP_

#include <pcl/gpu/kinfu_large_scale/cyclical_buffer.h>


bool 
pcl::gpu::CyclicalBuffer::checkForShift (const pcl::gpu::TsdfVolume::Ptr volume, const pcl::gpu::ColorVolume::Ptr color, const Eigen::Affine3f &cam_pose, const double distance_camera_target, const bool perform_shift, const bool last_shift, const bool force_shift, const bool extract_world)
{
  bool result = false;

  // project the target point in the cube
  pcl::PointXYZ targetPoint;
  targetPoint.x = 0.0f;
  targetPoint.y = 0.0f;
  targetPoint.z = distance_camera_target; // place the point at camera position + distance_camera_target on Z
  targetPoint = pcl::transformPoint (targetPoint, cam_pose);
  
  // check distance from the cube's center  
  pcl::PointXYZ center_cube;
  center_cube.x = buffer_.origin_metric.x + buffer_.volume_size.x/2.0f;
  center_cube.y = buffer_.origin_metric.y + buffer_.volume_size.y/2.0f;
  center_cube.z = buffer_.origin_metric.z + buffer_.volume_size.z/2.0f;
    
  if (force_shift || pcl::euclideanDistance (targetPoint, center_cube) > distance_threshold_)
    result = true;
  
  if (!perform_shift)
    return (result);

  // perform shifting operations
  if (result)
    performShift (volume, color, targetPoint, last_shift, extract_world);

  return (result);
}


void
pcl::gpu::CyclicalBuffer::performShift (const pcl::gpu::TsdfVolume::Ptr volume, const pcl::gpu::ColorVolume::Ptr color, const pcl::PointXYZ &target_point, const bool last_shift, const bool extract_world)
{
  // compute new origin and offsets
  int offset_x, offset_y, offset_z;
  computeAndSetNewCubeMetricOrigin (target_point, offset_x, offset_y, offset_z);
    
  PointCloud<PointXYZI>::Ptr current_slice (new PointCloud<PointXYZI>);

  if ( extract_world ) {
	  // extract current slice from the TSDF volume (coordinates are in indices! (see fetchSliceAsCloud() )
	  DeviceArray<PointXYZ> points;
	  DeviceArray<float> intensities;
	  int size;   
	  if(!last_shift)
	  {
		size = volume->fetchSliceAsCloud (cloud_buffer_device_xyz_, cloud_buffer_device_intensities_, &buffer_, offset_x, offset_y, offset_z); 
	  }
	  else
	  {
		size = volume->fetchSliceAsCloud (cloud_buffer_device_xyz_, cloud_buffer_device_intensities_, &buffer_, buffer_.voxels_size.x - 1, buffer_.voxels_size.y - 1, buffer_.voxels_size.z - 1);
	  }
	  points = DeviceArray<PointXYZ> (cloud_buffer_device_xyz_.ptr (), size);
	  intensities = DeviceArray<float> (cloud_buffer_device_intensities_.ptr(), size);

	  PointCloud<PointXYZ>::Ptr current_slice_xyz (new PointCloud<PointXYZ>);
	  PointCloud<PointIntensity>::Ptr current_slice_intensities (new PointCloud<PointIntensity>);

	  // Retrieving XYZ 
	  points.download (current_slice_xyz->points);
	  current_slice_xyz->width = (int) current_slice_xyz->points.size ();
	  current_slice_xyz->height = 1;

	  // Retrieving intensities
	  // TODO change this mechanism by using PointIntensity directly (in spite of float)
	  // when tried, this lead to wrong intenisty values being extracted by fetchSliceAsCloud () (padding pbls?)
	  std::vector<float , Eigen::aligned_allocator<float> > intensities_vector;
	  intensities.download (intensities_vector);
	  current_slice_intensities->points.resize (current_slice_xyz->points.size ());
	  for(int i = 0 ; i < current_slice_intensities->points.size () ; ++i)
		current_slice_intensities->points[i].intensity = intensities_vector[i];

	  current_slice_intensities->width = (int) current_slice_intensities->points.size ();
	  current_slice_intensities->height = 1;

	  // Concatenating XYZ and Intensities
	  pcl::concatenateFields (*current_slice_xyz, *current_slice_intensities, *current_slice);
	  current_slice->width = (int) current_slice->points.size ();
	  current_slice->height = 1;

	  // transform the slice from local to global coordinates
	  Eigen::Affine3f global_cloud_transformation; 
	  global_cloud_transformation.translation ()[0] = buffer_.origin_GRID_global.x;
	  global_cloud_transformation.translation ()[1] = buffer_.origin_GRID_global.y;
	  global_cloud_transformation.translation ()[2] = buffer_.origin_GRID_global.z;
	  global_cloud_transformation.linear () = Eigen::Matrix3f::Identity ();
	  transformPointCloud (*current_slice, *current_slice, global_cloud_transformation);
  }

  // retrieve existing data from the world model
  PointCloud<PointXYZI>::Ptr previously_existing_slice (new  PointCloud<PointXYZI>);
  double min_bound_x  = buffer_.origin_GRID_global.x + buffer_.voxels_size.x - 1;
  double new_origin_x = buffer_.origin_GRID_global.x + offset_x;
  double new_origin_y = buffer_.origin_GRID_global.y + offset_y;
  double new_origin_z = buffer_.origin_GRID_global.z + offset_z;

  if ( extract_world ) {
	  world_model_.getExistingData (buffer_.origin_GRID_global.x, buffer_.origin_GRID_global.y, buffer_.origin_GRID_global.z,
									offset_x, offset_y, offset_z,
									buffer_.voxels_size.x - 1, buffer_.voxels_size.y - 1, buffer_.voxels_size.z - 1,
									*previously_existing_slice);
  
	  //replace world model data with values extracted from the TSDF buffer slice
	  world_model_.setSliceAsNans (buffer_.origin_GRID_global.x, buffer_.origin_GRID_global.y, buffer_.origin_GRID_global.z,
								   offset_x, offset_y, offset_z,
								   buffer_.voxels_size.x, buffer_.voxels_size.y, buffer_.voxels_size.z);

  	  cout << current_slice->points.size() << endl;

	  PCL_INFO ("world contains %d points after update\n", world_model_.getWorldSize ());
	  world_model_.cleanWorldFromNans ();                               
	  PCL_INFO ("world contains %d points after cleaning\n", world_model_.getWorldSize ());
  }

  // clear buffer slice and update the world model
  PCL_DEBUG( "in clearSlice : [%d, %d, %d] + [%d, %d, %d]\n", buffer_.origin_GRID.x, buffer_.origin_GRID.y, buffer_.origin_GRID.z, offset_x, offset_y, offset_z );

  {
	  int3 minBounds, maxBounds;
	  int shiftX = offset_x;
	  int shiftY = offset_y;
	  int shiftZ = offset_z;
	  int newX = buffer_.origin_GRID.x + shiftX;
      int newY = buffer_.origin_GRID.y + shiftY;

	if ( shiftX >= 0 ) {
		minBounds.x = buffer_.origin_GRID.x;
		maxBounds.x = newX - 1;
	} else {
		minBounds.x = newX;
		maxBounds.x = buffer_.origin_GRID.x - 1;
	}
	if ( minBounds.x < 0 ) {
		minBounds.x += buffer_.voxels_size.x;
		maxBounds.x += buffer_.voxels_size.x;
	}

	if ( shiftY >= 0 ) {
		minBounds.y = buffer_.origin_GRID.y;
		maxBounds.y = newY - 1;
	} else {
		minBounds.y = newY;
		maxBounds.y = buffer_.origin_GRID.y - 1;
	}
	if ( minBounds.y < 0 ) {
		minBounds.y += buffer_.voxels_size.y;
		maxBounds.y += buffer_.voxels_size.y;
	}
    //Z
     minBounds.z = buffer_.origin_GRID.z;
     maxBounds.z = shiftZ;
	  PCL_DEBUG( "In clearTSDFSlice:\n" );
	  PCL_DEBUG("Origin : %d, %d %d\n", buffer_.origin_GRID.x, buffer_.origin_GRID.y, buffer_.origin_GRID.z );
	  PCL_DEBUG("Origin global : %f, %f %f\n", buffer_.origin_GRID_global.x, buffer_.origin_GRID_global.y, buffer_.origin_GRID_global.z );
	  PCL_DEBUG( "Offset : %d, %d, %d\n", shiftX, shiftY, shiftZ );
      PCL_DEBUG ("X bound: [%d - %d]\n", minBounds.x, maxBounds.x);
      PCL_DEBUG ("Y bound: [%d - %d]\n", minBounds.y, maxBounds.y);
      PCL_DEBUG ("Z bound: [%d - %d]\n", minBounds.z, maxBounds.z);
	  int size = buffer_.tsdf_memory_end - buffer_.tsdf_memory_start + 1;
	  PCL_DEBUG ( "Size is : %x\n", size );
  }
  pcl::device::clearTSDFSlice (volume->data (), &buffer_, offset_x, offset_y, offset_z);
  //zc:
  if(color){
      pcl::device::clearColorSlice (color->data (), &buffer_, offset_x, offset_y, offset_z);
      PCL_WARN("@CyclicalBuffer.performShift--clearColorSlice color volume VALID\n");
  }
  else{
      PCL_WARN("@CyclicalBuffer.performShift--clearColorSlice: color volume INVALID---------------\n");
  }

  if ( extract_world ) {
	  // insert current slice in the world if it contains any points
	  if (current_slice->points.size () != 0) {
		world_model_.addSlice(current_slice);
        PCL_INFO ("world contains %d points after add slice\n", world_model_.getWorldSize ());
	  }
  }

  //zc:
  if(color){
      // shift buffer addresses
      shiftOrigin (volume, color, offset_x, offset_y, offset_z);
      PCL_WARN("@CyclicalBuffer.performShift--shiftOrigin: color volume VALID\n");
  }
  else{
      PCL_WARN("@CyclicalBuffer.performShift--shiftOrigin: color volume INVALID---------------\n");
  }

  if ( extract_world ) {
	  // push existing data in the TSDF buffer
	  if (previously_existing_slice->points.size () != 0 ) {
		volume->pushSlice(previously_existing_slice, getBuffer () );
	  }
  }
}

void 
pcl::gpu::CyclicalBuffer::computeAndSetNewCubeMetricOrigin (const pcl::PointXYZ &target_point, int &shiftX, int &shiftY, int &shiftZ)
{
  // compute new origin for the cube, based on the target point
  float3 new_cube_origin_meters;
  new_cube_origin_meters.x = target_point.x - buffer_.volume_size.x/2.0f;
  new_cube_origin_meters.y = target_point.y - buffer_.volume_size.y/2.0f;
  new_cube_origin_meters.z = target_point.z - buffer_.volume_size.z/2.0f;
  PCL_INFO ("The old cube's metric origin was    (%f, %f, %f).\n", buffer_.origin_metric.x, buffer_.origin_metric.y, buffer_.origin_metric.z);
  PCL_INFO ("The new cube's metric origin is now (%f, %f, %f).\n", new_cube_origin_meters.x, new_cube_origin_meters.y, new_cube_origin_meters.z);

  // deduce each shift in indices
  shiftX = (int)( (new_cube_origin_meters.x - buffer_.origin_metric.x) * ( buffer_.voxels_size.x / (float) (buffer_.volume_size.x) ) );
  shiftY = (int)( (new_cube_origin_meters.y - buffer_.origin_metric.y) * ( buffer_.voxels_size.y / (float) (buffer_.volume_size.y) ) );
  shiftZ = (int)( (new_cube_origin_meters.z - buffer_.origin_metric.z) * ( buffer_.voxels_size.z / (float) (buffer_.volume_size.z) ) );

  new_cube_origin_meters.x = buffer_.origin_metric.x + shiftX / ( buffer_.voxels_size.x / (float) (buffer_.volume_size.x) );
  new_cube_origin_meters.y = buffer_.origin_metric.y + shiftY / ( buffer_.voxels_size.y / (float) (buffer_.volume_size.y) );
  new_cube_origin_meters.z = buffer_.origin_metric.z + shiftZ / ( buffer_.voxels_size.z / (float) (buffer_.volume_size.z) );
  PCL_INFO ("The new cube's metric origin is coerced to (%f, %f, %f).\n", new_cube_origin_meters.x, new_cube_origin_meters.y, new_cube_origin_meters.z);

  // update the cube's metric origin 
  buffer_.origin_metric = new_cube_origin_meters;
}

#endif // PCL_CYCLICAL_BUFFER_IMPL_HPP_
