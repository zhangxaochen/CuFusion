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

#include "device.hpp"

//#include "pcl/gpu/utils/device/vector_math.hpp"

namespace pcl
{
  namespace device
  {
    __global__ void
    initColorVolumeKernel (PtrStep<uchar4> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < VOLUME_X && y < VOLUME_Y)
      {
        uchar4 *pos = volume.ptr (y) + x;
        int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
        for (int z = 0; z < VOLUME_Z; ++z, pos += z_step)
          *pos = make_uchar4 (0, 0, 0, 0);
      }
    }

    __global__ void
    clearColorSliceKernel (PtrStep<uchar4> color_volume, pcl::gpu::tsdf_buffer buffer, int3 minBounds, int3 maxBounds)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
           
      //compute relative indices
      int idX, idY;
      
      if(x < minBounds.x)
        idX = x + buffer.voxels_size.x;
      else
        idX = x;
      
      if(y < minBounds.y)
        idY = y + buffer.voxels_size.y;
      else
        idY = y;	 
              
      if ( x < buffer.voxels_size.x && y < buffer.voxels_size.y)
      {
          if( (idX >= minBounds.x && idX <= maxBounds.x) || (idY >= minBounds.y && idY <= maxBounds.y) )
          {
              // BLACK ZONE => clear on all Z values
         
              ///Pointer to the first x,y,0			
              uchar4 *pos = color_volume.ptr(y) + x;
              
              ///Get the step on Z
              int z_step = buffer.voxels_size.y * color_volume.step / sizeof(*pos);
                                  
              ///Get the size of the whole TSDF memory
              int size = buffer.color_memory_end - buffer.color_memory_start + 1;
                                
              ///Move along z axis
    #pragma unroll
              for(int z = 0; z < buffer.voxels_size.z; ++z, pos+=z_step)
              {
                ///If we went outside of the memory, make sure we go back to the begining of it
                if(pos > buffer.color_memory_end)
                  pos = pos - size;
                  
                *pos = make_uchar4 (0, 0, 0, 0);
              }
           }
           else /* if( idX > maxBounds.x && idY > maxBounds.y)*/
           {
             
              ///RED ZONE  => clear only appropriate Z
             
              ///Pointer to the first x,y,0
              uchar4 *pos = color_volume.ptr(y) + x;
              
              ///Get the step on Z
              int z_step = buffer.voxels_size.y * color_volume.step / sizeof(*pos);
                           
              ///Get the size of the whole TSDF memory 
              int size = buffer.color_memory_end - buffer.color_memory_start + 1;
                            
              ///Move pointer to the Z origin
              pos+= minBounds.z * z_step;
              
              ///If the Z offset is negative, we move the pointer back
              if(maxBounds.z < 0)
                pos += maxBounds.z * z_step;
                
              ///We make sure that we are not already before the start of the memory
              if(pos < buffer.color_memory_start)
                  pos = pos + size;

              int nbSteps = abs(maxBounds.z);
              
          #pragma unroll				
              for(int z = 0; z < nbSteps; ++z, pos+=z_step)
              {
                ///If we went outside of the memory, make sure we go back to the begining of it
                if(pos > buffer.color_memory_end)
                  pos = pos - size;
                  
                *pos = make_uchar4 (0, 0, 0, 0);
              }
           } //else /* if( idX > maxBounds.x && idY > maxBounds.y)*/
       } // if ( x < VOLUME_X && y < VOLUME_Y)
    } // clearColorSliceKernel
  }
}


void
pcl::device::initColorVolume (PtrStep<uchar4> color_volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

  initColorVolumeKernel<<<grid, block>>>(color_volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    struct ColorVolumeImpl
    {
      enum
      {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,

        ONE_VOXEL = 0
      };

      Intr intr;

      PtrStep<float> vmap;
      PtrStepSz<uchar3> colors;

      Mat33 R_inv;
      float3 t;

      float3 cell_size;
      float tranc_dist;

      int max_weight;

      mutable PtrStep<uchar4> color_volume;

      __device__ __forceinline__ int3
      getVoxel (float3 point) const
      {
        int vx = __float2int_rd (point.x / cell_size.x);                // round to negative infinity
        int vy = __float2int_rd (point.y / cell_size.y);
        int vz = __float2int_rd (point.z / cell_size.z);

        return make_int3 (vx, vy, vz);
      }

      __device__ __forceinline__ float3
      getVoxelGCoo (int x, int y, int z) const
      {
        float3 coo = make_float3 (x, y, z);
        coo += 0.5f;                 //shift to cell center;

        coo.x *= cell_size.x;
        coo.y *= cell_size.y;
        coo.z *= cell_size.z;

        return coo;
      }

      __device__ __forceinline__ void
      operator () ( pcl::gpu::tsdf_buffer buffer ) const
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

        for (int z = 0; z < VOLUME_X; ++z)
        {
          float3 v_g = getVoxelGCoo (x, y, z);

          float3 v = R_inv * (v_g - t);

          if (v.z <= 0)
            continue;

          int2 coo;                   //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (coo.x >= 0 && coo.y >= 0 && coo.x < colors.cols && coo.y < colors.rows)
          {
            float3 p;
            p.x = vmap.ptr (coo.y)[coo.x];

            if (isnan (p.x))
              continue;

            p.y = vmap.ptr (coo.y + colors.rows    )[coo.x];
            p.z = vmap.ptr (coo.y + colors.rows * 2)[coo.x];

            bool update = false;
            if (ONE_VOXEL)
            {
              int3 vp = getVoxel (p);
              update = vp.x == x && vp.y == y && vp.z == z;
            }
            else
            {
              float dist = norm (p - v_g);
              update = dist < tranc_dist;
            }

            if (update)
            {
              uchar4 *ptr = color_volume.ptr (VOLUME_Y * z + y) + x;

			  shift_color_pointer( &ptr, buffer );

              uchar3 rgb = colors.ptr (coo.y)[coo.x];
              uchar4 volume_rgbw = *ptr;

              int weight_prev = volume_rgbw.w;

              const float Wrk = 1.f;
              float new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
              float new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
              float new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

              int weight_new = weight_prev + 1;

              uchar4 volume_rgbw_new;
              volume_rgbw_new.x = min (255, max (0, __float2int_rn (new_x)));
              volume_rgbw_new.y = min (255, max (0, __float2int_rn (new_y)));
              volume_rgbw_new.z = min (255, max (0, __float2int_rn (new_z)));
              volume_rgbw_new.w = min (max_weight, weight_new);

              *ptr = volume_rgbw_new;
            }
          }           /* in camera image range */
        }         /* for(int z = 0; z < VOLUME_X; ++z) */
      }       /* void operator() */
    };

    __global__ void
    updateColorVolumeKernel (const ColorVolumeImpl cvi, pcl::gpu::tsdf_buffer buffer) {
      cvi (buffer);
    }
  }
}

void
pcl::device::updateColorVolume (const Intr& intr, float tranc_dist, const Mat33& R_inv, const float3& t,
                                const MapArr& vmap, const PtrStepSz<uchar3>& colors, const float3& volume_size, PtrStep<uchar4> color_volume, pcl::gpu::tsdf_buffer* buffer, int max_weight)
{
  ColorVolumeImpl cvi;
  cvi.vmap = vmap;
  cvi.colors = colors;
  cvi.color_volume = color_volume;

  cvi.R_inv = R_inv;
  cvi.t = t;
  cvi.intr = intr;
  cvi.tranc_dist = tranc_dist;
  cvi.max_weight = min (max (0, max_weight), 255);

  cvi.cell_size.x = volume_size.x / VOLUME_X;
  cvi.cell_size.y = volume_size.y / VOLUME_Y;
  cvi.cell_size.z = volume_size.z / VOLUME_Z;

  dim3 block (ColorVolumeImpl::CTA_SIZE_X, ColorVolumeImpl::CTA_SIZE_Y);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

  updateColorVolumeKernel<<<grid, block>>>(cvi, *buffer);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    __global__ void
    extractColorsKernel (const float3 cell_size, const PtrStep<uchar4> color_volume, pcl::gpu::tsdf_buffer buffer, const PtrSz<PointType> points, uchar4 *colors)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx < points.size)
      {
        int3 v;
        float3 p = *(const float3*)(points.data + idx);
        v.x = __float2int_rd (p.x / cell_size.x);        // round to negative infinity
        v.y = __float2int_rd (p.y / cell_size.y);
        v.z = __float2int_rd (p.z / cell_size.z);

        const uchar4* tmp_pos = &(color_volume.ptr (buffer.voxels_size.y * v.z + v.y)[v.x]);
        uchar4* pos = const_cast<uchar4*> (tmp_pos);
        shift_color_pointer (&pos, buffer);
		colors[idx] = make_uchar4 ( pos->z, pos->y, pos->x, 0 ); //bgra

        //uchar4 rgbw = color_volume.ptr (VOLUME_Y * v.z + v.y)[v.x];
        //colors[idx] = make_uchar4 (rgbw.z, rgbw.y, rgbw.x, 0); //bgra
      }
    }
  }
}

void
pcl::device::exctractColors (const PtrStep<uchar4>& color_volume, const pcl::gpu::tsdf_buffer* buffer, const float3& volume_size, const PtrSz<PointType>& points, uchar4* colors)
{
  const int block = 256;
  float3 cell_size = make_float3 (volume_size.x / VOLUME_X, volume_size.y / VOLUME_Y, volume_size.z / VOLUME_Z);
  extractColorsKernel<<<divUp (points.size, block), block>>>(cell_size, color_volume, *buffer, points, colors);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::clearColorSlice (PtrStep<uchar4> color_volume, pcl::gpu::tsdf_buffer* buffer, int shiftX, int shiftY, int shiftZ)
{
    int newX = buffer->origin_GRID.x + shiftX;
    int newY = buffer->origin_GRID.y + shiftY;

    int3 minBounds, maxBounds;
    
    //X
    if(newX >= 0)
    {
     minBounds.x = buffer->origin_GRID.x;
     maxBounds.x = newX;    
    }
    else
    {
     minBounds.x = newX + buffer->voxels_size.x; 
     maxBounds.x = buffer->origin_GRID.x + buffer->voxels_size.x;
    }
    
    if(minBounds.x > maxBounds.x)
     std::swap(minBounds.x, maxBounds.x);
      
   
    //Y
    if(newY >= 0)
    {
     minBounds.y = buffer->origin_GRID.y;
     maxBounds.y = newY;
    }
    else
    {
     minBounds.y = newY + buffer->voxels_size.y; 
     maxBounds.y = buffer->origin_GRID.y + buffer->voxels_size.y;
    }
    
    if(minBounds.y > maxBounds.y)
     std::swap(minBounds.y, maxBounds.y);
    
    //Z
     minBounds.z = buffer->origin_GRID.z;
     maxBounds.z = shiftZ;
  
    // call kernel
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (buffer->voxels_size.x, block.x);      
    grid.y = divUp (buffer->voxels_size.y, block.y);
    
    clearColorSliceKernel<<<grid, block>>>(color_volume, *buffer, minBounds, maxBounds);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
   
}
