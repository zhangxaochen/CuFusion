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
#include <fstream>

using namespace pcl::device;

/*__global__ */__device__
const float COS30 = 0.8660254f
    ,COS45 = 0.7071f
    ,COS60 = 0.5f
    ,COS75 = 0.258819f
    ,COS80 = 0.173649f
    ,COS120 = -0.5f
    ,COS150 = -0.8660254f
    ;

namespace pcl
{
  namespace device
  {
    template<typename T>
    __global__ void
    initializeVolume (PtrStep<T> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      
      
      if (x < VOLUME_X && y < VOLUME_Y)
      {
          T *pos = volume.ptr(y) + x;
          int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
          for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
             pack_tsdf (0.f, 0, *pos);
      }
    }
    
    //zc: 这个模板 T 其实主要用作 bool
    template<typename T>
    __global__ void
    initFlagVolumeKernel(PtrStep<T> volume){
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      
      
      if (x < VOLUME_X && y < VOLUME_Y)
      {
          T *pos = volume.ptr(y) + x;
          int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
          for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
             //pack_tsdf (0.f, 0, *pos);
             *pos = false; //仅改此处?
      }
    }//initFlagVolumeKernel

    //zc: 这个模板 T 主要用于 char3, char4
    template<typename T>
    __global__ void
    initVrayPrevVolumeKrnl (PtrStep<T> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      
      
      if (x < VOLUME_X && y < VOLUME_Y)
      {
          T *pos = volume.ptr(y) + x;
          int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
          for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step){
              (*pos).x = 0;
              (*pos).y = 0;
              (*pos).z = 0;
              (*pos).w = 0; //T 目前必然用于 char4 (因为 host 中按 int 存储), 所以放心用 w 域 //2017-2-15 16:53:43
                   //↑- 【本来 xyz 用作 tsdf-v8 策略; 【现在启用 w, 用作仿照 bool flagVolume; 此处约定: 0-false-瞎猜, 1-true-看到; 默认仍为 0,
          }
      }
    }//initVrayPrevVolumeKrnl


        template<typename T>
    __global__ void
    clearSliceKernel (PtrStep<T> volume, pcl::gpu::tsdf_buffer buffer, int3 minBounds, int3 maxBounds)
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
              T *pos = volume.ptr(y) + x;
              
              ///Get the step on Z
              int z_step = buffer.voxels_size.y * volume.step / sizeof(*pos);
                                  
              ///Get the size of the whole TSDF memory
              int size = buffer.tsdf_memory_end - buffer.tsdf_memory_start + 1;
                                
              ///Move along z axis
    #pragma unroll
              for(int z = 0; z < buffer.voxels_size.z; ++z, pos+=z_step)
              {
                ///If we went outside of the memory, make sure we go back to the begining of it
                if(pos > buffer.tsdf_memory_end)
                  pos = pos - size;
                  
                pack_tsdf (0.f, 0, *pos);
              }
           }
           else /* if( idX > maxBounds.x && idY > maxBounds.y)*/
           {
             
              ///RED ZONE  => clear only appropriate Z
             
              ///Pointer to the first x,y,0
              T *pos = volume.ptr(y) + x;
              
              ///Get the step on Z
              int z_step = buffer.voxels_size.y * volume.step / sizeof(*pos);
                           
              ///Get the size of the whole TSDF memory 
              int size = buffer.tsdf_memory_end - buffer.tsdf_memory_start + 1;
                            
              ///Move pointer to the Z origin
              pos+= minBounds.z * z_step;
              
              ///If the Z offset is negative, we move the pointer back
              if(maxBounds.z < 0)
                pos += maxBounds.z * z_step;
                
              ///We make sure that we are not already before the start of the memory
              if(pos < buffer.tsdf_memory_start)
                  pos = pos + size;

              int nbSteps = abs(maxBounds.z);
              
          #pragma unroll				
              for(int z = 0; z < nbSteps; ++z, pos+=z_step)
              {
                ///If we went outside of the memory, make sure we go back to the begining of it
                if(pos > buffer.tsdf_memory_end)
                  pos = pos - size;
                  
                pack_tsdf (0.f, 0, *pos);
              }
           } //else /* if( idX > maxBounds.x && idY > maxBounds.y)*/
       } // if ( x < VOLUME_X && y < VOLUME_Y)
    } // clearSliceKernel
       
  }
}

void
pcl::device::initVolume (PtrStep<short2> volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);      
  grid.y = divUp (VOLUME_Y, block.y);

  initializeVolume<<<grid, block>>>(volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

void
pcl::device::initFlagVolume(PtrStep<bool> volume){
    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);      
    grid.y = divUp (VOLUME_Y, block.y);

    //initializeVolume<<<grid, block>>>(volume);
    initFlagVolumeKernel<<<grid, block>>>(volume);

    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}//initFlagVolume

void
pcl::device::initVrayPrevVolume(PtrStep<char4> volume){
    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);      
    grid.y = divUp (VOLUME_Y, block.y);

    //initializeVolume<<<grid, block>>>(volume);
    //initFlagVolumeKernel<<<grid, block>>>(volume); //magCnt 仍用 initFlagVolumeKernel, 因为它是模板函数, 初始化 false 与 0 一致
    initVrayPrevVolumeKrnl<<<grid, block>>>(volume);

    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}//initVrayPrevVolume

namespace pcl
{
  namespace device
  {
    struct Tsdf
    {
      enum
      {
        CTA_SIZE_X = 32, CTA_SIZE_Y = 8,
        //MAX_WEIGHT = 1 << 7
        MAX_WEIGHT = 1 << 8
        //MAX_WEIGHT = 15
        //MAX_WEIGHT = 255
        //MAX_WEIGHT = 15

        ,MAX_WEIGHT_V13 = 1<<8
      };

      mutable PtrStep<short2> volume;
      float3 cell_size;

      Intr intr;

      Mat33 Rcurr_inv;
      float3 tcurr;

      PtrStepSz<ushort> depth_raw; //depth in mm

      float tranc_dist_mm;

      __device__ __forceinline__ float3
      getVoxelGCoo (int x, int y, int z) const
      {
        float3 coo = make_float3 (x, y, z);
        coo += 0.5f;         //shift to cell center;

        coo.x *= cell_size.x;
        coo.y *= cell_size.y;
        coo.z *= cell_size.z;

        return coo;
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

        short2 *pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(*pos);

        for (int z = 0; z < VOLUME_Z; ++z, pos += elem_step)
        {
          float3 v_g = getVoxelGCoo (x, y, z);            //3 // p

          //tranform to curr cam coo space
          float3 v = Rcurr_inv * (v_g - tcurr);           //4

          int2 coo;           //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)           //6
          {
            int Dp = depth_raw.ptr (coo.y)[coo.x];

            if (Dp != 0)
            {
              float xl = (coo.x - intr.cx) / intr.fx;
              float yl = (coo.y - intr.cy) / intr.fy;
              float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

              float sdf = 1000 * norm (tcurr - v_g) * lambda_inv - Dp; //mm

              sdf *= (-1);

              if (sdf >= -tranc_dist_mm)
              {
                float tsdf = fmin (1, sdf / tranc_dist_mm);

                int weight_prev;
                float tsdf_prev;

                //read and unpack
                unpack_tsdf (*pos, tsdf_prev, weight_prev);

                const int Wrk = 1;

                float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                int weight_new = min (weight_prev + Wrk, MAX_WEIGHT);

                pack_tsdf (tsdf_new, weight_new, *pos);
              }
            }
          }
        }
      }
    };

    __global__ void
    integrateTsdfKernel (const Tsdf tsdf) {
      tsdf ();
    }

    __global__ void
    tsdf2 (PtrStep<short2> volume, const float tranc_dist_mm, const Mat33 Rcurr_inv, float3 tcurr,
           const Intr intr, const PtrStepSz<ushort> depth_raw, const float3 cell_size)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      short2 *pos = volume.ptr (y) + x;
      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_x = Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z;
      float v_y = Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z;
      float v_z = Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z; ++z)
      {
        float3 vr;
        vr.x = v_g_x;
        vr.y = v_g_y;
        vr.z = (v_g_z + z * cell_size.z);

        float3 v;
        v.x = v_x + Rcurr_inv.data[0].z * z * cell_size.z;
        v.y = v_y + Rcurr_inv.data[1].z * z * cell_size.z;
        v.z = v_z + Rcurr_inv.data[2].z * z * cell_size.z;

        int2 coo;         //project to current cam
        coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
        coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);


        if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)         //6
        {
          int Dp = depth_raw.ptr (coo.y)[coo.x]; //mm

          if (Dp != 0)
          {
            float xl = (coo.x - intr.cx) / intr.fx;
            float yl = (coo.y - intr.cy) / intr.fy;
            float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

            float sdf = Dp - norm (vr) * lambda_inv * 1000; //mm


            if (sdf >= -tranc_dist_mm)
            {
              float tsdf = fmin (1.f, sdf / tranc_dist_mm);

              int weight_prev;
              float tsdf_prev;

              //read and unpack
              unpack_tsdf (*pos, tsdf_prev, weight_prev);

              const int Wrk = 1;

              float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
              int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

              pack_tsdf (tsdf_new, weight_new, *pos);
            }
          }
        }
        pos += elem_step;
      }       /* for(int z = 0; z < VOLUME_Z; ++z) */
    }      /* __global__ */
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth_raw, const Intr& intr, const float3& volume_size,
                                  const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, 
                                  PtrStep<short2> volume)
{
  Tsdf tsdf;

  tsdf.volume = volume;  
  tsdf.cell_size.x = volume_size.x / VOLUME_X;
  tsdf.cell_size.y = volume_size.y / VOLUME_Y;
  tsdf.cell_size.z = volume_size.z / VOLUME_Z;
  
  tsdf.intr = intr;

  tsdf.Rcurr_inv = Rcurr_inv;
  tsdf.tcurr = tcurr;
  tsdf.depth_raw = depth_raw;

  tsdf.tranc_dist_mm = tranc_dist*1000; //mm

  dim3 block (Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

#if 01
   //tsdf2<<<grid, block>>>(volume, tranc_dist, Rcurr_inv, tcurr, intr, depth_raw, tsdf.cell_size);
   integrateTsdfKernel<<<grid, block>>>(tsdf);
#endif
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}


namespace pcl
{
  namespace device
  {
    __global__ void
    scaleDepth (const PtrStepSz<ushort> depth, PtrStep<float> scaled, const Intr intr)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= depth.cols || y >= depth.rows)
        return;

      int Dp = depth.ptr (y)[x];

      float xl = (x - intr.cx) / intr.fx;
      float yl = (y - intr.cy) / intr.fy;
      float lambda = sqrtf (xl * xl + yl * yl + 1);

      float res = Dp * lambda/1000.f; //meters
      if ( intr.trunc_dist > 0 && res > intr.trunc_dist )
          scaled.ptr (y)[x] = 0;
      else
          scaled.ptr (y)[x] = res;
    }

    //重载版, 直接输入vmap_g   @2018-11-25 22:18:12
    //@param[in] vmap_g, in meters
    __global__ void
    scaleDepth_vmap (const PtrStepSz<float> vmap_g, float3 tvec, PtrStep<float> scaled, const Intr intr)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      int cols = vmap_g.cols;
      int rows = vmap_g.rows / 3;
      if (x >= cols || y >= rows)
        return;

      //const float qnan = numeric_limits<float>::quiet_NaN();
      float3 v_g;
      v_g.x = vmap_g.ptr(y)[x];
      if(isnan(v_g.x)){ //若 vmap 对应 px 是无效值
          scaled.ptr(y)[x] = 0;
          return;
      }

      v_g.y = vmap_g.ptr(y + rows)[x];
      v_g.z = vmap_g.ptr(y + rows * 2)[x];

      float res = norm(v_g - tvec); //in meters

      if ( intr.trunc_dist > 0 && res > intr.trunc_dist )
          scaled.ptr (y)[x] = 0;
      else
          scaled.ptr (y)[x] = res;
    }

    //zc: 重载, ushort->short, depth 分正负, 用于 rcFlag 沿视线 scale, 且 mm->m @2018-10-14 21:28:11
    //暂未用到 @2018-11-25 22:14:08
    __global__ void
    scaleDepth_short (const PtrStepSz<short> depth, PtrStep<float> scaled, const Intr intr)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= depth.cols || y >= depth.rows)
        return;

      int Dp = depth.ptr (y)[x];

      float xl = (x - intr.cx) / intr.fx;
      float yl = (y - intr.cy) / intr.fy;
      float lambda = sqrtf (xl * xl + yl * yl + 1);

      float res = Dp * lambda/1000.f; //meters
      if ( intr.trunc_dist > 0 && res > intr.trunc_dist )
          scaled.ptr (y)[x] = 0;
      else
          scaled.ptr (y)[x] = res;
    }

    __global__ void
    tsdf23 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
            //const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size, const pcl::gpu::tsdf_buffer buffer)
            const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size, const pcl::gpu::tsdf_buffer buffer, int3 vxlDbg) //zc: 调试
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= buffer.voxels_size.x || y >= buffer.voxels_size.y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos = volume.ptr (y) + x;
      
      // shift the pointer to relative indices
      shift_tsdf_pointer(&pos, buffer);
      
      int elem_step = volume.step * buffer.voxels_size.y / sizeof(short2);

//#pragma unroll
      for (int z = 0; z < buffer.voxels_size.z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos += elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        // As the pointer is incremented in the for loop, we have to make sure that the pointer is never outside the memory
        if(pos > buffer.tsdf_memory_end)
          pos -= (buffer.tsdf_memory_end - buffer.tsdf_memory_start + 1);
        
        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
		// old code
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("@tsdf23: Dp_scaled, sdf, tranc_dist: %f, %f, %f, %s\n", Dp_scaled, sdf, tranc_dist, 
                  sdf >= -tranc_dist ? "sdf >= -tranc_dist" : "");
          }

          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          {
            float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos, tsdf_prev, weight_prev);
            //v17, 为配合 v17 用 w 二进制末位做标记位, 这里稍作修改: unpack 时 /2, pack 时 *2; @2018-1-22 02:01:27
            //weight_prev = weight_prev >> 1;

            //v19.8.1
            int non_edge_ccnt = weight_prev % VOL1_FLAG_TH;
            weight_prev = weight_prev >> VOL1_FLAG_BIT_CNT;

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            if(doDbgPrint){
                printf("tsdf_prev & tsdf, ->tsdf_new: %f, %f, %f; w_p, w_new: %d, %d\n", tsdf_prev, tsdf, tsdf_new, weight_prev, weight_new);
            }

            //weight_new = weight_new << 1; //省略了+0, v17 的标记位默认值=0
            weight_new = (weight_new << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
            pack_tsdf (tsdf_new, weight_new, *pos);
          }
        }
        else{ //NOT (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)
            if(doDbgPrint){
                printf("vxlDbg.xyz:= (%d, %d, %d), coo.xy:= (%d, %d)\n", vxlDbg.x, vxlDbg.y, vxlDbg.z, coo.x, coo.y);
            }
        }

		/*
		// this time, we need an interpolation to get the depth value
		float2 coof = { v_x * inv_z + intr.cx, v_y * inv_z + intr.cy };
        int2 coo =
        {
          __float2int_rd (v_x * inv_z + intr.cx),
          __float2int_rd (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols - 1 && coo.y < depthScaled.rows - 1 )         //6
        {
          //float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters
		  float a = coof.x - coo.x;
		  float b = coof.y - coo.y;
		  float d00 = depthScaled.ptr (coo.y)[coo.x];
		  float d01 = depthScaled.ptr (coo.y+1)[coo.x];
		  float d10 = depthScaled.ptr (coo.y)[coo.x+1];
		  float d11 = depthScaled.ptr (coo.y+1)[coo.x+1];

          float Dp_scaled = 0;

		  if ( d00 != 0 && d01 != 0 && d10 != 0 && d11 != 0 && a > 0 && a < 1 && b > 0 && b < 1 )
		    Dp_scaled = ( 1 - b ) * ( ( 1 - a ) * d00 + ( a ) * d10 ) + ( b ) * ( ( 1 - a ) * d01 + ( a ) * d11 );

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          {
            float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos, tsdf_prev, weight_prev);

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            pack_tsdf (tsdf_new, weight_new, *pos);
          }		  
		}
		*/
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__ tsdf23

    __global__ void
    tsdf23_s2s (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
            const float tranc_dist, const float eta, //s2s (delta, eta)
            bool use_eta_trunc,
            //const Mat33 Rcurr_inv, const float3 tcurr, 
            const Mat33 Rcurr_inv, const float3 tcurr, const float3 volume000_gcoo,
            const Intr intr, const float3 cell_size, int3 vxlDbg) //zc: 调试
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      //float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      //float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      //float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;
      float v_g_x = (x + 0.5f) * cell_size.x + volume000_gcoo.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y + volume000_gcoo.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z + volume000_gcoo.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos = volume.ptr (y) + x;
      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos += elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        // old code
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              //printf("@tsdf23_s2s: Dp_scaled, sdf, tranc_dist: %f, %f, %f, %s; sdf/tdist: %f, coo.xy: (%d, %d)\n", Dp_scaled, sdf, tranc_dist, 
              //    sdf >= -tranc_dist ? "sdf >= -tranc_dist" : "", sdf/tranc_dist, coo.x, coo.y);
          }

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          //if (Dp_scaled != 0 && sdf >= -eta) //meters //比较 eta , 而非 delta (tdist)

          //↓-参数控制到底选谁做 tdist, 
          float tdist_real = use_eta_trunc ? eta : tranc_dist;
          if (Dp_scaled != 0 && sdf >= -tdist_real) //meters 
          {
            float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

            if(sdf < -tranc_dist)
                tsdf = -1.0f;

#if 10   //滑窗累加
            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos, tsdf_prev, weight_prev);
            //v17, 为配合 v17 用 w 二进制末位做标记位, 这里稍作修改: unpack 时 /2, pack 时 *2; @2018-1-22 02:01:27
            weight_prev = weight_prev >> VOL1_FLAG_BIT_CNT;

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            if(doDbgPrint){
                //printf("tsdf_prev, tsdf_curr, tsdf_new: %f, %f, %f; wp, wnew: %d, %d\n", tsdf_prev, tsdf, tsdf_new, weight_prev, weight_new);
            }
#elif 1 //直接 set volume 为当前 dmap 映射结果
            float tsdf_new = tsdf;
            int weight_new = 1;
#endif
            weight_new = weight_new << VOL1_FLAG_BIT_CNT; //省略了+0, v17 的标记位默认值=0
            pack_tsdf (tsdf_new, weight_new, *pos);
          }
          else{ //(Dp_scaled == 0 || sdf < -eta)
            //float tsdf_new = 0;
            //int weight_new = 0;
            //pack_tsdf (tsdf_new, weight_new, *pos);
            if(doDbgPrint)
                printf("NOT (Dp_scaled != 0 && sdf >= -eta)\n");
          }
        }
        else{ //NOT (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)
            //if(doDbgPrint){
            //    printf("vxlDbg.xyz:= (%d, %d, %d), coo.xy:= (%d, %d)\n", vxlDbg.x, vxlDbg.y, vxlDbg.z, coo.x, coo.y);
            //}
        }
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }//__global__ tsdf23_s2s

    enum{FUSE_KF_AVGE, //kf tsdf 原策略
        FUSE_RESET, //i 冲掉 i-1
        FUSE_IGNORE_CURR //忽视 i
        ,FUSE_FIX_PREDICTION //先负后正, 正冲掉负
        ,FUSE_CLUSTER   //二分类思路
    };

    __global__ void
    tsdf23_v11 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
#if 0
          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
#else
          //↓--v11.7: 按 wmap (weight) 动态设定 tranc_dist 上限, (但基准不变:
          //float tranc_dist_real = tranc_dist * weiFactor;
          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8

          if(doDbgPrint){
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          //if (Dp_scaled != 0 && -tranc_dist_real <= sdf && sdf < tranc_dist) //meters, v11.8
#endif
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            bool isInclined = (incidAngleMask.ptr(coo.y)[coo.x] != 0); //太倾斜了, 入射角太大
            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vrayPrev;
            //假设已归一化, 虽然 char->float 有误差, 但不再做归一化
            vrayPrev.x = 1.f * (*vrayPrev_pos).x / CHAR_MAX; //char2float
            vrayPrev.y = 1.f * (*vrayPrev_pos).y / CHAR_MAX;
            vrayPrev.z = 1.f * (*vrayPrev_pos).z / CHAR_MAX;

            //v11.3: 用 vrayPrev_pos[3] 做 hadSeenConfidence, 取代 hadSeen 布尔量: //2017-3-11 21:40:24
            signed char *seenConfid = &vrayPrev_pos->w;
            const int seenConfidTh = 15;

            float3 vray; //这次不用视线做主要判断, 此处只是用来测试 nmap 传参对错
                            //v11.2 改成都要做: 视线 & 表面法向双重判定 //2017-3-8 22:00:32
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snormPrev;
            snormPrev.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snormPrev.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snormPrev.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //v11.9: 有时候 snorm 被噪声错误地初始化, 真实值却难以再去修正 snorm @2017-4-11 17:03:51
            int snormPrevConfid = (*snorm_pos).w;
            const int snormPrevConfid_thresh = 5;

            //const bool hadSeen = *flag_pos; //别名 hadSeen, 不准确
            const bool hadSeen = (*seenConfid > seenConfidTh); //v11.3: 策略, 当连续 confid++, 达到阈值之后, 才标记 seen; 若达不到阈值, 还要--

            //bool isSnormPrevInit = (norm(snormPrev) > 1e-8);
            //bool isSnormPrevInit = ( (norm(snormPrev) > 1e-8) && (snormPrevConfid > snormPrevConfid_thresh) );
            bool isSnormPrevInit = (snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零

            if(doDbgPrint){
                printf("isInclined, %d\n", isInclined);
                printf("cos_vray_norm, %f; snorm_curr_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("(norm(snormPrev) == 0) == %s; (norm(snormPrev) < 1e-8) == %s\n",
                    norm(snormPrev) == 0 ? "T" : "F",
                    norm(snormPrev) < 1e-8 ? "T" : "F");
            }


            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);

            float tsdf_prev2nd = -123;
            int weight_prev2nd = -233;
            unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

            //const int w2ndCntThresh = 10; //w2nd 超过此阈值才能逆袭
            const int w2ndCntThresh = 10 * 10; //v11.4 用 weiFactor 之后

            if(doDbgPrint){
                printf("tsdf_prev: tsdf1st: %f, %d; tsdf2nd: %f, %d;\n", tsdf_prev1, weight_prev1, tsdf_prev2nd, weight_prev2nd);
            }

            int fuse_method = FUSE_KF_AVGE; //默认原策略
            bool doUpdateVrayAndSnorm = false;

            const float cosThreshVray = //0.8660254f; //cos(30°)
                //0.9396926f; //cos(20°) //当 largeIncidMask 取 80 阈值时, 此处应为 (90-x)*2
                0.9659258f; //cos(15°) //因为largeIncidMask 以 75°为阈值, 所以这里用 90-75=15 为阈值
                //0.996194698; //cos(5°)
            const float cosThreshSnorm = 0.8660254f; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_norm = dot(snormPrev, snorm_curr_g);
            float cos_vray = dot(vrayPrev, vray_normed);
            bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm && cos_vray < cosThreshVray); //snorm-init 之后才做 newFace 判定 @2017-4-21 00:42:00
            //bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm); //去掉 vray 判定, 别! 原因: vray 防止 *视角稳定但snorm 突变 (边缘etc.)* 情形, 不轻易 isNewFace=true

            if(doDbgPrint){
                printf("cos_norm: snormPrev, snorm_curr_g, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_norm, 
                    snormPrev.x, snormPrev.y, snormPrev.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);
                printf("\tcos_vray, vrayPrev, vray_normed, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_vray, 
                    vrayPrev.x, vrayPrev.y, vrayPrev.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("%s, snormPrevConfid, snormPrevConfid_thresh: %d, %d\n", isNewFace ? "isNewFace-T" : "isNewFace-F", snormPrevConfid, snormPrevConfid_thresh);
                printf("\t%s\n", cos_norm > cosThreshSnorm ? "cos_norm > cosThreshSnorm" : "cos_norm <= cosThreshSnorm");
                printf("\t%s\n", cos_vray > cosThreshVray ? "cos_vray > cosThreshVray" : "cos_vray <= cosThreshVray");
            }


#if 01   //v11.3, v11.4, 
            if(isInclined){ //若边缘, doUpdateVray 保持 false
                if(!hadSeen){ //若 seen-flag 未初始化过
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=F; ++FUSE_KF_AVGE\n");
                    fuse_method = FUSE_KF_AVGE;

                    //*seenConfid = max(0, *seenConfid - 1);
                    //↑-- 不要 -1 了, 只增不减, 但同时 seenConfidTh 阈值调高 (5 -> 15), 延缓其 flag=true   @2017-3-23 11:11:55
                }
                else{ //if(hadSeen) //若之前 seen
#if 0   //忘了 sdf < 0 这个判定为什么了, 目前感觉会导致有偏差, 放弃   @2017-3-9 15:06:22
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; %s; sdf: %f\n", sdf<0 ? "==FUSE_IGNORE_CURR" : "++FUSE_KF_AVGE", sdf);
                    if(sdf < 0)
                        fuse_method = FUSE_IGNORE_CURR;
                    else
                        fuse_method = FUSE_KF_AVGE;
#elif 1 //一律 ignore
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; \n");
                    fuse_method = FUSE_IGNORE_CURR;
#endif
                }
            }
            else{ //if(!isInclined){ //若非边缘, 在内部
                //*seenConfid = min(Tsdf::MAX_WEIGHT, *seenConfid + 1); //v11.4 用 weiFactor 之后, 这里反而是 BUG!!
                *seenConfid = min(SCHAR_MAX, *seenConfid + 1);

                if(!isSnormPrevInit){ //vray.prev 若未初始化, 用 < epsilon 判定
                    //if (*seenConfid > seenConfidTh) //这就是 hadSeen, 所以不要这么判定
                        //doUpdateVrayAndSnorm = true;
                }


                if(!hadSeen){ //若 seen-flag 未初始化过
#if 0   //< v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; >>FUSE_RESET\n");
                    *flag_pos = true;
                    fuse_method = FUSE_RESET;
#elif 1 //v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; seenConfid, seenConfidTh: %d, %d, ++FUSE_KF_AVGE~~~~~\n", *seenConfid, seenConfidTh); //别处也没有 reset 了
                    fuse_method = FUSE_KF_AVGE;
#endif
                    //if (*seenConfid > seenConfidTh) //既然 hadSeen 逻辑改过, 则此处必然一直 false
                    //    doUpdateVrayAndSnorm = true;
                }
                else{ //if(hadSeen) //若之前 seen, 必然经过过 【isInclined-F; hadSeen=F】阶段, 也必然 isSnormPrevInit->true, 不必再 if-isSnormPrevInit
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=T;\n");

                    //if(cos_norm > cosThresh ){ //夹角角度 <30°, 算作同视角
                    if(!isNewFace){ //同视角, 双 cos 联合判定
                        //TODO...
                        fuse_method = FUSE_KF_AVGE; //其实默认

                        //if (*seenConfid > seenConfidTh) //不必, 因为已在 if-hadSeen 分支内
                        if(cos_norm > cosThreshSnorm) //反之 cos_norm < th 时, 尽管 newFace=false, 但不应 update
                            doUpdateVrayAndSnorm = true;

                        if(!isSnormPrevInit)
                            doUpdateVrayAndSnorm = true;
                    }
                    else{ // >30°, 算作不同视角, 比如转过头之后
                        //if(!isSnormPrevInit) //newFace 改进之后, 这里不会再进入
                        //    doUpdateVrayAndSnorm = true;

#if 10   //三类不完善, 而且语义不明确, 放弃 @2017-3-24 17:50:24
                        //化简为三类
                        if(tsdf_curr < 0 && tsdf_curr < tsdf_prev1){
                            if(doDbgPrint)
                                printf("\ttsdf < 0 && tsdf < tsdf_prev1; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);

                            fuse_method = FUSE_IGNORE_CURR;
                        }
                        else if(tsdf_prev1 < 0 && tsdf_prev1 < tsdf_curr){
                            if(doDbgPrint){
                                printf("\ttsdf_prev1 < 0 && tsdf_prev1 < tsdf; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                                printf("\t\t, weight_prev1, w2ndCntThresh: %d, %d\n", weight_prev1, w2ndCntThresh);
                            }
#if 0   //weight_prev1 是否要与 w2ndCntThresh 对比?
                            if(weight_prev1 > w2ndCntThresh){
                                fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
                            }
                            else{
                                fuse_method = FUSE_KF_AVGE; //这里默认是否有问题
                            }
#elif 1 //1st 不与 w2ndCntThresh 对比, 因为下面做对比控制: weight_new2nd > w2ndCntThresh
                            fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
#endif
                            //doUpdateSnorm = true; //放到 FUSE_FIX_PREDICTION 里判定
                        }
                        else if(tsdf_curr >=0 && tsdf_prev1 >= 0){
                            if(doDbgPrint){
                                printf("\ttsdf >=0 && tsdf_prev1 >= 0; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                            }
                            fuse_method = FUSE_KF_AVGE;
                            doUpdateVrayAndSnorm = true;
                        }
#elif 1 //细分、增补为7类, @2017-3-24 17:51:03
                        if(tsdf_prev1 >= 0){
                            if(tsdf_curr <0){
                                fuse_method = FUSE_IGNORE_CURR;
                                doUpdateVrayAndSnorm = true;
                                
                                if(doDbgPrint)
                                    printf("+~~-,, ==FUSE_IGNORE_CURR\n");
                            }
                            else{//tsdf_curr >=0
                                if(sdf < tranc_dist){
                                    fuse_method = FUSE_KF_AVGE;

                                    if(doDbgPrint)
                                        printf("+~~↓+,, ++FUSE_KF_AVGE\n");
                                }
                                else{
                                    fuse_method = FUSE_IGNORE_CURR;

                                    if(doDbgPrint)
                                        printf("+~~↑+,, ==FUSE_IGNORE_CURR\n");
                                }
                            }
                        }
                        else{ //tsdf_prev1 <0
                            float abs_tsdfcurr = abs(tsdf_curr);
                            if(abs_tsdfcurr < abs(tsdf_prev1)){
                                fuse_method = FUSE_FIX_PREDICTION;

                                if(doDbgPrint){
                                    if(tsdf_curr < 0)
                                        printf("-~~↑-,, >>FUSE_FIX_PREDICTION\n");
                                    else
                                        printf("-~~↓+,, >>FUSE_FIX_PREDICTION\n");
                                }
                            }
                            else{
                                fuse_method = FUSE_IGNORE_CURR;

                                if(doDbgPrint){
                                    if(tsdf_curr < 0)
                                        printf("-~~↓-,, ==FUSE_IGNORE_CURR\n");
                                    else
                                        printf("-~~↑+,, ==FUSE_IGNORE_CURR\n");
                                }
                            }
                        }
#endif
                    }//cos vs. cosTh
                }//if-hadSeen
            }//if-isInclined
#elif 0 //v11.5; //不靠谱, 又忘了思路详情了... @2017-3-16 00:05:51
            if(isInclined){
                if(doDbgPrint)
                    printf("isInclined-T; ++FUSE_KF_AVGE\n");

                fuse_method = FUSE_KF_AVGE;
                doUpdateVrayAndSnorm = false;
            }
            else{ //if(!isInclined){ //若非边缘, 在内部
                if(doDbgPrint)
                    printf("isInclined-F;\n");

                bool isSnormPrevInit = (norm(snormPrev) > 1e-8);
                if(!isSnormPrevInit){ //vray.prev 若未初始化, 用 < epsilon 判定
                    if(doDbgPrint)
                        printf("\tisSnormPrevInit-F\n");

                    fuse_method = FUSE_KF_AVGE;
                    doUpdateVrayAndSnorm = true;
                }
                else{ //vray+snorm 都初始化过了
                    if(!isNewFace){ //同视角, 双 cos 联合判定
                        if(doDbgPrint)
                            printf("\tisNewFace-F\n");

                        fuse_method = FUSE_KF_AVGE; //其实默认
                        doUpdateVrayAndSnorm = true;
                    }
                    else{ // isNewFace, 算作不同视角, 比如转过头之后
                        if(doDbgPrint)
                            printf("\tisNewFace-T\n");

                        //化简为三类
                        if(tsdf_curr < 0 && tsdf_curr < tsdf_prev1){
                            if(doDbgPrint)
                                printf("\ttsdf < 0 && tsdf < tsdf_prev1; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);

                            fuse_method = FUSE_IGNORE_CURR;
                        }
                        else if(tsdf_prev1 < 0 && tsdf_prev1 < tsdf_curr){
                            if(doDbgPrint){
                                printf("\ttsdf_prev1 < 0 && tsdf_prev1 < tsdf; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                                printf("\t\t, weight_prev1, w2ndCntThresh: %d, %d\n", weight_prev1, w2ndCntThresh);
                            }
#if 0   //weight_prev1 是否要与 w2ndCntThresh 对比?
                            if(weight_prev1 > w2ndCntThresh){
                                fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
                            }
                            else{
                                fuse_method = FUSE_KF_AVGE; //这里默认是否有问题
                            }
#elif 1 //1st 不与 w2ndCntThresh 对比, 因为下面做对比控制: weight_new2nd > w2ndCntThresh
                            fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
#endif
                            //doUpdateSnorm = true; //放到 FUSE_FIX_PREDICTION 里判定
                        }
                        else if(tsdf_curr >=0 && tsdf_prev1 >= 0){
                            if(doDbgPrint){
                                printf("\ttsdf >=0 && tsdf_prev1 >= 0; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                            }
                            fuse_method = FUSE_KF_AVGE;
                            doUpdateVrayAndSnorm = true;
                        }
                    }//isNewFace
                }//vray+snorm 都初始化过了
            }
#elif 1 //v11.6: v11.5不靠谱, 改成 isInclined 只用于控制 vray+snorm 的更新; 去掉了 hadSeen-flag 控制
            //代码是简化了, 但是结果变差了, 详情略过
            bool isSnormPrevInit = (norm(snormPrev) > 1e-8);

            if(isInclined){
                doUpdateVrayAndSnorm = false;
            }
            else if(!isSnormPrevInit){
                doUpdateVrayAndSnorm = true;
            }

            if(!isSnormPrevInit){
                fuse_method = FUSE_KF_AVGE;
            }
            else{ //vray+snorm 都初始化过了
                if(!isNewFace){ //同视角, 双 cos 联合判定
                    if(doDbgPrint)
                        printf("\tisNewFace-F\n");

                    fuse_method = FUSE_KF_AVGE; //其实默认

                    if(!isInclined)
                        doUpdateVrayAndSnorm = true;
                }
                else{ // isNewFace, 算作不同视角, 比如转过头之后
                    if(doDbgPrint)
                        printf("\tisNewFace-T\n");

                    //化简为三类
                    if(tsdf_curr < 0 && tsdf_curr < tsdf_prev1){
                        if(doDbgPrint)
                            printf("\ttsdf < 0 && tsdf < tsdf_prev1; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);

                        fuse_method = FUSE_IGNORE_CURR;
                    }
                    else if(tsdf_prev1 < 0 && tsdf_prev1 < tsdf_curr){
                        if(doDbgPrint){
                            printf("\ttsdf_prev1 < 0 && tsdf_prev1 < tsdf; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                            printf("\t\t, weight_prev1, w2ndCntThresh: %d, %d\n", weight_prev1, w2ndCntThresh);
                        }
                        fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
                        //doUpdateSnorm = true; //放到 FUSE_FIX_PREDICTION 里判定
                    }
                    else if(tsdf_curr >=0 && tsdf_prev1 >= 0){
                        if(doDbgPrint){
                            printf("\ttsdf >=0 && tsdf_prev1 >= 0; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                        }
                        fuse_method = FUSE_KF_AVGE;

                        if(!isInclined)
                            doUpdateVrayAndSnorm = true;
                    }
                }//isNewFace
            }//vray+snorm 都初始化过了
#endif
            const int Wrk = max(15 * weiFactor, 1.f);
            if(FUSE_KF_AVGE == fuse_method){
                float tsdf_new1 = (tsdf_prev1 * weight_prev1 + Wrk * tsdf_curr) / (weight_prev1 + Wrk);
                int weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0

                pack_tsdf (tsdf_new1, weight_new1, *pos1);
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过

                if(doDbgPrint)
                    printf("++FUSE_KF_AVGE, weight_new1, weight_new2nd, %d, %d\n", weight_new1, weight_new2nd);
            }
            else if(FUSE_FIX_PREDICTION == fuse_method){ //取代粗暴 FUSE_RESET
#if 0   //factor/step 方式不行
//                   //const int pos_neg_factor = 8;
//                   int pos_neg_factor = min(weight_prev1 / 10, 1); //到这里时可能 w1 其实不大, 所以不能粗暴设定大步长
//                   int pnWrk = pos_neg_factor * Wrk;
//                   float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + pnWrk * tsdf) / (weight_prev2nd + pnWrk);
//                   int weight_new2nd = min (weight_prev2nd + pnWrk, Tsdf::MAX_WEIGHT);
// 
//                   int weight_new1 = max(weight_prev1 - pnWrk, 0);
// 
//                   if(weight_new2nd > weight_new1){ //若 2nd 逆袭, 则交换 1st/2nd, 永远保持 1st 为主
#elif 1
                float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + Wrk * tsdf_curr) / (weight_prev2nd + Wrk);
                int weight_new2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);

                //int weight_new1 = max(weight_prev1 - Wrk, 0);
                if(weight_new2nd > w2ndCntThresh){ //交换 1st/2nd, 永远保持 1st 为主 //这里改成: 2nd 不必逆袭 1st, 只要大于某常量阈值即可
#endif
                    if(doDbgPrint){
                        printf("weight_new2nd > w2ndCntThresh,,, exchanging 1st-2nd\n");
                    }
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos1); //new-2nd 放到 pos-1st 中
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos2nd);

                    doUpdateVrayAndSnorm = true; //直到 2nd 逆袭, 才用新的 snorm 更新当前 vxl
                }
                else{ //否则
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos1);
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos2nd);
                    doUpdateVrayAndSnorm = false;
                }

                if(doDbgPrint)
                    //printf("...>>FUSE_FIX_PREDICTION, weight_new1, weight_new2nd, %d, %d\n", weight_new1, weight_new2nd);
                    printf("...>>FUSE_FIX_PREDICTION, weight_new2nd, %d\n", weight_new2nd);

                //调试: 不管doDbgPrint, 全部输出, 看究竟有没有走到这一步的 vxl: @2017-3-11 21:22:59
                //答: 有!! 因为 FUSE_FIX_PREDICTION 目前针对 case: tsdf_prev1 < 0 && tsdf_prev1 < tsdf
                //printf("...>>FUSE_FIX_PREDICTION, weight_new2nd, %d,,, [xyz]=(%d, %d, %d)\n", weight_new2nd, x, y, z);
            }
            else if(FUSE_RESET == fuse_method){
                if(doDbgPrint)
                    printf(">>FUSE_RESET\n");

                pack_tsdf(tsdf_curr, 1, *pos1);
            }
            else if(FUSE_IGNORE_CURR == fuse_method){
                if(doDbgPrint)
                    printf("==FUSE_IGNORE_CURR\n");

                //DO-NOTHING!!! //×
                //IGN时, 也要 2nd 弄一下 @2017-3-16 03:53:08
                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过
            }

            if(doDbgPrint)
                printf("doUpdateSnorm: %d\n", doUpdateVrayAndSnorm);

            if(doUpdateVrayAndSnorm){
                //max (-DIVISOR, min (DIVISOR, (int)nearbyintf (tsdf * DIVISOR))); //@pack_tsdf
                //因为 vray_normed.xyz 必然均 <=1, 所以不必 max/min... ↑
                (*vrayPrev_pos).x = (int)nearbyintf(vray_normed.x * CHAR_MAX); //float2char
                (*vrayPrev_pos).y = (int)nearbyintf(vray_normed.y * CHAR_MAX);
                (*vrayPrev_pos).z = (int)nearbyintf(vray_normed.z * CHAR_MAX);

                //用了 pcc 求 nmap 方法之后, 边缘法向不准 (因为 sobel?), 要切掉; 否则导致一些坑洼 @2017-3-15 16:54:25
                //用 4:=7/2+1
                const int edgeMarg = 4;
                if(coo.x < edgeMarg || coo.x >= depthScaled.cols - edgeMarg || coo.y < edgeMarg || coo.y >= depthScaled.rows - edgeMarg){
                    if(doDbgPrint)
                        printf("+++++++++++++++at edge, dont-update-snorm; coo.xy: (%d, %d)\n", coo.x, coo.y);
                }
                else{
                    //(*snorm_pos).w += 1; //即 snormPrevConfid
                    (*snorm_pos).w = min(SCHAR_MAX, snormPrevConfid + 1);

                    if(!isSnormPrevInit || isNewFace){
                        if(doDbgPrint)
                            printf("\t(!isSnormPrevInit || isNewFace): %d, %d; snormPrevConfid: %d\n", isSnormPrevInit, isNewFace, (*snorm_pos).w);

                        (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                        (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                        (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                    }
                    else{ //isSnormPrevInit && !isNewFace //v11.6: 若snorm 初始化过了, 且当前没有突变, 则用 model 的法向, 因为其更稳定
                        if(doDbgPrint)
                            printf("\tisSnormPrevInit && !isNewFace\n");

                        float3 snorm_model_g;
                        snorm_model_g.x = nmap_model_g.ptr(coo.y)[coo.x];
                        snorm_model_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
                        snorm_model_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

                        float cos_vray_norm_model = dot(snorm_model_g, vray_normed);
                        if(cos_vray_norm_model > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                            //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                            //假设不保证外部已正确预处理：
                            snorm_model_g.x *= -1;
                            snorm_model_g.y *= -1;
                            snorm_model_g.z *= -1;
                        }
                        if(doDbgPrint)
                            printf("\t\tcos_vray_norm_model, %f; snorm_model_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm_model, snorm_model_g.x, snorm_model_g.y, snorm_model_g.z, vray_normed.x, vray_normed.y, vray_normed.z);

                        float cos_norm_model_and_prev = dot(snorm_model_g, snormPrev);
                        //↑--按理说, 此时 n_model, n_curr 应该夹角很小 (已经考虑了与视线做 ±1 乘法) //v11.7   @2017-3-17 15:52:25
                        //但若因为噪声, 导致 n_model 偏差过大, 则完全不更新:
                        //if(cos_norm_model_and_prev > cosThreshSnorm){
                        //if(1){ //发现 snormPrev 不好

                        //zc: 改逻辑: 若 snorm-model/curr 相近才更新 @2017-4-25 21:24:23
                        float cos_norm_model_and_curr = dot(snorm_model_g, snorm_curr_g);
                        if(cos_norm_model_and_curr > cosThreshSnorm){
                            //发现 __float2int_rd 因 round-down 存在退化问题, 数值不稳定, 改用 nearbyintf (搜来的)?  @2017-3-15 15:33:33
                            (*snorm_pos).x = (int)nearbyintf(snorm_model_g.x * CHAR_MAX); //float2char
                            (*snorm_pos).y = (int)nearbyintf(snorm_model_g.y * CHAR_MAX);
                            (*snorm_pos).z = (int)nearbyintf(snorm_model_g.z * CHAR_MAX);
                        }
                        else{
                            //DO-NOTHING!!!
                        }
                    }
                }//cut-edgeMarg

                if(doDbgPrint){
                    printf("newVray: [%d, %d, %d]\n", (*vrayPrev_pos).x, (*vrayPrev_pos).y, (*vrayPrev_pos).z);
                    printf("\tnewSnorm: [%d, %d, %d]\n", (*snorm_pos).x, (*snorm_pos).y, (*snorm_pos).z);
                }
            }//if-(doUpdateVrayAndSnorm)
          }//if- (Dp_scaled != 0 && sdf >= -tranc_dist)
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v11

    __global__ void
    tsdf23_v11_remake (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
#if 0
          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
#else
          //↓--v11.7: 按 wmap (weight) 动态设定 tranc_dist 上限, (但基准不变:
          //float tranc_dist_real = tranc_dist * weiFactor;
          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8

          if(doDbgPrint){
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          //if (Dp_scaled != 0 && -tranc_dist_real <= sdf && sdf < tranc_dist) //meters, v11.8
#endif
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            bool isInclined = (incidAngleMask.ptr(coo.y)[coo.x] != 0); //太倾斜了, 入射角太大
            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vrayPrev;
            //假设已归一化, 虽然 char->float 有误差, 但不再做归一化
            vrayPrev.x = 1.f * (*vrayPrev_pos).x / CHAR_MAX; //char2float
            vrayPrev.y = 1.f * (*vrayPrev_pos).y / CHAR_MAX;
            vrayPrev.z = 1.f * (*vrayPrev_pos).z / CHAR_MAX;

            //v11.3: 用 vrayPrev_pos[3] 做 hadSeenConfidence, 取代 hadSeen 布尔量: //2017-3-11 21:40:24
            signed char *seenConfid = &vrayPrev_pos->w;
            const int seenConfidTh = 15;

            float3 vray; //这次不用视线做主要判断, 此处只是用来测试 nmap 传参对错
                            //v11.2 改成都要做: 视线 & 表面法向双重判定 //2017-3-8 22:00:32
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snormPrev;
            snormPrev.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snormPrev.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snormPrev.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //v11.9: 有时候 snorm 被噪声错误地初始化, 真实值却难以再去修正 snorm @2017-4-11 17:03:51
            signed char *snormPrevConfid = &snorm_pos->w;
            const int snormPrevConfid_thresh = 5;

            //const bool hadSeen = *flag_pos; //别名 hadSeen, 不准确
            const bool hadSeen = (*seenConfid > seenConfidTh); //v11.3: 策略, 当连续 confid++, 达到阈值之后, 才标记 seen; 若达不到阈值, 还要--

            //bool isSnormPrevInit = (norm(snormPrev) > 1e-8);
            //bool isSnormPrevInit = ( (norm(snormPrev) > 1e-8) && (snormPrevConfid > snormPrevConfid_thresh) );
            bool isSnormPrevInit = (*snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零

            if(doDbgPrint){
                printf("isInclined, %d\n", isInclined);
                printf("cos_vray_norm, %f; snorm_curr_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("(norm(snormPrev) == 0) == %s; (norm(snormPrev) < 1e-8) == %s\n",
                    norm(snormPrev) == 0 ? "T" : "F",
                    norm(snormPrev) < 1e-8 ? "T" : "F");
            }


            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);

            float tsdf_prev2nd = -123;
            int weight_prev2nd = -233;
            unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

            //const int w2ndCntThresh = 10; //w2nd 超过此阈值才能逆袭
            const int w2ndCntThresh = 10 * 10; //v11.4 用 weiFactor 之后

            if(doDbgPrint){
                printf("tsdf_prev: tsdf1st: %f, %d; tsdf2nd: %f, %d;\n", tsdf_prev1, weight_prev1, tsdf_prev2nd, weight_prev2nd);
            }

            int fuse_method = FUSE_KF_AVGE; //默认原策略
            bool doUpdateVrayAndSnorm = false;

            const float cosThreshVray = //0.8660254f; //cos(30°)
                //0.9396926f; //cos(20°) //当 largeIncidMask 取 80 阈值时, 此处应为 (90-x)*2
                0.9659258f; //cos(15°) //因为largeIncidMask 以 75°为阈值, 所以这里用 90-75=15 为阈值
                //0.996194698; //cos(5°)
            const float cosThreshSnorm = 0.8660254f; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_norm = dot(snormPrev, snorm_curr_g);
            float cos_vray = dot(vrayPrev, vray_normed);
            bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm && cos_vray < cosThreshVray); //snorm-init 之后才做 newFace 判定 @2017-4-21 00:42:00
            //bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm); //去掉 vray 判定, 别! 原因: vray 防止 *视角稳定但snorm 突变 (边缘etc.)* 情形, 不轻易 isNewFace=true

            //zc: 增加判定, 若 weight-factor 太小(如, 边缘区域), 则直接均值, 且不 updateVray @2017-7-13 22:29:39
            if(weiFactor > 0.2){


            if(doDbgPrint){
                printf("cos_norm: snormPrev, snorm_curr_g, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_norm, 
                    snormPrev.x, snormPrev.y, snormPrev.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);
                printf("\tcos_vray, vrayPrev, vray_normed, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_vray, 
                    vrayPrev.x, vrayPrev.y, vrayPrev.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("%s, snormPrevConfid, snormPrevConfid_thresh: %d, %d\n", isNewFace ? "isNewFace-T" : "isNewFace-F", *snormPrevConfid, snormPrevConfid_thresh);
                printf("\t%s\n", cos_norm > cosThreshSnorm ? "cos_norm > cosThreshSnorm" : "cos_norm <= cosThreshSnorm");
                printf("\t%s\n", cos_vray > cosThreshVray ? "cos_vray > cosThreshVray" : "cos_vray <= cosThreshVray");
            }


            if(isInclined){ //若边缘, doUpdateVray 保持 false
                if(!hadSeen){ //若 seen-flag 未初始化过
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=F; ++FUSE_KF_AVGE\n");
                    fuse_method = FUSE_KF_AVGE;

                    //*seenConfid = max(0, *seenConfid - 1);
                    //↑-- 不要 -1 了, 只增不减, 但同时 seenConfidTh 阈值调高 (5 -> 15), 延缓其 flag=true   @2017-3-23 11:11:55
                }
                else{ //if(hadSeen) //若之前 seen
#if 0   //忘了 sdf < 0 这个判定为什么了, 目前感觉会导致有偏差, 放弃   @2017-3-9 15:06:22
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; %s; sdf: %f\n", sdf<0 ? "==FUSE_IGNORE_CURR" : "++FUSE_KF_AVGE", sdf);
                    if(sdf < 0)
                        fuse_method = FUSE_IGNORE_CURR;
                    else
                        fuse_method = FUSE_KF_AVGE;
#elif 1 //一律 ignore
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; \n");
                    fuse_method = FUSE_IGNORE_CURR;
#endif
                }
            }
            else{ //if(!isInclined){ //若非边缘, 在内部
                //*seenConfid = min(Tsdf::MAX_WEIGHT, *seenConfid + 1); //v11.4 用 weiFactor 之后, 这里反而是 BUG!!
                *seenConfid = min(SCHAR_MAX, *seenConfid + 1);

                if(!isSnormPrevInit){ //vray.prev 若未初始化, 用 < epsilon 判定
                    //if (*seenConfid > seenConfidTh) //这就是 hadSeen, 所以不要这么判定
                        //doUpdateVrayAndSnorm = true;
                }


                if(!hadSeen){ //若 seen-flag 未初始化过
#if 0   //< v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; >>FUSE_RESET\n");
                    *flag_pos = true;
                    fuse_method = FUSE_RESET;
#elif 1 //v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; seenConfid, seenConfidTh: %d, %d, ++FUSE_KF_AVGE~~~~~\n", *seenConfid, seenConfidTh); //别处也没有 reset 了
                    fuse_method = FUSE_KF_AVGE;
#endif
                    //if (*seenConfid > seenConfidTh) //既然 hadSeen 逻辑改过, 则此处必然一直 false
                    //    doUpdateVrayAndSnorm = true;
                }
                else{ //if(hadSeen) //若之前 seen, 必然经过过 【isInclined-F; hadSeen=F】阶段, 也必然 isSnormPrevInit->true, 不必再 if-isSnormPrevInit
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=T;\n");

                    //if(cos_norm > cosThresh ){ //夹角角度 <30°, 算作同视角
                    if(!isNewFace){ //同视角, 双 cos 联合判定
                        //TODO...
                        fuse_method = FUSE_KF_AVGE; //其实默认

                        //if (*seenConfid > seenConfidTh) //不必, 因为已在 if-hadSeen 分支内
#if 0
                        if(cos_norm > cosThreshSnorm) //反之 cos_norm < th 时, 尽管 newFace=false, 但不应 update
                            doUpdateVrayAndSnorm = true;

                        if(!isSnormPrevInit)
                            doUpdateVrayAndSnorm = true;
#elif 1 //改成必然 update @2017-7-13 15:45:12
                        doUpdateVrayAndSnorm = true;
#endif
                    }
                    else{ // >30°, 算作不同视角, 比如转过头之后
                        //if(!isSnormPrevInit) //newFace 改进之后, 这里不会再进入
                        //    doUpdateVrayAndSnorm = true;

#if 10   //三类不完善, 而且语义不明确, 放弃 @2017-3-24 17:50:24
                        //化简为三类
                        if(tsdf_curr < 0 && tsdf_curr < tsdf_prev1){
                            if(doDbgPrint)
                                printf("\ttsdf < 0 && tsdf < tsdf_prev1; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);

                            fuse_method = FUSE_IGNORE_CURR;
                        }
                        else if(tsdf_prev1 < 0 && tsdf_prev1 < tsdf_curr){
                            if(doDbgPrint){
                                printf("\ttsdf_prev1 < 0 && tsdf_prev1 < tsdf; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                                printf("\t\t, weight_prev1, w2ndCntThresh: %d, %d\n", weight_prev1, w2ndCntThresh);
                            }
#if 0   //weight_prev1 是否要与 w2ndCntThresh 对比?
                            if(weight_prev1 > w2ndCntThresh){
                                fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
                            }
                            else{
                                fuse_method = FUSE_KF_AVGE; //这里默认是否有问题
                            }
#elif 1 //1st 不与 w2ndCntThresh 对比, 因为下面做对比控制: weight_new2nd > w2ndCntThresh
                            fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
#endif
                            //doUpdateSnorm = true; //放到 FUSE_FIX_PREDICTION 里判定
                        }
                        else if(tsdf_curr >=0 && tsdf_prev1 >= 0){
                            if(doDbgPrint){
                                printf("\ttsdf >=0 && tsdf_prev1 >= 0; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                            }
                            fuse_method = FUSE_KF_AVGE;

                            //if(cos_norm > 0) //加约束: 法向突变不能超90°; 是为了防止薄片区域, 因原始深度图噪声, 导致旧的背面法向更新;  @2017-11-17 15:39:06
                            //↑--移到 v12 里, 作对照 @2017-12-3 22:09:36
                            doUpdateVrayAndSnorm = true;
                        }
#endif
                    }//cos vs. cosTh
                }//if-hadSeen
            }//if-isInclined
            }//if-(weiFactor > 0.2)

            const int Wrk = max(15 * weiFactor, 1.f);
            if(FUSE_KF_AVGE == fuse_method){
                float tsdf_new1 = (tsdf_prev1 * weight_prev1 + Wrk * tsdf_curr) / (weight_prev1 + Wrk);
                int weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0

                pack_tsdf (tsdf_new1, weight_new1, *pos1);
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过

                if(doDbgPrint)
                    printf("++FUSE_KF_AVGE, tsdf_new1, weight_new1; tsdf_prev2nd, weight_new2nd, (%f, %d), (%f, %d)\n", tsdf_new1, weight_new1, tsdf_prev2nd, weight_new2nd);
            }
            else if(FUSE_FIX_PREDICTION == fuse_method){ //取代粗暴 FUSE_RESET
#if 0   //factor/step 方式不行
//                   //const int pos_neg_factor = 8;
//                   int pos_neg_factor = min(weight_prev1 / 10, 1); //到这里时可能 w1 其实不大, 所以不能粗暴设定大步长
//                   int pnWrk = pos_neg_factor * Wrk;
//                   float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + pnWrk * tsdf) / (weight_prev2nd + pnWrk);
//                   int weight_new2nd = min (weight_prev2nd + pnWrk, Tsdf::MAX_WEIGHT);
// 
//                   int weight_new1 = max(weight_prev1 - pnWrk, 0);
// 
//                   if(weight_new2nd > weight_new1){ //若 2nd 逆袭, 则交换 1st/2nd, 永远保持 1st 为主
#elif 1
                float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + Wrk * tsdf_curr) / (weight_prev2nd + Wrk);
                int weight_new2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);

                //int weight_new1 = max(weight_prev1 - Wrk, 0);
                if(weight_new2nd > w2ndCntThresh){ //交换 1st/2nd, 永远保持 1st 为主 //这里改成: 2nd 不必逆袭 1st, 只要大于某常量阈值即可
#endif
                    if(doDbgPrint){
                        printf("weight_new2nd > w2ndCntThresh,,, exchanging 1st-2nd\n");
                    }
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos1); //new-2nd 放到 pos-1st 中
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos2nd);

                    doUpdateVrayAndSnorm = true; //直到 2nd 逆袭, 才用新的 snorm 更新当前 vxl
                }
                else{ //否则
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos1);
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos2nd);
                    doUpdateVrayAndSnorm = false;
                }

                if(doDbgPrint)
                    //printf("...>>FUSE_FIX_PREDICTION, weight_new1, weight_new2nd, %d, %d\n", weight_new1, weight_new2nd);
                    printf("...>>FUSE_FIX_PREDICTION, tsdf_new2nd, weight_new2nd, %f, %d\n", tsdf_new2nd, weight_new2nd);

                //调试: 不管doDbgPrint, 全部输出, 看究竟有没有走到这一步的 vxl: @2017-3-11 21:22:59
                //答: 有!! 因为 FUSE_FIX_PREDICTION 目前针对 case: tsdf_prev1 < 0 && tsdf_prev1 < tsdf
                //printf("...>>FUSE_FIX_PREDICTION, weight_new2nd, %d,,, [xyz]=(%d, %d, %d)\n", weight_new2nd, x, y, z);
            }
            else if(FUSE_RESET == fuse_method){
                if(doDbgPrint)
                    printf(">>FUSE_RESET\n");

                pack_tsdf(tsdf_curr, 1, *pos1);
            }
            else if(FUSE_IGNORE_CURR == fuse_method){
                if(doDbgPrint)
                    printf("==FUSE_IGNORE_CURR: weight_prev2nd, Wrk: %d, %d\n", weight_prev2nd, Wrk);

                //DO-NOTHING!!! //×
                //IGN时, 也要 2nd 弄一下 @2017-3-16 03:53:08
                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过
            }

            if(doDbgPrint)
                printf("doUpdateSnorm: %d\n", doUpdateVrayAndSnorm);

            if(doUpdateVrayAndSnorm){
                //max (-DIVISOR, min (DIVISOR, (int)nearbyintf (tsdf * DIVISOR))); //@pack_tsdf
                //因为 vray_normed.xyz 必然均 <=1, 所以不必 max/min... ↑
                (*vrayPrev_pos).x = (int)nearbyintf(vray_normed.x * CHAR_MAX); //float2char
                (*vrayPrev_pos).y = (int)nearbyintf(vray_normed.y * CHAR_MAX);
                (*vrayPrev_pos).z = (int)nearbyintf(vray_normed.z * CHAR_MAX);

                //用了 pcc 求 nmap 方法之后, 边缘法向不准 (因为 sobel?), 要切掉; 否则导致一些坑洼 @2017-3-15 16:54:25
                //用 4:=7/2+1
                const int edgeMarg = 4;
                if(coo.x < edgeMarg || coo.x >= depthScaled.cols - edgeMarg || coo.y < edgeMarg || coo.y >= depthScaled.rows - edgeMarg){
                    if(doDbgPrint)
                        printf("+++++++++++++++at edge, dont-update-snorm; coo.xy: (%d, %d)\n", coo.x, coo.y);
                }
                else{
                    //(*snorm_pos).w += 1; //即 snormPrevConfid
                    *snormPrevConfid = min(SCHAR_MAX, *snormPrevConfid + 1);

                    if(!isSnormPrevInit || isNewFace){
                        if(doDbgPrint)
                            printf("\t(!isSnormPrevInit || isNewFace): %d, %d; snormPrevConfid: %d\n", isSnormPrevInit, isNewFace, (*snorm_pos).w);

                        (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                        (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                        (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                    }
                    else{ //isSnormPrevInit && !isNewFace //v11.6: 若snorm 初始化过了, 且当前没有突变, 则用 model 的法向, 因为其更稳定
                        if(doDbgPrint)
                            printf("\tisSnormPrevInit && !isNewFace\n");

                        float3 snorm_model_g;
                        snorm_model_g.x = nmap_model_g.ptr(coo.y)[coo.x];
                        snorm_model_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
                        snorm_model_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

                        float cos_vray_norm_model = dot(snorm_model_g, vray_normed);
                        if(cos_vray_norm_model > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                            //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                            //假设不保证外部已正确预处理：
                            snorm_model_g.x *= -1;
                            snorm_model_g.y *= -1;
                            snorm_model_g.z *= -1;
                        }
                        if(doDbgPrint)
                            printf("\t\tcos_vray_norm_model, %f; snorm_model_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm_model, snorm_model_g.x, snorm_model_g.y, snorm_model_g.z, vray_normed.x, vray_normed.y, vray_normed.z);

                        float cos_norm_model_and_prev = dot(snorm_model_g, snormPrev);
                        //↑--按理说, 此时 n_model, n_curr 应该夹角很小 (已经考虑了与视线做 ±1 乘法) //v11.7   @2017-3-17 15:52:25
                        //但若因为噪声, 导致 n_model 偏差过大, 则完全不更新:
                        //if(cos_norm_model_and_prev > cosThreshSnorm){
                        //if(1){ //发现 snormPrev 不好

                        //zc: 改逻辑: 若 snorm-model/curr 相近才更新 @2017-4-25 21:24:23
                        float cos_norm_model_and_curr = dot(snorm_model_g, snorm_curr_g);
                        if(cos_norm_model_and_curr > cosThreshSnorm){
                            //发现 __float2int_rd 因 round-down 存在退化问题, 数值不稳定, 改用 nearbyintf (搜来的)?  @2017-3-15 15:33:33
                            (*snorm_pos).x = (int)nearbyintf(snorm_model_g.x * CHAR_MAX); //float2char
                            (*snorm_pos).y = (int)nearbyintf(snorm_model_g.y * CHAR_MAX);
                            (*snorm_pos).z = (int)nearbyintf(snorm_model_g.z * CHAR_MAX);
                        }
                        else{
                            //DO-NOTHING!!!
                        }
                    }//if-(isSnormPrevInit && !isNewFace)
                }//cut-edgeMarg

                if(doDbgPrint){
                    printf("newVray: [%d, %d, %d]\n", (*vrayPrev_pos).x, (*vrayPrev_pos).y, (*vrayPrev_pos).z);
                    printf("\tnewSnorm: [%d, %d, %d]\n", (*snorm_pos).x, (*snorm_pos).y, (*snorm_pos).z);
                }
            }//if-(doUpdateVrayAndSnorm)
          }//if- (Dp_scaled != 0 && sdf >= -tranc_dist)
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v11_remake


    __global__ void
    tsdf23_v12 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
          short diff_depth = diff_dmap.ptr(coo.y)[coo.x];
#if 0
          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
#else
          //↓--v11.7: 按 wmap (weight) 动态设定 tranc_dist 上限, (但基准不变:
          //float tranc_dist_real = tranc_dist * weiFactor;
          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8

          if(doDbgPrint){
              printf("\ttranc_dist_real, weiFactor: (%f, %f); diff_depth:= %d\n", tranc_dist_real, weiFactor, diff_depth);
          }

          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          //if (Dp_scaled != 0 && -tranc_dist_real <= sdf && sdf < tranc_dist) //meters, v11.8
#endif
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            bool isInclined = (incidAngleMask.ptr(coo.y)[coo.x] != 0); //太倾斜了, 入射角太大
            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vrayPrev;
            //假设已归一化, 虽然 char->float 有误差, 但不再做归一化
            vrayPrev.x = 1.f * (*vrayPrev_pos).x / CHAR_MAX; //char2float
            vrayPrev.y = 1.f * (*vrayPrev_pos).y / CHAR_MAX;
            vrayPrev.z = 1.f * (*vrayPrev_pos).z / CHAR_MAX;

            //v11.3: 用 vrayPrev_pos[3] 做 hadSeenConfidence, 取代 hadSeen 布尔量: //2017-3-11 21:40:24
            signed char *seenConfid = &vrayPrev_pos->w;
            const int seenConfidTh = 15;

            float3 vray; //这次不用视线做主要判断, 此处只是用来测试 nmap 传参对错
                            //v11.2 改成都要做: 视线 & 表面法向双重判定 //2017-3-8 22:00:32
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snormPrev;
            snormPrev.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snormPrev.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snormPrev.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //v11.9: 有时候 snorm 被噪声错误地初始化, 真实值却难以再去修正 snorm @2017-4-11 17:03:51
            signed char *snormPrevConfid = &snorm_pos->w;
            const int snormPrevConfid_thresh = 5;

            //const bool hadSeen = *flag_pos; //别名 hadSeen, 不准确
            const bool hadSeen = (*seenConfid > seenConfidTh); //v11.3: 策略, 当连续 confid++, 达到阈值之后, 才标记 seen; 若达不到阈值, 还要--

            //bool isSnormPrevInit = (norm(snormPrev) > 1e-8);
            //bool isSnormPrevInit = ( (norm(snormPrev) > 1e-8) && (snormPrevConfid > snormPrevConfid_thresh) );
            bool isSnormPrevInit = (*snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零

            if(doDbgPrint){
                printf("isInclined, %d\n", isInclined);
                printf("cos_vray_norm, %f; snorm_curr_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("(norm(snormPrev) == 0) == %s; (norm(snormPrev) < 1e-8) == %s\n",
                    norm(snormPrev) == 0 ? "T" : "F",
                    norm(snormPrev) < 1e-8 ? "T" : "F");
            }


            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);

            float tsdf_prev2nd = -123;
            int weight_prev2nd = -233;
            unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

            //const int w2ndCntThresh = 10; //w2nd 超过此阈值才能逆袭
            const int w2ndCntThresh = 10 * 10; //v11.4 用 weiFactor 之后

            if(doDbgPrint){
                printf("tsdf_prev: tsdf1st: %f, %d; tsdf2nd: %f, %d;\n", tsdf_prev1, weight_prev1, tsdf_prev2nd, weight_prev2nd);
            }

            int fuse_method = FUSE_KF_AVGE; //默认原策略
            bool doUpdateVrayAndSnorm = false;

            const float cosThreshVray = //0.8660254f; //cos(30°)
                //0.9396926f; //cos(20°) //当 largeIncidMask 取 80 阈值时, 此处应为 (90-x)*2
                0.9659258f; //cos(15°) //因为largeIncidMask 以 75°为阈值, 所以这里用 90-75=15 为阈值
                //0.996194698; //cos(5°)
            const float cosThreshSnorm = 0.8660254f; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_norm = dot(snormPrev, snorm_curr_g);
            float cos_vray = dot(vrayPrev, vray_normed);
            bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm && cos_vray < cosThreshVray); //snorm-init 之后才做 newFace 判定 @2017-4-21 00:42:00
            //bool isNewFace = (isSnormPrevInit && cos_norm < cosThreshSnorm); //去掉 vray 判定, 别! 原因: vray 防止 *视角稳定但snorm 突变 (边缘etc.)* 情形, 不轻易 isNewFace=true

            //zc: 增加判定, 若 weight-factor 太小(如, 边缘区域), 则直接均值, 且不 updateVray @2017-7-13 22:29:39
            if(weiFactor > 0.2){


            if(doDbgPrint){
                printf("cos_norm: snormPrev, snorm_curr_g, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_norm, 
                    snormPrev.x, snormPrev.y, snormPrev.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);
                printf("\tcos_vray, vrayPrev, vray_normed, %f, [%f, %f, %f], [%f, %f, %f]\n", cos_vray, 
                    vrayPrev.x, vrayPrev.y, vrayPrev.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("%s, snormPrevConfid, snormPrevConfid_thresh: %d, %d\n", isNewFace ? "isNewFace-T" : "isNewFace-F", *snormPrevConfid, snormPrevConfid_thresh);
                printf("\t%s\n", cos_norm > cosThreshSnorm ? "cos_norm > cosThreshSnorm" : "cos_norm <= cosThreshSnorm");
                printf("\t%s\n", cos_vray > cosThreshVray ? "cos_vray > cosThreshVray" : "cos_vray <= cosThreshVray");
            }


            if(isInclined){ //若边缘, doUpdateVray 保持 false
                if(!hadSeen){ //若 seen-flag 未初始化过
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=F; ++FUSE_KF_AVGE\n");
                    fuse_method = FUSE_KF_AVGE;

                    //*seenConfid = max(0, *seenConfid - 1);
                    //↑-- 不要 -1 了, 只增不减, 但同时 seenConfidTh 阈值调高 (5 -> 15), 延缓其 flag=true   @2017-3-23 11:11:55
                }
                else{ //if(hadSeen) //若之前 seen
#if 0   //忘了 sdf < 0 这个判定为什么了, 目前感觉会导致有偏差, 放弃   @2017-3-9 15:06:22
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; %s; sdf: %f\n", sdf<0 ? "==FUSE_IGNORE_CURR" : "++FUSE_KF_AVGE", sdf);
                    if(sdf < 0)
                        fuse_method = FUSE_IGNORE_CURR;
                    else
                        fuse_method = FUSE_KF_AVGE;
#elif 1 //一律 ignore
                    if(doDbgPrint)
                        printf("isInclined-T; hadSeen=T; \n");
                    fuse_method = FUSE_IGNORE_CURR;
#endif
                }
            }
            else{ //if(!isInclined){ //若非边缘, 在内部
                //*seenConfid = min(Tsdf::MAX_WEIGHT, *seenConfid + 1); //v11.4 用 weiFactor 之后, 这里反而是 BUG!!
                *seenConfid = min(SCHAR_MAX, *seenConfid + 1);

                if(!isSnormPrevInit){ //vray.prev 若未初始化, 用 < epsilon 判定
                    //if (*seenConfid > seenConfidTh) //这就是 hadSeen, 所以不要这么判定
                        //doUpdateVrayAndSnorm = true;
                }


                if(!hadSeen){ //若 seen-flag 未初始化过
#if 0   //< v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; >>FUSE_RESET\n");
                    *flag_pos = true;
                    fuse_method = FUSE_RESET;
#elif 1 //v11.3
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=F; seenConfid, seenConfidTh: %d, %d, ++FUSE_KF_AVGE~~~~~\n", *seenConfid, seenConfidTh); //别处也没有 reset 了
                    fuse_method = FUSE_KF_AVGE;
#endif
                    //if (*seenConfid > seenConfidTh) //既然 hadSeen 逻辑改过, 则此处必然一直 false
                    //    doUpdateVrayAndSnorm = true;
                }
                else{ //if(hadSeen) //若之前 seen, 必然经过过 【isInclined-F; hadSeen=F】阶段, 也必然 isSnormPrevInit->true, 不必再 if-isSnormPrevInit
                    if(doDbgPrint)
                        printf("isInclined-F; hadSeen=T;\n");

                    //if(cos_norm > cosThresh ){ //夹角角度 <30°, 算作同视角
                    if(!isNewFace){ //同视角, 双 cos 联合判定
                        //TODO...
                        fuse_method = FUSE_KF_AVGE; //其实默认

                        //if (*seenConfid > seenConfidTh) //不必, 因为已在 if-hadSeen 分支内
#if 0
                        if(cos_norm > cosThreshSnorm) //反之 cos_norm < th 时, 尽管 newFace=false, 但不应 update
                            doUpdateVrayAndSnorm = true;

                        if(!isSnormPrevInit)
                            doUpdateVrayAndSnorm = true;
#elif 1 //改成必然 update @2017-7-13 15:45:12
                        doUpdateVrayAndSnorm = true;
#endif
                    }
                    else{ // >30°, 算作不同视角, 比如转过头之后
                        //if(!isSnormPrevInit) //newFace 改进之后, 这里不会再进入
                        //    doUpdateVrayAndSnorm = true;

#if 10   //三类不完善, 而且语义不明确, 放弃 @2017-3-24 17:50:24
                        //化简为三类
                        if(tsdf_curr < 0 && tsdf_curr < tsdf_prev1){
                            if(doDbgPrint)
                                printf("\ttsdf < 0 && tsdf < tsdf_prev1; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);

                            fuse_method = FUSE_IGNORE_CURR;
                        }
                        else if(tsdf_prev1 < 0 && tsdf_prev1 < tsdf_curr){
                            if(doDbgPrint){
                                printf("\ttsdf_prev1 < 0 && tsdf_prev1 < tsdf; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                                printf("\t\t, weight_prev1, w2ndCntThresh: %d, %d\n", weight_prev1, w2ndCntThresh);
                            }
#if 0   //weight_prev1 是否要与 w2ndCntThresh 对比?
                            if(weight_prev1 > w2ndCntThresh){
                                fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
                            }
                            else{
                                fuse_method = FUSE_KF_AVGE; //这里默认是否有问题
                            }
#elif 0 //1st 不与 w2ndCntThresh 对比, 因为下面做对比控制: weight_new2nd > w2ndCntThresh
                            fuse_method = FUSE_FIX_PREDICTION; //用备用 volume, 缓慢-→+
#elif 0 //v12.1 改成: 正冲负时, 判断 diff_depth @2017-12-3 22:29:04
                            //不好, 没啥用 @2017-12-4 02:59:18
                            if(tsdf_curr <= 0){ //同负
                                fuse_method = FUSE_FIX_PREDICTION;
                            }
                            else{ //if(tsdf_curr > 0) //正冲负
                                if(diff_depth > 30) //diff足够大, 才允许FIX; 否则用AVG; //不存在 diff<0
                                    fuse_method = FUSE_FIX_PREDICTION;
                                else
                                    fuse_method = FUSE_KF_AVGE; //其实默认
                            }
#elif 1 //v12.2 薄片损坏, 是因正冲负时, 内部变成正, 内外都是正, 所以找不到过零点
                            //此策略思路: 两外壁(正)之间只有一个vox为负时, 若此vox新值为正, 则不要正冲负; 或许对 vox较大时比较适用; 仍不好 @2017-12-10 22:29:45

                            if(tsdf_curr < 0) //同负, 总是 FIX
                                fuse_method = FUSE_FIX_PREDICTION; 
                            else{
                                //以下处理 正冲负 情形:
                                int grid_dx, grid_dy, grid_dz;
                                grid_dx = grid_dy = grid_dz = 0;

                                //此阈值, 判断 vray 落在周围 27(实际26) 晶格的哪一个; 
                                //因不会都< sqrt(1/3), 故不用担心 dxyz=000
                                const float vray_which_grid_thresh = 0.577350269; //sqrt(1/3)

                                if(vray_normed.x > vray_which_grid_thresh)
                                    grid_dx = 1;
                                else if(vray_normed.x < -vray_which_grid_thresh)
                                    grid_dx = -1;
                                //else grid_dx = 0; //默认

                                if(vray_normed.y > vray_which_grid_thresh)
                                    grid_dy = 1;
                                else if(vray_normed.y < -vray_which_grid_thresh)
                                    grid_dy = -1;

                                if(vray_normed.z > vray_which_grid_thresh)
                                    grid_dz = 1;
                                else if(vray_normed.z < -vray_which_grid_thresh)
                                    grid_dz = -1;

                                int nbr_x, nbr_y, nbr_z;
                                nbr_x = min(VOLUME_X-1, max(0, x+grid_dx));
                                nbr_y = min(VOLUME_Y-1, max(0, y+grid_dy));
                                nbr_z = min(VOLUME_Z-1, max(0, z+grid_dz));

                                //volume1 中, 沿视线方向, 当前 vox 的邻接(nbr) vox:
                                short2 *nbr_pos1 = volume1.ptr(nbr_y) + nbr_x;
                                nbr_pos1 += nbr_z * elem_step;

                                float nbr_tsdf_prev1;
                                int nbr_weight_prev1;
                                unpack_tsdf(*nbr_pos1, nbr_tsdf_prev1, nbr_weight_prev1);

                                char4 *nbr_vrayPrev_pos = vrayPrevVolume.ptr(nbr_y) + nbr_x;
                                //int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);
                                nbr_vrayPrev_pos += nbr_z * vrayPrev_elem_step;

                                float3 nbr_vrayPrev;

                                //假设已归一化, 虽然 char->float 有误差, 但不再做归一化
                                nbr_vrayPrev.x = 1.f * (*nbr_vrayPrev_pos).x / CHAR_MAX; //char2float
                                nbr_vrayPrev.y = 1.f * (*nbr_vrayPrev_pos).y / CHAR_MAX;
                                nbr_vrayPrev.z = 1.f * (*nbr_vrayPrev_pos).z / CHAR_MAX;

                                float cos_vrayCurr_nbrPrev = dot(nbr_vrayPrev, vray_normed);

                                if(nbr_tsdf_prev1 < 0)
                                    fuse_method = FUSE_FIX_PREDICTION;
                                else{ //if(nbr_tsdf_prev1 >= 0) 
                                    if(cos_vrayCurr_nbrPrev >= 0)
                                        fuse_method = FUSE_FIX_PREDICTION;
                                    else //if(cos_vrayCurr_nbrPrev < 0) //此时不要 FIX, 以免表面两侧 tsdf 同正
                                        fuse_method = FUSE_KF_AVGE; 
                                }
                            }//if-(tsdf_curr >= 0)
#elif 1 //v12.3 //思路:沿视线, 反向退步到vox2, 看 vox2 是否稳定
                            //几点共识: 
                            //1, 正冲负时, 总是 diff>0
                            //2, 不稳定区域, 该冲; 稳定区域, 不该冲


#endif
                            //doUpdateSnorm = true; //放到 FUSE_FIX_PREDICTION 里判定
                        }
                        else if(tsdf_curr >=0 && tsdf_prev1 >= 0){
                            if(doDbgPrint){
                                printf("\ttsdf >=0 && tsdf_prev1 >= 0; [:=prev1, curr: %f, %f\n", tsdf_prev1, tsdf_curr);
                            }
                            fuse_method = FUSE_KF_AVGE;

                            if(cos_norm > 0) //加约束: 法向突变不能超90°; 是为了防止薄片区域, 因原始深度图噪声, 导致旧的背面法向更新;  @2017-11-17 15:39:06
                                doUpdateVrayAndSnorm = true;
                        }
#endif
                    }//cos vs. cosTh
                }//if-hadSeen
            }//if-isInclined
            }//if-(weiFactor > 0.2)

            const int Wrk = max(15 * weiFactor, 1.f);
            if(FUSE_KF_AVGE == fuse_method){
                float tsdf_new1 = (tsdf_prev1 * weight_prev1 + Wrk * tsdf_curr) / (weight_prev1 + Wrk);
                int weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0

                pack_tsdf (tsdf_new1, weight_new1, *pos1);
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过

                if(doDbgPrint)
                    printf("++FUSE_KF_AVGE, tsdf_new1, weight_new1; tsdf_prev2nd, weight_new2nd, (%f, %d), (%f, %d)\n", tsdf_new1, weight_new1, tsdf_prev2nd, weight_new2nd);
            }
            else if(FUSE_FIX_PREDICTION == fuse_method){ //取代粗暴 FUSE_RESET
#if 0   //factor/step 方式不行
//                   //const int pos_neg_factor = 8;
//                   int pos_neg_factor = min(weight_prev1 / 10, 1); //到这里时可能 w1 其实不大, 所以不能粗暴设定大步长
//                   int pnWrk = pos_neg_factor * Wrk;
//                   float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + pnWrk * tsdf) / (weight_prev2nd + pnWrk);
//                   int weight_new2nd = min (weight_prev2nd + pnWrk, Tsdf::MAX_WEIGHT);
// 
//                   int weight_new1 = max(weight_prev1 - pnWrk, 0);
// 
//                   if(weight_new2nd > weight_new1){ //若 2nd 逆袭, 则交换 1st/2nd, 永远保持 1st 为主
#elif 1
                float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + Wrk * tsdf_curr) / (weight_prev2nd + Wrk);
                int weight_new2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);

                //int weight_new1 = max(weight_prev1 - Wrk, 0);
                //if(weight_new2nd > w2ndCntThresh){ //交换 1st/2nd, 永远保持 1st 为主 //这里改成: 2nd 不必逆袭 1st, 只要大于某常量阈值即可
                if(weight_new2nd > weight_prev1 / 2){ //怎么评估 w2 的稳定度? 不再用常量阈值, 改用 w1/2 (仍是经验性尝试), 用意: 若 w1 很稳定, 则 w2 逆袭慢(难)点 @2017-12-10 22:42:57
#endif
                    if(doDbgPrint){
                        printf("weight_new2nd > w2ndCntThresh,,, exchanging 1st-2nd\n");
                    }
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos1); //new-2nd 放到 pos-1st 中
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos2nd);

                    doUpdateVrayAndSnorm = true; //直到 2nd 逆袭, 才用新的 snorm 更新当前 vxl
                }
                else{ //否则
                    //pack_tsdf(tsdf_prev1, weight_new1, *pos1);
                    pack_tsdf(tsdf_new2nd, weight_new2nd, *pos2nd);
                    doUpdateVrayAndSnorm = false;
                }

                if(doDbgPrint)
                    //printf("...>>FUSE_FIX_PREDICTION, weight_new1, weight_new2nd, %d, %d\n", weight_new1, weight_new2nd);
                    printf("...>>FUSE_FIX_PREDICTION, tsdf_new2nd, weight_new2nd, (%f, %d); tprev1, wprev1: (%f, %d)\n", tsdf_new2nd, weight_new2nd, tsdf_prev1, weight_prev1);

                //调试: 不管doDbgPrint, 全部输出, 看究竟有没有走到这一步的 vxl: @2017-3-11 21:22:59
                //答: 有!! 因为 FUSE_FIX_PREDICTION 目前针对 case: tsdf_prev1 < 0 && tsdf_prev1 < tsdf
                //printf("...>>FUSE_FIX_PREDICTION, weight_new2nd, %d,,, [xyz]=(%d, %d, %d)\n", weight_new2nd, x, y, z);
            }
            else if(FUSE_RESET == fuse_method){
                if(doDbgPrint)
                    printf(">>FUSE_RESET\n");

                pack_tsdf(tsdf_curr, 1, *pos1);
            }
            else if(FUSE_IGNORE_CURR == fuse_method){
                if(doDbgPrint)
                    printf("==FUSE_IGNORE_CURR: weight_prev2nd, Wrk: %d, %d\n", weight_prev2nd, Wrk);

                //DO-NOTHING!!! //×
                //IGN时, 也要 2nd 弄一下 @2017-3-16 03:53:08
                int weight_new2nd = max(weight_prev2nd - Wrk, 0); //--, 但防止 <0
                pack_tsdf(tsdf_prev2nd, weight_new2nd, *pos2nd); //不管 2nd 是否真正初始化过
            }

            if(doDbgPrint)
                printf("doUpdateSnorm: %d\n", doUpdateVrayAndSnorm);

            if(doUpdateVrayAndSnorm){
                //max (-DIVISOR, min (DIVISOR, (int)nearbyintf (tsdf * DIVISOR))); //@pack_tsdf
                //因为 vray_normed.xyz 必然均 <=1, 所以不必 max/min... ↑
                (*vrayPrev_pos).x = (int)nearbyintf(vray_normed.x * CHAR_MAX); //float2char
                (*vrayPrev_pos).y = (int)nearbyintf(vray_normed.y * CHAR_MAX);
                (*vrayPrev_pos).z = (int)nearbyintf(vray_normed.z * CHAR_MAX);

                //用了 pcc 求 nmap 方法之后, 边缘法向不准 (因为 sobel?), 要切掉; 否则导致一些坑洼 @2017-3-15 16:54:25
                //用 4:=7/2+1
                const int edgeMarg = 4;
                if(coo.x < edgeMarg || coo.x >= depthScaled.cols - edgeMarg || coo.y < edgeMarg || coo.y >= depthScaled.rows - edgeMarg){
                    if(doDbgPrint)
                        printf("+++++++++++++++at edge, dont-update-snorm; coo.xy: (%d, %d)\n", coo.x, coo.y);
                }
                else{
                    //(*snorm_pos).w += 1; //即 snormPrevConfid
                    *snormPrevConfid = min(SCHAR_MAX, *snormPrevConfid + 1);

                    if(!isSnormPrevInit || isNewFace){
                        if(doDbgPrint)
                            printf("\t(!isSnormPrevInit || isNewFace): %d, %d; snormPrevConfid: %d\n", isSnormPrevInit, isNewFace, (*snorm_pos).w);

                        (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                        (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                        (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                    }
                    else{ //isSnormPrevInit && !isNewFace //v11.6: 若snorm 初始化过了, 且当前没有突变, 则用 model 的法向, 因为其更稳定
                        if(doDbgPrint)
                            printf("\tisSnormPrevInit && !isNewFace\n");

                        float3 snorm_model_g;
                        snorm_model_g.x = nmap_model_g.ptr(coo.y)[coo.x];
                        snorm_model_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
                        snorm_model_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

                        float cos_vray_norm_model = dot(snorm_model_g, vray_normed);
                        if(cos_vray_norm_model > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                            //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                            //假设不保证外部已正确预处理：
                            snorm_model_g.x *= -1;
                            snorm_model_g.y *= -1;
                            snorm_model_g.z *= -1;
                        }
                        if(doDbgPrint)
                            printf("\t\tcos_vray_norm_model, %f; snorm_model_g: [%f, %f, %f], vray_normed: [%f, %f, %f]\n", cos_vray_norm_model, snorm_model_g.x, snorm_model_g.y, snorm_model_g.z, vray_normed.x, vray_normed.y, vray_normed.z);

                        float cos_norm_model_and_prev = dot(snorm_model_g, snormPrev);
                        //↑--按理说, 此时 n_model, n_curr 应该夹角很小 (已经考虑了与视线做 ±1 乘法) //v11.7   @2017-3-17 15:52:25
                        //但若因为噪声, 导致 n_model 偏差过大, 则完全不更新:
                        //if(cos_norm_model_and_prev > cosThreshSnorm){
                        //if(1){ //发现 snormPrev 不好

                        //zc: 改逻辑: 若 snorm-model/curr 相近才更新 @2017-4-25 21:24:23
                        float cos_norm_model_and_curr = dot(snorm_model_g, snorm_curr_g);
                        if(cos_norm_model_and_curr > cosThreshSnorm){
                            //发现 __float2int_rd 因 round-down 存在退化问题, 数值不稳定, 改用 nearbyintf (搜来的)?  @2017-3-15 15:33:33
                            (*snorm_pos).x = (int)nearbyintf(snorm_model_g.x * CHAR_MAX); //float2char
                            (*snorm_pos).y = (int)nearbyintf(snorm_model_g.y * CHAR_MAX);
                            (*snorm_pos).z = (int)nearbyintf(snorm_model_g.z * CHAR_MAX);
                        }
                        else{
                            //DO-NOTHING!!!
                        }
                    }//if-(isSnormPrevInit && !isNewFace)
                }//cut-edgeMarg

                if(doDbgPrint){
                    printf("newVray: [%d, %d, %d]\n", (*vrayPrev_pos).x, (*vrayPrev_pos).y, (*vrayPrev_pos).z);
                    printf("\tnewSnorm: [%d, %d, %d]\n", (*snorm_pos).x, (*snorm_pos).y, (*snorm_pos).z);
                }
            }//if-(doUpdateVrayAndSnorm)
          }//if- (Dp_scaled != 0 && sdf >= -tranc_dist)
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v12

    enum{   //v13.2
        SAME_SIDE_VIEW
        ,OPPOSITE_VIEW
        ,GRAZING_VIEW   //暂定: 不融合 @2017-12-22 14:44:08
        ,GRAZING_VIEW_POS
        ,GRAZING_VIEW_NEG
    };

    enum{
        WEIGHT_RESET_FLAG = -1
        ,WEIGHT_SCALE = 10 //尝试 w float 策略时, w<1 会被 int 截断, 在 unpack 后/pack 前 添加 scale, 避免中间运算时 int截断导致出错

        ,TDIST_MIN_MM = 5 //5mm
        ,TDIST_MAX_MM = 25 //25mm
    };
#define SLIGHT_POSITIVE 1e-2

    //参数暂时等同 v12, host 主调函数暂借用 v12 的    @2018-1-5 16:30:48
    __global__ void
    tsdf23_v13 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];

          //zc: 相比v11, 暂放弃 tranc_dist_real 控制, 试试看 @2017-12-13 10:54:29
          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          
          //↓--又用回 tranc_dist_real; 效果不错, 比单纯操作权重好, 边缘只能这样处理? @2017-12-29 10:58:14
          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          if(doDbgPrint) printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);

          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          {
            //几点改进:
            //1, tsdf=sdf_normed, 直接用 sdf 值, 不再 fmin (1.0f, sdf_normed);
            //2, snorm 更新机制: curr & prev_model 谁 abs-tsdf 小, 用谁的 norm?  //是最后更新？
            //3, 异同视角判定机制: 【放弃】vray, 只用 snorm; 暂定仍用压缩版 char4; c&prev snorm-angle >30°
            //  ↑--仍然用了 vray-snorm_p 夹角作为异同视角判定指标
            //4, 优先判断异同视角
            //5, FIX 策略不要用 volume2nd 影子, 直接用大权重
            //6, 不再用 wmap, incidMask, 平时等权重, 追求普通表面上光滑

            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);
            //float tsdf_curr = sdf_normed; //撤回原定义: tsdf 仍是截断, 凡用不截断的计算, 都用 sdf_normed @2017-12-25 01:53:06

            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vray;
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snorm_prev_g;
            snorm_prev_g.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snorm_prev_g.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snorm_prev_g.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //v11.9: 有时候 snorm 被噪声错误地初始化, 真实值却难以再去修正 snorm @2017-4-11 17:03:51
            signed char *snormPrevConfid = &snorm_pos->w;
            const int snormPrevConfid_thresh = 5;

            //bool isSnormPrevInit = (norm(snormPrev) > 1e-8);
            //bool isSnormPrevInit = ( (norm(snormPrev) > 1e-8) && (snormPrevConfid > snormPrevConfid_thresh) );
            bool isSnormPrevInit = (*snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零

            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);

            int fuse_method = FUSE_KF_AVGE; //默认原策略
            bool doUpdateVrayAndSnorm = false;

            const float COS30 = 0.8660254f
                       ,COS45 = 0.7071f
                       ,COS60 = 0.5f
                       ,COS75 = 0.258819f
                       ;
            const float cosThreshSnorm = COS30; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_snorm_p_c = dot(snorm_prev_g, snorm_curr_g);
            //bool isNewFace = (isSnormPrevInit && cos_snorm_p_c < cosThreshSnorm /*&& cos_vray < cosThreshVray*/); //snorm-init 之后才做 newFace 判定 @2017-4-21 00:42:00
            //↑--去掉 vray 判定
            //↑↑--不好, 若边缘起始时法向不稳定, 后续都无法更正 @2017-12-20 09:38:00
            float cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);
            //bool isNewFace = (isSnormPrevInit && cos_vray_norm_prev > 0); //因之前snorm校正操作, 所以认为: 同面视角下, cos(vray, n_p)<0
            int view_case = SAME_SIDE_VIEW; //尝试取代 isNewFace @2017-12-22 10:58:03
            if(isSnormPrevInit){ //若尚未 snorm-init, 仍算作默认 same-view
                if(abs(cos_vray_norm_prev) < COS75){ //斜视判定
                    view_case = GRAZING_VIEW; //v13.3: 【DEPRECATED】 坏, 若 p在边缘导致法向-视线夹角很大, 则无法用 c修正; 初始错,之后对,如何修复?

                    //if(abs(cos_vray_norm_curr) < COS75) //v13.3.2: 必须当前帧也较斜视, 否则保持 same-side 规则 【DEPRECATED】
                    //    view_case = GRAZING_VIEW;

                    //v13.4: 退回到 vray 只与 snorm-prev 比较, 但 启用 pos-neg-graz 两种策略分开处理, 且融合策略改为: 
                    //1. 若 p>0:: 且若: ① snorm-confid== MAX, 则无论 c正负, 都忽略 (wc=0); ② else: wc=1 融合;    snorm 均不更新
                    //2. 若 p<0, 且若 ① |p| > cellSz/tdist * α 【e.g.: 600mm/256=2.34mm, 再/25=0.09375 是归一化的晶格vox尺度; α是经验系数, 暂定 3, 即要 |p|>3晶格】
                    //                  则重置 vox: snorm=0, confid=0, tsdf=SLIGHT_POSITIVE(微>0, 为了有过零点, 提取表面; 但又很要小, 以便更容易被后来帧修正)
                    //             若 ② else, 则 c忽略
                    
                    //【仍用 GRAZING_VIEW, 不用 ENUM pos-neg-graz, 而在代码段内, 用 cos><0 & p><0 做判断】 @2017-12-24 23:53:48
                    //if(cos_vray_norm_prev < 0)
                    //    view_case = GRAZING_VIEW_POS;
                    //else
                    //    view_case = GRAZING_VIEW_NEG;
                }
                else if(cos_vray_norm_prev < -COS75){ //同面正视
                    view_case = SAME_SIDE_VIEW;
                }
                else{ //if(cos_vray_norm_prev > COS75) //背面正视
                    view_case = OPPOSITE_VIEW;
                }

            }


            if(doDbgPrint){
                printf("vray_normed: [%f, %f, %f]; cos_vray_norm_prev, %f; cos_vray_norm_curr, %f (%s, ALWAYS cos<0)\n", 
                    vray_normed.x, vray_normed.y, vray_normed.z, cos_vray_norm_prev, cos_vray_norm_curr, cos_vray_norm_curr>0? "×":"√");
                //这里打印 snorm 校正之前的 cos-vray-snorm_c (校正之后必然 cos <0 了); snorm 却是校正之后的 @2017-12-20 10:43:19
                printf("cos_snorm_p_c: %f ---snorm_prev_g, snorm_curr_g: [%f, %f, %f], [%f, %f, %f]\n", 
                    cos_snorm_p_c, snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);

                printf("isSnormPrevInit: %s, --snormPrevConfid: %d\n", 
                    isSnormPrevInit ? "TRUE":"FALSE", *snormPrevConfid);

                //printf("%s isNewFace:::", isNewFace? "YES":"NOT");
                printf("%s", view_case==SAME_SIDE_VIEW ? "SAME-SIDE" : (view_case==GRAZING_VIEW ? "GRAZING" : "OPPO-SIDE") );
                printf("::: tsdf_prev1, tsdf_curr: %f, %f\n", tsdf_prev1, tsdf_curr);
            }

            //1, weighting 策略
            //float weight_curr = 1; //AVG, FIX, IGN, 都放弃, 用权重决定一切 @2017-12-14 10:53:54
            float weight_curr = 0; //改用 view_case 控制量之后, 默认权重置零
            float tsdf_new1 = SLIGHT_POSITIVE; //存更新后的 tsdf & w
            int weight_new = WEIGHT_RESET_FLAG;
            bool grazing_reset = false;

            //if(!isNewFace){ //同视角, 
            if(SAME_SIDE_VIEW == view_case){
                /*
                //【DEPRECATED】
                if(tsdf_curr < 0 && tsdf_prev1 >= 0){ //负冲正
                    //可能: 1, curr 变浅; 2, prev很大, 如观测到远处表面正面, 或之前因为某些边缘全反射, 误判为远处表面, 需要修正
                    if(tsdf_prev1 > 1)
                        fuse_method = FUSE_FIX_PREDICTION;
                    //else //默认 AVG
                        //fuse_method = FUSE_KF_AVGE;
                }
                else if(tsdf_curr < 0 && tsdf_prev1 < 0){ //负冲负
                    //默认 AVG
                }
                else if(tsdf_curr >= 0 && tsdf_prev1 >= 0){ //正冲正
                    //默认 AVG
                }
                else{ //if(tsdf_curr >=0 && tsdf_prev1 < 0) //正冲负

                }
                */

                /*
                //【DEPRECATED】
                if(tsdf_prev1 >= 0){ //prev正
                    //weight_curr = min(1, tsdf_prev1 / min(1, abs(tsdf_curr)) ); //错, 不是 min
                    //weight_curr = max(1, tsdf_prev1 / max(1, abs(tsdf_curr)) ); //分母别复杂, 等价简化
                    weight_curr = max(1.f, min(tsdf_prev1, tsdf_prev1 / abs(tsdf_curr)) ); //

                    //↑-ReLU, 是取max; 还有 LReLU, 见: 
                    //http://blog.csdn.net/mao_xiao_feng/article/details/53242235?locationNum=9&fps=1
                }
                else{ //if-(tsdf_prev1 < 0)
                    //w_curr 默认=1
                    //若 c(+)>>p(-), 其实
                }
                */

                weight_curr = (abs(tsdf_prev1)<=1 && abs(tsdf_curr)<=1) ? 1 : abs(tsdf_prev1 / tsdf_curr); //若 tsdf 没用 sdf, 则此处恒等于1, 又 pack 时候强制归一化, 所以此处无效
                weight_curr = weight_curr * weight_curr;
            }
#if 0   //v13.4: GRAZING_VIEW, not graz-pos-neg
            else if(GRAZING_VIEW == view_case){//略复杂
                if(*snormPrevConfid > Tsdf::MAX_WEIGHT_V13 / 2.f){
                    weight_curr = 0;
                }
                else{//snormPrevConfid 较小
                    if(tsdf_prev1 > 0){
                        weight_curr = 1;
                    }
                    else{ //tsdf_prev1 <0
                        float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //暂定 3*vox 厚度
                        if(doDbgPrint){
                            printf("GRAZING_VIEW-snormPrevConfid(<Th)-tsdf_prev1(<0)-pendingFixThresh: %f\n", pendingFixThresh);
                            printf("\ttsdf_prev1 > -pendingFixThresh: %s; sdf_normed: %f\n", tsdf_prev1 > -pendingFixThresh ? "TTT" : "FFF", sdf_normed);
                        }

                        if(tsdf_prev1 > -pendingFixThresh){
                            weight_curr = 0;
                        }
                        else{//tsdf_prev1 < -pendingFixThresh
                            if(sdf_normed > 1){
                                *snormPrevConfid = 0;
                                //snorm=0？ 暂不管, 因为 confid 重置, 后面自然正常处理 snorm?

                                grazing_reset = true;
                                //tsdf_new1 = 0.1f; //放后面, 放这里会被后面覆盖坏掉
                                //weight_new = 0;
                            }
                        }
                    }

                }
            }
#elif 0 //v13.5: GRAZ 判定优先级: ① c≈1; if-c>>1, 则② graz-pos-neg; ③ ???
            //用 sdf-normed 算, 但融合用 tsdf
            else if(GRAZING_VIEW == view_case){//略复杂
                if(doDbgPrint){
                    printf("GRAZING_VIEW--sdf_normed=%f (%s)--cos_V_N_p=%f (%s)\
                           --tsdf_prev1=%f (%s)-pendingFixThresh: %f\n", 
                        sdf_normed, sdf_normed > 1 ? ">1" : "<=1", 
                        cos_vray_norm_prev, cos_vray_norm_prev < 0 ? "<0" : ">=0",
                        tsdf_prev1, tsdf_prev1 > 0 ? ">0" : "<=0",
                        pendingFixThresh);
                    printf("\tabs(tsdf_prev1) < pendingFixThresh: %s;\n", \
                        abs(tsdf_prev1) < pendingFixThresh ? "TTT" : "FFF");
                }
                if(sdf_normed <= 1){ //其实也即 |..|<=1, 因为本来有 ..>= -1
                    weight_curr = 1;
                }
                else{//sdf_normed > 1
                    if(cos_vray_norm_prev < 0){ //即之前废弃的 GRAZING_VIEW_POS
                        weight_curr = 0; //例: 1, 侧面全反射, 导致远面向内延长出部分有效px
                    }
                    else{ //cos_vray_norm_prev > 0, 即 graz-neg
                        if(tsdf_prev1 > 0){
                            weight_curr = 0;
                        }
                        else{ //tsdf_prev1 <0
                            //if(doDbgPrint){
                            //    printf("GRAZING_VIEW-(sdf_normed > 1)-(cos_vray_norm_prev < 0)-(tsdf_prev1 <0)-pendingFixThresh: %f\n", pendingFixThresh);
                            //    printf("\ttsdf_prev1 > -pendingFixThresh: %s; sdf_normed: %f\n", tsdf_prev1 > -pendingFixThresh ? "TTT" : "FFF", sdf_normed);
                            //}

                            if(tsdf_prev1 > -pendingFixThresh)
                                weight_curr = 0;
                            else{//tsdf_prev1 < -pendingFixThresh
                                //if(sdf_normed > 1){ //早已判定过
                                *snormPrevConfid = 0;
                                //snorm=0？ 暂不管, 因为 confid 重置, 后面自然正常处理 snorm?

                                grazing_reset = true;
                                //tsdf_new1 = 0.1f; //放后面, 放这里会被后面覆盖坏掉
                                //weight_new = 0;
                            }
                        }
                    }
                }
            }
#elif 1 //v13.6: 简化 v13.5 逻辑:: ① |sdf|<1, AVG(w=0); ②某多项约束 RESET(confid=0); ③else IGN(w=0);
            else if(GRAZING_VIEW == view_case){//略复杂
                weight_curr = 0; //逻辑块内首先全部置零

                if(doDbgPrint){
                    printf("GRAZING_VIEW--sdf_normed=%f (%s)--cos_V_N_p=%f (%s)"
                           "--tsdf_prev1=%f (%s)-pendingFixThresh: %f\n", 
                           sdf_normed, sdf_normed > 1 ? ">1" : "<=1", 
                           cos_vray_norm_prev, cos_vray_norm_prev < 0 ? "<0" : ">=0",
                           tsdf_prev1, tsdf_prev1 > 0 ? ">0" : "<=0",
                           pendingFixThresh);
                    printf("\tabs(tsdf_prev1) < pendingFixThresh: %s;\n", \
                        abs(tsdf_prev1) < pendingFixThresh ? "TTT" : "FFF");
                }
                if(sdf_normed <= 1){ //其实也即 |..|<=1, 因为本来有 ..>= -1; 此逻辑块内 sdf==tsdf
#if 0   //v13.6 仅仅 |sdf|<1 就 wc=1 不好; 举例: sdf_prev=-0.2, confid=127, sdf_curr=0.7, 怎么办? GRAZ 状态下, 此 curr 不该冲掉 prev
                    weight_curr = 1;
#elif 0 //v13.7【DEPRECATED】 之前为啥设定 wc=1？ 为了照顾graz假阳性, 边缘容易因为 法向误差, 判定为 graz, 需要平滑掉
                    //所以此逻辑也不行, 无法正确处理假阳性 @2017-12-29 09:11:04
                    if(cos_snorm_p_c > COS45)
                        weight_curr = 1;
                    else
                        weight_curr = 0;
#elif 0 //v13.8 考虑 w 是 confid 与 p-c-dist (Dpc) 双变量函数, 模拟高斯函数/钟形曲线: 
                    //① confid 越大, sigma越小, 即对新 curr 越严格; ② p-c-dist 越大, curr 离 mu 越远, 权重越小
                    //把曲线近似为: 三段折线:wc= min(0, max(1, 1-c*(Dpc-th_min)/(M*(TH-th)) ) ) 
                    //↑--①, if Dpc<th: wc=1; ②, elif Dpc>TH; wc=0; ③ else 中间状态: wc= 1-c*(Dpc-th_min)/[M*(TH-th)]

                    const float tsdf_th_min = 0.2, //e.g.: 25mm*0.2=5mm
                        tsdf_TH_max = 0.6;    //e.g.: 25mm*0.6=15mm

                    float dpc = abs(tsdf_curr - tsdf_prev1);
                    weight_curr = 1 - 1.f * *snormPrevConfid / SCHAR_MAX * (dpc - tsdf_th_min) / (tsdf_TH_max - tsdf_th_min);
                    weight_curr = max(0.f, min(1.f, weight_curr));
#elif 1 //v13.9, 修改分段函数形式, 不要定宽 th, TH; 要根据 confid 动态变化的 sigma
                    float dpc = abs(tsdf_curr - tsdf_prev1);
                    float sigma = 1 - 1.f * *snormPrevConfid / SCHAR_MAX; //confid~(0,127) --> sigma~(1,0)
                    sigma = 0.2 * sigma + 0.1; //(0,1)--> (0.1, 0.3)

                    weight_curr = 1 - 1.f * *snormPrevConfid / SCHAR_MAX * (dpc - sigma) / (2 * sigma); //分母即 3σ-σ=2σ
                    weight_curr = max(0.f, min(1.f, weight_curr));
#endif
                }
                else{//sdf_normed > 1 //对于近面侧视, 但看到远面正视的情形
                    if(cos_vray_norm_prev > 0 && tsdf_prev1 < -pendingFixThresh) //即, 1, 背面视角; 2, 很负, p<<0
                    //if(tsdf_prev1 < -pendingFixThresh) //v13.10, 仅判定 p<<0, 去掉【背面视角】约束 \
                            ↑--错！考虑 cos_vray_norm_prev 是因为: graz时, 有时边缘全反射, 导致远处面错误"看到", 单单 pendingFixThresh 不够, 因为这个值可能不稳定, 
                    {
                        //↓--需要核实3D法向侧“1/8球”邻域值全 <0, 确保不破坏过零点; 类似 v12
                        int sx = snorm_prev_g.x > 0 ? 1 : -1, //sign, 正负号
                            sy = snorm_prev_g.y > 0 ? 1 : -1,
                            sz = snorm_prev_g.z > 0 ? 1 : -1;
                        bool doBreak = false;
                        int nbr_x = -1,
                            nbr_y = -1,
                            nbr_z = -1;
                        float nbr_tsdf;
                        int nbr_weight;
                        for(int ix=0; ix<=1 && !doBreak; ix++){
                            for(int iy=0; iy<=1 && !doBreak; iy++){
                                for(int iz=0; iz<=1 && !doBreak; iz++){
                                    if(0==ix && 0==iy && 0==iz)
                                        continue;

                                    nbr_x = min(VOLUME_X-1, max(0, x + ix*sx));
                                    nbr_y = min(VOLUME_Y-1, max(0, y + iy*sy));
                                    nbr_z = min(VOLUME_Z-1, max(0, z + iz*sz));

                                    short2 *nbr_pos = volume1.ptr(nbr_y) + nbr_x;
                                    nbr_pos += nbr_z * elem_step;

                                    //float nbr_tsdf;
                                    //int nbr_weight;
                                    unpack_tsdf(*nbr_pos, nbr_tsdf, nbr_weight);
                                    if(WEIGHT_RESET_FLAG != nbr_weight && nbr_tsdf > 0){
                                        doBreak = true;
                                        break; //不显式中断其实也无所谓。。
                                    }
                                }
                            }
                        }//for-ix

                        if(doDbgPrint){
                            printf("\tdoBreak: %s\n", doBreak ? "doBreakTTT" : "doBreakFFF-grazing_reset");
                            printf("\tNBR-XYZ: %d, %d, %d; NBR-TSDF/w: %f, %d\n", nbr_x, nbr_y, nbr_z, nbr_tsdf, nbr_weight);
                        }

                        if(false == doBreak){
                            *snormPrevConfid = 0;
                            grazing_reset = true;
                        }
                        else
                            weight_curr = 0;
                    }//if-cos>0 & p<<0
                    else
                        weight_curr = 0;
                }//else-sdf_normed > 1
            }//elif-(GRAZING_VIEW == view_case)
#elif 1
            else if(GRAZING_VIEW_POS == view_case){
                if(snormPrevConfid < Tsdf::MAX_WEIGHT_V13 / 2.f)
                    weight_curr = 1;
                else
                    weight_curr = 0;
            }
            else if(GRAZING_VIEW_NEG == view_case){
            }
#endif
            //else{ //if-isNewFace //v13.2 放弃
            else if(OPPOSITE_VIEW == view_case){ //之前 if-isNewFace 
#if 0   //v13.old   之前: 根据 p, c tsdf 值, 调整 w 融合权重; 计算w 不稳定, 放弃; 理应简介明确 @2018-1-2 07:32:41

                //异视角, 原因有: 
                //1, 相机正常环绕到背后; 
                //2, 深度图因运动模糊/配准误差等, 导致不准, 影响到某些vox; 主要是: 某时刻 Dmap(i) 伸进(甚至跨过) 已有表面薄片结构时, 要特殊处理
                if(tsdf_prev1 >= 0){ //prev正
                    //weight_curr = max(0, tsdf_prev1 / tsdf_curr);
                    //↑-当 curr<0时, w=max(0, -X)=0; curr>0时, c<<p 则权重大
                    //weight_curr = min(tsdf_prev1, max(0, tsdf_prev1 / tsdf_curr) ); //不够好: 考虑 prev<1 情形
                    //weight_curr = min(max(1.f, tsdf_prev1), max(0.f, tsdf_prev1 / tsdf_curr) );
                    weight_curr = min(max(1.f, tsdf_prev1), max(0.f, tsdf_prev1 / (tsdf_curr + (tsdf_curr>0 ? 1 : -1) * 0.01)) ); //避免除零
                }
                else{ //if-(tsdf_prev1 < 0) //必然 p>-1, 不会太负
                    //w_curr 默认=1
                    //眼前: 看 tprev 反向延伸; 身后: 看 diffDmap //【放弃】

                    //if(tsdf_curr + tsdf_prev1 > 0) //当前帧可能有问题: 1, 背面观测, 但因整体配准, 导致局部深过头了; 2, 侧面观测到远处表面正面
                    //    weight_curr = 1;
                    //else
                    //    weight_curr = (tsdf_curr+1) / max(tsdf_prev1+1, 0.1);
                    weight_curr = tsdf_curr + tsdf_prev1 > 0 ? 
                        //1 : (tsdf_curr+1) / max(tsdf_prev1+1, 0.1); //1 不够好, 考虑: 若 tcurr 很大, 理应很小权重, 1太大
                        (-tsdf_prev1 / tsdf_curr) : (tsdf_curr+1) / max(tsdf_prev1+1, 0.1);
                }//if-tprev><0
#elif 1 //v13.10
                if(doDbgPrint){
                    printf("\tabs(tsdf_prev1) < abs(tsdf_curr): %s\n", abs(tsdf_prev1) < abs(tsdf_curr) ? "TTT-curr更远" : "FFF+curr更近");
                }

                //if(tsdf_prev1 >= 0)
                if(abs(tsdf_prev1) < abs(tsdf_curr)) //prev 更贴近表面
                    weight_curr = 0;
                else //curr 更贴近表面
                    weight_curr = 10;


#endif
            //}//if-isNewFace OR NOT
            }//if-OPPOSITE_VIEW
            
            if(doDbgPrint){
                printf("\tweight_prev1, weight_curr:: %d, %f\n", weight_prev1, weight_curr);
            }

            //2, 更新 tsdf, weight, snorm
            if(WEIGHT_RESET_FLAG != weight_prev1) //避免分母除零
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * weight_curr) 
                / (weight_prev1 + weight_curr);
            weight_new = weight_prev1; //默认不更新

            //if(!isNewFace){ //若同向, 
            if(SAME_SIDE_VIEW == view_case){
                //if(grazing_reset) //grazing_reset 局部变量, 不可用作判定
                if(WEIGHT_RESET_FLAG == weight_prev1 && sdf_normed > 1){ //GRAZ 时, sdf>1 时 grazing_reset 的结果, 
                    if(doDbgPrint)
                        printf("\tWEIGHT_RESET_FLAG == weight_prev1 && sdf_normed > 1\n");
                }
                else{ //① 正常 same-side, 未受过 grazing_reset 影响; 或 ② graz-reset, 但是 sdf<1;
                    //权重累积
                    weight_new = min(weight_prev1 + weight_curr, (float)Tsdf::MAX_WEIGHT_V13);

                    if(isSnormPrevInit){
                        //if(doDbgPrint) printf("snorm_curr_g-111: [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);

                        //逐步、稍微更新法向量
                        snorm_curr_g = (snorm_prev_g * weight_prev1 + snorm_curr_g * weight_curr) 
                            * (1./(weight_prev1 + weight_curr) ); //float3 没重载除法

                        //if(doDbgPrint) printf("snorm_curr_g-222: [%f, %f, %f], norm(snorm_curr_g):= %f\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, norm(snorm_curr_g));

                        //snorm_curr_g *= 1./norm(snorm_curr_g);
                        snorm_curr_g = normalized(snorm_curr_g);

                        //if(doDbgPrint) printf("snorm_curr_g-333: [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);
                    }

                    //会不会因为 char 存储, 前面的更新没意义? 不知道 @2017-12-18 00:55:39
                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);

                    //法向置信度+1
                    //(*snormPrevConfid) +=1; //要有上限!
                    *snormPrevConfid = min(SCHAR_MAX, *snormPrevConfid + 1);

                    if(doDbgPrint){
                        printf("\t*snormPrevConfid+1\n");
                        //printf("【snorm_pos.x】: %d, %d, %f, %f, %d\n", (*snorm_pos).x, snorm_pos->x, snorm_curr_g.x * CHAR_MAX, nearbyintf(snorm_curr_g.x * CHAR_MAX), (int)nearbyintf(snorm_curr_g.x * CHAR_MAX));
                        //printf("【snorm_pos.y】: %d, %d, %f, %f, %d\n", (*snorm_pos).y, snorm_pos->y, snorm_curr_g.y * CHAR_MAX, nearbyintf(snorm_curr_g.y * CHAR_MAX), (int)nearbyintf(snorm_curr_g.y * CHAR_MAX));
                        //printf("【snorm_pos.z】: %d, %d, %f, %f, %d\n", (*snorm_pos).z, snorm_pos->z, snorm_curr_g.z * CHAR_MAX, nearbyintf(snorm_curr_g.z * CHAR_MAX), (int)nearbyintf(snorm_curr_g.z * CHAR_MAX));
                    }
                }
            }
            //else{ //若异向, 
            else if(OPPOSITE_VIEW == view_case){
                //权重递减到一半 //暂行 @2017-12-17 23:56:00
                //weight_new = max(weight_prev1 - weight_curr, Tsdf::MAX_WEIGHT_V13 / 2.f);
                //↑-不好, 若权重没到 MAX/2 呢? //用 snorm-initialized-confidence-thresh, 因为达不到此 thresh 不会走到这个分支
                weight_new = max(int(weight_prev1 - weight_curr), snormPrevConfid_thresh);

                //仅当 w_curr 较大时, 即要 curr 冲 prev 时, 才鼓捣法向
                if(weight_curr > 1){
                    (*snormPrevConfid) -=1;

                    if(doDbgPrint){
                        printf("*snormPrevConfid---1\n");
                    }
                }
                if(*snormPrevConfid <= snormPrevConfid_thresh){
                    *snormPrevConfid = snormPrevConfid_thresh + 1;

                    //直接用 curr 覆盖:
                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                }
            }
            else if(GRAZING_VIEW == view_case){
                //DO-NOTHING
                if(grazing_reset){
                    tsdf_new1 = SLIGHT_POSITIVE;
                    weight_new = WEIGHT_RESET_FLAG; //-1, 是个标记, 表示 grazing_reset 过
                }
                else /*if(WEIGHT_RESET_FLAG != weight_new)*/{
                    //首先注意 WEIGHT_RESET_FLAG
                    if(WEIGHT_RESET_FLAG == weight_prev1)
                        weight_prev1 = 0;

                    //类似 same-side, 权重累积, norm 也缓慢校正, 前面 GRAZING_VIEW 代码段已经设置 weight_curr
                    weight_new = min(weight_prev1 + weight_curr, (float)Tsdf::MAX_WEIGHT_V13);

                    //逐步、稍微更新法向量
                    snorm_curr_g = (snorm_prev_g * weight_prev1 + snorm_curr_g * weight_curr) 
                        * (1./(weight_prev1 + weight_curr) ); //float3 没重载除法
                    snorm_curr_g = normalized(snorm_curr_g);

                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);

                    //graz 下，法向置信度【不+1】
                    //*snormPrevConfid = min(SCHAR_MAX, *snormPrevConfid + 1);

                }
            }//if-(GRAZING_VIEW == view_case)

            //if(WEIGHT_RESET_FLAG != weight_prev1)
                pack_tsdf(tsdf_new1, weight_new, *pos1);

            if(doDbgPrint){
                printf("\ttsdf_new1, weight_new:: %f, %d\n", tsdf_new1, weight_new);
                printf("\tnew-snorm(*snorm_pos): [%d, %d, %d]\n", snorm_pos->x, snorm_pos->y, snorm_pos->z);
                printf("\tnew-snorm(*snorm_pos): [%f, %f, %f]\n", 1.f * (*snorm_pos).x / CHAR_MAX, 1.f * (*snorm_pos).y / CHAR_MAX, 1.f * (*snorm_pos).z / CHAR_MAX);
            }

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v13

    //v13 考虑问题: 控制量 snormPrevConfid 行为与 weight_curr & weight_new 有没有分叉？ 如果始终一致，是否可用一个变量？    @2018-1-5 16:41:01
    //v14 失败: 教训:= ① 不要直接 reset!! 没有后悔药; ② 各向异性, 【确实】容易导致偏差 bias (2017那篇博士论文也提到); 可能不如高斯/滑动平均
    __global__ void
    tsdf23_v14 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];

          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          if(doDbgPrint) printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vray;
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snorm_prev_g;
            snorm_prev_g.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snorm_prev_g.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snorm_prev_g.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);

            //signed char *snormPrevConfid = &snorm_pos->w;
            //↑-v14 尝试去掉 snormPrevConfid 控制量, 用 w 本身替代
            //const int snormPrevConfid_thresh = 5;

            //bool isSnormPrevInit = (*snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零
            bool isSnormPrevInit = weight_prev1 > 0; //v14 尝试用 w 替代 snormPrevConfid 控制量

            const float COS30 = 0.8660254f
                       ,COS45 = 0.7071f
                       ,COS60 = 0.5f
                       ,COS75 = 0.258819f
                       ;
            const float cosThreshSnorm = COS30; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_snorm_p_c = dot(snorm_prev_g, snorm_curr_g);
            float cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);

            int view_case = SAME_SIDE_VIEW; //尝试取代 isNewFace @2017-12-22 10:58:03
            if(isSnormPrevInit){ //v14: 用了 w
                if(abs(cos_vray_norm_prev) < COS75){ //斜视判定
                    view_case = GRAZING_VIEW; //v13.3: 若 p在边缘导致法向-视线夹角很大, 初始错,之后对,如何修复?
                }
                else if(cos_vray_norm_prev < -COS75){ //同面正视
                    view_case = SAME_SIDE_VIEW;
                }
                else{ //if(cos_vray_norm_prev > COS75) //背面正视
                    view_case = OPPOSITE_VIEW;
                }
            }

            if(doDbgPrint){
                printf("vray_normed: [%f, %f, %f]; cos_vray_norm_prev, %f; cos_vray_norm_curr, %f (%s, ALWAYS cos<0)\n", 
                    vray_normed.x, vray_normed.y, vray_normed.z, cos_vray_norm_prev, cos_vray_norm_curr, cos_vray_norm_curr>0? "×":"√");
                //这里打印 snorm 校正之前的 cos-vray-snorm_c (校正之后必然 cos <0 了); snorm 却是校正之后的 @2017-12-20 10:43:19
                printf("cos_snorm_p_c: %f ---snorm_prev_g, snorm_curr_g: [%f, %f, %f], [%f, %f, %f]\n", 
                    cos_snorm_p_c, snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);

                printf("isSnormPrevInit: %s, \n", isSnormPrevInit ? "TTT" : "FFF");
                //printf("isSnormPrevInit: %s, --snormPrevConfid: %d\n", 
                //    isSnormPrevInit ? "TRUE":"FALSE", *snormPrevConfid);

                //printf("%s isNewFace:::", isNewFace? "YES":"NOT");
                printf("%s", view_case==SAME_SIDE_VIEW ? "SAME-SIDE" : (view_case==GRAZING_VIEW ? "GRAZING" : "OPPO-SIDE") );
                printf("::: tsdf_prev1, tsdf_curr: %f, %f\n", tsdf_prev1, tsdf_curr);
            }

            //1, weighting 策略
            //float weight_curr = 1; //AVG, FIX, IGN, 都放弃, 用权重决定一切 @2017-12-14 10:53:54
            float weight_curr = 0; //改用 view_case 控制量之后, 默认权重置零
            float tsdf_new1 = SLIGHT_POSITIVE; //存更新后的 tsdf & w
            int weight_new = WEIGHT_RESET_FLAG;
            bool grazing_reset = false;

            if(SAME_SIDE_VIEW == view_case){
                weight_curr = 1;
            }
            else if(GRAZING_VIEW == view_case){//略复杂
                weight_curr = 0; //逻辑块内首先全部置零

                if(doDbgPrint){
                    printf("GRAZING_VIEW--sdf_normed=%f (%s)--cos_V_N_p=%f (%s)"
                        "--tsdf_prev1=%f (%s)-pendingFixThresh: %f\n", 
                        sdf_normed, sdf_normed > 1 ? ">1" : "<=1", 
                        cos_vray_norm_prev, cos_vray_norm_prev < 0 ? "<0" : ">=0",
                        tsdf_prev1, tsdf_prev1 > 0 ? ">0" : "<=0",
                        pendingFixThresh);
                    printf("\tabs(tsdf_prev1) < pendingFixThresh: %s;\n", \
                        abs(tsdf_prev1) < pendingFixThresh ? "TTT" : "FFF");
                }
                if(sdf_normed <= 1){ //其实也即 |..|<=1, 因为本来有 ..>= -1; 此逻辑块内 sdf==tsdf
                    //v13.9, 钟形曲线, 修改分段函数形式, 不要定宽 th, TH; 要根据 confid 动态变化的 sigma
                    float dpc = abs(tsdf_curr - tsdf_prev1);
                    float sigma = 1 - 1.f * weight_prev1 / Tsdf::MAX_WEIGHT_V13; //confid~(0,127) --> sigma~(1,0)
                    sigma = 0.2 * sigma + 0.1; //(0,1)--> (0.1, 0.3)

                    weight_curr = 1 - 1.f * weight_prev1 / Tsdf::MAX_WEIGHT_V13 * (dpc - sigma) / (2 * sigma); //分母即 3σ-σ=2σ
                    weight_curr = max(0.f, min(1.f, weight_curr));
                }
                else{//sdf_normed > 1 //对于近面侧视, 但看到远面正视的情形
                    if(cos_vray_norm_prev > 0 && tsdf_prev1 < -pendingFixThresh) //即, 1, 背面视角; 2, 很负, p<<0
                        //if(tsdf_prev1 < -pendingFixThresh) //v13.10, 仅判定 p<<0, 去掉【背面视角】约束 \
                        ↑--错！考虑 cos_vray_norm_prev 是因为: graz时, 有时边缘全反射, 导致远处面错误"看到", 单单 pendingFixThresh 不够, 因为这个值可能不稳定, 
                    {
                        //↓--需要核实3D法向侧“1/8球”邻域值全 <0, 确保不破坏过零点; 类似 v12
                        int sx = snorm_prev_g.x > 0 ? 1 : -1, //sign, 正负号
                            sy = snorm_prev_g.y > 0 ? 1 : -1,
                            sz = snorm_prev_g.z > 0 ? 1 : -1;
                        bool doBreak = false;
                        int nbr_x = -1,
                            nbr_y = -1,
                            nbr_z = -1;
                        float nbr_tsdf;
                        int nbr_weight;
                        for(int ix=0; ix<=1 && !doBreak; ix++){
                            for(int iy=0; iy<=1 && !doBreak; iy++){
                                for(int iz=0; iz<=1 && !doBreak; iz++){
                                    if(0==ix && 0==iy && 0==iz)
                                        continue;

                                    nbr_x = min(VOLUME_X-1, max(0, x + ix*sx));
                                    nbr_y = min(VOLUME_Y-1, max(0, y + iy*sy));
                                    nbr_z = min(VOLUME_Z-1, max(0, z + iz*sz));

                                    short2 *nbr_pos = volume1.ptr(nbr_y) + nbr_x;
                                    nbr_pos += nbr_z * elem_step;

                                    //float nbr_tsdf;
                                    //int nbr_weight;
                                    unpack_tsdf(*nbr_pos, nbr_tsdf, nbr_weight);
                                    if(WEIGHT_RESET_FLAG != nbr_weight && nbr_tsdf > 0){
                                        doBreak = true;
                                        break; //不显式中断其实也无所谓。。
                                    }
                                }
                            }
                        }//for-ix

                        if(doDbgPrint){
                            printf("\tdoBreak: %s\n", doBreak ? "doBreakTTT" : "doBreakFFF-grazing_reset");
                            printf("\tNBR-XYZ: %d, %d, %d; NBR-TSDF/w: %f, %d\n", nbr_x, nbr_y, nbr_z, nbr_tsdf, nbr_weight);
                        }

                        if(false == doBreak){
                            //*snormPrevConfid = 0;
                            weight_new = WEIGHT_RESET_FLAG; //之后重入此 vox 仍然有效
                            grazing_reset = true; //仅当前循环中有效
                        }
                        else
                            weight_curr = 0;
                    }//if-cos>0 & p<<0
                    else
                        weight_curr = 0; //多写一遍, 好读, 其实默认
                }//else-sdf_normed > 1
            }//elif-(GRAZING_VIEW == view_case)
            else if(OPPOSITE_VIEW == view_case){ //之前 if-isNewFace 
                //v13.10
                if(doDbgPrint){
                    printf("\tabs(tsdf_prev1) < abs(tsdf_curr): %s\n", abs(tsdf_prev1) < abs(tsdf_curr) ? "TTT-curr更远" : "FFF+curr更近");
                }

                weight_curr = 0; //默认置零

                //if(tsdf_prev1 >= 0){ //若 p+, 无论 c+/- 都不能冲
                //    weight_curr = 0;
                //}
                //if(abs(tsdf_prev1) < abs(tsdf_curr)) //prev 更贴近表面
                //    weight_curr = 0;
                //else //curr 更贴近表面
                //    weight_curr = 10;

                if(tsdf_prev1 < 0 && abs(tsdf_prev1) > abs(tsdf_curr)){
                    //↑=仅当 p-, 且 |p|>|c|, 才【可能】c 冲 p; 且仍要判定沿 norm_p 方向, 邻域 nbr 全<0, 确保过零点

                    //拷贝自 上面 GRAZING_VIEW 逻辑块内 @2018-1-7 21:25:12
                    int sx = snorm_prev_g.x > 0 ? 1 : -1, //sign, 正负号
                        sy = snorm_prev_g.y > 0 ? 1 : -1,
                        sz = snorm_prev_g.z > 0 ? 1 : -1;
                    bool doBreak = false;
                    int nbr_x = -1,
                        nbr_y = -1,
                        nbr_z = -1;
                    float nbr_tsdf;
                    int nbr_weight;
                    for(int ix=0; ix<=1 && !doBreak; ix++){
                        for(int iy=0; iy<=1 && !doBreak; iy++){
                            for(int iz=0; iz<=1 && !doBreak; iz++){
                                if(0==ix && 0==iy && 0==iz)
                                    continue;

                                nbr_x = min(VOLUME_X-1, max(0, x + ix*sx));
                                nbr_y = min(VOLUME_Y-1, max(0, y + iy*sy));
                                nbr_z = min(VOLUME_Z-1, max(0, z + iz*sz));

                                short2 *nbr_pos = volume1.ptr(nbr_y) + nbr_x;
                                nbr_pos += nbr_z * elem_step;

                                //float nbr_tsdf;
                                //int nbr_weight;
                                unpack_tsdf(*nbr_pos, nbr_tsdf, nbr_weight);
                                if(WEIGHT_RESET_FLAG != nbr_weight && nbr_tsdf > 0){
                                    doBreak = true;
                                    break; //不显式中断其实也无所谓。。
                                }
                            }
                        }
                    }//for-ix

                    if(doDbgPrint){
                        printf("\tdoBreak: %s\n", doBreak ? "doBreakTTT" : "doBreakFFF-grazing_reset");
                        printf("\tNBR-XYZ: %d, %d, %d; NBR-TSDF/w: %f, %d\n", nbr_x, nbr_y, nbr_z, nbr_tsdf, nbr_weight);
                    }

                    if(false == doBreak){
                        //weight_curr = 10;

                        grazing_reset = true;
                    }

                }
            }//if-OPPOSITE_VIEW

            if(doDbgPrint){
                printf("\tweight_prev1, weight_curr:: %d, %f\n", weight_prev1, weight_curr);
            }

            //2, 更新 tsdf, weight, snorm
            if(WEIGHT_RESET_FLAG != weight_prev1) //避免分母除零
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * weight_curr) 
                / (weight_prev1 + weight_curr);
            weight_new = weight_prev1; //默认不更新

            if(SAME_SIDE_VIEW == view_case){
                //if(grazing_reset) //grazing_reset 局部变量, 不可用作判定
                if(WEIGHT_RESET_FLAG == weight_prev1 && sdf_normed > 1){ //GRAZ 时, sdf>1 时 grazing_reset 的结果, 
                    if(doDbgPrint)
                        printf("\tWEIGHT_RESET_FLAG == weight_prev1 && sdf_normed > 1\n");
                }
                else{ //① 正常 same-side, 未受过 grazing_reset 影响; 或 ② graz-reset, 但是 sdf<1;
                    //权重累积
                    if(WEIGHT_RESET_FLAG == weight_prev1)
                        weight_prev1 = 0;
                    weight_new = min(weight_prev1 + weight_curr, (float)Tsdf::MAX_WEIGHT_V13);

                    if(isSnormPrevInit){
                        //if(doDbgPrint) printf("snorm_curr_g-111: [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);

                        //逐步、稍微更新法向量
                        snorm_curr_g = (snorm_prev_g * weight_prev1 + snorm_curr_g * weight_curr) 
                            * (1./(weight_prev1 + weight_curr) ); //float3 没重载除法

                        //if(doDbgPrint) printf("snorm_curr_g-222: [%f, %f, %f], norm(snorm_curr_g):= %f\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, norm(snorm_curr_g));

                        //snorm_curr_g *= 1./norm(snorm_curr_g);
                        snorm_curr_g = normalized(snorm_curr_g);

                        //if(doDbgPrint) printf("snorm_curr_g-333: [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);
                    }

                    //会不会因为 char 存储, 前面的更新没意义? 不知道 @2017-12-18 00:55:39
                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                }
            }
            else if(GRAZING_VIEW == view_case){
                if(grazing_reset){
                    tsdf_new1 = SLIGHT_POSITIVE;
                    weight_new = WEIGHT_RESET_FLAG; //-1, 是个标记, 表示 grazing_reset 过
                }
                else /*if(WEIGHT_RESET_FLAG != weight_new)*/{
                    //首先注意 WEIGHT_RESET_FLAG
                    if(WEIGHT_RESET_FLAG == weight_prev1)
                        weight_prev1 = 0;

                    //类似 same-side, 权重累积, norm 也缓慢校正, 前面 GRAZING_VIEW 代码段已经设置 weight_curr
                    weight_new = min(weight_prev1 + weight_curr, (float)Tsdf::MAX_WEIGHT_V13);

                    //逐步、稍微更新法向量
                    snorm_curr_g = (snorm_prev_g * weight_prev1 + snorm_curr_g * weight_curr) 
                        * (1./(weight_prev1 + weight_curr) ); //float3 没重载除法
                    snorm_curr_g = normalized(snorm_curr_g);

                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);

                    //graz 下，法向置信度【不+1】
                    //*snormPrevConfid = min(SCHAR_MAX, *snormPrevConfid + 1);

                }
            }//if-(GRAZING_VIEW == view_case)
            else if(OPPOSITE_VIEW == view_case){
#if 0 //v14: 仅不断减小 w-new, 直到此 vox 变成 SAME 逻辑

                weight_new = max(int(weight_prev1 - weight_curr), 0);

                //仅当 w_curr 较大时, 即要 curr 冲 prev 时, 才鼓捣法向
                //if(weight_curr > 1){
                //    (*snormPrevConfid) -=1;

                //    if(doDbgPrint){
                //        printf("*snormPrevConfid---1\n");
                //    }
                //}

                //if(*snormPrevConfid <= snormPrevConfid_thresh){
                //    *snormPrevConfid = snormPrevConfid_thresh + 1;

                //    //直接用 curr 覆盖:
                //    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                //    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                //    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                //}
#elif 1 //v14.1: oppo 不再用渐变权重, 直接 reset
                if(grazing_reset){
                    tsdf_new1 = SLIGHT_POSITIVE;
                    weight_new = WEIGHT_RESET_FLAG; //-1, 是个标记, 表示 grazing_reset 过

                    (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                    (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                    (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
                }
#endif
            }//if-(OPPOSITE_VIEW == view_case)

            if(WEIGHT_RESET_FLAG != weight_prev1)
                pack_tsdf(tsdf_new1, weight_new, *pos1);

            if(doDbgPrint){
                printf("\ttsdf_new1, weight_new:: %f, %d\n", tsdf_new1, weight_new);
                printf("\tnew-snorm(*snorm_pos): [%d, %d, %d]\n", snorm_pos->x, snorm_pos->y, snorm_pos->z);
                printf("\tnew-snorm(*snorm_pos): [%f, %f, %f]\n", 1.f * (*snorm_pos).x / CHAR_MAX, 1.f * (*snorm_pos).y / CHAR_MAX, 1.f * (*snorm_pos).z / CHAR_MAX);
            }
          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v14

    //根据 v14 教训, 换思路: 【切断负值】; 当且仅当: ① p<0 负值区域; ② w够大, 即说明之前观测"够稳"; ③ cos-vray-n_p >cos75°, 即背面oppo观测, 非grazing; ④ 法向量 n_p 方向, 邻域确保存在过零点
    __global__ void
    tsdf23_v15 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

       //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];

          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          if(doDbgPrint) printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                return;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vray;
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            float3 snorm_prev_g;
            snorm_prev_g.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snorm_prev_g.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snorm_prev_g.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //read and unpack
            float tsdf_prev1;
            float weight_prev1;
            int weight_prev1_scaled;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1_scaled);
            weight_prev1 = 1.f * weight_prev1_scaled / WEIGHT_SCALE; //用于解决钟形曲线 float w<1 转 int 截断的错误

            //signed char *snormPrevConfid = &snorm_pos->w;
            //↑-v14 尝试去掉 snormPrevConfid 控制量, 用 w 本身替代
            //const int snormPrevConfid_thresh = 5;

            //bool isSnormPrevInit = (*snormPrevConfid > snormPrevConfid_thresh); //去掉 X>1e-8 判定, 因为 confid > th 时必然 X 已经初始化非零
            //bool isSnormPrevInit = weight_prev1 > 0; //v14 尝试用 w 替代 snormPrevConfid 控制量
            bool isSnormPrevInit = weight_prev1 > 1; //v15 因为 global_time_ == 0 时, 已经 w=1

            const float COS30 = 0.8660254f
                       ,COS45 = 0.7071f
                       ,COS60 = 0.5f
                       ,COS75 = 0.258819f
                       ;
            const float cosThreshSnorm = COS30; //cos(30°), 与 vray 区分开, 采用更宽容阈值 @2017-3-15 00:39:18

            float cos_snorm_p_c = dot(snorm_prev_g, snorm_curr_g);
            float cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);

            int view_case = SAME_SIDE_VIEW; //尝试取代 isNewFace @2017-12-22 10:58:03
            if(isSnormPrevInit){ //v14: 用了 w
#if 0   //OLD, 
                if(abs(cos_vray_norm_prev) < COS75){ //斜视判定
                    view_case = GRAZING_VIEW; //v13.3: 若 p在边缘导致法向-视线夹角很大, 初始错,之后对,如何修复?
                }
                else if(cos_vray_norm_prev < -COS75){ //同面正视
                    view_case = SAME_SIDE_VIEW;
                }
                else{ //if(cos_vray_norm_prev > COS75) //背面正视
                    view_case = OPPOSITE_VIEW;
                }
#elif 1 //v15.2: 为对应 oppo 二次截断, 放宽 graz 条件, 即 oppo 条件更严格
                if(cos_vray_norm_prev < -COS75){ //同面正视
                    view_case = SAME_SIDE_VIEW;
                }
                else if(abs(cos_vray_norm_prev) < COS75 || abs(cos_vray_norm_curr) < COS75){
                    view_case = GRAZING_VIEW; //v13.3: 若 p在边缘导致法向-视线夹角很大, 初始错,之后对,如何修复?
                }
                else{ //if(cos_vray_norm_prev > COS75) //背面正视
                    view_case = OPPOSITE_VIEW;
                }

#endif
            }

            if(doDbgPrint){
                printf("vray_normed: [%f, %f, %f]; cos_vray_norm_prev, %f; cos_vray_norm_curr, %f (%s, ALWAYS cos<0)\n", 
                    vray_normed.x, vray_normed.y, vray_normed.z, cos_vray_norm_prev, cos_vray_norm_curr, cos_vray_norm_curr>0? "×":"√");
                //这里打印 snorm 校正之前的 cos-vray-snorm_c (校正之后必然 cos <0 了); snorm 却是校正之后的 @2017-12-20 10:43:19
                printf("cos_snorm_p_c: %f ---snorm_prev_g, snorm_curr_g: [%f, %f, %f], [%f, %f, %f]\n", 
                    cos_snorm_p_c, snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z);

                printf("isSnormPrevInit: %s, \n", isSnormPrevInit ? "TTT" : "FFF");
                //printf("isSnormPrevInit: %s, --snormPrevConfid: %d\n", 
                //    isSnormPrevInit ? "TRUE":"FALSE", *snormPrevConfid);

                //printf("%s isNewFace:::", isNewFace? "YES":"NOT");
                printf("%s", view_case==SAME_SIDE_VIEW ? "SAME-SIDE" : (view_case==GRAZING_VIEW ? "GRAZING" : "OPPO-SIDE") );
                printf("::: tsdf_prev1, tsdf_curr: %f, %f\n", tsdf_prev1, tsdf_curr);
            }

            //1, weighting 策略
            //float weight_curr = 1; //AVG, FIX, IGN, 都放弃, 用权重决定一切 @2017-12-14 10:53:54
            float weight_curr = 0; //改用 view_case 控制量之后, 默认权重置零
            float tsdf_new1 = SLIGHT_POSITIVE; //存更新后的 tsdf & w
            float weight_new = WEIGHT_RESET_FLAG; //v15 还用 reset-flag 吗? 记不住了
            int weight_new_scaled;
            bool grazing_reset = false;

            if(SAME_SIDE_VIEW == view_case){
                weight_curr = 1;
            }
            else if(GRAZING_VIEW == view_case){
                //weight_curr = 1;    //v15.0: graz 时仍然 w=1, graz 仅用于???  @2018-1-9 14:53:21
                //↑-不行, 斜视背面时, e.g., -0.1 被 1 不断侵蚀, 

                //v15.1: 仍用 v13.9, 钟形曲线, 修改分段函数形式, 不要定宽 th, TH; 要根据 confid 动态变化的 sigma
                float dpc = abs(tsdf_curr - tsdf_prev1);
                float sigma = 1 - 1.f * weight_prev1 / Tsdf::MAX_WEIGHT_V13; //confid~(0,127) --> sigma~(1,0)
                sigma = 0.2 * sigma + 0.1; //(0,1)--> (0.1, 0.3)

                weight_curr = 1 - 1.f * weight_prev1 / Tsdf::MAX_WEIGHT_V13 * (dpc - sigma) / (2 * sigma); //分母即 3σ-σ=2σ
                weight_curr = max(0.f, min(1.f, weight_curr));

            }
            else if(OPPOSITE_VIEW == view_case){ //之前 if-isNewFace 
                //weight_curr = 0; //OLD, 改成: 不操纵 w, 因为总会造成 bias  
                if(tsdf_prev1 > 0){ //还是用 wc, 但是要么 0, 要么 -wp (本意是设置 w_new = 0, 二次截断)
                    weight_curr = 0; //正值不许被冲
                }
                else if(tsdf_prev1 < 0)
                    //&& weight_prev1 > 50) //经验值
                {
                    //根据 v14 教训, 换思路: 【切断负值】; 当且仅当: ① p<0 负值区域; ② w够大, 即说明之前观测"够稳"; ③ cos-vray-n_p >cos75°, 即背面【oppo】观测, 非grazing; ④ 法向量 n_p 方向, 邻域确保存在过零点

                    int sx = snorm_prev_g.x > 0 ? 1 : -1, //sign, 正负号
                        sy = snorm_prev_g.y > 0 ? 1 : -1,
                        sz = snorm_prev_g.z > 0 ? 1 : -1;
                    bool doBreak = false;
                    int nbr_x = -1,
                        nbr_y = -1,
                        nbr_z = -1;
                    float nbr_tsdf;
                    int nbr_weight;
                    for(int ix=0; ix<=1 && !doBreak; ix++){
                        for(int iy=0; iy<=1 && !doBreak; iy++){
                            for(int iz=0; iz<=1 && !doBreak; iz++){
                                if(0==ix && 0==iy && 0==iz)
                                    continue;

                                nbr_x = min(VOLUME_X-1, max(0, x + ix*sx));
                                nbr_y = min(VOLUME_Y-1, max(0, y + iy*sy));
                                nbr_z = min(VOLUME_Z-1, max(0, z + iz*sz));

                                short2 *nbr_pos = volume1.ptr(nbr_y) + nbr_x;
                                nbr_pos += nbr_z * elem_step;

                                //float nbr_tsdf;
                                //int nbr_weight;
                                unpack_tsdf(*nbr_pos, nbr_tsdf, nbr_weight);
                                //if(WEIGHT_RESET_FLAG != nbr_weight && nbr_tsdf > 0){
                                if(0 != nbr_weight && nbr_tsdf > 0){ //v15.0: w_new 不再填 WEIGHT_RESET_FLAG, 而是直接填零
                                    doBreak = true;
                                    break; //不显式中断其实也无所谓。。
                                }
                            }
                        }
                    }//for-ix

                    if(doDbgPrint){
                        printf("\tdoBreak: %s\n", doBreak ? "doBreakTTT=不动" : "doBreakFFF-可以reset");
                        printf("\tNBR-XYZ: %d, %d, %d; NBR-TSDF/w: %f, %d\n", nbr_x, nbr_y, nbr_z, nbr_tsdf, nbr_weight);
                    }

                    if(false == doBreak){
                        weight_curr = -weight_prev1;
                    }
                }//if=p<0 & w> th
            }//if-OPPOSITE_VIEW

            if(doDbgPrint){
                printf("\tweight_prev1, weight_curr:: %f, %f\n", weight_prev1, weight_curr);
            }

            //2, 更新 tsdf, weight, snorm
            weight_new = min(weight_prev1 + weight_curr, (float)Tsdf::MAX_WEIGHT_V13);
            if(0 == weight_new){
                tsdf_new1 = 0;
            }
            else{ //分母不为零
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * weight_curr) / weight_new;
            }
            weight_new_scaled = (int)nearbyintf(weight_new * WEIGHT_SCALE);
            pack_tsdf(tsdf_new1, weight_new_scaled, *pos1);

            //2.2 更新 snorm
            if(SAME_SIDE_VIEW == view_case){
                //逐步、稍微更新法向量
                if(0 != weight_new){
                    snorm_curr_g = (snorm_prev_g * weight_prev1 + snorm_curr_g * weight_curr) 
                        * (1./weight_new ); //float3 没重载除法
                    snorm_curr_g = normalized(snorm_curr_g);
                }
                (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
            }
            else if(GRAZING_VIEW == view_case){
                //DO-NOTHING
            }
            else if(OPPOSITE_VIEW == view_case){
                (*snorm_pos).x = 0;
                (*snorm_pos).y = 0;
                (*snorm_pos).z = 0;
            }

            if(doDbgPrint){
                printf("\ttsdf_new1, weight_new:: %f, %f\n", tsdf_new1, weight_new);
                printf("\tnew-snorm(*snorm_pos): [%d, %d, %d]\n", snorm_pos->x, snorm_pos->y, snorm_pos->z);
                printf("\tnew-snorm(*snorm_pos): [%f, %f, %f]\n", 1.f * (*snorm_pos).x / CHAR_MAX, 1.f * (*snorm_pos).y / CHAR_MAX, 1.f * (*snorm_pos).z / CHAR_MAX);
            }

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)

    }//tsdf23_v15

    //v16: 测试版, 测试仅用 tranc_dist_real 策略, 保持 tdist 较大, 边缘什么效果 @2018-1-18 10:31:39
    __global__ void
    tsdf23_v16 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;

       //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
              printf("coo.xy:(%d, %d)\n", coo.x, coo.y);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];

          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          //float tranc_dist_real = max(cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8

          if(doDbgPrint) printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          {
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);

            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos1, tsdf_prev, weight_prev);

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf_curr) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            if(doDbgPrint){
                printf("tsdf_prev, tsdf, tsdf_new: %f, %f, %f\n", tsdf_prev, tsdf_curr, tsdf_new);
            }

            pack_tsdf (tsdf_new, weight_new, *pos1);
          }
        }
        else{ //NOT (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)
            if(doDbgPrint){
                printf("vxlDbg.xyz:= (%d, %d, %d), coo.xy:= (%d, %d)\n", vxlDbg.x, vxlDbg.y, vxlDbg.z, coo.x, coo.y);
            }
        }
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__ tsdf23_v16

    //v13~v15 失败, 教训: 凡是直接 reset, 都是有偏的, 对噪声不鲁棒, 很容易导致 bias
    //v17 尝试策略: 双 tsdf, 长短 tdist, 动态选择, 哪个合适用哪个; 【缺点】： ① 可能效果仍然差, 过于乐观; ② raycast, march-cubes 可能需要随之大改动 @2018-1-18 15:26:21
    __global__ void
    tsdf23_v17 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度; //值是相对于 tranc_dist 归一化过的

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;
      const float tdist2nd_m = TDIST_MIN_MM / 1e3; //v17

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if(doDbgPrint)
            printf("coo.xy:(%d, %d)\n", coo.x, coo.y);

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];

          float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          if(doDbgPrint) printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters
          {
            //v17.3 求解 cos-vray-snorm_c 放在此块最前面, 但不是最外层 sdf 初始化位置 @2018-1-30 17:15:23
            float3 snorm_curr_g;
            snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
            if(isnan(snorm_curr_g.x)){
                if(doDbgPrint)
                    printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);

                //return; //错, v18.x 时才发现 @2018-3-8 15:29:28
                continue;
            }

            snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
            snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

            float3 vray;
            vray.x = v_g_x;
            vray.y = v_g_y;
            vray.z = v_g_z;
            //float vray_norm = norm(vray);
            float3 vray_normed = normalized(vray); //单位视线向量

            float cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
            if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                //假设不保证外部已正确预处理：
                snorm_curr_g.x *= -1;
                snorm_curr_g.y *= -1;
                snorm_curr_g.z *= -1;
            }

            //v17.3: sdf 按照 cos-vray-snorm_c 投影, 暂不管 snorm_p //已验证: 效果不错, 在表面(零值面)附近, 确实需要此法, 确保精确, 以免后面二次截断(neg_near_zero) 误判
            float sdf_cos = abs(cos_vray_norm_curr) * sdf;
            if(doDbgPrint){
                printf("snorm_curr_g, vray_normed: [%f, %f, %f], [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
                printf("sdf-orig: %f,, cos_vray_norm_curr: %f,, sdf_cos: %f\n", sdf, cos_vray_norm_curr, sdf_cos);
            }

            sdf = sdf_cos;
            float sdf_normed = sdf * tranc_dist_inv;
            float tsdf_curr = fmin (1.0f, sdf_normed);
            float sdf_normed_mm = sdf_normed * 1e3;


            float3 snorm_prev_g;
            snorm_prev_g.x = 1.f * (*snorm_pos).x / CHAR_MAX; //char2float
            snorm_prev_g.y = 1.f * (*snorm_pos).y / CHAR_MAX;
            snorm_prev_g.z = 1.f * (*snorm_pos).z / CHAR_MAX;

            //read and unpack
            float tsdf_prev1;
            int weight_prev1;
            unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
            bool use_tdist2nd = weight_prev1 % 2; //v17.1: 若 w二进制末位=1, 则用备用 tdist (目前策略是极小截断)
            weight_prev1 = weight_prev1 >> 1; //去掉末位, 后面用
            if(doDbgPrint)
                printf("use_tdist2nd-prev: %d,, tsdf_prev1: %f,, weight_prev1: %d\n", use_tdist2nd, tsdf_prev1, weight_prev1);


            float tsdf_prev1_real_m = tsdf_prev1 * (use_tdist2nd ? tdist2nd_m : tranc_dist); //

            int Wrk = 1; //默认1

            if(use_tdist2nd){
                //此块内仅修改 tsdf_curr
                tsdf_curr = fmin (1.0f, sdf / tdist2nd_m);
                if(sdf < -tdist2nd_m)
                    Wrk = 0;
            }

#if 0   //v17.0, 用 volume-2nd, 代码未完成; 但是发现用一个 vol 就够了 (因为只需要一比特“长短tdist标记位”), 所以此逻辑【【【暂放弃】】】
            float tsdf_prev2nd = -123;
            int weight_prev2nd = -233;
            unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

            //volume-2nd 直接 pack, 留作备用
            if(sdf >= -tdist2nd_m){
                const int Wrk = 1;
                float tsdf_curr2nd = fmin (1.0f, sdf / tdist2nd_m); //volume-2nd 设定就是 tdist=5mm 
                float tsdf_new2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr2nd * Wrk) / (weight_prev2nd + Wrk);
                int weight_new2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                pack_tsdf(tsdf_new2nd, weight_new2nd, *pos2nd);
            }

            //v17.0: 用 snorm_pos->w 记录 tdist, 每个 vox 独立, 量纲 mm 整数
            signed char *trunc_dist_mm = &snorm_pos->w;
            if(0 == *trunc_dist_mm) //若标记位 还没初始化, 就用函数参数初始化; 否则, 用已存的标记值
                *trunc_dist_mm = int(tranc_dist * 1e3 + 0.5);
            float trunc_dist_m = trunc_dist_mm / 1e3;
#endif

            //v17.2: 把之前"背面看负值vox, 若视线前方过零点, 则此vox 不动" 
            //改为: 正面看负值 vox, 若 w 达到某阈值, 且是 "最贴近零点", 则标记, 再当背面看时, 若有标记, 则不动
            //暂用 snorm_pos->w 做标记位, 暂不折腾 w(short) @2018-1-29 00:46:48
            //signed char *neg_near_zero = &snorm_pos->w;
            bool neg_near_zero = snorm_pos->w; //初始 0->false
            const int weight_neg_th = 30; 
            if(tsdf_prev1 < 0 && weight_prev1 > weight_neg_th && !neg_near_zero)//若: 负值, 且权重达到阈值, 且标记位尚未初始化
            {
                //边缘要不要判定, 以避免边缘不平滑? 不确定, 暂不, 
                //weiFactor

                if(tsdf_prev1_real_m > 1.1 * cell_size.x){ //虽负, 但很贴近表面 (零值面) //不用 max(x,y,z); 判定阈值仅 csz.x, 刻意的 //用 projTSDF 总有误判, 所以改用 sdf_cos
                    neg_near_zero = true;
                    snorm_pos->w = 1; //neg_near_zero=true
                }
            }

            const float COS30 = 0.8660254f
                ,COS45 = 0.7071f
                ,COS60 = 0.5f
                ,COS75 = 0.258819f
                ;

            float cos_snorm_p_c = dot(snorm_prev_g, snorm_curr_g);

            //v17.X: snorm-p-c 夹角 >60°, 认为干涉, 分情况, 可能的策略: 
            //① 远端干涉, 不要动; 
            //② 近表面: a, 负冲正, 别动; b, 正冲负, ??? 【错！这样总会导致 bias, 要考虑噪声!】

            if(doDbgPrint){
                printf("snorm_prev_g.xyz: (%f, %f, %f)\n", snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z);
                printf("snorm_curr_g.xyz: (%f, %f, %f); cos_snorm_p_c: %f\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, cos_snorm_p_c);
            }

            bool isSnormPrevInit = (norm(snorm_prev_g) > 1e-8);
            if(!isSnormPrevInit && sdf < tranc_dist){ //只有在近表面 (暂用 tdist 大的那个判定), 才初始化 snorm
                (*snorm_pos).x = (int)nearbyintf(snorm_curr_g.x * CHAR_MAX); //float2char
                (*snorm_pos).y = (int)nearbyintf(snorm_curr_g.y * CHAR_MAX);
                (*snorm_pos).z = (int)nearbyintf(snorm_curr_g.z * CHAR_MAX);
            }
            else if(isSnormPrevInit && cos_snorm_p_c < COS60){ //若 norm-p 初始化过, 且 cos-n-p-c 方向差异大
                //v17.0: 所有情形下, 都做立即缩减 tdist, 先把基础流程、控制量弄通 【尝试】
                //sdf //m, 当前帧
                //tranc_dist, tranc_dist_inv //m, 当前函数传入
                //tsdf_prev1 //0~1, 

                if(!use_tdist2nd){ //若2nd 标记位没有设置, 说明第一次进来, 要: 1, 设标记位; 2, w_p=0
                    use_tdist2nd = true;

                    //float tsdf_prev1_real_m = tsdf_prev1 * tranc_dist; //之前应该都用的函数参数 tranc_dist //上面要用, 所以放到外面
                    tsdf_prev1 = tsdf_prev1_real_m / tdist2nd_m; //暂不用 fmin(1, ..), 核心是 w=0
                    //if(tsdf_prev1_real_m < -tdist2nd_m){ //若 相比 tdist2nd, 太负, 则重置, 因为本来 -tdist2nd 以外区域也是未初始化状态
                    if(tsdf_prev1_real_m < -tdist2nd_m && !neg_near_zero){ 
                        weight_prev1 = 0;
                        tsdf_prev1 = 0; //其实不必, 显式写出, 帮助阅读

                        snorm_pos->x = snorm_pos->y = snorm_pos->z = 0;
                    }
                }
                //不论 use_tdist2nd T/F, t_curr 肯定要按 td-2nd 重算:
                tsdf_curr = fmin (1.0f, sdf / tdist2nd_m);
                if(sdf < -tdist2nd_m){
                    Wrk = 0;
                    tsdf_curr = 0; //其实不必, 显式写出, 帮助阅读
                }
                else{
                    ////v17.5
                    //if(sdf > tdist2nd_m //若: 观察到远端表面
                    //    && 0 != weight_prev1) //且此 vox 重置之后又被远端更新过 //use_tdist2nd 已经 true, 不能做判定指标; 用 weight_prev1 判定
                    //    Wrk = 0; //就不再更新, 
                    
                    //v17.6.1: 简单粗暴: 若 w >th, 认为 t_p 足够稳定, 且因外层逻辑 cos(n-p-c)<COS60, 所以直接舍弃当前: w_c = 0
                    if(tsdf_curr < tsdf_prev1 && weight_prev1 > weight_neg_th) //若: c<p
                        Wrk = 0;
                }

                //v17.2: 续
                if(neg_near_zero)
                    Wrk = 0;

                //v17.7: 17.5 移到外层, 即无论 isSnormPrevInit / cos_snorm_p_c 啥样, 只要 w_c !=0, 远端一律不叠加 w @2018-2-4 11:30:27
                if(sdf > (use_tdist2nd ? tdist2nd_m : tranc_dist) //若: 观察到远端表面
                    && 0 != weight_prev1) //且此 vox 重置之后又被远端更新过 //use_tdist2nd 已经 true, 不能做判定指标; 用 weight_prev1 判定
                    Wrk = 0; //就不再更新, 

                //v17.x: 远端干涉(正冲负), 一律不要动
                //v17.x: 远端干涉(正冲负), 按视线判定, 仅当"背后"视角时, 才缩减 tdist
            }//cos-norm-p-c < COS60

            //v17.4
            if(!neg_near_zero){ //当之前不太靠近表面时, 根据 t_c 调整权重
                //↓--若: wrk 非零; 且 {t_c} < {t_p}
                if(abs(Wrk) > 1e-5 && abs(tsdf_curr) < abs(tsdf_prev1) )
                {
                    float tpc_ratio = abs(tsdf_prev1) / (abs(tsdf_curr) + 1e-2); //此块内结果必然 >1; 分母trick为了避免除零
                    //v17.4.1: 直接用 ratio 做权重:
                    Wrk = (int)fmin(10.f, tpc_ratio);

                    //v17.4.2: 用 ratio^2, 目的: 在 t_c 并不太小时, 仍然加速 t_c 影响力
                    Wrk = (int)fmin(10.f, tpc_ratio * tpc_ratio);
                }
            }

            float tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
            int weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
            
            if(doDbgPrint){
                printf("【【tsdf_prev1: %f,, weight_prev1: %d; tsdf_prev1_real_m: %f, neg_near_zero: %s\n", tsdf_prev1, weight_prev1, tsdf_prev1_real_m, neg_near_zero ? "TTT":"FFF");
                printf("【【tsdf_curr: %f,, Wrk: %d; \n", tsdf_curr, Wrk);
                printf("tsdf_new1: %f,, weight_new1: %d;;; use_tdist2nd: %d\n", tsdf_new1, weight_new1, use_tdist2nd);
            }
            //pack 前, 最后 w_new 要加上标记位:
            weight_new1 = (weight_new1 << 1) + use_tdist2nd;

            pack_tsdf (tsdf_new1, weight_new1, *pos1);

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          else{
              if(doDbgPrint)
                  printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v17

    //for v18, 为了测试 krnl 是否 thread, block 如实遍历, 结果: OK
    __global__ void
    test_kernel (int3 vxlDbg){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if(vxlDbg.x == x && vxlDbg.y == y)
            printf("dbg@test_kernel>>>xy: %d, %d\n", x, y);

    }//test_kernel

    __global__ void
    tsdf23_v18 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<ushort> depthModel,
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      //printf("tsdf23_v18, xy: %d, %d\n", x, y);
      //if(vxlDbg.x == x && vxlDbg.y == y)
      //    printf("dbg@tsdf23_v18>>>xy: %d, %d\n", x, y);

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度; //值是相对于 tranc_dist 归一化过的

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      //我的控制量们:
      short2 *pos2nd = volume2nd.ptr(y) + x;
      const float tdist2nd_m = TDIST_MIN_MM / 1e3; //v17

      //hadSeen-flag:
      bool *flag_pos = flagVolume.ptr(y) + x;
      int flag_elem_step = flagVolume.step * VOLUME_Y / sizeof(bool);

      //vray.prev
      char4 *vrayPrev_pos = vrayPrevVolume.ptr(y) + x;
      int vrayPrev_elem_step = vrayPrevVolume.step * VOLUME_Y / sizeof(char4);

      //surface-norm.prev
      char4 *snorm_pos = surfNormVolume.ptr(y) + x;
      int snorm_elem_step = surfNormVolume.step * VOLUME_Y / sizeof(char4);

      //if(vxlDbg.x == x && vxlDbg.y == y)
      //    printf("dbg@tsdf23_v18-before-for-loop>>>xy: %d, %d\n", x, y);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step,

           pos2nd += elem_step,
           flag_pos += flag_elem_step,

           vrayPrev_pos += vrayPrev_elem_step,
           snorm_pos += snorm_elem_step)
      {
        //v18.2 【已解决, 此循环内不该用 return】
        //if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
        //    && vxlDbg.x == x && vxlDbg.y == y)// && vxlDbg.z == z)
        //{   //临时测试: 总有些 vox 无法定位到, 似乎根本不进入此逻辑块; @2018-3-1 22:47:15
        //    printf("dbg@for-loop>>>xyz: %d, %d, %d\n", x, y, z);
        //}
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if(doDbgPrint)
            printf("coo.xy:(%d, %d)\n", coo.x, coo.y);

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
          //float tranc_dist_real = max(2*cell_size.x, tranc_dist * weiFactor); //截断不许太短, v11.8
          float tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v18.4: 边缘可能 w_factor=0, 

          //↓--v18.20: 放在 sdf >= -tranc_dist_real 之前, @2018-8-13 16:21:48
          const float W_FACTOR_EDGE_THRESH = 0.99f;
          bool is_curr_edge = weiFactor < W_FACTOR_EDGE_THRESH;

          float3 snorm_curr_g;
          snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];

           if(isnan(snorm_curr_g.x)){
               if(doDbgPrint)
                   printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);
 
               //return;    //内循环, 每次都要走遍 z轴, 不该 跳出
               continue;    //v18.2
           }

          snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
          snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

          float3 vray;
          vray.x = v_g_x;
          vray.y = v_g_y;
          vray.z = v_g_z;
          //float vray_norm = norm(vray);
          float3 vray_normed = normalized(vray); //单位视线向量

          float cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
          if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
              //printf("ERROR+++++++++++++++cos_vray_norm > 0");

              //假设不保证外部已正确预处理：
              snorm_curr_g.x *= -1;
              snorm_curr_g.y *= -1;
              snorm_curr_g.z *= -1;
          }

          //float sdf_cos = abs(cos_vray_norm_curr) * sdf;
          float sdf_cos = max(COS75, abs(cos_vray_norm_curr)) * sdf; //v18.3: 乘数因子不许小于 COS75

          if(doDbgPrint){
              printf("snorm_curr_g, vray_normed: [%f, %f, %f], [%f, %f, %f]\n", snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
              printf("sdf-orig: %f,, cos_vray_norm_curr: %f,, sdf_cos: %f\n", sdf, cos_vray_norm_curr, sdf_cos);
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          sdf = sdf_cos; //v18.23: 若去掉, 只用 tranc_dist_real, 也不好, 边缘尖锐, 但代价是破碎 @2018-8-23 10:03:48

          //↓--v18.17: unpack 挪到外面
          //read and unpack
          float tsdf_prev1;
          int weight_prev1;
          unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
          bool prev_always_edge = weight_prev1 % 2; //【DEL v17.1】 //v18.15: 语义变为: 是否一直处于边缘 (初值:=0:=false) @2018-3-28 15:56:33
          weight_prev1 = weight_prev1 >> 1; //去掉末位, 只是为了与 v17 保持一致, 方便测试 【【因为 tsdf23 里 w*2 了
          if(doDbgPrint)
              printf("prev_always_edge-prev: %d, is_curr_edge: %d, tsdf_prev1: %f,, weight_prev1: %d\n", prev_always_edge, is_curr_edge, tsdf_prev1, weight_prev1);

          //↓--v18.20: 移到这里了 @2018-8-14 00:01:12
          if(weight_prev1 <= 1 && is_curr_edge){ //v18.18: 略改, 因懒, 因 global_time =0 时用的 tsdf23 直接 w+1 @2018-4-10 17:27:08
              prev_always_edge = true;
          }
          else if(!is_curr_edge && prev_always_edge){
              prev_always_edge = false;

              //weight_prev1 = min(weight_prev1, 30); //策略1: w-p 直接降权到 30≈1s; //不好, 若t-p=1, 则 1*30 期望仍很大, 难修正
              //weight_prev1 = min(weight_prev1, 5);	//v18.21: 直接降权不行, 易受尖峰噪声干扰, 下面改成缓降 @2018-8-17 10:53:36
              //weight_prev1 = max(5, weight_prev1 / 2); //v18.24: 去掉所有降权, 在 box-small 上, 内部噪声反而大大减小 (仍不完善) @2018-8-23 10:35:13

          }

          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v18.4
          //if (Dp_scaled != 0 && tranc_dist_real >= sdf && sdf >= -tranc_dist_real) //meters //v18.6: 测试正值远端截断; 【结果坏：内部降噪, 外部(尤其边缘)加噪; 改放在后面, 见 v18.7
          {
            float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);

            //↓--这里废弃, 挪到外面了 v18.17
            ////read and unpack
            //float tsdf_prev1;
            //int weight_prev1;
            //unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
            //bool prev_always_edge = weight_prev1 % 2; //【DEL v17.1】 //v18.15: 语义变为: 是否一直处于边缘 (初值:=0:=false) @2018-3-28 15:56:33
            //weight_prev1 = weight_prev1 >> 1; //去掉末位, 只是为了与 v17 保持一致, 方便测试 【【因为 tsdf23 里 w*2 了
            //if(doDbgPrint)
            //    printf("prev_always_edge-prev: %d,, tsdf_prev1: %f,, weight_prev1: %d\n", prev_always_edge, tsdf_prev1, weight_prev1);

            //const int Wrk = 1;
            int Wrk = 1; //v18.5: 考虑全反射: diff_dmap + 大入射角 (用 nmap_model_g, 不用 nmap-curr 判定) @2018-3-11 11:58:55
            short diff_c_p = diff_dmap.ptr(coo.y)[coo.x]; //mm, curr-prev, +正值为当前更深
            ushort depth_prev = depthModel.ptr(coo.y)[coo.x];

            const int diff_c_p_thresh = 20; //20mm
            if(doDbgPrint)
                printf("depth_prev: %u; diff_c_p: %d\n", depth_prev, diff_c_p);

            if(depth_prev > 0 //首先要 model 上 px 有效（已初始化）
                && diff_c_p > diff_c_p_thresh){
                float3 snorm_prev_g;
                snorm_prev_g.x = nmap_model_g.ptr(coo.y)[coo.x];
                if(isnan(snorm_prev_g.x)){
                    if(doDbgPrint)
                        printf("\t+++++isnan(snorm_prev_g.x)\n");

                    Wrk = 0;
                }
                else{
                    snorm_prev_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
                    snorm_prev_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

                    float cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);
                    if(doDbgPrint)
                        printf("\tsnorm_prev_g.xyz: (%f, %f, %f), cos_vray_norm_prev: %f\n", 
                            snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, cos_vray_norm_prev);

                    if(abs(cos_vray_norm_prev) < COS75)
                        Wrk = 0;
                }
            }//if-(diff_c_p > diff_c_p_thresh)

            //v18.7: 改为: 第一次(w=0)观测到远端, 禁止初始化; 
            //结果：1, 内/外均优于 v18.6, 内部优于 v18.5, 2, 但是外部仍有部分碎片噪声; 3, 法向图(raycast结果)很难看!    【暂存】
//             if(0 == weight_prev1 && sdf > tranc_dist_real){
//                 Wrk = 0;
//             }

            //const float W_FACTOR_EDGE_THRESH = 0.99f; //v18.20: 移到前面了
            //bool is_curr_edge = weiFactor < W_FACTOR_EDGE_THRESH;

            //↓--v18.20 移到上面了
            //if(Wrk != 0){
            //    //if(0 == weight_prev1 && is_curr_edge){ //若 w-prev尚未初始化，且 curr 在边缘
            //    if(weight_prev1 <= 1 && is_curr_edge){ //v18.18: 略改, 因懒, 因 global_time =0 时用的 tsdf23 直接 w+1 @2018-4-10 17:27:08
            //        prev_always_edge = true;
            //    }
            //    else if(!is_curr_edge && prev_always_edge){
            //        prev_always_edge = false;

            //        //weight_prev1 = min(weight_prev1, 30); //策略1: w-p 直接降权到 30≈1s; //不好, 若t-p=1, 则 1*30 期望仍很大, 难修正
            //        weight_prev1 = min(weight_prev1, 5);
            //    }
            //}

            float tsdf_new1 = tsdf_prev1;
            int weight_new1 = weight_prev1;
            if(Wrk > 0)
                //&& !(!prev_always_edge && is_curr_edge && tsdf_curr > 0.99) ) //若: prev确认非边缘, curr是边缘, 且 t-c确实大, 则不更新 t, w
                //&& (prev_always_edge || !is_curr_edge || tsdf_curr <= 0.99) ) //同义, 
            {
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
            }

            if(doDbgPrint){
                //printf("【【tsdf_prev1: %f,, weight_prev1: %d; tsdf_prev1_real_m: %f, neg_near_zero: %s\n", tsdf_prev1, weight_prev1, tsdf_prev1_real_m, neg_near_zero ? "TTT":"FFF");
                printf("【【tsdf_prev1: %f,, weight_prev1: %d;\n", tsdf_prev1, weight_prev1);
                printf("【【tsdf_curr: %f,, Wrk: %d; \n", tsdf_curr, Wrk);
                printf("tsdf_new1: %f,, weight_new1: %d;;; prev_always_edge: %d\n", tsdf_new1, weight_new1, prev_always_edge);
            }

            if(weight_new1 == 0)
                tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

            //pack 前, 最后 w_new 要加上标记位:
            weight_new1 = (weight_new1 << 1) + prev_always_edge;

            pack_tsdf (tsdf_new1, weight_new1, *pos1);

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
//           else{
//               if(doDbgPrint)
//                   printf("NOT (Dp_scaled != 0 && sdf >= -tranc_dist)\n");
//           }
          //else if(Dp_scaled != 0 && sdf < -tranc_dist) { //v18.12: 此处+v18.8; 若某vox曾经看见过一眼（因噪声、全反射，持续时间短）, 
                                                            //但其后长时间不可见, 则慢慢降权(消亡); 【结果：很好, 优于 v18.11, 但有时候看见一眼未必是噪声, 要改
          else if(Dp_scaled != 0 
              && sdf < -tranc_dist &&  sdf > -4*tranc_dist   //v18.13: 改-2*tdist +v18.8, 排除 v18.12 的问题 //v18.14 改-4*tdist, 并去掉 v18.8, 就用原来 marching cubes
              //&&  sdf > -4*tranc_dist   //v18.19: else 已隐含 sdf >= -tranc_dist_real @2018-8-13 15:38:07
              && !prev_always_edge  //v18.17: 仅对非边缘执行 "-1 策略", 若总是边缘(如, 细棍子), 则不 -1 @2018-4-8 02:32:39
            )
          {
              //↓-v18.17: 挪到 if 外面了
              //float tsdf_prev1;
              //int weight_prev1;
              //unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
              //bool prev_always_edge = weight_prev1 % 2;
              //weight_prev1 = weight_prev1 >> 1; //去掉末位, 

              const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
              if(/*tsdf_prev1 >= 0.999 ||*/ //若 t_p 之前存"远端", 非近表面
                  tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
              {
                  //weight_prev1 = max(0, weight_prev1-1);
                  weight_prev1 = max(01, weight_prev1-1); //v18.22
                  //printf("^^^^^^^^^^^^in else if(Dp_scaled != 0 ------weight_prev1 < POS_VALID_WEIGHT_TH\n"); //若 POS_VALID_WEIGHT_TH置零, 则不会进入此逻辑 @2018-8-13 11:45:51

                  if(doDbgPrint){
                      printf("】】tsdf_prev1: %f,, weight_prev1-=1: %d;\n", tsdf_prev1, weight_prev1);
                  }
              }

              if(weight_prev1 == 0)
                  tsdf_prev1 = 0; //严谨点, 避免调试绘制、marching cubes意外
              weight_prev1 = (weight_prev1 << 1) + prev_always_edge;

              pack_tsdf (tsdf_prev1, weight_prev1, *pos1);
          }
        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v18

    //v19: 输出变量到文件, 外部调试
    //输出: tsdf, sn-p-c(1or3轴?), is/non-edge, diff-c-p(+dp), tdist/real-td
    //__device__ float tsdf_curr_dev;
    __device__ float sdf_orig_dev;
    __device__ float cos_dev;
    __device__ float sdf_cos_dev;
    __device__ float tdist_real_dev;
    __device__ bool snorm_oppo_dev; //opposite
    __device__ bool is_curr_edge_dev;
    __device__ bool is_non_edge_near0_dev;
    __device__ short depth_curr_dev;
    __device__ short depth_prev_dev;
    __device__ short diff_cp_dev;

    __global__ void
    tsdf23_v19 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<ushort> depthModel,
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度; //值是相对于 tranc_dist 归一化过的

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      short2* pos2nd = volume2nd.ptr (y) + x;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step
           ,pos2nd += elem_step

           )
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if(doDbgPrint)
            printf("coo.xy:(%d, %d)\n", coo.x, coo.y);

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
          float tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v18.4: 边缘可能 w_factor=0, 

          //↓--v18.20: 放在 sdf >= -tranc_dist_real 之前, @2018-8-13 16:21:48
          const float W_FACTOR_EDGE_THRESH = 0.99f;
          bool is_curr_edge_wide = weiFactor < W_FACTOR_EDGE_THRESH;
          is_curr_edge_wide = weiFactor < 0.3;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.3: 去掉 v19.6.2, 并重新用 0.3, wide 实际边 narrow @2018-9-6 15:56:23
          //is_curr_edge_wide = weiFactor < 0.1;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.9: 改 0.1 @2018-9-9 22:12:29
          bool is_curr_edge_narrow = weiFactor < 0.3; //v19.2.2: 阈值 0.6~0.8 都不行, 0.3 凑合 @2018-8-25 11:33:56

          float3 snorm_curr_g;
          snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
          float3 snorm_prev_g;
          snorm_prev_g.x = nmap_model_g.ptr(coo.y)[coo.x];

          if(isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x)){
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x), weiFactor: %f\n", weiFactor);

              //return;    //内循环, 每次都要走遍 z轴, 不该 跳出
              continue;    //v18.2
          }

          bool sn_curr_valid = false,
               sn_prev_valid = false; //如果没有 continue 跳出, 走到下面至少有一个 true

          if(!isnan(snorm_curr_g.x) ){
              snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_curr_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);
          }

          if(!isnan(snorm_prev_g.x)){
              snorm_prev_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_prev_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_prev_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("\t+++++isnan(snorm_prev_g.x), weiFactor\n", weiFactor);
          }

          float3 vray;
          vray.x = v_g_x;
          vray.y = v_g_y;
          vray.z = v_g_z;
          //float vray_norm = norm(vray);
          float3 vray_normed = normalized(vray); //单位视线向量

          float cos_vray_norm_curr = -11;
          if(sn_curr_valid){
              cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
              if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_curr_g.x *= -1;
                  snorm_curr_g.y *= -1;
                  snorm_curr_g.z *= -1;

                  cos_vray_norm_curr *= -1;
              }
          }

          float cos_vray_norm_prev = -11; //-11 作为无效标记(有效[-1~+1]); 到这里 c/p 至少有一个有效
          if(sn_prev_valid){
              cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);
              if(cos_vray_norm_prev > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_prev_g.x *= -1;
                  snorm_prev_g.y *= -1;
                  snorm_prev_g.z *= -1;

                  cos_vray_norm_prev *= -1;
              }
          }

          //v19.1: 以 c/p 较大夹角（较小 abs-cos）为准
          float cos_abs_min = min(-cos_vray_norm_curr, -cos_vray_norm_prev);
          float cos_factor = max(COS75, abs(cos_abs_min));

          //float sdf_cos = abs(cos_vray_norm_curr) * sdf;
          float sdf_cos = cos_factor * sdf; //v18.3: 乘数因子不许小于 COS75
          //float sdf_cos = max(COS75, min(abs(cos_abs_min), weiFactor) ) * sdf; //v19.5.2 并不好, 不应 @2018-9-3 01:39:02 

          if(doDbgPrint){
              printf("sn_c, sn_p, vray_normed = [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", 
                  snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, 
                  snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
              printf("sdf-orig: %f,, cos_vray_norm_curr: %f,, sdf_cos: %f\n", sdf, cos_abs_min, sdf_cos);
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          float tsdf_prev1;
          int weight_prev1;
          unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
          //↓-原 prev_always_edge  //v19.1: 语义变为: 非边缘区域(即使)标记, 当 "大入射角全反射/突变/curr边缘"时, 加锁保护, 避免噪声导致碎片 @2018-8-23 11:12:35
          //bool non_edge_near0 = weight_prev1 % 2;

          //v19.8.1: 改使用 NON_EDGE_TH 检测 “稳定” 的non-edge-area, 暂用 3bits @2018-9-11 13:13:55
          int non_edge_ccnt = weight_prev1 % VOL1_FLAG_TH;
          bool non_edge_near0 = (non_edge_ccnt + 1 == VOL1_FLAG_TH);
          weight_prev1 = weight_prev1 >> VOL1_FLAG_BIT_CNT;
          if(doDbgPrint){
              printf("non_edge_ccnt: %d, NON_EDGE_TH: %d\n", non_edge_ccnt, VOL1_FLAG_TH);
              printf("non_edge_near0: %d, is_curr_edge: %d, tsdf_prev1: %f,, weight_prev1: %d\n", non_edge_near0, is_curr_edge_wide, tsdf_prev1, weight_prev1);
          }

          const float shrink_dist_th = 0.02; //20mm

#define TSDF_ORIG 0
#if !TSDF_ORIG
          //sdf = sdf_cos; //v18~19.5

          //↓--v19.6.1 sdf & tdist 都根据是否边缘调整 @2018-9-6 01:47:59
          if(!non_edge_near0){ //必然一直是 edge //改了: 语义为: non_edge_ccnt 未达到 NON_EDGE_TH
              if(doDbgPrint)
                  printf("if(!non_edge_near0)\n");

              //v19.8.1: 
              //if(!is_curr_edge_wide && abs(sdf) < 4 * tranc_dist)
              if(!is_curr_edge_wide && abs(sdf) < shrink_dist_th)
                  non_edge_ccnt = min(7, non_edge_ccnt + 1);
              else
                  non_edge_ccnt = max(0, non_edge_ccnt - 1);

              if(doDbgPrint)
                  printf("\tAFTER-non_edge_ccnt: %d\n", non_edge_ccnt);

              //边缘上： 希望既用小 tdist 避免棱边膨大, 又不会bias导致腐蚀
              sdf = sdf_cos;
              tranc_dist_real = tranc_dist * cos_factor;
              //tranc_dist_real = tranc_dist; //v19.6.5 对比上面, 不好 @2018-9-8 01:23:46
          }
          else{ //non_edge_near0, 确认非contour
//               if(-tranc_dist * 1.2 < sdf && sdf < -tranc_dist){  【错，放弃】 @2018-9-17 11:17:59
//                   if(doDbgPrint)
//                       printf("")
//                   continue;
//               }
              //if(0 == weight_prev1 && sdf > tranc_dist * 1.2){ //v19.8.9: non-edge & wp==0 说明之前是背面 -td~-shrink_dist_th 区域, 现在 sdf>某th, 则不要 fuse @2018-9-18 17:49:31
              if(0 == weight_prev1 && sdf > +shrink_dist_th){ //v19.8.10: 背面法向误判可解决, 其他问题无解 @2018-9-18 17:51:32
                  if(doDbgPrint)
                      printf("non-edge, p<-td, c>td*factor, DONT fuse\n");

                  continue;
              }

              if(is_curr_edge_wide){ //但当前是edge
                  tranc_dist_real = tranc_dist * cos_factor;
                  //tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v19.6.2: 尝试更小的 //没大区别
              }
              else{//且当前在内部
                  if(sdf < 0) //v19.6.4, good @2018-9-8 19:39:56
                  //v19.7.4 试试去掉上面↑ 19.6.4, 变厚, 19.7.3 的破碎解决, 凹凸噪点未解决 @2018-9-10 07:01:21
                      sdf = sdf_cos;
                  tranc_dist_real = tranc_dist;
              }
          }

          if(doDbgPrint)
              printf("AFTER-sdf: %f, tranc_dist_real: %f; sdf>-td: %s\n", sdf, tranc_dist_real, sdf > -tranc_dist_real ? "sdfTTT": "sdfFFF");

#endif

          //v19.3: 尝试二类,单维度,聚类(类比混合高斯) binary-gmm
          float tsdf_prev2nd = -123;
          int weight_prev2nd = -233;
          unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

          //weight_prev2nd = weight_prev2nd >> VOL1_FLAG_BIT_CNT;
          const float min_thickness = max(0.003, cell_size.x*1.11); //薄壁最小可表征厚度>=3mm,
          float tsdf_mut_thresh = min_thickness * tranc_dist_inv; //mutation-threshold, 归一化的突变阈值, 

          //if(!is_curr_edge_wide && abs(sdf) < 4 * tranc_dist){ //or tranc_dist_real?
          //    non_edge_near0 = true;
          //}
          //↑--v19.8.1 注释掉

          float tsdf_new1 = tsdf_prev1; //放在 if之前
          int weight_new1 = weight_prev1;
          if(doDbgPrint){
              printf("【【tsdf_prev1: %f,, weight_prev1: %d;\n", tsdf_prev1, weight_prev1);
              printf("\t【【tsdf_prev2nd: %f,, weight_prev2nd: %d;\n", tsdf_prev2nd, weight_prev2nd);

              if(01){ //输出到外部文件, 调试 @2018-8-30 09:54:53
                  //输出: sdf(not tsdf), tdist/real-td, sn-p-c(1or3轴?), is/non-edge, diff-c-p(+dp),
                  //tsdf_curr_dev = fmin (1.0f, sdf * tranc_dist_inv);
                  sdf_orig_dev = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);
                  cos_dev = max(COS75, abs(cos_abs_min));
                  //sdf_cos_dev = sdf;
                  sdf_cos_dev = sdf_orig_dev * cos_dev; //改, 上面不对
                  tdist_real_dev = tranc_dist_real;

                  depth_curr_dev = Dp_scaled * 1e3; //需要 float(m)->short(mm)
                  depth_prev_dev = depthModel.ptr(coo.y)[coo.x];
                  diff_cp_dev = diff_dmap.ptr(coo.y)[coo.x];

                  is_curr_edge_dev = is_curr_edge_wide;
                  is_non_edge_near0_dev = non_edge_near0;
                  { //sn_oppo
                  int which_axis_larger = -1; //snorm_curr_g.x > snorm_curr_g.y ? 0 : (snorm_curr_g.y > snorm_curr_g.z ? 1 : 2);
                  float sn_curr_dir = 0.f;

                  float absnx = abs(snorm_curr_g.x),
                      absny = abs(snorm_curr_g.y),
                      absnz = abs(snorm_curr_g.z);
                  if(absnx > absny){
                      which_axis_larger = absnx > absnz ? 0 : 2;
                      sn_curr_dir = absnx > absnz ? snorm_curr_g.x : snorm_curr_g.z;
                  }
                  else{ //x<=y
                      which_axis_larger = absny > absnz ? 1 : 2;
                      sn_curr_dir = absny > absnz ? snorm_curr_g.y : snorm_curr_g.z;
                  }

                  float sn_prev_dir = 0.f;

                  float Fn, Fp;
                  int Wn = 0, Wp = 0;
                  switch(which_axis_larger){
                  case 2:
                      //unpack_tsdf(*(pos1 + volume1.step/sizeof(short2)), Fn, Wn);
                      unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                      unpack_tsdf (*(pos1 - elem_step), Fp, Wp); //二邻域都取出, 以防临界情况时, 某邻域未初始化; @2018-8-27 17:20:53
                      //走到这里不存在二邻域均未初始化情形
                      break;

                  case 1:
                      unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                      unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);
                      break;

                  case 0:
                      unpack_tsdf (*(pos1 + 1), Fn, Wn);
                      unpack_tsdf (*(pos1 - 1), Fp, Wp);
                      break;
                  }
                  float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);
                  if(Wn == 0) Fn = tsdf_curr;
                  if(Wp == 0) Fp = tsdf_curr; //避免无效vox

                  sn_prev_dir = Fn - Fp; //确实是 n-p, 与表面前正后负一致

                  snorm_oppo_dev = sn_curr_dir * sn_prev_dir < 0 ? true : false;
                  }
              }
          }//if-(doDbgPrint)

#if TSDF_ORIG
          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters //v18.4
          {
            float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);
            int Wrk = 1;
            tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
            weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

          }
          else if(Dp_scaled != 0 && non_edge_near0 && !is_curr_edge_wide
              //&& sdf > -4*tranc_dist_real
              && sdf > -4*tranc_dist    //good
              )
          {
              //要不要 if-- tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH 条件？待定 @2018-8-24 01:08:46
              //v19.2: 要, 
              const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
              if(tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
              {
                  weight_new1 = max(0, weight_new1-1); //v18.22
                  if(weight_new1 == 0)
                      tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

                  if(doDbgPrint)
                      printf("】】tsdf_new1: %f,, weight_new1-=1: %d;\n", tsdf_new1, weight_new1);
              }

          }//elif-(-4*td < sdf < -td)

#elif 1
          //if (Dp_scaled != 0 && sdf >= -tranc_dist) //v19.5.3 棱边膨大严重, 不可取 @2018-9-4 19:26:57
          //if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v18.4
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * cos_factor) //meters //v19.5.4 棱边膨大 @2018-9-6 00:21:27
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * (non_edge_near0 ? 1 : cos_factor)) //meters //v19.5.5 @2018-9-6 00:21:27
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v19.6
          {
            float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);
            int Wrk = 1;
            short diff_c_p = diff_dmap.ptr(coo.y)[coo.x]; //mm, curr-prev, +正值为当前更深
            ushort depth_prev = depthModel.ptr(coo.y)[coo.x];

            const int diff_c_p_thresh = 20; //20mm
            if(doDbgPrint){
                printf("【【tsdf_curr: %f,, Wrk: %d; \n", tsdf_curr, Wrk);
                printf("depth_prev: %u; diff_c_p: %d\n", depth_prev, diff_c_p);
            }

            if(non_edge_near0 && is_curr_edge_wide  //v19.2.1: edge_wide加上深度突变判定, 首先要 model 上 px 有效（已初始化）
                && depth_prev > 0
                && diff_c_p > diff_c_p_thresh
            //if(non_edge_near0 && is_curr_edge_narrow  //v19.2.2: 较难稳定融合临近边缘区域, 暂定经验阈值: 【0.3, 60】 @2018-8-24 17:18:56
            //    && weight_prev1 < 60
                )
            {
                if(doDbgPrint)
                    printf("P-non-edge & C-edge & d_prev+ & diff_cp > TH\n");

                Wrk = 0;
            }

            //v19.3
            const int weight_confid_th = 60;
            //if(
            //    //weight_prev1 > weight_confid_th
            //    tsdf_prev1 + tsdf_curr < 0
            //    //↑--v19.5: 放弃 weight_confid_th 判定, 改为: tp+tc<0 (理论上反面观测一定和为负,除非抖动)
            //    //&& abs(tsdf_curr - tsdf_prev1) > tsdf_mut_thresh
            //    //↑--v19.4: ① 放弃此 tsdf突变 thresh 判定, ② 放弃下面交换 pos1/pos2 方案, ①②都太武断; ③使用 norm 顺/逆(prev用F梯度/c用nmap)判定 @2018-8-27 16:17:18
            //    && non_edge_near0 && !is_curr_edge_wide //已累积足够可信, 且prev是内部, 且curr也非边缘
            //    )

            //↓--v19.7.1: 主要修复超薄穿透问题, 沿视线检测过零点-p>>+c @2018-9-9 19:18:37
            if(non_edge_near0 && !is_curr_edge_wide) //prev是内部, 且curr也非边缘
            {
                if(doDbgPrint){
                    printf("】】】】 non_edge_near0 && !is_curr_edge_wide; tsdf_curr: %f\n", tsdf_curr);
                    printf("\tw2nd: %d\n", weight_prev2nd);
                }

                //Wrk = min(int(abs(tsdf_prev1) / (abs(tsdf_curr)+0.01)), 10); //v19.3.1: 上限 5、10都不行, 放弃 @2018-8-26 00:48:15
#if 0   //尝试为了效率, norm顺逆仅判定单轴, 而非俩norm三轴内积 @2018-8-27 16:21:05
                int which_axis_larger = -1; //snorm_curr_g.x > snorm_curr_g.y ? 0 : (snorm_curr_g.y > snorm_curr_g.z ? 1 : 2);
                float sn_curr_dir = 0.f;

                float absnx = abs(snorm_curr_g.x),
                    absny = abs(snorm_curr_g.y),
                    absnz = abs(snorm_curr_g.z);
                if(absnx > absny){
                    which_axis_larger = absnx > absnz ? 0 : 2;
                    sn_curr_dir = absnx > absnz ? snorm_curr_g.x : snorm_curr_g.z;
                }
                else{ //x<=y
                    which_axis_larger = absny > absnz ? 1 : 2;
                    sn_curr_dir = absny > absnz ? snorm_curr_g.y : snorm_curr_g.z;
                }

                float sn_prev_dir = 0.f;

                float Fn, Fp;
                int Wn = 0, Wp = 0;
                switch(which_axis_larger){
                case 2:
                    //unpack_tsdf(*(pos1 + volume1.step/sizeof(short2)), Fn, Wn);
                    unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                    unpack_tsdf (*(pos1 - elem_step), Fp, Wp); //二邻域都取出, 以防临界情况时, 某邻域未初始化; @2018-8-27 17:20:53
                                                                 //走到这里不存在二邻域均未初始化情形
                    break;

                case 1:
                    unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                    unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);
                    break;

                case 0:
                    unpack_tsdf (*(pos1 + 1), Fn, Wn);
                    unpack_tsdf (*(pos1 - 1), Fp, Wp);
                    break;
                }
                if(0 == (Wn >> 1)) //v19.6.6: 若邻域无效, 则用 t-curr, 排除无效邻域 @2018-9-9 20:07:49
                    Fn = tsdf_curr;
                else if(0 == (Wp >> 1))
                    Fp = tsdf_curr;

                sn_prev_dir = Fn - Fp; //确实是 n-p, 与表面前正后负一致

                if(doDbgPrint)
                    printf("\t【【sn_curr_dir, sn_prev_dir: %f, %f; prod:= %f\n", sn_curr_dir, sn_prev_dir, sn_curr_dir * sn_prev_dir);

                if(sn_curr_dir * sn_prev_dir < 0){ //单轴顺逆判定

#elif 1 //v19.7.5: 改用三轴判定顺逆, 单轴不稳定
                //↓--from @tsdf23normal_hack
                const float qnan = numeric_limits<float>::quiet_NaN();
                float3 vox_normal = make_float3(qnan, qnan, qnan); //隐函数梯度做法向

                float Fn, Fp;
                int Wn = 0, Wp = 0;
                unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                unpack_tsdf (*(pos1 - elem_step), Fp, Wp);

                const int W_NBR_TH = 4;
                if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                    vox_normal.z = (Fn - Fp)/cell_size.z;

                unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);

                if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                    vox_normal.y = (Fn - Fp)/cell_size.y;

                unpack_tsdf (*(pos1 + 1), Fn, Wn);
                unpack_tsdf (*(pos1 - 1), Fp, Wp);

                if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                    vox_normal.x = (Fn - Fp)/cell_size.x;

                if(doDbgPrint){ //一般值如: (55.717842, 82.059792, 71.134979), 
                    printf("vox_normal.xyz: %f, %f, %f\n", vox_normal.x, vox_normal.y, vox_normal.z);
                    //printf("vnx==qnan: %d\n", vox_normal.x == qnan);
                }

                bool setup_cluster2 = false; //背面观测标志位 
                float cluster_margin_th = 0.8; //v19.8.8: 二类判定阈值之一 @2018-9-16 00:40:13

                if(sdf < tranc_dist_real){ //v19.7.6: 增加 sdf < td 判定 //移到最外面, 因 19.8.4 也用 @2018-9-12 18:36:55
                    //if (vox_normal.x != qnan && vox_normal.y != qnan && vox_normal.z != qnan){ //错 qnan 不能用 != 比较
                    if (!isnan(vox_normal.x) && !isnan(vox_normal.y) && !isnan(vox_normal.z)){
                        if(doDbgPrint)
                            printf("vox_normal VALID\n");

                        float vox_norm2 = dot(vox_normal, vox_normal);
                        if (vox_norm2 >= 1e-10)
                        {
                            vox_normal *= rsqrt(vox_norm2); //归一化

                            float cos_sn_c_p = dot(snorm_curr_g, vox_normal);
                            if(doDbgPrint)
                                printf("\tcos_sn_c_p: %f\n", cos_sn_c_p);

                            //if(cos_sn_c_p < COS120){
                            //if(cos_sn_c_p < COS120 && sdf < tranc_dist_real){ //v19.7.6: 增加 sdf < td 判定
                            //if(abs(tsdf_prev1 - tsdf_curr) - cos_sn_c_p > 1 && sdf < tranc_dist_real){ //19.8.7: 若 ①|tp-tc|越大 ②cos越负, 则越可能 setup-2nd @2018-9-15 13:05:01
                            if(sdf < tranc_dist_real  //19.8.8: @2018-9-15 23:37:55
                                && (abs(tsdf_prev1 - tsdf_curr) > cluster_margin_th || cos_sn_c_p < COS120) 
                                )
                            {
                                if(doDbgPrint)
                                    printf("setup_cluster2, cos_sn_c_p, %f\n", cos_sn_c_p);
                                setup_cluster2 = true; //否则默认 false: 当 ① vox_n 不够稳定 (各邻域 W<16), 或② vox_norm2 太小, 或③夹角小于 COS120
                            }
                        }
                    }
                    //↓--v19.8.4 很差, 因未考虑初始时刻, 所有 vox w都很小 @2018-9-12 18:33:50
                    else{ //说明 vox 及其邻域在观测"边缘" 很少被扫到, 尚未稳定
                        if(doDbgPrint)
                            printf("vox_normal has qnan\n");

                        //if(abs(tsdf_prev1 - tsdf_curr) > 0.5){
                        if(abs(tsdf_prev1 - tsdf_curr) > cluster_margin_th){
                             if(doDbgPrint)
                                 printf("setup_cluster2, |tp-tc|>: %f, tp: %f, tc: %f\n", abs(tsdf_prev1 - tsdf_curr), tsdf_prev1, tsdf_curr);
                             setup_cluster2 = true;
                         }

                    }
                }
                else{
                    if(doDbgPrint)
                        printf("NOT if(sdf < tranc_dist_real)\n");
                }

                if(setup_cluster2){
                //if(setup_cluster2 || 0 != weight_prev2nd){ //逻辑错, 非常差 @2018-9-17 10:50:10
#endif
                    if(doDbgPrint)
                        printf("weight_prev2nd %s 0\n", 0 == weight_prev2nd ? "====" : ">>>>");

                    if(0 == weight_prev2nd){ //vol-2nd 尚未初始化

                        pack_tsdf (tsdf_curr, Wrk, *pos2nd);
                    }
                    else{ //wp2nd > 0; 若 vol-2nd 已初始化
                        //二类聚类
                        //float tsdf_1_2_mid = (tsdf_prev1 + tsdf_prev2nd) * 0.5;

                        //↓--v19.3.2: 假设两个高斯 sigma 一样宽, 不考虑已累积权重
                        float tsdf_new_tmp;
                        int weight_new_tmp;
                        if(abs(tsdf_curr - tsdf_prev1) < abs(tsdf_curr - tsdf_prev2nd)){
                            if(doDbgPrint)
                                printf("\t<<<tcurr~~~tp1\n");

                            tsdf_new_tmp = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                            weight_new_tmp = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                            //pack 前, 最后 w_new 要加上标记位:    //放在 if之后
                            //weight_new_tmp = (weight_new1 << 1) + non_edge_near0;
                            weight_new_tmp = (weight_new_tmp << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1; 【注: 且之前误用变量 weight_new1
                            pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos1);
                        }
                        else{ //if-tc near tp2nd 
                            if(doDbgPrint)
                                printf("\t<<<tcurr~~~tp2nd\n");

                            tsdf_new_tmp = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                            weight_new_tmp = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);

                            if(doDbgPrint)
                                printf("\twnew2nd: %d; weight_confid_th: %d\n", weight_new_tmp, weight_confid_th);

                            //v19.3.2: 只要 w2>th 就交换 pos1, pos2nd 存储值
                            //|--即使结合 v19.3.3 仍不稳定, 【放弃】。 效果见: https://i.stack.imgur.com/1dd18.png @2018-8-27 00:54:48
                            //if(weight_new_tmp > weight_confid_th){

                            //v19.5.1: 尝试 w2>w1 就交换 @2018-8-28 10:36:40
                            //if(weight_new_tmp > weight_prev1){

                            //19.6.8: w1 & w2 >th, 且仅当 abs-F2 小才交换 @2018-9-9 21:52:50
                            if(weight_new_tmp > weight_prev1 && abs(tsdf_new_tmp) < abs(tsdf_prev1) ){
                                if(doDbgPrint)
                                    printf("【【【【【Exchanging... w2>w1 and |t2|<|t1|\n");

                                pack_tsdf (tsdf_prev1, weight_prev1, *pos2nd); //(t1,w1)=>pos2

                                weight_new_tmp = (weight_new_tmp << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                                pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos1); //(tnew, wnew)=>pos1
                            }
                            else{
                                pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos2nd); //v19.6.7: 忘了pack-pos2, 补上, 结合 19.5.1, 变坏了 @2018-9-9 21:45:41
                            }
                        }//if-tc near tp2nd 
                    }//if-wp2nd > 0

                    continue; //结束本次循环, 下面不执行
                }//if-(setup_cluster2)
                else{ //否则, 若同侧视角
                    if(doDbgPrint)
                        //printf("】】sn_curr_dir * sn_prev_dir >= 0\n");
                        printf("NOT setup_cluster2\n");

                    //v19.7.1: 若法线负方向, 邻域-p>>+c, 则跳过, 不融合
                    //v19.7.3: 难兼顾对抗噪声, 注掉 @2018-9-10 02:21:07
                    if(tsdf_curr > 0 && tsdf_prev1 < 0){ //-p>>+c
                        //int snorm_x_sgn = snorm_curr_g.x > 0 ? 1 : -1,
                        //    snorm_y_sgn = snorm_curr_g.y > 0 ? 1 : -1,
                        //    snorm_z_sgn = snorm_curr_g.z > 0 ? 1 : -1;
                        int snorm_x_sgn_neg = snorm_curr_g.x > 0 ? -1 : +1,
                            snorm_y_sgn_neg = snorm_curr_g.y > 0 ? -1 : +1,
                            snorm_z_sgn_neg = snorm_curr_g.z > 0 ? -1 : +1;

                        //                     unpack_tsdf (*(pos1 + elem_step), Fn, Wn); //仅参考下
                        //                     unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                        //                     unpack_tsdf (*(pos1 + 1), Fn, Wn);
                        short2* pos1_nbr_forward = pos1 
                            + snorm_z_sgn_neg * elem_step 
                            + snorm_y_sgn_neg * volume1.step/sizeof(short2) 
                            + snorm_x_sgn_neg * 1;

                        float F_nbr;
                        int W_nbr = 0;
                        unpack_tsdf(*(pos1_nbr_forward), F_nbr, W_nbr);
                        if(F_nbr > 0)
                            continue; //不融合
                    }//-p>>+c
                }//else if-not setup_cluster2
            }//if-(P & C non-edge)
            tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
            weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

            //v19.3.3: 若上面没 continue 跳出, 这里对 2nd 衰减 @2018-8-27 00:32:26
            //v19.7.2: 既然已有二类, 不要衰减, @2018-9-10 01:35:12
            //if(weight_prev2nd > 0){
            if(weight_prev2nd > 0 && weight_prev2nd < weight_confid_th){ //v19.8.6: 若 w2 不稳定时才衰减 @2018-9-14 22:37:38
                /*printf()*/
                float tsdf_new2nd = tsdf_prev2nd;
                int wnew2nd = max(weight_prev2nd - Wrk, 0);
                if(0 == wnew2nd){
                    tsdf_new2nd = 0;

                    if(doDbgPrint)
                        printf("################RESET vox2nd\n");
                }

                pack_tsdf (tsdf_new2nd, wnew2nd, *pos2nd);
            }

            if(doDbgPrint){
                printf("【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
            }

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          //else if(Dp_scaled != 0 && non_edge_near0 && !is_curr_edge_wide
          //else if(Dp_scaled != 0 && non_edge_ccnt > 0 && !is_curr_edge_wide //v19.8.2
          else if(Dp_scaled != 0 && !is_curr_edge_wide //v19.8.3
              //&& sdf > -4*tranc_dist_real
              //&& sdf > -4*tranc_dist    //good
              && sdf > -shrink_dist_th   //good
            )
          {
              //要不要 if-- tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH 条件？待定 @2018-8-24 01:08:46
              //v19.2: 要, 
              const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
              if(tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
              //if(weight_prev1 < POS_VALID_WEIGHT_TH) //v19.8.5: 去掉 tp+ 判定， +-一律衰减 @2018-9-12 20:00:24
              {
                  weight_new1 = max(0, weight_new1-1); //v18.22
                  if(weight_new1 == 0)
                      tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

                  if(doDbgPrint)
                      printf("】】tsdf_new1: %f,, W1-UNSTABLE SHRINK, weight_new1-=1: %d;\n", tsdf_new1, weight_new1);
              }

          }//elif-(-4*td < sdf < -td)
#endif  //切换 tsdf-orig & v19

          //pack 前, 最后 w_new 要加上标记位:    //放在 if之后
          //weight_new1 = (weight_new1 << 1) + non_edge_near0;
          weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
          pack_tsdf (tsdf_new1, weight_new1, *pos1);

        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v19

    __global__ void
    tsdf23_v20 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<ushort> depthModel,
        const PtrStepSz<short> diff_dmap, //v12.1
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度; //值是相对于 tranc_dist 归一化过的

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      short2* pos2nd = volume2nd.ptr (y) + x;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step
           ,pos2nd += elem_step

           )
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if(doDbgPrint)
            printf("coo.xy:(%d, %d)\n", coo.x, coo.y);

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
          float tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v18.4: 边缘可能 w_factor=0, 

          //↓--v18.20: 放在 sdf >= -tranc_dist_real 之前, @2018-8-13 16:21:48
          const float W_FACTOR_EDGE_THRESH = 0.99f;
          bool is_curr_edge_wide = weiFactor < W_FACTOR_EDGE_THRESH;
          is_curr_edge_wide = weiFactor < 0.3;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.3: 去掉 v19.6.2, 并重新用 0.3, wide 实际边 narrow @2018-9-6 15:56:23
          //is_curr_edge_wide = weiFactor < 0.1;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.9: 改 0.1 @2018-9-9 22:12:29
          bool is_curr_edge_narrow = weiFactor < 0.3; //v19.2.2: 阈值 0.6~0.8 都不行, 0.3 凑合 @2018-8-25 11:33:56

          float3 snorm_curr_g;
          snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
          float3 snorm_prev_g;
          snorm_prev_g.x = nmap_model_g.ptr(coo.y)[coo.x];

          if(isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x)){
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x), weiFactor: %f\n", weiFactor);

              //return;    //内循环, 每次都要走遍 z轴, 不该 跳出
              continue;    //v18.2
          }

          bool sn_curr_valid = false,
               sn_prev_valid = false; //如果没有 continue 跳出, 走到下面至少有一个 true

          if(!isnan(snorm_curr_g.x) ){
              snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_curr_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);
          }

          if(!isnan(snorm_prev_g.x)){
              snorm_prev_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_prev_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_prev_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("\t+++++isnan(snorm_prev_g.x), weiFactor\n", weiFactor);
          }

          float3 vray;
          vray.x = v_g_x;
          vray.y = v_g_y;
          vray.z = v_g_z;
          //float vray_norm = norm(vray);
          float3 vray_normed = normalized(vray); //单位视线向量

          float cos_vray_norm_curr = -11;
          if(sn_curr_valid){
              cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
              if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_curr_g.x *= -1;
                  snorm_curr_g.y *= -1;
                  snorm_curr_g.z *= -1;

                  cos_vray_norm_curr *= -1;
              }
          }

          float cos_vray_norm_prev = -11; //-11 作为无效标记(有效[-1~+1]); 到这里 c/p 至少有一个有效
          if(sn_prev_valid){
              cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);
              if(cos_vray_norm_prev > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_prev_g.x *= -1;
                  snorm_prev_g.y *= -1;
                  snorm_prev_g.z *= -1;

                  cos_vray_norm_prev *= -1;
              }
          }

          //v19.1: 以 c/p 较大夹角（较小 abs-cos）为准
          float cos_abs_min = min(-cos_vray_norm_curr, -cos_vray_norm_prev);
          float cos_factor = max(COS75, abs(cos_abs_min));

          //float sdf_cos = abs(cos_vray_norm_curr) * sdf;
          float sdf_cos = cos_factor * sdf; //v18.3: 乘数因子不许小于 COS75
          //float sdf_cos = max(COS75, min(abs(cos_abs_min), weiFactor) ) * sdf; //v19.5.2 并不好, 不应 @2018-9-3 01:39:02 

          if(doDbgPrint){
              printf("sn_c, sn_p, vray_normed = [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", 
                  snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, 
                  snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
              printf("sdf-orig: %f,, cos_vray_norm_curr: %f,, sdf_cos: %f\n", sdf, cos_abs_min, sdf_cos);
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          float tsdf_prev1;
          int weight_prev1;
          unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
          //↓-原 prev_always_edge  //v19.1: 语义变为: 非边缘区域(即使)标记, 当 "大入射角全反射/突变/curr边缘"时, 加锁保护, 避免噪声导致碎片 @2018-8-23 11:12:35
          //bool non_edge_near0 = weight_prev1 % 2;

          //v19.8.1: 改使用 NON_EDGE_TH 检测 “稳定” 的non-edge-area, 暂用 3bits @2018-9-11 13:13:55
          int non_edge_ccnt = weight_prev1 % VOL1_FLAG_TH;
          bool non_edge_near0 = (non_edge_ccnt + 1 == VOL1_FLAG_TH);
          weight_prev1 = weight_prev1 >> VOL1_FLAG_BIT_CNT;
          if(doDbgPrint){
              printf("non_edge_ccnt: %d, NON_EDGE_TH: %d\n", non_edge_ccnt, VOL1_FLAG_TH);
              printf("non_edge_near0: %d, is_curr_edge: %d, tsdf_prev1: %f,, weight_prev1: %d\n", non_edge_near0, is_curr_edge_wide, tsdf_prev1, weight_prev1);
          }

          const float shrink_dist_th = 0.02; //20mm
          //v19.3
          //const int WEIGHT_CONFID_TH = 60; //改放到 internal.h 中 @2018-9-30 15:03:13

          //sdf = sdf_cos; //v18~19.5

          //↓--v19.6.1 sdf & tdist 都根据是否边缘调整 @2018-9-6 01:47:59
          if(!non_edge_near0){ //必然一直是 edge //改了: 语义为: non_edge_ccnt 未达到 NON_EDGE_TH
              if(doDbgPrint)
                  printf("if(!non_edge_near0)\n");

              //v19.8.1: 
              //if(!is_curr_edge_wide && abs(sdf) < 4 * tranc_dist)
              if(!is_curr_edge_wide && abs(sdf) < shrink_dist_th)
                  non_edge_ccnt = min(7, non_edge_ccnt + 1);
              else
                  non_edge_ccnt = max(0, non_edge_ccnt - 1);

              //v20.1.16: 这里先写入, 以免后面漏掉
              int weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1; 【注: 且之前误用变量 weight_new1
              pack_tsdf(tsdf_prev1, weight_new_tmp, *pos1);

              if(doDbgPrint)
                  printf("\tAFTER-non_edge_ccnt: %d\n", non_edge_ccnt);

              //边缘上： 希望既用小 tdist 避免棱边膨大, 又不会bias导致腐蚀
              if(sdf < 0)   //v20.1.4: 试试; 逻辑: 尽管边缘, 仍仅希望 负值*cos @2018-9-21 00:14:58
                  sdf = sdf_cos;
              tranc_dist_real = tranc_dist * cos_factor;
              //tranc_dist_real = tranc_dist; //v19.6.5 对比上面, 不好 @2018-9-8 01:23:46
          }
          else{ //non_edge_near0, 确认非contour
//               if(-tranc_dist * 1.2 < sdf && sdf < -tranc_dist){  【错，放弃】 @2018-9-17 11:17:59
//                   if(doDbgPrint)
//                       printf("")
//                   continue;
//               }
              //if(0 == weight_prev1 && sdf > tranc_dist * 1.2){ //v19.8.9: non-edge & wp==0 说明之前是背面 -td~-shrink_dist_th 区域, 现在 sdf>某th, 则不要 fuse @2018-9-18 17:49:31
              if(0 == weight_prev1 && sdf > +shrink_dist_th){ //v19.8.10: 背面法向误判可解决, 其他问题无解 @2018-9-18 17:51:32
                  if(doDbgPrint)
                      printf("non-edge, p<-td, c>td*factor, DONT fuse\n");

                  continue;
              }

              if(is_curr_edge_wide){ //但当前是edge
                  //tranc_dist_real = tranc_dist * cos_factor;
                  //tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v19.6.2: 尝试更小的 //没大区别
                  tranc_dist_real = tranc_dist; //v20.1.13: P-nonedge, 但 C-edge, 则 c不稳定, 希望降低其干扰 @2018-9-26 17:03:53
              }
              else{//且当前在内部
                  if(sdf < 0) //v19.6.4, good @2018-9-8 19:39:56
                  //v19.7.4 试试去掉上面↑ 19.6.4, 变厚, 19.7.3 的破碎解决, 凹凸噪点未解决 @2018-9-10 07:01:21
                      sdf = sdf_cos;
                  tranc_dist_real = tranc_dist;
              }
          }

          if(doDbgPrint)
              printf("AFTER-sdf: %f, tranc_dist_real: %f; sdf>-td: %s\n", sdf, tranc_dist_real, sdf > -tranc_dist_real ? "sdfTTT": "sdfFFF");

          //v19.3: 尝试二类,单维度,聚类(类比混合高斯) binary-gmm
          float tsdf_prev2nd = -123;
          int weight_prev2nd = -233;
          unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

          //weight_prev2nd = weight_prev2nd >> VOL1_FLAG_BIT_CNT;
          const float min_thickness = max(0.003, cell_size.x*1.11); //薄壁最小可表征厚度>=3mm,
          float tsdf_mut_thresh = min_thickness * tranc_dist_inv; //mutation-threshold, 归一化的突变阈值, 

          float tsdf_new1 = tsdf_prev1; //放在 if之前
          int weight_new1 = weight_prev1;
          if(doDbgPrint){
              printf("【【tsdf_prev1: %f,, weight_prev1: %d;\n", tsdf_prev1, weight_prev1);
              printf("\t【【tsdf_prev2nd: %f,, weight_prev2nd: %d;\n", tsdf_prev2nd, weight_prev2nd);

              if(01){ //输出到外部文件, 调试 @2018-8-30 09:54:53
                  //暂不
              }
          }//if-(doDbgPrint)

//#if TSDF_ORIG //暂不

                    //if (Dp_scaled != 0 && sdf >= -tranc_dist) //v19.5.3 棱边膨大严重, 不可取 @2018-9-4 19:26:57
          //if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v18.4
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * cos_factor) //meters //v19.5.4 棱边膨大 @2018-9-6 00:21:27
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * (non_edge_near0 ? 1 : cos_factor)) //meters //v19.5.5 @2018-9-6 00:21:27
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v19.6
          {
            float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);
            int Wrk = 1;
            short diff_c_p = diff_dmap.ptr(coo.y)[coo.x]; //mm, curr-prev, +正值为当前更深
            ushort depth_prev = depthModel.ptr(coo.y)[coo.x];

            //const int diff_c_p_thresh = 20; //20mm    //v20 改用 shrink_dist_th @2018-9-20 21:35:40
            if(doDbgPrint){
                printf("【【tsdf_curr: %f,, Wrk: %d; \n", tsdf_curr, Wrk);
                printf("depth_prev: %u; diff_c_p: %d\n", depth_prev, diff_c_p);
            }

            //v20.1.2: 发现有必要, 加回来; 逻辑: 薄带区p非边缘,c边缘, 若深度突变,很可能是噪声(全反射,or边缘高不确定性) @2018-9-20 22:44:02
            if(non_edge_near0 && is_curr_edge_wide 
                && depth_prev > 0 && diff_c_p > shrink_dist_th * 1e3 //m2mm
                )
            {
                if(doDbgPrint)
                    printf("P-non-edge & C-edge & d_prev+ & diff_cp > TH\n");

                //if(164 == x && 60 == y && 112 == z)
                //    printf("【【@(%d, %d, %d):P-non-edge & C-edge & d_prev+ & diff_cp > TH\n\n", x, y, z);

                //Wrk = 0;
                continue;
            }
            else if(doDbgPrint)
                printf("non_edge_near0: %d, is_curr_edge_wide: %d, depth_prev>0?:= %d, diff_c_p>th?:= %d\n", non_edge_near0, is_curr_edge_wide, depth_prev > 0, diff_c_p > shrink_dist_th * 1e3);

            //↓--v19.7.1: 主要修复超薄穿透问题, 沿视线检测过零点-p>>+c @2018-9-9 19:18:37
            //if(non_edge_near0 && !is_curr_edge_wide) //prev是内部, 且curr也非边缘 【【存疑, 看看能否去掉 @2018-9-19 18:25:40  //v20.1.7: 需要有
            if(non_edge_near0) //P-non_edge //v20.1.17: 改用嵌套 if-if @2018-9-27 16:26:45
            {
                if(!is_curr_edge_wide){ //P & C-non_edge
                    if(doDbgPrint){
                        printf("】】】】 non_edge_near0 && !is_curr_edge_wide; tsdf_curr: %f\n", tsdf_curr);
                        printf("\tw2nd: %d\n", weight_prev2nd);
                    }

                    //#elif 1 //v19.7.5: 改用三轴判定顺逆, 单轴不稳定
                    //↓--from @tsdf23normal_hack
                    const float qnan = numeric_limits<float>::quiet_NaN();
                    float3 vox_normal = make_float3(qnan, qnan, qnan); //隐函数梯度做法向
                    //const int W_NBR_CONFID_TH = 4;
                    const int W_NBR_CONFID_TH = WEIGHT_CONFID_TH / 2; //@v20.1.5 nbr-confid-th 太小不稳定, 改了有效果 @2018-9-24 17:23:27

                    float Fn, Fp;
                    int Wn = 0, Wp = 0;

#if 0   //v19 旧的
                    unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                    unpack_tsdf (*(pos1 - elem_step), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.z = (Fn - Fp)/cell_size.z;

                    unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                    unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.y = (Fn - Fp)/cell_size.y;

                    unpack_tsdf (*(pos1 + 1), Fn, Wn);
                    unpack_tsdf (*(pos1 - 1), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.x = (Fn - Fp)/cell_size.x;

#else   //v20.1.3 改为, 若某nbr 无效, 则用中心点本身
                    unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                    unpack_tsdf (*(pos1 - elem_step), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.z = (Fn - Fp)/(2*cell_size.z);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.z = (Fn - tsdf_prev1)/cell_size.z;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.z = (tsdf_prev1 - Fp)/cell_size.z;

                    unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                    unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.y = (Fn - Fp)/(2*cell_size.y);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.y = (Fn - tsdf_prev1)/cell_size.y;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.y = (tsdf_prev1 - Fp)/cell_size.y;

                    unpack_tsdf (*(pos1 + 1), Fn, Wn);
                    unpack_tsdf (*(pos1 - 1), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.x = (Fn - Fp)/(2*cell_size.x);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.x = (Fn - tsdf_prev1)/cell_size.x;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.x = (tsdf_prev1 - Fp)/cell_size.x;

#endif

                    if(doDbgPrint){ //一般值如: (55.717842, 82.059792, 71.134979), 
                        printf("vox_normal.xyz: %f, %f, %f\n", vox_normal.x, vox_normal.y, vox_normal.z);
                        //printf("vnx==qnan: %d\n", vox_normal.x == qnan);
                    }

                    bool setup_cluster2 = false; //背面观测标志位 

                    //if(0 < weight_prev2nd){ //w2≠0 本身作为一个 FLAG
                    if(0 < weight_prev2nd && weight_prev2nd < WEIGHT_CONFID_TH){ //v20.1.13: 改为: ①w2∈某区间, 就要一直 c2=true; 且②: 达到阈值后 w2 不清空  @2018-9-26 20:13:53
                        if(doDbgPrint)
                            printf("setup_cluster2; ++++w2>0\n");
                        setup_cluster2 = true;
                    }
                    else if(!isnan(vox_normal.x) && !isnan(vox_normal.y) && !isnan(vox_normal.z)){
                        //else if(sdf < shrink_dist_th && !isnan(vox_normal.x) && !isnan(vox_normal.y) && !isnan(vox_normal.z)){ //v20.1.10: 单看 cos120, 当远表面的norm干扰时, 影响cos判定, 所以改: 先判定 sdf<th @2018-9-25 20:20:14
                        if(doDbgPrint)
                            printf("vox_normal VALID\n");

                        if(sdf < shrink_dist_th){ //v20.1.10: 改放内部
                            float vox_norm2 = dot(vox_normal, vox_normal);
                            //if (vox_norm2 >= 1e-10)
                            //if (vox_norm2 >= 10) //v20.1.11: 因求 vox-norm 时用的 m量纲分母, 值应较大, 所以增大此阈值, 避免噪声 @2018-9-25 21:59:07
                            if (vox_norm2 >= 2500) //v20.1.12: 因为是 norm^2, 希望至少存在某轴 >50, 因为假设 cell-size 2mm, 希望至少存在某轴 nbr-diff > 0.1 @2018-9-26 11:10:20
                            {
                                vox_normal *= rsqrt(vox_norm2); //归一化

                                float cos_sn_c_p = dot(snorm_curr_g, vox_normal);
                                if(doDbgPrint)
                                    printf("\tcos_sn_c_p: %f\n", cos_sn_c_p);

                                //if(cos_sn_c_p < COS120)
                                if(cos_sn_c_p < COS150) //v20.1.11: 并且改用 cos150 @2018-9-26 10:07:37
                                {
                                    if(doDbgPrint)
                                        printf("setup_cluster2, cos_sn_c_p <cos120: %f\n", cos_sn_c_p);

                                    //if(164 == x && 60 == y && 112 == z)
                                    //    printf("【【@(%d, %d, %d): setup_cluster2, cos_sn_c_p <cos120: %f\n", x, y, z, cos_sn_c_p);

                                    setup_cluster2 = true; //否则默认 false: 当 ① vox_n 不够稳定 (各邻域 W<16), 或② vox_norm2 太小, 或③夹角小于 COS120
                                }
                                else if(weight_prev1 >= WEIGHT_CONFID_TH && weight_prev2nd >= WEIGHT_CONFID_TH){
                                    //v20.1.15: 即便 cos > COS150, 但因 w1, w2 都 confid了, 所以直接 true, 走分类流程 @2018-9-26 22:02:26
                                    if(doDbgPrint)
                                        printf("setup_cluster2, w1_w2_confid: %f\n", cos_sn_c_p);

                                    setup_cluster2 = true;
                                }
                                else{
                                    if(doDbgPrint)
                                        printf("\tNO-setup_cluster2, cos_sn_c_p>cos120\n");
                                }
                            }
                        }
                    }//if-vox_normal VALID
                    //else if(weight_prev1 > weight_confid_th){ //若 w1 稳定了, 但是 vox 邻域在观测"边缘" 很少被扫到, 尚未稳定, 且 w2==0
                    else if(weight_prev1 > WEIGHT_CONFID_TH && abs(tsdf_curr - tsdf_prev1) > 0.5){ //v20.1.14: 改: 20.1.13 之后, w∈某区间策略之后, 这里也要对应改 @2018-9-26 21:56:46
                        if(doDbgPrint)
                            printf("setup_cluster2; w1>th & w2==0 & vox_normal INVALID: xyz: %d, %d, %d\n", x, y, z);

                        //if(164 == x && 60 == y && 112 == z)
                        //    printf("【【@(%d, %d, %d): setup_cluster2; w1>th & w2==0 & vox_normal INVALID\n", x, y, z);

                        setup_cluster2 = true;
                    }

                    float tsdf_new_tmp;
                    int weight_new_tmp;

                    if(setup_cluster2){
                        //                     if(0 == weight_prev2nd){ //vol-2nd 尚未初始化
                        // 
                        //                         pack_tsdf (tsdf_curr, Wrk, *pos2nd);
                        //                     }
                        //                     else{ //wp2nd > 0; 若 vol-2nd 已初始化
                        //                     }
                        if(weight_prev2nd < WEIGHT_CONFID_TH){ //v20.1.1: 策略改为: 只要
                            if(doDbgPrint)
                                printf("w2<CONFID\n");

                            tsdf_new_tmp = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                            weight_new_tmp = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                            pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos2nd);

                            if(doDbgPrint)
                                printf("\twnew2nd: %d; weight_confid_th: %d\n", weight_new_tmp, WEIGHT_CONFID_TH);
                        }
                        else{ //w2>th
                            if(doDbgPrint)
                                printf("w2>=CONFID\n");

                            if(abs(tsdf_curr - tsdf_prev1) < abs(tsdf_curr - tsdf_prev2nd)){
                                if(doDbgPrint)
                                    printf("\t<<<tcurr~~~tp1; w1+++\n");

                                tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                                weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                                weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                                pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
                            }
                            else{ //if-tc near tp2nd 
                                if(doDbgPrint)
                                    printf("\t<<<tcurr~~~tp2nd; w2+++\n");

                                tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                                weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                                pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);
                            }

                            //默认 vol1 存的 abs-t 应较小, 若不然, 则交换:
                            if(abs(tsdf_prev2nd) < abs(tsdf_prev1)){
                                if(doDbgPrint)
                                    printf("【【【【【Exchanging... |t2|<|t1|: tp1:= %f, tp2:= %f\n", tsdf_prev1, tsdf_prev2nd);

                                pack_tsdf (tsdf_prev1, weight_prev1, *pos2nd); //(t1,w1)=>pos2
                                //↑--v20.1.13: 重新启用交换策略, 暂放弃置零策略 @2018-9-26 20:19:31

                                weight_new_tmp = (weight_prev2nd << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                                pack_tsdf (tsdf_prev2nd, weight_new_tmp, *pos1); //(tnew, wnew)=>pos1
                            }

                            //pack_tsdf (0, 0, *pos2nd); //pos2, 置零
                        }//elif-(w2 >= confid_th)

                        //if(164 == x && 60 == y && 112 == z)
                        //    printf("【【@(%d, %d, %d): if-setup_cluster2\n", x, y, z);

                        //continue; //v20.1.4: 只要setup_cluster2, 无论 w2++ or 交换, 都跳出, 不管 w1    @2018-9-21 00:08:55
                        //↑-v20.1.16: 去掉 continue, 分支分别写, 逻辑清晰一点 @2018-9-27 10:42:12
                    }//if-(setup_cluster2)
                    else{ //NO-setup_cluster2
                        //v20.1.16: 重复代码:
                        tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                        weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                        if(doDbgPrint){
                            printf("P_C_non_edge_setup_cluster2_false:【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
                        }

                        weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                        pack_tsdf (tsdf_new1, weight_new1, *pos1);

                    }
                }//if=!is_curr_edge_wide
                else{ //P-non_edge & C-edge
                    //DONT FUSE!! @2018-9-27 16:29:52
                    if(doDbgPrint)
                        printf("P_non_edge_C_edge_DONT_FUSE\n");
                }
            }//if=non_edge_near0
            else{ //P-edge
                //v20.1.16: w2不稳, 或 w1/2都稳定且视线夹角较小(因既是edge又视线夹角大时,噪声太大)
                //if(weight_prev2nd < weight_confid_th || (weight_prev2nd >= weight_confid_th && cos_abs_min >= COS60)){
//                 if(weight_prev2nd < weight_confid_th || cos_abs_min >= COS60){ //合并简化
//                     tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
//                     weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
//                     if(doDbgPrint){
//                         printf("【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
//                     }
// 
//                     weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
//                     pack_tsdf (tsdf_new1, weight_new1, *pos1);
//                 }
//                 else{
//                     if(doDbgPrint){
//                         printf("P_C_edge && w2>th && cos<cos60\n");
//                     }
//                 }

                //v20.1.17: 前面逻辑改成 if=non_edge_near0 单判定
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
                if(doDbgPrint){
                    printf("P_edge: 【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
                }

                weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                pack_tsdf (tsdf_new1, weight_new1, *pos1);
            }
            //if(164 == x && 60 == y && 112 == z)
            //    printf("【【@(%d, %d, %d):tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", x, y, z, tsdf_new1, weight_new1, non_edge_near0);

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
          //else if(Dp_scaled != 0 && !is_curr_edge_wide //v19.8.3
          else if(Dp_scaled != 0 //v20.1.8: curr-edge 判定移到下面, 因为 w2-shrink 不需要此判定 @2018-9-25 11:02:09
              //&& sdf > -4*tranc_dist_real
              //&& sdf > -4*tranc_dist    //good
              && sdf > -shrink_dist_th   //good
              )
          {
              //要不要 if-- tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH 条件？待定 @2018-8-24 01:08:46
              //v19.2: 要, 
              const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
              //if(tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
              if(!is_curr_edge_wide && tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //v20.1.8: curr-edge 判定移到这里 @2018-9-25 11:02:51
                  //if(weight_prev1 < POS_VALID_WEIGHT_TH) //v19.8.5: 去掉 tp+ 判定， +-一律衰减 @2018-9-12 20:00:24
              {
                  weight_new1 = max(0, weight_new1-1); //v18.22
                  if(weight_new1 == 0)
                      tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

                  if(doDbgPrint)
                      printf("】】tsdf_new1: %f,, W1-UNSTABLE SHRINK, weight_new1-=1: %d;\n", tsdf_new1, weight_new1);

                  //v20.1.16:
                  weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                  pack_tsdf (tsdf_new1, weight_new1, *pos1);
              }

//               if(weight_prev2nd > 0){ //即 setup_cluster2 仍在
//                   //v20.1.6: v20 连续 w2 赋值的策略下, 若负值临界区域, 扫不到就断断续续递增 w2, 转过面之后会造成误判, 因此改成断续就衰减 @2018-9-25 00:16:04
//                   //weight_prev2nd = max(0, weight_prev2nd-1);
//                   weight_prev2nd --;
//                   if(weight_prev2nd == 0)
//                       tsdf_prev2nd = 0;
//                   pack_tsdf(tsdf_prev2nd, weight_prev2nd, *pos2nd);
// 
//                   if(doDbgPrint)
//                       printf("】】】w2-SHRINK, tsdf_prev2nd: %f, weight_prev2nd-1: %d\n", tsdf_prev2nd, weight_prev2nd);

              if(0 < weight_prev2nd && weight_prev2nd < WEIGHT_CONFID_TH){ //即 setup_cluster2 仍在

                  //v20.1.9: 上面负值断续就衰减,不好用, 改成: 即使断续, 只要在薄带范围内, 就以 t2=-1 递增 w2  @2018-9-25 17:15:59
                  tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + (-1)) / (weight_prev2nd + 1);
                  weight_prev2nd = min (weight_prev2nd + 1, Tsdf::MAX_WEIGHT);
                  pack_tsdf(tsdf_prev2nd, weight_prev2nd, *pos2nd);
                  if(doDbgPrint)
                      printf("】】】w2+++, sdf<-TH, tsdf_prev2nd: %f, weight_prev2nd: %d\n", tsdf_prev2nd, weight_prev2nd);
              }
          }//elif-(-4*td < sdf < -td)

          //weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
          //pack_tsdf (tsdf_new1, weight_new1, *pos1);
          //↑--v20.1.16: 之前放外面, 主要是为了 任何条件下确保 non_edge_ccnt 写入; 但因逻辑难看清, 暂去掉, 在各分支内操作 @2018-9-27 11:28:19

        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v20


    __global__ void
    tsdf23_v21 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume1, 
        PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, const PtrStepSz<unsigned char> incidAngleMask,
        const PtrStep<float> nmap_curr_g, const PtrStep<float> nmap_model_g,
        /*↑--实参顺序: volume2nd, flagVolume, surfNormVolume, incidAngleMask, nmap_g,*/
        const PtrStep<float> weight_map, //v11.4
        const PtrStepSz<ushort> depth_not_scaled, //v21.4: 用没 scale的 dmap (mm)
        const PtrStepSz<ushort> depthModel,
        const PtrStepSz<short> diff_dmap, //v12.1
        const PtrStepSz<ushort> depthModel_vol2, //v21.6.0: 核心是, 在 vol2 上 rcast 求 sdf_forward, 而非用 dcurr 瞬时结果 @2018-11-29 17:54:01
        //const PtrStepSz<ushort> rc_flag_map, //v21
        const PtrStepSz<short> rc_flag_map, //v21
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size
        , int3 vxlDbg)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;
      float pendingFixThresh = cell_size.x * tranc_dist_inv * 3; //v13.4+ 用到: 暂定 3*vox 厚度; //值是相对于 tranc_dist 归一化过的

      short2* pos1 = volume1.ptr (y) + x;
      int elem_step = volume1.step * VOLUME_Y / sizeof(short2);

      short2* pos2nd = volume2nd.ptr (y) + x;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos1 += elem_step
           ,pos2nd += elem_step

           )
      {
        bool doDbgPrint = false;
        if(x > 0 && y > 0 && z > 0 //参数默认 000, 做无效值, 所以增加此检测
            && vxlDbg.x == x && vxlDbg.y == y && vxlDbg.z == z)
            doDbgPrint = true;

        float v_z_real = (v_z + Rcurr_inv.data[2].z * z_scaled);
        float inv_z = 1.0f / v_z_real;
        if(doDbgPrint)
            printf("inv_z:= %f\n", inv_z);

        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if(doDbgPrint)
            printf("coo.xy:(%d, %d)\n", coo.x, coo.y);

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters
          ushort Dp_not_scaled = depth_not_scaled.ptr (coo.y)[coo.x]; //mm

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if(doDbgPrint){
              printf("Dp_scaled, sdf, tranc_dist, %f, %f, %f\n", Dp_scaled, sdf, tranc_dist);
          }

          float weiFactor = weight_map.ptr(coo.y)[coo.x];
          float tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v18.4: 边缘可能 w_factor=0, 

          //↓--v18.20: 放在 sdf >= -tranc_dist_real 之前, @2018-8-13 16:21:48
          const float W_FACTOR_EDGE_THRESH = 0.99f;
          bool is_curr_edge_wide = weiFactor < W_FACTOR_EDGE_THRESH;
          //is_curr_edge_wide = weiFactor < 0.3;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.3: 去掉 v19.6.2, 并重新用 0.3, wide 实际边 narrow @2018-9-6 15:56:23
          //is_curr_edge_wide = weiFactor < 0.1;//W_FACTOR_EDGE_THRESH;
          //↑--v19.6.9: 改 0.1 @2018-9-9 22:12:29
          //↑--v21.3.7: wide 就用 0.99    @2018-10-21 23:20:40
          //is_curr_edge_wide = weiFactor < 0.3;//W_FACTOR_EDGE_THRESH;
          //↑--v21.3.7.d: 因 v21.3.7.bc 在 cup6down1c 不够平滑, 为了排查错误, 尝试此 @2018-11-19 20:03:32
          bool is_curr_edge_narrow = weiFactor < 0.3; //v19.2.2: 阈值 0.6~0.8 都不行, 0.3 凑合 @2018-8-25 11:33:56

          float3 snorm_curr_g;
          snorm_curr_g.x = nmap_curr_g.ptr(coo.y)[coo.x];
          float3 snorm_prev_g;
          snorm_prev_g.x = nmap_model_g.ptr(coo.y)[coo.x];

          if(isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x)){
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x) && isnan(snorm_prev_g.x), weiFactor: %f\n", weiFactor);

              //return;    //内循环, 每次都要走遍 z轴, 不该 跳出
              continue;    //v18.2
          }

          bool sn_curr_valid = false,
               sn_prev_valid = false; //如果没有 continue 跳出, 走到下面至少有一个 true

          if(!isnan(snorm_curr_g.x) ){
              snorm_curr_g.y = nmap_curr_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_curr_g.z = nmap_curr_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_curr_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("+++++++++++++++isnan(snorm_curr_g.x), weiFactor: %f\n", weiFactor);
          }

          if(!isnan(snorm_prev_g.x)){
              snorm_prev_g.y = nmap_model_g.ptr(coo.y + depthScaled.rows)[coo.x];
              snorm_prev_g.z = nmap_model_g.ptr(coo.y + 2 * depthScaled.rows)[coo.x];

              sn_prev_valid = true;
          }
          else{
              if(doDbgPrint)
                  printf("\t+++++isnan(snorm_prev_g.x), weiFactor\n", weiFactor);
          }

          float3 vray;
          vray.x = v_g_x;
          vray.y = v_g_y;
          vray.z = v_g_z;
          //float vray_norm = norm(vray);
          float3 vray_normed = normalized(vray); //单位视线向量

          float cos_vray_norm_curr = -11;
          if(sn_curr_valid){
              cos_vray_norm_curr = dot(snorm_curr_g, vray_normed);
              if(cos_vray_norm_curr > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_curr_g.x *= -1;
                  snorm_curr_g.y *= -1;
                  snorm_curr_g.z *= -1;

                  cos_vray_norm_curr *= -1;
              }
          }

          float cos_vray_norm_prev = -11; //-11 作为无效标记(有效[-1~+1]); 到这里 c/p 至少有一个有效
          if(sn_prev_valid){
              cos_vray_norm_prev = dot(snorm_prev_g, vray_normed);
              if(cos_vray_norm_prev > 0){ //当做assert, 要求必须: 夹角>90°, 即法向必须朝向相机这头
                  //printf("ERROR+++++++++++++++cos_vray_norm > 0");

                  //假设不保证外部已正确预处理：
                  snorm_prev_g.x *= -1;
                  snorm_prev_g.y *= -1;
                  snorm_prev_g.z *= -1;

                  cos_vray_norm_prev *= -1;
              }
          }

          //v19.1: 以 c/p 较大夹角（较小 abs-cos）为准
          float cos_abs_min = min(-cos_vray_norm_curr, -cos_vray_norm_prev);
          float cos_factor = max(COS75, abs(cos_abs_min));

          //float sdf_cos = abs(cos_vray_norm_curr) * sdf;
          float sdf_cos = cos_factor * sdf; //v18.3: 乘数因子不许小于 COS75
          //float sdf_cos = max(COS75, min(abs(cos_abs_min), weiFactor) ) * sdf; //v19.5.2 并不好, 不应 @2018-9-3 01:39:02 

          if(doDbgPrint){
              printf("sn_c, sn_p, vray_normed = [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", 
                  snorm_curr_g.x, snorm_curr_g.y, snorm_curr_g.z, 
                  snorm_prev_g.x, snorm_prev_g.y, snorm_prev_g.z, vray_normed.x, vray_normed.y, vray_normed.z);
              printf("sdf-orig: %f,, cos_vray_norm_curr: %f,, sdf_cos: %f\n", sdf, cos_abs_min, sdf_cos);
              printf("\ttranc_dist_real, weiFactor: %f, %f\n", tranc_dist_real, weiFactor);
          }

          float tsdf_prev1;
          int weight_prev1;
          unpack_tsdf (*pos1, tsdf_prev1, weight_prev1);
          //↓-原 prev_always_edge  //v19.1: 语义变为: 非边缘区域(即使)标记, 当 "大入射角全反射/突变/curr边缘"时, 加锁保护, 避免噪声导致碎片 @2018-8-23 11:12:35
          //bool non_edge_near0 = weight_prev1 % 2;

          //v19.8.1: 改使用 NON_EDGE_TH 检测 “稳定” 的non-edge-area, 暂用 3bits @2018-9-11 13:13:55
          const int non_edge_ccnt_max = VOL1_FLAG_TH - 1;
          int non_edge_ccnt = weight_prev1 % VOL1_FLAG_TH;
          bool non_edge_near0 = (non_edge_ccnt == non_edge_ccnt_max);
          weight_prev1 = weight_prev1 >> VOL1_FLAG_BIT_CNT;
          if(doDbgPrint){
              printf("non_edge_ccnt: %d, NON_EDGE_TH: %d\n", non_edge_ccnt, VOL1_FLAG_TH);
              printf("non_edge_near0: %d, is_curr_edge: %d, tsdf_prev1: %f,, weight_prev1: %d\n", non_edge_near0, is_curr_edge_wide, tsdf_prev1, weight_prev1);
          }

          int Wrk = 1; //v21.6.0.a: 移到前面, 根据 C-in/edge 变化   @2018-12-2 19:36:08

          const float shrink_dist_th = 0.02; //20mm
          //v19.3
          //const int WEIGHT_CONFID_TH = 60; //改放到 internal.h 中 @2018-9-30 15:03:13

          //sdf = sdf_cos; //v18~19.5

          //↓--v19.6.1 sdf & tdist 都根据是否边缘调整 @2018-9-6 01:47:59
          if(!non_edge_near0){ //必然一直是 edge //改了: 语义为: non_edge_ccnt 未达到 NON_EDGE_TH
              if(doDbgPrint)
                  printf("if(!non_edge_near0)\n");

              //v19.8.1: 
              //if(!is_curr_edge_wide && abs(sdf) < 4 * tranc_dist)
              //if(!is_curr_edge_wide && abs(sdf) < shrink_dist_th)
              if(!is_curr_edge_wide && abs(sdf_cos) < shrink_dist_th) //v21.2.2: 改用 sdf_cos 作比较 @2018-10-8 10:58:57
                  non_edge_ccnt = min(non_edge_ccnt_max, non_edge_ccnt + 1);
              else
                  non_edge_ccnt = max(0, non_edge_ccnt - 1);

              //v20.1.16: 这里先写入, 以免后面漏掉
              int weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1; 【注: 且之前误用变量 weight_new1
              pack_tsdf(tsdf_prev1, weight_new_tmp, *pos1);

              if(doDbgPrint)
                  printf("\tAFTER-non_edge_ccnt: %d\n", non_edge_ccnt);
          }
          else{ //non_edge_near0, 确认非contour
              //               if(-tranc_dist * 1.2 < sdf && sdf < -tranc_dist){  【错，放弃】 @2018-9-17 11:17:59
              //                   if(doDbgPrint)
              //                       printf("")
              //                   continue;
              //               }
              //if(0 == weight_prev1 && sdf > tranc_dist * 1.2){ //v19.8.9: non-edge & wp==0 说明之前是背面 -td~-shrink_dist_th 区域, 现在 sdf>某th, 则不要 fuse @2018-9-18 17:49:31
              if(0 == weight_prev1 && sdf > +shrink_dist_th){ //v19.8.10: 背面法向误判可解决, 其他问题无解 @2018-9-18 17:51:32
                  if(doDbgPrint)
                      printf("non-edge, p<-td, c>td*factor, DONT fuse\n");

                  continue;
              }
          }

#if 0   //< v21, 之前希望 sdf<0 时用 cos; >0时用 orig, 保持薄壁, 但是可能有偏差
          if(!non_edge_near0){
              //边缘上： 希望既用小 tdist 避免棱边膨大, 又不会bias导致腐蚀
              if(sdf < 0)   //v20.1.4: 试试; 逻辑: 尽管边缘, 仍仅希望 负值*cos @2018-9-21 00:14:58
                  sdf = sdf_cos;
              //tranc_dist_real = tranc_dist * cos_factor;
              //tranc_dist_real = tranc_dist; //v19.6.5 对比上面, 不好 @2018-9-8 01:23:46
          }
          else{ //non_edge_near0, 确认非contour

              if(is_curr_edge_wide){ //但当前是edge
                  //tranc_dist_real = tranc_dist * cos_factor;
                  //tranc_dist_real = max(0.3, weiFactor) * tranc_dist; //v19.6.2: 尝试更小的 //没大区别
                  tranc_dist_real = tranc_dist; //v20.1.13: P-nonedge, 但 C-edge, 则 c不稳定, 希望降低其干扰 @2018-9-26 17:03:53
              }
              else{//且当前在内部
                  if(sdf < 0) //v19.6.4, good @2018-9-8 19:39:56
                  //v19.7.4 试试去掉上面↑ 19.6.4, 变厚, 19.7.3 的破碎解决, 凹凸噪点未解决 @2018-9-10 07:01:21
                      sdf = sdf_cos;
                  tranc_dist_real = tranc_dist;
              }
          }
#elif 0 //v21.3.4 回到最初, sdf乘以cos, tdist用原本的     @2018-10-10 20:22:38
          sdf = sdf_cos; //v18~19.5

//           if(!(non_edge_near0 && is_curr_edge_wide)){ //v21.4.2 当 P-in & C-edge 时, sdf用 orig; 其他时候才 *cos    @2018-10-15 02:47:54
//               sdf = sdf_cos; 
//           }

          tranc_dist_real = tranc_dist; //td-real 本来与 weiMap 相关, 这里不管 wmap, 用 orig
#elif 0 //v21.3.5
          if(is_curr_edge_wide){ //C-edge
              //sdf.orig
              //tdist∝ wmap

              //↓-v21.3.6: 稍改,      @2018-10-18 11:00:31
              //逻辑: C-edge 情形下, ① P-in, 就 td∝wmap, 小截断, 避免棱边膨大 ② P-edge, 说明一直 edge, 就 td.orig, 使纯边缘不要变成尖锐锯齿
              if(!non_edge_near0) //P-edge:
                  tranc_dist_real = tranc_dist; //td.orig
          }
          else{ //C-in
              sdf = sdf_cos; 
              tranc_dist_real = tranc_dist; //td.orig
          }
#elif 10 //v21.3.7   重新理一遍
          if(is_curr_edge_narrow){ //C-edge-narrow,
              if(non_edge_near0){ //P-in-wide
                  //sdf.orig
                  //tdist∝ wmap
              }
              else{ //P-edge-wide
                  //sdf.orig
                  tranc_dist_real = tranc_dist; //td.orig

                  //↓--v21.3.7.c: 只要sdf_cos在 td 范围之内, 就用 sdf_cos; 目的: 使 cup6down1c 在 v21.3.7 也能把手连续不断开 @2018-11-19 16:43:29
                  //结果: 能达到"把手连续不断开", 但是表面不如 v21.3.7 光滑
                  if(sdf_cos < tranc_dist)
                  if(sdf > tranc_dist && sdf_cos < tranc_dist) //v21.3.7.f: 前面 v21.3.7.c 逻辑瑕疵: 即P&C-edge时, 希望 sdf_cos 是"替补", 而不是优先用
                      sdf = sdf_cos;
              }
          }
          else{ //C-in-narrow,
              //if(non_edge_near0) //v21.3.8: 仅当 P-in, 采用 sdf-cos
                  sdf = sdf_cos; 
              //↑--v21.3.9：试验：完全去掉 sdf-cos, 只用 orig, 会怎样？ 答：
              tranc_dist_real = tranc_dist; //td.orig

              Wrk = 2; //v21.6.0.a
          }
#elif 0 //v21.3.11: 尝试改成回到 v21.3.4, 一律用 sdf-cos, 但 td 策略等同 v21.3.7 根据 P-in & C-edge 调整
          //考虑: cup6down1c 数据, 试图让"杯子把手" 连续, 但 cluster策略的 if(abs(sdf_forward) < abs(sdf_backward)) 策略不行, 见 mesh-v21.3.11.ply
          //v21.3.12: 启用 v21.3.10, 即 cluster 再暂改成 if(1) //不如 v21.3.10, 【【【【说明 v21.3.11 并不好 @2018-10-22 17:12:12
          sdf = sdf_cos; 
          if(non_edge_near0 && is_curr_edge_narrow){
              //tdist∝ wmap
          }
          else{
              tranc_dist_real = tranc_dist; //td.orig
          }
#endif

          if(doDbgPrint)
              printf("AFTER-sdf: %f, tranc_dist_real: %f; sdf>-td: %s\n", sdf, tranc_dist_real, sdf > -tranc_dist_real ? "sdfTTT": "sdfFFF");

          //v19.3: 尝试二类,单维度,聚类(类比混合高斯) binary-gmm
          float tsdf_prev2nd = -123;
          int weight_prev2nd = -233;
          unpack_tsdf (*pos2nd, tsdf_prev2nd, weight_prev2nd);

          //weight_prev2nd = weight_prev2nd >> VOL1_FLAG_BIT_CNT;
          //const float min_thickness = max(0.003, cell_size.x*1.11); //薄壁最小可表征厚度>=3mm,
          //float tsdf_mut_thresh = min_thickness * tranc_dist_inv; //mutation-threshold, 归一化的突变阈值, 

          float tsdf_new1 = tsdf_prev1; //放在 if之前
          int weight_new1 = weight_prev1;

          int rc_flag = rc_flag_map.ptr(coo.y)[coo.x]; //v21.1.1  //ushort->int @2018-10-3 22:19:06
                                                          //v21.3.3 改成了 正负深度值, rcast ->+ 过零点才终止 @2018-10-14 20:11:56
          //if(rc_flag != 0)
          //    printf("【【rc_flag: %u; coo.xy: (%d, %d)\n", rc_flag, coo.x, coo.y);

          if(doDbgPrint){
              printf("【【rc_flag: %d; coo.xy: (%d, %d)\n", rc_flag, coo.x, coo.y);
              printf("【【tsdf_prev1: %f,, weight_prev1: %d;\n", tsdf_prev1, weight_prev1);
              printf("\t【【tsdf_prev2nd: %f,, weight_prev2nd: %d;\n", tsdf_prev2nd, weight_prev2nd);

              if(01){ //输出到外部文件, 调试 @2018-8-30 09:54:53
                  //暂不
              }
          }//if-(doDbgPrint)

//#if TSDF_ORIG //暂不

                    //if (Dp_scaled != 0 && sdf >= -tranc_dist) //v19.5.3 棱边膨大严重, 不可取 @2018-9-4 19:26:57
          //if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v18.4
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * cos_factor) //meters //v19.5.4 棱边膨大 @2018-9-6 00:21:27
          //if (Dp_scaled != 0 && sdf >= -tranc_dist * (non_edge_near0 ? 1 : cos_factor)) //meters //v19.5.5 @2018-9-6 00:21:27
          if (Dp_scaled != 0 && sdf >= -tranc_dist_real) //meters //v19.6
          {
            float tsdf_curr = fmin (1.0f, sdf * tranc_dist_inv);
            //float tsdf_curr = fmin (1.0f, sdf_cos * tranc_dist_inv); //v21.5.10: buddhahead 数据上, 佛头因为P-edge-wide, C-edge-narrow, sdf负值较大, 导致偏差, 产生凹凸噪声 //其他数据并不好 (如 box-small数据)
            //int Wrk = 1; //v21.6.0.a: 改移到前面, if 外面
            short diff_c_p = diff_dmap.ptr(coo.y)[coo.x]; //mm, curr-prev, +正值为当前更深
            ushort depth_prev = depthModel.ptr(coo.y)[coo.x];

            //const int diff_c_p_thresh = 20; //20mm    //v20 改用 shrink_dist_th @2018-9-20 21:35:40
            if(doDbgPrint){
                printf("【【tsdf_curr: %f,, Wrk: %d; \n", tsdf_curr, Wrk);
                printf("depth_prev: %u; diff_c_p: %d\n", depth_prev, diff_c_p);
            }

#if 0   //v21.1: 继承自 v20.x, 按 p-in/edge & c-in/edge 分四类, 仅在 p&c-in 块内处理 setup_c2, @2018-10-4 21:59:00
            //v20.1.2: 发现有必要, 加回来; 逻辑: 薄带区p非边缘,c边缘, 若深度突变,很可能是噪声(全反射,or边缘高不确定性) @2018-9-20 22:44:02
            if(non_edge_near0 && is_curr_edge_wide 
                && depth_prev > 0 && diff_c_p > shrink_dist_th * 1e3 //m2mm
                )
            {
                if(doDbgPrint)
                    printf("P-non-edge & C-edge & d_prev+ & diff_cp > TH\n");

                //if(164 == x && 60 == y && 112 == z)
                //    printf("【【@(%d, %d, %d):P-non-edge & C-edge & d_prev+ & diff_cp > TH\n\n", x, y, z);

                //Wrk = 0;
                continue;
            }
            else if(doDbgPrint)
                printf("non_edge_near0: %d, is_curr_edge_wide: %d, depth_prev>0?:= %d, diff_c_p>th?:= %d\n", non_edge_near0, is_curr_edge_wide, depth_prev > 0, diff_c_p > shrink_dist_th * 1e3);

            //↓--v19.7.1: 主要修复超薄穿透问题, 沿视线检测过零点-p>>+c @2018-9-9 19:18:37
            //if(non_edge_near0 && !is_curr_edge_wide) //prev是内部, 且curr也非边缘 【【存疑, 看看能否去掉 @2018-9-19 18:25:40  //v20.1.7: 需要有
            if(non_edge_near0) //P-non_edge //v20.1.17: 改用嵌套 if-if @2018-9-27 16:26:45
            {
                if(!is_curr_edge_wide){ //P & C-non_edge
                    if(doDbgPrint){
                        printf("】】】】 non_edge_near0 && !is_curr_edge_wide; tsdf_curr: %f\n", tsdf_curr);
                        printf("\tw2nd: %d\n", weight_prev2nd);
                    }

                    //#elif 1 //v19.7.5: 改用三轴判定顺逆, 单轴不稳定
                    //↓--from @tsdf23normal_hack
                    const float qnan = numeric_limits<float>::quiet_NaN();
                    float3 vox_normal = make_float3(qnan, qnan, qnan); //隐函数梯度做法向
                    //const int W_NBR_CONFID_TH = 4;
                    const int W_NBR_CONFID_TH = WEIGHT_CONFID_TH / 2; //@v20.1.5 nbr-confid-th 太小不稳定, 改了有效果 @2018-9-24 17:23:27

                    float Fn, Fp;
                    int Wn = 0, Wp = 0;

#if 0   //v19 旧的
                    unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                    unpack_tsdf (*(pos1 - elem_step), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.z = (Fn - Fp)/cell_size.z;

                    unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                    unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.y = (Fn - Fp)/cell_size.y;

                    unpack_tsdf (*(pos1 + 1), Fn, Wn);
                    unpack_tsdf (*(pos1 - 1), Fp, Wp);

                    if ((Wn >> VOL1_FLAG_BIT_CNT) > W_NBR_TH && (Wp >> VOL1_FLAG_BIT_CNT) > W_NBR_TH) 
                        vox_normal.x = (Fn - Fp)/cell_size.x;

#else   //v20.1.3 改为, 若某nbr 无效, 则用中心点本身
                    unpack_tsdf (*(pos1 + elem_step), Fn, Wn);
                    unpack_tsdf (*(pos1 - elem_step), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.z = (Fn - Fp)/(2*cell_size.z);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.z = (Fn - tsdf_prev1)/cell_size.z;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.z = (tsdf_prev1 - Fp)/cell_size.z;

                    unpack_tsdf (*(pos1 + volume1.step/sizeof(short2) ), Fn, Wn);
                    unpack_tsdf (*(pos1 - volume1.step/sizeof(short2) ), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.y = (Fn - Fp)/(2*cell_size.y);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.y = (Fn - tsdf_prev1)/cell_size.y;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.y = (tsdf_prev1 - Fp)/cell_size.y;

                    unpack_tsdf (*(pos1 + 1), Fn, Wn);
                    unpack_tsdf (*(pos1 - 1), Fp, Wp);
                    Wn >>= VOL1_FLAG_BIT_CNT;
                    Wp >>= VOL1_FLAG_BIT_CNT;

                    if (Wn > W_NBR_CONFID_TH && Wp > W_NBR_CONFID_TH) 
                        vox_normal.x = (Fn - Fp)/(2*cell_size.x);
                    else if(Wn > W_NBR_CONFID_TH) //but Wp <th
                        vox_normal.x = (Fn - tsdf_prev1)/cell_size.x;
                    else if(Wp > W_NBR_CONFID_TH) //but Wn <th
                        vox_normal.x = (tsdf_prev1 - Fp)/cell_size.x;

#endif

                    if(doDbgPrint){ //一般值如: (55.717842, 82.059792, 71.134979), 
                        printf("vox_normal.xyz: %f, %f, %f\n", vox_normal.x, vox_normal.y, vox_normal.z);
                        //printf("vnx==qnan: %d\n", vox_normal.x == qnan);
                    }

                    bool setup_cluster2 = false; //背面观测标志位 

                    int rc_flag = rc_flag_map.ptr(coo.y)[coo.x]; //v21.1.1  //ushort->int @2018-10-3 22:19:06

                    if(0 < weight_prev2nd){ //w2≠0 本身作为一个 FLAG
                    //if(0 < weight_prev2nd && weight_prev2nd < WEIGHT_CONFID_TH){ //v20.1.13: 改为: ①w2∈某区间, 就要一直 c2=true; 且②: 达到阈值后 w2 不清空  @2018-9-26 20:13:53
                        if(doDbgPrint)
                            printf("setup_cluster2; ++++w2>0\n");
                        setup_cluster2 = true;
                    }
                    else if(rc_flag == 127){
                        if(doDbgPrint)
                            printf("setup_cluster2; rc_flag_127\n");

                        setup_cluster2 = true;
                    }

                    float tsdf_new_tmp;
                    int weight_new_tmp;

                    if(setup_cluster2){
                        //                     if(0 == weight_prev2nd){ //vol-2nd 尚未初始化
                        // 
                        //                         pack_tsdf (tsdf_curr, Wrk, *pos2nd);
                        //                     }
                        //                     else{ //wp2nd > 0; 若 vol-2nd 已初始化
                        //                     }
                        if(doDbgPrint)
                            printf("w2_le_CONFID\n");

                        tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                        weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                        pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);

                        if(doDbgPrint)
                            printf("\twnew2nd: %d; weight_confid_th: %d\n", weight_prev2nd, WEIGHT_CONFID_TH);
                        
                        if(weight_prev2nd > WEIGHT_CONFID_TH){ //v20.1.1: 策略改为: 只要
                            if(doDbgPrint)
                                printf("w2_gt_CONFID\n");

                            if(abs(tsdf_curr - tsdf_prev1) < abs(tsdf_curr - tsdf_prev2nd)){
                                if(doDbgPrint)
                                    printf("\t<<<tcurr_near_tp1; w1+++\n");

                                tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                                weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                                weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                                pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
                            }
                            else{ //if-tc near tp2nd 
                                if(doDbgPrint)
                                    printf("\t<<<tcurr_near_tp2nd; w2+++\n");

                                tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                                weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                                pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);
                            }

                            //默认 vol1 存的 abs-t 应较小, 若不然, 则交换:
                            if(abs(tsdf_prev2nd) < abs(tsdf_prev1)){
                                if(doDbgPrint)
                                    printf("【【【【【Exchanging... |t2|<|t1|: tp1:= %f, tp2:= %f\n", tsdf_prev1, tsdf_prev2nd);

                                pack_tsdf (tsdf_prev1, weight_prev1, *pos2nd); //(t1,w1)=>pos2
                                //↑--v20.1.13: 重新启用交换策略, 暂放弃置零策略 @2018-9-26 20:19:31

                                weight_new_tmp = (weight_prev2nd << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                                pack_tsdf (tsdf_prev2nd, weight_new_tmp, *pos1); //(tnew, wnew)=>pos1
                            }

                            //pack_tsdf (0, 0, *pos2nd); //pos2, 置零
                        }//elif-(w2 >= confid_th)

                        //if(164 == x && 60 == y && 112 == z)
                        //    printf("【【@(%d, %d, %d): if-setup_cluster2\n", x, y, z);

                        //continue; //v20.1.4: 只要setup_cluster2, 无论 w2++ or 交换, 都跳出, 不管 w1    @2018-9-21 00:08:55
                        //↑-v20.1.16: 去掉 continue, 分支分别写, 逻辑清晰一点 @2018-9-27 10:42:12
                    }//if-(setup_cluster2)
                    else{ //NO-setup_cluster2
                        //v20.1.16: 重复代码:
                        tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                        weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
                        weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                        pack_tsdf (tsdf_new1, weight_new1, *pos1);

                        if(doDbgPrint){
                            printf("P_C_non_edge_setup_cluster2_false:【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
                        }
                    }
                }//P & C-non_edge
                else{ //P-non_edge & C-edge
                    //DONT FUSE!! @2018-9-27 16:29:52
                    if(doDbgPrint)
                        printf("P_non_edge_C_edge_DONT_FUSE\n");
                }
            }//if=non_edge_near0
            else{ //P-edge
                //v20.1.16: w2不稳, 或 w1/2都稳定且视线夹角较小(因既是edge又视线夹角大时,噪声太大)
                //if(weight_prev2nd < weight_confid_th || (weight_prev2nd >= weight_confid_th && cos_abs_min >= COS60)){
//                 if(weight_prev2nd < weight_confid_th || cos_abs_min >= COS60){ //合并简化
//                     tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
//                     weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
//                     if(doDbgPrint){
//                         printf("【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
//                     }
// 
//                     weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
//                     pack_tsdf (tsdf_new1, weight_new1, *pos1);
//                 }
//                 else{
//                     if(doDbgPrint){
//                         printf("P_C_edge && w2>th && cos<cos60\n");
//                     }
//                 }

                //v20.1.17: 前面逻辑改成 if=non_edge_near0 单判定
                tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
                weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                pack_tsdf (tsdf_new1, weight_new1, *pos1);
                if(doDbgPrint){
                    printf("P_edge: 【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
                }
            }
#elif 0 //v21.2: 改成: setup-c2 仅以 rcFlag 为判定指标 @2018-10-4 21:59:48
        //且 setup_cluster2 语义改为: c2 保护标志位, 即 w2<th时, 仅 w2++, 不分类; w2>th后才分类

//             if(!non_edge_near0){ //P-edge: 包括: 1, P确实一直边缘, 2, 或 P的 ccnt 尚未累积达到阈值
//                 tsdf_new1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
//                 weight_new1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
//                 weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
//                 pack_tsdf (tsdf_new1, weight_new1, *pos1);
//                 if(doDbgPrint){
//                     printf("P_edge: 【【tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", tsdf_new1, weight_new1, non_edge_near0);
//                 }
//             }
//             else if(is_curr_edge_wide){ //P-in, C-edge
//                 if(doDbgPrint)
//                     printf("P_non_edge_C_edge_DONT_FUSE\n");
//                 continue;
//             }

            //if(non_edge_near0 && is_curr_edge_wide){ //P-in, C-edge
            //    continue;

            //    if(doDbgPrint)
            //        printf("P_non_edge_C_edge_DONT_FUSE\n");
            //}
            //↑--此逻辑不对, 不要随便跳过, 会导致 in/edge 交界处, tsdf 偏差!! 实际前面 v19.8.10已经正确处理过了    @2018-10-9 00:29:37

            bool setup_cluster2 = false; //背面观测标志位 

            float tsdf_new_tmp;
            int weight_new_tmp;

            //int rc_flag = rc_flag_map.ptr(coo.y)[coo.x]; //v21.1.1  //ushort->int @2018-10-3 22:19:06
            //if(doDbgPrint)
            //    printf("【【rc_flag: %u", rc_flag);

            //if(rc_flag == 127 || rc_flag == 128 || rc_flag == 130){ //v21.1~3
            rc_flag + Dp_scaled * 1000
            if(rc_flag)
//                 if(doDbgPrint)
//                     printf("setup_cluster2; rc_flag_127\n");

                //setup_cluster2 = true;

                if(non_edge_near0){ //P-in
                    tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                    weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                    pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);

                    if(doDbgPrint)
                        printf("rc_flag_127-P_in-【【w2++; t2_new: %f, w2_new: %d\n", tsdf_prev2nd, weight_prev2nd);

                    if(weight_prev2nd > WEIGHT_CONFID_TH 
                        && abs(tsdf_prev2nd) < abs(tsdf_prev1)) //默认 vol1 存的 abs-t 应较小, 若不然, 则交换 exch
                    {

                        if(doDbgPrint)
                            printf("【【【【【Exchanging... w2_gt_TH && |t2|<|t1|: tp1:= %f, tp2:= %f\n", tsdf_prev1, tsdf_prev2nd);

                        pack_tsdf (tsdf_prev1, weight_prev1, *pos2nd); //(t1,w1)=>pos2
                        //↑--v20.1.13: 重新启用交换策略, 暂放弃置零策略 @2018-9-26 20:19:31

                        weight_new_tmp = (weight_prev2nd << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                        pack_tsdf (tsdf_prev2nd, weight_new_tmp, *pos1); //(tnew, wnew)=>pos1
                    }

                }
                else{ //P-edge
                    tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                    weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                    weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                    pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);

                    if(doDbgPrint)
                        printf("rc_flag_127-P_edge-【w1++; t1_new: %f, w1_new: %d\n", tsdf_prev1, weight_new_tmp);

                }
            }
            else{ //rc_flag 顺, 或新, 或已经交换
                if(weight_prev2nd == 0){ //w2 无, 说明 rc 顺
                    tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                    weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                    weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                    pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);

                    if(doDbgPrint)
                        printf("【rc_flag_forward-w2_eq_0: %d, pack1, t1_new: %f, w1_new: %d\n", rc_flag, tsdf_prev1, weight_prev1);
                }
                else{ //判定 w2>0 or >th？？【【待定
                    if(doDbgPrint){
                        printf("【【rc_flag_forward-w2_gt_0: %d, cluster\n", rc_flag);

                        if(weight_prev2nd < WEIGHT_CONFID_TH)
                            printf("【【【【【【【w2_le_TH: %d, ERRORRRRRRRRRRRRRR\n", weight_prev2nd);
                    }

                    if(abs(tsdf_curr - tsdf_prev1) < abs(tsdf_curr - tsdf_prev2nd)){
                        if(doDbgPrint)
                            printf("\t<<<tcurr_near_tp1; w1+++\n");

                        tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                        weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                        weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                        pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
                    }
                    else{ //if-tc near tp2nd 
                        if(doDbgPrint)
                            printf("\t<<<tcurr_near_tp2nd; w2+++\n");

                        tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                        weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                        pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);
                    }
                }
            }
            //↓-【放弃，改仅以 rc_flag 为判定标志, 当 w2>th, exch 之后, rc_flag 自然重置 @2018-10-5 17:03:20
//             //if(0 < weight_prev2nd){ //w2≠0 本身作为一个 FLAG
//             if(0 < weight_prev2nd && weight_prev2nd < WEIGHT_CONFID_TH){ //v20.1.13: 改为: ①w2∈某区间, 就要一直 c2=true; 且②: 达到阈值后 w2 不清空  @2018-9-26 20:13:53
//                 if(doDbgPrint)
//                     printf("setup_cluster2; ++++w2>0\n");
// 
//                 setup_cluster2 = true;
//             }
//             else if(weight_prev2nd > WEIGHT_CONFID_TH){
//                 if(doDbgPrint)
//                     printf("DONT_setup_cluster2_w2_gt_CONFID\n");
// 
//                 setup_cluster2 = false;
//             }


//             if(setup_cluster2){
//                 if(doDbgPrint)
//                     printf("w2_le_CONFID--pack2\n");
// 
//                 tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
//                 weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
//                 pack_tsdf (tsdf_new_tmp, weight_prev2nd, *pos2nd);
//                 continue;
//             }
//             else{} 

//             if(abs(tsdf_curr - tsdf_prev1) < abs(tsdf_curr - tsdf_prev2nd)){
//                 if(doDbgPrint)
//                     printf("\t<<<tcurr_near_tp1; w1+++\n");
// 
//                 tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
//                 weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
// 
//                 weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
//                 pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
//             }
//             else{ //if-tc near tp2nd 
//                 if(doDbgPrint)
//                     printf("\t<<<tcurr_near_tp2nd; w2+++\n");
// 
//                 tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
//                 weight_prev2nd = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
//                 pack_tsdf (tsdf_prev2nd, weight_prev2nd, *pos2nd);
//             }

#elif 1 //v21.4.1   @2018-10-14 19:32:37
            int fuse_method = FUSE_KF_AVGE; //默认原策略
            if(rc_flag >= 0){// && depthScaled > 0) //去掉, 因为此大 if内本来就 d_curr >0 
                if(doDbgPrint)
                    printf("FUSE_KF_AVGE:=rc_flag_ge_0: rc_flag: %d\n", rc_flag);

                //v21.5.11: 以 long1-jlong-c为例, 内部碎片原因: F~-0.8, w1~30 时, 斜视大正值, 变成正值, 且权重增大, 后面无法衰减 @2018-10-27 21:34:08
                float cos_p_c = dot(snorm_curr_g, snorm_prev_g);
                float cos_abs_min3 = min(abs(cos_p_c), cos_abs_min);
                if(tsdf_prev1 < -0.5 && non_edge_near0 && cos_abs_min3 < COS80){
                    fuse_method = FUSE_IGNORE_CURR;
                }
            }
            else{ //if-rc<0 背面过零点了
                //float diff = Dp_scaled + rc_flag * 1e-3; //in meters //rc 本来是 short 毫米尺度
                float diff = (Dp_not_scaled + rc_flag) * 1e-3; //meters, 改用非 scale 值
                if(doDbgPrint)
                    printf("diff: %f, Dp_not_scaled + rc_flag: %d, %d\n", diff, Dp_not_scaled, rc_flag);

                //const float MIN_THICKNESS = 0.010; //-cell_size.x; //in meters
                const float MIN_THICKNESS = 0.001; //v21.3.7.e: 改 ≈0；//cup6down1c 把手断掉跟这里有关吗？ @2018-11-20 01:26:13
                //const float MIN_THICKNESS = shrink_dist_th; //v21.5.7.1.2: 不好, 太薄, 穿透
                //const float MIN_THICKNESS = cell_size.x * 4; //v21.5.7.1.3
                if(diff >= MIN_THICKNESS){ //c 甚至比 p背面过零点还要深, 
#if 10   //dc>inv_dp 时,一律 IGN
                    fuse_method = FUSE_IGNORE_CURR;
                    if(doDbgPrint)
                        printf("diff_gt_MIN_THICKNESS==FUSE_IGNORE_CURR\n");
#elif 1 //v21.3.7.g: 加判定: 仅当 P-in 时, 才根据diff 做 IGN, 若一直P-edge, 就仍默认 AVG, 确保细物体边缘光滑, 如cup6down1c @2018-11-20 11:16:11
                    if(non_edge_near0){
                        fuse_method = FUSE_IGNORE_CURR;
                        if(doDbgPrint)
                            printf("diff_gt_MIN_THICKNESS==FUSE_IGNORE_CURR--if_P_in\n");
                    }
#elif 0   //v21.4.5 实验: IGN 再换成 AVG
                    //fuse_method = FUSE_IGNORE_CURR;
                    //↑--v21.4.5: 去掉 diff>TH 时的 IGN 判定 @2018-10-16 10:16:14
                    //边缘 +>- 搞定, 但薄片孔洞, 破损重新出现; 怎么区分边缘、薄片？【未解决】 @2018-10-16 10:17:05
                    if(doDbgPrint)
                        //printf("diff_gt_MIN_THICKNESS==FUSE_IGNORE_CURR\n");
                        printf("diff_gt_MIN_THICKNESS==still_FUSE_KF_AVGE\n");
#elif 0 //实验: 因边缘 和斜视(2D img上也检出为边缘) 难以区分, 所以 c+ 在背面视角如何判定很难, 要么破碎, 要么边缘难圆滑
                    //① 思路一: 以 WEIGHT_CONFID_TH 区分, 粗暴, 可能有错 @2018-10-16 19:59:07
                    //不行, 例子: box [149,54,96,1] 调试点, 在 f464 第一次 diff>TH, 但 w1 已经很高, 只会判定 IGN, 所以 w1 不可靠
                    if(weight_prev1 > WEIGHT_CONFID_TH){ //类似 21.4.4
                        fuse_method = FUSE_IGNORE_CURR;
                        if(doDbgPrint)
                            printf("diff_gt_MIN_THICKNESS==FUSE_IGNORE_CURR\n");
                    }
                    else{
                        if(doDbgPrint)
                            printf("diff_gt_MIN_THICKNESS==still_FUSE_KF_AVGE\n");
                    }
#elif 1 //尝试在 dmap_model 上做边缘检测, 然后 2D dist-transform, 再判定 3D vxl 到边缘真实距离, 作为


#endif
                }
                else if(diff <= - shrink_dist_th){ //c 很浅
                    fuse_method = FUSE_KF_AVGE;
                    if(doDbgPrint)
                        printf("FUSE_KF_AVGE:=diff∈(-∞, -shrink_dist_th]\n");
                }
                else{ //diff∈(-shrink_dist_th, MIN_THICKNESS)
                    fuse_method = FUSE_CLUSTER;
                    if(doDbgPrint)
                        printf("FUSE_CLUSTER:=diff∈(-shrink_dist_th, MIN_THICKNESS)\n");
                }
            }//if-rc<0 

            //v21.4.3: rc 判定之后, 再以 in-edge 判定边缘情况, 直接覆盖前面决策
            //效果很差, 因为 C-edge 判定, 在临界区域, 非常不稳定!!    @2018-10-15 05:40:33
//             if(!non_edge_near0){ //若 一直边缘
//                 if(doDbgPrint){
//                     if(FUSE_IGNORE_CURR == fuse_method)
//                         printf("FUSE_IGNORE_CURR_2_FUSE_KF_AVGE+++++++++++++++\n");
//                     else if(FUSE_CLUSTER == fuse_method)
//                         printf("FUSE_CLUSTER_2_FUSE_KF_AVGE+++++++++++++++\n");
//                 }
// 
//                 fuse_method = FUSE_KF_AVGE;
//             }
//             else if(is_curr_edge_wide){ //P-in, C-edge
//                 if(weight_prev1 > WEIGHT_CONFID_TH){ //v21.4.4
//                     if(doDbgPrint){
//                         if(FUSE_KF_AVGE == fuse_method)
//                             printf("FUSE_KF_AVGE_2_FUSE_IGNORE_CURR==========\n");
//                         else if(FUSE_CLUSTER == fuse_method)
//                             printf("FUSE_CLUSTER_2_FUSE_IGNORE_CURR==========\n");
//                     }
//                     fuse_method = FUSE_IGNORE_CURR;
//                 }
//                 else{
//                     if(doDbgPrint)
//                         printf("P_in_C_edge_weight_prev1: %f\n", weight_prev1);
// 
//                     fuse_method = FUSE_KF_AVGE;
//                 }
//             }

            float tsdf_new_tmp;
            int weight_new_tmp;

            if(FUSE_KF_AVGE == fuse_method){
                tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
            }
            if(FUSE_CLUSTER == fuse_method){
                //纯以 rc 正负深度判定, 不以 tsdf 值正负、大小判定 @2018-10-14 21:11:25
                //float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm); //参考
                //float v_g_z_mm = v_g_z * 1e3; //mm, z本是米, 转毫米也未必整数
                float v_c_z_mm = v_z_real * 1e3; //mm, z本是米, 转毫米也未必整数
#if 10   //v21.6 之前
                float sdf_forward = Dp_not_scaled - v_c_z_mm;
#elif 1   //v21.6.x
                ushort Dp_not_scaled_vol2 = depthModel_vol2.ptr (coo.y)[coo.x]; //mm
                float sdf_forward = Dp_not_scaled_vol2 - v_c_z_mm;
#endif

                float sdf_backward = v_c_z_mm + rc_flag;
                if(doDbgPrint){
                    printf("FUSE_CLUSTER:=sdf_forward: %f, sdf_backward: %f; //Dp_not_scaled, v_c_z_mm, rc_flag: %f, %f, %f\n", sdf_forward, sdf_backward, (float)Dp_not_scaled, v_c_z_mm, (float)rc_flag);
                    printf("Dp_not_scaled, v_c_z_mm, rc_flag: %u, %f, %d\n", Dp_not_scaled, v_c_z_mm, rc_flag);
                }

#if 01   //v21.3.7 ~ v21.3.12 “怂二分类”, 
                //在 cup6up1c 数据上, 不如 v21.5.3 @2018-10-23 13:54:48
                //if(abs(sdf_forward) < abs(sdf_backward)){ //离正面近
                if(abs(sdf_forward) < abs(sdf_backward) || !non_edge_near0){ //离正面近 //v21.3.7.b: 同时要判定是否边缘	@2018-11-19 09:47:57
                //if(1){ //v21.3.10: 在 v21.3.7 基础上, 暂去掉 cluster 逻辑;   //在马克杯 cup6down1c 数据上不错, 杯子把手连贯 (v21.3.7 断了) @2018-10-22 15:42:13
                    //仍是 w1++
                    //Wrk = 2; //v21.4.3: cluster 时提高权重, 使 c更容易冲掉 p   @2018-10-15 04:12:16
                    //↑-并不好, 类似潜在导致 bias 的操作都不好 @2018-10-21 21:24:38
                    tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                    weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                    weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                    pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);
                    if(doDbgPrint)
                        printf("FUSE_CLUSTER__sdf_forward<<<<<sdf_backward\n");
                }
                else{ //离背面近
                    if(doDbgPrint)
                        printf("FUSE_CLUSTER__ignore...\n");
                }
#elif 1 //v21.5.x
                if(weight_prev2nd < WEIGHT_CONFID_TH){
                    tsdf_new_tmp = (tsdf_prev2nd * weight_prev2nd + tsdf_curr * Wrk) / (weight_prev2nd + Wrk);
                    weight_new_tmp = min (weight_prev2nd + Wrk, Tsdf::MAX_WEIGHT);
                    pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos2nd);
                    if(doDbgPrint)
                        printf("【【w2++:: t2_new, w2_new: %f, %d, \n", tsdf_new_tmp, weight_new_tmp);
                }
                else{
#if 0   //v21.5.1
                    float floor_p = 0.2; //记号: flrp
#elif 0 //v21.5.2: 若曾 P-in, 则钟形曲线形如 "_/\_"
                    float floor_p = 1.0; //记号: flrp
                    if(non_edge_near0)
                        floor_p = 0.2;

                    float p1 = max(floor_p, 1 - abs(tsdf_prev1)),
                        p2 = max(floor_p, 1 - abs(tsdf_prev2nd));

#elif 0 //v21.5.4: 仍然要考虑 (t)sdf forward、backward 关系, 以决定 flrp 下限
                    float floor_p = 0.0; //记号: flrp
                    //if(abs(sdf_forward) < abs(sdf_backward)){ //不好, 如果 vox 甚至在 sdf-back 之后(即此vox在背面曾在表面外) 怎么办?
                    if(sdf_forward > sdf_backward){
                        floor_p = 1.0;
                        if(doDbgPrint)
                            printf("sdf_forward_near, floor_p_UUUUUUUUUUUUP\n");
                    }
                    float p1 = max(floor_p, 1 - abs(tsdf_prev1)),
                        p2 = max(floor_p, 1 - abs(tsdf_prev2nd));
#elif 0 //v21.5.5   @2018-10-24 10:49:44
                    float p1 = 1.f,
                        p2 = 1.f;
                    if(!is_curr_edge_narrow){ //C-in-narrow 时候, 才操纵 p2; 一直边缘时候(比如细把手、铅笔) 不操纵 p2
                        if(sdf_forward < sdf_backward){
                            //p2 = 0.1; //暂不置零
                            p2 = 0.f; //置零

                        }
                    }
#elif 0 //v21.5.6: 考虑 ①, 不用 C-in/edge 判定; ②, sdf forw-backw 判定时, 也应影响 p1    @2018-10-24 19:18:14
                    float p1 = 1.f,
                        p2 = 1.f;
                    float floor_p = 0.0; //记号: flrp

                    if(sdf_forward < sdf_backward){ //离背面近
                        p2 = floor_p;
                    }
                    else{ //离正面近
                        if(tsdf_prev1 < -0.5) //且 t1 确实“很负”, 可能较大“背面观测”成分
                            p1 = floor_p;
                    }

#elif 1 //v21.6.0: 核心是, 在 vol2 上 rcast 求 sdf_forward, 而非用 dcurr 瞬时结果 @2018-11-29 17:54:01
                    float sdf_forward_le0 = min(0.f, sdf_forward); //取非正 <=0 部分
                    float sdf_backward_le0 = min(0.f, sdf_backward);
                    float p1 = sdf_backward_le0 / (sdf_forward_le0 + sdf_backward_le0); //trick: 分子分母都负值, 所以无所谓
                    float p2 = 1 - p1;
#endif


                    //tsdf_new_tmp = (weight_prev1 * p1 * tsdf_prev1 + weight_prev2nd * p2 * tsdf_prev2nd) / (weight_prev1 + weight_prev2nd);
                    tsdf_new_tmp = (weight_prev1 * p1 * tsdf_prev1 + weight_prev2nd * p2 * tsdf_prev2nd) / (weight_prev1 * p1 + weight_prev2nd * p2); //v21.5.3: 上面分母有错, 导致偏差, 凹凸噪声; 【已修正，已验证 @2018-10-23 12:26:20
                    //weight_new_tmp = min(weight_prev1 + weight_prev2nd, Tsdf::MAX_WEIGHT);
                    weight_new_tmp = min(int(weight_prev1 * p1 + weight_prev2nd * p2), Tsdf::MAX_WEIGHT); //v21.5.6: w_new 计算时带上 p1, p2
                    if(doDbgPrint)
                        printf("【【w2_fuse_w1: w1: %d, p1: %f, t1: %f; 【【w2: %d, p2: %f, t1: %f; w_new: %d, t_new: %f\n", 
                        weight_prev1, p1, tsdf_prev1, weight_prev2nd, p2, tsdf_prev2nd, weight_new_tmp, tsdf_new_tmp);

                    weight_new_tmp = (weight_new_tmp << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                    pack_tsdf (tsdf_new_tmp, weight_new_tmp, *pos1);

                    pack_tsdf (0, 0, *pos2nd); //pos2 重置清空
                }
#endif //二分类落实哪种策略
            }
#endif //v21.1~ v21.2~ v21.4.x


            //if(164 == x && 60 == y && 112 == z)
            //    printf("【【@(%d, %d, %d):tsdf_new1: %f,, weight_new1: %d;;; non_edge_near0: %d\n", x, y, z, tsdf_new1, weight_new1, non_edge_near0);

          }//if-(Dp_scaled != 0 && sdf >= -tranc_dist) 
#if 0   //v19.x~v20.x
          //else if(Dp_scaled != 0 && !is_curr_edge_wide //v19.8.3
          else if(Dp_scaled != 0 //v20.1.8: curr-edge 判定移到下面, 因为 w2-shrink 不需要此判定 @2018-9-25 11:02:09
              //&& sdf > -4*tranc_dist_real
              //&& sdf > -4*tranc_dist    //good
              && sdf > -shrink_dist_th   //good
              )
          {
              //要不要 if-- tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH 条件？待定 @2018-8-24 01:08:46
              //v19.2: 要, 
              const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
              //if(tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
              if(!is_curr_edge_wide && tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //v20.1.8: curr-edge 判定移到这里 @2018-9-25 11:02:51
                  //if(weight_prev1 < POS_VALID_WEIGHT_TH) //v19.8.5: 去掉 tp+ 判定， +-一律衰减 @2018-9-12 20:00:24
              {
                  weight_new1 = max(0, weight_new1-1); //v18.22
                  if(weight_new1 == 0)
                      tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

                  if(doDbgPrint)
                      printf("】】tsdf_new1: %f,, W1-UNSTABLE SHRINK, weight_new1-=1: %d;\n", tsdf_new1, weight_new1);

                  //v20.1.16:
                  weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                  pack_tsdf (tsdf_new1, weight_new1, *pos1);
              }

//               if(weight_prev2nd > 0){ //即 setup_cluster2 仍在
//                   //v20.1.6: v20 连续 w2 赋值的策略下, 若负值临界区域, 扫不到就断断续续递增 w2, 转过面之后会造成误判, 因此改成断续就衰减 @2018-9-25 00:16:04
//                   //weight_prev2nd = max(0, weight_prev2nd-1);
//                   weight_prev2nd --;
//                   if(weight_prev2nd == 0)
//                       tsdf_prev2nd = 0;
//                   pack_tsdf(tsdf_prev2nd, weight_prev2nd, *pos2nd);
// 
//                   if(doDbgPrint)
//                       printf("】】】w2-SHRINK, tsdf_prev2nd: %f, weight_prev2nd-1: %d\n", tsdf_prev2nd, weight_prev2nd);

              //v20.1.9: 上面负值断续就衰减,不好用, 改成: 即使断续, 只要在薄带范围内, 就以 t2=-1 递增 w2  @2018-9-25 17:15:59
//               if(0 < weight_prev2nd && weight_prev2nd < WEIGHT_CONFID_TH){ //即 setup_cluster2 仍在
//                   tsdf_prev2nd = (tsdf_prev2nd * weight_prev2nd + (-1)) / (weight_prev2nd + 1);
//                   weight_prev2nd = min (weight_prev2nd + 1, Tsdf::MAX_WEIGHT);
//                   pack_tsdf(tsdf_prev2nd, weight_prev2nd, *pos2nd);
//                   if(doDbgPrint)
//                       printf("】】】w2+++, sdf<-TH, tsdf_prev2nd: %f, weight_prev2nd: %d\n", tsdf_prev2nd, weight_prev2nd);
//               }
          }//elif-(-4*td < sdf < -td)
#elif 1 //稍微整理一下, 改嵌套 if-if, 以便兼容 v21.5.x
          else if(Dp_scaled != 0){
              //if(sdf > -shrink_dist_th){
              if(non_edge_near0){ //v21.5.8: owl2 全反射导致cu内部噪声, 原 sdf>-TH 判定无法解决, 尝试此法   @2018-10-25 03:26:26
              //if(1){ //v21.5.9: tooth1 上 v21.5.8 仍不行, 所以尝试 always-true @2018-10-25 04:24:59
                  //要不要 if-- tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH 条件？待定 @2018-8-24 01:08:46
                  //v19.2: 要, 
                  const int POS_VALID_WEIGHT_TH = 30; //30帧≈一秒
                  //if(tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //或, 若 t_p 正值但是尚不稳定
                  //if(!is_curr_edge_wide && tsdf_prev1 > 0 && weight_prev1 < POS_VALID_WEIGHT_TH) //v20.1.8: curr-edge 判定移到这里 @2018-9-25 11:02:51
                  if(!is_curr_edge_wide /*&& tsdf_prev1 > 0*/ && weight_prev1 < POS_VALID_WEIGHT_TH) //v20.1.8: curr-edge 判定移到这里 @2018-9-25 11:02:51
                      //if(weight_prev1 < POS_VALID_WEIGHT_TH) //v19.8.5: 去掉 tp+ 判定， +-一律衰减 @2018-9-12 20:00:24
                  {
                      //weight_new1 = max(0, weight_new1-1); //v18.22
                      weight_new1 = max(0, weight_new1-Wrk); //v21.6.0.a
                      if(weight_new1 == 0)
                          tsdf_new1 = 0; //严谨点, 避免调试绘制、marching cubes意外

                      if(doDbgPrint)
                          printf("】】tsdf_new1: %f,, W1-UNSTABLE SHRINK, weight_new1-=1: %d;\n", tsdf_new1, weight_new1);

                      //v20.1.16:
                      weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
                      pack_tsdf (tsdf_new1, weight_new1, *pos1);
                  }

                  //v21.5.11
//                   if(tsdf_prev1 < -0.5 /*&& non_edge_near0 */ && sdf >= -1.2 * tranc_dist_real) //meters 
//                   {
//                       const float tsdf_curr = -1;
//                       const int Wrk = 1;
//                       tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
//                       weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);
// 
//                       weight_new1 = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
//                       pack_tsdf (tsdf_prev1, weight_new1, *pos1);
// 
//                       if(doDbgPrint)
//                           printf("tp1_neg_AND_P_in_AND_sdf_gt_1.5th: 【【t1_new: %f, w1_new: %d\n", tsdf_prev1, weight_prev1);
//                   }
              }//if-(non_edge_near0)

              //v21.5.7: 接上面 v21.5.x (~v21.5.6): 考虑: 若二分类, w2<TH 时, 突然视角已过, w2++停下了, 怎么处理?
              //问题: v21.5.6 当w2 终止后, 转过大圈, 又见到正面时, 结果误判 w2 冲掉 w1, 导致凹凸噪声
              //总体思路: 扔掉 w2; 或找机会强制刷新; //终止时常见情形: ① diff 规则导致 IGN 【diff_gt_MIN_THICKNESS==FUSE_IGNORE_CURR】 ② sdf <tdist导致 【sdfFFF】, sdfFFF又分 >or< shrink_dist_th 情形
              //① 无效区域, w2--; ② 暂无

              //v21.5.7.1: 只要 dp!=0, 不管 shrink_dist_th, 直接 w2--; 其实雷同 v20.1.6 
              if(weight_prev2nd > 0){ //即 setup_cluster2 仍在
                  //v20.1.6: v20 连续 w2 赋值的策略下, 若负值临界区域, 扫不到就断断续续递增 w2, 转过面之后会造成误判, 因此改成断续就衰减 @2018-9-25 00:16:04
                  //weight_prev2nd = max(0, weight_prev2nd-1);
                  //weight_prev2nd --;
                  weight_prev2nd -= Wrk;

                  if(weight_prev2nd == 0)
                      tsdf_prev2nd = 0;
                  pack_tsdf(tsdf_prev2nd, weight_prev2nd, *pos2nd);

                  if(doDbgPrint)
                      printf("】】】w2-SHRINK, tsdf_prev2nd: %f, weight_prev2nd-1: %d\n", tsdf_prev2nd, weight_prev2nd);
              }
          }//if-(Dp_scaled != 0)
#endif

#if 0   //v21.4.6: 专门处理 v21.4.5 的问题, 思路: 当 depthModel 有值, 但 d_curr 无效, 很可能是斜视、结构光边缘导致; 
          //希望在远离表面处, 若 d_mod 有值, 就 t1=+1; 这里判定相当于先 interpmax_xy, 再xxxxxx   @2018-10-17 09:53:44
          //【暂放弃, 缺点：①下面代码没有真用到 interpmax_xy; 而是取自 depthModel 的值, 若model上边缘噪声, 再用自身,导致无解
          //②就算用 interpmax_xy, 也不好, 深度无效区域并不总是结构光视差导致, 随便用插值深度值作参考, 导致误判 @2018-10-24 22:19:24
          else if(Dp_scaled == 0){
              if(doDbgPrint)
                  printf("Dp_scaled_eq_0:=");

              ushort depth_prev = depthModel.ptr(coo.y)[coo.x];
              if(depth_prev > 0){
                  float v_c_z_mm = v_z_real * 1e3; //mm, z本是米, 转毫米也未必整数
                  //float sdf_forward = Dp_not_scaled - v_c_z_mm; //错, d_curr 已是0, 该用 d_prev
                  float sdf_forward = depth_prev - v_c_z_mm; //

                  if(doDbgPrint)
                      printf("depth_prev_gt_0:=sdf_forward: %f; //depth_prev, v_c_z_mm, rc_flag: %u, %f, %d\n", sdf_forward, depth_prev, v_c_z_mm, rc_flag);

                  if(sdf_forward > shrink_dist_th * 1e3){ //远离表面处
                      float tsdf_curr = +1.f;
                      int Wrk = 1;

                      float tsdf_new_tmp;
                      int weight_new_tmp;

                      tsdf_prev1 = (tsdf_prev1 * weight_prev1 + tsdf_curr * Wrk) / (weight_prev1 + Wrk);
                      weight_prev1 = min (weight_prev1 + Wrk, Tsdf::MAX_WEIGHT);

                      weight_new_tmp = (weight_prev1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt;
                      pack_tsdf (tsdf_prev1, weight_new_tmp, *pos1);

                      if(doDbgPrint)
                          printf("sdf_forward_gt_shrink_dist_th; 【【tsdf_new1: %f\n", tsdf_prev1);
                  }
                  else{ //近表面处, 不 fuse
                      if(doDbgPrint)
                          printf("sdf_forward_LEEEEEEE_shrink_dist_th\n");
                  }
                  //weight_prev1
              }
              else{ //d_prev ==0
                  if(doDbgPrint)
                      printf("depth_prev_EQQQQQ_0\n");
              }
          }//if-(Dp_scaled == 0)
#endif
#if 1   
        
        
        
#endif 

          //weight_new1 = (weight_new1 << VOL1_FLAG_BIT_CNT) + non_edge_ccnt; //v19.8.1
          //pack_tsdf (tsdf_new1, weight_new1, *pos1);
          //↑--v20.1.16: 之前放外面, 主要是为了 任何条件下确保 non_edge_ccnt 写入; 但因逻辑难看清, 暂去掉, 在各分支内操作 @2018-9-27 11:28:19

        }//if- 0 < (x,y) < (cols,rows)
      }// for(int z = 0; z < VOLUME_Z; ++z)
    }//tsdf23_v21

    __global__ void
    tsdf23normal_hack (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
                  const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
            return;

        const float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
        const float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
        float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

        float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

        float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
        float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
        float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

        float z_scaled = 0;

        float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
        float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

        float tranc_dist_inv = 1.0f / tranc_dist;

        short2* pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(short2);

        //#pragma unroll
        for (int z = 0; z < VOLUME_Z;
            ++z,
            v_g_z += cell_size.z,
            z_scaled += cell_size.z,
            v_x += Rcurr_inv_0_z_scaled,
            v_y += Rcurr_inv_1_z_scaled,
            pos += elem_step)
        {
            float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
            if (inv_z < 0)
                continue;

            // project to current cam
            int2 coo =
            {
                __float2int_rn (v_x * inv_z + intr.cx),
                __float2int_rn (v_y * inv_z + intr.cy)
            };

            if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
            {
                float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

                float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

                if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
                {
                    float tsdf = fmin (1.0f, sdf * tranc_dist_inv);                                              

                    bool integrate = true;
                    if ((x > 0 &&  x < VOLUME_X-2) && (y > 0 && y < VOLUME_Y-2) && (z > 0 && z < VOLUME_Z-2))
                    {
                        const float qnan = numeric_limits<float>::quiet_NaN();
                        float3 normal = make_float3(qnan, qnan, qnan);

                        float Fn, Fp;
                        int Wn = 0, Wp = 0;
                        unpack_tsdf (*(pos + elem_step), Fn, Wn);
                        unpack_tsdf (*(pos - elem_step), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.z = (Fn - Fp)/cell_size.z;

                        unpack_tsdf (*(pos + volume.step/sizeof(short2) ), Fn, Wn);
                        unpack_tsdf (*(pos - volume.step/sizeof(short2) ), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.y = (Fn - Fp)/cell_size.y;

                        unpack_tsdf (*(pos + 1), Fn, Wn);
                        unpack_tsdf (*(pos - 1), Fp, Wp);

                        if (Wn > 16 && Wp > 16) 
                            normal.x = (Fn - Fp)/cell_size.x;

                        if (normal.x != qnan && normal.y != qnan && normal.z != qnan)
                        {
                            float norm2 = dot(normal, normal);
                            if (norm2 >= 1e-10)
                            {
                                normal *= rsqrt(norm2);

                                float nt = v_g_x * normal.x + v_g_y * normal.y + v_g_z * normal.z;
                                float cosine = nt * rsqrt(v_g_x * v_g_x + v_g_y * v_g_y + v_g_z * v_g_z);

                                if (cosine < 0.5)
                                    integrate = false;
                            }
                        }
                    }

                    if (integrate)
                    {
                        //read and unpack
                        float tsdf_prev;
                        int weight_prev;
                        unpack_tsdf (*pos, tsdf_prev, weight_prev);

                        const int Wrk = 1;

                        float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                        int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

                        pack_tsdf (tsdf_new, weight_new, *pos);
                    }
                }
            }
        }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // tsdf23normal_hack
  }//namespace device

    __global__ void
    tsdf23test (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
            const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size, const pcl::gpu::tsdf_buffer buffer)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= buffer.voxels_size.x || y >= buffer.voxels_size.y)
        return;

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos = volume.ptr (y) + x;
      
      // shift the pointer to relative indices
      shift_tsdf_pointer(&pos, buffer);
      
      int elem_step = volume.step * buffer.voxels_size.y / sizeof(short2);

//#pragma unroll
      for (int z = 0; z < buffer.voxels_size.z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos += elem_step)
      {
        
        // As the pointer is incremented in the for loop, we have to make sure that the pointer is never outside the memory
        if(pos > buffer.tsdf_memory_end)
          pos -= (buffer.tsdf_memory_end - buffer.tsdf_memory_start + 1);
        
        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
		// old code
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

          if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
          {
            float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

            //read and unpack
            float tsdf_prev;
            int weight_prev;
            unpack_tsdf (*pos, tsdf_prev, weight_prev);

            const int Wrk = 1;

            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
            int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

            pack_tsdf (tsdf_new, weight_new, *pos);
          }
        }
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth, const Intr& intr,
                                  const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr, 
                                  float tranc_dist,
                                  //PtrStep<short2> volume, const pcl::gpu::tsdf_buffer* buffer, DeviceArray2D<float>& depthScaled)
                                  PtrStep<short2> volume, const pcl::gpu::tsdf_buffer* buffer, DeviceArray2D<float>& depthScaled, int3 vxlDbg) //zc: 调试
{
  depthScaled.create (depth.rows, depth.cols);

  dim3 block_scale (32, 8);
  dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

  //scales depth along ray and converts mm -> meters. 
  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
  cudaSafeCall ( cudaGetLastError () );

  float3 cell_size;
  cell_size.x = volume_size.x / buffer->voxels_size.x;
  cell_size.y = volume_size.y / buffer->voxels_size.y;
  cell_size.z = volume_size.z / buffer->voxels_size.z;

  //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 block (16, 16);
  dim3 grid (divUp (buffer->voxels_size.x, block.x), divUp (buffer->voxels_size.y, block.y));

  //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, *buffer);    
  tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, *buffer, vxlDbg);    

//  for ( int i = 0; i < 100; i++ )
//    tsdf23test<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, *buffer);    

  //tsdf23normal_hack<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

void
pcl::device::integrateTsdfVolume_s2s (/*const PtrStepSz<ushort>& depth,*/ const Intr& intr,
    const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr, const float3& volume000_gcoo, float tranc_dist, float eta, bool use_eta_trunc,
    PtrStep<short2> volume, DeviceArray2D<float>& depthScaled, int3 vxlDbg) //zc: 调试
{
    //depthScaled.create (depth.rows, depth.cols);

    //dim3 block_scale (32, 8);
    //dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

    ////scales depth along ray and converts mm -> meters. 
    //scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    //cudaSafeCall ( cudaGetLastError () );

    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

    //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
    dim3 block (16, 16);
    dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

    //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, *buffer);    
    tsdf23_s2s<<<grid, block>>>(depthScaled, volume, tranc_dist, eta, use_eta_trunc,
        Rcurr_inv, tcurr, volume000_gcoo, intr, cell_size, vxlDbg);    

    //  for ( int i = 0; i < 100; i++ )
    //    tsdf23test<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, *buffer);    

    //tsdf23normal_hack<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}//integrateTsdfVolume_s2s

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::integrateTsdfVolume_v11 (const PtrStepSz<ushort>& depth, const Intr& intr, const float3& volume_size, 
    const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume, 
    PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, DeviceArray2D<unsigned char> incidAngleMask, 
    const MapArr& nmap_curr_g, const MapArr &nmap_model_g,
    const MapArr &weight_map, //v11.4
    DeviceArray2D<float>& depthScaled, int3 vxlDbg)
{
    depthScaled.create (depth.rows, depth.cols);

    dim3 block_scale (32, 8);
    dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

    //scales depth along ray and converts mm -> meters. 
    scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall ( cudaGetLastError () );

    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

    //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
    dim3 block (16, 16);
    dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

    printf("vxlDbg@integrateTsdfVolume_v11: [%d, %d, %d]\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);

    //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);    
    //tsdf23_v11<<<grid, block>>>(depthScaled, volume, 
    tsdf23_v11_remake<<<grid, block>>>(depthScaled, volume, 
        volume2nd, flagVolume, surfNormVolume, vrayPrevVolume, incidAngleMask, 
        nmap_curr_g, nmap_model_g,
        weight_map,
        tranc_dist, Rcurr_inv, tcurr, intr, cell_size, vxlDbg);    
}//integrateTsdfVolume_v11

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::integrateTsdfVolume_v12 (const PtrStepSz<ushort>& depth, const Intr& intr, const float3& volume_size, 
    const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume, 
    PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, DeviceArray2D<unsigned char> incidAngleMask, 
    const MapArr& nmap_curr_g, const MapArr &nmap_model_g,
    const MapArr &weight_map, //v11.4
    const PtrStepSz<ushort>& depth_model,
    DeviceArray2D<short>& diffDmap,
    DeviceArray2D<float>& depthScaled, int3 vxlDbg)
{
    depthScaled.create (depth.rows, depth.cols);

    dim3 block_scale (32, 8);
    dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

    //scales depth along ray and converts mm -> meters. 
    scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall ( cudaGetLastError () );

    //v12 加一步: 求 diffDmap = depth(raw)-depth_model @2017-12-3 22:06:24
    //DeviceArray2D<short> diffDmap; //short, 而非 ushort
    //↑--局部变量会导致: Error: unspecified launch failure       ..\..\..\gpu\containers\src\device_memory.cpp:276 //在: DeviceMemory2D::release() 出错
    diffDmap.create(depth.rows, depth.cols);
    diffDmaps(depth, depth_model, diffDmap); //仍 mm


    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

    //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
    dim3 block (16, 16);
    dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

    printf("vxlDbg@integrateTsdfVolume_v12: [%d, %d, %d]\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);

    //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);    
    //tsdf23_v11<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v11_remake<<<grid, block>>>(depthScaled, volume, 
    tsdf23_v12<<<grid, block>>>(depthScaled, volume, 
         volume2nd, flagVolume, surfNormVolume, vrayPrevVolume, incidAngleMask, 
         nmap_curr_g, nmap_model_g,
         weight_map,
         diffDmap,
         tranc_dist, Rcurr_inv, tcurr, intr, cell_size, vxlDbg);    
}//integrateTsdfVolume_v12


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::integrateTsdfVolume_v13 (const PtrStepSz<ushort>& depth, const Intr& intr, const float3& volume_size, 
    const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume, 
    PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, DeviceArray2D<unsigned char> incidAngleMask, 
    const MapArr& nmap_curr_g, const MapArr &nmap_model_g,
    const MapArr &weight_map, //v11.4
    const PtrStepSz<ushort>& depth_model,
    DeviceArray2D<short>& diffDmap,
    DeviceArray2D<float>& depthScaled, int3 vxlDbg)
{
    depthScaled.create (depth.rows, depth.cols);

    dim3 block_scale (32, 8);
    dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

    //scales depth along ray and converts mm -> meters. 
    scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall ( cudaGetLastError () );

    //v12 加一步: 求 diffDmap = depth(raw)-depth_model @2017-12-3 22:06:24
    //DeviceArray2D<short> diffDmap; //short, 而非 ushort
    //↑--局部变量会导致: Error: unspecified launch failure       ..\..\..\gpu\containers\src\device_memory.cpp:276 //在: DeviceMemory2D::release() 出错
    diffDmap.create(depth.rows, depth.cols);
    diffDmaps(depth, depth_model, diffDmap); //仍 mm

    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

    //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
    dim3 block (16, 16);
    dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

    printf("vxlDbg@integrateTsdfVolume_v13: [%d, %d, %d]\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);

    //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);    
    //tsdf23_v11<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v13<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v14<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v15<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v16<<<grid, block>>>(depthScaled, volume,  //测试 tranc_dist_real 用的
    tsdf23_v17<<<grid, block>>>(depthScaled, volume,  //长短 tdist, 晶格独立存 tdist
        volume2nd, flagVolume, surfNormVolume, vrayPrevVolume, incidAngleMask, 
        nmap_curr_g, nmap_model_g,
        weight_map,
        diffDmap,
        tranc_dist, Rcurr_inv, tcurr, intr, cell_size, vxlDbg);    
}//integrateTsdfVolume_v13

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::integrateTsdfVolume_v18 (const PtrStepSz<ushort>& depth, const Intr& intr, const float3& volume_size, 
    const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume, 
    PtrStep<short2> volume2nd, PtrStep<bool> flagVolume, PtrStep<char4> surfNormVolume, PtrStep<char4> vrayPrevVolume, DeviceArray2D<unsigned char> incidAngleMask, 
    const MapArr& nmap_curr_g, const MapArr &nmap_model_g,
    const MapArr &weight_map, //v11.4
    const PtrStepSz<ushort>& depth_model,
    DeviceArray2D<short>& diffDmap,
    const PtrStepSz<ushort>& depth_model_vol2, //v21.6.0: 核心是, 在 vol2 上 rcast 求 sdf_forward, 而非用 dcurr 瞬时结果 @2018-11-29 17:54:01
    //const PtrStepSz<ushort>& rc_flag_map,
    const PtrStepSz<short>& rc_flag_map,
    DeviceArray2D<float>& depthScaled, int3 vxlDbg)
{
    depthScaled.create (depth.rows, depth.cols);

    dim3 block_scale (32, 8);
    dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

    //scales depth along ray and converts mm -> meters. 
    scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
    cudaSafeCall ( cudaGetLastError () );

    //v12 加一步: 求 diffDmap = depth(raw)-depth_model @2017-12-3 22:06:24
    //DeviceArray2D<short> diffDmap; //short, 而非 ushort
    //↑--局部变量会导致: Error: unspecified launch failure       ..\..\..\gpu\containers\src\device_memory.cpp:276 //在: DeviceMemory2D::release() 出错
    diffDmap.create(depth.rows, depth.cols);
    diffDmaps(depth, depth_model, diffDmap); //仍 mm

    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

    //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
    dim3 block (16, 16);
    dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

    printf("vxlDbg@integrateTsdfVolume_v18: [%d, %d, %d]\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);

    //tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);    
    //tsdf23_v11<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v13<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v14<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v15<<<grid, block>>>(depthScaled, volume, 
    //tsdf23_v16<<<grid, block>>>(depthScaled, volume,  //测试 tranc_dist_real 用的
    //tsdf23_v17<<<grid, block>>>(depthScaled, volume,  //长短 tdist, 晶格独立存 tdist
    //test_kernel<<<grid, block>>>(vxlDbg); //v18.2
    //tsdf23_v18<<<grid, block>>>(depthScaled, volume,  
    //tsdf23_v19<<<grid, block>>>(depthScaled, volume,  
//     tsdf23_v20<<<grid, block>>>(depthScaled, volume,  
//         volume2nd, flagVolume, surfNormVolume, vrayPrevVolume, incidAngleMask, 
//         nmap_curr_g, nmap_model_g,
//         weight_map,
//         depth_model, //v18.5, 新增形参, 主要判定 isnan
//         diffDmap,
//         tranc_dist, Rcurr_inv, tcurr, intr, cell_size, vxlDbg);    
    tsdf23_v21<<<grid, block>>>(depthScaled, volume,  
        volume2nd, flagVolume, surfNormVolume, vrayPrevVolume, incidAngleMask, 
        nmap_curr_g, nmap_model_g,
        weight_map,
        depth, //新增, 因为 rc_flag_map, depth_model 等都是没 scale的, 所以 kernel 中 rc_flag 处理逻辑统一不 scale
        depth_model, //v18.5, 新增形参, 主要判定 isnan
        diffDmap,
        depth_model_vol2, //v21.6.0: 核心是, 在 vol2 上 rcast 求 sdf_forward, 而非用 dcurr 瞬时结果 @2018-11-29 17:54:01
        rc_flag_map,
        tranc_dist, Rcurr_inv, tcurr, intr, cell_size, vxlDbg);    

    cudaSafeCall (cudaDeviceSynchronize ());

//     static std::ofstream fout("kf-dbg-log.txt");
//     //if(!fout.good())
//     float sdf_orig_host;
//     float cos_host;
//     float sdf_cos_host;
//     float tdist_real_host;
//     bool snorm_oppo_host;
//     bool is_curr_edge_host;
//     bool is_non_edge_near0_host;
//     short depth_curr_host;
//     short depth_prev_host;
//     short diff_cp_host;
//     cudaSafeCall(cudaMemcpyFromSymbol(&sdf_orig_host, sdf_orig_dev, sizeof(sdf_orig_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&cos_host, cos_dev, sizeof(cos_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&sdf_cos_host, sdf_cos_dev, sizeof(sdf_cos_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&tdist_real_host, tdist_real_dev, sizeof(tdist_real_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&snorm_oppo_host, snorm_oppo_dev, sizeof(snorm_oppo_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&is_curr_edge_host, is_curr_edge_dev, sizeof(is_curr_edge_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&is_non_edge_near0_host, is_non_edge_near0_dev, sizeof(is_non_edge_near0_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&depth_curr_host, depth_curr_dev, sizeof(depth_curr_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&depth_prev_host, depth_prev_dev, sizeof(depth_prev_host)) );
//     cudaSafeCall(cudaMemcpyFromSymbol(&diff_cp_host, diff_cp_dev, sizeof(diff_cp_host)) );
// 
//     fout << sdf_orig_host << ',' << cos_host << ',' << sdf_cos_host 
//          << ',' << tdist_real_host << ',' << snorm_oppo_host
//          << ',' << is_curr_edge_host << ',' << is_non_edge_near0_host 
//          << ',' << depth_curr_host << ',' << depth_prev_host << ',' << diff_cp_host 
//          << std::endl;
}//integrateTsdfVolume_v18


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::clearTSDFSlice (PtrStep<short2> volume, pcl::gpu::tsdf_buffer* buffer, int shiftX, int shiftY, int shiftZ)
{
    int newX = buffer->origin_GRID.x + shiftX;
    int newY = buffer->origin_GRID.y + shiftY;

    int3 minBounds, maxBounds;
    
	/*
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
	 */
	if ( shiftX >= 0 ) {
		minBounds.x = buffer->origin_GRID.x;
		maxBounds.x = newX - 1;
	} else {
		minBounds.x = newX;
		maxBounds.x = buffer->origin_GRID.x - 1;
	}
	if ( minBounds.x < 0 ) {
		minBounds.x += buffer->voxels_size.x;
		maxBounds.x += buffer->voxels_size.x;
	}

	if ( shiftY >= 0 ) {
		minBounds.y = buffer->origin_GRID.y;
		maxBounds.y = newY - 1;
	} else {
		minBounds.y = newY;
		maxBounds.y = buffer->origin_GRID.y - 1;
	}
	if ( minBounds.y < 0 ) {
		minBounds.y += buffer->voxels_size.y;
		maxBounds.y += buffer->voxels_size.y;
	}
    //Z
     minBounds.z = buffer->origin_GRID.z;
     maxBounds.z = shiftZ;
  
    // call kernel
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (buffer->voxels_size.x, block.x);      
    grid.y = divUp (buffer->voxels_size.y, block.y);
    
    clearSliceKernel<<<grid, block>>>(volume, *buffer, minBounds, maxBounds);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
   
}//clearTSDFSlice

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//拷贝自 contour_cue_impl.cu, 因为手动无法添加到此工程, 必须重新 cmake, 导致混乱; 所以直接源码拷贝
namespace zc{

//@brief gpu kernel function to generate the Contour-Correspondence-Candidates
//@param[in] angleThreshCos, MAX cosine of the angle threshold
//@注意 kernel 函数参数必须为 GPU 内存指针或对象拷贝，e.g., 必须为 float3 而非 float3&
__global__ void 
cccKernel(const float3 camPos, const PtrStep<float> vmap, const PtrStep<float> nmap, float angleThreshCos, PtrStepSz<_uchar> outMask){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("### %d, %d\n", x, y);

    int cols = outMask.cols,
        rows = outMask.rows;

    if(!(x < cols && y < rows))
        return;

    outMask.ptr(y)[x] = 0;

    if(isnan(nmap.ptr(y)[x]) || isnan(vmap.ptr(y)[x])){
        //printf("\tisnan: %d, %d\n", x, y);
        return;
    }

    float3 n, vRay;
    n.x = nmap.ptr(y)[x];
    n.y = nmap.ptr(y + rows)[x];
    n.z = nmap.ptr(y + 2 * rows)[x];

    vRay.x = camPos.x - vmap.ptr(y)[x];
    vRay.y = camPos.y - vmap.ptr(y + rows)[x];
    vRay.z = camPos.z - vmap.ptr(y + 2 * rows)[x];

    double nMod = norm(n); //理论上恒等于1？
    double vRayMod = norm(vRay);
    //printf("@@@ %f, %f\n", nMod, vRayMod);

    double cosine = dot(n, vRay) / (vRayMod * nMod);
    if(abs(cosine) < angleThreshCos)
        outMask.ptr(y)[x] = UCHAR_MAX;
}//cccKernel

void contourCorrespCandidate(const float3 &camPos, const MapArr &vmap, const MapArr &nmap, int angleThresh, pcl::device::MaskMap &outMask ){
    int cols = vmap.cols();
    int rows = vmap.rows() / 3;
    
    outMask.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    const float angleThreshCos = cos(angleThresh * 3.14159265359f / 180.f);
    //printf("vmap, nmap shape: [%d, %d], [%d, %d]\n", vmap.rows(), vmap.cols(), nmap.rows(), nmap.cols()); //test OK
    cccKernel<<<grid, block>>>(camPos, vmap, nmap, angleThreshCos, outMask);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize()); //tmp, 暂时企图避免阻塞 @2017-12-6 22:03:13
}//contourCorrespCandidate

__global__ void
calcWmapKernel(int rows, int cols, const PtrStep<float> vmapLocal, const PtrStep<float> nmapLocal, const PtrStepSz<_uchar> contMask, PtrStepSz<float> wmap_out){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    const float qnan = pcl::device::numeric_limits<float>::quiet_NaN();
    if(!(x < cols && y < rows))
        return;

    //调试用:
    bool doDbgPrint = false;
//     if(x == 388 && y == 292)
//         doDbgPrint = true;

    wmap_out.ptr(y)[x] = 0; //默认初始权重=0

    float3 vray; //local
    vray.x = vmapLocal.ptr(y)[x];
    if(isnan(vray.x))
        return;

    vray.y = vmapLocal.ptr(y + rows)[x];
    vray.z = vmapLocal.ptr(y + 2 * rows)[x]; //meters

    float3 snorm;
    snorm.x = nmapLocal.ptr(y)[x];
    snorm.y = nmapLocal.ptr(y + rows)[x];
    snorm.z = nmapLocal.ptr(y + 2 * rows)[x];

    //假设不确保 normalize 过: 要归一化, 不确保 snorm 朝向视点: 要 abs
    float cosine = dot(vray, snorm) / (norm(snorm) * norm(vray));
    cosine = abs(cosine);

#if 0   //v0: KinectFusion 论文提到的 "正比于...", 粗糙
    //wmap_out.ptr(y)[x] = cosine * zmin / max(zmin, vray.z); //缩放不受控制, 范围过于广 ( cos->0, z->+inf ); 下面加入 minXXfactor 约束 ↓
#elif 10 //v1: 有控制缩放范围
    const float minCosFactor = .5f; //cos min乘数因子, 即便 90°, 乘数也做 1/2, 以免太小
    const float cosMin = 0.5f; //60°, 若 theta<60°, 乘数因子固定为 1, 即完全信任倾角 0~60°时的深度值
    float cosFactor = 1;
    if(cosine < cosMin)
        cosFactor = 1 - (1 - 2 * cosine) * (1 - minCosFactor) / 1; //这个分母的 1= (1-0)

    const float minZfactor = .5f; //深度值 min乘数因子
    const float zmin = 0.5f,
                zmax = 3.f; //meters, zmax 此处并不会真的限定最大有效深度, 只是确保 zmax 处, 乘数为 minZfactor (原本比如=1/6)

    float oldMinZfactor = zmin / zmax;
    //float zFactor = 1 - (1 - vray.z) * (1 - minZfactor)/ (1 - rawMinZfactor); //×
    float zFactor = zmin / min(zmax, max(zmin, vray.z)); //1/6 <= factor <= 1
    //↓--[1/6, 1] -> [.5, 1]
    zFactor = 1 - (1 - zFactor) * (1 - minZfactor) / (1 - oldMinZfactor);

    float contFactor = 1;
    if(contMask.ptr(y)[x] != 0) //边缘降权, 防止翻翘
        contFactor = 0.3f;

    wmap_out.ptr(y)[x] = cosFactor * zFactor * contFactor;
#endif 

}//calcWmapKernel

//@brief v2, 之前 contMask 算作二分类权重 mask, 改成浮点型, 平滑过渡权重 (控制量 edgeDistMap)
//@param[in] edgeDistMap, 到边缘像素距离 mat: 值越小,离边缘越近, tsdf权重以及tsdf截断阈值越小; 需要结合 vmap.z 转化成物理尺度距离,
//@param[in] fxy, 是一个焦距约数, 量纲像素 
__global__ void
calcWmapKernel(int rows, int cols, const PtrStep<float> vmapLocal, const PtrStep<float> nmapLocal, const PtrStepSz<float> edgeDistMap, float fxy, PtrStepSz<float> wmap_out){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    const float qnan = pcl::device::numeric_limits<float>::quiet_NaN();
    if(!(x < cols && y < rows))
        return;

    //调试用:
    bool doDbgPrint = false;
//     if(x == 388 && y == 292)
//         doDbgPrint = true;

    wmap_out.ptr(y)[x] = 0; //默认初始权重=0

    float3 vray; //local
    vray.x = vmapLocal.ptr(y)[x];
    if(isnan(vray.x))
        return;

    vray.y = vmapLocal.ptr(y + rows)[x];
    vray.z = vmapLocal.ptr(y + 2 * rows)[x]; //meters

    float3 snorm;
    snorm.x = nmapLocal.ptr(y)[x];
    snorm.y = nmapLocal.ptr(y + rows)[x];
    snorm.z = nmapLocal.ptr(y + 2 * rows)[x];

    //假设不确保 normalize 过: 要归一化, 不确保 snorm 朝向视点: 要 abs
    float cosine = dot(vray, snorm) / (norm(snorm) * norm(vray));
    cosine = abs(cosine); //取锐角

#if 0   //v0: KinectFusion 论文提到的 "正比于...", 粗糙
    //wmap_out.ptr(y)[x] = cosine * zmin / max(zmin, vray.z); //缩放不受控制, 范围过于广 ( cos->0, z->+inf ); 下面加入 minXXfactor 约束 ↓
#elif 10 //v1: 有控制缩放范围
    const float minCosFactor = .3f; //cos min乘数因子, 即便 90°, 乘数也做 1/2, 以免太小
    const float cosMin = 0.5f; //60°, 若 theta<60°, 乘数因子固定为 1, 即完全信任倾角 0~60°时的深度值
    float cosFactor = 1;
    if(cosine < cosMin) //确保不要 cos >1
        cosFactor = 1 - (1 - 2 * cosine) * (1 - minCosFactor) / 1; //这个分母的 1= (1-0)

    const float minZfactor = .5f; //深度值 min乘数因子
    const float zmin = 0.5f,
                zmax = 3.f; //meters, zmax 此处并不会真的限定最大有效深度, 只是确保 zmax 处, 乘数为 minZfactor (原本比如=1/6)

    float oldMinZfactor = zmin / zmax;
    //float zFactor = 1 - (1 - vray.z) * (1 - minZfactor)/ (1 - rawMinZfactor); //×
    float zFactor = zmin / min(zmax, max(zmin, vray.z)); //1/6 <= factor <= 1
    //↓--[1/6, 1] -> [.5, 1]
    zFactor = 1 - (1 - zFactor) * (1 - minZfactor) / (1 - oldMinZfactor);

#if 0   //contMask 做控制量
    float contFactor = 1;
    if(contMask.ptr(y)[x] != 0) //边缘降权, 防止翻翘
        contFactor = 0.3f;

    wmap_out.ptr(y)[x] = cosFactor * zFactor * contFactor;
#elif 1 //edgeDistMap 做控制量
    const float maxEdgeDist = 30; //in mm
    float edgeDistMm = edgeDistMap.ptr(y)[x] / fxy * vray.z * 1e3; //in mm

    float edgeDistFactor = 1.f;
    if(edgeDistMm < maxEdgeDist) //不许超过 1
        edgeDistFactor = edgeDistMm / maxEdgeDist;

    wmap_out.ptr(y)[x] = cosFactor * zFactor * edgeDistFactor;
#endif

#endif //缩放有无控制


}//calcWmapKernel-v2

//@param[in] vmapLocal, 其实只要个 dmap 就行, 暂不改, 与之前 calcWmapKernel 保持一致
__global__ void
edge2wmapKernel(int rows, int cols, const PtrStep<float> vmapLocal, const PtrStepSz<float> edgeDistMap, float fxy, float maxEdgeDist, PtrStepSz<float> wmap_out){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    const float qnan = pcl::device::numeric_limits<float>::quiet_NaN();
    if(!(x < cols && y < rows))
        return;

    wmap_out.ptr(y)[x] = 0; //默认初始权重=0

    float3 vray; //local
    vray.x = vmapLocal.ptr(y)[x];
    if(isnan(vray.x))
        return;

    vray.y = vmapLocal.ptr(y + rows)[x];
    vray.z = vmapLocal.ptr(y + 2 * rows)[x]; //meters

    //const float maxEdgeDist = 10; //in mm //30mm 太长 //改为函数形参 @2018-8-17 17:39:15
    float edgeDistMm = edgeDistMap.ptr(y)[x] / fxy * vray.z * 1e3; //in mm

    float edgeDistFactor = min(1.f, edgeDistMm / maxEdgeDist);
    wmap_out.ptr(y)[x] = edgeDistFactor;
}//edge2wmapKernel

void calcWmap(const MapArr &vmapLocal, const MapArr &nmapLocal, const pcl::device::MaskMap &contMask, MapArr &wmap_out){
    int cols = vmapLocal.cols(),
        rows = vmapLocal.rows() / 3;

    wmap_out.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    calcWmapKernel<<<grid, block>>>(rows, cols, vmapLocal, nmapLocal, contMask, wmap_out);
    
    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize()); //tmp, 暂时企图避免阻塞 @2017-12-6 22:03:13
}//calcWmap

void calcWmap(const MapArr &vmapLocal, const MapArr &nmapLocal, const DeviceArray2D<float> &edgeDistMap, const float fxy, MapArr &wmap_out){
    int cols = vmapLocal.cols(),
        rows = vmapLocal.rows() / 3;

    wmap_out.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    calcWmapKernel<<<grid, block>>>(rows, cols, vmapLocal, nmapLocal, edgeDistMap, fxy, wmap_out);
    
    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize()); //tmp, 暂时企图避免阻塞 @2017-12-6 22:03:13
}//calcWmap

void edge2wmap(const MapArr &vmapLocal, const DeviceArray2D<float> &edgeDistMap, const float fxy, float maxEdgeDist, MapArr &wmap_out){
    int cols = vmapLocal.cols(),
        rows = vmapLocal.rows() / 3;

    wmap_out.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    edge2wmapKernel<<<grid, block>>>(rows, cols, vmapLocal, edgeDistMap, fxy, maxEdgeDist, wmap_out);

    cudaSafeCall(cudaGetLastError());

}//edge2wmap

__global__ void
transformVmapKernel(int rows, int cols, const PtrStep<float> vmap_src, const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    const float qnan = pcl::device::numeric_limits<float>::quiet_NaN();
    if(!(x < cols && y < rows))
        return;

    float3 vsrc, vdst = make_float3(qnan, qnan, qnan);
    vsrc.x = vmap_src.ptr(y)[x];

    if(!isnan(vsrc.x)){
        vsrc.y = vmap_src.ptr(y + rows)[x];
        vsrc.z = vmap_src.ptr(y + 2 * rows)[x];

        vdst = Rmat * vsrc + tvec;

        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
    }

    //确实应放在这里！无论是否 isnan(vdst.x)
    vmap_dst.ptr(y)[x] = vdst.x;
}//transformVmapKernel

void transformVmap( const MapArr &vmap_src, const Mat33 &Rmat, const float3 &tvec, MapArr &vmap_dst ){
    int cols = vmap_src.cols(),
        rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    
    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    transformVmapKernel<<<grid, block>>>(rows, cols, vmap_src, Rmat, tvec, vmap_dst);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize()); //tmp, 暂时企图避免阻塞 @2017-12-6 22:03:13
}//transformVmap

}//namespace zc