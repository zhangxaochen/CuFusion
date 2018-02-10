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
//#include <pcl/gpu/features/device/eigen.hpp>

#include <limits.h>

namespace pcl
{
  namespace device
  {
    enum
    {
      kx = 7,
      ky = 7,
      STEP = 1
    };

    __global__ void
    computeNmapKernelEigen (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

      if (isnan (vmap.ptr (v)[u]))
        return;

      int ty = min (v - ky / 2 + ky, rows - 1);
      int tx = min (u - kx / 2 + kx, cols - 1);

      float3 centroid = make_float3 (0.f, 0.f, 0.f);
      int counter = 0;
#if 10   //orig
      for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
        for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
        {
          float v_x = vmap.ptr (cy)[cx];
          if (!isnan (v_x))
          {
            centroid.x += v_x;
            centroid.y += vmap.ptr (cy + rows)[cx];
            centroid.z += vmap.ptr (cy + 2 * rows)[cx];
            ++counter;
          }
        }
#else   //zc: 改成, 若某邻域 pt.z 距离中心 p0.z 超过某阈值, 不计入PCA计算. 目的: 边缘点,排除大深度差造成错的法向估计
      //@2017-12-22 17:18:10    
      //貌似不管用 @2017-12-23 16:58:05
      float p0z = vmap.ptr (v + 2 * rows)[u];
      for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP){
        for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP){
          float v_x = vmap.ptr (cy)[cx];
          if (!isnan (v_x))
          {
            float v_y = vmap.ptr (cy + rows)[cx],
                  v_z = vmap.ptr (cy + 2 * rows)[cx];
            if(abs(v_z - p0z) <= 0.02){ //vmap 量纲应该是 meters
                centroid.x += v_x;
                centroid.y += v_y;
                centroid.z += v_z;
                ++counter;
            }
          }//if-isnan-vx
        }//for-cx
      }//for-cy
#endif

      if (counter < kx * ky / 2)
        return;

      centroid *= 1.f / counter;

      float cov[] = {0, 0, 0, 0, 0, 0};

      for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
        for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
        {
          float3 v;
          v.x = vmap.ptr (cy)[cx];
          if (isnan (v.x))
            continue;

          v.y = vmap.ptr (cy + rows)[cx];
          v.z = vmap.ptr (cy + 2 * rows)[cx];

          float3 d = v - centroid;

          cov[0] += d.x * d.x;               //cov (0, 0)
          cov[1] += d.x * d.y;               //cov (0, 1)
          cov[2] += d.x * d.z;               //cov (0, 2)
          cov[3] += d.y * d.y;               //cov (1, 1)
          cov[4] += d.y * d.z;               //cov (1, 2)
          cov[5] += d.z * d.z;               //cov (2, 2)
        }

      typedef Eigen33::Mat33 Mat33;
      Eigen33 eigen33 (cov);

      Mat33 tmp;
      Mat33 vec_tmp;
      Mat33 evecs;
      float3 evals;
      eigen33.compute (tmp, vec_tmp, evecs, evals);

      float3 n = normalized (evecs[0]);

      u = threadIdx.x + blockIdx.x * blockDim.x;
      v = threadIdx.y + blockIdx.y * blockDim.y;

      nmap.ptr (v       )[u] = n.x;
      nmap.ptr (v + rows)[u] = n.y;
      nmap.ptr (v + 2 * rows)[u] = n.z;
    }

    __global__ void
    computeNormalsContourcueKernel(const PtrStepSz<ushort> src,const PtrStepSz<float> grandient_x,const PtrStepSz<float> grandient_y, PtrStep<float> dst,float fx, float fy, float cx, float cy)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int row=src.rows,col=src.cols;
        if (x<col && y<row)
        {
            float dx=0,dy=0,dz=-1;
            //先计算逆矩阵
            float depth = src.ptr(y)[x];
            float du = grandient_x.ptr(y)[x],dv=grandient_y.ptr(y)[x];
            float m00 = 1.0/fx*(depth+(x-cx)*du),
                m01 = 1.0/fx*(x-cx)*dv,
                m10 = 1.0/fy*(y-cy)*du,
                m11 = 1.0/fy*(depth+(y-cy)*dv);
            float det = m00*m11-m01*m10;
            //printf("%f \n",det);
            if(abs(det) < 1e-5)
            {
                const float qnan = numeric_limits<float>::quiet_NaN();
                dx = dy = dz = qnan;
            }
            else
            {
                float m00_inv = m11/det,
                    m01_inv = -m01/det,
                    m10_inv = -m10/det,
                    m11_inv = m00/det;
                dx = du*m00_inv+dv*m10_inv;
                dy = du*m01_inv+dv*m11_inv;
                //normalize
                float norm=sqrt(dx*dx+dy*dy+dz*dz);
                dx = dx/norm;
                dy = dy/norm;
                dz = dz/norm;
            }
            dst.ptr(y)[x]=dx;
            dst.ptr(y+row)[x]=dy;
            dst.ptr(y+2*row)[x]=dz;
            /*printf("%f %f %f \n",dx,dy,dz);*/
        }
    }//computeNormalsContourcueKernel

    __global__ void
    diffVmapsKernel(int rows, int cols, const PtrStepSz<float> vmap1, const PtrStepSz<float> vmap2, const Mat33 Rmat, PtrStepSz<short> diffDmapOut){
        int x = threadIdx.x + blockIdx.x * blockDim.x,
            y = threadIdx.y + blockIdx.y * blockDim.y;

        if(!(x < cols && y < rows))
            return;

        float3 v1, v2;
        v1.x = vmap1.ptr(y)[x];
        v2.x = vmap2.ptr(y)[x];

        //默认初始化为无效值
        diffDmapOut.ptr(y)[x] = SHRT_MIN;

        if(isnan(v1.x) || isnan(v2.x))
            return;

        v1.y = vmap1.ptr(y + rows)[x];
        v1.z = vmap1.ptr(y + 2 * rows)[x];

        v2.y = vmap2.ptr(y + rows)[x];
        v2.z = vmap2.ptr(y + 2 * rows)[x];

        float3 dv = v1-v2; //deltaV, 无位置, 转换坐标系时只需要 R*dv, 不需要带 t
        dv = Rmat * dv; //其实只要 Rmat 第三行, 求 z 就够了, 懒得弄

        diffDmapOut.ptr(y)[x] = short(dv.z);
    }//diffVmapsKernel


    __global__ void
    diffDmapsKernel(int rows, int cols, const PtrStepSz<ushort> dmap1, const PtrStepSz<ushort> dmap2, PtrStepSz<short> diffDmapOut){
        int x = threadIdx.x + blockIdx.x * blockDim.x,
            y = threadIdx.y + blockIdx.y * blockDim.y;

        if(!(x < cols && y < rows))
            return;

        diffDmapOut.ptr(y)[x] = short(dmap1.ptr(y)[x] - dmap2.ptr(y)[x]); //有正负
    }//diffDmapsKernel

    __global__ void
    test_depth_uncertainty_kernel(int rows, int cols, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, const Mat33 dRmat_i_i1, const float3 dTvec_i_i1, PtrStepSz<float> uncertaintyMap){
        int x = threadIdx.x + blockIdx.x * blockDim.x,
            y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x >= cols || y >= rows)
            return;

        uncertaintyMap.ptr(y)[x] = 1e5; //默认暂定 ？; 比如 vmap 无效区域, 

        float3 vcurr, ncurr;
        vcurr.x = vmap.ptr(y)[x];
        ncurr.x = nmap.ptr(y)[x];
        if(isnan(vcurr.x) || isnan(ncurr.x))
            return;

        vcurr.y = vmap.ptr(y + rows)[x];
        vcurr.z = vmap.ptr(y + 2 * rows)[x];

        float3 vcp = dRmat_i_i1 * vcurr + dTvec_i_i1; //curr pt at prev cam coo

        //用什么表征 uncertainty?
        const int M2MM = 1000;
        //V1: pt depth(z) curr-prev; 与配准后深度差图区别是: 配准可能有系统误差? 但此处 dR, dt 其实是先配准求得的, 可能已包含系统误差?   @2017-12-8 10:56:38
        //uncertaintyMap.ptr(y)[x] = M2MM * (vcurr.z - vcp.z); //m2mm

        //V2: vc-vp 向量在法向上的投影, 毫米尺度
        ncurr.y = nmap.ptr(y + rows)[x];
        ncurr.z = nmap.ptr(y + 2 * rows)[x];

        if(ncurr.z > 0) //确保 local 法向量都朝向相机视点
            ncurr *= -1;

        uncertaintyMap.ptr(y)[x] = dot(vcurr - vcp, ncurr) * M2MM;
    }//test_depth_uncertainty_kernel

  }//namespace device
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::computeNormalsEigen (const MapArr& vmap, MapArr& nmap)
{
  int cols = vmap.cols ();
  int rows = vmap.rows () / 3;

  nmap.create (vmap.rows (), vmap.cols ());

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeNmapKernelEigen<<<grid, block>>>(rows, cols, vmap, nmap);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::device::computeNormalsContourcue(const Intr& intr, const DepthMap& depth,const MapArr& grandient_x, const MapArr& grandient_y, MapArr& nmap)
{
    nmap.create(depth.rows()*3,depth.cols());
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    float fx=intr.fx,fy=intr.fy,cx=intr.cx,cy=intr.cy;
    computeNormalsContourcueKernel<<<grid,block>>>(depth, grandient_x, grandient_y, nmap, fx,fy,cx,cy);

    cudaSafeCall ( cudaGetLastError () );
}

void pcl::device::diffVmaps(const MapArr &vmap1, const MapArr &vmap2, const Mat33 &Rmat, DeviceArray2D<short> &diffDmapOut){
    int cols = vmap1.cols(),
        rows = vmap1.rows() / 3;

    diffDmapOut.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    diffVmapsKernel<<<grid, block>>>(rows, cols, vmap1, vmap2, Rmat, diffDmapOut);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}//diffVmaps


void pcl::device::diffDmaps(const DepthMap &dmap1, const DepthMap &dmap2, DeviceArray2D<short> &diffDmapOut){
    int cols = dmap1.cols(),
        rows = dmap1.rows();

    diffDmapOut.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    diffDmapsKernel<<<grid, block>>>(rows, cols, dmap1, dmap2, diffDmapOut);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}//diffDmaps

//overload
void pcl::device::diffDmaps(const PtrStepSz<ushort> &dmap1, const PtrStepSz<ushort> &dmap2, DeviceArray2D<short> &diffDmapOut){
    int cols = dmap1.cols,
        rows = dmap1.rows;

    diffDmapOut.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    diffDmapsKernel<<<grid, block>>>(rows, cols, dmap1, dmap2, diffDmapOut);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}//diffDmaps

void pcl::device::test_depth_uncertainty(const MapArr &vmap, const MapArr &nmap, const Mat33 &dRmat_i_i1, const float3 & dTvec_i_i1, DeviceArray2D<float> &uncertaintyMap){
    int cols = vmap.cols(),
        rows = vmap.rows() / 3;

    uncertaintyMap.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    test_depth_uncertainty_kernel<<<grid, block>>>(rows, cols, vmap, nmap, dRmat_i_i1, dTvec_i_i1, uncertaintyMap);

    cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}//test_depth_uncertainty
