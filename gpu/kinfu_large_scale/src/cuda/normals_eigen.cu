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
            //ÏÈ¼ÆËãÄæ¾ØÕó
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