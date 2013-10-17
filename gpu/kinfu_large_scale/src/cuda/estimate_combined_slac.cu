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

//#include <pcl/gpu/utils/device/block.hpp>
//#include <pcl/gpu/utils/device/funcattrib.hpp>
#include "device.hpp"

namespace pcl
{
	namespace device
	{
		//typedef double float_type;
		typedef float float_type;

		struct Combined3
		{
			enum
			{
				CTA_SIZE_X = 32,
				CTA_SIZE_Y = 8,
				CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
			};

			struct plus
			{
				__forceinline__ __device__ float
					operator () (const float_type &lhs, const volatile float_type& rhs) const 
				{
					return (lhs + rhs);
				}
			};

			Mat33 Rcurr;
			float3 tcurr;

			PtrStep<float> vmap_curr;
			PtrStep<float> nmap_curr;

			Mat33 Rprev_inv;
			float3 tprev;

			Intr intr;

			PtrStep<float> vmap_g_prev;
			PtrStep<float> nmap_g_prev;

			float distThres;
			float angleThres;

			int cols;
			int rows;

			mutable PtrStep<float_type> gbuf;
			mutable float_type* gbuf_slac_triangle;
			mutable float_type* gbuf_slac_block;
			mutable float_type* gbuf_slac_b;

			__device__ __forceinline__ bool
				search (int x, int y, float3& n, float3& d, float3& s) const
			{
				float3 ncurr;
				ncurr.x = nmap_curr.ptr (y)[x];

				if (isnan (ncurr.x))
					return (false);

				float3 vcurr;
				vcurr.x = vmap_curr.ptr (y       )[x];
				vcurr.y = vmap_curr.ptr (y + rows)[x];
				vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

				float3 vcurr_g = Rcurr * vcurr + tcurr;

				float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);         // prev camera coo space

				int2 ukr;         //projection
				ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);      //4
				ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);                      //4

				if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
					return (false);

				float3 nprev_g;
				nprev_g.x = nmap_g_prev.ptr (ukr.y)[ukr.x];

				if (isnan (nprev_g.x))
					return (false);

				float3 vprev_g;
				vprev_g.x = vmap_g_prev.ptr (ukr.y       )[ukr.x];
				vprev_g.y = vmap_g_prev.ptr (ukr.y + rows)[ukr.x];
				vprev_g.z = vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

				float dist = norm (vprev_g - vcurr_g);
				if (dist > distThres)
					return (false);

				ncurr.y = nmap_curr.ptr (y + rows)[x];
				ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

				float3 ncurr_g = Rcurr * ncurr;

				nprev_g.y = nmap_g_prev.ptr (ukr.y + rows)[ukr.x];
				nprev_g.z = nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

				float sine = norm (cross (ncurr_g, nprev_g));

				if (sine >= angleThres)
					return (false);
				n = nprev_g;
				d = vprev_g;
				s = vcurr_g;
				return (true);
			}

			__device__ __forceinline__ void
				operator () () const
			{
				int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
				int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

				float3 n, d, s;
				bool found_coresp = false;

				if (x < cols && y < rows)
					found_coresp = search (x, y, n, d, s);

				int coo[3];         //projection
				coo[0] = __float2int_rd ( ( s.x + 1.5 ) / 0.375 );      //4
				coo[1] = __float2int_rd ( ( s.y + 1.5 ) / 0.375 );                      //4
				coo[2] = __float2int_rd ( ( s.z - 0.3 ) / 0.375 );

				float row[7];

				if (found_coresp && coo[0]>=0 && coo[0]<8 && coo[1]>=0 && coo[1]<8 && coo[2]>=0 &&coo[2]<8 )
				{
					*(float3*)&row[0] = cross (s, n);
					*(float3*)&row[3] = n;
					row[6] = dot (n, d - s);
					for ( int i = 0; i < 6; i++ )
						for ( int j = 0; j < 24; j++ )
							atomicAdd( gbuf_slac_block + i * 2187 + coo[0] * 81 + coo[1] * 9 + coo[2], 1.0 );
				}
				else
					row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

				int tid = Block::flattenedThreadId ();

				int shift = 0;
#pragma unroll
				for (int i = 0; i < 6; ++i)        //rows
				{
#pragma unroll
					for (int j = i; j < 7; ++j)          // cols + b
					{
						gbuf.ptr (shift++)[ (blockIdx.x + gridDim.x * blockIdx.y) * CTA_SIZE + tid ] = row[i]*row[j];
					}
				}
			}
		};

		__global__ void
			combinedKernel3 (const Combined3 cs) 
		{
			cs ();
		}

		struct TranformReduction3
		{
			enum
			{
				CTA_SIZE = 512,
				STRIDE = CTA_SIZE,

				B = 6, COLS = 6, ROWS = 6, DIAG = 6,
				UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
				TOTAL = UPPER_DIAG_MAT + B,

				GRID_X = TOTAL
			};

			PtrStep<float_type> gbuf;
			int length;
			mutable float_type* output;

			__device__ __forceinline__ void
				operator () () const
			{
				const float_type *beg = gbuf.ptr (blockIdx.x);
				const float_type *end = beg + length;

				int tid = threadIdx.x;

				float_type sum = 0.f;
				for (const float_type *t = beg + tid; t < end; t += STRIDE)
					sum += *t;

				__shared__ float_type smem[CTA_SIZE];

				smem[tid] = sum;
				__syncthreads ();

				Block::reduce<CTA_SIZE>(smem, Combined3::plus ());

				if (tid == 0)
					output[blockIdx.x] = smem[0];
			}
		};

		__global__ void
			TransformEstimatorKernel3 (const TranformReduction3 tr) 
		{
			tr ();
		}
	}
}

void pcl::device::estimateCombinedEx (const Mat33& Rcurr, const float3& tcurr, 
	const MapArr& vmap_curr, const MapArr& nmap_curr, 
	const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
	const MapArr& vmap_g_prev, const MapArr& nmap_g_prev, 
	float distThres, float angleThres,
	DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, 
	float_type* matrixA_host, float_type* vectorB_host,
	DeviceArray<float>& gbuf_slac_triangle, DeviceArray<float>& gbuf_slac_block, DeviceArray<float>& gbuf_slac_b,
	float* matrixSLAC_A_host, float* matrixSLAC_block_host, float* vectorSLAC_b_host)
{
	int cols = vmap_curr.cols ();
	int rows = vmap_curr.rows () / 3;
	dim3 block (Combined3::CTA_SIZE_X, Combined3::CTA_SIZE_Y);
	dim3 grid (1, 1, 1);
	grid.x = divUp (cols, block.x);
	grid.y = divUp (rows, block.y);

	Combined3 cs3;

	cs3.Rcurr = Rcurr;
	cs3.tcurr = tcurr;

	cs3.vmap_curr = vmap_curr;
	cs3.nmap_curr = nmap_curr;

	cs3.Rprev_inv = Rprev_inv;
	cs3.tprev = tprev;

	cs3.intr = intr;

	cs3.vmap_g_prev = vmap_g_prev;
	cs3.nmap_g_prev = nmap_g_prev;

	cs3.distThres = distThres;
	cs3.angleThres = angleThres;

	cs3.cols = cols;
	cs3.rows = rows;

	cs3.gbuf = gbuf;
	cs3.gbuf_slac_triangle = gbuf_slac_triangle;
	cs3.gbuf_slac_block = gbuf_slac_block;
	cs3.gbuf_slac_b = gbuf_slac_b;

	combinedKernel3<<<grid, block>>>(cs3);
	cudaSafeCall ( cudaGetLastError () );

	TranformReduction3 tr3;
	tr3.gbuf = gbuf;
	tr3.length = cols * rows;
	tr3.output = mbuf;

	TransformEstimatorKernel3<<<TranformReduction3::TOTAL, TranformReduction3::CTA_SIZE>>>(tr3);
	cudaSafeCall (cudaGetLastError ());
	cudaSafeCall (cudaDeviceSynchronize ());

	float_type host_data[TranformReduction3::TOTAL];
	mbuf.download (host_data);

	int shift = 0;
	for (int i = 0; i < 6; ++i) {		//rows
		for (int j = i; j < 7; ++j) {   // cols + b
			float_type value = host_data[shift++];
			if (j == 6)       // vector b
				vectorB_host[i] = value;
			else
				matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
		}
	}
}
