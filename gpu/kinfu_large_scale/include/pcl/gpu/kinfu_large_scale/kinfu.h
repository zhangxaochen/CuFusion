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

#ifndef PCL_KINFU_KINFUTRACKER_HPP_
#define PCL_KINFU_KINFUTRACKER_HPP_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu_large_scale/pixel_rgb.h>
#include <pcl/gpu/kinfu_large_scale/tsdf_volume.h>
#include <pcl/gpu/kinfu_large_scale/color_volume.h>
#include <pcl/gpu/kinfu_large_scale/raycaster.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <vector>

#include <pcl/gpu/kinfu_large_scale/cyclical_buffer.h>
#include <pcl/gpu/kinfu_large_scale/standalone_marching_cubes.h>

#include <pcl/io/ply_io.h>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <AHCPlaneFitter.hpp> //peac代码: http://www.merl.com/demos/point-plane-slam
#include "zcAhcUtility.h"
//#include <pcl/kdtree/kdtree_flann.h> //zc: 别放这里: error C2872: 'flann' : ambiguous symbol ...could be 'flann' ...or 'cv::flann'

//////////////////////////////
typedef pcl::PointXYZ PtType;
typedef pcl::PointCloud<PtType> CloudType;
PCL_EXPORTS CloudType::Ptr cvMat2PointCloud(const cv::Mat &dmat, const pcl::device::Intr &intr);

//平面拟合, 提取分割
//以下拷贝自 plane_fitter.cpp
// pcl::PointCloud interface for our ahc::PlaneFitter
template<class PointT>
struct OrganizedImage3D {
	const pcl::PointCloud<PointT>& cloud;
	//NOTE: pcl::PointCloud from OpenNI uses meter as unit,
	//while ahc::PlaneFitter assumes mm as unit!!!
	const double unitScaleFactor;

	OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1000) {}
	int width() const { return cloud.width; }
	int height() const { return cloud.height; }
	bool get(const int row, const int col, double& x, double& y, double& z) const {
		const PointT& pt=cloud.at(col,row);
		x=pt.x; y=pt.y; z=pt.z;
		return pcl_isnan(z)==0; //return false if current depth is NaN
	}
};
//typedef OrganizedImage3D<pcl::PointXYZRGBA> RGBDImage;
typedef OrganizedImage3D<PtType> RGBDImage;
typedef ahc::PlaneFitter<RGBDImage> PlaneFitter;

PCL_EXPORTS void dbgAhcPeac_kinfu( const RGBDImage *rgbdObj, PlaneFitter *pf);


//////////////////////////////
namespace pcl
{
  namespace gpu
  {        
    struct FramedTransformation {
	  enum RegistrationType { Kinfu = 0, DirectApply = 1, InitializeOnly = 2, IncrementalOnly = 3 };
	  enum ActionFlag {
		  ResetFlag = 0x1,					// if reset at the very beginning
		  IgnoreRegistrationFlag = 0x2,		// if discard the registration
		  IgnoreIntegrationFlag = 0x4,		// if discard integration
		  PushMatrixHashFlag = 0x8,			// if push the transformation matrix into the hash table
		  SavePointCloudFlag = 0x10,		// if save point cloud after execution
		  SaveAbsoluteMatrix = 0x20,		// if save absolute matrix, work with IgnoreIntegrationFlag
		  ExtractSLACMatrix = 0x40,			// if extract SLAC matrix
	  };

      int id1_;
	  int id2_;
      int frame_;
	  RegistrationType type_;
	  int flag_;
      Eigen::Matrix4f transformation_;
	  FramedTransformation() : type_( Kinfu ), flag_( 0 ) {}
      FramedTransformation( int id1, int id2, int f, Eigen::Matrix4f t ) : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_( t ), type_( DirectApply ), flag_( 0 ) {}
      FramedTransformation( int id1, int id2, int f, Eigen::Matrix4f t, RegistrationType tp, int flg ) 
		  : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_( t ), type_( tp ), flag_( flg ) {}
    };

	struct Coordinate {
	public:
		int idx_[ 8 ];
		float val_[ 8 ];
		float nval_[ 8 ];
	};

	class ControlGrid
	{
	public:
		ControlGrid(void) {};
		~ControlGrid(void) {};

	public:
		std::vector< Eigen::Vector3f > ctr_;
		int resolution_;
		float length_;
		float unit_length_;
		int min_bound_[ 3 ];
		int max_bound_[ 3 ];
		Eigen::Matrix4f init_pose_;
		Eigen::Matrix4f init_pose_inv_;

	public:
		void ResetBBox() { min_bound_[ 0 ] = min_bound_[ 1 ] = min_bound_[ 2 ] = 100000000; max_bound_[ 0 ] = max_bound_[ 1 ] = max_bound_[ 2 ] = -100000000; }
		void RegulateBBox( float vi, int i ) {
			int v0 = ( int )floor( vi / unit_length_ ) - 1;
			int v1 = ( int )ceil( vi / unit_length_ ) + 1;
			if ( v0 < min_bound_[ i ] ) {
				min_bound_[ i ] = v0;
			}
			if ( v1 > max_bound_[ i ] ) {
				max_bound_[ i ] = v1;
			}
		}
		inline int GetIndex( int i, int j, int k ) {
			return i + j * ( resolution_ + 1 ) + k * ( resolution_ + 1 ) * ( resolution_ + 1 );
		}
		inline bool GetCoordinate( const Eigen::Vector3f & pt, Coordinate & coo ) {
			int corner[ 3 ] = {
				( int )floor( pt( 0 ) / unit_length_ ),
				( int )floor( pt( 1 ) / unit_length_ ),
				( int )floor( pt( 2 ) / unit_length_ )
			};

			if ( corner[ 0 ] < 0 || corner[ 0 ] >= resolution_
				|| corner[ 1 ] < 0 || corner[ 1 ] >= resolution_
				|| corner[ 2 ] < 0 || corner[ 2 ] >= resolution_ )
				return false;

			float residual[ 3 ] = {
				pt( 0 ) / unit_length_ - corner[ 0 ],
				pt( 1 ) / unit_length_ - corner[ 1 ],
				pt( 2 ) / unit_length_ - corner[ 2 ]
			};
			// for speed, skip sanity check
			coo.idx_[ 0 ] = GetIndex( corner[ 0 ], corner[ 1 ], corner[ 2 ] );
			coo.idx_[ 1 ] = GetIndex( corner[ 0 ], corner[ 1 ], corner[ 2 ] + 1 );
			coo.idx_[ 2 ] = GetIndex( corner[ 0 ], corner[ 1 ] + 1, corner[ 2 ] );
			coo.idx_[ 3 ] = GetIndex( corner[ 0 ], corner[ 1 ] + 1, corner[ 2 ] + 1 );
			coo.idx_[ 4 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ], corner[ 2 ] );
			coo.idx_[ 5 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ], corner[ 2 ] + 1 );
			coo.idx_[ 6 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ] + 1, corner[ 2 ] );
			coo.idx_[ 7 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ] + 1, corner[ 2 ] + 1 );

			coo.val_[ 0 ] = ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			coo.val_[ 1 ] = ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] );
			coo.val_[ 2 ] = ( 1 - residual[ 0 ] ) * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			coo.val_[ 3 ] = ( 1 - residual[ 0 ] ) * ( residual[ 1 ] ) * ( residual[ 2 ] );
			coo.val_[ 4 ] = ( residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			coo.val_[ 5 ] = ( residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] );
			coo.val_[ 6 ] = ( residual[ 0 ] ) * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			coo.val_[ 7 ] = ( residual[ 0 ] ) * ( residual[ 1 ] ) * ( residual[ 2 ] );

			return true;
		}
		inline void GetPosition( const Coordinate & coo, Eigen::Vector3f & pos ) {
			//cout << coo.idx_[ 0 ] << ", " << coo.idx_[ 1 ] << ", " << coo.idx_[ 2 ] << ", " << coo.idx_[ 3 ] << ", " << coo.idx_[ 4 ] << ", " << coo.idx_[ 5 ] << ", " << coo.idx_[ 6 ] << ", " << coo.idx_[ 7 ] << ", " << endl;
			//cout << coo.val_[ 0 ] << ", " << coo.val_[ 1 ] << ", " << coo.val_[ 2 ] << ", " << coo.val_[ 3 ] << ", " << coo.val_[ 4 ] << ", " << coo.val_[ 5 ] << ", " << coo.val_[ 6 ] << ", " << coo.val_[ 7 ] << ", " << endl;
			//std::cout << ctr_[ coo.idx_[ 0 ] ] << std::endl;
			pos = coo.val_[ 0 ] * ctr_[ coo.idx_[ 0 ] ] + coo.val_[ 1 ] * ctr_[ coo.idx_[ 1 ] ]
				+ coo.val_[ 2 ] * ctr_[ coo.idx_[ 2 ] ] + coo.val_[ 3 ] * ctr_[ coo.idx_[ 3 ] ]
				+ coo.val_[ 4 ] * ctr_[ coo.idx_[ 4 ] ] + coo.val_[ 5 ] * ctr_[ coo.idx_[ 5 ] ]
				+ coo.val_[ 6 ] * ctr_[ coo.idx_[ 6 ] ] + coo.val_[ 7 ] * ctr_[ coo.idx_[ 7 ] ];
		}
		inline void Init( int res, double len, const Eigen::Matrix4f & init_pose ) {
			init_pose_ = init_pose;
			init_pose_inv_ = init_pose.inverse();
			resolution_ = res;
			length_ = len;
			unit_length_ = length_ / resolution_;

			int total = ( res + 1 ) * ( res + 1 ) * ( res + 1 );
			ctr_.resize( total );
			for ( int i = 0; i <= resolution_; i++ ) {
				for ( int j = 0; j <= resolution_; j++ ) {
					for ( int k = 0; k <= resolution_; k++ ) {
						Eigen::Vector4f pos( i * unit_length_, j * unit_length_, k * unit_length_, 1 );
						Eigen::Vector4f ppos = pos; // * 1.05;
						ctr_[ GetIndex( i, j, k ) ]( 0 ) = ppos( 0 );
						ctr_[ GetIndex( i, j, k ) ]( 1 ) = ppos( 1 );
						ctr_[ GetIndex( i, j, k ) ]( 2 ) = ppos( 2 );
					}
				}
			}
		}
	};

	struct SLACPoint {
	public:
		int idx_[ 8 ];
		float n_[ 3 ];
		float val_[ 8 ];
		float nval_[ 8 ];
		float p_[ 3 ];
	};

	class SLACPointCloud
	{
	public:
		typedef boost::shared_ptr< SLACPointCloud > Ptr;

	public:
		SLACPointCloud( int index = 0, int resolution = 12, float length = 3.0f ) {
			resolution_ = resolution;
			length_ = length;
			unit_length_ = length / resolution;
			index_ = index;
			nper_ = ( resolution_ + 1 ) * ( resolution_ + 1 ) * ( resolution_ + 1 ) * 3;
			offset_ = index * nper_;
		}

		~SLACPointCloud(void) {}

	public:
		int resolution_;
		int nper_;
		int offset_;
		int index_;
		float length_;
		float unit_length_;

	public:
		std::vector< SLACPoint > points_;

	public:
		void Init( PointCloud< PointXYZ >::Ptr pc, PointCloud< PointXYZ >::Ptr nc ) {
			for ( int i = 0; i < ( int )pc->points.size(); i++ ) {
				float x[ 6 ];
				x[ 0 ] = pc->points[ i ].x;
				x[ 1 ] = pc->points[ i ].y;
				x[ 2 ] = pc->points[ i ].z;
				x[ 3 ] = nc->points[ i ].x;
				x[ 4 ] = nc->points[ i ].y;
				x[ 5 ] = nc->points[ i ].z;
				points_.resize( points_.size() + 1 );
				if ( GetCoordinate( x, points_.back() ) == false ) {
					printf( "Error!!\n" );
					return;
				}
			}
		}

	public:
		bool IsValidPoint( int i ) {
			if ( _isnan( points_[ i ].p_[ 0 ] ) || _isnan( points_[ i ].p_[ 1 ] ) || _isnan( points_[ i ].p_[ 2 ] ) || 
				_isnan( points_[ i ].n_[ 0 ] ) || _isnan( points_[ i ].n_[ 1 ] ) || _isnan( points_[ i ].n_[ 2 ] ) )
				return false;
			else
				return true;
		}

	public:
		void UpdateAllNormal( const Eigen::VectorXd & ctr ) {
			for ( int i = 0; i < ( int )points_.size(); i++ ) {
				UpdateNormal( ctr, points_[ i ] );
			}
		}

		void UpdateAllPointPN( const Eigen::VectorXd & ctr ) {
			for ( int i = 0; i < ( int )points_.size(); i++ ) {
				UpdateNormal( ctr, points_[ i ] );
				Eigen::Vector3f pos = UpdatePoint( ctr, points_[ i ] );
				points_[ i ].p_[ 0 ] = pos( 0 );
				points_[ i ].p_[ 1 ] = pos( 1 );
				points_[ i ].p_[ 2 ] = pos( 2 );
			}
		}

		inline int GetIndex( int i, int j, int k ) {
			return i + j * ( resolution_ + 1 ) + k * ( resolution_ + 1 ) * ( resolution_ + 1 );
		}

		inline void UpdateNormal( const Eigen::VectorXd & ctr, SLACPoint & point ) {
			for ( int i = 0; i < 3; i++ ) {
				point.n_[ i ] = 0.0f;
				for ( int j = 0; j < 8; j++ ) {
					point.n_[ i ] += point.nval_[ j ] * ( float )ctr( point.idx_[ j ] + i + offset_ );
				}
			}
			float len = sqrt( point.n_[ 0 ] * point.n_[ 0 ] + point.n_[ 1 ] * point.n_[ 1 ] + point.n_[ 2 ] * point.n_[ 2 ] );
			point.n_[ 0 ] /= len;
			point.n_[ 1 ] /= len;
			point.n_[ 2 ] /= len;
		}

		inline void UpdatePose( const Eigen::Matrix4f & inc_pose ) {
			Eigen::Vector4f p, n;
			for ( int i = 0; i < ( int )points_.size(); i++ ) {
				SLACPoint & point = points_[ i ];
				//cout << point.p_[ 0 ] << endl << point.p_[ 1 ] << endl << point.p_[ 2 ] << endl << point.p_[ 3 ] << endl << point.p_[ 4 ] << endl << point.p_[ 5 ] << endl;
				p = inc_pose * Eigen::Vector4f( point.p_[ 0 ], point.p_[ 1 ], point.p_[ 2 ], 1 );
				n = inc_pose * Eigen::Vector4f( point.n_[ 0 ], point.n_[ 1 ], point.n_[ 2 ], 0 );
				point.p_[ 0 ] = p( 0 );
				point.p_[ 1 ] = p( 1 );
				point.p_[ 2 ] = p( 2 );
				point.n_[ 0 ] = n( 0 );
				point.n_[ 1 ] = n( 1 );
				point.n_[ 2 ] = n( 2 );
				//cout << point.p_[ 0 ] << endl << point.p_[ 1 ] << endl << point.p_[ 2 ] << endl << point.p_[ 3 ] << endl << point.p_[ 4 ] << endl << point.p_[ 5 ] << endl;
				//cout << endl;
			}
		}

		inline Eigen::Vector3f UpdatePoint( const Eigen::VectorXd & ctr, SLACPoint & point ) {
			Eigen::Vector3f pos;
			for ( int i = 0; i < 3; i++ ) {
				pos( i ) = 0.0;
				for ( int j = 0; j < 8; j++ ) {
					pos( i ) += point.val_[ j ] * ( float )ctr( point.idx_[ j ] + i + offset_ );
				}
			}
			return pos;
		}

		inline bool GetCoordinate( float pt[ 6 ], SLACPoint & point ) {
			point.p_[ 0 ] = pt[ 0 ];
			point.p_[ 1 ] = pt[ 1 ];
			point.p_[ 2 ] = pt[ 2 ];

			int corner[ 3 ] = {
				( int )floor( pt[ 0 ] / unit_length_ ),
				( int )floor( pt[ 1 ] / unit_length_ ),
				( int )floor( pt[ 2 ] / unit_length_ )
			};

			if ( corner[ 0 ] < 0 || corner[ 0 ] >= resolution_
				|| corner[ 1 ] < 0 || corner[ 1 ] >= resolution_
				|| corner[ 2 ] < 0 || corner[ 2 ] >= resolution_ )
				return false;

			float residual[ 3 ] = {
				pt[ 0 ] / unit_length_ - corner[ 0 ],
				pt[ 1 ] / unit_length_ - corner[ 1 ],
				pt[ 2 ] / unit_length_ - corner[ 2 ]
			};
			// for speed, skip sanity check
			point.idx_[ 0 ] = GetIndex( corner[ 0 ], corner[ 1 ], corner[ 2 ] ) * 3;
			point.idx_[ 1 ] = GetIndex( corner[ 0 ], corner[ 1 ], corner[ 2 ] + 1 ) * 3;
			point.idx_[ 2 ] = GetIndex( corner[ 0 ], corner[ 1 ] + 1, corner[ 2 ] ) * 3;
			point.idx_[ 3 ] = GetIndex( corner[ 0 ], corner[ 1 ] + 1, corner[ 2 ] + 1 ) * 3;
			point.idx_[ 4 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ], corner[ 2 ] ) * 3;
			point.idx_[ 5 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ], corner[ 2 ] + 1 ) * 3;
			point.idx_[ 6 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ] + 1, corner[ 2 ] ) * 3;
			point.idx_[ 7 ] = GetIndex( corner[ 0 ] + 1, corner[ 1 ] + 1, corner[ 2 ] + 1 ) * 3;

			point.val_[ 0 ] = ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			point.val_[ 1 ] = ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] );
			point.val_[ 2 ] = ( 1 - residual[ 0 ] ) * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			point.val_[ 3 ] = ( 1 - residual[ 0 ] ) * ( residual[ 1 ] ) * ( residual[ 2 ] );
			point.val_[ 4 ] = ( residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			point.val_[ 5 ] = ( residual[ 0 ] ) * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] );
			point.val_[ 6 ] = ( residual[ 0 ] ) * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] );
			point.val_[ 7 ] = ( residual[ 0 ] ) * ( residual[ 1 ] ) * ( residual[ 2 ] );

			pt[ 3 ] /= unit_length_;
			pt[ 4 ] /= unit_length_;
			pt[ 5 ] /= unit_length_;
			point.nval_[ 0 ] = 
				- pt[ 3 ] * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 4 ] * ( 1 - residual[ 0 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 5 ] * ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] );
			point.nval_[ 1 ] = 
				- pt[ 3 ] * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] ) 
				- pt[ 4 ] * ( 1 - residual[ 0 ] ) * ( residual[ 2 ] ) 
				+ pt[ 5 ] * ( 1 - residual[ 0 ] ) * ( 1 - residual[ 1 ] );
			point.nval_[ 2 ] = 
				- pt[ 3 ] * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] ) 
				+ pt[ 4 ] * ( 1 - residual[ 0 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 5 ] * ( 1 - residual[ 0 ] ) * ( residual[ 1 ] );
			point.nval_[ 3 ] = 
				- pt[ 3 ] * ( residual[ 1 ] ) * ( residual[ 2 ] ) 
				+ pt[ 4 ] * ( 1 - residual[ 0 ] ) * ( residual[ 2 ] ) 
				+ pt[ 5 ] * ( 1 - residual[ 0 ] ) * ( residual[ 1 ] );
			point.nval_[ 4 ] = 
				  pt[ 3 ] * ( 1 - residual[ 1 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 4 ] * ( residual[ 0 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 5 ] * ( residual[ 0 ] ) * ( 1 - residual[ 1 ] );
			point.nval_[ 5 ] = 
				  pt[ 3 ] * ( 1 - residual[ 1 ] ) * ( residual[ 2 ] ) 
				- pt[ 4 ] * ( residual[ 0 ] ) * ( residual[ 2 ] ) 
				+ pt[ 5 ] * ( residual[ 0 ] ) * ( 1 - residual[ 1 ] );
			point.nval_[ 6 ] = 
				  pt[ 3 ] * ( residual[ 1 ] ) * ( 1 - residual[ 2 ] ) 
				+ pt[ 4 ] * ( residual[ 0 ] ) * ( 1 - residual[ 2 ] ) 
				- pt[ 5 ] * ( residual[ 0 ] ) * ( residual[ 1 ] );
			point.nval_[ 7 ] = 
				  pt[ 3 ] * ( residual[ 1 ] ) * ( residual[ 2 ] ) 
				+ pt[ 4 ] * ( residual[ 0 ] ) * ( residual[ 2 ] ) 
				+ pt[ 5 ] * ( residual[ 0 ] ) * ( residual[ 1 ] );

			point.n_[ 0 ] = pt[ 3 ];
			point.n_[ 1 ] = pt[ 4 ];
			point.n_[ 2 ] = pt[ 5 ];

			return true;
		}
	};

	typedef std::pair< int, int > CorrespondencePair;
	struct Correspondence {
	public:
		typedef boost::shared_ptr< Correspondence > Ptr;

	public:
		int idx0_, idx1_;
		Eigen::Matrix4f trans_;
		std::vector< CorrespondencePair > corres_;
	public:
		Correspondence( int i0, int i1 ) : idx0_( i0 ), idx1_( i1 ) {}
	};
    /** \brief KinfuTracker class encapsulates implementation of Microsoft Kinect Fusion algorithm
      * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
      */
    class PCL_EXPORTS KinfuTracker
    {
      public:

        /** \brief Pixel type for rendered image. */
        typedef pcl::gpu::PixelRGB PixelRGB;

        typedef DeviceArray2D<PixelRGB> View;
        typedef DeviceArray2D<unsigned short> DepthMap;

        typedef pcl::PointXYZ PointType;
        typedef pcl::Normal NormalType;

        void 
        performLastScan (){perform_last_scan_ = true; PCL_WARN ("Kinfu will exit after next shift\n");}
        
		bool
		intersect( int bbox[ 6 ] );

        bool
        isFinished (){return (finished_);}

		int round( double x ) {
			return static_cast< int >( floor( x + 0.5 ) );
		}

		bool UVD2XYZ( int u, int v, unsigned short d, double & x, double & y, double & z ) {
			if ( d > 0 ) {
				z = d / 1000.0;
				x = ( u - cx_ ) * z / fx_;
				y = ( v - cy_ ) * z / fy_;
				return true;
			} else {
				return false;
			}
		}

		bool XYZ2UVD( double x, double y, double z, int & u, int & v, unsigned short & d ) {
			if ( z > 0 ) {
				u = round( x * fx_ / z + cx_ );
				v = round( y * fy_ / z + cy_ );
				d = static_cast< unsigned short >( round( z * 1000.0 ) );
				return ( u >= 0 && u < 640 && v >= 0 && v < 480 );
			} else {
				return false;
			}
		}

		void deformDepthImage( std::vector<unsigned short> & depth ) {
			slac_mutex_.lock();

			//cout << "deform" << endl;
			std::vector< unsigned short > temp;
			temp.resize( depth.size() );
			for ( int i = 0; i < 640 * 480; i++ ) {
				temp[ i ] = depth[ i ];
				depth[ i ] = 0;
			}
			int uu, vv;
			unsigned short dd;
			double x, y, z;
			for ( int v = 0; v < 480; v += 1 ) {
				for ( int u = 0; u < 640; u += 1 ) {
					unsigned short d = temp[ v * 640 + u ];
					if ( UVD2XYZ( u, v, d, x, y, z ) ) {
						Eigen::Vector4f dummy = grid_.init_pose_ * Eigen::Vector4f( x, y, z, 1 );
						Coordinate coo;
						Eigen::Vector3f pos;
						if ( grid_.GetCoordinate( Eigen::Vector3f( dummy( 0 ), dummy( 1 ), dummy( 2 ) ), coo ) ) {		// in the box, thus has the right coo
							grid_.GetPosition( coo, pos );
							Eigen::Vector4f reproj_pos = grid_.init_pose_inv_ * Eigen::Vector4f( pos( 0 ), pos( 1 ), pos( 2 ), 1.0 );
							if ( XYZ2UVD( reproj_pos( 0 ), reproj_pos( 1 ), reproj_pos( 2 ), uu, vv, dd ) ) {
								unsigned short ddd = depth[ vv * 640 + uu ];
								if ( ddd == 0 || ddd > dd ) {
									depth[ vv * 640 + uu ] = dd;
									//cout << u << ", " << v << ", " << d << " : " << uu << ", " << vv << ", " << dd << endl;
								}
							}
						}
					}
				}
			}

			slac_mutex_.unlock();
		}

        /** \brief Constructor
          * \param[in] volumeSize physical size of the volume represented by the tdsf volume. In meters.
          * \param[in] shiftingDistance when the camera target point is farther than shiftingDistance from the center of the volume, shiting occurs. In meters.
          * \note The target point is located at (0, 0, 0.6*volumeSize) in camera coordinates.
          * \param[in] rows height of depth image
          * \param[in] cols width of depth image
          */
        KinfuTracker (const Eigen::Vector3f &volumeSize, const float shiftingDistance, int rows = 480, int cols = 640);

        /** \brief Sets Depth camera intrinsics
          * \param[in] fx focal length x 
          * \param[in] fy focal length y
          * \param[in] cx principal point x
          * \param[in] cy principal point y
          */
        void
        setDepthIntrinsics (float fx, float fy, float cx = -1, float cy = -1);

        /** \brief Sets initial camera pose relative to volume coordiante space
          * \param[in] pose Initial camera pose
          */
        void
        setInitialCameraPose (const Eigen::Affine3f& pose);
                        
		/** \brief Sets truncation threshold for depth image for ICP step only! This helps 
		  *  to filter measurements that are outside tsdf volume. Pass zero to disable the truncation.
          * \param[in] max_icp_distance_ Maximal distance, higher values are reset to zero (means no measurement). 
          */
        void
        setDepthTruncationForICP (float max_icp_distance = 0.f);

		void
		setDepthTruncationForIntegrate (float max_integrate_distance = 0.f);

        /** \brief Sets ICP filtering parameters.
          * \param[in] distThreshold distance.
          * \param[in] sineOfAngle sine of angle between normals.
          */
        void
        setIcpCorespFilteringParams (float distThreshold, float sineOfAngle);
        
        /** \brief Sets integration threshold. TSDF volume is integrated iff a camera movement metric exceedes the threshold value. 
          * The metric represents the following: M = (rodrigues(Rotation).norm() + alpha*translation.norm())/2, where alpha = 1.f (hardcoded constant)
          * \param[in] threshold a value to compare with the metric. Suitable values are ~0.001          
          */
        void
        setCameraMovementThreshold(float threshold = 0.001f);

        /** \brief Performs initialization for color integration. Must be called before calling color integration. 
          * \param[in] max_weight max weighe for color integration. -1 means default weight.
          */
        void
        initColorIntegration(int max_weight = -1);        

		void
		initSLAC( int slac_num );

		void initOnlineSLAC( int slac_num );

        /** \brief Returns cols passed to ctor */
        int
        cols ();

        /** \brief Returns rows passed to ctor */
        int
        rows ();

        /** \brief Processes next frame.
          * \param[in] Depth next frame with values in millimeters
          * \return true if can render 3D view.
          */
		bool slac (const DepthMap& depth_raw, const DepthMap& depth, const View * pcolor = NULL);

		bool bdrodometry( const DepthMap & depth, const View * pcolor = NULL );
		
		//zc:
		bool cuOdometry( const DepthMap &depth, const View *pcolor = NULL);

		//全不行... @2017-4-2 14:30:29
		void dbgAhcPeac( const DepthMap &depth_raw, const View *pcolor = NULL);
		void dbgAhcPeac2( const CloudType::Ptr depCloud);
		void dbgAhcPeac3( const CloudType::Ptr depCloud, PlaneFitter *pf);
		void dbgAhcPeac4( const RGBDImage *rgbdObj, PlaneFitter *pf);
		void dbgAhcPeac5( const RGBDImage *rgbdObj, PlaneFitter *pf);

		template<typename Func>
		void dbgAhcPeac6( const RGBDImage *rgbdObj, PlaneFitter *pf, Func f){
			f(rgbdObj, pf);
		}
		
		bool kdtreeodometry( const DepthMap & depth, const View * pcolor = NULL );
		cv::Mat bdrodometry_interpmax( cv::Mat depth );
		cv::Mat bdrodometry_getOcclusionBoundary( cv::Mat depth, float dist_threshold = 0.05f );

        bool operator() (const DepthMap& depth, const View * pcolor = NULL, FramedTransformation * frame_ptr = NULL);

        bool operator() ( 
			const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
			const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
			const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
			const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
			const DepthMap& depth, const View * pcolor = NULL, FramedTransformation * frame_ptr = NULL
			);

		bool rgbdodometry(
			const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
			const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
			const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
			const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
			const DepthMap& depth, const View * pcolor = NULL, FramedTransformation * frame_ptr = NULL
		);

        /** \brief Processes next frame (both depth and color integration). Please call initColorIntegration before invpoking this.
          * \param[in] depth next depth frame with values in millimeters
          * \param[in] colors next RGB frame
          * \return true if can render 3D view.
          */
        //bool operator() (const DepthMap& depth, const View& colors);

        /** \brief Returns camera pose at given time, default the last pose
          * \param[in] time Index of frame for which camera pose is returned.
          * \return camera pose
          */
        Eigen::Affine3f
        getCameraPose (int time = -1) const;

		Eigen::Matrix4f
		getInitTrans() const;

        /** \brief Returns number of poses including initial */
        size_t
        getNumberOfPoses () const;

        /** \brief Returns TSDF volume storage */
        const TsdfVolume& volume() const;

        /** \brief Returns TSDF volume storage */
        TsdfVolume& volume();

        /** \brief Returns color volume storage */
        const ColorVolume& colorVolume() const;

        /** \brief Returns color volume storage */
        ColorVolume& colorVolume();
        
        /** \brief Renders 3D scene to display to human
          * \param[out] view output array with image
          */
        void
        getImage (View& view) const;

		/** \brief Returns point cloud abserved from last camera pose
          * \param[out] cloud output array for points
          */
        void
        getLastFrameCloud (DeviceArray2D<PointType>& cloud) const;

        /** \brief Returns point cloud abserved from last camera pose
          * \param[out] normals output array for normals
          */
        void
        getLastFrameNormals (DeviceArray2D<NormalType>& normals) const;
        
        
        /** \brief Returns pointer to the cyclical buffer structure
          */
        pcl::gpu::tsdf_buffer* 
        getCyclicalBufferStructure () 
        {
          return (cyclical_.getBuffer ());
        }
        
        /** \brief Extract the world and mesh it.
          */
        void
        extractAndMeshWorld ();

		/** \brief Force shifting.
		  */
		void
		forceShift()
		{
			PCL_INFO( "Immediate shifting required.\n" );
			force_shift_ = true;
		}

		void clearForceShift() {
			force_shift_ = false;
		}

		bool shiftNextTime() {
			return force_shift_;
		}

		int getGlobalTime() {
			return global_time_;
		}

		void toggleExtractWorld() {
			extract_world_ = true;
		}

		Eigen::Matrix<double, 6, 6, Eigen::RowMajor> getCoVarianceMatrix() {
			return A_.cast< double >();
		}

		float amplifier_;
		//vector<float> cuOdoAmpVec_;
		float w_f2mkr_;
		float e2c_weight_;

		//zc: 已知立方体三邻边长度, 命令行参数 -cusz //2017-1-2 11:57:19
		vector<float> cuSideLenVec_; //meters
		bool isUseCube_; //若 cuSideLenVec_.size() >0, 则 true
		Eigen::Affine3f camPoseUseCu_; //用立方体定位方式得到的相机姿态
		bool isLastFoundCrnr_; //上一帧有没有看到顶角

		vector<vector<double>> cubeCandiPoses_; //内vec必有size=12=(t3+R9), 是cube在相机坐标系的姿态; 外vec表示多个候选顶角的姿态描述
		size_t crnrIdx_; //因自动定位立方体不够好, 所以 kinfu_app 中鼠标选点, 初始化时手动指定【用哪个顶角】

		float e2c_dist_;
		bool with_nmap_; //@耿老师 要求加 nmap 惩罚项优化 R, 代码放在 -cusz 控制中, 命令行 "-nmap"

		//用于控制对比多惩罚项时, 看哪个项起实际作用
		bool term_123_;
		bool term_12_;
		bool term_13_;
		bool term_23_;
		bool term_1_;
		bool term_2_;
		bool term_3_;

		//tsdf 融合策略, 详见: 《pcl, kinfu, tsdfVolume 笔记》; 《三维重建中体素融合方案改进结果（寒假201702）.docx》；《三维重建结果与vivid 3D 扫描仪groundtruth真值对比测试.docx》(小节3)
		//用 float 而不用 int 是因为要区分很多子版本, 
		float tsdf_version_;
		bool isTsdfVer(float verNum){ return abs(tsdf_version_ - verNum) < 1e-6; }

		//bool dbgKf_;
		int dbgKf_;

		//有时候,某帧迭代配准会严重漂移, 需要step-in跟进调试观察; 如m7/05-f126失败
		int regCuStepId_;

		vector<int> logIdVec_;

		//@brief 关于: 对后端 fusion 策略改进
		float tsdfErodeRad_;

		//想要调试观察的体素 vxl
		vector<int> vxlDbg_;

		//用于 tsdf-v9, v11, 大入射角 mask的角度阈值, 之前 pcc 设定为 75°
		float incidAngleThresh_;

		//立方体定位算法相关     //2017-1-1 21:03:02
		bool isFirstFoundCube_;// = true; //若第一次定位到立方体: 1, 设定 cu2gPose_
		bool isFoundCu4pts_; //=false; //if-false, 持续平面拟合,立方体定位; if-true, 初始化全局 cu, 且停止平面拟合 @2017-4-22 21:29:14
		bool isCuInitialized_;// = false; //若第一次定位到立方体: 1, 设定 cu2gPose_

		Eigen::Affine3f cu2gPose_; //cube pose, cu->global, 全局唯一
		Cube cube_g_;
		//pcl::KdTreeFLANN<PointXYZ> cuEdgeTree_;
		pcl::PointCloud<PointXYZ>::Ptr cuContCloud_; //global, 所有棱边, 无论是否可见
		pcl::PointCloud<pcl::Normal>::Ptr cuContCloudNormal_; //global, 与 cuEdgeCloud_ 逐点对应, 但是在不可见位置, cont 法向其实错了, (因为映射到像素 uv 查找的), 只是不在乎

		//耿老师要求生成合成数据, 无噪声, 立方体坐标系为世界坐标系 @2017-5-8 20:00:52
		bool genSynData_;

		//以某平面(ABCD四参数, 量纲米)为分割 filter, 法向量同侧的保留 @2017-8-13 19:10:58
		//耿老师要求: 立方体上平面作为分割界面, 仅融合上面的
		bool isPlFilt_; //命令行参数
		float plFiltShiftMM_; //参考面上下平移控制, 其实没用在 kinfu.cpp, 备用
		Eigen::Vector4f planeFiltParam_; //世界坐标系下的

      private:

		DeviceArray2D<unsigned short> bdr_temp_depth_;
		DeviceArray2D<float4> kdtree_cloud_;
		DeviceArray2D<float4> kdtree_normal_;
        std::vector<DeviceArray2D<float4>> kdtree_curr_;

		//Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A_;
        //Eigen::Matrix<double, 6, 1> b_;
		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_;
        Eigen::Matrix<float, 6, 1> b_;

		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> AA_;
		Eigen::Matrix<float, 6, 1> bb_;

		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> AAA_;
		Eigen::Matrix<float, 6, 1> bbb_;

		//zc: for cu-odo
		//frame2model
		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_f2mod_;
		Eigen::Matrix<float, 6, 1> b_f2mod_;

		//frame2marker (3d cuboid as fiducial marker)
		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_f2mkr_;
		Eigen::Matrix<float, 6, 1> b_f2mkr_;

		//edge2contour (cuboid's occluding contour)
		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_e2c_;
		Eigen::Matrix<float, 6, 1> b_e2c_;


		bool extract_world_;

		/** \brief Immediate shifting */
		bool force_shift_;
        
        /** \brief Cyclical buffer object */
        pcl::gpu::CyclicalBuffer cyclical_;
		pcl::gpu::CyclicalBuffer cyclical2_;
        
        
        /** \brief Number of pyramid levels */
        enum { LEVELS = 3 };

        /** \brief ICP Correspondences  map type */
        typedef DeviceArray2D<int> CorespMap;

        /** \brief Vertex or Normal Map type */
        typedef DeviceArray2D<float> MapArr;
        
        typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
        typedef Eigen::Vector3f Vector3f;

        /** \brief Height of input depth image. */
        int rows_;
        /** \brief Width of input depth image. */
        int cols_;
        /** \brief Frame counter */
        int global_time_;

        /** \brief Truncation threshold for depth image for ICP step */
        float max_icp_distance_;

		float max_integrate_distance_;

        /** \brief Intrinsic parameters of depth camera. */
        float fx_, fy_, cx_, cy_;

        /** \brief Tsdf volume container. */
        TsdfVolume::Ptr tsdf_volume_;
		TsdfVolume::Ptr tsdf_volume2_;
        ColorVolume::Ptr color_volume_;
                
        /** \brief Initial camera rotation in volume coo space. */
        Matrix3frm init_Rcam_;

        /** \brief Initial camera position in volume coo space. */
        Vector3f   init_tcam_;

		Eigen::Matrix4f init_rev_;
		Eigen::Matrix4f init_trans_;

        /** \brief array with IPC iteration numbers for each pyramid level */
        int icp_iterations_[LEVELS];
        /** \brief distance threshold in correspondences filtering */
        float  distThres_;
        /** \brief angle threshold in correspondences filtering. Represents max sine of angle between normals. */
        float angleThres_;
        
        /** \brief Depth pyramid. */
        std::vector<DepthMap> depths_curr_;
        /** \brief Vertex maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_curr_;
        /** \brief Normal maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_curr_;

        /** \brief Vertex maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_prev_;
        /** \brief Normal maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_prev_;
                
        /** \brief Vertex maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> vmaps_curr_;
        /** \brief Normal maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> nmaps_curr_;

        //zc: cu 可见区域的*全局* vmap、nmap
        std::vector<DepthMap> depths_cu_;
        std::vector<MapArr> vmaps_cu_g_prev_;
        std::vector<MapArr> nmaps_cu_g_prev_;
        //以下都仅是为了避免局部变量导致控制台调试输出 "[CUDA] Allocating memory..."
        MapArr vmap_model_, nmap_model_;
        MapArr vmap_g_model_, nmap_g_model_;
        DepthMap dmapModel_, dmapModel_inp_;
        DeviceArray2D<short> diffDmap_; //zc: 用于处理 motionBlur   @2017-12-3 23:36:56
        pcl::device::MaskMap largeIncidMask_model_; //大入射角mask, vmap_g_model_ 的
        pcl::device::MaskMap largeIncidMask_curr_; //大入射角mask, dmap-curr 的
        pcl::device::MaskMap largeIncidMask_total_;
        MapArr nmap_filt_, nmap_filt_g_;
        DeviceArray2D<float> edgeDistMap_device_;
        MapArr wmap_; //weight-map, 按: 1, 入射角cos; 2, D(u); 3, 到边缘距离 来加权

        //bdr方案求解nmap 用到
        DepthMap synCuDmap_device_;
        MapArr gx_device_, gy_device_;

        DeviceArray2D<float> gbuf_f2mkr_;
        DeviceArray<float> sumbuf_f2mkr_;

        /** \brief Array of buffers with ICP correspondences for each pyramid level. */
        std::vector<CorespMap> coresps_;
        
        /** \brief Buffer for storing scaled depth image */
        DeviceArray2D<float> depthRawScaled_;
        
        /** \brief Temporary buffer for ICP */
        DeviceArray2D<float> gbuf_;
        /** \brief Buffer to store MLS matrix. */
        DeviceArray<float> sumbuf_;

		/** \brief Temporary buffer for SLAC */
        DeviceArray<float> gbuf_slac_triangle_;
		/** \brief Temporary buffer for SLAC */
        DeviceArray<float> gbuf_slac_block_;  // last row is gbuf_slac_b_;

		Eigen::Matrix<float, 6591, 6591, Eigen::ColMajor> slac_A_;
		Eigen::Matrix<float, 6591, 7, Eigen::ColMajor> slac_block_;
		Eigen::VectorXf slac_init_ctr_;
		Eigen::VectorXf slac_this_ctr_;
		Eigen::MatrixXf slac_base_mat_;
		Eigen::MatrixXf slac_full_mat_;
		Eigen::VectorXf slac_full_b_;

		DeviceArray<float> ctr_buf_;

		inline int getSLACIndex( int i, int j, int k ) {
			return ( i * ( slac_resolution_ + 1 ) * ( slac_resolution_ + 1 ) + j * ( slac_resolution_ + 1 ) + k ) * 3;
		}

		void initSLACMatrices();
		void addRegularizationTerm();
		Eigen::Matrix3f GetRotation( const int idx, const std::vector< int > & idxx, const Eigen::VectorXf & ictr, const Eigen::VectorXf & ctr );
		Eigen::Matrix3d GetRotationd( const int idx, const std::vector< int > & idxx, const Eigen::VectorXd & ictr, const Eigen::VectorXd & ctr );

		bool use_slac_;
		int slac_resolution_;
		int slac_num_;
		int slac_matrix_size_;
		int slac_lean_matrix_size_;
		int slac_lean_matrix_size_gpu_2d_;
		int slac_full_matrix_size_;
		std::vector< Eigen::Matrix4f > slac_trans_mats_;

		ControlGrid grid_;
		//std::vector< PointCloud< PointNormal >::Ptr > fragments_;
		std::vector< PointCloud< PointXYZ >::Ptr > fragments_dummy_;
		std::vector< SLACPointCloud::Ptr > fragments_;
		std::vector< Correspondence::Ptr > corres_;
		std::vector< Eigen::Matrix4f > base_pose_;
		Eigen::Matrix4f base_pose_cur_;
		Eigen::Matrix4f base_pose_cur_inv_;
		double dist_thresh_;

		DeviceArray<PointXYZ> cloud_buffer_device_;
		DeviceArray<Normal> normals_device_;
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

		void OptimizeSLAC();
		boost::mutex slac_mutex_;

		/** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Matrix3frm> rmats_;
        
        /** \brief Array of camera translations for each moment of time. */
        std::vector<Vector3f> tvecs_;

        /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
        float integration_metric_threshold_;
        
        /** \brief Allocates all GPU internal buffers.
          * \param[in] rows_arg
          * \param[in] cols_arg          
          */
        void
        allocateBufffers (int rows_arg, int cols_arg);

        /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
          */
        void
        reset ();     
                
        /** \brief When set to true, KinFu will extract the whole world and mesh it. */
        bool perform_last_scan_;
        
        /** \brief When set to true, KinFu notifies that it is finished scanning and can be stopped. */
        bool finished_;

        /** \brief // when the camera target point is farther than DISTANCE_THRESHOLD from the current cube's center, shifting occurs. In meters . */
        float shifting_distance_;

        /** \brief Size of the TSDF volume in meters. */
        float volume_size_;
        
      public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
  }
};

#endif /* PCL_KINFU_KINFUTRACKER_HPP_ */
