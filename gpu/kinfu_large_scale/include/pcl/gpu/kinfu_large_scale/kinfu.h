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
        bool operator() (const DepthMap& depth, const View * pcolor = NULL, FramedTransformation * frame_ptr = NULL);

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

      private:

		//Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A_;
        //Eigen::Matrix<double, 6, 1> b_;
		Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_;
        Eigen::Matrix<float, 6, 1> b_;


		bool extract_world_;

		/** \brief Immediate shifting */
		bool force_shift_;
        
        /** \brief Cyclical buffer object */
        pcl::gpu::CyclicalBuffer cyclical_;
        
        
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

		Eigen::Matrix<float, 2187, 2187, Eigen::ColMajor> slac_A_;
		Eigen::Matrix<float, 2187, 7, Eigen::ColMajor> slac_block_;
		Eigen::VectorXf slac_init_ctr_;
		Eigen::VectorXf slac_this_ctr_;
		Eigen::MatrixXf slac_base_mat_;
		Eigen::MatrixXf slac_full_mat_;
		Eigen::VectorXf slac_full_b_;

		DeviceArray<float> ctr_buf_;

		void initSLACMatrices();
		void addRegularizationTerm();
		Eigen::Matrix3f GetRotation( const int idx, const std::vector< int > & idxx, const Eigen::VectorXf & ictr, const Eigen::VectorXf & ctr );

		bool use_slac_;
		int slac_resolution_;
		int slac_num_;
		int slac_matrix_size_;
		int slac_lean_matrix_size_;
		int slac_lean_matrix_size_gpu_2d_;
		int slac_full_matrix_size_;
		std::vector< Eigen::Matrix4f > slac_trans_mats_;

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
