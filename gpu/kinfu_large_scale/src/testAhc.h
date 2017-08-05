#ifndef _TEST_AHC_H_
#define _TEST_AHC_H_

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/gpu/containers/device_array.h>

#include <AHCPlaneFitter.hpp> //peac代码: http://www.merl.com/demos/point-plane-slam
#include "zcAhcUtility.h"

//#include "internal.h"
//#include "kinfu.h"

//////////////////////////////
//typedef pcl::PointXYZ PtType;
//typedef pcl::PointCloud<PtType> CloudType;
//PCL_EXPORTS CloudType::Ptr cvMat2PointCloud(const cv::Mat &dmat, const pcl::device::Intr &intr);
//
////平面拟合, 提取分割
////以下拷贝自 plane_fitter.cpp
//// pcl::PointCloud interface for our ahc::PlaneFitter
//template<class PointT>
//struct OrganizedImage3D {
//	const pcl::PointCloud<PointT>& cloud;
//	//NOTE: pcl::PointCloud from OpenNI uses meter as unit,
//	//while ahc::PlaneFitter assumes mm as unit!!!
//	const double unitScaleFactor;
//
//	OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1000) {}
//	int width() const { return cloud.width; }
//	int height() const { return cloud.height; }
//	bool get(const int row, const int col, double& x, double& y, double& z) const {
//		const PointT& pt=cloud.at(col,row);
//		x=pt.x; y=pt.y; z=pt.z;
//		return pcl_isnan(z)==0; //return false if current depth is NaN
//	}
//};
////typedef OrganizedImage3D<pcl::PointXYZRGBA> RGBDImage;
//typedef OrganizedImage3D<PtType> RGBDImage;
//typedef ahc::PlaneFitter<RGBDImage> PlaneFitter;

PCL_EXPORTS void dbgAhcPeac_testAhc( const RGBDImage *rgbdObj, PlaneFitter *pf);

#endif //_TEST_AHC_H_
