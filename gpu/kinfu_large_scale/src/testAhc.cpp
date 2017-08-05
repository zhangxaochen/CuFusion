#include <pcl/gpu/kinfu_large_scale/kinfu.h>

#include "testAhc.h"

void dbgAhcPeac_testAhc( const RGBDImage *rgbdObj, PlaneFitter *pf){
	cv::Mat dbgSegMat(rgbdObj->height(), rgbdObj->width(), CV_8UC3); //分割结果可视化
	vector<vector<int>> idxss;

	pf->minSupport = 1000;
	pf->run(rgbdObj, &idxss, &dbgSegMat);
	annotateLabelMat(pf->membershipImg, &dbgSegMat);
	const char *winNameAhc = "dbgAhc@dbgAhcPeac_testAhc";
	imshow(winNameAhc, dbgSegMat);
}//dbgAhcPeac_testAhc

