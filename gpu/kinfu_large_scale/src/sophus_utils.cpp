#include <iostream>
#include <sophus/se3.hpp> //with older eigen (from pcl 160), error C2144: syntax error : 'int' should be preceded by ';'
#include <sophus/sophus.hpp>
#include "sophus_utils.h"

using namespace std;

void foo(){
	Matrix3frm Rcurr = Matrix3frm::Identity();
	Vector3f   tcurr;
	tcurr << 1,0,0;

	Sophus::SE3f se_rt(Rcurr, tcurr);
	cout<<se_rt.matrix()<<endl;
}

//@param[in] twist, Lie algebra (u, w)
//@param[out] Rmat, rotation-mat
//@param[out] tvec, translation-vec
void get_rt_from_twist(const Vector6f &twist, Matrix3f &Rmat, Vector3f &tvec){
//void get_rt_from_twist(const Vector6f &twist, Matrix3frm &Rmat, Vector3f &tvec){
	Sophus::SE3f se3f_exp = Sophus::SE3f::exp(twist);
	Rmat = se3f_exp.rotationMatrix();
	tvec = se3f_exp.translation();
}//get_rt_from_twist

void get_rt_from_twist(const Vector6f &twist, Matrix3frm &Rmat, Vector3f &tvec){
	Sophus::SE3f se3f_exp = Sophus::SE3f::exp(twist);
	Rmat = se3f_exp.rotationMatrix();
	tvec = se3f_exp.translation();
}//get_rt_from_twist

//@param[in] Rmat, rotation-mat
//@param[in] tvec, translation-vec
//@return the 6DOF twist
Vector6f get_twist_from_rt(const Matrix3f &Rmat, const Vector3f &tvec){
	Sophus::SE3f se3f_rt(Rmat, tvec);
	return se3f_rt.log();
}//get_twist_from_rt

Vector6f get_twist_from_rt(const Matrix3frm &Rmat, const Vector3f &tvec){
	Sophus::SE3f se3f_rt(Rmat, tvec);
	return se3f_rt.log();
}//get_twist_from_rt
