#ifndef SOPHUS_UTILS_H_
#define SOPHUS_UTILS_H_

#include <Eigen/Core>

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
typedef Eigen::Matrix<float,6,1> Vector6f;

using namespace Eigen;

void get_rt_from_twist(const Vector6f &twist, Matrix3f &Rmat, Vector3f &tvec);
void get_rt_from_twist(const Vector6f &twist, Matrix3frm &Rmat, Vector3f &tvec);
Vector6f get_twist_from_rt(const Matrix3f &Rmat, const Vector3f &tvec);
Vector6f get_twist_from_rt(const Matrix3frm &Rmat, const Vector3f &tvec);


#endif //SOPHUS_UTILS_H_