#ifndef ZC_CUDA_UTILS_HPP_
#define ZC_CUDA_UTILS_HPP_

#include <cuda_runtime.h>

struct /*__device_builtin__*/ float6 //dev-builtin raises error : incomplete type is not allowed
{ 
    float x, y, z, a, b, c;
};

typedef __device_builtin__ struct float6 float6;

__device__ __forceinline__ float
dot(const float6& v1, const float6& v2)
{
    return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z + v1.a*v2.a + v1.b*v2.b + v1.c*v2.c;
}

__device__ __forceinline__ float6&
operator*=(float6& vec, const float& v)
{
    vec.x *= v;  vec.y *= v;  vec.z *= v; vec.a *= v; vec.b *= v; vec.c *= v;
    return vec;
}

#endif /*ZC_CUDA_UTILS_HPP_*/
