
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ float hw(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float p(float f) {
 float y=fmaf(f,0.0790209f,0.2249424f); y=fmaf(f,y,0.6960656f); return fmaf(f,y,0.9999893f);
}
extern "C" __global__ void old_floor(const float* in,float* out) {
 int i=threadIdx.x+blockIdx.x*blockDim.x; float x=in[i];
 if(x>120.f){out[i]=hw(x);return;}
 x=fmaxf(x,-126.f); float j=floorf(x); float f=x-j;
 out[i]=__int_as_float(__float_as_int(p(f))+((int)j<<23));
}
extern "C" __global__ void bias_floor(const float* in,float* out) {
 int i=threadIdx.x+blockIdx.x*blockDim.x; float x=in[i];
 if(x>120.f){out[i]=hw(x);return;}
 x=fmaxf(x,-126.f); float t=__fadd_rd(x,12582912.f);
 float j=__fadd_rn(t,-12582912.f); float f=__fsub_rn(x,j);
 out[i]=__uint_as_float(__float_as_uint(p(f))+(__float_as_uint(t)<<23));
}
