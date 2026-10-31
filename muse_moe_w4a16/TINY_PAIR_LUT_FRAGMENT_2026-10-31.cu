
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
static __device__ __forceinline__ uint32_t mul2(uint32_t a,uint32_t b) {
  __nv_bfloat162 x=*reinterpret_cast<__nv_bfloat162*>(&a);
  __nv_bfloat162 y=*reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 z=__hmul2(x,y);
  return *reinterpret_cast<uint32_t*>(&z);
}
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}



extern "C" __device__ __noinline__ uint4 arithmetic_core(uint32_t q,uint32_t sg,uint32_t sg8) {
  uint32_t a,b,c,d;dq4(q,a,b,c,d);
  return make_uint4(mul2(a,sg),mul2(b,sg8),mul2(c,sg),mul2(d,sg8));
}
extern "C" __device__ __noinline__ uint4 table_core(uint32_t q,uint32_t sg,uint32_t sg8,
                                                   uint32_t base,uint32_t lane) {
  uint32_t r[4];
  const uint32_t origin=base+((lane&15u)<<2);
#pragma unroll
  for (int j=0;j<4;++j) {
    const uint32_t address=origin+(((q>>(8*j))&255u)<<6);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(r[j]):"r"(address):"memory");
  }
  return make_uint4(mul2(r[0],sg),mul2(r[1],sg8),mul2(r[2],sg),mul2(r[3],sg8));
}
extern "C" __global__ void parent_fragment(const uint32_t* words,const uint2* scales,uint4* out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){uint2 s=scales[i];out[i]=arithmetic_core(words[i],s.x,s.y);}
}
extern "C" __global__ void table_fragment(const uint32_t* natural_words,const uint2* scales,uint4* out,int n) {
  __shared__ uint32_t lut[4096];
  for(unsigned i=threadIdx.x;i<4096;i+=blockDim.x) {
    unsigned k=i>>4;
    unsigned a=((k&8u)<<12)|((k&7u)<<6);
    unsigned b=(((k>>4)&8u)<<12)|(((k>>4)&7u)<<6);
    lut[i]=a|(b<<16);
  }
  __syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    uint2 s=scales[i];
    out[i]=table_core(natural_words[i],s.x,s.y,(uint32_t)__cvta_generic_to_shared(lut),threadIdx.x&31);
  }
}
