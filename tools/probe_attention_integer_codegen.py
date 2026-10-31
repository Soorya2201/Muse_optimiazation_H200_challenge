"""Archive matched native/integer-quadratic E4M3 pack32 CUDA codegen."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from attention_integer_exp import cuda_helper

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'muse_attention_decode/INTEGER_EXP_COMPILER_2026-10-31.json.gz'
assert not target.exists()
source='#include <cuda_runtime.h>\n#include <stdint.h>\n'+cuda_helper()+r'''
__device__ __forceinline__ float hw(float x) {
  float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;
}
template<int FRACTION,int SLOT>
__device__ __forceinline__ float choose_exp(float x) {
  if constexpr(SLOT<FRACTION)return x>120.f?hw(x):aq_integer_exp(x);
  return hw(x);
}
template<int FRACTION>
__device__ __forceinline__ void body(const float* input,unsigned* output) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  float x[32];
#pragma unroll
  for(int j=0;j<32;++j)x[j]=input[t*32+j];
  unsigned p[8];
#pragma unroll
  for(int j=0;j<8;++j) {
    float a=choose_exp<FRACTION,0>(x[j*4]);
    float b=choose_exp<FRACTION,1>(x[j*4+1]);
    float c=choose_exp<FRACTION,2>(x[j*4+2]);
    float d=choose_exp<FRACTION,3>(x[j*4+3]);
    unsigned short lo,hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(lo):"f"(a),"f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(hi):"f"(c),"f"(d));
    p[j]=unsigned(lo)|(unsigned(hi)<<16);
  }
#pragma unroll
  for(int j=0;j<8;++j)output[t*8+j]=p[j];
}
extern "C" __global__ void native_pack32(const float* x,unsigned* y){body<0>(x,y);}
extern "C" __global__ void integer25_pack32(const float* x,unsigned* y){body<1>(x,y);}
extern "C" __global__ void integer50_pack32(const float* x,unsigned* y){body<2>(x,y);}
extern "C" __global__ void integer100_pack32(const float* x,unsigned* y){body<4>(x,y);}
'''
request=dict(source=source,options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),
    headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
except (urllib.error.URLError,TimeoutError) as error:
    result=dict(code=None,transport_error=repr(error))
with gzip.open(target,'wt') as f:json.dump(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Standalone pack32 entries, no full attention resources, GPU correctness or timing.',
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),request=request,response=result),f)
print('COMPILER',result.get('code'))
for row in result.get('stderr',[]):print(row['text'])
for name,device in result.get('devices',{}).items():print(name,len(device['asm']))
