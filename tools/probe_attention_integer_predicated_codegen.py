"""Matched pack32 codegen for predicated25/37.5/50/100 percent offload."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from attention_integer_exp_predicated import cuda_helper

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initialized',action='store_true')
parser.add_argument('--final-operands',action='store_true')
args=parser.parse_args()
assert not args.final_operands or args.initialized
if args.initialized:
    from attention_integer_exp_initialized import cuda_helper as initialized_helper
    def cuda_helper():return initialized_helper(final_operands=args.final_operands)

ROOT=Path(__file__).resolve().parents[1]
label='INITIALIZED' if args.initialized else 'PREDICATED'
if args.final_operands:label+='FINAL'
target=ROOT/f'muse_attention_decode/INTEGER_EXP_{label}_COMPILER_2026-10-31.json.gz'
assert not target.exists()
source='#include <cuda_runtime.h>\n#include <stdint.h>\n'+cuda_helper()+r'''
__device__ __forceinline__ float hw(float x) {
  float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;
}
template<int FRACTION,int SLOT>
__device__ __forceinline__ float choose_exp(float x) {
  if constexpr(SLOT<FRACTION)return aq_integer_predicated(x);
  return hw(x);
}
__device__ __forceinline__ unsigned pack4(float a,float b,float c,float d) {
  unsigned short lo,hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(lo):"f"(a),"f"(b));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(hi):"f"(c),"f"(d));
  return unsigned(lo)|(unsigned(hi)<<16);
}
template<int FRACTION>
__device__ __forceinline__ void body(const float* input,unsigned* output) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  float x[32];
#pragma unroll
  for(int j=0;j<32;++j)x[j]=input[t*32+j];
  unsigned p[8];
#pragma unroll
  for(int j=0;j<4;++j) {
    p[2*j]=pack4(choose_exp<FRACTION,0>(x[8*j]),choose_exp<FRACTION,1>(x[8*j+1]),
                 choose_exp<FRACTION,2>(x[8*j+2]),choose_exp<FRACTION,3>(x[8*j+3]));
    p[2*j+1]=pack4(choose_exp<FRACTION,4>(x[8*j+4]),choose_exp<FRACTION,5>(x[8*j+5]),
                   choose_exp<FRACTION,6>(x[8*j+6]),choose_exp<FRACTION,7>(x[8*j+7]));
  }
#pragma unroll
  for(int j=0;j<8;++j)output[t*8+j]=p[j];
}
extern "C" __global__ void native_pack32(const float* x,unsigned* y){body<0>(x,y);}
extern "C" __global__ void pred25_pack32(const float* x,unsigned* y){body<2>(x,y);}
extern "C" __global__ void pred375_pack32(const float* x,unsigned* y){body<3>(x,y);}
extern "C" __global__ void pred50_pack32(const float* x,unsigned* y){body<4>(x,y);}
extern "C" __global__ void pred100_pack32(const float* x,unsigned* y){body<8>(x,y);}
'''
request=dict(source=source,options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130u1/compile',data=json.dumps(request).encode(),
    headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
except (urllib.error.URLError,TimeoutError) as error:
    result=dict(code=None,transport_error=repr(error))
with gzip.open(target,'wt') as f:json.dump(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    compiler='nvcc130u1',scope='Standalone pack32, not full attention resources or GPU validation/timing.',
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),request=request,response=result),f)
print('COMPILER',result.get('code'),result.get('transport_error'))
for row in result.get('stderr',[]):print(row['text'])
for name,device in result.get('devices',{}).items():print(name,len(device['asm']))
