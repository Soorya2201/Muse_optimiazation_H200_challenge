"""Compare native exp/FP8 and direct log2-to-FP8 packing, same NVCC build."""
import datetime
import argparse
import gzip
import json
from pathlib import Path
import urllib.error
import urllib.request

from attention_direct_fp8 import cuda_helper, cuda_branchless_helper

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--branchless', action='store_true')
args = parser.parse_args()
helper = cuda_branchless_helper() if args.branchless else cuda_helper()
source = '#include <cuda_runtime.h>\n#include <stdint.h>\n' + helper + r'''
__device__ __forceinline__ float hw(float x) {
 float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ unsigned native4(float a, float b, float c, float d) {
 unsigned short lo, hi;
 a=hw(a); b=hw(b); c=hw(c); d=hw(d);
 asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;":"=h"(lo):"f"(a),"f"(b));
 asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;":"=h"(hi):"f"(c),"f"(d));
 return unsigned(lo)|(unsigned(hi)<<16);
}
template<bool DIRECT, int WORDS>
__device__ __forceinline__ void body(const float* in, unsigned* out) {
 int t=threadIdx.x+blockIdx.x*blockDim.x;
 unsigned lane_lut=aq_direct_lut[threadIdx.x&31];
 float s[WORDS*4];
 #pragma unroll
 for(int j=0;j<WORDS*4;++j)s[j]=in[t*WORDS*4+j];
 unsigned p[WORDS];
 #pragma unroll
 for(int j=0;j<WORDS;++j) {
   if constexpr(DIRECT)p[j]=aq_direct_fp8x4(s[j*4],s[j*4+1],s[j*4+2],s[j*4+3],lane_lut);
   else p[j]=native4(s[j*4],s[j*4+1],s[j*4+2],s[j*4+3]);
 }
 #pragma unroll
 for(int j=0;j<WORDS;++j)out[t*WORDS+j]=p[j];
}
extern "C" __global__ void native_pack4(const float* x,unsigned* y){body<false,1>(x,y);}
extern "C" __global__ void direct_pack4(const float* x,unsigned* y){body<true,1>(x,y);}
extern "C" __global__ void native_pack32(const float* x,unsigned* y){body<false,8>(x,y);}
extern "C" __global__ void direct_pack32(const float* x,unsigned* y){body<true,8>(x,y);}
'''
request=dict(source=source,options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
label='DIRECT_FP8_BRANCHLESS' if args.branchless else 'DIRECT_FP8'
target=ROOT/f'muse_attention_decode/{label}_COMPILER_2026-10-31.json.gz'
assert not target.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',
    data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=180) as r:response=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:
    response=dict(code=None,transport_error=repr(e))
with gzip.open(target,'wt') as f:json.dump(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Standalone pack4/pack32, not full attention resource or runtime evidence.',
    request=request,response=response),f)
print('Compiler status:',response.get('code'),target.relative_to(ROOT))
for row in response.get('stderr',[]):print(row.get('text','')[:300])
for name,dev in response.get('devices',{}).items():
    print('DEVICE',name,'rows',len(dev.get('asm',[])))
