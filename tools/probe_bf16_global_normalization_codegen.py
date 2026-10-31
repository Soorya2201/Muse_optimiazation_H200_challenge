"""Compile old/fixed-window decoder and FP32 restore in one sm90a unit."""
import ast
import argparse
import datetime
import gzip
import json
from pathlib import Path
import urllib.error
import urllib.request

from bf16_global_normalization import DECODE, DECODE_LUT

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--fused-lut',action='store_true');args=parser.parse_args()
s=(ROOT/'muse_bf16_proj/submission.py').read_text()
t=ast.parse(s)
native=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='CUDA_SRC')
start=native.index('__device__ __forceinline__ uint32_t prmt(')
end=native.index('\ntemplate <typename T>',start)
native=native[start:end].replace('decode8(', 'decode_old(',1)
source='#include <cuda_runtime.h>\n#include <stdint.h>\n'+native+(DECODE_LUT if args.fused_lut else DECODE).replace('decode8(', 'decode_norm(',1)
for name in ['old','norm']:
    source+=f'''
extern "C" __global__ void {name}_decode(const uint4* in,uint4* out) {{
 int i=threadIdx.x+blockIdx.x*blockDim.x;uint4 v=in[i];uint32_t r[4];
 decode_{name}(v.x,v.y,v.z,v.w,r);out[i]=make_uint4(r[0],r[1],r[2],r[3]);
}}
'''
source+=r'''
extern "C" __global__ void restore_partial(const float4* in,float4* out,unsigned b4) {
 int i=threadIdx.x+blockIdx.x*blockDim.x;float4 v=in[i];
 float factor=__uint_as_float(b4&0x7f800000u);
 out[i]=make_float4(v.x*factor,v.y*factor,v.z*factor,v.w*factor);
}
'''
if args.fused_lut:
    source=source.replace('__uint_as_float(b4&0x7f800000u)', '__uint_as_float((2u*gn_original_base(b4)+15u)<<23)')
request=dict(source=source,options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
label='GLOBAL_NORMALIZATION_LUT' if args.fused_lut else 'GLOBAL_NORMALIZATION'
target=ROOT/f'muse_bf16_proj/{label}_COMPILER_2026-10-31.json.gz'
assert not target.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),
    headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as r:result=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:
    result=dict(code=None,transport_error=repr(e))
with gzip.open(target,'wt') as f:json.dump(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Standalone register decoder/restore only; not full-kernel resource/runtime proof.',request=request,response=result),f)
print('Compiler status',result.get('code'))
for row in result.get('stderr',[]):print(row.get('text','')[:300])
for name,dev in result.get('devices',{}).items():
    if name.startswith('SASS'):
        for row in dev.get('asm',[]):print(row.get('text',''))
