"""Archive NVCC13/sm90a whole-fused-body reduced-width scale-sharing builds."""
import argparse
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from prefill_scale_native import scale_shared_native

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('arm',choices=['parent','shared']);p.add_argument('--width',type=int,choices=[32,64,128,160,192,256],default=256);args=p.parse_args()
parent=(ROOT/'muse_prefill_layer/submission.py').read_text();t=ast.parse(parent)
native=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_CUDA_SRC' for x in n.targets))
if args.arm=='shared':native=scale_shared_native(native)
source=native.split('// ---- routing plan:')[0]
# Only this width is instantiated; full CUDA/GLUE compilation is still required.
a=source.index('      switch (ti.z) {',source.index('__global__ void __launch_bounds__(MOE_FUSED_THREADS'))
b=source.index('      if (fa.dbg && threadIdx.x == MOE_CB)',a)
source=source[:a]+f'      consumer_item<{args.width}, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);\n'+source[b:]
source+='\n} // namespace okc_moe\nvoid instantiate_prefill(okc_moe::FusedArgs fa, cudaStream_t stream) {\n okc_moe::moe_fused_persistent<0><<<132, 384, 232448, stream>>>(fa);\n}\n'
source=re.sub(r'^DEVI\b','static DEVI',source,flags=re.M)
source=re.sub(r'^__noinline__ __device__','static __noinline__ __device__',source,flags=re.M)
flags='-O3 -gencode=arch=compute_90a,code=sm_90a -std=c++20 --use_fast_math -Xptxas -v'
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
path=ROOT/f'muse_prefill_layer/SCALE_CODEGEN_{args.arm}_N{args.width}_2026-10-31.json.gz';assert not path.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
 with urllib.request.urlopen(req,timeout=180) as r:result=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:result=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Fused-kernel body including real producer/publisher/consumer roles, reduced to one token width/FINE0; not full dispatch or linked judge extension. Never executed.',arm=args.arm,width=args.width,parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),native_sha256=hashlib.sha256(native.encode()).hexdigest(),request=request,response=result)
with gzip.open(path,'wt') as f:json.dump(out,f)
print(path.relative_to(ROOT),'code',result.get('code'))
for row in result.get('stderr',[]):
 s=row.get('text','')
 if any(x in s for x in ['ptxas','spill','stack','error','Error','C75']):print(s[:300])
for name,d in result.get('devices',{}).items():print(name,len(d.get('asm',[])),'clipped',any('[truncated' in r.get('text','') for r in d.get('asm',[])))
