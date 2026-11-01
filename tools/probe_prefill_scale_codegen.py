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
from prefill_ss_native import shared_weight_native
from prefill_ss_k64_native import scale_reused_shared_native
from prefill_wide_batch_native import wide_batch_native

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('arm',choices=['parent','shared','ss','ss_all','ss_k64','wide_batch']);p.add_argument('--width',type=int,choices=[32,64,128,160,192,256],default=256)
p.add_argument('--full-switch',action='store_true');p.add_argument('--fine',type=int,choices=[0,1],default=0)
p.add_argument('--ss-consumer-regs',type=int,choices=[184,216],default=184)
p.add_argument('--label',default='');p.add_argument('--date',default='2026-10-31');args=p.parse_args();assert re.fullmatch('[a-z0-9_]*',args.label)
assert re.fullmatch(r'\d{4}-\d{2}-\d{2}',args.date)
parent=(ROOT/'muse_prefill_layer/submission.py').read_text();t=ast.parse(parent)
native=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_CUDA_SRC' for x in n.targets))
if args.arm=='wide_batch':native=wide_batch_native(native)
if args.arm=='shared':native=scale_shared_native(native)
if args.arm in ('ss','ss_all'):native=shared_weight_native(native,args.ss_consumer_regs,args.arm=='ss_all')
if args.arm=='ss_k64':native=scale_reused_shared_native(native)
source=native.split('// ---- routing plan:')[0]
# Only this width is instantiated; full CUDA/GLUE compilation is still required.
if not args.full_switch:
 a=source.index('      switch (ti.z) {',source.index('__global__ void __launch_bounds__(MOE_FUSED_THREADS'))
 b=source.index('      if (fa.dbg && threadIdx.x == MOE_CB)',a)
 fn='consumer_item_ss' if args.arm in ('ss_all','ss_k64') or args.arm=='ss' and args.width>=160 else 'consumer_item'
 source=source[:a]+f'      {fn}<{args.width}, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);\n'+source[b:]
source+=f'\n}} // namespace okc_moe\nvoid instantiate_prefill(okc_moe::FusedArgs fa, cudaStream_t stream) {{\n okc_moe::moe_fused_persistent<{args.fine}><<<132, 384, 232448, stream>>>(fa);\n}}\n'
source=re.sub(r'^DEVI\b','static DEVI',source,flags=re.M)
source=re.sub(r'^__noinline__ __device__','static __noinline__ __device__',source,flags=re.M)
flags='-O3 -gencode=arch=compute_90a,code=sm_90a -std=c++20 --use_fast_math -Xptxas -v'
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
width_label='all' if args.full_switch else str(args.width)
label=f'_FINE{args.fine}' if args.fine or args.full_switch else ''
if args.arm in ('ss','ss_all') and args.ss_consumer_regs!=184:label+=f'_R{args.ss_consumer_regs}'
if args.label:label+='_'+args.label
path=ROOT/f'muse_prefill_layer/SCALE_CODEGEN_{args.arm}_N{width_label}{label}_{args.date}.json.gz';assert not path.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
 with urllib.request.urlopen(req,timeout=180) as r:result=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:result=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=f'Fused-kernel body including real producer/publisher/consumer roles, width {width_label}/FINE{args.fine}; not linked judge extension. Never executed.',arm=args.arm,width=width_label,fine=args.fine,full_switch=args.full_switch,parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),native_sha256=hashlib.sha256(native.encode()).hexdigest(),request=request,response=result)
with gzip.open(path,'wt') as f:json.dump(out,f)
print(path.relative_to(ROOT),'code',result.get('code'))
for row in result.get('stderr',[]):
 s=row.get('text','')
 if any(x in s for x in ['ptxas','spill','stack','error','Error','C75']):print(s[:300])
for name,d in result.get('devices',{}).items():print(name,len(d.get('asm',[])),'clipped',any('[truncated' in r.get('text','') for r in d.get('asm',[])))
