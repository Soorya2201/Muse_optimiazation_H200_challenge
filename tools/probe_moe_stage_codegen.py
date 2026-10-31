"""Compare isolated segment codegen without compiling/executing Python kernels."""
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
from moe_stage_compact_variant import compact_stage,zero_mma_stage,scalar_zero_stage
from decoder_wgs_variants import register_split

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('variant',choices=['parent','compact','zero_mma','scalar_zero'])
parser.add_argument('--ntt',type=int,choices=range(1,9),default=8)
parser.add_argument('--regsplit',action='store_true')
parser.add_argument('--label',default='')
args=parser.parse_args()
assert re.fullmatch('[a-z0-9_]*',args.label)
p=ROOT/'muse_moe_w4a16/submission.py';raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
tree=ast.parse(raw);native=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
      and any(isinstance(x,ast.Name) and x.id=='CUDA_SRC' for x in n.targets))
if args.variant=='compact':native=compact_stage(native)
if args.variant=='zero_mma':native=zero_mma_stage(native)
if args.variant=='scalar_zero':native=scalar_zero_stage(native)
if args.regsplit:native=register_split(native,2)
source=native.split('#include <torch/extension.h>')[0]
source=re.sub(r'^__device__\b','static __device__',source,flags=re.MULTILINE)
source+=f'''
// COMPILER-ONLY: not a runnable kernel or full dispatch resource check.
namespace okc {{
__global__ void __launch_bounds__(NTL,1) stage_codegen_probe(Params p,int u,int s0,int s1,int slot,int seq,WgsPend pd,int* sink) {{
  if(threadIdx.x>=NT) {{
    {'asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");' if args.regsplit else ''}
    return;
  }}
  {'asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" ::: "memory");' if args.regsplit else ''}
  Smem& S=*reinterpret_cast<Smem*>(smem_raw);
  unsigned char* ring=smem_raw+((sizeof(Smem)+127)/128)*128;
  wgs_seg<false,{args.ntt}>(p,S,ring,u,s0,s1,slot,seq,pd);
  sink[threadIdx.x]=seq+pd.kind+pd.u;
}}
}}
'''
flags='-O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo -std=c++17 -DOKC_NS=3 -DOKC_NSF=3 -DOKC_NW=16 -DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_PRMTSC=1 -Xptxas -v'
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),
    filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
suffix=('_regsplit' if args.regsplit else '')+('_'+args.label if args.label else '')
target=ROOT/'muse_moe_w4a16'/f'STAGE_CODEGEN_{args.variant}_NTT{args.ntt}_2026-10-31{suffix}.json.gz'
assert not target.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:
    response=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Isolated phase-A wgs_seg, arbitrary segment bounds/pending record, same launch bounds; no routing/producer/full BIG dispatch; never executed.',
    variant=args.variant,ntt=args.ntt,regsplit=args.regsplit,parent_sha256=hashlib.sha256(raw).hexdigest(),
    modifier_sha256=hashlib.sha256((ROOT/'tools/moe_stage_compact_variant.py').read_bytes()).hexdigest(),
    native_sha256=hashlib.sha256(native.encode()).hexdigest(),request=request,response=response)
with gzip.open(target,'wt') as f:json.dump(out,f)
print(target.relative_to(ROOT),'code',response.get('code'))
for line in response.get('stderr',[]):
    text=line.get('text','')
    if any(x in text for x in ['ptxas','bytes stack','error','Error','C75']):print(text[:250])
for name,device in response.get('devices',{}).items():
    asm=device.get('asm',[]);truncated=any('[truncated' in r.get('text','') for r in asm)
    print(name,'rows',len(asm),'truncated',truncated)
