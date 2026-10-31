"""Compile reduced diagnostic bodies; never execute torch or the full module."""
import argparse
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import types
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('version',choices=['v27_phase_trace','v28_lazy_phase_trace'])
parser.add_argument('--label',default='',help='Preserve earlier diagnostic responses after an unsubmitted source repair')
args=parser.parse_args();assert re.fullmatch('[a-z0-9_]*',args.label)
p=ROOT/'muse_moe_w4a16'/('submission_'+args.version+'_probe.py');tree=ast.parse(p.read_text());env={}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CUDA_SRC','CPP_SRC','MAIN_NS','MAIN_DEFINES','DQ7','MAIN_FALLBACK_DEFINES','_MP_LAZY_POINTER']:
        env[n.targets[0].id]=ast.literal_eval(n.value)
env.setdefault('_MP_LAZY_POINTER',False);env['_MPHash']=hashlib
captured={}
def capture_build(**kwargs):
    captured.update(kwargs)
    return types.SimpleNamespace(okc_set_pdl=lambda v:None)
env['load_inline']=capture_build
for name in ['_main_defines','_mp_build']:
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),'inspected-own-builder','exec'),env)
env['_mp_build']({'sfx':('-DOKC_PRMTSC=1',)})
native=captured['cuda_sources'];source=native.split('#include <torch/extension.h>')[0]
source=re.sub(r'^__device__\b','static __device__',source,flags=re.MULTILINE)
start=source.index('    switch (ntt) {');end=source.index('\n    gs += s1 - s0;',start)
source=source[:start]+'    wgs_seg<PB,8>(p,S,ring,u,s0,s1,slot,seq,pd);'+source[end:]
source+='\nvoid instantiate_trace(okc::Params p,cudaStream_t s){okc::moe_fused<true><<<132,okc::NTL,231168,s>>>(p);}\n'
flags=' '.join(captured['extra_cuda_cflags']+['-std=c++17','-Xptxas','-v'])
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
target=ROOT/'muse_moe_w4a16'/('TRACE_CODEGEN_'+args.version+'_NTT8_2026-10-31'+('_'+args.label if args.label else '')+'.json.gz');assert not target.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:
    response=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
archive=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Reduced NTT8/BIGtrue/internal-linkage compiler diagnostic, not full H200 extension or performance result',
             source=str(p.relative_to(ROOT)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
             full_native_sha256=hashlib.sha256(native.encode()).hexdigest(),compile_source_sha256=hashlib.sha256(source.encode()).hexdigest(),request=request,response=response)
with gzip.open(target,'wt') as f:json.dump(archive,f)
print(target.relative_to(ROOT),'code',response.get('code'))
stderr='\n'.join(r.get('text','') for r in response.get('stderr',[]))
for line in stderr.splitlines():
    if any(k in line for k in ['ptxas','bytes stack','error','Error','C751']):print(line[:240])
for name,device in response.get('devices',{}).items():print(name,'rows',len(device.get('asm',[])),'truncated',any('[truncated' in r.get('text','') for r in device.get('asm',[])))
