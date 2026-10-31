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
parser=argparse.ArgumentParser();parser.add_argument('version',choices=['v27_phase_trace','v28_lazy_phase_trace','v29_unit_trace','v30_flush_trace'])
parser.add_argument('--label',default='',help='Preserve earlier diagnostic responses after an unsubmitted source repair')
parser.add_argument('--unit-control',action='store_true',help='V29/V30 with detail recording disabled, same register allocation/layout')
parser.add_argument('--ntt',type=int,choices=range(1,9),default=8)
parser.add_argument('--full-switch',action='store_true',help='Keep all eight NTT instantiations; may exceed public compiler limits')
parser.add_argument('--skip-asm',action='store_true',help='Request compiler statistics without assembly, as documented by CE API')
parser.add_argument('--host-asm',action='store_true',help='Use host assembly view while retaining device compilation/statistics')
args=parser.parse_args();assert re.fullmatch('[a-z0-9_]*',args.label)
p=ROOT/'muse_moe_w4a16'/('submission_'+args.version+'_probe.py');tree=ast.parse(p.read_text());env={}
submission_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CUDA_SRC','CPP_SRC','MAIN_NS','MAIN_DEFINES','DQ7','MAIN_FALLBACK_DEFINES','_MP_LAZY_POINTER','_UT_REGISTER_SPLIT']:
        env[n.targets[0].id]=ast.literal_eval(n.value)
env.setdefault('_MP_LAZY_POINTER',False);env['_MPHash']=hashlib
captured={}
def capture_build(**kwargs):
    captured.update(kwargs)
    return types.SimpleNamespace(okc_set_pdl=lambda v:None)
env['load_inline']=capture_build
names=['_main_defines','_mp_build']
if args.version=='v29_unit_trace':names=['_ut_register_split','_ut_native']+names
if args.version=='v30_flush_trace':names=['_uf_register_split','_uf_native']+names
for name in names:
    n=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    exec(compile(ast.Module(body=[n],type_ignores=[]),'inspected-own-builder','exec'),env)
if args.unit_control:
    assert args.version in ['v29_unit_trace','v30_flush_trace']
    env['_mp_build']({'sfx':('-DOKC_PRMTSC=1',)},False)
else:
    env['_mp_build']({'sfx':('-DOKC_PRMTSC=1',)})
native=captured['cuda_sources'];source=native.split('#include <torch/extension.h>')[0]
source=re.sub(r'^__device__\b','static __device__',source,flags=re.MULTILINE)
if not args.full_switch:
    start=source.index('    switch (ntt) {');end=source.index('\n    }',start)+len('\n    }')
    source=source[:start]+f'    wgs_seg<PB,{args.ntt}>(p,S,ring,u,s0,s1,slot,seq,pd);'+source[end:]
source+='\nvoid instantiate_trace(okc::Params p,cudaStream_t s){okc::moe_fused<true><<<132,okc::NTL,231168,s>>>(p);}\n'
flags=' '.join(captured['extra_cuda_cflags']+['-std=c++17','-Xptxas','-v'])
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
if args.skip_asm:request['options']['compilerOptions']['skipAsm']=True
if args.host_asm:request['options']['filters']['binary']=False
variant='FULL' if args.full_switch else f'NTT{args.ntt}'
target=ROOT/'muse_moe_w4a16'/('TRACE_CODEGEN_'+args.version+'_'+variant+'_2026-10-31'+('_'+args.label if args.label else '')+'.json.gz');assert not target.exists()
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:
    response=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
archive=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=f'{variant}/BIGtrue/internal-linkage compiler diagnostic, not full H200 extension or performance result',
             unit_recording=not args.unit_control,
             source=str(p.relative_to(ROOT)),source_sha256=submission_sha256,
             full_native_sha256=hashlib.sha256(native.encode()).hexdigest(),compile_source_sha256=hashlib.sha256(source.encode()).hexdigest(),request=request,response=response)
with gzip.open(target,'wt') as f:json.dump(archive,f)
print(target.relative_to(ROOT),'code',response.get('code'))
stderr='\n'.join(r.get('text','') for r in response.get('stderr',[]))
for line in stderr.splitlines():
    if any(k in line for k in ['ptxas','bytes stack','error','Error','C751']):print(line[:240])
for name,device in response.get('devices',{}).items():print(name,'rows',len(device.get('asm',[])),'truncated',any('[truncated' in r.get('text','') for r in device.get('asm',[])))
