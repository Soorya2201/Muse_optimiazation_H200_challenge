"""Archive full kernel instantiation codegen; never execute source."""
import argparse,ast,datetime,gzip,hashlib,json,re,urllib.error,urllib.request
from pathlib import Path
from moe_two_cta_native import two_cta_native
from moe_two_cta_wide_native import two_cta_wide_native
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('arm',choices=['parent','two_cta','two_cta_wide']);p.add_argument('--big',type=int,choices=[0,1],default=1);p.add_argument('--label',default='');p.add_argument('--stats-only',action='store_true');p.add_argument('--compact-display',action='store_true');p.add_argument('--host-assembly',action='store_true');p.add_argument('--resource-linkfail',action='store_true');a=p.parse_args();assert re.fullmatch('[a-z0-9_]*',a.label)
parent=(root/'muse_moe_w4a16/submission.py').read_text();native=next(ast.literal_eval(n.value) for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
if a.arm=='two_cta':native=two_cta_native(native)
if a.arm=='two_cta_wide':native=two_cta_wide_native(native)
source=native.split('#include <torch/extension.h>')[0];source=re.sub(r'^__device__\b','static __device__',source,flags=re.M)
source+='\nconstexpr unsigned compiled_dynamic_smem=((sizeof(okc::Smem)+127)/128)*128+okc::WGS_NS*okc::UNIT_BYTES+okc::WGS_NX*okc::WGS_XS;\n'
if a.arm=='two_cta':source+='static_assert(compiled_dynamic_smem==110848 || compiled_dynamic_smem==110976);\nstatic_assert(okc::NTL==320);\n'
if a.arm=='two_cta_wide':source+='static_assert(compiled_dynamic_smem==113408);\nstatic_assert(sizeof(okc::Smem)==10976);\nstatic_assert(okc::NTL==320 && okc::FT_T==64);\nstatic_assert(3*okc::EMAX<=okc::UNIT_BYTES);\n'
source+=f'\nvoid instantiate_topology(okc::Params p,cudaStream_t s){{okc::moe_fused<{"true" if a.big else "false"}><<<{132 if a.arm=="parent" else 264},okc::NTL,0,s>>>(p);}}\n'
flags='-O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -DOKC_NS=3 -DOKC_NSF=3 -DOKC_NW='+('16' if a.arm=='parent' else '8')+' -DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_PRMTSC=1 -DOKC_DQ7=1 -Xptxas -v'
request=dict(source=source,options=dict(userArguments=flags,compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
if a.stats_only:request['options']['compilerOptions']['skipAsm']=True
if a.compact_display:request['options']['filters'].update(directives=True,commentOnly=True,trim=True)
if a.host_assembly:request['options']['filters']['binary']=False
if a.resource_linkfail:
 # Upstream NvccCompiler only adds -keep/device-asm extraction inputs for
 # non-executable builds. Ask for link mode but force a unique unresolved
 # symbol: no executable can be produced or run. The expected host link
 # failure may still preserve complete ptxas device-resource diagnostics.
 assert not a.stats_only and not a.host_assembly
 request['source']+='\nextern int muse_resource_probe_intentionally_unresolved_20261101();\nint main(){return muse_resource_probe_intentionally_unresolved_20261101();}\n'
 request['options']['filters']['execute']=True
path=root/f'muse_moe_w4a16/TWO_CTA_CODEGEN_{a.arm}_BIG{a.big}_2026-11-01{("_"+a.label) if a.label else ""}.json.gz';assert not path.exists()
try:
 req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
except (urllib.error.URLError,TimeoutError) as e:response=dict(code=None,transport_error=repr(e),stderr=[dict(text=str(e))])
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Full fused producer/consumer/routing/final body with all reachable NTT widths; internal linkage, no torch host binding, not GPU execution or measured occupancy.',intentional_host_link_failure=a.resource_linkfail,arm=a.arm,big=a.big,parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),native_sha256=hashlib.sha256(native.encode()).hexdigest(),request=request,response=response)
with gzip.open(path,'wt') as f:json.dump(out,f)
print(path.relative_to(root),'code',response.get('code'))
for r in response.get('stderr',[]):
 s=r.get('text','')
 if any(x in s for x in ('ptxas','error','Error','C751')):print(s[:400])
for k,v in response.get('devices',{}).items():print(k,len(v.get('asm',[])),'clipped',any('[truncated' in r.get('text','') for r in v.get('asm',[])))
