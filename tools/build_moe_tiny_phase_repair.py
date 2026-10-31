"""Repair only V33's missing host import and dry-run its build setup on CPU."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
original=root/'muse_moe_w4a16/submission_v33_tiny_phase_probe.py';old=original.read_text()
assert hashlib.sha256(old.encode()).hexdigest()=='ec031d29e2218830859b0b452d72b638d3a3541323761decf2089c216d8f9270'
assert old.count('import hashlib as _TPHash')==0 and old.count('tag=hashlib.sha256')==1
new=old.replace('import base64 as _TP64','import base64 as _TP64\nimport hashlib as _TPHash',1)
new=new.replace('tag=hashlib.sha256','tag=_TPHash.sha256',1)
target=root/'muse_moe_w4a16/submission_v34_tiny_phase_import_repair.py';assert not target.exists()

def build_node(source):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_tp_build')

tree=ast.parse(old);env={}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        name=n.targets[0].id
        if name in ('CUDA_SRC','CPP_SRC','TINY_DEFINES','MAIN_NS','DQ7'):env[name]=ast.literal_eval(n.value)
captures=[]
class StubModule:
    def okc_set_pdl(self,value):assert value==0

def capture(*args,**kwargs):captures.append(dict(name=args[0],**kwargs));return StubModule()

env.update(load_inline=capture)
exec(compile(ast.Module(body=[build_node(old)],type_ignores=[]),'<original build setup>','exec'),env)
try:env['_tp_build']({'sfx':('-DOKC_PRMTSC=1',)},False)
except NameError as error:assert 'hashlib' in str(error)
else:raise AssertionError('Expected to reproduce original host failure')
env['_TPHash']=hashlib
exec(compile(ast.Module(body=[build_node(new)],type_ignores=[]),'<repaired build setup>','exec'),env)
for suffix in ((),('-DOKC_PRMTSC=1',)):
    for trace in (False,True):env['_tp_build']({'sfx':suffix},trace)
assert len(captures)==4
for i in (0,2):
    a,b=captures[i:i+2]
    assert a['cpp_sources']==b['cpp_sources'] and a['functions']==b['functions']
    assert a['extra_cuda_cflags']==b['extra_cuda_cflags']
    assert a['cuda_sources'].replace('#define OKC_DEBUG 0','#define OKC_DEBUG 1',1)==b['cuda_sources']
assert all('okc_tiny_resources' in c['functions'] for c in captures)
parent=(root/'muse_moe_w4a16/submission.py').read_text();assert new.startswith(parent)
oldkernels=[ast.dump(n)for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)and n.name=='kernel_fn']
assert oldkernels==[ast.dump(n)for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)and n.name=='kernel_fn']
target.write_text(new);ast.parse(new)
out=dict(scope='CPU execution of exact host build function with extension compiler stubbed; not CUDA compilation or GPU validation.',
    original_sha256=hashlib.sha256(old.encode()).hexdigest(),source_sha256=hashlib.sha256(new.encode()).hexdigest(),
    original_host_nameerror_reproduced=True,build_setups_executed=4,
    normal_and_fallback_scale_flags_checked=True,trace_control_only_debug_define_differs=True,
    canonical_prefix_and_kernels_unchanged=True,
    builds=[dict(name=c['name'],cpp_sha256=hashlib.sha256(c['cpp_sources'].encode()).hexdigest(),native_sha256=hashlib.sha256(c['cuda_sources'].encode()).hexdigest(),flags=c['extra_cuda_cflags'])for c in captures])
p=root/'muse_moe_w4a16/TINY_PHASE_REPAIR_PREFLIGHT_2026-10-31.json';assert not p.exists()
p.write_text(json.dumps(out,indent=2)+'\n')
print(target.relative_to(root),out['source_sha256']);print('Original NameError reproduced; all four repaired host build setups complete; GPU code unchanged.')
