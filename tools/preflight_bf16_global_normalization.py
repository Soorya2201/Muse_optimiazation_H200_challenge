"""Execute only source-generation/host-call setup; never import CUDA parent."""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import types

ROOT=Path(__file__).resolve().parents[1]
parent=(ROOT/'muse_bf16_proj/submission.py').read_text()
candidate=ROOT/'muse_bf16_proj/submission_v43_global_normalization_probe.py'
source=candidate.read_text()
assert source.startswith(parent)
assert hashlib.sha256(parent.encode()).hexdigest()=='290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90'
ns={'_CPRe':re}
tree=ast.parse(source)
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        try:ns[n.targets[0].id]=ast.literal_eval(n.value)
        except (ValueError,TypeError,SyntaxError):pass
config_node=next(n.value for n in tree.body if isinstance(n,ast.Assign)
    and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SK2_CONFIGS')
assert all(isinstance(n,(ast.List,ast.Tuple,ast.Constant,ast.Name,ast.Load)) for n in ast.walk(config_node))
ns['SK2_CONFIGS']=eval(compile(ast.Expression(config_node),'<static config>','eval'),{'__builtins__':{}},ns)
functions=['_sk2_tokr','_sk2_name','_sk2_geom','_gn_metadata','_gn_body',
           '_cp_escape_body','_cp_specs','_cp_source','_cp_run']
for name in functions:
    n=next(n for n in reversed(tree.body) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(candidate),'exec'),ns)
specs=ns['_cp_specs']()
original=json.loads((ROOT/'muse_bf16_proj/C12_PERMUTE_PREFLIGHT_2026-10-31.json').read_text())
assert json.loads(json.dumps(specs))==original['specs']
assert all(s['cfg'][-1]==1 for s in specs)
native=ns['_cp_source']()
assert native.count('v[e] *= gn_restore;')==4
assert native.count('const uint32_t bq = gn_original_base(base4);')==2
assert native.count('const uint32_t bq = base4 & 0xFFu;')==2
assert native.count('lop3.b32 %0, %1, 0x87878787, %2, 0xe2;')==4
assert '@@' not in native
gzip_path=ROOT/'muse_bf16_proj/GLOBAL_NORMALIZATION_NATIVE_2026-10-31.cu.gz'
assert not gzip_path.exists()
with gzip.open(gzip_path,'wt') as f:f.write(native)
seen=[]
ns['torch']=types.SimpleNamespace(bfloat16='bf16',empty=lambda *a,**kw:object())
ns['_cp_pack']=lambda st,spec:(None,None,4,st['base'])
mod=types.SimpleNamespace(cp_run=lambda *a:seen.append(a))
x=types.SimpleNamespace(shape=(16,8192),device='synthetic')
for b in range(1,120):
    st={'base':b*0x01010101,'dummy':None}
    for arm in (0,1):
        ns['_cp_run'](mod,st,specs[2],dict(x=x,config={'n':2304}),None,arm)
        p=seen[-1][5]
        assert p==(st['base'] if arm==0 else ns['_gn_metadata'](st['base']))
        if arm:
            assert (p&7)|((p>>5)&0x38)|((p>>1)&0x40)==b
receipt=json.loads((ROOT/'muse_bf16_proj/run_receipts/b89f4e5d007047d9.json').read_text())
cells={(r['layer'],r['m']) for r in specs}
def model(cut):
    factors=[]
    for load in sorted({r['case']['load'] for r in receipt['benchmarks']}):
        rows=[r for r in receipt['benchmarks'] if r['case']['load']==load]
        before=sum(r['median_ns']*r['case']['weight'] for r in rows)
        after=sum(r['median_ns']*r['case']['weight']*(1-cut if (r['case']['layer'],r['case']['m']) in cells else 1) for r in rows)
        factors.append(before/after)
    return math.prod(factors)**(1/len(factors))
lo,hi=0.,1.
for _ in range(60):
    mid=(lo+hi)/2
    if model(mid)<1.03:lo=mid
    else:hi=mid
report=dict(scope='CPU AST/native generation, exact host-call metadata and same-job scenario screen; no GPU performance prediction.',
    source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    native_sha256=hashlib.sha256(native.encode()).hexdigest(),native_bytes=len(native),
    specs=specs,host_calls=len(seen),stage_counts_unchanged=True,
    source_changes='Packed1 decoder/metadata/base extraction and pre-correction partial rescale only; original timed prefix unchanged.',
    opportunity_receipt='b89f4e5d007047d9',phase_zero_board=model(1),
    uniform_cut_to_gain={str(cut):model(cut) for cut in [.05,.1,.15,.25]},
    uniform_target_cell_time_cut_needed_for_3pct=hi,
    limitations='Decoder share is not measured. Earlier sign-layout-only transfer found no eligible gain. This additional normalization may still lose to memory bandwidth, restore overhead or register pressure.')
target=ROOT/'muse_bf16_proj/GLOBAL_NORMALIZATION_PREFLIGHT_2026-10-31.json'
assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
print('PASS',len(specs),'specs;',len(seen),'host calls;',len(native),'native bytes')
print('Board models',report['uniform_cut_to_gain'],'phase zero',model(1),'needed',hi)
