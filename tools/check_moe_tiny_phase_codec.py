"""Stress exact telemetry retention before spending an H200 check."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import struct
from moe_tiny_phase_codec import encode_report,decode_report

root=Path(__file__).resolve().parents[1]
path=root/'muse_moe_w4a16/submission_v33_tiny_phase_probe.py';source=path.read_text()
parent=(root/'muse_moe_w4a16/submission.py').read_text();assert source.startswith(parent)
tree=ast.parse(source)
original=[ast.dump(n) for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
assert original==[ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
codec=(root/'tools/moe_tiny_phase_codec.py').read_text()
node=next(n for n in ast.parse(codec).body if isinstance(n,ast.FunctionDef) and n.name=='encode_report')
assert ast.get_source_segment(codec,node).replace('def encode_report(','def _tp_encode_report(') in source
fixture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tp_fixture')
cpu=ast.get_source_segment(source,fixture).split('    gen=torch.Generator')[0]+'    return ids\n'
env={'_TPRandom':random};exec(cpu,env)
fixtures=0
for m in (1,2,3,4):
    for lo in (0,13,224):
        for seed in (3101,7907):
            for nlocal in (0,1,2,4,8):
                rows=env['_tp_fixture'](m,seed,nlocal,None,lo)
                assert len(rows)==m
                assert all(len(set(r))==8 and sum(lo<=e<lo+32 for e in r)==nlocal for r in rows)
                fixtures+=1
rng=random.Random(1337)
f32=lambda value:struct.unpack('<f',struct.pack('<f',value))[0]
errors=lambda:{k:[f32(rng.random()*.0001),f32(rng.random()*.001)] for k in ('old','trace','a1')}
report=dict(status='complete',stage='complete',rows=[],nearby=[],traces=[],
    clock_columns=['entry','routed','phase_a_end','phase_b_end','grid_barrier_end','exit'],
    arms=['a0','old','trace','a1'],rounds=8)
for pad in (0,1,3,7):
    for seed in (3101,7907):
        for n in (0,1,2,4,8):
            report['rows'].append(dict(seed=seed,nlocal=n,pad_mib=pad,
                ns={k:[rng.randrange(10000,70000) for _ in range(8)] for k in report['arms']},
                error=errors(),fresh=errors(),change=f32(rng.random()*20),scratch_mod32m=rng.randrange(1<<25)))
for m in (2,3,4):report['nearby'].append(dict(m=m,error=errors(),fresh=errors(),change=f32(12.)))
for seed in (3101,7907):
    clocks=[]
    for c in range(132):
        current=(1<<63)+rng.randrange(5000);row=[current]
        for col in range(5):current+=rng.randrange(4000,50000);row.append(current)
        clocks.append(row)
    report['traces'].append(dict(seed=seed,nlocal=2,pad_mib=0,grid=132,clocks_ns=clocks))
wire=encode_report(report);assert decode_report(wire)==report
assert len(wire)<14000,(len(wire),'payload must leave room for profiler tail')
# Exact u32 event boundaries, u64 origins, zero-width columns, empty errors.
edge=copy.deepcopy(report);edge['rows']=edge['rows'][:1];edge['traces']=edge['traces'][:1]
for k in edge['arms']:edge['rows'][0]['ns'][k]=[1,2,31,32,65535,65536,2**31,2**32-1]
edge['traces'][0]['clocks_ns']=[[2**64-100]*6 for _ in range(132)]
assert decode_report(encode_report(edge))==edge
failed=dict(status='failed',stage='build',rows=[],nearby=[],traces=[],error='test failure')
assert decode_report(encode_report(failed))==failed
bad=copy.deepcopy(edge);bad['rows'][0]['change']=.1
try:encode_report(bad)
except AssertionError:pass
else:raise AssertionError('must reject lossy float64 metric encoding')
out=dict(scope='CPU-only report/fixture/source preflight, not CUDA correctness or timing.',
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),canonical_prefix_exact=True,
    kernel_functions_unchanged=True,cpu_routing_fixtures=fixtures,stress_rows=len(report['rows']),
    exact_event_timestamps=40*32,exact_u64_clocks=2*132*6,payload_chars=len(wire),
    retention_limit=16000,edge_roundtrip=True,lossy_metric_rejected=True,
    rejected_initial_protocol=dict(payload_chars=20388,reason='Exceeded 16000 tail; repaired before GPU launch without dropping samples.'))
p=root/'muse_moe_w4a16/TINY_PHASE_PREFLIGHT_2026-10-31.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
