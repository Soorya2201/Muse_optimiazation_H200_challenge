"""Pure host roundtrip and exact protocol-source checks before H200 launch."""
import ast
import base64
import collections
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import zlib

root=Path(__file__).resolve().parents[1];track=root/'muse_prefill_layer'
p=track/'submission_v34_wide_batch_drained_probe.py';source=p.read_text();parent=(track/'submission.py').read_text();assert source.startswith(parent)
tree=ast.parse(source);assert len([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'])==len([n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'])
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wb_wire')
env=dict(_KSCopy=copy,_KSJSON=json,_KSStruct=struct,_KS64=base64,_KSZlib=zlib)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<wire pure host>','exec'),env)
orders=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))
for shift in range(8):
 rows=[tuple((a+shift)%4 for a in orders[i%4]) for i in range(8)]
 assert set(collections.Counter((a,i) for r in rows for i,a in enumerate(r)).values())=={2}
 assert set(collections.Counter((a,b) for r in rows for a,b in zip(r,r[1:])).values())=={2}
probe=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wb_probe'))
assert 'orders[rep%4]' in probe and 'if rep&1:order.reverse()' not in probe
assert 'identical untimed warm-up before cold read' in probe
assert "report['weight_layout_unchanged']=True" in probe and '_ss_movement' not in probe
assert "_WBMath.isfinite(ms) and ms>0" in probe
rng=random.Random(0xB47C)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def errors():return {k:[f32(rng.random()*.001),f32(rng.random()*.009)] for k in ('old','new','a1')}
report=dict(status='complete',stage='complete',rows=[],nearby=[])
for m,seed,pad in itertools.product((753,3012,16384),(3208,7911),(0,1,3,7)):
 raw={k:[bits(rng.uniform(.01,8)) for _ in range(8)] for k in ('a0','old','new','a1')}
 report['rows'].append(dict(m=m,seed=seed,pad_mib=pad,event_ms_bits=raw,error=errors(),fresh=errors(),change=f32(rng.random()),xmod=rng.randrange(32<<20)))
for m in (31,33,65,257,2048):report['nearby'].append(dict(m=m,error=errors(),fresh=errors(),change=f32(rng.random())))
wire=env['_wb_wire'](report);payload=zlib.decompress(base64.b64decode(wire));length=struct.unpack_from('<I',payload)[0];d=json.loads(payload[4:4+length]);frames=[struct.unpack_from('<32I13f',payload,4+length+row*180) for row in range(24)];raw=[v for frame in frames for v in frame[:32]]
assert len(payload)==4+length+24*180
expected=[v for row in report['rows'] for k in ('a0','old','new','a1') for v in row['event_ms_bits'][k]];assert raw==expected
assert d['rows']==[{k:v for k,v in row.items() if k not in ('event_ms_bits','error','fresh','change')} for row in report['rows']]
for row,frame in zip(report['rows'],frames):
 expected_metrics=[v for field in ('error','fresh') for name in ('old','new','a1') for v in row[field][name]]+[row['change']]
 assert list(frame[32:])==expected_metrics
assert d['nearby']==report['nearby']
print('complete stress chars',len(wire))
assert len(wire)<12000
bad=copy.deepcopy(report);bad['agreement_finite']={k:dict(nan=16777216,inf=16777216) for k in ('a0','old','new','a1')};bad['comparison_arm']='new';bad['agreement_phase']='fresh';bad.update(status='failed',error='compiler diagnostic '+''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(2500)),error_tail=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(1000)),trace=''.join('File /tmp/own_extension.py, line123, in _ks_build\n' for _ in range(20)))
failure_wire=env['_wb_wire'](bad)
print('failure stress chars',len(failure_wire))
assert len(failure_wire)<12000,('failure payload can lose primary evidence',len(failure_wire))
out=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),canonical_prefix_unchanged=True,no_new_timed_kernel=True,
         balanced_order_shifts=8,stress_groups=24,exact_float32_event_values=768,event_bits_roundtrip_exact=True,
         all_error_and_fresh_metrics_roundtrip_exact=True,payload_chars=len(wire),failure_payload_chars=len(failure_wire),
         preserves_primary_and_secondary_exception=True,
         scope='Pure host/source/codec checks only; no proof of native binding, GPU arithmetic, allocation or speed.')
(track/'WIDE_BATCH_DRAINED_PROBE_PREFLIGHT_2026-11-01.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
