"""Offline ILC source, state isolation, copy ownership, schedule and lossless codec."""
import ast
import base64
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
import zlib

from bf16_ilc_allocator import CPP,CUDA

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_bf16_proj'
p=TRACK/'submission_v48_ilc_probe.py';s=p.read_text();parent=(TRACK/'submission.py').read_text()
assert s.startswith(parent)
nodes=ast.parse(s).body
names={'_ILC_CPP','_ILC_CUDA','_ILC_SPECS','_ILC_ORDERS'}
env={n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)
     and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names}
assert env['_ILC_CPP']==CPP and env['_ILC_CUDA']==CUDA
assert len([n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'])==len(
    [n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'])
env.update(_ILCStruct=struct,_ILCJSON=json,_ILCZlib=zlib,_ILCB64=base64)
for name in ('_ilc_bits','_ilc_wire','_ilc_replace'):
    fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pure ILC host functions>','exec'),env)

source={'raw':{'o':object(),'pooler':object()},'cnt':object()};saved=dict(source['raw']);other=object()
copied=env['_ilc_replace'](source,('raw','o'),other)
assert source['raw']==saved and copied['raw']['o'] is other
assert copied['cnt'] is source['cnt'] and copied['raw']['pooler'] is source['raw']['pooler']

for shift in range(13):
    orders=[tuple((a+shift)%4 for a in env['_ILC_ORDERS'][r%4]) for r in range(8)]
    positions=collections.Counter((a,pos) for row in orders for pos,a in enumerate(row))
    pairs=collections.Counter((a,b) for row in orders for a,b in zip(row,row[1:]))
    assert set(positions)==set(itertools.product(range(4),repeat=2)) and set(positions.values())=={2}
    assert set(pairs)=={(a,b) for a in range(4) for b in range(4) if a!=b} and set(pairs.values())=={2}

# Actual grid-stride copy geometry: every vector/tail byte exactly once.
copy_cases=0
for count in list(range(1,513))+[4095,4096,4097,65535,65536,65537,131072,1048591]:
    blocks=min(max((count+4095)//4096,1),132*8);threads=blocks*256
    vector_count=count//16;seen=[]
    for thread in range(threads):
        seen.extend(range(thread,vector_count,threads))
    assert sorted(seen)==list(range(vector_count))
    assert list(range(count%16))==[thread for thread in range(threads) if thread<count%16]
    assert vector_count*16+count%16==count
    copy_cases+=1

rounding_cases=0
for g0,g1 in itertools.product((65536,131072,2097152),repeat=2):
    gran=abs(g0*g1)//math.gcd(g0,g1)
    for logical in (1,15,16,17,65535,65536,65537,1<<26):
        physical=((logical-1)//gran+1)*gran
        assert physical>=logical and physical%g0==physical%g1==0 and physical-logical<gran
        rounding_cases+=1

rng=random.Random(0x11c);rows=[]
for pad,seed,case in itertools.product(range(2),(3209,7909),range(8)):
    # High-entropy accepted FP32 metrics stress the binary representation.
    def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
    pairs=[[f32(rng.random()*1e-5),f32(rng.random()*1e-4)] for _ in range(8)]
    raw=[[env['_ilc_bits'](rng.randrange(400,1000)*32/1e6) for _ in range(8)] for _ in range(4)]
    rows.append(dict(id=case,pad=pad,seed=seed,allocation=[1<<26,1<<26,65536,0,1],
        virtual_address_mod_32m=pad*65536,fp64=[rng.random()*.002,rng.random()*.003,0.],
        normal=pairs[0],compressed=pairs[1],fresh=pairs[2:4],remaps=pairs[4:8],event_ms_bits=raw))
report=dict(status='complete',rows=rows)
def decode(wire):
    raw=zlib.decompress(base64.b64decode(wire));length=struct.unpack_from('<I',raw)[0]
    meta=json.loads(raw[4:4+length]);offset=4+length;decoded=[]
    for row in meta['rows']:
        codec=row['time_codec'];n=64 if codec=='ns16_exact' else 128
        values=list(struct.unpack_from('<32H' if n==64 else '<32I',raw,offset));offset+=n
        bits=[env['_ilc_bits'](v/1e6) for v in values] if n==64 else values
        metrics=struct.unpack_from('<16f3d',raw,offset);offset+=88
        decoded.append((bits,list(metrics)))
    assert offset==len(raw)
    return meta,decoded
wire=env['_ilc_wire'](report);meta,back=decode(wire)
assert len(wire)<8000,('full stress payload too large before GPU',len(wire))
for row,(bits,metrics) in zip(rows,back):
    assert bits==sum(row['event_ms_bits'],[])
    assert metrics==row['normal']+row['compressed']+sum(row['fresh'],[])+sum(row['remaps'],[])+row['fp64']
# Force exact fallback, including a sub65us sample not representable as integer ns.
fallback=dict(rows[0]);fallback['event_ms_bits']=[a[:] for a in rows[0]['event_ms_bits']]
fallback['event_ms_bits'][0][0]=env['_ilc_bits'](.013456789)
other=env['_ilc_wire'](dict(rows=[fallback]));m,b=decode(other)
assert m['rows'][0]['time_codec']=='float32_bits'
assert b[0][0]==sum(fallback['event_ms_bits'],[])

out=dict(scope='CPU source/geometry/state/codec proofs only; native allocator/driver support/lifetime and GPU correctness/performance unproven.',
    source_sha256=hashlib.sha256(s.encode()).hexdigest(),canonical_prefix_unchanged=True,
    no_new_timed_kernel_definition=True,new_state_dictionary_isolated=True,
    copy_geometry_cases=copy_cases,allocation_rounding_cases=rounding_cases,
    balanced_order_shifts=13,full_stress_groups=32,exact_event_float32_values=1024,
    exact_metric_values=32*19,payload_chars=len(wire),fallback_payload_chars=len(other),
    event_and_metric_roundtrip_exact=True,
    source_lifetime_invariants=['all VMM owners created as shared_ptr before driver mutation',
      'tensor from_blob deleter retains arena owner','context synchronized before every unmap',
      'native destructor restores stored CUDA context and synchronizes before releasing handles',
      'weight copies and VMM remaps only in independent prepare diagnostic',
      'returned state is unchanged original prepare result; no experimental allocations retained'])
(TRACK/'ILC_PREFLIGHT_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',out)
