"""Stress exact unit matrices/four-arm timing rows before a GPU check."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import zlib
from moe_unit_trace_codec import matrix,rows,trace

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'muse_moe_w4a16/submission_v29_unit_trace_probe.py';tree=ast.parse(p.read_text())
env={'_MP64':base64,'_MPStruct':struct}
for name in ['_ut_pack','_mp_pack_rows']:
    n=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    exec(compile(ast.Module(body=[n],type_ignores=[]),'own-unit-codec','exec'),env)
r=random.Random(102319);f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
records=[]
for pad in (0,1,3,7):
    for seed in (3101,7907):
        for m in (32,128,256):
            ns={k:[r.randrange(10000,400000) for _ in range(5)] for k in ['a0','trace','a1','canonical']}
            records.append(dict(m=m,seed=seed,pad_mib=pad,scratch_mod32m=r.randrange(32<<20),ns=ns,
                error={k:[f32(r.random()/1000) for _ in range(2)] for k in ['trace','a1','canonical']},
                fresh={k:[f32(r.random()/1000) for _ in range(2)] for k in ['trace','a1','canonical']},
                ratio=statistics.median(math.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace'])),
                aa=statistics.median(a/b for a,b in zip(ns['a0'],ns['a1'])),
                baseline_speedup_vs_canonical=statistics.median(c/math.sqrt(a*b) for a,b,c in zip(ns['a0'],ns['a1'],ns['canonical'])),change=f32(r.random()+1)))
wire=dict(status='complete',stage='complete',row_count=24,row_format='<24I12f4d',
          rows_b64=env['_mp_pack_rows'](records),resources=[0,96,0,640,231168]*2,
          baseline_resources=[0,96,0,640,231168]*2,traces=[],seconds=90.105)
assert rows(wire)==records
for m in (32,256):
    origin=9_123_456_789_100;phase=[[r.randrange(1,300000) for _ in range(8)] for _ in range(4)]
    original=[];flat=[];counts=[]
    for c in (0,33,66,99):
        fs=[]
        for i in range(24):
            start=r.randrange(1,300000);flush=start+r.randrange(32,2000);fe=flush+r.randrange(32,4000)
            wait=fe+r.randrange(3000,20000);end=wait+r.randrange(64,2000)
            f=[i%2,r.randrange(1024),0,16,r.randrange(1,9),2,1,2,r.randrange(1024),256|r.randrange(1,17),start,wait,end,flush,fe,0]
            fs.append(f)
        counts.append(len(fs));flat.extend(fs)
        original.append(dict(cta=c,frames=[[origin+x-1 if j in range(10,15) and x else x for j,x in enumerate(f)] for f in fs]))
    cuts=[v for c in range(132) for v in (c*64,(c+1)*64,c*32,(c+1)*32)]
    tr=dict(kind='sampled_unit_trace_v1',origin_ns=origin,grid=132,ctas=[0,33,66,99],
        phase_clocks=env['_ut_pack'](phase),frames=env['_ut_pack'](flat),frame_counts=counts,
        frame_fields=['phase','unit','s0','s1','ntt','prior_kind','after_kind','flush_kind','flush_unit','flush_nst','segment_start','compute_wait_end','segment_end','flush_start','flush_end','final_flush'],
        metadata_header=[17,7,120000,160000],cuts_u16_le_b64=base64.b64encode(struct.pack('<528H',*cuts)).decode(),
        jobs=[[j,r.randrange(1,65)] for j in range(24)],m=m,seed=3101,pad_mib=0)
    assert matrix(tr['frames'])==flat
    decoded=trace(tr);assert decoded['groups']==original
    assert decoded['phase_clocks_ns']==[[origin+x for x in row] for row in phase]
    wire['traces'].append(tr)
for test in [[[0,1,2**32-1]]*3,[[0]*16 for _ in range(4)],[[2**32-1]*8]]:
    assert matrix(env['_ut_pack'](test))==test
payload=base64.b64encode(zlib.compress(json.dumps(wire,separators=(',',':')).encode(),9))
assert json.loads(zlib.decompress(base64.b64decode(payload)))==wire
assert len(payload)<12000,len(payload)
out=dict(status='pass',scope='CPU stress data, no GPU timing',rows=24,traces=2,frames_per_trace=96,
    payload_chars=len(payload),payload_plus_1200_tail_chars=len(payload)+1200,
    exact_rows=True,exact_clock_and_metadata_matrices=True,zero_width_and_u32_boundary_cases=True,
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
target=ROOT/'muse_moe_w4a16/UNIT_TRACE_CODEC_CPU_2026-10-31.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
