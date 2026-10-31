"""Exercise exact row/clock roundtrips and the 16KB log-tail budget offline."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import random
import statistics
import math
import struct
import zlib
from moe_phase_trace_codec import decode_rows,decode_trace

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'muse_moe_w4a16/submission_v27_phase_trace_probe.py'
t=ast.parse(p.read_text());n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_pack_rows')
ns={'_MPStruct':struct,'_MP64':base64}
exec(compile(ast.Module(body=[n],type_ignores=[]),'probe-row-packer','exec'),ns)
r=random.Random(9817)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
rows=[]
for pad in [0,1,3,7]:
    for seed in [3101,7907]:
        for m in [32,128,256]:
            times={k:[r.randrange(2000,500000) for _ in range(5)] for k in ['a0','trace','a1']}
            rows.append(dict(m=m,seed=seed,pad_mib=pad,
                error={k:[f32(r.random()/1e6) for _ in range(2)] for k in ['trace','a1']},
                fresh={k:[f32(r.random()/1e6) for _ in range(2)] for k in ['trace','a1']},
                ns=times,ratio=statistics.median(math.sqrt(a*b)/c for a,b,c in zip(times['a0'],times['a1'],times['trace'])),
                aa=statistics.median(a/b for a,b in zip(times['a0'],times['a1'])),change=f32(r.random()+1),
                scratch_mod32m=r.randrange(32<<20)))
wire=dict(status='complete',stage='complete',resources=[0,96,0,576,231168]*2,
          row_count=24,row_format='<19I8f3d',rows_b64=ns['_mp_pack_rows'](rows),traces=[],seconds=41.193)
assert decode_rows(wire)==rows
columns=['entry','routed','phase_a_end','phase_b_end','grid_barrier_end','exit','producer_join_end','gather_end']
for m in [32,256]:
    clocks=[];origin=123456789010000
    for c in range(132):
        t0=origin+r.randrange(0,1000);t1=t0+r.randrange(3000,6000);t7=t1+r.randrange(1000,5000)
        t6=t7+r.randrange(100,500);t2=t6+r.randrange(40000,130000);t3=t2+r.randrange(40000,100000)
        t4=origin+250000+r.randrange(0,2000);t5=t4+r.randrange(1000,6000)
        clocks.append([t0,t1,t2,t3,t4,t5,t6,t7])
    deltas=[];order=[0,1,7,6,2,3,4,5]
    for row in clocks:
        prev=origin;dr=[]
        for col in order:
            v=row[col]-prev;prev=row[col];assert 0<=v<2**32
            dr.append(v)
        deltas.append(dr)
    minima=[min(row[k] for row in deltas) for k in range(8)]
    widths=[(max(row[k] for row in deltas)-minima[k]).bit_length() for k in range(8)]
    packed=0;position=0
    for k in range(8):
        for row in deltas:
            packed|=(row[k]-minima[k])<<position;position+=widths[k]
    blob=packed.to_bytes((position+7)//8,'little')
    metadata=[[i*64,(i+1)*64,i*32,(i+1)*32,17,7,120000,160000] for i in range(132)]
    cuts=[v for row in metadata for v in row[:4]]
    tr=dict(m=m,seed=3101,pad_mib=0,origin_ns=origin,grid=132,
            clock_bitpacked_le_b64=base64.b64encode(blob).decode(),clock_minima=minima,clock_widths=widths,clock_order=order,
            metadata_header=metadata[0][4:],cuts_u16_le_b64=base64.b64encode(struct.pack('<528H',*cuts)).decode(),
            jobs=[[i,r.randrange(1,65)] for i in range(24)],clock_columns=columns)
    decoded=decode_trace(tr);assert decoded['clocks_ns']==clocks and decoded['metadata']==metadata
    wire['traces'].append(tr)
payload=base64.b64encode(zlib.compress(json.dumps(wire,separators=(',',':')).encode(),9))
assert json.loads(zlib.decompress(base64.b64decode(payload)))==wire
assert len(payload)<12000,len(payload)
out=dict(status='pass',scope='CPU generated stress fixtures, not H200 performance',
         rows=24,traces=2,ctas_per_trace=132,clock_values=2112,
         payload_chars=len(payload),payload_plus_1000_tail_chars=len(payload)+1000,
         exact_integer_times=True,exact_fp32_error_metrics=True,exact_fp64_ratios=True,
         exact_u64_clock_roundtrip=True,exact_metadata_roundtrip=True,
         source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
target=ROOT/'muse_moe_w4a16/PHASE_TRACE_CODEC_CPU_2026-10-31.json'
assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
