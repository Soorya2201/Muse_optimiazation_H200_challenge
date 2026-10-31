"""Check sparse frame fidelity and the service-tail budget before H200."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import random
import struct
import zlib
from moe_flush_trace_codec import trace
from moe_unit_trace_codec import matrix,rows

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'muse_moe_w4a16/submission_v30_flush_trace_probe.py';tree=ast.parse(p.read_text())
env={'_MP64':base64,'_MPStruct':struct}
for name in ['_ut_pack','_mp_pack_rows']:
    n=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    exec(compile(ast.Module(body=[n],type_ignores=[]),'own-flush-codec','exec'),env)
# Preserve the real row magnitudes and wire schema from the previous run;
# these are telemetry tests, not new GPU observations.
previous=json.loads((ROOT/'muse_moe_w4a16/UNIT_TRACE_e038ff14796944dc.json').read_text())
records=previous['rows'];wire=dict(status='complete',stage='complete',row_count=24,
 row_format='<24I12f4d',rows_b64=env['_mp_pack_rows'](records),resources=[0,96,0,576,231168]*2,
 baseline_resources=[0,96,0,576,231168]*2,traces=[],seconds=99.)
assert rows(wire)==records
r=random.Random(103130)
for m in (32,256):
    origin=1_793_434_766_633_418_496
    phase=[[r.randrange(300001) for _ in range(8)] for _ in range(4)]
    fs=[];expected=[dict(cta=c,frames=[]) for c in (0,1,66,67)]
    for ci in range(4):
        for pb in (0,1):
            for u in range(12):
                start=r.randrange(300001);end=start+r.randrange(1,6000)
                kind=r.randrange(1,4);nst=r.randrange(17)
                fs.append([ci,pb,u,kind,nst,start,end])
                expected[ci]['frames'].append([pb,u,kind,nst,origin+start,origin+end])
    template=previous['wire']['traces'][m==256]
    tr={k:template[k] for k in ['grid','metadata_header','cuts_u16_le_b64','jobs']}
    tr.update(kind='sampled_flush_trace_v1',origin_ns=origin,ctas=[0,1,66,67],
       phase_clocks=env['_ut_pack'](phase),frames=env['_ut_pack'](fs),
       frame_fields=['sample','phase','unit','expected_kind','expected_nst','start','end'],m=m,seed=3101,pad_mib=0)
    assert matrix(tr['frames'])==fs
    d=trace(tr);assert d['groups']==expected
    assert d['phase_clocks_ns']==[[origin+x for x in c] for c in phase]
    wire['traces'].append(tr)
for test in [[[0,1,2**32-1]]*3,[[0]*7 for _ in range(4)],[[2**32-1]*8]]:
    assert matrix(env['_ut_pack'](test))==test
payload=base64.b64encode(zlib.compress(json.dumps(wire,separators=(',',':')).encode(),9))
assert json.loads(zlib.decompress(base64.b64decode(payload)))==wire
assert len(payload)+1600<16000
out=dict(status='pass',scope='CPU synthetic frames and existing V29 timing-row roundtrip, no new GPU data',
    rows=24,traces=2,frames_per_trace=96,payload_chars=len(payload),payload_plus_1600_tail_chars=len(payload)+1600,
    exact_rows=True,exact_uint64_clocks=True,zero_width_u32_boundaries=True,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
target=ROOT/'muse_moe_w4a16/FLUSH_TRACE_CODEC_CPU_2026-10-31.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
