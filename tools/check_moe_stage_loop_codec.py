"""Check candidate report retention/fidelity before spending an OKC check."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib
from moe_unit_trace_codec import rows

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'muse_moe_w4a16/submission_v31_stage_loop_probe.py';tree=ast.parse(p.read_text())
n=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_pack_rows'][-1]
env={'_MP64':base64,'_MPStruct':struct}
exec(compile(ast.Module(body=[n],type_ignores=[]),'own-stage-codec','exec'),env)
# Existing rows supply realistic magnitudes only; no candidate GPU evidence.
records=json.loads((ROOT/'muse_moe_w4a16/FLUSH_TRACE_b22622ebb61946df.json').read_text())['rows']
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
for i,r in enumerate(records):
    for field in ['error','fresh']:
        for name in r[field]:r[field][name]=[f32(.0001*(1+i%5)),f32(.001*(1+i%7))]
nearby=[dict(m=m,seed=seed,error=records[0]['error'],fresh=records[0]['fresh'],change=1.25)
        for m,seed in [(40,7823),(96,9533),(192,11239)]]
wire=dict(status='complete',stage='complete',rows_b64=env['_mp_pack_rows'](records),row_count=24,
    row_format='<24I12f4d',nearby=nearby,resources=[0,96,0,640,231168]*2,
    control_resources=[0,96,0,640,231168]*2,seconds=123.25,
    arm_roles=dict(a0='canonical parent',a1='canonical parent repeat',trace='REG112 + zero-MMA uniform stage loop',canonical='REG112 original stage loop'),
    timed_dispatch='unchanged canonical parent')
assert rows(wire)==records
payload=base64.b64encode(zlib.compress(json.dumps(wire,separators=(',',':')).encode(),9))
assert json.loads(zlib.decompress(base64.b64decode(payload)))==wire
assert len(payload)+1600<16000
out=dict(status='pass',scope='CPU telemetry stress only, no candidate GPU evidence',rows=24,nearby=3,
    payload_chars=len(payload),payload_plus_1600_tail_chars=len(payload)+1600,
    exact_uint32_timings=True,exact_fp32_metrics=True,exact_fp64_ratios=True,
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
target=ROOT/'muse_moe_w4a16/STAGE_LOOP_CODEC_CPU_2026-10-31.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
