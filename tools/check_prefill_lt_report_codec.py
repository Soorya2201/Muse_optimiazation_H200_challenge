"""CPU-only report sizing and exact round-trip audit before another GPU run."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import random
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / 'muse_prefill_layer/submission_v25_public_lt_binary_part_a.py'
source = source_path.read_text()
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_lp_pack_report')
ns = {'_LPStruct':struct,'_LPBase64':base64}
exec(compile(ast.Module([fn],[]),'<isolated own report codec>','exec'),ns)
round_f32 = lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
results = []
for count in (40,80):
    rng = random.Random(3503)
    rows = []
    for index in range(count):
        cell, group, pad_index = index//8,(index//4)%2,index%4
        times = {k:[32*rng.randrange(700,60000) for _ in range(9)]
                 for k in ('a0','a1','candidate')}
        errors = {k:round_f32(rng.random()*.001) for k in ('a0','a1','candidate')}
        rows.append(dict(T=(753,3012,16384)[cell%3],
            name=('qkv','o','router','pooler','unpooler')[cell%5],group=group,
            pad_mib=(0,1,3,7)[pad_index],seed=3503,ns=times,
            fp64_errors=errors,agreement=round_f32(rng.random()*.001),fresh_errors=errors,
            fresh_agreement=round_f32(rng.random()*.001),parent_change=round_f32(rng.random()+1),
            ratio=round(rng.random()+.5,6),aa=round(rng.random()*.01+.995,6)))
    packed,quick = ns['_lp_pack_report'](dict(status='complete',stage='finished',rows=rows,plans=[]))
    payload = base64.b64encode(zlib.compress(json.dumps(packed,separators=(',',':')).encode(),9))
    summary = json.dumps(quick,separators=(',',':'))
    raw = base64.b64decode(packed['binary_rows_b64'])
    stride = struct.calcsize('<27I9f2d')
    assert len(raw)==count*stride
    for index,row in enumerate(rows):
        v = struct.unpack_from('<27I9f2d',raw,index*stride)
        assert list(v[:27])==sum([row['ns'][k] for k in ('a0','a1','candidate')],[])
        expected_metrics = list(row['fp64_errors'].values())+[row['agreement']]
        expected_metrics += list(row['fresh_errors'].values())+[row['fresh_agreement'],row['parent_change']]
        expected_metrics += [row['ratio'],row['aa']]
        assert list(v[27:])==expected_metrics
    results.append(dict(rows=count,payload_chars=len(payload),summary_chars=len(summary),
        below_12000_with_summary=len(payload)+len(summary)<12000,all_values_exact=True))
assert results[0]['below_12000_with_summary']
assert not results[1]['below_12000_with_summary']
report = dict(scope='Synthetic CPU report-codec stress fixture, not GPU performance or an absolute compression-size bound.',
    source=str(source_path.relative_to(ROOT)),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    seed=3503,cases=results,decision='Split ten candidate signatures into two five-signature runs, keeping both seeds and all four pads per candidate.')
out = ROOT / 'muse_prefill_layer/LT_REPORT_CODEC_2026-10-31.json'
assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
