"""Repair the truncated diagnostic without changing its GPU work or dispatch."""
import ast
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('part', choices=['a','b'])
args = parser.parse_args()
parent = ROOT / "muse_prefill_layer/submission_v24_public_lt_probe.py"
version = 25 if args.part=='a' else 26
target = ROOT / f"muse_prefill_layer/submission_v{version}_public_lt_binary_part_{args.part}.py"
source = parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "3e6606ec33a14142378293d8d80ebb5d3f2620fe85e5f738350897d9d6ae1f9e"
assert not target.exists()
helper = r'''
import struct as _LPStruct

def _lp_pack_report(report):
    packed = bytearray()
    meta = []
    summary = {}
    for row in report['rows']:
        times = [t for key in ('a0','a1','candidate') for t in row['ns'][key]]
        metrics = [row['fp64_errors'][key] for key in ('a0','a1','candidate')]
        metrics += [row['agreement']]
        metrics += [row['fresh_errors'][key] for key in ('a0','a1','candidate')]
        metrics += [row['fresh_agreement'],row['parent_change']]
        # These are GPU FP32 scalar results: binary32 is exact, not rounding.
        assert all(_LPStruct.unpack('<f',_LPStruct.pack('<f',v))[0] == v for v in metrics)
        packed += _LPStruct.pack('<27I9f2d',*times,*metrics,row['ratio'],row['aa'])
        meta.append({k:row[k] for k in ('T','name','group','pad_mib','seed')})
        key = str(row['T'])+'/'+row['name']
        item = summary.setdefault(key,dict(n=0,min=1e99,max=0.,max_aa_offset=0.,max_fp64=0.,max_agreement=0.))
        item['n'] += 1
        item['min'] = min(item['min'],row['ratio'])
        item['max'] = max(item['max'],row['ratio'])
        item['max_aa_offset'] = max(item['max_aa_offset'],max(row['aa'],1/row['aa'])-1)
        item['max_fp64'] = max(item['max_fp64'],*row['fp64_errors'].values(),*row['fresh_errors'].values())
        item['max_agreement'] = max(item['max_agreement'],row['agreement'],row['fresh_agreement'])
    out = {k:v for k,v in report.items() if k!='rows'}
    out.update(binary_rows_format='<27I9f2d',binary_rows_meta=meta,
               binary_rows_b64=_LPBase64.b64encode(packed).decode())
    quick = dict(status=report['status'],stage=report['stage'],rows=len(meta),summary=summary)
    return out,quick

'''
needle = 'def _lp_probe(st):'
assert source.count(needle)==1
source = source.replace(needle,helper+needle)
part_selection = "\n_LP_PINS = dict(list(_LP_PINS.items())[" + (":5" if args.part=='a' else "5:") + "])\n"
source = source.replace(needle,part_selection+needle)
old = "    payload = _LPBase64.b64encode(_LPZlib.compress(_LPJSON.dumps(report,separators=(',',':')).encode(),9)).decode()\n    print('PREFILL_PUBLIC_LT_Z '+payload,flush=True)"
new = "    packed,quick = _lp_pack_report(report)\n    payload = _LPBase64.b64encode(_LPZlib.compress(_LPJSON.dumps(packed,separators=(',',':')).encode(),9)).decode()\n    quick['payload_chars'] = len(payload)\n    quick['raw_fits_12000'] = len(payload) <= 12000\n    print('PREFILL_PUBLIC_LT_Z '+payload,flush=True)\n    print('PREFILL_PUBLIC_LT_SUMMARY '+_LPJSON.dumps(quick,separators=(',',':')),flush=True)"
assert source.count(old)==1
source = source.replace(old,new)
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT),hashlib.sha256(source.encode()).hexdigest())
