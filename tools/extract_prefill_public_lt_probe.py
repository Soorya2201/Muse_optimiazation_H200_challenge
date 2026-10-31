"""Audit raw Lt comparisons and bound fixed combinations on the same job."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("job")
args = parser.parse_args()
track = ROOT / "muse_prefill_layer"
receipt = json.loads((track / "run_receipts" / (args.job+".json")).read_text())
matches = re.findall(r"PREFILL_PUBLIC_LT_Z ([A-Za-z0-9+/=]+)", receipt.get("log_tail", ""))
assert len(matches)==1, "missing/incomplete final report; inspect the same job's logs, do not restart"
report = json.loads(zlib.decompress(base64.b64decode(matches[0])))
if 'binary_rows_format' in report:
    assert report['binary_rows_format']=='<27I9f2d'
    raw = base64.b64decode(report.pop('binary_rows_b64'))
    meta = report.pop('binary_rows_meta')
    stride = struct.calcsize(report['binary_rows_format'])
    assert len(raw)==len(meta)*stride
    rows = []
    for i,row in enumerate(meta):
        values = struct.unpack_from(report['binary_rows_format'],raw,i*stride)
        row['ns'] = {k:list(values[j*9:(j+1)*9]) for j,k in enumerate(('a0','a1','candidate'))}
        row['fp64_errors'] = dict(zip(('a0','a1','candidate'),values[27:30]))
        row['agreement'] = values[30]
        row['fresh_errors'] = dict(zip(('a0','a1','candidate'),values[31:34]))
        row['fresh_agreement'],row['parent_change'],row['ratio'],row['aa'] = values[34:38]
        rows.append(row)
    report['rows'] = rows
    summaries = re.findall(r'PREFILL_PUBLIC_LT_SUMMARY (.+)',receipt.get('log_tail',''))
    assert len(summaries)==1, 'missing independent capture summary'
    quick = json.loads(summaries[0])
    assert quick['status']==report['status'] and quick['rows']==len(rows)
    assert quick['payload_chars']==len(matches[0]) and quick['raw_fits_12000']
    for key,item in quick['summary'].items():
        subset = [r for r in rows if str(r['T'])+'/'+r['name']==key]
        assert item['n']==len(subset)
        assert item['min']==min(r['ratio'] for r in subset)
        assert item['max']==max(r['ratio'] for r in subset)
print("DIAGNOSTIC",report["status"],report.get("stage"),report.get("elapsed_s"))
print("PLANS",json.dumps(report.get("plans"),indent=2))
if report["status"]!="complete":
    print("ERROR",report.get("error"),report.get("trace"))
groups = {}
summary = {}
for row in report["rows"]:
    key = (row["T"],row["name"])
    group = (row["group"],row["pad_mib"])
    assert (key,group) not in groups
    groups[key,group] = row
    for times in row["ns"].values():
        assert len(times)==9 and all(isinstance(v,int) and v>0 for v in times)
    a,b,c = (row["ns"][k] for k in ("a0","a1","candidate"))
    ratio = statistics.median(math.sqrt(x*y)/z for x,y,z in zip(a,b,c))
    assert abs(ratio-row["ratio"])<.0000006
    assert row["parent_change"]>.001
    tol = .00005 if row["name"]=="router" else .005
    assert max(*row["fp64_errors"].values(),*row["fresh_errors"].values(),
               row["agreement"],row["fresh_agreement"])<tol
    row["saved_us"] = statistics.median((math.sqrt(x*y)-z)/1000 for x,y,z in zip(a,b,c))

keys = sorted({(r["T"],r["name"]) for r in report["rows"]})
expected = set(itertools.product(range(2),(0,1,3,7)))
for key in keys:
    rows = [r for (k,g),r in groups.items() if k==key]
    if report["status"]=="complete":
        assert {g for k,g in groups if k==key}==expected
    ratios = [r["ratio"] for r in rows]
    offsets = [max(r["aa"],1/r["aa"])-1 for r in rows]
    summary[str(key)] = dict(n=len(rows),min=min(ratios),median=statistics.median(ratios),max=max(ratios),
        min_saved_us=min(r["saved_us"] for r in rows),max_saved_us=max(r["saved_us"] for r in rows),
        max_aa_offset=max(offsets),component_gate=bool(len(rows)==8 and min(ratios)>1.02 and max(offsets)<.01))

models = []
if report["status"]=="complete":
    selected = {(p['T'],p['name']) for p in report['plans'] if p['index']>=0 and not p['same_as_parent']}
    assert set(keys)==selected
    for enabled in itertools.product((False,True),repeat=len(keys)):
        chosen = [k for k,on in zip(keys,enabled) if on]
        factors = []
        component_ok = all(summary[str(k)]["component_gate"] for k in chosen)
        for group in sorted(expected):
            loads = {}
            for b in receipt["benchmarks"]:
                c = b["case"]
                T = c["batch"]*c["q_len"]
                a = b["median_ns"]/1000
                saving = sum(groups[k,group]["saved_us"] for k in chosen if k[0]==T)
                old,new = loads.get(c["load"],(0.,0.))
                loads[c["load"]] = old+c["weight"]*a,new+c["weight"]*max(a-saving,.001)
            factors.append(math.exp(statistics.mean(math.log(a/b) for a,b in loads.values())))
        models.append(dict(chosen=chosen,worst=min(factors),median=statistics.median(factors),best=max(factors),
                           qualified=bool(chosen and component_ok and min(factors)>=1.03)))

output = dict(job=args.job,cluster=receipt.get("cluster"),diagnostic=report,paired_summary=summary,
    model_scope="Optimistic sum of isolated dense graph savings on same-job weighted cells; router/pooler concurrency and cross-component interference unmeasured. Not a full-layer improvement. Fixture pads leave checkpoint-weight addresses fixed.",
    fixed_combinations=models)
path = track/("PUBLIC_LT_REPORT_"+args.job+".json")
assert not path.exists()
path.write_text(json.dumps(output,indent=2)+"\n")
print("SUMMARY",json.dumps(summary,indent=2))
print("QUALIFIED",[m for m in models if m['qualified']])
if len(models)>1:
    print("BEST_NONEMPTY",max((m for m in models if m['chosen']),key=lambda m:m['worst']))
print(path.relative_to(ROOT))
