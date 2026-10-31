"""Same-job prefill expert-duration opportunity model, never a gain forecast."""
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'muse_prefill_layer/run_receipts/0848e5e1b2ba40e5.json'
d=json.loads(receipt.read_text());profiles={p['index']:p for p in d['profiles']}
rows=[]
for i,b in enumerate(d['benchmarks']):
    report=profiles[i]['report']
    matches=[float(m.group(1)) for line in report.splitlines() if 'moe_fused_persistent' in line
             for m in [re.search(r'\s([\d.]+) µs\s',line)] if m]
    assert len(matches)==1
    rows.append(dict(case=b['case'],benchmark_us=b['median_ns']/1000,
                     expert_profile_us=matches[0]))

def model(cut):
    grouped={}
    for r in rows:
        c=r['case'];a=r['benchmark_us'];b=a-r['expert_profile_us']*cut
        assert b>0
        g=grouped.setdefault(c['load'],[0,0]);g[0]+=a*c['weight'];g[1]+=b*c['weight']
    return math.prod(a/b for a,b in grouped.values())**(1/len(grouped))
lo,hi=0.,1.
for _ in range(60):
    mid=(lo+hi)/2
    if model(mid)<1.03:lo=mid
    else:hi=mid
out=dict(scope='Optimistic same-job expert-profile-duration subtraction. Durations contain PDL waits/overlap; not critical-path isolated costs or a practical KPERM speedup prediction.',
         job=d['job'],cluster=d['cluster'],timed_path='Unchanged canonical v20 parent',rows=rows,
         model_by_fraction_removed={str(x):model(x) for x in [.05,.1,.15,.2,1.]},
         expert_duration_fraction_for_3pct_board_proxy=hi,
         candidate_status='Encoding proof only; timed X packing and GEMM1 staging required before GPU candidate. No observed runtime reduction.',
         caution='Do not equate six fewer scale-construction instructions with this fraction of elapsed expert time. No physical bandwidth or traffic saving inferred.')
(ROOT/'muse_prefill_layer/SCALE_PERMUTATION_SCOPE_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['model_by_fraction_removed','expert_duration_fraction_for_3pct_board_proxy']}))
