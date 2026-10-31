"""Archive candidate/control rows and conservatively model all seven policies."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zlib
from moe_unit_trace_codec import rows

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
r=json.loads((ROOT/'muse_moe_w4a16/run_receipts'/f'{args.job}.json').read_text())
target=ROOT/'muse_moe_w4a16'/f'STAGE_LOOP_{args.job}.json';assert not target.exists()
payloads=re.findall(r'^MOE_STAGE_LOOP_Z (\S+)$',r['log_tail'],re.MULTILINE)
assert len(payloads)==1,'Retain capture status UNKNOWN; do not reconstruct missing observations'
wire=json.loads(zlib.decompress(base64.b64decode(payloads[0])))
summary=json.loads(re.findall(r'^MOE_STAGE_LOOP_SUMMARY (.+)$',r['log_tail'],re.MULTILINE)[0])
assert summary['chars']==len(payloads[0]) and summary['rows']==wire['row_count']
assert summary['status']==wire['status'] and summary['nearby']==len(wire['nearby'])
records=rows(wire)
out=dict(job=args.job,cluster=r['cluster'],summary=summary,wire=wire,rows=records,
 scope='Synthetic independent four-arm cold-L2 comparisons. Timed dispatch is unchanged canonical. Not an optimized whole-run score.')
if wire['status']=='complete':
    assert len(records)==24 and len(wire['nearby'])==3
    assert {(x['m'],x['seed'],x['pad_mib']) for x in records}==set(itertools.product((32,128,256),(3101,7907),(0,1,3,7)))
    for x in records:
        ns=x['ns'];assert all(len(v)==5 and min(v)>0 for v in ns.values())
        ratio=statistics.median(math.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace']))
        assert math.isclose(ratio,x['ratio'],rel_tol=1e-12)
        control=statistics.median(c/math.sqrt(a*b) for a,b,c in zip(ns['a0'],ns['a1'],ns['canonical']))
        assert math.isclose(control,x['baseline_speedup_vs_canonical'],rel_tol=1e-12)
        assert math.isclose(statistics.median(a/b for a,b in zip(ns['a0'],ns['a1'])),x['aa'],rel_tol=1e-12)
        x['register_control_time_vs_parent']=x['baseline_speedup_vs_canonical']
        x['same_register_loop_speedup']=statistics.median(a/b for a,b in zip(ns['canonical'],ns['trace']))
        assert x['change']>.001
    out['max_aa_fraction']=max(abs(x['aa']-1) for x in records)
    out['aa_gate']=out['max_aa_fraction']<=.02
    out['numerical_gate']=all(math.isfinite(v[0]) and math.isfinite(v[1]) and v[0]<=.002 and v[1]<=.01
        for x in records+wire['nearby'] for field in ['error','fresh'] for v in x[field].values())
    out['resource_gate']=all(wire[k][0]==wire[k][5]==0 and wire[k][2]==wire[k][7]==0 for k in ['resources','control_resources'])
    out['max_error']={field:max(v for x in records+wire['nearby'] for pair in x[field].values() for v in pair) for field in ['error','fresh']}
    out['ratios']=[dict(m=m,min=min(x['ratio'] for x in records if x['m']==m),
       median=statistics.median(x['ratio'] for x in records if x['m']==m),max=max(x['ratio'] for x in records if x['m']==m),
       same_register_min=min(x['same_register_loop_speedup'] for x in records if x['m']==m),
       same_register_max=max(x['same_register_loop_speedup'] for x in records if x['m']==m)) for m in (32,128,256)]
    policies=[]
    for mask in range(1,8):
        included=[m for i,m in enumerate((32,128,256)) if mask&(1<<i)]
        gains=[]
        for seed,pad in itertools.product((3101,7907),(0,1,3,7)):
            ratios=[next(x['ratio'] for x in records if (x['m'],x['seed'],x['pad_mib'])==(m,seed,pad)) for m in included]
            gains.append(math.prod(ratios)**(1/6))
        component_gate=all(x['ratio']>=1.02 and x['same_register_loop_speedup']>=1.02 for x in records if x['m'] in included)
        eligible=out['aa_gate'] and out['numerical_gate'] and out['resource_gate'] and component_gate and min(gains)>=1.03
        policies.append(dict(included=included,component_gate=component_gate,min_proxy=min(gains),median_proxy=statistics.median(gains),max_proxy=max(gains),eligible=eligible))
    out['policies']=policies;out['eligible_policies']=sum(p['eligible'] for p in policies)
    out['best_optimistic_policy']=max(policies,key=lambda p:p['median_proxy'])
out['limits']=['Five rounds/group and only two random seeds/four scratch pads; no claim of a universal bound.',
 'Model applies synthetic component ratios to six equally weighted loads; it is not an official score or joint changed-dispatch measurement.',
 'Resources are full native attributes; instruction counts from separate compiler-only kernels are not full native instruction counts.',
 'Instruction-cache counters were not captured; speed alone does not identify a cache mechanism.',
 'Whole challenge tests/timings use unchanged canonical; candidate nearby and fresh-value checks are independent prepare-time validations.']
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(ROOT))
print(json.dumps({k:v for k,v in out.items() if k not in ['wire','rows','limits']},indent=2))
