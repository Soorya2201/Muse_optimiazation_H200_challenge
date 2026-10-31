"""Decode all raw WGS timing groups and enumerate the eight changed policies."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zlib

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
args = parser.parse_args()
receipt_path = ROOT / 'muse_decode_layer/run_receipts' / (args.job+'.json')
receipt = json.loads(receipt_path.read_text())
output = ROOT / 'muse_decode_layer' / ('WGS_REGISTER_SPLIT_'+args.job+'.json')
assert not output.exists(), output
log = receipt['log_tail']
matches = re.findall(r'^DECODER_WGS_REGS_Z (\S+)$',log,re.MULTILINE)
assert len(matches)==1, 'Missing or clipped report; do not reconstruct lost measurements'
payload = matches[0]
report = json.loads(zlib.decompress(base64.b64decode(payload)))
summaries = re.findall(r'^DECODER_WGS_REGS_SUMMARY (.+)$',log,re.MULTILINE)
assert len(summaries)==1
summary = json.loads(summaries[0])
assert summary['status']==report['status'] and summary['rows']==len(report['rows'])
assert summary['chars']==len(payload) and summary['nearby']==len(report['nearby'])
analysis = dict(job=args.job,cluster=receipt['cluster'],payload_chars=len(payload),report=report,summary=summary,
    scope='Independent synthetic, cold-L2 graph timings. Timed whole decoder is the incumbent. Phase-subtraction models are optimistic estimates; synthetic routing and PDL critical paths can differ.')
if report['status']=='complete':
    assert len(report['rows'])==16 and len(report['nearby'])==3
    grouped = {}
    for row in report['rows']:
        ns = row['ns']
        assert set(ns)=={'a0','a1','gs2','gs4'}
        assert all(len(v)==9 and all(isinstance(t,int) and t>0 for t in v) for v in ns.values())
        for arm in ('gs2','gs4'):
            r = statistics.median(math.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns[arm]))
            assert math.isclose(r,row['speedup'][arm],rel_tol=1e-12)
            grouped.setdefault((row['m'],arm),[]).append(r)
        for field in ('error','fresh'):
            assert all(e[0]<=.002 and e[1]<=.01 for e in row[field].values())
        assert row['change']>.001
    cells = []
    for (m,arm),values in sorted(grouped.items()):
        assert len(values)==8
        matching = [r for r in report['rows'] if r['m']==m]
        assert {(r['seed'],r['pad_mib']) for r in matching}==set(itertools.product((3108,7909),(0,1,3,7)))
        cells.append(dict(m=m,arm=arm,min=min(values),median=statistics.median(values),max=max(values),qualified=min(values)>1.02))
    analysis['cells'] = cells
    analysis['max_aa_fraction'] = max(abs(r['aa']-1) for r in report['rows'])
    analysis['max_error'] = {field:[max(r[field][a][i] for r in report['rows'] for a in r[field]) for i in (0,1)] for field in ('error','fresh')}
    profiles = {json.dumps(p['case'],sort_keys=True):p['report'] for p in receipt['profiles']}
    loads = {}
    for b in receipt['benchmarks']:
        case = b['case']; m = case['batch']*case.get('q_len',1)
        profile = profiles[json.dumps(case,sort_keys=True)]
        expert_us = sum(float(x) for x in re.findall(r'^.*okc::moe_fused.*?\s+([\d.]+) µs\s',profile,re.MULTILINE))
        loads.setdefault(case['load'],[]).append(dict(m=m,weight=case['weight'],parent_us=b['median_ns']/1000,expert_us=expert_us))
    assert len(loads)==6
    policies = []
    stats = {(r['m'],r['arm']):r for r in cells}
    for arms in itertools.product(('parent','gs2','gs4'),repeat=2):
        if arms==('parent','parent'):continue
        policy=dict(zip((128,256),arms)); factors={}
        for field in ('min','median','max'):
            ratios=[]
            for rows in loads.values():
                total=sum(r['parent_us']*r['weight'] for r in rows)
                changed=total
                for r in rows:
                    arm=policy.get(r['m'],'parent')
                    if arm!='parent':changed-=r['expert_us']*r['weight']*(1-1/stats[(r['m'],arm)][field])
                ratios.append(total/changed)
            factors[field]=math.prod(ratios)**(1/6)
        component_gate=all(arm=='parent' or stats[(m,arm)]['qualified'] for m,arm in policy.items())
        policies.append(dict(policy=policy,model=factors,component_gate=component_gate,
                             qualified=component_gate and factors['min']>=1.03))
    analysis['policies']=policies
    analysis['qualified_policies']=sum(p['qualified'] for p in policies)
    analysis['model_source']='Parent benchmark/profile rows from this same diagnostic job; M32 unchanged. Raw event latencies include graph overhead; no guessed subtraction.'
output.write_text(json.dumps(analysis,indent=2)+'\n')
print(output.relative_to(ROOT))
print(json.dumps({k:v for k,v in analysis.items() if k in ('job','cluster','payload_chars','summary','cells','qualified_policies','max_aa_fraction','max_error')},indent=2))
