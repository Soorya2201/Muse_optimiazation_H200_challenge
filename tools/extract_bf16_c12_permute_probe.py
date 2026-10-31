"""Decode complete C12-permutation evidence and screen all fixed policies."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import statistics
import zlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'muse_bf16_proj'
receipt = json.loads((folder / 'run_receipts' / (args.job+'.json')).read_text())
payloads = [s.split('BF16_C12_PERMUTE_Z ',1)[1] for s in receipt['log_tail'].splitlines()
            if s.startswith('BF16_C12_PERMUTE_Z ')]
assert len(payloads)==1, 'Require one complete final diagnostic, never treat missing logs as a pass'
d = json.loads(zlib.decompress(base64.b64decode(payloads[0])))
out = dict(job=args.job,cluster=receipt['cluster'],source_sha256=receipt['source_sha256'],
    scope='Independent synthetic diagnostic; timed challenge dispatch remains parent. Policy models are not measured board gains.',
    payload_chars=len(payloads[0]),diagnostic=d,summaries=[],policies=[])
if d['status']=='complete':
    assert receipt['summary']['passed'] and len(d['nearby'])==5 and len(d['cases'])==72
    assert {(r['pad'],r['id'],r['seed']) for r in d['cases']} == set(itertools.product(range(4),range(9),(3119,7919)))
    for spec in d['specs']:
        rows=[r for r in d['cases'] if r['id']==spec['id']]
        ratios,matched,controls,aa=[],[],[],[]
        for row in rows:
            t=row['ns'];assert len(t)==4 and all(len(v)==5 for v in t)
            row['ratios']={
                'parent_new':statistics.median(math.sqrt(a*z)/c for a,_,c,z in zip(*t)),
                'old_new':statistics.median(b/c for _,b,c,_ in zip(*t)),
                'parent_old':statistics.median(math.sqrt(a*z)/b for a,b,_,z in zip(*t)),
                'aa':statistics.median(a/z for a,_,_,z in zip(*t))}
            q=row['ratios'];ratios.append(q['parent_new']);matched.append(q['old_new'])
            controls.append(abs(q['parent_old']-1));aa.append(abs(q['aa']-1))
        resource=d['resources'][str(spec['id'])]
        zero_local=all(x[1]==0 for x in resource)
        summary=dict(id=spec['id'],layer=spec['layer'],m=spec['m'],resources=resource,
            parent_new_min=min(ratios),parent_new_median=statistics.median(ratios),parent_new_max=max(ratios),
            old_new_min=min(matched),old_new_median=statistics.median(matched),old_new_max=max(matched),
            max_control_offset=max(controls),max_aa_offset=max(aa),zero_local=zero_local,
            all_old_new_bit_identical=all(r['identical'] for r in rows))
        summary['qualifies']=(zero_local and min(ratios)>1.02 and min(matched)>1.02
                              and max(controls)<.03 and max(aa)<.02)
        out['summaries'].append(summary)
    benches=receipt['benchmarks'];assert len(benches)==30
    loads=sorted({r['case']['load'] for r in benches})
    eligible={r['id'] for r in out['summaries'] if r['qualifies']}
    target={(r['layer'],r['m']):r['id'] for r in d['specs']}
    for mask in range(1,1<<9):
        chosen={i for i in range(9) if mask>>i&1};models=[]
        for pad,seed in itertools.product(range(4),(3119,7919)):
            factors={r['id']:r['ratios']['parent_new'] for r in d['cases'] if r['pad']==pad and r['seed']==seed}
            loadratios=[]
            for load in loads:
                rows=[r for r in benches if r['case']['load']==load]
                before=sum(r['median_ns']*r['case']['weight'] for r in rows);after=0
                for r in rows:
                    sid=target.get((r['case']['layer'],r['case']['m']))
                    after+=r['median_ns']*r['case']['weight']/(factors[sid] if sid in chosen else 1)
                loadratios.append(before/after)
            models.append(math.prod(loadratios)**(1/len(loadratios)))
        out['policies'].append(dict(ids=sorted(chosen),worst=min(models),median=statistics.median(models),
            best=max(models),component_gates=chosen<=eligible,qualifies=chosen<=eligible and min(models)>=1.03))
target=folder/('C12_PERMUTE_REPORT_'+args.job+'.json')
assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n')
print('Diagnostic',d['status'],'elapsed',d.get('elapsed_s'),'payload',out['payload_chars'],'error',d.get('error'))
for row in out['summaries']:print(json.dumps(row))
if out['policies']:
    print('Best worst model:',max(out['policies'],key=lambda r:r['worst']))
    print('Eligible policies:',sum(r['qualifies'] for r in out['policies']))
print(target.relative_to(ROOT))
