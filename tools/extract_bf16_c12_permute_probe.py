"""Decode complete C12-permutation evidence and screen all fixed policies."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import zlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'muse_bf16_proj'
receipt = json.loads((folder / 'run_receipts' / (args.job+'.json')).read_text())
markers=('BF16_C12_PERMUTE_Z ','BF16_C12_DEVICE_Z ','BF16_C12_NORM_Z ')
lines=[s for s in receipt['log_tail'].splitlines() if s.startswith(markers)]
payloads=[s.split(' ',1)[1] for s in lines]
assert len(payloads)==1, 'Require one complete final diagnostic, never treat missing logs as a pass'
wire=zlib.decompress(base64.b64decode(payloads[0]))
if lines[0].startswith(markers[1:]):
    size=struct.unpack_from('<I',wire)[0];d=json.loads(wire[4:4+size]);buf=wire[4+size:]
    count=len(d['cases'])*4*d['rounds'];width=d.get('time_word_bytes',4)
    assert width in(2,4) and len(buf)==count*width
    times=iter(struct.unpack('<'+str(count)+('H'if width==2 else'I'),buf))
    for row in d['cases']:row['ns']=[[next(times) for _ in range(d['rounds'])] for _ in range(4)]
else:d=json.loads(wire)
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
            t=row['ns'];assert len(t)==4 and all(len(v)==d.get('rounds',5) for v in t)
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
label='GLOBAL_NORMALIZATION_REPORT_' if lines[0].startswith(markers[2]) else 'C12_PERMUTE_REPORT_'
target=folder/(label+args.job+'.json')
assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n')
print('Diagnostic',d['status'],'elapsed',d.get('elapsed_s'),'payload',out['payload_chars'],'error',d.get('error'))
for row in out['summaries']:print(json.dumps(row))
if out['policies']:
    print('Best worst model:',max(out['policies'],key=lambda r:r['worst']))
    print('Eligible policies:',sum(r['qualifies'] for r in out['policies']))
print(target.relative_to(ROOT))
