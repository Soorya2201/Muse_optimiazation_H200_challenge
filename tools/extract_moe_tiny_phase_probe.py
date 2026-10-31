"""Audit independent TINY controls and retain all coarse phase observations."""
import argparse
import itertools
import json
import math
from pathlib import Path
import re
import statistics as stats
from moe_tiny_phase_codec import decode_report

p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');args=p.parse_args()
root=Path(__file__).resolve().parents[1];folder=root/'muse_moe_w4a16'
receipt=json.loads((folder/'run_receipts'/f'{args.job}.json').read_text())
payloads=re.findall(r'^MOE_TINY_PHASE_Z (\S+)$',receipt['log_tail'],re.M)
assert len(payloads)==1,'Require one intact report; do not restart a job for missing telemetry'
d=decode_report(payloads[0])
out=dict(job=args.job,cluster=receipt['cluster'],source_sha256=receipt['source_sha256'],
    scope='Independent synthetic tiny-kernel attribution only. Timed challenge dispatch unchanged. Counterfactual subtractions are not critical-path bounds or measured speedups.',
    payload_chars=len(payloads[0]),diagnostic=d,summary_by_local_routes=[],trace_analysis=[],attribution_gate=False)
for row in d['rows']:
    a,b,t,z=(row['ns'][name] for name in ('a0','old','trace','a1'))
    assert all(len(v)==8 and all(x>0 for x in v) for v in (a,b,t,z))
    row['ratios']=dict(parent_old=stats.median(math.sqrt(x*y)/v for x,y,v in zip(a,z,b)),
        old_trace=stats.median(v/w for v,w in zip(b,t)),
        aa=stats.median(x/y for x,y in zip(a,z)))
    row['control_event_ns']=stats.median(math.sqrt(x*y) for x,y in zip(a,z))

if d['status']=='complete':
    assert receipt['summary']['passed']
    assert {(r['seed'],r['pad_mib'],r['nlocal']) for r in d['rows']}==set(itertools.product((3101,7907),(0,1,3,7),(0,1,2,4,8)))
    assert len(d['nearby'])==3 and len(d['traces'])==2
    resources=d['resources'];local_gate=True
    for key in ('old','trace'):
        a=resources[key];assert a[0]==a[5]==0
        local_gate &= a[2]==a[7]==0
    for row in d['rows']+d['nearby']:
        assert row['change']>.001
        for field in ('error','fresh'):
            for values in row[field].values():
                assert all(math.isfinite(v) for v in values) and values[0]<=.002 and values[1]<=.01
    for nlocal in (0,1,2,4,8):
        rows=[r for r in d['rows'] if r['nlocal']==nlocal];s=dict(nlocal=nlocal,groups=len(rows))
        for key in ('parent_old','old_trace','aa'):
            v=[r['ratios'][key] for r in rows]
            s[key]=dict(min=min(v),median=stats.median(v),max=max(v),max_offset=max(max(q,1/q)-1 for q in v))
        s['control_event_us_median']=stats.median(r['control_event_ns'] for r in rows)/1000
        s['attribution_gate']=bool(local_gate and s['parent_old']['max_offset']<.02
            and s['aa']['max_offset']<.02 and s['old_trace']['min']>=.95 and s['old_trace']['max']<=1.05)
        out['summary_by_local_routes'].append(s)
    out['attribution_gate']=all(s['attribution_gate'] for s in out['summary_by_local_routes'])
    names=('startup_to_route','phase_a','phase_b','grid_barrier_and_helper_join','final')
    for tr in d['traces']:
        clocks=tr['clocks_ns'];assert len(clocks)==tr['grid']
        assert all(len(c)==6 and c==sorted(c) and c[0]>0 for c in clocks)
        origin=min(c[0] for c in clocks);span=max(c[-1] for c in clocks)-origin
        stats_by_phase={}
        for k,name in enumerate(names):
            values=[c[k+1]-c[k] for c in clocks]
            stats_by_phase[name]=dict(min_ns=min(values),median_ns=stats.median(values),max_ns=max(values))
        row=next(r for r in d['rows'] if (r['seed'],r['pad_mib'],r['nlocal'])==(tr['seed'],tr['pad_mib'],tr['nlocal']))
        models=[]
        for mask in range(1,32):
            selected=[k for k in range(5) if mask>>k&1]
            # Hold starts, other durations and dependency ordering fixed. This
            # deliberately simplified deletion model is not a runtime forecast.
            remaining=max(c[-1]-origin-sum(c[k+1]-c[k] for k in selected) for c in clocks)
            saved=span-remaining;after=row['control_event_ns']-saved
            models.append(dict(phases=[names[k] for k in selected],observed_span_ns=span,
                counterfactual_span_ns=remaining,assumed_saving_ns=saved,
                one_load_other_five_unchanged_proxy=(row['control_event_ns']/after)**(1/6) if after>0 else None,
                attribution_accepted=out['attribution_gate']))
        out['trace_analysis'].append(dict(seed=tr['seed'],nlocal=tr['nlocal'],span_ns=span,
            control_event_ns=row['control_event_ns'],phase_intervals=stats_by_phase,conditional_deletion_models=models))
path=folder/f'TINY_PHASE_REPORT_{args.job}.json';assert not path.exists()
path.write_text(json.dumps(out,indent=2)+'\n')
print('DIAGNOSTIC',d['status'],d.get('stage'),d.get('seconds'),'payload',out['payload_chars'])
print('RESOURCES',d.get('resources'),'ATTRIBUTION_GATE',out['attribution_gate'])
if d['status']!='complete':print('ERROR',d.get('error'),d.get('trace'))
for s in out['summary_by_local_routes']:print(json.dumps(s))
for s in out['trace_analysis']:
    print('TRACE',s['seed'],s['span_ns'],json.dumps(s['phase_intervals']))
    print('SINGLE_DELETIONS',[(x['phases'],x['one_load_other_five_unchanged_proxy']) for x in s['conditional_deletion_models'] if len(x['phases'])==1])
print(path.relative_to(root))
