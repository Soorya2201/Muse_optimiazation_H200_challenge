"""Archive all flush observations and reject perturbed timing attribution."""
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
from moe_flush_trace_codec import trace

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
receipt=json.loads((ROOT/'muse_moe_w4a16/run_receipts'/f'{args.job}.json').read_text())
target=ROOT/'muse_moe_w4a16'/f'FLUSH_TRACE_{args.job}.json';assert not target.exists()
payloads=re.findall(r'^MOE_FLUSH_TRACE_Z (\S+)$',receipt['log_tail'],re.MULTILINE)
assert len(payloads)==1,'Complete report missing; preserve UNKNOWN instead of reconstructing observations'
wire=json.loads(zlib.decompress(base64.b64decode(payloads[0])))
summaries=re.findall(r'^MOE_FLUSH_TRACE_SUMMARY (.+)$',receipt['log_tail'],re.MULTILINE)
assert len(summaries)==1;summary=json.loads(summaries[0])
assert summary['status']==wire['status'] and summary['rows']==wire['row_count']
assert summary['traces']==len(wire['traces']) and summary['chars']==len(payloads[0])
records=rows(wire);traces=[trace(t) for t in wire['traces']]
out=dict(job=args.job,cluster=receipt['cluster'],summary=summary,wire=wire,rows=records,traces=traces,
 scope='Prepare-only flush observations from WG0 at CTAs0/1/66/67, matched REG112 redistribution; canonical is the fourth arm. Timed path is unchanged incumbent. Intervals include the sampling instructions and concurrent GPU work. Kind and stage count are schedule-derived labels, not independently sampled values. Phase-A partial readback is unknown.')
if wire['status']=='complete':
    assert len(records)==24 and len(traces)==2
    assert {(r['m'],r['seed'],r['pad_mib']) for r in records}==set(itertools.product((32,128,256),(3101,7907),(0,1,3,7)))
    for r in records:
        ns=r['ns'];assert set(ns)=={'a0','a1','trace','canonical'} and all(len(v)==5 and min(v)>0 for v in ns.values())
        ratio=statistics.median(math.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace']))
        canonical=statistics.median(c/math.sqrt(a*b) for a,b,c in zip(ns['a0'],ns['a1'],ns['canonical']))
        assert math.isclose(ratio,r['ratio'],rel_tol=1e-12)
        assert math.isclose(canonical,r['baseline_speedup_vs_canonical'],rel_tol=1e-12)
        assert math.isclose(statistics.median(a/b for a,b in zip(ns['a0'],ns['a1'])),r['aa'],rel_tol=1e-12)
        assert r['change']>.001
    out['ratios']=[dict(m=m,trace_min=min(r['ratio'] for r in records if r['m']==m),
        trace_median=statistics.median(r['ratio'] for r in records if r['m']==m),trace_max=max(r['ratio'] for r in records if r['m']==m),
        control_min=min(r['baseline_speedup_vs_canonical'] for r in records if r['m']==m),
        control_max=max(r['baseline_speedup_vs_canonical'] for r in records if r['m']==m)) for m in (32,128,256)]
    out['trace_gate']=all(.95<=r['ratio']<=1.05 for r in records)
    out['control_gate']=all(.98<=r['baseline_speedup_vs_canonical']<=1.02 for r in records)
    out['resource_gate']=all(wire[k][0]==wire[k][5]==0 and wire[k][2]==wire[k][7]==0 for k in ['resources','baseline_resources'])
    out['numerical_gate']=all(all(math.isfinite(x) for x in v) and v[0]<=.002 and v[1]<=.01
        for r in records for k in ['error','fresh'] for v in r[k].values())
    out['timing_attribution_gate']=out['trace_gate'] and out['control_gate'] and out['resource_gate'] and out['numerical_gate']
    out['max_aa_fraction']=max(abs(r['aa']-1) for r in records)
    out['max_error']={k:max(x for r in records for v in r[k].values() for x in v) for k in ['error','fresh']}
    for tr in traces:
        observations=[]
        for group in tr['groups']:
            c=group['cta'];expected={}
            for pb in (0,1):
                g0,g1=tr['metadata'][c][2*pb:2*pb+2]
                while g0<g1:
                    u,s0=divmod(g0,16);s1=min(16,s0+g1-g0)
                    if s0 or s1<16:expected[(pb,u)]=(3 if pb and s0==0 else 2,s1-s0)
                    elif not pb:expected[(pb,u)]=(1,0)
                    g0+=s1-s0
            assert len(group['frames'])==len(expected)
            seen=set();group['observations']=[]
            for pb,u,kind,raw_nst,start,end in group['frames']:
                key=(pb,u);assert key in expected and key not in seen;seen.add(key)
                nst,readback=raw_nst&255,bool(raw_nst&256)
                assert (kind,nst)==expected[key]
                if kind==1:assert not readback
                assert not readback # wire stores schedule count only, no readback bit
                j=u>>(5-pb);ntt=(tr['jobs'][j][1]+7)//8
                obs=dict(cta=c,phase=pb,unit=u,expected_kind=kind,expected_nst=nst,readback=(kind==3 if pb else None),ntt=ntt,flush_us=(end-start)/1000)
                group['observations'].append(obs);observations.append(obs)
        tr['interval_summary']=[]
        for pb in (0,1):
            for label,pred in [('whole_publication',lambda r:r['expected_kind']==1),
                               ('split_mixed_readback_unknown',lambda r:not r['phase'] and r['expected_kind']==2),
                               ('split_sender',lambda r:r['phase'] and r['expected_kind']==2),
                               ('split_finisher',lambda r:r['phase'] and r['expected_kind']==3)]:
                vals=[r['flush_us'] for r in observations if r['phase']==pb and pred(r)]
                if vals:tr['interval_summary'].append(dict(phase=pb,kind=label,n=len(vals),min=min(vals),median=statistics.median(vals),max=max(vals)))
    out['limits']=['Only four CTAs/WG0, one full replay for each M32/M256 fixture; M128 not traced.',
       'Aggregate timing gates do not prove identical internal schedules or isolate causal critical-path costs.',
       'Kind/stage labels are inferred from metadata. Phase-A partial readback is not observed.',
       'Phase-A flush and phase-B wait may overlap. Summing observed flush intervals is not a removable-time bound.']
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(ROOT))
print(json.dumps({k:v for k,v in out.items() if k in ['job','cluster','summary','ratios','trace_gate','control_gate','resource_gate','numerical_gate','timing_attribution_gate','max_aa_fraction','max_error']},indent=2))
for t in traces:print(json.dumps(dict(m=t['m'],interval_summary=t.get('interval_summary')),indent=2))
