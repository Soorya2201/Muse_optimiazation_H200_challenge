"""Archive sampled unit intervals with explicit ownership and timing gates."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zlib
from moe_unit_trace_codec import rows,trace

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
receipt=json.loads((ROOT/'muse_moe_w4a16/run_receipts'/f'{args.job}.json').read_text())
target=ROOT/'muse_moe_w4a16'/f'UNIT_TRACE_{args.job}.json';assert not target.exists()
payloads=re.findall(r'^MOE_UNIT_TRACE_Z (\S+)$',receipt['log_tail'],re.MULTILINE)
assert len(payloads)==1,'Complete report missing; preserve UNKNOWN instead of reconstructing observations'
wire=json.loads(zlib.decompress(base64.b64decode(payloads[0])))
summaries=re.findall(r'^MOE_UNIT_TRACE_SUMMARY (.+)$',receipt['log_tail'],re.MULTILINE)
assert len(summaries)==1;summary=json.loads(summaries[0])
assert summary['status']==wire['status'] and summary['rows']==wire['row_count']
assert summary['traces']==len(wire['traces']) and summary['chars']==len(payloads[0])
records=rows(wire);traces=[trace(t) for t in wire['traces']]
out=dict(job=args.job,cluster=receipt['cluster'],summary=summary,wire=wire,rows=records,traces=traces,
 scope='Prepare-only sampled WG0 observations from CTAs0/33/66/99. Instrumented and matched control use REG112; canonical is a fourth arm. No measured production optimization. Intervals contain instrumentation and overlapping GPU work, not exact isolated hardware-operation latencies.')
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
        assert all(v[0]<=.002 and v[1]<=.01 for k in ['error','fresh'] for v in r[k].values())
    out['ratios']=[dict(m=m,trace_min=min(r['ratio'] for r in records if r['m']==m),
        trace_median=statistics.median(r['ratio'] for r in records if r['m']==m),trace_max=max(r['ratio'] for r in records if r['m']==m),
        control_min=min(r['baseline_speedup_vs_canonical'] for r in records if r['m']==m),
        control_max=max(r['baseline_speedup_vs_canonical'] for r in records if r['m']==m)) for m in (32,128,256)]
    out['trace_gate']=all(.95<=r['ratio']<=1.05 for r in records)
    out['control_gate']=all(.98<=r['baseline_speedup_vs_canonical']<=1.02 for r in records)
    out['resource_gate']=all(wire[k][0]==wire[k][5]==0 and wire[k][2]==wire[k][7]==0 for k in ['resources','baseline_resources'])
    out['timing_attribution_gate']=out['trace_gate'] and out['control_gate'] and out['resource_gate']
    out['max_aa_fraction']=max(abs(r['aa']-1) for r in records)
    out['max_error']={k:max(x for r in records for v in r[k].values() for x in v) for k in ['error','fresh']}
    for tr in traces:
        metadata=tr['metadata'];jobs=tr['jobs'];stats=[]
        assert tr['m'] in (32,256) and tr['seed']==3101 and tr['pad_mib']==0
        for group in tr['groups']:
            c=group['cta'];frames=group['frames'];group['observations']=[]
            expected=[]
            for pb in (0,1):
                g0,g1=metadata[c][2*pb:2*pb+2]
                sp=((g1-1)//16)*16 if pb==0 and g1%16 and ((g1-1)//16)*16>g0 else g0
                for a,b in ([(sp,g1),(g0,sp)] if sp>g0 else [(g0,g1)]):
                    while a<b:
                        u,s0=divmod(a,16);s1=min(16,s0+b-a)
                        expected.append((pb,u,s0,s1,0));a+=s1-s0
                expected.append((pb,0,0,0,1))
            assert [(f[0],f[1],f[2],f[3],f[15]) for f in frames]==expected,'Missing/reordered unit record'
            previous=None
            for f in frames:
                pb,u,s0,s1,ntt,prior,after,kind,fu,raw_nst,start,wait,end,fs,fe,final=f
                assert pb in (0,1) and prior in range(4) and after in range(4) and kind in range(4)
                assert start<=end
                if previous is not None and previous[0]==pb:assert prior==previous[6]
                else:assert prior==0
                nst,readback=raw_nst&255,bool(raw_nst&256)
                assert nst<=16 and raw_nst<512
                if not final:
                    assert start<=wait<=end and 1<=ntt<=8
                    j=u>>(5-pb);assert ntt==(jobs[j][1]+7)//8
                else:assert wait==0 and s0==s1==ntt==0 and after==0
                if prior:
                    assert kind==prior and start<=fs<=fe<=end
                    if not final:assert fe<=wait
                    assert previous is not None and fu==previous[1]
                    if kind==1:assert nst==0 and not readback
                    else:assert nst==previous[3]-previous[2]
                    if kind==3:assert readback
                else:assert kind==fs==fe==raw_nst==0
                obs=dict(phase=pb,unit=u,s0=s0,s1=s1,ntt=ntt,final=bool(final),
                    prior_kind=prior,after_kind=after,flush_kind=kind,flush_unit=fu,flush_nst=nst,readback=readback,
                    segment_us=(end-start)/1000,compute_interval_us=(wait-start)/1000 if wait else None,
                    epilogue_interval_us=(end-wait)/1000 if wait else None,flush_us=(fe-fs)/1000 if prior else 0.)
                group['observations'].append(obs);stats.append(obs);previous=f
        summary_stats=[]
        for pb in (0,1):
            for label,predicate,field in [
                ('whole_segment',lambda r:not r['final'] and r['s0']==0 and r['s1']==16,'segment_us'),
                ('whole_epilogue',lambda r:not r['final'] and r['s0']==0 and r['s1']==16,'epilogue_interval_us'),
                ('whole_publication',lambda r:r['flush_kind']==1,'flush_us'),
                ('split_no_readback',lambda r:r['flush_kind']==2 and not r['readback'],'flush_us'),
                ('split_readback',lambda r:r['readback'],'flush_us'),
                ('final_flush',lambda r:r['final'],'segment_us')]:
                vals=[r[field] for r in stats if r['phase']==pb and predicate(r)]
                if vals:summary_stats.append(dict(phase=pb,kind=label,n=len(vals),min=min(vals),median=statistics.median(vals),max=max(vals)))
        tr['interval_summary']=summary_stats
    out['limits']=['Four fixed CTAs and WG0 only, one full replay per traced fixture; no M128 unit trace.',
       'Quarter-position CTAs may have special boundary alignment; unsampled unit/CTA behavior is not inferred.',
       'The compute interval includes weight waits, MMA and a previous deferred flush; subtracting the flush does not prove a removable critical-path delay.',
       'Register allocation differs from canonical; passing aggregate gates does not prove identical per-phase schedules.',
       'Readback flag is bit8 of raw frame column9; low8 bits preserve the stage count.']
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(ROOT))
print(json.dumps({k:v for k,v in out.items() if k in ['job','cluster','summary','ratios','trace_gate','control_gate','resource_gate','timing_attribution_gate','max_aa_fraction','max_error']},indent=2))
for t in traces:print(json.dumps(dict(m=t['m'],interval_summary=t.get('interval_summary')),indent=2))
