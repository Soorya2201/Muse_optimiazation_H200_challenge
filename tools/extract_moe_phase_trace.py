"""Decode complete trace evidence and bound balance/split-fixup hypotheses."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zlib
from moe_phase_trace_codec import decode_rows,decode_trace

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('job');args=parser.parse_args()
receipt=json.loads((ROOT/'muse_moe_w4a16/run_receipts'/f'{args.job}.json').read_text())
output=ROOT/'muse_moe_w4a16'/f'PHASE_TRACE_{args.job}.json';assert not output.exists()
matches=re.findall(r'^MOE_PHASE_TRACE_Z (\S+)$',receipt['log_tail'],re.MULTILINE)
assert len(matches)==1,'Missing complete diagnostic; preserve UNKNOWN, never reconstruct lost observations'
wire=json.loads(zlib.decompress(base64.b64decode(matches[0])))
summaries=re.findall(r'^MOE_PHASE_TRACE_SUMMARY (.+)$',receipt['log_tail'],re.MULTILINE)
assert len(summaries)==1;summary=json.loads(summaries[0])
assert summary['status']==wire['status'] and summary['rows']==wire['row_count']
assert summary['traces']==len(wire['traces']) and summary['chars']==len(matches[0])
rows=decode_rows(wire);traces=[decode_trace(t) for t in wire['traces']]
analysis=dict(job=args.job,cluster=receipt['cluster'],scope='Prepare-only instrumented synthetic traces. Timed official kernel is the incumbent. Metadata is exact for each fixture; clocks can be perturbed by instrumentation and are not proof of causal hardware stalls.',
    summary=summary,wire=wire,rows=rows,traces=traces)
if wire['status']=='complete':
    assert len(rows)==24 and len(traces)==2
    assert {(r['m'],r['seed'],r['pad_mib']) for r in rows}==set(itertools.product((32,128,256),(3101,7907),(0,1,3,7)))
    for row in rows:
        assert all(len(v)==5 and all(isinstance(x,int) and x>0 for x in v) for v in row['ns'].values())
        ns=row['ns'];ratio=statistics.median(math.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace']))
        assert math.isclose(ratio,row['ratio'],rel_tol=1e-12)
        assert math.isclose(statistics.median(a/b for a,b in zip(ns['a0'],ns['a1'])),row['aa'],rel_tol=1e-12)
        assert row['change']>.001
        for field in ['error','fresh']:
            assert all(v[0]<=.002 and v[1]<=.01 for v in row[field].values())
    analysis['perturbation']=[dict(m=m,min=min(r['ratio'] for r in rows if r['m']==m),
        median=statistics.median(r['ratio'] for r in rows if r['m']==m),
        max=max(r['ratio'] for r in rows if r['m']==m)) for m in (32,128,256)]
    analysis['max_aa_fraction']=max(abs(r['aa']-1) for r in rows)
    analysis['max_error']={field:max(v[i] for r in rows for v in r[field].values() for i in (0,1)) for field in ['error','fresh']}
    phase_pairs={'route':(0,1),'gather':(1,7),'producer_join':(7,6),'phase_a':(6,2),'phase_b':(2,3),'grid_barrier':(3,4),'final':(4,5)}
    cost_tables=[[112,135,140,158,173,185,200,207],[100,104,115,123,132,144,156,174]]
    for tr in traces:
        clocks=tr['clocks_ns'];meta=tr['metadata'];jobs=tr['jobs'];grid=len(clocks)
        assert len(meta)==grid and all(len(r)==8 for r in clocks+meta)
        assert len(jobs)==meta[0][4]+meta[0][5] and all(r[4:]==meta[0][4:] for r in meta)
        nj=len(jobs)
        for pb in (0,1):
            ranges=[r[2*pb:2*pb+2] for r in meta]
            assert ranges[0][0]==0 and ranges[-1][1]==nj*(512>>pb)
            assert all(a<=b for a,b in ranges) and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
            expected=sum(cost_tables[pb][(cnt+7)//8-1]*(512>>pb) for e,cnt in jobs)
            assert meta[0][6+pb]==expected
        origin=min(c[0] for c in clocks)
        tr['phase_us']={}
        for name,(a,b) in phase_pairs.items():
            vals=[(c[b]-c[a])/1000 for c in clocks];assert min(vals)>=0
            tr['phase_us'][name]=dict(min=min(vals),mean=statistics.mean(vals),median=statistics.median(vals),max=max(vals))
        tr['clock_span_us']=(max(c[5] for c in clocks)-origin)/1000
        tr['entry_spread_us']=(max(c[0] for c in clocks)-origin)/1000
        tr['phase_b_absolute_end_max_minus_mean_us']=(max(c[3] for c in clocks)-statistics.mean(c[3] for c in clocks))/1000
        split_summary=[]
        for pb in (0,1):
            boundaries=[meta[c][2*pb] for c in range(1,grid)]
            split_edges=sum(s%16!=0 for s in boundaries)
            # Endpoint counts, not measured unit-end times. Rounding both
            # sides changes ownership and work balance; this is only screening.
            alternatives=[]
            for tolerance in (2,4,8):
                def align(s):
                    rem=s%16
                    return s-rem if rem<=tolerance else s+16-rem if rem>=16-tolerance else s
                cuts=[align(s) for s in boundaries]
                assert all(a<=b for a,b in zip(cuts,cuts[1:]))
                alternatives.append(dict(tolerance=tolerance,split_edges=sum(s%16!=0 for s in cuts),
                    max_stage_shift=max(abs(a-b) for a,b in zip(boundaries,cuts))))
            split_summary.append(dict(phase=pb,split_edges=split_edges,
                nonempty_ctas=sum(r[2*pb]<r[2*pb+1] for r in meta),
                max_stages=max(r[2*pb+1]-r[2*pb] for r in meta),alignments=alternatives))
        tr['split_summary']=split_summary
    analysis['timing_attribution_gate']=all(.95<=r['min'] and r['max']<=1.05 for r in analysis['perturbation']) and wire['resources'][2]==wire['resources'][7]==0
    analysis['timing_attribution_limit']='Passing the coarse perturbation gate does not make instrumented timings exact parent timings or prove a particular stall cause. Only two individual full traces; M128 has performance controls but no full CTA trace.'
output.write_text(json.dumps(analysis,indent=2)+'\n')
print(output.relative_to(ROOT))
print(json.dumps({k:v for k,v in analysis.items() if k in ['job','cluster','summary','perturbation','max_aa_fraction','max_error','timing_attribution_gate']},indent=2))
for t in traces:
    print(json.dumps({k:v for k,v in t.items() if k in ['m','jobs','phase_us','clock_span_us','entry_spread_us','phase_b_absolute_end_max_minus_mean_us','split_summary']},indent=2))
