"""Integer work/cut models from exact routed metadata, independent of clocks."""
import argparse
import heapq
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
src=ROOT/'muse_moe_w4a16'/f'PHASE_TRACE_{args.job}.json'
a=json.loads(src.read_text());assert a['wire']['status']=='complete'
target=ROOT/'muse_moe_w4a16'/f'UNIT_PARTITION_MODEL_{args.job}.json';assert not target.exists()
tables=[[112,135,140,158,173,185,200,207],[100,104,115,123,132,144,156,174]]
out=dict(job=args.job,scope='Exact integer scheduling model for two captured synthetic route rosters. No GPU timing estimates; timestamps are intentionally unused. Source cost coefficients are not a proven hardware-time model.',cases=[])
for tr in a['traces']:
    jobs=tr['jobs'];grid=len(tr['metadata'])
    for pb in (0,1):
        stride=512>>pb;prefix=[0]
        for _,count in jobs:prefix.append(prefix[-1]+stride*tables[pb][(count+7)//8-1])
        def work_to(stage):
            if stage==len(jobs)*stride:return prefix[-1]
            j,k=divmod(stage,stride)
            return prefix[j]+k*tables[pb][(jobs[j][1]+7)//8-1]
        original=[tr['metadata'][0][2*pb]]+[r[2*pb+1] for r in tr['metadata']]
        assert len(original)==grid+1 and original[0]==0 and original[-1]==len(jobs)*stride
        def stats(cuts):
            costs=[work_to(b)-work_to(a) for a,b in zip(cuts,cuts[1:])]
            assert sum(costs)==prefix[-1] and min(costs)>=0
            return dict(total_cost=sum(costs),max_cost=max(costs),mean_cost=statistics.mean(costs),
                max_over_mean=max(costs)/statistics.mean(costs),
                max_stages=max(b-a for a,b in zip(cuts,cuts[1:])),
                split_cuts=sum(s%16!=0 for s in cuts[1:-1]),
                empty_ctas=sum(x==0 for x in costs))
        base=stats(original);alignments=[]
        for q in (2,4,8):
            def align(s):
                rem=s%16
                return s-rem if rem<=q else s+16-rem if rem>=16-q else s
            cuts=[align(s) for s in original]
            assert all(a<=b for a,b in zip(cuts,cuts[1:]))
            st=stats(cuts);st.update(tolerance=q,max_cost_over_parent=st['max_cost']/base['max_cost'],
                max_stage_shift=max(abs(a-b) for a,b in zip(original,cuts)))
            alignments.append(st)
        units=[16*tables[pb][(count+7)//8-1] for _,count in jobs for _ in range(stride//16)]
        heap=[(0,c) for c in range(grid)];heapq.heapify(heap)
        for cost in sorted(units,reverse=True):
            load,c=heapq.heappop(heap);heapq.heappush(heap,(load+cost,c))
        full_costs=[v for v,c in heap]
        whole=dict(algorithm='offline longest-processing-time-first whole-unit assignment; device planner unimplemented',
            max_cost=max(full_costs),mean_cost=statistics.mean(full_costs),max_over_mean=max(full_costs)/statistics.mean(full_costs),
            max_cost_over_parent=max(full_costs)/base['max_cost'],units=len(units))
        if len(set(units))==1:
            lower=((len(units)+grid-1)//grid)*units[0]
            assert max(full_costs)==lower
            whole['homogeneous_exact_minimax_max_cost']=lower
            whole['proof']='Identical indivisible units imply some CTA owns at least ceil(units/CTAs); LPT reaches that bound in this scalar cost model.'
        out['cases'].append(dict(m=tr['m'],seed=tr['seed'],phase=pb,parent=base,alignments=alignments,whole_unit=whole))
out['limits']=['No input precision or route is changed; all arithmetic totals are conserved.',
 'Cut alignment changes stage owners; native inverse-owner mapping and producer/consumer order would also need a coherent implementation.',
 'Fixed work coefficients omit reduction costs, memory locality, readiness stalls, clock/power variation and input-dependent interphase progress.',
 'Whole-unit lower bounds do not prove a runtime regression; they disprove assuming that eliminating splits is free.',
 'No model here is a measured speedup or a qualified candidate.']
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(ROOT))
for c in out['cases']:
    print(c['m'],c['phase'],'parent max/mean',round(c['parent']['max_over_mean'],6),
      'alignment max-cost factors',[(x['tolerance'],round(x['max_cost_over_parent'],6),x['split_cuts']) for x in c['alignments']],
      'whole-unit factor',round(c['whole_unit']['max_cost_over_parent'],6))
