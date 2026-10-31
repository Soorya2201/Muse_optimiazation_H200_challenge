"""Explicitly limited planning proxies from the two sampled V30 fixtures."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
job='b22622ebb61946df'
p=json.loads((ROOT/'muse_moe_w4a16'/f'FLUSH_TRACE_{job}.json').read_text())
assert p['timing_attribution_gate']
models=[]
for label,predicate in [('phase_a_whole_publication',lambda r:r['phase']==0 and r['expected_kind']==1),
                        ('phase_a_all_flush',lambda r:r['phase']==0),
                        ('all_flush',lambda r:True)]:
    fixtures=[]
    for tr in p['traces']:
        row=next(r for r in p['rows'] if (r['m'],r['seed'],r['pad_mib'])==(tr['m'],tr['seed'],tr['pad_mib']))
        # Same-job matched control, raw CUDA graph event time. No cross-job
        # subtraction and no inferred graph-overhead correction.
        event_us=statistics.median(math.sqrt(a*b) for a,b in zip(row['ns']['a0'],row['ns']['a1']))/1000
        sums=[sum(r['flush_us'] for r in g['observations'] if predicate(r)) for g in tr['groups']]
        a_us=[(c[2]-c[6])/1000 for c in tr['phase_clocks_ns']]
        fixtures.append(dict(m=tr['m'],event_us=event_us,per_cta_selected_flush_sum_us=sums,
            mean_flush_sum_us=statistics.mean(sums),max_flush_sum_us=max(sums),sampled_phase_a_us=a_us))
    for reduction in ['mean_flush_sum_us','max_flush_sum_us']:
        factors=[f['event_us']/(f['event_us']-f[reduction]) for f in fixtures]
        models.append(dict(scope=label,assumed_removed=reduction,fixtures=fixtures,
            two_fixtures_other_four_loads_unchanged_proxy=math.prod(factors)**(1/6)))
out=dict(job=job,status='planning_only',scope='Two individual synthetic M32/M256 fixtures, WG0 at four CTAs; no M128 full trace.',
    models=models,limits=[
      'These are counterfactual scheduling proxies, not measured optimization gains or rigorous board/critical-path upper bounds.',
      'Assumes every selected interval can be removed in full without replacement costs, lost overlap, extra work or memory-layout changes.',
      'An interval can overlap WGMMA or a dependency elsewhere. Summing durations does not identify exposed latency.',
      'Mean/max cover four selected CTAs only, not all CTAs/warpgroups or nearby routings.',
      'REG112 instrumented scheduling differs from canonical; aggregate timing gates do not establish per-phase equivalence.',
      'Synthetic fixture event times are not official scored inputs; M128 and the other four loads cannot be inferred from this sample.'
    ])
target=ROOT/'muse_moe_w4a16'/f'FLUSH_SCOPE_MODELS_{job}.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n')
print(target.relative_to(ROOT))
for m in models:print(m['scope'],m['assumed_removed'],m['two_fixtures_other_four_loads_unchanged_proxy'],[(f['m'],round(f['mean_flush_sum_us'],3),round(f['max_flush_sum_us'],3)) for f in m['fixtures']])
