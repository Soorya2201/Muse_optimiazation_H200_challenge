"""Archive compressed independent attention diagnostics and fixed compositions."""
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
root = Path(__file__).resolve().parents[1]
folder = root / 'muse_attention_decode'
receipt = json.loads((folder / 'run_receipts' / (args.job + '.json')).read_text())
lines = receipt['log_tail'].splitlines()
payloads = [l.split('ATTN_PK2_HYBRID_Z ', 1)[1] for l in lines if l.startswith('ATTN_PK2_HYBRID_Z ')]
assert len(payloads) == 1, 'Need one intact compressed diagnostic report'
report = json.loads(zlib.decompress(base64.b64decode(payloads[0])))
out = dict(job=args.job, cluster=receipt['cluster'], source_sha256=receipt['source_sha256'],
           scope='Independent import-time fixtures; timed path remains parent. Modeled gains are not measured board gains.',
           diagnostic=report, summaries=[], compositions=[])
if report['status'] == 'complete':
    assert receipt['summary']['passed']
    assert len(report['nearby']) == 6
    cases = report['cases']
    assert len(cases) == 16
    assert {(r['pad'], r['b'], r['seed']) for r in cases} == set(itertools.product(range(4), (16, 32), (3209, 7909)))
    for batch, arm in itertools.product((16, 32), (1, 2)):
        rows = [r for r in cases if r['b'] == batch]
        ratios = [r['r'][arm - 1] for r in rows]
        aa = max(abs(r['aa'] - 1) for r in rows)
        out['summaries'].append(dict(batch=batch, arm=arm, min_ratio=min(ratios),
                                    variant=report.get('variant_names', {}).get(str(arm), str(arm)),
                                    median_ratio=statistics.median(ratios), max_ratio=max(ratios),
                                    max_duplicate_offset=aa, qualifies=min(ratios) > 1.02 and aa < .01))
    profiles = receipt['benchmarks']
    assert len(profiles) == 12
    loads = sorted({r['case']['load'] for r in profiles})
    for arms in itertools.product((0, 1, 2), repeat=2):
        chosen = dict(zip((16, 32), arms))
        models = []
        for pad, seed in itertools.product(range(4), (3209, 7909)):
            ratios = {r['b']: [1] + r['r'] for r in cases if r['pad'] == pad and r['seed'] == seed}
            per_load = []
            for load in loads:
                rows = [r for r in profiles if r['case']['load'] == load]
                before = sum(r['median_ns'] * r['case']['weight'] for r in rows)
                after = 0
                for row in rows:
                    c = row['case']
                    factor = ratios[c['batch']][chosen[c['batch']]] if (
                        c['batch'] in chosen and c['layer'] == 'full' and c.get('q_len', 1) == 8) else 1
                    after += row['median_ns'] * c['weight'] / factor
                per_load.append(before / after)
            models.append(math.prod(per_load) ** (1 / len(loads)))
        gates = all(s['qualifies'] for s in out['summaries'] if chosen[s['batch']] == s['arm'])
        out['compositions'].append(dict(arms=arms, worst_model=min(models),
                                        median_model=statistics.median(models), best_model=max(models),
                                        component_gates=gates,
                                        qualifies=any(arms) and gates and min(models) >= 1.03))
target = folder / ('PK2_HYBRID_REPORT_' + args.job + '.json')
assert not target.exists(), 'Preserve archived observations'
target.write_text(json.dumps(out, indent=2) + '\n')
print(target.relative_to(root))
print('Diagnostic:', report['status'], 'elapsed_s:', report['elapsed_s'], 'error:', report.get('error'))
print('Resources:', report['resources'])
print(json.dumps(out['summaries'], indent=2))
if out['compositions']:
    print('Best worst-case model:', max(out['compositions'], key=lambda x: x['worst_model']))
    print('Qualifies:', sum(x['qualifies'] for x in out['compositions']))
