"""Screen public routing byte pools; no GPU timing or critical-path proof.

Use archived external tuples, verify all histograms, then combine fractions
with our own scored phase-duration planning model. Different fixtures are
not matched comparisons. No field is a predicted achievable speedup.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/public/trxphxle_72db638/experiments/experts-sched/round4-tuples.csv'
PHASES = ROOT / 'muse_decode_layer/PHASE_OPPORTUNITY_2026-10-31.json'
OUTPUT = ROOT / 'muse_decode_layer/PUBLIC_REREAD_SCOPE_2026-10-31.json'
BYTE_JOB = 28_311_552
labels = {'decode 32':'D32', 'verify 16':'V16', 'verify 32':'V32'}
rows = list(csv.DictReader(SOURCE.open()))
assert len(rows) == 18
verified = []
for row in rows:
    hist = json.loads(row['n_e'])
    assert len(hist) == 32 and all(isinstance(n,int) and n >= 0 for n in hist)
    touched = sum(n > 0 for n in hist)
    j64 = sum((n+63)//64 for n in hist)
    j128 = sum((n+127)//128 for n in hist)
    assert touched == int(row['K']) and j64 == int(row['J64']) and j128 == int(row['J128'])
    assert sum(hist) == int(row['R'])
    assert BYTE_JOB * touched == int(row['unique_weight_bytes'])
    assert BYTE_JOB * j64 == int(row['planned_weight_bytes'])
    assert BYTE_JOB * (j64-touched) == int(row['duplicated_planned_weight_bytes'])
    assert BYTE_JOB * (j64-j128) == int(row['J128_removed_weight_bytes'])
    assert int(row['metric_pass_count']) == 1 and int(row['profiler_warmup_pass_count']) == 0
    assert abs(float(row['q']) - (j64-touched)/j64) < 1e-12
    verified.append(dict(case_id=row['case_id'],load=row['load'],layer=row['layer'],
                         weight_seed=int(row['weight_seed']),histogram=hist,
                         touched=touched,j64=j64,j128=j128,
                         full_dedup_fraction=(j64-touched)/j64,
                         j128_fraction=(j64-j128)/j64,
                         external_dram_read_bytes=int(row['measured_dram_read_bytes']),
                         external_duration_ns=int(row['measured_duration_ns'])))
phase = json.loads(PHASES.read_text())
models = []
for seed in sorted({r['weight_seed'] for r in verified}):
    fixtures = {(r['load'],r['layer']):r for r in verified if r['weight_seed']==seed}
    for field in ('full_dedup_fraction','j128_fraction'):
        grouped = {}
        for row in phase['rows']:
            case = row['case']; load = case['load']; initial = row['benchmark_us']
            fraction = fixtures[(labels[load],case['layer'])][field] if load in labels else 0
            changed = initial - row['profile_phase_us']['experts'] * fraction
            assert changed > 0
            group = grouped.setdefault(load,[0,0])
            group[0] += initial * case['weight'];group[1] += changed * case['weight']
        ratios = {k:v[0]/v[1] for k,v in grouped.items()}
        models.append(dict(external_weight_seed=seed,mechanism=field,
                           per_load_proxy=ratios,
                           six_load_proxy=math.prod(ratios.values())**(1/len(ratios))))
out = dict(scope='Conditional planning only: external route byte fractions scaled against our scored phase durations. Different inputs/jobs; profile durations contain waits/overlap. Not measured gain, rigorous upper bound or causal traffic-to-time law.',
           source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           source_url='https://github.com/trxphxle/okc-2026-muse-kernels/blob/72db638fa2718bc0ea5645fdd5c52a5511594579/experiments/experts-sched/round4-tuples.csv',
           parent_job=phase['job'],parent_sha256=phase['source_sha256'],
           verified_rows=verified,models=models,
           decision='No >=3% whole-decoder case established for reread-only optimization on these tuples. Larger service-efficiency changes remain a separate hypothesis. Zero-duplication fixtures have no reread byte pool. Do not generalize V32 average to every routing.')
OUTPUT.write_text(json.dumps(out,indent=2)+'\n')
for row in models:
    print(row['external_weight_seed'],row['mechanism'],f"{row['six_load_proxy']:.9f}")
