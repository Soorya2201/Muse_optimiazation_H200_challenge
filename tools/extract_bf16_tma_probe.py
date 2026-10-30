"""Preserve paired TMA diagnostics and model fixed shape-only compositions."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("job")
parser.add_argument("--allow-partial", action="store_true",
                    help="Archive explicitly missing rows; never qualify incomplete evidence")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
track = root / "muse_bf16_proj"
receipt = json.loads((track / "run_receipts" / (args.job + ".json")).read_text())
assert receipt["summary"]["passed"], "Do not model a failed diagnostic as valid"
logs = list((track / "profiles").glob("*." + args.job + ".log"))
log = logs[0].read_text() if logs else receipt["log_tail"]
records = [json.loads(line.split("BF16_TMA_PROBE ", 1)[1])
           for line in log.splitlines() if line.startswith("BF16_TMA_PROBE ")]
if len(records) != 32:
    records = [json.loads(line.split("BF16_TMA_PROBE ", 1)[1])
               for line in receipt["log_tail"].splitlines()
               if line.startswith("BF16_TMA_PROBE ")]
expected = {(p, l, m, s) for p in range(4) for l in ('pooler', 'unpooler')
            for m in (128, 256) for s in (3119, 7919)}
observed = {(r['p'], r['l'], r['m'], r['s']) for r in records}
assert len(observed) == len(records), "Duplicate diagnostic records"
missing = sorted(expected - observed)
assert args.allow_partial or not missing, f"Missing diagnostics: {missing}"
assert not observed - expected
keys = sorted({(r['l'], r['m']) for r in records})
summaries = []
for layer, m in keys:
    rows = [r for r in records if (r['l'], r['m']) == (layer, m)]
    ratios = [r['r'] for r in rows]
    aa = max(abs(r['aa'] - 1) for r in rows)
    summaries.append(dict(layer=layer, m=m, min_ratio=min(ratios),
                          max_ratio=max(ratios), median_ratio=statistics.median(ratios),
                          max_duplicate_offset=aa,
                          observations=len(rows),
                          qualifies=len(rows) == 8 and min(ratios) > 1.02 and aa < 0.01))
profiles = receipt['benchmarks']
assert len(profiles) == 30
load_names = sorted({p['case']['load'] for p in profiles})
compositions = []
complete_groups = [(p, s) for p, s in itertools.product(range(4), (3119, 7919))
                   if all((p, l, m, s) in observed for l, m in keys)]
assert complete_groups, "No complete paired layout/seed group"
for flags in itertools.product((False, True), repeat=4):
    selected = {k for k, f in zip(keys, flags) if f}
    models = []
    for pad, seed in complete_groups:
        ratios = {(r['l'], r['m']): r['r'] for r in records if r['p'] == pad and r['s'] == seed}
        load_ratios = []
        for load in load_names:
            cells = [p for p in profiles if p['case']['load'] == load]
            before = sum(p['median_ns'] * p['case']['weight'] for p in cells)
            after = sum(p['median_ns'] * p['case']['weight'] /
                        (ratios[(p['case']['layer'], p['case']['m'])]
                         if (p['case']['layer'], p['case']['m']) in selected else 1)
                        for p in cells)
            load_ratios.append(before / after)
        models.append(math.prod(load_ratios) ** (1 / len(load_names)))
    gates = all(s['qualifies'] for s in summaries if (s['layer'], s['m']) in selected)
    compositions.append(dict(selected=sorted(selected), worst_model=min(models),
                             median_model=statistics.median(models),
                             max_model=max(models), component_gates=gates,
                             qualifies=bool(selected) and gates and min(models) >= 1.03))
payload = dict(job=args.job, cluster=receipt['cluster'], source_sha256=receipt['source_sha256'],
               scope="Prepare-only synthetic diagnostics. Timed kernel is unchanged parent. Same-job official parent cells weight a conservative raw-event ratio model; not a measured board gain.",
               missing_records=missing, complete_model_groups=complete_groups,
               truncation_caveat="One initial row was truncated by the service log tail" if missing else None,
               records=records, summaries=summaries, compositions=compositions)
target = track / ("TMA_PROBE_REPORT_" + args.job + ".json")
assert not target.exists(), "Preserve archived evidence"
target.write_text(json.dumps(payload, indent=2) + "\n")
print(target.relative_to(root))
print(json.dumps(summaries, indent=2))
print("Best fixed model:", max(compositions, key=lambda c: c['worst_model']))
print("Qualified compositions:", sum(c['qualifies'] for c in compositions))
