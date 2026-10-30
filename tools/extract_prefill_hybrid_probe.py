"""Archive all prefill hybrid probes and evaluate fixed class combinations.

Models are planning evidence, not whole-layer performance measurements.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("job")
parser.add_argument("--variants", type=int, nargs="+", default=[44, 45])
args = parser.parse_args()
assert re.fullmatch(r"[0-9a-f]{16}", args.job)
track = ROOT / "muse_prefill_layer"
receipt = json.loads((track / "run_receipts" / (args.job + ".json")).read_text())
assert receipt["summary"]["passed"]
records = {}
resources = None
texts = [receipt["log_tail"]]
local_log = track / "profiles" / (Path(receipt["source"]).stem + "." + args.job + ".log")
if local_log.exists():
    texts.append(local_log.read_text())
for text in texts:
    for line in text.splitlines():
        if line.startswith("PREFILL_PIPE_RESOURCES "):
            observed = json.loads(line.split(" ", 1)[1])
            if resources is not None:
                assert observed == resources
            resources = observed
            continue
        if not line.startswith("PREFILL_PIPE_EXP_PROBE "):
            continue
        row = json.loads(line.split(" ", 1)[1])
        assert "skipped" not in row, row
        key = (row["case"], row["group"])
        if key in records:
            assert records[key] == row
        records[key] = row
labels = {
    "t1f": (1, 753, 80000, "full_short"),
    "t4f": (4, 753, 80000, "full_batch"),
    "first0": (1, 16384, 0, "full_long"),
    "first32768": (1, 16384, 32768, "full_long"),
    "first65536": (1, 16384, 65536, "full_long"),
}
assert set(records) == {(label, group) for label in labels for group in (0, 1)}, "incomplete diagnostic logs"
for (label, group), row in records.items():
    assert (row["batch"], row["q_len"], row["ctx"]) == labels[label][:3]
    assert row["repetitions"] == 9

# Each diagnostic times parent43 in the official full-layer call. Use this
# same job's profile as the model baseline, preserving cluster matching.
baseline = receipt
classes = ("full_short", "full_batch", "full_long")
eligible = {kind: {43: True} for kind in classes}
eligibility_detail = {}
for kind in classes:
    for variant in args.variants:
        key = str(variant)
        checks = []
        for label, shape in labels.items():
            if shape[3] != kind:
                continue
            for group in (0, 1):
                row = records[label, group]
                validation = row["validation"].get(key, {})
                ratio = row["paired_speedup"].get(key, {}).get("median", 0)
                aa = row["aa_ratio_median"]
                checks.append({"case": label, "group": group, "ratio": ratio, "aa": aa,
                               "pass": validation.get("finite") is True
                               and validation.get("relerr_vs_parent", 1) < 0.01
                               and 1 / 1.01 <= aa <= 1.01 and ratio > 1.02})
        eligible[kind][variant] = all(check["pass"] for check in checks)
        eligibility_detail[f"{kind}/{variant}"] = checks

shape_rows = []
for benchmark, profile in zip(baseline["benchmarks"], baseline["profiles"]):
    case = benchmark["case"]
    label = None
    phase = 0.0
    if case["layer"] == "full":
        label = next(k for k, v in labels.items()
                     if v[:3] == (case["batch"], case["q_len"], case["ctx"]))
        line = next(l for l in profile["report"].splitlines() if "attn_fwd_kernel<" in l)
        phase = float(re.search(r"([0-9.]+) µs", line).group(1))
    shape_rows.append({"load": case["load"], "weight": case["weight"], "label": label,
                       "shape_us": benchmark["median_ns"] / 1000, "forward_us": phase})

combinations = []
for variants in itertools.product((43, *args.variants), repeat=3):
    choice = dict(zip(classes, variants))
    valid = all(eligible[kind][variant] for kind, variant in choice.items())
    factors = []
    if valid:
        for group in (0, 1):
            old, new = {}, {}
            for shape in shape_rows:
                cut = 0.0
                if shape["label"]:
                    kind = labels[shape["label"]][3]
                    variant = choice[kind]
                    if variant != 43:
                        ratio = records[shape["label"], group]["paired_speedup"][str(variant)]["median"]
                        cut = shape["forward_us"] * (1 - 1 / ratio)
                load, weight = shape["load"], shape["weight"]
                old[load] = old.get(load, 0) + weight * shape["shape_us"]
                new[load] = new.get(load, 0) + weight * (shape["shape_us"] - cut)
            factors.append(math.prod(old[k] / new[k] for k in old) ** (1 / len(old)))
    combinations.append({"choice": choice, "eligible": valid, "modeled_factors": factors,
                         "clears_3percent_model_gate": valid and min(factors) > 1.03})
best = max((c for c in combinations if c["eligible"]), key=lambda c: min(c["modeled_factors"]))
payload = {
    "job": args.job, "cluster": receipt["cluster"], "source_sha256": receipt["source_sha256"],
    "timed_dispatch": "parent sliding8/full43; diagnostic score is not a hybrid score",
    "method": "independent synthetic fixtures; two seeds/groups; nine counterbalanced cold-L2 graph replays; duplicate parent controls",
    "gates": {"max_phase_relative_l2": 0.01, "max_duplicate_control_offset": 0.01,
              "min_phase_gain_each_group_case": 0.02, "min_modeled_board_gain_each_group": 0.03},
    "model_limits": "Uses this same job's timed-parent native forward durations and independent microbenchmark ratios. Overlap, allocator layout and end-to-end effects are not isolated; actual full-layer checks required before promotion.",
    "records": [records[k] for k in sorted(records)],
    "compiled_resources": resources,
    "eligibility": eligibility_detail, "fixed_class_combinations": combinations, "best_eligible_model": best,
}
target = track / ("HYBRID_PIPE_REPORT_" + args.job + ".json")
assert not target.exists(), "do not replace archived diagnostic evidence"
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in json.dumps(payload, indent=2).splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], capture_output=True, text=True, check=True)
print(target.relative_to(ROOT))
for row in payload["records"]:
    print(json.dumps({k: row[k] for k in ("case", "group", "aa_ratio_median", "paired_speedup", "validation")}))
print("BEST_ELIGIBLE_MODEL", json.dumps(best))
