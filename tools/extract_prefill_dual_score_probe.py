"""Decode, audit, and model the independent two-score prefill diagnostic."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import zlib

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("job")
args = p.parse_args()
track = ROOT / "muse_prefill_layer"
receipt = json.loads((track / "run_receipts" / (args.job + ".json")).read_text())
matches = re.findall(r"PREFILL_DUAL_SCORE_Z ([A-Za-z0-9+/=]+)", receipt.get("log_tail", ""))
assert len(matches) == 1, "expected one complete final report; do not restart job if logs are incomplete"
report = json.loads(zlib.decompress(base64.b64decode(matches[0])))
print("DIAGNOSTIC", report["status"], report.get("stage"), report.get("elapsed_s"))
print("RESOURCES", report.get("resources"))
if report["status"] != "complete":
    print("ERROR", report.get("error"), report.get("trace"))

groups = {}
for row in report["rows"]:
    key = (row["group"], row["pad_mib"], row["case"])
    assert key not in groups
    groups[key] = row
    for times in row["us"].values():
        assert len(times) == 9 and all(math.isfinite(v) and v > 0 for v in times)
    for arm, ratio in row["speedup"].items():
        recomputed = statistics.median(math.sqrt(a*b)/c for a,b,c in
            zip(row["us"]["a0"], row["us"]["a1"], row["us"][arm]))
        assert abs(recomputed-ratio) < .0000006
        assert row["errors"][arm] < .01 and row["fresh_error"][arm] < .01
    assert row["parent_change"] > .001

geometries = ("t1f", "t4f", "first0", "first32768", "first65536")
if report["status"] == "complete":
    assert set(groups) == set(itertools.product(range(2), (0,1,3,7), geometries))
    assert len(report["rows"]) == 40

def label(c):
    if c["layer"] == "sliding":
        return None
    if c["q_len"] == 753:
        return "t1f" if c["batch"] == 1 else "t4f"
    return "first" + str(c["ctx"])

cells = []
for b, profile in zip(receipt["benchmarks"], receipt["profiles"]):
    assert b["case"] == profile["case"]
    forward = sum(float(m[1]) for line in profile["report"].splitlines()
                  if "attn_fwd_kernel" in line and
                  (m := re.search(r"\s([0-9.]+) µs\s+[0-9.]+%", line)))
    cells.append(dict(case=b["case"], us=b["median_ns"]/1000,
                      forward_us=forward, label=label(b["case"])))

models = []
if report["status"] == "complete":
    for choices in itertools.product((43,48,49), repeat=3):
        qualified = True
        factors = []
        for group,pad in itertools.product(range(2), (0,1,3,7)):
            loads = {}
            for cell in cells:
                c, lab = cell["case"], cell["label"]
                choice = choices[0 if lab == "t1f" else 1 if lab == "t4f" else 2]
                saving = 0.
                if lab and choice != 43:
                    r = groups[group,pad,lab]
                    ratio = r["speedup"].get(str(choice))
                    if ratio is None:
                        qualified = False
                    else:
                        aa = max(r["aa"],1/r["aa"])-1
                        qualified &= ratio > 1.02 and aa < .01
                        saving = cell["forward_us"] * (1-1/ratio)
                old,new = loads.get(c["load"], (0.,0.))
                loads[c["load"]] = (old+c["weight"]*cell["us"],
                    new+c["weight"]*max(cell["us"]-saving,.001))
            factors.append(math.exp(statistics.mean(math.log(a/b) for a,b in loads.values())))
        models.append(dict(choices=choices, worst=min(factors), median=statistics.median(factors),
                           best=max(factors), qualified=bool(qualified and min(factors)>=1.03)))

summary = {}
for arm,lab in itertools.product(("48","49"), geometries):
    values = [r["speedup"][arm] for r in report["rows"] if r["case"]==lab and arm in r["speedup"]]
    if values:
        summary[f"{arm}/{lab}"] = dict(n=len(values), min=min(values), median=statistics.median(values), max=max(values))
output = dict(job=args.job, cluster=receipt.get("cluster"),
    scope="Independent fixtures only. Whole-layer timed dispatch remains parent. Phase model uses this job's forward profile and cannot establish a full-layer gain.",
    diagnostic=report, paired_summary=summary, same_job_cells=cells, fixed_combinations=models)
path = track / ("DUAL_SCORE_REPORT_"+args.job+".json")
assert not path.exists()
path.write_text(json.dumps(output,indent=2)+"\n")
print("SUMMARY",json.dumps(summary,indent=2))
print("QUALIFIED",[m for m in models if m['qualified']])
if models:
    print("BEST_NONEMPTY",max((m for m in models if tuple(m['choices'])!=(43,43,43)),key=lambda m:m['worst']))
print("REPORT",path.relative_to(ROOT))
