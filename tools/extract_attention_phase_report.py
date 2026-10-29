"""Preserve the actual phase-probe output and a bounded interpretation for research."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "muse_attention_decode"
job = "13bc760e093343f3"
lines = (folder / f"profiles/submission_v17_phase_probe.{job}.log").read_text().splitlines()
records = [json.loads(s.removeprefix("PHASE ")) for s in lines if s.startswith("PHASE {")]
assert len(records) == 10
assert max(r["relerr"] for r in records) < 0.002
for row in records:
    for values in row["min_median_max_us"].values():
        assert 0 <= values[0] <= values[1] <= values[2]
payload = {"job": job, "source_sha256": "703205792c1797613d674bc02d250415218644fcc72384cc0b808849a058f1d1",
           "timed_parent_speedup": 2.605394354316633,
           "scope": "Single independent synthetic fixture per shape, not the judge timing inputs. Traced module is separate from the unchanged timed parent.",
           "records": records}
intro = """# Attention phase evidence, October 29

Check `13bc760e093343f3` passed all 16 tests and 12 timings at **2.605394x**,
1445.578 us, CI [2.566313, 2.644475]. The timed path is the parent; no algorithmic
speedup is claimed. Both modules built in 47.4 seconds; the full probe reported 47.6.
Raw extracted observations are in `phase_probe_13bc760e093343f3.json`.

The table uses global phase timestamps relative to the earliest CTA entry, in microseconds.
Raw graph times include replay overhead. These are diagnostic measurements on one independent
fixture per shape, with normal random FP8 keys/values and scattered page-1 addresses. They
are not the scored coherent-key fixtures. Debug output differs from control by at most
1.58e-5 relative L2; measured debug overhead ranges from 0.93 to 2.70 us.

|Batch|Query length|Window|Parent/debug raw us|First K/V median|Last head-0 loop mark|Last partial issue|Last merge-ready mark|Last finalizer-exit mark|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
rows = []
for r in records:
    t = r["min_median_max_us"]
    rows.append(f'|{r["B"]}|{r["L"]}|{r["window"]}|{r["parent_raw_us"]:.3f}/{r["instrumented_raw_us"]:.3f}|'
                f'{t["first_KV_ready"][1]:.3f}|{t["consumer_loop_end"][2]:.3f}|'
                f'{t["partial_issue_end"][2]:.3f}|{t["merge_ready"][2]:.3f}|{t["output_issued"][2]:.3f}|')
outro = """

Interpretation and limits:

- At B1 sliding decode, the last partial-issue mark is 2.944 us and merge-ready is
  4.224 us: a 1.280 us global-stage interval. For B1 sliding verify it is 4.416 to
  6.048 us, or 1.632 us. Differences between median markers (1.50/2.37 us) also include
  CTA imbalance and must not be called pure counter latency.
- Batched full verify spends most of its elapsed time before the loop marks finish.
  Head-0 loop marks reach 103.328/186.848 us at B16/B32; finalizer-exit marks reach
  107.072/191.200 us. This supports a loop/scheduling experiment, but does not establish
  whether SFU, tensor cores or memory wait dominates that loop.
- The loop marker samples thread0/head0, not every consumer group. The last-to-last
  intervals also include other-head work and coordination. First markers are per-CTA
  first-segment observations; last markers can span later segments.
- Instrumentation perturbs register allocation, scheduling and memory traffic. Differences
  between graph timing and one cold eager trace do not justify subtracting a universal
  overhead from every phase. `output_issued` marks finalizer exit; some CTAs own no rows,
  and this is not a fence proving every output store has reached HBM.

Actions from this evidence:

1. The one-time flat-L8 softmax offset was checked next, passed at 2.593449x and did not
   improve the aggregate. Keep it as negative/inconclusive component evidence.
2. Test B1 sliding-L1 completion flags against the original contended arrival counter.
   Preserve the existing FP32 reductions, CTA publication barrier and finalizer. One warp
   polls at most 32 split flags in parallel; all flags reset before kernel completion.
3. If that protocol wins, validate it on different context lengths/page sizes and graph
   replays before considering a separate distributed-finalizer version for L8. Do not
   assume one CTA finalizing all 128 L8 rows will improve its cooperative parent.
"""
for name, content in [("phase_probe_13bc760e093343f3.json",json.dumps(payload,indent=2)+"\n"),
                      ("PHASE_PROFILE_2026-10-29.md", intro+"\n".join(rows)+outro)]:
    target = "muse_attention_decode/"+name
    assert not (ROOT / target).exists(), "Do not replace an archived report"
    patch="*** Begin Patch\n*** Add File: "+target+"\n"+"\n".join("+"+s for s in content.splitlines())+"\n*** End Patch\n"
    subprocess.run(["apply_patch",patch],cwd=ROOT,check=True)
print("Preserved all 10 trace records and bounded stage interpretation")
