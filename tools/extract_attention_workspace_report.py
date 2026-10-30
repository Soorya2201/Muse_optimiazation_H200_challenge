"""Archive the six independent workspace-probe records without changing kernels."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "muse_attention_decode"
job = "acf6f11e935049db"
lines = (folder / f"profiles/submission_v20_workspace_shard_probe.{job}.log").read_text().splitlines()
records = [json.loads(line[7:]) for line in lines if line.startswith("SHARDS {")]
assert len(records) == 6
trials = [trial for record in records for trial in record["trials"]]
assert len(trials) == 50
assert all(len(trial["ratios"]) == 6 for trial in trials)
assert not any(trial["qualifies"] for trial in trials)
assert max(trial["error"] for trial in trials) <= 0.002
payload = {
    "job": job, "cluster": "sfo3",
    "source_sha256": "b2d660f4a494ce226b0c5180ab487659608154255297aa96fe237214df09b866",
    "scope": "Independent synthetic inputs, three fixtures and two alternating cold-L2 groups each. Timed judge path remains the parent.",
    "records": records,
}
report = """# Workspace and finalizer probe, October 29

Check `acf6f11e935049db`, cluster `sfo3`, passed 16 tests and 12 timed shapes
at 2.616172x (CI 2.576929-2.655415), 1439.622 us. The timed path is the parent;
this aggregate is not evidence of an optimization. Import-time probe finished in
48.5 s. All 50 alternative plans passed the synthetic correctness bound, with
maximum relative error 0.000146724. Zero qualified for adoption.

For each shape, this table selects the alternative with the largest minimum
parent/candidate time ratio across six groups. Ratios above one favor the candidate.
This selection is descriptive; the same data selected and evaluated the alternatives.

|Batch|Query length|Trials|Best alternative nc/coop|Minimum ratio|Maximum ratio|
|---:|---:|---:|---:|---:|---:|
"""
for record in records:
    trial = max(record["trials"], key=lambda trial: min(trial["ratios"]))
    report += (f'|{record["B"]}|{record["L"]}|{len(record["trials"])}|'
               f'{trial["nc"]}/{trial["coop"]}|{min(trial["ratios"]):.6f}|{max(trial["ratios"]):.6f}|\n')
report += """
B16L8 nc2/coop1 is a mixed, sub-1.5% result, not a consistent win. It is
incorrect to describe every alternative as slower. Some larger copy counts
regress severely: B1L8 nc32/coop0 ratios are 0.217536-0.219366. The fixed split
counts, synthetic normal-random keys/values, and one machine limit generalization.
The probe does not prove the parent is globally optimal.

Per CODEX_NEXT_STEPS.md, close the attention B1 merge/protocol family. Do not
submit this diagnostic file or spend more checks on these copy/finalizer variants.
Raw observations are in `workspace_probe_acf6f11e935049db.json`.
"""
for name, content in [
    (f"workspace_probe_{job}.json", json.dumps(payload, indent=2) + "\n"),
    ("WORKSPACE_PROBE_2026-10-29.md", report),
]:
    target = "muse_attention_decode/" + name
    assert not (ROOT / target).exists(), "Do not overwrite archived observations"
    patch = "*** Begin Patch\n*** Add File: " + target + "\n"
    patch += "\n".join("+" + line for line in content.splitlines()) + "\n*** End Patch\n"
    subprocess.run(["apply_patch", patch], cwd=ROOT, check=True)
print("Archived 50 alternatives over six shapes; zero qualified")
