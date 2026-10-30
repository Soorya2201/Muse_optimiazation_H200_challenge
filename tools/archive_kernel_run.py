"""Preserve terminal judge evidence, including cluster and every kernel profile."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("job")
args = parser.parse_args()
raw = subprocess.run(["okc", "status", args.job, "--json"], check=True, capture_output=True, text=True)
job = json.loads(raw.stdout)
assert job["status"] == "done", job["status"]
result = job["result"]
records = result.get("records", [])
source = Path(job["profile"]["json"])
profile = json.loads(source.read_text())
submission = Path(profile["submission"])
payload = {
    "job": args.job,
    "board": job["board"],
    "mode": job["mode"],
    "cluster": job.get("cluster"),
    "machine": result.get("machine"),
    "created_at": job.get("created_at"),
    "source": str(submission.relative_to(Path.cwd())),
    "source_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
    "duplicate_of": job.get("duplicate_of"),
    "summary": profile["summary"],
    "setup": [row for row in records if row.get("event") == "setup"],
    "benchmarks": [
        {key: row[key] for key in ["case", "median_ns", "baseline_ns", "paired_ratio", "speedup", "status"] if key in row}
        for row in profile["benchmarks"]
    ],
    "profiles": profile["profiles"],
    "log_tail": result.get("log_tail", ""),
}
target = Path(job["board"]) / "run_receipts" / (args.job + ".json")
assert not target.exists(), "Do not replace archived measurements"
content = json.dumps(payload, indent=2) + "\n"
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in content.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], check=True, capture_output=True)
print(target)
print(json.dumps({"cluster": payload["cluster"], **payload["summary"], "setup": payload["setup"]}))
