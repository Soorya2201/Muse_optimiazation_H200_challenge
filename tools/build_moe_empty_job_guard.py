"""Isolate a licensed empty-job range guard; requires a future H200 check."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parent=root/'muse_moe_w4a16/submission.py';source=parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
target=root/'muse_moe_w4a16/submission_v35_empty_job_guard_candidate.py';assert not target.exists()
anchor='  const int nj = S.njF + S.njS;\n  wgs_prefix(S, nj);'
guard='''  const int nj = S.njF + S.njS;
  // Adapted from trxphxle/okc-2026-muse-kernels, Apache-2.0,
  // commit72db638, kernels/moe.py: a shard may have no local work.
  if (nj == 0) {
    S.wtot = 0;
    S.wtotB = 0;
    S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;
    return;
  }
  wgs_prefix(S, nj);'''
assert source.count(anchor)==1
new=source.replace(anchor,guard,1)
name='okc_moe_w4a16_v216x_{tag}_sm{arch}'
assert new.count(name)==1
new=new.replace(name,'okc_moe_w4a16_v216x_empty_job_guard_{tag}_sm{arch}',1)
# Exact reversal proves there are no changes to packing, arithmetic,
# synchronization, dispatch, routing or precision in nonempty work.
assert new.replace(guard,anchor,1).replace('okc_moe_w4a16_v216x_empty_job_guard_{tag}_sm{arch}',name,1)==source
ast.parse(new)
target.write_text(new)
out=dict(scope='Source-isolated draft, no GPU correctness/performance result and no promotion.',
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
    attribution='trxphxle/okc-2026-muse-kernels@72db638fa2718bc0ea5645fdd5c52a5511594579 kernels/moe.py, Apache-2.0',
    exact_source_reversal_passed=True,syntax_passed=True,
    mechanism='For zero jobs, initialize both totals/all four stage cuts to zero and return before wgs_stage_at reads a nonexistent job; other source is unchanged except collision-safe extension identity.',
    caveats=['No actual parent empty-WGS failure has been measured.',
        'Current TINY phase diagnostic uses non-WGS kernels; its empty-route coverage does not validate WGS.',
        'GPU validation must cover empty32/128/256 and nearby/fresh routes, fallback scales, normal nonempty controls and full challenge tests.',
        'No additional Oct31 routine MoE checks remain; do not use the owner reserve.'])
p=root/'muse_moe_w4a16/EMPTY_JOB_GUARD_PREFLIGHT_2026-10-31.json';assert not p.exists()
p.write_text(json.dumps(out,indent=2)+'\n')
print(target.relative_to(root),out['candidate_sha256'],'unvalidated draft; canonical unchanged')
