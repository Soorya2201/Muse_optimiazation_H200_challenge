# Serving experiment ledger

## Nov 9: initial implementation and source review

- User replaced the kernel objective with serving and revoked earlier AGENTS
  instructions. Goal API replacement failed because the previous goal is
  unfinished. Recorded OBJECTIVE.md; did not falsely complete the old goal.
- Fresh official init succeeds at297e54fd,400KB cap, and announces Nov12opening.
  Old a55ab7d5prototype/rules are retained as historical work. Serving status
  has no priority and jobs is empty. No check/submit POST attempted.
- Web direct rules/update fetches return403. Search cache shows Nov10opening;
  date discrepancy retained. Public discovery expanded to155public repo hits,
  with six selected exact trees and bounded reads, not an exhaustive audit.
- Candidate 01 applies the licensed global-route-density fix to the new path.
  Its public H100 layer measurements are external only. CPU65,540policy cases
  pass, including unchanged EP1behavior; arithmetic/GPU/model quality untested.
- Candidate 02 combines maximum/value token metadata into one collective.
  The initial patch attempt used an obsolete import anchor and failed atomically;
  re-read the new baseline and corrected the anchor before any test/run.
- CPU preflight initially attempted a zero-sized sampler batch. Stock itself
  rejects its ambiguous reshape. Confirmed the candidate has the same error,
  and retained72matching rejection cases rather than silently dropping the edge.
  All360nonempty sampler cases and786,432transport/selection cases pass.
- Isolated-index application, source restrictions and Python AST checks pass
  for both isolated candidates and their7,065-byte combined patch. No Triton
  compiler, NCCL, CUDA graph or serving-performance result is claimed locally.
- A real multi-GPU component script is prepared and unrun. It uses PyTorch NCCL,
  not the complete SGLang coordinator. Full official validation remains required.

Every future official run must record UTC time, candidate SHA, mode, handle,
service budget, terminal result, quality/TTFT/per-load data and available logs.
Poll an existing handle rather than restarting after an observation timeout.
Preserve invalid/truncated evidence; no success or throughput claim from it.

## Nov 9 follow-up: public sources and route setup

- Six additional public-filtered GitHub queries saved in TARGETED_DISCOVERY;
  three exact new trees and ten selected files hash-verified. Reviewed bounded
  source sections, not all files or entrants. Added the optimization library
  with exact sources, constraints, negative evidence and prioritized experiments.
- Traced `layer.workspace` from NVFP4 weight preparation into Marlin's lock
  protocol. Considered removing the per-forward zero/allocation. Rejected
  before coding after finding the official generic runner's explicit per-call
  storage requirement to avoid captured-graph lock aliasing/deadlocks. Final
  slice lock reset alone is insufficient. No performance experiment claimed.
- At23:07:40 UTC, requested official candidate 01check with `--no-wait`.
  CLI returned exit1: serving is closed until Nov 12 at 9 am PT; no handle returned.
  Exact SHA/request/response saved. Jobs returned[]afterward. Budget remaining
  was not supplied. This was a rejected request, not a submitted/running check.
- Candidate 04 fuses int32 narrowing, global-to-local ID lookup and negative-ID
  handling; keeps per-call scratch, stock CPU/noncontiguous fallback and EP1.
 480 CPU wrapper fallback cases plus fresh-input calls pass, including int64
  narrowing wraparound and negative sentinels. The Triton branch is unrun.
  Prepared a real CUDA graph/correctness/paired-timing tool, unrun locally.
- Candidate 06 adapts the local-alignment/prefix approach from lee101's Apache2
  public write-up to 297e54fd. Our source trace confirms `is_ep=True` disables
  both sorted-ID and inactive-prefix specializations in the wrapper. Already
  localized IDs allow a local consumer without changing distributed routing.
 420 exact-method CPU contract cases/840 instrumented GEMM calls pass across
  EP1/2/4/8, all ranks, random/local/peer/dropped routes and seven row counts.
  CPU doubles do not establish native alignment, numerical correctness or speed.
- The new method-test first failed because a branch-local import survived
  removal of only top-level imports. No candidate failure was inferred. Made
  import removal recursive and corrected the GEMM argument observer's offset
  before rerunning. These are fixture repairs; source candidate unchanged.
- New isolated04/06and combinations05/07pass base-application/source packaging
  and Python AST checks. Existing01/02/03preserved byte-for-byte. Development
  checkout/current patch now match07: 12,186 bytes, 4 files, 86 byte longest addition.
  No combination has measured speed/quality evidence. No source was published
  publicly, model precision changed, external hardware rented or peer harness run.
- First official check sequence when open: 01, 06, 02. Test04 separately as budget
  or an existing development node permits. Only choose a scored composition
  after sufficient evidence; avoid treating a small check as all-load validation.
