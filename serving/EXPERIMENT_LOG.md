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

## Nov 9 continuation: native validation and pinned Humming feasibility

- Previous goal turn was progress: two isolated candidates and documented
  sources were pushed as6640357. Continuing the owner's serving objective;
  the old kernel tracker text was not treated as a new task replacement.
- New native MoE component harness prepares exact stock/01/04/06/07methods
  from frozen patches. CPU source-preparation receipts preserve the initial
  plan, explicit unchanged-finalization assertion, and expanded row coverage
  for seven/eight positions per request at concurrency1/4/16/32. Five source
  hashes match the prior manifests; all ten methods compile. This is source
  preparation only, not a GPU or numerical pass. No candidate diff changed.
- Inspected pinned upstream `test_muse_marlin_inactive_routes`: its strong
  poisoned-buffer/dense-reference tests also assert old EP flags/count shape.
  Eight parameter combinations would conflict with06before checking arithmetic.
  This is static evidence; no native test run or quality waiver is claimed.
  No upstream test was edited. The new harness adds actual native projections,
  normalization, stock pairing, poison masks, dense references and fresh-input
  graphs, with incremental receipts/errors and optional balanced paired timing.
- Found official SGLang pyproject already declaresHumming0.1.12cu13. Verified
  and inventoried the exact PyPI source archive by SHA256,291regular files;
  six initial/nine additional selected files fetched. Read bounded schema/API/
  tuning sections, not every source file. No install/build/import/execution.
- Also inventoried current Apache2publicHumming3f2fa223,309files/eightselected
  downloads. Current small-row tuning uses newer abstractions absent from the
  pinned tuner; latest advertised H200 figures cannot be assigned to0.1.12.
  The current header tree alone is533,181bytes, exceeding the400KBdiffcap.
  The old fullhumming tree is816,019bytes; archive compression is irrelevant
  to submission source size. Existing dependency is the practical first route.
- Humming's generic SGLang runner directly weights/sums down projections,
  without Muse post-expert normalization. Direct substitution is unsuitable.
  Future spike must use the projection API, one canonical weight layout,
  unchanged BF16/FP32 precision, stock gate/up scales, existing finalization
  and demonstrated graph lock ownership. No speculative Humming patch added.
- A documentation patch first failed atomically on an incorrect library-file
  context; inspected status and reapplied with correct anchors. No evidence or
  source candidate was overwritten. No official check request was repeated;
  serving remains unopened, no live GPU handle or new score.

## Nov 9 continuation: precision-constrained Humming adapter

- Previous turn was progress: native validation preparation and the pinned
  Humming feasibility audit were pushed as b7499c2. No live GPU process/job
  was claimed. Continued the owner's serving objective from current files.
- Deeper pinned0.1.12 audit found a material precision distinction: FP32 MMA
  does not imply FP32 Stream-K reduction. The epilogue converts partial outputs
  to16bits and adds BF16/FP16 pairs across slices. Stock Muse requests FP32
  Marlin reduction. Excluded that default path; the development adapter sets
  no Stream-K and rejects every tactic that enables it or FP16 accumulation.
  Shared-memory reduction retains FP32 values for FP32 MMA in the inspected
  code. No compiled arithmetic or quality equivalence is inferred from this.
- Implemented development-only `serving/prototypes/humming_moe.py`, separate
  from all seven frozen submission diffs. It uses the existing0.1.12 API,
  packed FP4 weights/FP8 group scales, stock gate-column/BF16 outer-scale
  rounding, separate alignment tables when G1/G2 blocks differ, masked SiLU
  and the exact stock finalizer. No full-model checkpoint aliases/preparation
  hook, native run, performance win or novelty claim is provided.
- Initially considered expanding group scales to BF16; dropped before tests
  because it adds one byte per16weights, about11.1% to packed-weight/group-scale
  storage. Retained FP8 instead. The adapter replaces conventional parameters
  rather than retaining a Marlin fallback layout. Transient native repacking
  and full-model memory/cache effects still require measurement.
- CPU protocol doubles passed96cases/8mock preparations, EP1/2/4/8 last ranks,
  four row counts and random/local/dropped routes, equal/different block sizes.
  Verified gate-scale choice/rounding, six parameters, released references to
  old parameter objects, route coverage and call flow. Two forbidden precision
  configurations rejected. Fake tensor CUDA flags and fake operators are used;
  no actual Humming package or GPU kernel was imported/executed. API bind shapes
  come from pinned source with annotations/default values omitted, not changed
  required/optional arguments. The rerun with repo-pinned sources also passed.
- Pinned nine unchanged API/precision/source files including the package's
  LICENSE (no NOTICE exists in this archive), with archive/file hashes and
  explicit bounded read scope. This avoids dependence on temporary downloads
  for reproducing the CPU API checks. Full package extraction under/tmp was
  read-only and not an audit/execution of all133package files.
- Native comparison tool now has optional `--humming-prototype`, using the
  stock finalizer and the same poisoned slots/dense/paired/graph/timing gates.
  Only its source-plan mode ran. A function-local importlib binding that would
  shadow the earlier loader reference was caught in review and removed before
  native-plan validation; no GPU result or numerical failure was fabricated.
- All original candidate hashes and current candidate07patch are unchanged.
  No official request repeated, rented node ordered or service score obtained.
  Next gate is real native validation/timing; a no-Stream-K regression must be
  recorded/rejected rather than relaxed to lower precision for a faster score.

## Nov 9 execution-blocker revalidation

- Previous turn was progress (f0338b6), not a verified wait. This turn adds
  no optimization: it revalidates the missing execution route after three
  consecutive checkpoints with the same constraint.
- At23:48:02UTC requested candidate01again; exit1, no handle, service opens
  Nov12at9amPacific. Full response/SHA saved in EXECUTION_GATE_RECHECK.
  Fresh jobs/status/resource observations saved in EXECUTION_RESOURCE_SNAPSHOT:
  empty serving jobs, null priorities, local arm64Darwin, zero CUDA devices.
  No live process exists to poll. Listed kernel quotas do not establish serving
  eligibility or a development GPU, and were not used as a workaround.
- The current CLI exposes no documented serving development-node command.
  No existing remote GPU has been supplied/identified; no node rented.
  Remaining performance/quality/graph/TTFT decisions require native hardware.
  Further unmeasured source variants would not establish the next improvement.
- Saved an explicit blocked/completion audit and resume criteria. Preserve
  source/candidate hashes and CPU-only evidence. Goal status `blocked` is to
  be requested after this checkpoint is pushed; neither the stale kernel
  tracker objective nor current serving objective is complete.
