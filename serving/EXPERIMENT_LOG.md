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

## Nov 10 UTC / Nov 9 Pacific: owner-requested concept research

- Owner explicitly requested continued learning/approach development while
  execution is unavailable. Resumed useful offline research, not a GPU job
  or a claim that the execution blocker has disappeared.
- Reviewed bounded sections from 12 primary sources: DFlash, original
  speculative decoding, Sequoia, SGLang/RadixAttention, PagedAttention, FA3,
  CUDA PDL, NCCL graph requirements, pinned HiCache design, Sarathi-Serve,
  pinned DeepEP README and prior pinned Humming sources. Versions/read scopes
  saved in CONCEPT_SOURCES; six official-base source hashes recorded separately.
  No whole-paper/device-source audit or exhaustive-web claim.
- Developed an 18-family playbook connecting mechanisms to fixed workload,
  precision, output, memory, capture/ordering and TTFT constraints. Important
  distinction: distributional speculative correctness is insufficient to
  assume stock-output/RNG agreement after changing verification cadence.
- Source trace: projection/router work is already arranged on separate
  streams. Marlin pre-expert normalization happens after the router join.
  Prepared a design to move that same out-of-place norm before the join, with
  ideal saving min(N,max(0,R-P)). Contention/host gaps may erase or reverse it.
  No new patch implemented; only profile-confirmed slack justifies the test.
- Implemented analytical cost models: whole-step Amdahl sensitivity, exact
  accepted-prefix survival/histogram identity with bonus/tail cap, explicit
  full-block DFlash cost, actual fixed active batch size distinct from client
  concurrency, supplied cost bounds and no automatic runtime policy.
  Full-block histograms are counterfactual screens, not evidence of future
  steady-state acceptance. Real graph-bucket/rollback/RNG costs remain needed.
- Initial163math checks passed. Extended with seven actual-batch/concurrency
  unit cases; final170checks pass. Both unmeasured-template runs correctly
  return needs_measurements without candidate ranking. Latest receipts include
  tool/input hashes. These synthetic calculations are not GPU benchmarks.
- DeepEP metadata request initially used a trailing slash and returned404;
  corrected the URL, then pinned public README commit/hash. Not an access or
  code-availability failure. CurrentV2/API examples are not a drop-in SGLang
  adapter. No external device code imported, installed or executed.
- Public released-model config request returnedHTTP401 after the web tool
  could not open it. Preserved failure; no access bypass or weight download.
  Did not invent full-model dimensions from historical kernel floors. The
  old MoE TASK explicitly uses busiest-rank skew; averages are insufficient.
- No official request repeated, new submission diff, native timing, serving
  score or rank. Frozen candidate/prototype hashes are preserved. Publish
  research/model/design artifacts to the existing private branch and draft PR.

## Nov 10 UTC: implement the norm-overlap experiment

- User explicitly resumed approach development; new goal tracker is active.
  Prior goal turn was concrete research progress, not a verified wait.
- Candidate08 moves the same pre-expert norm before the existing router join
  only for non-SP CUDA NVFP4 Marlin with a norm. The later call receives None
  for that invocation; module state, routing, weights and precision are unchanged.
  Source packaging passes: 1,343 bytes, one file, longest added line74 bytes.
- Exact stock/candidate coordinator bodies and actual stock norm bodies execute
  on CPU:4,320 paired output/input matches,128 eligible relocations,3,328
  nonempty fallbacks and864 empty cases. Stream/routing/expert doubles explicitly
  exclude CUDA, native Marlin arithmetic, capture correctness and performance.
- First fixture failed on stock record_stream because fake packed route IDs
  were a tuple. Corrected that double to a tensor; no candidate change. One
  documentation anchor failed atomically and was corrected. Development JSON
  preserves these failures. AST hashes vary by Python schema; final receipts
  include the interpreter and portable source SHA256. Earlier receipts retained.
- Optional --norm-overlap native mode now prepares the exact coordinators with
  actual router/fused routing/BF16 projection/norm and stock Marlin. It plans
  eager norm/route comparisons, fresh-input/bias graph replay, paired AB/BA
  whole-coordinator timing and separate optional profiler timelines. Only its
  source-plan mode ran. One-GPU synthetic EP fixture is not full serving.
- Candidate07/current patch and Humming prototype hashes unchanged. No composite
  candidate, official request, GPU job, paid hardware or performance claim.
  Candidate08 needs representative slack/contended-timing evidence and native
  validation before promotion; first official checks remain01/06/02.

## Nov10: correct DFlash2 targeting and use existing observability

- Prior candidate08 turn was progress, not a live wait. Fresh read-only status
  is null/empty jobs at06:06UTCNov10; no check, submit or GPU job requested.
- Traced existing CPU acceptance copies -> per-request histogram -> native
  response metadata and existing profile endpoints/step annotations. Built
  offline JSON/JSONL/SSE intake with230 synthetic checks using actual official
  histogram/metric bodies. Counts finals once, validates weighted counters,
  preserves incomplete captures and separates raw A+1 from client C/V. No
  generated tokens, active-batch inference, latency, quality or score evidence.
- Important planning correction: public unauthenticated DFlash2 config fetched
  and pinned at46c90f4 declares DFlash2DraftModel with rank256/top-k16 selector.
  Base source creates CandidateSelector; worker selects that sampler before
  the plain class changed by02. CPU constructor doubles confirm that branch.
  The deployed service revision/effective TP/path remain unobserved. Earlier
  first-day01/06/02 was incomplete; new plan01/06/09.02 retained as fallback.
- Candidate09 packs FP32 bits +31-bit globally bounded ID into int64:8vs12B
  per candidate and one metadata collective instead of two; unchanged local
  radix, dense projection, global top-k, selector, RNG and verification. All
  rank predicates use shared model/platform facts; partition/LSE/large-ID/CPU/
  HIP/FP64/TP1 paths keep original code.4,170B/2files packaging passes.
- CPU196608protocol values and2304function cases pass:216fast,2088fallback.
  Projection/collectives/Triton/admission are doubles and local top-k is Torch;
  actual Triton/NCCL/FlashInfer/capture unrun. First builder fixture missed a
  logger and raised NameError; fixed fixture only. Native eight-Hopper
  head/radix/Triton/PyTorchNCCL harness passes source preparation only. Plans
  native bits, fresh-input graphs and rank-max AB/BA timing; not full lattice
  or SGLang GroupCoordinator/serving validation.
- A documentation context failed atomically and was corrected; guessed old
  source paths/glob misses were resolved by rg discovery. Web raw-config
  retrieval failed, public HTTP resolve succeeded200 and pinned bytes match.
  Audits record these failures; no native regression or win invented. Raw
  model README kept temporary; only factual config/API/pin evidence published.
- Candidate01..08/current07/Humming remain frozen; no composite or winning
  submission selected. Need actual path/profile/native/quality/TTFT/all-load
  evidence; no rank or throughput claim. Push corrected approach privately.

## Nov10: actual offline Triton compilation gate

- Previous6e3eff8turn was concrete source-path/intake/selector progress, not a
  verified wait. Local Mac still lacks CUDA and Docker/Podman/Colima/Lima/NVCC;
  fresh read-only jobs remain empty. No official check/submit was requested.
- Found a useful execution route: private repository Actions is enabled. Prepare
  an8-minute-bounded CPU job compiling the five exact added Triton kernels from
  frozen02/04/09 for an explicit Hopper target. This is actual code generation/
  assembly, not another CPU protocol double and not a GPU numerical test.
- SGLang pins Torch2.14.1; its Linux metadata requests Triton~=3.8.0. Pin the
  exact3.8.0CPython3.12/Linuxx86wheel hash; no Torch/model/GPUdriver install.
  Action commit SHAs verified via public GitHub API. Deployed image still unknown.
- First local source extractor dropped the final newline before a following
  diff header; existing source-hash validation rejected it. Fixed extraction,
  preserved TRITON_AOT_PLAN_FAILED receipt; all five source/AST hashes now match.
  Source plan passes25specializations,0compiled locally. No frozen patch changed.
- Push the compiler/workflow, follow its specific run handle to terminal, and
  preserve failure/assembly artifacts. GPU/NCCL/graphs/quality/TTFT/performance
  remain required even if offline compilation succeeds. No fastest composition
  or new serving rank is implied.

### Offline compiler result

- CIrun42768219289 oncfbaf73 failed workflow parsing before any job. Local
  Ruby parsing reproduces the YAML mapping error at the unquoted pip flag's
  colon. Changed that command to a block scalar; no kernel/candidate change.
- Specific next run42768362132 on446889a completed successfully. Actual
  Triton3.8.0/bundledptxas12.9.86 compiled and assembled25specializations
  across five kernel bodies:14selector,9greedy,2map. Generated PTX8.8
  targets sm_90a. This is not an audit of the organizer's CUDA13 image.
- Downloaded private artifacts;125assembly/IR hashes match the report,25
  cubins have CUDA ELFmachine190, and source/tool/CIcommit identities match.
  Permanent253721Bzip and rawCIlog archived privately, along with compiled/
  verified/source-audit JSON. Both CI runs are terminal; no live wait remains.
- This closes actual code-generation/assembly for those signatures. It does
  not run GPU numerics, collectives, SGLang host wrappers, graphs, quality,
  TTFT or timing. All nine candidate/current07/Humming source hashes unchanged.
  No native runtime win, scored submission or rank claim. Continue through
  real native/full-model gates when official/existing hardware is available.
- One final documentation patch used a stale wrapped-line anchor and failed
  atomically; corrected the anchor, with no partial edits or source changes.

- Final git whitespace verification rejected trailing whitespace in the raw
  provider log atline148. Preserved identical raw bytes as gzip with hashes
  instead of editing that log or bypassing the check. No source/compiler change.

## Nov10: native execution blocker audit after compilation

- Previous953b663goal turn was concrete progress: actual offline Hopper code
  generation/assembly, not a live wait or measured serving gain.
- One official01check at06:34UTCNov10 after that gate again returned exit1:
  serving opensNov12at9amPT. Stdout empty, no job/handle. Exact source SHA
  and raw response preserved in OFFICIAL_EXECUTION_RECHECK_2026-11-10.json.
- Fresh read-only serving jobs are empty; local Torch2.15reports CUDAfalse/0
  devices onDarwinarm64. CompilerCI42768362132terminalsuccess; no existing
  authorizedHopper node supplied. No GPU process or CI wait remains live.
- New blocked audit maps source/CPU/compiler evidence to full requirements.
  Native numerics/precision/bounds, SGLang/NCCL/graphs, actual workload profiles,
  all-load stockanswers/220tokens/TTFT, strongest measured composition and
  final winning result remain unverified. No completion/score/rank claim.
- The same runtime gate persisted through resumedc005d9e/6e3eff8/953b663
  and this fresh audit. Earlier turns made offline progress; the current next
  empirical action needs an external execution-state change. Fresh blocked
  threshold satisfied without counting old pre-resume goal turns.
- Nine frozen patch hashes/current07/compiler archive remain intact. Publish
  audit privately, then requestblocked. Resume on official opening or existing
  authorizedHopper; revalidate live rules/base/budget before01/06/09. Poll only
  an accepted returned handle. No future automatic task or GPU rental created.
