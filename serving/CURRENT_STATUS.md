# Serving checkpoint — Nov 9 2026

The [offline Triton compiler gate](TRITON_OFFLINE_COMPILATION.md) now passes
actual Hopper code generation/assembly in private CPU CI: five kernels from
02/04/09 across25specializations. Verified125assembly hashes and25CUDAELF
cubins against the frozen sources/tool/CI commit. Triton3.8.0/bundledptxas12.9.86
is not a deployed-image audit. No GPU kernel was launched. This closes a
compilation gap; numerical behavior, NCCL/SGLang, graphs and performance remain
unproven. Run42768362132 succeeded after fixing a recorded workflow syntax
failure in42768219289; neither run remains live. Source patches are unchanged.

Latest Nov10 source correction: the public DFlash2 configuration declares a
rank256/top-k16 candidate selector. The pinned worker therefore selects its
selector sampler rather than candidate02's plain greedy sampler. See
[the reachability audit and corrected approach](SELECTOR_COLLECTIVE.md).
First isolated official checks are now **01/06/09**, subject to current
rules/budget and confirming the deployed model/path. Earlier01/06/02 guidance
below and in historical receipts is superseded;02 remains a generic fallback.

[Candidate09](candidates/09_selector_single_collective.diff) is4,170bytes and
packs unchanged FP32 value bits plus a bounded vocabulary ID into one int64.
It replaces two selector metadata all-gathers with one,8vs12bytes/candidate,
then restores the original values/ID order for unchanged global top-k. It does
not change the selector model, dense head, local top-k or sampling.196,608 CPU
protocol values and2,304 exact-function cases pass;216 use one collective and
2,088 keep stock fallbacks. Offline Triton compilation passes separately;
CUDA numerics/NCCL/graphs and serving gains remain unproven. The eight-Hopper
native head-path harness has passed source-plan
mode only.09 is isolated; no new combined/current patch was selected.

[Measurement intake](MEASUREMENT_INTAKE.md) now uses existing response metadata
and profiling controls.230 synthetic parser checks pass, including final
snapshot deduplication and stock acceptance-counter semantics. The fixtures
contain no generated output/GPU run. This avoids inventing a histogram from a
mean acceptance length or treating client concurrency as real active batch.
Fresh read-only jobs are empty and priority is null at06:06UTCNov10; no new
official check or GPU job was requested. This turn and the previous candidate08
turn made concrete source/implementation progress; neither is a live GPU wait.

Nov 10 UTC approach-development continuation: isolated
[candidate08](candidates/08_marlin_norm_overlap.diff) now moves the existing
Marlin pre-norm before the router join. It is 1,343 bytes, applies to the
official base and passes 4,320 paired CPU coordinator/norm cases. This includes
128 eligible nonempty relocations; the other 4,192 empty/fallback cases retain
stock ordering. Simulated streams/experts are not CUDA/Marlin validation.
The [design and gates](designs/HOPPER_NORM_OVERLAP.md) explain its conditional
benefit and why it can regress. An optional native whole-coordinator experiment
with actual router/Marlin, fresh-input graphs, paired timing and profiler traces
is prepared; only source-plan mode ran. Receipts end in
`NORM_OVERLAP_CPU_CHECKED_2026-11-10.json` and
`NORM_OVERLAP_NATIVE_PLAN_CHECKED_2026-11-10.json`.
Candidate07/current patch and the Humming prototype are unchanged. Candidate08
is not part of a combined submission and does not displace the first planned
official checks without representative evidence. No official run or GPU job
was requested in this continuation. The previous goal turn was research
progress; this resumed turn adds an executable experiment, not a verified wait.

Owner requested continued concept research while GPU execution is unavailable.
The [new research playbook](RESEARCH_PLAYBOOK_2026-11-10.md) covers 18 concept
families from 12 primary sources, with bounded read scopes. It adds a
source-traced pre-norm overlap design and an analytical cost-screening tool.
The tool passes 170 math identity/boundary checks; its empty measurement
template produces `needs_measurements`, not a winning candidate. No runtime
patch, native run or score was added in this research pass. Dates in the new
receipts use Nov 10 UTC (Nov 9 Pacific).

The project objective is now [winning the serving challenge](OBJECTIVE.md).
There is no official serving run, score or verified rank for this work yet.
Two check requests were rejected before a job was created because serving is closed.
The latest live recheck at23:48UTC confirms November12at9amPacific. This is an
execution blocker, not a running wait. See [the audit](EXECUTION_BLOCKER_AUDIT_2026-11-09.md)
and resource snapshot. The prepared native experiments need official execution
or an existing authorized Hopper node; no such node has been supplied.

Private review: [draft PR1](https://github.com/Soorya2201/Muse_optimiazation_H200_challenge/pull/1),
branch `serving/20261109`. Initial implementation/research checkpoint46e0902.

Fresh `okc serve init` pins **297e54fd105e974cdf401e5b4d20253a22c16f90**.
Its rules allow400KB of source diff under `python/sglang/`, with no added line
over400characters, and3checks/1scored run per day. The initializer says
**Nov12,9am Pacific** for opening. The cached Nov3 web announcement says Nov10;
direct page fetches returned403. At23:07 UTC Nov 9, the actual check endpoint
confirmed Nov 12 at 9 am Pacific and rejected candidate 01without a handle. The
init notice and current RULES hash are in INITIALIZATION_2026-11-09.json.

The new flags include draft attention `fa4`, CPU multimodal feature transport
and160GB HiCache. The original a55ab7d5/720KB serving plan is historical.
The new working checkout is ignored at `serving/current/sglang`; the source
diffs and downloaded current rules are committed and reproducible.

## Implemented candidates

| Candidate | Mechanism | State |
|---|---|---|
| [01_ep_route_density.diff](candidates/01_ep_route_density.diff) | Correct Marlin's expected expert occupancy using global expert count for the global token batch. Adapted from hariappi's licensed public fix to the new module path. | CPU policy checks pass; no own GPU/serving result |
| [02_draft_single_collective.diff](candidates/02_draft_single_collective.diff) | Transport the draft head's maximum value and full int64 token ID in one all-gather; fuse global selection. Preserve stock first-rank ties/NaNs and unchanged dense GEMM. | CPU protocol/class and offline Triton compilation pass; NCCL/GPU/graphs pending |
| [03_combined.diff](candidates/03_combined.diff) | Both mechanisms together | Packaging passes; evaluate only after isolated results |
| [04_fused_expert_map.diff](candidates/04_fused_expert_map.diff) | Fuse int32 cast, local expert mapping and dropped-route selection into one CUDA kernel; retain CPU/noncontiguous fallback |480 CPU fallback cases and offline Triton compilation pass; GPU execution unrun |
| [05_combined_with_expert_map.diff](candidates/05_combined_with_expert_map.diff) |01+02+04 | Packaging only |
| [06_local_alignment_prefix.diff](candidates/06_local_alignment_prefix.diff) | Align local expert buckets and enable existing Marlin sorted/inactive-prefix specialization after global-to-local mapping |420 exact-method CPU contract cases/840 instrumented GEMM calls pass; native kernels untested |
| [07_all_candidates.diff](candidates/07_all_candidates.diff) | All four mechanisms | Packaging only; not a proven strongest submission |
| [08_marlin_norm_overlap.diff](candidates/08_marlin_norm_overlap.diff) | Move the same pre-expert norm before the existing router join on non-SP CUDA Marlin |4,320 paired CPU cases and packaging pass; native scheduling/math/performance unproven |
| [09_selector_single_collective.diff](candidates/09_selector_single_collective.diff) | One smaller lossless collective in the released DFlash2 selector's candidate top-k |196,608 CPU protocol values/2,304 exact-function cases and offline Hopper compilation pass; native runtime validation pending |

Candidate 03 is 7,065 bytes across 3 files. The latest development checkout and
`current/patch.diff` match candidate 07, 12,186 bytes across 4 files. Longest added
line 86 bytes. Each of the seven frozen candidates
applies cleanly to the exact official base. No weight copies or external build
dependencies are added. The sampler intentionally sends16 bytes per rank/token,
versus stock's10 bytes at BF16: one fewer collective is a latency hypothesis,
not a bandwidth reduction or measured speedup.

Validation evidence:

- 65,540 exact Marlin policy cases through16,384tokens, EP1/2/4/8; EP1 unchanged.
- 786,432 CPU transport/selection columns: exhaustive FP16/BF16 encodings,
 random FP32 encodings, NaNs/infinities/signed zeros, non-power-of-two TP sizes,
 first-rank ties with nonmonotonic IDs and token IDs above2^54.
- 360 exact sampler-class CPU flow cases with fresh inputs, shrinking/growing
 batches, output pointer reuse, TP1/2/3/4/8 and FP64 fallback.72 zero-batch cases
 reproduce the same stock/candidate rejection; zero batches are not supported.
- The new remap wrapper passes 480 CPU fallback cases. The local-prefix method
 passes 420 CPU route-contract cases with 840 instrumented GEMM calls. These
 tests do not execute native alignment or Marlin arithmetic.
- All seven patches pass source-path/mode/size/line limits, isolated-index
 application and Python syntax checks. Reports: CPU_PREFLIGHT_2026-11-09.json,
 EXPERT_MAP_CPU_2026-11-09.json, LOCAL_PREFIX_CPU_2026-11-09.json and the three
 PATCH_MANIFEST reports.

CPU tests use Torch2.15 and emulate the two small Triton operations and the
collective. They do not execute GPU code or validate the deployed image. The
developer GPU component script uses real Triton/PyTorch NCCL and graphs but is
unrun; it still cannot replace full SGLang/serving validation.

The new [native MoE validation harness](NATIVE_VALIDATION.md) is also prepared
and GPU-unrun. Its CPU plan applies four isolated/combined MoE variants,
extracts ten exact methods including stock, and confirms identical finalization
AST hashes. It plans real NVFP4 preparation, native projections and finalizers,
poisoned inactive slots, dense references, fresh-input graphs and paired timing.
The pinned upstream unit test's EP-policy assertions conflict with candidate06;
that static incompatibility is documented, not hidden or called a GPU failure.

## Next execution steps

1. Recheck current rules/opening and available serving checks. With the current
   evidence, check candidate 01 first; use the stock comparison supplied by the
   official run. Record the exact handle and poll it to terminal.
2. Check candidate06 and candidate09 independently. Candidate02 affects the
   plain greedy sampler, bypassed by the released DFlash2 selector; inspect
   runtime path evidence before spending a check on that fallback. Check04 when
   budget or existing development hardware permits; no combination is a winner
   until measured. The first-day sequence01/06/09 uses the three-check limit.
   Reject failed quality/TTFT paths and preserve their evidence. Component wins
   need not translate into full-model wins.
3. Use the1/day scored budget for the strongest validated patch, then independently
   repeat its full-node result. Archive all results and compare the actual loads.
   A short check covers only one rung: all-load quality/TTFT/performance remain
   unproven until the full scored run returns. Do not mistake that check for
   complete validation, or make cosmetic resubmissions to fish for scores.
4. Profile the remaining bottleneck before expanding the patch. Higher-impact
   research directions: acceptance/cost-aware prefix verification with the same
   released 8-position drafter; single-layout expert fusion without duplicate
   model weights; communication overlap that preserves state/stream correctness.
   These are unimplemented hypotheses, not novelty or performance claims.
   The [Humming feasibility audit](HUMMING_FEASIBILITY_2026-11-09.md) found the
   base already declares Humming0.1.12. Its exact source/API was inspected
   separately from current upstream: no package install or execution.
   Next larger-kernel experiment is that pinned backend with unchanged BF16/
   FP32 precision and Muse normalization, avoiding duplicated model layouts.

An [isolated Humming adapter](HUMMING_PROTOTYPE.md) is now prepared outside
all submission diffs. It rejects Stream-K because the pinned implementation
adds BF16 partial outputs, unlike stock's FP32 reduction. It keeps packed FP4/
FP8 scales and stock finalization, and uses separate route tables if G1/G2
block sizes differ. 96 CPU protocol cases and 8 mock preparations pass against
the pinned API shapes; two forbidden precision configurations are rejected.
Real library/GPU execution and native checkpoint integration remain unproven.

The new [optimization library](OPTIMIZATION_LIBRARY.md) records implemented
mechanisms, precise public sources, rejected workspace reuse (graph-lock
alias/deadlock risk), constraints and prioritized next work. The public review
includes nine selected project trees plus upstream Humming source/archive
screens and two upstream vLLM files; downloaded does not mean fully audited.
No peer harness was executed, and external performance is not our score.

No rented hardware has been ordered. An asynchronous question about existing
development hardware is pending; work assumes official runs only unless the
owner supplies an existing machine. That question does not block source work.

The old isolated MoE binding remains non-integrated. Its weight duplication,
post-norm contract and old baseline prevent direct full-model adoption. Its
kernel-track timings are not serving throughput evidence. See MOE_PORT_STATUS.md.
