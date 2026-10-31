# MoE phase/scheduling follow-up

Canonical source remains exact alygg77 v24, SHA-256
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`,
scored `ce241acf7ab84c04`,2.372309x. The preceding decoder goal turn made
progress: a complete H200 experiment rejected the register/GS4 combination.
That result changes our next action; do not duplicate that family on MoE.

## Fresh upstream and component review

`COMPONENT_REPO_HEADS_2026-10-31.json` records fresh public heads. Adopted
alygg776cae49a, vjeuxf35a41e and Madhavan721c627 are still the current
default-branch heads. There is no newly published revision of those
kernels to claim or copy. Other branches/private repositories are outside
this check. This supplements the earlier broad account/code searches.

Fresh indexed code search returned217 unique paths/13repositories, versus
203/12 in the previous capture. Fourteen new paths all belong to
[Kushagrabainsla/muse-spark-open](https://github.com/Kushagrabainsla/muse-spark-open/tree/0778ecd8293e375ffe33faa466db0231929aaac3):
eight MoE submissions, one local benchmark, and five decoder paths left
for that track's later review. All nine downloaded MoE files match their
indexed Git blob hashes at commit0778ecd. Headers, bit layout, config,
prepare/dispatch and selected native branches were read, not executed.
Root license and file license were not established; source remains in
temporary research storage and no implementation was copied into this
repository's submissions. URLs/hashes/review scope are preserved in
`KUSHAGRA_NEW_SOURCE_SCREEN_2026-10-31.json`.

At07:48:45UTC this owner is#703/1225,1.7036x, ours#157,2.3723x. Its split-K
GEMM1/GEMM2/final-norm design, mixed-block scale sharing, activation reuse,
per-CTA routing and WGMMA component are useful comparison points, but the
current fused RS-WGMMA/KPERM3 parent already addresses the major listed
intermediate and atomic costs. The author's H100 vector-atomic/fused-norm
wins regressed on H200; those external measurements are scope-specific,
not our own controls. No new additive gain is established by this review.
This is not a blanket compliance endorsement: for example, the inspected
prepare stores a precomputed FP32 norm-weight expression, whereas our
canonical retains the supplied BF16 norm weights. No such storage change
was adopted.

Two additional component libraries are pinned below, with source files,
root licenses, exact commit/file hashes and original URLs in each
`PROVENANCE.json`. None was executed or imported into the submission.

### b12x

[local-inference-lab/b12x](https://github.com/local-inference-lab/b12x/tree/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88),
commit2cc7f66, Apache-2.0. The repository is archived. Seven selected
files are retained under `research/public/b12x_2cc7f66/`.

- The W4A16 path explicitly retains BF16 activations, so it is a meaningful
  numerical-contract lead. It is incorrect to exclude the entire library
  merely because other paths use FP4/FP8 activations.
- Its kernel's native-NVFP4 BF16 scale path calls the explicitly
  SM120-targeted `fp8x4_e4m3_to_bfloat2x2_native_sm120`. The helper emits
  `cvt.rn.bf16x2.e4m3x2` and requires an architecture-specific target.
  This is not directly executable H200 source. A complete Hopper port is
  unimplemented/unbenchmarked; no claim that every portable variant loses.
- The generic E2M1 decoder widens four values through masks, shifts and
  ORs. It does not establish an operation-count improvement over the
  specialized eight-value DQ7/previously tested eight-operation family.
  No new decoder GPU check is justified by counting this source alone.
- Its execution-model distinction is useful: availability/readiness and
  load balance are different scheduler concerns. A materialized queue may
  only balance work; an incrementally published queue must also enforce
  dependencies. Our parent already publishes phase-B stage readiness.
  Substituting an ordinary dynamic queue would not eliminate that need.
- Its optional prefill fused reduction reduces scratch and atomically
  accumulates BF16-rounded route contributions into FP32. That is a
  different output/norm contract from this task's per-expert norm before
  route combination. No throughput claim is made by that scratch result.
  Our parent already fuses the final expert-output normalization/sum.

### DistMoE

[meta-pytorch/dist_moe](https://github.com/meta-pytorch/dist_moe/tree/285763b5a0e01059a62c96b0990754a60a6ed94a),
commit285763b. Root BSD-3-Clause and bundled Apache license are retained
in `research/public/dist_moe_285763b/`. Four selected files are pinned.

The NVFP4 design quantizes source activations and uses `tcgen05` FP4xFP4
with TMEM accumulation, plus quantized W2 inputs. That execution path is
incompatible with the H200/BF16-activation contract. Its generic scheduler
file is still useful research: static/dynamic assignment, a shared work
record relay, and explicit readiness/lifetime protocols. A relay does not
publish another warp's stores or replace the existing per-stage credits.
No kernel port or performance claim follows from inspecting it.

## Scheduling literature and candidate boundaries

[Stream-K](https://arxiv.org/html/2301.03598v1), Osama/Merrill/Cecka/Garland/Owens,
2023, CC-BY4.0, section5.2: full-domain iteration balancing can create
cache skew; hybrid schedules restrict split work and combine it with
whole-tile waves. Its two-tile variant hides partial-sum handling better
than splitting only a small final wave. AppendixA.1 models reduction costs.
These are dense-GEMM results, not a measured speedup for routed,
heterogeneous, dependent SwiGLU phases. Current code already uses
cost-weighted Stream-K and phase-A partial reordering; merely adding the
name or a copied cost table is not new work. See also the scheduling
reading index for Twill's separate applicability limits.

Remaining questions are concrete: how many current cut points split
units, where phases end across CTAs, and whether reducing split work could
outweigh the additional work imbalance and changed readiness order.
Nearest-unit rounding, arbitrary whole-unit tails and new cost tables are
not assumed faster. Public dynamic-tail/refit/fence negatives remain valid
within their scopes. No alignment implementation is being promoted from
an analytical model alone.

## V27 current-parent trace protocol

Check `f48193f0c16044b9` uses
`submission_v27_phase_trace_probe.py`, SHA-256
`1bd537e503233b5ba994fc96b28ca9d904e245ee2008a1c7c463fa300f733042`.
Its exact parent prefix and timed `kernel_fn` are unchanged. Only an
independent prepare-time module enables the parent's existing eight debug
clock slots. The idle x-producer lane exports its existing job roster and
stage ranges. No clocks/stores are inserted in the dequant/MMA stage body,
but the build can still change register allocation or scheduling.

Twenty-four cold-L2 graph comparisons coverM32/128/256, two seeds, four
retained scratch pads and five alternating rounds, with two parent graphs.
Weights are shared between pads; these are four scratch layouts, not four
independent weight allocations. Initial and fresh activation/route/weight
replays must agree within.002relativeL2/.01normalized row. Full132-CTA
clock/range/job traces are retained only forM32/256, seed3101,pad0. M128
has performance/correctness controls, not a full trace. Each trace is one
representative replay, not a distribution or a hardware-counter capture.

The [PTX globaltimer documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)
defines the nanosecond clock. Its observation overhead remains part of the
instrumented build. Parent-relative ratios must stay within.95–1.05 in
every group and local allocation must stay zero before even coarse timing
attribution. Passing that gate still does not make timestamps exact parent
durations or identify a causal stall. Metadata can be used separately.

The first CPU stress layout produced18,388 characters; a varint revision
still produced12,456. Both were rejected before any GPU check. The final
lossless layout uses exact uint32 nanosecond timings, exact FP32 error
metrics, FP64 ratios, per-column bit-packed clock deltas and uint16 cuts.
It roundtrips all24rows/2traces/2,112clock values in10,768characters plus
tail headroom. Evidence: `PHASE_TRACE_CODEC_CPU_2026-10-31.json` and
`tools/check_moe_phase_trace_codec.py`. No raw rounds or precision were
dropped to shrink the log. Absolute 64-bit clock origins are retained.

Phase-duration/imbalance models remain planning estimates. Main-path
phase-zero headroom exceeds3%; this check is for establishing current
loss budgets and candidate scope, not a scored optimization. No unchanged
source resubmission is planned.

## V27 terminal result and compiler repair

V27/checkf48193f0c16044b9,sfo3,passes5tests/6timings at2.372579771x,
5668.303us,CI2.336991–2.408168,prepare216.580s. Probe51.597s,complete9660
character report,all24groups/2traces. Initial and fresh agreement is zero;
A/Aoffset<=.9972%. However instrumented-to-parent speedup ratios are
.881529–.896613 atM32,.889420–.916980 atM128,.907538–.931651 atM256.
Both native BIG instantiations report96registers/**296localbytes**.
The timing-attribution gate fails. Preserve its clocks as perturbed data;
do not report its6.656/11.008us end gaps as current-parent imbalance.

Four standalone NVCC13.0,explicitsm90a,fixedNTT8/BIGtrue compiler responses
are archived as `TRACE_CODEGEN_*.json.gz`. They are reduced bodies,
not the judge extension; SASS is clipped at5000lines.

1. V27 reproduces296byteframe,60spill-store/64spill-load bytes.
2. Initial shared-pointer alias attempt does not compile: the WS-only
   member is absent in a WGS build. Caught before a GPU check; do not
   conceal the failed attempt or claim the absent field is reusable.
3. A dedicated shared pointer with volatile reloads alone leaves288byte
   frame,52store/48load bytes. The live debug pointer is not the entire cause.
4. Metadata reads of `p.dbg` inside the outlined range helper require an
   addressable copy of the whole by-value Params object. Reading the
   dedicated shared pointer instead makes that formal Params argument
   unused again, as in the parent. Now the reduced compile has
   **0frame/0spill stores/0spill loads**,96registers. The full H200 build
   and timing controls below independently confirm the repair's effect.

The final V28 file is immutable SHA
`678ec161b8552570fad648d04ad1af4addfa4d4b73fc73afa74aac51ca8aff91`.
The H200 comparison protocol/codec/fixtures are AST-identical toV27. Its
computational loop is unchanged. The new pointer fits the existing shared
allocation's rounding: dynamic shared remains231,168bytes and576threads.
This is a diagnostic repair, not an optimization of the already zero-local
canonical source. Full requests preserve the unsubmitted failed versions.

## V28 terminal result and useful evidence

Check **f8208a2105ab4940**,sfo3,passes5tests/6timings at2.361522034x,
5694.844us,CI2.326099–2.396945,prepare187.263s. These official timings
still measure the parent. Probe44.295s; complete9080-character report.
Both full BIG variants now have96regs/**0localbytes**. All24comparison
groups pass the5% perturbation gate:

|Tokens|Minimum / median / maximum parent-to-trace ratio|
|---|---:|
|32|.993303 / .998764 / 1.003081|
|128|.992691 / .998699 / 1.001808|
|256|.993068 / .995984 / 1.002879|

All initial/fresh relative errors are zero; A/Aoffset<=1.1405%. This
permits coarse phase investigation, not exact parent timing attribution.
The full clock traces are only two individual replays of one seed; use
neither untracedM128 nor all nearby routings as inferred observations.

|Observed fixture|Mean phaseA|Mean phaseB|PhaseB absolute end max−mean|Clock span|
|---|---:|---:|---:|---:|
|M32,seed3101|79.703us|42.368us|2.816us|135.040us|
|M256,seed3101|120.024us|60.107us|4.352us|200.704us|

All12profiles,complete local/remote logs,48rawcomparison groups and four
full clock/range records from these two checks were read/decoded and
archived. Receipts and `PHASE_TRACE_<job>.json` preserve all measurements.
No canonical source/scored submit changed. MoE budget2/8checks,zero submits.

## Candidate cost screening

`UNIT_PARTITION_MODEL_<job>.json` uses exact route/cut metadata and the
current integer cost table, intentionally independent of clock accuracy.
Both runs have identical rosters/cuts for the two selected fixtures.

|Fixture/phase|Original split boundaries|Remaining after±4-stage alignment|Largest-CTA weighted-work factor|
|---|---:|---:|---:|
|M32 A|120|60|1.057143|
|M32 B|120|60|1.114286|
|M256 A|115|54|1.055361|
|M256 B|123|60|1.142857|

Eliminating every split by nearest-unit rounding raises that work factor
to1.142857/1.371429 atM32 A/B and1.156627/1.220779 atM256. Offline
longest-processing-time-first whole-unit assignment still gives
1.142857/1.371429 and1.068781/1.209235. In the homogeneousM32 roster,
the whole-unit result is the exact scalar-model minimax lower bound from
indivisible tile counts. It is not a hardware-time theorem. These models
show why fewer fixups alone cannot be assumed faster; they omit readiness,
locality, actual reduction costs and a device planner's overhead. None
is a native implementation or measured speedup. No blind rounding/LPT
GPU sweep is qualified by these numbers.

`PHASE_SCOPE_MODELS_f8208a2105ab4940.json` keeps the same synthetic fixture
parent event timings (including graph overhead), with four other loads
held unchanged. Hypothetically subtracting the observed phaseB end gaps
models only1.006944x across six loads. Removing all observed non-stream
mean durations models1.019103x. A20% reduction of mean phaseA instead
models1.041475x. These are planning proxies, not official-input board
bounds, measured critical-path removals or promised gains. They prioritize
work throughout the phase-A stream over another small tail tweak.

Next useful MoE experiment: retain this validated zero-local diagnostic
design, sample a few CTAs' per-segment/unit-boundary times with bounded
output, and distinguish whole-unit epilogues from split fixups and deferred
flushes on the current KPERM3 base. Require resource/perturbation controls
again; adding clocks while accumulators are live can change codegen.
Only a distinct mechanism with a credible>=3% whole-board model should
advance to candidate dispatch, held-out/fresh checks and independent
whole-kernel repeats. Keep prior fence/ring/tail/register/dequant negatives
closed. The rank1 goal remains open.
