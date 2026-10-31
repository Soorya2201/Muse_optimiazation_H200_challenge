# Main-loop compaction and broader public-source discovery

The preceding goal turn made progress: V30 established spill-free full native
flush profiling with passing numerical and perturbation controls. It found
no practical >=3% publication-only lead in the two sampled synthetic loads.
This cycle investigates the main stage loop instead. Canonical remains
`submission.py`, SHA-256
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`,
scored `ce241acf7ab84c04` at 2.372309331x. Rank1 remains unachieved.

## New branch, PR and gist coverage

`BRANCH_GIST_DISCOVERY_2026-10-31.json` preserves paginated public metadata
for all 13 repositories in the earlier MoE indexed search: 26 branches,
21 tags and 169 pull requests. It also screens public gist metadata for
43 owners: the current MoE top30 and the indexed repository owners.
42 accounts resolve, containing 188 public gists; `Abarbir12` returns404
and remains unresolved. Description/filename screening, including CUDA,
Triton, GEMM, quantization and challenge names, finds no relevant gist.
No gist contents were downloaded. This does not establish that all public
gists or private/unpublished submissions everywhere were inspected.

`ROHAN_BRANCH_PATH_SCREEN_2026-10-31.json` preserves all MoE paths from
14 distinct Rohan branch/PR17 heads, with recursive-tree truncation flags
all false. Comparing their blobs to the earlier default-branch index
finds no additional candidate submission Python source. New-to-that-index
blobs are evaluation/spec/starter metadata, not new implementations.
The Claude cohort, Gemini cohort and PR17 head do not expose a stronger
unseen MoE source. Ozgur's additional branch/PR concerns BF16 and is not
treated as a MoE port. PostTrainLLM's published 1.2503x result is also not
evidence of a stronger MoE implementation.

## Component literature and pinned sources

Exact commits, original URLs, Git-blob identities, SHA-256 hashes and root
licenses are preserved in each directory's `PROVENANCE.json`. Nothing
below was imported/executed or copied into the timed submission.

|Source|Pin / license|What was inspected and what remains|
|---|---|---|
|[FLUTE](https://github.com/HanGuo97/flute/tree/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761)|`research/public/flute_9eb83a1/`, six files, Apache-2.0|Kernel/config/conversion/packing sections: BF16 support, vectorized pair lookup and duplicated shared tables. A task-compatible integrated LUT port is unimplemented.|
|[qgemm-mx](https://github.com/titoatwork/qgemm-mx/tree/f83c8511e49fa7fa7feef960fc1cc2e8b7588513)|`research/public/qgemm_mx_f83c851/`, six files, MIT|README/status/device decode/layout headers. Published status says Hopper WGMMA/TMA performance kernel is not implemented; its reference kernel is correctness-only. Not a faster replacement.|
|[Unsloth PR13840](https://github.com/unslothai/unsloth/pull/13840)|`research/public/unsloth_nvfp4_ea03b13/`, two files at PR head `ea03b135b5202634054a116001b1105986bcdd72`, Apache-2.0|Tiled in-GEMM dequantization, exact-bit E2M1 upcast, FP32 accumulation and conservative per-GPU dispatch. Its published dispatch has no H100 win and no default SM90 gate. Current fused parent already avoids the full dequantized-weight temporary.|
|[Metrale](https://github.com/SeedSource/metrale-inference/tree/f7b8b4d70fd23f50079e6f5a7716c3d5c8025f72)|`research/public/metrale_f7b8b4d/`, five files, MIT OR Apache-2.0|The selected GEMM uses BF16 MMA and scalar LUT decode. Separate MoE GEMV stages have a different shared-expert/output contract. `e2m1_branchless.cu` is the opposite conversion, FP32-to-FP4 quantization; it is not a faster decoder and must not be adopted for activations. No compatible faster Hopper port established.|

The initial guessed Metrale repository URL404 was resolved through public
repository discovery to `SeedSource/metrale-inference`. Its root uses
`LICENSE-APACHE`/`LICENSE-MIT`, not `LICENSE`; that initial404 was repaired
and both exact licenses are preserved. No partial, unlicensed pin is claimed.

[FLUTE's paper](https://arxiv.org/html/2407.10960v2), Guo et al., 2024,
explains pair lookup, table duplication, offline packing and Stream-K.
Its benchmarks use A100/A6000, other quantization/group settings and other
baselines. Those headline gains do not predict a gain over this H200
NVFP4 fused kernel. The paper's integration does not use CUDA graphs;
this does not prove an adapted kernel cannot be captured.

Source-level applicability: one BF16 pair table is256*4=1024 bytes; the
source's 8/16/32-copy modes require8/16/32 KiB for the duplicated pair
tables alone. Our main shared allocation is231168 bytes, leaving1280
below the existing232448-byte block cap. One copy can fit arithmetically,
but duplicated modes need a storage/lifetime redesign. Lookup also adds
address calculation and dependent shared reads, whereas the current DQ7
is specialized for fixed E2M1. Neither fewer issued instructions nor a
speedup is established by that comparison. No blind LUT GPU check.

## A distinct main-loop hypothesis

The current `wgs_seg` specializes one complete first-stage body and another
steady-stage body. The first body selects `mma0` and runs a deferred flush;
both otherwise perform the same 16 k16 operations. A shared stage body
could reduce instruction footprint. This is separate from the closed
GS4/register-only, ring-depth, fence and tail experiments.

NVIDIA's [H100 instruction-cache case study](https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/)
shows why excess unrolling can hurt. It also distinguishes hot-code
footprint from total executable size and warns that reduced unrolling can
cost scheduling/occupancy. We have no instruction-cache counters for this
MoE, so cache starvation is a hypothesis, not a diagnosis or speed claim.

`tools/moe_stage_compact_variant.py` preserves two unsuccessful compiler
approaches and the revised candidate:

1. A runtime `FIRST` merges the first/steady loop. Reduced NTT8 compilation
   shrinks code but adds8frame/4spill-store/4spill-load bytes and three
   C7519 compiler-injected warpgroup-arrive warnings. Not sent to H200.
2. One extra zero-A `mma0` initializes async accumulators before a uniform
   accumulation loop. It removes the conditional accumulator merge without
   scalar accesses to async accumulator registers. At the original budget,
   NTT8 still has24frame/20store/40load bytes. Not sent to H200 unchanged.
3. Combine that initialization with the existing REG112 producer/consumer
   redistribution. This is a new loop architecture plus a matched REG112
   control, not a retry of redistribution alone. Isolated narrow/wide cases
   compile with zero frame/spills and no C75 warnings.

The candidate still multiplies every actual weight/activation at the same
precision and preserves the original accumulation order after a zero
initialization. It adds one WGMMA per segment: +1/256 operations for a whole
16-stage unit, up to+1/16 for a one-stage partial segment, plus an initial
activation-slot wait/fence. These costs may outweigh any code-size benefit.

## Compiler evidence and preflight limits

Eleven complete NVCC13/sm90a requests/responses are archived as
`STAGE_CODEGEN_*.json.gz`. The new probe instantiates only phase-A
`wgs_seg` with arbitrary segment bounds/pending metadata and matched launch
bounds. It omits routing, producers, phaseB and full-dispatch instantiations
and is never executed. Unlike the earlier whole-kernel probes, these SASS
listings are complete and untruncated. Counts are retained in
`STAGE_CODEGEN_ANALYSIS_2026-10-31.json`.

|Isolated REG112 case|Parent instructions|Candidate instructions|Parent/candidate static HGMMA|Frame/spills|
|---|---:|---:|---:|---|
|NTT1|1061|863|32/17|Zero both|
|NTT8|1475|1202|32/17|Zero both|

Static instruction count drops18.66%/18.51%. This is not dynamic work
reduction: the candidate adds an MMA dynamically, and the original already
loops over a hot steady body. It does not establish a full-kernel cache
benefit, resources, runtime or a new score. Final generated native prefixes
match these compiler requests exactly, excluding the new host resource getter.

The prior same-job model shows >3% phase-zero headroom and a conditional
20% phase-A cut gives1.041475x in two synthetic loads. That motivates one
controlled structural experiment; it does not predict20% improvement from
an18.5% static code-size reduction. A production candidate still needs
actual repeated >=3% board-model evidence and full checks before promotion.

## V31 independent H200 experiment

Check `c364de8b14af4f36` completed on `sfo3`: five challenge tests and
six timing cells pass, 2.375219067x, 5662.004us, CI2.339590781–2.410847353,
prepare265.245s. Fifth of eight reserved MoE checks, zero submits. Immutable
source `submission_v31_stage_loop_probe.py`, SHA-256
`0453862337d3ba2f04535c470ed1edc66c37207cb33f6839b6b02a7783a11363`.
Timed `kernel_fn` remains canonical; the point score is not a gain.

The full99.821s independent diagnostic retains all3nearby fixtures and24
four-arm timing groups in3500characters. Both candidate/control, both BIG
instantiations:96static registers, zero local bytes,640threads,231168dynamic
shared bytes. Initial, fresh-input and nearby output agreement errors are
zero. A/A maximum median offset1.6671% passes the chosen2% gate. Every raw
round, all6profiles and full available logs were read and archived.

|M|Canonical/candidate speedup range|Same-REG112 control/candidate range|
|---|---:|---:|
|32|0.944454–0.962790|0.941493–0.956787|
|128|0.910065–0.935259|0.911458–0.936466|
|256|0.908642–0.922796|0.903601–0.920915|

The candidate takes approximately4–10% longer. All7nonempty changed-load
policies reject. Even the least bad median board proxy is0.991248x; enabling
all three is0.964284x. Static code-size reduction did not produce a runtime
gain. No cache counters were captured, so neither a cache benefit nor a
specific causal stall is inferred. Canonical/scored code remains unchanged.

## Bounded initialization follow-up

Before closing the approach, two compiler-only checks separate the
initialization issue:

- Runtime `FIRST` plus REG112 removes its spills but retains three C7519
  injected-arrive warnings. It was not sent to H200.
- Scalar zero initialization plus an explicit fence removes the extra zero-A
  MMA and initial activation-slot wait. Contrary to the blanket concern in
  an older source comment, this isolated loop structure compiles with zero
  frame/spills and no C75 warnings at NTT1/8. NTT8 SASS contains16HGMMAs,
  and the32accumulator zeros lower to16CS2R instructions. This is reduced
  compiler evidence, not proof about the complete kernel or runtime.

The original scalar source and its comment-corrected final NTT1/8 requests
are both preserved. No code-generation result was discarded or overwritten.
The final native prefix exactly matches the final compiler request.

V32/check `72dff8caec5047d8` completed on `sfo3`: all5challenge tests
and6timing cells pass, 2.361042674x, 5696.000us, CI2.325627034–2.396458314,
prepare264.468s. Immutable `submission_v32_scalar_stage_probe.py`, SHA
`ff0ca7c6116e0f4103d9b35019fc5ac9ba51af089b0e86df290aab26bb6fcde8`.
Sixth of eight reserved MoE checks, zero submits. Timed dispatch is still
canonical, so this score is not a new algorithm result.

The complete97.599s independent diagnostic again retains3nearby fixtures
and24timing groups in3500characters. Fullcandidate/control attributes are
96registers/0local/640threads/231168dynamicshared in both BIG modes.
All initial/fresh/nearby agreement errors are0, A/A maximum median offset
is1.27887%, and all numerical/resource/stability gates pass. All6profiles,
full available logs and24raw timing groups were read and archived.

|M|Canonical/candidate speedup range|Same-REG112 control/candidate range|
|---|---:|---:|
|32|0.955689–0.970549|0.957852–0.970581|
|128|0.921098–0.933850|0.917088–0.936376|
|256|0.919313–0.930443|0.916247–0.928704|

The repair remains approximately3–9%longer, including against its matched
REG112 original-loop control. All7changed-load policies reject; best median
board proxy0.993269x, all-three0.968380x. Cross-job comparisons betweenV31
andV32 are not a causal estimate of the initialization change. Removing the
extra MMA/slot wait did not reverse the regression.

Close these two uniform-loop implementations after the two independent GPU
negatives. Keep canonical; no scored submit or source promotion. No GPU job
is live. RuntimeFIRST/REG112 is only a compiler result with extra ownership
fences, not a measured loser or a new qualified performance lead. Reopening
needs new evidence, not a parameter sweep. The smaller static footprint
alone does not establish a cache advantage, and fullnative zero-local status
rules out the reduced-probe spill problem as an explanation for these full
candidate slowdowns. Loop scheduling/initialization and other costs remain
unisolated without hardware counters or further justified ablations.

All14nonempty per-run load combinations were rejected. These use same-job
ratios and six equally weighted loads; they are synthetic planning models,
not official scores or a cross-job jointly changed-dispatch result.

## Exact payload arithmetic for future prioritization

`WEIGHT_PAYLOAD_MODELS_f8208a2105ab4940.json`, reproduced by
`tools/model_moe_weight_payload.py`, uses V28's full132-CTA metadata and
same-job controls. These two fixtures each have one job per active expert:
18 atM32 and26 atM256. Every stage transfers36864weight bytes. Exact
W13/W2 payloads are339738624/169869312bytes and490733568/245366784bytes.
The payload divided by raw control graph-event time is3.59935/3.58640TB/s.
This is an end-to-end payload quotient, not measured isolated DRAM bandwidth.

At an assumed4.3TB/s, moving only those weights takes118.51/171.19us,
versus raw control events141.58/205.25us. The two-fixture, other-four-loads
unchanged planning proxy is1.061717x; nominal4.8TB/s gives1.101370x. Both
omit all compute, other traffic, synchronization and launch costs. The
empirical stream rates are not hard hardware ceilings, these are not
board upper bounds, and M128/tiny loads are not inferred. They are a useful
constraint against assuming an arbitrary20%main-phase improvement will be
available from a code-size change alone. Small-load phase accounting is
still a distinct open measurement direction for a later MoE cycle.
