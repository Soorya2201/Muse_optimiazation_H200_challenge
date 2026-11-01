# BF16 projection optimization research library

Nov1 latest: [Hopper inline-compression experiment and full result](ILC_RESEARCH_2026-10-31.md).
V48failedbuild usingC++17; repairedV49/checkbd4aecd37f164283 compiles20,
passesparent9tests/30timings andindependent6bytechecks/32groups/1024exact
event values/608metrics. HardwareVMM+compression support/actualflags pass,
all outputagreements0. No matchedcomponentgroup gains2%;all8cells and255
fixedpoliciesreject. AAmax2.1059%alsofails2%,bestchangedworstconditional
proxy1.000598461x. No causaltraffic/compressionratio ornewscoredclaim.
CloseeightlargeBF16cellallocationform;canonicalunchanged,no submit.
Nov1BF16routine1/8afterreset,others0;Oct31BF16ended5/8. Allprofiles/logs/
rawevidencearchived;externaldriveabsent,internalbackupauthoritative.
Nextdecoder. Newsourcepins/limits: [NWC,ZipServ,NVIDIA](LOSSLESS_SOURCE_REVIEW_LATE_2026-10-31.md).

Oct31 late [global exponent normalization](GLOBAL_NORMALIZATION_2026-10-31.md)
is now terminal. V43/check3aa502d1eb8f4854 passes9tests/30timings with the
unchanged timed parent1.557816x. The actual new SK2 path passes5nearby and
72independent groups with zero initial/fresh differences,all old/new outputs
bit-identical,and zero full-native local allocation. Despite a12->7 decoder
core reduction,no route/511fixed policies qualify;best worst-group board
model.999847554x. Full raw times/resources/numerics,30profiles and available
logs inspected. No scored change;BF16 routine checks4/8,submits0. Close
this tested normalization form. Newly pinned BSD-3-Clause
[Unweight source](https://github.com/cloudflareresearch/unweight-kernels/tree/db0e6d3d4505cf71379f6b9b4567acf598bc186c)
and our independent byte/geometry screen supply no stronger direct port.
Next sequential track decoder;the dated report preserves the complete result.

Oct31 terminal update: V39/V40/V41 all pass the unchanged timed parent;
all nine-route transfer policy sets reject. The final read-flush run
completes 72 groups with best worst-group board proxy 1.000443x and no
eligible route. No promotion or score gain. The transfer is closed; see
[terminal method/results](DEVICE_EVENT_METHOD_2026-10-31.md) and
[public-source audit](PUBLIC_SOURCE_FOLLOWUP_2026-10-31.md). All 90 profiles
and 216 independent timing groups were inspected. Current BF16 score stays
1.535843x; Oct31 routine checks 3/8, submits 0.

Oct31 continuation overrides historical score/queue labels below: canonical
is exact alygg77 v34,1.535843x. Read
[C12 layout transfer](C12_PERMUTE_RESEARCH_2026-10-31.md) for the current
bounded experiment and its terminal outcome. It transfers an already
embedded MIT decoder's sign permutation to nine older SK2 routes,retaining
precision and pipelines. CPU223904cases prove identical fragments;
standalone sm90a decode12to9instructions is not a runtime gain claim.
The old/new synthetic comparison shares one weight pointer and uses
actual/matched/duplicate controls,nearby shapes and FP64 references. No
canonical promotion is implied by the diagnostic parent's point score.
The exact Chen/Ali controls are complete. TMA v38's configurations and the
cluster/compression negatives remain closed. Fresh public code index244
paths/13repos,including five new BF16 files atKushagrabainsla0778ecd,
does not identify a stronger replacement;links/hashes and static review
are in`KUSHAGRA_SOURCE_SCREEN_2026-10-31.json`.

Last updated: 2026-10-29. This is a living, evidence-ranked library for the
`muse_bf16_proj` track. It covers legitimate, CUDA-graph-capturable methods that preserve BF16
weights exactly (or use the task's allowed FP32-router hi+lo representation). It is not a claim
that every published GPU technique applies to this workload. Methods are ranked against the
actual H200, cold-L2, six-load judge contract.

## Goal and constraints

- Current scored result: **1.521867x**, job `d1bd752954ff4d21`; exact Nathen v18 is promoted.
- Current leader at the latest refresh: **1.6367x**.
- Score: geometric mean of six equally important load ratios. Each load contains qkv, o,
  pooler, unpooler, and router calls weighted 62 times.
- Every replay starts with cold L2. Decode/verify-small projections are primarily weight-stream
  problems; cache-dependent tricks do not survive the judge protocol.
- The timed function must be CUDA-graph capturable and cannot cache input-derived work.
- Dense BF16 weights must remain BF16 or be compressed losslessly. Router FP32 may use the
  allowed BF16 hi+lo split with FP32 accumulation.
- The Mac host has no NVIDIA GPU. Local evidence is syntax, packing emulation, source generation,
  and NVCC/PTXAS resource analysis; real correctness and timing require OKC's H200.

## Evidence labels

| Label | Meaning |
|---|---|
| Judged | Passed OKC correctness and has H200 timing. |
| Public judged | A public repository includes an attributable OKC receipt/source hash. |
| Compiled | NVCC/PTXAS accepted the exact device source for `sm_90a`; no H200 runtime timing. |
| Emulated | CPU/PyTorch tests establish layout or exact reconstruction, not GPU performance. |
| Proposed | Supported by primary documentation or papers but not implemented here. |
| Rejected | Measured locally or publicly and not competitive for this workload. |

## Workload regimes

| Regime | Shapes | Dominant issue | Best architecture family |
|---|---|---|---|
| Tiny M | m=1,8 | Cold-HBM weight bytes, ramp latency, launch tail | Direct vectorized GEMV, lossless C12, exact sparse corrections |
| Small M | m=16,32 | Weight bandwidth plus enough reuse for Tensor Cores | Tuned RS/SS WGMMA or carefully sized split-K GEMV |
| Large M | m=128,256 | Tensor Core throughput, TMA/WGMMA pipeline, tail balance | Warp-specialized SS-WGMMA, clusters, tuned cuBLASLt fallback |
| Router | all M | Exact FP32-equivalent accuracy at small N=256 | BF16 hi+lo, FP32 accumulation, fused or single-pass reduction |

No single kernel family is expected to win every row. The strongest public v18 source uses
different routes by shape and layer, and our candidates preserve that portfolio approach.

## Technique catalog

### 1. Backend portfolio and shape-local autotuning

**Method.** Prepare several legitimate implementations, capture each with CUDA Graphs, flush L2,
measure alternating replay order, and retain the fastest route for each broad shape/layer range.
Candidate families include cuBLAS/cuBLASLt, direct GEMV, WGMMA, TinyGEMM, and cluster kernels.

**Why it can win.** Library heuristics are excellent for ordinary GEMMs but not uniformly optimal
for skinny or cold-HBM shapes. Conversely, custom GEMV loses badly once M provides enough matrix
reuse. NVIDIA recommends querying and reusing cuBLASLt algorithms rather than paying heuristic
selection on every call.

**Evidence here.** Judged. v7/v10/v19 and public v18 use per-shape selection. v19 also guards
activation with a 1% and 0.08 us minimum win. v30 extends the candidate set without changing
nearby-shape fallbacks.

**Decision.** Foundation technique. Keep every new kernel behind correctness and cold-L2 gates.

### 2. Direct vectorized GEMV for M=1/8

**Method.** Give each warp contiguous output rows, vector-load weights, retain x in registers or
shared memory, accumulate in FP32, and coalesce the final BF16 store. Tune output rows per warp,
K unroll, warp count, CTA count, and vector width.

**Why it can win.** Tiny-M dense projections read far more weight bytes than activations or
outputs. Avoiding TMA setup and a general GEMM scheduler reduces ramp and tail latency.

**Evidence here.** Judged. v3 is the current scored kernel; v18's selected qkv/pooler/o routes are
more advanced direct/compressed descendants. v8 showed that merely increasing consumer warps can
help one shape and hurt another, so geometry must be shape-local.

**Decision.** Keep v18's selected routes. Future work should change bytes or overlap, not perform
another blind warp-count sweep.

### 3. Software-prefetched register pipeline

**Method.** Load the next weight group before decoding/MMA of the current group. Use two or more
register stages, bounded unrolling, and enough independent accumulators to hide HBM latency.

**Why it can win.** Direct global loads avoid shared-memory staging, but a dependent load/decode/
MMA loop exposes HBM latency. Software prefetch creates instruction-level overlap.

**Evidence here.** Public judged in Nathen v18; proposed in our simplified v14. Public evidence
reports material qkv/pooler/o gains when combined with C12 packing and a dedicated correction
path. Register pressure is the controlling risk.

**Decision.** High priority only through exact v18 first. Inspect PTXAS registers/spills for every
new geometry; v30 WR2 qkv32 and v31 WR4 pooler32 were removed after spills were proved offline.

### 4. Warp-specialized TMA plus WGMMA

**Method.** Producer warp(s) issue TMA loads into a multistage shared ring; consumer warpgroups
issue WGMMA; mbarriers manage full/empty phases. Choose ping-pong or cooperative scheduling,
stage depth, cluster shape, raster order, and swizzle.

**Why it can win.** Hopper TMA removes address-generation work from CUDA cores, while WGMMA uses
warpgroup Tensor Cores and overlaps transfers with MMA. CUTLASS Hopper examples use this design.

**Evidence here.** Public judged in v18 and compiled in v20-v22/v32. Our v10 stage sweep showed
that deeper staging is not automatically faster for tiny-M routes. v21/v22 specialize the proven
large-M qkv machinery for pooler/unpooler.

**Decision.** Highest large-M priority after exact v18. Measure the prepared shapes before
creating more stage variants.

### 5. WGMMA with padded M=64 for m=16/32

**Method.** Pad x to a legal WGMMA M tile, use SS-WGMMA or RS-WGMMA, and mask stores. Compare
M=64 tiles against direct GEMV and cuBLAS on each layer.

**Why it can win.** Tensor Cores can offset 2-4x padded arithmetic when the operation remains
weight-bandwidth-bound and reduces scalar reduction overhead.

**Evidence here.** Proposed. Generic Triton m16/m32 v9 was rejected at 0.9239x because it used an
ill-suited one-CTA-per-output-tile schedule. That result does not reject a Hopper-native
warp-specialized WGMMA kernel.

**Decision.** Medium-high priority after the reset portfolio is measured. Start with pooler m32,
then qkv m32; require a strict win over v18, not merely over our old v3.

### 6. Split-K with global workspace reduction

**Method.** Multiple CTAs compute K slices, write FP32 partials, and run a second reduction kernel
or a last-CTA semaphore epilogue.

**Why it can win.** It supplies enough CTAs for M=1 and large K.

**Evidence here.** Rejected as a general direction. v1 and public campaigns show that extra
launches, workspace traffic, counters, and an 8-9.5 us reduction tail erase the parallelism gain.
The Stream-K paper is useful background, but the challenge's tiny outputs and graph-adjusted
microsecond regime make its fix-up cost unusually expensive.

**Decision.** Do not retry global fix-up. Use cluster DSMEM or a no-reduction ownership schedule.

### 7. Cluster split-K with distributed shared memory

**Method.** Place K-split CTAs in one thread-block cluster. Send FP32 partials directly to the
owner CTA's distributed shared-memory mailbox with cluster-scoped asynchronous stores and an
mbarrier. The owner converts and writes the output once.

**Why it can win.** It keeps split-K reduction off global memory, eliminates the second kernel,
and gives deterministic bounded synchronization inside a resident cluster.

**Evidence here.** Judged in v3/v18 qkv routes; compiled in v20-v22/v32. V32's static audit proves
mailbox coverage, alignment, byte counts, and output ownership for the pooler-m128 prototype.

**Decision.** Core rank-one architecture. Prefer two-CTA clusters when they provide enough grid
parallelism without destroying residency.

### 8. No-reduction output ownership

**Method.** Assign each CTA or warpgroup complete output rows and the full K range, avoiding any
cross-CTA reduction. Increase independent N tiles to fill the H200.

**Why it can win.** Removes reduction tail and global/shared communication. It is attractive when
N is large enough to expose at least one wave of independent tiles.

**Evidence here.** Compiled in v20-v22 unpooler. Public C12 diagnostics ended within 1.5-2.2 us
without global split-K versus 8-9.5 us with it.

**Decision.** Preferred for unpooler and potentially qkv/pooler when N tiling exposes sufficient
parallelism. Measure v20/v21 before changing geometry.

### 9. Persistent work-centric tile scheduling

**Method.** Launch roughly one CTA per resident slot and pull tiles from a device queue, balancing
work by K iterations/bytes instead of static tile count. Keep producer and consumer roles alive
across tiles.

**Why it can win.** Reduces waves and launch tail, balances irregular tile durations, and can
retain pipeline state. Triton's persistent-kernel tutorial explicitly targets runtime drift;
Stream-K describes work-centric decomposition.

**Evidence here.** Proposed for dense projection kernels; proven useful in the project's MoE
track but not yet competitive here. The BF16 shapes are regular, so queue overhead may dominate.

**Decision.** Lower than C12-cluster. Revisit only if profiles show a large CTA completion spread
after reduction tails are removed.

### 10. Lossless 12-bit BF16 compression (C12)

**Method.** Preserve each BF16 low byte and encode sign plus a three-bit exponent offset in four
high bits. Store rare exact exponent corrections separately. Reconstruct the exact 16-bit BF16
pattern using `lop3`/`prmt`.

**Why it can win.** Reduces the dominant weight stream by nearly 25% while using otherwise idle
integer issue capacity. The task explicitly permits lossless compression.

**Evidence here.** Public judged in v18 at 1.518570x aggregate. Our v5/v6 proved that compression
alone is insufficient: scalar/shared decode regressed to 0.9951x/1.0316x. The emulator proves
v32's representation across all 65,536 BF16 bit patterns and patch boundaries.

**Decision.** Highest-upside technique. The decoder must be overlapped and the correction stream
must not force raw-block fallback or a serial tail.

### 11. Direct C12 decode into register fragments

**Method.** Stream packed groups directly from global memory, reconstruct WGMMA/MMA register
fragments, prefetch group j+1, and avoid a decoded BF16 shared copy.

**Why it can win.** Minimizes shared capacity/bandwidth and preserves most of the 25% HBM saving.

**Evidence here.** Public judged in v18 for selected small-M routes; our v14 is emulated but not
H200-tested. Public large-M experiments show register/shared C12 WGMMA can lose to uncompressed
shared/shared WGMMA because decode and global reduction become critical.

**Decision.** Use for tiny/small M. Do not assume it is best at m128/256.

### 12. C12 decode into shared memory, then SS-WGMMA

**Method.** A decoder warpgroup expands packed C12 into a swizzled BF16 shared tile, applies exact
patches, and signals WGMMA consumers. Retain a proven SS-WGMMA and DSMEM epilogue.

**Why it can win.** Combines compressed HBM traffic with the faster shared/shared WGMMA path and
decouples integer decode from Tensor Core consumers.

**Evidence here.** Compiled and emulated in v32. NVCC 13 reports 122 registers, three barrier
resources, zero stack, and zero spills. The six-stage pooler-m128 kernel requests 201,888 bytes
dynamic shared memory. H200 runtime correctness and timing remain unknown.

**Decision.** Highest-priority original candidate after v20-v22 establish the uncompressed
cluster baseline.

### 13. Concurrent exact-exception warp

**Method.** Store a compact stream of rare correction records and assign a dedicated warp to
apply their dot-product contribution or patch decoded shared memory while the main path runs.

**Why it can win.** Keeps the common packed path branch-free and prevents one escape from forcing
an entire block back to raw BF16.

**Evidence here.** Public judged in v18 for small-M. v32 patches decoded shared tiles before the
WGMMA barrier with capacity eight; sampled normal-like tiles needed at most four.

**Decision.** Required companion to compression. Prefer stream-ordered per-warp records for direct
decode and per-tile compact patches for decode-to-shared.

### 14. Exact 11-bit BF16 compression (C11)

**Method.** Use an eight-bit low field and three-bit sign/exponent code with a narrower exponent
window plus exact escapes.

**Why it might win.** Saves one byte per eight weights versus C12 before corrections.

**Evidence here.** Rejected by modeling. Eight-byte correction records leave only about
1.10-1.68% ideal stream savings over C12, and router-low can become larger. Even unrealistic
four-byte zero-cost corrections do not independently reach the leader.

**Decision.** Do not implement before the reset queue. Revisit only with a nonduplicated compact
four-byte exception stream and a measured C12 decoder baseline.

### 15. Shared-memory swizzle and bank-conflict control

**Method.** Store BF16 tiles in WGMMA-compatible 128-byte swizzles; pad DSMEM rows; index mailboxes
so each float4 transaction is aligned and disjoint.

**Why it can win.** Prevents shared-bank serialization and makes WGMMA descriptors legal. Padding
can be cheaper than resolving conflicts dynamically.

**Evidence here.** Compiled/emulated/static-audited in v32. The compact DSMEM mailbox reduces the
mailbox from 17,408 to 16,384 bytes and enables a theoretical seven-stage layout under 232,448 B.

**Decision.** Mandatory correctness/performance plumbing, not an independent experiment.

### 16. Stage-depth, cluster-shape, raster, and cache-policy tuning

**Method.** Tune TMA ring depth, CTA cluster dimensions, number of consumer warpgroups, raster
direction, swizzle, and cache hints. Compile variants together and time them in one prepare pass.

**Why it can win.** The optimum balances latency hiding against shared memory, registers,
residency, and tail waves. CUTLASS explicitly exposes raster and cluster tuning.

**Evidence here.** Judged mixed result. v10 found the existing tiny-M stage depths best. v22 is
the prepared bounded sweep for four large routes. v31 WR4 was removed for 104 local bytes/thread;
v30 WR2 was removed for a 32-byte spill path.

**Decision.** Tune only around a measured winning architecture. Never treat more stages or more
threads as inherently faster.

### 17. cuBLASLt algorithm enumeration and pinned plans

**Method.** Enumerate legal cuBLASLt algorithms in `prepare()`, graph-time them with cold L2, and
store the winning descriptor/workspace for each shape.

**Why it can win.** Vendor kernels can beat custom code at m128/256, while the default heuristic
may not choose the best graph/cold-L2 algorithm for every narrow shape.

**Evidence here.** Public judged in v18. Generic DeepGEMM and TinyGEMM recombinations in public
campaigns plateaued near 1.15-1.18x aggregate, below v18.

**Decision.** Preserve v18 plans as the fallback and comparator. Do not replace them wholesale
with one third-party backend.

### 18. Router BF16 hi+lo with FP32 accumulation

**Method.** Split FP32 W into BF16 hi and BF16 residual lo; multiply both by BF16 x and accumulate
in FP32. Fuse the two streams/reduction or concatenate prepared weights when that wins.

**Why it can win.** Meets the router's 0.01% aggregate tolerance while using Tensor Cores or a
single custom pass. Plain BF16 router weights are not accurate enough.

**Evidence here.** Judged. v3 router is already 1.8-4.5x faster on relevant shapes; v18 adds tuned
routes. Public campaigns confirm concatenated hi+lo helps m128/256 but cannot close the aggregate
leader gap alone.

**Decision.** Keep v18/v30 router routes. Router is no longer the main score bottleneck.

### 19. Output-row partitioning and reduction removal

**Method.** Assign each consumer warp a disjoint output-row slice and let it traverse all K
slices, eliminating intra-CTA atomics.

**Why it can win.** Avoids shared atomics and a separate epilogue while preserving MMA count.

**Evidence here.** Judged v12 at 1.1179x. It helped qkv m8 but did not strictly improve the full
aggregate; v11's per-warp partials plus explicit reduction regressed to 1.0986x.

**Decision.** v13's m8-only specialization remains a fallback, not a pre-v18 priority.

### 20. Epilogue fusion and output-vector stores

**Method.** Convert FP32 accumulators to BF16 and write coalesced vectors in the owner CTA; fuse
simple bias/activation only where the task actually contains it.

**Why it can win.** Reduces extra launches and output traffic.

**Evidence here.** The projection task is just matmul, so fusion opportunity is limited. Router
hi+lo fusion is relevant; general activation fusion belongs in the whole-layer track.

**Decision.** Keep stores vectorized. Move cross-operation fusion to `muse_decode_layer` after the
BF16 track reaches its stop condition.

### 21. Programmatic Dependent Launch (PDL)

**Method.** Let a dependent kernel begin independent preamble work before its predecessor fully
completes, then synchronize at the true data dependency.

**Why it can win.** Hopper supports PDL in CUDA Graphs and can hide launch/preamble latency.

**Evidence here.** Proposed, but a single projection call has little internal multi-kernel work
after global split-K was rejected. It is more promising when composing projections in the whole
decoder layer.

**Decision.** Do not complicate isolated BF16 submissions with PDL. Carry the technique into
`muse_decode_layer` and serving work.

### 22. Cross-projection prefetch and overlap

**Method.** In the whole-layer kernel, begin fetching the next projection's weights while the
current projection finishes its epilogue or collective.

**Why it can win.** The judge flushes L2 before the whole timed call, not between kernels inside
that call. Legitimate in-call prefetch can overlap HBM ramp/tails.

**Evidence here.** Proposed. It cannot be expressed across separate calls in this BF16 problem
without doing forbidden work outside the timed function.

**Decision.** Preserve as a decode-layer/serving idea; do not attempt a benchmark-specific cache
hack in the standalone projection track.

## Public challenge implementations

| Source | License | Best public evidence | Transferable ideas | Decision |
|---|---|---:|---|---|
| `NathenFernandes/muse-bf16-proj` v18 | MIT | 1.518570x check, 9/9 | Direct C12, exact corrections, WGMMA/TMA/clusters, pinned cuBLASLt, router portfolio | First reset check, exact bytes |
| `Madhavan113/kerneloptimizer` | Public source/notice preserved locally | Historical ~1.3x class | C12 diagnostics, route profiles, exact correction layouts | Ideas incorporated; not ahead of v18 |
| `lee101/openkernels` | Apache-2.0 | README best BF16 check 1.149x | Native CUDA diagnostics, C12 RS-WGMMA tail evidence | Do not spend pre-v18 check |
| `ozgureyilmaz/okc-muse-kernels` | MIT | 1.180353x check; 1.171773x scored | TinyGEMM/DeepGEMM/router composition and negative results | Dominated by v18 |
| Current leader `colinlikescode` | No public source found | 1.6367x board score | Unknown | Do not infer or claim private method |

Other public starter/campaign repositories were searched, including `vjeux/okc-2026` and
`lilaclilac09/open-kernel-challenge`; none provided evidence above v18 in the latest audit.

## Research and official implementation index

1. NVIDIA CUDA Programming Guide, thread-block clusters and distributed shared memory:
   https://docs.nvidia.com/cuda/cuda-programming-guide/
2. NVIDIA Hopper Tuning Guide, TMA, clusters, asynchronous barriers:
   https://docs.nvidia.com/cuda/hopper-tuning-guide/
3. NVIDIA Hopper architecture overview:
   https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
4. CUTLASS Hopper warp-specialized GEMM example 48:
   https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu
5. CUTLASS Hopper grouped GEMM example 57:
   https://github.com/NVIDIA/cutlass/blob/main/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
6. CUTLASS 3.x GEMM API and dispatch policies:
   https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md
7. Triton/Gluon WGMMA tutorial:
   https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html
8. Triton/Gluon persistent-kernel tutorial:
   https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html
9. Stream-K, work-centric GEMM decomposition:
   https://arxiv.org/abs/2301.03598
10. cuBLAS/cuBLASLt documentation and heuristic caching:
    https://docs.nvidia.com/cuda/cublas/
11. CUDA Programmatic Dependent Launch:
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
12. FlashInfer GEMM backends and TinyGEMM/router APIs:
    https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/gemm
13. DeepGEMM source:
    https://github.com/deepseek-ai/DeepGEMM
14. Lossless compression via exponent concentration:
    https://arxiv.org/abs/2510.02676
15. Low-precision neural-component lossless compression survey/design:
    https://arxiv.org/abs/2508.19263
16. Exact public Muse BF16 v18:
    https://github.com/NathenFernandes/muse-bf16-proj
17. Public native-CUDA campaign:
    https://github.com/lee101/openkernels
18. Public measured variant campaign:
    https://github.com/ozgureyilmaz/okc-muse-kernels

The compression papers support exponent concentration as a general principle, but their entropy
coders are not automatically suitable for a microsecond GPU kernel. C12's fixed-width, random-
access decode is favored here because it avoids serial variable-length decoding.

## Measured failures that should not be repeated

| Failure | Evidence | Lesson |
|---|---|---|
| Global Stream-K/fix-up | v1 and public C12 diagnostics | Reduction launch/tail costs more than added parallelism |
| Scalar/shared C12 decode | v5 0.9951x, v6 1.0316x | Fewer bytes are useless if decode is serialized |
| More consumer warps everywhere | v8 1.1018x | Geometry is shape-specific; occupancy/registers matter |
| Generic Triton small GEMM | v9 0.9239x | Use Hopper-native WGMMA and legal skinny scheduling |
| More TMA stages by default | v10 1.1080x | Stage depth trades shared memory/residency against latency |
| Per-warp partial buffer + epilogue | v11 1.0986x | Extra shared traffic and reduction tail can exceed atomic cost |
| Unbounded geometry variants | Offline spill audits | PTXAS spills disqualify a candidate before H200 timing |
| C11 with eight-byte corrections | Exact byte/score model | Nominal bit width is not end-to-end stream size |
| Generic TinyGEMM/DeepGEMM composition | Public 1.15-1.18x campaigns | Vendor backends help selected shapes, not the entire portfolio |

## Ranked experiment queue

This is the order to use after the reserved daily counter resets. Do not reorder based on an
unmeasured intuition.

1. **Exact public v18.** Highest-confidence jump: published 1.518570x versus our 1.1158x. Check
   exact vendored bytes and attribution. If it passes and strictly wins, promote and submit.
2. **v30 all-small portfolio.** Starts from v25 and independently gates qkv16, o, qkv32, and
   public-profile M1 routes. Fall back v29 -> v28 -> v25 only on resource/prepare failure.
3. **v31 pooler32 WR2.** Isolated spill-free geometry. Combine only after a measured win.
4. **v20 cluster dense m128.** Establish that the specialized qkv-derived pooler/unpooler cluster
   architecture compiles, passes, and beats v18.
5. **v21 m128/m256 cluster portfolio.** Separately gate four large pooler/unpooler routes.
6. **v22 bounded large-M autotune.** Only after v21 compiles/runs; compare cache/TMA and adjacent
   legal stage depths in one check.
7. **v32 C12 decode-to-shared pooler128.** Measure only after the uncompressed cluster baseline,
   so its gain/loss is attributable.
8. **Combine strict winners.** Recheck one integrated candidate; do not combine statistically tied
   routes or failures.
9. **Broaden C12-cluster.** If v32 wins, extend first to pooler256, then qkv128/256 and
   unpooler128/256. Keep every route separately gated. A seven-stage v32 is allowed only if the
   six-stage profile is latency-limited rather than register/occupancy-limited.
10. **Hopper-native m32 WGMMA.** If the combined score remains below the leader, implement padded
    M64 SS/RS-WGMMA for pooler32 and qkv32, using v18 as the comparator.
11. **Broader direct C12 small-M.** If small shapes remain limiting, extend stream-ordered direct
    compression with a concurrent exact-correction warp, beginning with pooler/qkv m8/16.

## Rank-one feasibility gates

- v18 alone is a large ranking jump but remains about 7.2% below the 1.6367x leader in speedup.
- Improving only the four v21/v22 large pooler/unpooler routes cannot reach rank 1; even replacing
  them with optimistic task floors models to about 1.6076x.
- v30+v31 small targets alone also cannot reach rank 1 at their optimistic floors.
- The prepared small-plus-large portfolio can cross the leader only with broad, unusually strong
  gains. Therefore the decisive architecture is likely lossless C12 across multiple projection
  families, not another isolated router or geometry tweak.
- A candidate is promoted only after all correctness tests pass and its gain exceeds measurement
  noise. Rank is never inferred from a local estimate.

## Maintenance protocol

For every new source or idea:

1. Record the primary URL, license, hardware, precision, and whether the result is judged.
2. Map it to one of the four workload regimes; do not transfer a result across regimes blindly.
3. State exactness and CUDA-graph implications.
4. Estimate aggregate score impact, not just one kernel's speedup.
5. Search this file and `IMPROVISATIONS.md` for an earlier measured failure.
6. Require syntax, source-hash, packing/emulation, NVCC resource, and preflight checks as applicable.
7. Log every OKC invocation in root `RESULTS.md`, including failures and duplicates.
8. Read the saved profile after every passing check and update this library with measured evidence.
9. Push documentation and source together, while leaving `submission.py` on the best verified
   scored version until a qualifying check passes.

## Oct 29 evidence update

The reset queue has now been executed through its integration gate. Exact public v18 reproduced
and scored **1.521867x** (`d1bd752954ff4d21`), so it replaces v3 as the foundation. V31, v20,
v21, v22, and v32 contributed no selected kernel: their proposed routes all lost their internal
cold-L2 timing comparisons. This is strong judged evidence against further cluster-DSMEM tuning
for pooler/unpooler m128/m256 and against the six-stage C12 cluster pooler pipeline.

V30 is mixed evidence. Its first check was 1.525964x (`cf48c6d2aeb84753`), but an independent
source-comment-only confirmation was 1.504394x (`8880201062034c1a`). Both passed 9/9. The M1
module itself reproduced useful local selections for qkv, unpooler, and router, but the enclosing
v25/v28/v29 portfolio selected other routes differently between runs. Therefore v30 is not a
qualified aggregate winner. Research priority changes from “combine the queue” to “isolate only
the three M1 winners on exact v18.”

Measured failure additions:

| Failure | Evidence | Lesson |
|---|---|---|
| Pooler32 doubled-CTA geometry | v31: 20.74 vs 20.45 us | Same total warps with fewer CTAs does not offset scheduling/resource cost. |
| Dense m128/m256 cluster portfolio | v20-v22, every route lost | DSMEM/TMA synchronization and occupancy dominate; stop parameter sweeps. |
| C12 cluster pooler128 | v32: 49.22 vs 23.68 us, occupancy 1 | Lossless compression cannot rescue an occupancy-1 cluster pipeline. |
| Broad prepare-time route portfolio | v30: 1.5260x then 1.5044x | Microsecond route races need isolation and repeatability before promotion. |

The required isolation is implemented in the current
`submission_v30_all_small_qkv32.py` working candidate (SHA-256 `2d133f…`). Its final bindings use
fixed v18 for prepare and fallback, compile only the previously judged M1 module, and gate exactly
qkv/unpooler/router at m1. This removes the causal ambiguity from v25-v30's unrelated route races.
It is syntax/static-audit complete but awaits a fresh-budget H200 aggregate check.

## Above-ranked public-source audit (Oct 29)

The GitHub API audit of all 149 accounts above rank #150 found one verified public BF16 source:
[chenwainuo/okc](https://github.com/chenwainuo/okc), board rank #78 at capture. The exact audited
commit is `3d418e1625f6de2075b5684a27ddbc3d1442422d`, Apache-2.0. Its README reports 3133.1 us /
1.542x. The complete account ledger and false-positive dispositions are in
`ABOVE_RANKED_GITHUB_AUDIT.md`; notably, rank #1's public `Open-Inference` repository does not
contain the challenge submission.

The source advances beyond our public v18 foundation in three high-leverage areas: direct
predicated-XOR C12 GEMV for small m, register-A RS128/RS32 WGMMA for selected m16/m32/m128
routes, and a one-node router kernel. Its per-thread staged escape cursor avoids scanning empty
escape blocks, while exact-shape paired graph selection preserves cuBLASLt where custom paths
lose. Its own closed-track records warn against RS128 m256, fewer-CTA decompositions, per-warp
reduction, and cluster multicast.

The exact source is pinned locally as `submission_rank78_chen_3d418e1.py`, SHA-256
`bb50079be2ed74d65bb7109f1432fbc7a49bbe70913d247c254717cd0f27f712`. Python compilation
passes. First fresh-budget action: check it unchanged. Then integrate only the routes that improve
over each exact parent's judged profile; do not wholesale replace our cells that are already
faster.

Offline source comparison confirms it is not the same lineage as our v18: only 171 matching lines
and a 0.024761 line-sequence ratio. Do not build a dual-stack candidate before the exact control;
compiling and autotuning both complete systems risks the 300-second prepare limit. If the control
reproduces, use it as the new parent and transplant only a demonstrated incumbent winner.

### Full upstream research audit

The implemented next combined candidate is `submission_v33_chen_m1_hybrid.py`: exact Chen parent
plus the unchanged, already-judged M1 CUDA module for QKV/unpooler/router. It replaces the old
sum-flush gate with the parent's read-flush paired estimator and requires two separate 3% wins.
Offline integration checks pass; H200 verification is pending. See `STRONGEST_SUBMISSION_PLAN.md`
for the new ordered experiments, cross-source references, and a model proving that an impossible
per-cell union of both parents reaches only 1.57522x. Broad non-router gains are still necessary.

`CHENWAINUO_REPO_DEEP_DIVE.md` now catalogs the rest of the repository: 188 projection candidate
directories, four clean-room implementations, and a frozen history of 74 official jobs. The
strong lineage progressed through C12/GV, self-zeroing paths, shorter QKV chains, RS128, RS32,
RTK, predicated-XOR repair, per-thread escape cursors, and judge-faithful paired preparation.

The 3133.1-us export is an 87%-similar descendant of the archived 3157.04-us
`gv_predxor_pw45_v2` source. Its 1,354 inserted lines add validated cold-L2 paired baking,
conservative proven selections, forced cursor paths at selected m16/m32/m128 shapes, an O32 HALF
twin, and later router refinements. This is a source-level inference, not a formal score ablation;
the public repository does not state the final job ID.

Preserved non-shipping leads include unpooler16 X prefetch (+0.576 us local but an official whole
check loss), GV53 early issue (+0.192 us local but a scored whole loss), tiny descriptor-offset and
early-half observations, and a new constant-depth packet design. Closed evidence rules out C11/
variable codecs, small-m TMA rings, cluster reductions, RS128 m256, stock CUTLASS/DeepGEMM BF16,
reverse-operand ladders, and the four clean-room full submissions. Use the deep-dive table before
opening any BF16 branch.
