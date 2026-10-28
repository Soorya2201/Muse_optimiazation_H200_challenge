# `muse_bf16_proj` optimization record

The current scored and verified submission is `submission.py` =
`submission_v3_cluster.py`.

## Current result and profile

Scored job `167d6e5c85f74682` passed at **1.1158x** (SGLang 4,830.8 us, ours
4,329.6 us; 95% CI 1.0990-1.1325). As of 2026-10-28 this was rank #1071;
first place was 1.6367x.

The custom kernel wins primarily on m=1 projections and on the router. It uses a TMA-streamed
shared-memory ring, swap-AB BF16 `mma.sync`, cluster split-K, and a DSMEM reduction. The router
is especially favorable because its exact BF16 hi+lo representation is evaluated in one pass,
while SGLang launches two GEMMs plus an add. For m=16/32, custom qkv/pooler kernels were slower
than cuBLAS, so the verified choice table selects cuBLAS there. At m=128/256 it also selects
cuBLAS, whose Hopper `nvjet_sm90` kernels dominate the profile.

The longest decode weight stream is the 67.1 MB pooler/unpooler. The practical cold-L2 floor is
about 15-17 us at the measured 4.1-4.3 TB/s, not the nominal 14 us. The custom path is still
limited by TMA issue rate, dequant-free BF16 weight movement, MMA fragment consumption, and
cluster reduction tail. The m=16/32 custom prototype measured roughly 23.5 us for qkv versus
13.6 us cuBLAS and 27.8 us for pooler versus 20.8 us cuBLAS, so selection rather than forced use
is essential.

## Optimization inventory

| Optimization | Status | Result |
|---|---|---|
| Lossless MMA-fragment weight repack in `prepare()` | Tried, retained | Makes every consumer lane's BF16 fragment contiguous. |
| TMA bulk weight streaming through a deep SMEM ring | Tried, retained | Raises cold-HBM bandwidth over register-direct v0. |
| Spread activation slice copies across the producer warp | Tried, retained | Avoids one thread issuing every x copy. |
| Swap-AB `mma.sync` for skinny M | Tried, retained | Efficient for m<=8 without padding M to 64. |
| Cluster split-K and DSMEM reduction | Tried, retained | Removes the 2-4 us global fix-up chain. |
| Plain `__syncthreads` when cluster size is one | Tried, retained | Avoids unnecessary cluster barriers. |
| Exact router FP32 as BF16 hi+lo with FP32 accumulation | Tried, retained | Passes the 1e-4 and probe-row bounds. |
| Per-shape custom/cuBLAS selection | Tried, retained | Prevents m=16/32 and m>=128 regressions. |
| Stream-K global fix-up | Tried earlier, rejected | Serial counter/fix-up cost erased the gain. |
| Generic Triton tensor-core GEMM for m=16/32 | Tried, rejected | One CTA/output tile was far too slow on long-K qkv/pooler (about 45/38 us). A warp-specialized TMA/WGMMA kernel is still untried. |
| CUTLASS/DeepGEMM-style WGMMA for m=128/256 | Not tried | High effort; cuBLAS is already strong. |
| Exact 12-bit BF16 compression | TMA-ring variants rejected; direct-load v14 prepared | v5/v6 put unpacking and raw-block fallback on the consumer critical path. Public H200 evidence shows direct vector loads, a `prmt`/`lop3`-optimized layout, software prefetch, and a concurrent exception warp can instead deliver 13-20% per-shape gains. |
| 16-warp pooler TMA stages | Tried, rejected | m=1 improved only ~0.3 us while m=8 regressed by ~4.4 us. |
| Refreshed old shape choices | Tried, rejected | Graph timing did not reproduce the older eager-sweep advantage. |
| Per-warp shared partials instead of atomics | Tried, rejected | Explicit partial stores and epilogue summation regressed to 1.0986x; Hopper shared atomics were cheaper. |
| Row-partitioned qkv consumers | Tried, promising only for m=8 | Removed intra-CTA atomics with unchanged MMA count; v12 reached 1.1179x and profiled qkv m8 at 11.7 us, but m1/aggregate did not strictly beat v3. Next specialize only m8. |
| Direct global-load compressed GEMV | Prepared, untested | v14 targets qkv/pooler m1/m8 with 16 streaming warps, one prefetched 64-K group per warp, 12-bit fragment decode, and a seventeenth exact-correction warp. Await the next reserved check window. |
| Nathen Fernandes v18 public kernel | Vendored, awaiting our check | Exact MIT-licensed judged artifact: 1.5186x, 9/9, CI 1.4958-1.5413 (`66d8b007f6c641ee`). It combines stream-ordered exact C12 GEMV, per-warp in-register escape patches, WGMMA/TMA/cluster kernels, a scalar FP32 router specialization, and pinned cuBLASLt plans. |
| vjeux public BF16 kernel | Researched, not queued | The public `vjeux/okc-2026` source is architecturally advanced, but its authoritative leaderboard entry is 1.5014x at rank 175 (`08c28ade42034919`), below Nathen v18's 1.5186x/rank 121. Do not spend the first reset check on it. |
| [lee101 public v22/diagnostics](https://github.com/lee101/openkernels) | Researched, not queued | Apache-2.0 source and detailed judge diagnostics. Current board entry is 1.4627x/rank 214 (`b55e82a9c7f04856`); published v22 evidence is 1.459x. Its later RS-WGMMA experiments isolate global split-K reduction as the m128 bottleneck. |
| [`jkbpvsc` board entry](https://github.com/jkbpvsc) | Score only; no reusable source | New current #2 at 1.5933x (`6b16112fadec4058`). Web search, GitHub code search, and the owner's public repository list contained no matching challenge source as of Oct 28. |
| [rohanarun board entry](https://github.com/rohanarun/Auto-research-agent) | Score only; no reusable source | Current board entry is 1.5503x/rank 43 (`d1e1635ef48f4850`), stronger than Nathen v18, but the current public repository contains only the research application/prompts and no matching kernel or receipt. Do not attribute or reconstruct unavailable source. |
| v19 route-generation autotune | Prepared, untested | Starts with the exact v18 artifact, then during `prepare()` compares its complete v15-v18 dispatch generations with cold-L2 CUDA-graph timing on all 30 scored shapes. It keeps v18 unless an older generation wins by both 1% and 0.08 us, and always uses fixed v18 for nearby/held-out shapes. |
| v23 full-history route autotune | Prepared, untested | Supersedes v19's search without new CUDA compilation. Reconstructs every v9-v18 route-change boundary: 23 scored shapes changed, with 49 genuinely distinct candidates total. Seven unchanged shapes and all held-out shapes stay fixed v18; the same 1% and 0.08 us cold-L2 gate applies. |
| v20 cluster-dense m128 | Prepared, untested | Generalizes v18's judged qkv m128 cluster-DSMEM kernel to pooler m128 (64 two-CTA clusters, K split in half) and unpooler m128 (128 independent CTAs, no reduction). Random and one-hot GPU self-tests plus cold-L2 timing gate each route during `prepare()`; otherwise v19 remains active. |
| v21 cluster-dense m128/m256 | Prepared, untested | Extends v20 with v18's dormant TM256 WGMMA schedule. Two generated modules independently gate pooler/unpooler at m128 and m256, retaining captured outputs during graph timing. All four routes use 128 CTAs and fit Hopper's dynamic shared-memory limit; any non-winning route falls back to v19. |
| v22 cluster pipeline autotune | Prepared, untested | Subsumes v21 without compiling it first. For each of its four shapes, compares the base 2D-TMA schedule, v18's evidenced 3D-TMA/normal-weight-cache policy, and one adjacent legal stage depth. All 12 candidates require random and exact one-hot correctness; only a 1% and 0.08 us cold-L2 win over v19 activates. |

## Version and OKC history

| Version | Job | Result | Notes |
|---|---|---:|---|
| starter/custom development | `2ddc741cc8f5428f` check | development | Earlier TMA/cluster sweep. |
| v3 cluster | `87850e2a50124d20` check | 1.122x | Recovered best check. |
| v3 cluster | `3f64c8489d024353` check | 1.1189x | Fresh authoritative verification. |
| v3 / current `submission.py` | `167d6e5c85f74682` submit | **1.1158x** | Current scored result. |
| v4 diagnostic | `0540fd64be8648ef` check | 1.1154x | Same kernel plus untimed exact-compression analysis. |
| v5 exact BF16-12 | `211b39873ae4497f` check | 0.9951x | Exact fixed 12-bit stream with raw-block fallback; scalar unpack was too expensive. |
| v6 PRMT BF16-12 | `3762b11e7ff64f2e` check | 1.0316x | Pairwise PRMT unpack recovered some loss but stayed far below v3. |
| v7 choice refresh | `5e56c7e77fb34524` check | 1.1120x | Statistically tied/slightly worse; not promoted. |
| v8 pooler NW=16 | `5f6f3910b4bb4ac9` check | 1.1018x | Wider TMA stages hurt m=8. |
| v9 Triton m=16/32 | `7897be35f5c548f9` check | 0.9239x | Correct, but long-K qkv/pooler tiles were much slower than cuBLAS. |
| v10 stage diagnostic | `cddcfa69cbbf4c5a` check | 1.1080x | Cold-L2 graph sweep confirmed qkv's current four-stage ring wins m1/m8; pooler's current ring wins m8 and is within 0.03 us at m1. No dispatch change retained. |
| v11 per-warp partial reduction | `fb91e9bfbf7e43d4` check | 1.0986x | Correct, but separate per-warp shared stores and the explicit epilogue sum cost more than the original same-address shared atomics. qkv m1/m8 paired times were 12.80/13.57 us; rejected. |
| v12 qkv row partition | `f73b3d06dcef4761` check | 1.1179x | Correct; two consumer warps each owned one 16-row slice and traversed all four K slices, removing intra-CTA atomics without adding MMA work. qkv m1/m8 profiles were 9.9/11.7 us. Aggregate 4321.5 us, CI 1.1011-1.1346: recovered v11 but did not strictly beat the prior 1.1189-1.122 checks. Do not promote; next isolate this mapping to m=8. |
| v13 qkv row partition only at m=8 | not run | Prepared from v12; m=1 uses the original four-consumer topology and only qkv m=8 uses row ownership. Held for the next daily check budget. |
| v14 direct compressed qkv/pooler m1/m8 | not run | New architecture based on public H200 evidence: matrix-wide 3-bit exponent window, exact sparse FP32 corrections, direct non-allocating global loads, software-prefetched decode/MMA, and a concurrent exception warp. CPU layout and lossless reconstruction tests pass; CUDA/H200 validation pending. |
| upstream Nathen v18 | upstream job `66d8b007f6c641ee`; not yet run by us | Exact source SHA-256 `111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44`, vendored from commit `86aff75e6aca72511bf9b41b6dc190bf1e763373`. Published result 1.518570x, CI 1.4958-1.5413, 9/9. First candidate for our next reset; preserve attribution and license. |
| v19 Nathen route autotune | not run | `submission_v19_nathen_route_autotune.py`; it copies upstream v18 with one non-semantic trailing-space cleanup, followed by only the autotune overlay. Syntax passes locally. Run only after exact v18 establishes the current-pod baseline. |
| v20 cluster-dense m128 | not run | `submission_v20_cluster_dense_m128.py`; v19 is an exact 637,846-byte prefix. Generated pooler/unpooler CUDA has no unresolved markers or qkv-specific row constants, and the file is 646,721 bytes. CUDA/H200 compilation and timing are pending. |
| v21 cluster-dense m128/m256 | not run | `submission_v21_cluster_dense_m128_m256.py`; v19 is an exact 637,846-byte prefix. Static generation produced two-case pooler/unpooler sources with no unresolved markers, m128/m256 swap-WGMMA calls, and 163,920-213,136 bytes of dynamic shared memory. CUDA/H200 compilation and timing are pending. |
| v22 cluster pipeline autotune | not run | `submission_v22_cluster_dense_autotune.py`; v21 is its exact prefix but its final prepare bypasses v21 and starts directly from v19, so only two six-template extensions compile. Generated sources contain all six cases and use 155,712-221,344 bytes of dynamic shared memory. CUDA/H200 compilation and timing are pending. |
| v23 full-history route autotune | not run | `submission_v23_full_history_autotune.py`; v19 is an exact 637,846-byte prefix, but v23 bypasses v19's sweep and starts from exact v18 prepare. AST reconstruction proves 23 changed shapes, 20 two-way comparisons, three three-way comparisons, and 49 candidate graphs total. CUDA/H200 timing is pending. |

## Lossless 12-bit BF16 feasibility result

Job `0540fd64be8648ef` measured the real judge weights in 128-value blocks. For each block it
chose the best consecutive window of 15 BF16 exponents; code 15 was reserved for an exact escape.

| Matrix | Escape rate | Blocks with escape | Estimated bits/weight |
|---|---:|---:|---:|
| qkv_gate | 0.0102% | 1.30% | 12.064 |
| o | 0.0098% | 1.25% | 12.064 |
| pooler | 0.0103% | 1.31% | 12.064 |
| unpooler | 0.0100% | 1.27% | 12.064 |

This is far below the handoff's 1% escape threshold. The representation preserves the sign and
all seven mantissa bits inline, reconstructs the exact 8-bit exponent from a block base plus a
4-bit code, and stores rare original exponents in a side structure. No precision changes.

The first implementation experiment changed the conclusion to "encoding promising, TMA-ring
decode path rejected." v5 stored fixed 12-byte records for each eight BF16 values and read the
original packed fragment for the 1.3% of blocks whose exponent span exceeded 15. It passed every
correctness test, including exact probe rows, but qkv m=1 rose from 9.5 to 21.4 us. v6 used PRMT
and pairwise exponent reconstruction; qkv still took 18.9 us and pooler 27.2 us. On Hopper, the
TMA handshake, block-fallback reads, and integer dependency chain dominated before the 24% byte
reduction could help.

Public research on 2026-10-28 materially changed the next action. A published Muse BF16 kernel
guide reports a 1.343x check / 1.331x submit using the same 12-bit premise, but with a different
hot loop: direct `ld.global.nc.L1::no_allocate` vector loads instead of a TMA shared-memory ring;
a pack-time bit layout decoded by `prmt`/`lop3`; the next compressed group loaded before current
decode/MMA; and one extra warp applying rare exact corrections concurrently. Its report measures
13-20% improvements on qkv/pooler/o at m=1/8/16. Sources:
[format and decoder](https://thethings.ai/p/okc/muse-bf16-proj-guide/07-format),
[compressed GEMV](https://thethings.ai/p/okc/muse-bf16-proj-guide/08-gemvz), and
[measured report](https://thethings.ai/p/okc/muse-bf16-proj-report).
No public implementation from the current rank-1 team `colinlikescode` was found; this is a
public former-leading design, not a claim about the private current winner.

After that note was written, the underlying MIT-licensed repository was located at
[NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj). Its newer
v18 is substantially stronger than the tutorial snapshot: **1.518570x** on judged job
`66d8b007f6c641ee` with CI 1.4958-1.5413 and all 9 tests passing. The exact artifact and evidence
are vendored under `upstream_nathen_v18/`; its source hash matches the repository manifest.
This still is not the current #1 team's source, but it is authoritative evidence from a kernel
that historically reached #1 and now sits only about 7.8% in speedup ratio behind today's leader.

V18's per-shape evidence shows where the extra jump over the 1.33x tutorial design came from:
router m=1/8/16/32 at 3.94/4.34/4.23/5.04 us, pooler m=128/256 at 20.49/24.19 us, and dense
m=16 paths at 10.25/5.48/13.98/13.74 us. Its C12 escape handling is packed per streaming warp
and patched directly into decoded registers rather than evaluated as a separate arithmetic tail.
It also retains specialized WGMMA/TMA/cluster and cuBLASLt routes for larger shapes. These are
the largest remaining architectural gaps in our v3/v14 line.

### Quantified gap from v18 to the current leader

The current leader is 1.6367x versus v18's 1.518570x. Holding baseline times fixed, matching it
requires reducing every v18 load total by **7.22%**. The required savings are 2.56 us (decode 1),
3.44 us (decode 16), 3.95 us (decode 32), 2.70 us (verify 1), 4.83 us (verify 16), and 5.92 us
(verify 32). Pooler is 29-38% of every load; qkv is 21-26%; unpooler is 28-29% except at m=1/8.
Router is only 9-12% despite its large individual speedups. A 10% router-only improvement would
raise the aggregate by only about 1.0%, while an 8% improvement across the four non-router
projections models to approximately **1.6363x**. Therefore post-v18 work should prioritize dense
projection route selection and memory traffic, not another router rewrite.

The full public history of `vjeux/okc-2026` confirms v141 is the 1.5014x board artifact and that
its cleaned, kernel-equivalent version scored 1.4905x. Its most transferable technique is not its
whole-file implementation but cold-L2, graph-replay route selection across independently valid
families. V19 applies that technique to the already-prepared v15-v18 generations inside the
stronger Nathen artifact, with a conservative win threshold. A targeted GitHub search found no
public implementation belonging to current leader `colinlikescode` as of Oct 28; do not imply
that any public design reproduces the private leader.

V23 tightens and broadens that same route-selection idea. Static reconstruction of v18's route
history shows only 23 of 30 scored shapes changed between v9 and v18. Twenty have two distinct
kernels and three have three, for 49 meaningful candidate graphs—not the 300 graphs implied by
blindly timing ten dispatchers on every shape. Because v18's unified prepare already compiles and
self-validates all historical modules, v23 adds no extension build. It directly compares only
the generation boundaries where a route changed, keeps fixed v18 on seven unchanged and every
held-out shape, and requires both the v19 relative and absolute win margins. This fully subsumes
v19's meaningful search with less prepare-time work and less duplicate-measurement noise.

### Public-source landscape and the next architectural target

An authenticated GitHub code search on Oct 28 found eight repositories containing this problem.
The live score/source ladder is: `colinlikescode` 1.6367x (#1, no source found), `jkbpvsc`
1.5933x (#2, no matching source found), `rohanarun` 1.5503x (#43, no matching kernel
published), Nathen v18 1.5186x (#121, exact MIT source), vjeux v141 1.5014x (#175, Apache
source), and lee101 1.4627x (#214, Apache source). Lower published implementations do not
justify spending a whole-file check before v18/v23.

The valuable new evidence is lee101's later v26-v30 diagnostic series. Its lossless C12
RS-WGMMA kernel shares x through TMA and decodes weights directly into register-sourced WGMMA
fragments—the same broad direction needed to close the m32/m128 gap—but m128 remained 1.6-1.7x
behind cuBLAS. Clock profiles showed CTA start spread below 0.15 us and no dominant decode/wait
phase. With global split-K, CTAs ended across an **8-9.5 us reduction tail**; without split-K,
they ended within 1.5-2.2 us. Decode-ahead depth four and two-CTA-per-SM geometry did not fix it.
Therefore a future compressed m128 kernel should not repeat global counter/workspace reduction.
The next defensible architecture is cluster DSMEM/st.async aggregation or a no-split schedule
that keeps enough independent N tiles resident; deeper decode pipelining alone is a rejected path.

V20 implements that scheduling conclusion by specializing the already judged qkv m128 source in
Nathen v18 rather than inventing a new reduction primitive. The original qkv route uses 36
three-CTA clusters. Pooler instead uses 64 two-CTA clusters, splitting K in half and aggregating
through cluster DSMEM; unpooler uses 128 independent CTAs and needs no reduction. Its generator
specializes matrix dimensions, output strides, K chunk counts, and tensor maps, then rejects any
unresolved template marker or qkv-specific row constant. During `prepare()`, random and one-hot
correctness tests precede a conservative cold-L2 timing gate against v19. This is prepared static
evidence only: CUDA compilation, correctness, and performance still require an H200 check.

V18's judged shape data also leaves a larger m256 opportunity: pooler takes 24.19 us and
unpooler 23.32 us, versus 17.4 us floors for both. Its source already contains a dormant TM256
swap-WGMMA schedule, although that experiment was written for qkv and was not enabled there.
V21 applies the same specialization method to pooler/unpooler at both m128 and m256. It compiles
one two-case module per layer, uses 64 two-CTA clusters for pooler and 128 independent CTAs for
unpooler, and gates each of the four routes separately. Static shared-memory calculations are
213,136/196,688 bytes for pooler m128/m256 and 196,752/163,920 bytes for unpooler m128/m256,
all below Hopper's 227 KiB opt-in limit. This remains a hypothesis until the remote compile,
self-tests, and cold-L2 comparisons pass.

V22 bounds the next tuning step instead of multiplying whole-file checks. Each v21 route is
compared in one extension against two alternatives: the 3D-TMA plus normal weight-cache policy
encoded by v18's `PIPE=6683` qkv experiment (translated to `6363` for TM256), and one adjacent
stage depth that remains launchable. Pooler tries S8/S8-3D/S7 at m128 and S4/S4-3D/S3 at m256;
unpooler tries S8/S8-3D/S9 and S4/S4-3D/S5. The geometry, FP32 accumulation, cluster reduction,
and output conversion are unchanged. Static generation confirms six cases per layer, and shared
memory spans 155,712-221,344 bytes. V22 starts directly from v19 rather than running v21's
prepare, then validates all 12 candidates and times four-way cold-L2 graph replays per shape.

The exact v18 shape CSV also bounds this line of work. Making all four targeted routes 5%, 10%,
15%, or 20% faster raises the reconstructed 1.518570x score to 1.533735x, 1.549524x, 1.565984x,
or 1.583167x respectively. Replacing them with the TASK.md floors reaches only 1.607610x, still
below the 1.6367x leader—and those floors are optimistic for cold-HBM execution. Therefore v22
can be a meaningful bridge but cannot by itself reach #1; subsequent work must also improve the
small-m dense projections, especially qkv/pooler, rather than endlessly tuning these four routes.

## Novel next ideas

### 1. Direct-load 12-bit GEMV with concurrent exact exceptions

Existing v5/v6: TMA a compressed ring into shared memory, decode in the consumer, and read a raw
BF16 fragment whenever any value in its block escapes. Proposed v14: store one raw low byte plus
sign/3-bit exponent-window code per weight, stream 48 instead of 64 bytes per lane directly into
registers, prefetch group j+1 before decoding/MMA of group j, and let a dedicated warp evaluate
the roughly 0.01% exact corrections in parallel. Estimated gain from the public H200 measurements:
**13-20% on qkv/pooler m=1/8**, materially larger than our earlier 6-12% estimate. Risk: medium:
the layout is CPU-emulation verified, but register allocation, launch bounds, and judge-weight
exception counts still require H200 validation. Start with v14's four shapes, then add o and m=16
only after a strict check win.

### 2. WGMMA M=64 with masked/padded m=16/32

Existing: the custom kernel uses warp MMA and loses to cuBLAS at m=16/32. Proposed: TMA x and
weight tiles into swizzled shared memory, pad activation M to 64, and use one WGMMA consumer
warpgroup with specialized producer warps. Estimated gain: **5-15% on m=16/32 projection time**.
Risk: medium-high because padding wastes 2-4x compute and WGMMA descriptors/swizzles are fragile.
Start with pooler m=32, where weight bandwidth dominates and the 67 MB stream amortizes setup.

### 3. Extend direct compression to m=16 with selective cluster reduction

Existing v14 deliberately covers only qkv/pooler m=1/8 with one CTA per 16 output rows. Proposed:
after that path is validated, use the public tuned m=16 mappings and compare its global semaphore
tail with our proven cluster/DSMEM split-K for qkv. Estimated gain: **6-18% on individual m=16
layers**, with smaller aggregate impact. Risk: high: DSMEM clusters can reduce residency, while
global last-CTA reduction adds a serial tail. Preserve direct register streaming; do not restore
the rejected TMA ring merely to reuse DSMEM.

### 4. Cross-projection weight prefetch in the whole layer

Existing: each projection independently ramps HBM from a cold start. Proposed in the whole-layer
and serving tracks: use PDL so the next projection's producer CTA can issue its first TMA stages
while the previous epilogue/all-reduce tail finishes. Estimated gain: **2-6% of projection time**.
Risk: medium; concurrent streams may reduce rather than increase HBM efficiency, so measure power,
bandwidth, and overlap on the H200.
