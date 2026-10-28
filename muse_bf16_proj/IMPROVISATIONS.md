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
| v24 cross-small C12 routes | Prepared, untested | Reuses v18's final qkv/pooler m1 plans at m8 and its final o m8 plan at m1. Their templates use dynamic m with MT=1, and v18 prepare already verifies every plan at m=1,4,8. Three cold-L2 comparisons add no compilation; only strict wins activate. |
| v25 history + cross autotune | Prepared, untested | Primary post-v18 candidate. Combines v23 and v24 in one no-compile sweep: 52 meaningful graphs over 23 changed shapes, fixed v18 elsewhere, and the same conservative gate. V23/v24 are now isolation diagnostics rather than separate default checks. |
| v26 compressed `o` CTA geometry | Prepared, untested | Keeps v18's exact direct-load C12 decoder and tests only legal WR=2/4 geometries at o m1/m8: 512 threads/256 CTAs and 1024 threads/128 CTAs instead of 256 threads/512 CTAs. Every candidate is exact-one-hot and random validated over m=1,4,8, then must clear the 1%/0.08 us cold-L2 gate. |
| v27 qkv16 clustered CTA geometry | Prepared, untested | Keeps v18's exact C12 decode and four-block cluster reduction while changing WR=1→2/4. All variants execute 2,304 warps, but the grid falls from 576 to 288/144 CTAs. Candidate-specific launch bounds, no-local-spill checks, exhaustive one-hot/random validation, and the conservative timing gate protect fixed v18. |
| v28 history/cross + qkv16 | Prepared, untested | Primary post-v18 candidate. Runs v25 first, then compares validated WR=2/4 qkv16 routes against v25's selected qkv16 winner. Thus one check can retain history/cross and geometry gains independently; every other route remains v25/fixed-v18. |
| v29 all small-route autotune | Prepared, untested | Adds v26's independently gated o m1/m8 geometry after v28. One check can retain history/cross, qkv16, and o gains shape by shape. V28 remains the single-new-extension fallback if the second compilation threatens prepare time. |
| v30 all-small plus qkv32 | Prepared, untested | Adds WR=2/4 qkv32 geometry after v29, targeting a 12.16 us route with an 8.0 us stated floor. Its three independent experimental extensions now compile concurrently before the original sequential validation/autotune order. This is the maximal check-efficient candidate; v29→v28→v25 form the staged fallback ladder. |
| v31 pooler32 geometry | Prepared, untested | Isolated test of WR=2/4 on v18's no-reduction C12 route: 128/64 CTAs versus 256, with 2,048 total warps unchanged. Kept outside v30 until measured. |
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
| v24 cross-small C12 routes | not run | `submission_v24_cross_small_routes.py`; v19 is an exact 637,846-byte prefix and v24 starts from exact v18 prepare. Static inspection proves the reused qkv/pooler/o templates are dynamic over m=1..8 and their existing prepare verification includes both cross-tested endpoints. CUDA/H200 timing is pending. |
| v25 history + cross autotune | not run | `submission_v25_history_cross_autotune.py`; v23 is an exact 643,845-byte prefix, but final prepare starts directly from exact v18 and performs only the combined v25 sweep. Static checks confirm 23 shapes and 52 candidate graphs. CUDA/H200 timing is pending. |
| v26 compressed `o` CTA geometry | not run | `submission_v26_o_c12_geometry.py`; v19 is an exact 637,846-byte prefix. Static checks prove WR=2/4 use 512/1024 threads, 256/128 CTAs, exact K coverage, and valid output tiling. The file is 643,418 bytes. CUDA compilation, escape-capacity validation on judge weights, exact GPU self-tests, and timing are pending. |
| v27 qkv16 clustered CTA geometry | not run | `submission_v27_qkv16_c12_geometry.py`; v19 is an exact 637,846-byte prefix. Generated WR=2/4 instances use 256/512 threads, 288/144 CTAs, 2,304 total warps, and 8,192/16,384 bytes of reduction storage. Source-anchor, geometry, syntax, and size checks pass; CUDA/H200 evidence is pending. |
| v28 history/cross + qkv16 | not run | `submission_v28_history_cross_qkv16.py`; v25 is an exact 649,127-byte prefix and the complete file is 655,295 bytes. The overlay validates and gates qkv16 against v25 after its 52-graph search. Exact v18's published 53.8 s cold prepare leaves headroom under the 300 s prepare limit, but combined H200 preparation and timing remain unverified. |
| v29 all small-route autotune | not run | `submission_v29_all_small_autotune.py`; v28 is an exact 655,295-byte prefix and the complete file is 659,779 bytes. The appended o routes use 512/1024 threads, 256/128 CTAs, and 4,096 total warps each. Syntax, prefix, geometry, and size checks pass; its two added extension compilations versus exact v18 have not been timed on H200. |
| v30 all-small plus qkv32 | not run | `submission_v30_all_small_qkv32.py`; v29 is an exact 659,779-byte prefix and the complete file is 666,501 bytes. Qkv32 WR=2/4 use 256/512 threads, 288/144 CTAs, and the same 2,304 total warps as v18. Its three independent NVCC builds run through the existing prepare thread-pool pattern, followed by the unchanged sequential qkv16/o/qkv32 validation and cold-L2 gates. Source anchors, syntax, mocked state/order, prefixes, geometry, and size pass; H200 preparation remains unverified. |
| v31 pooler32 geometry | not run | `submission_v31_pooler32_geometry.py`; v19 is an exact prefix and the file is 643,310 bytes. WR=2/4 use 512/1024 threads and 128/64 CTAs, preserve 2,048 warps, and require exact/random validation, positive capacity, no local spill, and a strict timing win. |

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

V24 exploits a separate omission in static dispatch. V18's final qkv and pooler plans are exposed
only at m1, while its final o plan is exposed only at m8, but all three compiled instances have
`CM=0`, `MT=1` and are therefore valid for m=1..8. More strongly, their existing prepare path
already runs random and exact one-hot checks at m=1,4,8. V24 cross-tests qkv1→m8,
pooler1→m8, and o8→m1 against fixed v18 with 16 alternating cold-L2 graph replays and the same
1%/0.08 us gate. No new CUDA source, packing, or precision behavior is introduced.

V25 combines v23 and v24 so their information costs one check rather than two. It appends the
three cross-small candidates to the exact route-history map, yielding 52 graphs over the same 23
shapes. The final prepare bypasses both earlier autotuners and calls exact v18 prepare once; seven
unchanged scored shapes and every held-out shape remain fixed v18. V23 and v24 should be run only
to isolate a surprising combined result, not by default.

V26 explores the smallest adjacent geometry space around the judged direct-load C12 `o` route
rather than changing its proven decoder. V18 uses WR=1/WK=8/KU=2: 256 threads and 512 CTAs for
the 8,192-by-1,024 projection. WR=2 and WR=4 preserve the same 1,024-wide K traversal and total
work while producing 512-thread/256-CTA and 1,024-thread/128-CTA launches. The hypothesis is that
fewer, larger CTAs reduce block scheduling and tail cost on this short-K matrix; the risk is lower
occupancy or worse per-CTA issue efficiency. The extension contains only those two specializations,
requires exact reconstruction of all 1,024 weight columns plus random tests at m=1,4,8, and times
them against fixed v18 at m=1/8. It activates nothing without both a 1% and 0.08 us win. Pooler's
existing C12 route is already a 1,024-thread, 128-CTA one-wave launch, so there is no analogous
higher-WR candidate. This is prepared static evidence only, not a measured improvement.

V27 applies the same bounded geometry question to qkv m16, where the upside is larger. V18's
`push_d9_cap6` C12 route uses WR=1/WK=4/KU=8/S=4: 128 threads, 576 CTAs, and four-block cluster
reduction. WR=2 and WR=4 preserve its decoder, FP32 accumulation path, split-4 topology, and
exactly 2,304 total warps while reducing the grid to 288 and 144 CTAs. Their reduction storage is
only 8 and 16 KiB. The WR=2 specialization retains the original 768 resident-thread launch-bound
target; WR=4 uses a one-block minimum because imposing three 512-thread blocks would distort
register allocation. Both variants must fit their per-warp C12 escape lists, report positive
cluster capacity and zero local spill, reproduce all 8,192 weight columns exactly, and pass random
m=9/15/16 tests. Only qkv m16 is timed or dispatched, and fixed v18 remains the fallback. The
hypothesis is fewer scheduling waves and a shorter clustered tail; the risks are reduced latency
hiding and extra within-CTA synchronization. This remains unmeasured until an H200 check.

V28 combines the two independent, highest-priority no-regression searches. It first runs v25's
history and cross-small selection, then compiles and validates v27's qkv16 geometries and compares
them against whichever qkv16 route v25 selected. A qkv geometry win overrides only that shape;
otherwise v25 is returned unchanged. This saves one check while preserving isolation in the
prepare log. The upstream receipt reports only 53.8 seconds for exact v18 cold preparation, so the
additional no-compile v25 sweep and two-template extension have substantial nominal room under the
300-second limit. That is feasibility evidence, not a measured combined prepare time.

V29 adds v26's o m1/m8 geometry after v28, comparing against the route already selected by the
history/cross search. Its WR=2/4 variants both execute 4,096 total warps while reducing the grid
from v18's 512 CTAs to 256/128. They reuse the exact C12 format, exhaustively verify m=1,4,8, and
cannot override v28 without the same 1%/0.08 us win. This is the most check-efficient candidate,
but it compiles a second new extension; v28 is deliberately retained as the fallback if v29 hits
the 300-second prepare ceiling or exposes a resource/compilation issue.

V30 applies the qkv16 geometry hypothesis to qkv32, whose v18 time is 12.16 us against the 8.0 us
stated floor. The original and WR=2/4 variants all execute 2,304 warps, but use 576/288/144 CTAs.
The new plans retain exact C12 decode and split-4 cluster reduction, reject local spill, reproduce
all 8,192 columns exactly, and validate random m=9,16,17,24,32 inputs. They are compared against
v29's already-selected qkv32 route. V30 is the maximal combined candidate; if its third extension
causes compilation or prepare-time trouble, descend through v29, v28, then no-compile v25. To
reduce that risk, v30 now starts from v25 once, compiles the qkv16, o, and qkv32 extensions in
parallel using the already-established prepare pool, and only then performs the original
validation and autotune sequence on the main CUDA stream. A mocked call-order test proves that
the same state keys and qkv16→o→qkv32 timing dependencies are preserved; only build wall time can
change.

V31 isolates pooler m32 rather than burdening v30 with a fourth extension. V18's no-reduction C12
route uses 256 CTAs × 256 threads; WR=2 gives a near one-wave 128-CTA grid and WR=4 deliberately
tests the 64-CTA underfill boundary, with 2,048 total warps unchanged. Both preserve exact decode
and validate m=9,16,17,24,32. Combine only after a measured win.

The latest portfolio model sets a hard priority boundary. Improving v30 plus v31 targets by
2/5/10/15/20% raises v18 from 1.51857x to only 1.52378/1.53168/1.54515/1.55899/1.57322x.
Even replacing all five with their optimistic TASK floors reaches 1.58896x. Adding the four
v21/v22 pooler/unpooler m128/m256 targets changes the picture: their combined floor model reaches
1.68212x, but uniform 5/10/15/20% wins across all nine targets yield
1.54698/1.57665/1.60767/1.64014x. Thus roughly 20% broad improvement is required just to edge the
current 1.6367 leader. Stop preparing adjacent small-shape geometries now; measure v30, v31, and
the cluster line, then combine only demonstrated wins. If they fall short, the missing lever must
be a broader compression/scheduling architecture, not another one-shape tile adjustment.

The [public BF16 strategy guide](https://thethings.ai/p/okc/muse-bf16-proj-guide/03-strategy)
also narrows future work: direct C12 streaming saves 13-20% on suitable small shapes, while L2
prefetch, x-stationary/multiple-row-block GEMV, persistent TMA rings, and compressed WGMMA were
measured losers. Do not spend checks repeating those rejected variants; adjacent geometry around
the direct-load C12 plans and removal of split-K tails remain the defensible small-m directions.

A fresh public-source audit on Oct 28 found no newly published kernel from `colinlikescode`,
`jkbpvsc`, or another current top entry. The apparent “1.67x” on the guide strategy page is the
per-shape router m128/256 result, not a new overall score; the guide still identifies its shipped
artifact as 1.343x check / 1.331x scored. Its linked
[project report](https://thethings.ai/p/okc/muse-bf16-proj-report) does sharpen the post-v22 design:
the failed compressed WGMMA kernel spent about 190 ns per 64-wide stage handshake, ran register-A
WGMMA near 50% of peak, and paid a 4-6 us split-K wait/reduction. The report's proposed remedies
are 128-wide K stages, a decode warpgroup that writes BF16 tiles to shared memory, shared/shared
WGMMA at CUTLASS-like efficiency, and cluster/DSMEM reduction. This is public evidence supporting
our no-global-tail direction, not evidence about the private leader. Implement that broader design
only after v21/v22 measurements; do not mistake the report for a ready 1.67x submission.

The Oct 28 pre-reset static audit then exercised the deferred CUDA generators rather than merely
running `py_compile`. The exact v18 SHA-256 still matches its manifest; the v25→v28→v29→v30 and
v21→v22 byte-prefix chains remain intact; v30/v31 resolve both replacement dispatch cases; and
v21/v22 generate all pooler/unpooler cases with no leftover placeholders or qkv output constants.
An initial manual reading incorrectly treated the config's group count as `NR` and therefore
overestimated unpooler shared memory. Evaluating the kernel's exact formula showed the largest
v22 plan uses 221,344 B, below the
[NVIDIA Hopper limit](https://docs.nvidia.com/cuda/hopper-tuning-guide/) of 227 KiB per block. The schedules were
left unchanged. V20-v22 now compute and reject oversized configurations during source generation,
so a future tile edit cannot silently spend an H200 check on an impossible launch. No OKC run was
used and the promoted submission remains untouched.

A later Oct 28 search covered the newly risen #3-#5 teams (`harshagundala`, `sudodaksh`, and
`schrepa`) and still found no attributable public kernel. A fresh independent
[Guild reproduction](https://guild.1nterface.ai/t/5?credible=all) of the exact MIT v18 source did
provide useful variance evidence: 1.53155x check and 1.52107x submit (`0babd4cb0aca4510`), which
supports keeping exact v18 first after reset. The same discussion reported two narrower ideas.
Programmatic dependent launch improved its lower-scoring global producer/reducer kernel by 4-9%
on router and about 2% on small pooler, but decompression and inspection of every v18 native source
found no PDL instruction or equivalent global semaphore/atomic reducer in the active design; v18
already routes through cluster/no-split kernels or library GEMMs. Adding PDL would therefore mean
reintroducing a different global-tail architecture, not toggling a free launch attribute.

The discussion's original-row-major `W @ X.T` path improved its own o M8-32 cases by 7-13%, but
its published M8 predecessor was 6.88 us. Even the full 13% gain models to about 5.99 us, slower
than v18's measured 5.496 us at M8; v18 is also already at 5.476/5.807 us for M16/M32. With no
exact faster M16/M32 times or source, this does not justify a check ahead of v30. Preserve it as a
post-measurement comparison only. No OKC run was used for this audit.

Applying the public claims optimistically to v18's exact shape CSV makes the priority decision
quantitative. A full 9% reduction on every router shape plus 2% on every small-M pooler shape would
raise 1.518570x only to 1.539329x. A full 13% reduction on o M8/M16/M32 would reach 1.530820x.
Their union reaches only 1.551944x, still 5.46% below the 1.6367 leader even before accounting for
the demonstrated non-transfer at o M8. These techniques may complement a future winner, but they
cannot replace the broad compression/scheduling work required for rank one.

The final v30 preparation boundary now treats qkv16, o, and qkv32 extension compilation as three
independent experiments. Previously, an exception from any one build discarded exact v25 and both
other builds, wasting the entire check before validation or timing. V30 now records a bounded build
error, installs an empty plan for only the failed module, and continues through the unchanged
qkv16→o→qkv32 validation/autotune order. Python compilation, the exact v25→v28→v29→v30
prefix chain, the normal mocked prepare path, and all three one-failure mocked paths pass. This is
a check-efficiency safeguard, not measured H200 performance; no OKC run was consumed.

An authenticated GitHub code refresh found the newly indexed
[`Madhavan113/kerneloptimizer`](https://github.com/Madhavan113/kerneloptimizer/tree/b7520fb174f788818c4936babb3743f756b1649b/muse_bf16_proj)
source. Its header attributes vjeux and applies Apache-2.0 to the modifications. The live board entry
is 1.4933x; the repository separately publishes a 9/9 profile at 1.488179x, 3246.11 us, job
`b1508a1e59694b0f`. The current source SHA-256 is
`09817f808584f3da1775d2ea002fce0e38a79b38afc01db24e6cf3e121fcded8`, but the profile does not
record a source hash, so the two artifacts must not be described as an authenticated exact pair.

The public implementation uses direct lossless C12 decode into register-sourced MMA fragments,
shared-memory activation staging, split-K shared/global reductions, and special router kernels. Its
published profile is slower than v18 overall but appears faster on ten shapes when each is compared
to v18's exact CSV: o m1 (5.51%); qkv m1/m16/m32 (8.04/1.90/7.75%); pooler m32 (3.81%);
unpooler m1/m32/m128 (22.89/3.88/2.55%); and router m1/m256 (15.78/17.19%). Cross-run node and
baseline variation means these are candidate signals, not causal measurements. Replacing every
v18 shape by the faster published time is an oracle upper bound of **1.555216x**, versus
1.518570x for v18 (+2.41%), still far below 1.6367x.

Therefore do not spend a check on the weaker whole file. Exact v18 and v30 remain first. Read the
v30 profile afterward and extract only routes that fail to match the strongest public reference
times: qkv m1 8.59 us, qkv m32 11.22 us, unpooler m1 2.87 us, and router m1 3.32 us. This avoids
duplicating v30's qkv32 search and turns the source into a targeted attributed follow-up rather than
another broad speculative check. No OKC run was used for this audit.

The public qkv-m32 route was then audited down to its exact `qkv32_r208` specialization. It owns
24 output rows per CTA, launches 96 CTAs of 288 threads (four consumer warps, one producer warp,
and four escape-correction warps), and processes eight 64-column units per stage. Each stage is
51,200 bytes: 32,768 bytes of activations plus 18,432 bytes of compressed weights. A 208 KiB ring
fits four stages; barriers, correction buffers, the escape table, and the zero row bring the exact
dynamic-shared-memory requirement to **219,552 bytes per CTA**. The route is CL=1, so it avoids a
cross-CTA reduction, and its row-balancing permutation evens escape work across the 96 CTAs.

This is not a small geometry transplant. A standalone extraction needs roughly 41 KiB of CUDA
common/mbarrier/TMA/kernel code plus its packing, row balancing, escape-table construction, tensor
map, and launch wrapper. For comparison, v30's already prepared qkv-m32 CUDA module is about 35 KiB
and tests 256- and 512-thread direct C12 geometries. Adding the public route before measuring v30
would create a fifth extension build, increase concurrent compiler memory pressure, and duplicate
the same shape without evidence that v30 misses the 11.22 us public threshold. Defer it as a
post-v30 isolated candidate: add it only if v30's selected qkv-m32 time is above 11.22 us (allowing
for cross-run noise), or if v30's qkv module fails while the rest of v30 succeeds. No OKC run was
used for this audit.

The targeted follow-up is now folded into hardened v30 rather than left as a plan. One new
independent extension contains the Apache-attributed public M1 kernels only: row-major lossless C12
GEMV for qkv, o, and the 1,024-row unpooler slice, plus the original-FP32 router GEMV. Pooler M1 is
deliberately omitted because its public time loses to v18. The three projection packers prove every
non-escape bit pattern exactly, prove each escape correction reconstructs the original BF16 value,
and build per-CTA correction tables. Prepare then tests random inputs and three one-hot columns per
route. Each shape is independently cold-L2 graph-timed against the already selected v30 base and
requires both 1% and 0.08 us before dispatch changes. The timed call branches only on public shape
metadata and remains graph-capturable.

Using the four public M1 times as an impossible perfect-transfer oracle raises v18 from 1.518570x
to only **1.537273x** (+1.23%); the module is a useful portfolio bridge, not a rank-one claim. It
compiles concurrently with the qkv16, o, and qkv32 experiments, and its build failure is isolated in
the same way. Python syntax, the v25→v28→v29→v30 byte-prefix chain, balanced CUDA-source braces,
all required instantiations, independent C12 code/table emulation for the three matrix shapes, the
normal mocked prepare order, and all four one-build-failure paths pass. CUDA compilation,
correctness, actual timings, and total prepare duration still require the reset H200 check. No OKC
run was consumed while implementing it.

A follow-up compile-risk audit matched the three C12 launch geometries and the router launch to the
published source, verified balanced CUDA braces and one common barrier per cgemv thread, and added
shape plus immediate CUDA-launch guards to every specialization. The M1 autotuner originally
synchronized after each of its 128 samples; it now records the same alternating cold-L2 event stream
and synchronizes once per route, as the established v25-v30 tuners do. Static guards prove event
collection precedes the batch synchronization and elapsed-time reads. This changes prepare overhead,
not the kernels or activation gate. Apple clang cannot compile CUDA and no nvcc is installed locally,
so local end-to-end extension compilation is unavailable and the remote H200 remains authoritative.
No OKC run was consumed.

Because the candidate is already public on GitHub, the isolated device translation unit was then
sent to [Compiler Explorer](https://godbolt.org/) with minimal CUDA-only wrappers that instantiate
all four kernels. NVCC 13.0 accepted `-O3 -std=c++20 -arch=sm_90a -c` with no diagnostics. A second
`-Xptxas=-v` build reported: router M1, 32 registers/32 B shared; o M1 C12, 32 registers/24 B;
unpooler M1 C12, 40 registers/40 B; qkv M1 C12, 40 registers/72 B. Every specialization uses one
barrier and reports zero stack frame, zero spill stores, and zero spill loads. This proves CUDA
device syntax, template instantiation, and ptxas code generation; it does not compile the ATen
binding or establish graph capture, correctness, prepare time, or H200 performance. Those remain
gated on the remote check. No OKC run was consumed.

A third compile pass retained the complete CUDA translation unit and replaced only the unavailable
ATen headers with a minimal interface stub for `Tensor::size`, typed/untyped `data_ptr`, and the
current CUDA stream. The first stub correctly failed because it omitted ATen's untyped `data_ptr()`
overload; after matching the real interface, NVCC compiled every host wrapper, shape guard, launch,
and device specialization cleanly and reproduced the same ptxas resources. This adds host-wrapper
syntax coverage, but deliberately does not claim PyTorch ABI or generated pybind coverage.

The installed CPU-only Torch 2.8 package supplied real C++ and pybind headers for the remaining
binding audit. `load_inline` generated the exact wrapper source but could not invoke a compiler
because Ninja is not installed; no package was installed or environment changed. Compiling that
generated source directly initially exposed two audit-command omissions—the matching Python 3.9
header path and Ninja's usual `TORCH_EXTENSION_NAME` define. With those standard flags restored,
Apple clang syntax-compiled and linked the wrapper against real `libtorch`, `libtorch_cpu`,
`libtorch_python`, and `libc10`; Python imported it and found both `v30m1_cgemv` and
`v30m1_rgemv` callables. This validates the exact prototypes, Torch tensor type casters, pybind
generation, symbol naming, linking, and module loading on Torch 2.8. The judge uses Torch 2.14 with
CUDA, so its full extension build and execution remain the authoritative gate.

A prepare-time audit traced the final v30 call graph rather than counting every textual
`load_inline` occurrence. Exact v18 already schedules 21 preparation families through a four-worker
pool and then builds its two v18 additions. Its authenticated `results/v18.json` receipt records
**53.798846702 seconds** in `prepare()` for job `66d8b007f6c641ee`, leaving about 246 seconds under
the 300-second prepare cap. V30 starts from the same exact-v18/v25 base, then compiles its four
independent qkv16/o/qkv32/M1 modules concurrently with four workers before sequential validation
and bounded graph timing. That does not prove v30 will finish, but it provides substantially more
headroom than the earlier source-size-only risk estimate. Keep full v30 second in the reset order;
use v29/v28/v25 only if the judge reports an actual top-level prepare/resource failure. Both exact
v18 and v30 pass local Python compilation, and the vendored v18 SHA-256 remains
`111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44`. No OKC run was consumed.

A separate timed-path compliance audit started from the final `kernel_fn` and inspected only its
reachable new v28-v30 helpers. Dispatch depends on `config` shape/layer metadata and immutable
prepare-time choices; it never reads activation values on the host. `_v30_launch` and
`_v30m1_launch` allocate only their returned output tensors and call the selected extension on the
current CUDA stream, matching the graph-capturable allocation pattern already used by judged v18.
The M1 CUDA/host translation unit contains zero device/stream synchronizations, `cudaMalloc`,
`cudaFree`, `cudaMemcpy`, `printf`, or host I/O. Its `TORCH_CHECK`s inspect tensor shapes, and its
`C10_CUDA_KERNEL_LAUNCH_CHECK`s report immediate host launch status without waiting for GPU work.
All weight inspection, integer conversion, lossless-pack proof, correctness evaluation, event
timing, synchronization, and printing remain inside `prepare()`. The new projection routes retain
BF16 activations and reconstruct exact BF16 weights; the router reads the original FP32 weights and
returns FP32. No OKC run was consumed.

An Oct 28 authenticated GitHub code-search refresh found no public repository belonging to any of
the current top-five BF16 teams. It did surface two previously unrecorded lower-scoring artifacts.
`PostTrainLLM/posttrainllm` commit `07897a4d0d5d99b9eb63ea3b49fc71df43159749` publishes an exact
source hash `774a28dae97b364a64704edc54d3ad811aaa71fea3918fc5ff8ab4f207845269` and check job
`1095cf9b43bf49a0`, which passed at only **1.067341x** (CI 1.05133-1.08335). It uses a conventional
Triton m1 GEMV, a 16-way small-router partial buffer plus reduction, and the SGLang dispatch for
other dense shapes. All of those paths are dominated by the existing v18/v30 portfolio.

`Ranoobaba/okc-campaign` publishes a Triton/cuBLAS autotuner with BF16 hi+lo router weights; its live
entry is **1.2331x**, rank 589 at the refresh, job `c4be07e843a74e33`. Its current source hash is
`fa245a5288d38d91cff9242184c550d8c86523c4bcbeabc36df70e75a615929a`. It has no lossless weight
compression, cluster reduction, or per-shape timing receipt that beats the public v18 thresholds.
Neither artifact warrants an OKC check or changes the reset order. No OKC run was consumed.

### Public-source landscape and the next architectural target

An authenticated GitHub code search on Oct 28 found eight repositories containing this problem.
The live score/source ladder is: `colinlikescode` 1.6367x (#1, no source found), `jkbpvsc`
1.5933x (#2, no matching source found), `harshagundala` 1.5899x (#3), `sudodaksh` 1.5893x
(#4), and `schrepa` 1.5876x (#5), all without attributable matching source; `rohanarun` is
1.5503x (#43, no matching kernel published), Nathen v18 is 1.5186x (#121, exact MIT source),
vjeux v141 is 1.5014x (#175, Apache source), and lee101 is 1.4627x (#214, Apache source).
Lower published implementations do not
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

### Exact 11-bit compression bound (Oct 28, no OKC run)

The public Madhavan source contains an unshipped 11-bit diagnostic, so this was evaluated as a
possible post-v30 architecture. The exact representation keeps the existing eight-bit low byte
(seven mantissa bits plus exponent bit zero) and replaces C12's four-bit high code with a packed
three-bit code: one sign bit plus a two-bit offset for the upper seven exponent bits. Together
with the raw low exponent bit, one window covers eight consecutive full exponents. Eight weights
therefore occupy 11 bytes instead of C12's 12 bytes. This does **not** require packing seven-bit
mantissas, but decoding eight codes now requires extracting eight three-bit fields from three
bytes instead of reading nibble-aligned codes.

A CPU distribution estimate using the task seed (not the exact CUDA RNG stream, so it is a
distribution estimate rather than judge evidence) found best-window escape rates of 1.355% qkv,
1.249% o, 1.354% pooler, and 1.247% unpooler. With a hypothetical globally compact four-byte
escape record, the four projection streams shrink by about 4.72-5.01% relative to C12; six-byte
records leave only 2.91-3.34%. The actual public exact-correction path uses `int2` records—eight
bytes per table entry, plus a count/padding record and sometimes per-CTA replication—so the
idealized stream reduction is only 1.10% qkv, 1.67% o, 1.11% pooler, and 1.68% unpooler before
metadata or extra decode instructions. Router-low is a poor candidate: its estimated escape rate
is 4.016%, making even the compact form larger than C12.

Per-block exponent windows do not rescue it. An 8,192-block Monte Carlo on the same BF16 normal
distribution measured 1.109/1.200/1.244/1.265% escapes for 32/64/128/256-weight blocks. Charging
only one base byte per block and ideal four-byte exceptions leaves 3.29/4.09/4.50/4.70% savings;
with the existing eight-byte records it leaves just 0.34/0.89/1.18/1.33%. At 512-8192 weights,
the escape rate returns to about 1.25%, so the window is effectively matrix-global. Adaptation
therefore trades nearly all of its small coverage improvement for base metadata.

The authoritative v18 CSV reconstructs its published 1.518570x exactly. If impossible zero-cost
four-byte escape records made every projection time scale with bytes, C11 would reach only
1.587559x; applying it only to m<=32 reaches 1.564078x. With a 1.5-2.0 us fixed ramp/drain term,
those bounds fall to 1.577829-1.574614x and 1.556634-1.554172x. With the existing eight-byte
record format, the corresponding all-shape bound is only 1.536956x with zero fixed cost and
1.534332-1.533459x with the fixed term; small-m-only is 1.530654x or 1.528625-1.527950x. These
models are still optimistic because they charge no three-bit unpack work or correction tail.

Decision: do not put an 11-bit kernel ahead of exact v18, v30, v31, or the prepared cluster
measurements. It cannot close the leader gap alone and is marginal with the current correction
table. Reconsider only as a measured follow-up if a compact four-byte global exception stream can
avoid per-CTA duplication and overlap corrections, or if the dormant `loads-11/12` diagnostic
produces a much larger H200 timing delta than the byte model predicts.

## Novel next ideas

### 0. C12 decode-to-shared with cluster DSMEM (post-reset priority)

Public evidence now isolates why neither large-m line is sufficient alone: Madhavan's C12
register/shared WGMMA reaches 15.24/22.66/19.46 us for qkv/pooler/unpooler m128, while exact v18's
uncompressed shared/shared cluster/no-reduction routes are already faster at
13.99/20.49/19.37 us. The proposed hybrid bulk-loads C12, expands it with a decoder warpgroup into
a swizzled BF16 shared tile, patches exact exceptions before a barrier, and retains shared/shared
WGMMA plus cluster DSMEM reduction. Modeled shared memory with eight bounded patch records per
stage is 202,080 B at m128 and 222,448 B at m256 for pooler, under the 232,448 B opt-in limit.
Uniform 20% improvement over all eight large
dense shapes reaches 1.62272x; 25% reaches 1.65364x, so it can be a rank-one component only when
broadly successful or combined with v30/v31. Full pipeline, exactness, resource arithmetic, and
rollout are in `DESIGN_C12_CLUSTER.md`. This is not implemented or measured and must follow the
reset baseline checks.
The offline contract is now executable: `tools/c12_cluster_emulator.py` passed exhaustive BF16
bits, every patch-field boundary, 4,096 fixed-global-base normal-like tiles, adversarial values,
and the WGMMA swizzle. The sampled patch maximum was four versus the reserved capacity eight.
CUDA instruction mapping and performance remain unimplemented.

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
