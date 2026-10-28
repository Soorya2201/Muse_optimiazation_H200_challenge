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
| vjeux public BF16 kernel | Researched, not queued | The public `vjeux/okc-2026` source is architecturally advanced, but its authoritative leaderboard entry is 1.5014x at rank 173 (`08c28ade42034919`), below Nathen v18's 1.5186x/rank 121. Do not spend the first reset check on it. |

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
