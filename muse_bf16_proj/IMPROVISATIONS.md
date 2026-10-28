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
| Exact 12-bit BF16 compression | Tried, rejected in current decode form | Encoding is exact and only ~0.01% values escape, but scalar and PRMT unpack variants were ALU-bound and slower than raw BF16. |
| 16-warp pooler TMA stages | Tried, rejected | m=1 improved only ~0.3 us while m=8 regressed by ~4.4 us. |
| Refreshed old shape choices | Tried, rejected | Graph timing did not reproduce the older eager-sweep advantage. |

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

The implementation experiment changed the conclusion from "promising" to "encoding promising,
decode path rejected." v5 stored fixed 12-byte records for each eight BF16 values and read the
original packed fragment for the 1.3% of blocks whose exponent span exceeded 15. It passed every
correctness test, including exact probe rows, but qkv m=1 rose from 9.5 to 21.4 us. v6 used PRMT
and pairwise exponent reconstruction; qkv still took 18.9 us and pooler 27.2 us. On Hopper, the
extra integer dependency chain dominated before the 24% byte reduction could help. A future
version should only revisit this if reconstruction can be folded into a warp-specialized producer
and hidden behind independent WGMMA work, rather than sitting on the consumer's critical path.

## Novel next ideas

### 1. Fixed-layout 12-bit BF16 with raw-block fallback

Existing: every cold decode GEMV reads 16 bits/weight from HBM. Proposed: for each 128-weight
block store 128 sign+mantissa bytes, 64 exponent-code bytes, and a base/metadata record. Blocks
with any escape (only ~1.3%) fall back to a separate raw BF16 array, indexed by metadata. This
avoids a variable escape loop in the hot path while remaining exact for arbitrary weights.
Estimated gain: **6-12% on decode loads**, with about 21-24% fewer average weight bytes after
metadata/fallbacks. Risk: medium-high; integer reconstruction can become the bottleneck and the
TMA layout must still deliver MMA fragments efficiently. Implement qkv m=1 first and compare
achieved HBM bytes/s and integer issue utilization.

### 2. WGMMA M=64 with masked/padded m=16/32

Existing: the custom kernel uses warp MMA and loses to cuBLAS at m=16/32. Proposed: TMA x and
weight tiles into swizzled shared memory, pad activation M to 64, and use one WGMMA consumer
warpgroup with specialized producer warps. Estimated gain: **5-15% on m=16/32 projection time**.
Risk: medium-high because padding wastes 2-4x compute and WGMMA descriptors/swizzles are fragile.
Start with pooler m=32, where weight bandwidth dominates and the 67 MB stream amortizes setup.

### 3. Compressed TMA plus DSMEM split-K

Existing: compression and cluster reduction are separate ideas. Proposed: TMA the fixed-layout
12-bit stream into each CTA's shared ring, reconstruct BF16 fragments in consumer registers, and
retain the proven DSMEM reduction. Estimated gain: **8-15% on m<=8**. Risk: high: shared layout,
barrier phases, and reconstruction scheduling interact. Build it as a new version; retain v3 as
the fallback for every unverified shape.

### 4. Cross-projection weight prefetch in the whole layer

Existing: each projection independently ramps HBM from a cold start. Proposed in the whole-layer
and serving tracks: use PDL so the next projection's producer CTA can issue its first TMA stages
while the previous epilogue/all-reduce tail finishes. Estimated gain: **2-6% of projection time**.
Risk: medium; concurrent streams may reduce rather than increase HBM efficiency, so measure power,
bandwidth, and overlap on the H200.
