# `muse_moe_w4a16` optimization record

This file records measured results, including regressions. Estimates are explicitly marked.
The current verified submission is `submission.py` = `submission_v9.py`.

## Current result and profile

Scored job `63e72333221040c0` passed all tests at **1.6092x** (SGLang 13,448.5 us,
ours 8,357.3 us; 95% CI 1.5850-1.6333). It ranks #503 on the problem board as of
2026-10-27. The first-place score was 2.6196x.

| Shape | Ours (us) | TASK floor (us) | Floor / ours | Main profile phase |
|---|---:|---:|---:|---|
| bs=1 | 39.47 | 11.42 | 28.9% | gemm1 20.8 us (55.0%) |
| bs=16 | 110.70 | 52.52 | 47.4% | gemm1 73.1 us (61.5%) |
| bs=32 | 166.23 | 70.76 | 42.6% | gemm1 134.4 us (63.9%) |
| bs=8 | 86.38 | 37.32 | 43.2% | gemm1 40.0 us (61.6%) |
| bs=128 | 290.67 | 101.34 | 34.9% | gemm1 184.0 us (61.9%) |
| bs=256 | 429.38 | 108.94 | 25.4% | gemm1 294.1 us (64.6%) |

The TASK floor assumes 4.8 TB/s and discounts the observed number of touched experts. A
more realistic weight-only floor from the diagnostic routing is about 136 us at bs=128 and
142 us at bs=256. Even against those floors the large shapes achieve only about 47% and 33%,
respectively. The longest phase is always W13/gemm1 because it streams twice as many weight
rows as W2, dequantizes NVFP4 with integer instructions, and computes both gate and up
projections. Gemm2 is 35-39% of profiled GPU time; final RMS normalization and routed summation
is only 0.6-5.6%.

## Optimization inventory

| Optimization | Status | Measured or estimated effect |
|---|---|---|
| Lossless lane/MMA-fragment weight repack in `prepare()` | Tried, retained | Enables vector streaming and about one integer dequant op per weight. |
| In-register E2M1 and E4M3-scale conversion to BF16 | Tried, retained | Required for H200, which has no FP4 tensor cores. |
| Swap-AB `mma.sync` (weight rows as M, tokens as N) | Tried, retained | Avoids padding 15/16 of a WGMMA M=64 tile on decode. |
| Fuse W13 gate+up and SwiGLU epilogue | Tried, retained | Avoids writing the 8192-wide W13 result. |
| Deterministic warp-local routing lists | Tried in v5, retained | Replaced contended shared atomics and improved large-shape scheduling. |
| Equal-byte `(expert, output tile, token chunk)` work items | Tried in v5, retained | Keeps skewed experts from assigning unequal weight bytes to CTAs. |
| Schedule token chunks adjacent so duplicate weight requests overlap in L2 | Tried in v5, retained | Large improvement at bs=256 versus the earlier v1-class run. |
| Deeper register weight prefetch | Tried in v9, retained | WP 4 -> 6 improved check 1.5771x -> 1.5993x and submit 1.5543x -> 1.6092x. |
| 32-token items / NT=4 at bs=256 | Tried in v6, rejected | Registers rose to 255 and occupancy fell to 2 blocks/SM; score 1.5436x. |
| Two resident grid waves at bs>=128 | Tried in v7, rejected | CTA overhead outweighed tail reduction; score 1.5678x. |
| NT=1 at bs=128 | Tried in v8, rejected | Extra weight rereads outweighed occupancy; score 1.5575x. |
| Eight-stage TMA shared-memory ring | Tried in v4, rejected for now | Self-bench was slower and an invalid launch poisoned the check; needs redesign, not retuning. |
| PDL across gemm1, gemm2, finalize | Prototyped in v4 | No accepted win yet; launch gaps are small beside weight streaming for large shapes. |
| WGMMA for hot experts | Not tried | Estimated 10-25% on hot-expert compute, but requires a separate path and routing-dependent dispatch. |
| Cluster multicast of one weight tile to token-chunk CTAs | Not tried | Estimated 15-35% on bs=128/256 if duplicate HBM reads are eliminated. High implementation risk. |
| Fuse gemm2 last-arriver normalization/finalize | Not tried | Estimated 1-3%; removes one launch and reread, but needs graph-safe per-pair counters. |

## Version and OKC history

| Version | Job | Result | What changed |
|---|---|---:|---|
| v1-clean / old `submission.py` | `d0805799cf2245be` submit | 1.5543x | Three-launch custom NVFP4 kernel; original scored baseline. |
| v5 | `3469d3b6396a4309` check | 1.5771x | Deterministic routing and chunk-fastest equal-byte scheduling. |
| v6 | `946ef633000a4cf0` check | 1.5436x | NT=4 / 32-token chunks at bs=256; rejected. |
| v7 | `aea96697ffa14137` check | 1.5678x | Two grid waves at bs>=128; rejected. |
| v8 | `830ac92b8a134d07` check | 1.5575x | NT=1 at bs=128; rejected. |
| v9 | `6df3524f92aa430c` check | 1.5993x | Six-stage register weight pipeline; accepted. |
| v9 / current `submission.py` | `63e72333221040c0` submit | **1.6092x** | Scored confirmation of v9. |

All runs, including regressions, are also recorded in the root `RESULTS.md`.

## Novel next ideas

### 1. Cluster-multicast hot-expert weights

Existing: every 16-token CTA chunk issues the same packed weight loads. Chunk-fastest ordering
encourages L2 coalescing, but each CTA still owns an independent register stream.

Proposed: form a Hopper thread-block cluster per hot expert/output tile. One producer CTA uses
TMA multicast to stage each 1152-byte K step into every cluster member's shared memory; each
consumer CTA computes a different 8- or 16-token chunk. This makes one HBM transaction feed up
to eight chunks. Estimated gain: **15-35% on bs=128/256**. Risk: high; cluster residency,
multicast masks, mbarrier phases, and mixed cold/hot scheduling must all be correct. Implement
a hot-only kernel first, selected from shape/config and a device routing-count pass inside the
timed graph; retain v9 for cold experts.

### 2. WGMMA hot path with TMA producer warps

Existing: warp-level `mma.sync` is ideal for one to sixteen tokens but underuses tensor cores
when a hot expert has 64-128 tokens.

Proposed: batch 64 tokens for a hot expert into WGMMA M=64 tiles. Producer warps TMA packed
weights and activation tiles into a swizzled shared ring; consumer warpgroups dequantize and
issue WGMMA. Estimated gain: **10-25% for the hot-expert portion**, perhaps 5-12% overall.
Risk: high because H200 cannot consume FP4 directly, so dequantization must not become the
producer bottleneck. Start with W13 for one expert and compare achieved bytes/s and tensor-core
utilization against v9.

### 3. Last-arriver gemm2 finalization

Existing: gemm2 writes BF16 `ybuf`, atomically accumulates sum-of-squares, then a third kernel
normalizes and sums the eight routes.

Proposed: add graph-replay-safe per-pair counters in the existing workspace. After a gemm2 tile
publishes its BF16 outputs and sum, the last tile for a pair normalizes that row and atomically
accumulates FP32 token output; a tiny conversion epilogue writes BF16. Estimated gain: **1-3%**.
Risk: medium: counters must be reset in-kernel every replay without races, and FP32 output
atomics add traffic. Prototype only after the cluster hot path because finalize is currently
under 1% on the important large shapes.

### 4. Cross-layer selected-expert prefetch

Existing: this standalone kernel starts W13 reads only after routing is available, from cold L2.

Proposed for the whole decoder/serving track: once top-8 routing is known, launch a PDL-enabled
prefetch of the selected experts' first packed tiles while the pooler projection or communication
tail is still running. The benchmark flushes L2 only before the whole timed call, so prefetching
inside that call is legitimate. Estimated gain: **5-15% of MoE latency** if overlap survives L2
pressure. Risk: medium-high; prefetch must not evict activations or duplicate more traffic than it
hides. Measure with and without the pooler stream and cap prefetched bytes per hot expert.
