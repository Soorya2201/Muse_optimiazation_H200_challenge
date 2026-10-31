# 05. Small measured facts (appendix)

Short items that did not need a full note. Each item gives its evidence and how far it goes.

## 1. Clusters of 4 at one CTA per SM launched in two waves

**Setup.** A split-K projection kernel using thread-block clusters, with shared memory sized for one CTA per SM, ran as grids of 126–128 CTAs. I compared the start times of all CTAs (`%globaltimer`, one cold replay).

| Cluster size | Grid | Max active clusters reported by `cudaOccupancyMaxActiveClusters` | CTA start spread |
|---|---|---|---|
| 2 | 128 | – | 0.0 µs |
| 4 | 128 (32 clusters) | 30 (= 120 CTAs) | 20.2, 23.8, 25.9 µs (three configurations) |
| 7 | 126 (18 clusters) | not recorded | 19.7 µs |

**Finding.** With clusters of 4 at one CTA per SM, only 30 clusters (120 of 132 SMs) can be resident at once. The last two clusters of a 32-cluster grid start when the first ones finish, 20–26 µs later. The occupancy API predicted this. My first reading of the same log took a nonzero answer to mean "fits", because 32 × 4 = 128 ≤ 132, but the number itself (30 < 32) already said there would be a second wave.

**Inference.** CUDA co-schedules all CTAs of a cluster on one GPC. If the GPCs' SM counts are not multiples of 4, some SMs cannot host a 4-CTA cluster at one CTA per SM. Clusters of 2 had no such loss here.

**Implication.** For a one-wave grid of clusters, size the grid from `cudaOccupancyMaxActiveClusters` for that exact cluster size and shared-memory footprint, not from the SM count.

## 2. FP8 `mma.sync` on sm_90 runs on the FP16 tensor-core path

**Source.** Khattak and Mikaitis report that on Hopper, FP8 inputs to `mma.sync` "are first converted to fp16, after which the HMMA instruction is invoked at the SASS level". Only `wgmma` reaches the native FP8 path. I did not dump SASS to confirm this myself.

**What I measured (indirect).** Three implementations of the 8-token verify attention over the FP8 KV cache, at batch 16, full layers:

| Implementation | Time per 64-key stage | Forward pass end |
|---|---|---|
| FP8 `wgmma`, K and V consumed as FP8, Q and P rounded to E4M3 | 2.0 µs | 193 µs |
| FP16 `mma.sync` after one explicit FP8 → FP16 conversion per K/V element | 2.9 µs | 253 µs |
| FP16 `wgmma` after the same conversion | 3.4 µs | 312 µs |

These kernels also differ in pipelining, so the table shows only that the native FP8 path was the fastest. It does not isolate the conversion cost.

**Implication.** On Hopper, FP8 operands need `wgmma` to get the FP8 rate. If the source is right, FP8 `mma.sync` offers no throughput advantage over converting to FP16 yourself.

## 3. Dependent load chains

**Measured.** On a cold start the first data of a kernel arrives 1.2–2.3 µs after the request (note 02, Table 3). Any step that must wait for one load before issuing the next pays a round trip, and a chain of them multiplies it.

**Example.** A 12-bit lossless weight encoding needs a rare-value patch step: a list of escape records per row, each corrected against the activations. The first version used one warp that, for each record and each token, loaded an activation from global memory and then added it into shared memory with a compare-and-swap loop. That is a chain of dependent load → atomic steps. The fix issued every record and activation load first (`cp.async`, 4 bytes each, 32 records per group), then applied them with lanes over tokens and no atomics. Graph µs (in-process timer), one run before and one after:

| Shape | Before: patches on | Before: patches off | After: patches on | After: patches off |
|---|---|---|---|---|
| qkv+gate, m = 32 | 41.25 | 19.36 | 21.25 | 20.03 |
| in-proj, m = 16 | 29.47 | 21.57 | 22.82 | 22.69 |
| in-proj, m = 32 (config A) | 54.88 | 23.01 | 25.79 | 24.86 |
| in-proj, m = 32 (config B) | 76.86 | 28.06 | 26.27 | 27.68 |
| router, m = 128 | 184.83 | 20.32 | 29.41 | 20.29 |

The patch phase's last CTA finished at 47.6 → 9.0 µs (in-proj m = 32, config A) and 176.9 → 21.2 µs (router m = 128). The total work did not change; only the dependency structure did.

**Estimate, not measured separately.** A split-K "last CTA finalizes" tail costs about three dependent round trips (fence, counter atomic, reload of the partials), roughly 2–3 µs. That is comparable to the ~1.1 µs node it replaces (note 04) plus the finalize kernel's own run time. So fusing a reduction into its producer saves less than the node count suggests.

## 4. CTA end-time spread is the largest identified loss in the two-phase MoE at small batch

**Measured.** `%globaltimer` end stamps of every CTA in one cold replay (note 03 design, tuned configurations).

| Tokens | Kernel | CTA end: min / p50 / max (µs) | p50 → max |
|---|---|---|---|
| 1 | `gemm1` | 2.3 / – / 19.2 | – |
| 8 | `gemm1` | 20.0 / 36.6 / 45.1 | 8.5 µs |
| 8 | `gemm2` | 46.8 / 64.6 / 69.3 | 4.7 µs |
| 16 | persistent form, both phases | 82.9 / 93.2 / 109.7 | 16.5 µs |
| 32 | persistent form, both phases | 100.8 / 108.6 / 120.5 | 11.9 µs |

At 1 token the `gemm1` CTA body averaged 6.3 µs, but the longest took 15.2 µs.

**Compared with the whole loss.** Real consumers add 20.7–22.8 µs per call over the no-op skeleton (note 03, Table 2). The median-to-last-CTA spread alone is 8.5–16.5 µs per kernel at 8–32 tokens, and the next kernel cannot use its inputs until the last CTA ends (note 04).

**Explanations tested and rejected:**
- *The `gemm2` items' acquire load.* In the persistent form, a variant that skips the per-item `ld.acquire` for experts already seen was bit-exact, but did not shrink the spread. The last CTA ended at 69.0 vs 69.4 µs at 8 tokens and 125.7 vs 119.9 µs at 16 tokens.
- *Uneven CTA placement.* A `%smid` probe showed breadth-first placement: in grids of 264, 396 and 528 CTAs (2–4 per SM), no two consecutive block indices among the first 132 shared an SM (0 of 131 pairs). An SM-balanced first-round table measured 39.6 vs 40.7 µs at 1 token (tuner), which is within noise.
- *Work fetch order.* A static first round with dynamic fetch after it gave no gain. At 1 token the `gemm1` body was still 17.2 µs at most against 7.2 µs on average.

**Inference.** At these batch sizes each CTA processes only a few large work items (a `gemm1` item is 16 stages of 9216 bytes), so per-SM byte counts differ by whole items. Smaller or stealable tail items are the obvious next test. I did not run it.

## 5. L2 prefetch gave no gain in four separate attempts

| Attempt | Phase | Result |
|---|---|---|
| `cp.async.bulk.prefetch.L2` of MoE weight blocks 4 stages ahead | bandwidth-bound | Harness: slower at every batch size, +3% to +15% (note 03). |
| Prefetch of the next MoE kernel's first weights from the previous kernel's tail | tail of a bandwidth-bound kernel | Tuner: slower in 8 of 9 paired configurations, by 0.6–3.6 µs; one was 0.9 µs faster. |
| `prefetch.global.L2` of attention K/V rows 8 stages ahead | latency-bound attention | Tuner: within ±0.3 µs on the four small shapes (15.6 vs 15.6, 23.7 vs 23.9, 20.6 vs 20.9, 14.4 vs 14.5 µs) and 214.4 vs 216.0 µs at verify B = 16 full; slower at decode B = 16 and B = 32 full (139.6 vs 116.7 and 257.0 vs 209.7 µs). |
| Prefetch of the next projection's weights on a forked graph branch during attention | latency-bound attention | Tuner: slower by 1.9–4.1 µs (note 04). |

**Why the attention prefetch could not help.** A probe that maps every key into a region small enough to stay in L2 cut the time per 64-key stage by only 1–15%:
- decode B = 1 full: 1686 vs 1990 ns;
- decode B = 16 full: 1441 vs 1676 ns;
- decode B = 32 full: 1423 vs 1585 ns;
- verify B = 16 full: 2069 vs 2081 ns.

The stages were not waiting on DRAM latency, so moving data closer could not shorten them.

**Implication.** Before adding a prefetch, measure the stage time with the data already L2-resident. If that barely moves, the prefetch has nothing to hide, and it still costs issue slots and DRAM bandwidth.

## Related work

- NVIDIA, CUDA C++ Programming Guide (12.8), *Thread Block Clusters*: thread blocks in a cluster are "guaranteed to be co-scheduled on a GPU Processing Cluster (GPC)" ([guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#thread-block-clusters)).
- F. A. Khattak and M. Mikaitis, *Accurate Models of NVIDIA Tensor Cores*, §4.1.6. FP8 `mma.sync` on Hopper converts to FP16 and runs HMMA ([arXiv:2512.07004](https://arxiv.org/abs/2512.07004)).
- W. Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*. Hopper memory latencies ([arXiv:2402.13499](https://arxiv.org/abs/2402.13499)).
