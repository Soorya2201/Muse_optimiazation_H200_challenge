# Prefill-layer optimization record

## Optimization inventory

| Optimization | Status | Measured/estimated effect |
|---|---|---|
| SGLang Marlin NVFP4 experts | retained for T=753 | custom direct decode was slower (872 vs 828 us in v8 autotune) |
| custom in-register NVFP4 grouped GEMM | tried (A/C/E) | useful at larger T, but slower than decode-once BF16 path |
| decode touched experts to timed BF16 scratch, then grouped GEMM | shipped in v5 | T=3012: ~2.19 vs 2.47 ms Marlin; T=16384: ~7.37 vs 13.31 ms |
| fuse SwiGLU into W13 epilogue | shipped | avoids an activation-sized read/write and launch |
| router high/low BF16 split with FP32 accumulation | shipped | matches SGLang's large-T fast path |
| router on side stream beside pooler | shipped | hides most routing time |
| overlap W2 decode with GEMM1 | tried in v6, rejected | contention; 1.125x versus v5 check 1.136x |
| stage-4 BN64 GEMM1 | tried in v7, rejected | required 262,144 B shared memory, above 232,448 B limit |
| autotune decode row tile/warps | tried in v8 | BN32/4 won mid, BN64/8 won big by only 10/6 us; submit variance erased gain |
| WGMMA/TMA warp-specialized expert GEMM | not tried | estimated 10--25% of expert phase |
| FA3 configuration/kernel tuning | not tried | most useful for full-context shapes; estimated 5--15% there |
| fuse residual/norm boundaries | not tried | estimated 1--3% at T=3012, less at T=16384 |

## Version and run history

| Version | Job | Mode | Speedup | Outcome |
|---|---|---|---:|---|
| v3 | 0f121aecd65e4aba | check | 0.998x | custom Triton compilation failed; fell back to baseline |
| v5 | d3a6811b14844cb6 | submit | 1.065x | recovered original scored result |
| v5 | 4b32ae57017d46ba | check | 1.136x | 5/5; fresh baseline profile |
| v5 | b6fbcde2c4aa4418 | submit | **1.118x** | best scored result; CI 1.102--1.135 |
| v6 | 45b8e49eb9c148a9 | check | 1.125x | W2-decode overlap did not beat v5 check |
| v7 | 46bf22c0f7f141ca | check | 1.104x | stage-4 candidate failed SMEM; timed path was v5 |
| v8 | 0b0b78bbc24c445c | check | 1.146x | decode-tile autotune; best check |
| v8 | 836dccd9085345fe | submit | 1.116x | did not beat v5 score; v5 restored locally |
| v9 | 6042d9b1cbc3469d | check | 1.132x | serial W2 decode after GEMM1 did not improve cache locality; reverted to v8 order |
| v10 | 1709aa488709430e | check | 1.127x | two queued persistent GEMM waves lost end to end; removed |
| v11 | 3b8e370cc41146c4 | check | 1.124x | group=4/16 tile-order sweep lost end to end; removed |
| v12 | b9956e0dc9e34639 | check | 1.119x | DeepGEMM warmed up, but the custom epilogue referenced a non-constexpr Python global; safe v8 fallback passed 5/5 |
| v13 | 310b5b1b67374ed5 | check | 1.110x | DeepGEMM reached the final unpermute, whose scalar scale load incorrectly used a vector mask; safe fallback passed 5/5 |
| v14 | 8eb0710179014513 | check | 1.116x | valid masked BF16 DeepGEMM was 35% slower mid and 58% slower big than the current custom path; rejected and removed |
| v15 | 3e1e425a05654c05 | check | 1.132x | route-count hybrid was correct but mixing direct/BF16 paths in one kernel more than doubled expert time; rejected |
| v16 | bed381d5de1d42d1 | check | 1.128x | split hot/cold kernels won synthetic timing by 46/55 us but lost end to end on real routing and four launches; archived, not promoted |

## Profile and roofline analysis

In the strong v5 check, T=753 sliding took 1.05 ms and Marlin experts consumed 691 us
(66%). Full attention added about 421 us in the full case. T=3,012 sliding took 3.14 ms
GPU-busy time: expert decode 717 us, GEMM1 932 us, and GEMM2 395 us together consumed
about 65%; full attention added 1.65 ms in the full case.

At T=16,384 without cached context, the custom expert phase was approximately 6.8 ms
(4.4 ms GEMM1, 1.9 ms GEMM2, 0.54 ms decode), while four dense projections totaled
roughly 3.9 ms. With 65K cached tokens, FA3 became the largest single kernel at 8.35 ms,
followed by the 5.6 ms custom expert compute. Thus the bottleneck changes by shape:
experts dominate sliding/short-context prefill; KV attention dominates the longest full
context.

The challenge floors imply v5 reaches about 24% of roofline at T=753 sliding
(257 us / 1,056 us), 35% at T=3012 sliding (1,036 / 2,967), and 43--46% on the first-turn
sliding/ctx0 shapes. The remaining gap is primarily tensor-core utilization in expert
GEMMs and paged-cache efficiency in full attention, not launch overhead.

v9 combined v8's decode-tile autotune with a serial W2 locality experiment: decode W13,
run GEMM1, then decode W2 directly before GEMM2. This avoided v6's HBM contention but did
not help: job `6042d9b1cbc3469d` passed 5/5 at 1.1318x (342.275 ms), below v8's 1.1460x
check. W2 decode returned to its original pre-GEMM1 position; v8 remains the local base.

v10 added a 264-CTA/two-wave copy of G. The synthetic T=3012 tuner preferred it by only
about 5 us, but job `1709aa488709430e` regressed to 1.1265x (343.900 ms) and the scored
turn-4 sliding shape slowed. The extra wave was removed; unlike decode T=32, these long
GEMMs already amortize their persistent loops and benefit from the maximum-resident grid.

v11 swept M-group ordering: group=4 for mid-size expert reuse and group=16 to match the
rough per-expert block count at T=16384. The synthetic tuner selected group=16 for the big
bucket, but job `3b8e370cc41146c4` regressed to 1.1240x (344.665 ms). Neither variant was
retained; the full workload did not reproduce the microbenchmark's small locality gain.

v12 began the architectural DeepGEMM experiment: losslessly decode only touched NVFP4
experts to BF16, use SGLang's masked grouped BF16 DeepGEMM for W13/W2, and keep the
global scales plus SwiGLU/unpermute in small Triton epilogues. The pinned environment had
the required APIs and completed DeepGEMM warmup, but Triton rejected `EPI_SCALE` as an
ordinary Python global in the first epilogue. The exception was contained by autotuning,
so job `b9956e0dc9e34639` ran the v8 fallback, passed 5/5, and scored 1.1193x. The fix is
to pass both `EPI_SCALE` and `LOCAL_EXPERTS` as explicit constexpr launch parameters;
the first run therefore measures fallback behavior, not DeepGEMM performance.

v13 fixed the constexpr launches. Both masked DeepGEMM calls and the custom SwiGLU
epilogue then compiled, exposing a final Triton type error: the per-route expert scale is
a scalar pointer but was loaded with the vector lane mask used for the 256-wide output
tile. Job `310b5b1b67374ed5` again contained the exception, passed 5/5 through the v8
fallback, and scored 1.1099x. The unpermute now computes scalar `route_valid`, substitutes
expert zero for invalid routes, loads one scalar scale, and uses the vector mask only for
the output tile.

v14 completed the masked BF16 DeepGEMM path and validated its result against Marlin.
It was not competitive: at synthetic T=3012 it took 2954 us versus 2192 us for G32
(+34.8%), and at T=16384 it took 11603 us versus 7327 us for G64W8 (+58.4%). The two
full NVFP4-to-BF16 expert decodes remain necessary, while expert grouping/padding and two
extra epilogues add overhead; DeepGEMM therefore cannot recover enough tensor-core gain.
Autotuning rejected it, job `8eb0710179014513` passed 5/5 through the v8 choices at
1.1156x, and the unused experimental code was removed. This rules out the off-the-shelf
masked BF16 route; a competitive architectural step must decode directly into a
warp-specialized WGMMA pipeline rather than materialize then regroup the same weights.

v15 tested the route-count-aware hybrid proposed below. Experts with at least two or
four BM128 route blocks were decoded once to BF16; colder experts stayed on the direct
NVFP4 path. The numerical result was correct, but the runtime branch placed both weight
load/decode paths in each Triton kernel and sharply increased register/code pressure.
At T=3012 J2/J4 took 4651/4222 us versus G32's 2189 us; at T=16384 they took
19408/19201 us versus G64W8's 7372 us. Job `3e1e425a05654c05` therefore selected the
existing paths and passed 5/5 at 1.1320x. The final variant separates hot and cold work
into statically specialized launches to determine whether branch-induced pressure, rather
than the materialization policy itself, caused the loss.

v16 statically specialized hot and cold work into separate launches. This removed the
mixed-path register penalty: threshold four won the synthetic tuner at both sizes,
2148 us versus G32's 2194 us at T=3012 and 7233 us versus G64W8's 7288 us at T=16384.
The full workload did not preserve that 2.1%/0.8% microbenchmark gain. Its four expert
GEMM launches and the scored routing distributions produced 343.402 ms and 1.1281x
(CI 1.1112--1.1451) in job `bed381d5de1d42d1`, below v8's 338.05 ms / 1.1460x check.
It was archived but not promoted or submitted. A future hybrid must compact hot/cold
tiles into disjoint work queues so each phase keeps one launch and avoids scanning/skipping
the other class; merely splitting the same persistent tile space is insufficient.

## Novel next ideas

### 1. TMA producer + WGMMA consumer expert pipeline

Existing G first expands every touched expert to a large BF16 HBM scratch, then rereads
it for `tl.dot`. Instead, two producer warps would TMA-load NVFP4 codes/scales into a
multistage shared ring, decode into shared BF16 tiles, and let a consumer warpgroup issue
WGMMA. W13's epilogue would retain fused SwiGLU. This removes the 0.5--0.8 ms scratch
write/read and gives WGMMA a steady pipeline. Estimated gain: 15--30% of the expert phase.
Risk: high (CuTe/CUDA implementation and shared-memory pressure). Plan: prototype W2
first at BM128/BN128/BK128, then add W13's paired gate/up tiles and persistent expert work
queues.

### 2. Route-count-aware hybrid representation

G decodes every touched expert fully even when an expert has only one token block; direct
NVFP4 kernels repeatedly decode hot experts. Build the device histogram already implied
by `moe_align_block_size`, classify each expert by route blocks, and use direct in-register
decode for cold experts while materializing only hot experts to BF16 scratch. Estimated
gain: 5--12% of the expert phase, especially at T=3012. Risk: medium; the extra partition
must not serialize scheduling. Plan: compact hot/cold expert-block lists on device, launch
the two paths into disjoint output rows, then share the existing finalizer.

### 3. Full-context page-1 FP8 prefill attention

FA3 is generic paged attention and consumes 8.35 ms at ctx65K. A dedicated page-size-1
kernel would coalesce both KV heads into 128-byte cache-line transactions, use deep TMA or
`cp.async` gather staging, and tile the long query block through FP8 WGMMA while retaining
online softmax. Estimated gain: 15--25% of the attention phase, or roughly 6--10% on the
ctx65K shape. Risk: high correctness/engineering burden. Plan: validate QK FP8 against the
existing tolerance, implement split-KV partial softmax states, and fuse the final gated
head norm into the combine.

### 4. Cross-phase persistent schedule

The current layer serializes dense projections, attention, and experts except for the
router/pooler overlap. A persistent device scheduler could begin expert metadata and
selected-weight prefetch as soon as routing completes, while the pooler finishes, and
pipeline post-expert rows into the unpooler. This is legitimate work inside the timed call
and avoids caching outputs. Estimated gain: 3--8% end to end. Risk: medium-high because
co-resident kernels compete for HBM, as v6 demonstrated. Plan: use PDL/TMA prefetch only
for metadata and the first hot-expert tiles, cap residency, and benchmark bandwidth
contention before expanding the overlap.

