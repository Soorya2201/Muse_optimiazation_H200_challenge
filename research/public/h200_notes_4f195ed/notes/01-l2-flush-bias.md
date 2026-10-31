# 01. Write-flush vs read-flush: how the L2 flush biases cold-cache timings

## Question

A cold-L2 microbenchmark has to evict L2 before each timed call. Common practice is to *write* a large buffer (`buf.zero_()`, `cudaMemsetAsync`). The alternative is to *read* one (`buf.sum()`). On an H200, does the choice change the measured time of memory-bound kernels, and can it change which of two kernels looks faster?

## Setup

- Dense BF16 projections at m = 1 on four layers with 8.4–67.1 MB of weights (see note 00 for shapes).
- Kernels: the cuBLAS GEMM, two GEMVs of my own (one output row per CTA, and four rows per CTA), and a split kernel of my own that reads a 12-bit lossless encoding of the same weights (50.7 MB instead of 67.1 MB for in-proj, 28.5 MB instead of 37.7 MB for qkv+gate).
- Four runs on 2026-10-24:

  | Run | UTC | Tuner flush |
  |---|---|---|
  | W1 | 16:50 | write: `zero_()` on a 64 MiB int32 buffer |
  | W2 | 17:25 | write, same |
  | R1 | 17:50 | read: `sum()` over the same buffer |
  | R2 | 18:20 | read, same |

  The tuner's timing loop is otherwise identical in all four runs (note 00, timer 2). The cuBLAS calls and the GEMVs are the same code in all four.
- The independent harness (note 00, timer 1) flushes by reading in every run.

## Method

For each (kernel, layer, run) I take the tuner's raw replay time and subtract the harness time of the same kernel in the same run:

    excess = tuner_raw − harness

If the two timers saw the same cache state, the excess would be roughly constant. It would be the graph-launch cost that the harness subtracts and the tuner does not, plus small differences in statistics. The excess should not depend on the kernel or its byte count.

## Data

**Table 1. cuBLAS GEMM, m = 1, excess in µs.**

| Layer | Weights (MB) | W1 | W2 | R1 | R2 |
|---|---|---|---|---|---|
| out-proj slice | 8.4 | 4.64 | 4.61 | 3.15 | 3.17 |
| o-proj | 16.8 | 5.24 | 4.93 | 3.23 | 3.49 |
| qkv+gate | 37.7 | 6.48 | 6.66 | 2.98 | 3.62 |
| in-proj | 67.1 | 8.86 | 9.63 | 3.15 | 2.98 |

The raw values behind one row, for scale: in-proj tuner 30.08 / 30.78 µs under the write flush, 24.61 / 24.61 µs under the read flush. The harness timed the same call at 21.15–21.63 µs in all four runs.

**Table 2. Other kernels, m = 1, excess in µs.** A dash means the harness did not time that kernel in that run.

| Kernel | Layer | Bytes read (MB) | W1 | W2 | R1 | R2 |
|---|---|---|---|---|---|---|
| GEMV, 1 row/CTA | out-proj slice | 8.4 | 4.96 | 4.86 | 3.24 | 3.30 |
| GEMV, 4 rows/CTA | o-proj | 16.8 | 5.34 | – | 3.26 | 3.57 |
| GEMV, 1 row/CTA | qkv+gate | 37.7 | 7.68 | – | 3.58 | – |
| GEMV, 1 row/CTA | in-proj | 67.1 | 9.56 | – | – | – |
| 12-bit split kernel | qkv+gate | 28.5 | – | 5.12 | – | – |
| 12-bit split kernel | in-proj | 50.7 | – | 3.91 | – | – |

**Table 3. One decision the write flush got wrong (in-proj, m = 1).**

| | Tuner, write flush (W2) | Harness |
|---|---|---|
| 12-bit split kernel | 24.77 µs | 20.86 µs (W2) |
| GEMV, 1 row/CTA | 26.18 µs | 16.78 µs (W1; not timed in W2) |
| Difference | split kernel 1.41 µs *faster* | split kernel 4.08 µs *slower* |

The same thing happened on qkv+gate. The tuner had the split kernel 0.48 µs ahead (17.22 vs 17.70 µs). The harness had it 2.12 µs behind (12.10 µs in W2 vs 9.98 µs for the GEMV in W1). In both cases the tuner chose the split kernel.

**Table 4. MoE layer, tuner only.** These are two runs (2026-10-24, 19:00 and 21:10 UTC) on the same three routing samples per batch size and the same tuner code. The flush changed (write: 96 MiB `zero_()`; read: `sum()` over it), and the kernels gained one pointer argument that these configurations leave null. Times are tuner µs, each the mean over 3 routings of the median of 15 replays.

| Tokens | Kernel family | Config | Write flush | Read flush | Shift |
|---|---|---|---|---|---|
| 1 | two-kernel, mma.sync | a | 44.3 | 40.7 | −3.6 |
| 1 | two-kernel, mma.sync | b | 44.7 | 44.3 | −0.4 |
| 8 | two-kernel, mma.sync | c | 80.4 | 75.2 | −5.2 |
| 16 | two-kernel, mma.sync | d | 127.5 | 130.7 | +3.2 |
| 128 | two-kernel, warpgroup MMA | e | 244.3 | 244.1 | −0.2 |
| 256 | two-kernel, warpgroup MMA | e | 294.7 | 294.4 | −0.3 |
| 256 | two-kernel, warpgroup MMA | f | 301.0 | 298.8 | −2.2 |

For scale, the tuner's repeatability: in three later runs that shared one code version and the read flush, config a measured 41.4, 41.3 and 41.3 µs, config c 75.2, 75.1 and 75.4 µs, and config e at 256 tokens 294.7, 293.4 and 293.6 µs.

Data: [`data/l2_flush_offsets.csv`](../data/l2_flush_offsets.csv), [`data/moe_tuner_flush.csv`](../data/moe_tuner_flush.csv). Figure: [`figures/fig1_flush_excess.png`](../figures/fig1_flush_excess.png).

![Excess of tuner time over harness time vs weight bytes, write and read flush](../figures/fig1_flush_excess.png)

## Finding

1. **With a read flush, the excess is flat.** All 14 (kernel, layer, run) pairs in Tables 1–2 fall within 2.98–3.62 µs (mean 3.29 µs), from 8.4 MB to 67.1 MB of weights. Once the flush matched the harness, my tuner tracked harness time plus about 3.3 µs for every kernel.
2. **With a write flush, the excess grows with the bytes the kernel reads.** For cuBLAS (means of W1 and W2):

   | Weights (MB) | Write excess (µs) | Read excess (µs) | Extra under write (µs, derived) |
   |---|---|---|---|
   | 8.4 | 4.62 | 3.16 | 1.46 |
   | 16.8 | 5.08 | 3.36 | 1.72 |
   | 37.7 | 6.57 | 3.30 | 3.27 |
   | 67.1 | 9.25 | 3.06 | 6.18 |

   The GEMVs show the same trend (4.9 µs at 8.4 MB to 9.6 µs at 67.1 MB).
3. **The bias depends on the kernel, so it can reverse a comparison.** On the same layer and flush, the 12-bit kernel's excess was 3.91 µs and the GEMV's 9.56 µs. That 5.7 µs difference in bias exceeded the 1.4 µs margin by which the tuner preferred the 12-bit kernel. On the harness, the 12-bit kernel was 4.1 µs slower (Table 3).
4. **The MoE shifts are kernel-dependent too, but not proportional to bytes.** Changing the flush moved configurations by −5.2 to +3.2 µs (Table 4), against a repeatability of about ±0.1–0.7 µs. At one token, configs a and b went from 0.4 µs apart to 3.6 µs apart. These kernels read 81–700 MiB, more than L2 holds, and the shift does not grow with batch size. I did not isolate the mechanism for this case.

**Limits.**
- Four runs on one day. The two GEMV harness times in Table 3 come from run W1 and are compared with run W2.
- The mechanism is an inference that I did not observe directly (no DRAM write counters). `zero_()` leaves the buffer's last ~50 MB as dirty lines in a write-back L2, and the timed kernel's reads evict them, which forces write-backs that compete for DRAM bandwidth. That fits findings 1–3: the penalty grows with the bytes read, and a kernel that reads fewer bytes is penalized less. I have no account of the positive MoE shift (+3.2 µs).

## Implications

- **Match the flush to how the kernel will run.** If production starts from a cold, clean L2 (or you want to compare against a harness that reads), flush by reading a buffer larger than L2. I used 64–96 MiB on a 50 MB L2.
- **Write-flush timings favour kernels that move fewer bytes.** Compression schemes, smaller tiles and split designs look better than they are. In this data the effect was large enough to choose the wrong kernel, by about 4 µs on a 17–21 µs call.
- **Calibrate your timer against an independent one on a fixed kernel.** A library GEMM timed by both makes the offset visible. If the offset depends on the kernel, the cache state differs between the two timers.
- **Published cold-cache numbers from write-flush tools may include a write-back penalty.** This is an inference from my H200 data, not a measurement of those tools: absolute times for memory-bound kernels in the 10–30 µs range could be inflated by several µs.

## Related work

- Triton's `do_bench` clears L2 with `zero_()` on a 256 MB buffer before each timed call: [testing.py](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/testing.py), [driver.py](https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/backend/driver.py).
- NVBench's cold measurements flush with `cudaMemsetAsync` over an L2-sized buffer: [l2flush.cuh](https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh).
- Speechmatics, "How to Accurately Time CUDA Kernels in Pytorch", flushes with `zero_()` and notes that the L2 is write-back: [blog](https://blog.speechmatics.com/cuda-timings).
- Nsight Compute flushes caches between replay passes by default (`--cache-control`): [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- I found no published measurement of read-flush vs write-flush bias.
