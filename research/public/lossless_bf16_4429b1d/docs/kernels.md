# Kernels

Four kernel families read the 12-bit format. The `.cu` files keep the short CUDA namespaces of the measured
sources:

| kernel | file / namespace | tokens m | layout | grid | threads per CTA |
|---|---|---|---|---|---|
| `rowpair` | `zr.cu` / `zr` | 1 | `rowpair` | N / (2·TPC) CTAs, TPC row pairs each | 32·NW |
| `fullk` | `zg.cu` / `zg` | ≤ 32 | `rowtile` | N / (16·RT) CTAs | 288 (8 compute + 1 escape warp) |
| `fullk_tma` | `zt.cu` / `zt` | ≤ 16 | `rowtile` | N / (16·RT) CTAs | 320 (8 consumer + producer + escape warp) |
| `splitk` | `zs.cu` / `zs` | ≤ 32 | `blocked` | G = number of SMs (or 128), 1 CTA per SM, + finalize grid | 32·(NCW + 1) |

The configuration ids and template parameters are listed in
[`lossless_bf16/configs.py`](../lossless_bf16/configs.py). Every kernel handles one token tile of 8·NT tokens, so
a decoded weight fragment feeds NT MMAs. `tools/compile_check.sh` prints registers and shared memory per
instantiation. With nvcc 12.8, ptxas reports small spills for two `fullk` configurations: 32 B for config 0 and 16 B
for config 7. Config 10 is config 7 with one chunk in flight instead of two, and has no spill.

## Shared inner loop

- **Weights.** `fullk` and `rowpair` load each lane's 48 bytes per tile with three
  `ld.global.nc.L1::no_allocate.v4.u32`, issuing U chunks before the first MMA. `fullk_tma` and `splitk` copy whole
  stages with `cp.async.bulk` (TMA) under an L2 evict-first policy into an mbarrier ring, then read them with
  `LDS.128`.
- **Activations.** The column order (see [format.md](format.md#column-order-inside-a-chunk)) makes each lane's B
  operand for a chunk 32 contiguous bytes:
  - `fullk` reads them with two 16-byte `__ldg` per token;
  - `fullk_tma` copies 1 KB of x per token per stage;
  - `splitk` loads the x tile per stage with one 2-D tensor-map TMA (128-byte swizzle, rows ≥ m zero-filled);
  - `rowpair` treats the 8 K-segments of x as 8 "tokens".
- **Decode.** 1 `SHF` + 2 `LOP3` + 4 `PRMT` per 8 weights per lane. The register results go straight into
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; [format.md](format.md#tile-layout-16-rows--64-columns-in-1536-bytes)
  shows the code. The compiled SASS (nvcc 12.8, `tools/compile_check.sh`) has one `SHF.R.W.U32` per decoded k-step:
  - `zg_kernel<1,1,2>`: 8 HMMA, 8 SHF.R.W;
  - `zg_kernel<2,2,2>`: 32 HMMA, 16 SHF.R.W, so each decode feeds 2 token tiles;
  - `zr_kernel<4,2,1,false,8>`: 8 HMMA, 8 SHF.R.W, 32 PRMT.
- **Scale.** The per-row 2^-s is applied once per output, after the K loop.

## The four kernels

**`fullk` (zg).** Each CTA owns RT row tiles (16·RT rows) and the whole K range. The 8 compute warps take 64-column
chunks round-robin (warp w takes chunks w, w+8, …), with U chunks of loads in flight per warp, so a CTA holds
8·U·RT·1536 bytes in flight in registers (24.6 KB for config 6). A ninth warp applies the escapes while the others
stream:
1. it loads the record counts of its row tiles and all records up front;
2. for each group of 32 records it fetches the needed activation values with 4-byte `cp.async`;
3. it adds the corrections with lanes over tokens, so no two lanes write the same word and no atomics are needed.

At the end, the 8 warps' fp32 accumulators are reduced through shared memory and
`out = fmaf(sum, 2^-s, corrections)`. There is no cross-CTA reduction.

**`fullk_tma` (zt).** The same work split, fed by a producer warp. For each stage of 8 consecutive chunks, it issues
one bulk copy per row tile (contiguous in the `rowtile` layout) and one 1 KB copy of activations per token into an
ST-deep mbarrier ring. Consumer warp w takes chunk w of every stage.

**`rowpair` (zr), m = 1.** The `rowpair` layout views W as V = W.reshape(8N, K/8), so a 16-row tile holds two rows
of W, each split into 8 K-segments. With x viewed as 8 tokens (its 8 segments), one MMA yields D[i][s] = (row i of
V)·(segment s of x); the wanted values are the diagonal entries D[s][s] and D[8+s][s]. 7/8 of the MMA columns are
wasted, but the tensor pipe is idle at m = 1. In exchange, the weights go from 12 bits into A fragments with no
conversion or FMA work, and the grid has thousands of small CTAs that balance well over 132 SMs. One warp per tile
loads the first 32 escape records at kernel start. It fetches their activation values after the first weight loads
are issued, and walks any further records after the loop.

**`splitk` (zs).** One CTA per SM (G CTAs). The `blocked` layout orders tiles as (row block, chunk, row tile), and
CTA c owns tiles [c·U/G, (c+1)·U/G), so every SM streams the same bytes ±1 tile (±1536 B).
- **Producer.** One lane issues, per stage, one bulk copy of the CTA's contiguous tiles and one tensor-map TMA of
  the activation tile.
- **Consumers.** Warp w owns RW row tiles of each row block. At the end of a row block (or of its range) it stores
  its fp32 accumulators, unscaled and in fragment order, to a partial slot. Every slot is fully written on every
  call: no zeroing, atomics or counters.
- **Finalize.** A second kernel, launched with programmatic stream serialization: the main grid executes
  `griddepcontrol.launch_dependents` at its start. Before `griddepcontrol.wait` the finalize loads scales and
  computes the escape sums, overlapping the stream. After the wait it sums the partials with all loads in flight and
  writes BF16. `configs.splitk_plan` mirrors the launch plan the host code re-checks.

No instruction needs the sm_90a feature set (no `wgmma`, no `setmaxnreg`). The sources compile for plain sm_90 too
(`tools/compile_check.sh --arch sm_90`). They are built for sm_90a because that is what was measured. Any sm_90a-only
PTX added later belongs behind `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`.

## What was measured

H200 SXM, 132 SMs; method in the [README](../README.md#results-h200-sxm). Data in [`results/`](../results/).

### Where the time goes: split-K timeline

The split-K kernel records `%globaltimer` at four points per CTA: start, first stage ready, loop end, partials stored.
The finalize records entry, escape sums done, after the wait, and end. One cold-L2 replay, run R2, µs from the first
CTA start:

| N×K | m | cfg/G | first stage (mean / max) | loop end (mean / max) | finalize end | per-SM GB/s (mean / min) | aggregate TB/s |
|---|---:|---|---|---|---:|---|---:|
| 2304×8192 | 8 | 3/132 | 1.83 / 2.30 | 9.71 / 10.50 | 12.29 | 22.2 / 20.3 | 2.70 |
| 2304×8192 | 16 | 1/132 | 1.57 / 2.05 | 9.00 / 9.73 | 11.26 | 23.8 / 21.9 | 2.91 |
| 2304×8192 | 32 | 2/128 | 1.94 / 2.30 | 9.97 / 13.06 | 15.10 | 22.2 / 16.9 | 2.17 |
| 8192×1024 | 8 | 3/128 | 2.16 / 2.82 | 4.68 / 5.12 | 6.66 | 21.1 / 19.2 | 2.46 |
| 8192×1024 | 16 | 1/128 | 1.85 / 2.30 | 5.36 / 5.63 | 7.42 | 19.2 / 18.3 | 2.23 |
| 8192×1024 | 32 | 2/128 | 1.83 / 2.05 | 5.80 / 6.14 | 8.19 | 17.0 / 16.0 | 2.05 |
| 4096×8192 | 16 | 1/128 | 1.77 / 2.56 | 14.77 / 15.36 | 17.15 | 26.9 / 25.6 | 3.28 |
| 4096×8192 | 32 | 2/128 | 1.77 / 2.30 | 16.66 / 19.46 | 21.50 | 23.7 / 20.2 | 2.59 |
| 8192×4096 | 16 | 4/128 | 2.27 / 3.07 | 14.13 / 15.62 | 17.41 | 28.3 / 25.6 | 3.22 |
| 8192×4096 | 32 | 2/132 | 1.96 / 2.82 | 17.15 / 20.22 | 22.02 | 22.3 / 18.9 | 2.49 |

![split-K timeline](../results/splitk_timeline.png)

On this GPU `%globaltimer` advanced in steps of about 0.26 µs, so single readings carry that uncertainty; means over
CTAs are finer. Across all three runs:
- **Startup.** The first stage landed 1.4–2.3 µs after CTA start, on average.
- **Tail.** Partial stores, the PDL hand-off (0.26–0.51 µs) and the finalize took 1.5–3.1 µs after the last CTA's
  loop ended.
- **Graph launch.** The paired harness measured a fixed graph-replay cost of 3.6–4.4 µs; every kernel pays it,
  cuBLAS included.

The ideal stream times of these matrices are 3.5–14 µs: their BF16 bytes at the nominal 4.8 TB/s, or 2.6–10.5 µs
for the 12-bit bytes. Next to that, 1.4–2.3 µs of startup plus 1.5–3.1 µs of tail is large, and largest for the
8192×1024 projection, where the 12-bit kernels lost.

### What limits each regime

**Bytes in flight (Little's law).** An SM sustains rate R only if it keeps R × latency bytes in flight. In earlier
measurements on the same GPU (not part of `results/`):
- dependent global loads under load took about 1–2 µs;
- per-SM streaming saturated near 35 GB/s (about 4.6 TB/s over 132 SMs) with several CTAs per SM.

That calls for roughly 35–70 KB in flight per SM. `fullk` config 6 keeps 24.6 KB per CTA in registers and runs
about 2 CTAs per SM on the 256-CTA grid of 4096×8192. At m = 1 that shape took 14.57 µs for 50.4 MB in the paired
harness: 3.46 TB/s of compressed bytes, against 3.14 TB/s for cuBLAS on 67.1 MB of BF16. `splitk` has a ring of
about 210 KB per CTA, yet with one CTA per SM it streamed only 16.5–28.3 GB/s per SM (2.0–3.3 TB/s aggregate). A
deep ring in a single CTA did not reach the rate of several CTAs per SM. We did not isolate the cause; a single
producer lane issuing every copy is one candidate.

**Balance over 132 SMs.** 2304 rows are 144 row tiles, so a full-K grid with one tile per CTA leaves 12 SMs with
twice the work. At m = 16 on 2304×8192, the equal-byte `splitk` (CTA weight bytes 213.5–215.0 KB) took 15.87 µs.
The best `fullk` took 17.79 µs (a 5-replay screen) and cuBLAS 17.47 µs. At m = 1, many-CTA designs balance well:
`rowpair` won on 8192×1024 and 1024×4096. On those shapes, and on 2304×8192, an uncompressed many-CTA GEMV was within
3% of the best 12-bit kernel at m = 1.

**Activation traffic at m = 32.** In each 64-column chunk, a CTA reads RT × 1536 B of weights from DRAM and
m × 128 B of activations from L2. With `fullk`'s 2-row-tile configurations at m = 32 that is 4096 B of
activations per 3072 B of weights. With `splitk`'s 8 or 16 row tiles per block it falls to 1/3 or 1/6 of the weight
bytes, but `splitk` runs into the one-CTA-per-SM rate above. At m = 32, the 12-bit kernels beat cuBLAS only on
4096×8192 (1.09×, `fullk` config 8). They lost on 2304×8192 (0.97×), 8192×4096 (0.83×) and 8192×1024 (0.76×).

**Escape handling.** Escapes are rare, but their handling must not create serial chains. An earlier cluster split-K
variant (not included here) used the same escape-warp code as `fullk` / `fullk_tma`:
- The first version did one global activation load and one shared-memory CAS loop per record and token. It took
  54.9 µs at 4096×8192, m = 32, against 23.0 µs with escapes disabled.
- Fetching all activations of 32 records with 4-byte `cp.async` and assigning lanes to tokens instead of using
  atomics brought this to 25.8 µs (24.9 µs with escapes disabled).

Both figures are same-process medians, launch cost included.

### When each kernel won (run R2, fastest 12-bit configuration)

| m | 2304×8192 | 8192×1024 | 4096×8192 | 8192×4096 | 1024×4096 |
|---|---|---|---|---|---|
| 1 | `fullk_tma` 0 | `rowpair` 6 | `fullk` 6 | — | `rowpair` 3 |
| 8 | `fullk` 6 | `fullk` 1 (lost) | `fullk` 6 | — | `fullk` 6 |
| 16 | `splitk` 1, G=132 | `splitk` 1, G=128 (lost) | `fullk` 3 | `fullk` 10 | — |
| 32 | `splitk` 2, G=128 (lost) | `splitk` 2, G=128 (lost) | `fullk` 8 | `fullk` 8 (lost) | — |

"lost" means cuBLAS was faster; "—" means the shape was not measured at that m. `configs.MEASURED` holds this table
as the default selection; `tune()` re-times every candidate on your GPU.

### Variants that were measured and dropped

- **Stream-K with atomic partials** (`zk`, not included). CTAs split the (row block, chunk) sequence evenly; partial
  sums go to a workspace with `red.add.f32`, and arrival counters make the last CTA write the output. In the paired
  harness it ran 1.01× cuBLAS at 4096×8192, m = 1 (20.86 vs 21.15 µs) and 1.16× at 2304×8192, m = 1. Run R2's
  `fullk` reached 1.47× on the former. The tail after the loop (fence, counter, read-back) is a chain of dependent
  global round trips.
- **Cluster split-K with a DSMEM reduction** (`zc`, not included). It was exact on the GPU for clusters of 2 to 7 CTAs,
  but at m = 32 it stayed at cuBLAS level, with about 2.5 µs to the first stage and about 3 µs of cluster-reduce tail
  in its timeline. With 4-CTA clusters at one CTA per SM, CTAs started in waves (start spread 20–26 µs), although
  `cudaOccupancyMaxActiveClusters` allowed the full grid.

### A note on measurement

Flush L2 by *reading* a buffer. An earlier tuner flushed by zeroing a 64 MiB buffer. That left dirty lines whose
write-back landed in the next timed kernel and penalized kernels that read more bytes. With the zeroing flush,
same-process times ran about 6 µs above the paired harness; after switching to a reading flush the offset was about
3 µs. [`lossless_bf16/timing.py`](../lossless_bf16/timing.py) and
[`bench/bench_linear.py`](../bench/bench_linear.py) use the reading flush.
