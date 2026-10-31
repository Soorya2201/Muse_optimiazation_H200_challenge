# lossless-bf16

Exact 12-bit storage for BF16 weight matrices, decoded in registers by NVIDIA Hopper (sm_90a) GEMV/GEMM kernels.

At decode time, an LLM projection reads every weight once per step and does almost no arithmetic per byte, so its
latency is set by DRAM traffic. This repository stores BF16 weights in a fixed-length 12-bit format that
reconstructs every one of the 65,536 BF16 bit patterns exactly, and provides kernels that decode it inside
memory-bound GEMV and skinny GEMM, straight into `mma.sync` operand registers. On an H200 SXM, using projection shapes
from a 2026 open MoE LLM, the 12-bit kernels ran faster than cuBLAS at every measured point with 1–16 tokens on
the three matrices of 37 MB or more: 1.10–1.33× in same-process timing, 1.10–1.47× with launch cost removed.
cuBLAS was faster on the 17 MB 8192×1024 projection at 8–32 tokens, and at 32 tokens on two of the three larger
matrices.

## The idea

Scale each row by a power of two so that its largest exponent becomes 127. A weight whose exponent then lies in
[112, 127], the 16 binades below the row maximum, has a BF16 high byte of the form `s 0111 e3 e2 e1`. Only 4 of
those bits vary, so the weight is stored as 12 bits:

```
L byte  (8 bits) = e0 m6 m5 m4 m3 m2 m1 m0     the low byte of the scaled BF16 value
nibble  (4 bits) = s e3 e2 e1                  the varying bits of the high byte
```

Everything else is an **escape**: zeros, subnormals, inf, NaN, and values more than 15 binades below the row
maximum. An escape is stored as the all-zero code, which decodes to +2^-15. A sparse record keeps the original bits,
and the kernel adds `x*w - x*2^-15*2^-s` for it. On synthetic N(0, 1/K) weights, 4.7 to 7.3 weights in 100,000
are escapes. Tiles of 16 rows × 64 columns are stored in `mma.sync.m16n8k16` A-fragment order, so a warp turns 48 loaded
bytes per lane into the A fragments for 4 MMAs with 7 integer instructions per 8 weights:

```cuda
const uint32_t Ha = (Nw & 0x87878787u) | 0x38383838u;                          // LOP3
const uint32_t Hb = (__funnelshift_r(Nw, Nw, 4) & 0x87878787u) | 0x38383838u;  // SHF + LOP3
const uint32_t a0 = __byte_perm(La, Ha, 0x5140), a1 = __byte_perm(La, Ha, 0x7362);  // 4 x PRMT
const uint32_t a2 = __byte_perm(Lb, Hb, 0x5140), a3 = __byte_perm(Lb, Hb, 0x7362);
mma_bf16(acc, a0, a1, a2, a3, b0, b1);                                         // fp32 accumulate
```

The per-row scale is applied once per output, after the K loop. Details: [docs/format.md](docs/format.md),
[docs/kernels.md](docs/kernels.md).

## Results (H200 SXM)

![speedup over cuBLAS](results/speedup.png)

| Weight N×K | MB, BF16 → 12-bit | m | cuBLAS µs | 12-bit µs | kernel | speedup | speedup, paired |
|---|---|---:|---:|---:|---|---:|---:|
| 2304×8192 | 37.7 → 28.3 | 1 | 17.34 | 13.86 | fullk_tma | 1.25 | — |
| 2304×8192 | 37.7 → 28.3 | 8 | 17.73 | 15.07 | fullk | 1.18 | 1.21 |
| 2304×8192 | 37.7 → 28.3 | 16 | 17.47 | 15.87 | splitk | 1.10 | 1.10 |
| 2304×8192 | 37.7 → 28.3 | 32 | 17.86 | 18.43 | splitk | **0.97** | — |
| 8192×1024 | 16.8 → 12.6 | 1 | 10.02 | 9.02 | rowpair | 1.11 | 1.26 |
| 8192×1024 | 16.8 → 12.6 | 8 | 9.92 | 10.24 † | fullk | **0.97** | — |
| 8192×1024 | 16.8 → 12.6 | 16 | 9.98 | 12.10 † | splitk | **0.82** | — |
| 8192×1024 | 16.8 → 12.6 | 32 | 9.98 | 13.12 † | splitk | **0.76** | — |
| 4096×8192 | 67.1 → 50.3 | 1 | 24.51 | 18.37 | fullk | 1.33 | 1.47 |
| 4096×8192 | 67.1 → 50.3 | 8 | 24.48 | 18.46 | fullk | 1.33 | 1.44 |
| 4096×8192 | 67.1 → 50.3 | 16 | 24.48 | 19.84 | fullk | 1.23 | 1.30 |
| 4096×8192 | 67.1 → 50.3 | 32 | 25.18 | 23.07 | fullk | 1.09 | 1.13 |
| 8192×4096 | 67.1 → 50.3 | 16 | 21.70 | 19.42 | fullk | 1.12 | 1.17 |
| 8192×4096 | 67.1 → 50.3 | 32 | 21.63 | 26.14 † | fullk | **0.83** | — |
| 1024×4096 | 8.4 → 6.3 | 1 | 10.18 | 7.30 | rowpair | 1.39 | 1.81 |
| 1024×4096 | 8.4 → 6.3 | 8 | 10.27 | 9.06 | fullk | 1.13 | 1.23 |

The shapes are the decode-time projections of one GPU's tensor-parallel shard: a fused Q/K/V/gate projection
(2304×8192), the attention output projection (8192×1024), and the projections into and out of the MoE block
(4096×8192, 8192×4096, plus the 1/8 slice 1024×4096 used at up to 8 tokens). m is the number of tokens.
Weights are synthetic: N(0, 1/K), rounded to BF16. MB counts weight bytes only, at 12 bits per weight for the
12-bit format. Per-row scales and escape tables add another 0.2–0.7% in the row-tile layout, and up to 3.7% in the
other layouts (see [docs/format.md](docs/format.md#size)).

**How these numbers were measured.** All numbers come from one run on an H200 SXM (141 GB, 132 SMs). Each call
was captured in a CUDA graph. Before every replay, a buffer larger than L2 was read (64 MiB in the same-process
timing), so the weights came from DRAM (cold L2). Each replay was timed with CUDA events.
- *cuBLAS µs, 12-bit µs, speedup*: cuBLAS with its default heuristic (`torch.nn.functional.linear`) and every
  12-bit configuration, timed in the same process. Each value is the median of 24 replays; at m = 1 it is the
  median of three 24-replay medians. † marks a 5-replay screening median: that configuration was never among the
  fastest candidates, so it was not timed again. These times include the fixed cost of a graph launch (about 3–4 µs)
  on both sides, which pulls ratios toward 1.
- *speedup, paired*: from a separate harness in the same run. It alternates cuBLAS and 12-bit replays on fresh
  inputs, 60 per side, subtracts a measured fixed replay cost (3.6–4.4 µs) from both sides, and reports the median
  of the paired ratios. It exists only where the 12-bit kernel was the one dispatched in that run. A dash means
  something else was faster there: cuBLAS, another cuBLASLt algorithm, or an uncompressed kernel.

Two more runs of the identical kernels are in [results/](results/). Across the three runs, same-process speedups
vary by at most 0.035.

**Where cuBLAS wins, and why.**
- *8192×1024 at m ≥ 8.* The matrix is small, so fixed costs dominate. In the split-K timeline at m = 16, the
  first stage reaches shared memory after 1.9 µs, streaming ends at 5.6 µs, and partial stores plus the finalize
  kernel run until 7.4 µs. Reading 4.2 MB less cannot offset that, and with K = 1024 each row tile has only 16
  chunks of work.
- *m = 32 on 2304×8192 and 8192×4096.* The full-K kernel re-reads the activations from L2 in every CTA. For a
  CTA with 2 row tiles at m = 32, that is 4096 bytes of activations per 3072 bytes of weights in each 64-column
  chunk. The split-K kernel shares each activation tile across 128–256 rows, but with one CTA per SM it streamed
  only 17–24 GB/s per SM.
  [results/splitk_timeline.png](results/splitk_timeline.png) shows the breakdown.
- *m > 32* has no 12-bit kernel here.

**What the comparison does not show.** At m = 1 on the three smaller matrices, and on 1024×4096 at m = 8, an
uncompressed BF16 kernel of ours (not in this repository) measured within 3% of the 12-bit kernel in the same
runs (for example 13.76 vs 13.86 µs at 2304×8192). There, the gain over cuBLAS comes from the kernel schedule, not
from the 25% byte saving. Compression pays where streaming dominates: on 4096×8192 at m = 1–16 and on 2304×8192 at
m = 8–16, no uncompressed candidate was among the four fastest; on 8192×4096 at m = 16 the fastest uncompressed
candidate, a cuBLASLt algorithm, took 21.09 µs against 19.42 µs. Non-default cuBLASLt
algorithms, timed in the same process, beat the default by up to 6% on these shapes
([results/h200_same_process.csv](results/h200_same_process.csv)).

## Kernels

| kernel | CUDA source | tokens | structure | where it was the fastest 12-bit kernel and beat cuBLAS (run R2) |
|---|---|---|---|---|
| `rowpair` | `zr` | 1 | a 16-row tile is 2 rows × 8 K-segments; the useful outputs are the diagonal of the MMA; many small CTAs | 8192×1024, 1024×4096 |
| `fullk` | `zg` | ≤ 32 | 1–4 row tiles per CTA over the whole K; 8 warps take 64-column chunks round-robin with 16-byte loads; one warp applies escapes | 4096×8192 (all m), 2304×8192 m=8, 8192×4096 m=16, 1024×4096 m=8 |
| `fullk_tma` | `zt` | ≤ 16 | `fullk` fed by a TMA bulk-copy ring | 2304×8192 m=1 |
| `splitk` | `zs` | ≤ 32 | one CTA per SM, equal byte shares (±1 tile), fp32 partials, finalize launched with programmatic dependent launch | 2304×8192 m=16 |

## Use and reproduce

```python
import torch
from lossless_bf16 import pack, unpack_reference, linear, tune

W = weight.to("cuda", torch.bfloat16)                    # [N, K], N % 16 == 0, K % 64 == 0
p = pack(W, layouts="all")                                # rowtile + blocked (split-K) + rowpair (m = 1)
assert torch.equal(unpack_reference(p).view(torch.int16), W.view(torch.int16))   # bit-exact
y = linear(x, p)                                          # x: [m, K] bf16, 1 <= m <= 32 -> [m, N] bf16
tune(p, W, m=16)                                          # optional: time all configurations for this m
```

```bash
pip install -e .                       # PyTorch with CUDA >= 12.5 and nvcc; the extension builds on first use
pytest                                 # CPU tests everywhere; GPU tests run only on a Hopper GPU
python bench/bench_linear.py --tune --paired     # the measurement method above, on your GPU
tools/compile_check.sh                 # compile-only check in Docker (nvcc 12.8), no GPU needed
python results/plot_results.py         # regenerate the plots from the CSVs
```

## Status and limitations

- The four `.cu` files are byte-identical to the kernel sources measured on the H200 (SHA-256 in
  [lossless_bf16/csrc/README.md](lossless_bf16/csrc/README.md)). The packing code produces byte-identical tensors to
  the packing code used in those runs; this was checked on CPU against that code.
- **This package has not been run on a GPU.** The Python wrapper, the bindings, the `torch.utils.cpp_extension.load`
  build, the tuner and the benchmark script are new. They compile without errors for sm_90a and sm_90 with nvcc 12.8
  in Docker; ptxas register counts of the `rowpair` kernels match those the measured build reported on the H200. The
  GPU tests in `tests/test_gpu.py` are written but have not been run.
- Measured on one GPU type (H200 SXM) with synthetic weights. No other GPU and no trained checkpoint was measured.
- The format applies to BF16 with N % 16 == 0 and K % 64 == 0 (`rowpair`: K % 512 == 0). The kernels cover
  1 ≤ m ≤ 32. They are built for sm_90a only, so they run on compute capability 9.0 (H100, H200) and nowhere else;
  only the H200 was measured.
- Storage is exact for every bit pattern. The GEMM uses each weight's exact value with fp32 accumulation, but the
  kernels are built with `--use_fast_math`, as measured. So subnormal weights contribute zero, and fp32 results
  below 2^-126 flush to zero. In `fullk` / `fullk_tma`, an escaped weight more than about 39 binades below its row's
  top binade can lose its product to fp32 rounding. `splitk` and `rowpair` keep escape products apart and are exact
  for one-hot inputs. See [docs/format.md](docs/format.md#numerics-of-the-kernels).
- The extra layouts (`blocked`, `rowpair`) each hold another 12-bit copy of the matrix. Pack only the ones you use.

## Related work

- **DFloat11** ([arXiv 2504.11651](https://arxiv.org/abs/2504.11651)) codes exponents with Huffman codes, at about
  11 bits per weight, and decompresses to BF16 before a standard GEMM.
- **ZipNN** ([arXiv 2411.05239](https://arxiv.org/abs/2411.05239)) separates exponent bytes and entropy-codes them,
  for model storage and transfer.
- **NeuZip** ([arXiv 2410.20650](https://arxiv.org/abs/2410.20650)) ANS-codes exponents and decompresses layer by
  layer to reduce memory.
- **Unweight** ([Cloudflare, 2026](https://research.cloudflare.com/papers/unweight-2026.pdf)) uses palette- and
  Huffman-coded exponents with Hopper kernels that rebuild BF16 in shared memory for WGMMA. It reports losses
  against cuBLAS at small batch on H100.
- **ZipServ** ([arXiv 2603.17435](https://arxiv.org/abs/2603.17435)) is a fixed-length 3-bit exponent-window code
  with a bitmap fallback (about 11.3 bits per weight), decoded into `mma.sync` registers. It reports results on
  RTX 4090 and L40S at batch 8–32.
- **Marlin** ([arXiv 2408.11743](https://arxiv.org/abs/2408.11743)) reshuffles quantized weights offline into
  `mma.sync` fragment order and streams them with wide loads and evict-first hints. The tile layout here applies
  the same idea to an exact format.
- **Stream-K** ([arXiv 2301.03598](https://arxiv.org/abs/2301.03598)) divides the work evenly across CTAs. The
  `splitk` kernel uses equal byte shares in the same spirit. A stream-K variant of this work was measured and
  dropped ([docs/kernels.md](docs/kernels.md#variants-that-were-measured-and-dropped)).

Like ZipServ, this format is fixed-length and decoded in registers. It differs in using a 16-binade window per row
with a power-of-two row scale, and sparse escape records instead of a fallback bitmap.

## Citation

```bibtex
@software{mrnirmata_lossless_bf16_2026,
  author  = {MrNirmata},
  title   = {lossless-bf16: exact 12-bit storage for {BF16} weights, decoded in registers on {Hopper} GPUs},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/MrNirmata/lossless-bf16}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
