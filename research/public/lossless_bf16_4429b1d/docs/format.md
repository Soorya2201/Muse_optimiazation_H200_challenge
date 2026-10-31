# The 12-bit format

Notation: W is an [N, K] BF16 matrix (N % 16 == 0, K % 64 == 0). A BF16 pattern is `s | e7..e0 | m6..m0`: sign,
8-bit biased exponent e, 7 mantissa bits. e = 0 means ±0 or subnormal; e = 255 means ±inf or NaN; otherwise the
value is normal. Code: [`lossless_bf16/format.py`](../lossless_bf16/format.py) (encoder and reference decoder) and
[`lossless_bf16/packing.py`](../lossless_bf16/packing.py) (layouts).

## Per-row power-of-two scale

For row r, let e_max be the largest exponent among the row's normal weights. The row shift is
s = 127 − e_max (s = 0 if the row has no normal weight), so s ∈ [−127, 126]. Multiplying the row by 2^s moves its
largest exponent to 127 and changes only exponent fields. The kernels receive `rscale[r] = 2^-s` as fp32, which is
always a normal fp32 number.

## The 16-binade window and the 12-bit code

A weight is stored directly when it is normal and its scaled exponent e' = e + s lies in [112, 127], i.e.
e ≥ e_max − 15. Then e' = 112 + E with E = e3 e2 e1 e0 ∈ [0, 15], and the scaled BF16 pattern is

```
high byte = s 0 1 1 1 e3 e2 e1
low byte  = e0 m6 m5 m4 m3 m2 m1 m0
```

Bits 6..3 of the high byte are always `0111`. The format stores the other 12 bits:

| field | bits | content |
|---|---|---|
| L byte | 8 | the low byte, `e0 m6..m0` |
| nibble | 4 | `s e3 e2 e1` |

The decoder rebuilds the high byte as `(nibble bits) | 0x38`. The original pattern is the scaled one with
e' − s written back into the exponent field.

## Escapes

All other weights are escapes:
- ±0 and subnormals (e = 0);
- ±inf and NaN (e = 255);
- normal weights with e < e_max − 15. These are more than 15 binades below the row's top binade, i.e.
  |w| < 2^-15 · 2^(e_max − 127).

An escape's 12 bits are all zero, which decodes to the BF16 pattern 0x3800 = +2^-15 (in scaled units). A record
stores the weight's position, its 16 bits widened to fp32 (`bits << 16`), and nv = −2^-15 · 2^-s as fp32. For each
escape, the kernel adds `x·w + x·nv`. The term x·nv cancels the placeholder the MMA accumulated: it is x times a
power of two, so it is exact in fp32 whenever the result is a normal number. The term x·w adds the weight's true
contribution.

## Tile layout: 16 rows × 64 columns in 1536 bytes

Tiles are stored in the order in which `mma.sync.m16n8k16.row.col` consumes its A operand. Write lane l of a warp as
g = l / 4, t = l % 4. A 64-column chunk holds four k-steps j = 0..3 of 16 columns each. For k-step j, lane (g, t)
holds the A fragments a0 (row g, columns 2t, 2t+1), a1 (row g+8, same columns), a2 (row g, columns 2t+8, 2t+9) and
a3 (row g+8, same columns).

**Bytes 0–1023: L bytes.** Lane l owns 16 bytes at offset `512·q + 16·l` for q = 0 (k-steps 0, 1) and q = 1
(k-steps 2, 3). Within those 16 bytes, 32-bit word `2·(j & 1) + hi8` holds 4 low bytes. hi8 = 0 is for columns
2t, 2t+1 and hi8 = 1 for columns 2t+8, 2t+9. Byte i of the word belongs to row g + 8·(i >> 1) and stored column
16j + 8·hi8 + 2t + (i & 1). With c = 16j + 8·hi8 + 2t, the order is therefore (g, c), (g, c+1), (g+8, c),
(g+8, c+1).

**Bytes 1024–1535: nibble words.** Lane l owns the 16 bytes at `1024 + 16·l`; word j holds the 8 nibbles of k-step
j, with byte i matching byte i of the L words:

| weight | bit positions in the 32-bit word |
|---|---|
| hi8 = 0, byte i | sign → bit 8i+7, e3 e2 e1 → bits 8i+2 .. 8i |
| hi8 = 1, byte i | e3 e2 e1 → bits 8i+6 .. 8i+4, sign → bit 8·((i+1) mod 4)+3 |

Every bit of the word is used, and a right rotation by 4 maps the hi8 = 1 fields onto the hi8 = 0 positions. One
k-step for 8 weights per lane therefore decodes in 7 integer instructions. This is the code in all four kernels:

```cuda
const uint32_t La = word_of(wq[j >> 1], (j & 1) * 2), Lb = word_of(wq[j >> 1], (j & 1) * 2 + 1);
const uint32_t Nw = word_of(wq[2], j);
const uint32_t Ha = (Nw & 0x87878787u) | 0x38383838u;                          // LOP3
const uint32_t Hb = (__funnelshift_r(Nw, Nw, 4) & 0x87878787u) | 0x38383838u;  // SHF.R.W + LOP3
const uint32_t a0 = __byte_perm(La, Ha, 0x5140), a1 = __byte_perm(La, Ha, 0x7362);   // PRMT
const uint32_t a2 = __byte_perm(Lb, Hb, 0x5140), a3 = __byte_perm(Lb, Hb, 0x7362);   // PRMT
```

`0x5140` interleaves bytes 0, 1 of L and H into two BF16 values (low byte from L, high byte from H); `0x7362`
does the same for bytes 2, 3. [`format.decode_chunk_major`](../lossless_bf16/format.py) runs exactly these operations
in PyTorch, and the CPU tests decode every layout through it.

## Column order inside a chunk

At stored position `16j + 8h + 2t + e` inside a 64-column chunk (k-step j, mma column 2t + e + 8h), the format puts
actual column `16t + 4j + 2h + e`. With this order, the activations lane t needs for the chunk's 4 k-steps are the
32 contiguous bytes `x[16t .. 16t+15]`: two 16-byte loads per token instead of eight 4-byte loads. The kernels read
x in its natural order; only W is permuted. See `packing.k_permutation`.

## Layouts

All layouts share the encoding above and the per-row scales. Each is a separate copy of the 12-bit tiles.

| layout | tile order | escape table | kernels |
|---|---|---|---|
| `rowtile` | [N/16][K/64] | per row tile: int32 [N/16][emax][4]. Entry 0 is {count}; entries 1..count are {column, row in tile, bits(w), bits(nv)} | `fullk`, `fullk_tma` |
| `blocked` (RT = 8, 16) | [N/(16·RT)][K/64][RT], the same tiles as `rowtile` reordered so that a CTA's equal share is one contiguous byte range | per row: int32 [N][emr][4] {column or −1, bits(w), bits(nv), 0} | `splitk` |
| `rowpair` | `rowtile` encoding of V = W.reshape(8N, K/8). A 16-row tile of V is rows 2t, 2t+1 of W, each split into 8 K-segments; the shift s is chosen per (row, segment) | per tile, as `rowtile`, columns in V coordinates | `rowpair` |

## Why it is lossless

For a directly stored weight, the sign, all 8 exponent bits (e = e' − s) and all 7 mantissa bits are recovered.
For every other pattern, the escape record holds all 16 bits. So every one of the 65,536 BF16 patterns round-trips
exactly, including ±0, subnormals, ±inf, and NaNs with any payload, and there are no preconditions on the input.
The CPU tests ([`tests/test_format.py`](../tests/test_format.py)) check this for every layout on:
- all 65,536 patterns, sorted, shuffled, and with every row holding all of them;
- Gaussian rows at scales from 2^-126 to 2^120;
- rows whose magnitudes span 267 binades;
- the special values.

They also check that the escape set equals the rule above. `unpack_reference` reconstructs the matrix with the
kernels' own decode operations. The packed tensors are byte-identical to those produced by the packing code of the
measured runs; this was checked on CPU against that code, including adversarial rows.

## Numerics of the kernels

The storage is exact. What the kernels compute from it is y = x Wᵀ with fp32 accumulation on tensor cores, rounded to
BF16 (round to nearest even), plus the following points:

- In-window weights enter the MMA as w·2^s, and the fp32 accumulator is multiplied by 2^-s once per output. Scaling by
  a power of two is exact unless the result leaves the fp32 range.
- An escape's placeholder 2^-15 does take part in the fp32 accumulation before the correction removes it. The
  accumulator's rounding can therefore differ, at fp32 level, from a GEMM that never saw the placeholder.
- The order of the escape terms differs per kernel:
  - `fullk`, `fullk_tma`: `out = fmaf(acc, 2^-s, Σ fmaf(x, w, x·nv))`; an escape warp builds the sum.
  - `splitk`: `out = fmaf(Σ partials, 2^-s, Σ x·nv) + Σ x·w`; the finalize kernel does this.
  - `rowpair`: `out = (acc·2^-s + Σ x·nv) + Σ x·w`.

  With a one-hot activation (a power of two at one column), every kernel returns the weight column bit-exactly
  (compared as values, so +0 equals −0), with one exception. In `fullk` and `fullk_tma`, an escaped weight more
  than about 39 binades below its row's top binade loses x·w to rounding inside `fmaf(x, w, x·nv)`, and the output
  is 0 (`tests/test_emulation.py::test_fullk_escape_order_limit`). For dense activations such a term lies far below
  BF16 output resolution.
- The kernels are compiled with `-O3 --use_fast_math`, the flags of the measured build. Fast math flushes fp32
  subnormals to zero. So subnormal weights, which are always escapes, contribute 0, and fp32 intermediates or
  results below 2^-126 become 0. Scaled weights lie in [2^-15, 2), so this matters only for extreme activations
  or results.
- ±inf and NaN weights propagate through x·w as in IEEE arithmetic.

**Measured accuracy on the H200.**
- *Random activations.* With N(0, 1) activations, the relative L2 error against an fp64 reference was 1.66–1.67·10^-3
  on the four BF16 shapes. This was an early GPU self-test of a stream-K variant (same format, same escape
  arithmetic as `fullk`). Rounding the exact product to BF16 alone gives 1.67·10^-3 for this distribution, computed
  on CPU.
- *Checks before timing.* Every configuration timed in the runs of [`results/`](../results/) first passed two
  checks on the GPU with that run's weights:
  - one-hot probes bit-exact at a full and at a partial token tile; for `splitk` and `rowpair`, also at escaped
    columns;
  - random-activation error below 5·10^-3.

  The `zs` (splitk) self-test also covered an escape-heavy synthetic matrix with a 10^-20 weight, a row with 40
  zeros, and 37-CTA grids that make CTAs span several row blocks: 96 runs, no failure. That test ran in a separate
  integration run with the same kernel source.
- *Checks during timing.* The paired harness checked every timed output: relative L2 error ≤ 10^-2 overall, and
  ≤ 10^-3 on rows with a single power-of-two activation.

## Size

Bytes of each layout for synthetic N(0, 1/K) weights (packed on CPU; escape tables padded to the longest row tile
or row):

| N×K | BF16 MB | 12-bit tiles MB | `rowtile` total | `blocked` (RT 8) | `rowpair` |
|---|---:|---:|---:|---:|---:|
| 2304×8192 | 37.75 | 28.31 | 28.37 (+0.20%) | 28.51 (+0.68%) | 28.51 (+0.72%) |
| 8192×1024 | 16.78 | 12.58 | 12.66 (+0.65%) | 12.88 (+2.34%) | 13.04 (+3.65%) |
| 4096×8192 | 67.11 | 50.33 | 50.42 (+0.18%) | 50.61 (+0.55%) | 50.66 (+0.65%) |
| 8192×4096 | 67.11 | 50.33 | 50.46 (+0.26%) | 50.89 (+1.11%) | 50.99 (+1.30%) |
| 1024×4096 | 8.39 | 6.29 | 6.31 (+0.21%) | 6.35 (+0.85%) | 6.37 (+1.17%) |

The overhead consists of 4 bytes per row (per row segment in `rowpair`), 16 bytes per escape record, and the table
padding. `PackedWeight.nbytes(layout)` reports it for any matrix.

## Escape rate

Counted on the GPU at pack time in 15 runs ([`results/escape_counts.csv`](../results/escape_counts.csv)). The weights
were synthetic N(0, 1/K), rounded to BF16 and regenerated in every run:

| N×K | weights | escapes, min–max | rate |
|---|---:|---:|---:|
| 2304×8192 | 18,874,368 | 1222–1342 | 6.5–7.1·10^-5 |
| 8192×1024 | 8,388,608 | 391–463 | 4.7–5.5·10^-5 |
| 4096×8192 | 33,554,432 | 2239–2438 | 6.7–7.3·10^-5 |
| 8192×4096 | 33,554,432 | 1948–2051 | 5.8–6.1·10^-5 |

**Trained checkpoints were not measured.** The rate depends on each row's dynamic range: a row whose largest weight
is far above the rest pushes more weights out of its window. Studies of LLM weights report BF16 exponents
concentrated in a few values (DFloat11, ZipServ), but that is a per-tensor statistic, and we have not measured this
per-row window on real weights. `pack(W).n_escapes` gives the count for any matrix. At 16 bytes per record, escapes
cost about 0.1% extra bytes at a rate of 10^-4, and 1% at 10^-3.
