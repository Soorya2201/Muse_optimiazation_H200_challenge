# muse_moe_w4a16: the hardware and the model, as they matter to this kernel

This note explains two things: the parts of the H200 (Hopper, sm_90a) that decide how fast this
kernel can run, and how Muse Spark Open's routed-expert block is built, including why the
benchmark shapes look the way they do. Numbers marked "measured" come from okc runs in this
folder. Everything else is a datasheet number or arithmetic.

---

## 1. The computation in one paragraph

Current-implementation clarification (Oct 29): the promoted v18 runs routing, W13,
SwiGLU, W2, normalization and routed summation in one persistent launch. The sections
below also describe earlier multi-launch versions. In the WGS path, phase-A epilogues
release-add credits to `doneAs[job, B-K-stage]`; the phase-B activation producer waits
for the eight contributing warpgroup credits with acquire loads. There is no full-grid
A-to-B barrier. The grid barrier protects the final full-row norm/sum and scratch reset.

The main WGS build uses 231,168 bytes of shared memory, 576 threads, and one CTA per SM.
Both the original and the tested register-redistribution build report zero local-memory
allocation. The latter releases producer registers and requests 112 consumer registers,
but it remains at one CTA per SM and did not improve the aggregate score. Adding another
shared-memory ring is not a free tuning parameter at this capacity.

Minimal NVCC 13.0 device SASS shows that the old nine-op dequantizer's two literal shifts
already use IMAD.SHL. The eight-op port removes one IMAD while retaining the same five
LOP3 and two SHF operations. Identical-input measurements show at most small benefits on
larger batches and slight regressions on tiny batches. Source instruction counts alone
do not predict a whole-kernel speedup. See PUBLIC_RESEARCH.md for the measured controls.

Every token picks 8 of 256 experts. This GPU owns experts 0-31 (8-way expert parallelism, EP8)
and computes only the (token, expert) pairs that land on them. For each pair:

```
h     = x[t] @ W13[e]^T            # [4096] x [8192 x 4096]^T -> 8192  (gate | up)
inter = silu(h[:4096]) * h[4096:]  # SwiGLU -> 4096
y     = inter @ W2[e]^T            # [4096] x [4096 x 4096]^T -> 4096
y     = y * (1 + post_norm_w)      # post-expert RMSNorm weight
y     = y / rms(y)                 # RMSNorm across the 4096 outputs of this pair
out[t] += g[t, slot] * y           # weighted sum over the token's local experts
```

W13 and W2 are NVFP4. Activations are BF16. The output is BF16 `[M, 4096]`.

## 2. NVFP4 storage and its cost per expert

NVFP4 stores one weight as `e2m1 * block_scale * global_scale`:

| piece | format | granularity | bits per weight |
|---|---|---|---|
| code | E2M1 (sign, 2 exponent bits, 1 mantissa bit): values 0, 0.5, 1, 1.5, 2, 3, 4, 6 | per weight | 4 |
| block scale | FP8 E4M3 | 16 consecutive weights along K | 0.5 |
| global scale | FP32 | whole tensor (per expert here) | ~0 |

That is 4.5 bits, or 0.5625 bytes, per weight. One expert has 3 x 4096 x 4096 = 50.3 M
weights, so it is **28.3 MB**: W13 is 18.9 MB and W2 is 9.4 MB. All 32 local experts
together are 906 MB, about 18x the 50 MB L2. Weights therefore stream from HBM on every call,
and the benchmark's cold-L2 replays make sure of it.

## 3. H200 (Hopper) features that matter here

### 3.1 Memory hierarchy

| level | size | bandwidth (approx.) | role in this kernel |
|---|---|---|---|
| HBM3e | 141 GB | **4.8 TB/s** | weights stream from here once per call: the roofline |
| L2 | 50 MB (two partitions) | several TB/s | activations (x, inter, y') stay resident; cold before each replay |
| SMEM / L1 | 228 KB per SM, unified with L1 (256 KB) | ~128 B/clk/SM | staging buffers for TMA rings, reductions, routing lists |
| registers | 64 K x 32-bit per SM (256 KB) | n/a | accumulators, dequantized A fragments, register prefetch |

With 132 SMs at about 1.98 GHz, each SM's share of HBM bandwidth is about 36 GB/s, or roughly
18 bytes per clock. **Little's law** gives the bytes each SM must keep in flight:
`36 GB/s x ~1.5-3 us loaded latency = 55-110 KB per SM`. A kernel that holds only one or two
16-byte loads per thread in flight can't reach that. This is the main reason the first versions
here ran at 2-3 TB/s instead of about 4.5.

### 3.2 SM compute resources

* 4 sub-partitions (SMSPs) per SM. Each issues 1 warp-instruction per clock.
* FP32 pipes: 128 lanes per SM. **INT32 pipe: 64 lanes per SM**, which is half rate. LOP3, SHF
  and PRMT, the ops that do dequant bit work, run on that half-rate pipe.
* Tensor cores: BF16 dense peak is about 989 TFLOP/s with WGMMA. Warp-level `mma.sync`
  (the Ampere-style instruction used here) gets a lower fraction, roughly 60-70%.

**Dequant budget.** HBM brings in 4.8e12 / 0.5625 = 8.5e12 weights/s. The INT pipe executes
528 SMSPs x 16 lanes x 1.98e9 = 1.67e13 lane-ops/s. So a kernel can afford only
**about 2 integer ops per weight** before the dequant work, not the memory, sets the speed.
The issue rate (about 3.9 instructions per weight in total) is a second cap. This is why the
bit layout of the repacked weights matters (section 5).

### 3.3 No FP4 tensor cores

Hopper tensor cores take FP16, BF16, TF32, FP8 (E4M3/E5M2) and INT8. They do **not** take FP4.
Blackwell (sm_100) added `cvt.e2m1x2` and FP4 MMA. On H200, every E2M1 code has to be
converted in registers to a BF16 value, multiplied by its block scale, and only then fed to a
BF16 MMA. This is the "W4A16" path: 4-bit weights and 16-bit activations. FP8 MMA isn't an
option either, because the activations must stay BF16.

### 3.4 TMA, mbarriers, WGMMA, clusters

* **TMA** (`cp.async.bulk`) moves bytes from global memory to shared memory asynchronously.
  It is issued by one thread and completes on an **mbarrier** with a transaction count. The 1-D
  bulk form needs no tensor map, only a 16-byte-aligned address and a size. Because the weights
  are repacked into one contiguous stream per (expert, item, warp), a single instruction can
  put kilobytes in flight without spending registers. That is how v4 reaches Little's-law
  depth.
* **WGMMA** is the warpgroup-wide (128-thread) asynchronous MMA. It reads B, and optionally A,
  from shared memory, with M = 64 per warpgroup. For decode, the token dimension is 1-8, so M
  would be mostly padding. It pays off only for "hot" experts with 64 or more tokens, which do
  appear in verify shapes (section 4.3).
* **Thread-block clusters / DSMEM** let up to 8 or 16 CTAs share shared memory, and TMA can
  multicast one load to several CTAs. Here each weight byte is used by exactly one CTA, so
  multicast doesn't help the weight stream. Clusters could help split-K reductions.

### 3.5 CUDA graphs and launch overhead

The benchmark captures `kernel_fn` as a CUDA graph and subtracts about 4 us of fixed replay
cost. Every kernel boundary inside the graph still costs roughly 1-3 us: launch latency, the
tail where the last CTAs finish, and the ramp where the next kernel's first loads are still in
flight. The decode-1 floor is only 11.4 us, so three or four kernel boundaries are a large
share of the total.

**Programmatic dependent launch** (PDL: `griddepcontrol.launch_dependents` / `.wait`) lets
the next kernel's CTAs start during the previous kernel's tail. They can do work that doesn't
depend on it, such as routing and weight prefetch, and block only before they read its
outputs. PDL is captured into graphs as programmatic edges.

## 4. Muse Spark Open's MoE design and why the shapes look the way they do

### 4.1 Expert geometry

* 256 routed experts, top-8, hidden width H = 4096, expert intermediate width I = 4096. Each
  expert is as wide as the model, which is unusually large; many MoEs use I much smaller than H.
  So one expert is a full-size FFN (50 M parameters), and reading it costs 28.3 MB.
* SwiGLU: W13 stacks gate rows `[0, I)` over up rows `[I, 2I)`. A kernel that computes the
  matching gate and up rows in the same CTA can apply SwiGLU in the epilogue and never write
  `h` (8192 wide) to memory.
* **Post-expert RMSNorm**: each expert's output is scaled by `(1 + w)` and RMS-normalized
  before the router weight is applied. This is unusual. Most MoEs sum the raw expert outputs.
  It means the expert's 4096 outputs need a reduction across the full row (sum of squares)
  before any of them can be weighted. That cross-CTA dependency is why there is a separate
  finalize step.

### 4.2 Expert parallelism (EP8)

With 8 GPUs, each holds 256 / 8 = 32 experts and receives every token. On average a token
sends 8 x 32 / 256 = 1 pair to each GPU. A step waits for the busiest GPU, and the benchmark
rotates expert ids so that this GPU is the busiest. Each GPU then all-reduces its partial
outputs; that all-reduce is outside this problem.

### 4.3 Skewed routing

The router scores are uniform noise plus a heavy-tailed (Student-t, 2 degrees of freedom)
popularity per expert. The heavy tail matters: a few experts get a popularity boost large
enough that they land in almost every token's top-8. Measured with the same recipe in
`prepare()` (these are diagnostics, not the judge's data):

| bs | local pairs | experts touched | largest expert's tokens |
|---|---|---|---|
| 1 | 2 | 2 | 1 |
| 16 | 21 | 6 | 13 |
| 32 | 70 | 12 | 32 |
| 8 | 13 | 8 | 5 |
| 128 | 268 | 23 | **128** |
| 256 | 452 | 24 | **91** |

What this means for the kernel:

1. **Bytes scale with experts touched, not with tokens.** bs 128 and bs 256 touch almost the
   same 23-24 experts, so their weight traffic is nearly identical (about 650-680 MB). That is
   why their floors are close (101 us vs 109 us).
2. **Per-expert token counts are bimodal.** Most experts see 1-10 tokens (a GEMV-like regime
   where weight bandwidth is the limit), but one or two "hot" experts see 64-128 tokens (a
   small-GEMM regime where tensor-core throughput and dequant reuse matter). One kernel
   configuration has to handle both, or pick per expert at run time.
3. A design that splits a hot expert's tokens into chunks and runs the chunks *one after
   another* re-reads that expert's weights once per chunk. v1 did this and lost up to 2x on
   bs 128 and bs 256.

### 4.4 Decode vs. verify shapes

* **Decode** (bs 1, 16, 32): one token per request. Pairs per expert are mostly 1. Pure weight
  streaming; launch latency and tail effects dominate at bs 1.
* **Verify** (bs 8, 128, 256 = 1, 16, 32 requests x 8 draft tokens): speculative-decoding
  verification checks an 8-token draft per request in one step. A request's 8 tokens tend to
  route to similar experts, so per-expert counts grow faster than with independent tokens. This
  is where hot experts and compute start to matter.

### 4.5 The floors

`TASK.md`'s floor is `max(bytes / 4.8 TB/s, flops / 989 TFLOP/s)`. It counts only
0.75 x the expected number of distinct experts (`ROUTING_SHARE`). The real touched count, and
so the real byte floor, is higher. For bs 256 that is 24 experts x 28.3 MB = 680 MB, or
**142 us**, against the quoted 109 us. Compute isn't the limit even at bs 256:
452 pairs x 2 x 50.3 M = 45 GFLOP, about 46 us at WGMMA peak. With `mma.sync` and the dequant
work in the same SMs, though, compute is no longer negligible.

## 5. How this submission maps onto the hardware

* **Repack in `prepare()`** (untimed). Each (expert, 32-row work item, 64-wide K step) becomes
  1152 contiguous bytes: 32 lanes x 16 B of codes for each of the two m16 tiles, plus 128 B of
  the original E4M3 block scales. Each lane's 16 bytes are exactly the codes its `mma.m16n8k16`
  A-fragments need for four k16 sub-steps. K is permuted inside each 16-block, so a lane's
  activations for one sub-step are 4 contiguous BF16 values (one 8-byte load). The weight
  permutation matches. Bits stay lossless: 4 per weight.
* **Dequant with roughly 1 integer op per weight.** Inside each 4-bit code, the bits are placed
  so that one `rotate` and one `and` produce the BF16 bit pattern `e2m1 x 2^-126` (sign at bit
  15, the two exponent bits and the mantissa bit at bits 8..6), with zero, subnormal and sign
  handling all exact. A brute-force search found a layout that covers two 32-bit words with 8
  registers at 16 ALU ops per 16 weights (section 2 of IMPROVISATIONS.md). Marlin's FP4 path
  needs about 1.75.
* **Block scale.** `prmt` + `imad` turn two E4M3 bytes into a BF16x2 equal to
  `scale x 2^119`. The integer add of the exponent bias is exact for normal E4M3 values, and
  `prepare()` checks whether any are subnormal; if so, a CVT-based variant is used. A BF16
  `HMUL2` with half-broadcast applies the scale. The product `e2m1 x scale x 2^-7` is exact in
  BF16, so the only rounding is in the FP32 accumulation.
* **Swap-AB MMA.** Weights are the M = 16 operand and tokens are N = 8, so a decode token
  wastes 7/8 of the N lanes, not 15/16 of M.
* **Split-K across 4 warps** per item, reduced through shared memory. This gives enough
  parallelism for bs 1 (2 experts x 256 items).
* **Hot-expert chunks as parallel items.** Chunk index varies fastest in the item order, so
  the CTAs working on different token chunks of the same weights run at the same time, and the
  second and later reads of those weights hit L2.
* **Three kernels with PDL.** gemm1 (SwiGLU epilogue), then gemm2 (`(1 + w)`, BF16 y' store,
  atomic sum of squares per pair), then finalize (rsqrt, router-weight sum, BF16 out).
