# muse_decode_layer: the hardware and the layer

Study notes for the Open Kernel Challenge problem `muse_decode_layer`. The task is one
decoder layer of Muse Spark Open as a single H200 runs it under TP8/EP8, timed at decode
and speculative-verify batch sizes. Numbers marked "(est.)" are my own back-of-envelope
estimates. The other numbers come from TASK.md, the reference, or measured okc runs.

---

## Part 1: H200 (SM90, Hopper) as it matters to a decode layer

### 1.1 The headline numbers

| Resource | H200 SXM | Why it matters here |
|---|---|---|
| HBM3e bandwidth | 4.8 TB/s | A decode layer is almost entirely bandwidth-bound: about 130–200 MB of weights per step plus the KV cache. |
| SMs | 132 | Any kernel with fewer than ~132 CTAs (or fewer than 2–4 waves' worth of loads in flight) leaves bandwidth unused. |
| L2 | 50 MB (two partitions) | The judge flushes L2 before every replay, so weights are always cold. L2 only helps *within* a step (for example re-reading an expert's weights for a second token block). |
| Shared memory | up to 228 KB per SM | Space for deep cp.async/TMA pipelines (4–8 stages of 16–32 KB). |
| Registers | 64K × 32-bit per SM | Limits how many dequantized tiles a CTA can hold. |
| BF16 dense tensor | ~989 TFLOP/s (wgmma) | Only matters for verify-32 (T=256) attention and MoE. |
| FP8 dense tensor | ~1979 TFLOP/s | FA3 runs QK^T and PV in FP8 on the FP8 KV cache. |
| Clock | 1.98 GHz (judge machine) | |

**Roofline crossover.** 989e12 / 4.8e12 ≈ 206 FLOP/byte. A BF16 GEMV at T tokens does
2T FLOP per 2 weight bytes (T FLOP/byte), so the dense projections stay bandwidth-bound
until T ≈ 200. An NVFP4 weight does 2T FLOP per 0.5625 bytes (3.6T FLOP/byte), so the
expert GEMMs cross over at about T_e ≈ 57 tokens per expert. Only the hottest experts at
verify-32 get there.

**The ALU trap of W4A16.** Every 4-bit weight has to be turned into BF16 before a BF16 MMA
can use it. At 128 int lanes per SM per clock, one decode token's 3 experts (~150 M
codes) cost about 150e6 × ops_per_code / (132 × 128 × 1.98e9) s. At 8 ops per code that
is ~36 µs, which is *more* than the 16 µs it takes to stream those bytes. So the dequant
has to cost ~1–2 integer ops per code (packed bit tricks, `lop3`, `prmt`, `bf16x2`
multiplies), or it becomes the bottleneck even at T=1. Marlin does this in CUDA. A naïve
Triton dequant does not (see IMPROVISATIONS.md, v4 lab).

### 1.2 Memory-system features

* **TMA (Tensor Memory Accelerator).** An asynchronous bulk copy engine. One thread issues
  a 2D/3D tile copy global→shared, completion is signalled on an mbarrier, and no
  registers are spent on addresses. It is ideal for dense weight streaming. It cannot
  gather the scattered 128-byte rows of a `page_size = 1` KV cache, except with
  im2col/gather4 tricks on newer parts. On H200 the KV gather uses `cp.async` (16 B per
  thread) or plain vectorized loads.
* **cp.async + multi-stage pipelines.** These keep 3–6 tiles in flight per CTA. Little's
  law for HBM: 4.8 TB/s × ~0.8 µs latency ≈ 4 MB must be in flight GPU-wide, or ~30 KB per
  SM. A GEMV CTA with a single 8 KB tile in flight cannot saturate HBM.
* **L2 and sector granularity.** L2 lines are 128 B with 32 B sectors. A KV row at
  page_size=1 is 128 B (2 heads × 64 × FP8), so reading both heads of a token in one CTA
  touches whole lines. Splitting heads across CTAs makes two 64 B half-line reads at
  different times.
* **Distributed shared memory / thread-block clusters (up to 8 or 16 CTAs).** CTAs in a
  cluster can read each other's shared memory and multicast TMA loads. This is useful for
  split-K or split-KV reductions without a global-memory round trip, and for sharing one
  weight tile across CTAs that work on different token blocks of the same expert.

### 1.3 Compute features

* **WGMMA (warpgroup MMA).** 4 warps issue one asynchronous MMA with M=64 and N in
  8…256. B always comes from shared memory, and A from shared memory or registers. For
  decode GEMVs the trick is **swap-AB**: put the weights on the M=64 side and the few
  tokens on the N side (N=8 or 16), so the tensor core does no padding waste on M.
* **mma.sync (Ampere-style).** Synchronous, m16n8k16. Marlin still uses it on Hopper,
  which is one reason it reaches only a fraction of the H200's peak at verify-32.
* **Mixed-input MMA does not exist in hardware.** BF16 × NVFP4 always means "dequantize
  in registers, then BF16 MMA". FP4 tensor-core MMA (with block scales) is Blackwell
  only (SM100). FP8 × FP8 is native but is not allowed for activations here.

### 1.4 Launch overhead, CUDA graphs and PDL

* The judge captures `kernel_fn` as a **CUDA graph** and replays it, with the ~4 µs
  replay cost subtracted. Graphs remove CPU launch latency, but each kernel node still
  costs roughly 1–3 µs of GPU-side "tail + ramp": the last wave drains, the next grid is
  scheduled, and its prologue loads start cold.
* The SGLang layer runs **~36–40 kernels** per step. At T=1 the actual data movement is
  ~35–40 µs, but SGLang takes ~170 µs (measured). Launch gaps and under-filled kernels
  account for a large share of that.
* **PDL (Programmatic Dependent Launch, `cudaLaunchAttributeProgrammaticStreamSerialization`
  plus `griddepcontrol.wait` / `launch_dependents`)** lets kernel N+1 launch and run its
  prologue (weight prefetch into shared memory or L2, address setup) while kernel N is
  still finishing. For a chain of bandwidth-bound GEMVs whose *weights* do not depend on
  the previous kernel, the next GEMV can start streaming weights before its activation
  input is ready. This is the single biggest generic lever for small-T decode. Triton
  exposes it through `launch_pdl` / `gdc_wait` in recent versions. In CUDA it is
  `cudaLaunchKernelEx` with the PDL attribute, and the attribute survives graph capture.
* **Side streams inside a graph** become parallel branches. SGLang already overlaps
  the router GEMV and top-8 with the pooler GEMM this way.
* **Persistent megakernels** (one launch for the whole layer, with a device-side task
  queue and grid barriers or flags between phases) remove all launch gaps and can
  overlap phases, for example prefetching the o-projection weights while attention is
  still running. This is the end state the top leaderboard entries likely approach.

### 1.5 Cold-L2 timing consequences

Every replay starts with a flushed L2, so:

* every weight byte comes from HBM once per step at best;
* tiny metadata reads (page table, `cache_seqlens`, routing ids) pay full HBM latency
  (~600–800 ns), so kernels whose first instruction is a dependent load chain are
  latency-bound at small sizes;
* there is no benefit from "the previous step left it in L2".

---

## Part 2: the Muse Spark Open layer (per GPU, TP8/EP8)

### 2.1 Configuration on one GPU

| Quantity | Value | Notes |
|---|---|---|
| hidden | 8192 | residual stream width |
| query heads / KV heads / head_dim | 16 / 2 / 64 | 128 Q heads and 16 KV heads model-wide, split TP8; GQA group 8 |
| sliding window | 2048 tokens | sliding layers use interleaved (GPT-J) RoPE, θ = 500 000 |
| full layers | no RoPE | attend to the whole context (80K in the scored loads, up to 1M in tests) |
| experts | 256 total, top-8, 32 local (EP8) | NVFP4 weights, `local_start` = busiest 32-range |
| expert width | 4096 (`D_EXPERT`) | each expert: w13 [8192 × 4096] (gate then up), w2 [4096 × 4096] |
| KV cache | FP8 E4M3, page size 1 in scored cases | layout [pages, page, 2 heads, 64] = 128 B per token per K or V |

### 2.2 Data flow with shapes (T = tokens = batch × q_len)

```
inputs: stream s  FP32 [T, 8192]   (residual stream)
        x = attn_input BF16 [T, 8192]   (= rmsnorm(s)·(1+w) from the previous layer)

ATTENTION BLOCK
 1. qkv_gate = x @ Wqkvg^T            BF16 [T, 2304]   W: 2304×8192 BF16 (37.7 MB)
       split -> q [T,16,64], k [T,2,64], v [T,2,64], g [T,16·64] (output gate)
 2. QK-norm: q,k <- rmsnorm_per_head(q,k)  (eps 1e-5, no weight)
    sliding layers: interleaved RoPE on q,k at position seq_len+i
 3. KV write: fp8(k / k_scale), fp8(v / v_scale) -> cache rows cache_loc[T]
 4. attention, causal inside the verify block, GQA 8:1:
      scores = (q·5.473)·k^T · (1/8)  (qk_scale_factor folded into the softmax scale)
      sliding: keys (pos-2047 .. pos], full: keys [0 .. pos]
      o_h = softmax(scores)·V          [T,16,64]
 5. gated output: o = rmsnorm_per_head(o_h) · sigmoid(g)    BF16 [T,1024]
 6. a = o @ Wo^T                      [T, 8192]       Wo: 8192×1024 BF16 (16.8 MB)
       (all-reduce across TP8 -> identity here)

POST-ATTENTION BOUNDARY (one fused SGLang kernel)
 7. s1 = α1·s + β1·rmsnorm(a)   (eps 1e-8), α,β per channel from the residual gate:
       β = sigmoid(w/0.3), α = sqrt(max(sigmoid(-w/0.3)·(1+β), 1e-3))   "norm-preserving"
 8. h = rmsnorm(s1)·(1+w_pre_ffn)     BF16 [T, 8192] (plus an FP32 copy for the router at T≤8)

MoE BLOCK
 9. router: logits = h @ Wr^T  FP32 [T,256]   Wr: 256×8192 FP32 (8.4 MB)
       T≤8: FP32 GEMV; T>8: BF16 hi+lo split, two BF16 GEMMs, FP32 accumulate
10. scores = sigmoid(0.177·logits); pick top-8 of (scores + bias); weights = picked scores
       renormalized to sum 1 (the bias only chooses, never weights)
11. pooler: p = h @ Wp^T  [T, 4096]      Wp: 4096×8192 BF16 (67 MB), replicated on every GPU
12. pre-expert norm: p = rmsnorm(p)·(1+w)   BF16
13. local experts (the routes whose expert is in local_start..local_start+31):
       u = p @ W13_e^T     [T_e, 8192]   NVFP4: 16.8 MB codes + 2.1 MB scales per expert
       i = silu(u[:, :4096]) · u[:, 4096:]   BF16 [T_e, 4096]
       y = i @ W2_e^T      [T_e, 4096]   NVFP4: 8.4 MB + 1 MB per expert
       post-expert norm: y' = rmsnorm(y·(1+w_post)) (eps 1e-8)
       m[t] = Σ_slots weight · y'          BF16 [T, 4096]  (all-reduce -> identity)
14. unpooler: T≤8: f = m @ Wu[:1024]^T, only this GPU's 1/8 of the columns
              (8.4 MB; the all-gather is the identity, zeros elsewhere)
              T>8: f = m @ Wu^T, the full 8192×4096 BF16 (67 MB)

POST-FFN BOUNDARY (one fused SGLang kernel)
15. s2 = α2·s1 + β2·rmsnorm(f·(1+w_post_ffn))
16. x_next = rmsnorm(s2)·(1+w_next_pre_attn)   BF16 [T, 8192]

outputs: x_next BF16 [T,8192], s2 FP32 [T,8192], topk_ids int32 [T,8], + the KV rows written
```

### 2.2.1 v6 large-token expert path

For `T = 128/256`, v6 replaces the align + two Marlin GEMMs + activation + post-norm
sequence in step 13 with the proven Track-1 CUDA path. The router's global IDs are first
mapped to local IDs on device (`0..31`, `-1` for non-local), so `local_start` never crosses
to the host and CUDA-graph capture stays safe. The weights are losslessly repacked once:
E2M1 codes and E4M3 block scales remain unchanged but are laid out in lane/MMA consumption
order. Three timed launches perform fused W13+SwiGLU, W2+gain+sum-of-squares, and normalized
weighted finalization. T<=32 retains autotuned Marlin, where it is as fast or faster.

### 2.3 Where the bytes are (per step, one GPU)

| Item | T ≤ 8 | T = 128 / 256 |
|---|---|---|
| qkv_gate + o (BF16) | 54.5 MB | 54.5 MB |
| pooler (BF16) | 67 MB | 67 MB |
| unpooler (BF16) | 8.4 MB (slice) | 67 MB |
| router (FP32) | 8.4 MB | 8.4 MB (BF16 hi+lo = same bytes) |
| experts | ~28 MB × touched (T=1: ~1–3 touched) | ~28 MB × (20–32 touched) |
| KV, sliding | B × 2048 × 256 B (B=32: 16.8 MB) | same |
| KV, full 80K | B × 80K × 256 B (B=32: 655 MB) | same |
| activations | negligible (T × 8192 × 6 B × 2) | ~25 MB at T=256 |

So: **decode-1 is ~140 MB of weights (≈ 29 µs at 4.8 TB/s), and full-attention batch 32
is dominated by 655 MB of KV (≈ 136 µs)**. The verify loads add expert *compute*. At
T=256 the busiest GPU sees roughly 400–1000 local routes over 20–32 experts, so the expert
GEMMs approach the compute side of the roofline for the hot experts.

### 2.4 The design features one by one

* **Gated attention.** The attention output is multiplied by `sigmoid(g)`, where `g` is a
  fourth projection fused into the same GEMM as q, k and v (that is why qkv_gate has 2304
  rows). Fusion opportunity: the gate and per-head norm can be applied in the attention
  epilogue or the split-KV combine kernel. My kernel does this.
* **QK-norm.** Per-head RMSNorm of q and k before RoPE. Queries and keys therefore have
  unit RMS, which bounds the logits (×5.473/8). That is what makes FP8 queries and an FP8
  KV cache numerically acceptable.
* **Sliding / full layer interleave.** Sliding layers (window 2048, RoPE) have tiny,
  fixed KV traffic, so they are weight-bound. Full layers (no RoPE, NoPE-style long
  context) are KV-bound at large batch. The scored mix weights sliding shapes 46 and full
  shapes 16 per load.
* **GQA 16:2.** Each KV head serves 8 query heads, so a decode step reads each KV byte
  once for 8 heads. For verify (L=8) one KV tile serves 64 query rows, which turns decode
  attention from a GEMV into a small GEMM (M=64), and tensor cores start to matter.
* **Router: sigmoid + bias top-8.** DeepSeek-V3-style auxiliary-loss-free balancing: a
  per-expert bias shifts *selection* only. Scores are sigmoids, not a softmax, and the
  chosen ones are renormalized. With FP32 weights, the router is the only FP32 GEMV in
  the layer. Above 8 tokens SGLang uses the BF16 hi/lo split (hi = bf16(w),
  lo = bf16(w − hi)), which keeps ~16 mantissa bits while using BF16 tensor cores.
* **Pooler → experts → unpooler.** A low-rank "MoE bottleneck": the 8192-wide stream is
  projected to 4096 (the pooler, replicated, BF16), all experts operate in the
  4096-wide space, and the unpooler maps back. Under TP8 each GPU computes only its
  1024-column slice for T≤8. Because the pooler is replicated it costs 67 MB on every GPU
  every step, which is the biggest single dense read at decode.
* **Pre-/post-expert norms.** RMSNorm before the experts and *per-expert-output* RMSNorm
  after (with gain 1+w), then a weighted sum. The post-expert norm makes each expert's
  contribution scale-free, which is why per-expert global scales can vary wildly without
  hurting the mix.
* **Norm-preserving scaled residuals (α, β).** `s ← α·s + β·rmsnorm(branch)`, with
  α² ≈ sigmoid(−w)(1+sigmoid(w)) chosen so that the stream's expected norm is preserved.
  The checks compare the *residual update* `s_out − α1α2·s_in`, because the carried part
  dominates the norm.
* **TP8 / EP8.** Attention heads and dense projections are split 8 ways (all-reduces
  after o and after the experts). Experts are split 8 ways (32 per GPU). In this problem
  the communication is the identity. The judge times the busiest GPU (`local_start` = the
  32-range with the most routes), and the router seed is chosen so that this GPU gets its
  realistic share of expert work.

### 2.5 What SGLang launches (measured, v0 profile, decode-1 sliding: 175.7 µs GPU time)

Marlin MoE 40.8 µs (2 calls) · qkv + pooler cuBLAS GEMV 33.1 µs · FA3 9.8 µs ·
o-proj 7.1 µs · router FP32 GEMV (cublas dot) 5.1 µs · split-K reduces 5.0 µs ·
~26 small elementwise, norm and routing kernels making up the rest (each 2–6 µs). The
largest *structural* inefficiency at T=1 is therefore the number of kernels, not any one
kernel. At T≥128 it is Marlin (409–579 µs, i.e. 55–74% of the step). At batch 16/32 full
layers it is FA3 at page_size 1 (194 / 376 µs against a 68 / 136 µs KV floor).
