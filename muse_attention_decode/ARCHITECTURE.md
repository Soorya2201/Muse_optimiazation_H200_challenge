# H200 and Muse Spark Open attention: what matters for `muse_attention_decode`

This note covers the two things you need to reason about this kernel: the machine (an NVIDIA
H200, which is a Hopper GH100 die with HBM3e), and the model's attention block (Muse Spark Open's
gated attention, as SGLang serves it under TP8). Every number below is either a published spec or
is derived from the problem's own shapes. Where a number is an estimate, it says so.

---

## Part 1. The H200 (Hopper, sm_90a) as this kernel sees it

### 1.1 The shape of the chip

| Resource | H200 SXM | Why it matters here |
|---|---|---|
| SMs | 132 (out of 144 on the full die) | The grid has to fill 132 SMs. A batch-1 decode has only 1 request x 2 KV heads, so the work must be split along the key axis (split-KV / flash-decoding) to use more than a handful of SMs. |
| GPCs | 8 (about 16-18 SMs each) | A thread-block cluster must fit inside one GPC. That limits cluster size (8 is portable, 16 is "non-portable" and needs an opt-in). |
| Clock | about 1.98 GHz boost (the judge reported 1980 MHz) | Converts cycles to µs: 1 µs is about 2,000 cycles. |
| Registers | 64K x 32-bit per SM, at most 255 per thread | Bounds how many warps can be resident and so how much latency hiding is possible. |
| Shared memory | 228 KB per SM (up to 227 KB per block, 1 KB reserved per block) | Holds the K/V pipeline stages. Two CTAs per SM need to fit in 228 KB together. |
| L2 | 50 MB (two partitions joined by a crossbar) | Big enough to hold split-KV partials, but the judge flushes it before each replay, so the KV cache always comes from HBM. |
| HBM3e | 141 GB, about 4.8 TB/s | This is the roofline for the long-context full-attention layer. |

### 1.2 The memory hierarchy and its latencies (approximate, for order-of-magnitude reasoning)

| Level | Latency (cycles) | Bandwidth |
|---|---|---|
| Registers | 0 | n/a |
| Shared memory / L1 (256 KB combined per SM) | about 30 | about 128 B/clk/SM |
| Distributed shared memory (another SM in the same cluster) | about 200 | high; SM-to-SM network inside a GPC |
| L2 hit | about 250-300 | several TB/s aggregate |
| HBM (cold L2, which is what the judge measures) | about 600-1000 (roughly 0.4-0.6 µs) | 4.8 TB/s peak |

Two consequences drive the whole design.

1. **Little's law.** To stream at 4.8 TB/s with about 0.5 µs of latency, the chip needs about
   4.8e12 x 0.5e-6 = 2.4 MB of loads in flight, which is about 18 KB per SM. A kernel that issues a
   load, waits for it, and then issues the next one runs at a small fraction of peak. That is why the
   K/V tiles go through a multi-stage `cp.async` pipeline.
2. **Dependent round trips cost real time.** Each layer of indirection costs one full HBM latency.
   This kernel has three by nature: `seq_lens[b]` (where the window starts), then
   `page_table[b, pos]` (which physical row), then the K/V row itself. For a 2,048-token sliding
   window that moves only 0.5 MB, these round trips, not bandwidth, set the time. Removing one is worth
   about 0.5 µs, which is large next to a 0.1 µs bandwidth floor.

### 1.3 Gather granularity: why "4.8 TB/s" is not reachable on this cache

The scored cases use a **page size of 1**, and `generate_input` scatters physical pages with a
random permutation. So every token is its own random cache row:

- K row for one token and all KV heads on the GPU: `Hkv x 64 B = 128 B` (2 KV heads under TP8).
- V row: another 128 B, in a different tensor, so a different random address.

A 128 B access is 4 L2 sectors of 32 B, which is exactly one L2 line, but in DRAM terms it is a
random burst into a different DRAM page almost every time. HBM achieves its peak on long sequential
bursts that reuse an open row. Random 128 B gathers pay row activate and precharge costs, so the
practical ceiling is well below 4.8 TB/s. FA3 in the baseline reaches only about 1.6 TB/s on the
batch-32 full layer (384 µs for 0.62 GB). The problem's "floor" column assumes 4.8 TB/s, so it is a
loose lower bound for these shapes. Two layout facts help:

- Read both KV heads of a row in one CTA, so each cache row is one full 128 B line and not two
  64 B half-lines fetched by different CTAs at different times.
- Keep many independent row requests in flight (deep pipeline, enough resident CTAs).

### 1.4 The SM's compute engines

| Unit | Rate per SM per clock (dense) | Use here |
|---|---|---|
| Tensor cores, `wgmma` (Hopper warpgroup MMA, async, operands from shared memory) | about 1024 BF16 FMAs, 2048 FP8 FMAs | 989 TFLOP/s BF16 / 1979 TFLOP/s FP8 chip-wide. |
| Tensor cores, legacy `mma.sync` (warp-level, register operands) | roughly half of wgmma throughput in practice | Simple to use for small M (8-64 query rows), which is the decode case. |
| FP32 FMA | 128 lanes | Softmax scaling, masking, rescale of the accumulator. |
| MUFU (ex2, rsqrt, sin/cos) | 16 per clock | `exp2` of every score. `ex2.approx.f16x2` does two per instruction. |

Decode attention is normally memory-bound: each K/V byte is used by only 8 query heads x L tokens.
The arithmetic intensity is about `4 x 8L x 64 / 128` FLOP per KV byte, so 16 FLOP/B at L=1 and
128 FLOP/B at L=8 (per KV head pair). The machine balance is about 989e12 / 4.8e12 = 206 FLOP/B for
BF16 wgmma, but much lower for `mma.sync` plus the exp/convert work around it. So **verify (L=8) on the
full layer is close to compute-bound** for a simple `mma.sync` kernel, while decode (L=1) is
memory-bound. The softmax `exp2` is a real cost at L=8: B=32 x 80K keys x 2 KV heads x 64 rows is
about 330M exponentials, which is 75 µs at 16/clk/SM. Packed `f16x2` exponentials halve that.

### 1.5 Hopper features and how they apply

- **cp.async (Ampere-style async copy).** A 16 B global-to-shared copy that does not block the
  thread. It is ideal for gathers, because each 16 B chunk can come from an arbitrary address (here,
  8 lanes cover one 128 B cache row). Grouped with `commit_group` / `wait_group N` it forms an
  N-stage software pipeline.
- **TMA (Tensor Memory Accelerator).** Hopper's bulk-copy engine. It moves whole boxes of a
  tensor described by a tensor map, with one thread issuing, and it signals an mbarrier. It is the
  right tool for contiguous tiles (page size 64). On Hopper it has **no gather mode** (row gather,
  "gather4", is a Blackwell feature), so for page size 1 each row would need its own TMA
  instruction. That is why `cp.async` is used for this cache layout.
- **wgmma.** Warpgroup (4 warps) MMA reading A and B from shared memory with M = 64 per warpgroup.
  With 8 query heads per KV head, decode has only 8 useful rows (L=1) or 64 rows (L=8, an exact fit).
  At L=1, 56 of 64 rows would be padding. The usual trick is the transposed formulation
  (S^T = K Q^T, so M = keys).
- **FP8 tensor cores.** E4M3 x E4M3 with FP32 accumulate, at twice the BF16 rate. Because the KV
  cache is already E4M3, `QK^T` can consume raw cache bytes with no conversion if the query is also
  rounded to E4M3, which FA3 does and the rules allow. `PV` in FP8 would need P in FP8, which loses
  precision on small probabilities. The kernel here keeps P in FP16.
- **Thread-block clusters and distributed shared memory (DSMEM).** Up to 8 CTAs (16 with an
  opt-in) are co-scheduled on one GPC and can read each other's shared memory. For split-KV this
  allows the partial softmax states of up to 16 splits to be merged on-chip, without a round trip
  through global memory and a second kernel.
- **Programmatic dependent launch (PDL).** Lets kernel N+1 start its prologue while kernel N is
  finishing. That matters when a pipeline has to be several kernels. A single fused kernel avoids
  the issue.

### 1.6 CUDA graphs and launch overhead

The judge captures `kernel_fn` into a CUDA graph and replays it, subtracting about 4 µs of fixed
replay cost. What remains per kernel node is about 1-2 µs of launch, drain and ramp gap between
dependent kernels, plus the kernel's own critical path. SGLang's chain is 5+ kernels: QK-norm,
cache store, q cast, FA3 (plus FA3's split combine on full layers), norm-gate. On the sliding
layer, which moves about 0.5 MB, that chain costs about 20 µs while the data movement is about
0.1 µs. **Most of the sliding-layer time is launch gaps and dependent-latency chains, not
bandwidth.** Fusing everything into one launch is the largest single win, and the sliding shapes carry
weight 46 versus 16 for the full layer in every load.

---

## Part 2. Muse Spark Open's attention block

### 2.1 What the block computes

Per token, the fused QKV projection emits one BF16 vector of `(2 Hq + 2 Hkv) x 64` values. Under
TP8 that is 16 query heads, 2 key heads, 2 value heads and 16 gate heads (2304 values). Then:

```
q, k, v, gate = split(qkv)
q = RMSNorm(q) * 5.473        # per head, no learned weight ("QK-norm")
k = RMSNorm(k)
q, k = RoPE(q, pos), RoPE(k, pos)        # sliding-window layers only (theta = 500000, interleaved pairs)
cache[pos] = fp8_e4m3(k), fp8_e4m3(v)    # paged KV cache, per-tensor scale (1 in scored cases)
a = softmax(q . K^T / 8) . V             # over the window (sliding) or the whole context (full)
out = RMSNorm(a) * sigmoid(gate)         # per head, then the output projection (not part of this problem)
```

### 2.2 The design choices and why they exist

- **QK-norm (RMSNorm on q and k, then a fixed 5.473 gain on q).** It bounds every logit:
  `|q.k|/8 <= 5.473 x 64 / 8 ~ 43.8` in the worst case, and much less in practice. This removes
  attention-logit growth, which is what makes long-context training stable. It also makes the cache
  FP8-friendly: normalized keys are unit-RMS, so E4M3 (max 448, 3 mantissa bits) represents them
  with no scale. That is why SGLang serves with `kv_scale = 1`.
- **Gated attention output (`RMSNorm(a) x sigmoid(gate)`).** A per-head, per-channel learned gate
  after attention. It lets a head switch itself off cheaply (avoiding "attention sinks", where heads
  dump probability on token 0 to do nothing) and adds a non-linearity before the output projection.
  The RMSNorm on `a` removes the dependence of the output scale on how diffuse the attention was. For
  the kernel this is a **free epilogue**: the attention output is still in registers, and the gate
  is one more 128 B read per head.
- **Interleaved sliding-window and full layers.** Most layers use a 2,048-token window with RoPE.
  Their KV read is bounded (0.5 MB per request under TP8), whatever the context length. A minority of
  layers attend to the entire context with no RoPE (NoPE), which gives position-agnostic long-range
  retrieval. The weights in the problem table (46 sliding versus 16 full per step) are the layer counts
  of one decode step. The sliding layers are numerous and latency-bound. The full layers are few and
  bandwidth-bound, and they hold almost all of the bytes at 80K context.
- **GQA 16:2 under TP8 (128 query heads, 16 KV heads in the whole model).** Each KV head serves 8
  query heads. For decode this is the main data-reuse opportunity: one K/V tile read from HBM is used
  by 8 query heads (x L tokens in verify). A kernel that packs all 8 heads (and all L tokens) of a
  KV head into one tensor-core tile reads the cache once instead of 8 times. The "dp" test
  configuration (128 q / 16 kv heads on one GPU) keeps the same 8:1 ratio.
- **Paged FP8 cache, page size 1.** SGLang's radix-tree prefix cache allocates token by token
  (page size 1), so any token's KV can live anywhere. That gives perfect prefix sharing and no
  fragmentation, but it turns the attention read into a pure gather of 128 B rows (Part 1.3).
- **Speculative decoding with an 8-token verify block.** The draft model proposes 7 tokens, and the
  target model scores 8 positions in one step, with causal masking inside the block. For attention
  this multiplies query rows by 8 at the same KV read. Arithmetic intensity rises 8x, which is why
  verify (L=8) moves from memory-bound toward compute-bound, and why the 8 x 8 = 64 query rows per KV
  head exactly fill an M=64 tensor-core tile.

### 2.3 Why these benchmark shapes

| Load | What it represents | What limits it |
|---|---|---|
| decode 1 / verify 1 | Single-user latency (interactive chat) | Sliding layer: launch and dependent-latency chain. Full layer: 20 MB of gathers spread over only 2 KV heads, so it needs split-KV across about 128-256 CTAs plus a merge. |
| decode 16 / verify 16 | Moderate serving batch | Full layer: 330 MB of random 128 B gathers. |
| decode 32 / verify 32 | Throughput serving | Full layer: 660 MB of gathers. Verify adds 8x the math (tensor-core and exp throughput). |
| ctx 80K, agentic spread (plus or minus 7.5%) | Long agent transcripts | Load balance across requests of unequal length. |

The score is a geometric mean over loads, and each load weighs the sliding layer 46 to 16 against
the full layer. So the latency of the tiny sliding-layer kernel matters about as much as the
bandwidth of the huge full-layer kernel.

### 2.4 The data each step must touch (TP8, per request)

| Item | Bytes |
|---|---|
| qkv input (L tokens x 2304 BF16) | 4.6 KB x L |
| sliding layer K+V read (2,048 x 2 heads x 64 x 2) | 512 KB |
| full layer K+V read at 80K | 20.5 MB |
| new cache rows written | 256 B x L |
| page-table entries read (int32 per token) | 8 KB (sliding) / 320 KB (full), about 1.6% of the K/V bytes |
| output (L x 1024 BF16) | 2 KB x L |
