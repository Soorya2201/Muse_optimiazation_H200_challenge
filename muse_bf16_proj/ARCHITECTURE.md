# ARCHITECTURE: H200 and Muse Spark Open, as they matter to `muse_bf16_proj`

These are study notes. They cover the hardware facts and model facts that decide how fast a
skinny BF16 GEMM (`out[m, n] = x[m, k] @ W[n, k]^T` with m = 1 to 256) can run on one H200.
Numbers marked *measured* come from my own runs on the judge's H200 (job ids are in
IMPROVISATIONS.md). Everything else is a vendor figure or a widely cited microbenchmark value,
so treat it as approximate.

---

## 1. Hopper (H200 SXM) in one page

| Resource | Value | Why it matters here |
|---|---|---|
| SMs | 132 (8 GPCs, ~16-18 SMs each) | Work must be cut into >= 132 balanced pieces, ideally a small multiple. |
| HBM3e | 141 GB, **4.8 TB/s** peak | Every decode shape here is bound by weight bytes. |
| Practical read bandwidth (*measured*, cold L2, simple streaming kernel) | 8 MB in 2.6 us (3.2 TB/s), 16 MB in 4.4 us (3.8 TB/s), 37.75 MB in 9.6 us (4.1 TB/s), 64 MB in 15.5 us (4.3 TB/s) | This is the realistic floor. Small transfers never reach peak because a fixed ~1 us of ramp-up and tail is paid every time. |
| L2 | 50 MB (two partitions joined by a crossbar), roughly 2.5x HBM bandwidth | The judge flushes L2 before every replay, so weights always come from HBM. x and partial sums are small and stay L2-resident after first touch. |
| Shared memory / L1 | 256 KB per SM combined; up to **227 KB** dynamic smem per CTA | Holds a deep ring of weight tiles in flight (TMA destination). |
| Register file | 64 K x 32-bit per SM (256 KB) | Bounds how many loads a register-direct kernel can keep in flight. |
| Max threads / warps per SM | 2048 / 64 | |
| Tensor cores, BF16 dense | ~989 TFLOP/s through `wgmma`; legacy `mma.sync` reaches roughly 60% of that | For m <= 32 compute is irrelevant. At m = 256, pooler/unpooler become compute-heavy (17 GFLOP each). |
| SM clock | ~1.98 GHz (*measured*, `nvidia-smi` in the judge) | |

### 1.1 Memory latency and Little's law

DRAM latency on an idle H100/H200 is about 500-700 ns and rises past 1 us under full load. To
keep 4.8 TB/s flowing with ~1 us of latency you need about **4.8 MB in flight across the chip,
or ~36 KB per SM**, and twice that to be safe. That one number explains most of what follows:

* **Register-direct loads** (`ld.global.nc.v4` straight into MMA fragments) get their in-flight
  bytes from many warps times a few 16-byte loads each. Each warp waits for its batch, computes,
  then issues the next batch, so the in-flight volume is bursty and limited by registers.
  My v0 kernel reached ~3 TB/s with this approach.
* **TMA bulk copies** (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx`) let a
  single thread start a multi-KB copy into shared memory. A ring of 48-200 KB per SM, guarded
  by mbarriers, keeps the memory system saturated while warps compute on earlier stages.

### 1.2 TMA, mbarriers, async proxy

* `cp.async.bulk` (1-D) needs no tensor map: a contiguous byte range of global memory goes to
  shared memory, 16-byte aligned, with sizes a multiple of 16. Completion is signalled on an
  mbarrier via its transaction count (`expect_tx` bytes).
* `cp.async.bulk.tensor` (2-D to 5-D, with a `CUtensorMap`) can gather strided tiles and
  swizzle them. It is useful when the source is not contiguous; x's per-token rows are an
  example.
* Producer/consumer pattern: `full[s]` (count 1 + tx bytes) and `empty[s]` (count = consumer
  warps). The consumer waits `full[s]` with phase parity, reads, then arrives on `empty[s]`.
  The producer waits `empty[s]` before reusing the slot.
* Issue cost is real. Every bulk copy is one instruction, and the TMA unit handles requests
  serially. 33 small copies per stage from one thread was a bottleneck in my kernel (section 3
  of IMPROVISATIONS.md).

### 1.3 Thread-block clusters and distributed shared memory (DSMEM)

* A cluster of up to 8 CTAs (16 with `cudaFuncAttributeNonPortableClusterSizeAllowed`) is
  co-scheduled on one GPC. Any CTA can read or write a peer's shared memory (`mapa` +
  `ld.shared::cluster`). `barrier.cluster.arrive.release` / `wait.acquire` synchronize the
  cluster in a few hundred ns.
* For split-K, this replaces the classic global-memory fix-up (fp32 atomics into a workspace,
  `__threadfence`, an atomic counter, a last-arriver re-read). That chain costs 2-4 us of
  serial latency, which is as long as the whole 8 MB GEMV.

### 1.4 WGMMA vs mma.sync

* `wgmma.mma_async` (warpgroup = 4 warps) takes A from registers or smem and B from smem, with
  M = 64 per warpgroup and N up to 256. It is the only way to approach 989 TFLOP/s and is what
  cuBLAS (`nvjet_sm90_*` kernels) uses for m >= 64.
* `mma.sync.m16n8k16` is register-only and simple. With swap-AB (weights as the M = 16
  operand, tokens as N = 8), it wastes nothing on padding at m = 1 to 8, because we are
  bandwidth-bound and the MMA costs only a few cycles.

### 1.5 CUDA graphs and launch overhead

* The judge captures each shape as a CUDA graph and subtracts a ~4 us fixed replay cost.
  (*Measured*: in my own event timing an empty graph costs 4.6-5.3 us event to event.)
* Within a kernel, what remains is CTA launch (~0.5 us), the first-load latency including TLB
  misses on a cold working set (~1 us), and the tail: the last bytes, the reduction and the
  output stores (~0.5-1 us). For the 8 MB shapes (unpooler slice, router) this fixed part is as
  big as the transfer itself. That is why the best kernels on the board are judged on fixed
  costs as much as on bandwidth.
* Each extra kernel in a graph (memset, split-K reduce, a second GEMM) adds ~1-2 us. SGLang's
  router path above 8 tokens launches **three** kernels (two GEMMs plus an add), which is why
  it is the easiest shape to win.

---

## 2. Muse Spark Open, the parts this kernel serves

Deployment assumed by the judge: **8x H200, tensor parallel 8 (TP8) for attention and dense
projections, expert parallel 8 (EP8) for the routed experts**. Every number below is the
per-GPU slice (from `reference.py`). Interpretations not stated in the spec are marked
*(inferred)*.

| Layer | Per-GPU W [n, k] | Bytes | Role |
|---|---|---|---|
| `qkv_gate` | 2304 x 8192 BF16 | 37.7 MB | Fused projection: q (16 heads x 64) + k (2 x 64) + v (2 x 64) + output gate (16 x 64). Column-parallel: each GPU owns 1/8 of the heads, so hidden 8192 -> 2304. |
| `o` | 8192 x 1024 BF16 | 16.8 MB | Attention output, row-parallel: 16 heads x 64 = 1024 inputs -> hidden 8192, followed by an all-reduce (not part of this problem). |
| `pooler` | 4096 x 8192 BF16 | 67.1 MB | hidden 8192 -> MoE width 4096, **replicated** on every rank. The experts then work in a narrower 4096-wide space *(inferred: a down-projection that makes the 256 routed experts cheaper)*. |
| `unpooler` | 8192 x 4096 BF16 | 67.1 MB | MoE width 4096 -> hidden 8192, replicated. **Up to 8 tokens** SGLang computes only this GPU's 1/8 slice (1024 rows, 8.4 MB) and all-gathers. Above 8 tokens it runs the whole weight, because the all-gather would cost more than the redundant math. |
| `router` | 256 x 8192 **FP32** | 8.4 MB | Logits for 256 experts. FP32 because top-k expert selection is sensitive to small logit errors. SGLang runs FP32 `F.linear` up to 8 tokens, and above that two BF16 GEMMs on a hi/lo split with FP32 accumulation (`forward_split2`). |

### 2.1 Where these sit in a decode step (per layer)

```
x (hidden 8192, BF16)
 -> RMSNorm
 -> qkv_gate GEMM          [this problem]     -> RoPE, KV-cache write (FP8), gated attention decode
 -> o GEMM                  [this problem]     -> all-reduce (TP8) -> residual add
 -> RMSNorm
 -> router GEMM (FP32)      [this problem]     -> top-k over 256 experts
 -> pooler GEMM             [this problem]     -> routed experts (NVFP4, EP8: muse_moe_w4a16)
 -> unpooler GEMM           [this problem]     -> (all-gather if sliced) -> residual add
```

Every one of these runs 62 times per decode step (the "weight" column in TASK.md), so a step
streams about 62 x (37.7 + 16.8 + 67.1 + 8.4/67.1 + 8.4) MB, roughly **8.1-12.3 GB per step
per GPU**, from these projections alone. At 4.8 TB/s that is about 1.7-2.6 ms of pure weight
streaming per step. This is why the projections matter even though each GEMM is tiny.

### 2.2 Why these shapes

* **m = 1, 16, 32 (decode, spec-decode off):** one token per running request. Batch 1, 16 or 32
  requests per GPU.
* **m = 8, 128, 256 (verify):** SGLang's default speculative recipe checks an 8-token draft per
  request in one step. Batch 1, 16 or 32 requests gives 8, 128 or 256 rows.
* At m <= 32 every shape is **memory-bound by 30-200x** (arithmetic intensity is ~m FLOP/byte
  against ~206 FLOP/byte at the ridge). The kernel is a streaming problem.
* At m = 128 the floor is still memory (14.6 us for pooler versus 8.7 us of math). At m = 256
  pooler/unpooler hit the compute roof (17.4 us at 989 TFLOP/s), so a wgmma-class kernel is
  needed there.
* The unpooler's slice rule (n = 1024 at m <= 8) and the router's FP32/split rule (switch at
  8 tokens) come straight from SGLang. The baseline therefore uses different kernels across
  that boundary, and the m = 8 vs m = 16 router times differ by design.

### 2.3 Precision rules that shape the kernel

* Weights stay BF16 (repacking is allowed and lossless). Output BF16, FP32 accumulation.
* Router: FP32 weights may run as BF16 hi + lo with FP32 accumulation at **any** m. With
  hi + lo concatenated along K, `[x | x] @ [hi | lo]^T` is one K = 16384 GEMM. Probe rows (a
  single power-of-two entry in x) then see a relative error of ~2^-17, inside the 1e-3 probe
  bound and the 1e-4 router bound. *Measured* rel-L2: 2.5e-6.
