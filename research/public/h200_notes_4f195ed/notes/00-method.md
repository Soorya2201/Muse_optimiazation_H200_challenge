# 00. Hardware, workloads and measurement method

This note describes what every other note assumes. Where a later note departs from it, that note says so.

## Hardware

- One NVIDIA H200 SXM per run: sm_90a, 132 SMs, 141 GB HBM3e (4.8 TB/s nominal), 50 MB L2.
- Runs were scheduled on a pool of such machines. Two runs may not have used the same physical GPU, so I prefer comparisons made inside one run, and I flag every cross-run comparison.
- I did not lock or record the SM clock. Where a cycle budget is needed, I **assume 1.75 GHz** under load and say so.
- Toolchain: CUDA 13 compiler, PyTorch host code. Kernels are CUDA C++ with inline PTX, built for `sm_90a`.
- Measurements were taken between 2026-10-23 and 2026-10-25.

## Workloads

All are decode-time (one to a few hundred tokens per step) pieces of one transformer layer, at the shapes a single GPU sees under 8-way parallelism.

| Name in these notes | What it is |
|---|---|
| Dense projections | BF16 `x @ W^T` with `m` = 1–256 tokens. N×K: qkv+gate 2304×8192 (37.7 MB), o-proj 8192×1024 (16.8 MB), in-proj 4096×8192 (67.1 MB), out-proj 8192×4096 (67.1 MB; at ≤ 8 tokens only a 1024×4096 slice, 8.4 MB), router 256×8192 in FP32 (8.4 MB). |
| MoE layer | 256 experts, top-8 routing, hidden width 4096, expert intermediate width 4096, SwiGLU, an RMSNorm on each expert output, 32 experts local to the GPU. Weights are NVFP4: E2M1 codes, one E4M3 scale per 16 weights, one FP32 scale per tensor; 0.5625 bytes per weight, 28.31 MB per expert. Activations stay BF16. Batches of 1, 8, 16, 32, 128 and 256 tokens with skewed expert popularity. |
| Attention | Paged attention over an FP8 (E4M3) KV cache, page size 1. GQA with 16 query heads, 2 KV heads, head dim 64. Sliding-window layers (window 2048) and full layers, contexts up to about 80K tokens. Decode (1 query token) and 8-token verify. |
| Decoder layer | All of the above plus the glue between them (norms, residuals, RoPE, KV-cache writes), captured as one CUDA graph per call. At batch 1 a call is about 13 kernels (13 in the profile I checked). |

## Timers

Three timers appear in these notes. They measure different things and are never mixed in one table without a label.

**1. Harness time (independent).** A timing harness that I did not write captures each workload shape as a CUDA graph and replays it:
- before every timed replay it flushes L2 by *reading* a buffer larger than L2;
- it alternates replays of the kernel under test with replays of a reference implementation on the same fresh inputs, 20–100 repetitions per shape;
- it reports the reference's median time divided by the median of the paired time ratios (the order inside each pair alternates);
- it subtracts a graph-replay fixed cost of about 4 µs from both sides. It calibrates that cost on every shape from replays of a one-node and a two-node graph (note 04 uses these calibrations).

I treat harness time as the ground truth for "how long does this kernel take on a cold L2". Repeating the same code in separate runs moved the 80–110 µs MoE shapes (8 and 16 tokens) by about ±1.5–3 µs.

**2. In-process tuner time (mine).** Inside the process, before timing, I choose among kernel variants with my own timer:
- two eager warm-up calls, then one call captured per CUDA graph;
- before each replay an L2 flush over a 64 MiB (dense projections) or 96 MiB (MoE) buffer;
- CUDA events around each replay; the median of 15–24 replays, averaged over 3 routing samples for the MoE.

Tuner times are *raw*: they include the graph-launch fixed cost that the harness subtracts. Note 01 is about how the flush step biases this timer.

**3. In-kernel timelines (mine).** Selected kernels write `%globaltimer` stamps per CTA (start, first stage ready, loop end, stores done, and similar) into a debug buffer during one or a few cold-L2 replays.
- These are single samples on fixed seeds. Per-CTA stage traces varied by about ±15% between runs.
- Timeline values in some runs fall on multiples of 32 ns and in others on multiples of about 256 ns. That spacing is the effective resolution, so I do not interpret sub-microsecond differences from the coarser runs.

## Conventions

- µs and ns are wall time. MB = 10^6 bytes, MiB = 2^20 bytes, GB/s and TB/s use 10^9 and 10^12.
- "Measured" means a number printed by one of the three timers. "Derived" means arithmetic on measured numbers; I show the arithmetic. "Estimate" or "inference" marks anything else.
- "Stage" is one unit of a shared-memory ring: one bulk copy (or one group of copies) that a producer issues and consumers retire.
- A "no-op consumer" variant runs the full producer, ring, barriers and scheduling of a kernel but skips the math. It measures what the data movement alone costs.
