# Prefill-layer architecture

## Current verified public-source integration (Oct29)

`submission.py` is exact Apache-2.0 vjeux f35a41e,archived as
`submission_v17_vjeux_f35a41e.py`;checkce90e6fc5b4f49ed passed5tests/8timings at1.785870x.
It supersedes the historical v5 expert strategy below. Expert weights stay NVFP4 in
persistent state;the persistent RS-WGMMA kernel decodes fragments in registers and
computes both expert projections inside the timed call. Gather+plan fusion prepares
expert-contiguous BF16 activations and heterogeneous remainder tiles. Custom staged
paged attention covers every visible key,uses FP8 queries/cache,and fuses norm/gate.
Residual reconstruction at large T avoids storing/reloading the first FP32 stream by
rebuilding it from the current input/branch and saved RMS factor during the timed call.
cuBLASLt serves dense GEMMs;router high+low halves produce FP32 logits. The additional
lee1018-op repack/converter candidate passes correctness but has no established performance
gain over this parent. Attribution,exact hashes,tests,and caveats: PUBLIC_RESEARCH.md.

## Hardware facts that shape the implementation

The target is one NVIDIA H200 (Hopper): 132 SMs, roughly 50 MB L2, nominal 4.8 TB/s
HBM3e, and about 228 KB usable shared memory per SM (the compiler reports a 232,448-byte
kernel limit here). Measured kernels usually sustain about 3.2--4.3 TB/s. Hopper supplies
TMA, WGMMA, FP8 tensor cores, thread-block clusters and distributed shared memory, but it
has no native FP4 arithmetic. Muse's NVFP4 weights therefore require E2M1 unpacking and
E4M3 block-scale application before BF16/FP8 tensor-core math. CUDA-graph replay makes
host launch overhead small at these prefill sizes; throughput and memory movement dominate.

## Workload and sharding

This is one Muse Spark Open decoder layer on one GPU of TP8/EP8 serving. Attention owns
16 query heads and two KV heads (head dimension 64). The model has 256 experts globally;
this GPU stores 32 and computes the routes in its `local_start..local_start+31` range.
Cross-GPU all-reduce/all-gather operations are identity in the challenge.

Scored token counts are 753 (batch 1), 3,012 (batch 4), and 16,384. Each request has
either sliding attention over 2,048 keys or full attention over as many as 81,920 keys.
The KV cache is FP8 E4M3 and scored pages contain one token.

## Layer data flow

For `T = batch * q_len`:

1. BF16 `attn_input[T,8192]` is multiplied by `qkv_gate[2304,8192]`.
2. Q/K receive per-head RMSNorm; sliding layers also apply interleaved RoPE.
3. new K/V rows are converted to FP8 and stored in the paged cache.
4. FA3 computes causal paged attention, using all required keys.
5. per-head output RMSNorm and `sigmoid(gate)` are fused, followed by the BF16 output
   projection to 8,192 channels.
6. a fused residual/post-norm/pre-norm boundary produces the FFN input.
7. the FP32 router (implemented as BF16 high+low GEMMs with FP32 accumulation here) runs
   beside the BF16 pooler `8192 -> 4096`; biased sigmoid routing selects top 8.
8. local NVFP4 experts compute W13, fused SwiGLU, W2, per-route post-expert RMSNorm, and
   weighted slot reduction.
9. the BF16 unpooler maps `4096 -> 8192`, followed by the second fused residual boundary.

## v5 expert strategy

The implementation chooses an expert path by token bucket during untimed `prepare()`:

- T=753: SGLang Marlin wins and remains selected.
- T=3,012: decode the touched experts to a timed BF16 scratch and run custom grouped
  GEMMs (`G`), around 2.19 ms versus Marlin's 2.47 ms in prepare-time measurements.
- T=16,384: the same `G` path is around 7.37 ms versus Marlin's 13.31 ms.

The scratch is filled on every timed call; it is not a cross-input cache. NVFP4 remains
the persistent weight representation. W13/SwiGLU and W2 are separate Triton kernels;
the latter is followed by SGLang's correct post-expert normalization and weighted sum.

