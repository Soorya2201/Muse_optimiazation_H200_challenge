# MoE serving port checkpoint — 2026-10-30

This is an **isolated offline prototype**, not a working serving optimization.
No serving check or submit was run. The kernel boards retain their checked sources.

`okc_init/patch.diff` contains two new files under `python/sglang/`: a CUDA
header and a Python `load_jit` wrapper. Neither changes a model call site.
Setting `SGLANG_OKC_MOE=1` only permits the explicit developer entry points;
it does **not** switch the server away from Marlin.

## What was implemented and checked

- Pinned donor: alygg77 `6cae49a`, exact checked MoE v24, Apache-2.0 with
  vjeux attribution. Hashes are in `MOE_PORT_MANIFEST.json`; upstream license
  and NOTICE are retained in `../research/public/alygg77_6cae49a/`.
- Replaced the PyTorch C++ binding with 21 explicit TVM-FFI TensorView
  arguments plus three integers. The stream comes from `TVMFFIEnvGetStream`
  for the activation's CUDA device. PDL stays disabled, as in the donor.
- Preserved all 17 packing/setup helper bodies and the variant/chunk dispatch
  by AST comparison. Device code differs only in nine inline-assembly
  operand-list line wraps to satisfy the 400-character serving rule.
- Developer API accepts **32 local raw experts**, a separate global expert
  offset, BF16 activations, eight global IDs and FP32 routing scores. It
  returns the local weighted sum; cross-rank communication is the caller's job.
- Python syntax, exact source hashes, helper/dispatch ASTs, device-source
  whitespace equivalence, flag rejection, patch paths/size/line limits and
  reverse-apply checks pass. The patch is **165,452 bytes**, below 720 KiB.

Reproduce after `okc serve init` at SGLang `a55ab7d5`:

```bash
python3 tools/build_serving_moe_prototype.py
okc serve diff serving/okc_init
python3 tools/verify_serving_moe_prototype.py
```

The generator prints its manifest and refuses to overwrite changed generated
files. The committed manifest identifies this checkpoint. No CUDA compiler or
H200 is available locally, so none of these checks proves GPU correctness.

## Why a direct model hook would be wrong

1. **Return contract.** `_marlin_expert_outputs` returns per-route expert rows
   **and local IDs**. Our kernel already applies the post-expert norm and
   routing weights and returns `[tokens, 4096]`. The candidate integration
   point is `_forward_marlin`, after pre-expert normalization. Preserve the
   surrounding model's TP/EP collectives and all unsupported-shape fallbacks.
2. **Norm representation.** The prototype retains the kernel-track input:
   original BF16 `post_norm_w`, with `1 + w` formed in FP32. SGLang's
   `set_post_expert_norm` caches a BF16 **effective weight** and supports
   weight-first/weight-after forms. Passing that cached tensor to this
   prototype would add one twice. Subtracting one and rounding to BF16 is
   also not a lossless adapter. Port explicit effective-weight semantics,
   validate epsilon/form and compare against stock before installing a hook.
3. **Weight representation/scales.** `process_weights_after_loading` first
   deinterleaves w13 and then repacks it for Marlin; it can collapse distinct
   gate/up global scales. The prototype expects the challenge's raw NVFP4
   tensors, not Marlin tensors. A real port needs a quantization post-load
   hook before repacking, correct scale interpretation and shard mapping.
   Do not change `srt/model_loader/`, requantize, or assume distinct gate/up
   scales can be silently collapsed without checking stock semantics.
4. **Resident memory.** For 32 local experts at H=I=4096, raw code bytes are
   805,306,368 and scale bytes are 100,663,296. One layout is **864 MiB**;
   four code/scale layouts consume **3.375 GiB per layer**, before scratch,
   stock fallback weights, other model weights and KV cache. Sixty such
   layers would consume **202.5 GiB** just for these copies. That cannot fit
   one H200. A single WGS layout is still 50.625 GiB across sixty layers,
   on top of fallback state. Start with an explicitly selected single-layer
   pilot, then prove a lower-memory full-model design and KV budget.

## Required GPU work before promotion

Compile every PRMT/LUT and small/large variant on sm90a. Compare numerics
against stock for all token counts 1–32, DFlash verification sizes through
256 and nearby chunk boundaries, all eight expert offsets, skewed/no-local
routes, graph capture/replay and fresh inputs. Check independent concurrent
state/stream use; scratch must never be shared by overlapping calls.

Only after those pass should a model hook be added and a full-node memory
and output-agreement check run with the fixed serving flags. Measure warm-L2
behavior, TP/EP communication, prefill TTFT and all four serving loads.
The official serving track opens Nov5; no serving throughput improvement has
been measured. This prototype does not establish rank1 or a serving score.
