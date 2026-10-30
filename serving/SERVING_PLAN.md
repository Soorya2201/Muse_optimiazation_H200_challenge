# Serving track plan (opens Nov 5 2026, noon PT)

All $100K of prize money is in this track. The kernel boards only decide who gets priority on
the node queue (top 50 overall). This file is the starting point. It was written on Oct 29 from
the official `okc serve init` checkout. Nothing here has run on a GPU yet.

Oct30 checkpoint: see [MOE_PORT_STATUS.md](MOE_PORT_STATUS.md). An isolated,
default-off `load_jit` prototype and source-only patch are prepared and checked
locally. It is not integrated or GPU-compiled. The direct `_marlin_expert_outputs`
replacement suggested below has a different return contract; use the checkpoint's
integration, norm, raw-weight and memory gates before adding model hooks.

## 1. What is already set up

- `serving/okc_init/` comes from `okc serve init okc_init`:
  - `sglang/`: the pinned SGLang at `a55ab7d5`, on branch `my-patch`. It is git-ignored in our
    repo, so re-create it with the same command on another machine.
  - `RULES.md`: the official patch rules and how a run is scored. Read it first.
  - `patch.diff`: empty.
- Workflow: edit `serving/okc_init/sglang` → `okc serve diff okc_init` → `okc serve check okc_init/patch.diff`
  (3 a day, one short rung, unscored) → `okc serve submit okc_init/patch.diff` (1 a day, a whole
  8×H200 node for up to 90 minutes).

## 2. How a run is scored (from RULES.md)

- The workload is multi-turn agentic coding conversations at concurrency 1, 4, 16 and 32.
- Every request returns exactly 220 tokens.
- The score is throughput at stock SGLang's interactivity on the same node.
- Outputs must match stock's (about as well as stock matches itself).
- Time to first token must stay within 2× stock's at every load. A load over that limit can't
  raise the score.
- The judge's server flags are fixed. The patch can change what they do, not the flags:
  `--tp-size 8 --ep-size 8 --quantization modelopt_fp4 --moe-runner-backend marlin
  --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 --enable-hierarchical-cache
  --chunked-prefill-size 16384 --context-length 131072 --max-running-requests 32
  --cuda-graph-max-bs-decode 32 --random-seed 1234`.
- The kernel problems map onto these loads directly. The kernel loads "decode 1/16/32" and
  "verify 1/16/32" are the serving concurrencies. Verify is the DFlash 8-token target check,
  which is what serving actually runs every step.

## 3. Where each kernel plugs in (paths under `okc_init/sglang/python/sglang/`)

| Our kernel track | SGLang call site at a55ab7d5 | What it replaces |
|---|---|---|
| `muse_moe_w4a16` (persistent fused NVFP4 MoE) | `srt/layers/muse_spark_fp4_moe.py`: `MuseSpark1xFp4Experts._marlin_expert_outputs` (two `moe_wna16_marlin_gemm` calls, around lines 451-560); runner selection at lines 133-200 | Marlin W4A16 G1/G2 pair. RULES.md calls it the biggest share of a decode step from 16 requests up. **Port this first.** |
| `muse_attention_decode` (fused qk-norm, RoPE, FP8 KV write, attention, norm, gate) | `srt/models/muse_spark.py`: `MuseSpark1xAttention.gated_attention` (lines ~455-484); `srt/layers/attention/flashattention_backend.py`: `forward_decode` (~1958) and `forward_extend` (~1390, target-verify mode) | FA3 plus the separate `_fused_qk_norm` / `_fused_o_norm_gate` launches |
| `muse_bf16_proj` (lossless C12 GEMV, router) | `qkv_proj` / `o_proj` linears and `bf16_gemm_dispatch` (`srt/layers/quantization/unquant.py`) used by `pre_expert_proj` (the pooler) in `MuseSpark1xMoE._forward_experts` (~806); router: `MuseSpark1xRouter.forward` / `forward_split2` (~256-293) and `kernels/ops/muse_spark_v12/router_gemm.py` | cuBLAS GEMMs at m ≤ 32 |
| `muse_decode_layer` (glue fusion, PDL chain) | `MuseSpark1xDecoderLayer.forward` (~1056) and `kernels/ops/muse_spark_v12/fused_norm_residual.py` | separate norm, residual and routing launches |

Keep the precision rules exactly as stock has them:
- **Router:** at ≤ 8 tokens stock uses an FP32 activation with `F.linear`; above 8 it uses
  BF16 hi+lo (`_use_split_bf16_router`, `router_in_fp32`). See the decoder compliance item in
  `CODEX_NEXT_STEPS.md`.
- **FA3:** stock's FP8 query in FA3 stays as it is.
- **KV cache:** stays FP8.

## 4. How custom CUDA gets into a patch

- The prebuilt `sgl-kernel` binaries are not rebuilt, and the patch may contain only source
  files (.py/.cu/.cuh/.h/.cpp/.json).
- Use SGLang's own JIT path: `sglang.kernels.jit.utils.load_jit(name, *args,
  cuda_files=[...], cuda_wrappers=[...])`, with the `.cuh` placed under
  `python/sglang/kernels/jit/csrc/`. Copy the pattern of `kernels/ops/muse_spark_v12/router_gemm.py`.
  Our kernel files use `torch.utils.cpp_extension.load_inline`. Turn each one into a `.cuh`
  plus a thin Python wrapper.
- JIT compile time happens at server start and counts toward the 90-minute node budget, so
  keep the number of template instantiations small.
- Weight repacking (NVFP4 re-layout such as KPERM3, C12 compression) must not touch
  `srt/model_loader/`. Do it in the layer's post-load hook (the quant method's
  `process_weights_after_loading` or the layer's own init after weights land), the same way
  stock builds `_split2_hi/_lo` lazily.
- Every new path goes behind an environment switch that defaults to stock. Then one patch can
  A/B each kernel, and a broken path can be turned off without a new diff.

## 5. Differences from the kernel problems (don't assume transfer)

- The kernel judge flushes L2 before every replay. A real server runs warm, and successive
  layers share L2.
- The kernel problems time one GPU's slice. In serving, all-reduce and EP dispatch/combine sit
  between the kernels, and CUDA graphs cover the whole step.
- Batch sizes vary step to step (agentic spread, finished requests), so every kernel needs
  graph-safe shape handling for every bs ≤ 32 and verify lengths of 8 per request, not only
  the 6 scored shapes.
- TTFT matters: the prefill path (chunked 16384) can't regress past 2× stock, so the
  `muse_prefill_layer` work protects TTFT.

## 6. Ordered work list

1. **Before Nov 5, offline (no GPU):**
   - (a) Port the MoE kernel to a `load_jit` module plus a wrapper with the same signature as
     `_marlin_expert_outputs`, behind `SGLANG_OKC_MOE=1`.
   - (b) Do the same for attention (`SGLANG_OKC_ATTN=1`), covering decode and target-verify.
   - (c) Write a unit test per op that compares against the stock op at bs 1..32 and verify
     lengths of 8.
   - (d) `py_compile` everything and keep `okc serve diff` under 720 KB with no line over 400
     characters.
2. **Decide on hardware (owner):** test on a rented 8×H200 node (or at least one H200 for op
   tests) in `lmsysorg/sglang:dev-muse-spark-open`. Another public team rents a judge-faithful
   H200 for about $5-6/hour. With only 3 serving checks a day, op-level testing on a rented
   GPU is the only way to iterate safely.
3. **Day 1 (Nov 5):**
   - `okc serve check` an empty patch to learn the stock numbers.
   - Then check the MoE port alone.
4. **Next days:** attention port → dense projections → glue fusion. One mechanism per check, so
   any regression can be attributed.
5. **DFlash and drafting** (`srt/speculative/dflash_worker_v2.py`, `dflash_utils.py`):
   - Make the drafter forward faster (same weights).
   - Consider adding prompt-lookup / n-gram drafts from the conversation. Agentic coding
     repeats code heavily, so acceptance could rise.
   - The rules allow both. Output equality with stock must be re-checked, because acceptance
     changes alter batch composition.
6. **Scheduling / CUDA graphs** (`srt/managers/scheduler.py`, `model_executor/cuda_graph_config.py`):
   check that verify steps at every concurrency hit a captured graph.

## 7. Open questions for the owner

- Budget for a rented H200 / 8×H200 node before Nov 5?
- Kernel deadline vs serving prep: how much time goes to serving porting versus the remaining
  kernel-board checks?
