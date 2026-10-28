# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _base_prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _base_kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# =====================================================================================
# Custom kernels (Triton). Everything below replaces pieces of SGLang's layer above.
# Each custom piece is self-tested in prepare() against a torch reference and timed
# against SGLang's kernel (CUDA-graph replays, cold L2); the faster correct one is used
# per shape. If a test fails or a kernel does not compile, SGLang's kernel is used.
# =====================================================================================
import time
import traceback

import triton
import triton.language as tl

TWO63 = tl.constexpr(9.223372036854775808e18)
LOG2E = 1.4426950408889634


# ---------------------------------- MoE: NVFP4 W4A16 ----------------------------------
# E2M1 -> BF16 by bit placement: sign to bit 15, the 3 magnitude bits to bits 6..8. The
# BF16 value is then exactly e2m1 * 2^-126 (code 1 = +-0.5 is a BF16 subnormal). One BF16
# multiply by (E4M3 block scale * 2^63) gives e2m1 * scale * 2^-63 exactly (<= 6
# significant bits, normal range); the FP32 epilogue multiplies by global scale * 2^63.

@triton.jit
def _deq8(words, sc):
    """'join' layout: words int32 [BN, W] (natural NVFP4 packing), sc bf16 [BN, W]
    -> bf16 [BN, W, 2, 2, 2] that flattens to natural K order."""
    o0, o1, o2, o3, o4, o5, o6, o7 = tl.inline_asm_elementwise(
        asm="""
{
.reg .b32 t, a, c, s2, r;
mov.b32 s2, {$9, $9};
shl.b32 t, $8, 12;
and.b32 a, t, 0x80008000;
and.b32 c, t, 0x70007000;
shr.b32 c, c, 6;
or.b32 r, a, c;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$0, $4}, r;
shl.b32 t, $8, 8;
and.b32 a, t, 0x80008000;
and.b32 c, t, 0x70007000;
shr.b32 c, c, 6;
or.b32 r, a, c;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$1, $5}, r;
shl.b32 t, $8, 4;
and.b32 a, t, 0x80008000;
and.b32 c, t, 0x70007000;
shr.b32 c, c, 6;
or.b32 r, a, c;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$2, $6}, r;
and.b32 a, $8, 0x80008000;
and.b32 c, $8, 0x70007000;
shr.b32 c, c, 6;
or.b32 r, a, c;
mul.rn.bf16x2 r, r, s2;
mov.b32 {$3, $7}, r;
}
""",
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[words, sc.to(tl.int16, bitcast=True)],
        dtype=(tl.int16, tl.int16, tl.int16, tl.int16, tl.int16, tl.int16, tl.int16, tl.int16),
        is_pure=True,
        pack=1,
    )
    i0 = tl.join(o0, o4)
    i1 = tl.join(o1, o5)
    i2 = tl.join(o2, o6)
    i3 = tl.join(o3, o7)
    m0 = tl.join(i0, i2)
    m1 = tl.join(i1, i3)
    return tl.join(m0, m1).to(tl.bfloat16, bitcast=True)


@triton.jit
def _nib_bf16(nib):
    bits = ((nib & 7) << 6) | ((nib & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _moe_align_kernel(ids_ptr, ls_ptr, sorted_ptr, bexp_ptr,
                      R: tl.constexpr, R_P2: tl.constexpr, MAXP: tl.constexpr, MAXP_P2: tl.constexpr,
                      NBM: tl.constexpr, NBM_P2: tl.constexpr, BM: tl.constexpr):
    """One CTA: groups this GPU's routes by local expert, each expert padded to BM rows.
    sorted[p] = route index (token * 8 + slot) or R (padding); bexp[block] = expert or -1."""
    offs = tl.arange(0, R_P2)
    rm = offs < R
    ids = tl.load(ids_ptr + offs, mask=rm, other=0)
    ls = tl.load(ls_ptr)
    loc = (ids - ls + 256) % 256
    loc = tl.where(rm, loc, 999)
    p = tl.arange(0, MAXP_P2)
    tl.store(sorted_ptr + p, tl.full([MAXP_P2], R, tl.int32), mask=p < MAXP)
    bb = tl.arange(0, NBM_P2)
    tl.store(bexp_ptr + bb, tl.full([NBM_P2], -1, tl.int32), mask=bb < NBM)
    tl.debug_barrier()
    off = 0
    for e in tl.static_range(32):
        m = (loc == e).to(tl.int32)
        cnt = tl.sum(m, 0)
        rank = tl.cumsum(m, 0) - 1
        tl.store(sorted_ptr + off + rank, offs, mask=m > 0)
        nb = (cnt + BM - 1) // BM
        tl.store(bexp_ptr + off // BM + bb, tl.full([NBM_P2], e, tl.int32), mask=bb < nb)
        off += nb * BM


@triton.jit
def _moe_gemm_join_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, bexp_ptr, R,
                          N: tl.constexpr, K: tl.constexpr, WROWS: tl.constexpr, A_DIV: tl.constexpr,
                          GATED: tl.constexpr, NBM: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Grouped NVFP4 GEMM, swap-AB (weights are the MMA's M side), natural packing,
    PTX dequant of 8 codes per 32-bit word. GATED: rows [0,N) gate, [N,2N) up ->
    silu(g)*u in BF16; else FP32 out."""
    pid = tl.program_id(0)
    pid_m = pid % NBM
    pid_n = pid // NBM
    e = tl.load(bexp_ptr + pid_m)
    if e < 0:
        return
    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    rmask = rows < R
    arow = tl.where(rmask, rows // A_DIV, 0)
    WD: tl.constexpr = BK // 8
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_w = tl.arange(0, WD)
    offs_k = tl.arange(0, BK)
    e64 = e.to(tl.int64)
    w_base = w_ptr + e64 * (WROWS * (K // 8)) + offs_n[:, None] * (K // 8) + offs_w[None, :]
    s_base = sf_ptr + e64 * (WROWS * (K // 16)) + offs_n[:, None] * (K // 16) + (offs_w // 2)[None, :]
    a_base = a_ptr + arow[None, :].to(tl.int64) * K + offs_k[:, None]
    UPW: tl.constexpr = N * (K // 8)
    UPS: tl.constexpr = N * (K // 16)
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    acc2 = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        at = tl.load(a_base + k0, mask=rmask[None, :], other=0.0)
        words = tl.load(w_base + k0 // 8)
        sc = (tl.load(s_base + k0 // 16).to(tl.float32) * TWO63).to(tl.bfloat16)
        acc = tl.dot(tl.reshape(_deq8(words, sc), (BN, BK)), at, acc)
        if GATED:
            words2 = tl.load(w_base + UPW + k0 // 8)
            sc2 = (tl.load(s_base + UPS + k0 // 16).to(tl.float32) * TWO63).to(tl.bfloat16)
            acc2 = tl.dot(tl.reshape(_deq8(words2, sc2), (BN, BK)), at, acc2)
    gs = tl.load(gs_ptr + e) * TWO63
    out_off = rows[None, :].to(tl.int64) * N + offs_n[:, None]
    if GATED:
        g = acc * gs
        u = acc2 * gs
        h = g / (1.0 + tl.exp(-g)) * u
        tl.store(out_ptr + out_off, h.to(tl.bfloat16), mask=rmask[None, :])
    else:
        tl.store(out_ptr + out_off, acc * gs, mask=rmask[None, :])


@triton.jit
def _split_step(wp, sp, alo, ahi, acc, BN: tl.constexpr, USE_DOT: tl.constexpr):
    """One 128-wide K step in the 'split' layout: byte j of the step holds element j (low
    nibble) and element j + 64 (high nibble), so the two halves are two plain K=64 tiles."""
    byt = tl.load(wp).to(tl.int32)
    slo = (tl.load(sp).to(tl.float32) * TWO63).to(tl.bfloat16)
    shi = (tl.load(sp + 4).to(tl.float32) * TWO63).to(tl.bfloat16)
    wl = tl.reshape(tl.reshape(_nib_bf16(byt & 15), (BN, 4, 16)) * slo[:, :, None], (BN, 64))
    wh = tl.reshape(tl.reshape(_nib_bf16(byt >> 4), (BN, 4, 16)) * shi[:, :, None], (BN, 64))
    if USE_DOT:
        acc = tl.dot(wl, alo, acc)
        acc = tl.dot(wh, ahi, acc)
    else:
        acc += tl.sum(wl.to(tl.float32)[:, :, None] * alo.to(tl.float32)[None, :, :], axis=1)
        acc += tl.sum(wh.to(tl.float32)[:, :, None] * ahi.to(tl.float32)[None, :, :], axis=1)
    return acc


@triton.jit
def _moe_gemm_split_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, bexp_ptr, R,
                           N: tl.constexpr, K: tl.constexpr, WROWS: tl.constexpr, A_DIV: tl.constexpr,
                           GATED: tl.constexpr, NBM: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, USE_DOT: tl.constexpr):
    """Same GEMM on the 'split' repacked codes (lossless nibble permutation done in
    prepare): no join/reshape of K, two K=64 MMAs (or FMA reductions) per 128-wide step."""
    BK: tl.constexpr = 128
    HB: tl.constexpr = 64
    pid = tl.program_id(0)
    pid_m = pid % NBM
    pid_n = pid // NBM
    e = tl.load(bexp_ptr + pid_m)
    if e < 0:
        return
    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    rmask = rows < R
    arow = tl.where(rmask, rows // A_DIV, 0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_b = tl.arange(0, HB)
    e64 = e.to(tl.int64)
    w_base = w_ptr + e64 * (WROWS * (K // 2)) + offs_n[:, None] * (K // 2) + offs_b[None, :]
    s_base = sf_ptr + e64 * (WROWS * (K // 16)) + offs_n[:, None] * (K // 16) + tl.arange(0, 4)[None, :]
    a_base = a_ptr + arow[None, :].to(tl.int64) * K + offs_b[:, None]
    UPB: tl.constexpr = N * (K // 2)
    UPS: tl.constexpr = N * (K // 16)
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    acc2 = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        alo = tl.load(a_base + k0, mask=rmask[None, :], other=0.0)
        ahi = tl.load(a_base + k0 + HB, mask=rmask[None, :], other=0.0)
        acc = _split_step(w_base + k0 // 2, s_base + k0 // 16, alo, ahi, acc, BN, USE_DOT)
        if GATED:
            acc2 = _split_step(w_base + UPB + k0 // 2, s_base + UPS + k0 // 16, alo, ahi, acc2, BN, USE_DOT)
    gs = tl.load(gs_ptr + e) * TWO63
    out_off = rows[None, :].to(tl.int64) * N + offs_n[:, None]
    if GATED:
        g = acc * gs
        u = acc2 * gs
        h = g / (1.0 + tl.exp(-g)) * u
        tl.store(out_ptr + out_off, h.to(tl.bfloat16), mask=rmask[None, :])
    else:
        tl.store(out_ptr + out_off, acc * gs, mask=rmask[None, :])


@triton.jit
def _bw_probe_kernel(w_ptr, out_ptr, bexp_ptr, K: tl.constexpr, WROWS: tl.constexpr, NBM: tl.constexpr,
                     BN: tl.constexpr):
    """Bandwidth probe: streams the same gate+up bytes a GEMM1 CTA would, no math."""
    pid = tl.program_id(0)
    pid_m = pid % NBM
    pid_n = pid // NBM
    e = tl.load(bexp_ptr + pid_m)
    if e < 0:
        return
    offs_n = pid_n * BN + tl.arange(0, BN)
    base = w_ptr + e.to(tl.int64) * (WROWS * (K // 2)) + offs_n[:, None] * (K // 2) + tl.arange(0, 64)[None, :]
    acc = tl.zeros((BN, 64), dtype=tl.int32)
    for k0 in range(0, K, 128):
        acc += tl.load(base + k0 // 2).to(tl.int32)
        acc += tl.load(base + (WROWS // 2) * (K // 2) + k0 // 2).to(tl.int32)
    tl.store(out_ptr + pid * BN + tl.arange(0, BN), tl.sum(acc, 1))


@triton.jit
def _moe_finalize_kernel(y_ptr, ids_ptr, wt_ptr, ls_ptr, pw_ptr, out_ptr, D: tl.constexpr):
    """Per token: sum over its local slots of weight * rmsnorm(y * (1 + w_post))."""
    t = tl.program_id(0)
    d = tl.arange(0, D)
    pw = 1.0 + tl.load(pw_ptr + d).to(tl.float32)
    ls = tl.load(ls_ptr)
    acc = tl.zeros([D], tl.float32)
    for s in tl.static_range(8):
        e = tl.load(ids_ptr + t * 8 + s)
        loc = (e - ls + 256) % 256
        ok = loc < 32
        w = tl.load(wt_ptr + t * 8 + s).to(tl.float32)
        y = tl.load(y_ptr + (t * 8 + s).to(tl.int64) * D + d, mask=(d < D) & ok, other=0.0)
        v = y * pw
        r = tl.rsqrt(tl.sum(v * v, 0) / D + 1e-8)
        acc += tl.where(ok, w * r, 0.0) * v
    tl.store(out_ptr + t * D + d, acc.to(tl.bfloat16))


def _np2(x):
    return 1 << max(0, (int(x) - 1).bit_length())


def _repack_split(w):
    """Lossless nibble permutation: within each 128-element K block, byte j holds element j
    (low nibble) and element j + 64 (high nibble). Same bytes, same NVFP4 codes."""
    out = torch.empty_like(w)
    n, kb = w.shape[1], w.shape[2]
    for e in range(w.shape[0]):
        x = w[e]
        codes = torch.stack((x & 15, x >> 4), -1).reshape(n, kb * 2).view(n, kb * 2 // 128, 2, 64)
        out[e] = (codes[:, :, 0, :] | (codes[:, :, 1, :] << 4)).reshape(n, kb)
    return out


def _moe_align(topk_ids, local_start, tokens, BM):
    R = tokens * TOP_K
    NBM = -(-(R + LOCAL_EXPERTS * (BM - 1)) // BM)
    dev = topk_ids.device
    sorted_ids = torch.empty(NBM * BM, dtype=torch.int32, device=dev)
    bexp = torch.empty(NBM, dtype=torch.int32, device=dev)
    _moe_align_kernel[(1,)](topk_ids, local_start, sorted_ids, bexp, R=R, R_P2=_np2(R), MAXP=NBM * BM,
                            MAXP_P2=_np2(NBM * BM), NBM=NBM, NBM_P2=_np2(NBM), BM=BM, num_warps=4)
    return sorted_ids, bexp, NBM


def _moe_gemm(kind, a, which, out, sorted_ids, bexp, R, NBM, cfg, state):
    """kind: 'join' or 'split'. which: 13 (gated) or 2."""
    m = state["moe"]
    gated = which == 13
    BM, BN = cfg[1], cfg[2]
    grid = (NBM * (D_EXPERT // BN),)
    common = dict(N=D_EXPERT, K=D_EXPERT, WROWS=(2 if gated else 1) * D_EXPERT, A_DIV=TOP_K if gated else 1,
                  GATED=gated, NBM=NBM, BM=BM, BN=BN)
    if kind == "join":
        return _moe_gemm_join_kernel[grid](a, m[f"w{which}"], m[f"sf{which}"], m[f"gs{which}"], out, sorted_ids,
                                           bexp, R, BK=128, num_warps=cfg[3], num_stages=cfg[4], **common)
    return _moe_gemm_split_kernel[grid](a, m[f"w{which}s"], m[f"sf{which}"], m[f"gs{which}"], out, sorted_ids,
                                        bexp, R, USE_DOT=cfg[5], num_warps=cfg[3], num_stages=cfg[4], **common)


def _moe_forward(x, topk_weights, topk_ids, local_start, state, cfg):
    """x [T, 4096] bf16 (pre-expert-normed), topk ids/weights [T, 8] -> [T, 4096] bf16.
    cfg = (kind, BM, BN, warps, stages[, use_dot])."""
    tokens = x.shape[0]
    R = tokens * TOP_K
    dev = x.device
    ids = topk_ids.to(torch.int32).contiguous()
    sorted_ids, bexp, NBM = _moe_align(ids, local_start, tokens, cfg[1])
    inter = torch.empty((R, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _moe_gemm(cfg[0], x, 13, inter, sorted_ids, bexp, R, NBM, cfg, state)
    y = torch.empty((R, D_EXPERT), dtype=torch.float32, device=dev)
    _moe_gemm(cfg[0], inter, 2, y, sorted_ids, bexp, R, NBM, cfg, state)
    out = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _moe_finalize_kernel[(tokens,)](y, ids, topk_weights.float().contiguous(), local_start,
                                    state["moe"]["post_w"], out, D=D_EXPERT, num_warps=8)
    return out


MOE_CANDS = (
    ("join", 16, 64, 4, 3),
    ("split", 16, 64, 4, 3, True),
    ("split", 16, 128, 8, 3, True),
    ("split", 4, 64, 8, 2, False),
    ("split", 1, 64, 4, 3, False),
    ("split", 32, 64, 4, 3, True),
    ("split", 64, 128, 8, 3, True),
)


# ------------------------------------- attention --------------------------------------

@triton.jit
def _attn_split_kernel(q_ptr, k_ptr, v_ptr, pt_ptr, cs_ptr, ks_ptr, vs_ptr, op_ptr, lse_ptr,
                       pt_stride, qk_scale, NS, CH,
                       L: tl.constexpr, MQ: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
                       BN: tl.constexpr):
    """Split-KV decode/verify attention for one (request, KV head, split): the 8 query heads
    of the group times the L block tokens form MQ rows. FP8 cache read through the page
    table (next tile's page ids prefetched), BF16 queries and probabilities, FP32
    accumulation. Writes the split's normalized partial output and its log2-sum-exp."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    G: tl.constexpr = 8
    D: tl.constexpr = 64
    end = tl.load(cs_ptr + b)
    seq = end - L
    if WINDOW > 0:
        start = tl.maximum(seq - WINDOW + 1, 0)
    else:
        start = seq * 0
    lo = start + s * CH
    hi = tl.minimum(lo + CH, end)
    r = tl.arange(0, MQ)
    i = r // G
    g = r % G
    rvalid = i < L
    qpos = seq + i
    d = tl.arange(0, D)
    q = tl.load(q_ptr + ((b * L + i) * 1024 + (h * G + g) * D)[:, None] + d[None, :],
                mask=rvalid[:, None], other=0.0)
    sc_scale = qk_scale * tl.load(ks_ptr)
    m_i = tl.full([MQ], float("-inf"), tl.float32)
    l_i = tl.zeros([MQ], tl.float32)
    acc = tl.zeros([MQ, D], tl.float32)
    pt_row = pt_ptr + b.to(tl.int64) * pt_stride
    n = lo + tl.arange(0, BN)
    pg = tl.load(pt_row + n // PAGE, mask=n < hi, other=0)
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        nm = n < hi
        row = pg.to(tl.int64) * PAGE + (n % PAGE)
        kv_off = row[:, None] * 128 + h * D + d[None, :]
        kt = tl.load(k_ptr + kv_off, mask=nm[:, None], other=0.0)
        vt = tl.load(v_ptr + kv_off, mask=nm[:, None], other=0.0)
        nn = n + BN
        pg = tl.load(pt_row + nn // PAGE, mask=nn < hi, other=0)
        sc = tl.dot(q, tl.trans(kt.to(tl.bfloat16))) * sc_scale
        valid = nm[None, :] & (n[None, :] <= qpos[:, None])
        if WINDOW > 0:
            valid = valid & (n[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(valid, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt.to(tl.bfloat16))
        m_i = m_new
    vsc = tl.load(vs_ptr)
    ok = l_i > 0
    l_safe = tl.where(ok, l_i, 1.0)
    o = acc / l_safe[:, None] * vsc
    lse = tl.where(ok, tl.where(m_i == float("-inf"), 0.0, m_i) + tl.log2(l_safe), float("-inf"))
    base = ((b * 2 + h) * NS + s) * MQ
    tl.store(op_ptr + (base + r)[:, None] * D + d[None, :], o)
    tl.store(lse_ptr + base + r, lse)


@triton.jit
def _attn_combine_kernel(op_ptr, lse_ptr, gate_ptr, gate_stride, out_ptr, NS,
                         L: tl.constexpr, MQ: tl.constexpr, NS_P2: tl.constexpr):
    """Merge the splits of one (token, query head), then the per-head RMSNorm x sigmoid(gate)
    (SGLang's separate FusedRMSNormGate kernel, fused here)."""
    t = tl.program_id(0)
    hq = tl.program_id(1)
    D: tl.constexpr = 64
    b = t // L
    i = t % L
    h = hq // 8
    g = hq % 8
    r = i * 8 + g
    s = tl.arange(0, NS_P2)
    sm = s < NS
    base = ((b * 2 + h) * NS + s) * MQ + r
    lse = tl.load(lse_ptr + base, mask=sm, other=float("-inf"))
    mx = tl.max(lse, 0)
    w = tl.where(sm & (lse > float("-inf")), tl.exp2(lse - mx), 0.0)
    d = tl.arange(0, D)
    o = tl.load(op_ptr + base[:, None] * D + d[None, :], mask=sm[:, None], other=0.0)
    o = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    gt = tl.load(gate_ptr + t * gate_stride + hq * D + d).to(tl.float32)
    o = o * tl.sigmoid(gt)
    tl.store(out_ptr + t * 1024 + hq * D + d, o.to(tl.bfloat16))


def _attn_plan(batch, length, window, max_span_full, bn, target):
    span = (window + length - 1) if window else max_span_full
    ns = max(1, -(-target // (2 * batch)))
    ch = max(2 * bn, -(-span // ns))
    ch = -(-ch // bn) * bn
    ns = max(1, -(-span // ch))
    return ns, ch


def _attention_forward(q, k_cache, v_cache, page_table, cache_seqlens, k_scale, v_scale, gate,
                       batch, length, window, scale, bn=64, target=264):
    """q [T, 1024] bf16 (normed, roped), gate view [T, 1024] -> gated, normed output [T, 1024] bf16."""
    tokens = batch * length
    page = k_cache.shape[1]
    mq = max(16, _np2(8 * length))
    ns, ch = _attn_plan(batch, length, window, page_table.shape[1] * page, bn, target)
    dev = q.device
    op = torch.empty((batch, 2, ns, mq, 64), dtype=torch.float32, device=dev)
    lse = torch.empty((batch, 2, ns, mq), dtype=torch.float32, device=dev)
    _attn_split_kernel[(batch, 2, ns)](
        q, k_cache, v_cache, page_table, cache_seqlens, k_scale, v_scale, op, lse,
        page_table.stride(0), scale * LOG2E, ns, ch,
        L=length, MQ=mq, PAGE=page, WINDOW=window, BN=bn, num_warps=4, num_stages=2)
    out = torch.empty((tokens, 1024), dtype=torch.bfloat16, device=dev)
    _attn_combine_kernel[(tokens, 16)](op, lse, gate, gate.stride(0), out, ns,
                                       L=length, MQ=mq, NS_P2=max(2, _np2(ns)), num_warps=2)
    return out


def _fa3_forward(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, k_scale, v_scale, gate,
                 batch, length, window, scale, scaled, sched, state):
    tokens = batch * length
    descale = {}
    if scaled:
        descale = dict(k_descale=k_scale.reshape(1, 1).expand(batch, 2),
                       v_descale=v_scale.reshape(1, 1).expand(batch, 2))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, 16, 64).to(FP8), k_cache=k_cache, v_cache=v_cache, page_table=page_table,
        cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=length,
        softmax_scale=scale, causal=True, window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=sched, **descale)
    if isinstance(out, tuple):
        out = out[0]
    return state["o_norm_gate"](out.reshape(tokens, 1024).to(torch.bfloat16), gate)


def _attn_choice(state, batch, length, window):
    c = state["attn_cfg"].get((batch, length, window > 0))
    if c is not None:
        return c
    if window or batch == 1:
        return ("mine", 64, 264)
    return ("fa3",)


ATTN_CANDS = (("mine", 64, 264), ("mine", 128, 528), ("mine", 64, 1056), ("mine", 128, 1056),
              ("mine", 64, 1584), ("mine", 128, 1584))


# ------------------------------------ self tests --------------------------------------

def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _selftest_moe(weights, state, cfgs):
    from nvfp4 import dequantize
    dev = weights["w13"].device
    gen = torch.Generator(device=dev).manual_seed(1234)
    tokens = 12
    x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
    picks = [0, 5, 31]
    ls = 64
    ids = torch.empty((tokens, TOP_K), dtype=torch.int32)
    for t in range(tokens):
        ids[t] = torch.tensor([ls + picks[t % 3], ls + picks[(t + 1) % 3]] + list(range(128, 134)))
    ids = ids.to(dev)
    wts = torch.rand((tokens, TOP_K), generator=gen, device=dev) + 0.1
    wts = wts / wts.sum(-1, keepdim=True)
    local_start = torch.tensor([ls], dtype=torch.int32, device=dev)
    pw = 1.0 + weights["post_expert_norm"].float()
    ref = torch.zeros((tokens, D_EXPERT), device=dev)
    for e in picks:
        w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e])
        w2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e])
        for t in range(tokens):
            for s in range(2):
                if int(ids[t, s]) - ls != e:
                    continue
                hh = x[t].float() @ w13.T
                inter = (torch.nn.functional.silu(hh[:D_EXPERT]) * hh[D_EXPERT:]).to(torch.bfloat16).float()
                yy = (inter @ w2.T) * pw
                yy = yy * torch.rsqrt(yy.pow(2).mean() + 1e-8)
                ref[t] += wts[t, s] * yy
        del w13, w2
    errs = {}
    for c in cfgs:
        try:
            out = _moe_forward(x, wts, ids, local_start, state, c)
            errs[c] = _rel(out, ref)
        except Exception:
            print(f"moe cand {c} FAILED\n" + traceback.format_exc()[-1500:])
    return errs


def _attn_ref(q, kc, vc, pt, cs, ks, vs, gate, batch, length, window, scale):
    page = kc.shape[1]
    kf = kc.view(-1, 2, 64).float() * ks
    vf = vc.view(-1, 2, 64).float() * vs
    qv = q.view(batch, length, 2, 8, 64).float()
    outs = []
    for b in range(batch):
        end = int(cs[b])
        seq = end - length
        t = torch.arange(end, device=q.device)
        rows = pt[b].long()[t // page] * page + t % page
        kk, vv = kf[rows], vf[rows]
        qp = seq + torch.arange(length, device=q.device)
        vis = t[None, :] <= qp[:, None]
        if window:
            vis &= t[None, :] > qp[:, None] - window
        att = torch.einsum("lhgd,thd->lhgt", qv[b], kk) * scale
        att = att.masked_fill(~vis[:, None, None, :], float("-inf")).softmax(-1)
        outs.append(torch.einsum("lhgt,thd->lhgd", att, vv).reshape(length, 16, 64))
    o = torch.cat(outs).reshape(batch * length, 16, 64)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)
    return o.reshape(batch * length, 1024) * torch.sigmoid(gate.float())


def _selftest_attn(device):
    worst = {}
    gen = torch.Generator(device=device).manual_seed(77)
    scale = 0.125 * 5.4730064863875
    for page in (1, 64):
        for window in (0, 2048):
            for length in (1, 8):
                lens = [1500, 3000, 37] if not window else [5000, 2100, 37]
                batch = len(lens)
                max_pages = (max(lens) + length + page - 1) // page + 2
                total = batch * max_pages
                perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
                pt = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
                for b, n in enumerate(lens):
                    np_ = (n + length + page - 1) // page
                    pt[b, :np_] = perm[b * max_pages: b * max_pages + np_]
                kc = torch.randn((total, page, 2, 64), generator=gen, device=device).to(FP8)
                vc = torch.randn((total, page, 2, 64), generator=gen, device=device).to(FP8)
                cs = torch.tensor([n + length for n in lens], dtype=torch.int32, device=device)
                tokens = batch * length
                q = torch.randn((tokens, 16, 64), generator=gen, device=device)
                q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True))).to(torch.bfloat16).reshape(tokens, 1024)
                qkv = torch.randn((tokens, 2304), generator=gen, device=device).to(torch.bfloat16)
                gate = qkv[:, 1280:]
                ks = torch.tensor(0.5, device=device)
                vs = torch.tensor(0.25, device=device)
                ref = _attn_ref(q, kc, vc, pt, cs, 0.5, 0.25, gate, batch, length, window, scale)
                for c in ATTN_CANDS:
                    out = _attention_forward(q, kc, vc, pt, cs, ks, vs, gate, batch, length, window, scale,
                                             bn=c[1], target=c[2])
                    worst[c] = max(worst.get(c, 0.0), _rel(out, ref))
    return worst


# ------------------------------- prepare-time autotuning / lab -------------------------

def _bench_graph(fn, flush, reps=15):
    """Median µs of one CUDA-graph replay of fn, with a cold L2 before each replay
    (includes the graph launch overhead, ~4-8 µs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    times = []
    for _ in range(reps):
        flush.add_(1.0)
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) * 1000.0)
    del g
    times.sort()
    return times[len(times) // 2]


def _kstats(k):
    try:
        return f"regs={k.n_regs} spills={k.n_spills} smem={k.metadata.shared}"
    except Exception:
        return "stats?"


def _autotune_attention(state, dev, flush, ok_cands):
    gen = torch.Generator(device=dev).manual_seed(5)
    ctx, page = 80000, 1
    maxlen = int(ctx * 1.075) + 1
    bmax = 32
    max_pages = maxlen + 8
    total = bmax * max_pages
    kc = torch.empty((total, page, 2, 64), dtype=FP8, device=dev)
    vc = torch.empty((total, page, 2, 64), dtype=FP8, device=dev)
    for cache in (kc, vc):
        flat = cache.view(-1, 128)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            flat[at:at + n] = torch.randn((n, 128), generator=gen, device=dev).to(FP8)
    pt_all = torch.randperm(total, generator=gen, device=dev).to(torch.int32).view(bmax, max_pages)
    u = torch.rand((bmax,), generator=gen, device=dev)
    lens_all = (ctx * (1 + 0.075 * (2 * u - 1))).long()
    scale = 0.125 * 5.4730064863875
    one = torch.tensor(1.0, device=dev)
    for batch in (1, 16, 32):
        for length in (1, 8):
            for window in (2048, 0):
                tokens = batch * length
                pt = pt_all[:batch].contiguous()
                cs = (lens_all[:batch] + length).to(torch.int32)
                cu = torch.arange(0, tokens + 1, length, dtype=torch.int32, device=dev)
                q = torch.randn((tokens, 1024), generator=gen, device=dev).to(torch.bfloat16)
                gate = torch.randn((tokens, 2304), generator=gen, device=dev).to(torch.bfloat16)[:, 1280:]
                res = {("fa3",): _bench_graph(lambda: _fa3_forward(
                    q, kc, vc, pt, cs, cu, one, one, gate, batch, length, window, scale, False, None, state), flush)}
                for c in ok_cands:
                    res[c] = _bench_graph(lambda: _attention_forward(
                        q, kc, vc, pt, cs, one, one, gate, batch, length, window, scale, bn=c[1], target=c[2]),
                        flush)
                best = min(res, key=res.get)
                state["attn_cfg"][(batch, length, window > 0)] = best
                print(f"attn B={batch} L={length} {'sliding' if window else 'full'}: " +
                      " ".join(f"{'/'.join(map(str, k))}={v:.1f}" for k, v in res.items()) + f" -> {best}")
    del kc, vc


def _synthetic_routing(gen, dev, tokens, pop):
    ids = torch.multinomial(pop.expand(tokens, N_EXPERTS), TOP_K, replacement=False, generator=gen).to(torch.int32)
    counts = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ls = LOCAL_EXPERTS * int(counts.argmax())
    wts = torch.rand((tokens, TOP_K), generator=gen, device=dev) + 0.1
    return ids, torch.tensor([ls], dtype=torch.int32, device=dev), wts / wts.sum(-1, keepdim=True), ls


def _moe_lab(state, dev, flush, ok_cfgs):
    """Per-kernel timings of the MoE pieces (printed for the write-up)."""
    gen = torch.Generator(device=dev).manual_seed(11)
    rank_p = 1.0 / torch.arange(1, N_EXPERTS + 1, device=dev).float() ** 1.1
    pop = rank_p[torch.randperm(N_EXPERTS, generator=gen, device=dev)]
    for tokens in (1, 32, 256):
        ids, local_start, wts, ls = _synthetic_routing(gen, dev, tokens, pop)
        x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        R = tokens * TOP_K
        line = [f"lab T={tokens}"]
        inter = torch.zeros((R, D_EXPERT), dtype=torch.bfloat16, device=dev)
        for c in ok_cfgs:
            sorted_ids, bexp, NBM = _moe_align(ids, local_start, tokens, c[1])
            nreal = int((bexp >= 0).sum())
            k = _moe_gemm(c[0], x, 13, inter, sorted_ids, bexp, R, NBM, c, state)
            t1 = _bench_graph(lambda: _moe_gemm(c[0], x, 13, inter, sorted_ids, bexp, R, NBM, c, state), flush)
            line.append(f"{'/'.join(map(str, c))}: gemm1={t1:.1f} blocks={nreal} [{_kstats(k)}]")
        sorted_ids, bexp, NBM = _moe_align(ids, local_start, tokens, 16)
        line.append(f"align16={_bench_graph(lambda: _moe_align(ids, local_start, tokens, 16), flush):.1f}")
        probe = torch.empty(NBM * 64 * 64, dtype=torch.int32, device=dev)
        line.append("bwprobe_gemm1(BM16)=%.1f" % _bench_graph(lambda: _bw_probe_kernel[(NBM * 64,)](
            state["moe"]["w13s"], probe, bexp, K=D_EXPERT, WROWS=2 * D_EXPERT, NBM=NBM, BN=64, num_warps=4,
            num_stages=3), flush))
        nexp = int((bexp >= 0).sum())
        line.append(f"(gemm1 bytes {nexp * 2 * D_EXPERT * D_EXPERT * (0.5 + 1 / 16) / 1e6:.0f} MB if one block/expert)")
        line.append("empty_graph=%.1f" % _bench_graph(lambda: None, flush))
        print("\n  ".join(line))


def _autotune_moe(state, dev, flush, ok_cfgs):
    gen = torch.Generator(device=dev).manual_seed(9)
    # Skewed routing: Zipf-like expert popularity ("a few experts take most tokens").
    rank_p = 1.0 / torch.arange(1, N_EXPERTS + 1, device=dev).float() ** 1.1
    pop = rank_p[torch.randperm(N_EXPERTS, generator=gen, device=dev)]
    for tokens in (1, 8, 16, 32, 128, 256):
        ids, local_start, wts, ls = _synthetic_routing(gen, dev, tokens, pop)
        x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        loc = (ids.long() - ls) % N_EXPERTS
        nloc = int((loc < LOCAL_EXPERTS).sum())
        ndist = int(torch.unique(loc[loc < LOCAL_EXPERTS]).numel())
        res = {"marlin": _bench_graph(lambda: _experts(x, wts, (ids - local_start) % N_EXPERTS, state), flush)}
        for c in ok_cfgs:
            if c[1] > 16 and tokens < 16:
                continue
            if c[1] < 16 and tokens > 32:
                continue
            res[c] = _bench_graph(lambda: _moe_forward(x, wts, ids, local_start, state, c), flush)
        best = min(res, key=res.get)
        state["moe_cfg"][tokens] = best
        print(f"moe T={tokens} routes={nloc} experts={ndist}: " +
              " ".join(f"{k if isinstance(k, str) else '/'.join(map(str, k))}={v:.1f}" for k, v in res.items())
              + f" -> {best}")


def prepare(weights):
    state = _base_prepare(weights)
    dev = weights["qkv_gate"].device
    state["moe"] = {
        "w13": weights["w13"].contiguous().view(torch.int32),
        "w13s": _repack_split(weights["w13"].contiguous()),
        "sf13": weights["w13_sf"].contiguous(),
        "gs13": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous().view(torch.int32),
        "w2s": _repack_split(weights["w2"].contiguous()),
        "sf2": weights["w2_sf"].contiguous(),
        "gs2": weights["w2_gs"].float().contiguous(),
        "post_w": weights["post_expert_norm"].contiguous(),
    }
    state["moe_cfg"] = {}
    state["attn_cfg"] = {}
    print("triton", triton.__version__, "torch", torch.__version__)
    t_all = time.time()
    ok_moe = []
    try:
        errs = _selftest_moe(weights, state, MOE_CANDS)
        ok_moe = [c for c, e in errs.items() if e < 2e-2]
        print("selftest moe:", {"/".join(map(str, c)): f"{e:.2e}" for c, e in errs.items()})
    except Exception:
        print("selftest moe FAILED\n" + traceback.format_exc()[-3000:])
    state["use_moe"] = bool(ok_moe)
    ok_attn = []
    try:
        errs = _selftest_attn(dev)
        ok_attn = [c for c, e in errs.items() if e < 2e-2]
        print("selftest attn:", {"/".join(map(str, c)): f"{e:.2e}" for c, e in errs.items()})
    except Exception:
        print("selftest attn FAILED\n" + traceback.format_exc()[-3000:])
    state["use_attn"] = bool(ok_attn)
    flush = torch.empty(96 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, fn, ok in (("moe lab", _moe_lab, ok_moe), ("attn", _autotune_attention, ok_attn),
                         ("moe", _autotune_moe, ok_moe)):
        if ok and time.time() - t_all < 200:
            try:
                fn(state, dev, flush, ok)
            except Exception:
                print(f"{name} FAILED\n" + traceback.format_exc()[-3000:])
    del flush
    print(f"prepare extras {time.time() - t_all:.1f}s")
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    choice = _attn_choice(state, batch, length, window) if state["use_attn"] else ("fa3",)
    if choice[0] == "mine":
        o = _attention_forward(q.reshape(tokens, hq * d), k_cache, v_cache, data["page_table"],
                               data["cache_seqlens"], data["k_scale"], data["v_scale"], gate,
                               batch, length, window, scale, bn=choice[1], target=choice[2])
    else:
        o = _fa3_forward(q, k_cache, v_cache, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"],
                         data["k_scale"], data["v_scale"], gate, batch, length, window, scale, scaled,
                         data["scheduler_metadata"], state)
    attn_output = _gemm(o, dense["o"])

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    x_moe = state["pre_expert_norm"](expert_input)
    moe_cfg = state["moe_cfg"].get(tokens, "marlin") if state["use_moe"] else "marlin"
    if moe_cfg == "marlin":
        local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
        expert_output = _experts(x_moe, topk_weights, local_ids, state)
    else:
        expert_output = _moe_forward(x_moe, topk_weights, topk_ids, data["local_start"], state, moe_cfg)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# =====================================================================================
# v5: Marlin autotuning + small-kernel fusion (later definitions override earlier ones)
# =====================================================================================

@triton.jit
def _local_ids_kernel(ids_ptr, ls_ptr, out_ptr, R: tl.constexpr, R_P2: tl.constexpr):
    """expert_map[(ids - local_start) % 256] in one launch: 0..31 local, -1 elsewhere."""
    o = tl.arange(0, R_P2)
    m = o < R
    ids = tl.load(ids_ptr + o, mask=m, other=0)
    loc = (ids - tl.load(ls_ptr) + 256) % 256
    tl.store(out_ptr + o, tl.where(loc < 32, loc, -1).to(tl.int32), mask=m)


@triton.jit
def _kv_store_kernel(k_ptr, k_stride, v_ptr, v_stride, kc_ptr, vc_ptr, loc_ptr, ks_ptr, vs_ptr):
    """Both caches' new rows in one launch: fp8(k / k_scale), fp8(v / v_scale)."""
    t = tl.program_id(0)
    d = tl.arange(0, 128)
    k = tl.load(k_ptr + t * k_stride + d).to(tl.float32) / tl.load(ks_ptr)
    v = tl.load(v_ptr + t * v_stride + d).to(tl.float32) / tl.load(vs_ptr)
    row = tl.load(loc_ptr + t).to(tl.int64)
    tl.store(kc_ptr + row * 128 + d, k.to(tl.float8e4nv))
    tl.store(vc_ptr + row * 128 + d, v.to(tl.float8e4nv))


@triton.jit
def _ffn_boundary_sliced_kernel(s_ptr, u_ptr, al_ptr, be_ptr, wf_ptr, wn_ptr, s_out, h_out,
                                H: tl.constexpr, NZ: tl.constexpr):
    """T <= 8: the post-FFN boundary reading the unpooler's 1024-column slice directly (the
    all-gather's zero columns are implicit), replacing F.pad + the boundary kernel."""
    t = tl.program_id(0)
    d = tl.arange(0, H)
    f = tl.load(u_ptr + t * NZ + d, mask=d < NZ, other=0.0).to(tl.float32) * tl.load(wf_ptr + d)
    f = f * tl.rsqrt(tl.sum(f * f, 0) / H + 1e-8)
    s = tl.load(al_ptr + d) * tl.load(s_ptr + t * H + d) + tl.load(be_ptr + d) * f
    tl.store(s_out + t * H + d, s)
    hh = s * tl.rsqrt(tl.sum(s * s, 0) / H + 1e-5) * tl.load(wn_ptr + d)
    tl.store(h_out + t * H + d, hh.to(tl.bfloat16))


def _experts_cfg(x, topk_weights, local_ids, state, block_size_m, atomic):
    """_experts with a chosen Marlin block size / atomic-add mode; local_ids already mapped."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    topk_weights = topk_weights.to(torch.float32).contiguous()
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    alloc = torch.zeros if atomic else torch.empty
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache1 = alloc((num_tokens * TOP_K, gemm1_n), device=x.device, dtype=x.dtype)
    cache3 = alloc((num_tokens * TOP_K, size_k), device=x.device, dtype=x.dtype)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _default_bm(tokens):
    for bm in BLOCK_SIZE_M_CANDIDATES:
        if tokens * TOP_K / LOCAL_EXPERTS / bm < 0.9:
            return bm
    return BLOCK_SIZE_M_CANDIDATES[-1]


def _local_ids(topk_ids, local_start):
    R = topk_ids.numel()
    out = torch.empty_like(topk_ids, dtype=torch.int32)
    _local_ids_kernel[(1,)](topk_ids, local_start, out, R=R, R_P2=max(2, _np2(R)), num_warps=1)
    return out


MARLIN_CANDS = tuple((bm, at) for bm in (8, 16, 32, 48, 64) for at in (False, True))


def _selftest_marlin(weights, state):
    """Each Marlin config vs the SGLang default config on the same routing (they must agree)."""
    dev = weights["w13"].device
    gen = torch.Generator(device=dev).manual_seed(4321)
    ok = []
    for tokens in (3, 40):
        x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(N_EXPERTS, generator=gen, device=dev)[:TOP_K] for _ in range(tokens)])
        ids = (ids % 48).to(torch.int32) if False else ids.to(torch.int32)
        ids[:, 0] = torch.arange(tokens, device=dev) % 5 + 32  # make sure locals exist
        ids[:, 1] = torch.where(ids[:, 1] == ids[:, 0], ids[:, 1] + 1, ids[:, 1])
        ls = torch.tensor([32], dtype=torch.int32, device=dev)
        wts = torch.rand((tokens, TOP_K), generator=gen, device=dev) + 0.1
        wts = wts / wts.sum(-1, keepdim=True)
        ref = _experts(x, wts, (ids - ls) % N_EXPERTS, state)
        loc = _local_ids(ids, ls)
        lref = state["bank"]["expert_map"][((ids - ls) % N_EXPERTS).long()]
        assert torch.equal(loc, lref), "local ids mismatch"
        res = {}
        for c in MARLIN_CANDS:
            try:
                res[c] = _rel(_experts_cfg(x, wts, loc, state, c[0], c[1]), ref)
            except Exception:
                res[c] = 1.0
                print(f"marlin {c} failed: " + traceback.format_exc()[-400:])
        print(f"selftest marlin T={tokens}:", {f"{c[0]}/{int(c[1])}": f"{e:.1e}" for c, e in res.items()})
        ok.append({c for c, e in res.items() if e < 1e-2})
    return sorted(set.intersection(*ok))


def _selftest_fusions(weights, state):
    dev = weights["w13"].device
    gen = torch.Generator(device=dev).manual_seed(99)
    # KV store
    tokens = 12
    k = torch.randn((tokens, 128), generator=gen, device=dev).to(torch.bfloat16)
    v = torch.randn((tokens, 2304), generator=gen, device=dev).to(torch.bfloat16)[:, 1152:1280]
    kc = torch.zeros((64, 1, 2, 64), dtype=FP8, device=dev)
    vc = torch.zeros_like(kc)
    loc = torch.randperm(64, generator=gen, device=dev)[:tokens]
    one = torch.tensor(1.0, device=dev)
    _kv_store_kernel[(tokens,)](k, k.stride(0), v, v.stride(0), kc, vc, loc, one, one, num_warps=1)
    kv_ok = (torch.equal(kc.view(-1, 128)[loc].view(torch.uint8), k.to(FP8).view(torch.uint8)) and
             torch.equal(vc.view(-1, 128)[loc].view(torch.uint8), v.to(FP8).view(torch.uint8)))
    # sliced FFN boundary vs SGLang's (pad + fused boundary)
    s = torch.randn((5, HIDDEN), generator=gen, device=dev)
    u = torch.randn((5, UNPOOL_SLICE), generator=gen, device=dev).to(torch.bfloat16)
    ref_h, ref_s, _ = state["ffn_boundary"](s.clone(), F.pad(u, (0, HIDDEN - UNPOOL_SLICE)),
                                            branch_dtype=torch.bfloat16)
    h_out, s_out = _ffn_boundary_sliced(s, u, state)
    carried = state["fb"]["alpha"] * s
    e_s = _rel(s_out - carried, ref_s - carried)
    e_h = _rel(h_out, ref_h)
    print(f"selftest fusions: kv_store exact={kv_ok} ffn_boundary update err={e_s:.2e} hidden err={e_h:.2e}")
    return kv_ok, (e_s < 5e-3 and e_h < 5e-3)


def _ffn_boundary_sliced(s, u, state):
    fb = state["fb"]
    tokens = s.shape[0]
    s_out = torch.empty_like(s)
    h_out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=s.device)
    _ffn_boundary_sliced_kernel[(tokens,)](s, u, fb["alpha"], fb["beta"], fb["wf"], fb["wn"], s_out, h_out,
                                           H=HIDDEN, NZ=UNPOOL_SLICE, num_warps=16)
    return h_out, s_out


def _autotune_marlin(state, dev, flush, cands):
    gen = torch.Generator(device=dev).manual_seed(9)
    rank_p = 1.0 / torch.arange(1, N_EXPERTS + 1, device=dev).float() ** 1.1
    pop = rank_p[torch.randperm(N_EXPERTS, generator=gen, device=dev)]
    for tokens in (1, 8, 16, 32, 128, 256):
        res_all = {}
        for rep in range(2):  # two routing draws, summed, to reduce draw noise
            ids, local_start, wts, ls = _synthetic_routing(gen, dev, tokens, pop)
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
            loc = _local_ids(ids, local_start)
            for c in cands:
                t = _bench_graph(lambda: _experts_cfg(x, wts, loc, state, c[0], c[1]), flush, reps=9)
                res_all[c] = res_all.get(c, 0.0) + t / 2
        default = (_default_bm(tokens), False)
        best = min(res_all, key=res_all.get)
        if res_all[best] > 0.97 * res_all.get(default, 1e9):
            best = default  # only move off SGLang's choice for a clear win
        state["marlin_cfg"][tokens] = best
        print(f"marlin T={tokens}: " + " ".join(f"{c[0]}/{int(c[1])}={v:.1f}" for c, v in res_all.items())
              + f" default={default} -> {best}")


def _decode_prepare_v6_base(weights):
    state = _base_prepare(weights)
    dev = weights["qkv_gate"].device
    alpha1, beta1 = None, None
    w = weights["ffn_residual_gate"].float() / 0.3
    beta = torch.sigmoid(w)
    state["fb"] = {
        "alpha": torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(),
        "beta": beta.contiguous(),
        "wf": (1.0 + weights["post_ffn_norm"].float()).contiguous(),
        "wn": (1.0 + weights["next_pre_attn_norm"].float()).contiguous(),
    }
    state["attn_cfg"] = {}
    state["marlin_cfg"] = {}
    print("triton", triton.__version__, "torch", torch.__version__)
    t_all = time.time()
    state["use_kv"], state["use_fb"] = False, False
    try:
        state["use_kv"], state["use_fb"] = _selftest_fusions(weights, state)
    except Exception:
        print("selftest fusions FAILED\n" + traceback.format_exc()[-2000:])
    marlin_ok = []
    try:
        marlin_ok = _selftest_marlin(weights, state)
    except Exception:
        print("selftest marlin FAILED\n" + traceback.format_exc()[-2000:])
    state["use_local_kernel"] = bool(marlin_ok)
    ok_attn = []
    try:
        errs = _selftest_attn(dev)
        ok_attn = [c for c, e in errs.items() if e < 2e-2]
        print("selftest attn:", {"/".join(map(str, c)): f"{e:.2e}" for c, e in errs.items()})
    except Exception:
        print("selftest attn FAILED\n" + traceback.format_exc()[-3000:])
    state["use_attn"] = bool(ok_attn)
    flush = torch.empty(96 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, fn, ok in (("attn", _autotune_attention, ok_attn), ("marlin", _autotune_marlin, marlin_ok)):
        if ok and time.time() - t_all < 200:
            try:
                fn(state, dev, flush, ok)
            except Exception:
                print(f"{name} FAILED\n" + traceback.format_exc()[-3000:])
    del flush
    print(f"prepare extras {time.time() - t_all:.1f}s")
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if state["use_kv"] and k.stride(-1) == 1 and v.stride(-1) == 1:
        _kv_store_kernel[(tokens,)](k, k.stride(0), v, v.stride(0), k_cache, v_cache, data["cache_loc"],
                                    data["k_scale"], data["v_scale"], num_warps=1)
    else:
        kk, vv = (k / data["k_scale"], v / data["v_scale"]) if scaled else (k, v)
        store_cache(kk.to(FP8).view(torch.uint8), vv.to(FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    choice = _attn_choice(state, batch, length, window) if state["use_attn"] else ("fa3",)
    if choice[0] == "mine":
        o = _attention_forward(q.reshape(tokens, hq * d), k_cache, v_cache, data["page_table"],
                               data["cache_seqlens"], data["k_scale"], data["v_scale"], gate,
                               batch, length, window, scale, bn=choice[1], target=choice[2])
    else:
        o = _fa3_forward(q, k_cache, v_cache, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"],
                         data["k_scale"], data["v_scale"], gate, batch, length, window, scale, scaled,
                         data["scheduler_metadata"], state)
    attn_output = _gemm(o, dense["o"])

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        if state["use_local_kernel"]:
            local_ids = _local_ids(topk_ids, data["local_start"])
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
        if state["use_local_kernel"]:
            local_ids.record_stream(main_stream)
    x_moe = state["pre_expert_norm"](expert_input)
    # The embedded Track-1 kernel beats Marlin at T=8/16. This check-only v9
    # candidate also tests T=32 with the single-token-thread occupancy point;
    # keep Marlin at T=1 and retain the established path for T=128/256.
    if state.get("use_fast_moe", False) and (tokens in (8, 16, 32) or tokens >= 128):
        expert_output = _fast_moe_forward({
            "hidden_states": x_moe,
            "topk_ids": local_ids,
            "topk_weights": topk_weights,
            "config": {"local_start": 0, "local_count": LOCAL_EXPERTS},
        }, state["fast_moe"])
    elif state["use_local_kernel"]:
        bm, atomic = state["marlin_cfg"].get(tokens, (_default_bm(tokens), False))
        expert_output = _experts_cfg(x_moe, topk_weights, local_ids, state, bm, atomic)
    else:
        expert_output = _experts(x_moe, topk_weights, (topk_ids - data["local_start"]) % N_EXPERTS, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        if state["use_fb"]:
            next_attn_input, stream = _ffn_boundary_sliced(stream, unpooled, state)
            return next_attn_input, stream, topk_ids
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
# Embedded Track-1 fused MoE implementation (namespaced with fast_moe prefixes below).
"""Routed experts (NVFP4 W4A16) for H200: three launches, no host sync.

  gemm1 : per-CTA routing in smem, then (expert, 16 gate rows + 16 up rows) work items,
          4 warps split K, mma.sync bf16 with in-register NVFP4 dequant, SwiGLU epilogue
          -> inter[pair, 4096] bf16 (pair = token*8 + slot).
  gemm2 : same engine over W2 (32 output rows per item), epilogue applies (1 + w),
          writes y' bf16 and atomically accumulates sum(y'^2) per pair.
  final : out[t] = sum_slots g * y'[pair] * rsqrt(mean(y'^2) + eps).

FastMoeWeights are repacked once in _fast_moe_prepare() into a lane-ordered stream: per (expert, item, k64 step)
1152 contiguous bytes = 2 x 512 B of E2M1 codes (one 16-byte vector per lane, codes placed at
bit positions chosen so that a rotate + mask turns them into bf16 bit patterns) + 128 B of the
untouched E4M3 block scales. Still 4 bits per weight + 1 E4M3 byte per 16 weights (lossless).
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class FastMoeWeights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class FastMoeConfig(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class FastMoeInputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8]
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: FastMoeConfig


H = 4096
I_DIM = 4096
LOCAL = 32
TOP_K = 8

FAST_MOE_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define KDIM 4096
#define NWARPS 4
#define SPW 16          // k64 steps per warp (64 steps / 4 warps)
#define STEP_U4 72      // uint4 per (item, k64 step): 32 + 32 code vectors + 8 scale vectors
#define ITEM_U4 (64 * STEP_U4)
#define WP 6            // deeper register-prefetch pipeline for HBM latency hiding
#define AP 2            // activation prefetch depth

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_act(const __nv_bfloat16* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t rotl(uint32_t x, int L) { return __funnelshift_l(x, x, L); }

// w (two bf16 = e2m1 * 2^-126) times the low (H=0) or high (H=1) bf16 of s, broadcast.
template <int HI>
__device__ __forceinline__ uint32_t bmul(uint32_t w, uint32_t s) {
  __nv_bfloat162 wv = *reinterpret_cast<__nv_bfloat162*>(&w);
  __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 sb = HI ? __high2bfloat162(sv) : __low2bfloat162(sv);
  __nv_bfloat162 r = __hmul2(wv, sb);
  return *reinterpret_cast<uint32_t*>(&r);
}

// Two E4M3 scales (bytes 2*HI, 2*HI+1 of word) -> bf16x2 equal to scale * 2^119.
template <int HI, bool SAFE>
__device__ __forceinline__ uint32_t cvt_scale(uint32_t word) {
  if (!SAFE) {
    uint32_t p = __byte_perm(word, 0u, HI ? 0x4342 : 0x4140);
    return p * 16u + 0x77807780u;
  } else {
    __nv_fp8x2_storage_t two = (__nv_fp8x2_storage_t)((word >> (HI * 16)) & 0xFFFFu);
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(two, __NV_E4M3);
    __half2 h = *reinterpret_cast<__half2*>(&hr);
    float2 f = __half22float2(h);
    const float k = 6.64613997892457936e35f;  // 2^119
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x * k, f.y * k);
    return *reinterpret_cast<uint32_t*>(&b);
  }
}

// Decode two packed words (codes of sub-steps 2p, 2p+1) into 8 A-fragment registers.
__device__ __forceinline__ void dequant_pair(uint32_t A, uint32_t B, uint32_t* r0, uint32_t* r1) {
  const uint32_t M = 0x81C081C0u;
  uint32_t V = (A & 0xC180C180u) | (B & ~0xC180C180u);
  r0[0] = rotl(A, 6) & M;
  r0[1] = rotl(A, 2) & M;
  r0[2] = rotl(A, 12) & M;
  r0[3] = rotl(V, 14) & M;
  r1[0] = rotl(B, 4) & M;
  r1[1] = B & M;
  r1[2] = rotl(B, 10) & M;
  r1[3] = rotl(V, 8) & M;
}

template <int MODE, int NT, bool SAFE>
__global__ void __launch_bounds__(128) moe_gemm(
    const __nv_bfloat16* __restrict__ act, const int* __restrict__ ids, int M, int local_start,
    const uint4* __restrict__ wpk, const float* __restrict__ gsc, const float* __restrict__ normw,
    __nv_bfloat16* __restrict__ outp, float* __restrict__ ss) {
  constexpr int TILES = MODE == 1 ? 256 : 128;
  extern __shared__ int smem[];
  float* red = reinterpret_cast<float*>(smem);
  int* s_list = smem + NWARPS * 2 * NT * 4 * 32;
  __shared__ int s_cnt[32], s_off[32], s_fill[32], s_touched[32], s_ntouch;

  constexpr int CH = 8 * NT;
  __shared__ int s_ibase[32], s_nch[32], s_total;
  __shared__ int s_wcnt[NWARPS][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int npairs = M * 8;
  reinterpret_cast<int*>(s_wcnt)[tid] = 0;
  if (MODE == 1) {
    for (int i = blockIdx.x * 128 + tid; i < npairs; i += gridDim.x * 128) ss[i] = 0.f;
  }
  // Deterministic routing (identical list order in every CTA): warp w owns pairs [lo, hi).
  const int per = (npairs + NWARPS - 1) / NWARPS;
  const int lo = min(npairs, warp * per), hi = min(npairs, lo + per);
  __syncthreads();
  for (int i = lo + lane; i < hi; i += 32) {
    int e = ids[i] - local_start;
    if ((unsigned)e < 32u) atomicAdd(&s_wcnt[warp][e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) c += s_wcnt[w][lane];
    s_cnt[lane] = c;
    const int nch = (c + CH - 1) / CH;
    int incl = c, inci = nch * TILES;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, incl, o);
      int w = __shfl_up_sync(0xffffffffu, inci, o);
      if (lane >= o) { incl += v; inci += w; }
    }
    int base = incl - c;
    s_off[lane] = base;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) { const int t = s_wcnt[w][lane]; s_wcnt[w][lane] = base; base += t; }
    s_nch[lane] = nch;
    s_ibase[lane] = inci - nch * TILES;
    unsigned m = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) s_touched[__popc(m & ((1u << lane) - 1u))] = lane;
    if (lane == 31) { s_ntouch = __popc(m); s_total = inci; }
  }
  __syncthreads();
  for (int b0 = lo; b0 < hi; b0 += 32) {
    const int i = b0 + lane;
    const int e = i < hi ? ids[i] - local_start : -1;
    const bool loc = (unsigned)e < 32u;
    const unsigned peers = __match_any_sync(0xffffffffu, loc ? e : 64 + lane);
    const int rank = __popc(peers & ((1u << lane) - 1u));
    if (loc) s_list[s_wcnt[warp][e] + rank] = i;
    __syncwarp();
    if (loc && rank == 0) s_wcnt[warp][e] += __popc(peers);
    __syncwarp();
  }
  __syncthreads();

  // Items: (touched expert, tile, token chunk), chunk fastest so CTAs sharing weights run together.
  const int total = s_total;
  const int r = lane >> 2, cq = lane & 3;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    int k_ = 0;
    while (k_ + 1 < s_ntouch && s_ibase[s_touched[k_ + 1]] <= item) k_++;
    const int e = s_touched[k_];
    const int rem = item - s_ibase[e];
    const int nch = s_nch[e];
    const int tile = rem / nch;
    const int cnt = s_cnt[e], off = s_off[e];
    const uint4* wb = wpk + ((size_t)e * TILES + tile) * ITEM_U4 + (size_t)warp * SPW * STEP_U4;
    const float g = gsc[e];
    {
      const int c0 = (rem - tile * nch) * CH;
      const int nv = min(cnt - c0, 8 * NT);
      const int ntl = (nv + 7) >> 3;
      const __nv_bfloat16* bp[NT];
      bool bv[NT];
#pragma unroll
      for (int j = 0; j < NT; j++) {
        int idx = c0 + 8 * j + r;
        bv[j] = idx < cnt;
        int p = bv[j] ? s_list[off + idx] : 0;
        int row = MODE == 1 ? (p >> 3) : p;
        bp[j] = act + (size_t)row * KDIM + warp * (SPW * 64) + cq * 4;
      }
      float acc[2][NT][4];
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int v = 0; v < 4; v++) acc[a][j][v] = 0.f;

      uint4 pw0[WP], pw1[WP], psc[WP];
      uint2 pa[AP][NT][4];
#pragma unroll
      for (int s = 0; s < WP; s++) {
        pw0[s] = ld_stream(wb + s * STEP_U4 + lane);
        pw1[s] = ld_stream(wb + s * STEP_U4 + 32 + lane);
        psc[s] = ld_stream(wb + s * STEP_U4 + 64 + r);
      }
#pragma unroll
      for (int s = 0; s < AP; s++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int q = 0; q < 4; q++)
            pa[s][j][q] = bv[j] ? ld_act(bp[j] + s * 64 + q * 16) : make_uint2(0u, 0u);

#pragma unroll
      for (int ks = 0; ks < SPW; ks++) {
        const int ws = ks % WP, as = ks % AP;
        const uint4 w0 = pw0[ws], w1 = pw1[ws], sc = psc[ws];
        uint2 xb[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int q = 0; q < 4; q++) xb[j][q] = pa[as][j][q];
        if (ks + WP < SPW) {
          pw0[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + lane);
          pw1[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + 32 + lane);
          psc[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + 64 + r);
        }
        if (ks + AP < SPW) {
#pragma unroll
          for (int j = 0; j < NT; j++)
#pragma unroll
            for (int q = 0; q < 4; q++)
              pa[as][j][q] = bv[j] ? ld_act(bp[j] + (ks + AP) * 64 + q * 16) : make_uint2(0u, 0u);
        }
        // scales: [tile][row r / r+8][kb pair]
        uint32_t sc_[2][2][2];
        sc_[0][0][0] = cvt_scale<0, SAFE>(sc.x); sc_[0][0][1] = cvt_scale<1, SAFE>(sc.x);
        sc_[0][1][0] = cvt_scale<0, SAFE>(sc.y); sc_[0][1][1] = cvt_scale<1, SAFE>(sc.y);
        sc_[1][0][0] = cvt_scale<0, SAFE>(sc.z); sc_[1][0][1] = cvt_scale<1, SAFE>(sc.z);
        sc_[1][1][0] = cvt_scale<0, SAFE>(sc.w); sc_[1][1][1] = cvt_scale<1, SAFE>(sc.w);
        uint32_t R[2][4][4];  // [tile][substep][reg]
        dequant_pair(w0.x, w0.y, R[0][0], R[0][1]);
        dequant_pair(w0.z, w0.w, R[0][2], R[0][3]);
        dequant_pair(w1.x, w1.y, R[1][0], R[1][1]);
        dequant_pair(w1.z, w1.w, R[1][2], R[1][3]);
#pragma unroll
        for (int s = 0; s < 4; s++) {
#pragma unroll
          for (int t = 0; t < 2; t++) {
            const uint32_t s_r = sc_[t][0][s >> 1], s_r8 = sc_[t][1][s >> 1];
            uint32_t a0, a1, a2, a3;
            if (s & 1) {
              a0 = bmul<1>(R[t][s][0], s_r); a1 = bmul<1>(R[t][s][1], s_r8);
              a2 = bmul<1>(R[t][s][2], s_r); a3 = bmul<1>(R[t][s][3], s_r8);
            } else {
              a0 = bmul<0>(R[t][s][0], s_r); a1 = bmul<0>(R[t][s][1], s_r8);
              a2 = bmul<0>(R[t][s][2], s_r); a3 = bmul<0>(R[t][s][3], s_r8);
            }
#pragma unroll
            for (int j = 0; j < NT; j++)
              if (j < ntl) mma16816(acc[t][j], a0, a1, a2, a3, xb[j][s].x, xb[j][s].y);
          }
        }
      }

      // cross-warp split-K reduction
      __syncthreads();
#pragma unroll
      for (int t = 0; t < 2; t++)
#pragma unroll
        for (int j = 0; j < NT; j++)
          if (j < ntl)
#pragma unroll
            for (int v = 0; v < 4; v++) red[(((warp * 2 + t) * NT + j) * 4 + v) * 32 + lane] = acc[t][j][v];
      __syncthreads();
      for (int j = warp; j < ntl; j += NWARPS) {
        float h[2][4];
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int v = 0; v < 4; v++) {
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < NWARPS; w++) sum += red[(((w * 2 + t) * NT + j) * 4 + v) * 32 + lane];
            h[t][v] = sum * g;
          }
        if (MODE == 1) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int idx = c0 + 8 * j + 2 * cq + (v & 1);
            if (idx < cnt) {
              const int p = s_list[off + idx];
              const float gt = h[0][v], ut = h[1][v];
              const float val = gt / (1.f + __expf(-gt)) * ut;
              outp[(size_t)p * KDIM + tile * 16 + (v >> 1) * 8 + r] = __float2bfloat16(val);
            }
          }
        } else {
          float ssp[2] = {0.f, 0.f};
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int idx = c0 + 8 * j + 2 * cq + (v & 1);
            const int row0 = tile * 32 + (v >> 1) * 8 + r;
            const float y0 = h[0][v] * normw[row0];
            const float y1 = h[1][v] * normw[row0 + 16];
            if (idx < cnt) {
              const int p = s_list[off + idx];
              outp[(size_t)p * KDIM + row0] = __float2bfloat16(y0);
              outp[(size_t)p * KDIM + row0 + 16] = __float2bfloat16(y1);
            }
            ssp[v & 1] += y0 * y0 + y1 * y1;
          }
#pragma unroll
          for (int q = 0; q < 2; q++) {
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 4);
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 8);
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 16);
          }
          if (r == 0) {
#pragma unroll
            for (int q = 0; q < 2; q++) {
              const int idx = c0 + 8 * j + 2 * cq + q;
              if (idx < cnt) atomicAdd(&ss[s_list[off + idx]], ssp[q]);
            }
          }
        }
      }
    }
  }
}

__global__ void __launch_bounds__(256) moe_finalize(
    const __nv_bfloat16* __restrict__ ybuf, const float* __restrict__ ss, const int* __restrict__ ids,
    const float* __restrict__ tw, int local_start, __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, tid = threadIdx.x;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
#pragma unroll 1
  for (int k = 0; k < 8; k++) {
    const int p = t * 8 + k;
    const int e = ids[p] - local_start;
    if ((unsigned)e < 32u) {
      const float f = tw[p] * rsqrtf(ss[p] * (1.f / 4096.f) + 1e-8f);
      const uint4* yp = reinterpret_cast<const uint4*>(ybuf + (size_t)p * KDIM + tid * 16);
      uint4 a = yp[0], b = yp[1];
      const __nv_bfloat162* av = reinterpret_cast<const __nv_bfloat162*>(&a);
      const __nv_bfloat162* bv = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 fa = __bfloat1622float2(av[i]);
        float2 fb = __bfloat1622float2(bv[i]);
        acc[2 * i] += f * fa.x; acc[2 * i + 1] += f * fa.y;
        acc[8 + 2 * i] += f * fb.x; acc[8 + 2 * i + 1] += f * fb.y;
      }
    }
  }
  uint4 o[2];
  __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
  for (int i = 0; i < 8; i++) ov[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  uint4* op = reinterpret_cast<uint4*>(out + (size_t)t * KDIM + tid * 16);
  op[0] = o[0];
  op[1] = o[1];
}

template <int MODE, int NT, bool SAFE>
static void launch_t(const torch::Tensor& act, const torch::Tensor& ids, int M, int ls, const torch::Tensor& wpk,
                     const torch::Tensor& gsc, const torch::Tensor& normw, torch::Tensor& outp, torch::Tensor& ss,
                     int grid) {
  auto kern = moe_gemm<MODE, NT, SAFE>;
  const int smem = (NWARPS * 2 * NT * 4 * 32 + M * 8) * 4;
  static int configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem > 48 * 1024 ? smem : 48 * 1024);
    configured = smem;
  }
  kern<<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), ids.data_ptr<int>(), M, ls,
      reinterpret_cast<const uint4*>(wpk.data_ptr()), gsc.data_ptr<float>(), normw.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(outp.data_ptr()), ss.data_ptr<float>());
}

template <int MODE, int NT, bool SAFE>
static int occ_t(int M) {
  int n = 0;
  const int smem = (NWARPS * 2 * NT * 4 * 32 + M * 8) * 4;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(moe_gemm<MODE, NT, SAFE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_gemm<MODE, NT, SAFE>, 128, smem);
  return n;
}

#define DISPATCH(FN, ...)                                                        \
  if (mode == 1) {                                                               \
    if (safe) { if (nt == 1) FN<1, 1, true>(__VA_ARGS__); else if (nt == 2) FN<1, 2, true>(__VA_ARGS__); else FN<1, 4, true>(__VA_ARGS__); } \
    else { if (nt == 1) FN<1, 1, false>(__VA_ARGS__); else if (nt == 2) FN<1, 2, false>(__VA_ARGS__); else FN<1, 4, false>(__VA_ARGS__); } \
  } else {                                                                       \
    if (safe) { if (nt == 1) FN<2, 1, true>(__VA_ARGS__); else if (nt == 2) FN<2, 2, true>(__VA_ARGS__); else FN<2, 4, true>(__VA_ARGS__); } \
    else { if (nt == 1) FN<2, 1, false>(__VA_ARGS__); else if (nt == 2) FN<2, 2, false>(__VA_ARGS__); else FN<2, 4, false>(__VA_ARGS__); } \
  }

void gemm(int64_t mode, int64_t nt, bool safe, torch::Tensor act, torch::Tensor ids, int64_t M, int64_t ls,
          torch::Tensor wpk, torch::Tensor gsc, torch::Tensor normw, torch::Tensor outp, torch::Tensor ss,
          int64_t grid) {
  DISPATCH(launch_t, act, ids, (int)M, (int)ls, wpk, gsc, normw, outp, ss, (int)grid)
}

int64_t occupancy(int64_t mode, int64_t nt, bool safe, int64_t M) {
  int n = 0;
  if (mode == 1) {
    if (safe) { n = nt == 1 ? occ_t<1, 1, true>(M) : nt == 2 ? occ_t<1, 2, true>(M) : occ_t<1, 4, true>(M); }
    else { n = nt == 1 ? occ_t<1, 1, false>(M) : nt == 2 ? occ_t<1, 2, false>(M) : occ_t<1, 4, false>(M); }
  } else {
    if (safe) { n = nt == 1 ? occ_t<2, 1, true>(M) : nt == 2 ? occ_t<2, 2, true>(M) : occ_t<2, 4, true>(M); }
    else { n = nt == 1 ? occ_t<2, 1, false>(M) : nt == 2 ? occ_t<2, 2, false>(M) : occ_t<2, 4, false>(M); }
  }
  return n;
}

std::vector<int64_t> kinfo(int64_t mode, int64_t nt, bool safe) {
  cudaFuncAttributes a;
  if (mode == 1) {
    if (safe) { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<1, 1, true>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<1, 2, true>); else cudaFuncGetAttributes(&a, moe_gemm<1, 4, true>); }
    else { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<1, 1, false>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<1, 2, false>); else cudaFuncGetAttributes(&a, moe_gemm<1, 4, false>); }
  } else {
    if (safe) { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<2, 1, true>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<2, 2, true>); else cudaFuncGetAttributes(&a, moe_gemm<2, 4, true>); }
    else { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<2, 1, false>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<2, 2, false>); else cudaFuncGetAttributes(&a, moe_gemm<2, 4, false>); }
  }
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes};
}

void finalize(torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ids, torch::Tensor tw, int64_t ls, torch::Tensor out) {
  const int M = out.size(0);
  moe_finalize<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(ybuf.data_ptr()), ss.data_ptr<float>(), ids.data_ptr<int>(),
      tw.data_ptr<float>(), (int)ls, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
}
"""

FAST_MOE_CPP_SRC = r"""
void gemm(int64_t mode, int64_t nt, bool safe, torch::Tensor act, torch::Tensor ids, int64_t M, int64_t ls,
          torch::Tensor wpk, torch::Tensor gsc, torch::Tensor normw, torch::Tensor outp, torch::Tensor ss,
          int64_t grid);
int64_t occupancy(int64_t mode, int64_t nt, bool safe, int64_t M);
std::vector<int64_t> kinfo(int64_t mode, int64_t nt, bool safe);
void finalize(torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ids, torch::Tensor tw, int64_t ls, torch::Tensor out);
"""

_FAST_MOE_EXT = None


def _fast_moe_ext():
    global _FAST_MOE_EXT
    if _FAST_MOE_EXT is None:
        _FAST_MOE_EXT = load_inline(
            name="muse_moe_w4a16_v9",
            cpp_sources=FAST_MOE_CPP_SRC,
            cuda_sources=FAST_MOE_CUDA_SRC,
            functions=["gemm", "occupancy", "kinfo", "finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    return _FAST_MOE_EXT


# ---------------------------------------------------------------- packing (_fast_moe_prepare only)
_FAST_MOE_FA = (7, 8, 14, 15)
# Output registers of a pair of k16 sub-steps: (word 0=A / 1=B / 2=V, rotate-left amount).
_FAST_MOE_SLOTS = [(0, 6), (0, 2), (0, 12), (2, 14), (1, 4), (1, 0), (1, 10), (2, 8)]
_FAST_MOE_TGT = (6, 7, 8, 15)  # e2m1 bit (m, e0, e1, sign) -> bit of the bf16 half


def _fast_moe_bitmap():
    out, used = [], set()
    for idx, (w, L) in enumerate(_FAST_MOE_SLOTS):
        s_local, reg = divmod(idx, 4)
        for half in range(2):
            for b in range(4):
                src = (16 * half + _FAST_MOE_TGT[b] - L) % 32
                ww = w if w != 2 else (0 if (src % 16) in _FAST_MOE_FA else 1)
                assert (ww, src) not in used
                used.add((ww, src))
                out.append((s_local, reg, half, b, ww, src))
    assert len(used) == 64
    return out


def _fast_moe_pack_gemm(wq, wsf, row_idx):
    """wq uint8 [E, N, K/2]; wsf uint8 [E, N, K/16]; row_idx long [ITEMS, 2, 16] -> uint8 [E, ITEMS, K/64, 1152]."""
    E, N, Kh = wq.shape
    K = Kh * 2
    KS = K // 64
    ITEMS = row_idx.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), dim=-1).reshape(E, N, K)
    c = codes[:, row_idx.reshape(-1)].reshape(E, ITEMS, 2, 2, 8, KS, 2, 2, 4, 4)
    c = c.permute(0, 1, 5, 2, 4, 8, 6, 7, 3, 9).contiguous().to(torch.int32)
    words = torch.zeros((E, ITEMS, KS, 2, 8, 4, 2, 2), dtype=torch.int32, device=wq.device)
    for (sl, reg, half, b, ww, src) in _fast_moe_bitmap():
        rh = reg % 2
        q = (reg // 2) * 2 + half
        words[..., ww] |= ((c[..., sl, rh, q] >> b) & 1) << src
    wbytes = words.reshape(E, ITEMS, KS, 2, 32, 4).contiguous().view(torch.uint8).reshape(E, ITEMS, KS, 1024)
    s = wsf[:, row_idx.reshape(-1)].reshape(E, ITEMS, 2, 2, 8, KS, 4)
    s = s.permute(0, 1, 5, 4, 2, 3, 6).contiguous().reshape(E, ITEMS, KS, 128)
    return torch.cat((wbytes, s), dim=-1).contiguous()


def _fast_moe_pack_all(wq, wsf, row_idx, chunk=4):
    parts = []
    for e0 in range(0, wq.shape[0], chunk):
        parts.append(_fast_moe_pack_gemm(wq[e0:e0 + chunk], wsf[e0:e0 + chunk], row_idx))
    return torch.cat(parts, dim=0).contiguous()


def _fast_moe_needs_safe(sf_u8):
    exp = (sf_u8 >> 3) & 0xF
    return bool(((exp == 0) | (sf_u8 >= 0x7F)).any().item())


def _fast_moe_prepare(weights):
    dev = weights["w13"].device
    ext = _fast_moe_ext()
    sl = slice(0, LOCAL)
    w13 = weights["w13"][sl]
    sf13 = weights["w13_sf"][sl].contiguous().view(torch.uint8)
    w2 = weights["w2"][sl]
    sf2 = weights["w2_sf"][sl].contiguous().view(torch.uint8)
    i = torch.arange(I_DIM // 16, device=dev)
    base = i[:, None] * 16 + torch.arange(16, device=dev)[None, :]
    rows1 = torch.stack((base, base + I_DIM), dim=1)
    i2 = torch.arange(H // 32, device=dev)
    rows2 = (i2[:, None] * 32 + torch.arange(32, device=dev)[None, :]).reshape(-1, 2, 16)
    pk1 = _fast_moe_pack_all(w13, sf13, rows1)
    pk2 = _fast_moe_pack_all(w2, sf2, rows2)
    safe = _fast_moe_needs_safe(sf13) or _fast_moe_needs_safe(sf2)
    state = {
        "pk1": pk1, "pk2": pk2, "safe": safe,
        "gsc1": (weights["w13_gs"][sl].float() * 128.0).contiguous(),
        "gsc2": (weights["w2_gs"][sl].float() * 128.0).contiguous(),
        "normw": (weights["post_norm_w"].float() + 1.0).contiguous(),
        "nsm": torch.cuda.get_device_properties(dev).multi_processor_count,
        "occ": {},
    }
    torch.cuda.synchronize()
    print(f"[prep] safe={safe} pk1={tuple(pk1.shape)} pk2={tuple(pk2.shape)}")
    if FAST_MOE_DEBUG:
        for mode in (1, 2):
            for nt in (1, 2, 4):
                print(f"[kinfo] mode={mode} nt={nt} regs/local={list(ext.kinfo(mode, nt, safe))} "
                      f"occ(M=256)={ext.occupancy(mode, nt, safe, 256)}")
        _fast_moe_selftest(state, weights)
    return state


def _fast_moe_nt_for(M):
    # A single token-thread already wins at T=8/16. Test that occupancy point at
    # T=32 independently of the known-losing two-thread standalone measurement.
    return 1 if M <= 32 else 2


def _fast_moe_grid(state, mode, nt, M):
    key = (mode, nt, M)
    g = state["occ"].get(key)
    if g is None:
        occ = max(1, int(_fast_moe_ext().occupancy(mode, nt, state["safe"], M)))
        tiles = 256 if mode == 1 else 128
        g = occ * state["nsm"]
        # Two queued waves shorten each CTA's persistent item loop. This wins
        # at T=32; a measured T=128/256 expansion regressed and was reverted.
        if M == 32:
            # Two waves are optimal among 1x/2x/3x at T=32. The same choice
            # regressed at T=16 and T>=128, so keep this shape-specific.
            g *= 2
        state["occ"][key] = g
    return g


FAST_MOE_DEBUG = False


def _fast_moe_route(bs, seed, dev):
    """Diagnostics only: synthetic skewed routing (Student-t expert popularity), GPU 0 busiest."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, H), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    v, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    cfg = {"bs": bs, "d_hidden": H, "d_expert": I_DIM, "n_experts": 256, "local_start": 0,
           "local_count": 32, "top_k": 8, "sharding": "ep8"}
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (v / v.sum(-1, keepdim=True)).float(), "config": cfg}


def _fast_moe_torch_ref(data, weights):
    from nvfp4 import dequantize
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    out = torch.zeros_like(x)
    w = weights["post_norm_w"].float()
    for e in torch.unique(ids).tolist():
        if e >= LOCAL:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        it = torch.nn.functional.silu(h[:, :I_DIM]) * h[:, I_DIM:]
        y = (it @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T) * (1 + w)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out


def _fast_moe_selftest(state, weights):
    dev = weights["w13"].device
    for bs in (1, 33, 257):
        d = _fast_moe_route(bs, 100 + bs, dev)
        o = _fast_moe_forward(d, state).float()
        r = _fast_moe_torch_ref(d, weights)
        err = ((o - r).norm() / r.norm()).item()
        rows = ((o - r).norm(dim=-1) / torch.maximum(r.norm(dim=-1), 0.1 * r.norm(dim=-1).mean())).max().item()
        print(f"[selftest] bs={bs} rel_l2={err:.3e} worst_row={rows:.3e}")
    flush = torch.empty((96 * 1024 * 1024,), dtype=torch.int32, device=dev)
    for bs in (1, 16, 32, 8, 128, 256):
        d = _fast_moe_route(bs, 7 + bs, dev)
        ids = d["topk_ids"]
        loc = ids[ids < 32].long()
        cnt = torch.bincount(loc, minlength=32)
        _fast_moe_forward(d, state)
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            _fast_moe_forward(d, state)
        ts = []
        for _ in range(30):
            flush.add_(1)
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            gph.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        ts.sort()
        touched = int((cnt > 0).sum())
        floor = touched * 3 * 4096 * 4096 * 0.5625 / 4.8e12 * 1e6
        print(f"[bench] bs={bs} med={ts[len(ts)//2]:.1f}us min={ts[0]:.1f}us touched={touched} "
              f"pairs={int(cnt.sum())} maxcnt={int(cnt.max())} bwfloor={floor:.1f}us")
    del flush


def _fast_moe_forward(data, state):
    cfg = data["config"]
    ext = _fast_moe_ext()
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    tw = data["topk_weights"].contiguous()
    if tw.dtype != torch.float32:
        tw = tw.float()
    M = x.shape[0]
    ls = int(cfg["local_start"])
    assert ls == 0 and int(cfg["local_count"]) == LOCAL
    nt = _fast_moe_nt_for(M)
    dev = x.device
    inter = torch.empty((M * TOP_K, I_DIM), dtype=torch.bfloat16, device=dev)
    ybuf = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
    ss = torch.empty((M * TOP_K,), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    safe = state["safe"]
    ext.gemm(1, nt, safe, x, ids, M, ls, state["pk1"], state["gsc1"], state["normw"], inter, ss,
             _fast_moe_grid(state, 1, nt, M))
    ext.gemm(2, nt, safe, inter, ids, M, ls, state["pk2"], state["gsc2"], state["normw"], ybuf, ss,
             _fast_moe_grid(state, 2, nt, M))
    ext.finalize(ybuf, ss, ids, tw, ls, out)
    return out


def _selftest_fast_moe(weights, state):
    """Check the embedded fused expert path against the selected Marlin path."""
    dev = weights["w13"].device
    gen = torch.Generator(device=dev).manual_seed(6129)
    tokens = 33
    x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
    ls = torch.tensor([64], dtype=torch.int32, device=dev)
    ids = torch.randint(0, N_EXPERTS, (tokens, TOP_K), generator=gen, device=dev, dtype=torch.int32)
    ids[:, 0] = torch.arange(tokens, device=dev, dtype=torch.int32) % LOCAL_EXPERTS + ls
    wts = torch.rand((tokens, TOP_K), generator=gen, device=dev) + 0.1
    wts /= wts.sum(-1, keepdim=True)
    local_ids = _local_ids(ids, ls)
    bm, atomic = state["marlin_cfg"].get(tokens, (_default_bm(tokens), False))
    ref = _experts_cfg(x, wts, local_ids, state, bm, atomic)
    out = _fast_moe_forward({
        "hidden_states": x,
        "topk_ids": local_ids,
        "topk_weights": wts,
        "config": {"local_start": 0, "local_count": LOCAL_EXPERTS},
    }, state["fast_moe"])
    err = _rel(out, ref)
    print(f"selftest fast_moe T={tokens}: rel={err:.3e}")
    return err < 2e-2


def prepare(weights):
    state = _decode_prepare_v6_base(weights)
    state["use_fast_moe"] = False
    if not state["use_local_kernel"]:
        return state
    try:
        moe_weights = dict(weights)
        moe_weights["post_norm_w"] = weights["post_expert_norm"]
        state["fast_moe"] = _fast_moe_prepare(moe_weights)
        state["use_fast_moe"] = _selftest_fast_moe(weights, state)
    except Exception:
        print("fast_moe FAILED\n" + traceback.format_exc()[-3000:])
    torch.cuda.synchronize()
    return state
