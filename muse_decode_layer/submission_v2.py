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
# Each custom piece is self-tested in prepare() against a torch reference; if a test
# fails (or the kernel does not compile) the layer falls back to SGLang's kernel.
# =====================================================================================
import math
import time
import traceback

import triton
import triton.language as tl

TWO63 = tl.constexpr(9.223372036854775808e18)
LOG2E = 1.4426950408889634

# E2M1 -> BF16 for 8 nibbles of one 32-bit word (nibble j = bits 4j..4j+3 = element j).
# Nibble j goes to bits 12..15 of the low half, nibble j+4 to bits 12..15 of the high half;
# the sign stays at bit 15 and the 3 magnitude bits move to bits 6..8, which makes the BF16
# value e2m1 * 2^-126 exactly (code 1 is a BF16 subnormal). One bf16x2 multiply by
# (block scale * 2^63) then gives e2m1 * scale * 2^-63, exact and normal; the FP32
# epilogue multiplies by global scale * 2^63.
_DEQ_ASM = """
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
"""


@triton.jit
def _deq8(words, sc):
    """words int32 [BN, W], sc bf16 [BN, W] -> bf16 [BN, W, 2, 2, 2] (flattens to K order)."""
    o0, o1, o2, o3, o4, o5, o6, o7 = tl.inline_asm_elementwise(
        asm=_DEQ_ASM,
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
def _moe_gemm_kernel(a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr, sorted_ptr, bexp_ptr, R,
                     N: tl.constexpr, K: tl.constexpr, WROWS: tl.constexpr, A_DIV: tl.constexpr,
                     GATED: tl.constexpr, NBM: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Grouped W4A16 (NVFP4) GEMM, swap-AB: acc[n, m] = sum_k W[e, n, k] * A[row(m), k].
    GATED: W rows [0, N) are the gate and [N, 2N) the up projection; out = silu(g) * u (BF16).
    Else out = acc (FP32)."""
    pid = tl.program_id(0)
    pid_m = pid % NBM
    pid_n = pid // NBM
    e = tl.load(bexp_ptr + pid_m)
    if e < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    rows = tl.load(sorted_ptr + offs_m)
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
        wt = tl.reshape(_deq8(words, sc), (BN, BK))
        acc = tl.dot(wt, at, acc)
        if GATED:
            words2 = tl.load(w_base + UPW + k0 // 8)
            sc2 = (tl.load(s_base + UPS + k0 // 16).to(tl.float32) * TWO63).to(tl.bfloat16)
            wt2 = tl.reshape(_deq8(words2, sc2), (BN, BK))
            acc2 = tl.dot(wt2, at, acc2)
    # The dequantized tile was e2m1 * sf * 2^-63: the global scale and 2^63 remain.
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


def _moe_bm(tokens):
    if tokens <= 32:
        return 16
    if tokens <= 128:
        return 32
    return 64


def _moe_forward(x, topk_weights, topk_ids, local_start, state, cfg=None):
    """x [T, 4096] bf16 (pre-expert-normed), topk ids/weights [T, 8] -> [T, 4096] bf16."""
    m = state["moe"]
    tokens = x.shape[0]
    R = tokens * TOP_K
    if cfg is None:
        cfg = state["moe_cfg"].get(tokens) or (_moe_bm(tokens), 64, 3)
    BM, BN, STAGES = cfg
    BK = 128
    NBM = -(-(R + LOCAL_EXPERTS * (BM - 1)) // BM)
    dev = x.device
    sorted_ids = torch.empty(NBM * BM, dtype=torch.int32, device=dev)
    bexp = torch.empty(NBM, dtype=torch.int32, device=dev)
    ids = topk_ids.to(torch.int32).contiguous()
    _moe_align_kernel[(1,)](ids, local_start, sorted_ids, bexp, R=R, R_P2=_np2(R), MAXP=NBM * BM,
                            MAXP_P2=_np2(NBM * BM), NBM=NBM, NBM_P2=_np2(NBM), BM=BM, num_warps=4)
    inter = torch.empty((R, D_EXPERT), dtype=torch.bfloat16, device=dev)
    grid = (NBM * (D_EXPERT // BN),)
    _moe_gemm_kernel[grid](x, m["w13"], m["sf13"], m["gs13"], inter, sorted_ids, bexp, R,
                           N=D_EXPERT, K=D_EXPERT, WROWS=2 * D_EXPERT, A_DIV=TOP_K, GATED=True, NBM=NBM,
                           BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=STAGES)
    y = torch.empty((R, D_EXPERT), dtype=torch.float32, device=dev)
    _moe_gemm_kernel[grid](inter, m["w2"], m["sf2"], m["gs2"], y, sorted_ids, bexp, R,
                           N=D_EXPERT, K=D_EXPERT, WROWS=D_EXPERT, A_DIV=1, GATED=False, NBM=NBM,
                           BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=STAGES)
    out = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _moe_finalize_kernel[(tokens,)](y, ids, topk_weights.float().contiguous(), local_start, m["post_w"], out,
                                    D=D_EXPERT, num_warps=8)
    return out


# ------------------------------- attention -------------------------------------------

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
    """Merge the splits of one (token, query head), then the per-head RMSNorm x sigmoid(gate)."""
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


# ------------------------------- self tests ------------------------------------------

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
        out = _moe_forward(x, wts, ids, local_start, state, c)
        errs[c] = _rel(out, ref)
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


ATTN_CANDS = (("mine", 64, 264), ("mine", 128, 528), ("mine", 64, 1056), ("mine", 128, 1056))


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


# ------------------------------- prepare-time autotuning -----------------------------

def _bench_graph(fn, flush, reps=15):
    """Median µs of one CUDA-graph replay of fn, with a cold L2 before each replay."""
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
                res = {}
                res[("fa3",)] = _bench_graph(lambda: _fa3_forward(
                    q, kc, vc, pt, cs, cu, one, one, gate, batch, length, window, scale, False, None, state), flush)
                for c in ok_cands:
                    res[c] = _bench_graph(lambda: _attention_forward(
                        q, kc, vc, pt, cs, one, one, gate, batch, length, window, scale, bn=c[1], target=c[2]),
                        flush)
                best = min(res, key=res.get)
                state["attn_cfg"][(batch, length, window > 0)] = best
                print(f"attn B={batch} L={length} {'sliding' if window else 'full'}: " +
                      " ".join(f"{'/'.join(map(str, k))}={v:.1f}" for k, v in res.items()) + f" -> {best}")
    del kc, vc


def _autotune_moe(state, dev, flush, ok_cfgs):
    gen = torch.Generator(device=dev).manual_seed(9)
    # Skewed routing: Zipf-like expert popularity, as "a few experts take most tokens".
    rank_p = 1.0 / torch.arange(1, N_EXPERTS + 1, device=dev).float() ** 1.1
    pop = rank_p[torch.randperm(N_EXPERTS, generator=gen, device=dev)]
    for tokens in (1, 8, 16, 32, 128, 256):
        x = torch.randn((tokens, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        ids = torch.multinomial(pop.expand(tokens, N_EXPERTS), TOP_K, replacement=False, generator=gen).to(torch.int32)
        counts = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
        ls = LOCAL_EXPERTS * int(counts.argmax())
        local_start = torch.tensor([ls], dtype=torch.int32, device=dev)
        wts = torch.rand((tokens, TOP_K), generator=gen, device=dev) + 0.1
        wts = wts / wts.sum(-1, keepdim=True)
        loc = (ids.long() - ls) % N_EXPERTS
        nloc = int((loc < LOCAL_EXPERTS).sum())
        ndist = int(torch.unique(loc[loc < LOCAL_EXPERTS]).numel())
        res = {}
        res["marlin"] = _bench_graph(lambda: _experts(x, wts, (ids - local_start) % N_EXPERTS, state), flush)
        for c in ok_cfgs:
            if c[0] > 16 and tokens < 16:
                continue
            res[c] = _bench_graph(lambda: _moe_forward(x, wts, ids, local_start, state, c), flush)
        best = min(res, key=res.get)
        state["moe_cfg"][tokens] = best
        print(f"moe T={tokens} routes={nloc} experts={ndist}: " +
              " ".join(f"{k if isinstance(k, str) else '/'.join(map(str, k))}={v:.1f}" for k, v in res.items())
              + f" -> {best}")


MOE_CANDS = ((16, 64, 3), (16, 64, 4), (32, 64, 3), (64, 64, 3), (64, 128, 3))


def prepare(weights):
    state = _base_prepare(weights)
    dev = weights["qkv_gate"].device
    state["moe"] = {
        "w13": weights["w13"].contiguous().view(torch.int32),
        "sf13": weights["w13_sf"].contiguous(),
        "gs13": weights["w13_gs"].float().contiguous(),
        "w2": weights["w2"].contiguous().view(torch.int32),
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
        print("selftest moe:", {str(c): f"{e:.2e}" for c, e in errs.items()})
    except Exception:
        print("selftest moe FAILED\n" + traceback.format_exc()[-3000:])
    state["use_moe"] = bool(ok_moe)
    ok_attn = []
    try:
        errs = _selftest_attn(dev)
        ok_attn = [c for c, e in errs.items() if e < 2e-2]
        print("selftest attn:", {str(c): f"{e:.2e}" for c, e in errs.items()})
    except Exception:
        print("selftest attn FAILED\n" + traceback.format_exc()[-3000:])
    state["use_attn"] = bool(ok_attn)
    flush = torch.empty(96 * 1024 * 1024, dtype=torch.float32, device=dev)
    if ok_attn:
        try:
            _autotune_attention(state, dev, flush, ok_attn)
        except Exception:
            print("autotune attn FAILED\n" + traceback.format_exc()[-3000:])
    if ok_moe:
        try:
            _autotune_moe(state, dev, flush, ok_moe)
        except Exception:
            print("autotune moe FAILED\n" + traceback.format_exc()[-3000:])
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
    moe_cfg = state["moe_cfg"].get(tokens) if state["use_moe"] else "marlin"
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
