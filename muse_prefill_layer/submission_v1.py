# okc: problem=muse_prefill_layer
"""Muse Spark Open prefill layer on one H200 (TP8/EP8 slice).

Starts from SGLang's composition (baseline.py) and replaces the routed experts
(moe_wna16_marlin_gemm x2 + silu_and_mul) with two persistent Triton grouped GEMMs that
dequantize NVFP4 in registers:

  * prepare() repacks each expert's E2M1 codes (lossless bit permutation, still 4 bits per
    weight, same E4M3 block scales and FP32 global scale) so that one 32-bit word decodes into
    four bf16x2 registers with a handful of and/shift ops plus one mul.bf16x2 that applies the
    block scale (e2m1 * e4m3 is exact in bf16).
  * swap-AB: weights are the MMA's M side (64-row WGMMA), tokens the N side, so experts with
    few tokens waste little.
  * GEMM1 computes gate and up tiles together and applies SiLU(gate) * up in the epilogue.
  * The FP32 global scale (and the 2^9 used to keep the decode exact) is applied in FP32 on the
    accumulator.

Everything else is SGLang's modules. prepare() self-tests the custom MoE against the Marlin
path on random inputs and falls back to Marlin if it disagrees.
"""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0

# ----------------------------------------------------------------------------------------------
# NVFP4 grouped GEMM (Triton)
# ----------------------------------------------------------------------------------------------
REPACK_BK = 128          # K elements per repacked block: bytes j hold elements j and j + 64
SCALE_BIAS = float(2.0 ** 117)   # block scale * 2^117 (bf16); decoded e2m1 is value * 2^-126
EPI_SCALE = 512.0        # 2^9 = 2^126 / 2^117, applied with the global scale in FP32



@triton.jit
def _deq(x, sl, sh):
    return tl.inline_asm_elementwise(
        asm="""
{
.reg .b32 t0, t1, t2, t3, t4;
and.b32 t0, $4, 0x81C081C0;
mul.rn.bf16x2 $0, t0, $5;
shl.b32 t1, $4, 3;
and.b32 t1, t1, 0x81C081C0;
mul.rn.bf16x2 $1, t1, $6;
shl.b32 t2, $4, 6;
and.b32 t2, t2, 0x81C081C0;
mul.rn.bf16x2 $2, t2, $7;
shl.b32 t3, $4, 1;
and.b32 t3, t3, 0x80008000;
shr.b32 t4, $4, 3;
and.b32 t4, t4, 0x01800180;
or.b32 t3, t3, t4;
shr.b32 t4, $4, 7;
and.b32 t4, t4, 0x00400040;
or.b32 t3, t3, t4;
mul.rn.bf16x2 $3, t3, $8;
}
""", constraints="=r,=r,=r,=r,r,r,r,r,r", args=[x, sl, sh],
        dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)


@triton.jit
def _load_w(wp, sp, BN: tl.constexpr, G: tl.constexpr, BK2: tl.constexpr, SB: tl.constexpr):
    x = tl.load(wp)                       # uint8 [BN, G, 16]
    s_lo = tl.load(sp).to(tl.float32)     # [BN, G] e4m3 -> fp32
    s_hi = tl.load(sp + G).to(tl.float32)
    sl = (s_lo * SB).to(tl.bfloat16)
    sh = (s_hi * SB).to(tl.bfloat16)
    sl = tl.broadcast_to(sl[:, :, None], (BN, G, 16))
    sh = tl.broadcast_to(sh[:, :, None], (BN, G, 16))
    w0, w1 = _deq(x, sl, sh)
    return tl.reshape(w0, (BN, BK2)), tl.reshape(w1, (BN, BK2))


@triton.jit
def _tile_coords(tile, num_lm, num_n, GROUP_M: tl.constexpr):
    width = GROUP_M * num_n
    first_m = (tile // width) * GROUP_M
    gsz = tl.minimum(num_lm - first_m, GROUP_M)
    r = tile % width
    return first_m + r % gsz, r // gsz


@triton.jit
def _moe_gemm1(A, W, S, GS, C, sorted_ids, expert_ids, npost_ptr, num_valid,
               K: tl.constexpr, NH: tl.constexpr, TOPK: tl.constexpr, NLOCAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
               MAXMB: tl.constexpr, NUM_PROGS: tl.constexpr, SB: tl.constexpr, EPI: tl.constexpr):
    """inter[r, n] = silu(x W_gate^T) * (x W_up^T) for each routed row r (token*TOPK+slot)."""
    pid = tl.program_id(0)
    BK2: tl.constexpr = BK // 2
    G: tl.constexpr = BK2 // 16
    KS: tl.constexpr = K // 16
    KB: tl.constexpr = K // 2
    num_m = tl.load(npost_ptr) // BM
    mb = tl.arange(0, MAXMB)
    eids = tl.load(expert_ids + mb, mask=mb < num_m, other=NLOCAL)
    num_lm = tl.sum((eids < NLOCAL).to(tl.int32), axis=0)
    num_n: tl.constexpr = NH // BN
    total = num_lm * num_n
    offs_n = tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_j = tl.arange(0, BK2)
    offs_g = tl.arange(0, G)
    offs_i = tl.arange(0, 16)
    for tile in range(pid, total, NUM_PROGS):
        pid_m, pid_n = _tile_coords(tile, num_lm, num_n, GROUP_M)
        e = tl.load(expert_ids + pid_m).to(tl.int64)
        rows = tl.load(sorted_ids + pid_m * BM + offs_m)
        rmask = rows < num_valid
        tok = tl.where(rmask, rows // TOPK, 0)
        a_ptr = A + tok[:, None].to(tl.int64) * K + offs_j[None, :]
        n_g = pid_n * BN + offs_n
        wg = W + e * (2 * NH * KB) + n_g[:, None, None] * KB + offs_g[None, :, None] * 16 + offs_i[None, None, :]
        wu = wg + NH * KB
        sg = S + e * (2 * NH * KS) + n_g[:, None] * KS + offs_g[None, :]
        su = sg + NH * KS
        acc_g = tl.zeros((BN, BM), dtype=tl.float32)
        acc_u = tl.zeros((BN, BM), dtype=tl.float32)
        for kb in range(0, K // BK):
            a0 = tl.load(a_ptr + kb * BK, mask=rmask[:, None], other=0.0)
            a1 = tl.load(a_ptr + kb * BK + BK2, mask=rmask[:, None], other=0.0)
            g0, g1 = _load_w(wg + kb * BK2, sg + kb * (BK // 16), BN, G, BK2, SB)
            u0, u1 = _load_w(wu + kb * BK2, su + kb * (BK // 16), BN, G, BK2, SB)
            a0t = tl.trans(a0)
            a1t = tl.trans(a1)
            acc_g = tl.dot(g0, a0t, acc_g)
            acc_g = tl.dot(g1, a1t, acc_g)
            acc_u = tl.dot(u0, a0t, acc_u)
            acc_u = tl.dot(u1, a1t, acc_u)
        gs = tl.load(GS + e) * EPI
        g = acc_g * gs
        u = acc_u * gs
        h = g * tl.sigmoid(g) * u
        cp = C + rows[None, :].to(tl.int64) * NH + n_g[:, None]
        tl.store(cp, h.to(tl.bfloat16), mask=rmask[None, :])


@triton.jit
def _moe_gemm2(A, W, S, GS, C, sorted_ids, expert_ids, npost_ptr, num_valid,
               K: tl.constexpr, N: tl.constexpr, NLOCAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
               MAXMB: tl.constexpr, NUM_PROGS: tl.constexpr, SB: tl.constexpr, EPI: tl.constexpr):
    """y[r, n] = inter[r] W2^T for each routed row r."""
    pid = tl.program_id(0)
    BK2: tl.constexpr = BK // 2
    G: tl.constexpr = BK2 // 16
    KS: tl.constexpr = K // 16
    KB: tl.constexpr = K // 2
    num_m = tl.load(npost_ptr) // BM
    mb = tl.arange(0, MAXMB)
    eids = tl.load(expert_ids + mb, mask=mb < num_m, other=NLOCAL)
    num_lm = tl.sum((eids < NLOCAL).to(tl.int32), axis=0)
    num_n: tl.constexpr = N // BN
    total = num_lm * num_n
    offs_n = tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_j = tl.arange(0, BK2)
    offs_g = tl.arange(0, G)
    offs_i = tl.arange(0, 16)
    for tile in range(pid, total, NUM_PROGS):
        pid_m, pid_n = _tile_coords(tile, num_lm, num_n, GROUP_M)
        e = tl.load(expert_ids + pid_m).to(tl.int64)
        rows = tl.load(sorted_ids + pid_m * BM + offs_m)
        rmask = rows < num_valid
        r_safe = tl.where(rmask, rows, 0)
        a_ptr = A + r_safe[:, None].to(tl.int64) * K + offs_j[None, :]
        n_g = pid_n * BN + offs_n
        wp = W + e * (N * KB) + n_g[:, None, None] * KB + offs_g[None, :, None] * 16 + offs_i[None, None, :]
        sp = S + e * (N * KS) + n_g[:, None] * KS + offs_g[None, :]
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for kb in range(0, K // BK):
            a0 = tl.load(a_ptr + kb * BK, mask=rmask[:, None], other=0.0)
            a1 = tl.load(a_ptr + kb * BK + BK2, mask=rmask[:, None], other=0.0)
            w0, w1 = _load_w(wp + kb * BK2, sp + kb * (BK // 16), BN, G, BK2, SB)
            acc = tl.dot(w0, tl.trans(a0), acc)
            acc = tl.dot(w1, tl.trans(a1), acc)
        y = acc * (tl.load(GS + e) * EPI)
        cp = C + rows[None, :].to(tl.int64) * N + n_g[:, None]
        tl.store(cp, y.to(tl.bfloat16), mask=rmask[None, :])


def _repack_nvfp4(packed, bk=REPACK_BK):
    """Lossless bit permutation of standard NVFP4 packing (element 2i low nibble, 2i+1 high)
    into the layout _DEQ_ASM decodes. Still 4 bits per weight; shape and dtype unchanged."""
    out = torch.empty_like(packed)
    n, kh = packed.shape[-2:]
    k = kh * 2
    for e in range(packed.shape[0]):
        p = packed[e]
        lo = (p & 0xF).to(torch.int64)
        hi = (p >> 4).to(torch.int64)
        codes = torch.stack((lo, hi), -1).reshape(n, k // bk, 2, bk // 8, 4)
        del lo, hi
        a = codes[:, :, 0]
        b = codes[:, :, 1]

        def s(c):
            return (c >> 3) & 1

        def m(c):
            return c & 7

        w = (s(a[..., 0]) << 15) | (m(a[..., 0]) << 6) | (s(a[..., 1]) << 31) | (m(a[..., 1]) << 22)
        w |= (s(a[..., 2]) << 12) | (m(a[..., 2]) << 3) | (s(a[..., 3]) << 28) | (m(a[..., 3]) << 19)
        w |= (s(b[..., 0]) << 9) | m(b[..., 0]) | (s(b[..., 1]) << 25) | (m(b[..., 1]) << 16)
        w |= (s(b[..., 2]) << 14) | ((m(b[..., 2]) >> 1) << 10) | ((m(b[..., 2]) & 1) << 13)
        w |= (s(b[..., 3]) << 30) | ((m(b[..., 3]) >> 1) << 26) | ((m(b[..., 3]) & 1) << 29)
        by = torch.stack([(w >> (8 * q)) & 0xFF for q in range(4)], -1).to(torch.uint8)
        out[e] = by.reshape(n, kh)
        del codes, w, by
    return out


def _moe_config(tokens):
    """(BM, gemm1 (BN, warps, stages), gemm2 (BN, warps, stages), progs per SM)."""
    if tokens >= 8192:
        return dict(BM=128, g1=(64, 8, 3), g2=(128, 8, 3), occ=1, group=8)
    return dict(BM=64, g1=(64, 4, 3), g2=(128, 4, 3), occ=2, group=8)


def _custom_experts(x, topk_weights, lid, slot_ids, state, cfg=None):
    """x [T, 4096] bf16 (pre-expert-normed), lid [T, 8] int32 experts relative to local_start
    (0..255, this GPU's are 0..31), slot_ids [T, 8] (-1 off this GPU). Returns [T, 4096] bf16."""
    moe = state["moe"]
    t = x.shape[0]
    cfg = cfg or _moe_config(t)
    bm = cfg["BM"]
    sorted_ids, expert_ids, npost = moe_align_block_size(lid, bm, N_EXPERTS)
    maxmb = triton.next_power_of_2(expert_ids.numel())
    num_valid = t * TOP_K
    inter = torch.empty((num_valid, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((num_valid, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    nsm = moe["num_sms"]
    bn1, w1, s1 = cfg["g1"]
    progs = nsm * cfg["occ"]
    _moe_gemm1[(progs,)](
        x, moe["w13"], moe["w13_sf"], moe["w13_gs"], inter, sorted_ids, expert_ids, npost, num_valid,
        K=D_EXPERT, NH=D_EXPERT, TOPK=TOP_K, NLOCAL=LOCAL_EXPERTS, BM=bm, BN=bn1, BK=REPACK_BK,
        GROUP_M=cfg["group"], MAXMB=maxmb, NUM_PROGS=progs, SB=SCALE_BIAS, EPI=EPI_SCALE,
        num_warps=w1, num_stages=s1)
    bn2, w2, s2 = cfg["g2"]
    _moe_gemm2[(progs,)](
        inter, moe["w2"], moe["w2_sf"], moe["w2_gs"], y, sorted_ids, expert_ids, npost, num_valid,
        K=D_EXPERT, N=D_EXPERT, NLOCAL=LOCAL_EXPERTS, BM=bm, BN=bn2, BK=REPACK_BK,
        GROUP_M=cfg["group"], MAXMB=maxmb, NUM_PROGS=progs, SB=SCALE_BIAS, EPI=EPI_SCALE,
        num_warps=w2, num_stages=s2)
    return fused_post_expert_norm_weighted_sum(
        y.view(t, TOP_K, D_EXPERT), topk_weights.to(torch.float32).contiguous(),
        norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=slot_ids)


# ----------------------------------------------------------------------------------------------
# SGLang modules (as baseline.py)
# ----------------------------------------------------------------------------------------------
def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
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


def _marlin_experts(x, topk_weights, topk_ids, state):
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


def _self_test(state, device):
    """Custom MoE vs the Marlin path on random routed inputs, for both tile configs."""
    gen = torch.Generator(device=device).manual_seed(1234)
    ok = True
    for t, cfg in ((512, _moe_config(512)), (1024, _moe_config(1 << 14))):
        x = torch.randn((t, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        scores = torch.rand((t, N_EXPERTS), generator=gen, device=device)
        scores[:, :LOCAL_EXPERTS] += 0.5   # plenty of local routes
        w, ids = scores.topk(TOP_K, dim=-1)
        ids = ids.to(torch.int32)
        w = (w / w.sum(-1, keepdim=True)).float()
        slot_ids = state["bank"]["expert_map"][ids].contiguous()
        ref = _marlin_experts(x, w, ids, state).float()
        got = _custom_experts(x, w, ids, slot_ids, state, cfg).float()
        torch.cuda.synchronize()
        err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
        finite = bool(torch.isfinite(got).all())
        print(f"[self-test] T={t} BM={cfg['BM']} rel_l2 vs marlin = {err:.3e} finite={finite}", flush=True)
        ok = ok and finite and err < 2e-2
    return ok


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
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
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    state["moe"] = {
        "w13": _repack_nvfp4(weights["w13"].contiguous()),
        "w13_sf": weights["w13_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].float().contiguous(),
        "w2": _repack_nvfp4(weights["w2"].contiguous()),
        "w2_sf": weights["w2_sf"].contiguous(),
        "w2_gs": weights["w2_gs"].float().contiguous(),
        "num_sms": torch.cuda.get_device_properties(device).multi_processor_count,
    }
    torch.cuda.synchronize()
    try:
        state["use_custom"] = _self_test(state, device)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[self-test] custom MoE failed: {exc!r}", flush=True)
        state["use_custom"] = False
    print(f"[prepare] use_custom={state['use_custom']}", flush=True)
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
    lid = (topk_ids - data["local_start"]) % N_EXPERTS
    x_e = state["pre_expert_norm"](expert_input)
    if state["use_custom"]:
        slot_ids = state["bank"]["expert_map"][lid].contiguous()
        expert_output = _custom_experts(x_e, topk_weights, lid.to(torch.int32).contiguous(), slot_ids, state)
    else:
        expert_output = _marlin_experts(x_e, topk_weights, lid, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
