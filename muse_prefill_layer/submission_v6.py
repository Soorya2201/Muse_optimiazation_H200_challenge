# okc: problem=muse_prefill_layer
"""Muse Spark Open prefill layer on one H200 (TP8/EP8 slice).

SGLang's composition (baseline.py), with the routed experts (two moe_wna16_marlin_gemm calls
plus silu_and_mul) replaced by two persistent Triton grouped GEMMs that dequantize NVFP4 in
registers:

  * prepare() permutes the bits of each 32-bit word of E2M1 codes (lossless; still 4 bits per
    weight, same E4M3 block scales and FP32 global scales). Each word holds 8 consecutive K
    elements. An inline-PTX decoder turns a word into eight bf16 values with and/shift ops
    and one mul.bf16x2 per pair, which applies the block scale. e2m1 * e4m3 is exact in bf16.
  * tl.join + reshape put the eight values back in K order. The repack is laid out so this
    order comes out natural, so one wgmma-sized tl.dot covers each 128-wide K block.
  * GEMM1 computes gate and up tiles together and applies SiLU(gate) * up in the epilogue.
    The FP32 global scale (and the 2^9 that keeps the decode exact) is applied in FP32.
  * Persistent grid. Only m-blocks of this GPU's 32 experts are visited: moe_align sorts by
    expert, so they come first.

prepare() self-tests the custom MoE against the Marlin path and falls back to Marlin if it
disagrees.
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
BK_MOE = 128                     # K per pipeline step (16 words of 8 values per weight row)
SCALE_BIAS = float(2.0 ** 117)   # block scale * 2^117 (bf16); decoded e2m1 bits are value * 2^-126
EPI_SCALE = 512.0                # 2^9, applied with the FP32 global scale on the accumulator
BITREV = (0, 4, 2, 6, 1, 5, 3, 7)  # decoder output s holds element 8w + BITREV[s]


@triton.jit
def _deq8(x, s):
    """x: int32 word of 8 E2M1 codes (repacked); s: bf16 block scale * 2^117.
    Returns 8 bf16 values (value * scale * 2^-9), outputs in decoder slot order."""
    return tl.inline_asm_elementwise(
        asm="""
{
.reg .b32 t0, t1, t2, t3, t4, s2, r0, r1, r2, r3;
mov.b32 s2, {$9, $9};
and.b32 t0, $8, 0x81C081C0;
mul.rn.bf16x2 r0, t0, s2;
shl.b32 t1, $8, 3;
and.b32 t1, t1, 0x81C081C0;
mul.rn.bf16x2 r1, t1, s2;
shl.b32 t2, $8, 6;
and.b32 t2, t2, 0x81C081C0;
mul.rn.bf16x2 r2, t2, s2;
shl.b32 t3, $8, 1;
and.b32 t3, t3, 0x80008000;
shr.b32 t4, $8, 3;
and.b32 t4, t4, 0x01800180;
or.b32 t3, t3, t4;
shr.b32 t4, $8, 7;
and.b32 t4, t4, 0x00400040;
or.b32 t3, t3, t4;
mul.rn.bf16x2 r3, t3, s2;
mov.b32 {$0, $1}, r0;
mov.b32 {$2, $3}, r1;
mov.b32 {$4, $5}, r2;
mov.b32 {$6, $7}, r3;
}
""",
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h", args=[x, s],
        dtype=(tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16,
               tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16),
        is_pure=True, pack=1)


@triton.jit
def _load_wtile(wp, sp, BN: tl.constexpr, NW: tl.constexpr, SB: tl.constexpr):
    """wp: int32 pointers [BN, NW]; sp: e4m3 pointers [BN, NW] (group = word // 2).
    Returns the bf16 tile [BN, NW * 8] in natural K order (times 2^-9)."""
    x = tl.load(wp)
    s = (tl.load(sp).to(tl.float32) * SB).to(tl.bfloat16)       # [BN, NW]: word w uses group w // 2
    o0, o1, o2, o3, o4, o5, o6, o7 = _deq8(x, s)
    z = tl.join(tl.join(tl.join(o0, o1), tl.join(o2, o3)), tl.join(tl.join(o4, o5), tl.join(o6, o7)))
    return tl.reshape(z, (BN, NW * 8))


@triton.jit
def _deq_test_kernel(W, S, OUT, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    NW: tl.constexpr = BK // 8
    offs_n = pid * BN + tl.arange(0, BN)
    offs_w = tl.arange(0, NW)
    offs_k = tl.arange(0, BK)
    for kb in range(0, K // BK):
        wp = W + offs_n[:, None] * (K // 8) + kb * NW + offs_w[None, :]
        sp = S + offs_n[:, None] * (K // 16) + kb * (BK // 16) + (tl.arange(0, NW) // 2)[None, :]
        t = _load_wtile(wp, sp, BN, NW, SB)
        tl.store(OUT + offs_n[:, None] * K + kb * BK + offs_k[None, :], t)


@triton.jit
def _deq_experts_kernel(W, S, OUT, expert_ids, npost_ptr, N: tl.constexpr, K: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, SB: tl.constexpr, BM: tl.constexpr,
                        MAXMB: tl.constexpr):
    """Decode expert e's NVFP4 rows [pid_n*BN, +BN) to bf16 (value * scale * 2^-9), only for
    experts that have at least one routed block this step (read on device)."""
    pid_n = tl.program_id(0)
    e = tl.program_id(1)
    num_m = tl.load(npost_ptr) // BM
    mb = tl.arange(0, MAXMB)
    eids = tl.load(expert_ids + mb, mask=mb < num_m, other=-1)
    used = tl.sum((eids == e).to(tl.int32), axis=0)
    if used > 0:
        NW: tl.constexpr = BK // 8
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_w = tl.arange(0, NW)
        offs_k = tl.arange(0, BK)
        e64 = e.to(tl.int64)
        for kb in range(0, K // BK):
            wp = W + e64 * (N * (K // 8)) + offs_n[:, None] * (K // 8) + kb * NW + offs_w[None, :]
            sp = S + e64 * (N * (K // 16)) + offs_n[:, None] * (K // 16) + kb * (BK // 16) + (tl.arange(0, NW) // 2)[None, :]
            t = _load_wtile(wp, sp, BN, NW, SB)
            tl.store(OUT + e64 * (N * K) + offs_n[:, None] * K + kb * BK + offs_k[None, :], t)


@triton.jit
def _tile_coords(tile, num_lm, num_n, GROUP_M: tl.constexpr):
    width = GROUP_M * num_n
    first_m = (tile // width) * GROUP_M
    gsz = tl.minimum(num_lm - first_m, GROUP_M)
    r = tile % width
    return first_m + r % gsz, r // gsz


@triton.jit
def _num_local_blocks(expert_ids, npost_ptr, BM: tl.constexpr, NLOCAL: tl.constexpr, MAXMB: tl.constexpr):
    num_m = tl.load(npost_ptr) // BM
    mb = tl.arange(0, MAXMB)
    eids = tl.load(expert_ids + mb, mask=mb < num_m, other=NLOCAL)
    return tl.sum(((eids >= 0) & (eids < NLOCAL)).to(tl.int32), axis=0)


@triton.jit
def _moe_gemm1(A, W, S, GS, C, sorted_ids, expert_ids, npost_ptr, num_valid,
               K: tl.constexpr, NH: tl.constexpr, TOPK: tl.constexpr, NLOCAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
               MAXMB: tl.constexpr, NUM_PROGS: tl.constexpr, SB: tl.constexpr, EPI: tl.constexpr,
               DEQ: tl.constexpr):
    """inter[r, n] = silu(x W_gate^T)[n] * (x W_up^T)[n] for each routed row r = token*TOPK+slot."""
    pid = tl.program_id(0)
    NW: tl.constexpr = BK // 8
    KW: tl.constexpr = K // 8
    KS: tl.constexpr = K // 16
    num_lm = _num_local_blocks(expert_ids, npost_ptr, BM, NLOCAL, MAXMB)
    num_n: tl.constexpr = NH // BN
    total = num_lm * num_n
    offs_n = tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_w = tl.arange(0, NW)
    for tile in range(pid, total, NUM_PROGS):
        pid_m, pid_n = _tile_coords(tile, num_lm, num_n, GROUP_M)
        e = tl.load(expert_ids + pid_m).to(tl.int64)
        rows = tl.load(sorted_ids + pid_m * BM + offs_m)
        rmask = rows < num_valid
        tok = tl.where(rmask, rows // TOPK, 0)
        a_ptr = A + tok[:, None].to(tl.int64) * K + offs_k[None, :]
        n_g = pid_n * BN + offs_n
        wg = W + e * (2 * NH * KW) + n_g[:, None] * KW + offs_w[None, :]
        wu = wg + NH * KW
        sg = S + e * (2 * NH * KS) + n_g[:, None] * KS + (tl.arange(0, NW) // 2)[None, :]
        su = sg + NH * KS
        bg = W + e * (2 * NH * K) + n_g[:, None] * K + offs_k[None, :]
        bu = bg + NH * K
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // BK):
            a = tl.load(a_ptr + kb * BK, mask=rmask[:, None], other=0.0)
            if DEQ:
                tg = _load_wtile(wg + kb * NW, sg + kb * (BK // 16), BN, NW, SB)
                tu = _load_wtile(wu + kb * NW, su + kb * (BK // 16), BN, NW, SB)
            else:
                tg = tl.load(bg + kb * BK)
                tu = tl.load(bu + kb * BK)
            acc_g = tl.dot(a, tl.trans(tg), acc_g)
            acc_u = tl.dot(a, tl.trans(tu), acc_u)
        gs = tl.load(GS + e) * EPI
        g = acc_g * gs
        u = acc_u * gs
        h = g * tl.sigmoid(g) * u
        cp = C + rows[:, None].to(tl.int64) * NH + n_g[None, :]
        tl.store(cp, h.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _moe_gemm2(A, W, S, GS, C, sorted_ids, expert_ids, npost_ptr, num_valid,
               K: tl.constexpr, N: tl.constexpr, NLOCAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
               MAXMB: tl.constexpr, NUM_PROGS: tl.constexpr, SB: tl.constexpr, EPI: tl.constexpr,
               DEQ: tl.constexpr):
    """y[r, n] = inter[r] W2^T for each routed row r."""
    pid = tl.program_id(0)
    NW: tl.constexpr = BK // 8
    KW: tl.constexpr = K // 8
    KS: tl.constexpr = K // 16
    num_lm = _num_local_blocks(expert_ids, npost_ptr, BM, NLOCAL, MAXMB)
    num_n: tl.constexpr = N // BN
    total = num_lm * num_n
    offs_n = tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_w = tl.arange(0, NW)
    for tile in range(pid, total, NUM_PROGS):
        pid_m, pid_n = _tile_coords(tile, num_lm, num_n, GROUP_M)
        e = tl.load(expert_ids + pid_m).to(tl.int64)
        rows = tl.load(sorted_ids + pid_m * BM + offs_m)
        rmask = rows < num_valid
        r_safe = tl.where(rmask, rows, 0)
        a_ptr = A + r_safe[:, None].to(tl.int64) * K + offs_k[None, :]
        n_g = pid_n * BN + offs_n
        wp = W + e * (N * KW) + n_g[:, None] * KW + offs_w[None, :]
        sp = S + e * (N * KS) + n_g[:, None] * KS + (tl.arange(0, NW) // 2)[None, :]
        bp = W + e * (N * K) + n_g[:, None] * K + offs_k[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // BK):
            a = tl.load(a_ptr + kb * BK, mask=rmask[:, None], other=0.0)
            if DEQ:
                t = _load_wtile(wp + kb * NW, sp + kb * (BK // 16), BN, NW, SB)
            else:
                t = tl.load(bp + kb * BK)
            acc = tl.dot(a, tl.trans(t), acc)
        y = acc * (tl.load(GS + e) * EPI)
        cp = C + rows[:, None].to(tl.int64) * N + n_g[None, :]
        tl.store(cp, y.to(tl.bfloat16), mask=rmask[:, None])


def _repack_nvfp4(packed):
    """Lossless bit permutation within each 32-bit word (8 consecutive K elements) of standard
    NVFP4 packing into the layout _deq8 decodes. Returns int32 [E, N, K/8]."""
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] // 4,), dtype=torch.int32, device=packed.device)
    n, kh = packed.shape[-2:]
    for e in range(packed.shape[0]):
        p = packed[e]
        lo = (p & 0xF).to(torch.int64)
        hi = (p >> 4).to(torch.int64)
        c = torch.stack((lo, hi), -1).reshape(n, kh // 4, 8)
        del lo, hi
        sl = [c[..., BITREV[s]] for s in range(8)]

        def sg(x):
            return (x >> 3) & 1

        def mg(x):
            return x & 7

        w = (sg(sl[0]) << 15) | (mg(sl[0]) << 6) | (sg(sl[1]) << 31) | (mg(sl[1]) << 22)
        w |= (sg(sl[2]) << 12) | (mg(sl[2]) << 3) | (sg(sl[3]) << 28) | (mg(sl[3]) << 19)
        w |= (sg(sl[4]) << 9) | mg(sl[4]) | (sg(sl[5]) << 25) | (mg(sl[5]) << 16)
        w |= (sg(sl[6]) << 14) | ((mg(sl[6]) >> 1) << 10) | ((mg(sl[6]) & 1) << 13)
        w |= (sg(sl[7]) << 30) | ((mg(sl[7]) >> 1) << 26) | ((mg(sl[7]) & 1) << 29)
        w = torch.where(w >= (1 << 31), w - (1 << 32), w)
        out[e] = w.to(torch.int32)
        del c, sl, w
    return out


MOE_CANDIDATES = {
    "A": dict(BM=64, g1=(64, 4, 3), g2=(128, 4, 3), occ=2, group=8),
    "B": dict(BM=64, g1=(64, 8, 3), g2=(128, 8, 3), occ=2, group=8),
    "C": dict(BM=128, g1=(64, 8, 3), g2=(128, 8, 3), occ=1, group=8),
    "D": dict(BM=64, g1=(128, 8, 3), g2=(128, 8, 4), occ=1, group=8),
    "E": dict(BM=128, g1=(128, 8, 3), g2=(256, 8, 3), occ=1, group=8),
    "F": dict(BM=128, g1=(64, 8, 4), g2=(128, 8, 4), occ=1, group=16),
    # per-call decode of touched experts to a BF16 scratch, then a plain BF16 grouped GEMM
    "G": dict(BM=128, g1=(64, 8, 3), g2=(128, 8, 3), occ=1, group=8, bf16=True),
    "G2": dict(BM=128, g1=(128, 8, 2), g2=(256, 8, 2), occ=1, group=8, bf16=True),
    "H": dict(BM=64, g1=(64, 4, 3), g2=(128, 4, 3), occ=1, group=8, bf16=True),
}
BUCKETS = (("small", 1024, ("A", "G", "H")), ("mid", 6144, ("C", "G", "G2")),
           ("big", 1 << 30, ("E", "G", "G2")))
BENCH_TOKENS = {"small": 753, "mid": 3012, "big": 16384}


def _bucket(tokens):
    for name, limit, _ in BUCKETS:
        if tokens <= limit:
            return name
    return BUCKETS[-1][0]


def _moe_config(tokens):
    return MOE_CANDIDATES["C" if tokens >= 8192 else "A"]


def _custom_experts(x, topk_weights, lid, slot_ids, state, cfg=None):
    """x [T, 4096] bf16 (pre-expert-normed); lid [T, 8] int32 experts relative to local_start
    (0..255, this GPU's are 0..31); slot_ids [T, 8] (-1 off this GPU). Returns [T, 4096] bf16."""
    moe = state["moe"]
    t = x.shape[0]
    cfg = cfg or _moe_config(t)
    bm = cfg["BM"]
    sorted_ids, expert_ids, npost = moe_align_block_size(lid, bm, N_EXPERTS)
    maxmb = triton.next_power_of_2(expert_ids.numel())
    num_valid = t * TOP_K
    inter = torch.empty((num_valid, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((num_valid, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    progs = moe["num_sms"] * cfg["occ"]
    deq = not cfg.get("bf16", False)
    w13, w2b = moe["w13"], moe["w2"]
    if not deq:
        w13, w2b = moe["w13_bf"], moe["w2_bf"]
        main_stream, decode_stream = torch.cuda.current_stream(), state["moe_decode_stream"]
        _deq_experts_kernel[(2 * D_EXPERT // 64, LOCAL_EXPERTS)](
            moe["w13"], moe["w13_sf"], w13, expert_ids, npost, N=2 * D_EXPERT, K=D_EXPERT, BN=64, BK=BK_MOE,
            SB=SCALE_BIAS, BM=bm, MAXMB=maxmb, num_warps=4)
        # W2 is not consumed until GEMM2. Decode it on a captured side stream while GEMM1
        # consumes the already-decoded W13, then join immediately before GEMM2.
        decode_stream.wait_stream(main_stream)
        with torch.cuda.stream(decode_stream):
            _deq_experts_kernel[(D_EXPERT // 64, LOCAL_EXPERTS)](
                moe["w2"], moe["w2_sf"], w2b, expert_ids, npost, N=D_EXPERT, K=D_EXPERT, BN=64, BK=BK_MOE,
                SB=SCALE_BIAS, BM=bm, MAXMB=maxmb, num_warps=4)
    bn1, w1, s1 = cfg["g1"]
    _moe_gemm1[(progs,)](
        x, w13, moe["w13_sf"], moe["w13_gs"], inter, sorted_ids, expert_ids, npost, num_valid,
        K=D_EXPERT, NH=D_EXPERT, TOPK=TOP_K, NLOCAL=LOCAL_EXPERTS, BM=bm, BN=bn1, BK=BK_MOE,
        GROUP_M=cfg["group"], MAXMB=maxmb, NUM_PROGS=progs, SB=SCALE_BIAS, EPI=EPI_SCALE, DEQ=deq,
        num_warps=w1, num_stages=s1)
    if not deq:
        main_stream.wait_stream(decode_stream)
    bn2, w2, s2 = cfg["g2"]
    _moe_gemm2[(progs,)](
        inter, w2b, moe["w2_sf"], moe["w2_gs"], y, sorted_ids, expert_ids, npost, num_valid,
        K=D_EXPERT, N=D_EXPERT, NLOCAL=LOCAL_EXPERTS, BM=bm, BN=bn2, BK=BK_MOE,
        GROUP_M=cfg["group"], MAXMB=maxmb, NUM_PROGS=progs, SB=SCALE_BIAS, EPI=EPI_SCALE, DEQ=deq,
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


def _dequant_ref(packed, sf):
    """Plain-torch NVFP4 decode (no global scale), float32 [N, K]."""
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    mag = grid[codes & 7]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    return (vals.view(vals.shape[0], -1, 16) * sf.float().unsqueeze(-1)).view(vals.shape)


def _self_test(state, weights, device):
    moe = state["moe"]
    # 1. Decoder alone on expert 0's w2.
    n_rows = 256
    out = torch.empty((n_rows, D_EXPERT), device=device, dtype=torch.bfloat16)
    _deq_test_kernel[(n_rows // 64,)](moe["w2"][0], moe["w2_sf"][0], out, K=D_EXPERT, BN=64, BK=BK_MOE,
                                      SB=SCALE_BIAS, num_warps=4)
    ref = _dequant_ref(weights["w2"][0, :n_rows], weights["w2_sf"][0, :n_rows]) * 2.0 ** -9
    got = out.float()
    bad = (got != ref)
    print(f"[self-test] decode mismatches {int(bad.sum())}/{bad.numel()} max|d|={float((got - ref).abs().max()):.3e}",
          flush=True)
    if bad.any():
        print("[self-test] got row0[:16]", [round(v, 4) for v in (got[0, :16] * 512).tolist()], flush=True)
        print("[self-test] ref row0[:16]", [round(v, 4) for v in (ref[0, :16] * 512).tolist()], flush=True)
    decode_ok = not bad.any()
    # 2. Whole MoE vs Marlin.
    gen = torch.Generator(device=device).manual_seed(1234)
    ok = decode_ok
    for t, cfg in ((512, MOE_CANDIDATES["A"]),):
        x = torch.randn((t, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        scores = torch.rand((t, N_EXPERTS), generator=gen, device=device)
        scores[:, :LOCAL_EXPERTS] += 0.5
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


def _synthetic_routing(t, gen, device):
    """Skewed routing like the scored cases: ~2 local routes per token over 16 hot experts."""
    probs = 0.75 ** torch.arange(16, device=device, dtype=torch.float32)
    loc = torch.multinomial(probs.expand(t, 16).contiguous(), 2, replacement=False, generator=gen)
    rest = torch.rand((t, N_EXPERTS - LOCAL_EXPERTS), generator=gen, device=device).topk(TOP_K - 2).indices + LOCAL_EXPERTS
    ids = torch.cat((loc, rest), 1).to(torch.int32).contiguous()
    w = torch.rand((t, TOP_K), generator=gen, device=device) + 0.1
    return ids, (w / w.sum(-1, keepdim=True)).float().contiguous()


def _time_ms(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _autotune(state, device):
    """Untimed: pick the fastest expert implementation per token bucket on synthetic routing."""
    gen = torch.Generator(device=device).manual_seed(77)
    choice = {}
    for name, _, cands in BUCKETS:
        t = BENCH_TOKENS[name]
        x = torch.randn((t, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        ids, w = _synthetic_routing(t, gen, device)
        slot_ids = state["bank"]["expert_map"][ids].contiguous()
        results = {"marlin": _time_ms(lambda: _marlin_experts(x, w, ids, state))}
        ref = _marlin_experts(x, w, ids, state).float()
        for c in cands:
            try:
                got = _custom_experts(x, w, ids, slot_ids, state, MOE_CANDIDATES[c]).float()
                torch.cuda.synchronize()
                err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
                if not (err < 2e-2 and bool(torch.isfinite(got).all())):
                    print(f"[autotune] {name} {c} wrong: rel_l2={err:.3e}", flush=True)
                    continue
                results[c] = _time_ms(lambda: _custom_experts(x, w, ids, slot_ids, state, MOE_CANDIDATES[c]))
            except Exception as exc:  # noqa: BLE001
                print(f"[autotune] {name} {c} failed: {str(exc)[-300:]}", flush=True)
        best = min(results, key=results.get)
        choice[name] = best
        print(f"[autotune] {name} T={t}: " + ", ".join(f"{k}={v * 1000:.0f}us" for k, v in results.items())
              + f" -> {best}", flush=True)
    return choice


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
        "moe_decode_stream": torch.cuda.Stream(device),
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
        # Scratch for the per-call decode path (filled inside kernel_fn every call, never reused
        # across calls as a cache).
        "w13_bf": torch.empty((LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT), device=device, dtype=torch.bfloat16),
        "w2_bf": torch.empty((LOCAL_EXPERTS, D_EXPERT, D_EXPERT), device=device, dtype=torch.bfloat16),
    }
    torch.cuda.synchronize()
    try:
        state["use_custom"] = _self_test(state, weights, device)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[self-test] custom MoE failed: {str(exc)[-1500:]}", flush=True)
        state["use_custom"] = False
    state["moe_choice"] = {name: "marlin" for name, _, _ in BUCKETS}
    if state["use_custom"]:
        try:
            state["moe_choice"] = _autotune(state, device)
        except Exception as exc:  # noqa: BLE001
            print(f"[autotune] failed: {str(exc)[-500:]}", flush=True)
    print(f"[prepare] use_custom={state['use_custom']} choice={state['moe_choice']}", flush=True)
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
    lid = ((topk_ids - data["local_start"]) % N_EXPERTS).to(torch.int32)
    x_e = state["pre_expert_norm"](expert_input)
    impl = state["moe_choice"][_bucket(tokens)]
    if impl != "marlin":
        slot_ids = state["bank"]["expert_map"][lid].contiguous()
        expert_output = _custom_experts(x_e, topk_weights, lid.contiguous(), slot_ids, state, MOE_CANDIDATES[impl])
    else:
        expert_output = _marlin_experts(x_e, topk_weights, lid, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
