"""Extension build and the GPU entry point linear(x, packed)."""
from __future__ import annotations

import os

import torch

from . import configs as C
from .packing import PackedWeight

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SOURCES = ("bindings.cpp", "zg.cu", "zt.cu", "zr.cu", "zs.cu")
# The flags of the measured build. The kernels use only sm_90 PTX (mma.sync, cp.async.bulk[.tensor], mbarrier,
# griddepcontrol); they are compiled for sm_90a, as measured.
CUDA_FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"]

_EXT = None
_WS = {}


def load_extension(verbose: bool = False):
    """JIT-build (once) and return the CUDA extension. Needs CUDA >= 12.5 (nvcc and runtime) to build and a
    compute capability 9.0 GPU (H100 / H200) to run."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="lossless_bf16_ext", sources=[os.path.join(_CSRC, s) for s in SOURCES],
                    extra_cflags=["-O3"], extra_cuda_cflags=CUDA_FLAGS, verbose=verbose)
    return _EXT


def _splitk_workspace(device, floats: int) -> torch.Tensor:
    """fp32 partials buffer, reused across calls and grown on demand. The kernel never needs it zeroed. Grow it
    outside graph capture (call linear() once per configuration before capturing)."""
    key = torch.device(device)
    ws = _WS.get(key)
    if ws is None or ws.numel() < floats:
        ws = torch.empty((max(floats, 1),), dtype=torch.float32, device=device)
        _WS[key] = ws
    return ws


def _num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def launch(x: torch.Tensor, p: PackedWeight, c: C.Config, out: torch.Tensor) -> torch.Tensor:
    """Run configuration c: out[m, N] (bf16) = x[m, K] @ W^T. x and out contiguous bf16 on the GPU."""
    ext = load_extension()
    m, k = x.shape
    n = p.n
    if not C.fits(c, n, k, m):
        raise ValueError(f"config {c} does not fit N={n}, K={k}, m={m}")
    if c.kernel == "fullk":
        r = p.rowtile
        ext.zg_gemm(x, r["wp"], r["rscale"], r["tab"], r["emax"], out, m, k, n // 16, n, c.cfg)
    elif c.kernel == "fullk_tma":
        r = p.rowtile
        ext.zt_gemm(x, r["wp"], r["rscale"], r["tab"], r["emax"], out, m, k, n // 16, n, c.cfg)
    elif c.kernel == "rowpair":
        if p.rowpair is None:
            raise ValueError("pack with layouts including 'rowpair' to use the rowpair kernel")
        r = p.rowpair
        ext.zr_gemv(x, r["wp"], r["rscale"], r["tab"], r["emax"], out, None, k, n // 2, c.cfg)
    elif c.kernel == "splitk":
        if p.blocked is None:
            raise ValueError("pack with layouts including 'blocked' to use the splitk kernel")
        G = c.grid or _num_sms(x.device)
        pl = C.splitk_plan(n, k, c.cfg, G)
        ws = _splitk_workspace(x.device, pl["ws_floats"])
        b = p.blocked
        ext.zs_gemm(x, b["wpb"][pl["rt"]], p.rowtile["rscale"], b["erow"], b["emr"], out, ws, m, k, n, c.cfg,
                    pl["G"], pl["jmax"], pl["tpp"], pl["npt"], 1, None)
    else:
        raise ValueError(c.kernel)
    return out


def select(p: PackedWeight, m: int) -> C.Config:
    """Tuned configuration for m's token bucket if tune() ran, else the measured / default choice."""
    c = p.tuned.get(C.bucket(m))
    if c is not None and C.fits(c, p.n, p.k, m):
        return c
    return C.default_config(p.n, p.k, m, p.layouts)


def linear(x: torch.Tensor, p: PackedWeight, config: C.Config | None = None) -> torch.Tensor:
    """y = x @ W^T for x [..., K] bf16 (1 <= tokens <= 32) and W packed in the 12-bit format. Returns bf16 [..., N].

    Accumulation is fp32 on tensor cores (mma.sync m16n8k16), so results match a BF16 GEMM with fp32 accumulation
    up to summation order. For more than 32 tokens use a dense GEMM on unpacked weights."""
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise TypeError("x must be a bfloat16 CUDA tensor")
    if x.shape[-1] != p.k:
        raise ValueError(f"x has K = {x.shape[-1]}, weight has K = {p.k}")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, p.k).contiguous()
    m = x2.shape[0]
    c = config or select(p, m)
    out = torch.empty((m, p.n), dtype=torch.bfloat16, device=x.device)
    launch(x2, p, c, out)
    return out.reshape(*lead, p.n)


def probe_exact(p: PackedWeight, w: torch.Tensor, c: C.Config, m: int, seed: int = 0, escaped: bool = False) -> float:
    """Fraction of outputs that are bit-exact for one-hot activations (one +-2^3..2^7 entry per token row): each
    output row must equal the scaled weight column exactly. escaped=True puts the entries on escaped columns."""
    gen = torch.Generator(device=w.device).manual_seed(seed + m)
    k = p.k
    if escaped and p.n_escapes:
        tab = p.rowtile["tab"]
        valid = torch.arange(tab.shape[1] - 1, device=tab.device)[None, :] < tab[:, 0, 0].long()[:, None]
        esc_cols = torch.unique(tab[:, 1:, 0][valid]).long()
        cols = esc_cols[torch.randint(esc_cols.numel(), (m,), generator=gen, device=w.device)]
    else:
        cols = torch.randint(k, (m,), generator=gen, device=w.device)
    val = ((torch.randint(2, (m,), generator=gen, device=w.device) * 2 - 1)
           * 2.0 ** torch.randint(3, 8, (m,), generator=gen, device=w.device)).float()
    x = torch.zeros((m, k), dtype=torch.float32, device=w.device)
    x[torch.arange(m, device=w.device), cols] = val
    x = x.to(torch.bfloat16)
    out = torch.full((m, p.n), float("nan"), dtype=torch.bfloat16, device=w.device)
    launch(x, p, c, out)
    ref = (val.double()[:, None] * w.double()[:, cols].T).to(torch.bfloat16)
    same = (out == ref) | (torch.isnan(out) & torch.isnan(ref))     # value equality: +0 == -0
    return float(same.float().mean())
