"""CPU emulation of the kernels' arithmetic on the packed data (no GPU): fragment decode, scales, escape terms,
and the split-K launch plan. The emulation reads only the packed tensors, the way the kernels do."""
import math

import pytest
import torch

from lossless_bf16 import configs as C
from lossless_bf16 import format as fmt
from lossless_bf16 import k_permutation, pack


def decoded_scaled_f64(p):
    """Decoded (scaled) weights of the rowtile layout as float64 [N, K] in actual column order."""
    n, k = p.n, p.k
    nrt, kcn = n // 16, k // 64
    cm = p.rowtile["wp"].view(nrt, kcn, 1536).permute(1, 0, 2).contiguous().view(-1)
    bits = fmt.decode_chunk_major(cm, n, k)
    vals = (bits << 16).view(torch.float32).double()
    out = torch.empty_like(vals)
    out[:, k_permutation(k)] = vals
    return out


def escapes(p):
    tab = p.rowtile["tab"]
    nrt, emax = tab.shape[0], tab.shape[1]
    valid = torch.arange(emax - 1)[None, :] < tab[:, 0, 0].long()[:, None]
    recs = tab[:, 1:, :][valid]
    rts = torch.arange(nrt)[:, None].expand(nrt, emax - 1)[valid]
    rows = rts * 16 + recs[:, 1].long()
    cols = recs[:, 0].long()
    w = recs[:, 2].view(torch.float32).double()
    nv = recs[:, 3].view(torch.float32).double()
    return rows, cols, w, nv


def emulate_fullk(p, x):
    """out = (sum_k x * decoded) * 2^-s + sum_escapes (x * w + x * nv), in float64."""
    dec = decoded_scaled_f64(p)
    acc = x.double() @ dec.T
    out = acc * p.rowtile["rscale"].double()[None, :]
    rows, cols, w, nv = escapes(p)
    xd = x.double()
    for r, c, wv, nvv in zip(rows.tolist(), cols.tolist(), w.tolist(), nv.tolist()):
        out[:, r] += xd[:, c] * wv + xd[:, c] * nvv
    return out


def synthetic(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((n, k), generator=g) / math.sqrt(k)).to(torch.bfloat16)
    w[:, ::37] = 0                                                       # zero columns
    w[::7, 3::53] = (w[::7, 3::53].float() * 2.0 ** -16).to(torch.bfloat16)   # just below the window
    w[5, :40] = 0
    w[3, 5] = 1e-20                                                      # far below the window
    return w


@pytest.mark.parametrize("m", [1, 5, 8, 16, 32])
def test_fullk_arithmetic_matches_fp64(m):
    w = synthetic(128, 1024, 1)
    p = pack(w)
    g = torch.Generator().manual_seed(m)
    x = torch.randn((m, 1024), generator=g).to(torch.bfloat16)
    ref = x.double() @ w.double().T
    out = emulate_fullk(p, x)
    assert float((out - ref).norm() / ref.norm()) < 1e-12


def test_one_hot_rows_are_exact_in_fp32():
    """One-hot activations (a power of two at one column): with fp32 arithmetic in the kernel's order, every output
    equals the weight column exactly. Escaped columns included."""
    w = synthetic(64, 512, 2)
    p = pack(w)
    dec = decoded_scaled_f64(p)                 # exact (bf16 values)
    rscale = p.rowtile["rscale"].double()
    rows, cols, wv, nv = escapes(p)
    esc_cols = torch.unique(cols).tolist()
    for c in list(range(0, 512, 7)) + esc_cols:
        for v in (8.0, -32.0, 128.0):
            acc = (v * dec[:, c]).float()                              # exact in fp32
            out = (acc.double() * rscale).float()                      # 2^-s scaling: exact
            sel = cols == c
            corr = torch.zeros(64, dtype=torch.float64)
            wsum = torch.zeros(64, dtype=torch.float64)
            corr.index_add_(0, rows[sel], v * nv[sel])                 # x * nv: exact (power of two)
            wsum.index_add_(0, rows[sel], v * wv[sel])                 # x * w: exact (bf16 times power of two)
            got = ((out.double() + corr).float().double() + wsum).float()   # zs / zr order: (s*scale + N) + W
            want = (v * w[:, c].double()).float()
            assert torch.equal(got, want) or bool(((got == want) | (got.isnan() & want.isnan())).all())


def test_fullk_escape_order_limit():
    """Documents a known limit of the fullk / fullk_tma escape arithmetic, out = fmaf(acc, 2^-s, fmaf(x, w, x nv)):
    when |w| is more than ~39 binades below the row's top binade, fp32 rounds x * w away inside the inner fmaf and
    a one-hot output comes out as 0 instead of x * w. The splitk and rowpair kernels keep sum(x * w) apart and are
    exact. For dense activations the lost term is far below BF16 output resolution."""
    w = synthetic(64, 512, 2)
    assert float(w[3, 5]) != 0.0 and abs(float(w[3, 5])) < 1e-19
    p = pack(w)
    rows, cols, wv, nv = escapes(p)
    sel = (rows == 3) & (cols == 5)
    v = 8.0
    w_, nv_ = float(wv[sel]), float(nv[sel])
    scale = float(p.rowtile["rscale"][3])
    acc = v * 2.0 ** -15                                   # the decoded placeholder times x
    f32 = lambda t: float(torch.tensor(t, dtype=torch.float64).float())   # noqa: E731
    inner = f32(v * w_ + f32(v * nv_))                     # fmaf(x, w, x * nv)
    fullk = f32(acc * scale + inner)                       # fmaf(acc, scale, corr)
    split = f32(f32(f32(acc * scale) + f32(v * nv_)) + f32(v * w_))
    assert fullk == 0.0
    assert split == f32(v * float(w[3, 5]))


@pytest.mark.parametrize("n,k", [(2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096), (256, 512)])
@pytest.mark.parametrize("cfg", sorted(C.SPLITK))
def test_splitk_plan(n, k, cfg):
    _, ncw, rw, _ = C.SPLITK[cfg]
    rt = ncw * rw
    if (n // 16) % rt:
        pytest.skip("row block height does not divide N/16")
    for G in (132, 128, 37):
        pl = C.splitk_plan(n, k, cfg, G)
        U, G = pl["U"], pl["G"]
        ranges = [(c * U // G, (c + 1) * U // G) for c in range(G)]
        assert ranges[0][0] == 0 and ranges[-1][1] == U
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        sizes = [b - a for a, b in ranges]
        assert max(sizes) - min(sizes) <= 1                       # equal byte shares, +-1 tile
        kcn = k // 64
        jm = 0
        for rb in range(pl["nrb"]):
            first, last = rb * kcn * rt, rb * kcn * rt + kcn * rt - 1
            owners = {c for c, (a, b) in enumerate(ranges) if a <= last and b > first}
            jm = max(jm, len(owners))
            assert owners == set(range(C.splitk_owner(first, G, U), C.splitk_owner(last, G, U) + 1))
        assert jm == pl["jmax"]
        assert pl["tpp"] * pl["npt"] >= pl["jmax"] and pl["npt"] in (4, 8)


def test_measured_configs_fit_their_shapes():
    for (n, k, b), c in C.MEASURED.items():
        assert C.fits(c, n, k, b), (n, k, b, c)
        assert c in C.candidates(n, k, b, ("rowtile", "blocked", "rowpair"), 132) or c.grid == 0


@pytest.mark.parametrize("m", [1, 2, 8, 9, 16, 17, 32])
def test_default_config_without_optional_layouts(m):
    for n, k in [(2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096), (1024, 4096), (48, 128)]:
        c = C.default_config(n, k, m, ("rowtile",))
        assert C.LAYOUT_OF[c.kernel] == "rowtile" and C.fits(c, n, k, m)


def test_bucket_bounds():
    assert [C.bucket(m) for m in (1, 2, 8, 9, 16, 17, 32)] == [1, 8, 8, 16, 16, 32, 32]
    with pytest.raises(ValueError):
        C.bucket(33)
