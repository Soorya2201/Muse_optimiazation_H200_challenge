"""The 12-bit tile format: encoder, escape records and a bit-level reference decoder.

A BF16 weight matrix W [N, K] (N % 16 == 0, K % 64 == 0) is stored as 16 x 64 tiles of 1536 bytes, 12 bits per
weight, plus one power-of-two scale per row and a list of escaped weights. Every BF16 bit pattern round-trips
exactly. See docs/format.md for the layout; the names below follow it.

Per row r, let e_max be the largest biased exponent among the row's normal (finite, non-zero, non-subnormal)
weights and s = 127 - e_max (s = 0 if the row has no normal weight). After scaling by 2^s, a weight whose exponent
lies in the 16-binade window [112, 127] is stored as

    L byte  = e0 | m6..m0          (exponent bit 0, then the 7 mantissa bits: the low byte of the scaled BF16)
    nibble  = sign | e3 e2 e1      (e3..e0 = scaled exponent - 112)

and decodes to the BF16 value with high byte  sign | 0111 | e3 e2 e1. Everything else (exponent more than 15
binades below the row maximum, +-0, subnormals, +-inf, NaN) is an escape: its 12 bits are all zero, which decodes
to +2^-15, and an escape record stores the original bits so the kernel can add x * w - x * 2^-15 * 2^-s.

All functions are pure PyTorch and run on CPU or GPU tensors.
"""
from __future__ import annotations

import torch

BLOCK_BYTES = 1536          # one 16 x 64 tile
TILE_ROWS = 16
TILE_COLS = 64
WINDOW_LO = 112             # scaled exponents 112..127 are stored in 4 bits
WINDOW_HI = 127


def _bits16(w: torch.Tensor) -> torch.Tensor:
    """BF16 tensor -> its 16-bit patterns as int32 in [0, 65535]."""
    if w.dtype != torch.bfloat16:
        raise TypeError(f"expected a bfloat16 tensor, got {w.dtype}")
    return w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def row_shift(bits: torch.Tensor) -> torch.Tensor:
    """Per-row exponent shift s (int32 [N]): s = 127 - (largest normal exponent of the row), 0 for rows without a
    normal weight. Scaling a row by 2^s puts its largest exponent at 127."""
    e = (bits >> 7) & 0xFF
    normal = (e > 0) & (e < 255)
    emax = torch.where(normal, e, torch.zeros_like(e)).amax(dim=1)
    return torch.where(emax > 0, 127 - emax, torch.zeros_like(emax))


def in_window(bits: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Mask of weights stored directly (not escaped): normal, with scaled exponent in [112, 127]."""
    e = (bits >> 7) & 0xFF
    ep = e + s[:, None]
    return (e > 0) & (e < 255) & (ep >= WINDOW_LO) & (ep <= WINDOW_HI)


def encode_chunk_major(w: torch.Tensor) -> dict:
    """Encode W [N, K] bf16 into tiles stored chunk-major: wp uint8 [K/64][N/16][1536].

    Returns dict(wp, rscale f32 [N] = 2^-s, esc int32 [E, 4] = {row, col, bits(w as f32), bits(-2^-15 * 2^-s)}
    sorted by (row tile, 64-column chunk), n_esc, nrt, kcn). Columns are in the order given; callers that want the
    kernel K order permute W first (see packing.k_permutation).
    """
    n, k = w.shape
    if n % TILE_ROWS or k % TILE_COLS:
        raise ValueError(f"shape {tuple(w.shape)}: N must be a multiple of 16 and K of 64")
    nrt, kcn = n // TILE_ROWS, k // TILE_COLS
    dev = w.device
    b = _bits16(w)
    s = row_shift(b)
    ok = in_window(b, s)
    e = (b >> 7) & 0xFF
    e4 = torch.where(ok, e + s[:, None] - WINDOW_LO, torch.zeros_like(e))   # 0..15
    sgn = torch.where(ok, b >> 15, torch.zeros_like(b))
    lbyte = torch.where(ok, ((e4 & 1) << 7) | (b & 0x7F), torch.zeros_like(b))
    e3 = e4 >> 1
    del e4, e
    rscale = torch.ldexp(torch.ones(n, dtype=torch.float64, device=dev), (-s).to(torch.float64)).to(torch.float32)

    # Index of W inside a tile as the mma.sync m16n8k16 A fragments see it:
    #   row = 16 rt + 8 half + g, col = 64 kc + 16 j + 8 hi8 + 2 t + pair,  j = 2 jq + jr
    # L bytes go to [kc, rt, jq, g, t, jr, hi8, half, pair]: lane l = 4 g + t reads 16 B at jq * 512 + 16 l, and its
    # 32-bit word 2 jr + hi8 holds the low bytes of (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1) at column offset 8 hi8.
    lb = lbyte.to(torch.uint8).view(nrt, 2, 8, kcn, 2, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8)
    lb = lb.contiguous().view(kcn, nrt, 1024)
    # Nibble words: one 32-bit word per (lane, k-step j). Slot = 2 half + pair; hi8 = 0 weights use byte `slot`
    # bits {7: sign, 2..0: e3e2e1}; hi8 = 1 weights use byte `slot` bits 6..4 (e3e2e1) and byte (slot+1)%4 bit 3
    # (sign), so a 4-bit rotate maps the second set onto the first.
    half = torch.arange(2, device=dev).view(1, 2, 1, 1, 1, 1, 1, 1, 1)
    hi8 = torch.arange(2, device=dev).view(1, 1, 1, 1, 1, 1, 2, 1, 1)
    pair = torch.arange(2, device=dev).view(1, 1, 1, 1, 1, 1, 1, 1, 2)
    slot = half * 2 + pair
    sh_e = torch.where(hi8 == 0, 8 * slot, 8 * slot + 4).to(torch.int64)
    sh_s = torch.where(hi8 == 0, 8 * slot + 7, 8 * ((slot + 1) % 4) + 3).to(torch.int64)
    e3v = e3.to(torch.int64).view(nrt, 2, 8, kcn, 2, 2, 2, 4, 2)
    sv = sgn.to(torch.int64).view(nrt, 2, 8, kcn, 2, 2, 2, 4, 2)
    contrib = (e3v << sh_e) | (sv << sh_s)
    del e3v, sv
    words = contrib.sum(dim=(1, 6, 8))                       # disjoint bits: sum == or.  [rt, g, kc, jq, jr, t]
    del contrib
    words = words.permute(2, 0, 1, 5, 3, 4).contiguous()     # [kc, rt, g, t, jq, jr]
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    nb = words.view(kcn, nrt, 128).view(torch.uint8)        # [kc, rt, 512] (little-endian words)
    wp = torch.cat([lb, nb], dim=2).contiguous().view(-1)

    # escape records, sorted by (row tile, chunk)
    idx = (~ok).nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    key = (rows // TILE_ROWS) * kcn + cols // TILE_COLS
    order = torch.argsort(key, stable=True)
    rows, cols = rows[order], cols[order]
    wbits = b[rows, cols].to(torch.int64) << 16               # the weight as fp32 bits (exact for every pattern)
    wbits = torch.where(wbits >= 2 ** 31, wbits - 2 ** 32, wbits).to(torch.int32)
    nv = -(2.0 ** -15) * rscale[rows]                         # cancels the decoded placeholder +2^-15, exactly
    esc = torch.stack([rows.to(torch.int32), cols.to(torch.int32), wbits, nv.view(torch.int32)], dim=1).contiguous()
    return {"wp": wp, "rscale": rscale, "esc": esc, "n_esc": int(idx.shape[0]), "nrt": nrt, "kcn": kcn,
            "shift": s}


def _byte_perm(x: torch.Tensor, y: torch.Tensor, sel: int) -> torch.Tensor:
    """CUDA __byte_perm(x, y, sel) on int64 tensors holding 32-bit values (selector values 0..7)."""
    src = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(x)
    for i in range(4):
        out = out | (src[(sel >> (4 * i)) & 7] << (8 * i))
    return out


def decode_chunk_major(wp: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Reference decoder: run the kernels' register decode (1 SHF + 2 LOP3 + 4 PRMT per 8 weights) on the stored
    bytes and place each 16-bit result where the mma.sync A fragment puts it. Returns the scaled BF16 patterns as
    int32 [N, K] (escapes decode to 0x3800 = +2^-15)."""
    nrt, kcn = n // TILE_ROWS, k // TILE_COLS
    dev = wp.device
    blk = wp.view(kcn, nrt, BLOCK_BYTES)
    lq = blk[:, :, :1024].reshape(kcn, nrt, 2, 32, 4, 4).to(torch.int64)   # [kc, rt, jq, lane, word, byte]
    lw = lq[..., 0] | (lq[..., 1] << 8) | (lq[..., 2] << 16) | (lq[..., 3] << 24)
    nq = blk[:, :, 1024:].reshape(kcn, nrt, 32, 4, 4).to(torch.int64)      # [kc, rt, lane, j, byte]
    nw = nq[..., 0] | (nq[..., 1] << 8) | (nq[..., 2] << 16) | (nq[..., 3] << 24)
    mask32 = 0xFFFFFFFF
    out = torch.zeros(n, k, dtype=torch.int64, device=dev)
    lane = torch.arange(32, device=dev)
    g, t = lane // 4, lane % 4
    rt_idx = torch.arange(nrt, device=dev)[None, :, None]
    kc_idx = torch.arange(kcn, device=dev)[:, None, None]
    for j in range(4):
        la = lw[:, :, j // 2, :, (j & 1) * 2]
        lb = lw[:, :, j // 2, :, (j & 1) * 2 + 1]
        nwj = nw[:, :, :, j]
        ha = (nwj & 0x87878787) | 0x38383838                               # LOP3
        rot = ((nwj >> 4) | (nwj << 28)) & mask32                           # SHF (funnel shift = rotate)
        hb = (rot & 0x87878787) | 0x38383838                               # LOP3
        regs = [_byte_perm(la, ha, 0x5140), _byte_perm(la, ha, 0x7362),    # 4 x PRMT
                _byte_perm(lb, hb, 0x5140), _byte_perm(lb, hb, 0x7362)]
        # a0: (g, 2t..2t+1), a1: (g+8, 2t..), a2: (g, 2t+8..), a3: (g+8, 2t+8..)
        for ri, (dr, dc) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
            for h in range(2):
                v = (regs[ri] >> (16 * h)) & 0xFFFF                        # [kc, rt, lane]
                rr = rt_idx * 16 + g[None, None, :] + dr
                cc = kc_idx * 64 + j * 16 + 2 * t[None, None, :] + dc + h
                out[rr.expand_as(v), cc.expand_as(v)] = v
    return out.to(torch.int32)


def unscale_bits(scaled: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Scaled BF16 patterns (exponent field in [112, 127]) -> original patterns, by subtracting the row shift from
    the exponent field. This is the exact integer form of multiplying by 2^-s."""
    e = (scaled >> 7) & 0xFF
    return (scaled & 0x807F) | ((e - shift[:, None]) << 7)


def shift_from_rscale(rscale: torch.Tensor) -> torch.Tensor:
    """rscale = 2^-s (fp32) -> s (int32)."""
    e = (rscale.view(torch.int32) >> 23) & 0xFF
    return (127 - e).to(torch.int32)
