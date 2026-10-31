"""Kernel layouts of the 12-bit format, PackedWeight, pack() and unpack_reference().

Three layouts are derived from the same encoding (format.encode_chunk_major):

  rowtile  K permuted inside each 64-column chunk, tiles stored row-tile-major [N/16][K/64][1536], one escape
           table per row tile. Used by the full-K kernels (fullk, fullk_tma) and the canonical layout.
  blocked  the rowtile tiles reordered for the split-K kernel: [N/(16 RT)][K/64][RT][1536] for RT in {8, 16}, so
           each CTA's share is one contiguous byte range; escapes as one table per row. A pure permutation of the
           rowtile bytes (same scales, same 12-bit codes).
  rowpair  the rowtile encoding of W viewed as V = W.reshape(8 N, K / 8): a 16-row tile of V is a pair of rows of W
           split into 8 segments, with one power-of-two scale per segment. Used by the m = 1 row GEMV.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import format as fmt

BLOCKED_RTS = (8, 16)


def k_permutation(k: int, device=None) -> torch.Tensor:
    """Stored column position p -> actual column. Inside a 64-column chunk, k-step j, mma column 2t + e + 8h holds
    column 16t + 4j + 2h + e, so lane t's activation operand for the chunk's 4 k-steps is the contiguous run
    [16t, 16t + 16): two 16-byte loads per token instead of eight 4-byte loads."""
    p = torch.arange(k, device=device)
    kc, c = p // 64, p % 64
    j, cc = c // 16, c % 16
    h, t, e = cc // 8, (cc % 8) // 2, cc % 2
    return kc * 64 + 16 * t + 4 * j + 2 * h + e


def pack_rowtile(w: torch.Tensor) -> dict:
    """[N, K] bf16 -> rowtile layout: wp uint8 [N/16 * K/64 * 1536], rscale f32 [N], tab int32 [N/16, emax, 4]
    ({count, 0, 0, 0}, then {actual column, row in tile, bits(w), bits(-2^-15 * 2^-s)} per escape), emax, n_esc."""
    n, k = w.shape
    perm = k_permutation(k, w.device)
    enc = fmt.encode_chunk_major(w[:, perm].contiguous())
    nrt, kcn = n // 16, k // 64
    wp = enc["wp"].view(kcn, nrt, fmt.BLOCK_BYTES).permute(1, 0, 2).contiguous().view(-1)
    esc = enc["esc"][: enc["n_esc"]]
    rows, kperm = esc[:, 0].long(), esc[:, 1].long()
    rt = rows // 16
    dev = w.device
    counts = torch.bincount(rt, minlength=nrt) if rows.numel() else torch.zeros(nrt, dtype=torch.long, device=dev)
    emax = int(counts.max().item()) + 1
    tab = torch.zeros((nrt, emax, 4), dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    if rows.numel():
        order = torch.argsort(rt, stable=True)
        rows, kperm, rt = rows[order], kperm[order], rt[order]
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=dev) - start[rt] + 1
        tab[rt, pos, 0] = perm[kperm].to(torch.int32)
        tab[rt, pos, 1] = (rows % 16).to(torch.int32)
        tab[rt, pos, 2] = esc[order, 2]
        tab[rt, pos, 3] = esc[order, 3]
    return {"wp": wp, "rscale": enc["rscale"], "tab": tab.contiguous(), "emax": emax, "n_esc": enc["n_esc"]}


def pack_rowpair(w: torch.Tensor) -> dict:
    """[N, K] bf16 -> rowpair layout (rowtile layout of W.reshape(8N, K/8)); needs N even and K % 512 == 0."""
    n, k = w.shape
    if n % 2 or k % 512:
        raise ValueError(f"rowpair layout needs N even and K % 512 == 0, got {tuple(w.shape)}")
    p = pack_rowtile(w.contiguous().reshape(n * 8, k // 8))
    p["ntiles"] = n // 2
    return p


def _row_escape_table(tab: torch.Tensor, n: int):
    """rowtile table [N/16, emax, 4] -> per-row table [N, emr, 4] {column or -1, bits(w), bits(nv), 0}, and emr."""
    nrt, emax = tab.shape[0], tab.shape[1]
    dev = tab.device
    cnt = tab[:, 0, 0].long()
    valid = torch.arange(emax - 1, device=dev)[None, :] < cnt[:, None]
    recs = tab[:, 1:, :][valid]
    rts = torch.arange(nrt, device=dev)[:, None].expand(nrt, emax - 1)[valid]
    rows = rts * 16 + recs[:, 1].long()
    counts = torch.bincount(rows, minlength=n) if rows.numel() else torch.zeros(n, dtype=torch.long, device=dev)
    emr = max(1, int(counts.max().item()) if rows.numel() else 1)
    er = torch.zeros((n, emr, 4), dtype=torch.int32, device=dev)
    er[:, :, 0] = -1
    if rows.numel():
        order = torch.argsort(rows, stable=True)
        rows, recs = rows[order], recs[order]
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=dev) - start[rows]
        er[rows, pos, 0] = recs[:, 0]
        er[rows, pos, 1] = recs[:, 2]
        er[rows, pos, 2] = recs[:, 3]
    return er.contiguous(), emr


def blocked_from_rowtile(rowtile: dict, n: int, k: int, rts=BLOCKED_RTS) -> dict:
    """rowtile layout -> blocked layout for the split-K kernel (one copy of the tiles per row-block height RT)."""
    nrt, kcn = n // 16, k // 64
    wpb = {}
    for rt in rts:
        if nrt % rt == 0:
            wpb[rt] = rowtile["wp"].view(nrt // rt, rt, kcn, fmt.BLOCK_BYTES).permute(0, 2, 1, 3).contiguous().view(-1)
    erow, emr = _row_escape_table(rowtile["tab"], n)
    return {"wpb": wpb, "erow": erow, "emr": emr}


@dataclass
class PackedWeight:
    """A BF16 weight [N, K] in the 12-bit format. `rowtile` is always present; `blocked` and `rowpair` only if
    requested at pack time (each adds about 0.75 bytes per weight per copy)."""
    n: int
    k: int
    rowtile: dict
    blocked: dict | None = None
    rowpair: dict | None = None
    tuned: dict = field(default_factory=dict)   # token bucket -> Config (filled by tune())

    @property
    def n_escapes(self) -> int:
        return self.rowtile["n_esc"]

    @property
    def layouts(self) -> tuple:
        return tuple(name for name in ("rowtile", "blocked", "rowpair") if getattr(self, name) is not None)

    @property
    def device(self) -> torch.device:
        return self.rowtile["wp"].device

    def nbytes(self, layout: str = "rowtile") -> int:
        """Bytes the kernels read for this layout: tiles + scales + escape table."""
        if layout == "rowtile":
            r = self.rowtile
            return r["wp"].numel() + r["rscale"].numel() * 4 + r["tab"].numel() * 4
        if layout == "rowpair":
            r = self.rowpair
            return r["wp"].numel() + r["rscale"].numel() * 4 + r["tab"].numel() * 4
        if layout.startswith("blocked"):
            rt = int(layout[len("blocked"):] or BLOCKED_RTS[0])
            b = self.blocked
            return b["wpb"][rt].numel() + self.rowtile["rscale"].numel() * 4 + b["erow"].numel() * 4
        raise ValueError(layout)

    def to(self, device) -> "PackedWeight":
        def mv(d):
            if d is None:
                return None
            out = {}
            for key, v in d.items():
                if isinstance(v, torch.Tensor):
                    out[key] = v.to(device)
                elif isinstance(v, dict):
                    out[key] = {kk: vv.to(device) for kk, vv in v.items()}
                else:
                    out[key] = v
            return out
        return PackedWeight(self.n, self.k, mv(self.rowtile), mv(self.blocked), mv(self.rowpair), dict(self.tuned))


def pack(w: torch.Tensor, layouts=("rowtile",)) -> PackedWeight:
    """Pack a BF16 weight W [N, K] (N % 16 == 0, K % 64 == 0) into the 12-bit format.

    layouts: any of "rowtile" (always built), "blocked" (split-K kernel), "rowpair" (m = 1 row GEMV), or "all".
    Packing runs wherever W lives; on a GPU it takes well under a second for the shapes in this repository."""
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise TypeError("pack expects a 2-D bfloat16 tensor")
    if layouts == "all":
        layouts = ("rowtile", "blocked", "rowpair")
    unknown = set(layouts) - {"rowtile", "blocked", "rowpair"}
    if unknown:
        raise ValueError(f"unknown layouts {sorted(unknown)}")
    n, k = w.shape
    rt = pack_rowtile(w)
    blocked = blocked_from_rowtile(rt, n, k) if "blocked" in layouts else None
    rowpair = pack_rowpair(w) if "rowpair" in layouts else None
    return PackedWeight(n, k, rt, blocked, rowpair)


def _unpack_rowtile_bits(wp: torch.Tensor, rscale: torch.Tensor, tab: torch.Tensor, n: int, k: int) -> torch.Tensor:
    nrt, kcn = n // 16, k // 64
    chunk_major = wp.view(nrt, kcn, fmt.BLOCK_BYTES).permute(1, 0, 2).contiguous().view(-1)
    scaled = fmt.decode_chunk_major(chunk_major, n, k)              # stored (permuted) column order
    bits_p = fmt.unscale_bits(scaled, fmt.shift_from_rscale(rscale))
    bits = torch.empty_like(bits_p)
    bits[:, k_permutation(k, wp.device)] = bits_p
    cnt = tab[:, 0, 0].long()
    valid = torch.arange(tab.shape[1] - 1, device=tab.device)[None, :] < cnt[:, None]
    recs = tab[:, 1:, :][valid]
    if recs.numel():
        rts = torch.arange(nrt, device=tab.device)[:, None].expand(nrt, tab.shape[1] - 1)[valid]
        rows = rts * 16 + recs[:, 1].long()
        bits[rows, recs[:, 0].long()] = (recs[:, 2] >> 16) & 0xFFFF
    return bits


def _to_bf16(bits: torch.Tensor) -> torch.Tensor:
    b = bits.to(torch.int32)
    return torch.where(b >= 2 ** 15, b - 2 ** 16, b).to(torch.int16).view(torch.bfloat16)


def unpack_reference(p: PackedWeight, layout: str = "rowtile") -> torch.Tensor:
    """Reconstruct W [N, K] bf16 bit-exactly from one layout, using the kernels' register-decode bit operations.

    This is a reference (slow, memory-hungry); it exists to verify the format, not to serve inference."""
    n, k = p.n, p.k
    if layout == "rowtile":
        r = p.rowtile
        return _to_bf16(_unpack_rowtile_bits(r["wp"], r["rscale"], r["tab"], n, k))
    if layout == "rowpair":
        r = p.rowpair
        v = _unpack_rowtile_bits(r["wp"], r["rscale"], r["tab"], 8 * n, k // 8)
        return _to_bf16(v.reshape(n, k))
    if layout.startswith("blocked"):
        rt = int(layout[len("blocked"):] or BLOCKED_RTS[0])
        b = p.blocked
        nrt, kcn = n // 16, k // 64
        wp = b["wpb"][rt].view(nrt // rt, kcn, rt, fmt.BLOCK_BYTES).permute(0, 2, 1, 3).contiguous().view(-1)
        empty = torch.zeros((nrt, 1, 4), dtype=torch.int32, device=wp.device)
        bits = _unpack_rowtile_bits(wp, p.rowtile["rscale"], empty, n, k)
        er = b["erow"]
        rows = torch.arange(n, device=er.device)[:, None].expand(n, er.shape[1])
        sel = er[:, :, 0] >= 0
        bits[rows[sel], er[:, :, 0][sel].long()] = (er[:, :, 1][sel] >> 16) & 0xFFFF
        return _to_bf16(bits)
    raise ValueError(f"unknown layout {layout!r}")
