"""Kernel configurations, the split-K launch plan, and the per-shape config selector.

Kernel families (CUDA namespace in csrc/ in brackets):
  rowpair    [zr] m = 1. Row-pair x 8-segment tiles, diagonal mma, many small CTAs.        layout: rowpair
  fullk      [zg] m <= 32. RT row tiles per CTA, whole K per CTA, 8 compute warps
             round-robin over 64-column chunks with direct 16-byte loads, plus an escape warp.  layout: rowtile
  fullk_tma  [zt] m <= 16. As fullk, but a producer warp fills an mbarrier ring with TMA bulk copies.
                                                                                               layout: rowtile
  splitk     [zs] m <= 32. One CTA per SM, equal byte shares of the (row block, chunk, row tile) sequence,
             fp32 partials, finalize kernel launched with programmatic dependent launch.   layout: blocked
"""
from __future__ import annotations

from dataclasses import dataclass

BLK = 1536
SPLITK_FIN_THREADS = 128

# id -> (NT token tiles of 8, RT row tiles per CTA, U chunks in flight per warp); as ZG_CFG_LIST in csrc/zg.cu
FULLK = {0: (1, 1, 4), 1: (1, 2, 2), 2: (2, 1, 4), 3: (2, 2, 2), 4: (4, 1, 2), 5: (4, 2, 1), 6: (1, 1, 2),
         7: (2, 4, 2), 8: (4, 2, 2), 10: (2, 4, 1)}
# id -> (NT, RT, ST ring stages of 8 chunks); as ZT_CFG_LIST in csrc/zt.cu
FULLK_TMA = {0: (1, 1, 4), 1: (1, 1, 6), 2: (1, 2, 3), 3: (2, 1, 4), 4: (2, 2, 3), 5: (1, 1, 8)}
# id -> (NW warps per CTA, WPT warps per tile, U chunks in flight, MINB min CTAs per SM); as ZR_CFG_LIST in
# csrc/zr.cu. Ids 9-11 of the source (PAIR = true: fp32 output summing the two rows of a tile) are not exposed.
ROWPAIR = {0: (8, 8, 2, 3), 1: (4, 4, 4, 4), 2: (16, 16, 1, 2), 3: (8, 8, 1, 4), 4: (4, 4, 2, 6),
           5: (8, 4, 2, 3), 6: (4, 2, 1, 8), 7: (4, 1, 2, 6), 8: (8, 2, 1, 4)}
# id -> (NT, NCW consumer warps, RW row tiles per warp, ST ring stages); RT = NCW * RW; as ZS_CFG_LIST in csrc/zs.cu
SPLITK = {0: (1, 8, 1, 16), 1: (2, 8, 1, 15), 2: (4, 8, 1, 13), 3: (1, 8, 2, 8), 4: (2, 8, 2, 8), 5: (4, 8, 2, 7),
          6: (4, 4, 4, 7), 7: (2, 4, 4, 8)}

KERNELS = {"rowpair": ROWPAIR, "fullk": FULLK, "fullk_tma": FULLK_TMA, "splitk": SPLITK}
LAYOUT_OF = {"rowpair": "rowpair", "fullk": "rowtile", "fullk_tma": "rowtile", "splitk": "blocked"}
MAX_M = 32


@dataclass(frozen=True)
class Config:
    kernel: str          # rowpair | fullk | fullk_tma | splitk
    cfg: int             # id in the kernel's table
    grid: int = 0        # splitk only: CTAs (0 = number of SMs)

    def __str__(self) -> str:
        grid = f"g{self.grid}" if self.grid else "gSM"
        return f"{self.kernel}{self.cfg}" + (grid if self.kernel == "splitk" else "")


def bucket(m: int) -> int:
    """Token bucket of m: 1, 8, 16 or 32."""
    if m < 1 or m > MAX_M:
        raise ValueError(f"the 12-bit kernels cover 1 <= m <= {MAX_M}, got m = {m}")
    return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32


def tokens(c: Config) -> int:
    """Largest m the configuration handles (one token tile)."""
    if c.kernel == "rowpair":
        return 1
    return 8 * KERNELS[c.kernel][c.cfg][0]


def fits(c: Config, n: int, k: int, m: int) -> bool:
    if c.cfg not in KERNELS[c.kernel] or m > tokens(c) or n % 16 or k % 64:
        return False
    if c.kernel == "rowpair":
        nw, wpt = ROWPAIR[c.cfg][0], ROWPAIR[c.cfg][1]
        return m == 1 and n % 2 == 0 and k % 512 == 0 and (n // 2) % (nw // wpt) == 0
    if c.kernel in ("fullk", "fullk_tma"):
        return (n // 16) % KERNELS[c.kernel][c.cfg][1] == 0
    _, ncw, rw, _ = SPLITK[c.cfg]
    return (n // 16) % (ncw * rw) == 0


def splitk_owner(b: int, G: int, U: int) -> int:
    return ((b + 1) * G + U - 1) // U - 1


def splitk_plan(n: int, k: int, cfg: int, G: int) -> dict:
    """Launch plan of the split-K kernel, as zs_gemm re-checks it: G clamped to [1, U] (U = N/16 * K/64 tiles),
    jmax (most CTAs sharing a row block = partial slots per row block), finalize threads per output position tpp
    (power of two) and partials per thread npt (4 or 8, tpp * npt >= jmax), workspace floats, and the min / max
    weight bytes of one CTA."""
    nt, ncw, rw, _ = SPLITK[cfg]
    rt = ncw * rw
    nrt, kcn = n // 16, k // 64
    if n % (16 * rt) or k % 64:
        raise ValueError(f"splitk cfg {cfg} needs N % {16 * rt} == 0 and K % 64 == 0")
    U = nrt * kcn
    G = max(1, min(int(G), U))
    nrb = nrt // rt
    jmax = max(splitk_owner(rb * kcn * rt + kcn * rt - 1, G, U) - splitk_owner(rb * kcn * rt, G, U) + 1
               for rb in range(nrb))
    tpp = 1
    while jmax > 8 * tpp:
        tpp *= 2
    npt = 4 if jmax <= 4 * tpp else 8
    P = nrt * nt * 32
    fg = -(-P * tpp // SPLITK_FIN_THREADS)
    blocks = [(c + 1) * U // G - c * U // G for c in range(G)]
    return {"G": G, "U": U, "nrb": nrb, "rt": rt, "jmax": jmax, "tpp": tpp, "npt": npt, "P": P, "fg": fg,
            "ws_floats": nrb * jmax * rt * nt * 32 * 4, "bmin": min(blocks) * BLK, "bmax": max(blocks) * BLK}


def candidates(n: int, k: int, m: int, layouts, num_sms: int) -> list:
    """Configurations worth timing for (n, k, m) given the packed layouts: the smallest token tile that holds m."""
    b = bucket(m)
    out = []
    if "rowpair" in layouts and m == 1:
        out += [Config("rowpair", c) for c in ROWPAIR]
    if "rowtile" in layouts:
        out += [Config("fullk", c) for c, v in FULLK.items() if 8 * v[0] == max(8, b)]
        if b <= 16:
            out += [Config("fullk_tma", c) for c, v in FULLK_TMA.items() if 8 * v[0] == max(8, b)]
    if "blocked" in layouts and m > 1:
        for c, v in SPLITK.items():
            if 8 * v[0] == max(8, b):
                out += [Config("splitk", c, g) for g in sorted({num_sms, 128})]
    return [c for c in out if fits(c, n, k, m)]


# Fastest 12-bit configuration per (N, K, token bucket) in the H200 runs (results/h200_same_process.csv, run R2;
# where runs disagreed the alternatives were within 2%). grid 0 = number of SMs (132 on H200).
MEASURED = {
    (2304, 8192, 1): Config("fullk_tma", 0), (2304, 8192, 8): Config("fullk", 6),
    (2304, 8192, 16): Config("splitk", 1, 0), (2304, 8192, 32): Config("splitk", 2, 128),
    (8192, 1024, 1): Config("rowpair", 6), (8192, 1024, 8): Config("fullk", 1),
    (8192, 1024, 16): Config("splitk", 1, 128), (8192, 1024, 32): Config("splitk", 2, 128),
    (4096, 8192, 1): Config("fullk", 6), (4096, 8192, 8): Config("fullk", 6),
    (4096, 8192, 16): Config("fullk", 3), (4096, 8192, 32): Config("fullk", 8),
    (8192, 4096, 16): Config("fullk", 10), (8192, 4096, 32): Config("fullk", 8),
    (1024, 4096, 1): Config("rowpair", 3), (1024, 4096, 8): Config("fullk", 6),
}


def default_config(n: int, k: int, m: int, layouts) -> Config:
    """Measured choice for known shapes when its layout is packed, else a fullk configuration (rowtile layout)."""
    b = bucket(m)
    c = MEASURED.get((n, k, b))
    if c is not None and LAYOUT_OF[c.kernel] in layouts and fits(c, n, k, m):
        return c
    order = {1: (6, 0, 1), 8: (6, 0, 1), 16: (3, 2, 10, 7), 32: (8, 5, 4)}[b]
    for cfg in order:
        c = Config("fullk", cfg)
        if fits(c, n, k, m):
            return c
    raise ValueError(f"no 12-bit configuration fits N={n}, K={k}, m={m}")
