# Extract from rohanarun/Auto-research-agent at 018b7a102348ea3b73701e8be52d24f82d8c91ee
# Copyright (c) 2026 Rohan Arun; MIT, see LICENSE.
# Source header also credits the Apache-2.0 vjeux base.
import triton
import triton.language as tl

@triton.jit
def rq_deep_tma_pooler(DX, DW, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     EVEN: tl.constexpr = False):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(K // BK):
        a = DX.load([pm * BM, kb * BK])
        b = DW.load([kb * BK, pn * BN])
        acc = tl.dot(a, b, acc)
    mm = pm * BM + tl.arange(0, BM)
    nn = pn * BN + tl.arange(0, BN)
    if EVEN:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16),
                 (mm[:, None] < M) & (nn[None, :] < N))
