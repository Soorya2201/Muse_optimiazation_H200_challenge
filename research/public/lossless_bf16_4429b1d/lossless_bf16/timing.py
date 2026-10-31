"""Cold-L2 CUDA-graph timing and the per-shape tuner.

The method is the one used for every number in results/: capture one call in a CUDA graph; before each replay,
read (sum) a buffer larger than L2 so the weights come from DRAM; time each replay with CUDA events; report the
median. The read-based flush matters: zeroing a buffer instead leaves dirty lines whose write-back is charged to
the next kernel, which penalizes kernels that read more bytes.
"""
from __future__ import annotations

import torch

from . import configs as C
from .ops import _num_sms, launch, probe_exact
from .packing import PackedWeight

FLUSH_BYTES = 64 << 20      # 64 MiB, larger than the L2 of H100 / H200


def l2_flush_buffer(device="cuda", nbytes: int = FLUSH_BYTES) -> torch.Tensor:
    return torch.empty((nbytes // 4,), dtype=torch.int32, device=device).fill_(1)


def time_graph(fn, flush: torch.Tensor, replays: int = 24) -> float:
    """Median time in microseconds of `replays` cold-L2 replays of a CUDA graph holding one call of fn.
    The time includes the fixed cost of a graph launch (a few microseconds on H200)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for i in range(replays):
        flush.sum()
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[replays // 2]


def time_paired(fn_a, fn_b, flush: torch.Tensor, replays: int = 60) -> dict:
    """Alternate cold-L2 replays of two graphs (A then B, then B then A, ...) and return the median times and the
    median of the per-pair ratios t_A / t_B. Pairing cancels slow drifts (clocks, temperature) between the two."""
    graphs = []
    for fn in (fn_a, fn_b):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs.append(g)
    torch.cuda.synchronize()
    ta, tb = [], []
    for i in range(replays):
        order = (0, 1) if i % 2 == 0 else (1, 0)
        for which in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            s.record()
            graphs[which].replay()
            e.record()
            (ta if which == 0 else tb).append((s, e))
    torch.cuda.synchronize()
    a = [s.elapsed_time(e) * 1000.0 for s, e in ta]
    b = [s.elapsed_time(e) * 1000.0 for s, e in tb]
    ratios = sorted(x / y for x, y in zip(a, b))
    return {"a_us": sorted(a)[replays // 2], "b_us": sorted(b)[replays // 2], "ratio": ratios[replays // 2]}


def tune(p: PackedWeight, w: torch.Tensor | None, m: int, replays: int = 24, screen_replays: int = 5, top: int = 8,
         flush: torch.Tensor | None = None, verbose: bool = False) -> list:
    """Time every candidate configuration for m tokens and remember the fastest in p.tuned[bucket(m)].

    Candidates are screened with `screen_replays` replays; the best `top` are re-timed with `replays`. If w (the
    original BF16 weight, on the same GPU) is given, a configuration must first reproduce one-hot probes bit-exactly.
    Returns [(us, Config)] sorted, re-timed entries first."""
    dev = p.device
    flush = l2_flush_buffer(dev) if flush is None else flush
    x = torch.randn((m, p.k), device=dev).to(torch.bfloat16)
    out = torch.empty((m, p.n), dtype=torch.bfloat16, device=dev)
    cands = C.candidates(p.n, p.k, m, p.layouts, _num_sms(dev))
    screened = []
    for c in cands:
        if w is not None and probe_exact(p, w, c, m) < 1.0:
            if verbose:
                print(f"  {c}: one-hot probe not exact, skipped")
            continue
        us = time_graph(lambda c=c: launch(x, p, c, out), flush, screen_replays)
        screened.append((us, c))
    screened.sort(key=lambda t: t[0])
    timed = sorted(((time_graph(lambda c=c: launch(x, p, c, out), flush, replays), c) for _, c in screened[:top]),
                   key=lambda t: t[0])
    if timed:
        p.tuned[C.bucket(m)] = timed[0][1]
    if verbose:
        print("  " + " ".join(f"{c}={us:.2f}" for us, c in timed))
    return timed + screened[top:]
