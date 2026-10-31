"""Exact 12-bit storage for BF16 weights, decoded in registers by Hopper (sm_90a) GEMV/GEMM kernels.

    from lossless_bf16 import pack, unpack_reference, linear
    p = pack(W.cuda(), layouts="all")       # W: [N, K] bfloat16, N % 16 == 0, K % 64 == 0
    assert torch.equal(unpack_reference(p).view(torch.int16), W.cuda().view(torch.int16))
    y = linear(x, p)                        # x: [m, K] bfloat16, 1 <= m <= 32
"""
from .configs import Config, bucket, candidates, default_config, splitk_plan
from .packing import PackedWeight, k_permutation, pack, unpack_reference

__all__ = ["Config", "PackedWeight", "bucket", "candidates", "default_config", "k_permutation", "linear",
           "load_extension", "pack", "probe_exact", "splitk_plan", "tune", "unpack_reference"]
__version__ = "0.1.0"


def __getattr__(name):
    # GPU entry points import lazily so the format code works on machines without CUDA.
    if name in ("linear", "load_extension", "probe_exact", "launch", "select"):
        from . import ops
        return getattr(ops, name)
    if name in ("tune", "time_graph", "time_paired", "l2_flush_buffer"):
        from . import timing
        return getattr(timing, name)
    raise AttributeError(name)
