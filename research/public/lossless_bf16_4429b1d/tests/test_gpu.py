"""GPU tests (skipped without a CUDA device of compute capability 9.0). They build the extension on first use.

For every candidate configuration of each shape: one-hot activations must reproduce the weight column bit-exactly
(random columns, and escaped columns for the splitk / rowpair kernels), and random activations must match an fp64
reference to within 5e-3 relative L2 error (BF16 output rounding alone gives about 1.7e-3)."""
import math

import pytest
import torch

cuda_ok = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = pytest.mark.skipif(not cuda_ok, reason="needs a CUDA device with compute capability 9.0 (Hopper)")

SHAPES = [(256, 1024), (2304, 8192), (8192, 1024), (1024, 4096)]


def synthetic(n, k, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn((n, k), generator=g, device=device) / math.sqrt(k)).to(torch.bfloat16)
    w[:, ::509] = 0                                                         # zero columns
    w[::7, 3::211] = (w[::7, 3::211].float() * 2.0 ** -16).to(torch.bfloat16)   # just below the window
    w[5, :40] = 0                                                           # a row with many escapes
    return w


@pytest.fixture(scope="module")
def ext():
    from lossless_bf16 import load_extension
    return load_extension()


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 5, 8, 13, 16, 29, 32])
def test_every_candidate(ext, n, k, m):
    from lossless_bf16 import candidates, pack, probe_exact
    from lossless_bf16.ops import _num_sms, launch
    w = synthetic(n, k)
    p = pack(w, layouts="all")
    x = torch.randn((m, k), device="cuda").to(torch.bfloat16)
    ref = x.double() @ w.double().T
    for c in candidates(n, k, m, p.layouts, _num_sms(w.device)):
        assert probe_exact(p, w, c, m) == 1.0, c
        if c.kernel in ("splitk", "rowpair"):
            assert probe_exact(p, w, c, m, escaped=True) == 1.0, c
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
        launch(x, p, c, out)
        torch.cuda.synchronize()
        err = float((out.double() - ref).norm() / ref.norm())
        assert err < 5e-3, (c, err)


def test_linear_default_and_graph(ext):
    from lossless_bf16 import linear, pack
    w = synthetic(2304, 8192, seed=1)
    p = pack(w, layouts="all")
    for m in (1, 8, 16, 32):
        x = torch.randn((m, 8192), device="cuda").to(torch.bfloat16)
        y = linear(x, p)                      # first call outside capture (workspace, attributes)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = linear(x, p)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, yg)
        ref = x.double() @ w.double().T
        assert float((y.double() - ref).norm() / ref.norm()) < 5e-3


def test_unpack_on_gpu_is_exact():
    from lossless_bf16 import pack, unpack_reference
    w = synthetic(1024, 4096, seed=2)
    p = pack(w, layouts="all")
    for layout in ("rowtile", "rowpair", "blocked8", "blocked16"):
        assert torch.equal(unpack_reference(p, layout).view(torch.int16), w.view(torch.int16)), layout
