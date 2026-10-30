"""Add prepare-only independent-fixture integration checks to passing v28."""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_moe_w4a16/submission_v28_loadjit_validation.py"
target = ROOT / "muse_moe_w4a16/submission_v29_loadjit_integration_checks.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "41458546626ef0243a9378d40f86670bee9e35f73bb5548831c09edbb160253f"
assert not target.exists(), "preserve archived candidates"
suffix = '''

# All fixtures below are independent setup diagnostics. None is returned in
# state or used by kernel_fn. No challenge input is available to prepare.
import copy as _PortCopy
import json as _PortJSON

_validated_prepare_parent = prepare

def _port_integration_checks(state):
    dev = state["w"][0].device
    gen = torch.Generator(device=dev).manual_seed(10302026)
    worst = 0.0
    count = 0

    def error(a, b):
        return float((a.float() - b.float()).norm().item()) / max(float(b.float().norm().item()), 1e-12)

    def fixture(m, remote_only=False):
        x = torch.randn((m, 4096), device=dev, dtype=torch.bfloat16, generator=gen)
        ids = torch.randint(32 if remote_only else 0, 256, (m, 8), device=dev, dtype=torch.int32, generator=gen)
        if not remote_only:
            # Independent fixed routing ensures at least one local expert per row.
            ids[:, 0] = torch.arange(m, device=dev, dtype=torch.int32) % 32
        scores = torch.rand((m, 8), device=dev, dtype=torch.float32, generator=gen)
        scores /= scores.sum(dim=1, keepdim=True)
        return x, ids, scores

    def data(x, ids, scores, lo=0):
        return {"hidden_states": x, "topk_ids": ids, "topk_weights": scores,
                "config": {"local_start": lo, "local_count": 32}}

    for m in (1, 16, 33):
        x, ids, scores = fixture(m)
        reference = kernel_fn(data(x, ids, scores), state)
        torch.cuda.synchronize(dev)
        for lo in range(0, 256, 32):
            shifted_state = dict(state, lo=lo)
            shifted_ids = (ids + lo) % 256
            actual = kernel_fn(data(x, shifted_ids, scores, lo), shifted_state)
            torch.cuda.synchronize(dev)
            rel = error(actual, reference)
            assert rel <= 5e-4, ("expert-offset invariant", m, lo, rel)
            worst = max(worst, rel)
            count += 1

    x, ids, scores = fixture(3, remote_only=True)
    empty = kernel_fn(data(x, ids, scores), state)
    torch.cuda.synchronize(dev)
    assert torch.count_nonzero(empty).item() == 0, "remote-only routes must produce zero local output"

    # A real deep copy, not a shallow view of scratch. The fresh thread owns
    # this copy; its graph never overlaps with another user of that scratch.
    copied = _PortCopy.deepcopy(state)
    assert copied["scratch"][0].data_ptr() != state["scratch"][0].data_ptr()
    assert copied["w"][12].data_ptr() != state["w"][12].data_ptr()
    x, ids, scores = fixture(33)
    replacements = [x.clone(), torch.randn(x.shape, device=dev, dtype=x.dtype, generator=gen)]
    stream = torch.cuda.Stream(device=dev)
    torch.cuda.synchronize(dev)
    result, failures = {}, []

    def worker():
        try:
            ctx = _C.c_void_p()
            _context_check(_context_api().cuCtxGetCurrent(_C.byref(ctx)), "cuCtxGetCurrent(probe)")
            result["initial_context"] = bool(ctx.value)
            _bind_context(x)
            with torch.cuda.stream(stream):
                # Warm before capture, then replay with two fresh activation values.
                kernel_fn(data(x, ids, scores), copied)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = kernel_fn(data(x, ids, scores), copied)
                maximum = 0.0
                for replacement in replacements:
                    x.copy_(replacement)
                    graph.replay()
                    expected = kernel_fn(data(x, ids, scores), state)
                    stream.synchronize()
                    rel = error(out, expected)
                    assert rel <= 5e-4, ("fresh-thread copied-state graph", rel)
                    maximum = max(maximum, rel)
                result["graph_replays"] = len(replacements)
                result["graph_max_relerr"] = maximum
        except BaseException as exc:
            failures.append(exc)

    thread = _ContextThreading.Thread(target=worker)
    thread.start()
    thread.join()
    if failures:
        raise failures[0]
    torch.cuda.synchronize(dev)
    result.update(offset_cases=count, offset_max_relerr=worst, remote_only_zero=True,
                  independent_state_copy=True, prepare_only_fixtures=True)
    print("OKC_LOADJIT_INTEGRATION " + _PortJSON.dumps(result, sort_keys=True))


def prepare(weights):
    state = _validated_prepare_parent(weights)
    _port_integration_checks(state)
    return state
'''
text = parent.read_text() + suffix
ast.parse(text)
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in text.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], capture_output=True, text=True, check=True)
print(target.relative_to(ROOT), hashlib.sha256(text.encode()).hexdigest())
