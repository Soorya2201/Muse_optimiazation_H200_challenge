"""Build an immutable prepare-only TMA comparison; timed dispatch stays parent."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_bf16_proj/submission_v34_alygg77_6cae49a.py"
target = ROOT / "muse_bf16_proj/submission_v38_tma_layout_probe.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90"
assert not target.exists(), "Preserve immutable submitted candidates"
public = ROOT / "research/public/rohanarun_018b7a1"
suffix = '\n\n# Standalone BF16 TMA operator from rohanarun/Auto-research-agent\n'
suffix += '# commit 018b7a102348ea3b73701e8be52d24f82d8c91ee.\n'
suffix += '\n'.join('# ' + line for line in (public / 'LICENSE').read_text().splitlines()) + '\n'
suffix += (public / 'rq_deep_tma_pooler.py').read_text()
suffix += '''
from triton.tools.tensor_descriptor import TensorDescriptor as _RQDescriptor
import copy as _RQCopy
import json as _RQJson
import statistics as _RQStats
import math as _RQMath

_rq_parent_prepare = prepare
_rq_parent_kernel = kernel_fn
_RQ_PROBE_CONFIGS = {
    ("pooler", 128): (64, 64, 128, 6),
    ("pooler", 256): (64, 64, 128, 6),
    ("unpooler", 128): (64, 128, 128, 4),
    ("unpooler", 256): (128, 128, 64, 6),
}


def _rq_probe_run(x, wt, config):
    bm, bn, bk, stages = config
    m, k = x.shape
    n = wt.shape[1]
    assert k % bk == 0
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    dx = _RQDescriptor.from_tensor(x, [bm, bk])
    dw = _RQDescriptor.from_tensor(wt, [bk, bn])
    rq_deep_tma_pooler[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dx, dw, out, m, n, k, bm, bn, bk,
        EVEN=(m % bm == 0 and n % bn == 0),
        num_warps=4, num_stages=stages)
    return out


def _rq_capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _rq_accuracy(out, ref, hot):
    delta = out.double() - ref
    norms = ref.norm(dim=1)
    den = torch.maximum(norms, norms.mean() * 0.1).clamp_min(1e-30)
    rel = float(delta.norm() / ref.norm().clamp_min(1e-30))
    row = float((delta.norm(dim=1) / den).max())
    hoterr = float((delta[:hot].norm(dim=1) / den[:hot]).max())
    finite = bool(torch.isfinite(out).all())
    assert finite and rel <= 0.01 and row <= 0.05 and hoterr <= 0.001, (rel, row, hoterr)
    return [round(rel, 8), round(row, 8), round(hoterr, 8)]


def _rq_measure(graphs, flush, sink, group):
    times = [[], [], []]
    for rep in range(9):
        order = [(rep + group + i) % 3 for i in range(3)]
        if (rep + group) % 2:
            order.reverse()
        for i in order:
            # Cold L2 before EVERY arm. Events bracket only one graph replay.
            sink.copy_(flush.sum(dtype=torch.float32))
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[i][0].replay()
            b.record()
            b.synchronize()
            times[i].append(a.elapsed_time(b) * 1000.0)
    ratios = [_RQMath.sqrt(a * c) / b for a, b, c in zip(*times)]
    aa = _RQStats.median(a / c for a, _, c in zip(*times))
    # Raw event/graph latency, no guessed graph-cost subtraction. Ratios are
    # conservative for a faster arm. Preserve every paired event measurement.
    return {"us": [[round(t, 4) for t in ts] for ts in times],
            "r": round(_RQStats.median(ratios), 6), "aa": round(aa, 6)}


def _rq_probe(state):
    flush = torch.ones(64 << 20, dtype=torch.float16, device="cuda")
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    retained = []
    for pad_index, pad_mib in enumerate((0, 1, 3, 7)):
        # All pads/copies remain live to ensure four independent allocations.
        # Clones reuse compiled modules/plans, not prepare/autotuning results
        # from the probe. No fixture enters returned state.
        pad = torch.empty(pad_mib << 20, dtype=torch.uint8, device="cuda")
        st = _RQCopy.deepcopy(state)
        retained.append((pad, st))
        for (layer, m), config in _RQ_PROBE_CONFIGS.items():
            w = st["raw"][layer]
            wt = st["klarge_rawT"][layer]
            n, k = w.shape
            assert torch.equal(wt, w.T), "transpose must preserve every BF16 bit"
            for group, seed in enumerate((3119, 7919)):
                gen = torch.Generator(device=w.device).manual_seed(seed)
                x = torch.randn((m, k), device=w.device, dtype=torch.bfloat16, generator=gen)
                if group:
                    x[:, ::31] *= 16
                hot = max(1, m // 8)
                columns = torch.randint(k, (hot,), device=w.device, generator=gen)
                x[:hot].zero_()
                x[torch.arange(hot, device=w.device), columns] = 2.0
                data = {"x": x, "config": {"layer": layer, "m": m, "n": n, "k": k}}
                parent_fn = lambda: _rq_parent_kernel(data, st)
                candidate_fn = lambda: _rq_probe_run(x, wt, config)
                graphs = [_rq_capture(parent_fn), _rq_capture(candidate_fn), _rq_capture(parent_fn)]
                ref = x.double() @ w.double().T
                errors = [_rq_accuracy(g[1], ref, hot) for g in graphs[:2]]
                # Replays must read updated activation values, not stale output.
                x.mul_(-1)
                for graph, _ in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                for _, out in graphs:
                    _rq_accuracy(out, -ref, hot)
                record = _rq_measure(graphs, flush, sink, group + pad_index)
                record.update(p=pad_index, l=layer, m=m, s=seed, e=errors,
                              wm=wt.data_ptr() % (32 << 20), xm=x.data_ptr() % (32 << 20))
                print("BF16_TMA_PROBE " + _RQJson.dumps(record, separators=(",", ":")), flush=True)
                del graphs, ref, data, x
    # Everything above is independent synthetic validation and benchmarking.
    # No output/activation/graph/alternate state is returned or cached.


def prepare(weights):
    state = _rq_parent_prepare(weights)
    _rq_probe(state)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return state
'''
source = parent.read_text() + suffix
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT), hashlib.sha256(source.encode()).hexdigest())
