"""Interleaved A/B: stock marlin block-size heuristic vs the EP-aware one.

Both variants are captured as CUDA graphs once, then replayed alternately for
REPS rounds; the median per variant is reported, so clock drift hits both.
"""

import json
import statistics
import sys

import torch

import harness as H

REPS = 9

raw = H.make_raw_weights(seed=0, device="cuda")
layer = H.build_marlin_layer(raw)
ref = H.Reference(raw)
del raw


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


tokens = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else
                           "1,4,16,32,64,128,192,288,384,512,768,1024,2048,4096").split(",")]
out = {}
for t in tokens:
    x, scores, ids = H.make_inputs(t, seed=t, device="cuda")
    fns = {
        "stock": lambda: H.marlin_expert_outputs(layer, x, scores, ids),
        "ep": lambda: H.marlin_expert_outputs(layer, x, scores, ids, block_size_m="ep"),
    }
    got, local_ids = fns["ep"]()
    err = H.compare(got, ref(x, local_ids), local_ids) if t <= 1024 else {}
    graphs = {k: capture(f) for k, f in fns.items()}
    iters = 100 if t <= 1024 else 20
    samples = {k: [] for k in graphs}
    for _ in range(REPS):
        for k, g in graphs.items():
            samples[k].append(H.time_fn(g.replay, iters))
    med = {k: statistics.median(v) for k, v in samples.items()}
    spread = {k: (max(v) - min(v)) / statistics.median(v) for k, v in samples.items()}
    out[t] = {"median_us": med, "spread": spread, "rel_l2": err.get("rel_l2")}
    print(f"T={t:5d} stock {med['stock']:8.1f}  ep {med['ep']:8.1f}  "
          f"x{med['stock'] / med['ep']:4.2f}  spread {spread['stock']:.2f}/{spread['ep']:.2f}  "
          f"rel_l2={err.get('rel_l2', float('nan')):.4f}", flush=True)
    del graphs

with open("ab_blocksize.json", "w") as f:
    json.dump(out, f, indent=1)
