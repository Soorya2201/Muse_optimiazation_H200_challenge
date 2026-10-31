"""Benchmark the 12-bit kernels against cuBLAS (torch.nn.functional.linear) on cold-L2 CUDA-graph replays.

Method (the one behind results/h200_same_process.csv): one call captured in a CUDA graph; before every replay a
64 MiB buffer is read (sum) so the weights come from DRAM; each replay is timed with CUDA events; the median of
--replays replays is reported. Times include the fixed cost of a graph launch on both sides.

  python bench/bench_linear.py                    # the shapes in results/, measured default configurations
  python bench/bench_linear.py --tune             # time every candidate configuration first (a few seconds/shape)
  python bench/bench_linear.py --paired           # also: 60 alternating replays, median of per-pair ratios
  python bench/bench_linear.py --shape 4096x8192 --m 1 16 --csv out.csv

Weights are synthetic N(0, 1/K) in BF16, as in results/; the format's size and speed depend on the weights only
through the escape count.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lossless_bf16 as lb  # noqa: E402
from lossless_bf16.timing import l2_flush_buffer, time_graph, time_paired, tune  # noqa: E402

# (N, K, token counts) of the measured runs; 1024 x 4096 is the 1/8 tensor-parallel slice used at m <= 8
DEFAULT = [((2304, 8192), (1, 8, 16, 32)), ((8192, 1024), (1, 8, 16, 32)), ((4096, 8192), (1, 8, 16, 32)),
           ((8192, 4096), (16, 32)), ((1024, 4096), (1, 8))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", nargs="*", default=None, help="NxK, e.g. 4096x8192")
    ap.add_argument("--m", nargs="*", type=int, default=None, help="token counts (1..32)")
    ap.add_argument("--replays", type=int, default=24)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU (Hopper, sm_90)")
    dev = torch.device("cuda")
    if args.shape:
        ms = tuple(args.m or (1, 8, 16, 32))
        plan = [(tuple(int(v) for v in s.lower().split("x")), ms) for s in args.shape]
    else:
        plan = [(s, tuple(args.m) if args.m else ms) for s, ms in DEFAULT]
    props = torch.cuda.get_device_properties(dev)
    print(f"# {props.name}, {props.multi_processor_count} SMs, torch {torch.__version__}, CUDA {torch.version.cuda}")
    lb.load_extension()
    flush = l2_flush_buffer(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rows = []
    hdr = f"{'N x K':>11} {'m':>3} {'cuBLAS us':>10} {'12-bit us':>10} {'speedup':>8}  config"
    if args.paired:
        hdr += "   paired ratio"
    print(hdr)
    for (n, k), ms in plan:
        w = (torch.randn((n, k), generator=gen, device=dev) / math.sqrt(k)).to(torch.bfloat16)
        p = lb.pack(w, layouts="all")
        for m in ms:
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            if args.tune:
                tune(p, w, m, flush=flush)
            cfg = lb.select(p, m)
            if lb.probe_exact(p, w, cfg, m) < 1.0:
                sys.exit(f"{n}x{k} m={m} {cfg}: one-hot probe not bit-exact")
            y = lb.linear(x, p, cfg)
            ref = x.double() @ w.double().T
            err = float((y.double() - ref).norm() / ref.norm())
            t_ref = time_graph(lambda: F.linear(x, w), flush, args.replays)
            t_our = time_graph(lambda: lb.linear(x, p, cfg), flush, args.replays)
            row = {"n": n, "k": k, "m": m, "cublas_us": round(t_ref, 2), "ours_us": round(t_our, 2),
                   "speedup": round(t_ref / t_our, 3), "config": str(cfg), "rel_err": f"{err:.2e}",
                   "weight_MB_bf16": round(n * k * 2 / 1e6, 2), "weight_MB_12bit": round(p.nbytes() / 1e6, 2)}
            line = f"{n:>5}x{k:<5} {m:>3} {t_ref:>10.2f} {t_our:>10.2f} {t_ref / t_our:>7.3f}x  {cfg}"
            if args.paired:
                r = time_paired(lambda: F.linear(x, w), lambda: lb.linear(x, p, cfg), flush)
                row["paired_ratio"] = round(r["ratio"], 3)
                line += f"   {r['ratio']:.3f}"
            print(line)
            rows.append(row)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
