"""Reproduce the isolated, self-contained v23 MoE experiment from checked sources."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_moe_w4a16/submission_v18_vjeux_f35a41e.py").read_text()
paired = (ROOT / "muse_moe_w4a16/submission_v22_paired_dequant_gate.py").read_text()
tree = ast.parse(paired)
helpers = "\n\n".join(ast.get_source_segment(paired, n) for n in tree.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name in ("_paired_run", "_paired_graph_times"))
assert len(helpers) > 1500
tail = r'''

# Local experiment: whole-W2 L2 prefetch during W13 on single-token calls.
# Parent: vjeux f35a41e (Apache-2.0); its notice/license are retained.
# This adds cache hints only. It does not change packing, precision, or routing.
_prepare_before_prefetch = prepare
_kernel_before_prefetch = kernel_fn
_W2_PREFETCH = r"""
  // At most 4 * 9 MiB of W2, leaving L2 capacity for live activations.
  // The helper warp already issues W13; issue W2 independently, with no waits.
  if (p.M == 1 && tid == 0 && S.njS <= 4) {
    for (int unit = blockIdx.x; unit < S.njS * RB_B; unit += gridDim.x) {
      const int job = unit / RB_B;
      const size_t block = (size_t)S.jE[job] * RB_B + (unit % RB_B);
      const unsigned char* src = p.wBu + block * UNIT_BYTES;
      asm volatile("cp.async.bulk.prefetch.L2.global%HINT% [%0], %1%OPERAND%;\n"
                   ::"l"(src), "r"(UNIT_BYTES)%POLICY% : "memory");
    }
  }
"""


def prepare(weights):
    import concurrent.futures
    import math
    import time
    started = time.perf_counter()
    base = _prepare_before_prefetch(weights)
    answer = {"base": base, "selected": None}
    if time.perf_counter() - started > 175:
        print("W2 prefetch: skip compilation; retain parent", flush=True)
        return answer
    try:
        definitions = {}
        for name, keep in (("default", False), ("keep", True)):
            hook = (_W2_PREFETCH.replace("%HINT%", ".L2::cache_hint" if keep else "")
                    .replace("%OPERAND%", ", %2" if keep else "")
                    .replace("%POLICY%", ', "l"(0x14F0000000000000ull)' if keep else ""))
            assert CUDA_SRC.count("  route(p, S);\n") == 1
            ALT_SRCS["w2_" + name] = CUDA_SRC.replace("  route(p, S);\n", "  route(p, S);\n" + hook)
            defs = tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0")
            # Keep W2 in L2; mark streaming W13 loads evict-first in the keep variant.
            definitions[name] = defs + (f"-DOKC_L2H={int(keep)}",) + base.get("sfx", ()) + ("@w2_" + name,)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            built = list(pool.map(_ext, definitions.values()))
        modules = dict(zip(definitions, built))
        control = _ext(tuple(TINY_DEFINES) + base.get("sfx", ()))
        dev = weights["w13"].device
        data_sets = [_gen_input(1, 40039 + 7919 * i, dev) for i in range(3)]
        touched = [int(torch.unique(d["topk_ids"][(d["topk_ids"] >= 0)
                   & (d["topk_ids"] < LOCAL_EXPERTS)]).numel()) for d in data_sets]
        print(f"W2 prefetch synthetic touched experts: {touched}", flush=True)
        flush = torch.zeros((64 << 20,), dtype=torch.float32, device=dev)
        best = 1.03
        for name, ext in modules.items():
            if time.perf_counter() - started > 270:
                print(f"W2 prefetch {name}: budget skip", flush=True)
                continue
            ratios = []
            worst = 0.0
            for data in data_sets:
                fn0 = lambda: _paired_run(data, base, control)
                fn1 = lambda: _paired_run(data, base, ext)
                old, new = fn0().float(), fn1().float()
                error = float((old - new).norm() / old.norm().clamp_min(1e-12))
                worst = max(worst, error if math.isfinite(error) else float("inf"))
                del old, new
                if worst > 0.002:
                    break
                for _ in range(2):
                    t0, t1 = _paired_graph_times(fn0, fn1, flush, repetitions=20)
                    ratios.append(t0 / t1)
            qualifies = (worst <= 0.002 and len(ratios) == 6
                         and all(math.isfinite(v) and v > best for v in ratios))
            print(f"W2 prefetch {name}: relerr={worst:.3e}, ratios={ratios}, qualifies={qualifies}", flush=True)
            if qualifies:
                best = min(ratios)
                answer["selected"] = ext
                answer["name"] = name
        del flush, data_sets
    except Exception as exc:
        answer["selected"] = None
        print(f"W2 prefetch: exception, retain parent: {str(exc)[:1000]}", flush=True)
    for buffer in base["scratch"]:
        buffer.zero_()
    print(f"W2 prefetch prepare {time.perf_counter()-started:.1f}s: {answer.get('name', 'parent')}", flush=True)
    return answer


def kernel_fn(data, state):
    if data["hidden_states"].shape[0] == 1 and state["selected"] is not None:
        return _paired_run(data, state["base"], state["selected"])
    return _kernel_before_prefetch(data, state["base"])
'''
target = "muse_moe_w4a16/submission_v23_w2_prefetch.py"
assert not (ROOT / target).exists(), "Do not overwrite an archived experiment"
source = parent.replace("okc_moe_w4a16_v216x_", "okc_moe_w4a16_v223_w2_") + "\n" + helpers + tail
ast.parse(source)
patch = "*** Begin Patch\n*** Add File: " + target + "\n" + "\n".join("+" + s for s in source.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], cwd=ROOT, check=True)
