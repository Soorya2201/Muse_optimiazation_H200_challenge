"""Measure existing cubic exp on the selected prefill attention pipeline.

Diagnostic timed dispatch stays on parent variant43. Only independent
prepare fixtures exercise the new combinations. No inputs/graphs persist.
"""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_prefill_layer/submission_v20_context_bind.py"
target = ROOT / "muse_prefill_layer/submission_v21_hybrid_pipeline_probe.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5"
assert not target.exists(), "do not overwrite archived candidates"
suffix = '''

# Existing Apache-2.0 vjeux FP32 cubic, combined with the already selected
# RRS/PIPE full-attention schedule. This does not port Blackwell instructions.
import json as _PipeJSON
import math as _PipeMath
import statistics as _PipeStats

_pipe_case = "    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;"
_pipe_attr = "  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);"
assert GLUE.count(_pipe_case) == 1 and GLUE.count(_pipe_attr) == 1
GLUE = GLUE.replace(_pipe_case, _pipe_case + "\\n" +
                    _pipe_case.replace("case 43:", "case 44:").replace("<3, 0,", "<3, 16,") + "\\n" +
                    _pipe_case.replace("case 43:", "case 45:").replace("<3, 0,", "<3, 32,"))
GLUE = GLUE.replace(_pipe_attr, _pipe_attr + "\\n" +
                    _pipe_attr.replace("<3, 0,", "<3, 16,") + "\\n" +
                    _pipe_attr.replace("<3, 0,", "<3, 32,"))
# Both IDs keep bit8 set: the C++ dispatch's NWG computation therefore stays3.
assert (44 & 8) and (45 & 8)
ATTN_VARIANTS.update({44: (3, 16), 45: (3, 32)})
assert ATTN_FIXED == {"slide": 8, "full_long": 43, "full_short": 43, "full_batch": 43}

_pipe_parent_prepare = prepare

def _pipe_capture(st, data, cfg, q8, qkv, variant, nsplit):
    # Fresh captured intermediate buffers per arm, including an independent
    # duplicate parent graph to expose allocation/address or order sensitivity.
    def run():
        return _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _pipe_measure(graphs, group, flush, sink, reps=9):
    names = list(graphs)
    times = {name: [] for name in names}
    pairs = []
    for rep in range(reps):
        shift = (rep + 2 * group) % len(names)
        order = names[shift:] + names[:shift]
        if (rep + group) % 2:
            order.reverse()
        sample = {}
        for name in order:
            # Untimed cold-L2 sweep. Events bracket only graph replay.
            sink.copy_(flush.sum(dtype=torch.float32))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) * 1000.0
            sample[name] = elapsed
            times[name].append(elapsed)
        pairs.append(sample)
    aa = [p["a0"] / p["a1"] for p in pairs]
    ratios = {}
    for name in names:
        if name in ("a0", "a1"):
            continue
        values = [_PipeMath.sqrt(p["a0"] * p["a1"]) / p[name] for p in pairs]
        ratios[name] = {"median": _PipeStats.median(values), "min": min(values), "max": max(values)}
    return {"medians_us": {k: _PipeStats.median(v) for k, v in times.items()},
            "aa_ratio_median": _PipeStats.median(aa), "paired_speedup": ratios,
            "repetitions": reps}


def _pipe_probe(st):
    valid = _validate_attn(st, [43, 44, 45])
    assert 43 in valid, "parent full-attention pipeline failed independent validation"
    arms = [v for v in (44, 45) if v in valid]
    if not arms:
        print("PREFILL_PIPE_EXP_PROBE " + _PipeJSON.dumps({"valid": valid, "skipped": "no valid hybrid"}))
        return
    flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    geometries = (("t1f", 1, 753, 80000), ("t4f", 4, 753, 80000),
                  ("first0", 1, 16384, 0), ("first32768", 1, 16384, 32768),
                  ("first65536", 1, 16384, 65536))
    for group, seed in enumerate((3101, 7907)):
        for label, batch, length, context in geometries:
            data, cfg, q8, qkv = _synthetic_attn_case(batch, length, context, 0, 1, seed)
            nsplit = _attn_splits(cfg, data, 43, st["nsm"])
            graphs = {"a0": _pipe_capture(st, data, cfg, q8, qkv, 43, nsplit)}
            graphs["a1"] = _pipe_capture(st, data, cfg, q8, qkv, 43, nsplit)
            errors = {}
            parent = graphs["a0"][1].float()
            for variant in arms:
                pair = _pipe_capture(st, data, cfg, q8, qkv, variant, nsplit)
                output = pair[1].float()
                rel = float((output - parent).norm() / parent.norm().clamp_min(1e-12))
                finite = bool(torch.isfinite(output).all())
                errors[str(variant)] = {"relerr_vs_parent": rel, "finite": finite}
                if finite and rel < 0.01:
                    graphs[str(variant)] = pair
            result = _pipe_measure(graphs, group, flush, sink)
            result.update(case=label, group=group, seed=seed, batch=batch, q_len=length,
                          ctx=context, nsplit=nsplit, validation=errors, valid_small=valid)
            print("PREFILL_PIPE_EXP_PROBE " + _PipeJSON.dumps(result, sort_keys=True), flush=True)
            del graphs, parent, data, cfg, q8, qkv
    # All fixture graphs/outputs/cache data leave scope here. Timed dispatch
    # remains the original fixed variants; no probe selection is installed.


def prepare(weights):
    state = _pipe_parent_prepare(weights)
    _pipe_probe(state)
    assert state["attn_choice"] == ATTN_FIXED
    return state
'''
text = parent.read_text() + suffix
ast.parse(text)
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in text.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], capture_output=True, text=True, check=True)
print(target.relative_to(ROOT), hashlib.sha256(text.encode()).hexdigest())
