"""Follow the measured three-WG regression with resource and two-WG controls."""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_prefill_layer/submission_v21_hybrid_pipeline_probe.py"
target = ROOT / "muse_prefill_layer/submission_v22_hybrid_resource_probe.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "1cda34bc6d5f67eeda2959288e2cca48365bced652bf4d9ac95c8802e2382952"
assert not target.exists()
base = ROOT / "muse_prefill_layer/submission_v20_context_bind.py"
base_source = base.read_text()
base_tree = ast.parse(base_source)
build_node = next(n for n in base_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
build = ast.get_source_segment(base_source, build_node)
assert build.count('"attn_set_prof"]') == 1
build = build.replace('"attn_set_prof"]', '"attn_set_prof", "attn_resources"]')
suffix = '''

# Resource diagnostics and a separate two-consumer-WG schedule control.
# Timed full-layer dispatch remains parent variant43 throughout this probe.
_two_cases = []
_two_attrs = []
for _variant, _exps in ((48, 0), (49, 16), (50, 32)):
    _two_cases.append(_pipe_case.replace("case 43:", f"case {_variant}:")
                      .replace("<3, 0,", f"<2, {_exps},").replace("dim3(512)", "dim3(384)"))
    _two_attrs.append(_pipe_attr.replace("<3, 0,", f"<2, {_exps},"))
    ATTN_VARIANTS[_variant] = (2, _exps)
    assert not (_variant & 8)
GLUE = GLUE.replace(_pipe_case, _pipe_case + "\\n" + "\\n".join(_two_cases))
GLUE = GLUE.replace(_pipe_attr, _pipe_attr + "\\n" + "\\n".join(_two_attrs))
GLUE += r\'\'\'
#include <vector>
std::vector<int64_t> k_attn_resources(int variant) {
  cudaFuncAttributes a = {};
  cudaError_t result = cudaErrorInvalidValue;
  switch (variant) {
    case 43: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>); break;
    case 44: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<3, 16, false, true, false, true>); break;
    case 45: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<3, 32, false, true, false, true>); break;
    case 48: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<2, 0, false, true, false, true>); break;
    case 49: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<2, 16, false, true, false, true>); break;
    case 50: result = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel<2, 32, false, true, false, true>); break;
  }
  return {(int64_t)variant, (int64_t)result, (int64_t)a.numRegs, (int64_t)a.localSizeBytes,
          (int64_t)a.sharedSizeBytes, (int64_t)a.maxDynamicSharedSizeBytes, (int64_t)a.maxThreadsPerBlock};
}
\'\'\'
CPP += r\'\'\'
std::vector<int64_t> k_attn_resources(int variant);
std::vector<int64_t> attn_resources(int64_t variant) { return k_attn_resources((int)variant); }
\'\'\'
'''
suffix += "\n\n" + build + "\n\n"
suffix += '''
def _pipe_probe(st):
    fields = ("variant", "cuda_error", "numRegs", "localSizeBytes", "sharedSizeBytes",
              "maxDynamicSharedSizeBytes", "maxThreadsPerBlock")
    resources = [dict(zip(fields, st["mod"].attn_resources(v))) for v in (43, 44, 45, 48, 49, 50)]
    print("PREFILL_PIPE_RESOURCES " + _PipeJSON.dumps(resources, sort_keys=True))
    assert all(row["cuda_error"] == 0 for row in resources)
    valid = _validate_attn(st, [43, 48, 49, 50])
    assert 43 in valid
    arms = [v for v in (48, 49, 50) if v in valid]
    flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    geometries = (("t1f", 1, 753, 80000), ("t4f", 4, 753, 80000),
                  ("first0", 1, 16384, 0), ("first32768", 1, 16384, 32768),
                  ("first65536", 1, 16384, 65536))
    for group, seed in enumerate((3101, 7907)):
        for label, batch, length, context in geometries:
            data, cfg, q8, qkv = _synthetic_attn_case(batch, length, context, 0, 1, seed)
            graphs = {}
            for name in ("a0", "a1"):
                ns = _attn_splits(cfg, data, 43, st["nsm"])
                graphs[name] = _pipe_capture(st, data, cfg, q8, qkv, 43, ns)
            errors, splits = {}, {"43": ns}
            parent = graphs["a0"][1].float()
            for variant in arms:
                ns = _attn_splits(cfg, data, variant, st["nsm"])
                splits[str(variant)] = ns
                pair = _pipe_capture(st, data, cfg, q8, qkv, variant, ns)
                output = pair[1].float()
                rel = float((output - parent).norm() / parent.norm().clamp_min(1e-12))
                finite = bool(torch.isfinite(output).all())
                errors[str(variant)] = {"relerr_vs_parent": rel, "finite": finite}
                if finite and rel < 0.01:
                    graphs[str(variant)] = pair
            result = _pipe_measure(graphs, group, flush, sink)
            result.update(case=label, group=group, seed=seed, batch=batch, q_len=length,
                          ctx=context, nsplit=splits, validation=errors, valid_small=valid)
            print("PREFILL_PIPE_EXP_PROBE " + _PipeJSON.dumps(result, sort_keys=True), flush=True)
            del graphs, parent, data, cfg, q8, qkv
'''
text = parent.read_text() + suffix
ast.parse(text)
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in text.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], capture_output=True, text=True, check=True)
print(target.relative_to(ROOT), hashlib.sha256(text.encode()).hexdigest())
