"""CPU source/patch checks. Does not validate CUDA compilation or performance."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
checkout = ROOT / "serving/okc_init/sglang"
manifest = json.loads((ROOT / "serving/MOE_PORT_MANIFEST.json").read_text())
donor = (ROOT / manifest["donor"]).read_text()
assert hashlib.sha256(donor.encode()).hexdigest() == manifest["donor_sha256"]
original = ast.parse(donor)
ported = checkout / "python/sglang/kernels/ops/muse_spark_v12/okc_moe_prototype.py"
new = ast.parse(ported.read_text())
for name, digest in manifest["outputs"].items():
    content = (checkout / name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest, name
    assert max(map(len, content.decode().splitlines())) <= 400, name
    if name.endswith("okc_moe_prototype.cuh"):
        header = content.decode()
    elif name.endswith("okc_moe_helper0.cuh"):
        helper_header = content.decode()
original_functions = {
    n.name: n for n in original.body if isinstance(n, ast.FunctionDef)
    and 3104 <= n.lineno <= 3359
}
# Match the current headers to the exact embedded sources exercised by OKC.
# This is source linkage only; numerical results still come from the receipts.
for version in (28, 29):
    candidates = list((ROOT / "muse_moe_w4a16").glob(f"submission_v{version}_loadjit*.py"))
    for candidate in candidates:
        candidate_tree = ast.parse(candidate.read_text())
        embedded = next(ast.literal_eval(n.value) for n in candidate_tree.body
                        if isinstance(n, ast.Assign) and any(isinstance(a, ast.Name)
                        and a.id == "_NATIVE_SOURCES" for a in n.targets))
        for name, content in embedded.items():
            expected = checkout / "python/sglang/kernels/jit/csrc/muse_spark_v12" / name
            assert content == expected.read_text(), (candidate, name)
new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
for name, node in original_functions.items():
    other = new_functions["_prepare_local_weights" if name == "prepare" else name]
    other.name = name
    assert ast.dump(node, include_attributes=False) == ast.dump(other, include_attributes=False), name
dispatch = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn")
new_dispatch = new_functions["_kernel"]
new_dispatch.name = "kernel_fn"
assert ast.dump(dispatch, include_attributes=False) == ast.dump(new_dispatch, include_attributes=False)
cuda = next(
    ast.literal_eval(n.value) for n in original.body if isinstance(n, ast.Assign)
    and any(isinstance(a, ast.Name) and a.id == "CUDA_SRC" for a in n.targets)
)
device = cuda.split("#include <torch/extension.h>")[0]
ported_device = header[header.index("\n// ---------------------------------------------------------------- roadmap"):]
ported_device = ported_device[:ported_device.index("\n\n\nstatic size_t okc_smem_bytes()")]
assert "".join(device.split()) == "".join(ported_device.split())
helper_cuda = next(
    ast.literal_eval(n.value) for n in original.body if isinstance(n, ast.Assign)
    and any(isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name)
            and a.value.id == "ALT_SRCS" and ast.literal_eval(a.slice) == "helper0"
            for a in n.targets)
)
helper_device = helper_cuda.split("#include <torch/extension.h>")[0]
ported_helper = helper_header[helper_header.index("\n// ---------------------------------------------------------------- roadmap"):]
ported_helper = ported_helper[:ported_helper.index("\n\n\nstatic size_t okc_smem_bytes()")]
assert "".join(helper_device.split()) == "".join(ported_helper.split())
# Compare final settings, including late augmented assignments, without running
# imports, native compilation, diagnostics or arbitrary donor module code.
def variant_settings(nodes):
    namespace = {}
    selected = []
    for node in nodes:
        if isinstance(node, ast.Assign) and 100 <= node.lineno <= 191:
            selected.append(node)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "MID_DEFINES":
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<variant settings>", "exec"), namespace)
    return {k: v for k, v in namespace.items() if k.endswith("_DEFINES")}
expected_settings = variant_settings(original.body)
new_settings = {}
for node in new.body:
    if isinstance(node, ast.Assign) and any(isinstance(a, ast.Name) and a.id in expected_settings for a in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<port settings>", "exec"), new_settings)
    elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "MID_DEFINES":
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<port settings>", "exec"), new_settings)
assert expected_settings == {k: new_settings[k] for k in expected_settings}
assert "torch::" not in header and "at::cuda" not in header
assert "TVMFFIEnvGetStream(kDLCUDA, x.device().device_id)" in header
# The flag must reject both public developer entry points before touching a tensor.
import os
namespace = {"os": os}
for name in ("prepare_kernel_track_weights", "run_kernel_track_moe"):
    node = new_functions[name]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ported), "exec"), namespace)
old_flag = os.environ.pop("SGLANG_OKC_MOE", None)
try:
    for name, args in (("prepare_kernel_track_weights", ({},)), ("run_kernel_track_moe", (None, None, None, None))):
        try:
            namespace[name](*args)
        except RuntimeError as error:
            assert "SGLANG_OKC_MOE=1" in str(error)
        else:
            raise AssertionError(f"{name} executed with flag absent")
finally:
    if old_flag is not None:
        os.environ["SGLANG_OKC_MOE"] = old_flag
patch = ROOT / "serving/okc_init/patch.diff"
assert patch.stat().st_size <= 720 * 1024
added_lines = [l[1:] for l in patch.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]
assert max(map(len, added_lines)) <= 400
paths = [l.split(" b/", 1)[1] for l in patch.read_text().splitlines() if l.startswith("diff --git ")]
assert sorted(paths) == sorted(manifest["outputs"]), paths
assert all(p.startswith("python/sglang/") and "/model_loader/" not in p for p in paths)
subprocess.run(["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)], check=True)
print(f"PASS: {len(original_functions)} packing/setup helpers, dispatch AST, final variant settings, both device sources, default-off guards, {len(paths)}-file {patch.stat().st_size}-byte patch")
print("Source checks do not validate GPU behavior; consult immutable run receipts. Model integration and serving throughput remain unvalidated.")
