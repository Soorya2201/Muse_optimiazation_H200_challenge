"""Embed the serving prototype's exact JIT sources in a kernel-track check.

This tests the native binding and numerical behavior, not a GPU algorithm gain.
Only our own compilation sources are written in the remote job.
"""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
checkout = ROOT / "serving/okc_init/sglang"
module = checkout / "python/sglang/kernels/ops/muse_spark_v12/okc_moe_prototype.py"
source = module.read_text()
tree = ast.parse(source)
files = {
    name: (checkout / "python/sglang/kernels/jit/csrc/muse_spark_v12" / name).read_text()
    for name in ("okc_moe_prototype.cuh", "okc_moe_helper0.cuh")
}
first = '''# okc: problem=muse_moe_w4a16
# Apache-2.0. Portions Copyright vjeux (f35a41e); alygg77 (6cae49a).
# Context binding attribution: Matsvei (kernarc matvey), via alygg77.
# Validation of a native binding port; no algorithmic speedup claim.
import tempfile
from pathlib import Path
'''
first += "_NATIVE_SOURCES = " + repr(files) + "\n"
first += '''
_SOURCE_DIRECTORY = None

def _jit_source(name):
    global _SOURCE_DIRECTORY
    if _SOURCE_DIRECTORY is None:
        _SOURCE_DIRECTORY = tempfile.TemporaryDirectory(prefix="okc_moe_loadjit_")
        for filename, content in _NATIVE_SOURCES.items():
            (Path(_SOURCE_DIRECTORY.name) / filename).write_text(content)
    return str(Path(_SOURCE_DIRECTORY.name) / name)

'''
nodes = []
for n in tree.body:
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
        continue  # Prototype module docstring is not the kernel-track contract.
    if isinstance(n, ast.FunctionDef) and n.name in ("prepare_kernel_track_weights", "run_kernel_track_moe"):
        continue
    nodes.append(n)
body = "\n\n".join(ast.get_source_segment(source, n) for n in nodes)
needle = 'cuda_files=["muse_spark_v12/" + filename]'
assert body.count(needle) == 1
body = body.replace(needle, 'cuda_files=[_jit_source(filename)]')
donor = (ROOT / "muse_moe_w4a16/submission_v24_alygg77_6cae49a.py").read_text()
context = donor[donor.index("# Per-thread primary-context bind"):]
context = context.replace("_context_parent_kernel = kernel_fn", "_context_parent_kernel = _kernel")
text = first + body + "\n\nprepare = _prepare_local_weights\n\n" + context
ast.parse(text)
target = ROOT / "muse_moe_w4a16/submission_v27_loadjit_validation.py"
assert not target.exists(), "do not overwrite an archived GPU candidate"
patch = "*** Begin Patch\n*** Add File: " + str(target) + "\n"
patch += "\n".join("+" + line for line in text.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], capture_output=True, text=True, check=True)
print(target.relative_to(ROOT), hashlib.sha256(text.encode()).hexdigest())
