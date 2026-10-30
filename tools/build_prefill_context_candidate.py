"""Preserve the prefill GPU path and port the attributed fresh-thread context binder."""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_prefill_layer/submission_v17_vjeux_f35a41e.py"
donor = ROOT / "muse_moe_w4a16/submission_v24_alygg77_6cae49a.py"
target = ROOT / "muse_prefill_layer/submission_v20_context_bind.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "9a12398e9db70abb59ee3a914e185a5e8010e2557056ece22528f6b0b8c24217"
assert hashlib.sha256(donor.read_bytes()).hexdigest() == "b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281"
assert not target.exists(), "Do not overwrite an archived candidate"
mapping = {
    "_C": "_PFCTX_C", "_CONTEXT_API": "_PFCTX_API", "_CONTEXT_TLS": "_PFCTX_TLS",
    "_context_api": "_pfctx_api", "_context_check": "_pfctx_check", "_bind_context": "_pfctx_bind",
}

class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = mapping.get(node.id, node.id)
        return node

    def visit_Global(self, node):
        node.names = [mapping.get(name, name) for name in node.names]
        return node

    def visit_FunctionDef(self, node):
        node.name = mapping.get(node.name, node.name)
        return self.generic_visit(node)

tree = ast.parse(donor.read_text())
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
           ("_context_api", "_context_check", "_bind_context")]
assert len(helpers) == 3
helper_text = "\n\n".join(ast.unparse(Rename().visit(node)) for node in helpers)
prefix = parent.read_text()
assert "_PFCTX_" not in prefix and "_pfctx_" not in prefix
suffix = '''

# Per-thread CUDA context binding follows Matsvei (kernarc user matvey),
# as published by alygg77/okc-muse-spark-kernels at 6cae49a (Apache-2.0).
# The exact vjeux f35a41e parent/native code above is retained unchanged.
import ctypes as _PFCTX_C
import threading as _PFCTX_Threading
_PFCTX_TLS = _PFCTX_Threading.local()
_PFCTX_API = None
'''
suffix += "\n" + helper_text + "\n"
suffix += '''

_PFCTX_PREPARE, _PFCTX_KERNEL = prepare, kernel_fn


def _pfctx_thread_probe(weights, state):
    """Validate the driver bind and one already-selected Lt plan on independent data."""
    device = weights["qkv_gate"].device
    choices = state.get("lt_choice", {})
    key = next(iter(choices), None)
    if key is not None:
        tokens, name = key
        weight_name, out_f32 = LT_GEMMS[name]
        weight = state[weight_name]
        generator = torch.Generator(device=device).manual_seed(20301030)
        x = torch.randn((tokens, weight.shape[1]), generator=generator, device=device,
                        dtype=torch.bfloat16)
        out = torch.empty((tokens, weight.shape[0]), device=device,
                          dtype=torch.float32 if out_f32 else torch.bfloat16)
        ref = torch.mm(x, weight.t(), out_dtype=torch.float32) if out_f32 else torch.mm(x, weight.t())
        torch.cuda.synchronize(device)
    errors, observations = [], []

    def worker():
        try:
            api = _pfctx_api()
            before = _PFCTX_C.c_void_p()
            _pfctx_check(api.cuCtxGetCurrent(_PFCTX_C.byref(before)), "probe cuCtxGetCurrent")
            _pfctx_bind(weights["qkv_gate"])
            index = _PFCTX_C.c_int(-1)
            _pfctx_check(api.cuCtxGetDevice(_PFCTX_C.byref(index)), "probe cuCtxGetDevice")
            expected = device.index if device.index is not None else 0
            if index.value != expected:
                raise RuntimeError("prefill thread bound to wrong device")
            if key is not None:
                plan, algorithm = choices[key]
                state["lt"].lt_run(plan, algorithm, x, weight, out, state["lt_ws"])
                torch.cuda.synchronize(device)
            observations.append(bool(before.value))
        except Exception as error:
            errors.append(error)

    thread = _PFCTX_Threading.Thread(target=worker)
    thread.start()
    thread.join()
    if errors:
        raise RuntimeError("prefill fresh-thread context probe failed") from errors[0]
    if len(observations) != 1:
        raise RuntimeError("prefill context probe did not finish")
    if key is not None:
        error = _lt_rel_err(out, ref.float())
        if not error < LT_ALGO_TOL:
            raise RuntimeError(f"prefill fresh-thread Lt error {error}")
    else:
        error = None
    print(f"[pfctx] fresh-thread bind passed: initial_context={observations[0]}, Lt={key}, relerr={error}", flush=True)


def prepare(weights):
    _pfctx_bind(weights["qkv_gate"])
    state = _PFCTX_PREPARE(weights)
    _pfctx_thread_probe(weights, state)
    return state


def kernel_fn(data, state):
    _pfctx_bind(data["attn_input"])
    return _PFCTX_KERNEL(data, state)
'''
content = prefix + suffix
assert content.startswith(prefix)
parsed = ast.parse(content)
assert ast.dump(ast.Module(body=parsed.body[:len(ast.parse(prefix).body)], type_ignores=[]), include_attributes=False) == ast.dump(ast.parse(prefix), include_attributes=False)
patch = "*** Begin Patch\n*** Add File: " + str(target.relative_to(ROOT)) + "\n"
patch += "\n".join("+" + line for line in content.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch", patch], cwd=ROOT, check=True, capture_output=True)
assert target.read_text().startswith(prefix)
print("Parent prefix/native constants unchanged; syntax and helper-isolation checks pass")
print(hashlib.sha256(target.read_bytes()).hexdigest())
