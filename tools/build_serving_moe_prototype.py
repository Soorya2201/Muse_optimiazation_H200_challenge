"""Build an isolated, default-off load_jit port; do not install model hooks.

This preserves kernel-track norm semantics. See serving/MOE_PORT_STATUS.md
before attempting to connect it to SGLang's model.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT / "muse_moe_w4a16/submission_v24_alygg77_6cae49a.py"
CHECKOUT = ROOT / "serving/okc_init/sglang"
EXPECTED = "b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281"
assert hashlib.sha256(DONOR.read_bytes()).hexdigest() == EXPECTED
assert subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
).strip() == "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
source = DONOR.read_text()
tree = ast.parse(source)
assignments = {
    a.id: n for n in tree.body if isinstance(n, ast.Assign)
    for a in n.targets if isinstance(a, ast.Name)
}
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
cuda = ast.literal_eval(assignments["CUDA_SRC"].value)
def port_cuda(cuda):
    device, host = cuda.split("#include <torch/extension.h>", 1)
    device_original = device
    wrapped = []
    for line in device.splitlines(keepends=True):
        if len(line.rstrip("\n")) > 400:
            # Only inline-assembly operand lists need wrapping. No strings change.
            assert line.lstrip().startswith(": ") and "//" not in line
            parts = line.rstrip("\n").split(", ")
            current = parts.pop(0)
            for part in parts:
                if len(current) + len(part) + 2 > 350:
                    wrapped.append(current + ",\n")
                    current = "                 " + part
                else:
                    current += ", " + part
            wrapped.append(current + "\n")
        else:
            wrapped.append(line)
    device = "".join(wrapped)
    assert "".join(device.split()) == "".join(device_original.split())
    host = host.replace("#include <ATen/cuda/CUDAContext.h>", "")
    host = host.replace(
        "static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n\n"
        "void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }", ""
    )
    signature = (
        "void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n"
        "             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {"
    )
    assert signature in host
    args = [f"tvm::ffi::TensorView w{i}" for i in range(14)]
    args += [f"tvm::ffi::TensorView s{i}" for i in range(3)]
    args += [f"tvm::ffi::TensorView {v}" for v in ("x", "ids", "tw", "out")]
    args += ["int64_t lo", "int64_t nloc", "int64_t grid"]
    replacement = "void okc_moe_prototype_run(\n    " + ",\n    ".join(args) + ") {\n"
    replacement += "  using namespace sglang::host;\n"
    replacement += "  std::array<tvm::ffi::TensorView, 14> w = {" + ", ".join(f"w{i}" for i in range(14)) + "};\n"
    replacement += "  std::array<tvm::ffi::TensorView, 3> scratch = {s0, s1, s2};\n"
    replacement += '''  RuntimeCheck(nloc == 32 && lo >= 0 && lo + nloc <= 256, "invalid local expert shard");
      RuntimeCheck(grid > 0 && grid < 160, "grid exceeds stream-K scratch slots");
      for (auto tensor : w) {
        RuntimeCheck(tensor.device().device_type == kDLCUDA && tensor.IsContiguous(), "invalid weight view");
        RuntimeCheck(tensor.device().device_id == x.device().device_id, "weight device mismatch");
      }
      for (auto tensor : scratch) {
        RuntimeCheck(tensor.device().device_type == kDLCUDA && tensor.IsContiguous(), "invalid scratch view");
        RuntimeCheck(tensor.device().device_id == x.device().device_id, "scratch device mismatch");
      }
      RuntimeCheck(x.device().device_type == kDLCUDA, "CUDA activation required");
      RuntimeCheck(x.ndim() == 2 && x.size(1) == okc::HD, "invalid activation shape");
      RuntimeCheck(x.size(0) >= 1 && x.size(0) <= 384, "invalid chunk size");
      auto device = SymbolicDevice{};
      device.set_options<kDLCUDA>();
      TensorMatcher({x.size(0), okc::HD}).with_dtype<bf16_t>().with_device(device).verify(x);
      TensorMatcher({x.size(0), 8}).with_dtype<int32_t>().with_device(device).verify(ids);
      TensorMatcher({x.size(0), 8}).with_dtype<fp32_t>().with_device(device).verify(tw);
      TensorMatcher({x.size(0), okc::HD}).with_dtype<bf16_t>().with_device(device).verify(out);
      RuntimeCheck(x.IsContiguous() && ids.IsContiguous() && tw.IsContiguous() && out.IsContiguous(),
                   "contiguous call buffers required");
    '''
    host = host.replace(signature, replacement)
    host = host.replace("TORCH_CHECK(", "sglang::host::RuntimeCheck(")
    host = host.replace("ids.is_contiguous()", "ids.IsContiguous()")
    host = host.replace("cudaStream_t st = at::cuda::getCurrentCUDAStream();", "cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));")
    start = host.index("  if (g_okc_pdl) {")
    end = host.index("  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<", start)
    host = host[:start] + host[end:]  # Donor PDL=False; keep that launch behavior.
    host = host.replace(
        "    (void)e0; (void)e1;", "    RuntimeCheck(e0 == cudaSuccess && e1 == cudaSuccess, \"dynamic shared memory setup failed\");"
    )
    host = host.replace("  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;", "  p.dbg = nullptr;")
    host += '\n'
    notice = (
        "// Apache-2.0. Portions Copyright vjeux (f35a41e); alygg77 (6cae49a).\n"
        "// Isolated host port; device section changes only nine operand-list line wraps.\n"
        "// Prototype only: kernel-track norm offsets, no SGLang model integration.\n"
    )
    header = notice + "#pragma once\n#include <array>\n#include <sgl_kernel/tensor.h>\n#include <sgl_kernel/utils.h>\n#include <tvm/ffi/container/tensor.h>\n#include <tvm/ffi/extra/c_env_api.h>\n" + device + host
    header = header.rstrip() + "\n"
    return header, device_original, device


helper_cuda = next(
    ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
    and any(isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name)
            and a.value.id == "ALT_SRCS" and ast.literal_eval(a.slice) == "helper0"
            for a in n.targets)
)
header, device_original, device = port_cuda(cuda)
helper_header, helper_original, helper_device = port_cuda(helper_cuda)
constants = "\n\n".join(
    ast.get_source_segment(source, n) for n in tree.body
    if isinstance(n, ast.Assign) and 100 <= n.lineno <= 191
)
helpers = "\n\n".join(
    ast.get_source_segment(source, n) for n in tree.body
    if isinstance(n, ast.FunctionDef) and 3104 <= n.lineno <= 3359
)
helpers = helpers.replace("def prepare(weights):", "def _prepare_local_weights(weights):")
python = '''# Apache-2.0. Portions Copyright vjeux (f35a41e); alygg77 (6cae49a).
"""Isolated kernel-track MoE port. Not a model adapter; no serving claim.

Weights use 32 local experts in raw checkpoint order. post_norm_w is the
original BF16 offset and the kernel forms 1 + w in FP32. SGLang's cached
effective norm tensor is NOT interchangeable with this input.
"""
import os
from functools import lru_cache
import torch
from sglang.kernels.jit.utils import load_jit

'''
python += constants + '''

# The donor appends this after declaring ALT_SRCS; retain the effective tuple.
MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")

class _ModuleAdapter:
    def __init__(self, module):
        self.module = module

    def okc_run(self, weights, scratch, x, ids, scores, out, lo, count, grid):
        self.module.run(*weights, *scratch, x, ids, scores, out, lo, count, grid)


@lru_cache(None)
def _ext(defines=None):
    defines = tuple(MAIN_DEFINES if defines is None else defines)
    alternate = [d[1:] for d in defines if d.startswith("@")]
    if alternate not in ([], ["helper0"]):
        raise ValueError("unknown alternate source")
    flags = ["-O3", "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_DQ7=1"]
    flags += [d for d in defines if not d.startswith("@")]
    tag = "_".join(d.replace("-D", "").replace("=", "_").replace("@", "src_").lower() for d in defines)
    filename = "okc_moe_helper0.cuh" if alternate else "okc_moe_prototype.cuh"
    return _ModuleAdapter(load_jit(
        "muse_spark_okc_moe_prototype", tag,
        cuda_files=["muse_spark_v12/" + filename],
        cuda_wrappers=[("run", "okc_moe_prototype_run")],
        extra_cuda_cflags=flags,
    ))

'''
python += helpers + '''

_POS = _stored_pos()

def prepare_kernel_track_weights(weights, local_start=0):
    """Developer entry point, before Marlin repacks raw checkpoint tensors.

    Allocates several GB for ONE layer. Never call for every model layer.
    This prototype deliberately installs no automatic post-load hook.
    """
    if os.environ.get("SGLANG_OKC_MOE") != "1":
        raise RuntimeError("isolated prototype requires SGLANG_OKC_MOE=1")
    if local_start not in range(0, 256, 32):
        raise ValueError("local_start must identify an ep8 expert shard")
    dev = weights["w13"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        raise ValueError("this port targets Hopper sm90a")
    for name, shape, dtype in (
        ("w13", (32, 8192, 2048), torch.uint8),
        ("w2", (32, 4096, 2048), torch.uint8),
        ("w13_sf", (32, 8192, 256), torch.float8_e4m3fn),
        ("w2_sf", (32, 4096, 256), torch.float8_e4m3fn),
        ("w13_gs", (32,), torch.float32),
        ("w2_gs", (32,), torch.float32),
        ("post_norm_w", (4096,), torch.bfloat16),
    ):
        tensor = weights[name]
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != dev:
            raise ValueError(f"unexpected raw local weight {name}")
    with torch.cuda.device(dev):
        state = _prepare_local_weights(weights)
    state["lo"] = local_start
    return state


def run_kernel_track_moe(x, ids, scores, state):
    """Local weighted sum only; caller owns any cross-rank collective."""
    if os.environ.get("SGLANG_OKC_MOE") != "1":
        raise RuntimeError("isolated prototype requires SGLANG_OKC_MOE=1")
    if x.ndim != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        raise ValueError("expected BF16 [tokens, 4096] activation")
    m = x.shape[0]
    if ids.shape != (m, 8) or scores.shape != (m, 8):
        raise ValueError("expected eight global expert IDs and scores per token")
    if ids.dtype not in (torch.int32, torch.int64) or scores.dtype != torch.float32:
        raise ValueError("expected integer IDs and FP32 routing scores")
    if any(t.device != state["w"][0].device for t in (x, ids, scores)):
        raise ValueError("activation, routing and weights must share a device")
    if m == 0:
        return torch.empty((0, HIDDEN), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        return _kernel({
            "hidden_states": x, "topk_ids": ids, "topk_weights": scores,
            "config": {"local_start": state["lo"], "local_count": state["cnt"]},
        }, state)
'''
# Keep the actual variant dispatch/chunking intact, without the donor's context wrapper.
original_kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn")
python += "\n\n" + ast.get_source_segment(source, original_kernel).replace("def kernel_fn(", "def _kernel(", 1) + "\n"
assert "load_inline" not in python
ast.parse(python)
outputs = {
    "python/sglang/kernels/jit/csrc/muse_spark_v12/okc_moe_prototype.cuh": header,
    "python/sglang/kernels/jit/csrc/muse_spark_v12/okc_moe_helper0.cuh": helper_header,
    "python/sglang/kernels/ops/muse_spark_v12/okc_moe_prototype.py": python,
}
for rel, content in outputs.items():
    assert max(map(len, content.splitlines())) <= 400, rel
    target = CHECKOUT / rel
    if target.exists():
        assert target.read_text() == content, "refuse to overwrite modified generated port"
        continue
    patch = f"*** Begin Patch\n*** Add File: {target}\n"
    patch += "\n".join("+" + line for line in content.splitlines()) + "\n*** End Patch\n"
    subprocess.run(["apply_patch", patch], check=True, capture_output=True)
manifest = {
    "status": "offline isolated prototype; not compiled on CUDA, not integrated into model",
    "sglang_commit": "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f",
    "donor": str(DONOR.relative_to(ROOT)), "donor_sha256": EXPECTED,
    "donor_device_section_sha256": hashlib.sha256(device_original.encode()).hexdigest(),
    "ported_device_section_sha256": hashlib.sha256(device.encode()).hexdigest(),
    "device_section_changes": "whitespace only; nine inline-assembly operand lists wrapped for 400-character patch limit",
    "helper0_donor_device_sha256": hashlib.sha256(helper_original.encode()).hexdigest(),
    "helper0_ported_device_sha256": hashlib.sha256(helper_device.encode()).hexdigest(),
    "outputs": {rel: hashlib.sha256(content.encode()).hexdigest() for rel, content in outputs.items()},
}
print(json.dumps(manifest, indent=2))
