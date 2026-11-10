# Compile the new kernels before a GPU run

The private repository now has a bounded CPU-runner workflow for actual
Triton-to-Hopper compilation. The local Mac has no CUDA, Linux container
runtime or CUDA compiler. Existing CPU protocol doubles do not prove that the
Triton kernels compile; this gate addresses that specific gap.

`tools/compile_serving_triton_offline.py` extracts the exact five JIT kernel
bodies from the frozen added-source diffs for02/04/09. It verifies full-source
hashes against the existing patch manifests and identical function ASTs after
extraction. It imports no host Torch wrappers and mocks no compiler/operator.
An explicit CUDA `GPUTarget('cuda',90,32)` compiles the functions without
launching them. The private artifact includes compiler IR, PTX, assembled CUDA
ELF cubins and a report of each input signature, source hash and output hash.
These binaries are validation artifacts, not part of a source submission.

The25 planned specializations cover FP16/BF16/FP32 value loads, int32/int64
IDs, ordinary/strided selector layouts, high vocabulary offsets, TP2/3/8
greedy selection and masked tails. This covers the five new kernels in these
three helpers; it does not compile stock Marlin or every runtime specialization.
The plain-greedy02 path remains a fallback rather than a first-day check.

The pinned SGLang pyproject specifies Torch2.14.1. That release's Linux PyPI
metadata requests Triton~=3.8.0, so the workflow pins Triton3.8.0 and the exact
CPython3.12/Linuxx86_64 wheel hash. It installs no Torch/model/image or GPU
driver. This is a compiler environment consistent with that declared
dependency; the deployed organizer image has not been inspected. Sources:
[Torch metadata](https://pypi.org/pypi/torch/2.14.1/json),
[Triton metadata](https://pypi.org/pypi/triton/3.8.0/json),
[Triton's AOT tool](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py).
Exact metadata/actions pins are in the compiler environment audit.

```sh
python tools/compile_serving_triton_offline.py --plan-only --output-dir /tmp/aot-plan-fresh
python tools/compile_serving_triton_offline.py --output-dir /tmp/aot-compile-fresh
```

Only source-plan mode has run at this initial checkpoint:25 planned,0 compiled,
no GPU execution. The first extractor attempt lost the added source's final
newline before the next diff header; the existing source hash check rejected
it. Fixed that parser and retained the failed plan receipt. No candidate or
GPU kernel changed.

The workflow runs on this serving branch only when its compiler inputs change,
with read-only repository permissions and an8-minute job timeout. It preserves
failed/partial artifacts and never submits an OKC job. Follow the specific CI
run to terminal after the push; do not restart it after an observation timeout.

A successful compiler result proves code generation/assembly for the recorded
compiler and signatures. It does not prove numerical behavior, pointer bounds,
rank agreement, NCCL/SGLang behavior, CUDA graph replay, quality, TTFT or speed.
Those native/full-model gates remain required. Compiler artifacts must not be
installed into the official runtime or treated as a fastest composition.
