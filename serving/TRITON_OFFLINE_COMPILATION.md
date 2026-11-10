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

The [actual compiler run](https://github.com/Soorya2201/Muse_optimiazation_H200_challenge/actions/runs/42768362132)
passed all25 specializations on commit446889a. Downloaded artifacts were
verified against the report:125 assembly/IR hashes,25 CUDA ELF machine190
cubins, Hopper PTX, matching source/tool/CI-commit identities and no GPU launch.
The report records Triton3.8.0 and bundled ptxas12.9.86. This compiler toolchain
is not an audit of the organizer's CUDA13 image. The durable private archive is
`research/serving/TRITON_AOT_ASSEMBLIES_2026-11-10.zip`, with reports
`TRITON_AOT_COMPILED_2026-11-10.json` and `TRITON_AOT_VERIFIED_2026-11-10.json`.
Coverage is14selector,9greedy and2map specializations; no runtime numerical
test occurred. The earlier local plan still truthfully records0compiled there.

The first extractor attempt lost the added source's final
newline before the next diff header; the existing source hash check rejected
it. Fixed that parser and retained the failed plan receipt. No candidate or
GPU kernel changed. The first CI run42768219289 then failed workflow parsing
before any job started: an unquoted pip flag contained a YAML colon delimiter.
Local parsing of the original workflow reproduces that error. A block scalar
fixed it, and the next source-changing push produced the successful run above.
The failed workflow receipt/log history is retained; it is not a compiler or
kernel failure. No live CI run remains from this experiment.

The workflow runs on this serving branch only when its compiler inputs change,
with read-only repository permissions and an8-minute job timeout. It preserves
failed/partial artifacts and never submits an OKC job. Follow any future
specific CI run to terminal; do not restart it after an observation timeout.

A successful compiler result proves code generation/assembly for the recorded
compiler and signatures. It does not prove numerical behavior, pointer bounds,
rank agreement, NCCL/SGLang behavior, CUDA graph replay, quality, TTFT or speed.
Those native/full-model gates remain required. Compiler artifacts must not be
installed into the official runtime or treated as a fastest composition.
