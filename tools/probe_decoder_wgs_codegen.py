"""Compile a public, standalone decoder MoE kernel on Compiler Explorer.

No GPU timing or torch imports. Only the already public native source is sent.
The PyTorch host wrapper is replaced with a minimal kernel launch. This makes
the result a code-generation diagnostic, not the judge's linked extension.
"""
import argparse
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["base", "explicit_scale_reuse"])
args = parser.parse_args()
parent = ROOT / "muse_decode_layer/submission_v17_alygg77_6cae49a.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"
tree = ast.parse(parent.read_text())
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MOE_CUDA_SRC" for t in n.targets))
native = ast.literal_eval(node.value)
source = native.split("#include <torch/extension.h>")[0]
if args.arm == "explicit_scale_reuse":
    before = "    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);"
    after = before + "\n    uint32_t reused_scales[4];\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n      const uint32_t sw = j < 2 ? scs.x : scs.y, sb = (uint32_t)(j & 1) * 2u;\n      reused_scales[j] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n    }"
    assert source.count(before) == 1
    source = source.replace(before, after)
    before = "          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;"
    assert source.count(before) == 1
    source = source.replace(before, "          const uint32_t sgp = reused_scales[js];")
source += "\nvoid instantiate_public_kernel(okc::Params p, cudaStream_t stream) {\n  okc::moe_fused<false><<<132, okc::NTL, 200000, stream>>>(p);\n  okc::moe_fused<true><<<132, okc::NTL, 200000, stream>>>(p);\n}\n"
flags = "-O3 -std=c++17 -arch=sm_90a -Xptxas -v -DOKC_NS=3 -DOKC_FNST=4 -DOKC_NSF=3 -DOKC_NW=16 -DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_DQ7=1 -DOKC_PRMTSC=1"
request = {"source": source, "options": {"userArguments": flags,
    "compilerOptions": {"executorRequest": False}, "filters": {"binary": False,
    "execute": False, "intel": True, "demangle": False, "directives": False,
    "labels": True, "commentOnly": True, "libraryCode": False, "trim": False}}}
output = ROOT / f"muse_decode_layer/WGS_CODEGEN_{args.arm}_2026-10-30.json.gz"
assert not output.exists(), output
req = urllib.request.Request("https://godbolt.org/api/compiler/nvcc130/compile",
    data=json.dumps(request).encode(), headers={"Content-Type": "application/json", "Accept": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    response = json.load(r)
archive = {"scope": "Standalone public-kernel compiler diagnostic; no GPU timing; host extension wrapper replaced.",
    "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(native.encode()).hexdigest(),
    "compile_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "public_source": "https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0",
    "request": request, "response": response}
with gzip.open(output, "wt") as f:
    json.dump(archive, f)
print("ARCHIVE", output.relative_to(ROOT), "code", response.get("code"))
for line in response.get("stderr", []):
    print(line.get("text", "")[:400])
for name, dev in response.get("devices", {}).items():
    asm = dev.get("asm", [])
    instructions = collections.Counter()
    for row in asm:
        text = row.get("text", "").strip()
        # SASS instruction rows contain an address, then the instruction.
        if "*/" in text:
            parts = text.split("*/", 1)[1].strip().split()
            if parts:
                instructions[parts[0]] += 1
    print("DEVICE", name, "rows", len(asm), "opcode_counts", instructions.most_common(20))
