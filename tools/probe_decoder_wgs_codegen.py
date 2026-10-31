"""Compile a public, standalone decoder MoE kernel on Compiler Explorer.

No GPU timing or torch imports. Only the already public native source is sent.
The PyTorch host wrapper is replaced with a minimal kernel launch. This makes
the result a code-generation diagnostic, not the judge's linked extension.
"""
import argparse
import ast
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from decoder_wgs_variants import register_split

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["base", "explicit_scale_reuse", "regsplit_gs2", "regsplit_gs4"])
parser.add_argument("--instantiation", choices=["both", "true", "false"], default="both")
parser.add_argument("--binary-host", action="store_true",
                    help="Avoid host fatbinary assembly; request object disassembly plus device views")
parser.add_argument("--internal-helpers", action="store_true",
                    help="Prevent CE's --keep-device-functions retaining unused external helpers")
parser.add_argument("--fixed-ntt", type=int, choices=range(1,9),
                    help="Diagnostic only: compile one token-tile width, never a runnable general kernel")
parser.add_argument("--explicit-gencode", action="store_true", default=True,
                    help="Match the judge's compute_90a -> sm_90a only, avoiding generic sm_90 fallback")
parser.add_argument("--legacy-arch-shorthand", action="store_true",
                    help="Reproduce the invalid generic compute_90 fallback in old diagnostics")
args = parser.parse_args()
parent = ROOT / "muse_decode_layer/submission_v17_alygg77_6cae49a.py"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"
tree = ast.parse(parent.read_text())
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MOE_CUDA_SRC" for t in n.targets))
native = ast.literal_eval(node.value)
source = native.split("#include <torch/extension.h>")[0]
if args.arm.startswith('regsplit_gs'):
    source = register_split(source, int(args.arm[-1]))
if args.arm == "explicit_scale_reuse":
    before = "    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);"
    after = before + "\n    uint32_t reused_scales[4];\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n      const uint32_t sw = j < 2 ? scs.x : scs.y, sb = (uint32_t)(j & 1) * 2u;\n      reused_scales[j] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n    }"
    assert source.count(before) == 1
    source = source.replace(before, after)
    before = "          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;"
    assert source.count(before) == 1
    source = source.replace(before, "          const uint32_t sgp = reused_scales[js];")
if args.internal_helpers:
    source = re.sub(r'^__device__\b','static __device__',source,flags=re.MULTILINE)
if args.fixed_ntt:
    start = source.index('    switch (ntt) {')
    end = source.index('\n    gs += s1 - s0;',start)
    assert source[start:end].strip().endswith('}')
    source = source[:start] + f'    wgs_seg<PB, {args.fixed_ntt}>(p, S, ring, u, s0, s1, slot, seq, pd);' + source[end:]
source += "\nvoid instantiate_public_kernel(okc::Params p, cudaStream_t stream) {\n"
for big in (['false','true'] if args.instantiation=='both' else [args.instantiation]):
    source += f"  okc::moe_fused<{big}><<<132, okc::NTL, 200000, stream>>>(p);\n"
source += "}\n"
flags = "-O3 -std=c++17 -arch=sm_90a -Xptxas -v -DOKC_NS=3 -DOKC_FNST=4 -DOKC_NSF=3 -DOKC_NW=16 -DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_DQ7=1 -DOKC_PRMTSC=1"
if args.explicit_gencode and not args.legacy_arch_shorthand:
    flags = flags.replace('-arch=sm_90a','-gencode=arch=compute_90a,code=sm_90a')
request = {"source": source, "options": {"userArguments": flags,
    "compilerOptions": {"executorRequest": False}, "filters": {"binary": args.binary_host,
    "execute": False, "intel": True, "demangle": False, "directives": False,
    "labels": True, "commentOnly": True, "libraryCode": False, "trim": False}}}
output = ROOT / (f"muse_decode_layer/WGS_CODEGEN_{args.arm}_2026-10-30.json.gz"
                 if args.instantiation=='both' else
                 f"muse_decode_layer/WGS_CODEGEN_{args.arm}_single_{args.instantiation}_2026-10-31.json.gz")
if args.binary_host:
    output = output.with_name(output.name.replace('.json.gz','_binary_host.json.gz'))
if args.internal_helpers:
    output = output.with_name(output.name.replace('.json.gz','_internal_helpers.json.gz'))
if args.fixed_ntt:
    output = output.with_name(output.name.replace('.json.gz',f'_fixed_ntt{args.fixed_ntt}.json.gz'))
if args.explicit_gencode and not args.legacy_arch_shorthand:
    output = output.with_name(output.name.replace('.json.gz','_explicit_gencode.json.gz'))
assert not output.exists(), output
req = urllib.request.Request("https://godbolt.org/api/compiler/nvcc130/compile",
    data=json.dumps(request).encode(), headers={"Content-Type": "application/json", "Accept": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        response = json.load(r)
except (urllib.error.URLError, TimeoutError) as exc:
    response = {'code': None, 'transport_error': type(exc).__name__,
                'http_status': getattr(exc,'code',None),
                'stderr':[{'text':str(exc)}]}
archive = {"scope": "Standalone public-kernel compiler diagnostic; no GPU timing; host extension wrapper replaced.",
    "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "instantiation": args.instantiation,
    "binary_host": args.binary_host,
    "internal_helpers": args.internal_helpers,
    "fixed_ntt_diagnostic_only": args.fixed_ntt,
    "explicit_gencode": args.explicit_gencode and not args.legacy_arch_shorthand,
    "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(native.encode()).hexdigest(),
    "compile_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "public_source": "https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0",
    "request": request, "response": response}
with gzip.open(output, "wt") as f:
    json.dump(archive, f)
print("ARCHIVE", output.relative_to(ROOT), "code", response.get("code"))
stderr = response.get('stderr',[])
print('STDERR_LINES',len(stderr))
visible = stderr if len(stderr)<=45 else stderr[:15]+stderr[-30:]
for line in visible:
    print(line.get("text", "")[:400])
for name, dev in response.get("devices", {}).items():
    asm = dev.get("asm", [])
    instructions = collections.Counter()
    for row in asm:
        text = row.get("text", "").strip()
        # SASS instruction rows contain an address, then the instruction.
        if row.get('opcodes'):
            parts = text.split()
            opcode = next((p for p in parts if not p.startswith('@')),None)
            if opcode:
                instructions[opcode] += 1
        elif "*/" in text:
            parts = text.split("*/", 1)[1].strip().split()
            if parts:
                instructions[parts[0]] += 1
    clipped = any('[truncated' in row.get('text','') for row in asm)
    print("DEVICE", name, "rows", len(asm), "truncated", clipped,
          "displayed_opcode_counts", instructions.most_common(20))
