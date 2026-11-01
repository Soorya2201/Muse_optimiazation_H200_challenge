"""Compile exact Smem layout only; full-body compiler service is inconclusive."""
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from moe_two_cta_wide_native import two_cta_wide_native

root = Path(__file__).resolve().parents[1]
parent = (root/'muse_moe_w4a16/submission.py').read_text()
original = next(ast.literal_eval(n.value) for n in ast.parse(parent).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
native = two_cta_wide_native(original)
end = native.index('\n};', native.index('struct Smem {'))+3
source = native[:end] + r'''
static_assert(sizeof(Smem)==10976);
constexpr unsigned metadata_bytes=((sizeof(Smem)+127)/128)*128;
static_assert(metadata_bytes==11008);
static_assert(metadata_bytes+2*18432+2*32768==113408);
static_assert(EMAX*3<=18432);
static_assert(NTL==320 && FT_T==64 && JMAX==88);
__global__ void layout_only(unsigned* out) {
  if(threadIdx.x==0) {out[0]=sizeof(Smem);out[1]=metadata_bytes;out[2]=113408;}
}
} // namespace okc
'''
flags = '-O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -DOKC_NS=3 -DOKC_NSF=3 -DOKC_NW=8 -DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_PRMTSC=1 -DOKC_DQ7=1 -Xptxas -v'
request = dict(source=source, options=dict(userArguments=flags, compilerOptions=dict(executorRequest=False),
                                         filters=dict(binary=True, execute=False, demangle=False, directives=True, labels=True, commentOnly=True, trim=True)))
try:
    req = urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile', data=json.dumps(request).encode(),
                                 headers={'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.urlopen(req, timeout=90) as r:
        response = json.load(r)
except (urllib.error.URLError, TimeoutError) as exc:
    response = dict(code=None, transport_error=repr(exc))
out = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           scope='NVCC structure-layout assertions only. Does not compile the full expert body or prove register budget, actual H200 residency, memory ordering, accuracy or performance.',
           native_sha256=hashlib.sha256(native.encode()).hexdigest(), request=request, response=response)
path = root/'muse_moe_w4a16/TWO_CTA_WIDE_LAYOUT_CODEGEN_2026-11-01.json.gz'
assert not path.exists()
with gzip.open(path, 'wt') as f:
    json.dump(out, f)
print(path.relative_to(root), 'code', response.get('code'))
for row in response.get('stderr', []):
    s = row.get('text', '')
    if any(x in s for x in ('error', 'Error', 'ptxas')):
        print(s[:300])
