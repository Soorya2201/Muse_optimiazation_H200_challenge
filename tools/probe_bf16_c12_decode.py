"""CPU losslessness proof and standalone sm90a instruction comparison."""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import random
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / 'muse_bf16_proj/submission.py'
raw = parent.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90'
t = ast.parse(raw)
assignments = {n.targets[0].id: n.value for n in t.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name)}
native = ast.literal_eval(assignments['CUDA_SRC'])
old = native[native.index('__device__ __forceinline__ void decode8('):native.index('\ntemplate <typename T>')]
new = r'''
// Sign-bit layout and decoder adapted from the pinned MIT Nathen decodediet
// component already embedded in this parent, now applied to SK2 fragments.
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(nw));
  const uint32_t ha = (nw & 0x87878787u) + base4;
  const uint32_t hb = (rotated & 0x87878787u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
'''


def permute(c):
    # The second sign travels to the NEXT byte's bit3 so rotr4 restores it
    # to its original byte's bit7; the final byte wraps within this u32.
    s1 = c & 0x80808080
    return ((c & 0x77777777) | ((c & 0x08080808) << 4) | (s1 << 4) | (s1 >> 28)) & 0xffffffff


def inverse(c):
    s1 = c & 0x08080808
    return ((c & 0x77777777) | ((c & 0x80808080) >> 4) | (s1 >> 4) | (s1 << 28)) & 0xffffffff


def highbytes(c, base):
    a = (((c & 0x07070707) | ((c << 4) & 0x80808080)) + base) & 0xffffffff
    b = ((((c >> 4) & 0x07070707) | (c & 0x80808080)) + base) & 0xffffffff
    return a, b


def highbytes_new(c, base):
    r = ((c >> 4) | (c << 28)) & 0xffffffff
    return ((c & 0x87878787) + base) & 0xffffffff, ((r & 0x87878787) + base) & 0xffffffff


samples = 0
for base in range(121):
    b4 = base * 0x01010101
    for c in range(256):
        for pos in range(4):
            code = c << (8 * pos)
            assert inverse(permute(code)) == code
            assert highbytes(code, b4) == highbytes_new(permute(code), b4)
            samples += 1
rng = random.Random(9331)
for _ in range(100000):
    code = rng.getrandbits(32)
    b4 = rng.randrange(121) * 0x01010101
    assert inverse(permute(code)) == code
    assert highbytes(code, b4) == highbytes_new(permute(code), b4)
    samples += 1
# Low-byte words and PRMT selectors are literally unchanged. Equal packed high
# bytes imply all eight BF16 outputs match, including arbitrary low-byte bits.
pre = native[native.index('__device__ __forceinline__ uint32_t prmt('):native.index('// Decode 8 weights:')]
old = old.replace('decode8(', 'decode_old(', 1)
new = new.replace('decode8(', 'decode_new(', 1)
source = '#include <cuda_runtime.h>\n#include <stdint.h>\n' + pre + old + new
for name in ('old', 'new'):
    source += f'''
extern "C" __global__ void {name}_probe(const uint4* in, uint4* out) {{
 int i=threadIdx.x+blockIdx.x*blockDim.x;uint4 v=in[i];uint32_t r[4];
 decode_{name}(v.x,v.y,v.z,v.w,r);out[i]=make_uint4(r[0],r[1],r[2],r[3]);
}}
'''
request = dict(source=source, options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False), filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
target = ROOT / 'muse_bf16_proj/C12_SIGN_PERMUTE_CODEGEN_2026-10-31.json'
assert not target.exists()
req = urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',
    data=json.dumps(request).encode(), headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
except (urllib.error.URLError, TimeoutError) as e:
    result = dict(code=None,transport_error=repr(e))
target.write_text(json.dumps(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Exhaustive byte-position/base and random-word CPU equivalence; standalone helper codegen, not H200 performance.',
    parent_sha256=hashlib.sha256(raw).hexdigest(),cpu_cases=samples,all_high_bytes_identical=True,
    development_correction='Initial per-byte sign swap failed CPU equivalence before compiler/GPU use; rotr4 crosses bytes. Corrected to a u32 sign permutation with explicit inverse.',
    request=request,response=result),indent=2)+'\n')
print('CPU equivalence PASS:',samples,'cases; Compiler status:',result.get('code'))
for line in result.get('stderr',[]): print(line['text'])
for name,dev in result.get('devices',{}).items():
    if name.startswith('SASS'):
        print(name)
        for row in dev['asm']: print(row['text'])
