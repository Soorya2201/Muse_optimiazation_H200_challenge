"""Archive standalone NVCC13 sm90a SASS for the independently derived helper."""
import ast
import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'muse_attention_decode/submission_v24_bit_exp_probe.py').read_text())
helper = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
source = '#include <cuda_runtime.h>\n#include <stdint.h>\n' + helper + r'''
__device__ __forceinline__ float hw(float x) {
 float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
extern "C" __global__ void hardware(const float* in,float* out) {
 int i=threadIdx.x+blockIdx.x*blockDim.x; out[i]=hw(in[i]);
}
extern "C" __global__ void bit_exp(const float* in,float* out) {
 int i=threadIdx.x+blockIdx.x*blockDim.x; float x=in[i];
 out[i]=x>120.f?hw(x):ex2_poly(x);
}
'''
request = dict(source=source, options=dict(userArguments='-O3 -std=c++17 -arch=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False), filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
target = ROOT / 'muse_attention_decode/BIT_EXP_COMPILER_2026-10-31.json'
assert not target.exists()
req = urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',
    data=json.dumps(request).encode(), headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        response = json.load(r)
except (urllib.error.URLError, TimeoutError) as e:
    response = dict(code=None, transport_error=repr(e))
target.write_text(json.dumps(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Standalone helper only; not full attention resource/performance evidence.', request=request,
    response=response), indent=2) + '\n')
print('Compiler status:', response.get('code'))
for row in response.get('stderr', []):
    print(row.get('text', '')[:250])
for name, dev in response.get('devices', {}).items():
    print('DEVICE', name)
    for row in dev.get('asm', []):
        print(row.get('text', ''))
