"""CPU codec/bank/capacity proof and standalone NVCC13 fragment comparison.

This is not a complete MoE port, H200 run, or speedup claim. A natural-byte
weight repack would be needed in a full kernel. No downloaded module is
imported; only the inspected, pure packing function is evaluated. Use
--offline to recheck CPU proofs and the archived source without a request.
"""
import argparse
import ast
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.error
import urllib.request

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--offline',action='store_true')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_moe_w4a16'
parent=(TRACK/'submission.py').read_text();tree=ast.parse(parent)
assert hashlib.sha256(parent.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
native=ast.literal_eval(node.value)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_stored_pos')
env={'DQ7':True};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<public pure bit map>','exec'),env)
positions=env['_stored_pos']()

def code_raw(n):return ((n&8)<<12)|((n&7)<<6)
table=[code_raw(i&15)|(code_raw(i>>4)<<16) for i in range(256)]
def rotl(x,n):return ((x<<n)|(x>>(32-n)))&0xffffffff
def parent_decode(q):return [q&0x81c081c0,(q*16)&0x81c081c0,
                             rotl(q,10)&0x81c081c0,
                             ((q*64)&0x80c080c0)|(rotl(q,14)&0x01000100)]
def encode_parent(natural):
    word=0
    for r in range(4):
        for up in range(2):
            n=(natural>>(8*r+4*up))&15
            for bit in range(4):word|=((n>>bit)&1)<<positions[r,up,bit]
    return word
def decode_lut(q):return [table[(q>>(8*i))&255] for i in range(4)]
for q in range(256):assert table[q]==code_raw(q&15)|(code_raw(q>>4)<<16)
cases=0
for byte in range(4):
    for pair in range(256):
        q=pair<<(byte*8);assert parent_decode(encode_parent(q))==decode_lut(q);cases+=1
rng=random.Random(0x4C55543136)
for _ in range(100000):
    q=rng.getrandbits(32);assert parent_decode(encode_parent(q))==decode_lut(q);cases+=1
assert sorted(positions.values())==list(range(32))

def waves(indices,copies):
    banks=[set() for _ in range(32)]
    for lane,index in enumerate(indices):
        word=index*copies+lane%copies
        banks[word%32].add(word)
    return max(map(len,banks))
bank_cases=[]
for copies in (1,8,16,32):
    assert waves([0]*32,copies)==1
    assert waves([2*i for i in range(32)],copies)<=32//copies
    histogram=collections.Counter()
    for _ in range(20000):
        v=waves([rng.randrange(256) for _ in range(32)],copies)
        assert 1<=v<=32//copies
        histogram[v]+=1
    bank_cases.append(dict(copies=copies,bytes=1024*copies,
        modeled_uniform_warp_word_waves=dict(sorted(histogram.items())),
        maximum_word_waves=32//copies))
cap=232448;tiny=214784;main=231168
capacity=[dict(copies=d,table_bytes=1024*d,tiny_sum=tiny+1024*d,
               tiny_fits=tiny+1024*d<=cap,main_sum=main+1024*d,
               main_fits=main+1024*d<=cap) for d in (1,8,16,32)]
assert tiny+16384==231168 and tiny+32768>cap

# Extract only the exact public DQ7 helper. A complete standalone table
# kernel initializes its own shared table, then calls a non-inline core so
# its issued fragment instructions can be distinguished from initialization.
a=native.index('__device__ __forceinline__ void dq4(')
b=native.index('// ---------------------------------------------------------------- routing prologue',a)
decoder=native[a:b]
source=r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
static __device__ __forceinline__ uint32_t mul2(uint32_t a,uint32_t b) {
  __nv_bfloat162 x=*reinterpret_cast<__nv_bfloat162*>(&a);
  __nv_bfloat162 y=*reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 z=__hmul2(x,y);
  return *reinterpret_cast<uint32_t*>(&z);
}
'''+decoder+r'''
extern "C" __device__ __noinline__ uint4 arithmetic_core(uint32_t q,uint32_t sg,uint32_t sg8) {
  uint32_t a,b,c,d;dq4(q,a,b,c,d);
  return make_uint4(mul2(a,sg),mul2(b,sg8),mul2(c,sg),mul2(d,sg8));
}
extern "C" __device__ __noinline__ uint4 table_core(uint32_t q,uint32_t sg,uint32_t sg8,
                                                   uint32_t base,uint32_t lane) {
  uint32_t r[4];
  const uint32_t origin=base+((lane&15u)<<2);
#pragma unroll
  for (int j=0;j<4;++j) {
    const uint32_t address=origin+(((q>>(8*j))&255u)<<6);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(r[j]):"r"(address):"memory");
  }
  return make_uint4(mul2(r[0],sg),mul2(r[1],sg8),mul2(r[2],sg),mul2(r[3],sg8));
}
extern "C" __global__ void parent_fragment(const uint32_t* words,const uint2* scales,uint4* out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){uint2 s=scales[i];out[i]=arithmetic_core(words[i],s.x,s.y);}
}
extern "C" __global__ void table_fragment(const uint32_t* natural_words,const uint2* scales,uint4* out,int n) {
  __shared__ uint32_t lut[4096];
  for(unsigned i=threadIdx.x;i<4096;i+=blockDim.x) {
    unsigned k=i>>4;
    unsigned a=((k&8u)<<12)|((k&7u)<<6);
    unsigned b=(((k>>4)&8u)<<12)|(((k>>4)&7u)<<6);
    lut[i]=a|(b<<16);
  }
  __syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    uint2 s=scales[i];
    out[i]=table_core(natural_words[i],s.x,s.y,(uint32_t)__cvta_generic_to_shared(lut),threadIdx.x&31);
  }
}
'''
request=dict(source=source,options=dict(userArguments='-O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v',
    compilerOptions=dict(executorRequest=False),filters=dict(binary=True,execute=False,intel=True,
    demangle=False,directives=False,labels=True,commentOnly=True,libraryCode=False,trim=False)))
path=TRACK/'TINY_PAIR_LUT_CODEGEN_2026-10-31.json.gz'
if args.offline:
    archived=json.load(gzip.open(path,'rt'))
    assert archived['request']==request
    assert (TRACK/'TINY_PAIR_LUT_FRAGMENT_2026-10-31.cu').read_text()==source
    assert archived['cpu_word_cases']==cases
    assert archived['capacity']==capacity
    assert archived['bank_model']==json.loads(json.dumps(bank_cases))
    assert archived['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
    print('CPU/source/archive PASS:',cases,'words,256 table entries,80000 modeled warps; no network or writes')
    raise SystemExit(0)
assert not path.exists()
(TRACK/'TINY_PAIR_LUT_FRAGMENT_2026-10-31.cu').write_text(source)
req=urllib.request.Request('https://godbolt.org/api/compiler/nvcc130/compile',data=json.dumps(request).encode(),
    headers={'Content-Type':'application/json','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
except (urllib.error.URLError,TimeoutError) as exc:
    result=dict(code=None,transport_error=repr(exc),stderr=[dict(text=str(exc))])
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Standalone decoder cores and initialized diagnostic kernels, not full fused MoE resources, a GPU correctness test, timing result or complete weight-repack integration.',
    parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    cpu_word_cases=cases,exhaustive_pair_table_entries=256,all_unscaled_bf16_operand_bits_equal=True,
    capacity=capacity,bank_model=bank_cases,
    bank_model_limit='32 four-byte banks, unique words per bank (same word broadcasts); uniform independent pair indices are synthetic, not actual weight statistics or observed device cycles.',
    tiny_resource_provenance='V34 linked tiny control/trace reports214784dynamic bytes; attribution failure does not invalidate resource attributes. Only that tiny build is covered.',
    request=request,response=result)
with gzip.open(path,'wt') as f:json.dump(out,f)
(TRACK/'TINY_PAIR_LUT_PREFLIGHT_2026-10-31.json').write_text(json.dumps({k:v for k,v in out.items() if k not in ('request','response')},indent=2)+'\n')
print('CPU WORDS',cases,'tiny16copybytes',tiny+16384,'compiler_code',result.get('code'))
for row in result.get('stderr',[]):print(row.get('text','')[:250])
for name,dev in result.get('devices',{}).items():
    print(name,'rows',len(dev.get('asm',[])),'clipped',any('[truncated' in x['text'] for x in dev.get('asm',[])))
