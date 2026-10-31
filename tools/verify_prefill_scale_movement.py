"""Check the actual proposed CUDA register/shmem mapping with symbolic words.

No GPU or latency claim. The simulation follows subgroup shuffle statements
and warp load/barrier/store phases, then checks independent index equations.
"""
import ast
import hashlib
import json
from pathlib import Path
from prefill_scale_permutation import logical_k, packed_k
from prefill_scale_native import HELPERS, scale_shared_native

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'muse_prefill_layer/submission.py').read_text()
tree=ast.parse(source)
native=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(x,ast.Name) and x.id=='_CUDA_SRC' for x in n.targets))
changed=scale_shared_native(native)
# Catch a missed per-lane selection in *every* initial/prefetch scale address.
body=changed[changed.index('template <int N, int CB = 128>'):changed.index('\ntemplate <bool GATED>',changed.index('template <int N, int CB = 128>'))]
assert 'sbase = st + KPS * XB + soff + 2 * t;' in body
assert body.count('const uint32_t scale = scale_pair_reused(nsc, m16);')==2
assert body.count('scale_pair_sel(')==0
assert body.count('k64_permute_staging<N>')==1
assert body.index('named_bar_sync(1, 256);\n    k64_permute_staging<N>') < body.index('fence.proxy.async.shared::cta;')
assert 'k64_gather_words(outw, tid & 31);' in changed


def transpose(v):
    a=[x[:] for x in v]
    # Execute the source's actual two butterfly stages, rather than an
    # independent mathematical transpose that could mask a coding error.
    for bit in (1,2):
        pairs=((0,1),(2,3)) if bit==1 else ((0,2),(1,3))
        nxt=[x[:] for x in a]
        for lo,hi in pairs:
            sent=[a[l][lo if l&bit else hi] for l in range(4)]
            for lane in range(4):
                nxt[lane][lo if lane&bit else hi]=sent[lane^bit]
        a=nxt
    return a


gather_words=0
for stage in range(64):
    # Four lanes hold sixteen consecutive BF16s apiece (eight raw words).
    original=[[((stage*64+lane*16+2*j+1)<<16)|(stage*64+lane*16+2*j)
               for j in range(8)] for lane in range(4)]
    even=transpose([[r[2*j] for j in range(4)] for r in original])
    odd=transpose([[r[2*j+1] for j in range(4)] for r in original])
    result=[even[l]+odd[l] for l in range(4)]
    flat=[r for row in original for word in row for r in (word&65535,word>>16)]
    for lane in range(4):
        values=[r for word in result[lane] for r in (word&65535,word>>16)]
        expected=[flat[logical_k(lane*16+j)] for j in range(16)]
        assert values==expected
        gather_words+=8

# Current GEMM1 staging: 128B rows, chunk XOR row%8. One warp per row,
# each lane loads a raw 32-bit pair before *any* lane overwrites the row.
records=[];raw_patterns=set()
for width in (32,64,96,128,160,192,224,256):
    for offset in (0,16384,32768,49152):
        old=[]
        for row in range(width):
            words=[0]*32
            for pair in range(32):
                a=(offset+row*64+2*pair)&65535;b=(a+1)&65535
                raw_patterns.update([a,b]);words[pair^((row&7)<<2)]=a|(b<<16)
            old+=words
        new=old[:];coverage=[0]*len(old)
        for warp in range(8):
            for row in range(warp,width,8):
                sw=(row&7)<<2
                # Atomic simulation phase boundary corresponds to __syncwarp.
                registers=[old[row*32+(lane^sw)] for lane in range(32)]
                for lane,value in enumerate(registers):
                    destination=((lane<<2)&31)|(lane>>3)
                    index=row*32+(destination^sw)
                    assert coverage[index]==0
                    coverage[index]=1;new[index]=value
        assert all(coverage)
        for row in range(width):
            sw=(row&7)<<2
            for pair in range(32):
                actual=new[row*32+(pair^sw)]
                expected=old[row*32+((logical_k(2*pair)//2)^sw)]
                assert actual==expected
        records.append(dict(token_width=width,raw_offset=offset,words=len(old),extra_global_bytes=0,extra_shared_allocation_bytes=0))
assert len(raw_patterns)==65536
out=dict(scope='CPU execution-model proof of subgroup shuffle and warp-owned in-place staging. No CUDA execution, resource or performance claim.',
         parent_sha256=hashlib.sha256(source.encode()).hexdigest(),native_before_sha256=hashlib.sha256(native.encode()).hexdigest(),
         native_after_sha256=hashlib.sha256(changed.encode()).hexdigest(),helper_sha256=hashlib.sha256(HELPERS.encode()).hexdigest(),
         gather_words=gather_words,staging_cases=records,staging_words=sum(r['words'] for r in records),
         bf16_raw_patterns=len(raw_patterns),scale_addresses_checked=True,
         synchronization='Consumer barrier after all stmatrix writers; warp reads then __syncwarp then disjoint row permutation stores; existing proxy fence and consumer barrier before original queue/bulk publish.',
         caveats=['Source address proof is not actual GPU race-checking.','Complete fused-kernel resource status is unknown.','No physical bank-conflict, throughput or whole-board gain is established.'])
path=ROOT/'muse_prefill_layer/SCALE_MOVEMENT_CPU_2026-10-31.json';path.write_text(json.dumps(out,indent=2)+'\n')
print('PASS',gather_words,'gather words,',out['staging_words'],'staging words,',len(raw_patterns),'raw BF16 patterns; no extra global pass or shared allocation in mapping')
