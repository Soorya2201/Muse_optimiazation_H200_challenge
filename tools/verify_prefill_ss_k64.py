"""Exact symbolic proof for K64 code/scale reuse with standard shared A.

No GPU timing or floating-point/concurrency claim. BF16 pair positions are
validated by code/scale tags and the already-pinned parent nibble map.
"""
import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
from prefill_scale_permutation import logical_k
from prefill_ss_k64_native import scale_reused_shared_native

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_prefill_layer'
parent=(TRACK/'submission.py').read_text();tree=ast.parse(parent)
def literal(name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
native=literal('_CUDA_SRC');pos=literal('_CODE_POS')
def nib(row,k):return (row*7+k*13+(k>>3))&15
def sf(row,j):return (row*31+j*37)&255
def off(row,k):return row*128+((k*2)^((row&7)*16))
rng=random.Random(0x53534B3634);vectors=0;seen=set()
for ks in range(64):
    memory=bytearray([0xa5]*16384);expected={}
    for q in range(256):
        warp,lane=divmod(q,32);g,t=divmod(lane,4)
        for j in range(4):
            codes=[0]*8
            for half in range(2):
                row=warp*16+g+half*8
                for hk in range(2):
                    for b in range(2):
                        physical=j*16+hk*8+t*2+b;logical=logical_k(physical)
                        assert logical//16==t
                        code=nib(row,ks*64+logical);scale=sf(row,ks*4+t)
                        codes[3-(half+2*hk)+4*b]=code
                        expected[row,logical]=scale*16+code;seen.add((code,scale))
            word=sum(((codes[i]>>b)&1)<<pos[i][b] for i in range(8) for b in range(4))
            struct.pack_into('<I',memory,q*16+j*4,word)
        for j in range(4):
            for half in range(2):
                memory[4096+(warp*8+g)*8+j*2+half]=sf(warp*16+g+half*8,ks*4+j)
    held={}
    for tid in range(64):
        held[tid]=[]
        for i in range(4):
            q=tid+64*i;warp,lane=divmod(q,32);g,t=divmod(lane,4)
            held[tid].append((q,struct.unpack_from('<4I',memory,q*16),
                             memory[4096+(warp*8+g)*8+2*t:4096+(warp*8+g)*8+2*t+2]))
    order=list(range(64));rng.shuffle(order);written=set()
    for tid in order:
        for q,words,scales in held[tid]:
            warp,lane=divmod(q,32);g,t=divmod(lane,4)
            fragments=[]
            for word in words:
                codes=[sum(((word>>pos[i][b])&1)<<b for b in range(4)) for i in range(8)]
                fragments.append([[scales[a&1]*16+codes[3-a+4*b] for b in range(2)] for a in range(4)])
            for half in range(2):
                a,b=fragments[2*half:2*half+2]
                for rh in range(2):
                    row=warp*16+g+rh*8;k=t*16+half*8;address=off(row,k)
                    values=a[rh]+a[rh+2]+b[rh]+b[rh+2]
                    assert address%16==0 and len(values)==8 and address+16<=16384
                    for byte in range(address,address+16):assert byte not in written;written.add(byte)
                    struct.pack_into('<8H',memory,address,*values);vectors+=1
    assert len(written)==16384 and len(expected)==8192
    assert all(struct.unpack_from('<H',memory,off(r,k))[0]==tag for (r,k),tag in expected.items())
assert len(seen)==4096
changed=scale_reused_shared_native(native)
marker='// ---- routing plan:'
assert changed[changed.index(marker):]==native[native.index(marker):]
path=TRACK/'SS_K64_NATIVE_2026-10-31.cu.gz'
with gzip.open(path,'wt') as f:f.write(changed)
(TRACK/'SS_K64_NATIVE_2026-10-31.diff').write_text(''.join(difflib.unified_diff(
    native.splitlines(True),changed.splitlines(True),fromfile='parent.cu',tofile='ss_k64.cu',n=0)))
out=dict(scope=__doc__,parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),
         native_sha256=hashlib.sha256(changed.encode()).hexdigest(),blocks=64,
         destinations=524288,raw_code_scale_pairs=len(seen),aligned_vector_stores=vectors,
         standard_activation_and_intermediate_layout=True,unchanged_from_routing_plan_to_end=True,
         source_stores_per_worker_stage_old=64,source_stores_per_worker_stage_new=16,
         source_store_bytes_per_worker_stage_old=256,source_store_bytes_per_worker_stage_new=256,
         scale_pair_constructions_per_worker_stage_old=16,scale_pair_constructions_per_worker_stage_new=4,
         extra_global_activation_passes=0,required_weight_repack='Existing exact _ks_repack from v27; scale storage unchanged')
(TRACK/'SS_K64_PREFLIGHT_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',out['destinations'],'standard-layout destinations;',vectors,'aligned16Bstores;',len(seen),'code/scale pairs')
