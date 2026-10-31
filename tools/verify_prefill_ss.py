"""CPU proof of producer-owned NVFP4 register fragments to shared A layout.

Symbolic code/scale tags validate bit and address assignment, not BF16
hardware multiplication, WGMMA execution, or concurrency on H200.
"""
import ast
import collections
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
from prefill_ss_native import shared_weight_native

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / 'muse_prefill_layer'
parent = (TRACK / 'submission.py').read_text()
tree = ast.parse(parent)
def literal(name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
native = literal('_CUDA_SRC'); pos = literal('_CODE_POS')
assert hashlib.sha256(parent.encode()).hexdigest() == '8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5'
def code(row,k): return (row*7+k*13+(k>>3)) & 15
def scale(row,j): return (row*31+j*37) & 255
def offset(row,k): return row*128 + ((k*2)^((row&7)*16))
def rotl(x,n): return ((x<<n)|(x>>(32-n))) & 0xffffffff

blocks = vector_pairs = 0
seen = set(); rng=random.Random(0x53534D4D41)
for ks in range(64):
    packed=bytearray([0xa5]*16384)
    expected={}
    for q in range(256):
        warp,lane=divmod(q,32); g,t=divmod(lane,4)
        for j in range(4):
            codes=[0]*8
            for half in range(2):
                row=warp*16+half*8+g
                for hk in range(2):
                    for b in range(2):
                        k=j*16+hk*8+t*2+b
                        nib=code(row,ks*64+k)
                        i=3-(half+2*hk)+4*b
                        codes[i]=nib
                        expected[(row,k)]=(scale(row,ks*4+j)<<4)|nib
                        seen.add((nib,scale(row,ks*4+j)))
            word=sum(((codes[i]>>bit)&1)<<pos[i][bit] for i in range(8) for bit in range(4))
            struct.pack_into('<I',packed,q*16+j*4,word)
            r=[word&0x81c081c0,(word*16)&0x81c081c0,
               rotl(word,10)&0x81c081c0,
               ((word*64)&0x80c080c0)|(rotl(word,14)&0x01000100)]
            for a in range(4):
                for b in range(2):
                    nib=codes[3-a+4*b]
                    assert ((r[a]>>(16*b))&0xffff)==((nib&8)<<12)|((nib&7)<<6)
        for j in range(4):
            for half in range(2):
                row=warp*16+half*8+g
                packed[4096+(warp*8+g)*8+j*2+half]=scale(row,ks*4+j)
    # Load the actual packed byte addresses for all 64 workers before writes.
    held={}; reads=collections.Counter()
    for tid in range(64):
        held[tid]=[]
        for i in range(4):
            q=tid+i*64; warp,lane=divmod(q,32);g=lane>>2
            c=struct.unpack_from('<4I',packed,q*16)
            s=bytes(packed[4096+(warp*8+g)*8:4096+(warp*8+g)*8+8])
            held[tid].append((q,c,s))
            reads.update(range(q*16,q*16+16))
            reads.update(range(4096+(warp*8+g)*8,4096+(warp*8+g)*8+8))
    assert all(reads[b]==1 for b in range(4096))
    assert all(reads[b]==4 for b in range(4096,4608))
    # Randomize post-barrier workers. All input is already held, so arbitrary
    # overwrite order must preserve the same complete A operand.
    workers=list(range(64));rng.shuffle(workers);written=set()
    for tid in workers:
        for q,words,s in held[tid]:
            warp,lane=divmod(q,32); g,t=divmod(lane,4)
            for j,word in enumerate(words):
                nibbles=[sum(((word>>pos[i][bit])&1)<<bit for bit in range(4)) for i in range(8)]
                for a in range(4):
                    row=warp*16+g+(a&1)*8;k=j*16+2*t+(a>>1)*8
                    for b in range(2):
                        off=offset(row,k+b)
                        assert off not in written and off%2==0 and off<16384
                        written.add(off)
                        tag=(s[j*2+(a&1)]<<4)|nibbles[3-a+4*b]
                        struct.pack_into('<H',packed,off,tag)
                    vector_pairs+=1
    assert len(written)==8192
    assert all(struct.unpack_from('<H',packed,offset(row,k))[0]==tag for (row,k),tag in expected.items())
    blocks+=1
assert len(seen)==4096

# Existing epilogue needs at most 34,816 bytes: G1 N*128, G2 one half's
# 128 rows at a 272-byte pitch. The four 49,152-byte slots fit exactly.
staging=[]
for n in (32,64,128,160,192,256):
    gated=n*128
    nongated=min(n,128)*272
    assert max(gated,nongated)<=34816
    stages=64//(4 if n==32 else 2 if n<=128 else 1)
    assert stages%8==0
    staging.append(dict(width=n,gated_bytes=gated,nongated_bytes=nongated,stages=stages))
assert 1024+4*49152+34816==232448
for _ in range(1000):
    widths=[rng.choice([32,64,128,160,192,256]) for _ in range(20)]
    raw=[0]*4;decoded=[0]*4
    for n in widths:
        assert all((x&1)==0 for x in raw+decoded)
        stages=64//(4 if n==32 else 2 if n<=128 else 1)
        for it in range(stages):
            slot=it&3;phase=(it>>2)&1
            assert raw[slot]&1==phase
            raw[slot]+=1
            if n>=160:
                assert decoded[slot]&1==phase
                decoded[slot]+=1

artifacts=[]
for regs in (184,216):
    changed=shared_weight_native(native,regs)
    # Parent conversion and the complete direct epilogue are reused verbatim.
    conversion=native[native.index('DEVI void convert8('):native.index('\n\nstruct Args')]
    assert conversion in changed
    path=TRACK/f'SS_NATIVE_R{regs}_2026-10-31.cu.gz'
    if not path.exists() or gzip.open(path,'rt').read()!=changed:
        with gzip.open(path,'wt') as f:f.write(changed)
    diff=''.join(difflib.unified_diff(native.splitlines(True),changed.splitlines(True),fromfile='parent.cu',tofile=f'ss_r{regs}.cu',n=0))
    (TRACK/f'SS_NATIVE_R{regs}_2026-10-31.diff').write_text(diff)
    pr=120 if regs==184 else 72
    artifacts.append(dict(consumer_regs=regs,producer_regs=pr,role_register_request=128*pr+256*regs,
                          native_sha256=hashlib.sha256(changed.encode()).hexdigest(),path=str(path.relative_to(ROOT))))
changed=shared_weight_native(native,184,True)
path=TRACK/'SS_NATIVE_ALL_R184_2026-10-31.cu.gz'
if not path.exists() or gzip.open(path,'rt').read()!=changed:
    with gzip.open(path,'wt') as f:f.write(changed)
(TRACK/'SS_NATIVE_ALL_R184_2026-10-31.diff').write_text(''.join(difflib.unified_diff(
    native.splitlines(True),changed.splitlines(True),fromfile='parent.cu',tofile='ss_all_r184.cu',n=0)))
artifacts.append(dict(mode='all widths SS, KPS1',consumer_regs=184,producer_regs=120,
                      role_register_request=62464,native_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                      path=str(path.relative_to(ROOT))))
out=dict(scope=__doc__,parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),
         blocks=blocks,raw_e2m1_patterns=16,scale_raw_byte_patterns=256,
         observed_code_scale_pairs=len(seen),symbolic_fragment_pairs=vector_pairs,
         symbolic_bf16_destinations=blocks*8192,staging=staging,
         metadata_readers=320,phase_sequences=1000,items_per_sequence=20,
         all_ss_stages_per_item=64,all_ss_decoded_barrier_cycles_per_item=16,
         dynamic_shared_bytes=232448,extra_global_activation_bytes=0,
         raw_load_before_overwrite='CPU all-worker capture then randomized stores; native named barrier requires independent GPU validation',
         artifacts=artifacts)
(TRACK/'SS_PREFLIGHT_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',blocks,'blocks,',vector_pairs,'fragment pairs,',len(seen),'code/scale pairs, 1000 phase sequences')
print('shared bytes',out['dynamic_shared_bytes'],'role budgets',[(x['consumer_regs'],x['role_register_request']) for x in artifacts])
