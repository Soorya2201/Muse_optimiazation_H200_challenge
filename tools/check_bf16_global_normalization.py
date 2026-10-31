"""Exact weight-bit recovery and bounded FP32 arithmetic before any GPU job."""
import ctypes
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

from bf16_global_normalization import metadata, metadata_lut

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--fused-lut',action='store_true');args=parser.parse_args()
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
u32 = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
flt = lambda x: struct.unpack('<f', struct.pack('<I', x))[0]
lib = ctypes.CDLL(None)
fma = lib.fmaf
fma.argtypes = [ctypes.c_float]*3
fma.restype = ctypes.c_float
count=0
for base in range(1,120):
    param=metadata(base*0x01010101)
    scale=flt(param & 0x7f800000)
    assert param & 255 == base
    assert scale == 2.**(2*base-112)
    lutparam=metadata_lut(base*0x01010101)
    recovered_base=(lutparam&7)|((lutparam>>5)&0x38)|((lutparam>>1)&0x40)
    assert recovered_base==base
    assert lutparam & (0xffffffff^0x87878787)==0x38383838
    for offset in range(8):
        for low in range(256):
            for sign in range(2):
                old=sign<<15 | (base+offset)<<8 | low
                normalized=sign<<15 | (56+offset)<<8 | low
                assert flt(old<<16)==flt(normalized<<16)*scale
                recovered=(normalized&0x80ff)|(((normalized>>8&127)+base-56)<<8)
                assert recovered==old
                count+=1
    for pos in range(4):
        for byte in range(256):
            nw=byte<<(8*pos)
            # Ternary 0xe2 implements (a&b)|(c&~b), source operand ordering.
            for word in (nw,((nw>>4)|(nw<<28))&0xffffffff):
                ternary=0
                for bit in range(32):
                    index=((word>>bit&1)<<2)|((0x87878787>>bit&1)<<1)|(lutparam>>bit&1)
                    ternary|=((0xe2>>index)&1)<<bit
                assert ternary==((word&0x87878787)|0x38383838)

rng=random.Random(622701)
dots=0
worst=0.
# Homogeneity is conditional on finite normal intermediate arithmetic.
# Include multiple accumulators and original-unit escape contributions.
for base in (45,48,51,54,56,58,61,64):
    scale=flt(metadata(base*0x01010101)&0x7f800000)
    for rep in range(64):
        old=[0.]*4;new=[0.]*4
        for k in range(256):
            o=rng.randrange(8);low=rng.randrange(256);sg=rng.randrange(2)
            w=flt((sg<<15 | (base+o)<<8 | low)<<16)
            wn=flt((sg<<15 | (56+o)<<8 | low)<<16)
            xb=(u32(rng.uniform(-8,8))+0x8000)&0xffff0000
            x=flt(xb)
            lane=k%4
            old[lane]=fma(w,x,old[lane]);new[lane]=fma(wn,x,new[lane])
        # Unscale after NACC reduction, before correction/reduction to BF16.
        for i in range(4):
            unscaled=f32(new[i]*scale)
            assert u32(unscaled)==u32(old[i]),(base,rep,i)
            correction=f32(rng.uniform(-.01,.01))
            assert u32(f32(correction+unscaled))==u32(f32(correction+old[i]))
        oldsum,newsum=old[0],new[0]
        for i in range(1,4):oldsum=f32(oldsum+old[i]);newsum=f32(newsum+new[i])
        assert u32(f32(newsum*scale))==u32(oldsum)
        dots+=1
report=dict(scope='CPU exact storage recovery plus bounded scalar FP32/libc-fmaf arithmetic; not GPU MMA validation.',
    status='pass',decoded_bit_patterns=count,global_base_range=[1,119],
    dot_fixtures=dots,terms_each=256,accumulators=4,
    lut_parameter_cases=119,ternary_word_cases=119*4*256*2,
    module_sha256=hashlib.sha256((ROOT/'tools/bf16_global_normalization.py').read_bytes()).hexdigest(),
    limitations=['FP32 homogeneity requires no intermediate overflow/underflow/FTZ crossing.',
        'Extreme BF16 activations and GPU MMA accumulation remain unproven.',
        'Escaped original weight bits/correction pipeline are unchanged; only main partials rescaled.',
        'Existing sign-layout-only experiment is closed; this tests additional global exponent normalization.'])
label='GLOBAL_NORMALIZATION_LUT' if args.fused_lut else 'GLOBAL_NORMALIZATION'
p=ROOT/f'muse_bf16_proj/{label}_MATH_2026-10-31.json'
assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print('PASS',count,'exact decoded patterns;',dots,'bounded FP32 dot fixtures')
