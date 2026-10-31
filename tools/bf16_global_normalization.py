"""Research component: fixed-window C12 decode with unscaled FP32 partials.

Sign layout is Nathen Fernandes' MIT component already embedded in the
Apache-2.0 parent. Fixed-window normalization follows the mechanism studied
in MrNirmata/lossless-bf16 (Apache-2.0). Here the original global base and
escape table are retained, and every MMA partial is unscaled BEFORE the
original escape corrections/reduction. No per-row scale or new escape math.
"""
import re

DECODE = r'''
// Nathen's MIT sign layout; fixed window [112,127] after exact pow2 scaling.
// Original global base remains available to unchanged escape corrections.
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1,
    uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(nw));
  const uint32_t ha = (nw & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (rotated & 0x87878787u) | 0x38383838u;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
'''


def metadata(base4):
    base = base4 & 255
    assert 1 <= base <= 119
    assert base4 == base * 0x01010101
    # FP32 2**(2*base-112) restores original units. Its low23 bits are zero,
    # so the original base byte can share this integer kernel parameter.
    return base | ((2*base+15) << 23)


def normalize_body(body):
    line = 'for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];'
    assert body.count(line) == 2
    body = body.replace(line, line + '\n'
        '          v[e] *= __uint_as_float(base4 & 0x7f800000u);')
    assert body.count('v[e] *= __uint_as_float') == 2
    return body


def metadata_lut(base4):
    base=base4 & 255
    assert 1<=base<=119 and base4==base*0x01010101
    # Fixed bits outside 0x87878787 establish the normalized exponent.
    # Original-base metadata lives exclusively in bits masked OFF by LOP3.
    return 0x38383838 | (base&7) | ((base&0x38)<<5) | ((base&0x40)<<1)


DECODE_LUT = DECODE[:DECODE.index('__device__')] + r'''
__device__ __forceinline__ unsigned gn_original_base(unsigned p) {
  return (p&7u) | ((p>>5)&0x38u) | ((p>>1)&0x40u);
}
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1,
    uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t rotated, ha, hb;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(nw));
  // (a & mask) | (param & ~mask); parameter's ignored bits carry metadata.
  asm("lop3.b32 %0, %1, 0x87878787, %2, 0xe2;" : "=r"(ha) : "r"(nw), "r"(base4));
  asm("lop3.b32 %0, %1, 0x87878787, %2, 0xe2;" : "=r"(hb) : "r"(rotated), "r"(base4));
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
'''


def normalize_body_lut(body):
    anchor='  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)'
    assert body.count(anchor)==1
    body=body.replace(anchor,anchor+'\n'
        '  const float gn_restore = __uint_as_float((2u*gn_original_base(base4)+15u)<<23);')
    anchor='const uint32_t bq = base4 & 0xFFu;'
    assert body.count(anchor)==1
    body=body.replace(anchor,'const uint32_t bq = gn_original_base(base4);')
    anchor='for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];'
    assert body.count(anchor)==2
    return body.replace(anchor,anchor+'\n          v[e] *= gn_restore;')
