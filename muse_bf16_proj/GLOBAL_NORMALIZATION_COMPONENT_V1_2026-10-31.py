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
