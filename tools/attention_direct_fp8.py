"""Independently derived log2 -> positive E4M3 quantizer, CPU model and CUDA.

Research only. Models ideal exp2 followed by RNE/satfinite, not the small
errors of hardware ex2.approx. No attention dispatch uses this module.
"""
from decimal import Decimal, localcontext
import math
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(x):
    return struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0]


def next_up(x):
    if x == 0:
        return value(1)
    return value(bits(x) + (1 if x > 0 else -1))


def ceil32(x):
    y = f32(float(x))
    return next_up(y) if Decimal.from_float(y) < x else y


def add_ru(a, b):
    # Sums used here have <=53 significant bits. Inputs have bounded,
    # non-extreme exponents; this is not a general float32-add emulator.
    exact = float(a) + float(b)
    y = f32(exact)
    return next_up(y) if y < exact else y


def decode(q):
    assert 0 <= q <= 126
    return q / 512 if q < 8 else math.ldexp(1 + (q & 7) / 8, (q >> 3) - 7)


def tables():
    with localcontext() as ctx:
        ctx.prec = 80
        ln2 = Decimal(2).ln()
        def log2(x):
            return x.ln() / ln2
        normal = [ceil32(log2(Decimal(31) / 32))]
        normal.extend(ceil32(log2(1 + (Decimal(m) - Decimal('.5')) / 8)
                             - Decimal(m) / 8) for m in range(1, 8))
        sub = [-math.inf] + [ceil32(log2((Decimal(q) - Decimal('.5')) / 512))
                            for q in range(1, 9)]
        all_boundaries = [ceil32(log2((Decimal.from_float(decode(q - 1))
                                      + Decimal.from_float(decode(q))) / 2))
                          for q in range(1, 127)]
    return normal + sub + [-math.inf] * 15, all_boundaries


LUT, BOUNDARIES = tables()


def quantize(x):
    """Return ideal E4M3 positive byte for FP32 log2 input (NaN excluded)."""
    x = min(f32(x), 9.)
    if x >= -6:
        t = f32(x + 1572864.)
        q = bits(t) - bits(1572864.) + 56
        base = f32(t - 1572864.)
        idx = q & 7
    elif x > -10:
        biased = add_ru(x, 384.)
        interpolated = value((bits(biased) << 8) + 0x7f800000)
        t = f32(interpolated + 24576.)
        q = bits(t) - bits(24576.)
        base, idx = 0., 8 + q
    else:
        q, base, idx = 0, 0., 8
    boundary = add_ru(base, LUT[idx])
    return min(q - int(x < boundary), 126)


def cuda_helper():
    lut = ', '.join(f'0x{bits(x):08x}u' for x in LUT)
    return r'''
// Independently derived exact ideal-exp2/RNE E4M3 quantizer, FP32 inputs.
// Not asserted bit-identical to ex2.approx + cvt near rounding boundaries.
__device__ __constant__ unsigned aq_direct_lut[32] = {''' + lut + r'''};
__device__ __forceinline__ unsigned aq_direct_code(float x, unsigned lane_lut) {
  x = fminf(x, 9.f);
  unsigned q, idx;
  float base;
  if (x >= -6.f) {
    float t = __fadd_rn(x, 1572864.f);
    q = __float_as_uint(t) - 0x49c00000u + 56u;
    base = __fadd_rn(t, -1572864.f);
    idx = q & 7u;
  } else if (x > -10.f) {
    float b = __fadd_ru(x, 384.f);
    float p = __uint_as_float((__float_as_uint(b) << 8) + 0x7f800000u);
    float t = __fadd_rn(p, 24576.f);
    q = __float_as_uint(t) - 0x46c00000u;
    base = 0.f; idx = 8u + q;
  } else {
    q = 0u; base = 0.f; idx = 8u;
  }
  // Every live warp lane participates, including masked/underflow inputs.
  float delta = __uint_as_float(__shfl_sync(0xffffffffu, lane_lut, idx));
  float boundary = __fadd_ru(base, delta);
  q -= unsigned(x < boundary);
  return min(q, 126u);
}
__device__ __forceinline__ unsigned aq_direct_fp8x4(
    float x0, float x1, float x2, float x3, unsigned lane_lut) {
  return aq_direct_code(x0, lane_lut)
       | (aq_direct_code(x1, lane_lut) << 8)
       | (aq_direct_code(x2, lane_lut) << 16)
       | (aq_direct_code(x3, lane_lut) << 24);
}
'''


def quantize_branchless(x):
    x = min(f32(x), 9.)
    nt = f32(max(x, -6.) + 1572864.)
    nq = bits(nt) - 0x49c00000 + 56
    b = add_ru(min(max(x, -10.), -6.), 384.)
    p = value((bits(b) << 8) + 0x7f800000)
    st = f32(p + 24576.)
    sq = bits(st) - 0x46c00000
    normal = x >= -6
    q = nq if normal else sq
    idx = (nq & 7) if normal else (8 + sq)
    base = f32(nt - 1572864.) if normal else 0.
    return min(q - int(x < add_ru(base, LUT[idx])), 126)


def cuda_branchless_helper():
    source = cuda_helper()
    start = source.index('  unsigned q, idx;')
    stop = source.index('  // Every live warp lane participates', start)
    source = source[:start] + r'''
  float nt = __fadd_rn(fmaxf(x, -6.f), 1572864.f);
  unsigned nq = __float_as_uint(nt) - 0x49c00000u + 56u;
  float sx = fminf(fmaxf(x, -10.f), -6.f);
  float b = __fadd_ru(sx, 384.f);
  float p = __uint_as_float((__float_as_uint(b) << 8) + 0x7f800000u);
  float st = __fadd_rn(p, 24576.f);
  unsigned sq = __float_as_uint(st) - 0x46c00000u;
  bool normal = x >= -6.f;
  unsigned q = normal ? nq : sq;
  unsigned idx = normal ? (nq & 7u) : (8u + sq);
  float base = normal ? __fadd_rn(nt, -1572864.f) : 0.f;
''' + source[stop:]
    return source
