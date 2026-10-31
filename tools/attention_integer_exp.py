"""Independent fixed-binade integer implementation of the known quadratic exp2.

Research only. All scored attention paths remain unchanged. The endpoint
quadratic family is prior art; this module changes its rounded evaluator.
"""
from attention_direct_fp8 import bits, f32, value

A=f32(.33976602809582634)
COEFFICIENT=round(A*(1<<25))


def integer_exp(x):
    """FP32 CPU model on x<=120, including masked -infinity; NaN excluded."""
    assert x<=120
    x=max(f32(x),-126.)
    u=bits(f32(x+384.))
    b=u&32767
    product=b*(32768-b)
    correction=(product*COEFFICIENT)>>32
    return value(((u<<8)+0x7f800000-correction)&0xffffffff)


def cuda_helper():
    return r'''
// Independently implemented integer evaluator of the known endpoint quadratic.
// Fixed coefficient from A=.33976602809582634, not a coefficient sweep.
// Caller keeps native ex2 above120; masked -inf rounds to zero after FP8.
__device__ __forceinline__ float aq_integer_exp(float x) {
  x=fmaxf(x,-126.f);
  unsigned u=__float_as_uint(__fadd_rn(x,384.f));
  unsigned b=u&32767u;
  unsigned product=b*(32768u-b);
  unsigned correction=__umulhi(product,''' + str(COEFFICIENT) + r'''u);
  return __uint_as_float((u<<8)+0x7f800000u-correction);
}
'''
