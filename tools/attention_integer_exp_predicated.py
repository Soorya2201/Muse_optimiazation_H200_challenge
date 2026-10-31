"""Same integer evaluator, explicit predication for the high-tail fallback.

No per-lane branch/reconvergence is needed for the rare native-exp arm.
All integer expressions match attention_integer_exp.integer_exp exactly.
"""
from attention_integer_exp import COEFFICIENT


def cuda_helper():
    return r'''
// Independent integer evaluator of the known endpoint-constrained quadratic.
// Explicit predication retains native exp2 above120 without lane divergence.
__device__ __forceinline__ float aq_integer_predicated(float x) {
  float out;
  asm volatile("{\n"
    ".reg .pred p;\n"
    ".reg .f32 z;\n"
    ".reg .b32 u,b,t,c,q;\n"
    "setp.gt.f32 p,%1,120.0;\n"
    "@!p max.f32 z,%1,-126.0;\n"
    "@!p add.rn.f32 z,z,384.0;\n"
    "@!p mov.b32 u,z;\n"
    "@!p and.b32 b,u,32767;\n"
    "@!p sub.u32 t,32768,b;\n"
    "@!p mul.lo.u32 t,b,t;\n"
    "@!p mul.hi.u32 c,t,''' + str(COEFFICIENT) + r''';\n"
    "@!p sub.u32 q,0x7f800000,c;\n"
    "@!p mad.lo.u32 q,u,256,q;\n"
    "@!p mov.b32 %0,q;\n"
    "@p ex2.approx.ftz.f32 %0,%1;\n"
    "}\n" : "=f"(out):"f"(x));
  return out;
}
'''
