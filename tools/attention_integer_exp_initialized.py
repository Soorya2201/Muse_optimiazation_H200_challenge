"""Define all core temporaries unconditionally; predicate only the final fallback."""
from attention_integer_exp import COEFFICIENT


def cuda_helper(final_operands=False):
    source=r'''
// Independent integer evaluator of the known endpoint quadratic.
// All temporaries are defined unconditionally, avoiding predicated SSA merges.
// For x>120 the unused core bits are arbitrary unsigned arithmetic; native
// ex2 overwrites the result. No FP operation consumes the unused float bits.
__device__ __forceinline__ float aq_integer_predicated(float x) {
  float out;
  asm volatile("{\n"
    ".reg .pred p;\n"
    ".reg .f32 z;\n"
    ".reg .b32 u,b,t,c,q;\n"
    "max.f32 z,%1,-126.0;\n"
    "add.rn.f32 z,z,384.0;\n"
    "mov.b32 u,z;\n"
    "and.b32 b,u,32767;\n"
    "sub.u32 t,32768,b;\n"
    "mul.lo.u32 t,b,t;\n"
    "mul.hi.u32 c,t,''' + str(COEFFICIENT) + r''';\n"
    "sub.u32 q,0x7f800000,c;\n"
    "mad.lo.u32 q,u,256,q;\n"
    "mov.b32 %0,q;\n"
    "setp.gt.f32 p,%1,120.0;\n"
    "@p ex2.approx.ftz.f32 %0,%1;\n"
    "}\n" : "=f"(out):"f"(x));
  return out;
}
'''
    if final_operands:
        # The C++ output constraint is written only after every input read.
        # This remains correct even if the compiler coalesces input/output.
        source=source.replace('"mov.b32 %0,q;\\n"', '"mov.b32 z,q;\\n"')
        source=source.replace('"@p ex2.approx.ftz.f32 %0,%1;\\n"',
            '"@p ex2.approx.ftz.f32 z,%1;\\n"\n    "mov.f32 %0,z;\\n"')
    return source
