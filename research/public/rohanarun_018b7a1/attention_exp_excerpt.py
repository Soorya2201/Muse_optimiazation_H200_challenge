# Static research excerpt; upstream module is never imported.
# Polynomial matches the Apache-2.0 Madhavan/vjeux source already pinned locally.
# Public mirror rohanarun/Auto-research-agent 018b7a102348ea3b73701e8be52d24f82d8c91ee (root MIT).
# See adjacent LICENSE and project Apache-2.0 LICENSE.

POLY_CPP = '__device__ __forceinline__ float ex2_poly(float x) {\n  x = fmaxf(x, -127.f);\n  const float j = floorf(x);\n  const float f = x - j;\n  float p = fmaf(f, 0.0790209f, 0.2249424f);\n  p = fmaf(f, p, 0.6960656f);\n  p = fmaf(f, p, 0.9999893f);\n  return __int_as_float(__float_as_int(p) + ((int)j << 23));\n}'
LEGACY_PK2_CONSTRAINT = 'constexpr bool PK2 = OKC_PK2 && PCHK && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_EXH8 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&'
