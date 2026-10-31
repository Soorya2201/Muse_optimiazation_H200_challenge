"""One-body stage-loop hypothesis; native source only, no timed dispatch."""


def compact_stage(source: str) -> str:
    start=source.index('// one segment (')
    end=source.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase',start)
    body=source[start:end]
    old='  auto stage = [&](auto first) {\n    constexpr bool FIRST = decltype(first)::value;'
    assert body.count(old)==1
    body=body.replace(old,'  auto stage = [&](bool FIRST) {')
    old='  stage(std::true_type{});\n#pragma unroll 1\n  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});'
    assert body.count(old)==1
    body=body.replace(old,'#pragma unroll 1\n  for (int st = s0; st < s1; ++st) stage(st == s0);')
    return source[:start]+body+source[end:]


def zero_mma_stage(source: str) -> str:
    """Initialize via one zero-A MMA, then use one uniform accumulator loop.

    Compiler hypothesis only: the extra MMA/first-slot wait can outweigh
    instruction-footprint savings. No scalar zeroing of async accumulators.
    """
    source=compact_stage(source)
    start=source.index('// one segment (')
    end=source.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase',start)
    body=source[start:end]
    old='  const unsigned char* pc0 = ring + wg * WGS_PIECE;\n'
    assert body.count(old)==1
    body=body.replace(old,old+'''  // An extra zero-A MMA initializes async accumulators without a scalar
  // access or a conditional MMA0/accumulate merge inside the stage loop.
  const int init_xs=seq%WGS_NX;
  wgs_wait(S.bars+BX_F+init_xs,(uint32_t)(seq/WGS_NX)&1u,5000000+seq);
  wg_fence();
  WGSMMA<NTT>::mma0(d,0u,0u,0u,0u,xd0+(uint64_t)(uint32_t)(init_xs*(WGS_XS>>4)));
  wg_commit();
''')
    old='''          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));'''
    assert body.count(old)==1
    body=body.replace(old,'''          WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                           xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));''')
    return source[:start]+body+source[end:]


def scalar_zero_stage(source: str) -> str:
    """Compiler-only alternative: scalar initialization plus explicit fence."""
    source=zero_mma_stage(source)
    old='''  const int init_xs=seq%WGS_NX;
  wgs_wait(S.bars+BX_F+init_xs,(uint32_t)(seq/WGS_NX)&1u,5000000+seq);
  wg_fence();
  WGSMMA<NTT>::mma0(d,0u,0u,0u,0u,xd0+(uint64_t)(uint32_t)(init_xs*(WGS_XS>>4)));
  wg_commit();'''
    assert source.count(old)==1
    source=source.replace(old,'''#pragma unroll
  for(int nt=0;nt<NTT;++nt) {
#pragma unroll
    for(int v=0;v<4;++v)d[nt][v]=0.f;
  }
  wg_fence();''')
    return source.replace('''  // An extra zero-A MMA initializes async accumulators without a scalar
  // access or a conditional MMA0/accumulate merge inside the stage loop.''',
    '''  // Scalar zero initialization precedes the explicit ownership fence;
  // no extra MMA or activation-slot wait is added before the stage loop.''')
