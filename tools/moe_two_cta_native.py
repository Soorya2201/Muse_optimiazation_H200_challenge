"""Experimental 128-output-row WGS units and two consumer warpgroups/CTA.

Uses the original packed WGS weight pointers: each new half unit selects one
contiguous half of a parent 36KiB stage. No weight precision/layout change.
Compile with NW8, FT32 and a264-block grid only after verifying2CTA residency.
"""

def two_cta_native(source: str) -> str:
    def replace(before,after,count=1):
        nonlocal source
        assert source.count(before)==count,(before,source.count(before))
        source=source.replace(before,after)
    replace('#define OKC_FT_T 64','#define OKC_FT_T 32')
    # Non-WGS functions are compiled away in this independent WGS module.
    replace('constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;',
            'constexpr int UNIT_CODES = 16384, UNIT_SC = 2048, UNIT_BYTES = UNIT_CODES + UNIT_SC;')
    replace('#define OKC_WGSNX 3', '#define OKC_WGSNX 2')
    replace('constexpr int WGS_SLOTS = 160;', 'constexpr int WGS_SLOTS = 320;')
    replace('WGS_SLOTS * 4', 'WGS_SLOTS * 2',count=7)
    # Both phases have twice as many128-row units, still16K256 stages/unit.
    replace('const int ubl = pb ? 4 : 5;', 'const int ubl = pb ? 5 : 6;',count=2)
    replace('constexpr int UBL = PB ? 4 : 5;', 'constexpr int UBL = PB ? 5 : 6;',count=3)
    replace('(9 - pb)', '(10 - pb)',count=4)
    # Prefix costs count pairs of new stages, preserving32-bit products
    # for384tokens/all-local routing at a264-block grid.
    replace('return (lo << (10 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);',
            'return (lo << (10 - pb)) + 2 * (int)((x - (uint32_t)pre[lo]) / cs);')
    replace('((s & ((1 << (10 - pb)) - 1)) + 1)',
            '(((s & ((1 << (10 - pb)) - 1)) >> 1) + 1)')
    replace('__launch_bounds__(NTL, 1)', '__launch_bounds__(NTL, 2)')
    replace('constexpr int WGS_FIXW = WGS_NT * 512;',
            'constexpr int WGS_FIXW = 8 * 512;')
    # FT32 chunks are8-aligned, so only the final chunk of each of32
    # experts needs padding. R<=EMAX+7*32; retain parent xB pointer offset.
    replace('constexpr int WGS_XROWS = EMAX + 8 * JMAX;',
            'constexpr int WGS_XROWS = EMAX + 8 * (32 + EMAX / 64 + 8);')
    replace('slot * 4 + wg','slot * 2 + wg',count=3)
    replace('(ub >> 1)', '(ub >> 2)',count=2)
    replace('* 16 + wg * 4 + wp', '* 8 + wg * 4 + wp',count=2)
    # The parent layout remains [old256-row unit][16stages][4WG].
    replace('const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;',
            'const unsigned char* src = base + ((size_t)(((S.jE[j] << (ubl - 1)) + (ub >> 1)) * 16 + st)) * (size_t)(2 * UNIT_BYTES) + (size_t)(ub & 1) * UNIT_BYTES;')
    # FT32 admits only NTT1..4. Remove unused larger specializations rather
    # than leave out-of-bounds d[4..7] references in explicit definitions.
    a=source.index('template <> struct WGSMMA<5>');b=source.index('// two producer warps:',a)
    source=source[:a]+source[b:]
    a=source.index('      case 5: wgs_seg<PB, 5>');b=source.index('\n    }',a)
    source=source[:a]+'      default: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;'+source[b:]
    # Existing empty-job guard, attributed to the pinned Apache2 source.
    replace('  const int nj = S.njF + S.njS;\n  wgs_prefix(S, nj);', '''  const int nj = S.njF + S.njS;
  // trxphxle/okc-2026-muse-kernels@72db638, Apache-2.0 empty-job guard.
  if (!nj) { S.wtot=S.wtotB=0; S.wgr[0]=S.wgr[1]=S.wgr[2]=S.wgr[3]=0; return; }
  wgs_prefix(S, nj);''')
    #256 consumers gather the same4096-wide row as512 parent consumers.
    #The two logical lane batches retain the exact KPERM3 destination map.
    a=source.index('  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");')
    b=source.index('\n  CTA_SYNC();',a)
    source=source[:a]+'''  static_assert(NT * 16 == HD, "two 16B chunks per thread per row");
  constexpr int GB = 4;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[2][GB]; uint4* d[2][GB];
#pragma unroll
    for (int part=0;part<2;++part) {
      const int ftid=tid+part*NT;
#pragma unroll
      for (int k=0;k<GB;++k) {
        const int q=qb+k; d[part][k]=nullptr;
        if (q<q1) {
          int jj=0; while (S.jrow[jj+1]<=q) ++jj;
          const int r=q-S.jrow[jj],cnt=S.jC[jj];
          if (r<cnt) {
            const int ntt=(cnt+7)>>3,G=ftid&31;
            d[part][k]=reinterpret_cast<uint4*>(p.xA+(size_t)S.jrow[jj]*HD)+((((ftid>>5)*ntt+(r>>3))*32+(ftid&31))*8+(r&7));
            const unsigned* xs32=reinterpret_cast<const unsigned*>(p.x+(size_t)(S.ent[S.jS[jj]+r]>>3)*HD+(ftid>>5)*256)+(8*((G>>1)&3)+2*(G>>3)+(G&1));
            v[part][k]=make_uint4(__ldg(xs32),__ldg(xs32+32),__ldg(xs32+64),__ldg(xs32+96));
          }
        }
      }
    }
#pragma unroll
    for (int part=0;part<2;++part)
#pragma unroll
      for (int k=0;k<GB;++k) if (d[part][k]) *d[part][k]=v[part][k];
  }'''+source[b:]
    #Final norm now owns16rather than8columns/thread. Complete rows/route
    #order/FP32 accumulation/mandatory final BF16 rounding are preserved.
    a=source.index('    static_assert(HD == NT * 8, "8 columns per thread");')
    b=source.index('\n#elif OKC_FINCOL',a)
    source=source[:a]+'''    static_assert(HD == NT * 16, "16 columns per thread");
    float* red=reinterpret_cast<float*>(xbuf);
    const int lane=tid&31,warp=WARP_IDX(tid);
    for(int tk=blockIdx.x;tk<p.M;tk+=gridDim.x) {
      int qv[8]; float ps[8];
#pragma unroll
      for(int sl=0;sl<8;++sl) {
        qv[sl]=S.pos[tk*8+sl];ps[sl]=0.f;
        if(qv[sl]>=0) {
          const float4* src=reinterpret_cast<const float4*>(p.ybuf+(size_t)qv[sl]*HD)+tid*4;
#pragma unroll
          for(int chunk=0;chunk<4;++chunk) {
            const float4 v=__ldcg(src+chunk);
            ps[sl]+=v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;
          }
        }
      }
#pragma unroll
      for(int sl=0;sl<8;++sl) if(qv[sl]>=0) {
#pragma unroll
        for(int o=16;o>0;o>>=1) ps[sl]+=__shfl_xor_sync(0xffffffffu,ps[sl],o);
      }
      if(lane==0) {
#pragma unroll
        for(int sl=0;sl<8;++sl) red[warp*8+sl]=ps[sl];
      }
      CTA_SYNC();
      if(tid<8) {float sum=0.f;for(int w=0;w<NW;++w) sum+=red[w*8+tid];red[NW*8+tid]=sum;}
      CTA_SYNC();
      // Two8-column output passes keep eight accumulators live, matching
      // the parent. One16-column pass spilled four registers in full codegen.
#pragma unroll
      for(int part=0;part<2;++part) {
        float acc[8]={};
#pragma unroll
        for(int sl=0;sl<8;++sl) if(qv[sl]>=0) {
          const float r=p.topk_w[tk*8+sl]*rsqrtf(red[NW*8+sl]*(1.f/HD)+1e-8f);
          const float4* src=reinterpret_cast<const float4*>(p.ybuf+(size_t)qv[sl]*HD)+(tid+part*NT)*2;
          const float4 a=__ldcg(src),b=__ldcg(src+1);
          acc[0]+=r*a.x;acc[1]+=r*a.y;acc[2]+=r*a.z;acc[3]+=r*a.w;
          acc[4]+=r*b.x;acc[5]+=r*b.y;acc[6]+=r*b.z;acc[7]+=r*b.w;
        }
        uint4 o;uint32_t* ow=reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for(int i=0;i<4;++i) {__nv_bfloat162 v=__floats2bfloat162_rn(acc[2*i],acc[2*i+1]);ow[i]=*reinterpret_cast<uint32_t*>(&v);}
        *reinterpret_cast<uint4*>(p.out+(size_t)tk*HD+(tid+part*NT)*8)=o;
      }
      CTA_SYNC();
    }
  }'''+source[b:]
    # PRMT fast builds never execute the LUT loop(tid>=0), but NVCC keeps
    # the signed-negative path. Remove it at preprocessing time to avoid
    # register pressure from unreachable floating-point conversion code.
    anchor='  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {'
    assert source.count(anchor)==1
    a=source.index(anchor);start=source.index('{',a);depth=1;b=start+1
    while depth:
        if source[b]=='{':depth+=1
        elif source[b]=='}':depth-=1
        b+=1
    source=source[:a]+'#if !OKC_PRMTSC\n'+source[a:b]+'\n#endif'+source[b:]
    # Rematerialize generation state after the compute phases. The original
    # common subexpressions kept ctr+3 and the initial generation live across
    # both phases, spilling them under the two-resident-block register cap.
    replace('    int* bc = p.ctr + 4 + (S.gen & 1);', r'''    int generation;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(generation) : "r"(smem_u32(&S.gen)) : "memory");
    int* bc = p.ctr + 4 + (generation & 1);''')
    replace('    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);', r'''    if (old == (int)gridDim.x - 1) {
      int* fresh_ctr=p.ctr;
      asm volatile("" : "+l"(fresh_ctr) :: "memory");
      atomicAdd(fresh_ctr + 3, 1);
    }''')
    # Keep the launch counter pointer in shared storage and reload at use,
    # rather than carrying one global pointer across the full compute body.
    replace('struct Smem {', 'struct Smem {\n  unsigned long long ctr_reload;')
    helper=r'''__device__ __forceinline__ int* two_cta_counter() {
  unsigned long long ptr;
  asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(ptr) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ctr_reload)) : "memory");
  return reinterpret_cast<int*>(ptr);
}
'''
    a=source.index('template <bool BIG>\n__global__ void __launch_bounds__(NTL, 2)');b=source.index('\n}  // namespace okc',a)
    kernel=source[a:b]
    assert kernel.count('p.ctr')>=5
    kernel=kernel.replace('p.ctr','two_cta_counter()')
    anchor='  const int tid = threadIdx.x;\n  DBG(0);'
    assert kernel.count(anchor)==1
    kernel=kernel.replace(anchor,'  const int tid = threadIdx.x;\n  if (tid==0) S.ctr_reload=(unsigned long long)p.ctr;\n  DBG(0);')
    source=source[:a]+helper+kernel+source[b:]
    # Only this experimental WGS module doubles the parent's one-per-SM
    # launch grid. Other variants retain the original host grid unchanged.
    replace('  okc::Params p;\n  p.wA =', '  grid *= 2;\n  okc::Params p;\n  p.wA =')
    return source
