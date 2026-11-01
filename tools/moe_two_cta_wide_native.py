"""Two-CTA/FT64 alternative: reuse routing scratch as the later weight ring.

Independent fast-PRMT WGS module only. Keeps original FT64 jobs and packed
weight pointers. Two weight slots/two X slots; actual residency and runtime
remain unknown until linked H200 validation. Does not modify immutable V41.
"""
from moe_two_cta_native import two_cta_native


def two_cta_wide_native(original: str) -> str:
    source = two_cta_native(original)

    def replace(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (before, source.count(before))
        source = source.replace(before, after)

    replace('#define OKC_FT_T 32', '#define OKC_FT_T 64')
    replace('constexpr int WGS_NS = 3;', 'constexpr int WGS_NS = 2;')
    # Restore exactly the original NTT5..8 implementations and dispatch.
    lo = original.index('template <> struct WGSMMA<5>')
    hi = original.index('// two producer warps:', lo)
    replace('// two producer warps:', original[lo:hi]+'// two producer warps:')
    lo = original.index('      case 5: wgs_seg<PB, 5>')
    hi = original.index('\n    }', lo)
    replace('      default: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;', original[lo:hi])
    replace('  uint32_t lut[256];', '#if !OKC_PRMTSC\n  uint32_t lut[256];\n#endif')
    replace('  signed char eid[EMAX];', '  // Route IDs occupy the future weight ring until gather completes.')
    replace('  short ent[EMAX];', '  // Sorted route entries use the following disjoint ring bytes.')
    source = source.replace('S.eid', 'two_cta_route_ids()').replace('S.ent', 'two_cta_route_entries()')
    helper = r'''
// IDs and entries coexist during routing, then both become dead after gather.
// The first weight stage overwrites their storage only after the full join.
static_assert(OKC_WGS && OKC_PRMTSC, "independent fast-scale WGS module only");
__device__ __forceinline__ signed char* two_cta_route_ids() {
  return reinterpret_cast<signed char*>(smem_raw + ((sizeof(Smem)+127)/128)*128);
}
__device__ __forceinline__ short* two_cta_route_entries() {
  return reinterpret_cast<short*>(smem_raw + ((sizeof(Smem)+127)/128)*128 + EMAX);
}
'''
    replace('extern __shared__ __align__(16) unsigned char smem_raw[];',
            'extern __shared__ __align__(16) unsigned char smem_raw[];\n'+helper)
    # All consumer generic reads/writes of the overlaid range must precede
    # the producer's async writes. Each consumer fences before the full-CTA
    # join; producer bulk copies are after that same join.
    replace('  wgs_gather(p, S);\n  DBG(7);\n  WS_JOIN();',
            '  wgs_gather(p, S);\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  DBG(7);\n  WS_JOIN();')
    return source
