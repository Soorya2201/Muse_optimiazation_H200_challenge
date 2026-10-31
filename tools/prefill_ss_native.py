"""Experimental producer dequantization + shared/shared WGMMA for prefill.

Only research builds may use this transform. Packed weights remain NVFP4;
dequantization occurs inside every timed call. No activation permutation,
extra global pass, allocation, routing or attention change is introduced.
"""

DEQUANT = r'''
// Two otherwise idle producer warps expand one packed weight block in
// place. Every packed word/scale is in registers before the 64-thread
// barrier permits any store to overwrite the packed input region.
DEVI void ss_expand(uint8_t* packed, int dtid) {
  uint4 codes[4];
  uint2 scales[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = dtid + i * 64;
    codes[i] = *reinterpret_cast<const uint4*>(packed + q * 16);
    const int warp = q >> 5, g = (q & 31) >> 2;
    scales[i] = *reinterpret_cast<const uint2*>(packed + 4096 + (warp * 8 + g) * 8);
  }
  named_bar_sync(2, 64);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = dtid + i * 64;
    const int lane = q & 31, row = (q >> 5) * 16 + (lane >> 2), kt = (lane & 3) * 2;
    const uint4 c = codes[i];
    const uint2 s = scales[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t word = j == 0 ? c.x : j == 1 ? c.y : j == 2 ? c.z : c.w;
      uint32_t a[4];
      convert8(word, scale_pair_sel(j < 2 ? s.x : s.y, j & 1), a);
      const int k = j * 16 + kt;
      // Same BF16 register fragments as the parent, laid out K-major SW128.
      const int lo = row * 128, hi = (row + 8) * 128;
      const int sw = (row & 7) * 16;
      *reinterpret_cast<uint32_t*>(packed + lo + ((k * 2) ^ sw)) = a[0];
      *reinterpret_cast<uint32_t*>(packed + hi + ((k * 2) ^ sw)) = a[1];
      *reinterpret_cast<uint32_t*>(packed + lo + (((k + 8) * 2) ^ sw)) = a[2];
      *reinterpret_cast<uint32_t*>(packed + hi + (((k + 8) * 2) ^ sw)) = a[3];
    }
  }
  // Each generic writer orders its own stores before async WGMMA reads.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
'''


def _mma(n):
    outputs = ', '.join('%' + str(i) for i in range(n // 2))
    constraints = ', '.join('"+f"(acc[' + str(i) + '])' for i in range(n // 2))
    return f'''
template <> DEVI void ss_mma<{n}>(float (&acc)[128], uint64_t ad, uint64_t bd) {{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{outputs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;"
               : {constraints} : "l"(ad), "l"(bd));
}}
'''


def shared_weight_native(source: str, consumer_regs: int = 184, all_widths: bool = False) -> str:
    assert consumer_regs in (184, 216)
    producer_regs = 120 if consumer_regs == 184 else 72
    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    replace('constexpr int SLOT = 43008;', 'constexpr int SLOT = 49152;')
    replace('constexpr int STG_BYTES = 45056;', 'constexpr int STG_BYTES = 34816;')
    replace('constexpr int NSLOT = 4;', 'constexpr int NSLOT = 4;\nstatic_assert(1024 + NSLOT * SLOT + 34816 <= 232448, "SS shared memory exceeds H200 block budget");')
    if all_widths:
        for width,old_value in ((32,4),(64,2),(128,2)):
            replace(f'#define MOE_KPS{width} {old_value}', f'#define MOE_KPS{width} 1')
        replace('DEVI int kps_for(int n) { return n >= 160 ? 1 : (n <= 32 ? MOE_KPS32 : (n <= 64 ? MOE_KPS64 : MOE_KPS128)); }',
                'DEVI int kps_for(int n) { return 1; }')
    anchor = 'template <int N, int CB = 128>\nDEVI void consumer_item('
    start = source.index(anchor)
    end = source.index('\ntemplate <bool GATED>', start)
    old = source[start:end]
    epi = old[old.index('  // ---- epilogue through the dedicated staging buffer ----'):]
    helper = DEQUANT + '\ntemplate <int N> DEVI void ss_mma(float (&acc)[128], uint64_t ad, uint64_t bd);\n'
    widths=(32,64,128,160,192,256) if all_widths else (160,192,256)
    helper += ''.join(_mma(n) for n in widths)
    helper += r'''
template <int N, int CB = 128>
DEVI void consumer_item_ss(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                          int e, int row0, int ntile, bool GATED, int* done = nullptr,
                          unsigned long long* t_mid = nullptr, volatile long long* pubq = nullptr, int nreal = 256) {
  static_assert(N == 160 || N == 192 || N == 256, "SS is restricted to wide, KPS1 tiles");
  const int ctid = threadIdx.x - CB, wg = ctid >> 7, w = (ctid >> 5) & 3;
  const int lane = ctid & 31, g = lane >> 2, t = lane & 3;
  float acc[128];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  int prev = -1;
#pragma unroll 1
  for (int it = 0; it < KS; ++it) {
    mbar_wait(bars + 128 + 8 * slot, ph);
    const uint32_t base = smem_u32(ring + slot * SLOT);
    const uint64_t ad = desc_sw128(base + N * 128 + wg * 8192);
    const uint64_t bd = desc_sw128(base);
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) ss_mma<N>(acc, ad + 2 * j, bd + 2 * j);
    wg_commit();
    wg_wait<1>();
    if (prev >= 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
    prev = slot;
    if (++slot == NSLOT) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>();
  mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  if (t_mid) {
    unsigned long long stamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp));
    *t_mid = stamp;
  }
''' + epi
    if all_widths:
        helper=helper.replace('static_assert(N == 160 || N == 192 || N == 256, "SS is restricted to wide, KPS1 tiles");',
                              'static_assert(N == 32 || N == 64 || N == 128 || N == 160 || N == 192 || N == 256, "SS supports the complete parent width set");')
    # Put helper after consumer_item so all parent epilogue macros exist.
    source = source[:end] + '\n' + helper + source[end:]

    start = source.index('__global__ void __launch_bounds__(MOE_FUSED_THREADS, 1) moe_fused_persistent(')
    end = source.index('// ---- routing plan:', start)
    body = source[start:end]
    assert body.count('mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);') == 1
    body = body.replace('mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);',
                        'mbar_init(ibar + 16 + 8 * k, 320);')
    anchor = '    for (int k = 0; k < 2; ++k) {'
    assert body.count(anchor) == 1
    body = body.replace(anchor, '    for (int s = 0; s < NSLOT; ++s) mbar_init(bars + 128 + 8 * s, 64);\n' + anchor)
    body = body.replace('setmaxnreg.dec.sync.aligned.u32 40;', f'setmaxnreg.dec.sync.aligned.u32 {producer_regs};')
    body = body.replace('setmaxnreg.inc.sync.aligned.u32 232;', f'setmaxnreg.inc.sync.aligned.u32 {consumer_regs};')
    # Two previously idle producer warps, leaving DMA and publisher unchanged.
    anchor = '''      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  } else {'''
    assert body.count(anchor) == 1
    body = body.replace(anchor, r'''      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    } else {
      static_assert(!MOE_CONS_FIRST && MOE_ALLLANE_RELEASE, "SS requires the original full producer WG and all-lane releases");
      const int dtid = threadIdx.x - 64;
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 8 * k, iph);
        const int item = iring[k];
        mbar_arrive(ibar + 16 + 8 * k); // each of 64 metadata readers releases its own read
        if (item >= nitems) break;
        k ^= 1;
        if (k == 0) iph ^= 1;
        const bool second = item >= n1;
        const Args& p = second ? fa.g2 : fa.g1;
        const int it2 = second ? item - n1 : item;
        const int4 ti = p.tiles[it2 / p.NT];
        // Every item has an even number of complete four-slot ring cycles.
        // Narrow items do not change decoded-barrier parity; wide items
        // have 64 stages and also restore that parity before the next item.
        if (ti.z < 160) continue;
#pragma unroll 1
        for (int it = 0; it < KS; ++it) {
          const int s = it & 3;
          const uint32_t ph = (uint32_t)(it >> 2) & 1u;
          mbar_wait(bars + 8 * s, ph);
          ss_expand(ring + s * SLOT + ti.z * 128, dtid);
          mbar_arrive(bars + 128 + 8 * s);
        }
      }
    }
  } else {''')
    # Inline full SS bodies for wide arms only. Preserve old narrow paths.
    body = body.replace('if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<160>',
                        'if constexpr (false) consumer_item_ni<160>')
    body = body.replace('if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<192>',
                        'if constexpr (false) consumer_item_ni<192>')
    if all_widths:
        body=body.replace('        if (ti.z < 160) continue;',
                          '        // Every width uses KPS1 and the SS/dequant pipeline in this arm.')
    for n in widths:
        before = f'consumer_item<{n}, MOE_CB>'
        assert body.count(before) == 1
        body = body.replace(before, f'consumer_item_ss<{n}, MOE_CB>')
    source = source[:start] + body + source[end:]
    return source
