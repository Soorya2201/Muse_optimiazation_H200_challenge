"""Exact, independently reproducible native-source edits for decoder WGS probes.

These are experiments, not the canonical submission. The register split needs
all 128 threads of the producer warpgroup, including 64 otherwise idle threads.
Those threads participate in the one initial full-block WS_JOIN before exiting.
"""


def register_split(source: str, group_steps: int) -> str:
    assert group_steps in (2, 4)

    def replace(before: str, after: str) -> None:
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    replace("constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
            "constexpr int NTL = NT + (OKC_WGS ? 128 : OKC_WS ? 32 : 0);")
    replace("""#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif""", """#if OKC_WGS
  static_assert(NT == 512, "register split requires four consumer warpgroups");
  if (tid >= NT) {
    // The whole fifth warpgroup releases registers before splitting by role.
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\\n" ::: "memory");
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else if (tid < NT + 64) wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else WS_JOIN();  // idle padding warps join the initial routing/gather barrier
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\\n" ::: "memory");
#endif""")
    if group_steps == 4:
        # Flush after eight k16 operations in both arms: GS2 CI3 -> GS4 CI1.
        replace("#define OKC_WGSFCI 3", "#define OKC_WGSFCI 1")
        replace("#define OKC_WGSGS 2  //", "#define OKC_WGSGS 4  //")
    return source


def delayed_activation_wait(source: str) -> str:
    """Decode the first independent A fragments before waiting for X.

    All generic weight loads remain after BW_F. The BX_F wait still precedes
    the first WGMMA access to X, and the operand fence/commit/release sequence
    is unchanged. This changes scheduling only, not precisions or work.
    """
    before = "    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n"
    assert source.count(before) == 1
    source = source.replace(before, "")
    anchor = """        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {"""
    assert source.count(anchor) == 1
    after = """        if (ci == 0)
          wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
""" + anchor
    return source.replace(anchor, after)


def rounded_widths(source: str) -> str:
    """Experimental four-width WGS body, preserving global activation layouts.

    Actual n-tiles 3/5/6/7 issue widths 4/8/8/8. The activation producer
    initializes only those additional shared-memory columns, in this call,
    after the consumers release the slot. All producer lanes participate.
    Valid outputs retain the parent's K order and BF16/FP32 types. Dummy
    columns never reach partial sums, epilogues, or readiness counters.
    CPU layout verification and compilation are not GPU validation.
    """
    def replace(before: str, after: str) -> None:
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    start = source.index('__device__ void wgs_xproducer(')
    end = source.index('// phase A\'s x gather', start)
    producer = source[start:end]
    assert producer.count('  if ((threadIdx.x & 31) != 0) return;') == 1
    producer = producer.replace('  if ((threadIdx.x & 31) != 0) return;',
                                '  const int xlane = threadIdx.x & 31;')
    before = '      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);'
    assert producer.count(before) == 1
    producer = producer.replace(before, '      if (n >= WGS_NX && xlane == 0) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);')
    producer = producer.replace('if (!pb && !(OKC_WGSX & 8))', 'if (xlane == 0 && !pb && !(OKC_WGSX & 8))')
    producer = producer.replace('if (pb && !(OKC_WGSX & 8))', 'if (xlane == 0 && pb && !(OKC_WGSX & 8))')
    before = '      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n"'
    assert producer.count(before) == 1
    producer = producer.replace(before, '''      // Lane 0 has acquired the old consumer release and global readiness.
      // Synchronize before any lane writes this reused shared slot.
      __syncwarp();
      const uint32_t ntiles = (uint32_t)npad >> 3;
      const uint32_t wide = ntiles <= 2 ? ntiles : ntiles <= 4 ? 4u : 8u;
      if (wide != ntiles) {
        for (uint32_t b = bytes + (uint32_t)xlane * 16u; b < wide * 4096u; b += 512u)
          *reinterpret_cast<uint4*>(xring + xs * WGS_XS + b) = make_uint4(0, 0, 0, 0);
        // Each writer orders its own generic stores before async WGMMA reads.
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
        __syncwarp();
      }
      if (xlane == 0) {
''' + before)
    before = '      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }'
    assert producer.count(before) == 1
    producer = producer.replace(before, '      }\n' + before)
    source = source[:start] + producer + source[end:]

    before = '''    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32'''
    after = '''    for (int nt = 0; nt < NTT; ++nt)
      if (nt < ((S.jC[u >> UBL] + 7) >> 3))
      asm volatile("red.global.add.v4.f32'''
    replace(before, after)
    replace('  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);',
            '  for (int nt = 0; nt < NTT; ++nt)\n    if (nt < ((S.jC[j] + 7) >> 3)) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);')
    start = source.index('    switch (ntt) {')
    end = source.index('\n    gs += s1 - s0;', start)
    source = source[:start] + '''    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3:
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }''' + source[end:]
    return source
