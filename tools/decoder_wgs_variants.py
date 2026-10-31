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
