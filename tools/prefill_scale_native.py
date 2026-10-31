"""Prospective prefill K64 scale-sharing source edits.

Only independent research builds may use this transform until full H200
validation. Timed weight formats/activations/routing precision are retained.
"""

HELPERS = r'''
DEVI uint32_t scale_pair_reused(uint16_t raw, uint32_t m16) {
#if MOE_FAST_SCALE
  return __byte_perm((uint32_t)raw, 0u, 0x4140) * m16 + 0x77807780u;
#else
  return scale_pair((uint32_t)raw);
#endif
}
// K64 permutation: original index -> physical packed index, independently
// verified against logical_k/packed_k in the CPU model.
DEVI int k64_packed(int k) {
  return (k & 1) | ((k & 2) << 2) | ((k & 12) << 2) | ((k & 48) >> 3);
}
// Distributed 4x4 transpose of uint32 values. Subgroups are the four lanes
// that hold one original K64 row chunk. Eight SHFLs total for the two planes.
DEVI void k64_transpose4(uint32_t (&v)[4], int lane) {
  uint32_t a = (lane & 1) ? v[0] : v[1];
  uint32_t b = (lane & 1) ? v[2] : v[3];
  a = __shfl_xor_sync(0xffffffffu, a, 1, 4);
  b = __shfl_xor_sync(0xffffffffu, b, 1, 4);
  if (lane & 1) { v[0] = a; v[2] = b; }
  else { v[1] = a; v[3] = b; }
  a = (lane & 2) ? v[0] : v[2];
  b = (lane & 2) ? v[1] : v[3];
  a = __shfl_xor_sync(0xffffffffu, a, 2, 4);
  b = __shfl_xor_sync(0xffffffffu, b, 2, 4);
  if (lane & 2) { v[0] = a; v[1] = b; }
  else { v[2] = a; v[3] = b; }
}
DEVI void k64_gather_words(uint32_t (&words)[8], int lane) {
  uint32_t even[4], odd[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { even[i] = words[2*i]; odd[i] = words[2*i+1]; }
  k64_transpose4(even, lane);
  k64_transpose4(odd, lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) { words[i] = even[i]; words[4+i] = odd[i]; }
}
// Each warp owns one complete 64-BF16 row. All old pairs are in registers
// before any lane overwrites their permuted destinations; no global pass or
// extra shared allocation. Caller synchronizes all stmatrix writers first.
template <int N>
DEVI void k64_permute_staging(uint8_t* stg, int ctid) {
  const int lane = ctid & 31;
  for (int row = ctid >> 5; row < N; row += 8) {
    const int swizzle = (row & 7) << 2;
    uint32_t* base = reinterpret_cast<uint32_t*>(stg + row * 128);
    const uint32_t value = base[lane ^ swizzle];
    __syncwarp();
    const int destination = ((lane << 2) & 31) | (lane >> 3);
    base[destination ^ swizzle] = value;
  }
}
'''


def scale_shared_native(source: str) -> str:
    def replace(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (before, source.count(before), count)
        source = source.replace(before, after)

    replace('DEVI uint32_t opaque_zero() {', HELPERS + '\nDEVI uint32_t opaque_zero() {')
    # Restrict consumer edits to its complete body. Old nonpersistent kernels
    # are not part of this experimental fused-only API.
    start=source.index('template <int N, int CB = 128>')
    end=source.index('\ntemplate <bool GATED>',start)
    body=source[start:end]
    body=body.replace('uint2 nsc = *reinterpret_cast<const uint2*>(',
                      'uint16_t nsc = *reinterpret_cast<const uint16_t*>(')
    body=body.replace('nsc = *reinterpret_cast<const uint2*>(',
                      'nsc = *reinterpret_cast<const uint16_t*>(')
    body=body.replace('const uint2 sc = nsc;', 'const uint32_t scale = scale_pair_reused(nsc, m16);')
    body=body.replace(' + soff)', ' + soff + 2 * t)')
    body=body.replace(' + soff;', ' + soff + 2 * t;')
    old='          const uint32_t sw = j < 2 ? sc.x : sc.y;'
    # The four call sites have different comments/indentation; strip only
    # these otherwise dead construction lines, preserving every arithmetic op.
    lines=body.splitlines()
    lines=[line for line in lines if 'const uint32_t sw = j < 2 ? sc.x : sc.y;' not in line]
    body='\n'.join(lines)+'\n'
    assert body.count('scale_pair_sel(sw, j & 1, m16)') == 3
    body=body.replace('scale_pair_sel(sw, j & 1, m16)', 'scale')
    assert ' sc.' not in body and 'const uint2 sc' not in body
    staging='''    if (pubq) {  // fused sm_90: the publisher warp copies the staged epilogue tile out'''
    # Use a stable full prefix; one extra consumer CTA barrier orders all
    # earlier stmatrix writes before any warp's in-place read/overwrite.
    anchor='''    if (pubq) {  // fused sm_90: the publisher warp copies the staged tile out (bulk copy) and then publishes `done`'''
    assert body.count(anchor)==1
    body=body.replace(anchor, '''    named_bar_sync(1, 256);
    k64_permute_staging<N>(stg, ctid);
''' + anchor)
    source=source[:start]+body+source[end:]
    # Normalize and round first, then move the existing BF16 pairs by shuffles.
    replace('''  const int stg = tid >> 2, ch0 = 2 * (tid & 3);
#pragma unroll
  for (int k = 0; k < 8; ++k) {''', '''  k64_gather_words(outw, tid & 31);
  const int stg = tid >> 2, ch0 = 2 * (tid & 3);
#pragma unroll
  for (int k = 0; k < 8; ++k) {''')
    return source
