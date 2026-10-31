"""Research combination: K64 packed weights -> standard shared BF16 operand.

Unlike the previous K64 prefill attempt, X and the GEMM1 output retain their
original layouts. The producer's expansion stores undo the K permutation.
Requires the matching lossless prepare-time K64 weight repack.
"""
from prefill_ss_native import DEQUANT, shared_weight_native

K64_DEQUANT = r'''
DEVI void ss_expand(uint8_t* packed, int dtid) {
  uint4 codes[4];
  uint16_t scales[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = dtid + i * 64;
    codes[i] = *reinterpret_cast<const uint4*>(packed + q * 16);
    const int warp = q >> 5, lane = q & 31, g = lane >> 2, t = lane & 3;
    scales[i] = *reinterpret_cast<const uint16_t*>(packed + 4096 + (warp * 8 + g) * 8 + 2 * t);
  }
  named_bar_sync(2, 64);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = dtid + i * 64;
    const int lane = q & 31, row = (q >> 5) * 16 + (lane >> 2), t = lane & 3;
    const uint4 c = codes[i];
    const uint32_t scale = scale_pair_sel((uint32_t)scales[i], 0);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t a[4], b[4];
      convert8(half ? c.z : c.x, scale, a);
      convert8(half ? c.w : c.y, scale, b);
      // Physical K64 steps (2*half,2*half+1) cover consecutive original
      // groups of eight values in the lane's original 16-value scale block.
      // Permutation preserves BF16 pairs, permitting aligned vector stores.
      const int k = t * 16 + half * 8;
      const int sw = (row & 7) * 16;
      const int lo = row * 128 + ((k * 2) ^ sw);
      const int hi = (row + 8) * 128 + ((k * 2) ^ sw);
      *reinterpret_cast<uint4*>(packed + lo) = make_uint4(a[0], a[2], b[0], b[2]);
      *reinterpret_cast<uint4*>(packed + hi) = make_uint4(a[1], a[3], b[1], b[3]);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
'''


def scale_reused_shared_native(source: str) -> str:
    source = shared_weight_native(source,184,True,strict_roles=True)
    assert source.count(DEQUANT)==1
    return source.replace(DEQUANT,K64_DEQUANT)
