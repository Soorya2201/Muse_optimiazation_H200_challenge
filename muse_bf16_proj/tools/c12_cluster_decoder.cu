// Standalone device-side contract for DESIGN_C12_CLUSTER.md.
// Compile only; the Mac development host has no NVIDIA GPU. The full candidate
// will embed these helpers into its producer/decoder/consumer pipeline.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t c12_prmt(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t result;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
  return result;
}

__device__ __forceinline__ uint32_t c12_lop3_select(uint32_t a, uint32_t b, uint32_t mask) {
  uint32_t result;
  // Per bit: mask ? b : a.
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(result) : "r"(a), "r"(b), "r"(mask));
  return result;
}

// Eight logical BF16 weights are encoded as eight low bytes followed by four
// bytes whose low nibbles are codes 0..3 and high nibbles are codes 4..7.
__device__ __forceinline__ void c12_decode8(
    uint32_t low0, uint32_t low1, uint32_t codes, uint32_t base4, uint32_t (&decoded)[4]) {
  const uint32_t high0 =
      c12_lop3_select(codes & 0x07070707u, codes << 4, 0x80808080u) + base4;
  const uint32_t high1 =
      c12_lop3_select((codes >> 4) & 0x07070707u, codes, 0x80808080u) + base4;
  decoded[0] = c12_prmt(low0, high0, 0x5140u);
  decoded[1] = c12_prmt(low0, high0, 0x7362u);
  decoded[2] = c12_prmt(low1, high1, 0x5140u);
  decoded[3] = c12_prmt(low1, high1, 0x7362u);
}

__device__ __forceinline__ uint32_t c12_swizzled_index(uint32_t logical) {
  const uint32_t row = logical >> 6;
  const uint32_t col = logical & 63u;
  return row * 64u + ((((col >> 3) ^ (row & 7u)) << 3) | (col & 7u));
}

// One 128-thread decoder warpgroup expands one 64x64 tile through dynamic
// shared memory. `packed` is 6,144 global bytes, `decoded` receives the
// 8,192-byte 128B-swizzled BF16 tile after its shared-memory construction, and
// each patch is (xor_high << 12) | logical_index. patch_count must be <= 8.
// Launch this standalone proof with 14,336 dynamic shared bytes.
extern "C" __global__ __launch_bounds__(128, 1) void c12_decode_tile(
    const uint8_t* __restrict__ packed,
    const uint32_t* __restrict__ patches,
    uint16_t* __restrict__ decoded,
    uint32_t base4,
    uint32_t patch_count) {
  extern __shared__ __align__(16) uint8_t stage[];
  uint8_t* packed_shared = stage;
  uint16_t* decoded_shared = reinterpret_cast<uint16_t*>(stage + 6144);
  const uint32_t tid = threadIdx.x;

#pragma unroll
  for (uint32_t chunk = tid; chunk < 6144u / 16u; chunk += 128u)
    reinterpret_cast<uint4*>(packed_shared)[chunk] = reinterpret_cast<const uint4*>(packed)[chunk];
  __syncthreads();

#pragma unroll
  for (uint32_t iteration = 0; iteration < 4; ++iteration) {
    const uint32_t group = tid + iteration * 128u;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(packed_shared + group * 12u);
    uint32_t values[4];
    c12_decode8(source[0], source[1], source[2], base4, values);

    const uint32_t row = group >> 3;
    const uint32_t logical_chunk = group & 7u;
    const uint32_t physical_chunk = logical_chunk ^ (row & 7u);
    uint32_t* destination =
        reinterpret_cast<uint32_t*>(decoded_shared + row * 64u + physical_chunk * 8u);
    destination[0] = values[0];
    destination[1] = values[1];
    destination[2] = values[2];
    destination[3] = values[3];
  }
  __syncthreads();

  if (tid < patch_count && tid < 8u) {
    const uint32_t patch = patches[tid];
    const uint32_t logical = patch & 0xFFFu;
    const uint32_t xor_high = (patch >> 12) & 0xFFu;
    const uint32_t physical = c12_swizzled_index(logical);
    reinterpret_cast<uint8_t*>(decoded_shared)[physical * 2u + 1u] ^= static_cast<uint8_t>(xor_high);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t chunk = tid; chunk < 8192u / 16u; chunk += 128u)
    reinterpret_cast<uint4*>(decoded)[chunk] = reinterpret_cast<const uint4*>(decoded_shared)[chunk];
}
