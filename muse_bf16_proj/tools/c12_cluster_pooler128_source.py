#!/usr/bin/env python3
"""Emit the compile-only pooler-m128 C12/SS-WGMMA/DSMEM pipeline prototype."""

from __future__ import annotations


def wgmma_ss_128() -> str:
    registers = ", ".join(f"%{index}" for index in range(64))
    outputs = ", ".join(f'"+f"(acc[{index}])' for index in range(64))
    return f'''__device__ __forceinline__ void wgmma_ss_128(float* acc, uint64_t weight, uint64_t x) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %66, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{{{registers}}}, %64, %65, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {outputs}
      : "l"(weight), "l"(x), "r"(1));
}}'''


CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t saddr(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("{ .reg .b64 state; mbarrier.arrive.expect_tx.shared.b64 state, [%0], %1; }"
               :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{ .reg .b64 state; mbarrier.arrive.shared.b64 state, [%0]; }" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                 " selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(bar), "r"(parity) : "memory");
  } while (!ready);
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;"
                 " selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(bar), "r"(parity) : "memory");
  } while (!ready);
}
__device__ __forceinline__ void bulk_g2s(uint32_t destination, const void* source, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(destination), "l"(source), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_x(
    uint32_t destination, const CUtensorMap* map, uint32_t bar, int k, int row) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3}], [%4];"
               :: "r"(destination), "l"(map), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t address) {
  uint64_t descriptor = static_cast<uint64_t>((address & 0x3FFFFu) >> 4);
  descriptor |= static_cast<uint64_t>(1) << 16;
  descriptor |= static_cast<uint64_t>(64) << 32;
  descriptor |= static_cast<uint64_t>(1) << 62;
  return descriptor;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t c12_prmt(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t result;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
  return result;
}
__device__ __forceinline__ uint32_t c12_lop3_select(uint32_t a, uint32_t b, uint32_t mask) {
  uint32_t result;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(result) : "r"(a), "r"(b), "r"(mask));
  return result;
}
__device__ __forceinline__ void c12_decode8(
    uint32_t low0, uint32_t low1, uint32_t codes, uint32_t base4, uint32_t (&decoded)[4]) {
  const uint32_t high0 = c12_lop3_select(codes & 0x07070707u, codes << 4, 0x80808080u) + base4;
  const uint32_t high1 = c12_lop3_select((codes >> 4) & 0x07070707u, codes, 0x80808080u) + base4;
  decoded[0] = c12_prmt(low0, high0, 0x5140u);
  decoded[1] = c12_prmt(low0, high0, 0x7362u);
  decoded[2] = c12_prmt(low1, high1, 0x5140u);
  decoded[3] = c12_prmt(low1, high1, 0x7362u);
}
__device__ __forceinline__ uint32_t swizzled_index(uint32_t logical) {
  const uint32_t row = logical >> 6;
  const uint32_t col = logical & 63u;
  return row * 64u + ((((col >> 3) ^ (row & 7u)) << 3) | (col & 7u));
}

@@WGMMA@@

extern "C" __global__ __launch_bounds__(288, 1) __cluster_dims__(2, 1, 1)
void c12_pooler128(
    const __grid_constant__ CUtensorMap map_x,
    const uint8_t* __restrict__ packed,
    const uint32_t* __restrict__ patches,
    const uint8_t* __restrict__ patch_counts,
    __nv_bfloat16* __restrict__ output,
    uint32_t base4) {
  constexpr int N = 4096;
  constexpr int K_UNITS = 128;
  constexpr int TM = 128;
  constexpr int NSTAGE = 6;
  constexpr int PACKED_BYTES = 6144;
  constexpr int DECODED_BYTES = 8192;
  constexpr int X_BYTES = 16384;
  constexpr int STAGE_BYTES = PACKED_BYTES + DECODED_BYTES + X_BYTES;
  constexpr int NACC = 64;
  constexpr int REDUCTION_STRIDE = NACC + 4;

  extern __shared__ __align__(1024) uint8_t raw[];
  const uint32_t raw_address = saddr(raw);
  const uint32_t base = (raw_address + 1023u) & ~1023u;
  uint8_t* shared = raw + (base - raw_address);
  const uint32_t full = base + NSTAGE * STAGE_BYTES;
  const uint32_t decoded_ready = full + NSTAGE * 8;
  const uint32_t empty = decoded_ready + NSTAGE * 8;
  const uint32_t qbar = empty + NSTAGE * 8;
  float* peer = reinterpret_cast<float*>(shared + NSTAGE * STAGE_BYTES + NSTAGE * 24 + 16);

  const int tid = threadIdx.x;
  const int rank = blockIdx.x & 1;
  const int row_block = blockIdx.x >> 1;
  const int kb0 = rank * (K_UNITS / 2);
  const int stages = K_UNITS / 2;

  if (tid == 256) {
    for (int slot = 0; slot < NSTAGE; ++slot) {
      mbar_init(full + slot * 8, 1);
      mbar_init(decoded_ready + slot * 8, 1);
      mbar_init(empty + slot * 8, 1);
    }
    mbar_init(qbar, 1);
    mbar_expect(qbar, 64 * NACC * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (tid >= 256) {
    if (tid == 256) {
      for (int stage_index = 0; stage_index < stages; ++stage_index) {
        const int slot = stage_index % NSTAGE;
        if (stage_index >= NSTAGE)
          mbar_wait(empty + slot * 8, ((stage_index / NSTAGE) - 1) & 1);
        const int kb = kb0 + stage_index;
        const uint32_t stage = base + slot * STAGE_BYTES;
        mbar_expect(full + slot * 8, PACKED_BYTES + X_BYTES);
        bulk_g2s(stage, packed + (static_cast<size_t>(row_block) * K_UNITS + kb) * PACKED_BYTES,
                 PACKED_BYTES, full + slot * 8);
        tma_x(stage + PACKED_BYTES + DECODED_BYTES, &map_x, full + slot * 8, kb * 64, 0);
      }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  if (tid >= 128) {
    const uint32_t decoder = tid - 128;
    for (int stage_index = 0; stage_index < stages; ++stage_index) {
      const int slot = stage_index % NSTAGE;
      const int kb = kb0 + stage_index;
      const uint32_t stage_address = base + slot * STAGE_BYTES;
      uint8_t* packed_stage = shared + slot * STAGE_BYTES;
      uint16_t* decoded_stage = reinterpret_cast<uint16_t*>(packed_stage + PACKED_BYTES);
      mbar_wait(full + slot * 8, (stage_index / NSTAGE) & 1);
#pragma unroll
      for (uint32_t iteration = 0; iteration < 4; ++iteration) {
        const uint32_t group = decoder + iteration * 128u;
        const uint32_t* source = reinterpret_cast<const uint32_t*>(packed_stage + group * 12u);
        uint32_t values[4];
        c12_decode8(source[0], source[1], source[2], base4, values);
        const uint32_t row = group >> 3;
        const uint32_t chunk = (group & 7u) ^ (row & 7u);
        uint32_t* destination = reinterpret_cast<uint32_t*>(decoded_stage + row * 64u + chunk * 8u);
        destination[0] = values[0]; destination[1] = values[1];
        destination[2] = values[2]; destination[3] = values[3];
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      const size_t tile = static_cast<size_t>(row_block) * K_UNITS + kb;
      const uint32_t count = patch_counts[tile];
      if (decoder < count && decoder < 8u) {
        const uint32_t patch = patches[tile * 8 + decoder];
        const uint32_t logical = patch & 0xFFFu;
        const uint32_t physical = swizzled_index(logical);
        reinterpret_cast<uint8_t*>(decoded_stage)[physical * 2u + 1u] ^=
            static_cast<uint8_t>((patch >> 12) & 0xFFu);
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (decoder == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(decoded_ready + slot * 8);
      }
      (void)stage_address;
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t4 = lane & 3;
  float accumulator[NACC];
#pragma unroll
  for (int index = 0; index < NACC; ++index) accumulator[index] = 0.0f;

  for (int stage_index = 0; stage_index < stages; ++stage_index) {
    const int slot = stage_index % NSTAGE;
    const uint32_t stage = base + slot * STAGE_BYTES;
    mbar_wait(decoded_ready + slot * 8, (stage_index / NSTAGE) & 1);
#pragma unroll
    for (int index = 0; index < NACC; ++index) asm volatile("" : "+f"(accumulator[index]) :: "memory");
    wg_fence();
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16)
      wgmma_ss_128(accumulator,
                   desc_sw128(stage + PACKED_BYTES + k16 * 32),
                   desc_sw128(stage + PACKED_BYTES + DECODED_BYTES + k16 * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int index = 0; index < NACC; ++index) asm volatile("" : "+f"(accumulator[index]) :: "memory");
    if (stage_index > 0 && tid == 0) mbar_arrive(empty + ((stage_index - 1) % NSTAGE) * 8);
  }
  wg_wait<0>();
  if (tid == 0) mbar_arrive(empty + ((stages - 1) % NSTAGE) * 8);

  const int keep_first = 2 * rank;
  const bool keep = (warp >> 1) == rank;
  if (!keep) {
    const int local_warp = warp & 1;
    const uint32_t local = saddr(peer + (local_warp * 32 + lane) * REDUCTION_STRIDE);
    uint32_t remote, remote_barrier;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_barrier) : "r"(qbar), "r"(rank ^ 1));
#pragma unroll
    for (int index = 0; index < NACC; index += 4)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                   "[%0], {%1, %2, %3, %4}, [%5];"
                   :: "r"(remote + index * 4),
                      "r"(__float_as_uint(accumulator[index + 0])),
                      "r"(__float_as_uint(accumulator[index + 1])),
                      "r"(__float_as_uint(accumulator[index + 2])),
                      "r"(__float_as_uint(accumulator[index + 3])), "r"(remote_barrier) : "memory");
    return;
  }

  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  mbar_wait_cluster(qbar, 0);
  const float* remote_values = peer + ((warp & 1) * 32 + lane) * REDUCTION_STRIDE;
#pragma unroll
  for (int index = 0; index < NACC; index += 4) {
    const float4 values = *reinterpret_cast<const float4*>(remote_values + index);
    accumulator[index + 0] += values.x; accumulator[index + 1] += values.y;
    accumulator[index + 2] += values.z; accumulator[index + 3] += values.w;
  }

  __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(shared);
  const int local_row = (warp - keep_first) * 16;
#pragma unroll
  for (int token_group = 0; token_group < 16; ++token_group) {
    const int token = token_group * 8 + 2 * t4;
    __nv_bfloat16* first = staging + token * 32 + local_row + g;
    first[0] = __float2bfloat16_rn(accumulator[4 * token_group + 0]);
    first[8] = __float2bfloat16_rn(accumulator[4 * token_group + 2]);
    first[32] = __float2bfloat16_rn(accumulator[4 * token_group + 1]);
    first[40] = __float2bfloat16_rn(accumulator[4 * token_group + 3]);
  }
  asm volatile("bar.sync 2, 64;" ::: "memory");

  const int worker = (warp - keep_first) * 32 + lane;
  for (int chunk = worker; chunk < TM * 4; chunk += 64) {
    const int row_chunk = chunk & 3;
    const int token = chunk >> 2;
    const uint4 value = *reinterpret_cast<const uint4*>(staging + token * 32 + row_chunk * 8);
    *reinterpret_cast<uint4*>(output + static_cast<size_t>(token) * N + row_block * 64 + rank * 32 + row_chunk * 8) = value;
  }
}
'''


def main() -> None:
    print(CUDA.replace("@@WGMMA@@", wgmma_ss_128()))


if __name__ == "__main__":
    main()
