#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "kittens.cuh"
#include "prototype.cuh"

namespace cg = cooperative_groups;
namespace tk = kittens;
namespace lcsf = tk::prototype::lcsf;
namespace lcf = tk::prototype::lcf;

typedef nv_bfloat16 bf16;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int16_t i16;
typedef float f32;
typedef u32 usize;
using block_barrier = cuda::barrier<cuda::thread_scope_block>;

typedef uint2 u32x2;
typedef uint4 u32x4;

#ifndef UNWEIGHT_CLUSTER_SIZE
#define UNWEIGHT_CLUSTER_SIZE 8
#endif

#ifndef UNWEIGHT_TILE_M
#define UNWEIGHT_TILE_M 64
#endif

#ifndef UNWEIGHT_TILE_K
#define UNWEIGHT_TILE_K 64
#endif

#ifndef UNWEIGHT_MAX_CODE_LEN
#define UNWEIGHT_MAX_CODE_LEN 12
#endif

#ifndef TILES_PER_METADATA_BATCH
#define TILES_PER_METADATA_BATCH 64
#endif

__host__ __device__ constexpr usize ceil_div(usize a, usize b)
{
    return (a + b - 1) / b;
}

constexpr usize U8S_PER_U32 = sizeof(u32) / sizeof(u8);
constexpr usize U32S_PER_U32X4 = sizeof(u32x4) / sizeof(u32);
constexpr usize BF16S_PER_U32X4 = sizeof(u32x4) / sizeof(bf16);
constexpr u32 TILE_M_SHIFT = 6;
constexpr u32 TILE_M_MASK = UNWEIGHT_TILE_M - 1;
constexpr u32 TILE_K_SHIFT = 6;
constexpr usize TILE_COLS_U32 = UNWEIGHT_TILE_K / U8S_PER_U32;
constexpr usize TILE_SIZE_U32 = UNWEIGHT_TILE_M * TILE_COLS_U32;
constexpr usize TILE_METADATA_SIZE_U32 = 4;
constexpr usize U32X4S_PER_TILE_ROW = UNWEIGHT_TILE_K / BF16S_PER_U32X4;
constexpr usize U32X4S_PER_TILE = UNWEIGHT_TILE_M * U32X4S_PER_TILE_ROW;
constexpr usize U32X4S_PER_TILE_OFFSETS =
    UNWEIGHT_TILE_M * sizeof(u16) / sizeof(u32x4);
constexpr u32 BF16_EXP_BITS = 8;
constexpr u8 BF16_SIGN_MASK = 0x80;
constexpr u8 BF16_MANTISSA_MASK = 0x7F;
constexpr usize EXP_HIST_BINS = 256;
constexpr i16 HUFFMAN_NO_CHILD = -1;
constexpr usize HUFFMAN_ENCODE_TABLE_SIZE = EXP_HIST_BINS + 1;
constexpr u32 BITS_PER_WORD = 32;
constexpr u32 BITS_PER_WORD_SHIFT = 5;
constexpr usize ENCODED_ROW_WORDS = 256;
constexpr usize HIGH_FREQ_PALETTE_SIZE = 16;
constexpr u8 PALETTE_SENTINEL = 255;
constexpr usize PERSISTENT_GRID_SIZE = 132;
constexpr usize MAX_THREADS_PER_BLOCK = 1024;
constexpr usize WARPS_PER_BLOCK =
    MAX_THREADS_PER_BLOCK / tk::WARP_THREADS;
constexpr usize WARPS_PER_WARPGROUP = 4;
constexpr int BATCH_DYNAMIC = -1;
constexpr int BATCH_STATIC = 1;
constexpr int DEPTH_DYNAMIC = -1;
constexpr int DEPTH_STATIC = 1;
constexpr int ROWS_DYNAMIC = -1;
constexpr int ROWS_STATIC = 1;
constexpr int COLS_DYNAMIC = -1;
constexpr int COLS_STATIC = 1;
constexpr bool SWIZZLE_ON = true;
constexpr bool SWIZZLE_OFF = false;

static_assert(UNWEIGHT_TILE_M == (1 << TILE_M_SHIFT));
static_assert(UNWEIGHT_TILE_K == (1 << TILE_K_SHIFT));
static_assert(TILE_COLS_U32 % U32S_PER_U32X4 == 0);
static_assert(BITS_PER_WORD == (1 << BITS_PER_WORD_SHIFT));

struct alignas(4) HuffmanDecodeEntry {
    u8 symbol;
    u8 length;
    u8 palette_idx;
    u8 _pad;
};

__device__ __forceinline__ u32x4 enforced_load_u32x4_global(const u32x4 *ptr)
{
    u32x4 val;
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(ptr));
    return val;
}

__device__ __forceinline__ void enforced_store_u32x4_global(u32x4 *ptr, u32x4 val)
{
    asm volatile(
        "st.global.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

struct ExpPair {
    u8 lo;
    u8 hi;

    __device__ static ExpPair from_bf16_pair(u32 bf16_pair)
    {
        u32 shifted = bf16_pair >> 7;
        u16 exps = __byte_perm(shifted, 0, 0x0020);

        return { (u8)(exps & 0xFF), (u8)(exps >> 8) };
    }
};

__device__ __forceinline__ u32 reconstruct_bf16_pair(u32 sm_exp_pair)
{
    u32 signs = (sm_exp_pair & 0x00800080u) << 8;
    u32 exponents = (sm_exp_pair & 0xFF00FF00u) >> 1;
    u32 mantissas = sm_exp_pair & 0x007F007Fu;

    return signs | exponents | mantissas;
}

struct TileCoord {
    usize row;
    usize col;

    __device__ static TileCoord from_row_outer_idx(usize idx, usize num_cols) {
        return {idx / num_cols, idx % num_cols};
    }
};

struct UnweightTileDims {
    usize num_tiles_m;
    usize num_tiles_k;
    usize num_tiles;
    usize total_rows;

    static UnweightTileDims from_dims(usize rows, usize cols) {
        usize m = rows / UNWEIGHT_TILE_M;
        usize k = cols / UNWEIGHT_TILE_K;

        return { m, k, m * k, m * k * UNWEIGHT_TILE_M };
    }
};

__device__ __forceinline__ u32 warp_broadcast(u32 value) {
    return __shfl_sync(tk::MASK_ALL, value, 0);
}

__device__ __forceinline__ bool is_verbatim_row(u64 verbatim_bitmask, usize row) {
    return ((verbatim_bitmask >> row) & 1ull) != 0;
}

struct TileMetadata {
    u32 tile_start_bit;
    usize verbatim_row_offset;
    u64 verbatim_bitmask;

    __device__ __forceinline__ static TileMetadata load(const u32 *tile_metadata, usize tile_idx)
    {
        u32x4 meta = reinterpret_cast<const u32x4 *>(tile_metadata)[tile_idx];

        return {
            meta.x,
            meta.y,
            (static_cast<u64>(meta.w) << 32) | meta.z
        };
    }

    __device__ __forceinline__ void store(u32 *tile_metadata, usize tile_idx) const
    {
        u32x4 meta = {
            tile_start_bit,
            verbatim_row_offset,
            static_cast<u32>(verbatim_bitmask),
            static_cast<u32>(verbatim_bitmask >> 32)
        };

        reinterpret_cast<u32x4 *>(tile_metadata)[tile_idx] = meta;
    }

    __device__ __forceinline__ bool is_verbatim(usize row) const {
        return is_verbatim_row(verbatim_bitmask, row);
    }

    __device__ __forceinline__ usize verbatim_row_count() const {
        return __popcll(verbatim_bitmask);
    }

    __device__ __forceinline__ usize verbatim_row_index(usize row) const {
        u64 prior_mask = (row == 0) ? 0ull : ((1ull << row) - 1ull);

        return __popcll(verbatim_bitmask & prior_mask);
    }
};

template<typename LcsfTemplate>
void lcsf_launch(
    typename LcsfTemplate::layout::globals globals,
    usize grid_x,
    cudaStream_t stream,
    usize cluster_size = 1)
{
    constexpr int BLOCK_SIZE = tk::prototype::detail::NUM_THREADS_v<LcsfTemplate>;
    constexpr unsigned long SMEM_SIZE = tk::MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(
        lcsf::kernel<LcsfTemplate>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_SIZE);

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, 1, 1);
    config.blockDim = dim3(BLOCK_SIZE, 1, 1);
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    config.attrs = nullptr;
    config.numAttrs = 0;

    cudaLaunchAttribute attrs[1];

    if (cluster_size > 1) {
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_size;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;

        config.attrs = attrs;
        config.numAttrs = 1;
    }

    cudaLaunchKernelEx(&config, lcsf::kernel<LcsfTemplate>, globals);
}

template<typename LcfTemplate>
void lcf_launch(typename LcfTemplate::layout::globals &globals, usize grid_x, cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = tk::prototype::detail::NUM_THREADS_v<LcfTemplate>;
    constexpr unsigned long SMEM_SIZE = tk::MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(
        lcf::kernel<LcfTemplate>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_SIZE);

    lcf::kernel<LcfTemplate><<<grid_x, BLOCK_SIZE, SMEM_SIZE, stream>>>(globals);
}
