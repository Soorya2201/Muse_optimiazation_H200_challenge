#pragma once

#include "common.cuh"
#include "transcode_to_palette.cuh"

__device__ void encode_row_to_palette(
    const bf16 *row_data,
    const u8 *symbol_to_palette_idx,
    u8 *palette_idx_out,
    usize cols)
{
    const u32x4 *row_u32x4 = reinterpret_cast<const u32x4 *>(row_data);

    #pragma unroll
    for (usize i = 0; i < U32X4S_PER_TILE_ROW; i++) {
        u32x4 v = __ldg(&row_u32x4[i]);

        u32 exp0 = (v.x >> 7) & 0xFF;
        u32 exp1 = (v.x >> 23) & 0xFF;
        u32 exp2 = (v.y >> 7) & 0xFF;
        u32 exp3 = (v.y >> 23) & 0xFF;
        u32 exp4 = (v.z >> 7) & 0xFF;
        u32 exp5 = (v.z >> 23) & 0xFF;
        u32 exp6 = (v.w >> 7) & 0xFF;
        u32 exp7 = (v.w >> 23) & 0xFF;

        u32 idx0 = symbol_to_palette_idx[exp0];
        u32 idx1 = symbol_to_palette_idx[exp1];
        u32 idx2 = symbol_to_palette_idx[exp2];
        u32 idx3 = symbol_to_palette_idx[exp3];
        u32 idx4 = symbol_to_palette_idx[exp4];
        u32 idx5 = symbol_to_palette_idx[exp5];
        u32 idx6 = symbol_to_palette_idx[exp6];
        u32 idx7 = symbol_to_palette_idx[exp7];

        u32 packed = pack_8_palette_indices(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7);

        reinterpret_cast<u32 *>(palette_idx_out)[i] = packed;
    }
}

template<u32 PERSISTENT_CTAS>
__global__ void encode_to_palette_kernel(
    const bf16 *__restrict__ data,
    const u8 *__restrict__ symbol_to_palette_idx,
    const u32 *__restrict__ tile_metadata,
    u8 *__restrict__ palette_idx_out,
    u8 *__restrict__ verbatim_exponents,
    u32 *__restrict__ next_row_idx,
    usize cols,
    usize num_tiles_m,
    usize num_tiles_k)
{
    usize total_rows = num_tiles_m * num_tiles_k * UNWEIGHT_TILE_M;
    usize k_tiles = num_tiles_k;

    while (true) {
        usize row_idx = atomicAdd(next_row_idx, 1);
        if (row_idx >= total_rows) {
            break;
        }

        usize tile_idx = row_idx >> TILE_M_SHIFT;
        usize row_in_tile = row_idx & TILE_M_MASK;

        auto meta = TileMetadata::load(tile_metadata, tile_idx);

        usize n_tile = tile_idx / k_tiles;
        usize k_tile = tile_idx % k_tiles;
        usize data_row = (n_tile << TILE_M_SHIFT) + row_in_tile;
        usize data_col = k_tile << TILE_K_SHIFT;
        const bf16 *row_data = data + data_row * cols + data_col;
        usize linear_offset = data_row * cols + data_col;
        u8 *out_base = palette_idx_out + (linear_offset >> 1);

        if (meta.is_verbatim(row_in_tile)) {
            usize verbatim_idx = meta.verbatim_row_offset + meta.verbatim_row_index(row_in_tile);
            u8 *verbatim_out = verbatim_exponents + verbatim_idx * UNWEIGHT_TILE_K;
            const u32x4 *row_u32x4 = reinterpret_cast<const u32x4 *>(row_data);

            #pragma unroll
            for (usize i = 0; i < U32X4S_PER_TILE_ROW; i++) {
                u32x4 v = __ldg(&row_u32x4[i]);

                u8 exp0 = (v.x >> 7) & 0xFF;
                u8 exp1 = (v.x >> 23) & 0xFF;
                u8 exp2 = (v.y >> 7) & 0xFF;
                u8 exp3 = (v.y >> 23) & 0xFF;
                u8 exp4 = (v.z >> 7) & 0xFF;
                u8 exp5 = (v.z >> 23) & 0xFF;
                u8 exp6 = (v.w >> 7) & 0xFF;
                u8 exp7 = (v.w >> 23) & 0xFF;

                verbatim_out[i * BF16S_PER_U32X4 + 0] = exp0;
                verbatim_out[i * BF16S_PER_U32X4 + 1] = exp1;
                verbatim_out[i * BF16S_PER_U32X4 + 2] = exp2;
                verbatim_out[i * BF16S_PER_U32X4 + 3] = exp3;
                verbatim_out[i * BF16S_PER_U32X4 + 4] = exp4;
                verbatim_out[i * BF16S_PER_U32X4 + 5] = exp5;
                verbatim_out[i * BF16S_PER_U32X4 + 6] = exp6;
                verbatim_out[i * BF16S_PER_U32X4 + 7] = exp7;
            }

            #pragma unroll
            for (usize i = 0; i < U32X4S_PER_TILE_ROW; i++) {
                reinterpret_cast<u32 *>(out_base)[i] = 0xFFFFFFFF;
            }
        } else {
            encode_row_to_palette(row_data, symbol_to_palette_idx, out_base, cols);
        }
    }
}

extern "C" int unweight_encode_to_palette_launch(
    const void *data,
    const u8 *symbol_to_palette_idx,
    const u32 *tile_metadata,
    u8 *palette_idx_out,
    u8 *verbatim_exponents,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);

    encode_to_palette_kernel<PERSISTENT_GRID_SIZE><<<grid_x, MAX_THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const bf16 *>(data),
        symbol_to_palette_idx,
        tile_metadata,
        palette_idx_out,
        verbatim_exponents,
        next_row_idx,
        cols,
        td.num_tiles_m,
        td.num_tiles_k);

    return cudaGetLastError();
}
