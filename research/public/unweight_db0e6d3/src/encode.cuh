#pragma once

#include "common.cuh"

__device__ __forceinline__ const bf16 *get_row_data(
    const bf16 *data,
    usize cols,
    usize tile_idx,
    usize row_in_tile,
    usize k_tiles)
{
    usize n_tile = tile_idx / k_tiles;
    usize k_tile = tile_idx % k_tiles;
    usize data_row = (n_tile << TILE_M_SHIFT) + row_in_tile;
    usize data_col = k_tile << TILE_K_SHIFT;

    return data + data_row * cols + data_col;
}

__device__ __forceinline__ void extract_exponents(
    const bf16 *row_data,
    u32 *exp_out)
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

        exp_out[i * 2] = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24);
        exp_out[i * 2 + 1] = exp4 | (exp5 << 8) | (exp6 << 16) | (exp7 << 24);
    }
}

__device__ void encode_row(
    const u32 *exponents,
    const HuffmanEncodeEntry *encode_table,
    u32 *encoded_out,
    usize tile_idx,
    usize row_in_tile,
    u32 tile_start_bit,
    const u16 *row_offsets)
{
    usize prev_idx = (tile_idx << TILE_M_SHIFT) + row_in_tile - 1;
    u32 row_start_bit = row_in_tile == 0 ? 0 : (u32)__ldg(&row_offsets[prev_idx]);
    u32 bit_offset = tile_start_bit + row_start_bit;
    u32 current_word_idx = bit_offset / BITS_PER_WORD;
    u32 current_bit_in_word = bit_offset % BITS_PER_WORD;
    u32 accumulated_word = 0;
    bool first_word = true;

    #pragma unroll
    for (usize i = 0; i < TILE_COLS_U32; i++) {
        u32 packed = exponents[i];

        #pragma unroll
        for (usize j = 0; j < U8S_PER_U32; j++) {
            u8 exp = (packed >> (j * 8)) & 0xFF;

            auto entry = encode_table[exp];
            u32 code = entry.code;
            u32 len = entry.length;

            accumulated_word |= code << current_bit_in_word;

            if (current_bit_in_word + len >= BITS_PER_WORD) {
                if (first_word) {
                    atomicOr(&encoded_out[current_word_idx], accumulated_word);
                    first_word = false;
                } else {
                    encoded_out[current_word_idx] = accumulated_word;
                }

                current_word_idx++;
                current_bit_in_word = (current_bit_in_word + len) - BITS_PER_WORD;

                u32 remaining = len - current_bit_in_word;

                accumulated_word = current_bit_in_word > 0 ? (code >> remaining) : 0;
            } else {
                current_bit_in_word += len;
            }
        }
    }

    if (current_bit_in_word > 0) {
        atomicOr(&encoded_out[current_word_idx], accumulated_word);
    }
}

__device__ __forceinline__ void copy_verbatim_row(
    const u32 *exponents,
    u8 *verbatim_out,
    usize verbatim_row_offset,
    usize verbatim_idx_in_tile)
{
    usize verbatim_dst = (verbatim_row_offset + verbatim_idx_in_tile) * UNWEIGHT_TILE_K;
    u32x4 *dst_u32x4 = reinterpret_cast<u32x4 *>(verbatim_out + verbatim_dst);

    #pragma unroll
    for (usize i = 0; i < TILE_COLS_U32 / U32S_PER_U32X4; i++) {
        dst_u32x4[i] = u32x4{
            exponents[i * U32S_PER_U32X4],
            exponents[i * U32S_PER_U32X4 + 1],
            exponents[i * U32S_PER_U32X4 + 2],
            exponents[i * U32S_PER_U32X4 + 3]};
    }
}

__device__ void process_row(
    usize global_row_idx,
    const bf16 *data,
    const u32 *tile_metadata,
    const u16 *row_offsets,
    const HuffmanEncodeEntry *encode_table,
    u32 *encoded_out,
    u8 *verbatim_out,
    usize cols,
    usize k_tiles)
{
    u32 exponents[TILE_COLS_U32];

    usize tile_idx = global_row_idx >> TILE_M_SHIFT;
    usize row_in_tile = global_row_idx & TILE_M_MASK;
    auto meta = TileMetadata::load(tile_metadata, tile_idx);
    const bf16 *row_data = get_row_data(data, cols, tile_idx, row_in_tile, k_tiles);

    extract_exponents(row_data, exponents);

    if (meta.is_verbatim(row_in_tile)) {
        usize verbatim_idx = meta.verbatim_row_index(row_in_tile);

        copy_verbatim_row(
            exponents,
            verbatim_out,
            meta.verbatim_row_offset,
            verbatim_idx);
    } else {
        encode_row(
            exponents,
            encode_table,
            encoded_out,
            tile_idx,
            row_in_tile,
            meta.tile_start_bit,
            row_offsets);
    }
}

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
unweight_encode(
    const bf16 *__restrict__ data,
    const u32 *__restrict__ tile_metadata,
    const u16 *__restrict__ row_offsets,
    const HuffmanEncodeEntry *__restrict__ encode_table_hbm,
    u32 *__restrict__ encoded_out,
    u8 *__restrict__ verbatim_out,
    u32 *__restrict__ next_row_idx,
    usize cols,
    usize k_tiles,
    usize total_rows)
{
    __shared__ HuffmanEncodeEntry encode_table[HUFFMAN_ENCODE_TABLE_SIZE];

    for (usize i = threadIdx.x; i < HUFFMAN_ENCODE_TABLE_SIZE; i += blockDim.x) {
        encode_table[i] = encode_table_hbm[i];
    }

    __syncthreads();

    while (true) {
        usize row_idx = atomicAdd(next_row_idx, 1);

        if (row_idx >= total_rows) {
            break;
        }

        process_row(
            row_idx,
            data,
            tile_metadata,
            row_offsets,
            encode_table,
            encoded_out,
            verbatim_out,
            cols,
            k_tiles);
    }
}

extern "C" void unweight_encode_launch(
    const bf16 *data,
    const u32 *tile_metadata,
    const u16 *row_offsets,
    const HuffmanEncodeEntry *encode_table,
    u32 *encoded_out,
    u8 *verbatim_out,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);

    unweight_encode<<<PERSISTENT_GRID_SIZE, MAX_THREADS_PER_BLOCK, 0, stream>>>(
        data,
        tile_metadata,
        row_offsets,
        encode_table,
        encoded_out,
        verbatim_out,
        next_row_idx,
        cols,
        td.num_tiles_k,
        td.total_rows);
}
