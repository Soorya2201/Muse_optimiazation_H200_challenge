#pragma once

#include "common.cuh"
#include "huffman_bit_reader.cuh"

__device__ __forceinline__ u32 pack_8_palette_indices(
    u32 idx0, u32 idx1, u32 idx2, u32 idx3,
    u32 idx4, u32 idx5, u32 idx6, u32 idx7)
{
    u32 b01 = idx0 + (idx1 << 4);
    u32 b23 = idx2 + (idx3 << 4);
    u32 b45 = idx4 + (idx5 << 4);
    u32 b67 = idx6 + (idx7 << 4);

    u32 lo = b01 + (b23 << 8);
    u32 hi = b45 + (b67 << 8);

    return lo + (hi << 16);
}

__device__ __forceinline__ u32x2 pack_16_palette_indices(
    u32 idx0, u32 idx1, u32 idx2, u32 idx3,
    u32 idx4, u32 idx5, u32 idx6, u32 idx7,
    u32 idx8, u32 idx9, u32 idx10, u32 idx11,
    u32 idx12, u32 idx13, u32 idx14, u32 idx15)
{
    return u32x2{
        pack_8_palette_indices(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7),
        pack_8_palette_indices(idx8, idx9, idx10, idx11, idx12, idx13, idx14, idx15)};
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ u32x2 transcode_palette16(HuffmanBitReader<DECODE_TABLE_SIZE> &reader)
{
    u8 idx0  = reader.decode().palette_idx;
    u8 idx1  = reader.decode().palette_idx;
    u8 idx2  = reader.decode().palette_idx;
    u8 idx3  = reader.decode().palette_idx;
    u8 idx4  = reader.decode().palette_idx;
    u8 idx5  = reader.decode().palette_idx;
    u8 idx6  = reader.decode().palette_idx;
    u8 idx7  = reader.decode().palette_idx;
    u8 idx8  = reader.decode().palette_idx;
    u8 idx9  = reader.decode().palette_idx;
    u8 idx10 = reader.decode().palette_idx;
    u8 idx11 = reader.decode().palette_idx;
    u8 idx12 = reader.decode().palette_idx;
    u8 idx13 = reader.decode().palette_idx;
    u8 idx14 = reader.decode().palette_idx;
    u8 idx15 = reader.decode().palette_idx;

    return pack_16_palette_indices(
        idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7,
        idx8, idx9, idx10, idx11, idx12, idx13, idx14, idx15);
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ void transcode_row_to_palette(
    usize global_row_idx,
    const u32 *tile_metadata,
    const u32 *encoded,
    const u16 *row_offsets,
    const HuffmanDecodeEntry *decode_table,
    u8 *palette_idx_out,
    usize n_tiles,
    usize k_tiles,
    usize decoded_cols)
{
    usize tile_idx = global_row_idx >> TILE_M_SHIFT;
    usize row_in_tile = global_row_idx & TILE_M_MASK;
    auto meta = TileMetadata::load(tile_metadata, tile_idx);

    if (meta.is_verbatim(row_in_tile)) {
        return;
    }

    usize n_tile = tile_idx / k_tiles;
    usize k_tile = tile_idx % k_tiles;
    usize decoded_row_base = n_tile << TILE_M_SHIFT;
    usize decoded_col_base = k_tile << TILE_K_SHIFT;
    usize linear_offset = (decoded_row_base + row_in_tile) * decoded_cols + decoded_col_base;
    u8 *out_base = palette_idx_out + (linear_offset >> 1);
    usize prev_idx = (tile_idx << TILE_M_SHIFT) + row_in_tile - 1;
    u32 row_start_bit = row_in_tile == 0 ? 0 : (u32)__ldg(&row_offsets[prev_idx]);
    u32 bit_offset = meta.tile_start_bit + row_start_bit;

    HuffmanBitReader<DECODE_TABLE_SIZE> reader;
    reader.init(encoded, decode_table, bit_offset);

    u32x4 *out = reinterpret_cast<u32x4 *>(out_base);

    #pragma unroll
    for (usize i = 0; i < 2; i++) {
        u32x2 lo = transcode_palette16(reader);
        u32x2 hi = transcode_palette16(reader);
        enforced_store_u32x4_global(&out[i], u32x4{lo.x, lo.y, hi.x, hi.y});
    }
}

template<u32 DECODE_TABLE_SIZE>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
unweight_transcode_to_palette(
    u8 *__restrict__ palette_idx_out,
    const u16 *__restrict__ row_offsets,
    const u32 *__restrict__ tile_metadata,
    const u32 *__restrict__ encoded,
    const HuffmanDecodeEntry *__restrict__ decode_table_hbm,
    u32 *__restrict__ next_row_idx,
    usize n_tiles,
    usize k_tiles,
    usize total_rows,
    usize decoded_cols)
{
    __shared__ HuffmanDecodeEntry decode_table[DECODE_TABLE_SIZE];

    for (usize i = threadIdx.x; i < DECODE_TABLE_SIZE; i += blockDim.x) {
        decode_table[i] = decode_table_hbm[i];
    }

    __syncthreads();

    while (true) {
        usize row_idx = atomicAdd(next_row_idx, 1);

        if (row_idx >= total_rows) {
            break;
        }

        transcode_row_to_palette<DECODE_TABLE_SIZE>(
            row_idx,
            tile_metadata,
            encoded,
            row_offsets,
            decode_table,
            palette_idx_out,
            n_tiles,
            k_tiles,
            decoded_cols);
    }
}

template<u32 DECODE_TABLE_SIZE>
static int launch_transcode_variant(
    u8 *palette_idx_out,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u32 *encoded,
    const HuffmanDecodeEntry *decode_table,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);

    unweight_transcode_to_palette<DECODE_TABLE_SIZE><<<grid_x, MAX_THREADS_PER_BLOCK, 0, stream>>>(
        palette_idx_out,
        row_offsets,
        tile_metadata,
        encoded,
        decode_table,
        next_row_idx,
        td.num_tiles_m,
        td.num_tiles_k,
        td.total_rows,
        cols);

    return (int)cudaGetLastError();
}

static int dispatch_transcode(
    u8 *palette_idx_out,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u32 *encoded,
    const HuffmanDecodeEntry *decode_table,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    usize decode_table_size,
    cudaStream_t stream)
{
    #define DISPATCH_CASE(SIZE) \
        case SIZE: return launch_transcode_variant<SIZE>( \
            palette_idx_out, row_offsets, \
            tile_metadata, encoded, decode_table, \
            next_row_idx, rows, cols, grid_x, stream)

    switch (decode_table_size) {
        DISPATCH_CASE(8);
        DISPATCH_CASE(16);
        DISPATCH_CASE(32);
        DISPATCH_CASE(64);
        DISPATCH_CASE(128);
        DISPATCH_CASE(256);
        DISPATCH_CASE(512);
        DISPATCH_CASE(1024);
        DISPATCH_CASE(2048);
        DISPATCH_CASE(4096);
        DISPATCH_CASE(8192);
        DISPATCH_CASE(16384);
        DISPATCH_CASE(32768);
        default:
            return (int)cudaErrorInvalidValue;
    }
    #undef DISPATCH_CASE
}

extern "C" int unweight_transcode_to_palette_launch(
    u8 *palette_idx_out,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u32 *encoded,
    const HuffmanDecodeEntry *decode_table,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    usize decode_table_size,
    cudaStream_t stream)
{
    return dispatch_transcode(
        palette_idx_out,
        row_offsets,
        tile_metadata,
        encoded,
        decode_table,
        next_row_idx,
        rows,
        cols,
        grid_x,
        decode_table_size,
        stream);
}
