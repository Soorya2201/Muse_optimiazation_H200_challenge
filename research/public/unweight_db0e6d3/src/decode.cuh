#pragma once

#include "common.cuh"
#include "huffman_bit_reader.cuh"

struct RowContext {
    usize tile_idx;
    usize row_in_tile;
    usize output_row;
    usize output_col;

    __device__ __forceinline__ static RowContext from_global_row(
        usize global_row_idx,
        usize k_tiles)
    {
        usize tile_idx = global_row_idx >> TILE_M_SHIFT;
        usize row_in_tile = global_row_idx & TILE_M_MASK;
        usize n_tile = tile_idx / k_tiles;
        usize k_tile = tile_idx % k_tiles;

        return {
            tile_idx,
            row_in_tile,
            (n_tile << TILE_M_SHIFT) + row_in_tile,
            k_tile << TILE_K_SHIFT
        };
    }

    __device__ __forceinline__ usize sm_row_offset() const
    {
        return tile_idx * TILE_SIZE_U32 + row_in_tile * TILE_COLS_U32;
    }
};

__device__ __forceinline__ static u32 pack_4_exp(u8 e0, u8 e1, u8 e2, u8 e3)
{
    return __byte_perm(__byte_perm(e0, e1, 0x0040), __byte_perm(e2, e3, 0x0040), 0x5410);
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ static u32x4
huffman_decode_exp16(HuffmanBitReader<DECODE_TABLE_SIZE> &reader)
{
    u8 e0  = reader.decode().symbol;
    u8 e1  = reader.decode().symbol;
    u8 e2  = reader.decode().symbol;
    u8 e3  = reader.decode().symbol;
    u8 e4  = reader.decode().symbol;
    u8 e5  = reader.decode().symbol;
    u8 e6  = reader.decode().symbol;
    u8 e7  = reader.decode().symbol;
    u8 e8  = reader.decode().symbol;
    u8 e9  = reader.decode().symbol;
    u8 e10 = reader.decode().symbol;
    u8 e11 = reader.decode().symbol;
    u8 e12 = reader.decode().symbol;
    u8 e13 = reader.decode().symbol;
    u8 e14 = reader.decode().symbol;
    u8 e15 = reader.decode().symbol;

    return u32x4{
        pack_4_exp(e0, e1, e2, e3),
        pack_4_exp(e4, e5, e6, e7),
        pack_4_exp(e8, e9, e10, e11),
        pack_4_exp(e12, e13, e14, e15)};
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ static void decode_and_reconstruct_bf16x16(
    HuffmanBitReader<DECODE_TABLE_SIZE> &reader,
    const u32x4 *sm_ptr,
    u32x4 *out)
{
    u32x4 exp = huffman_decode_exp16(reader);
    u32x4 sm = *sm_ptr;

    u32 pair_01 = __byte_perm(sm.x, exp.x, 0x5140);
    u32 pair_23 = __byte_perm(sm.x, exp.x, 0x7362);
    u32 pair_45 = __byte_perm(sm.y, exp.y, 0x5140);
    u32 pair_67 = __byte_perm(sm.y, exp.y, 0x7362);
    u32 pair_89 = __byte_perm(sm.z, exp.z, 0x5140);
    u32 pair_ab = __byte_perm(sm.z, exp.z, 0x7362);
    u32 pair_cd = __byte_perm(sm.w, exp.w, 0x5140);
    u32 pair_ef = __byte_perm(sm.w, exp.w, 0x7362);

    out[0] = u32x4{
        reconstruct_bf16_pair(pair_01),
        reconstruct_bf16_pair(pair_23),
        reconstruct_bf16_pair(pair_45),
        reconstruct_bf16_pair(pair_67)};
    out[1] = u32x4{
        reconstruct_bf16_pair(pair_89),
        reconstruct_bf16_pair(pair_ab),
        reconstruct_bf16_pair(pair_cd),
        reconstruct_bf16_pair(pair_ef)};
}

__device__ __forceinline__ static const u8 *get_verbatim_row_ptr(
    const u8 *verbatim_exponents,
    const TileMetadata &meta,
    usize row_in_tile)
{
    usize verbatim_idx = meta.verbatim_row_index(row_in_tile);

    return verbatim_exponents + (meta.verbatim_row_offset + verbatim_idx) * UNWEIGHT_TILE_K;
}

__device__ __forceinline__ static void copy_verbatim_exp_row(
    const u8 *verbatim_row,
    u8 *output_row)
{
    const u32x4 *src = reinterpret_cast<const u32x4 *>(verbatim_row);
    u32x4 *dst = reinterpret_cast<u32x4 *>(output_row);

    #pragma unroll 2
    for (usize i = 0; i < TILE_COLS_U32 / U32S_PER_U32X4; i++) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ static void reconstruct_verbatim_row(
    const u8 *verbatim_row,
    const u32 *sm_packed,
    usize sm_offset,
    bf16 *output_row)
{
    const u32x4 *exp_chunks = reinterpret_cast<const u32x4 *>(verbatim_row);
    const u32x4 *sm_chunks = reinterpret_cast<const u32x4 *>(sm_packed + sm_offset);
    u32x4 *out = reinterpret_cast<u32x4 *>(output_row);

    for (usize i = 0; i < TILE_COLS_U32 / U32S_PER_U32X4; i++) {
        u32x4 exp = exp_chunks[i];
        u32x4 sm = sm_chunks[i];
        u32 pair_01 = __byte_perm(sm.x, exp.x, 0x5140);
        u32 pair_23 = __byte_perm(sm.x, exp.x, 0x7362);
        u32 pair_45 = __byte_perm(sm.y, exp.y, 0x5140);
        u32 pair_67 = __byte_perm(sm.y, exp.y, 0x7362);
        u32 pair_89 = __byte_perm(sm.z, exp.z, 0x5140);
        u32 pair_ab = __byte_perm(sm.z, exp.z, 0x7362);
        u32 pair_cd = __byte_perm(sm.w, exp.w, 0x5140);
        u32 pair_ef = __byte_perm(sm.w, exp.w, 0x7362);

        out[i * 2 + 0] = u32x4{
            reconstruct_bf16_pair(pair_01),
            reconstruct_bf16_pair(pair_23),
            reconstruct_bf16_pair(pair_45),
            reconstruct_bf16_pair(pair_67)};
        out[i * 2 + 1] = u32x4{
            reconstruct_bf16_pair(pair_89),
            reconstruct_bf16_pair(pair_ab),
            reconstruct_bf16_pair(pair_cd),
            reconstruct_bf16_pair(pair_ef)};
    }
}

__device__ __forceinline__ static u32 get_row_bit_offset(
    const u16 *row_offsets,
    const TileMetadata &meta,
    usize tile_idx,
    usize row_in_tile)
{
    usize prev_idx = (tile_idx << TILE_M_SHIFT) + row_in_tile - 1;
    u32 row_start = row_in_tile == 0 ? 0 : (u32)__ldg(&row_offsets[prev_idx]);

    return meta.tile_start_bit + row_start;
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ static void decode_exp_row(
    HuffmanBitReader<DECODE_TABLE_SIZE> &reader,
    u8 *output_row)
{
    u32x4 *out = reinterpret_cast<u32x4 *>(output_row);

    #pragma unroll
    for (usize i = 0; i < TILE_COLS_U32 / U32S_PER_U32X4 / 2; i++) {
        u32x4 v0 = huffman_decode_exp16(reader);
        u32x4 v1 = huffman_decode_exp16(reader);

        enforced_store_u32x4_global(&out[i * 2 + 0], v0);
        enforced_store_u32x4_global(&out[i * 2 + 1], v1);
    }
}

template<u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ static void decode_and_reconstruct_row(
    HuffmanBitReader<DECODE_TABLE_SIZE> &reader,
    const u32 *__restrict__ sm_packed,
    usize sm_offset,
    bf16 *__restrict__ output_row)
{
    const u32x4 *__restrict__ sm_row_u32x4 = reinterpret_cast<const u32x4 *>(sm_packed + sm_offset);
    u32x4 *__restrict__ out = reinterpret_cast<u32x4 *>(output_row);

    #pragma unroll 1 
    for (usize i = 0; i < TILE_COLS_U32 / U32S_PER_U32X4; i++) {
        u32x4 sm = __ldg(&sm_row_u32x4[i]);
        u32x4 exp = huffman_decode_exp16(reader);

        u32 pair_01 = __byte_perm(sm.x, exp.x, 0x5140);
        u32 pair_23 = __byte_perm(sm.x, exp.x, 0x7362);
        u32 pair_45 = __byte_perm(sm.y, exp.y, 0x5140);
        u32 pair_67 = __byte_perm(sm.y, exp.y, 0x7362);
        u32 pair_89 = __byte_perm(sm.z, exp.z, 0x5140);
        u32 pair_ab = __byte_perm(sm.z, exp.z, 0x7362);
        u32 pair_cd = __byte_perm(sm.w, exp.w, 0x5140);
        u32 pair_ef = __byte_perm(sm.w, exp.w, 0x7362);

        out[i * 2 + 0] = u32x4{
            reconstruct_bf16_pair(pair_01),
            reconstruct_bf16_pair(pair_23),
            reconstruct_bf16_pair(pair_45),
            reconstruct_bf16_pair(pair_67)};
        out[i * 2 + 1] = u32x4{
            reconstruct_bf16_pair(pair_89),
            reconstruct_bf16_pair(pair_ab),
            reconstruct_bf16_pair(pair_cd),
            reconstruct_bf16_pair(pair_ef)};
    }
}

template<bool FULL_DECODE, u32 DECODE_TABLE_SIZE>
__device__ __forceinline__ void decode_row(
    usize global_row_idx,
    const u32 *sm_packed,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u32 *encoded,
    const u16 *row_offsets,
    const HuffmanDecodeEntry *decode_table,
    void *output,
    usize k_tiles,
    usize decoded_cols)
{
    auto ctx = RowContext::from_global_row(global_row_idx, k_tiles);
    auto meta = TileMetadata::load(tile_metadata, ctx.tile_idx);

    if (meta.is_verbatim(ctx.row_in_tile)) {
        const u8 *verbatim_row = get_verbatim_row_ptr(verbatim_exponents, meta, ctx.row_in_tile);

        if constexpr (FULL_DECODE) {
            bf16 *out_row = reinterpret_cast<bf16 *>(output)
                + ctx.output_row * decoded_cols + ctx.output_col;

            reconstruct_verbatim_row(verbatim_row, sm_packed, ctx.sm_row_offset(), out_row);
        } else {
            u8 *out_row = reinterpret_cast<u8 *>(output)
                + ctx.output_row * decoded_cols + ctx.output_col;

            copy_verbatim_exp_row(verbatim_row, out_row);
        }
        return;
    }

    u32 bit_offset = get_row_bit_offset(row_offsets, meta, ctx.tile_idx, ctx.row_in_tile);

    HuffmanBitReader<DECODE_TABLE_SIZE> reader;
    reader.init(encoded, decode_table, bit_offset);

    if constexpr (FULL_DECODE) {
        bf16 *out_row = reinterpret_cast<bf16 *>(output)
            + ctx.output_row * decoded_cols + ctx.output_col;

        decode_and_reconstruct_row(reader, sm_packed, ctx.sm_row_offset(), out_row);
    } else {
        u8 *out_row = reinterpret_cast<u8 *>(output)
            + ctx.output_row * decoded_cols + ctx.output_col;

        decode_exp_row(reader, out_row);
    }
}

template<bool FULL_DECODE, u32 DECODE_TABLE_SIZE>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
unweight_decode(
    const u32 *__restrict__ sm_packed,
    void *__restrict__ output,
    const u16 *__restrict__ row_offsets,
    const u32 *__restrict__ tile_metadata,
    const u8 *__restrict__ verbatim_exponents,
    const u32 *__restrict__ encoded,
    const HuffmanDecodeEntry *__restrict__ decode_table_hbm,
    u32 *__restrict__ next_row_idx,
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

        decode_row<FULL_DECODE, DECODE_TABLE_SIZE>(
            row_idx,
            sm_packed,
            tile_metadata,
            verbatim_exponents,
            encoded,
            row_offsets,
            decode_table,
            output,
            k_tiles,
            decoded_cols);
    }
}

template<bool FULL_DECODE, u32 DECODE_TABLE_SIZE>
static int launch_decode_tiles_variant(
    const u32 *sm_packed,
    void *output,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u32 *encoded,
    const HuffmanDecodeEntry *decode_table,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);

    unweight_decode<FULL_DECODE, DECODE_TABLE_SIZE><<<grid_x, MAX_THREADS_PER_BLOCK, 0, stream>>>(
        sm_packed,
        output,
        row_offsets,
        tile_metadata,
        verbatim_exponents,
        encoded,
        decode_table,
        next_row_idx,
        td.num_tiles_k,
        td.total_rows,
        cols);

    return (int)cudaGetLastError();
}

template<bool FULL_DECODE>
static int dispatch_decode_tiles(
    const u32 *sm_packed,
    void *output,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
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
        case SIZE: return launch_decode_tiles_variant< \
            FULL_DECODE, SIZE>( \
            sm_packed, output, row_offsets, \
            tile_metadata, verbatim_exponents, encoded, \
            decode_table, next_row_idx, \
            rows, cols, grid_x, stream)

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

__global__ void decoder_reset_row_index_counter(u32 *counter)
{
    *counter = 0;
}

extern "C" void unweight_decoder_reset_row_index_counter_launch(
    u32 *counter,
    cudaStream_t stream)
{
    decoder_reset_row_index_counter<<<1, 1, 0, stream>>>(counter);
}

extern "C" int unweight_decode_launch(
    const u32 *sm_packed,
    void *output,
    const u16 *row_offsets,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u32 *encoded,
    const HuffmanDecodeEntry *decode_table,
    u32 *next_row_idx,
    usize rows,
    usize cols,
    usize grid_x,
    u32 full_decode,
    usize decode_table_size,
    cudaStream_t stream)
{
    if (full_decode != 0) {
        return dispatch_decode_tiles<true>(
            sm_packed,
            output,
            row_offsets,
            tile_metadata,
            verbatim_exponents,
            encoded,
            decode_table,
            next_row_idx,
            rows,
            cols,
            grid_x,
            decode_table_size,
            stream);
    } else {
        return dispatch_decode_tiles<false>(
            sm_packed,
            output,
            row_offsets,
            tile_metadata,
            verbatim_exponents,
            encoded,
            decode_table,
            next_row_idx,
            rows,
            cols,
            grid_x,
            decode_table_size,
            stream);
    }
}