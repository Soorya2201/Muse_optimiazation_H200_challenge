#pragma once

namespace Layouts {
    using data_tile = tk::st<
        bf16,
        UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K,
        SWIZZLE_OFF>;

    using sm_tile = tk::st<
        f32,
        UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K / U8S_PER_U32,
        SWIZZLE_OFF>;

    using exp_tile = tk::st<
        f32,
        UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K / U8S_PER_U32,
        SWIZZLE_OFF>;

    using palette_idx_tile = tk::st<
        bf16,
        UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K / U8S_PER_U32,
        SWIZZLE_OFF>;

    using verbatim_exp_tile = tk::st<
        f32,
        1,
        UNWEIGHT_TILE_K / U8S_PER_U32,
        SWIZZLE_OFF>;

    using row_end_offsets_tile = tk::st<
        bf16,
        TILES_PER_METADATA_BATCH,
        UNWEIGHT_TILE_M,
        SWIZZLE_OFF>;

    using tile_offsets_tile = tk::st<
        f32,
        1,
        TILES_PER_METADATA_BATCH,
        SWIZZLE_OFF>;

    using metadata_tile = tk::st<
        f32,
        1,
        TILES_PER_METADATA_BATCH * TILE_METADATA_SIZE_U32,
        SWIZZLE_OFF>;

    using encoded_row_tile = tk::st<
        f32,
        1,
        ENCODED_ROW_WORDS,
        SWIZZLE_OFF>;

    using mat_tile = tk::st_bf<
        UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K>;

    template<int TILES_N>
    using mat_b_tile = tk::st_bf<
        TILES_N * UNWEIGHT_TILE_M,
        UNWEIGHT_TILE_K>;

    using data_gl = tk::gl<
        bf16,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        data_tile>;

    using sm_gl = tk::gl<
        f32,
        BATCH_DYNAMIC,
        DEPTH_DYNAMIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        sm_tile>;

    using exp_gl = tk::gl<
        f32,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        exp_tile>;

    using palette_idx_gl = tk::gl<
        bf16,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        palette_idx_tile>;

    using verbatim_exp_gl = tk::gl<
        f32,
        BATCH_DYNAMIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        verbatim_exp_tile>;

    using dummy_gl = tk::gl<
        bf16,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_STATIC,
        COLS_STATIC>;

    using row_end_offsets_gl = tk::gl<
        bf16,
        BATCH_DYNAMIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        row_end_offsets_tile>;

    using tile_offsets_gl = tk::gl<
        f32,
        BATCH_DYNAMIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        tile_offsets_tile>;

    using metadata_gl = tk::gl<
        f32,
        BATCH_DYNAMIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        metadata_tile>;

    using encoded_gl = tk::gl<
        f32,
        BATCH_DYNAMIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        encoded_row_tile>;

    using mat_gl = tk::gl<
        bf16,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        mat_tile>;

    template<int TILES_N>
    using mat_b_gl = tk::gl<
        bf16,
        BATCH_STATIC,
        DEPTH_STATIC,
        ROWS_DYNAMIC,
        COLS_DYNAMIC,
        mat_b_tile<TILES_N>>;

    template<int TILES_N>
    inline mat_b_gl<TILES_N> make_mat_b_gl(bf16 *data, usize rows, usize cols) {
        return mat_b_gl<TILES_N>(data, nullptr, nullptr, rows, cols);
    }

    inline data_gl make_data_gl(bf16 *data, usize rows, usize cols) {
        return data_gl(data, nullptr, nullptr, rows, cols);
    }

    inline sm_gl make_sm_gl(u32 *sm, usize num_tiles_m, usize num_tiles_k) {
        return sm_gl(
            reinterpret_cast<f32 *>(sm),
            num_tiles_m,
            num_tiles_k,
            UNWEIGHT_TILE_M,
            UNWEIGHT_TILE_K / U8S_PER_U32);
    }

    inline exp_gl make_exp_gl(u8 *exp, usize num_tiles_n, usize num_tiles_k) {
        return exp_gl(
            reinterpret_cast<f32 *>(exp),
            nullptr,
            nullptr,
            num_tiles_n * UNWEIGHT_TILE_M,
            num_tiles_k * (UNWEIGHT_TILE_K / U8S_PER_U32));
    }

    inline palette_idx_gl make_palette_idx_gl(
        u8 *palette_idx,
        usize num_tiles_n,
        usize num_tiles_k)
    {
        return palette_idx_gl(
            reinterpret_cast<bf16 *>(palette_idx),
            nullptr,
            nullptr,
            num_tiles_n * UNWEIGHT_TILE_M,
            num_tiles_k * (UNWEIGHT_TILE_K / U8S_PER_U32));
    }

    inline verbatim_exp_gl make_verbatim_exp_gl(u8 *verbatim_exp, usize total_rows) {
        return verbatim_exp_gl(
            reinterpret_cast<f32 *>(verbatim_exp),
            total_rows,
            nullptr,
            1,
            UNWEIGHT_TILE_K / U8S_PER_U32);
    }

    inline dummy_gl make_dummy_gl() {
        return dummy_gl(nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    inline row_end_offsets_gl make_row_end_offsets_gl(u16 *offsets, usize num_metadata_batches) {
        return row_end_offsets_gl(
            reinterpret_cast<bf16 *>(offsets),
            num_metadata_batches,
            nullptr,
            TILES_PER_METADATA_BATCH,
            UNWEIGHT_TILE_M);
    }

    inline tile_offsets_gl make_tile_offsets_gl(u32 *offsets, usize num_metadata_batches) {
        return tile_offsets_gl(
            reinterpret_cast<f32 *>(offsets),
            num_metadata_batches,
            nullptr,
            1,
            TILES_PER_METADATA_BATCH);
    }

    inline metadata_gl make_metadata_gl(u32 *metadata, usize num_metadata_batches) {
        return metadata_gl(
            reinterpret_cast<f32 *>(metadata),
            num_metadata_batches,
            nullptr,
            1,
            TILES_PER_METADATA_BATCH * TILE_METADATA_SIZE_U32);
    }

    inline encoded_gl make_encoded_gl(u32 *encoded, usize total_encoded_words) {
        return encoded_gl(
            reinterpret_cast<f32 *>(encoded),
            ceil_div(total_encoded_words, ENCODED_ROW_WORDS),
            nullptr,
            1,
            ENCODED_ROW_WORDS);
    }

    inline mat_gl make_mat_gl(bf16 *data, usize rows, usize cols) {
        return mat_gl(data, nullptr, nullptr, rows, cols);
    }
}
