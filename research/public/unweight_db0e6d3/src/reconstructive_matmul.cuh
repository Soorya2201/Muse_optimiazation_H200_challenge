#pragma once

#include <type_traits>

#include "common.cuh"
#include "layouts.cuh"

constexpr usize TILES_PER_CTA_M = 2;

template<bool USE_PALETTE, int TILES_PER_CTA_N,
    int INPUT_PIPE_STAGE_COUNT, int MAX_VERBATIM_ROWS_PARAM>
struct ReconstructiveMatmulLayout {
    static constexpr int MAX_VERBATIM_ROWS = MAX_VERBATIM_ROWS_PARAM;
    static constexpr bool HAS_VERBATIM = USE_PALETTE && (MAX_VERBATIM_ROWS > 0);

    using mat_tile = Layouts::mat_tile;
    using sm_tile = Layouts::sm_tile;
    using mat_b_tile = Layouts::mat_b_tile<TILES_PER_CTA_N>;

    using exp_tile = std::conditional_t<USE_PALETTE,
        Layouts::palette_idx_tile,
        Layouts::exp_tile>;

    using verbatim_exp_tile = std::conditional_t<HAS_VERBATIM,
        Layouts::verbatim_exp_tile,
        tk::alignment_dummy>;

    using mat_gl = Layouts::mat_gl;
    using sm_gl = Layouts::sm_gl;
    using mat_b_gl = Layouts::mat_b_gl<TILES_PER_CTA_N>;

    using exp_gl = std::conditional_t<USE_PALETTE,
        Layouts::palette_idx_gl,
        Layouts::exp_gl>;

    using verbatim_exp_gl = std::conditional_t<HAS_VERBATIM,
        Layouts::verbatim_exp_gl,
        Layouts::dummy_gl>;

    struct globals {
        mat_gl mat_a;
        mat_gl mat_c;
        u32 actual_m;
        u32 add_to_output;

        sm_gl mat_b_sm;
        exp_gl mat_b_exp;
        const u32 *tile_metadata;
        verbatim_exp_gl verbatim_exponents;
        const u8 *high_freq_palette;

        usize n_tiles;
        usize k_tiles;
    };

    struct input_block {
        mat_tile a[TILES_PER_CTA_M];
        sm_tile sm[TILES_PER_CTA_N];
        exp_tile exp[TILES_PER_CTA_N];
        verbatim_exp_tile verbatim_exp
            [TILES_PER_CTA_N][MAX_VERBATIM_ROWS > 0 ? MAX_VERBATIM_ROWS : 1];
        u64 verbatim_bitmask[TILES_PER_CTA_N];
        mat_b_tile b;
    };

    struct finish_block {
        mat_tile c[TILES_PER_CTA_M][TILES_PER_CTA_N];
    };

    static constexpr int INPUT_PIPE_STAGES = INPUT_PIPE_STAGE_COUNT;

    struct scratch_block {
        block_barrier reconstruction_barrier;
        alignas(16) u8 high_freq_palette[HIGH_FREQ_PALETTE_SIZE];
    };

    struct common_state {
        int2 tile_coord;
        u32 actual_m;
    };

    struct consumer_state {
        tk::rt_fl<16, TILES_PER_CTA_N * mat_tile::cols> accum;
    };
};

template<bool _USE_PALETTE, int _TILES_PER_CTA_N, int _INPUT_PIPE_STAGES, int _MAX_VERBATIM_ROWS>
struct ReconstructiveMatmulTemplate {
    static constexpr bool USE_PALETTE = _USE_PALETTE;
    static constexpr int TILES_PER_CTA_N = _TILES_PER_CTA_N;
    static constexpr int INPUT_PIPE_STAGES = _INPUT_PIPE_STAGES;
    static constexpr int MAX_VERBATIM_ROWS = _MAX_VERBATIM_ROWS;
    static constexpr bool HAS_VERBATIM = USE_PALETTE && (MAX_VERBATIM_ROWS > 0);
    static constexpr int NUM_CONSUMER_WARPS = TILES_PER_CTA_M * WARPS_PER_WARPGROUP;
    static constexpr int NUM_CONSUMER_WARPGROUPS = TILES_PER_CTA_M;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr usize GRID_SCHEDULE_GROUP = 12;
    static constexpr usize TILE_ROWS = UNWEIGHT_TILE_M;
    static constexpr usize PALETTE_TILE_COLS_U32 = UNWEIGHT_TILE_K / (2 * U8S_PER_U32);
    static constexpr usize PALETTE_TILE_SIZE_U32 = TILE_ROWS * PALETTE_TILE_COLS_U32;
    static constexpr usize PALETTE_WORDS_PER_CHUNK = U32S_PER_U32X4 / 2;
    static constexpr usize RECONSTRUCT_WARPS =
        TILES_PER_CTA_N == 1 ? WARPS_PER_WARPGROUP : NUM_CONSUMER_WARPS;
    static constexpr usize RECONSTRUCT_CHUNKS_PER_ROW = TILE_COLS_U32 / U32S_PER_U32X4;
    static constexpr u32 RECONSTRUCT_CHUNKS_PER_ROW_SHIFT = 2;
    static constexpr u32 RECONSTRUCT_CHUNKS_PER_ROW_MASK = RECONSTRUCT_CHUNKS_PER_ROW - 1;
    static constexpr usize RECONSTRUCT_THREADS = RECONSTRUCT_WARPS * tk::WARP_THREADS;
    static constexpr usize TOTAL_RECONSTRUCT_CHUNKS =
        TILES_PER_CTA_N * TILE_ROWS * RECONSTRUCT_CHUNKS_PER_ROW;
    static constexpr usize RECONSTRUCT_CHUNKS_PER_THREAD =
        (TOTAL_RECONSTRUCT_CHUNKS + RECONSTRUCT_THREADS - 1)
        / RECONSTRUCT_THREADS;

    static_assert(TILE_COLS_U32 % U32S_PER_U32X4 == 0);
    static_assert(RECONSTRUCT_CHUNKS_PER_ROW == (1 << RECONSTRUCT_CHUNKS_PER_ROW_SHIFT));
    static_assert(
        !USE_PALETTE || MAX_VERBATIM_ROWS == 0 ||
        MAX_VERBATIM_ROWS == 4 || MAX_VERBATIM_ROWS == 8 ||
        MAX_VERBATIM_ROWS == 16 || MAX_VERBATIM_ROWS == 32 ||
        MAX_VERBATIM_ROWS == 64);

    using layout = ReconstructiveMatmulLayout<
        USE_PALETTE, TILES_PER_CTA_N,
        INPUT_PIPE_STAGES, MAX_VERBATIM_ROWS>;
    using wide_c_tile = tk::st_bf<UNWEIGHT_TILE_M, UNWEIGHT_TILE_K * TILES_PER_CTA_N>;

    __device__ __forceinline__ static void enforced_store_reconstructed_row_bf16x8(
        typename layout::mat_b_tile &out,
        usize row,
        usize col,
        u32 v01,
        u32 v23,
        u32 v45,
        u32 v67)
    {
        u32 out_base = static_cast<u32>(__cvta_generic_to_shared(&out.data[0]));
        u32 swizzled = layout::mat_b_tile::idx(out_base, int2{(int)row, (int)col});

        asm volatile(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(swizzled), "r"(v01), "r"(v23), "r"(v45), "r"(v67)
            : "memory");
    }

    __device__ __forceinline__ static u32x4 enforced_load_u32x4_from_smem(u32 src)
    {
        u32x4 dst;

        asm volatile(
            "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
            : "r"(src));

        return dst;
    }

    __device__ __forceinline__ static u32x2 enforced_load_u32x2_from_smem(u32 src)
    {
        u32x2 dst;

        asm volatile(
            "ld.shared.v2.b32 {%0, %1}, [%2];\n"
            : "=r"(dst.x), "=r"(dst.y)
            : "r"(src));

        return dst;
    }

    __device__ __forceinline__ static u32 pack_palette_exp_word(
        u32 packed_indices,
        u32x4 palette_cache)
    {
        u32 idx_bytes =
            (packed_indices & 0x0000000Fu) |
            ((packed_indices << 4) & 0x00000F00u) |
            ((packed_indices << 8) & 0x000F0000u) |
            ((packed_indices << 12) & 0x0F000000u);

        u32 prmt_sel = packed_indices & 0x00007777u;
        u32 val_low = __byte_perm(palette_cache.x, palette_cache.y, prmt_sel);
        u32 val_high = __byte_perm(palette_cache.z, palette_cache.w, prmt_sel);
        u32 mask_high = ((idx_bytes >> 3) & 0x01010101u) * 0x000000FFu;

        return (val_low & ~mask_high) | (val_high & mask_high);
    }

    __device__ __forceinline__ static u32x4 decode_palette_indices(
        u32 exp_base,
        usize n_tile_idx,
        usize local_row,
        usize chunk_idx,
        u32x4 palette_cache)
    {
        usize exp_offset =
            n_tile_idx * PALETTE_TILE_SIZE_U32 +
            local_row * PALETTE_TILE_COLS_U32 +
            chunk_idx * PALETTE_WORDS_PER_CHUNK;

        u32 exp_addr = exp_base + exp_offset * sizeof(u32);
        u32x2 palette_idx_u32x2 = enforced_load_u32x2_from_smem(exp_addr);
        u32 w0 = palette_idx_u32x2.x;
        u32 w1 = palette_idx_u32x2.y;

        return u32x4{
            pack_palette_exp_word(w0, palette_cache),
            pack_palette_exp_word(w0 >> 16, palette_cache),
            pack_palette_exp_word(w1, palette_cache),
            pack_palette_exp_word(w1 >> 16, palette_cache)};
    }

    __device__ __forceinline__ static u32x4 load_verbatim_exp_chunk(
        const typename layout::verbatim_exp_tile
            verbatim_exp_tiles[TILES_PER_CTA_N][MAX_VERBATIM_ROWS],
        usize n_tile_idx,
        usize verbatim_row_idx,
        usize chunk_idx)
    {
        u32 verbatim_addr = static_cast<u32>(
            __cvta_generic_to_shared(&verbatim_exp_tiles[n_tile_idx][verbatim_row_idx].data[0]));

        verbatim_addr += chunk_idx * U32S_PER_U32X4 * sizeof(u32);

        return enforced_load_u32x4_from_smem(verbatim_addr);
    }

    __device__ __forceinline__ static void reconstruct_and_store_chunk(
        typename layout::mat_b_tile &out,
        usize global_row,
        usize col,
        u32x4 sm_u32x4,
        u32x4 exp_u32x4)
    {
        u32 pair_01 = __byte_perm(sm_u32x4.x, exp_u32x4.x, 0x5140);
        u32 pair_23 = __byte_perm(sm_u32x4.x, exp_u32x4.x, 0x7362);
        u32 pair_45 = __byte_perm(sm_u32x4.y, exp_u32x4.y, 0x5140);
        u32 pair_67 = __byte_perm(sm_u32x4.y, exp_u32x4.y, 0x7362);
        u32 pair_89 = __byte_perm(sm_u32x4.z, exp_u32x4.z, 0x5140);
        u32 pair_ab = __byte_perm(sm_u32x4.z, exp_u32x4.z, 0x7362);
        u32 pair_cd = __byte_perm(sm_u32x4.w, exp_u32x4.w, 0x5140);
        u32 pair_ef = __byte_perm(sm_u32x4.w, exp_u32x4.w, 0x7362);

        u32 bf16_01 = reconstruct_bf16_pair(pair_01);
        u32 bf16_23 = reconstruct_bf16_pair(pair_23);
        u32 bf16_45 = reconstruct_bf16_pair(pair_45);
        u32 bf16_67 = reconstruct_bf16_pair(pair_67);
        u32 bf16_89 = reconstruct_bf16_pair(pair_89);
        u32 bf16_ab = reconstruct_bf16_pair(pair_ab);
        u32 bf16_cd = reconstruct_bf16_pair(pair_cd);
        u32 bf16_ef = reconstruct_bf16_pair(pair_ef);

        enforced_store_reconstructed_row_bf16x8(
            out,
            global_row,
            col,
            bf16_01,
            bf16_23,
            bf16_45,
            bf16_67);

        enforced_store_reconstructed_row_bf16x8(
            out,
            global_row,
            col + (U32S_PER_U32X4 / 2) * U8S_PER_U32,
            bf16_89,
            bf16_ab,
            bf16_cd,
            bf16_ef);
    }

    __device__ __forceinline__ static u32x4 load_exp_chunk_palette(
        u32 exp_base,
        const typename layout::verbatim_exp_tile
            verbatim_exp_tiles[TILES_PER_CTA_N][MAX_VERBATIM_ROWS],
        const u64 verbatim_bitmasks[TILES_PER_CTA_N],
        u32x4 palette_cache,
        usize n_tile_idx,
        usize local_row,
        usize chunk_idx)
    {
        u64 verbatim_bitmask = verbatim_bitmasks[n_tile_idx];

        if (is_verbatim_row(verbatim_bitmask, local_row)) {
            if constexpr (HAS_VERBATIM) {
                u64 prior_mask = (local_row == 0) ? 0ull : ((1ull << local_row) - 1ull);
                u32 verbatim_row_idx = __popcll(verbatim_bitmask & prior_mask);

                return load_verbatim_exp_chunk(
                    verbatim_exp_tiles, n_tile_idx,
                    verbatim_row_idx, chunk_idx);
            } else {
                return u32x4{0, 0, 0, 0};
            }
        }

        return decode_palette_indices(exp_base, n_tile_idx, local_row, chunk_idx, palette_cache);
    }

    __device__ __forceinline__ static u32x4 load_exp_chunk_direct(
        u32 exp_base,
        usize n_tile_idx,
        usize local_row,
        usize col_u32)
    {
        usize exp_offset = n_tile_idx * TILE_SIZE_U32 + local_row * TILE_COLS_U32 + col_u32;
        u32 exp_addr = exp_base + exp_offset * sizeof(u32);

        return enforced_load_u32x4_from_smem(exp_addr);
    }

    __device__ static inline void warp_reconstruct_tile(
        const typename layout::sm_tile sm_tiles[TILES_PER_CTA_N],
        const typename layout::exp_tile exp_tiles[TILES_PER_CTA_N],
        const typename layout::verbatim_exp_tile
            verbatim_exp_tiles[TILES_PER_CTA_N][MAX_VERBATIM_ROWS],
        const u64 verbatim_bitmasks[TILES_PER_CTA_N],
        const u8 *high_freq_palette,
        typename layout::mat_b_tile &out,
        u32 warp_id)
    {
        if (warp_id >= RECONSTRUCT_WARPS) {
            return;
        }

        u32 sm_base = static_cast<u32>(__cvta_generic_to_shared(&sm_tiles[0].data[0]));
        u32 exp_base = static_cast<u32>(__cvta_generic_to_shared(&exp_tiles[0].data[0]));

        u32x4 palette_cache = u32x4{0, 0, 0, 0};
        if constexpr (USE_PALETTE) {
            u32 palette_addr = static_cast<u32>(__cvta_generic_to_shared(&high_freq_palette[0]));
            palette_cache = enforced_load_u32x4_from_smem(palette_addr);
        }

        usize tid = warp_id * tk::WARP_THREADS + tk::warp::laneid();

        #pragma unroll
        for (usize chunk_iter = 0; chunk_iter < RECONSTRUCT_CHUNKS_PER_THREAD; chunk_iter++) {
            usize chunk = tid + chunk_iter * RECONSTRUCT_THREADS;

            if (chunk >= TOTAL_RECONSTRUCT_CHUNKS) {
                continue;
            }

            usize global_row = chunk >> RECONSTRUCT_CHUNKS_PER_ROW_SHIFT;
            usize chunk_idx = chunk & RECONSTRUCT_CHUNKS_PER_ROW_MASK;
            usize col_u32 = chunk_idx * U32S_PER_U32X4;
            usize n_tile_idx = global_row >> TILE_M_SHIFT;
            usize local_row = global_row & TILE_M_MASK;
            usize sm_offset = n_tile_idx * TILE_SIZE_U32 + local_row * TILE_COLS_U32 + col_u32;
            u32 sm_addr = sm_base + sm_offset * sizeof(u32);
            u32x4 sm_u32x4 = enforced_load_u32x4_from_smem(sm_addr);

            u32x4 exp_u32x4;

            if constexpr (USE_PALETTE) {
                exp_u32x4 = load_exp_chunk_palette(
                    exp_base, verbatim_exp_tiles, verbatim_bitmasks, palette_cache,
                    n_tile_idx, local_row, chunk_idx);
            } else {
                exp_u32x4 = load_exp_chunk_direct(exp_base, n_tile_idx, local_row, col_u32);
            }

            reconstruct_and_store_chunk(
                out, global_row, col_u32 * U8S_PER_U32,
                sm_u32x4, exp_u32x4);
        }
    }

    __device__ static inline void load_mat_a_tiles(lcf::producer_load_args<layout> args)
    {
        for (usize i = 0; i < TILES_PER_CTA_M; i++) {
            tk::tma::load_async(
                args.input.a[i],
                args.globals.mat_a,
                {args.common.tile_coord.x + (int)i, args.iter},
                args.inputs_arrived);
        }
    }

    __device__ static inline void load_reconstruction_tiles(
        lcf::producer_load_args<layout> args,
        usize n_tile_base,
        usize k_iter)
    {
        if constexpr (USE_PALETTE) {
            for (usize i = 0; i < TILES_PER_CTA_N; i++) {
                usize n_tile = n_tile_base + i;
                usize tile_idx = n_tile * args.globals.k_tiles + k_iter;
                auto meta = TileMetadata::load(args.globals.tile_metadata, tile_idx);

                args.input.verbatim_bitmask[i] = meta.verbatim_bitmask;

                if constexpr (HAS_VERBATIM) {
                    usize verbatim_rows = meta.verbatim_row_count();
                    usize row_offset = meta.verbatim_row_offset;

                    for (usize verbatim_row_idx = 0; verbatim_row_idx < verbatim_rows; verbatim_row_idx++) {
                        tk::tma::load_async(
                            args.input.verbatim_exp[i][verbatim_row_idx],
                            args.globals.verbatim_exponents,
                            {(int)(row_offset + verbatim_row_idx), 0, 0, 0},
                            args.inputs_arrived);
                    }
                }
            }
        }

        for (usize i = 0; i < TILES_PER_CTA_N; i++) {
            usize n_tile = n_tile_base + i;

            tk::tma::load_async(
                args.input.sm[i],
                args.globals.mat_b_sm,
                {(int)n_tile, (int)k_iter, 0, 0},
                args.inputs_arrived);

            tk::tma::load_async(
                args.input.exp[i],
                args.globals.mat_b_exp,
                {(int)n_tile, (int)k_iter},
                args.inputs_arrived);
        }
    }

    __device__ static inline int2 compute_mat_c_coord(
        usize task_id,
        usize output_tiles_m,
        usize output_tiles_n)
    {
        usize total_output_tiles = output_tiles_m * output_tiles_n;

        if (task_id >= total_output_tiles) {
            return {-1, -1};
        }

        usize grouped_rows = (output_tiles_m / GRID_SCHEDULE_GROUP) * GRID_SCHEDULE_GROUP;
        usize grouped_tile_count = grouped_rows * output_tiles_n;

        if (task_id < grouped_tile_count) {
            usize group_period = GRID_SCHEDULE_GROUP * output_tiles_n;

            return {
                GRID_SCHEDULE_GROUP * (task_id / group_period) + task_id % GRID_SCHEDULE_GROUP,
                (task_id % group_period) / GRID_SCHEDULE_GROUP
            };
        } else {
            usize remainder_rows = output_tiles_m - grouped_rows;
            usize remainder_id = task_id - grouped_tile_count;

            return {
                grouped_rows + (remainder_id % remainder_rows),
                remainder_id / remainder_rows
            };
        }
    }

    __device__ static inline void common_setup(lcf::common_setup_args<layout> args)
    {
        usize output_tiles_m = args.globals.mat_c.rows() / (TILES_PER_CTA_M * UNWEIGHT_TILE_M);
        usize output_tiles_n = args.globals.mat_c.cols() / (TILES_PER_CTA_N * UNWEIGHT_TILE_K);
        usize task_id = args.task_iter * gridDim.x + blockIdx.x;
        int2 mat_c_coord = compute_mat_c_coord(task_id, output_tiles_m, output_tiles_n);
        u32 warpgroup_id = (u32)tk::warpgroup::groupid();
        u32 effective_id = (warpgroup_id == NUM_CONSUMER_WARPGROUPS) ? 0 : warpgroup_id;

        if (mat_c_coord.x < 0) {
            args.num_iters = -1;
            return;
        }

        args.num_iters = args.globals.mat_a.cols() / UNWEIGHT_TILE_K;
        args.common.actual_m = args.globals.actual_m;

        args.common.tile_coord = {
            (int)(mat_c_coord.x * TILES_PER_CTA_M + effective_id),
            (int)(mat_c_coord.y * TILES_PER_CTA_N)
        };

        if (args.task_iter == 0) {
            if (threadIdx.x == 0) {
                init(&args.scratch.reconstruction_barrier, NUM_CONSUMER_WARPS * tk::WARP_THREADS);
            }

            if constexpr (USE_PALETTE) {
                for (usize i = threadIdx.x; i < HIGH_FREQ_PALETTE_SIZE; i += blockDim.x) {
                    args.scratch.high_freq_palette[i] = __ldg(&args.globals.high_freq_palette[i]);
                }
            }

            __syncthreads();
        }
    }

    struct producer {
        __device__ static void setup(lcf::producer_setup_args<layout> args)
        {
            tk::warpgroup::decrease_registers<40>();
        }

        __device__ static void load(lcf::producer_load_args<layout> args)
        {
            if (tk::warpgroup::laneid() == 0) {
                u32 expected_bytes =
                    sizeof(args.input.a) +
                    sizeof(args.input.sm) +
                    sizeof(args.input.exp);

                if constexpr (HAS_VERBATIM) {
                    for (usize i = 0; i < TILES_PER_CTA_N; i++) {
                        usize n_tile = args.common.tile_coord.y + i;
                        usize tile_idx = n_tile * args.globals.k_tiles + args.iter;
                        auto meta = TileMetadata::load(args.globals.tile_metadata, tile_idx);

                        expected_bytes +=
                            meta.verbatim_row_count() *
                            tk::size_bytes<typename layout::verbatim_exp_tile>;
                    }
                }

                tk::tma::expect_bytes(args.inputs_arrived, expected_bytes);

                load_mat_a_tiles(args);
                load_reconstruction_tiles(args, args.common.tile_coord.y, args.iter);
            }
        }
    };

    struct consumer {
        __device__ static void setup(lcf::consumer_setup_args<layout> args)
        {
            tk::warpgroup::increase_registers<232>();
            tk::warp::zero(args.state.accum);
        }

        __device__ static void compute(lcf::consumer_compute_args<layout> args)
        {
            u32 warp_id = tk::warpgroup::groupid() * WARPS_PER_WARPGROUP + tk::warpgroup::warpid();

            warp_reconstruct_tile(
                args.input.sm,
                args.input.exp,
                args.input.verbatim_exp,
                args.input.verbatim_bitmask,
                args.scratch.high_freq_palette,
                args.input.b,
                warp_id);

            args.scratch.reconstruction_barrier.arrive_and_wait();

            tk::warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[tk::warpgroup::groupid()],
                args.input.b);

            tk::warpgroup::mma_async_wait();

            if (tk::warp::laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static void finish(lcf::consumer_finish_args<layout> args)
        {
            u32 gid = tk::warpgroup::groupid();

            tk::warpgroup::store(
                reinterpret_cast<wide_c_tile &>(args.finish.c[gid]),
                args.state.accum);

            tk::warpgroup::sync(gid + 4);

            u32 row_start = args.common.tile_coord.x * UNWEIGHT_TILE_M;

            if (tk::warpgroup::laneid() == 0 && row_start < args.common.actual_m) {
                for (usize i = 0; i < TILES_PER_CTA_N; i++) {
                    if (args.common.tile_coord.y + (int)i < (int)args.globals.n_tiles) {
                        if (args.globals.add_to_output != 0) {
                            tk::tma::store_add_async(
                                args.globals.mat_c,
                                args.finish.c[gid][i],
                                {args.common.tile_coord.x, args.common.tile_coord.y + (int)i});
                        } else {
                            tk::tma::store_async(
                                args.globals.mat_c,
                                args.finish.c[gid][i],
                                {args.common.tile_coord.x, args.common.tile_coord.y + (int)i});
                        }
                    }
                }

                tk::tma::store_async_read_wait();
            }

            tk::warp::zero(args.state.accum);

            if (tk::warp::laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

template<bool USE_PALETTE>
static inline auto make_reconstructive_matmul_exp_gl(
    const u8 *mat_b_exp,
    usize n_tiles,
    usize k_tiles)
{
    if constexpr (USE_PALETTE) {
        return Layouts::make_palette_idx_gl(const_cast<u8 *>(mat_b_exp), n_tiles, k_tiles);
    } else {
        return Layouts::make_exp_gl(const_cast<u8 *>(mat_b_exp), n_tiles, k_tiles);
    }
}

template<bool USE_PALETTE, int MAX_VERBATIM_ROWS>
static inline auto make_reconstructive_matmul_verbatim_gl(
    const u8 *verbatim_exponents,
    usize total_verbatim_rows)
{
    constexpr bool HAS_VERBATIM = USE_PALETTE && (MAX_VERBATIM_ROWS > 0);

    if constexpr (HAS_VERBATIM) {
        return Layouts::make_verbatim_exp_gl(
            const_cast<u8 *>(verbatim_exponents),
            total_verbatim_rows);
    } else {
        return Layouts::make_dummy_gl();
    }
}

template<typename Kernel>
static inline int launch_reconstructive_matmul_variant(
    bf16 *mat_a,
    bf16 *mat_c,
    usize actual_m,
    usize padded_m,
    usize n,
    usize k,
    u32 add_to_output,
    const u8 *mat_b_sm,
    const u8 *mat_b_exp,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u8 *high_freq_palette,
    usize total_verbatim_rows,
    usize grid_x,
    cudaStream_t stream)
{
    using globals = typename Kernel::layout::globals;
    constexpr unsigned long smem_size = tk::MAX_SHARED_MEMORY - 1024;

    usize n_tiles = n / UNWEIGHT_TILE_K;
    usize k_tiles = k / UNWEIGHT_TILE_K;

    globals g{
        Layouts::make_mat_gl(mat_a, padded_m, k),
        Layouts::make_mat_gl(mat_c, padded_m, n),
        (u32)actual_m,
        add_to_output,
        Layouts::make_sm_gl(
            const_cast<u32 *>(reinterpret_cast<const u32 *>(mat_b_sm)),
            n_tiles, k_tiles),
        make_reconstructive_matmul_exp_gl<Kernel::USE_PALETTE>(
            mat_b_exp, n_tiles, k_tiles),
        tile_metadata,
        make_reconstructive_matmul_verbatim_gl<Kernel::USE_PALETTE, Kernel::MAX_VERBATIM_ROWS>(
            verbatim_exponents,
            total_verbatim_rows),
        high_freq_palette,
        n_tiles,
        k_tiles
    };

    constexpr int num_threads = tk::prototype::detail::NUM_THREADS_v<Kernel>;

    cudaError_t err = cudaFuncSetAttribute(
        lcf::kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    if (err != cudaSuccess) {
        return (int)err;
    }

    lcf::kernel<Kernel><<<grid_x, num_threads, smem_size, stream>>>(g);

    return (int)cudaGetLastError();
}

enum class ReconstructiveMatmulVariant : u32 {
    N2P4 = 0,
    N2P3 = 1,
    N2P2 = 2,
    N3P2 = 3,
    N4P2 = 4,
    N1P6 = 5,
    N1P4 = 6,
    N1P2 = 7,
};

static inline usize variant_tiles_per_cta_n(ReconstructiveMatmulVariant v) {
    switch (v) {
        case ReconstructiveMatmulVariant::N1P6:
        case ReconstructiveMatmulVariant::N1P4:
        case ReconstructiveMatmulVariant::N1P2: return 1;
        case ReconstructiveMatmulVariant::N2P4:
        case ReconstructiveMatmulVariant::N2P3:
        case ReconstructiveMatmulVariant::N2P2: return 2;
        case ReconstructiveMatmulVariant::N3P2: return 3;
        case ReconstructiveMatmulVariant::N4P2: return 4;
        default: return 2;
    }
}

static inline usize variant_m_alignment(ReconstructiveMatmulVariant) {
    return TILES_PER_CTA_M * UNWEIGHT_TILE_M;
}

static inline usize variant_n_alignment(ReconstructiveMatmulVariant v) {
    return variant_tiles_per_cta_n(v) * UNWEIGHT_TILE_K;
}

template<bool USE_PALETTE, int MAX_VERBATIM_ROWS>
static inline int dispatch_variant(
    ReconstructiveMatmulVariant variant,
    bf16 *mat_a,
    bf16 *mat_c,
    usize actual_m,
    usize padded_m,
    usize n,
    usize k,
    u32 add_to_output,
    const u8 *mat_b_sm,
    const u8 *mat_b_exp,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u8 *high_freq_palette,
    usize total_verbatim_rows,
    usize grid_x,
    cudaStream_t stream)
{
    #define LAUNCH(N, P) \
        return launch_reconstructive_matmul_variant< \
            ReconstructiveMatmulTemplate< \
                USE_PALETTE, N, P, MAX_VERBATIM_ROWS>>( \
            mat_a, mat_c, actual_m, padded_m, \
            n, k, add_to_output, \
            mat_b_sm, mat_b_exp, tile_metadata, \
            verbatim_exponents, high_freq_palette, \
            total_verbatim_rows, grid_x, stream)

    if constexpr (MAX_VERBATIM_ROWS >= 32) {
        switch (variant) {
            case ReconstructiveMatmulVariant::N2P4:
            case ReconstructiveMatmulVariant::N2P3:
            case ReconstructiveMatmulVariant::N2P2: LAUNCH(2, 2);
            case ReconstructiveMatmulVariant::N3P2: LAUNCH(3, 2);
            case ReconstructiveMatmulVariant::N4P2: LAUNCH(4, 2);
            case ReconstructiveMatmulVariant::N1P6:
            case ReconstructiveMatmulVariant::N1P4:
            case ReconstructiveMatmulVariant::N1P2: LAUNCH(1, 2);
            default: return -1;
        }
    } else {
        switch (variant) {
            case ReconstructiveMatmulVariant::N2P4: LAUNCH(2, 4);
            case ReconstructiveMatmulVariant::N2P3: LAUNCH(2, 3);
            case ReconstructiveMatmulVariant::N2P2: LAUNCH(2, 2);
            case ReconstructiveMatmulVariant::N3P2: LAUNCH(3, 2);
            case ReconstructiveMatmulVariant::N4P2: LAUNCH(4, 2);
            case ReconstructiveMatmulVariant::N1P6: LAUNCH(1, 6);
            case ReconstructiveMatmulVariant::N1P4: LAUNCH(1, 4);
            case ReconstructiveMatmulVariant::N1P2: LAUNCH(1, 2);
            default: return -1;
        }
    }
    #undef LAUNCH
}

template<bool USE_PALETTE>
static inline int launch_reconstructive_matmul_source_variant(
    bf16 *mat_a,
    bf16 *mat_c,
    usize actual_m,
    usize padded_m,
    usize n,
    usize k,
    u32 add_to_output,
    const u8 *mat_b_sm,
    const u8 *mat_b_exp,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u8 *high_freq_palette,
    usize max_verbatim_per_tile,
    usize total_verbatim_rows,
    usize grid_x,
    ReconstructiveMatmulVariant variant,
    cudaStream_t stream)
{
    #define DISPATCH(MAX_ROWS) \
        return dispatch_variant<USE_PALETTE, MAX_ROWS>( \
            variant, mat_a, mat_c, actual_m, padded_m, n, k, add_to_output, \
            mat_b_sm, mat_b_exp, tile_metadata, verbatim_exponents, high_freq_palette, \
            total_verbatim_rows, grid_x, stream)

    if constexpr (USE_PALETTE) {
        if (max_verbatim_per_tile == 0)  DISPATCH(0);
        if (max_verbatim_per_tile <= 4)  DISPATCH(4);
        if (max_verbatim_per_tile <= 8)  DISPATCH(8);
        if (max_verbatim_per_tile <= 16) DISPATCH(16);
        if (max_verbatim_per_tile <= 32) DISPATCH(32);
        if (max_verbatim_per_tile <= 64) DISPATCH(64);
        return -1;
    } else {
        DISPATCH(0);
    }
    #undef DISPATCH
}

extern "C" int unweight_reconstructive_matmul_launch(
    bf16 *mat_a,
    bf16 *mat_c,
    usize actual_m,
    usize n,
    usize k,
    u32 add_to_output,
    const u8 *mat_b_sm,
    const u8 *mat_b_exp,
    const u32 *tile_metadata,
    const u8 *verbatim_exponents,
    const u8 *high_freq_palette,
    usize max_verbatim_per_tile,
    usize total_verbatim_rows,
    bool use_palette,
    ReconstructiveMatmulVariant variant,
    cudaStream_t stream)
{
    usize m_align = variant_m_alignment(variant);
    usize n_align = variant_n_alignment(variant);
    usize padded_m = ceil_div(actual_m, m_align) * m_align;
    usize output_tiles_m = padded_m / m_align;
    usize output_tiles_n = n / n_align;
    usize grid_x = min(output_tiles_m * output_tiles_n, PERSISTENT_GRID_SIZE);

    if (use_palette) {
        return launch_reconstructive_matmul_source_variant<true>(
            mat_a,
            mat_c,
            actual_m,
            padded_m,
            n,
            k,
            add_to_output,
            mat_b_sm,
            mat_b_exp,
            tile_metadata,
            verbatim_exponents,
            high_freq_palette,
            max_verbatim_per_tile,
            total_verbatim_rows,
            grid_x,
            variant,
            stream);
    }

    return launch_reconstructive_matmul_source_variant<false>(
        mat_a,
        mat_c,
        actual_m,
        padded_m,
        n,
        k,
        add_to_output,
        mat_b_sm,
        mat_b_exp,
        tile_metadata,
        verbatim_exponents,
        high_freq_palette,
        max_verbatim_per_tile,
        total_verbatim_rows,
        grid_x,
        variant,
        stream);
}