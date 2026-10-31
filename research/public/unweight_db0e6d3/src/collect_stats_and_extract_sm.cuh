#pragma once

#include "common.cuh"
#include "layouts.cuh"

struct CollectStatsAndExtractSmLayout {
    using data_tile = Layouts::data_tile;
    using sm_tile = Layouts::sm_tile;
    using data_gl = Layouts::data_gl;
    using sm_gl = Layouts::sm_gl;

    struct globals {
        data_gl data;
        sm_gl sm;
        u32 *hist;
        usize num_tiles_m;
        usize num_tiles_k;
    };

    struct input_block  { data_tile data; };
    struct output_block { sm_tile sm; };
    struct scratch_block { u32 hist[EXP_HIST_BINS]; };
    struct finish_block { };
    struct common_state {
        usize first_tile_idx;
    };
    struct producer_state { };
    struct consumer_state { };
};

struct CollectStatsAndExtractSmTemplate {
    using layout = CollectStatsAndExtractSmLayout;

    static constexpr int NUM_CONSUMER_WARPS = 2;
    static constexpr int NUM_PRODUCER_WARPS = 2;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int OUTPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    static constexpr int CLUSTER_BLOCKS = UNWEIGHT_CLUSTER_SIZE;

    __device__ static void update_hist(u32 bf16_pair, u32 *hist)
    {
        auto exps = ExpPair::from_bf16_pair(bf16_pair);

        atomicAdd(&hist[exps.lo], (u32)1);
        atomicAdd(&hist[exps.hi], (u32)1);
    }

    __device__ static u32 pack_u16(u16 lo, u16 hi)
    {
        return __byte_perm(lo, hi, 0x5410);
    }

    __device__ static u16 extract_sm_pair(u32 bf16_pair)
    {
        u32 mantissas = __byte_perm(bf16_pair, 0, 0x0020);
        u32 signs = __byte_perm(bf16_pair, 0, 0x0031);

        return (mantissas & 0x7F7F) | (signs & 0x8080);
    }

    __device__ static void cluster_reduce_hist(u32 *block_hist,
                                               u32 *__restrict__ hist,
                                               usize num_consumer_threads)
    {
        auto cluster = cg::this_cluster();

        cluster.sync();

        if (cluster.block_rank() == 0) {
            for (usize bin = threadIdx.x; bin < EXP_HIST_BINS; bin += num_consumer_threads) {
                u32 total = 0;

                for (usize i = 0; i < UNWEIGHT_CLUSTER_SIZE; i++) {
                    total += *cluster.map_shared_rank(&block_hist[bin], i);
                }

                atomicAdd(&hist[bin], total);
            }
        }

        cluster.sync();
    }

    __device__ static void tile_row_exp_hist_and_extract_sm(
        const layout::data_tile &data_tile,
        layout::sm_tile &sm_tile,
        u32 *__restrict__ hist,
        usize row)
    {
        const bf16 *data_row = &data_tile.data[row * UNWEIGHT_TILE_K];
        const u32x4 *data_row_u32x4 = reinterpret_cast<const u32x4 *>(data_row);

        u32x4 *sm_row_u32x4 = reinterpret_cast<u32x4 *>(
            &sm_tile.data[row * TILE_COLS_U32]);

        #pragma unroll
        for (usize i = 0; i < U32X4S_PER_TILE_ROW / 2; i++) {
            u32x4 lane0 = data_row_u32x4[i * 2];
            u32x4 lane1 = data_row_u32x4[i * 2 + 1];

            update_hist(lane0.x, hist);
            update_hist(lane0.y, hist);
            update_hist(lane0.z, hist);
            update_hist(lane0.w, hist);
            update_hist(lane1.x, hist);
            update_hist(lane1.y, hist);
            update_hist(lane1.z, hist);
            update_hist(lane1.w, hist);

            u16 sm0 = extract_sm_pair(lane0.x);
            u16 sm1 = extract_sm_pair(lane0.y);
            u16 sm2 = extract_sm_pair(lane0.z);
            u16 sm3 = extract_sm_pair(lane0.w);
            u16 sm4 = extract_sm_pair(lane1.x);
            u16 sm5 = extract_sm_pair(lane1.y);
            u16 sm6 = extract_sm_pair(lane1.z);
            u16 sm7 = extract_sm_pair(lane1.w);

            sm_row_u32x4[i] = u32x4{
                pack_u16(sm0, sm1),
                pack_u16(sm2, sm3),
                pack_u16(sm4, sm5),
                pack_u16(sm6, sm7)
            };
        }
    }

    __device__ static void common_setup(lcsf::common_setup_args<layout> args)
    {
        if (args.task_iter > 0) {
            args.num_iters = -1;
            return;
        }

        for (usize i = threadIdx.x; i < EXP_HIST_BINS; i += blockDim.x) {
            args.scratch.hist[i] = 0;
        }

        usize total_tiles = args.globals.num_tiles_m * args.globals.num_tiles_k;
        usize tiles_per_block = total_tiles / gridDim.x;
        usize remainder = total_tiles % gridDim.x;
        usize extra = (blockIdx.x < remainder) ? 1 : 0;

        args.common.first_tile_idx = blockIdx.x * tiles_per_block + min(blockIdx.x, remainder);
        args.num_iters = tiles_per_block + extra;
    }

    struct producer {
        __device__ static void setup(lcsf::producer_setup_args<layout> args) { }

        __device__ static void load(lcsf::producer_load_args<layout> args)
        {
            using producers = tk::group<NUM_PRODUCER_WARPS>;

            if (producers::warpid() == 0 && tk::warp::laneid() == 0) {
                usize tile_idx = args.common.first_tile_idx + args.iter;
                auto coord = TileCoord::from_row_outer_idx(tile_idx, args.globals.num_tiles_k);

                tk::tma::expect(args.inputs_arrived, args.input);

                tk::tma::load_async(
                    args.input.data,
                    args.globals.data,
                    {(int)coord.row, (int)coord.col},
                    args.inputs_arrived);
            }
        }

        __device__ static void store(lcsf::producer_store_args<layout> args)
        {
            using producers = tk::group<NUM_PRODUCER_WARPS>;

            if (producers::warpid() == 1 && tk::warp::laneid() == 0) {
                usize tile_idx = args.common.first_tile_idx + args.iter;
                auto coord = TileCoord::from_row_outer_idx(tile_idx, args.globals.num_tiles_k);

                tk::tma::store_async(
                    args.globals.sm,
                    args.output.sm,
                    {(int)coord.row, (int)coord.col, 0, 0});

                tk::tma::store_async_read_wait();

                arrive(args.outputs_finished);
            }
        }
    };

    struct consumer {
        __device__ static void setup(lcsf::consumer_setup_args<layout> args) { }

        __device__ static void compute(lcsf::consumer_compute_args<layout> args)
        {
            usize row = threadIdx.x;

            tile_row_exp_hist_and_extract_sm(
                args.input.data,
                args.output.sm,
                args.scratch.hist,
                row
            );

            __syncwarp();

            if (tk::warp::laneid() == 0) {
                arrive(args.inputs_finished);
                arrive(args.outputs_arrived);
            }
        }

        __device__ static void finish(lcsf::consumer_finish_args<layout> args)
        {
            cluster_reduce_hist(
                args.scratch.hist, args.globals.hist,
                NUM_CONSUMER_WARPS * tk::WARP_THREADS);

            if (tk::warp::laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

extern "C" void unweight_collect_stats_and_extract_sm_launch(
    const bf16 *data,
    u32 *sm,
    u32 *hist,
    usize rows,
    usize cols,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);
    usize grid_x = min(td.num_tiles, PERSISTENT_GRID_SIZE);
    grid_x = ceil_div(grid_x, (usize)UNWEIGHT_CLUSTER_SIZE) * UNWEIGHT_CLUSTER_SIZE;

    CollectStatsAndExtractSmLayout::globals globals{
        .data = Layouts::make_data_gl(const_cast<bf16 *>(data), rows, cols),
        .sm = Layouts::make_sm_gl(sm, td.num_tiles_m, td.num_tiles_k),
        .hist = hist,
        .num_tiles_m = td.num_tiles_m,
        .num_tiles_k = td.num_tiles_k
    };

    lcsf_launch<CollectStatsAndExtractSmTemplate>(globals, grid_x, stream, UNWEIGHT_CLUSTER_SIZE);
}
