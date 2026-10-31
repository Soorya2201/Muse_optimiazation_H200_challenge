#pragma once

#include "common.cuh"
#include "layouts.cuh"

struct ComputeTileMetadataLayout {
    static constexpr int NUM_CONSUMER_WARPS = 2;

    using data_tile = Layouts::data_tile;
    using row_end_offsets_tile = Layouts::row_end_offsets_tile;
    using metadata_tile = Layouts::metadata_tile;
    using data_gl = Layouts::data_gl;
    using row_end_offsets_gl = Layouts::row_end_offsets_gl;
    using metadata_gl = Layouts::metadata_gl;

    struct globals {
        data_gl data;
        row_end_offsets_gl row_end_offsets;
        metadata_gl metadata;
        HuffmanEncodeEntry *encode_table;
        u32 *max_verbatim_per_tile;
        usize num_tiles_m;
        usize num_tiles_k;
    };

    struct input_block {
        data_tile data;
    };

    struct scratch_block {
        row_end_offsets_tile row_end_offsets;
        metadata_tile metadata;
        u64 verbatim_bitmask[TILES_PER_METADATA_BATCH];
        HuffmanEncodeEntry encode_table[HUFFMAN_ENCODE_TABLE_SIZE];
    };

    struct finish_block { };

    struct common_state {
        usize first_tile_idx;
    };

    struct consumer_state { };
};

struct ComputeTileMetadataTemplate {
    using layout = ComputeTileMetadataLayout;

    static constexpr int NUM_CONSUMER_WARPS = layout::NUM_CONSUMER_WARPS;
    static constexpr int NUM_PRODUCER_WARPS = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;

    static constexpr usize TILES_PER_THREAD_IN_FINISH =
        TILES_PER_METADATA_BATCH / tk::WARP_THREADS;

    static constexpr usize U32X4S_PER_WARP1_OFFSETS =
        tk::WARP_THREADS * sizeof(u16) / sizeof(u32x4);

    __device__ static u32 warp_inclusive_prefix_sum(u32 val)
    {
        #pragma unroll
        for (usize delta = 1; delta < tk::WARP_THREADS; delta *= 2) {
            u32 n = __shfl_up_sync(0xFFFFFFFF, val, delta);

            if (tk::warp::laneid() >= delta) {
                val += n;
            }
        }

        return val;
    }

    __device__ static u16 fixup_warp1_offsets(
        u16 *offsets_u16,
        u32x4 *offsets_u32x4,
        usize slot)
    {
        u16 warp0_total = offsets_u16[slot * UNWEIGHT_TILE_M + tk::WARP_THREADS - 1];
        u32 total_packed = warp0_total | ((u32)warp0_total << 16);

        u32x4 *warp1_u32x4 = &offsets_u32x4[
            slot * U32X4S_PER_TILE_OFFSETS + U32X4S_PER_WARP1_OFFSETS];

        #pragma unroll
        for (usize i = 0; i < U32X4S_PER_WARP1_OFFSETS; i++) {
            u32x4 val = warp1_u32x4[i];

            val.x += total_packed;
            val.y += total_packed;
            val.z += total_packed;
            val.w += total_packed;

            warp1_u32x4[i] = val;
        }

        return warp0_total;
    }

    __device__ static u32 finalize_metadata(
        layout::scratch_block &scratch,
        usize num_tiles,
        u32 lane_id)
    {
        u16 *row_end_offsets_u16 = reinterpret_cast<u16 *>(scratch.row_end_offsets.data);
        u32x4 *row_end_offsets_u32x4 = reinterpret_cast<u32x4 *>(scratch.row_end_offsets.data);
        u32 *metadata = reinterpret_cast<u32 *>(scratch.metadata.data);

        u32 max_verbatim = 0;

        #pragma unroll
        for (usize t = 0; t < TILES_PER_THREAD_IN_FINISH; t++) {
            usize slot = lane_id + t * tk::WARP_THREADS;

            if (slot >= num_tiles) {
                break;
            }

            u16 warp1_local_total = row_end_offsets_u16[
                slot * UNWEIGHT_TILE_M + UNWEIGHT_TILE_M - 1];

            u16 warp0_total = fixup_warp1_offsets(row_end_offsets_u16, row_end_offsets_u32x4, slot);
            u64 verbatim_bitmask = scratch.verbatim_bitmask[slot];
            u32 tile_bit_len = (u32)warp0_total + (u32)warp1_local_total;
            u32 verbatim_count = (u32)__popcll(verbatim_bitmask);

            TileMetadata meta = {
                .tile_start_bit = tile_bit_len,
                .verbatim_row_offset = verbatim_count,
                .verbatim_bitmask = verbatim_bitmask
            };

            meta.store(metadata, slot);

            max_verbatim = max(max_verbatim, verbatim_count);
        }

        #pragma unroll
        for (usize delta = tk::WARP_THREADS / 2; delta > 0; delta /= 2) {
            max_verbatim = max(max_verbatim, __shfl_xor_sync(0xFFFFFFFF, max_verbatim, delta));
        }

        return max_verbatim;
    }

    __device__ static u32 compute_row_bit_len(
        const u32x4 *data_row_u32x4,
        const HuffmanEncodeEntry *encode_table,
        u64 *verbatim_bitmask,
        usize row)
    {
        u32 bit_len = 0;

        #pragma unroll
        for (usize i = 0; i < U32X4S_PER_TILE_ROW; i++) {
            u32x4 lane = data_row_u32x4[i];

            #pragma unroll
            for (usize j = 0; j < U32S_PER_U32X4; j++) {
                u32 pair = (&lane.x)[j];
                auto exp = ExpPair::from_bf16_pair(pair);

                u8 len0 = encode_table[exp.lo].length;
                u8 len1 = encode_table[exp.hi].length;

                if (len0 == 0 || len1 == 0) {
                    atomicOr(reinterpret_cast<unsigned long long *>(verbatim_bitmask), 1ULL << row);
                    return 0;
                }

                bit_len += len0 + len1;
            }
        }

        return bit_len;
    }

    __device__ static void common_setup(lcf::common_setup_args<layout> args)
    {
        usize total_tiles = args.globals.num_tiles_m * args.globals.num_tiles_k;
        usize total_batches = ceil_div(total_tiles, (usize)TILES_PER_METADATA_BATCH);
        usize batch_id = args.task_iter * gridDim.x + blockIdx.x;

        if (batch_id >= total_batches) {
            args.num_iters = -1;
            return;
        }

        usize first_tile = batch_id * TILES_PER_METADATA_BATCH;
        usize tiles_in_batch = min((usize)TILES_PER_METADATA_BATCH, total_tiles - first_tile);

        args.common.first_tile_idx = first_tile;
        args.num_iters = tiles_in_batch;

        for (usize i = threadIdx.x; i < TILES_PER_METADATA_BATCH; i += blockDim.x) {
            args.scratch.verbatim_bitmask[i] = 0;
        }

        if (args.task_iter == 0) {
            for (usize i = threadIdx.x; i < HUFFMAN_ENCODE_TABLE_SIZE; i += blockDim.x) {
                args.scratch.encode_table[i] = args.globals.encode_table[i];
            }
        }
    }

    struct producer {
        __device__ static void setup(lcf::producer_setup_args<layout> args) { }

        __device__ static void load(lcf::producer_load_args<layout> args)
        {
            if (tk::warp::laneid() == 0) {
                usize tile_idx = args.common.first_tile_idx + args.iter;
                auto coord = TileCoord::from_row_outer_idx(tile_idx, args.globals.num_tiles_k);

                tk::tma::expect(args.inputs_arrived, args.input.data);

                tk::tma::load_async(
                    args.input.data,
                    args.globals.data,
                    {(int)coord.row, (int)coord.col},
                    args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(lcf::consumer_setup_args<layout> args) { }

        __device__ static void compute(lcf::consumer_compute_args<layout> args)
        {
            using consumers = tk::group<NUM_CONSUMER_WARPS>;

            u32 warp_id = consumers::warpid();
            u32 lane_id = tk::warp::laneid();
            usize tile_idx = args.common.first_tile_idx + args.iter;
            usize row = warp_id * tk::WARP_THREADS + lane_id;
            usize slot = args.iter;

            const u32x4 *data_u32x4 = reinterpret_cast<const u32x4 *>(&args.input.data.data[0]);
            const u32x4 *data_row_u32x4 = &data_u32x4[row * U32X4S_PER_TILE_ROW];

            u32 row_len = compute_row_bit_len(
                data_row_u32x4,
                args.scratch.encode_table,
                &args.scratch.verbatim_bitmask[slot],
                row);

            u32 warp_sum = warp_inclusive_prefix_sum(row_len);
            u16 *row_end_offsets = reinterpret_cast<u16 *>(args.scratch.row_end_offsets.data);

            row_end_offsets[slot * UNWEIGHT_TILE_M + row] = (u16)warp_sum;

            __syncwarp();

            if (lane_id == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static void finish(lcf::consumer_finish_args<layout> args)
        {
            using consumers = tk::group<NUM_CONSUMER_WARPS>;

            u32 lane_id = tk::warp::laneid();

            if (consumers::warpid() == 0) {
                u32 batch_max_verbatim = finalize_metadata(args.scratch, args.num_iters, lane_id);

                __syncwarp();

                if (lane_id == 0) {
                    atomicMax(args.globals.max_verbatim_per_tile, batch_max_verbatim);

                    usize batch_id = args.task_iter * gridDim.x + blockIdx.x;

                    tk::tma::store_async(
                        args.globals.row_end_offsets,
                        args.scratch.row_end_offsets,
                        {(int)batch_id, 0, 0, 0});

                    tk::tma::store_async(
                        args.globals.metadata,
                        args.scratch.metadata,
                        {(int)batch_id, 0, 0, 0});

                    tk::tma::store_async_read_wait();
                }
            }

            if (lane_id == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

extern "C" void unweight_compute_tile_metadata_launch(
    bf16 *data,
    u16 *row_end_offsets,
    u32 *metadata,
    HuffmanEncodeEntry *encode_table,
    u32 *max_verbatim_per_tile,
    usize rows,
    usize cols,
    cudaStream_t stream)
{
    auto td = UnweightTileDims::from_dims(rows, cols);
    usize num_batches = ceil_div(td.num_tiles, (usize)TILES_PER_METADATA_BATCH);
    usize grid_x = min(num_batches, PERSISTENT_GRID_SIZE);

    ComputeTileMetadataLayout::globals globals{
        .data = Layouts::make_data_gl(data, rows, cols),
        .row_end_offsets = Layouts::make_row_end_offsets_gl(row_end_offsets, num_batches),
        .metadata = Layouts::make_metadata_gl(metadata, num_batches),
        .encode_table = encode_table,
        .max_verbatim_per_tile = max_verbatim_per_tile,
        .num_tiles_m = td.num_tiles_m,
        .num_tiles_k = td.num_tiles_k
    };

    lcf_launch<ComputeTileMetadataTemplate>(globals, grid_x, stream);
}
