#pragma once

#include "common.cuh"

__device__ u32 slice_prefix_sum_strided(u32 *metadata,
                                        usize start_tile,
                                        usize end_tile,
                                        usize field_offset)
{
    u32 sum = 0;

    for (usize tile = start_tile; tile < end_tile; tile++) {
        u32 *field = &metadata[tile * TILE_METADATA_SIZE_U32 + field_offset];
        u32 val = *field;
        *field = sum;
        sum += val;
    }

    return sum;
}

__device__ void apply_global_offset_strided(u32 *metadata,
                                            usize start_tile,
                                            usize end_tile,
                                            usize field_offset,
                                            u32 offset)
{
    for (usize tile = start_tile; tile < end_tile; tile++) {
        u32 *field = &metadata[tile * TILE_METADATA_SIZE_U32 + field_offset];
        *field += offset;
    }
}


__device__ __forceinline__ u32 warp_prefix_sum(u32 val) {
    u32 lane_id = threadIdx.x % tk::WARP_THREADS;

    #pragma unroll
    for (u32 delta = 1; delta < tk::WARP_THREADS; delta *= 2) {
        u32 other = __shfl_up_sync(0xffffffff, val, delta);

        if (lane_id >= delta) {
            val += other;
        }
    }

    return val;
}

__device__ void block_prefix_sum(u32 *data, u32 *warp_totals) {
    u32 tid = threadIdx.x;
    u32 lane_id = tid % tk::WARP_THREADS;
    u32 warp_id = tid / tk::WARP_THREADS;
    u32 val = data[tid];

    val = warp_prefix_sum(val);
    data[tid] = val;

    __syncthreads();

    if (lane_id == tk::WARP_THREADS - 1) {
        warp_totals[warp_id] = val;
    }

    __syncthreads();

    if (tid < WARPS_PER_BLOCK) {
        u32 warp_sum = warp_totals[tid];

        warp_sum = warp_prefix_sum(warp_sum);
        warp_totals[tid] = warp_sum;
    }

    __syncthreads();

    if (warp_id > 0) {
        data[tid] += warp_totals[warp_id - 1];
    }
}

__device__ void cluster_prefix_sum(u32 *block_data) {
    __shared__ u32 warp_totals[WARPS_PER_BLOCK];

    auto cluster = cg::this_cluster();

    cluster.sync();

    block_prefix_sum(block_data, warp_totals);

    __syncthreads();

    u32 block_total = block_data[MAX_THREADS_PER_BLOCK - 1];

    if (threadIdx.x == 0) {
        warp_totals[0] = block_total;
    }

    cluster.sync();

    u32 preceding_blocks_sum = 0;

    for (usize b = 0; b < blockIdx.x; b++) {
        preceding_blocks_sum += *cluster.map_shared_rank(&warp_totals[0], b);
    }

    block_data[threadIdx.x] += preceding_blocks_sum;

    cluster.sync();
}

__global__
__cluster_dims__(UNWEIGHT_CLUSTER_SIZE, 1, 1)
void unweight_v2_metadata_lens_to_offsets(u32 *__restrict__ metadata,
                                          usize tile_count,
                                          u32 *__restrict__ compressed_len_in_words_out,
                                          u32 *__restrict__ verbatim_row_count_out)
{
    __shared__ u32 block_sums_bits[MAX_THREADS_PER_BLOCK];
    __shared__ u32 block_sums_verbatim[MAX_THREADS_PER_BLOCK];

    constexpr u32 TOTAL_THREADS = UNWEIGHT_CLUSTER_SIZE * MAX_THREADS_PER_BLOCK;

    u32 tid = threadIdx.x;
    usize global_tid = blockIdx.x * MAX_THREADS_PER_BLOCK + tid;
    usize base_tiles_per_thread = tile_count / TOTAL_THREADS;
    usize remainder = tile_count % TOTAL_THREADS;
    usize extra = (global_tid < remainder) ? 1 : 0;
    usize tiles_before = base_tiles_per_thread * global_tid + min(global_tid, remainder);
    usize start_tile = tiles_before;
    usize end_tile = start_tile + base_tiles_per_thread + extra;
    u32 local_sum_bits = slice_prefix_sum_strided(metadata, start_tile, end_tile, 0);
    u32 local_sum_verbatim = slice_prefix_sum_strided(metadata, start_tile, end_tile, 1);

    block_sums_bits[tid] = local_sum_bits;
    block_sums_verbatim[tid] = local_sum_verbatim;

    cluster_prefix_sum(block_sums_bits);
    u32 global_offset_bits = block_sums_bits[tid] - local_sum_bits;
    apply_global_offset_strided(metadata, start_tile, end_tile, 0, global_offset_bits);

    cluster_prefix_sum(block_sums_verbatim);
    u32 global_offset_verbatim = block_sums_verbatim[tid] - local_sum_verbatim;
    apply_global_offset_strided(metadata, start_tile, end_tile, 1, global_offset_verbatim);

    auto cluster = cg::this_cluster();

    bool is_last_thread =
        cluster.block_rank() == UNWEIGHT_CLUSTER_SIZE - 1 &&
        tid == MAX_THREADS_PER_BLOCK - 1;

    if (is_last_thread) {
        u32 total_bits = block_sums_bits[tid];
        *compressed_len_in_words_out = ceil_div(total_bits, BITS_PER_WORD);
        *verbatim_row_count_out = block_sums_verbatim[tid];
    }
}

extern "C" void unweight_metadata_lens_to_offsets_launch(
    u32 *metadata,
    usize tile_count,
    u32 *compressed_len_in_words_out,
    u32 *verbatim_row_count_out,
    cudaStream_t stream)
{
    dim3 grid(UNWEIGHT_CLUSTER_SIZE, 1, 1);
    dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = UNWEIGHT_CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        unweight_v2_metadata_lens_to_offsets,
        metadata,
        tile_count,
        compressed_len_in_words_out,
        verbatim_row_count_out);
}