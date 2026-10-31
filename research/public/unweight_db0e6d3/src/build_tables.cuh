#pragma once

#include "common.cuh"

struct HuffmanNode {
    usize freq;
    i16 left;
    i16 right;
    u8 symbol;
};

struct alignas(4) HuffmanEncodeEntry {
    u16 code;
    u8 length;
};

struct HistEntry {
    u32 freq;
    u8 sym;
};

__device__ __forceinline__ usize sort_histogram_descending(
    const u32 *__restrict__ hist, HistEntry *sorted)
{
    usize count = 0;

    for (usize sym = 0; sym < EXP_HIST_BINS; sym++) {
        u32 freq = hist[sym];

        if (freq == 0) {
            continue;
        }

        usize j = count;

        while (j > 0 && sorted[j - 1].freq < freq) {
            sorted[j] = sorted[j - 1];
            j--;
        }

        sorted[j] = {freq, (u8)sym};
        count++;
    }

    return count;
}

__device__ __forceinline__ usize build_palette_from_sorted(
    const HistEntry *sorted,
    usize count,
    u8 *high_freq_palette,
    u8 *symbol_to_palette_idx)
{
    usize palette_count = (count < HIGH_FREQ_PALETTE_SIZE) ? count : HIGH_FREQ_PALETTE_SIZE;

    for (usize i = 0; i < palette_count; i++) {
        high_freq_palette[i] = sorted[i].sym;
        symbol_to_palette_idx[sorted[i].sym] = (u8)i;
    }

    return palette_count;
}

__device__ __forceinline__ void build_leaves_from_sorted(
    const HistEntry *sorted,
    usize palette_count,
    HuffmanNode *tree)
{
    for (usize i = 0; i < palette_count; i++) {
        auto e = sorted[palette_count - 1 - i];

        tree[i] = {e.freq, HUFFMAN_NO_CHILD, HUFFMAN_NO_CHILD, e.sym};
    }
}

__device__ __forceinline__ i16 pick_smallest(HuffmanNode *tree,
                                              usize leaf_count,
                                              usize node_count,
                                              usize *leaf_idx,
                                              usize *internal_idx)
{
    bool leaf_available = *leaf_idx < leaf_count;
    bool internal_available = *internal_idx < node_count;

    bool prefer_leaf = leaf_available &&
                       (!internal_available || tree[*leaf_idx].freq <= tree[*internal_idx].freq);

    return prefer_leaf ? (i16)(*leaf_idx)++ : (i16)(*internal_idx)++;
}

__device__ __forceinline__ usize build_huffman_tree(HuffmanNode *tree,
                                                     usize leaf_count)
{
    if (leaf_count <= 1) {
        return leaf_count;
    }

    usize leaf_idx = 0;
    usize internal_idx = leaf_count;
    usize node_count = leaf_count;

    for (usize i = 0; i < leaf_count - 1; i++) {
        i16 child1 = pick_smallest(tree, leaf_count, node_count, &leaf_idx, &internal_idx);
        i16 child2 = pick_smallest(tree, leaf_count, node_count, &leaf_idx, &internal_idx);

        usize freq = tree[child1].freq + tree[child2].freq;

        tree[node_count] = {freq, child1, child2, 0};

        node_count++;
    }

    return node_count;
}

__device__ __forceinline__ u8 build_encode_table(const HuffmanNode *tree,
                                                  usize node_count,
                                                  usize leaf_count,
                                                  HuffmanEncodeEntry *encode_table)
{
    u8 depths[HIGH_FREQ_PALETTE_SIZE * 2];
    u16 codes[HIGH_FREQ_PALETTE_SIZE * 2];

    i16 root = node_count - 1;

    depths[root] = 0;
    codes[root] = 0;

    for (i16 i = root; i >= (i16)leaf_count; i--) {
        u8 depth = depths[i];
        u16 code = codes[i];

        depths[tree[i].left] = depth + 1;
        depths[tree[i].right] = depth + 1;
        codes[tree[i].left] = code;
        codes[tree[i].right] = code | (1 << depth);
    }

    u8 max_depth = 0;

    for (usize i = 0; i < leaf_count; i++) {
        u8 sym = tree[i].symbol;

        encode_table[sym] = {codes[i], depths[i]};
        
        if (depths[i] > max_depth) {
            max_depth = depths[i];
        }
    }

    return max_depth;
}

extern "C" __global__ void unweight_build_encode_table_and_palette(
    const u32 *__restrict__ hist,
    HuffmanEncodeEntry *__restrict__ encode_table,
    u8 *__restrict__ high_freq_palette,
    u8 *__restrict__ symbol_to_palette_idx,
    u32 *__restrict__ decode_table_size)
{
    __shared__ HistEntry sorted[EXP_HIST_BINS];
    __shared__ HuffmanNode tree[HIGH_FREQ_PALETTE_SIZE * 2];

    if (threadIdx.x == 0) {
        usize count = sort_histogram_descending(hist, sorted);

        usize palette_count = build_palette_from_sorted(
            sorted, count,
            high_freq_palette, symbol_to_palette_idx);

        if (palette_count > 0) {
            build_leaves_from_sorted(sorted, palette_count, tree);

            usize node_count = build_huffman_tree(tree, palette_count);
            u8 max_code_len = build_encode_table(tree, node_count, palette_count, encode_table);

            *decode_table_size = 1u << max_code_len;
        } else {
            *decode_table_size = 0;
        }
    }
}

extern "C" __global__ void unweight_build_decode_table(
    const HuffmanEncodeEntry *__restrict__ encode_table,
    const u8 *__restrict__ symbol_to_palette_idx,
    HuffmanDecodeEntry *__restrict__ decode_table,
    u8 max_code_len)
{
    usize sym = threadIdx.x;
    auto enc = encode_table[sym];

    if (enc.length > 0) {
        usize pad_bits = max_code_len - enc.length;
        usize num_entries = 1 << pad_bits;

        HuffmanDecodeEntry entry;
        entry.symbol = (u8)sym;
        entry.length = enc.length;
        entry.palette_idx = symbol_to_palette_idx[sym];
        entry._pad = 0;

        for (usize j = 0; j < num_entries; j++) {
            usize idx = enc.code | (j << enc.length);

            decode_table[idx] = entry;
        }
    }
}

extern "C" void unweight_build_encode_table_and_palette_launch(
    const u32 *hist,
    void *encode_table,
    u8 *high_freq_palette,
    u8 *symbol_to_palette_idx,
    u32 *decode_table_size,
    cudaStream_t stream)
{
    unweight_build_encode_table_and_palette<<<1, 1, 0, stream>>>(
        hist,
        reinterpret_cast<HuffmanEncodeEntry *>(encode_table),
        high_freq_palette,
        symbol_to_palette_idx,
        decode_table_size);
}

extern "C" void unweight_build_decode_table_launch(
    const void *encode_table,
    const u8 *symbol_to_palette_idx,
    void *decode_table,
    u8 max_code_len,
    cudaStream_t stream)
{
    unweight_build_decode_table<<<1, HUFFMAN_ENCODE_TABLE_SIZE, 0, stream>>>(
        reinterpret_cast<const HuffmanEncodeEntry *>(encode_table),
        symbol_to_palette_idx,
        reinterpret_cast<HuffmanDecodeEntry *>(decode_table),
        max_code_len);
}