#pragma once

#include "common.cuh"

template<u32 DECODE_TABLE_SIZE>
struct HuffmanBitReader {
    static constexpr u32 WORD_SHIFT = 5;
    static constexpr u32 CODE_MASK = DECODE_TABLE_SIZE - 1;
    static_assert(BITS_PER_WORD == (1 << WORD_SHIFT));
    static_assert(
        (DECODE_TABLE_SIZE & (DECODE_TABLE_SIZE - 1)) == 0,
        "DECODE_TABLE_SIZE must be power of 2");

    u32 word_idx;
    u32 bit_idx;
    u32 current_word;
    u32 next_word;
    const u32 *encoded;
    const HuffmanDecodeEntry *decode_table;

    __device__ __forceinline__ void init(const u32 *enc,
                                         const HuffmanDecodeEntry *table,
                                         u32 start_bit_offset)
    {
        this->encoded = enc;
        this->decode_table = table;
        this->word_idx = start_bit_offset >> WORD_SHIFT;
        this->bit_idx = start_bit_offset & (BITS_PER_WORD - 1);
        this->current_word = __ldg(&enc[this->word_idx]);
        this->next_word = __ldg(&enc[this->word_idx + 1]);
    }

    __device__ __forceinline__ void advance_word()
    {
        this->word_idx++;
        this->current_word = this->next_word;
        this->next_word = __ldg(&this->encoded[this->word_idx + 1]);
    }

    __device__ __forceinline__ void consume_bits(u8 len)
    {
        this->bit_idx += len;

        if (this->bit_idx >= BITS_PER_WORD) {
            this->bit_idx -= BITS_PER_WORD;

            advance_word();
        }
    }

    __device__ __forceinline__ HuffmanDecodeEntry decode()
    {
        u32 bits = __funnelshift_r(this->current_word, this->next_word, this->bit_idx);
        auto entry = this->decode_table[bits & CODE_MASK];

        consume_bits(entry.length);

        return entry;
    }
};
