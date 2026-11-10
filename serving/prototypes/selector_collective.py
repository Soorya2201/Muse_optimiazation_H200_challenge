"""Lossless FP32 value / 31-bit vocabulary ID transport for DFlash2 top-k.

Development source mirrored in candidate09. The caller admits a globally
bounded vocabulary, not a rank-dependent fallback decision. No quantization.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(values, ids, packed, size: tl.constexpr, columns: tl.constexpr,
          vs0: tl.constexpr, vs1: tl.constexpr, is0: tl.constexpr,
          is1: tl.constexpr, start: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, col = pos//columns, pos % columns
    value = tl.load(values+row*vs0+col*vs1, pos < size, other=0).to(tl.float32)
    index = tl.load(ids+row*is0+col*is1, pos < size, other=0).to(tl.int64)
    bits = value.to(tl.uint32, bitcast=True).to(tl.uint64)
    pair = ((index+start).to(tl.uint64) << 32) | bits
    tl.store(packed+pos, pair.to(tl.int64), pos < size)


@triton.jit
def _unpack(packed, values, ids, size: tl.constexpr, columns: tl.constexpr,
            ps0: tl.constexpr, ps1: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, col = pos//columns, pos % columns
    pair = tl.load(packed+row*ps0+col*ps1, pos < size, other=0).to(tl.uint64)
    bits = pair.to(tl.uint32)
    tl.store(values+pos, bits.to(tl.float32, bitcast=True), pos < size)
    tl.store(ids+pos, (pair >> 32).to(tl.int64), pos < size)


def pack_selector_pairs(values, ids, vocab_start):
    assert values.ndim == ids.ndim == 2 and values.shape == ids.shape
    packed = torch.empty(values.shape, dtype=torch.int64, device=values.device)
    n = values.numel()
    if n:
        _pack[(triton.cdiv(n, 256),)](values, ids, packed, n, values.shape[1],
            *values.stride(), *ids.stride(), vocab_start, 256)
    return packed


def unpack_selector_pairs(packed):
    assert packed.ndim == 2
    values = torch.empty(packed.shape, dtype=torch.float32, device=packed.device)
    ids = torch.empty(packed.shape, dtype=torch.int64, device=packed.device)
    n = packed.numel()
    if n:
        _unpack[(triton.cdiv(n, 256),)](packed, values, ids, n, packed.shape[1],
            *packed.stride(), 256)
    return values, ids
