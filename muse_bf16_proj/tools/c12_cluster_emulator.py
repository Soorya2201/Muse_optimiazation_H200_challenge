#!/usr/bin/env python3
"""CPU proof for the proposed 64x64 C12 decode-to-shared tile.

The H200 kernel will bulk-copy a 6,144-byte packed tile, decode it into the
same 128-byte-swizzled BF16 layout produced by TMA, and apply at most eight
32-bit exact patches before exposing the tile to shared/shared WGMMA.

This tool has no CUDA dependency.  It proves the byte layout, patch encoding,
and swizzle for every possible BF16 bit pattern plus randomized normal-like
tiles.  It is not a performance model.
"""

from __future__ import annotations

import argparse
import random
import struct
from collections.abc import Iterable, Sequence


ROWS = 64
COLS = 64
VALUES = ROWS * COLS
PACKED_BYTES = VALUES * 3 // 2
PATCH_CAPACITY = 8


def choose_base(values: Sequence[int]) -> int:
    """Choose the best matrix/tile-wide window of eight upper exponent values."""
    hist = [0] * 128
    for value in values:
        hist[(value >> 8) & 0x7F] += 1
    return max(range(121), key=lambda base: sum(hist[base : base + 8]))


def encode_patch(index: int, xor_high: int) -> int:
    if not 0 <= index < 4096 or not 0 <= xor_high < 256:
        raise ValueError("patch field out of range")
    return index | (xor_high << 12)


def decode_patch(word: int) -> tuple[int, int]:
    return word & 0xFFF, (word >> 12) & 0xFF


def pack_tile(values: Sequence[int], base: int) -> tuple[bytes, list[int]]:
    """Pack logical row-major BF16 bits as 8 low bytes + 4 code bytes per group."""
    if len(values) != VALUES or not 0 <= base <= 120:
        raise ValueError("expected a 64x64 tile and a legal C12 base")
    packed = bytearray()
    patches: list[int] = []
    for start in range(0, VALUES, 8):
        group = values[start : start + 8]
        lows: list[int] = []
        codes: list[int] = []
        for offset, value in enumerate(group):
            value &= 0xFFFF
            low, high = value & 0xFF, value >> 8
            sign, exponent_high = high >> 7, high & 0x7F
            window = min(7, max(0, exponent_high - base))
            decoded_high = (sign << 7) | (base + window)
            lows.append(low)
            codes.append((sign << 3) | window)
            xor_high = high ^ decoded_high
            if xor_high:
                patches.append(encode_patch(start + offset, xor_high))
        packed.extend(lows)
        packed.extend(codes[i] | (codes[i + 4] << 4) for i in range(4))
    if len(packed) != PACKED_BYTES:
        raise AssertionError(f"bad packed size: {len(packed)}")
    return bytes(packed), patches


def decode_logical(packed: bytes, base: int, patches: Iterable[int]) -> list[int]:
    """Decode to logical row-major BF16 bits and apply exact high-byte XOR patches."""
    if len(packed) != PACKED_BYTES:
        raise ValueError("bad packed tile size")
    result = [0] * VALUES
    source = 0
    for start in range(0, VALUES, 8):
        lows = packed[source : source + 8]
        nibble_pairs = packed[source + 8 : source + 12]
        source += 12
        codes = [pair & 0xF for pair in nibble_pairs]
        codes.extend(pair >> 4 for pair in nibble_pairs)
        for offset, (low, code) in enumerate(zip(lows, codes)):
            high = ((code >> 3) << 7) | (base + (code & 7))
            result[start + offset] = (high << 8) | low
    for word in patches:
        index, xor_high = decode_patch(word)
        result[index] ^= xor_high << 8
    return result


def swizzle_128b(values: Sequence[int]) -> list[int]:
    """Match swizzle_tiles(): physical chunk = logical chunk XOR (row mod 8)."""
    if len(values) != VALUES:
        raise ValueError("expected one 64x64 tile")
    physical = [0] * VALUES
    for row in range(ROWS):
        for col in range(COLS):
            chunk, lane = divmod(col, 8)
            physical[row * COLS + ((chunk ^ (row & 7)) * 8 + lane)] = values[row * COLS + col]
    return physical


def unswizzle_128b(physical: Sequence[int]) -> list[int]:
    # XOR is its own inverse.
    return swizzle_128b(physical)


def decode_to_swizzled(packed: bytes, base: int, patches: Iterable[int]) -> list[int]:
    return swizzle_128b(decode_logical(packed, base, patches))


def float_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def assert_roundtrip(values: Sequence[int], base: int | None = None) -> int:
    chosen = choose_base(values) if base is None else base
    packed, patches = pack_tile(values, chosen)
    decoded = decode_logical(packed, chosen, patches)
    if decoded != [value & 0xFFFF for value in values]:
        index = next(i for i, (a, b) in enumerate(zip(decoded, values)) if a != (b & 0xFFFF))
        raise AssertionError(f"logical mismatch at {index}: {decoded[index]:04x} != {values[index]:04x}")
    swizzled = decode_to_swizzled(packed, chosen, patches)
    if unswizzle_128b(swizzled) != decoded:
        raise AssertionError("128-byte swizzle roundtrip failed")
    return len(patches)


def run_tests(seed: int, random_tiles: int) -> None:
    # Exhaust every BF16 bit pattern in sixteen complete tiles.
    for tile in range(16):
        values = [tile * VALUES + i for i in range(VALUES)]
        assert_roundtrip(values)

    # Prove every patch field independently, including maximum values.
    for index in range(VALUES):
        for xor_high in (0, 1, 0x7F, 0x80, 0xFF):
            if decode_patch(encode_patch(index, xor_high)) != (index, xor_high):
                raise AssertionError("patch encoding failed")

    rng = random.Random(seed)
    calibration = [float_to_bf16(rng.gauss(0.0, 1.0)) for _ in range(64 * VALUES)]
    global_base = choose_base(calibration)
    patch_counts = []
    over_capacity = 0
    for _ in range(random_tiles):
        values = [float_to_bf16(rng.gauss(0.0, 1.0)) for _ in range(VALUES)]
        count = assert_roundtrip(values, base=global_base)
        patch_counts.append(count)
        over_capacity += count > PATCH_CAPACITY

    # Force both signs, zeros, denormals, infinities, NaNs, and far-window exponents.
    adversarial = [((i * 7919) ^ (i << 7) ^ 0xA5A5) & 0xFFFF for i in range(VALUES)]
    assert_roundtrip(adversarial, base=60)

    print(f"exhaustive_bf16_patterns=65536 pass")
    print(f"patch_field_cases={VALUES * 5} pass")
    print(f"random_tiles={random_tiles} seed={seed} global_base={global_base} pass")
    if patch_counts:
        ordered = sorted(patch_counts)
        p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
        print(
            f"normal_like_patches min={ordered[0]} median={ordered[len(ordered)//2]} "
            f"p99={p99} max={ordered[-1]} over_capacity_{PATCH_CAPACITY}={over_capacity}"
        )
    print(f"packed_bytes={PACKED_BYTES} swizzled_bytes={VALUES * 2} pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5151)
    parser.add_argument("--random-tiles", type=int, default=2048)
    args = parser.parse_args()
    run_tests(args.seed, args.random_tiles)


if __name__ == "__main__":
    main()
