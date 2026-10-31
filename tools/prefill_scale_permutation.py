"""CPU model for a prospective KPERM-style prefill scale-sharing layout.

This is not a runnable GPU kernel. Both activation streams would require
matching timed packing, and floating accumulation order changes.
"""


def logical_k(packed_k: int) -> int:
    """Original K index consumed at a prospective permuted K position."""
    assert packed_k >= 0
    base, k = divmod(packed_k, 64)
    return base * 64 + ((k & 1) | ((k & 8) >> 2) | ((k & 48) >> 2) | ((k & 6) << 3))


def packed_k(logical: int) -> int:
    assert logical >= 0
    base, k = divmod(logical, 64)
    return base * 64 + ((k & 1) | ((k & 2) << 2) | ((k & 12) << 2) | ((k & 48) >> 3))


def e2m1_units(code: int) -> int:
    """Exact integer in units of one half, including signed zero as zero."""
    value = (0, 1, 2, 3, 4, 6, 8, 12)[code & 7]
    return -value if code & 8 else value


def e4m3_units(byte: int) -> int:
    """Exact finite E4M3FN integer in units of 1/512; NaNs are excluded."""
    assert 0 <= byte < 256 and (byte & 127) != 127
    exp, mant = (byte >> 3) & 15, byte & 7
    value = mant if exp == 0 else (8 + mant) << (exp - 1)
    return -value if byte & 128 else value
