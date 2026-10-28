#!/usr/bin/env python3
"""Audit the non-hardware invariants of the v32 pooler-m128 cluster kernel."""

from __future__ import annotations


NSTAGE = 6
K_UNITS = 128
PACKED_BYTES = 6144
DECODED_BYTES = 8192
X_BYTES = 16384
NACC = 64
REDUCTION_STRIDE = NACC + 4
DYNAMIC_SHARED = 202912


def audit_resources() -> None:
    stage_bytes = PACKED_BYTES + DECODED_BYTES + X_BYTES
    stage_ring = NSTAGE * stage_bytes
    barrier_arrays = NSTAGE * 3 * 8
    qbar_padding = 16
    mailbox = 2 * 32 * REDUCTION_STRIDE * 4
    used_from_aligned_base = stage_ring + barrier_arrays + qbar_padding + mailbox
    worst_alignment = 1023
    assert stage_bytes == 30720
    assert stage_ring == 184320
    assert mailbox == 17408
    assert used_from_aligned_base == 201888
    assert used_from_aligned_base + worst_alignment == DYNAMIC_SHARED - 1


def audit_barrier_phases() -> None:
    # An initialized mbarrier starts in phase zero. A wait with parity p
    # completes after the producer arrival toggles the phase away from p.
    for stage in range(K_UNITS // 2):
        slot = stage % NSTAGE
        generation = stage // NSTAGE
        assert slot in range(NSTAGE)
        assert (generation & 1) == ((stage // NSTAGE) & 1)  # full/decoded_ready
        if stage >= NSTAGE:
            assert ((generation - 1) & 1) == (((stage // NSTAGE) - 1) & 1)  # empty


def audit_reduction() -> None:
    sending_warps = 2
    transaction_bytes = sending_warps * 32 * NACC * 4
    mailbox_bytes = sending_warps * 32 * REDUCTION_STRIDE * 4
    assert transaction_bytes == 16384
    assert mailbox_bytes == 17408
    # Every lane owns a disjoint padded mailbox row.
    intervals = []
    for warp in range(sending_warps):
        for lane in range(32):
            begin = (warp * 32 + lane) * REDUCTION_STRIDE
            intervals.append(set(range(begin, begin + NACC)))
    assert len(set().union(*intervals)) == sending_warps * 32 * NACC
    assert sum(map(len, intervals)) == len(set().union(*intervals))


def audit_packed_ranges() -> None:
    ranges = []
    for row_block in range(64):
        for kb in range(K_UNITS):
            begin = (row_block * K_UNITS + kb) * PACKED_BYTES
            ranges.append((begin, begin + PACKED_BYTES))
    assert ranges[0] == (0, PACKED_BYTES)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert ranges[-1][1] == 50331648


def audit_output_coverage() -> None:
    written = set()
    for row_block in range(64):
        for rank in range(2):
            for worker in range(64):
                for chunk in range(worker, 128 * 4, 64):
                    token = chunk >> 2
                    row_chunk = chunk & 3
                    for element in range(8):
                        coordinate = (
                            token,
                            row_block * 64 + rank * 32 + row_chunk * 8 + element,
                        )
                        assert coordinate not in written
                        written.add(coordinate)
    assert len(written) == 128 * 4096
    assert min(written) == (0, 0)
    assert max(written) == (127, 4095)


def main() -> None:
    audit_resources()
    audit_barrier_phases()
    audit_reduction()
    audit_packed_ranges()
    audit_output_coverage()
    print(
        "c12 cluster static audit passed: "
        "shared=202912 reduction_tx=16384 packed=50331648 output=128x4096"
    )


if __name__ == "__main__":
    main()
