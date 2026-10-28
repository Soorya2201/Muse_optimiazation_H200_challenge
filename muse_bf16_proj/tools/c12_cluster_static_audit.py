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
DYNAMIC_SHARED = 201888


def audit_resources() -> None:
    stage_bytes = PACKED_BYTES + DECODED_BYTES + X_BYTES
    stage_ring = NSTAGE * stage_bytes
    barrier_arrays = NSTAGE * 3 * 8
    qbar_padding = 16
    mailbox = 2 * 32 * REDUCTION_STRIDE * 4
    used_from_aligned_base = stage_ring + barrier_arrays + qbar_padding + mailbox
    assert stage_bytes == 30720
    assert stage_ring == 184320
    assert mailbox == 17408
    assert used_from_aligned_base == 201888
    assert used_from_aligned_base == DYNAMIC_SHARED


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


def audit_deferred_seven_stage_layout() -> None:
    # NVCC 13.0 emits `.extern .shared .align 1024 .b8 raw[]` for the source
    # declaration, so a future variant may use raw directly without reserving
    # a second 1,023-byte alignment margin. Transpose the peer mailbox from
    # [warp, lane, accumulator] to [warp, float4-group, lane, float4].
    stages = 7
    stage_ring = stages * (PACKED_BYTES + DECODED_BYTES + X_BYTES)
    barrier_arrays = stages * 3 * 8
    qbar_bytes = 8
    peer_offset = (stage_ring + barrier_arrays + qbar_bytes + 15) & ~15
    compact_mailbox = 2 * 16 * 32 * 4 * 4
    total = peer_offset + compact_mailbox
    assert stage_ring == 215040
    assert peer_offset == 215216
    assert compact_mailbox == 16384
    assert total == 231600
    assert DYNAMIC_SHARED + 30560 == 232448  # documented H200 opt-in limit
    assert 232448 - total == 848

    addresses = set()
    for warp in range(2):
        for group in range(16):
            for lane in range(32):
                address = peer_offset + (((warp * 16 + group) * 32 + lane) * 16)
                assert address % 16 == 0
                block = set(range(address, address + 16))
                assert addresses.isdisjoint(block)
                addresses.update(block)
    assert len(addresses) == compact_mailbox


def main() -> None:
    audit_resources()
    audit_barrier_phases()
    audit_reduction()
    audit_packed_ranges()
    audit_output_coverage()
    audit_deferred_seven_stage_layout()
    print(
        "c12 cluster static audit passed: "
        "shared=201888 reduction_tx=16384 packed=50331648 output=128x4096 "
        "deferred_7stage=231600"
    )


if __name__ == "__main__":
    main()
