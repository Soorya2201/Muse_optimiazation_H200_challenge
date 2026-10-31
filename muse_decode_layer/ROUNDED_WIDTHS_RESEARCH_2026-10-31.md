# Decoder four-width specialization experiment

The preceding turn refreshed ranks only, so it made no optimization progress.
This cycle implements a distinct native candidate and produces new CPU,
compiler and primary-source evidence. No H200 job or scored submit was run.
The compliant canonical decoder remains SHA-256
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.

## Implemented mechanism

`tools/decoder_wgs_variants.py:rounded_widths` replaces eight WGS segment
specializations with four: actual n-tile counts 3/5/6/7 issue widths 4/8/8/8.
The hypothesis is reduced instruction-cache pressure from fewer inlined
segment bodies. No current measurement establishes that pressure as an
exposed bottleneck, and fewer source templates do not prove smaller binaries.
This is distinct from the old prefill narrower-MMA/composite-tile negatives.

The implementation keeps global xA/xB allocation, packing, strides, copy
sizes, readiness counts and scheduling cost tables unchanged. The activation
producer retains all 32 lanes. After lane 0 acquires the slot release and
input readiness, a warp synchronization precedes parallel zero stores to
the added shared n-tiles. Each writer executes a proxy fence before another
warp synchronization and lane-0 copy/publication. The zero range lies after
the unchanged copied prefix and within the existing 32KiB shared slot.
Original partial-tile padding is unchanged; only newly added columns are
initialized by this edit. Partial-sum atomics and direct epilogue calls are
guarded by actual n-tile count so dummy accumulators never contaminate
partial buffers or their reset protocol. Types and valid-column K order
are unchanged. This description is not a GPU synchronization proof.

NVIDIA documents the shared matrix descriptor and generic/async proxy
ordering separately; the producer cannot publish newly written padding
without considering both. See [PTX matrix descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)
and [proxy fences](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).

The exact complete generated native source, diff and SHA are archived as
`ROUNDED_WIDTHS_NATIVE_2026-10-31.cu.gz`, `.diff`, and
`ROUNDED_WIDTHS_PREFLIGHT_2026-10-31.json`. They are research artifacts,
not a promoted or GPU-validated submission.

## CPU evidence and costs

`python3 tools/model_decoder_rounded_widths.py` verifies all 64 nonempty
job sizes: 66,560 valid vector addresses and 14,336 added zero-vector
addresses. It checks disjoint coverage, bounds, all 16 stage bases, valid
partial publication/flush index agreement, unchanged gather/epilogue/flush
text, and unchanged cost tables through the complete host wrapper. It does
not cover empty-job behavior, native arithmetic or concurrent execution.
The first model invocation failed because its total-job byte bound omitted
BF16's two bytes per element; the assertion was repaired. Native source was
unchanged by that model fix. The failed invocation consumed no OKC budget.

All 18 pinned external trxphxle route histograms were recomputed. Logical
matrix-column work expands by **0–18.18%**, with **0–21MiB** extra shared
zero writes per whole expert call. These are operation/byte counts, not
latencies. HBM activation and weight bytes and scratch allocation do not
increase. Unchanged old-width cost tables can also misbalance the new
schedule; no improved scheduling claim is made.

The existing same-job phase model assigns the main WGS path to D32/V16/V32.
A uniform 20% expert-duration cut there models 1.057086x whole decoder;
roughly 11% is needed for 3%. This is conditional phase subtraction, not a
critical-path guarantee. Arithmetic/cache/resource evidence must support
such a practical cut before allocating a test.

## Compiler evidence and decision

Both exact requests/responses are archived under dated
`WGS_CODEGEN_*_skip_asm.json.gz`; inspect
`ROUNDED_WIDTHS_CODEGEN_SUMMARY_2026-10-31.json` for complete diagnostics.
NVCC13, explicit compute_90a->sm_90a, BIG=true, internal helper linkage,
complete width switch, minimal host launch anchor:

| Arm | Returned result | Resource evidence |
|---|---|---|
| Four-width candidate | compiler success | 96 registers; 16-byte stack; 16-byte spill stores and loads |
| Unchanged parent | CE `RangeError: Invalid string length` during response serialization | no usable matched statistics |

Candidate PTX has 61,916 returned rows, no truncation marker, and 130
WGMMA sites for each of N8/N16/N32/N64. SASS is clipped at 5,000 displayed
lines. `skipAsm` does not suppress the returned device views on this service.
There is **no matched complete binary-size comparison**, no H200 resource
measurement, no numerical GPU validation and no speedup result. CE's
parent serialization error is not an NVCC failure or a terminal GPU job.

**Defer this tested four-width implementation before H200.** Extra work,
spills and missing causal cache evidence do not establish the required
gain. Do not promote it or sweep width groupings blindly. Revisit only with
a distinct spill remedy and convincing evidence of recoverable instruction
fetch time; a zero-local reduced-width compile alone would still be weak.

## Public-source follow-up

Late indexed search remains 94 paths/10 repositories, zero new blobs.
The live 20:42UTC board identifies **pjmullin3**, not former leader mpeng19,
as current decoder leader at 2.4592x. The current leader's owner-repository
API returned zero public repositories. mpeng19 returned 11; its metadata
does not reveal a new challenge submission. This is a bounded public
metadata screen, not proof about private source or alternate accounts.
See `LEADER_PUBLIC_METADATA_LATE_2026-10-31.json` and
`GLOBAL_CODE_SEARCH_LATE_2026-10-31.json`.

Seven Apache-2.0 FlashInfer files at `9112f5b` (189,522 bytes), including
LICENSE/NOTICE, are pinned and hash-verified under
`research/public/flashinfer_monomoe_9112f5b/`. None was imported or executed.
The bounded source review and proposed transfer requirements are in
`RESEARCH_LIBRARY.md`. No claim to reproduce the leader's private method.

No canonical/scored changes; ranks last verified at20:42UTC. Oct31 routine checks remain
MoE8/8, attention1/8, BF164/8, decoder1/8, prefill5/8; all submits0,
reserve untouched. No GPU job is live. Next sequential track is prefill:
read its AGENTS/TASK and current K64/PIPE3/Lt negatives before a distinct
implementation. Goal remains active and incomplete.
