# Hopper inline compression: allocation-only BF16 experiment

The previous attention cycle completed a real H200 comparison and rejected
both integer-exp variants, pushed eedd990. This cycle works only on BF16.
Canonical source remains
`290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90` and
scored1.535843059x. No improvement is established by the preparation below.

## Why this differs from the closed software-decoder work

The earlier sign-layout and global-normalization SK2 experiments did not
qualify despite fewer decoder instructions. A broader review found NWC
and ZipServ; their code and transfer limits are recorded separately in
`LOSSLESS_SOURCE_REVIEW_LATE_2026-10-31.md`. Rather than adding another
software decoder, this experiment changes weight allocation properties.

NVIDIA's [Hopper inline-compression section](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression)
describes an optional hardware feature that can reduce memory traffic
without changing load instructions or mathematical values. It does not
reduce the reserved memory footprint and may choose no compression for a
particular pattern. Support on the actual judge and benefit for BF16 weight
bytes are unknown. We claim no compression ratio or physical-traffic result.

The [CUDA VMM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory)
requires querying `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`, opting
in with `CU_MEM_ALLOCATION_COMP_GENERIC`, and verifying the allocation's
returned properties. An accepted request is not proof that actual data
compressed. The official BSD-3-Clause
[cudaCompressibleMemory sample](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/3_CUDA_Features/cudaCompressibleMemory)
is pinned with its license; our wrapper is independently implemented.

## Candidate and ownership design

V48 SHA-256:
`ac071fac6362f2d4c93e176a93550fc7e592a09e863d50a8fc05dc150e35544b`.
The canonical file is an exact prefix; no new timed `kernel_fn` is defined.
The original prepare runs once; an independent diagnostic then returns
the same state. No experimental weight, allocation, activation or output
is retained in returned state. This is not a changed leaderboard submission.

`tools/bf16_ilc_allocator.py` supplies a C++ shared owner with two physical
handles (ordinary and compressible) and one reserved virtual address.
Allocation size is a common multiple of both queried granularities.
Every requested handle's actual compression flag must be0/1 respectively.
Remapping synchronizes the context first, unmaps, maps and sets device
access; it happens only outside synthetic timed graphs. Stored-context
guards support API calls on other host threads. Tensor views retain the
shared owner through their deleter. Cleanup synchronizes before unmapping
and releasing handles. Actual native binding/lifetime behavior still
requires GPU validation; CPU source checks alone do not prove it.

An SM vector-copy kernel preserves every byte, including its guarded tail.
Six GPU byte checks are planned: every65536BF16 bit pattern (as bytes,
including signed zeros/NaNs) and1/15/16/17/65537-byte sizes. Each alternates
normal/compressible mapping and payloads; no floating computation consumes
the all-pattern fixture.

The eight target cells are the existing four BF16 layers at M128/256.
Their exact selected readonly weight paths are recorded in `_ILC_SPECS`:
qbig, raw QKV, packed O128/O256, cuBLASLt pooler128/256 and unpooler256,
and transposed unpooler128. Missing pinned plans or a different dispatch
reject the diagnostic. Shallow dictionary replacement copies both dictionary
levels and preserves the original weight/state. No counter/workspace is
remapped. A perturbed weight copy must change the output, proving that the
selected path is actually used. Four alternating payload/mode replays then
check that the same captured graph, including TMA users, sees remapped data.
Original weight bits are never changed.

The driver links by its Linux SONAME `libcuda.so.1`, avoiding reliance on a
development-only `libcuda.so` symlink. All native code is newly compiled
with `load_inline`; no external process beyond its normal compiler is used.

## Local findings before GPU use

The first V44 draft changed fresh activations by-0.75, which would destroy
the power-of-two one-hot property required by the extra0.1% check. It was
corrected to-0.5 before any GPU job. V46's64-group stress payload was11860
characters, above its10000-character gate. No samples had been run: the
protocol was reduced before launch to32groups, retaining every target cell
and both seeds, with two0/7MiB allocation layouts. V44–V47 drafts remain
archived, never submitted. A promising result requires a broader four-layout
repeat, fresh-thread/deepcopy validation and real changed-dispatch checks
before promotion. We do not claim those checks have already passed.

The final CPU preflight covers520copy geometries,72granularity/rounding
cases,13balanced schedule shifts,32complete report groups,1024exact event
float32 values and608exact error metrics. Stress payload6108chars.
Event values use u16nanoseconds only when converting back reproduces the
original float32 bits exactly; otherwise u32bits are stored. Agreement
metrics retain native FP32 bits and FP64-reference metrics retain doubles.
An explicit sub65us noninteger-nanosecond case checks the fallback. No
quantization, sample removal or relaxed comparison is introduced.

## Opportunity and protocol declared before launch

From scored receipt a3d4758c1e174aba, eliminating all eight target cells
would give a loose2.228828x board proxy. A uniform target-cell cut9.328446%
models3%board gain;5/10/15%cuts model1.015638/1.032302/1.050117x.
These are conditional time scenarios, not predicted memory compression
ratios. Avoiding software decode offers a distinct way to test this gap,
but hardware may refuse the feature or find these weights incompressible.

One routine check, if launched, is BF16 check5/8 on Oct31 UTC. All scored
submits remain0; the owner's reserve is untouched. The challenge times the
unchanged parent. An unsupported device/allocation ends the independent
experiment explicitly; it is not a measured slowdown or a candidate pass.

If support and allocation flags pass,32groups cover eight cells, seeds3209/
7909 and0/7MiB retained pads. Four arms: actual-parent A0, ordinary VMM N,
compressible VMM C, duplicate actual-parent A1. N/C use **the same graph and
virtual address**, differing only in the physical allocation flag. Before
each measured replay, every arm receives an untimed graph warm-up followed
by read-only256MiB cache eviction. Start/stop event nodes are inside the
graph; no overhead constant is subtracted. Eight Williams-square rounds
balance positions and adjacent pairs. Store all1024raw event float32 values.

Initial and changed-input comparisons require finite outputs, FP64 global
<1%, row<5%, power-of-two one-hot<0.1%, and actual-parent agreement global
<0.02%/row<0.2%. Alternating weight payloads must also satisfy agreement.
Every selected component must beat both actual and matched-normal controls
by>2% in every group; A/A offsets<2%, actual/matched offsets<3%. All255
nonempty fixed eight-cell combinations are modeled; the worst group must
give>=3%board proxy. No dropped failing group or relaxed gate. Candidate
promotion additionally needs the broader repeat/ownership checks noted above.

## H200 result

Not launched when this protocol was written. Append terminal evidence,
every available profile/log, capability/allocation status and decision.
