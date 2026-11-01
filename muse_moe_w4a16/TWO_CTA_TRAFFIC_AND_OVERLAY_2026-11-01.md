# Two-CTA traffic risk and an FT64 storage alternative

The previous goal turn made concrete progress: broader leader-source review,
22 parser preflight cases and documentation pushed in7bb2767. This turn keeps
working on MoE while OKC maintenance prevents the first GPU check. No agents,
scored changes or new GPU jobs. The immutable V41 diagnostic remains first.

## What the saved routes establish

Two independent saved diagnostics, f8208a2105ab4940 and b22622ebb61946df,
contain exactly matching route/job metadata for M32 and M256, seed3101/pad0.
Their clocks are not used here. Reconstructing the exact fat-first/expert-order
job partition reproduces the saved original job lists. Counting every native
copy from the current prefix/cut policy gives:

| Fixture | Parent jobs → FT32 jobs | Weight TMA MiB, parent → FT32 | Activation TMA MiB, parent → FT32 |
|---|---:|---:|---:|
| M32,42 local routes |18 →18|486 →486|54 →108|
| M256,324 local routes |26 →28|702 →756|159 →318|

The M256 experts with40 and35 rows create the two extra FT32 jobs. The
7.69% extra weight requests and doubled X requests are **logical payloads**.
They are not measured HBM traffic: simultaneous chunks may share L2/in-flight
requests. The unique expert weight payload is unchanged. M128 and seed7907
route metadata are absent from these traces; no figures are invented for them.

The smaller output units also change split counts. Phase-A/B split units are
120/120→216/216 at M32 and115/123→230/231 at M256. Each new unit has two
warpgroups rather than four, so split-unit counts alone do not establish
more fixup bytes, more time or a causal bottleneck. The arithmetic and exact
scope are in `TWO_CTA_TRAFFIC_2026-11-01.json`.

A local dependency graph covers X-copy issue, first MMA group and X-slot
retirement for all792 CTA sequences across both fixtures/topologies. The
current two/three-slot rings are acyclic in this limited graph. A one-slot
ring with a two-stage segment has a cycle: the next copy waits for the old
slot, whose final group only retires behind the next stage's first group.
A drained one-stage segment does not have that cycle. This rules out simply
shrinking the X ring to one slot while preserving this retirement schedule.
It is not a proof of global readiness, split-finisher ordering, register
lifetime or actual GPU concurrency. Original H200 gates remain mandatory.

## A distinct follow-up: preserve FT64 by reusing dead routing storage

The original64-token partition avoids the extra weight jobs, but two X
slots need64KiB. Keeping the V41 metadata/three weight slots cannot fit two
blocks per SM. The new offline alternative instead:

- Retains original FT64 jobs and exact NTT1..8 WGMMA implementations.
- Uses two18KiB weight slots and two32KiB X slots, with two consumer
  warpgroups per320-thread block and264 blocks targeted.
- Places the3,072-byte expert-ID array and6,144-byte sorted-entry array
  in disjoint bytes of the future weight ring. Both are dead after routing
  and gather; weight copies begin after the full producer/consumer join.
- Removes the scale LUT field only in the fast-PRMT module. Existing fallback
  modules and canonical state remain unchanged.
- Adds a generic/async proxy fence from every consumer before the full join,
  before any producer TMA overwrite of those generic routing accesses.

The fence follows the generic/async handoff requirement described in
[NVIDIA's asynchronous-copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html).
The architecture resource limits are documented in the
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy).
Actual linked occupancy must still be queried; arithmetic is insufficient.

The native alternative is retained unchanged in
`TWO_CTA_WIDE_NATIVE_2026-11-01.cu.gz` and its diff. SHA256:
`c2880c4f24a8d1437a9513fe35852598fd43aab76f8c31978bd1e733f1c2f999`.
It is not a submission or a measured improvement. Global scratch offsets,
original packed weight pointers, precision and per-route semantics are kept.
The change still doubles logical X requests. Two weight slots can also
expose more copy latency; saving logical weight bytes is not a speedup proof.

## Validation and compiler limitations

An independent byte-addressed CPU routing/scatter model covers132 fixtures:
11 token counts through384, three local-expert offsets, random/hot/empty/
basis routes with distinct eight IDs per row. It compares the original16-warp
and new8-warp routing against a stable per-expert oracle. IDs remain intact
while entries are written, and metadata is disjoint from the overlaid range.
All pass. The resource and handoff model is in
`TWO_CTA_WIDE_PREFLIGHT_2026-11-01.json`.

A small **layout-only** NVCC13/sm90a compile validates the exact native Smem
definition:10,976-byte structure,11,008-byte aligned metadata and113,408-byte
dynamic allocation. Its kernel does not implement MoE; its register count
does not describe our candidate. The raw response is retained in
`TWO_CTA_WIDE_LAYOUT_CODEGEN_2026-11-01.json.gz`.

Five full-body resource requests fail inside Compiler Explorer serialization
with `RangeError: Invalid string length` at cachePut. They cover BIG1 normal,
BIG0 skipAsm, BIG1 compact display, BIG1 host assembly and a device-resource
attempt with an intentionally unresolved host symbol. The last request was
designed not to produce a runnable executable; it also yielded no device
resource diagnostics. These are inconclusive service failures, not CUDA
compilation failures or passing resource checks. All requests/responses are
retained in `TWO_CTA_WIDE_CODEGEN_ANALYSIS_2026-11-01.json` and its archives.

Selected current upstream Compiler Explorer NVCC/base compiler source was
read to understand device extraction and skipAsm/output handling. The
deployed revision is unknown; changing display/output mode did not repair
this service response. Stop these output-format retries. Full linked H200
resources, zero local allocation, two-block occupancy, numeric/nearby/fresh/
empty/FP64 checks and matched timings remain unresolved. No score is claimed.

The original V41 source/gates remain unchanged and eligible as the first
experiment when OKC resumes. The FT64 overlay is retained as a follow-up,
not a reason to replace an untested source silently or loosen validation.
