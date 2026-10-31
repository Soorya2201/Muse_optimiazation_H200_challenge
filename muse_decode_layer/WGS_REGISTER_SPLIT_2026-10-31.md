# Decoder WGS register and commit-group experiment

The canonical decoder remains the compliant alygg77 `6cae49a` source,
SHA-256 `8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.
This experiment targets the expert stream; it preserves FP32 router inputs
at T<=8 and all activation/weight precisions. Compiler diagnostics below
are separate from GPU timings and leaderboard scores.

## Corrected compiler evidence

`tools/probe_decoder_wgs_codegen.py` extracts the already public native
source and replaces the torch host wrapper with a minimal launch anchor.
The original two-instantiation experiment timed out. Subsequent full-body
requests hit Compiler Explorer's `RangeError: Invalid string length`.
One corrected-target request returned HTTP502 before transport archival
was implemented: its compiler status is UNKNOWN, not an inferred failure
or success. All other requests/responses are preserved, with exact source
hashes and options, in the dated `WGS_CODEGEN_*.json.gz` files. The summary
and missing-response provenance are in `WGS_CODEGEN_SUMMARY_2026-10-31.json`.

The reduced diagnostic compiles only BIG=true and one NTT switch arm,
with internal helper linkage to prevent unused external helper retention.
This is never a runnable general submission. `-arch=sm_90a` unexpectedly
also requested generic compute_90 PTX, where architecture-specific WGMMA
fails. The probe now defaults to the judge extension's explicit
`-gencode=arch=compute_90a,code=sm_90a`. The actual scored source already
used the correct flags. This was a diagnostic-tool fix, not a kernel gain.
See the [NVCC target documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch).
Compiler Explorer adds `--keep-device-functions` and extracts device code
separately: [its NVCC implementation](https://github.com/compiler-explorer/compiler-explorer/blob/ca34aede4d7cbf8233182579c062000ff0f63d22/lib/compilers/nvcc.ts).

For NTT8, base and explicit-scale-reuse arms both compile with **96
registers, zero stack frame, zero spill stores/loads**. PTX scale-constant
references fall from 80 to 32. In the displayed SASS, both have 16 matching
scale-construction IMAD sites; explicit source hoisting therefore has not
established a new machine-code scale reduction. Their instruction text
and displayed opcode fields differ: these are not identical binaries.

**SASS is truncated at 5,000 lines.** Both views happen to contain 4,767
displayed instructions; that equality is a clipping artifact, not a whole
kernel instruction-count result. Prefix hashes are not complete binary
fingerprints. No conclusion covers other NTT arms, complete switch bodies
or runtime performance. Do not spend a blind GPU check on scale hoisting
from PTX counts alone.

## New mechanism and resource calculation

The parent launches 512 consumer threads plus two producer warps. Its
GS4 schedule was previously closed because ptxas serialized narrow-N
WGMMA pipelines at the shared 96-register budget. This experiment changes
that resource mechanism before considering GS4 again.

Register redistribution alone is an existing negative control, not a new
invention: MoE v21/check83c410ffc02141f1 previously tied its parent at
2.324033x. That version moved range computation onto consumers. This
decoder control keeps the original x-producer range computation and adds
GS4 as the distinct scheduling combination; no claim of inventing the
32/112 split or of overcoming the prior negative is justified.

Pad the producer role to a full 128-thread warpgroup. All producer warps
execute `setmaxnreg.dec` to 32 registers before splitting into weight,
activation and idle roles. All four consumer warpgroups execute
`setmaxnreg.inc` to 112. The 64 idle threads participate in the initial
WS_JOIN barrier before returning. The x producer's barrier7 count remains
512+32, matching its original participants. Later consumer barriers still
count only 512. Shared-memory buffers, lossless KPERM3 mapping, input
reads and accumulation order are preserved.

Requested role budget: 512*112+128*32 = 61,440 registers, equal to
640*96 and below Hopper's 65,536-register SM capacity. This arithmetic is
a feasibility check; the runtime must also validate the compiler's full
allocation and role synchronization. Primary references:

- [PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg):
  the instruction must be uniform across every warp in its warpgroup;
  allocation can wait for the CTA register pool. Applying it to just the
  parent's two producer warps would not satisfy that full-group design.
- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy):
  register/shared-memory capacities and warp specialization with TMA.
- [NVIDIA CUTLASS register primitives](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/reg_reconfig.h):
  production examples of the ISA allocation/release operations. No CUTLASS
  implementation is imported into this experiment.

Reduced compiler results (BIG=true, internal helpers, explicit sm90a):

|Arm|NTT|Static register metadata|Frame / spill stores / spill loads|
|---|---:|---:|---:|
|Parent GS2|8|96|0 / 0 / 0 bytes|
|Explicit scale reuse GS2|8|96|0 / 0 / 0 bytes|
|Register split GS2|8|96|0 / 0 / 0 bytes|
|Register split GS4|8|96|0 / 0 / 0 bytes|
|Register split GS4|4|96|16 / 20 / 16 bytes|
|Register split GS4|2|96|0 / 0 / 0 bytes|

No C751x serialization warning is reported in these reduced successful
compiles. NTT4 spilling is counterevidence: more registers and fewer groups
do not guarantee a faster kernel. Static metadata of96 does not mean
consumers ignored their dynamic112 request; the PTX contains both role
instructions and visible SASS contains a CTA-pool112 allocation.

## Bound and GPU protocol

`WGS_TARGET_BOUND_2026-10-31.json` corrects the earlier broad expert model:
only decode32/verify16/verify32 use this main WGS path. In the scored
eff0b59 same-job phase model, reducing their expert durations by5%,10%,20%
models whole-board multipliers1.013393,1.027344,1.057086. Removing the
target entirely models1.456537. These are optimistic duration-subtraction
estimates: PDL waits and overlap prevent treating them as critical-path
proof. A small local instruction reduction alone cannot promise rank1.

V20/check `98ca0484be124a39` is a prepare-only diagnostic. The parent file
is its exact prefix and `kernel_fn` is unchanged. Compare parent GS2,
register split GS2, and register split GS4. GS4 flush index1 matches GS2
index3 after eight issued k16 operations. This isolates scheduling from
deferred-flush progress. Full compiler bodies, not fixed-NTT reductions,
are built on the H200.

The initial scope is M128/256 with the parent's main build. M32 uses its
separate B4FIN policy and is excluded from promotion/model assumptions.
For these two loads alone, the older same-job planning model gives1.019615
for10% less targeted expert duration and1.040866 for20%. Approximately
14.99% less expert duration is needed to model a3% board gain;41.47% would
be needed to close the captured1.0933x leader gap using only these loads.
Those numbers are goals for measurement, not forecasts.
Three nearby fixtures (40/96/192) validate outputs. Sixteen timing groups
cover two seeds, four retained memory pads, two token counts and nine
alternating cold-L2 graph rounds. Two parent graphs bound A/A variation.
Fresh activations, expert IDs and routing weights must change outputs and
retain agreement. Gates: relativeL2<=.002, maximum normalized row<=.01.
Every raw integer-nanosecond round is compressed into a final log report;
CPU report stress test is8,584 characters, below12K. No diagnostic fixture,
output or graph enters returned state. Original scratch is zeroed afterward.

Eight nonempty parent/GS2/GS4 policies are enumerated for M128/256. A
candidate must exceed1.02 in every seed/pad group and model at least1.03
on the board before a separate actual-dispatch check is considered. The
model uses this check's own parent benchmark/profile rows. Synthetic
routing and isolated timing still require whole-layer validation.

## H200 result: no eligible improvement

Check **98ca0484be124a39**, cluster **sfo3**, passed13 challenge tests and
all12 benchmark cells. Parent timed dispatch scored2.195102979x,
10,304.892us,95%CI2.162176434–2.228029524. Prepare187.956s, independent
probe84.367s. The score measures the incumbent, not either new variant.
The complete3592-character diagnostic survived both local and remote
logs. All12 profiles, all available logs and all16 raw timing groups were
read. Receipt and decoded data:
`run_receipts/98ca0484be124a39.json`,
`WGS_REGISTER_SPLIT_98ca0484be124a39.json`.

|Tokens|Arm|Minimum / median / maximum paired speedup|
|---|---|---:|
|128|Split GS2|0.997318 / 0.998689 / 0.999690|
|128|Split GS4|0.990645 / 0.992761 / 0.998470|
|256|Split GS2|0.996112 / 0.998732 / 1.004021|
|256|Split GS4|0.994567 / 0.997380 / 1.003779|

A/A median offset is at most0.4094%. All16 timing fixtures have zero
parent-relative error on initial and changed-input replays. Nearby
40/96/192 fixtures pass; the largest tiny fresh-replay difference is
2.438e-8, including parent-versus-parent variation. Each timing fixture
touches all32 local experts: local route counts432/448 atM128 and849/885
atM256. This independently generated routing is not a reproduction of
the judge distribution or a claim about every nearby routing pattern.

The actual full builds report96 static registers and640 maximum threads
for both BIG instantiations. GS2 has0 local bytes; **GS4 has32 local bytes**.
The reduced NTT8 no-spill result did not generalize to the complete switch.
The larger commit group still creates register pressure in the full build.
Attribution of the small runtime differences to spills alone is unproven;
there are no new hardware-counter measurements. Redistribution does not
change the known unit-end release/readiness protocol.

**All eight changed policies are rejected.** None clears the2% component
gate; the best worst/median/best-bound board estimates across policies are
0.999765 / 0.999885 / 1.000389. Even the optimistic bound is only+0.039%,
far below the3% planning gate. These models subtract same-job profiled
durations; they are not actual joint-dispatch measurements.

No canonical change or scored submission. Close this GS2/GS4 register
redistribution combination for the tested main path. Do not pursue a blind
register-count/group-size sweep or rerun the unchanged source for a better
point score. A future expert-stream redesign must address an independently
measured bottleneck beyond the already tested register allocation/grouping.
