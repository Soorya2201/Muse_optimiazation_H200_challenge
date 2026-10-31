# Prefill scale-sharing layout: encoding proof and integration cost

Canonical v20 remains unchanged at1.784958x, best scored job
`f62a284779b4447d`. SHA-256
`8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5`.
No new GPU job or scored submission in this research stage. The previous
decoder turn made progress through new source/measurement evidence and a
compiler screen; this turn works only on prefill.

## Mechanism

Our prefill expert consumer loads one uint2 containing four scale pairs.
Within a K64 block, step j uses the pair for that K16 region. Four threads
in each lane group share those same scales but each constructs four pairs.
The decoder's existing KPERM3 demonstrates the broader idea of permuting K
to make scale reuse follow consumer lanes. The prefill kernel has different
K64 staging, packing, normalization and GEMM1 epilogue; that code cannot be
copied over unchanged.

For a prospective adaptation, map each packed K position k within64 to:

`logical_k = (k&1) | ((k&8)>>2) | ((k&48)>>2) | ((k&6)<<3)`.

Keep the K64 block index. This is a six-bit permutation. A lane subgroup t
then consumes original scale group t across all four K16 steps: its four
per-step values cover the same original16-element group. It can load two
raw scale bytes (one per weight row), construct one BF16 scale pair and
reuse it across four MMA steps. The code-fragment bit encoding stays the
current parent's layout, and the four original scale pairs remain stored
in the same512-byte block; the lane selects a different pair.

Per128-weight-row/K64 block, storage stays4096 code bytes +512 scale bytes.
Logical consumer scale-byte requests would be2048->512, and pair
constructions per thread4->1. These are **not measured physical traffic
reductions**: the existing same-address shared loads can broadcast, and
HBM bytes are unchanged. Six fewer scale-construction instructions do not
mean six cycles saved. No instruction-count or throughput result from a
complete new native consumer exists yet.

This is a transfer/adaptation of established KPERM-style scale sharing,
not a new original codec or an established score improvement.

## Exact CPU evidence

`tools/prefill_scale_permutation.py` defines the forward/inverse index maps
and exact finite E2M1/E4M3FN integer representations.
`tools/verify_prefill_scale_permutation.py` independently reads the actual
canonical `_CODE_POS` map without importing the GPU module, then verifies:

- bijection/inverse across all4096 K positions;
- 65536 packed words, all524288 nibble/scale assignments for a128x4096 tile;
- the complete8192-row gated gate/up row permutation;
- 64 exact integer dot products at K64/128/256/4096 with random activations,
  E2M1 codes; the deterministic samples cover all16codes and all254finite
  E4M3FN scale byte patterns, with explicit coverage assertions.

The weight/scale assignment proof exercises arbitrary raw scale bytes;
the arithmetic tests exclude NaNs. Signed zero representation is retained
by nibble/byte checks; integer dot tests identify both zero signs as zero.
The exact products use integer units1/1024, avoiding a tolerance-based
CPU comparison. Output is archived in`SCALE_PERMUTATION_CPU_2026-10-31.json`.
This validates the representation model, not an executed new torch repack
or GPU kernel. K permutation changes FP32 accumulation order, so this is
not a claim of bit-identical floating outputs or H200 certification.

## Opportunity screen and practical obstacle

`tools/model_prefill_scale_scope.py` uses all eight benchmark/profile cells
from same-job `0848e5e1b2ba40e5`, sfo3, whose timed path is the unchanged
parent. Hypothetically remove5/10/15/20% of profiled expert duration:
whole-board proxies1.019650/1.040089/1.061365/1.083532x. Phase-zero1.628048x;
approximately7.5568% less expert duration models3% whole-board improvement.
These are optimistic phase-duration subtractions containing PDL waits and
overlap, not critical-path bounds or predictions for this particular edit.
`SCALE_PERMUTATION_SCOPE_2026-10-31.json` preserves every input cell.

**The representation proof is insufficient to allocate a check.** GEMM1's
normalized-input gather must permute X in the timed call, and GEMM1 must
write a correspondingly permuted intermediate for GEMM2. The current
gather stores each16-element chunk with two vector stores; a simple
permutation scatters its BF16 pairs and can require more store instructions.
More seriously, GEMM1 currently uses `stmatrix.trans` to write contiguous
8-column chunks before bulk publishing. This bit permutation spreads those
columns across chunks. Merely changing a pointer cannot implement it.
A replacement shuffle/staging/publisher design may erase all scale savings.
Neither path is implemented or timed; don't silently port only weights.

A cooperative scale-decode alternative could keep K order and exchange
one pair per lane through four subgroup shuffles, avoiding X repacking.
It reduces the nominal eight construction operations to two plus four
shuffles, a smaller saving with communication latency. It is an unimplemented
alternative, not another qualifying sweep or an assumed improvement.

## Next implementation gate

Stay on prefill. First derive and compare a concrete GEMM1 staging/publish
mapping that preserves normalized BF16 values without additional global
passes. Then implement the matching actual repack/gather/consumer in an
isolated candidate, prove stored weight/scales and intermediate index
coverage, and inspect complete native resources/ordering. Require a credible
practical>=3% case before an H200 diagnostic. Any H200 comparison must use
parent/matched/A-A controls, identical weights and routing, balanced arm
order, read flushes, multiple seeds/pads, nearby/one-hot/fresh graph inputs,
all raw samples and unchanged whole-layer timed dispatch until qualification.
No relaxed precision or routing changes. PIPE3, hybrid-exp and pin-only
negative families remain closed.

The goal remains active; implementation is incomplete and no GPU job is
live. Routine prefill usage stays4/8 checks, zero submits on Oct31 UTC.
