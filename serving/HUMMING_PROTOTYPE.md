# Isolated Humming projection prototype

The adapter at `prototypes/humming_moe.py` is development code. It is not in
any submission diff, has not executed real Humming/CUDA code, and has no
measured speedup. It consumes conventional raw w13/w2 NVFP4 parameters before
Marlin repacking. Native checkpoint aliases/loading are not integrated.

## Precision finding that changes the experiment

Pinned Humming 0.1.12 distinguishes MMA accumulator precision from Stream-K
output reduction precision. Its kernel generator selects FP32 MMA when
`use_f16_accum=False`, but the epilogue converts partial results to output
precision and `gmem_writer.cuh` adds BF16/FP16 pairs across Stream-K slices.
The output addition is not an FP32 scratch reduction. Stock Muse explicitly
passes `use_fp32_reduce=True` to Marlin. Merely selecting FP32 MMA is therefore
insufficient for a like-precision substitution.

The prototype explicitly uses `use_f16_accum=False`,
`use_batch_invariant=True`, and rejects every tactic whose `use_stream_k`
is not false. With no Stream-K slices, it passes no locks. The pinned shared
memory reducer retains FP32 values when the MMA accumulator is FP32. These
are source checks and guards, not proof of compiled numerical behavior.

Unchanged relevant sources, LICENSE and archive provenance are pinned at
`research/public/humming_0_1_12/`. The precision audit lists exact paths/hashes
under `research/serving/HUMMING_PRECISION_AUDIT_2026-11-09.json`. No library
was patched, installed or executed. This rules out adopting the default fast
path solely on the strength of current upstream benchmark claims.

## What the adapter does

- Keeps packed FP4 weights and FP8 group scales. A proposed BF16 scale-storage
  variant was dropped before validation: it adds one byte per 16 weights, roughly
 11.1% to packed-weight-plus-group-scale storage. No extra quantization or
  lower-precision activation path is used.
- Selects the gate column of two-column w13 outer scales, rounds outer scales
  through BF16 as stock does, then supplies the rounded values in Humming's
  FP32 outer-scale representation. Up/gate conversion still needs native paired
  testing; matching the data contract is not bitwise runtime equivalence.
- Replaces the six conventional parameter objects with one Humming layout.
  Does not retain a Marlin layout/fallback in that layer. Native repacking can
  have transient allocations; full-model memory/cache effects remain unmeasured.
- Estimates expected local routes from shape and EP ownership without reading
  live GPU route counts on the host. G1/G2 choose tactics separately. Each
  distinct block size gets its own valid-route table; the table is reused only
  if both projections have the same block size.
- Projects uncombined experts, runs masked SiLU between projections, and hands
  output/local IDs to the exact stock Muse finalizer. The generic Humming MoE
  runner's weighted sum without Muse normalization is not used.

## Evidence and commands

The CPU protocol tool uses tensor flags and library/operator doubles. It
binds calls against signatures extracted from exact pinned layer/forward
source; annotations and default values are omitted without changing required
versus optional arguments. It does not import or execute the real package.

The pinned-source run passes 96 protocol cases through 8 preparations: EP1/2/4/8,
the last rank of each EP size, rows 1/4/8/32, random/local/dropped routes and
equal/different G1/G2 block sizes. It checks gate-scale choice/rounding, six
parameter objects and released references to original parameter objects,
FP8 scale retention, projected route coverage and call flow. Two forbidden
precision configurations are rejected. Mock transformation does not establish
actual GPU storage size, arithmetic, graph safety or throughput.

```sh
python tools/preflight_humming_prototype.py --output /tmp/humming-cpu-protocol.json
python tools/benchmark_serving_moe_candidates.py --plan-only --humming-prototype --output /tmp/humming-source-plan.json
python tools/benchmark_serving_moe_candidates.py --benchmark --humming-prototype --output /tmp/humming-native.json
```

Only the first two modes ran locally. Their receipts preserve source hashes
and bounds. The actual native tool requires Hopper and the exact installed
0.1.12 package. It adds Humming as a separate variant, poisons its output slots
as well as Marlin's, and uses the existing dense-reference/paired/graph/timing
protocol. Both backend layouts coexist only for this synthetic A/B fixture;
that is not a proposed model deployment. The developer fixture's default small
dimensions cannot establish real-layer or serving performance.

Next gate: native numerical/graph validation, then representative layer timing.
No-Stream-K may lose badly at low concurrency. If it does, preserve that result
and reject the path instead of relaxing precision to obtain a faster number.
Only a demonstrated useful backend justifies a checkpoint-preparation/source
patch and the subsequent official quality, TTFT and all-load serving tests.
