# Marlin pre-expert normalization overlap

Status: isolated experiment implemented as
[candidate08](../candidates/08_marlin_norm_overlap.diff), 1,343 bytes against
official base 297e54fd. CPU coordinator/norm checks pass; native execution,
graph safety and speedup are unproven. It is not merged into candidate07 or
selected for submission. This applies established dataflow/stream overlap;
there is no historical novelty claim.

## Existing dependency graph

At official base 297e54fd, `models/muse_spark.py::_forward_experts` records the
side-stream dependency before enqueuing the expert-input projection. Router
and top-k run on the side stream. Main then waits for that stream and calls
the experts. Packed early-input preparation excludes `use_marlin`.

`muse_spark_fp4_moe.py::_forward_marlin` subsequently applies
`pre_expert_norm` before the two GEMMs. Thus the current Marlin normalization
is after the router join even though its input is the expert projection, not
the routing result. The baseline already permits projection/router overlap;
actual concurrent execution has not been measured.

Proposed dependency graph:

```mermaid
flowchart LR
    H[Hidden state ready] --> P[Expert-input projection]
    H --> R[Router and top-k]
    P --> N[Same pre-expert norm]
    N --> J[Existing stream join]
    R --> J
    J --> E[Marlin experts and unchanged finalizer]
```

The present graph puts the join before normalization. The proposed order
allows normalization to use remaining router slack on the main stream.

## Implemented experiment

After enqueuing router/top-k and before the main-stream wait, apply the exact
same pre-expert norm to `expert_input` on main when all of these hold: CUDA,
NVFP4, Marlin, no sequence-parallel path, and a norm is present. Pass `None`
as the later pre-norm argument only for that call. Other paths retain their
current argument and behavior. The patch changes no kernel arithmetic,
collective, weight or precision.

The inspected RMSNorm implementations return new tensors rather than mutating
the shared hidden state. That supports the dependency argument, but must be
verified against the actual norm instance and installed image. Do not infer
alias safety from the module name alone. Preserve existing stream events,
tensor lifetime/recording and captured graph boundaries; add no host data read.

## Ideal bound and why it can lose

Let P be projection duration, R the complete router/top-k branch duration and
N normalization duration from a common ready event. Ignoring interference:

`current = max(P,R) + N`

`proposed = max(P+N,R)`

`ideal saving = min(N, max(0,R-P))`.

If routing has already finished by the projection, the bound is zero. This
change can lose when the extra concurrency slows routing through shared
bandwidth/SM pressure or increases capture/host overhead. The formula holds
only when phase durations remain fixed; it is not a promised speedup.

## Evidence and remaining experiment

`NORM_OVERLAP_CPU_CHECKED_2026-11-10.json` executes the exact stock/candidate
`_forward_experts` bodies and both official norm classes on CPU. All 4,320
paired cases produce identical outputs, norm inputs/outputs and router/expert
inputs. Of these, 128 nonempty cases exercise relocation; 3,328 nonempty
fallbacks and 864 empty cases keep the stock event order. Both CPU/unavailable
CUDA and simulated CUDA modes, Marlin/TRT/BF16, SP on/off, projection on/off,
norm absent/two forms, split routing, capture flags, FP32/BF16 and row counts
0/1/7/32/129 are included. Streams, routing and experts are doubles. This proves
neither CUDA ordering nor native Marlin arithmetic. The CUDA-selected norm
body runs on CPU tensors, without using the SGLang CPU extension.

The first test run failed because its fake packed routing handle was a Python
tuple; stock requires a tensor for `record_stream`. The fixture was corrected,
not the candidate. Earlier receipts remain intact. Source/AST/size packaging
passes separately in `PATCH_MANIFEST_NORM_OVERLAP_2026-11-10.json`.

The optional `--norm-overlap` mode of the native MoE harness is now prepared.
It uses the actual router, fused routing, projection and pre-norm with exact
stock Marlin expert methods and the exact stock/candidate coordinators.
It plans eager norm/route observations, fresh-input/bias CUDA graph replay,
paired AB/BA whole-coordinator timing, and optional eager CUDA timelines.
Only source-plan mode ran; see
`NORM_OVERLAP_NATIVE_PLAN_CHECKED_2026-11-10.json`. This mode uses one Hopper
and synthetic weights/local EP ownership. It has no inter-rank collectives,
attention, checkpoint or serving validation; small fixture dimensions cannot
establish representative latency. The two norm forms above are CPU coverage;
the native fixture currently uses the deployed standard pre-norm and
weight-first post-norm.

Profile P/R/N on common event timelines at the four loads, real graph buckets
and representative contexts. Check that the norm runs exactly once and has
the same input/weight/epsilon/output dtype. Compare fresh route IDs, masked
inactive experts, both norm forms and growing/shrinking graph buckets against
stock. Validate capture/replay, input immutability and allocator lifetimes.

Measure paired whole-MoE latency with normalization in its original location
and before the join. Keep original projection/router implementations and
parameters. Record rank maxima and both branch timings to expose contention;
a faster norm kernel with slower MoE completion is a failed experiment.
Only a native win should advance to full-model answer/TTFT/all-load checks.
Candidate08 does not displace the planned first three official checks
(01/06/02) without stronger representative evidence. Preserve regressions
and inconclusive profiles; a zero ideal bound closes this overlap experiment
for that workload rather than justifying a composite submission.

Evidence: CONCEPT_LOCAL_SOURCE_AUDIT_2026-11-10.json records exact official
source hashes and read scope. CUDA's documented overlap semantics explain
why readiness matters; this design uses the existing streams and does not
require turning on PDL. [CUDA synchronization/PDL reference](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
