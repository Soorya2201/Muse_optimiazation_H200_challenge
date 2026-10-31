# Integer quadratic exponential: current-parent attention experiment

Previous goal turn completed the MoE pair-table screen and pushed97edd6e.
This cycle works only on attention. Canonical source remains
`5811d22bb985092ba5a29a8f4c8e266de189399ee708d7b14efcb9c9bf5a510e`,
scored2.603504573x. No rank improvement is established by CPU or compiler work.

## Distinct evaluator, same known polynomial family

The previous linear bit-exponential failed the fixed1% parent-output gate.
The endpoint-constrained quadratic previously derived in
`EXP_CURVATURE_MATH_2026-10-31.json` is also discussed in
[Fast Polynomial Transcendentals, appendix B.1](https://arxiv.org/html/2610.00049v1#A2.SS1).
Its earlier floating evaluator needs mantissa extraction and three FMAs.
We claim no novelty for that polynomial family and copy no author code.
This follow-up evaluates its correction with bounded integer products,
avoiding numeric F2I and all floating FMAs inside the exponential helper.
Scored Q/K/V formats, full context, packing, numerator/denominator operand,
max checks, state-rescaling exponentials and all synchronization remain.

For `u=bits(RN(x+384))`, `b=u&32767`, use fixed
`C=round(A*2**25)=11400656`, `A=FP32(.33976602809582634)`.
Then `p=b*(32768-b)` fits32 bits, and
`correction=(p*C)>>32`. Reinterpret
`((u<<8)+0x7f800000-correction) mod2**32` as FP32.
This realizes the known fractional polynomial
`1+(1-A)*f+A*f*f` on the bias-rounded fractional grid. NVIDIA's
[PTX integer arithmetic](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul)
defines the high-word product used here. Masked negative infinity is
clamped to-126, which rounds to E4M3 zero. No key/value load is skipped.
Above120 the original hardware exponential is retained, including overflow.

The first local checker used `math.exp2`, unavailable in the system's
Python3.9. Replacing it with `2.**x` repaired the checker before any GPU job.
All32769fraction states are monotone with exact endpoints1/2;1715scale
equivariance cases and375090additional distinct FP32 samples pass.
Worst sampled scalar relative error is0.002689394648. All bounded integer
products and corrections fit their stated fields. This is not exhaustive
FP32 coverage or an attention-output guarantee: signed-value cancellation
and FP8 rounding still require independent H200 checks.

## Compiler iterations and source hazards retained

Original NVCC13.0.0 request and its exact retry each returned terminal
HTTP502, not a CUDA compiler failure. Compiler discovery GETs worked;
the same source/flags compiled successfully with NVCC13.0.1 (`nvcc130u1`).
Every request/response, including both transport failures, is archived.

The first C++ guarded core introduces many BSSY/BSYNC pairs. Explicitly
predicating every temporary removes reconvergence but makes ptxas preserve
temporary values through SEL/FSEL chains. Initializing the core unconditionally
and predicating only the final native fallback removes those chains.
These are compiler results, not measured runtime regressions.

One further source audit caught the initialized draft writing the C++
assembly output before its last input read. With an ordinary `=f` constraint,
coalescing could be unsafe. The final helper keeps the intermediate in a
local PTX register and writes the output only after every input read. No
GPU job used the earlier draft. Final full standalone code compiles unchanged
in static instruction counts, zero frame/spills, and32registers for all
four changed entries. Native remains29registers.

|Pack32 static sites, including prologue/padding|Native|25%|37.5%|50%|100%|
|---|---:|---:|---:|---:|---:|
|Final initialized/predicated form|112|192|224|264|408|

All have16native packed F2FP sites. There are32static MUFU sites, but the
offloaded ones are high-tail-predicated fallbacks; do not interpret the
static count as32executed native exponentials on ordinary inputs. The
helper's ordinary core is six integer instructions plus clamp/add and a
fallback predicate. No runtime improvement follows from these counts.
Final PTX5325rows per target and SASS1215rows are complete/untruncated.
These are standalone resources, not the full linked attention specialization.

V25 is the fully predicated-temporary draft; V26 initializes temporaries;
V27 adds stage checkpoints and preserves primary/secondary errors; V28
also fixes the operand-lifetime issue. All sources remain archived. Only
V28 is eligible for this GPU diagnostic. Its SHA-256 is
`881f9a61d5f9f6343505680eae30b092e23f1c79090dee436102d03b388ad252`.
The final host preflight tests3pure source transforms,576exact event float32
bit roundtrips, ratio reconstruction and injected simultaneous build/cleanup
errors. It proves the original error remains in the emitted report.

## Non-free opportunity model and fixed GPU protocol

`INTEGER_EXP_OPPORTUNITY_2026-10-31.json` retains a simplified two-pipeline
model. Assuming an exposed40.5% exponential share from earlier external
evidence, integer/SFU throughput ratio4, six integer operations per replacement,
no shared issue competition and free overhead,37.5% offload yields a
1.043220x board proxy using actual scored-parent load weights.25% yields
1.027986x. These are optimistic scenarios, not bounds or current-parent
attribution. The [Hopper architecture table](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
lists64INT32 and128FP32 cores per SM, but does not prove the model's
IMAD.HI throughput or overlap. Register growth, predicates, integer throughput,
packing and already hidden SFU work could remove the entire potential gain.
This explains choosing25% as a lower-cost comparison and37.5% as the target;
it is not a blind coefficient or geometry sweep.

V28 keeps the entire canonical Python/native prefix and timed dispatch.
Independent prepare-time code builds parent/integer25/integer375, changing
only sixteen call sites inside FLAT q_len8 pack2. Eight fixed slot classes
select8or12of32scalar exponentials per consumer thread. Both query-row halves
use the same slot pattern. No data-value dispatch or key skipping is added.

Before timings require zero local-memory allocation in all full modules.
Six nearby fixtures cover pages1/64, scales1/64, TP/DP and batches2/3/5/20/33;
compare the unchanged parent and own PyTorch fallback, checking identical
new-cache bits. Parent-global<1%, parent-row<5%, fallback-global<15% and
fallback-row<75%, all finite. These are unchanged from the rejected linear
experiment, not relaxed to accept a result.

Sixteen timing groups: B16/B32, random/coherent seeds3209/7909, four retained
0/1/3/7MiB allocation pads, nine rotated/reversed cold-L2 graph rounds,
and two parent captures. Fresh qkv must change outputs and pass the same
agreement gate. Store all576event-duration values as their exact float32
millisecond bit patterns. Do not round/drop samples. Require every selected
component group>2% faster, duplicate-parent offsets<1%, and worst fixed
board-composition proxy>=3% before real changed-timed-path checks/repeats.
Any accuracy/resource/protocol failure rejects the diagnostic; no gate
relaxation or automatic retry. Parent challenge point score is never a
candidate gain. The raw-bit codec stress is1504chars before other metadata,
with an8000char emission gate inside the16000char log tail.

Fresh indexed search225paths/13repos adds0blobs; this is not exhaustive
private/unpublished-source coverage. Attention starts1/8routine checks on
Oct31 UTC. This diagnostic would be2/8, zero scored submits; reserve intact.
No rented GPU or agents. The global rank1 objective remains incomplete.

## H200 outcome

Check `d5959e1bdf8b4ad8` is terminal on sfo3. All16challenge tests and12timing
cells pass at2.597584427x,CI[2.558620661,2.649855915],1449.923999us. Those
challenge calls use the **unchanged parent**; this point is not an integer-exp
score. The independent142.304s diagnostic completes all6nearby fixtures,
16timing groups and576exact event-duration values. No primary/cleanup error.

All independent accuracy, finite-output, fresh-input and new-cache checks pass.
Maximum initial/fresh parent-global difference is0.00441105 and normalized
row difference0.01641259. Nearby fallback-global maximum0.0433148 and
row maximum0.10361274 pass their original gates. The integer evaluator fixes
the tested linear form's accuracy failure; it does not fix performance.

All three full linked modules report80registers, zero local/static-shared
bytes and384maxthreads. The reported49152dynamic-shared field is the
pre-launch allowance, not the eventual runtime shared allocation. No local
spill growth explains these observations; no causal hardware stall is measured.

|Batch|Parent/new ratio, integer25|Parent/new ratio, integer37.5|
|---|---:|---:|
|16|0.894681–0.917807|0.836752–0.867340|
|32|0.879055–0.898734|0.814045–0.836276|

Every retained group favors the parent. B16/random/pad3 also has a duplicate
parent offset1.166961%, exceeding the fixed1% gate; B32 maximum0.6296%.
The complete timing-control protocol therefore does not qualify for causal
attribution. Do not drop that group, relax the gate, or infer an SFU/INT32
stall explanation. The smaller offsets elsewhere do not rescue any candidate:
all8changed fixed compositions reject; best changed worst planning proxy is
0.985858478x (B16integer25only). The unchanged composition is1.0x.

All12local profile tables,83text-profile lines,5CLI log lines, every available
remote log line, complete decoded report, every raw event bit and reconstructed
ratio were inspected. Profile/read hashes and complete decoded microseconds
are retained in `INTEGER_EXP_PROFILE_READ_AUDIT_2026-10-31.json` and
`INTEGER_EXP_AUDIT_d5959e1bdf8b4ad8.json`; receipt and full terminal status
are in `run_receipts/`. All576event values are retained without rounding.
Ignored CLI JSON/text/log files are copied and checksummed to the internal backup.

Close this exact25/37.5% integer-quadratic form. No promotion, scored submit,
canonical change or automatic fraction/coefficient/architecture retry.
The optimistic two-pipeline scenario failed to establish a real gain;
better scalar accuracy and lower standalone register use were insufficient.
Preserve the validated integer evaluator as a component, with the measured
integration loss attached. A new use needs an independently justified phase
or architecture change, not a port to another track based on scalar tests.
Attention routine checks2/8, all scored submits0, reserve intact. No live
GPU job. Next sequential track: BF16. Rank1 objective remains active/incomplete.
