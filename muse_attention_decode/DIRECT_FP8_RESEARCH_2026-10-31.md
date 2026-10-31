# Direct FP8 probability generation: CPU and compiler screen

No canonical kernel, scored submission or OKC budget changed in this cycle.
Two independently derived log2-to-E4M3 quantizers pass CPU checks but fail
the current compiler-cost screen. GPU accuracy and speed remain untested.
Do not describe this as a measured H200 slowdown or a rank improvement.

The preceding invocation only refreshed ranks and counts as no optimization
progress. This continuation adds actual mathematical implementations,
numerical evidence, two successful sm90a compilations and new public sources.
All work is sequential; no agents or rented GPU used.

## Why fuse the conversion

The current parent SHA-256 is
`5811d22bb985092ba5a29a8f4c8e266de189399ee708d7b14efcb9c9bf5a510e`.
Its FLAT q_len8 PK2 lambda computes 32 scalar exponentials per consumer
thread and immediately packs them into eight E4M3 words. Both the numerator
and row sum subsequently use these packed probabilities in the same
`wgmma_m64n72k32_e4m3` operation. There is no separate FP32-probability
denominator to preserve in this path. Historical-state rescaling still
needs the original exponential.

The prior same-job load model in BIT_EXP_RESEARCH_2026-10-31.md gives a
loose67.44% board ceiling if both full-verify cells become free, and an
optimistic5.94% scenario if half of the externally reported40.5% exponential
share disappears at zero replacement cost. Those screens motivated studying
this phase; neither predicts a gain for the non-free helpers below.

This creates an opportunity to generate FP8 bytes directly from log2 scores,
removing exp2 and float-to-FP8 conversion together. Every K/V read, causal
mask, FP8 dtype, joint packed overflow check, offset move, repack, WGMMA wait
and state-rescaling operation must remain. A zero byte may follow ordinary
E4M3 rounding; it must not introduce a key/value skipping path.

NVIDIA specifies E4M3's sign/exponent/mantissa encoding, finite saturation
and nearest-even conversion. These define the target; they do not establish
our implementation's correctness or speed. Sources:
[CUDA E4M3 type](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html),
[PTX conversion](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).

## Implementations and rounding checks

`tools/attention_direct_fp8.py` contains the CPU models and generated CUDA
helpers. Normal values use a biased FP32 add to round log2 scores to eighths,
an integer candidate code, and eight repeating mantissa-boundary corrections.
Subnormals use an upward-rounded linear interpolation to select a candidate
and nine absolute log2 midpoint thresholds. One full-warp shuffle selects
the relevant correction; every live lane participates, including masked
lanes. Saturation to code126 happens after correction. The x=-10 midpoint
rounds to zero, as required by ties-to-even. Masks at negative infinity also
produce zero. NaN logits are excluded from the claimed domain.

Boundary constants are generated at 80 decimal digits and rounded upward to
FP32. These model ideal real exp2 followed by E4M3 rounding. Hardware
`ex2.approx` may produce a different byte near a rounding boundary, so there
is no universal bit-equivalence claim against the native parent.

The branchless refinement calculates bounded normal and subnormal candidates
unconditionally and selects the appropriate one before the common lookup.
It removes divergence/reconvergence without changing the intended mapping.

`DIRECT_FP8_MATH_2026-10-31.json` and
`DIRECT_FP8_BRANCHLESS_MATH_2026-10-31.json` record successful checks over
1,113,894 distinct FP32 values: ordered samples, 33 neighbors per format
boundary, all normal candidate-bin boundaries, all 131,073 relevant biased
subnormal-bin positions with neighbors, random finite bit patterns and
tails. Both models agree with the independent ideal-threshold classifier.
Ten thousand packed-word cases also verify byte ordering and the parent's
`+0x07070707` overflow predicate. This is substantial sampled coverage,
not exhaustive proof over all FP32 inputs or attention-output validation.

The first model's exact bytes are retained in
`DIRECT_FP8_CPU_V1_2026-10-31.py`, matching its report's module hash.
Recheck both current models with:

```bash
python3 tools/check_attention_direct_fp8.py --output /tmp/direct_fp8_recheck.json
```

## Compiler screen

Both complete Compiler Explorer NVCC13 requests/responses are preserved in
`DIRECT_FP8_COMPILER_2026-10-31.json.gz` and
`DIRECT_FP8_BRANCHLESS_COMPILER_2026-10-31.json.gz`.
Flags: `-O3 -std=c++17 -arch=sm_90a -Xptxas -v`. Neither response is truncated.
These are standalone four-value and 32-value packing kernels, not the
linked attention specialization. Counts include address/load/store
prologue and padding; they are static counts, not executed-path timings.

|Packing arm|Values/thread|Registers|Static SASS instructions|Reconvergence pairs|Shuffle lookups|
|---|---:|---:|---:|---:|---:|
|Native exp2 + FP8 conversion|4|14|32|0|0|
|Direct, branches|4|25|128|4|4|
|Direct, branchless|4|29|120|0|4|
|Native exp2 + FP8 conversion|32|29|112|0|0|
|Direct, branches|32|48|848|32|32|
|Direct, branchless|32|32|760|0|32|

All six builds report zero stack/spill bytes. The native 32-value kernel
contains 32 MUFU.EX2 and16 packed F2FP instructions. Direct packing removes
both, but adds clamp, candidate, select, lookup, threshold and byte-pack
work. Removing branches helps register use and instruction count without
providing a convincing full-kernel speedup case. The already tested
bias-range polynomial family remained 27–47% slower than the parent.
We therefore defer these exact quantizer implementations before GPU use.
This is an experiment-selection decision; instruction counts alone cannot
prove a runtime regression. A new implementation needs a substantially
cheaper mapping or independently supported overlap before another check.

## Additional derivation and newly found primary sources

An independently derived curvature form improves the old linear bit-exp
error: `EXP_CURVATURE_MATH_2026-10-31.json` records65,537 one-period FP32
samples using libc fmaf, monotonic output, exact endpoints1/2 and maximum
relative error0.002701730546. Its ideal polynomial is
`1+(1-A)f+A*f*f`, with FP32 A=0.3397660255432129. It adds mantissa extraction
and three FMAs to the bit reconstruction. Deferred without a GPU check;
the improved scalar error alone does not justify another polynomial sweep.

The later literature search found this same endpoint-constrained quadratic
family in [Fast Polynomial Transcendentals for LLMs, appendix B.1](https://arxiv.org/html/2610.00049v1#A2.SS1).
Its GB200 comparison separates fractional-core, complete-exp and integrated
attention measurements; short-core wins do not translate proportionally.
The [author's Apache-2.0 repository](https://github.com/MrHuff/fast-polynomial-transcendentals/tree/e022698f44c0e5816ca0cdb4d41bb64a1753c86e)
is pinned at e022698, six files/53,698 bytes including fit and matched raw
measurement data. These are GB200 results, not H200 measurements. We claim
no novelty for the quadratic family and port no reduced-precision helper.

[EFQ-Softmax](https://arxiv.org/html/2609.09721v1) directly generates
block-scaled E2M1 probabilities and uses the same operand for numerator and
denominator while retaining historical rescaling. Its format and A5 target
exclude a direct port here. The common-operand principle is already present
in our parent's FP8 row-sum WGMMA. No FP4 probability path is introduced.

The [Apache-2.0 SageAttention fork](https://github.com/zihaomu/SageAttention-int4/tree/7df231aa5c6c1e9972e69d2d6288269b31ce59cf)
contains a direct64-entry E4M3 LUT. Six source/license/benchmark files,
87,548 bytes, are pinned at7df231a with verified Git blobs and SHA-256 hashes.
Its README reports0.873–0.926x relative speed on RTX4090; it attributes the
loss to lookup/index/packing overhead and the already efficient native path.
Those are author measurements and explanations, not our H200 findings.

Static source review finds two incompatibilities: INT4/INT8 QK exceeds our
permitted query quantization, and `delta>=8.807` returns zero while ordinary
E4M3 would still represent probabilities at/below1 in its shifted domain.
That cutoff does not match format underflow. We retain the prior-art lesson
and do not port that cutoff, QK quantization or LUT kernel. No downloaded
module was imported or executed. Both repositories' provenance manifests
and original license/copyright notices are preserved under research/public.

The late challenge-specific code search still finds225 paths/13 repositories
and zero new blobs versus the earlier Oct31 capture. This is the indexed
search's scope, not proof that every entrant publishes code.

## Disposition and next step

Close the tested exact direct-mapping/compiler forms as deferred without
H200 execution. Keep the scalar derivations, primary sources, full compiler
responses and negative selection evidence. Attention remains1/8 routine
checks this UTC day; all scored sources and scores are unchanged. No live
GPU job or scored submit. Next sequential track is BF16 projections: read
its AGENTS/TASK and latest device-event/TMA reports before choosing a new,
distinct lead with a plausible >=3% whole-track opportunity.

Local verification corrections: the first curvature script used a None
tuple tie-break at zero error; initialization was fixed and the full sweep
then passed. The first V1 snapshot recovery appended an extra newline and
correctly failed its hash assertion; the archived bytes now match exactly.
Neither local error consumed GPU credit or changed a scored kernel.
