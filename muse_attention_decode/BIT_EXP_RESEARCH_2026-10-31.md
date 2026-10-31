# Fixed-binade bit approximation for packed attention probabilities

Current canonical attention remains vjeux f35a41e, SHA-256
`5811d22bb985092ba5a29a8f4c8e266de189399ee708d7b14efcb9c9bf5a510e`.
There is no verified improvement or rank-one result from this experiment yet.
The earlier degree-three Cody-Waite hybrids and B1 merge/workspace families
remain closed. This experiment replaces that polynomial with a distinct,
three-instruction interpolation core while retaining packed P.

## Primary sources and applicability

- [Schraudolph, Neural Computation 1999](https://schraudolph.org/pubs/Schraudolph99.pdf)
  explains IEEE exponent/mantissa interpolation and an adjustable bias.
  Historical CPU speedups are not evidence of an H200 speedup. The primary
  author's `nic.schraudolph.org` PDF timed out; the alternate author domain
  above was successfully read. The DOI endpoint was inaccessible this session.
- [NVIDIA developer discussion, September 2024](https://forums.developer.nvidia.com/t/overlap-between-tensorcore-gemm-operation-and-softmax-exp-operation/306988)
  contains Norbert Juffa's FP32 variants and an Orin user's report that numeric
  conversion shared the constrained math pipeline. That observation is about
  Orin, not a measured H200 pipeline assignment. Juffa also provides a limited
  range add-bias quadratic alternative. Neither supplied helper is copied here.
  We independently derive a fixed-binade linear interpolation with no F2I.

## Derivation and CPU evidence

Let `K=2/(e*ln(2))` and `delta=log2((1+K)/2)=0.04367744890360175`.
The ideal linear mantissa interpolation shifted by delta has relative error
between `-0.0298212058` and `+0.0298212058`. FP32 bias rounding adds error.
For the normal domain used here, `x+384-delta` stays within `[256,512)`, whose
FP32 bit-pattern slope is `2^15`. Shift those bits left eight, add
`0x7f800000` modulo 32 bits and reinterpret as FP32. This reconstructs an
exponent slope of `2^23` without an FP32-to-integer numeric conversion.

The helper clamps its negative tail to -126; masked negative infinity
therefore produces a tiny positive subnormal which rounds to E4M3 zero.
This performs no key skipping. Above120, the original hardware exponential
preserves overflow/max-update behavior. Online-state rescaling also retains
the hardware exponential. All arithmetic remains FP32; Q/K/V and P dtypes
are unchanged. This numerical approximation still needs independent output
validation and challenge checks; a scalar bound alone does not certify it.

`BIT_EXP_MATH_2026-10-31.json` preserves 262145 ordered FP32 samples, 52337
admissible random bit patterns and edge cases. Maximum observed relative
error is0.0298369480. Twenty independent normalized-weight fixtures have
maximum L1 distance0.0153826354. These are CPU arithmetic checks, not attention
output bounds or GPU performance results. NaN inputs are outside the finite
input contract; masking and online-state behavior require GPU validation.

## Compiler evidence, before any GPU check

`BIT_EXP_COMPILER_2026-10-31.json` preserves the complete Compiler Explorer
NVCC13 request/response, compiled for sm90a. Standalone helper has10registers,
zero frame/spill bytes. SASS contains a range predicate, then predicated
`FMNMX`, `FADD`, `LEA`; only the high-tail fallback contains `MUFU.EX2`.
No F2I or floor conversion occurs. This confirms the proposed instruction
change, not full-kernel register use or runtime improvement.

## Fixed diagnostic plan

`tools/build_attention_bit_exp_probe.py` creates immutable
`submission_v24_bit_exp_probe.py`, SHA-256
`3eed711fdf8586a7e92f2996c5f2740acc737d7ff737debe66c0c492cc1bb2e6`.
The exact parent Python/native prefix remains intact. The independent
diagnostic builds parent, bit50 and bit100; only the16 call sites inside
the existing packed-P lambda change, only in FLAT q_len8. PK2, joint max
checks, FP8 packing, WGMMA calls, all key reads and synchronization remain.

Reuse six nearby fixtures covering pages1/64, scales1/64, TP/DP and batches
2/3/5/20/33; compare parent and the file's own plain-PyTorch fallback and
require identical new-cache bits. Sixteen timing groups cover B16/B32,
random/coherent distributions, two seeds, four retained allocation pads,
nine rotated cold-L2 graph rounds and a duplicate parent. Fresh qkv replay
must change output and retain correctness. Fixtures and graphs leave scope.
Timed challenge calls always use the unchanged parent; their point score
cannot establish an optimization gain.

Declare numerical gates before H200: parent difference<1% global and<5%
per row, fallback<15%/<75%, all finite. These are diagnostic gates, not a
substitute for the challenge's adaptive FP32 bound. Require every component
group>2% faster, duplicate-parent offset<1%, and worst board proxy>=3%.
Only a qualifying candidate gets real timed-path challenge checks and an
independent repeat before promotion/submission. Never loosen gates after a
failure. This uses reserved OKC checks; no rented GPU is assumed.

The previously archived phase-zero scenario gives a loose67.4415% board
bound for eliminating both targeted full-verify cells. An optimistic model
removing half of the externally reported40.5% exponential fraction gives
5.9400%, assuming exposed work and free replacement. This passes the initial
research screen but is not a prediction; the new helper's overhead and
packing/max-check interactions must be measured.

## Terminal result: accuracy gate fails; performance remains unknown

Check `9c335063894c4abc`,cluster sfo3c,passes16challenge tests and12timed
cells at2.604626189x,CI[2.565556796,2.643695582],1446.004043us. Those timings
use the unchanged parent and establish no optimization gain. All12profiles
and the full available service log were read; immutable receipt and complete
decoded diagnostic are archived in `run_receipts/9c335063894c4abc.json` and
`PK2_HYBRID_REPORT_9c335063894c4abc.json`.

The target full-verify specialization in each full native module reports
80registers and0local bytes;
the resource getter's49152dynamic-shared value is its pre-launch allowance,
not the final launch allocation. The independent diagnostic lasts137.015s.
All six nearby coherent fixtures pass numerical/cache checks: parent global
differences are0.206–0.614%,maximum row2.058%; fallback global3.079–4.336%,
maximum row10.364%. Both new-cache tensors are bit-identical to the parent.

The first timing fixture,random B16/context86000/seed3209/pad0,then fails
the fixed parent-global<1% requirement before event measurements or fresh
replay. Bit50 has global1.461049%/row2.509642%;bit100 has
global2.007676%/row3.369145%. No timing groups are recorded. This is a
numerical robustness rejection,not proof of a challenge-rule violation or
a measured slowdown. The parent's challenge pass does not certify either
changed path. Do not loosen the gate or report a gain from its point score.

The small scalar relative-error bound did not ensure a sufficiently small
relative attention-output difference: weighted signed-value cancellation
and subsequent normalization make the mapping data-dependent. That is a
mechanistic explanation,not a measured causal decomposition. Coherent-only
validation would have missed this random-input failure. Close the tested
linear bit50/100 combination; a follow-up needs genuinely better accuracy
and credible instruction-cost/phase evidence before another GPU check.
No canonical/scored change or submit;attention Oct31checks1/8.
