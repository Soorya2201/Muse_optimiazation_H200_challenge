# Strongest supported BF16 candidate and remaining optimization plan

Updated 2026-10-29 18:07 UTC. Live BF16 position: #169/1374, 1.521867x; leader 1.6367x.

## Implemented candidate

`submission_v33_chen_m1_hybrid.py` starts with the byte-identical Chen 3133.1-us export and
adds the small M1 module already compiled and validated in our two v30 checks. It targets only
QKV m1, unpooler m1 (1024 output rows), and router m1. The CUDA strings and launch geometry
are unchanged from the donor. Parent license and donor notice are preserved in the project.

The parent supplies C12 direct GEMV, RS16/RS32/RS128 WGMMA, per-thread escape cursors,
the HALF O32 route, fused router paths, tuned cuBLASLt, and conservative paired selection.
The added module supplies direct C12 QKV/unpooler and original-FP32 router M1 paths.

Two independent eight-repetition paired graph comparisons must each show more than 3% improvement
against the parent before an added route is selected. Timing uses the parent's validated read
flush and graph fixed-cost subtraction. Outputs remain live during capture/replay. Packing and
random/probe validation run in prepare; nonfinite validation errors reject the module. If setup
fails or prepare time is short, the parent remains selected. Shape, dtype, device, contiguity,
alignment, and output-width guards protect the M1 dispatch.

The generator `tools/build_rank78_m1_hybrid.py` verifies both parent/donor hashes before emitting
an apply_patch payload. `tools/check_rank78_m1_hybrid.py` verifies deterministic composition,
unchanged CUDA/config, dispatch fallback, both timing margins, nonfinite ratios, and budget skip.
These offline checks pass. Full CUDA integration, prepare duration, and H200 aggregate performance
are pending. The candidate is **not yet a scored submission**.

## Quantitative gap

These are measured incumbent load sums from job `d1bd752954ff4d21` and published Chen sums.
They are from different jobs and must not be treated as paired comparisons.

| Tokens | Our load sum (us) | Chen published (us) | Main apparent deficits to investigate |
|---:|---:|---:|---|
| 1 | 36.41 | 35.98 | Tiny unpooler/router latency; QKV traffic. |
| 8 | 36.18 | 36.35 | Small differences; avoid selection regressions. |
| 16 | 47.59 | 49.06 | Our pooler/unpooler/router appear stronger; possible later selective transplant. |
| 32 | 55.06 | 52.28 | Chen QKV and pooler RS32 routes. |
| 128 | 66.18 | 61.28 | Chen QKV/pooler/unpooler RS128 routes. |
| 256 | 81.42 | 83.51 | Dense GEMM compute/operand delivery and router reduction. |

The current result needs a 7.546% speedup, equivalent to about 7.016% less normalized time,
to match 1.6367x. A model using our same-run baseline and Chen's rounded published cell times
reconstructs 1.54234x, consistent with Chen's reported 1.542x. Taking the minimum of both
parents at every cell—an impossible cross-job oracle—models to only **1.57522x**.
Thus ordinary route union alone does not explain a rank-one result.

Starting from the Chen published cells, modeled uniform cuts across every non-router projection
produce 1.58484x at 3%, 1.61451x at 5%, and 1.66114x at 8%. These are scenarios, not forecasts.
One microsecond saved in a small-token load is more valuable than one in a large-token load:
approximately 0.46% aggregate per microsecond at m1/m8 versus 0.20% at m256.

## Ranked optimization experiments

| Priority | Mechanism and target | Evidence and primary source | Smallest decisive experiment |
|---:|---|---|---|
| 1 | Reproduce Chen export and v33 hybrid | [Chen source](https://github.com/chenwainuo/okc), our judged M1 donor. | One unchanged-parent check followed by one hybrid check; inspect all actual picks, prepare times, and paired per-shape times. |
| 2 | Selected parent/our route transplants, especially m16 and O/router | [Nathen v18](https://github.com/NathenFernandes/muse-bf16-proj), our own 9/9 scored profile. | Transplant one route family into the verified parent; avoid compiling both entire stacks. |
| 3 | Shorten C12 decode/escape critical path without changing byte format | Chen's staged cursor and short high-byte apply; [Madhavan source](https://github.com/Madhavan113/kerneloptimizer). | Same payload, exact geometry, same math; move one load/repair issue or shorten one dependency chain; check registers and spills before H200. |
| 4 | Reduce fixed M1/M8 load latency and preserve one-launch dispatch | Our M1 results; Chen self-zeroing paths. | Compare full output-producing call against actual parent, including all reduction/zeroing work. |
| 5 | Tune m256 cuBLASLt plans against the strongest custom route | Nathen's pinned Lt plans and Chen's decision to drop losing m256 RQ kernels. | Bounded same-shape algorithm/layout shortlist; no comparison solely against torch.mm. |
| 6 | Single-CTA WGMMA schedule with better operand delivery, selected m128/m256 cells | [NVIDIA Hopper GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu), [Colfax pipeline study](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/). | Hold math and output tile fixed; measure one producer/consumer schedule against tuned Lt. Require register/smem residency evidence. |
| 7 | Minimize split-K epilogue work where the measured tail dominates | [lee101 diagnostics](https://github.com/lee101/openkernels), Chen's one-node reductions. | Instrument full mainloop and completion in one bounded shape. A new epilogue must reduce the observed tail without recreating failed cluster synchronization. |
| 8 | CTA scheduling only after proving wave underfill | [Stream-K paper](https://arxiv.org/abs/2301.03598), [Colfax work partitioning study](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/). | Calculate output tile count/residency first; test scheduling only when idle SMs are the measured cause. Our old global-fixup Stream-K already failed. |
| 9 | Constant-depth compressed packets co-designed with decode | Chen large-GV proposals and exact-SO instruction diagnostics. | CPU proof of all BF16 values, full metadata/escape byte accounting, bounded CUDA decode experiment. Quarter/sparse attempts increased instruction counts and lost. |
| 10 | Activation prefetch isolated to a proven no-spill cell | Chen unpooler16 local +0.576 us, followed by whole-policy regression. | Recheck one exact-parent cell with retained controls and output validation; reject spilled QKV prefetch variants before a broad run. |

These mechanisms are hypotheses until measured on our H200 job. Pipeline and scheduling papers
explain useful design principles; their reported gains are not transferable percentages for Muse.
All weight formats must remain lossless and activations retain their required precision.

## Known failures that constrain new designs

- v20–v22 dense cluster pooler/unpooler routes lost every prepare race.
- v32 decode-to-shared cluster pooler128 took 49.22 us versus 23.68 us; reject that occupancy-1 design.
- v30's broad portfolio checked at 1.5260x then 1.5044x; isolate changes and retain actual selected routes.
- Chen's C11 and variable codecs saved nominal bytes but added excessive repair/decode instructions.
- Chen's RS128 m256, small-m TMA rings, reverse-operand ladders, and four clean-room stacks lost to its actual incumbent.
- Positive local kernel measurements and counters do not establish a whole-score gain.

## Execution and budget

On Oct 29, the CLI shows 12/20 checks and 9/10 submits available, but the project reserves
capacity by limiting routine BF16 work to eight checks and two submits daily. All eight checks
are already consumed. A request to use up to two reserve checks for the parent and v33 has been
presented to the owner; no reserve check runs without their reply. Otherwise execute after the
00:00 UTC reset. Every actual OKC check/submit must be appended to root RESULTS.md.

After a passing check, read the profile and logs; verify 9/9 tests, complete 30-cell timing,
actual M1 selections, parent route stability, and prepare duration. A scored submission requires
the project improvement/confidence gate. An unmeasured candidate never replaces submission.py.
