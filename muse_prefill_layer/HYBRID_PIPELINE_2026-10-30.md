# Prefill full-attention pipeline × exponential split

The preceding goal turn made progress: the MoE JIT port passed full kernel
checks plus independent shard/thread/graph tests, and public-source leads
were screened. Those changes did not establish a leaderboard improvement.
No GPU job was live on this turn's resume. This cycle targets prefill
performance, one track at a time; the external drive remains absent and
the internal checkout is authoritative for the continuation.

## Research and source finding

[FlashAttention-3](https://arxiv.org/abs/2407.08608) develops Hopper schedules
that overlap matrix work and softmax using asynchronous execution.
[FlashAttention-4](https://arxiv.org/abs/2603.05451) describes software
exponentials among its methods for balancing non-matmul resources. Its
Blackwell-specific MMA/TMEM features are not being ported to H200 here.

Our existing Apache-2.0 vjeux prefill source already contains a FP32 cubic
`exp2_poly` and a periodic FMA/MUFU split in `softmax_core_rr`. The exposed
hybrid variants1/9 use a different schedule from selected full-attention
variant43. The latter instantiates
`attn_fwd_kernel<3,0,false,true,false,true>`: pipeline/RRS, hardware exps.
No exposed variant combined that schedule with EXPMODE16 or32.

The probe adds only GLUE launch cases/attributes44 and45 for those two
combinations. The CUDA template/helper bodies are unchanged. Python's
variant metadata agrees with the C++ three-warpgroup dispatch (both new
IDs retain bit8). Query/activation/cache precision, causal masks, the full
key range and normalization remain unchanged.

This is different from the closed decode-attention B1 merge family and
from the proposed new polynomial implementation for q_len8 decode verify.
It combines existing prefill machinery; it does not reopen those experiments.

## Phase-zero and resource planning before the GPU check

Source: terminal sfo3 receipt25a1e0e5df0d4949, all eight scored shapes.
For each load, sum `weight × shape_us`, subtract only the targeted full
attention forward duration, then take the geometric mean of the three
load factors. Overlapping combine/PDL waits are not counted twice.

- Eliminating **all full-attention forward work** gives a loose15.623%
  board-speedup bound. It is not an achievable prediction for this change.
- A25% cut in those forwards would model+3.487%;50% would model+7.232%.
  Actual exposed exponential time is unknown and must be measured.
- Do not transfer the published40.5% exponential fraction from the separate
  q_len8 decode-attention kernel to these prefill kernels.

The [NVIDIA arithmetic-throughput table](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-optimizations)
lists16 approximate FP32 exponential operations per SM clock on Hopper.
Using132SMs and the observed device maximum1.98GHz gives an **idealized
instruction-count floor**, not the exposed wall time. Counting16 query
heads and `B × 16 × [L×ctx + L(L+1)/2]` valid exponentials gives:

|Full case|Ideal exp-work floor|Measured forward|
|---|---:|---:|
|B1,L753,ctx80000|231.573us|350.2us|
|B4,L753,ctx80000|926.291us|1364.9us|
|B1,L16384,ctx0|513.567us|827.2us|
|B1,L16384,ctx32768|2567.711us|3805.5us|
|B1,L16384,ctx65536|4621.854us|6736.1us|

This ignores padded/masked operations, differing clocks, other SFU work,
instruction scheduling and overlap. Subtracting half this ideal exp-work
floor would model+4.748% on the board, **not a measured or guaranteed gain**.
It motivates measuring the combination instead of spending a check on a
sub1% instruction-count tweak. Additional FMA/integer work and register
pressure can erase the entire benefit.

## Measurement design

`submission_v21_hybrid_pipeline_probe.py` retains the exact hardened-v20
prefix. Its timed dispatch is still sliding8/full43. Only independent
prepare fixtures exercise44/45. There is no promotion or benchmark-based
dispatch selection in this diagnostic.

1. Validate43/44/45 against the existing FA3 operator on three independent
   small fixtures and split counts1/3. Reject invalid hybrids.
2. Use all five full-attention scored geometries, with independent seeds
   3101/7907 for two groups. No challenge inputs are available to prepare.
3. Capture two separate parent graphs A0/A1 and each valid hybrid, retaining
   separate intermediate buffers. Require finite output and <1% relative
   L2 versus the parent for each geometry before timing that hybrid.
4. Rotate/reverse arm order over nine repetitions. Sweep256MiB before each
   replay; events bracket only the graph. Report paired ratios against the
   geometric mean of A0/A1, duplicate-control ratios and min/median/max.
5. Keep every diagnostic result, including numerical failures or regressions.
   Fixture inputs/outputs/graphs are not returned in state. The fixed parent
   dispatch remains the full-layer timed implementation.

Microbenchmarks do not establish whole-layer performance. Require recurring
per-geometry gains beyond duplicate-control spread, model the weighted board
impact, and only then check a candidate on the actual full layer. No
unchanged-source resubmission to seek a favorable score.

## Run ledger

- Check `2d21c2cf18c84e2e`, completed on sfo3; second of eight reserved
  prefill checks today, one scored submit already used.
- Source SHA256 `1cda34bc6d5f67eeda2959288e2cca48365bced652bf4d9ac95c8802e2382952`.
- Builder: `tools/build_prefill_hybrid_pipeline_probe.py`.
- Local syntax, prefix and new launch/attribute expansion checks pass.
- All5tests/8timings pass; diagnostic parent-dispatch score1.788601593x,
  216595.972us,CI1.761773–1.815431,prepare57.13s. This is **not a hybrid score**.
- All eight profiles and the complete probe/prepare/remote logs were read;
  immutable receipt and `HYBRID_PIPE_REPORT_2d21c2cf18c84e2e.json` archived.
- Both hybrids validate against FA3(worst relative error0.0151). Across
  full geometries,hybrid-vs-parent relative errors are0.001009–0.001731,
  all finite and below the1% diagnostic gate.
- Both hybrids **lose in all ten geometry/group observations**. Speed ratios
  for25%FMA range0.829333–0.859606,meaning16.33–20.58% longer attention time.
  For50%FMA they range0.786251–0.821942,meaning21.66–27.19% longer time.
  Duplicate-parent median offsets are at most0.2302%,far below the loss.
- Of27 fixed class combinations,the unchanged43/43/43 choice is the only
  one meeting the measured phase gates. No full-layer hybrid candidate or
  scored submit is justified. Canonical/scored source remains unchanged.

## Adaptive follow-up: register-budget versus arithmetic cost

The loss is established;the cause is not. Polynomial range reduction and
extra FP32/integer instructions may contend with the pipelined kernel.
Register pressure is another hypothesis,not a measured explanation yet.

v22 adds a native `cudaFuncGetAttributes` bridge for43/44/45 and a two-consumer
warpgroup PIPE/RRS schedule:48 uses native exp,49 uses25%FMA,and50 uses50%FMA.
The existing CUDA source assigns232 registers to two-WG consumers versus160
for three-WG consumers through setmaxnreg. Attribute queries describe the
compiled function;they do not directly measure dynamic spills or occupancy.

The follow-up includes a feature-off48 control and the same five geometries,
two independent seeds,duplicate parent controls and numerical gates. Each
schedule uses the existing shape-only split heuristic. It retains parent43
for every timed full-layer call. This tests a concrete register/schedule
explanation for the observed loss,rather than sweeping polynomial fractions.

- Check `2f9fe01a6f8f4702`,pending at creation;third of eight reserved prefill
  checks today,one submit already used.
- Source SHA256 `9e7aee12321ad86c2b5a6615d4cab56c0569708e599d8c392553b7f901071ec4`.
- Builder `tools/build_prefill_hybrid_resource_probe.py`;extractor accepts
  `--variants 48 49 50` to evaluate all64 fixed class combinations.
- Only a recurring modeled board gain beyond3% would justify a subsequent
  whole-layer candidate check;resource observations alone cannot raise rank.
