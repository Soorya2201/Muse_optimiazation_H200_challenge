# 04. Kernel boundaries in CUDA graphs on H200: what a node costs, what PDL hides, and why concurrency did not help

## Question

A decode step replayed as a CUDA graph pays something at every kernel boundary. On an H200:
- how much does one more kernel node cost?
- what does programmatic dependent launch (PDL) recover?
- does running independent work on a forked graph branch, alongside a latency-bound kernel, recover any time?

## Setup

1. **Node cost.** The independent harness (note 00) calibrates every timed shape. It replays a graph with one node and a graph with two nodes. I collected all such calibrations: 1,278 pairs from 89 runs across the dense-projection, attention, MoE and decoder-layer workloads. The contents of the calibration nodes are the harness's, not mine.
2. **PDL, kernel pairs I wrote.**
   - **A split-K projection and its finalize kernel.** The main kernel executes `griddepcontrol.launch_dependents` at its start. The finalize kernel is launched with programmatic stream serialization. It does its input-independent work before `griddepcontrol.wait`, then sums the partials.
   - **A split-K FP32 router (256×8192, hi/lo BF16 split) and its finalize.** It was timed with and without the PDL launch attribute; the kernels are otherwise identical.
   - **The MoE chain `gemm1` → `gemm2` → combine** (note 03). Each dependent issues its weight loads before `griddepcontrol.wait`.
3. **Concurrency.** The full decoder-layer graph (note 00) was run in two forms:
   - *base*: the 2304-row qkv+gate projection runs before attention;
   - *shadow*: the projection is split. The 1280 q/k/v rows stay on the critical path. The 1024 gate rows (16.8 MB of weights, only needed after attention) move to a forked graph branch that runs during attention, and the gate is applied after a join.

   For each layer group (decode/verify × batch × sliding/full) I tried several gate-part kernels. On three groups I also tried an L2 prefetch of the next projection's weights on the same branch.

## Method

- **Node cost** = two-node replay − one-node replay, per calibration.
- **PDL gap** = (first finalize CTA past `griddepcontrol.wait`) − (last main CTA end), from `%globaltimer` stamps in one cold replay.
- **PDL on/off and shadow vs base** were timed as whole graphs with the in-process timer (read flush). The shadow A/B interleaved the replays, required both forms to produce the same output first, and kept a shadow only if it beat base in a screening pass and again in a confirming pass.

## Data

**Table 1. Cost of one extra graph node (harness calibrations, µs).**

| | n | Median | IQR | p5–p95 | Min–max |
|---|---|---|---|---|---|
| All | 1,278 | 1.088 | 1.024–1.152 | 0.896–1.280 | 0.672–1.568 |
| Dense projections | 630 | 1.088 | | | |
| Attention | 216 | 1.088 | | | |
| MoE | 156 | 1.072 | | | |
| Decoder layer | 252 | 1.056 | | | |

The one-node replay itself had a median of 5.12 µs (4.64–5.63). Extrapolating to zero nodes (2 × one − two) gives a median of 4.03 µs (3.49–4.74). That is the fixed cost the harness subtracts.

**Table 2. PDL gap, split-K projection → finalize (10 shapes, one replay each).**

| Shapes | PDL gap |
|---|---|
| qkv+gate m = 8, 16, 32; o-proj m = 16, 32; in-proj m = 32; out-proj m = 16, 32 | 0.26 µs |
| o-proj m = 8; in-proj m = 16 | 0.51 µs |

These timer values fall on multiples of about 0.256 µs, so the gap is one to two timer steps.

**Table 3. PDL on vs off, split-K router → finalize (tuner µs, two runs).**

| m | Main grid | PDL on | PDL off | On − off |
|---|---|---|---|---|
| 128 | 128 CTAs | 13.98 / 13.79 | 11.68 / 11.68 | +2.30 / +2.11 |
| 256 | 128 CTAs | 15.04 / 15.01 | 14.40 / 14.40 | +0.64 / +0.61 |
| 128 | 64 CTAs | 11.55 / 11.52 | – | (fastest variant) |

One timeline replay at m = 128 with the 128-CTA grid:

| | PDL gap | Finalize end |
|---|---|---|
| PDL on | 1.02 µs | 9.47 µs |
| PDL off | 1.54 µs | 7.42 µs |

**Table 4. MoE at 1 token, one replay, µs from the first `gemm1` CTA start.**

| Kernel | CTA starts | CTA ends |
|---|---|---|
| `gemm1` | 0.0–0.2 | 2.3–19.2 |
| `gemm2` | 2.7–19.4 | 20.7–26.9 |
| combine (1 CTA) | 20.4 | 28.9; it passed its wait at 27.5 |

`gemm2`'s first stage became ready on average 10.5 µs after its CTAs started, with a start spread of 16.7 µs.

**Table 5. Gate shadow vs base, whole decoder-layer graph (tuner µs, one run).**

| Group | Base | Best shadow | Shadow − base |
|---|---|---|---|
| decode, B = 1, sliding | 86.9 | 85.8 | −1.1 (confirm pass: 87.1 vs 85.9) |
| decode, B = 1, full | 100.2 | 101.1 | +0.9 |
| verify, B = 1, sliding | 137.7 | 138.9 | +1.2 |
| verify, B = 1, full | 149.7 | 151.2 | +1.5 |
| decode, B = 16, sliding | 164.4 | 170.3 | +5.9 |
| decode, B = 32, sliding | 219.6 | 225.7 | +6.1 |
| verify, B = 16, sliding | 290.4 | 294.7 | +4.3 |
| verify, B = 32, sliding | 401.2 | 406.2 | +5.0 |

Adding an L2 prefetch of the next projection's weights on the branch made it slower again:
- decode B = 1 sliding: 88.6 vs 86.7;
- decode B = 16 sliding: 176.8 vs 173.0;
- decode B = 32 sliding: 230.6 vs 226.5.

On the harness, the version that kept the shadow only for decode B = 1 sliding changed the overall decoder-layer time by +0.4%, inside the run's ±1.5% confidence interval.

Data: [`data/graph_node_calibration.csv`](../data/graph_node_calibration.csv), [`data/gate_shadow_ab.csv`](../data/gate_shadow_ab.csv). Figure: [`figures/fig4_node_cost.png`](../figures/fig4_node_cost.png).

![Distribution of the cost of one extra CUDA-graph node on H200](../figures/fig4_node_cost.png)

## Finding

1. **One more kernel node costs about 1.1 µs** (median 1.088 µs, IQR 1.02–1.15, n = 1,278), the same across all four workloads. The replay itself costs about 4 µs. My batch-1 decoder layer is 13 kernels, so about 12 × 1.09 ≈ 13 µs of its 91.3 µs harness time is node overhead. That figure is an estimate: it assumes real nodes cost what the calibration nodes cost, and PDL edges may cost less.
2. **PDL hides the launch, not the dependency.**
   - With the main kernel triggering at its start and the finalize preloading its independent inputs, the finalize passed its wait 0.26–0.51 µs after the last main CTA ended, against about 1.1 µs for a plain node boundary.
   - But `griddepcontrol.wait` returns only after every primary CTA has finished and flushed. In the MoE chain, `gemm2`'s CTAs were resident from 2.7 µs and their weights were already requested. They still could not use activations until the slowest `gemm1` CTA ended at 19.2 µs. The primary's tail (here a 17 µs spread of `gemm1` end times) is fully exposed. Note 05, §4 looks at that tail.
3. **PDL can make things slower.** With a 128-CTA main grid, enabling PDL on an otherwise identical router pair cost +2.1 to +2.3 µs at m = 128 and +0.6 µs at m = 256, in two runs. The gap at the boundary did shrink (1.54 → 1.02 µs), but the finalize ended later.
   - My inference: dependent CTAs launched early occupy SM resources and run their prologue next to the primary's main loop. The CUDA guide and practitioner write-ups describe this co-residency effect.
   - At m = 128, the PDL variant with a 64-CTA main grid was the fastest of all.
4. **Running independent weight streaming beside latency-bound attention made the layer slower.** Moving 16.8 MB of gate rows onto a branch during attention was slower in 7 of 8 groups, by +0.9 to +6.1 µs. It won 1.1 µs only at decode B = 1 sliding.
   - The attention in these groups is latency-bound. For example, at B = 1 sliding the KV cache read is about 0.5 MiB, about 0.12 µs at 4.2 TB/s (derived), yet the attention kernel alone takes about 6.3 µs on the harness. On paper there was idle bandwidth to fill.
   - Two costs outweighed the overlap. First, the split adds a kernel and its fixed cost. In the tuner the two halves timed 10.0 µs (1280 rows) and 9.2 µs (1024 rows) at m = 1, against 13.8 µs for the unsplit projection in another run. Second, the concurrent GEMV competes with attention for SM slots and memory.
   - An L2 prefetch on the same branch made things worse again, by 1.9–4.1 µs.

## Implications

- **Count nodes.** At about 1.1 µs each, a 13-node batch-1 step spends about 13 µs at boundaries. Fusing two kernels saves about that much only if the fused kernel adds no tail of its own: a last-arriver reduction costs 2–3 dependent round trips (note 05, §3).
- **Use PDL with the trigger early and the independent loads before the wait.** Then check the end time, not just the gap. The early dependent's CTAs must fit beside the primary's, or the primary slows down.
- **PDL does not shorten the primary's tail.** If the last primary CTA ends 10–17 µs after the median one, as in the MoE chain, the fix is inside the primary (balance, granularity), not in the launch mechanism.
- **"Latency-bound" does not mean "free to share".** A latency-bound kernel still holds SM slots and issues dependent loads whose latency grows under extra traffic. Measure the overlap; do not assume it.

## Related work

- NVIDIA, CUDA Programming Guide, *Programmatic Dependent Launch and Synchronization*: `cudaGridDependencySynchronize` (PTX `griddepcontrol.wait`) blocks until the primary grids "have completed and flushed results to global memory" ([guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)).
- Y. Yang, *Using Programmatic Dependent Launch (PDL) to Reduce End-to-End Latency*. Overlap requires both kernels to fit on the SM, and an early dependent's prologue can slow the primary's main loop ([blog](https://yang-yifan.github.io/blogs/pdl/pdl.html)).
- Hazy Research, *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B* (2025). On H100: about 2.1 µs per dummy-kernel launch on a stream and about 1.3 µs under CUDA graphs; PDL's grid-wide wait is too coarse for per-head dependencies ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)).
