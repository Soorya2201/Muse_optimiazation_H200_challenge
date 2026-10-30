# BF16 public-source follow-up and isolated TMA experiment

Current canonical source remains exact alygg77 v34, SHA-256
`290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90`.
No rank-one result or new speedup established by this research.

## Search and provenance

`GLOBAL_CODE_SEARCH_2026-10-30.json` archives 234 public GitHub code results
for `muse_bf16_proj in:file language:python`, including paths, blob hashes and
URLs. This broadens the previous repository-name/account screen. It is not
proof that every branch or every competitor's code is public or searched.
The earlier query with `is:public` returned zero even for known sources;
do not use that query as negative evidence.

- [Rohan Arun's public archive](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates):
  seven BF16 candidates, root MIT license. The Gemini candidate adds a simple
  BF16 TensorDescriptor/TMA GEMM on top of vjeux, plus unrelated raw cuBLASLt
  paths and an embedded Nathen v18 source. The decoded embedded source exactly
  matches our already pinned Nathen SHA `111630ff...dcb44`; it is not new work.
  Only the standalone `rq_deep_tma_pooler` operator is extracted, with license
  and SHA manifest in `research/public/rohanarun_018b7a1/`. Full upstream module
  is not executed. Original submission SHA is
  `eced0e32d609100a8946bd3727fb76cdfcbbd15f5c7e5536c6132c145bcbf2fd`.
- `ROHAN_PUBLIC_EVALUATION_SCREEN_2026-10-30.json` stores reported measurements
  for all seven candidates. **These are unverified external claims.** Both
  Gemini banks have baseline `100 + 15 * row_index` and exactly constant
  baseline/candidate ratio `1.5601` in all 30 rows. These values cannot support
  per-cell performance decisions; measurement provenance is unresolved. No
  inference about intent is made. Do not claim a 1.5601x reproduced score.
- [Ozgur's v027](https://github.com/ozgureyilmaz/okc-muse-kernels/tree/74daa3a6990168b52359c80955014a945fc7aa4e):
  Apache-2.0. Reported check 1.1804x/scored 1.1718x; the published per-cell
  values are slower than our current controls. FP32/register and hi+lo router
  paths, tinygemm and native DeepGEMM are useful background, not evidence of
  a new faster component here. No whole-source GPU check allocated.
- [muse-sputter](https://github.com/lachydotmcg/muse-sputter): MIT, explicitly
  reports a deliberately slow RTX3060 exercise; not an H200 speedup lead.
- `msaroufim/openkernelchallenge-audit`: audit/methodology project, no license
  established, no performance kernel lead. No harness tricks adopted, no
  source copied or executed.
- Madhavan's older `ParadigmKernelChallenge` name points at the already
  screened family; no distinct faster implementation established.

## Why test this operator

Distinct from Chen's native RQ128 kernels: this is a 17-line Triton BF16 dot
with TMA loads, a lossless transposed BF16 weight layout, FP32 accumulation,
and BF16 output. It changes data movement/scheduling, not weight precision.
The parent already owns the two transposed weights; no extra persistent
weight layout is needed. Public configurations target pooler/unpooler at
M=128/256. Four total cells, with all other operations retained.

The [official Triton persistent-matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
documents TensorDescriptor/TMA matmul and Hopper capability restrictions.
The [PyTorch TMA article](https://pytorch.org/blog/hopper-tma-unit/) explains
asynchronous global/shared transfers and reduced address/register work.
Its FP8-vs-FP16 speedups do not predict this BF16-vs-tuned-BF16 comparison.
This operator is ordinary tiled GEMM, not a new persistent implementation.

Phase-zero model uses the same-cluster repeat `9bc0fbb427aa45ac`, summing
all five cells per load (equal weight 62 cancels): verify16 parent 66.885273us,
verify32 82.902941us. Zeroing the four target cells gives 27.283353/34.804419us
and an intentionally loose **34.1924% board gain bound**. An optimistic
physical planning model uses max(64MiB/4.3TB/s, 2*M*33554432/989TFLOP/s)
plus an assumed 1.5us launch: projected load sums 61.496779/72.546318us,
**3.6905% board gain**. This is a best-case planning estimate, not a measured
expectation or guaranteed attainable floor. Other four loads stay fixed.

## v38 diagnostic plan

`tools/build_bf16_tma_probe.py` produces immutable
`submission_v38_tma_layout_probe.py`; exact parent is retained as a prefix.
The timed `kernel_fn` remains the parent's; the whole-run score therefore
does not measure the new TMA operator.

Prepare-only probe uses independent random and outlier activations, two
seeds, one-hot power-of-two rows, FP64 references, finite/global/per-row/
one-hot error gates, and changed-value graph replay checks. Four real
deepcopied parent states under retained 0/1/3/7MiB pads test allocation
sensitivity without repeating parent autotuning. Each candidate shares the
same BF16 transposed weight as its matching parent. Nine counterbalanced
cold-L2 graph repetitions compare parent/candidate/duplicate-parent.
Raw event times are preserved; no guessed graph overhead subtraction.
All fixtures, graphs and cloned states are discarded before prepare returns.

Promote only after consistent correctness and gains across seeds/layouts,
a useful board projection, and subsequent real timed-path check/repeat.
Do not submit a diagnostic or retry an unchanged algorithm for noisy scores.
Budget before diagnostic: 5/8 BF16 routine checks, 1/2 submits used Oct30 UTC.

## Measured result: rejected

Job `a8d2c8d0433a4829`, sfo3c, passed all 9 tests and 30 timed cells.
Prepare 165.146580s; four state copies, baseline sixteen. Whole-run score
1.556540x, CI 1.533192–1.579888, 3103.549956us, uses the **unchanged parent**.
It is not a TMA speedup and is not submitted to chase timing variation.
All thirty profiles and the complete available local/remote logs were read;
receipt and `TMA_PROBE_REPORT_a8d2c8d0433a4829.json` are archived.

The service retains only a 16,000-byte log tail. Thirty-one of the 32 printed
diagnostics survived; pad0/pooler128/seed3119 was truncated. Its numeric data
cannot be recovered from the returned artifacts. The probe completed all
assertions, but do not claim all per-case measurements are archived. Extract
with `--allow-partial`; the report explicitly records the missing row and
models only seven complete layout/seed groups. No extra GPU run is spent
repeating a rejected algorithm to fix logging. Future multi-case probes
should emit a compressed final report below 8KB to leave room for profiler
messages; preserve this immutable source and its logging failure.

| Target | Observed paired speedup range | Candidate latency change | Duplicate A/A max offset |
|---|---:|---:|---:|
| pooler128 | 0.979646–1.045515x (7 groups) | −4.35% to +2.08% | 2.252% |
| pooler256 | 0.651053–0.693833x (8 groups) | +44.13% to +53.60% | 1.133% |
| unpooler128 | 0.985890–1.018924x (8 groups) | −1.86% to +1.43% | 1.273% |
| unpooler256 | 0.989571–1.000000x (8 groups) | 0% to +1.05% | 0.940% |

Observed FP64-relative errors are at most 0.00166377 overall, 0.00172968
per row; one-hot error is zero. The updated-input graph assertions pass.
Four copied layouts were exercised (several share an address residue modulo
32MiB; this is not a claim of four distinct address colors).

All 15 nonempty fixed compositions fail the component gates. Across seven
complete groups, the best nonempty composition (pooler128 + unpooler128)
models 0.999844x worst, 1.001388x median, 1.002297x best board speedup.
The unchanged parent is the best worst-case model. No canonical change,
no new scored submission, no rank gain. BF16 budget now 6/8 routine checks,
1/2 submits used. The four published TMA configurations are closed for this
parent; do not blindly sweep them again.

What failed: generic TMA adoption adds no consistent advantage over this
already tuned SM90 parent. Pooler256's aggregate event latency is roughly
40.56us versus 27.4us for parent graphs; that regression is far larger than
the A/A variation. Exact cause (tile reuse, pipeline/occupancy or generated
instructions) is not established without counters/SASS. The small 128-token
differences are comparable to address/order variation and project far below
the board gate. Do not attribute the loss to precision or claim an isolated
hardware bottleneck from event timings alone.

Additional primary source: Bose et al.,
[When and Why to Use TMA](https://arxiv.org/html/2610.34907v1), Oct26 2026.
Their controlled H100 study treats load mechanism, tile size and sharing
protocol separately; memory-bound decode gets little benefit from changing
the load mechanism alone. This supports requiring matched controls and
counters. Their TMA-vs-cp.async gains do not predict our TMA-vs-tuned-SM90
comparison, and do not explain this particular regression by themselves.
