# Public attention source research and reproduction

## Oct 29 second cycle: isolated future-row prefetch

Current follow-up: diagnostic check `13bc760e093343f3` runs
`submission_v17_phase_probe.py`, SHA
`703205792c1797613d674bc02d250415218644fcc72384cc0b808849a058f1d1`.
The timed native source and Python entry path preserve the parent exactly, apart from
the extension namespace. A separately compiled traced module runs only on independently
generated import-time fixtures. Ten shapes expose entry/segment setup, page-index
readiness, copy issue, query setup, first K/V readiness, loop end, partial issue and merge.
It verifies debug outputs against the parent, reports instrumentation timing overhead,
drops all synthetic graphs/inputs/outputs and zeroes coordination scratch before real calls.
Markers perturb scheduling, so their intervals are diagnostic rather than exact parent
cycle counts. No performance gain is claimed. Builder/source isolation/syntax checks pass;
The check passed16tests/12timings at2.605394x,1445.578us,CI[2.566313,2.644475].
All10debug/control outputs agree within1.58e-5;instrumentation adds0.93–2.70us.
All12profiles/full logs were inspected. Raw records and careful stage interpretation are
preserved in `phase_probe_13bc760e093343f3.json` and `PHASE_PROFILE_2026-10-29.md`.
The timed path is parent,no gain claimed. Do not promote the diagnostic file.

Fourthcheck4bbfb48549864c60 tested `submission_v18_softmax_phase_offset.py`,
SHAc2d533e3b4b363a4766aed38ca8da328cc0d03585a14acd976357adb62598075.
Only flatL8 adds a one-time consumer-group offset;existing transpose fences/per-head
barriers remain. It passed16tests/12timings at2.593449x,1452.236us,
CI[2.554548,2.632351]. B16/B32fullverify100.76/184.30us versus parent's101.60/186.01us
are sub1% cross-job observations,not a proven gain;aggregatebelowparent. Not promoted.
All12profiles/emptylocal log and remoteprofiler markers read. No further TURN sweep is
justified by this result alone. The primary-paper/source mapping is in RESEARCH_LIBRARY.md.

### Trace-driven parallel completion flags (v19, running)

Fifthcheck5161a53a9b274da4 tests `submission_v19_parallel_completion_flags.py`,
SHA88472e7c187759c493f37f5a68265d87a0c7dc4dd7ec87a9260b654de8b21d7b.
The target is B1 sliding L1's measured coordination tail:after the last partial-issue
marker at2.944us,merge-ready is4.224us in the instrumented fixture. This1.280us interval
includes pending reductions,arrival coordination and instructions;it is not pure atomic
latency. The earlier median-marker gap1.50us also includes CTA imbalance.

The local change is restricted to one request,twoKVheads,nc1,no cluster,and<=32splits.
After the parent's existing CTA barrier,each split's thread0 release-stores a flag.
CTA0's first warp acquire-polls all active flags in parallel,then a CTA barrier propagates
the acquires to the unchanged `fin_nc1` finalizer. Existing FP32 atomic sums,normalization,
gating,precision,full attention and launch plans remain unchanged. Other CTAs return after
publication. One coalesced warp clears all32flags before kernel completion,including
inactive slots;serial graph/stream calls start with zero flags. No epoch counter is needed,
so there is no generation-wrap assumption or extra epoch-load latency.

The aligned flags31104..31135 fit inside the parent's minimum32768counter allocation and
do not overlap existing regions. H200 can hold this<=32CTA cohort. Empty splits are a
contiguous tail,excluded by `ceil(n_keys/chunk)`. CPU checks cover131328participant-range
cases,including context lengths up to1M,and source isolation/syntax pass. This arithmetic
check is not a proof of all GPU memory-order behavior;judge correctness/graph replay and
timing remain necessary. No synthetic fixture is retained by this performance candidate.

This is new local integration/protocol work,not a claim to invent completion flags.
Do not extend it to L8 without a separate design:oneCTA finalizing128rows may lose to
the parent's distributed cooperative finalizer. Budget5/8checks,1/2submits.

After MoE's second cycle, attention resumes at one check and one submit used today.
The full current source/profile/checks/reference files and track instructions were reviewed.
The promoted source already has FP8 WGMMA, FP8 PV for verify, producer register redistribution,
TMA page-index strips, and shape-specific fused merge. These are retained mechanisms.

Further inspection of the Apache-2.0 [Madhavan derivative](https://github.com/Madhavan113/kerneloptimizer/blob/c180924742f5ec30ee7434b1d99cf2c16d2aa777/muse_attention_decode/submission.py)
finds identical tuned launch overrides, plus many disabled experimental branches. Its public
2.5883x snapshot is below our 2.603505x, so the whole file is not assumed faster. Its
`OKC_PFL2` producer hook is absent from our pruned source: once future page-table strips
land, one lane per 128-byte K/V row prefetches the next tile into L2 before normal copies.
The experiment inventory lists distances one and two but supplies no win in that code.

`submission_v16_l2_lookahead.py` ports only distance one into the existing `TMAPG_ON`
branch (flat full-context L=1). It retains the same required cache reads, precision,
barriers, new-entry writes and launch plans. No synthetic tuning code, actual inputs,
or outputs are cached. Preconditions keep it within cached rows, page size one and
32-bit cache offsets; other native paths are unchanged.

Expected upside is modest because current B16/B32 full decode is already around the
published older fixture's gather ceiling. Possible losses are early producer blocking,
extra issue instructions/register pressure, and extra traffic. Neither a positive
source description nor a lower-ranked team's technique proves a gain here.

Builder `tools/build_attention_l2_lookahead_candidate.py` verifies that only the small
native hook and extension namespace differ from the exact checked parent. Python syntax
passes. SHA-256 `1e1ad7ff7266e6f9b99172d0e5703e7b6574668f1b5570a0f49de561aef9e23b`.
Check `5213b424b3eb47f5` passed all 16 tests and 12 timings at **2.482162x**,
1517.347 us, CI [2.444929, 2.519394]. It materially regresses the affected paths:

|Case|Scored parent paired us|Lookahead paired us|Parent/lookahead native profile us|
|---|---:|---:|---:|
|B16 full decode|86.29|92.99|86.5 / 94.9|
|B32 full decode|164.98|217.65|165.0 / 218.2|

All other paths stay near their parent timings. These are cross-job observations, but
the affected-path regressions are large, baseline times are similar, and the aggregate
confidence interval is below the parent's. The hook is rejected. Extra producer waits,
register pressure and traffic are possible causes; no hardware-counter proof isolates one.
All 12 profiles and the empty local log were inspected; remote log contains only profiler
markers. Two of eight checks and one of two submits used. Keep exact v15 promoted.

Additional lower-ranked source inspections in this cycle:

|Pinned source|Actual sections reviewed|Transfer decision|
|---|---|---|
|[operatorstack 6d9b776](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_attention_decode)|Published SHA `c5a94b6031bd1df0392773dcee7501a5dcc0d6de8e28eda4cec6d2109e29ad64`; actual producer/consumer loop, plan table, entry point, packing/merge declarations. MIT plus SGLang Apache portions in NOTICE. Public receipt `82034c2594ca4e55` reports 1.822823x, CI 1.795481–1.856356.|Fused split attention, two-head 128-byte reads, prefetched gate, and warp-specialized WGMMA are already present in our parent. Its verify consumer converts V into FP16 shared staging; ours already has FP8 PV and producer-side transpose. No demonstrated missing mechanism warrants replacing the faster parent.|
|[tommyyliu f134725](https://github.com/tommyyliu/okc/blob/f134725e27f368588bd9ef27cb5e97d023b415a5/muse_attention_decode/submission.py)|SHA `54b1499c1c50c7d50990d3e880b2f5b0f025c920bcc87c4888eac5e7d44e3a6b`; actual split assignment, two-level arrival merge, gate/epilogue, plan and entry point. README reports historical 1.559/1.717x, while the later account snapshot is 2.0669x; no source/receipt identity is asserted.|Its separate new-token split and hierarchical LSE merge are alternative synchronization choices. Our parent fuses new rows into each segment and already has cooperative or last-arriver merge, gate overlap and full-row gathers. A wholesale port has no measured advantage. No source copied.|

These are bounded inspections of named sections, not complete audits of those repositories.
The parent's 81 lines with trailing whitespace are retained byte-for-byte in the archived
v16 candidate; `git diff --check` reports those inherited lines. The builder's AST/native
isolation check and Python syntax check pass; no whitespace cleanup is applied to checked bytes.

## Account coverage

`RANKED_GITHUB_REPO_SCREEN.json` captures all 1064 ranked accounts on 2026-10-29: 877
above our then-rank#878, 186 below, and ourselves. 941 public-repository metadata snapshots
were reused from the MoE audit performed minutes earlier; each reused row records its capture
time. Other accounts were queried afresh. Four accounts remain unresolved (HTTP404):
accountforgoofystuff, Pinoeer-kingxi, gurrapu21, cyber199306. This is a public metadata screen,
not proof that private/unlabelled code is absent; matches still require source inspection.

Relevant public owners at capture: vjeux#83/2.6104x, ruwwww#90/2.6032x,
Madhavan113#108/2.5883x, tommyyliu#281/2.0669x, lee101#327/1.9907x,
infraax#484/1.7696x, and xharshvardhan#875/1.2990x. Whole scores do not establish route-level
superiority or tie the current repository bytes to a scored receipt.

## Exact first control

Local source `submission_v15_vjeux_f35a41e.py` is byte-identical to:

- https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_attention_decode.py
- https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_attention_decode/submission.py

SHA-256: `5811d22bb985092ba5a29a8f4c8e266de189399ee708d7b14efcb9c9bf5a510e`.
License is taken from the original vjeux Apache-2.0 repository and preserved in
`upstream_vjeux_APACHE_2.0.txt`; ruwwww's repository has no root license file, so originality
and permission were verified by comparing the complete source to the licensed original.

Check `385a2d2e58cd476d` passes all16 tests and12 timed shapes at **2.597212x**,
1450.132 us, CI[2.558253,2.647694]. All shapes profile as one fused launch. The source is
promoted byte-for-byte in submission.py. Scored confirmation `30c669b9702d434f` passes all16
tests and12timings at **2.603505x**,1446.627us,CI[2.564452,2.659499]. New rank#90/1066,
overall#354/1602. Both check and submit profiles/logs were read. B32 full decode164.98us and
full verify186.01us reproduce the large win.

Batch32 full decode is166.16us versus prior scored240.84us; full verify182.84us versus
390.08us. Sliding verify B16/B32 is11.45/14.97us versus prior29.70/34.31us. These cross-job
comparisons support the architectural change but are not formal per-feature ablations.

## Mechanisms missing from our incumbent

Our v13 uses fused warp-level attention for decode and a multi-launch FA3 path for most verify
shapes. The public source instead fuses Q/K normalization, RoPE, FP8 cache update, complete
paged attention, output normalization, and gating for every supported shape.

Its verify path packs query tokens and heads to use Hopper WGMMA; it keeps all visible keys,
applies the within-block causal mask, and uses online softmax. Decode uses FP8 QK and FP16 PV;
verify uses the FA3-style FP8 WGMMA probability/value path. Cache values stay in E4M3, normalized
queries round to E4M3, reductions use FP32. Numerical and cache-write validity must pass all16
judge tests, including 1M context, edge keys, alternate head counts/pages, and nonunit KV scales.

Full multi-request layers flatten work across requests to balance ragged sequence lengths;
plain split-KV, DSMEM, and cooperative finalization are chosen by shape and runtime occupancy.
Finalizers clear accumulators/counters between calls. The Python caches contain only device
properties, RoPE constants, compiled code, and reset scratch. No outputs are replayed or inputs
cached. Static review found no network, subprocess, harness import, tensor-value host branching,
or attention-weight-based key pruning in the active call.

## Sources and transfer limits

- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608): asynchronous tensor-core/TMA work,
  interleaved matmul/softmax, and FP8 processing are relevant mechanisms; its benchmark gains
  do not predict this paged decode workload's score.
- [Original source](https://github.com/vjeux/okc-2026): fused full computation and load balancing.
- [ruwwww report](https://github.com/ruwwww/open-kernel-challenge): 2.6032x reported attention
  receipt; README descriptions are checked against actual source rather than accepted as proof.
- [infraax archived experiments](https://github.com/infraax/okc-kernel-challenge): attention
  exp2 polynomial variants lost12–21%; measured exp was only7% of their path. This discourages
  speculative softmax approximations before profiling the actual current bottleneck.
- [lee101 experiments](https://github.com/lee101/openkernels): asynchronous WGMMA issue and
  avoiding unnecessary waits are useful diagnostic ideas, even when its whole score is weaker.

First reproduce the exact public source. After its profile, prioritize any remaining long-context
gather/merge cost and verify softmax/MMA scheduling. Do not repeat v7–v10 cluster/one-read sweeps
that already lost in our old warp-MMA architecture, or reduce required precision/context coverage.

## Remaining gap after the reproduced 2.597212x check

The current3.0542x leader requires about14.963% less normalized time. Modeling changes against
the actual check's weighted six-load sums gives:

| Hypothetical change | Modeled speedup |
|---|---:|
| All sliding cases10% faster | 2.684974x |
| All sliding cases20% faster | 2.780000x |
| All full-context cases10% faster | 2.785706x |
| All full-context cases20% faster | 3.005106x |

These are arithmetic scenarios, not measured gains. Even a20% full-context improvement alone
would fall short of first. Future work must combine full-context gather/consumer throughput
with the small-shape dependent-load and merge latency.

The lower-scoring lee101 source provides unusually useful counterevidence:
[gap analysis](https://github.com/lee101/openkernels/blob/main/docs/attention_gap_analysis.md)
and [fresh design/results](https://github.com/lee101/openkernels/blob/main/docs/attention_fresh_design.md).
Its measured random128B gather ceiling was3.87TB/s on its fixture; it identified seq/page/KV
dependent loads, consumer matmul-softmax chains, and2.5–5us split merge costs. Those figures
belong to its source/fixture, not ours. Its register-direct independent-warp/DSMEM design
predicted30–38% but achieved only2–4% and later variants lost. Our reproduced sliding
times are already below those earlier designs, so do not transplant them blindly.

Highest-value next bounded experiments after scoring: (1) instrument the actual current
full-verify producer/consumer/merge boundaries, (2) overlap QK of the next tile with softmax/PV
only if register/resource analysis permits it, (3) reduce a measured small-request merge or
dependent-load episode without adding a second launch, and (4) vary the split count only with
matching occupancy and complete-context correctness. The source already contains FP8 PV,
fusion, head/token reuse, and flat partitioning; count them as retained mechanisms rather
than new inventions.
