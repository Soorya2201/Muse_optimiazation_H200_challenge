# Public attention source research and reproduction

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
