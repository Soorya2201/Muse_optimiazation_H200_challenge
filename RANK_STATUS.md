# Verified leaderboard snapshot

Captured **2026-10-30 18:32:40 UTC** from the live OKC boards. Overall **#87/1711**.
No track has reached rank1. Positions change as others submit.

|Track|Our rank|Scored speedup|Leader speedup|Time cut to match leader|
|---|---:|---:|---:|---:|
|BF16 projections|186/1489|1.535843x|1.6498x|6.91%|
|Routed experts|118/1199|2.372309x|2.7578x|13.98%|
|Attention decode|146/1143|2.603505x|3.2221x|19.20%|
|Whole decoder|87/1059|2.211495x|2.4178x|8.53%|
|Prefill|144/705|1.784958x|2.0325x|12.18%|

The final column is 1−our_score/leader_score at a fixed leaderboard baseline;
it is a planning gap, not a prediction. Prefill has a separate board and does
not count toward overall.

Scored receipts: BF16 a3d4758c1e174aba,MoE ce241acf7ab84c04,attention
30c669b9702d434f,decoder eff0b59dc26c4c74,prefill f62a284779b4447d.
All promoted GPU sources passed the full challenge checks first. Decoder,
MoE and BF16 were independently repeated and compared with same-day controls.
Small gains have overlapping uncertainty; see RESULTS.md and each track's
comparison report for limitations and every per-shape time.

The earlier overall#50 position was real at its capture time; other entrants
moved ahead. Decoder d857144e59784ad0 was withdrawn for the small-T FP32-router
precision issue; its old rank/score is ineligible. The compliant replacement
scores2.211495x. Current local decoder/MoE/BF16 files match their pinned
Apache-2.0 public sources, with Nathen's MIT notices retained where used.

Prefill context-hardening check a77a2557e1a24fa7 and submit25a1e0e5df0d4949
both passed. Its native GPU path is unchanged; no speedup claim. The old
eligible1.784958x remains the board best. All five boards' public-owner
metadata screens are refreshed; each dated follow-up records its limits.
The isolated serving MoE load_jit binding passed two H200 kernel checks
at2.369600x/2.384810x,including independent shard-offset/deepcopy/fresh-thread
graph probes. These do not establish a speedup;canonical/scored kernels are
unchanged. The port is not model-integrated. See
serving/MOE_PORT_STATUS.md. The repository was reverified PRIVATE at capture.
Current test budgets and source-discovery scope are in CODEX_HANDOFF.md and
PUBLIC_SOURCE_INDEX.md. Rank1 work remains open.

Two new prefill performance probes passed5tests/8timings each but found no
eligible faster pipeline/exponential combination. Their official timings
use the parent dispatch;they are not new algorithm scores. Independent
diagnostics/resources and losing cases are archived in
muse_prefill_layer/HYBRID_PIPELINE_2026-10-30.md. No scored kernel changed.
