# Verified leaderboard snapshot

Captured **2026-10-30 05:13:00 UTC** from the live OKC boards. Overall **#54/1656**.
No track has reached rank1. Positions change as others submit.

|Track|Our rank|Scored speedup|Leader speedup|Time cut to match leader|
|---|---:|---:|---:|---:|
|BF16 projections|134/1437|1.535843x|1.6498x|6.91%|
|Routed experts|80/1151|2.372309x|2.6995x|12.12%|
|Attention decode|109/1102|2.603505x|3.2221x|19.20%|
|Whole decoder|54/1011|2.211495x|2.4178x|8.53%|
|Prefill|118/662|1.784958x|2.0325x|12.18%|

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

Prefill context-hardening check a77a2557e1a24fa7 is running. Its native GPU path
is unchanged; it is a robustness check, with no speedup claim. The repository
was verified PRIVATE. Current test budgets and source-discovery scope are in
CODEX_HANDOFF.md and PUBLIC_SOURCE_INDEX.md. Rank1 work remains open.
