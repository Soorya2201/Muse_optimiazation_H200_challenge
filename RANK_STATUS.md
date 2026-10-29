# Verified leaderboard snapshot

Captured2026-10-29 around20:10UTC from live OKC boards. Positions change as others
submit. Overall **#95/1610**. Prefill has a separate board and does not count toward overall.
No track has reached#1.

| Track | Our rank | Scored speedup | Leader speedup | Further normalized-time cut to match leader |
|---|---:|---:|---:|---:|
| BF16 projections |171/1394|1.521867x|1.6367x|7.02%|
| Routed experts |149/1118|2.324062x|2.6915x|13.65%|
| Attention decode |93/1072|2.603505x|3.0542x|14.76%|
| Whole decoder |95/985|2.127754x|2.4178x|12.00%|
| Prefill |103/645|1.784958x|2.0325x|12.18%|

The gap column is1−our_score/leader_score,with the leaderboard baseline held constant.
It is a planning target,not a guarantee of a future rank.

Latest scored receipts: BF16d1bd752954ff4d21,MoE9835951891204c56,
attention30c669b9702d434f,decoderd857144e59784ad0,prefillf62a284779b4447d.
All submitted sources passed judge checks first. Full measurements,confidence intervals,
failures,and source hashes are in RESULTS.md. PUBLIC_SOURCE_INDEX.md links each track's
ranked-account audit and actual reviewed repositories.

The new isolated lee1018-op prefill port passed at1.801190x off-board,but its0.858% edge
over the parent is inside uncertainty;it is not promoted or claimed as a proven gain.
Check75053ef427e34b93 passed but selected the parent everywhere: paired expert gains
were only0.06–1.58%,below its2% threshold. No incremental whole-layer gain is claimed.
BF16 has exhausted its8-check routine budget until00:00UTC;
exact Chen and the M1 hybrid remain queued. The GitHub repository was verified PRIVATE.
