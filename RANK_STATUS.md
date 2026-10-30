# Verified leaderboard snapshot

**Oct30 compliance correction:** decoder job `d857144e59784ad0` was withdrawn
after confirming BF16 rounding of its required FP32 small-T router input.
Its historical rank/score below are no longer eligible. Live decoder row after
withdrawal is #836 at 1.2326x; corrected-source check `521b039c454045da` is running.
Other rows below remain an Oct29 snapshot, not current ranks.

Captured2026-10-29 around20:44–20:46UTC from live OKC boards. Positions change as others
submit. Overall **#100/1612**. Prefill has a separate board and does not count toward overall.
No track has reached#1.

| Track | Our rank | Scored speedup | Leader speedup | Further normalized-time cut to match leader |
|---|---:|---:|---:|---:|
| BF16 projections |174/1398|1.521867x|1.6367x|7.02%|
| Routed experts |152/1120|2.324062x|2.6915x|13.65%|
| Attention decode |96/1074|2.603505x|3.0542x|14.76%|
| Whole decoder |100/988|2.127754x|2.4178x|12.00%|
| Prefill |105/647|1.784958x|2.0325x|12.18%|

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

MoE's second cycle used four additional checks (six today in total). Register redistribution
did not improve the aggregate; identical-input gates rejected all eight-op buckets; early
whole-W2 prefetch slowed M1 calls by 2.9–14.3%. All passed correctness and are documented,
but none is promoted. Attention's one-tile L2 lookahead passed correctness at2.482162x,
but regressed full decode and is rejected;scored2.603505x remains promoted.
These experiments do not change the scored ranks above.
