# Attention public-source delta, October31

Fresh GitHub public code search returns225unique paths in13repositories,
up from207in12 in the preceding capture. Index pagination totals and exact
URLs/blob hashes are in `GLOBAL_CODE_SEARCH_2026-10-31.json`; this is not an
exhaustive survey of unpublished/private code. Eighteen new path/blob entries
are all from [Kushagrabainsla/muse-spark-open](https://github.com/Kushagrabainsla/muse-spark-open/tree/0778ecd8293e375ffe33faa466db0231929aaac3).
Thirteen belong to attention and five reference decoder source. The latter
remain for the decoder track, preserving sequential work.

At capture this owner's attention rank is773,1.525x, job09b49ea4276e4306;
ours is173,2.6035x. A lower overall score does not rule out useful components.
All13 attention files were fetched into a temporary research directory and
verified with the Git blob SHA. Static architecture/dispatch review and the
author's experiment notes are summarized with exact SHA-256/URLs in
`KUSHAGRA_SOURCE_SCREEN_2026-10-31.json`. This is not an exhaustive per-line
conformance audit or our GPU validation. No source was executed or ported;
root license remains unestablished, so no whole-source mirror is published.

| Source family | Problem addressed | Applicability to current parent |
|---|---|---|
| A1/A2, fused Triton split-KV | Reduce launches, pair adjacent KV heads, parallelize combine | Author reports0.54–0.56x; current native parent already shares KV across all8verify rows and fuses every timed cell. No replacement established. |
| A3, fused prep+FA3 | Fuse QK normalization/RoPE/casts/cache stores | Already superseded by current one-launch native parent; historical1.106x is not a new candidate. |
| A4/A5, native FP16 MMA+prefetched page rows | Improve scattered gathers and remove repeated conversions | Earlier stage of current native family. Current parent already stages page metadata/uses shared KV and FP8 WGMMA verify. |
| A9/A10, WGMMA and warp specialization | Reuse8queries, overlap copy/transpose/score/PV, preserve async registers | Essential principles already implemented in current parent. Their full verify remained on FA3 after custom alternatives tied/lost. |
| A13/A14, cluster-pushed partials and compact merge | Reduce DSMEM pull latency and cold merge code cost | Push/merge fusion already in current parent. Idle-producer warming of a noinline merge is a research clue; its counter attribution is the author's, not our measurement. Requires a new phase-share/overhead bound before any port; closed B1 merge sweep stays closed. |
| A15, separate CUDA/PDL merges | Remove Triton combine or fuse prep for full decode | Author reports regressions and reverts both. Fullattnc uses `__ldcg` after dependency wait; correct ordering alone gives no speedup. Current parent already fuses finalization. |

Default `submission.py` and A14 are byte-identical. The
[primary experiment log](https://github.com/Kushagrabainsla/muse-spark-open/blob/0778ecd8293e375ffe33faa466db0231929aaac3/muse_attention_decode/EXPERIMENTS.md)
also reports negative FA3 split/short-chunk/page-prefetch tuning and raises
hybrid exponentials as a future lead. We already measured the degree-three
hybrids slower while preserving PK2; do not reopen them based on that note.
The independently derived bit-interpolation experiment is documented in
`BIT_EXP_RESEARCH_2026-10-31.md`, with its own H200 measurement required.

`CODE_WARMING_PLANNING_2026-10-31.json` screens the merge-warming clue against
existing current-parent phase data. Artificially removing the entire
head0-loop-end to finalizer-exit interval on both batched full-verify cells
gives only a1.007323x board planning proxy. This mixes independent synthetic
phase fixtures with scored-parent load weights; it is neither a rigorous
board bound nor an instruction-fetch counter measurement. The interval
includes other-head work and coordination; a warming helper cannot remove
it all. No >=3% full-verify-only warming candidate is established, so do not
spend a check on that speculative transfer. Sliding/B1 is outside this
model and remains subject to the closed merge/workspace-family instruction.

No newly published rank-one implementation or demonstrated additive gain is
established by this delta screen. Existing earlier ahead/behind account
screens remain historical coverage; this refresh is the public-code index,
not a refreshed all-account metadata survey.
