# BF16 public-source follow-up, October 30

Fresh metadata snapshot: `RANKED_GITHUB_REPO_SCREEN_2026-10-30.json`,
captured04:43:50UTC. All1433 ranked accounts were queried:191 above our
then-rank192,1241 below,and ourselves. Ten accounts were unresolved (recorded
in the JSON); a public-metadata screen cannot establish private-code absence.
The original Oct29 snapshot remains intact.

New name/description/topic matches compared with the older snapshot:

|Rank at capture|Public repository|Review and disposition|
|---:|---|---|
|47|https://github.com/adityuhkapoor/sglang|Default main tree inspected. Public compare against sgl-project/main shows0 commits ahead,23 behind,no changed files. No new standalone BF16 submission identified on this main branch. Branch listing was partial; no all-branches claim.|
|87|https://github.com/alygg77/okc-muse-spark-kernels|Exact Apache-2.0 BF16 source pinned and audited; queued after exact Chen. Full upstream SHA and notices preserved. Decoder/MoE sources already reproduced and scored in our account.|
|299|https://github.com/maxwbuckley/sglang|Default main tree inspected. Public compare against sgl-project/main shows0 ahead,30 behind,no changed files. Other named branches concern xielu; no BF16 winner identified in inspected source.|
|524|https://github.com/koladefaj/sglang|Complete default-tree listing inspected for standalone submission/OKC filenames; none matched. No claim that every file or branch was read.|
|1213|https://github.com/gaidar0yegor/w4a16-dequant-gemm|Metadata describes int4/NF4 dequant+GEMM. This is not a direct replacement for lossless BF16 projections. Retained as a future quantized-kernel research pointer; no code copied or performance claim.|
|1333|https://github.com/jiroamato/CLIProxyAPI|Description match is an unrelated API-proxy project; no kernel candidate.|

This refresh found one newly published standalone candidate family already
queued by P1 (alygg77), not a newly available rank-one submission. Repository
rankings are source-discovery context, not evidence a default branch contains
the code that earned a leaderboard score.

Exact Chen first control `f5ea143175564684` passed all9tests/30timings at
1.519434x on sfo3c,prepare196.84s. It did not beat our scored1.521867x.
Repeat `640d50a9220444e3` passed1.549794x on sfo3. Keep the clusters separate
when comparing performance; independently checked correctness remains useful.
