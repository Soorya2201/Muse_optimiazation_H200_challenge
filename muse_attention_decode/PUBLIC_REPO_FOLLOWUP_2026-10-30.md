# Attention public-source follow-up

Fresh snapshot covers1104 entrants:108 above our then-rank109,995 below,
and ourselves. Five accounts remain unresolved. Repository metadata reused
within1h keeps its original capture timestamp; board ranks are current.
Original1064-account snapshot remains intact.

Current leader tjiang-fair scores3.2221x,job7d1f783733cf49c6. A direct public
repository-list query returns zero repositories. No winning source was found;
its implementation cannot be inferred from its score or account affiliation.

New/updated matches examined or linked to the previous reviewed source:

|Repository|Disposition|
|---|---|
|https://github.com/alygg77/okc-muse-spark-kernels|Pinned attention Python AST/native constants equal our exact vjeux source. Their2.6530x point is not evidence of a different algorithm.|
|https://github.com/ZhimaoL/sglang|Updated fork; inspected main has0 commits ahead of upstream. No standalone attention winner identified there.|
|https://github.com/adityuhkapoor/sglang|New below-ranked match; inspected main has0 ahead of upstream.|
|https://github.com/maxwbuckley/sglang|New below-ranked match; main0 ahead. Inspected extra branch names concern xielu.|
|https://github.com/Madhavan113/ParadigmKernelChallenge|Updated721c627 family. Existing plan explicitly closes its tiny TMA merge/control variants; do not reopen them.|
|https://github.com/praveenperera/CLIProxyAPI and https://github.com/jiroamato/CLIProxyAPI|Unrelated API proxies; false positives.|
|https://github.com/Dbar2/basement-ready|Inspected app/UI tree; false positive.|
|https://github.com/sasobhabha/muse-charm|Newly published at05:18UTC,rank955. Inspected full default-tree listing and metadata: Android launcher/companion,not a CUDA kernel submission. No code copied.|

No additional reusable stronger attention artifact was identified in these
inspected areas. This is not an all-branches/all-file-content/private-code audit.
Keep the B1 merge/protocol stop condition; today uses zero attention GPU checks.

## New hypothesis rejected from workload structure

Cross-request shared-KV-page reuse could reduce reads only when physical pages
are shared. The provided reference generator at lines182-192 allocates
`total=batch*max_pages`,makes one random permutation,and assigns disjoint slices
using cumulative per-request offsets. Therefore physical page overlap across
requests is absent in these fixtures. The gain ceiling for this specific reuse
mechanism is zero here; no GPU experiment was spent. This says nothing about
serving workloads that genuinely share prefixes,or other attention optimizations.

The remaining hybrid FMA/MUFU-exp hypothesis and its precision/performance
gates stay in CODEX_NEXT_STEPS.md and NEXT_MEASUREMENTS_2026-10-30.md.
