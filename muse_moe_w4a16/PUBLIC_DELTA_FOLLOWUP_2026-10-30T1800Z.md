# MoE public-source delta review

The live board at18:00UTC had1,199 entrants,with our scored2.372309x at#116.
Leader dolanzhao had2.7578x. This supersedes the older MoE rank snapshot;
it does not change our verified scored source or its latency.

`PUBLIC_DELTA_2026-10-30T1800Z.json` records a fresh public-owner repository
metadata screen of157 accounts: all115 currently above us plus48 accounts
absent from the earlier1,151-account snapshot,six of which overlap the above
group. One account,Abarbir12,returned404. Repository metadata was fetched
again;no old cache was used. This is incremental coverage,not a fresh full
screen of every entrant,and not a review of every file/branch/private repo.

The six main source families(alygg77,Madhavan113,vjeux,NathenFernandes,
chenwainuo,lee101) still point to the previously inspected commits. No
stronger new standalone H200 MoE source was established in this review.

|Public match|Inspection and implication|
|---|---|
|[ZhimaoL/sglang](https://github.com/ZhimaoL/sglang)|Recently pushed fork. Main comparison against upstream reports0ahead/1,666behind. Main is not an additional implementation;this comparison does not cover every branch.|
|[goldbar123467/NVFP4-Group-Gemm](https://github.com/goldbar123467/NVFP4-Group-Gemm)|Tree and good_submission.py hardware-related lines inspected. It imports Blackwell helpers and uses tcgen05/TMEM/native FP4 Tensor Cores. This is a B200 implementation,not a directly executable H200 W4A16 kernel. No code copied.|
|[goldbar123467/nvfp4_group_gemm](https://github.com/goldbar123467/nvfp4_group_gemm)|README explicitly targets B200 and packed FP4 activations. Our activation contract is BF16;silently quantizing those activations is not an allowed port.|
|[CFC87/qad-exp](https://github.com/CFC87/qad-exp)|README describes quantization-aware distillation on B200 clusters. Training/requantization changes the checkpoint and does not satisfy lossless kernel-track weight rules.|
|[gaidar0yegor/w4a16-dequant-gemm](https://github.com/gaidar0yegor/w4a16-dequant-gemm)|Previously identified NF4 roofline/dequant study;useful general evidence,not an exact NVFP4 expert submission.|
|Other matched projects|Older SGLang forks,serving recipes,automation/forum projects and unrelated Muse/MuseScore names. Exact metadata links/timestamps are retained in the JSON;no claim that their every branch was audited.|

Earlier pending xharshvardhan MoE source review was also completed;see
`LOADJIT_VALIDATION_2026-10-30.md`. Its fused alignment/hot-cold Marlin grouping
does not establish a new winning component over our persistent fused parent.
The generic unit-boundary overlap idea was checked against actual WGS code:
operand ping-pong,wait_group1 and deferred previous-segment release already
exist. No blind parameter sweep or additional GPU check was spent on it.

The updated native binding passed two full H200 kernel checks and independent
integration probes,but produced no established algorithmic speedup. The
canonical/scored source remains the earlier exact alygg77 submission. Do not
resubmit unchanged algorithms merely to seek a favorable noisy score.
