# MoE broader public-code follow-up

The prior goal turn produced measured negative BF16 evidence. This cycle
works only on MoE, with canonical source SHA
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`
unchanged. No new GPU check or scored submission was spent in this review.
MoE routine budget remains 6/8 checks and 1/2 submits used Oct30 UTC.

## Coverage

`GLOBAL_CODE_SEARCH_2026-10-30.json` archives 203 unique public GitHub
file matches for `muse_moe_w4a16 in:file language:python`. Three search pages
reported totals 201/202/203 while indexing changed; this is coverage of the
returned unique paths, not a claim of a consistent full-index snapshot or
every branch. Each result explicitly had `repository.private == false`.
Twelve repositories matched; most files are previously reviewed campaigns,
intermediate versions, starter files or tools rather than new submissions.

The useful new family is
[rohanarun/Auto-research-agent](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates),
commit `018b7a102348ea3b73701e8be52d24f82d8c91ee`, root MIT license.
All seven candidate source SHA-256 values match their evaluation manifests.
`ROHAN_PUBLIC_EVALUATION_SCREEN_2026-10-30.json` preserves compact reported
per-case values/statuses. They are external claims, not independent results.

| Candidate family | Reported local geomeans | Assessment |
|---|---:|---|
| Haiku | 0.2034x, incomplete/failed | Plain grouped Triton path, no stronger component established |
| Opus | 1.1091x / 1.1350x | Marlin block tuning, already superseded by our fused persistent implementation |
| Sonnet | 1.0422x / 1.0311x | Mostly baseline Marlin path |
| GPT5.6 | 1.0031x / 1.0957x | Mostly baseline Marlin path |
| GPT6 Astra | 1.1719x / 1.1570x | GPU route compaction, already intrinsic to our persistent implementation |
| GPT6.1 | 1.4036x / 1.4155x | Single-token/local alignment and fused norm/sum; current parent already has both |
| Gemini | 2.3768x / 2.3768x claimed | Constant-ratio rows are not usable timing evidence; inspect source separately |

Both Gemini banks again use baseline `100 + 15*index`, identical paired
round values, and a constant ratio `2.3768` across all six cases. Provenance
is unresolved; no inference about intent. Do not use their claimed M128/256
67.32/73.63us values as real per-shape evidence or promote the whole source.

## Isolated multiply decoder

Gemini source SHA
`b59196ac2365cba30cc666cc74f38051ff8f5e19c8fbd9e771adfd4696c40765`
adds `OKC_MULDEC`: matching lossless bit packing, two funnel shifts, and
`((q & 0x001b001b) * 4228) & 0x81c081c0`. Its tiny and larger variants use
different packed states. Only `_stored_pos_mul` and that decode branch are
pinned in `research/public/rohanarun_018b7a1/moe_mul_decode.py`, retaining
root MIT and upstream Apache attribution. Whole upstream module was neither
imported nor executed, and its fixed-local-start prepare wrapper was not
adopted. This is a research excerpt, not a new submitted implementation.

`tools/verify_public_moe_decode_families.py` compares parent DQ7, lee101's
already tested eight-operation placement, and this new public placement.
All **35,136** packed-word cases pass: 34,848 pair/carry/random cases,
32 explicit bit basis cases, and 256 complete multiply-mask-domain cases.
Every output fragment has the same original BF16 bits. This is not runtime
enumeration of all 2^32 words; counts/scope/hashes are explicit in
`MOE_PUBLIC_DECODE_EQUIVALENCE_2026-10-30.json`.

`4228 == 0x1084`; both eight-operation versions have the same input/output
masks and five bitwise operations, two funnel shifts, one multiply. The
differences are shift constants/directions and corresponding packed positions.
This does **not** prove identical full-kernel SASS, register allocation or
performance. It does refute treating this as a newly reduced operation-count
decoder. Existing same-input/cold-L2 check `683c1eb4664a4396` measured the
eight-operation family at about 0.98–1.02x across loads and retained parent
everywhere; large-load improvements were only 0.44–1.56%. No new evidence
supports a >3% board gain, so no blind duplicate GPU check is allocated.

The parent also has newer KPERM3 WGS layout/scale reuse; source-first copying
of an older whole implementation could remove that gain. Keep the exact
current canonical kernel until a separately justified component wins.

## Other sources resolved

- [Ranoobaba/okc-campaign](https://github.com/Ranoobaba/okc-campaign/tree/3831ab3c3538a9878031b667624240bd24bb7082),
  commit `3831ab3c3538a9878031b667624240bd24bb7082`: public receipt files
  report 1.1483x/1.2627x/1.1338x. Their six per-cell medians are all slower
  than current controls; clusters differ, so this is screening evidence,
  not a causal comparison. Best reported receipt M1/16/32/8/128/256:
  50.07/160.01/207.82/112.79/343.97/468.94us. Root license not established;
  no implementation copied. Compact evidence/links in
  `RANOOBABA_PUBLIC_RECEIPT_SCREEN_2026-10-30.json`.
- [ozgureyilmaz/okc-muse-kernels](https://github.com/ozgureyilmaz/okc-muse-kernels/tree/74daa3a6990168b52359c80955014a945fc7aa4e):
  README explicitly limits work to BF16. Two MoE keyword hits do not identify
  optimized MoE submissions.
- [ruwwww/open-kernel-challenge](https://github.com/ruwwww/open-kernel-challenge/tree/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613):
  unchanged README/receipt family reports 2.3370x, below current 2.372309x;
  generic fusion/readiness descriptions duplicate current mechanisms.
  No new additive win or root license established, no source copied.

## Primary technical sources and applicability

- [vLLM NVFP4 GEMV](https://github.com/vllm-project/vllm/blob/main/vllm/models/flint/ops/nvfp4_gemv.py),
  inspected blob `54d04ef107fda6f02c195acd667d894a2e0378d7`, Apache-2.0.
  Header explicitly targets SM100; tuned on GB200. Its
  [primitives](https://github.com/vllm-project/vllm/blob/main/vllm/models/flint/ops/cute_primitives.py)
  blob `2094a5e1f62d640b08779d455b4d593a234ff513` uses native
  `cvt.rn.bf16x2.e2m1x2`. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  lists the native conversion's architecture requirements; this is not a
  directly executable SM90/H200 alternative. Generic swap-AB/PDL ideas
  already exist in our parent. No code copied or GPU check spent.
- [Humming's author report](https://vllm-project.github.io/2026/10/19/humming.html):
  its W4A8 activation conversion and native Blackwell W4A4 results cannot be
  adopted as BF16-activation H200 gains. No activation quantization introduced.
- Humming's W4A16 components were also reviewed separately at commit
  `452e0bb15f72cc3db68af583d04425bc997dbaaa` (Apache-2.0). Seven exact file
  hashes/URLs and applicability findings are preserved in
  `HUMMING_COMPONENT_SCREEN_2026-10-30.json`. Its native E2M1 conversion guard
  excludes SM90. Its generic floating conversion uses masked bit expansion;
  no additional operation-count reduction over our specialized parent is
  established. The fused MXFP4 helper only targets FP8/INT8, not BF16. Full
  alternative SM90 WNA16 schedules remain unbenchmarked against our parent;
  this component inspection does not prove the complete library is slower.
  No source copied into our submission and no GPU run allocated solely on
  incompatible native conversion or lower-precision benchmark headlines.
- [Multi-Scale Dequant](https://arxiv.org/abs/2605.13915): activation
  decomposition/quantization redesign is separate from exact checkpoint
  execution under this task. Its headline gains are not performance evidence
  for our unchanged BF16 activation contract. No code adopted.

Decision: preserve the canonical/scored source and two remaining routine
MoE checks for a distinct, sufficiently large opportunity. The broader search
resolved missed sources and an apparent new decoder into known/unsupported
families; it did not establish a score improvement. Next sequential source
review can apply the same broad code-search method to attention.
