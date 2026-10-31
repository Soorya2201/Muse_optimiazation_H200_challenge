# Prefill broader public-source review

The previous decoder review made research progress; this cycle targets
prefill only. Canonical v20 remains unchanged until a separately validated
full-layer candidate establishes a gain.

## Indexed discovery and source coverage

`GLOBAL_CODE_SEARCH_2026-10-31.json` preserves **118 unique public paths in
13 repositories**. Search page totals changed from113 to118 during paging;
this is coverage of returned unique paths, not a stable exhaustive index,
every branch or unpublished submissions. Public flags were checked.
Many matches are decode-only modules, baselines, starters or tools.

Seven additional prefill sources from
[rohanarun/Auto-research-agent, 018b7a1](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates)
were downloaded, AST-parsed and inspected without importing/executing the
whole modules. All seven SHA-256 values match their external evaluation
manifests. Root MIT license was already pinned in
`research/public/rohanarun_018b7a1/LICENSE`.
`ROHAN_PUBLIC_EVALUATION_SCREEN_2026-10-31.json` retains hashes, source URLs,
external baselines and recomputed load-weighted evaluation ratios.

| Family | External bank geomeans | Disposition |
|---|---:|---|
| Haiku | 0.05314 / 0.04192x | Plain PyTorch expert expansion, no stronger mechanism found |
| GPT6.1 | 1.00750 / 1.00325x | Baseline/Marlin integration; no stronger replacement established |
| GPT5.6 | 0.99658 / 1.00569x | Mostly baseline integration |
| GPT6 Astra | 1.00066 / 1.00029x | Local preprocessing changes, no material prefill gain established |
| Sonnet | 1.00446 / 1.00601x | Baseline/Marlin family |
| Opus | 1.00835 / 1.02665x | Fused local preprocessing, older expert path |
| Gemini | 1.9562 / 1.9562x claimed | Timing claims unusable and active FP8 dense branches violate precision |

The Gemini source hash is
`c251bd840e53551f7ccb40c23a419252e1a6355c3d1cbbfd809d6cf3a42fd1eb`.
Both banks use arithmetic baseline times100+15*index and a constant ratio
1.9562 across all eight geometries. They cannot establish real per-case
timings; provenance remains unresolved, with no inference about intent.
More decisively, inspected source sets `CIT=False`, `UNPOOL_FP8=True`,
`UNPOOL_FP8_MIN_T=1`, and enables FP8 pooler for T512..4096 (plus qualifying
larger contexts). Prepare converts original BF16 pooler/unpooler weights
to E4M3, while timed `_rowq8_kernel` quantizes BF16 activations before
`torch._scaled_mm`. This violates the task's weight/activation precision
rules even if tolerance checks pass. Do not submit or adopt it unchanged.

The isolated scheduling lead is its additional attention dispatches.
Inspection confirms our exact licensed parent already contains the dormant
`PIPE3` two-score implementation. Its host dispatch is being tested using
only our existing native bodies, preserving BF16 dense projections. See
`DUAL_SCORE_PIPELINE_2026-10-31.md`; the external1.9562 claim is not a reason
to expect that isolated pipeline to win.

`ADDITIONAL_PUBLIC_SOURCE_SCREEN_2026-10-31.json` pins four other prefill
sources and hashes:

- [ruwwww/open-kernel-challenge, 3b15eb8](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_prefill_layer/submission.py):
  source differs from the exact vjeux parent by only35 unified-diff lines:
  disables residual-stream recomputation, expands cuBLASLt pins from4 to14,
  and changes the tuning selection factor0.97 to1.0. Native kernel bodies
  are unchanged. New dense-algorithm signatures are a legitimate independent
  diagnostic lead; no performance gain follows from the source alone.
- [Ranoobaba/okc-campaign, 3831ab3](https://github.com/Ranoobaba/okc-campaign/blob/3831ab3c3538a9878031b667624240bd24bb7082/problems/muse_prefill_layer/submission.py):
  older SGLang/Marlin chain and FA3 prefill. No new persistent expert
  replacement identified. The decoder small-T FP32 issue is not a scored
  prefill claim: these prefill shapes are all above8 tokens.
- [lilaclilac09/open-kernel-challenge, d9bdcc9](https://github.com/lilaclilac09/open-kernel-challenge/blob/d9bdcc9f01dff1018f2f256a0a1a63911a16557e/muse_prefill_layer/submission.py)
  and [tommyyliu/okc, f134725](https://github.com/tommyyliu/okc/blob/f134725e27f368588bd9ef27cb5e97d023b415a5/muse_prefill_layer/submission.py):
  byte-identical10,692-byte plain PyTorch starters, SHA
  `1e294bac7f86bfcde5368f690695569e6561b0f56f77a720b19b3d642793aeeb`.
  Public HEAD is not evidence of their stronger scored source.

No implementation from these four was copied. Root licenses were not
established; source URLs/hashes suffice for research references. The newly
indexed Kushagrabainsla repository's six matches are all under the decoder
folder; they do not identify a distinct prefill source.

## Chord: relevant H200 schedule, incompatible direct weight contract

[Novita Chord](https://github.com/novitalabs/chord/tree/6199d44a863a165139d2fd8387b307bbe21f1ca9)
is pinned at `6199d44a863a165139d2fd8387b307bbe21f1ca9` in
`research/public/chord_6199d44/`. The library was neither installed nor run.
Three design/performance/API documents, five component files, root Apache-2.0
and vendored DeepGEMM MIT license are preserved with hashes.

Its [API contract](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/getting_started.md)
decodes weights as `code-8`, with BF16 group32 scales. The inspected native
decoder confirms centered INT4 arithmetic; Muse uses NVFP4 E2M1 with E4M3
block scales and separate FP32 global scales. Relabeling NVFP4 nibbles as
INT4 changes weights. Even a common linear scale cannot represent all
E2M1 levels in signed INT4: half-unit magnitudes extend to12, beyond its
integer range. A lossless port must replace its decode/scale contract.

Its [design](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/design.md)
uses register-A WGMMA, paired scale extraction, one MMA group in flight,
semi-static token tiles, TMA grouped work and FP32 stream-K partials. Our
parent already contains RS-WGMMA, pair-wise scale broadcasts, persistent
heterogeneous remainder tiles and TMA staging. Those shared concepts are
not newly missing optimizations. Its paired shared-load layout and complete
grouped schedule remain possible research directions, but neither has
been measured against our NVFP4 parent.

The [performance tables](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/performance.md)
compare Kimi K2.5 INT4 shapes against Humming, using a different benchmark
and routing distribution. They do not measure Muse or our parent and do
not justify claiming its headline gains as transferable milliseconds.
No Blackwell-only instruction or lower activation precision is adopted.

## Measurement order

1. Complete PIPE3 check `4c855e373fee4973`, read every available log/profile,
   decode all diagnostic cases and reject losses. Whole-layer timing stays
   on the parent and cannot be called a pipeline gain.
2. If no pipeline qualifies, investigate the distinct public cuBLASLt pin
   lead on identical BF16 inputs/weights with parent and duplicate controls,
   preserving current residual recomputation. Do not blindly combine pins
   with a boundary change or overwrite source controls.
3. Full Chord NVFP4 scheduling port remains unimplemented/unbenchmarked;
   establish a compatible exact decode and phase bound before a GPU test.

Existing FMA25/50 and single-score2WG negative results remain closed.
