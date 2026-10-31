# Decoder research library

This index supplements the historical reports; it does not claim exhaustive
coverage of public work or access to higher-ranked private implementations.
Every new idea must retain the TASK precision and full-attention contracts.

| Technique / evidence | Current disposition | Requirement before another GPU check |
|---|---|---|
| Four-width WGS specialization, own implementation | Compiler success with spills; extra arithmetic; deferred | Distinct spill remedy plus evidence that reduced code footprint can recover enough exposed time |
| Register redistribution GS2/GS4 | H200 negative | New resource mechanism; no repeat of the same sweep |
| Decode first A fragments before waiting for X | Reduced compile only | Exposed X-wait measurement sufficient for whole-decoder gain |
| Remove repeated expert reads | 18 external histogram planning screen below 3% | Additional service/scheduling benefit beyond saved bytes |
| Fewer publication fences / whole-unit offload | Existing scoped negative and small sampled opportunity | New measured critical-path mechanism; no extrapolation from opcode counts |
| Producer-warp sliced epilogues in MonoMoE | Primary source reviewed; direct port excluded | BF16/NVFP4 adapter, shared-memory lifetime proof, per-expert normalization and route semantics, realistic recoverable-time estimate |
| Data-carried activation-scale readiness in MonoMoE | Quantized-scale-specific protocol | Preserve BF16 payload and derive a valid same-call publication/reset protocol; do not assume a positive FP8 scale exists here |
| Hopper BF16 pull mega backend in FlashInfer | Architecture reference, not a compatible weight kernel | Decode NVFP4 without conversion of stored precision and match task semantics; benchmark actual adapted component |

## MonoMoE primary references

[MonoMoE paper](https://arxiv.org/abs/2609.04244) presents persistent,
weight-major decoding, small-N tensor-core token placement, readiness-based
handoffs and overlap of auxiliary work with weight streaming. Its H200
results concern evaluated FP8 models and baselines, not this NVFP4/BF16
challenge or our already-fused incumbent. It does not establish an additive
1.54x gain here. The parent already uses small-N WGMMA and persistent weight
streams. The potentially different component is work placement: prefetch
warps process sliced previous-expert epilogues during the next K loop.

Selected implementation:
[FlashInfer MonoMoE at 9112f5b](https://github.com/flashinfer-ai/flashinfer/tree/9112f5b919b7eed550b26f617a303820115c8905/csrc/fused_moe/monomoe).
Read complete main flow through output writeback, internal publication
helper and selected up/down deferred-epilogue bodies. This is not a complete
line-by-line audit or a build of the repository. Source has BS<=8 assertions,
FP8 intermediate quantization and scale publication; these cannot replace
our BF16 activation path directly. Source comments also rely on timing
margin for same-call zero-fill versus later atomic accumulation: that is
not a cross-CTA ordering proof to reuse. No code ported.

## FlashInfer Hopper BF16 backend

The [official architecture document](https://github.com/flashinfer-ai/flashinfer/blob/9112f5b919b7eed550b26f617a303820115c8905/docs/design_docs/moe_ep_architecture.md)
lists an SM90 BF16/BF16 pull backend and a reported H200 validation date.
Its BF16/NVFP4 mega backend targets SM100/SM103. The Hopper BF16 path is
therefore a scheduling reference; the document supplies neither an H200
NVFP4 adapter nor a matched measurement against our parent.

Pinned files retain root LICENSE/NOTICE and original contents. Their
`PROVENANCE.json` records exact commit, URLs, Git blobs and SHA-256 hashes.
Future source additions must record their own license and bounded review.

## Existing evidence to consult first

- `ROUNDED_WIDTHS_RESEARCH_2026-10-31.md`: new code/layout/compiler results.
- `PUBLIC_DELTA_AND_OPERAND_SCREEN_2026-10-31.md`: peer precision distinctions,
  source inventory, reread model and deferred operand-wait transform.
- `WGS_REGISTER_SPLIT_2026-10-31.md`: actual H200 negative and full resources.
- `PHASE_OPPORTUNITY_2026-10-31.json`: same-job phase planning data.
- `../muse_moe_w4a16/UNIT_FLUSH_FOLLOWUP_2026-10-31.md`: valid sampled flush
  attribution, limits and failed richer instrumentation.
