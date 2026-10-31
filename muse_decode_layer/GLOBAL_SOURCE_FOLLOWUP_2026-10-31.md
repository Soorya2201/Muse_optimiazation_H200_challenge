# Decoder public-source review and opportunity model

Canonical source remains exact alygg77 `6cae49a`, SHA-256
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.
No new decoder OKC check or scored submission was made in this review.
The UTC date rolled over during the review; all five tracks report zero
checks/submits used on Oct31. Oct30 usage remains archived in RESULTS.md.

## Additional public coverage

`GLOBAL_CODE_SEARCH_2026-10-30.json` preserves 87 public GitHub code matches
in nine repositories for `muse_decode_layer in:file language:python`.
Each result explicitly had `repository.private == false`. This covers the
returned indexed paths, not every branch or unpublished submission. Search
does not return every already known source (including our alygg77 parent),
so it supplements the account/repository audit rather than replacing it.

Six additional decoder candidates were downloaded at
[rohanarun/Auto-research-agent, 018b7a1](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates).
Root MIT license is retained in `research/public/rohanarun_018b7a1/LICENSE`.
The full modules were AST-parsed and inspected without importing/executing
them; no implementation was copied into a submission. Source hashes and
recomputed, load-weighted external evaluation summaries are archived in
`ROHAN_PUBLIC_EVALUATION_SCREEN_2026-10-30.json`.

| Candidate | External bank geomeans | Source assessment |
|---|---:|---|
| Haiku | 0.00645 / 0.00703x | Plain PyTorch starter, expands expert weights on each call |
| Opus | 1.04182 / 1.03964x | Marlin experts and fused cache/local-id preprocessing |
| Sonnet | 1.02282 / 1.02219x | Marlin and local-id fusion |
| GPT5.6 | 1.00337 / 1.00429x | Mostly baseline SGLang/Marlin path |
| GPT6 Astra | 1.11276 / 1.11291x | Triton sliding attention and GPU route compaction; Marlin experts |
| GPT6.1 | 1.10144 / 1.09208x | Split-K small projections, fused boundaries, packed Triton experts and side-stream router |

These are unverified external evaluation claims, not our judge results or
causal comparisons with our 2.211495x decoder. None establishes a stronger
replacement component. The two GPT6 implementations preserve the small-T
FP32 router activation in their inspected timed paths; that alone does not
establish complete compliance. No Gemini decoder candidate appeared in the
returned paths; do not transfer the other tracks' constant-ratio timing
claims to these six files.

`ADDITIONAL_PUBLIC_SOURCE_SCREEN_2026-10-31.json` pins commit, source hash,
URL and static signals for another 11 files in five repositories:

- [Ranoobaba/okc-campaign, 3831ab3](https://github.com/Ranoobaba/okc-campaign/tree/3831ab3c3538a9878031b667624240bd24bb7082):
  decoder uses older projection/Marlin integration. Its timed boundary requests
  `router_dtype=None`, discards the router activation and projects BF16
  `ffn_input` for routing even at T<=8. This has the small-T precision issue
  already fixed in our parent; do not adopt it unchanged.
- [PostTrainLLM/posttrainllm, f216fea](https://github.com/PostTrainLLM/posttrainllm/tree/f216fead93aebc68aeebe62d3ccb5b73811cb282):
  two indexed decoder copies are byte-identical. Marlin alignment, custom
  small-token finalize and FP32 small-T router are present. Their mechanisms
  do not establish an additive gain over our persistent fused expert path.
- [tommyyliu/okc, f134725](https://github.com/tommyyliu/okc/tree/f134725e27f368588bd9ef27cb5e97d023b415a5):
  main plus five historical candidates pinned. Main uses direct single-token
  experts and small-token finalize, otherwise Marlin; these mechanisms are
  already represented by stronger current components. No performance claim
  is inferred from historical filenames.
- [lilaclilac09/open-kernel-challenge, d9bdcc9](https://github.com/lilaclilac09/open-kernel-challenge/tree/d9bdcc9f01dff1018f2f256a0a1a63911a16557e):
  plain PyTorch starter, all local experts on all tokens with per-call weight
  expansion. Router receives an already BF16-rounded activation at small T;
  converting that tensor back to FP32 does not restore the lost precision.
- [xharshvardhan/open-kernel-challenge, 4958d31](https://github.com/xharshvardhan/open-kernel-challenge/tree/4958d3141aa977c4b6b26df7a8d06c90dd094ee9):
  fused local alignment, transposed attention alternatives, Marlin experts,
  and FP32 small-T routing. No new stronger persistent expert component found.

Only PostTrainLLM's root license was established as MIT by this metadata
screen; the other four returned no root license. No source from those four
was copied. A missing root license is not a performance finding.

The latest three commits of alygg77, Madhavan113 and lee101 were checked;
their heads remain the previously reviewed `6cae49a`, `721c627` and `c565e37`.
This review does not reveal the current leader's private implementation.

## Where a material gain could come from

`PHASE_OPPORTUNITY_2026-10-31.json` recomputes the six equally weighted loads
from scored, compliant job `eff0b59dc26c4c74`, cluster `sfo3`. Per-cell
benchmark medians and every reported profile phase duration are retained.
The model subtracts a fraction of one phase's profiled duration from the
whole-cell median, then recomputes the weighted-load geometric mean.

**This is an optimistic planning model.** PDL kernel durations include waits
and overlap; they are not isolated critical-path durations. These are not
predicted or measured improvements.

| Profile phase duration reduction | 10% less | 20% less | 30% less | Entire phase removed |
|---|---:|---:|---:|---:|
| Experts | 1.05081x | 1.10714x | 1.16997x | 1.95656x |
| Dense projections | 1.03732x | 1.07771x | 1.12159x | 1.59037x |
| Attention | 1.01603x | 1.03263x | 1.04983x | 1.19057x |
| Glue | 1.00559x | 1.01124x | 1.01696x | 1.05888x |

At Oct31 05:11:47 UTC our decoder is #118/1083, 2.2115x, and the leader
requires a 1.0933x speedup multiplier (about 8.53% less normalized latency).
Even the unrealistic removal of all profiled glue is insufficient in this
model. Expert pipeline work therefore remains the most valuable large
target; another small boundary fusion cannot plausibly close the whole gap.

## Compiler diagnostic: inconclusive, not a performance result

Inspected whether explicitly retaining four decoded KPERM scale pairs could
avoid repeated calculations. Existing KPERM3 already intends compiler reuse;
source-level repetition alone does not demonstrate redundant machine code.
`tools/probe_decoder_wgs_codegen.py` extracts only the exact public native
source, replaces its PyTorch host wrapper with two minimal launch anchors,
and requests NVCC13.0 `sm_90a` assembly from Compiler Explorer. It never
imports torch or runs a GPU kernel. Full request/response are compressed in
`WGS_CODEGEN_base_2026-10-30.json.gz`.

The compiler service returned **code143, processing-time limit/SIGKILL**.
Warnings identify unused symbols, but no completed device SASS/resource
result was obtained. No conclusion about scale hoisting, registers, spills
or performance follows. The script's explicit-reuse arm is **unrun**. No
blind GPU check was allocated on the strength of this failed compile.

## Next work

Close this decoder source-discovery cycle with the canonical unchanged.
Next review prefill's broader indexed public sources, the one track not yet
covered by this global-code-search pass. Within the decoder, any future
expert pipeline candidate must identify a mechanism absent from KPERM3,
pass exact-code/scale tests and preserve FP32 small-T routing. A design that
overlaps unit-end work with a second accumulator is an unimplemented
research hypothesis, with register capacity and consumer synchronization
still unresolved; it is not a promised improvement. Existing ring-depth,
side-stream, tail and fence negatives remain binding within their scopes.
