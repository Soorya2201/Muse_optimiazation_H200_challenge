# Prefill optimization research library

Current verified local source is context-safe v20; best board1.784958x.
Use this as an applicability index, not a list of promised speedups. Every
new implementation needs matching BF16/NVFP4 precision, complete attention,
fresh-input graph behavior and independent H200 measurements.

| Mechanism | Evidence and current decision |
|---|---|
| Persistent RS-WGMMA NVFP4 experts, fused SwiGLU, variable token widths | Already in canonical vjeux parent; avoid crediting it as a new optimization. |
| Lee eight-operation decode layout | CPU and paired H200 comparisons complete; recurring2% gate not met, not adopted. |
| KPERM-style scale reuse on prefill K64 blocks | New exact encoding proof; normalized-X gather and GEMM1 output staging remain unimplemented. See dated scale-share report. |
| Cooperative per-lane scale construction plus subgroup exchange | Proposed smaller alternative; no implementation or timing. |
| Three-WG/single-score hybrid exp | Measured negative on identical independent fixtures; closed. |
| Two-WG dual-score PIPE3 | Numerics/resource checks completed, slower than three-WG incumbent; closed configuration. |
| Additional public cuBLASLt pins | All ten signatures/80groups fail recurring2%gate; pin-only sweep closed. |
| Chord grouped W4A16 | Apache2 selected source pinned. INT4-centered/group32 scale contract needs a real NVFP4 port; no package run or direct gain. |
| Hugging Face finegrained-kernels, explicit BF16 activation branch | Compatible NVFP4 group16/E4M3/global-scale grouped arithmetic in principle; persistent Triton/TMA/gate mechanisms inspected. Published tuning/validation is B200; no H200/current-parent win. Keep activation/intermediate requantization off. |
| SonicMoE IO/MMA overlap and gather fusion | Primary paper and four Apache2 source/docs/license files pinned. Inspected upstream uses ordinary Quack weights; no packed-NVFP4 interface in reviewed forward path. Token-rounding routing changes and softmax router do not implement this challenge's sigmoid/top8 contract. Component study, no wholesale import. |
| DeepGEMM-for-sail W4A16 | Current fork targets T-Head PPU SDK/HGCC/HGRTC, not H200. Screened as incompatible direct backend; no install/port. |
| MegaMoE FP4/FP8 activation paths | Previously inspected SGLang paths quantize activations; incompatible direct replacement for this task. |
| trxphxle grouped expert implementation | Licensed source/manifest hash checked; reported prefill1.4774x is not our paired timing. Native conversion/persistence mechanisms already represented; no stronger additive component established. |

Detailed current work:
[SCALE_SHARE_RESEARCH_2026-10-31.md](SCALE_SHARE_RESEARCH_2026-10-31.md).
Current public-source delta and component contracts:
[PUBLIC_COMPONENT_FOLLOWUP_2026-10-31.md](PUBLIC_COMPONENT_FOLLOWUP_2026-10-31.md).
Earlier measured negatives remain in HYBRID_PIPELINE, DUAL_SCORE_PIPELINE,
PUBLIC_LT_PROBE and IMPROVISATIONS dated records. Source availability and a
permissive license do not establish compatible arithmetic or a runtime gain.
