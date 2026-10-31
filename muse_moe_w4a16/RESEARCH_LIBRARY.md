# MoE optimization research library

Canonical is exact alygg77 KPERM3, scored2.372309331x. This index records
applicability and measured failures; it is not a promise that every idea
will improve H200 performance. Every port must preserve exact NVFP4 weights,
BF16 activations, fresh-input CUDA graphs and per-route output norm.

| Mechanism | Current evidence and decision |
|---|---|
| Persistent route/GEMM1/SwiGLU/GEMM2/norm/sum, RS-WGMMA and TMA | Already in parent; no new adoption claim. |
| KPERM3 lossless scale sharing | Adopted after paired checks; current parent. |
| Lee/public eight-op multiply decoder | Valid packing, no recurring2% gain; closed measured forms. |
| Seven-op rotate/PRMT family | New bounded442368-selection feasibility screen finds no valid layout. Not a proof that all seven-op circuits are impossible. |
| Explicit hoist of KPERM scale construction | Existing reduced compiler already hoists four pairs across four groups; no new operation-count lead. |
| REG112/GS4, fourth slots/deeper rings, early W2 prefetch | Measured negative or insufficient; do not repeat parameter affixes. |
| Uniform stage body with zero-MMA or scalar initialization | Full local-memory controls/numerics pass, runtime loses3–10%; both tested forms closed. |
| Publication/fixup elimination | Accepted sampled phase models do not establish practical>=3% opportunity; no blind fence/tail rewrite. |
| FLUTE pair LUT | BF16 compatible in principle; duplication needs8–32KiB vs1280bytes free. Single-table gain unestablished. |
| qgemm-mx | Inspected Hopper performance kernel not implemented at pin. |
| b12x | BF16 path exists; inspected native scale conversion targets SM120. No faster SM90 port established. |
| Humming/vLLM native E2M1 paths | Reviewed native conversion requires newer architecture; generic alternatives not shown faster. |
| TensorRT-LLM Marlin NVFP4 | BF16 compatibility in principle, separate GEMMs already superseded by current fusion. No paired library win. |
| TensorRT CUDA-core FP4 path | Inspected path consumes FP4 activations; excluded direct port. |
| V100 QPN | FP16/Volta architecture; fragment packing/scale reuse studied, no direct BF16/H200 execution or gain. |
| trxphxle MoE | Manifest/source match2.3249x, external older pre-KPERM source; no stronger additive component. |
| ScrashOff MoE | Six versions/42external cells reviewed; Marlin/Triton family, no stronger component. License unestablished, no port/mirror. |
| Empty-job WGS range guard | New licensed source-isolated V35 draft prevents nonexistent-job lookup; CPU source/syntax only, no measured failure/gain or promotion. Needs future empty/fresh/full H200 validation. |
| One-token phase accounting | V34 complete40groups/2traces; all numerics/resources pass but A/A/control offsets exceed2%, global attribution rejected. Raw observations retained, no score gain or automatic rerun. See dated TINY_PHASE report. |

Sources/provenance and exact current findings:
[PUBLIC_COMPONENT_REVIEW_2026-10-31T1855Z.md](PUBLIC_COMPONENT_REVIEW_2026-10-31T1855Z.md).
Measured unit/flush and uniform-loop outcomes:
[UNIT_FLUSH_FOLLOWUP_2026-10-31.md](UNIT_FLUSH_FOLLOWUP_2026-10-31.md) and
[STAGE_LOOP_AND_BRANCH_RESEARCH_2026-10-31.md](STAGE_LOOP_AND_BRANCH_RESEARCH_2026-10-31.md).
