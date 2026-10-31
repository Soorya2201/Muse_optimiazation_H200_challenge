# MoE public components and decoder feasibility screen

Fresh indexed code search at this cycle returned217paths/13repos, no new
path/blob entries. Public owner metadata covers1249ranked accounts at
18:53UTC:177above/1071below/self, five unresolved404s (Abarbir12,
Pinoeer-kingxi,accountforgoofystuff,cyber199306,gurrapu21). Some repository
metadata is reused for at most one hour with original timestamps; ranks are
fresh. This is not a source audit of every file/branch/private submission.

Nineteen newly matched repositories versus the earlier Oct30 full inventory
include five SGLang forks. Their default branches compare0ahead/0changed
against their upstreams; other branches remain outside those comparisons.
Most remaining matches are already reviewed component libraries, unrelated
projects or older source families. Exact URLs and results are preserved in
RANKED_GITHUB_REPO_SCREEN and DEFAULT_FORK_DELTA timestamped JSON files.

## Selected source review

Initial26files/691225bytes were Git-blob/SHA256 verified in temporary
storage; two supplemental trxphxle source/manifest files were also verified.
No downloaded module was imported/executed in the submission. Only the
licensed V100 component is newly pinned, with original notices/licenses.

| Source | Scope and implication |
|---|---|
| [trxphxle MoE](https://github.com/trxphxle/okc-2026-muse-kernels/blob/72db638fa2718bc0ea5645fdd5c52a5511594579/kernels/moe.py) | Apache2. Actual source SHAee618300 matches publisher BEST manifest2.3249x; this is external, not reproduced. Header/packing/prepare/dispatch show the older vjeux family before our KPERM3. Added BF16 boundaries/empty-route validation do not establish an additive runtime win. Existing note about precision policy remains unresolved; no organizer ruling is inferred. |
| [ScrashOff MoE](https://github.com/ScrashOff/okc-muse-kernels/tree/d870912e7c5a030c52050bb2445418224c28d9de/history/muse_moe_w4a16) | Six sources/seven profile tables,42cells parsed. Published aggregates1.148–1.275x and all final Marlin cells are slower than current controls, without a causal cross-cluster comparison. Final8/16-row Marlin blocks/fused norm are already superseded here. Older Triton alternatives and rounded cached norm expression are not adopted. Root license unestablished; temporary review only, no mirroring/port. |
| [v100-skinny QPN](https://github.com/dnv2003/v100-skinny/blob/5b589c0dc81223e0ba65bcb3e755874723f8b515/kernels/skinny_kernels.cu) | MIT for kernels/docs, directory-scoped dual-license notice verified. QPN uses FP16 operands and Volta m8n8k4 quadpairs, so this is not a direct BF16/H200 kernel. Fragment-order packing, scale reuse and activation-stationary geometry are useful mechanisms, already partly represented here. Its separate int8-activation DP4A path is excluded. Six original source/docs/license files pinned under research/public/v100_skinny_5b589c0; no execution/H200 gain claim. |
| [TensorRT-LLM Marlin](https://github.com/NVIDIA/TensorRT-LLM/blob/2d1a7d170eb808bbe0e1203aa9d34dea702d52c5/cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_moe_template.h) | Apache2 source headers. Selected NVFP4 MoE GEMM/dispatch/dequant/scale/reduction sections reviewed; BF16 W4A16 is compatible in principle but parent already replaces separate Marlin expert GEMMs. No full library benchmark or additive scheduling gain established. |
| [TensorRT CUDA-core NVFP4](https://github.com/NVIDIA/TensorRT-LLM/blob/2d1a7d170eb808bbe0e1203aa9d34dea702d52c5/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.cu) | Despite its folder name, the inspected implementation loads packed FP4 activations and activation scales as well as FP4 weights. It is not a BF16-activation replacement. No activation conversion introduced. |

The [NVIDIA Hopper deployment guide](https://nvidia.github.io/TensorRT-LLM/1.4.0rc0/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.html)
confirms on-the-fly W4A16 fallback for NVFP4 on Hopper; it does not establish
that a particular TensorRT component beats this challenge parent. A native
FP4 or FP8 headline is not compatible performance evidence by itself.

## New bounded decoder search

`tools/screen_moe_decode_seven_ops.py` tests a distinct seven nominal-op
family:one masked raw output,two rotated/masked outputs,then one PRMT from
already available raw/rotated sources plus a final mask. The32packed bits
must remain independently recoverable at all eight BF16 sign/exponent/
mantissa positions. There are36disjoint rotation triples from465pairs;
all442368byte-source selections fail to cover the remaining eight bits.
No candidate is generated and no GPU check is spent on this family.

This is an exact bit-position feasibility result within that family, not
an optimality proof over all instructions, arithmetic encodings or circuits.
Other LOP3/IMAD/packing designs remain outside this search. The previously
closed eight-op family is not retried under a new name.

Existing reduced NTT8 parent SASS also shows eight PRMT sites for the two
static first/steady stage bodies:four scale constructions per body. The
compiler already hoists KPERM3 scales across all four load groups. Merely
moving those source expressions cannot claim new work removal. Scope is
isolated phaseA codegen, not dynamic full-kernel counters. Exact source/
archive hashes and PRMT lines are in DECODE_SEVEN_OP_SCREEN JSON.

The new concrete direction is independent TINY phase accounting on M1,
with controls and exact telemetry, before attempting a stream/synchronization
redesign. See TINY_PHASE_RESEARCH_2026-10-31.md. No new optimized dispatch,
canonical change or score improvement is established by this research.
