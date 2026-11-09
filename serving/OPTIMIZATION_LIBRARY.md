# Serving optimization library

Updated November 9, 2026. This is a bounded research and implementation library,
not an exhaustive web survey or evidence of a winning score. Exact public
commits, paths, Git blobs and download hashes are recorded in the discovery
and source metadata under `research/serving/`. No peer harness was executed.

## Implemented mechanisms

| Mechanism | Source/evidence | Our implementation and unresolved question |
|---|---|---|
| Global route density for Marlin tile selection | [hariappi's Apache2 patch and H100 layer experiments](https://github.com/hariappi/okc-muse-spark-kernels/tree/a3ebb0f2f027f07f38b063c6a74ac355a6b38848/serving) | Candidate 01. Correct denominator for globally replicated tokens. Public layer results do not establish our H200 model gain; skew and numerical behavior need official checks. |
| One collective for draft maximum and token ID | Established distributed-argmax/packing technique; implemented against our exact SGLang sampler | Candidate 02. Preserves FP32 value bits and int64 IDs; payload grows to16 bytes/rank/token. Need real collective/graph and end-to-end timing; fewer collectives do not guarantee lower latency. |
| Fused cast, local expert lookup and sentinel handling | Inspection of stock `_map_topk_ids_to_local_experts` and Muse's Marlin caller | Candidate 04. One Triton launch replaces several tensor operations for contiguous CUDA routes. Keeps scratch per call and EP1 unchanged.480 CPU fallback cases pass; CUDA branch uncompiled/unrun locally. |
| Local expert alignment and inactive-prefix specialization | [lee101's upstream analysis, item2](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/docs/upstream_ideas.md); independently traced exact official base | Candidate 06. Input IDs are already local, so align 32 local buckets at EP8 and publish the inactive block count. Invoke Marlin's local sorted/prefix specialization for both GEMMs. The base wrapper disables that specialization when `is_ep=True`, despite the caller certifying sorted IDs.420 CPU method/route-contract cases pass; native ordering, arithmetic and performance remain untested. |

Candidate 03 combines01+02;05 combines01+02+04;07 combines all four.
Combinations are packaging artifacts, not independently demonstrated wins.
All seven diffs apply to the official 297e54fd base. Candidate 07 is12,186bytes.

The prefix candidate changes the local GEMM specialization, not the distributed
model's expert parallelism. All global routes are still dispatched as before;
peer-owned routes remain -1 on this rank, and activation/finalization still use
the local-ID mask. Marlin reduction precision and model norms are unchanged.
The current CUDA source's sorted specialization advances directly by block ID;
its prefix variant reads `[padded_tokens, inactive_blocks]`. The original EP
specialization instead counts/scans inactive blocks. No microsecond estimate
from the public write-up is treated as our measured gain.

## Prioritized next research

| Priority | Approach and primary source | Concrete next experiment / constraint |
|---|---|---|
| 1 | Measure the four isolated mechanisms on the actual node | First official checks01,06,02; check04 separately when budget/hardware permits. Run a combined scored patch only after enough quality/TTFT evidence. Preserve failures and interactions. Three checks/day cannot validate every subset in one day. |
| 2 | Humming-style Hopper W4A16 scheduling | [vLLM priority source](https://github.com/vllm-project/vllm/blob/ef5f913afdf2af46559655633bf4c3f0960d14fc/vllm/model_executor/layers/quantization/utils/humming/priority.py) explicitly prefers Humming to Marlin at capability 90. [MoE adapter](https://github.com/vllm-project/vllm/blob/ef5f913afdf2af46559655633bf4c3f0960d14fc/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py) was downloaded; only selected support/configuration sections screened. Audit the actual NVFP4 kernel, packaging dependencies, scale/gate/post-norm contract and one-layout memory footprint before any port. No Humming speedup or compatibility established here. |
| 3 | Fuse/overlap layer operations and communication | [Meta Muse Mega](https://github.com/facebookresearch/sgl-muse-mega/tree/ddb7925ff45167250201f35743a19c5152254052) and [SGLang NVIDIA Mega](https://github.com/sgl-project/sgl-nvidia-mega/tree/750004efd8bc0db852cdd4efbebb807c73bedd4b) provide design leads: early loads, mailboxes, PDL and whole-step coverage. The latter's advertised model matrix is GB300/B300; a Muse directory does not demonstrate an H200-serving implementation. Need stream/order and capture proof; no blanket plugin enablement under different fixed flags. |
| 4 | Acceptance/cost-aware target verification of a prefix of the released 8-position draft | Retain the released drafter and target verification. Establish stock acceptance/verification costs and KV rollback/capture contracts before changing verification width. No unverified token emission, fixed validation-prompt routing or model change. Hypothesis only. |
| 5 | Request-derived prompt lookup | [Public DFlash lookup request](https://github.com/bstnxbt/dflash-mlx/issues/35), [Splash proposal](https://github.com/incoai/splash/issues/186), [adaptive-draft counterexamples](https://github.com/nathansutton/chad/blob/main/docs/configuration.md). Measure reuse on independent coding conversations, then target-verify every proposal. Different hardware and reasoning traffic can erase gains; this is established prior art. |
| 6 | Prefix-cache/HiCache transfer overlap | [Official SGLang deployment guide](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Meta/MuseSparkOpen.mdx). Profile actual hit rate, host-transfer stalls and TTFT before changing policy. Our flags already enable 160 GB HiCache; simply enabling an existing feature is not an optimization. |

## Rejected, constrained and negative evidence

- **Reuse the layer's Marlin workspace:** rejected before implementation. Weight
  preparation already allocates `layer.workspace`, and the final reduction
  slice resets its locks. However, the official generic Marlin runner explicitly
  allocates per-call storage to prevent captured graphs from aliasing those
  inter-block locks and deadlocking during capture. Reset-on-completion alone
  does not establish safe overlapping graph lifetimes. Source locations:
  `layers/moe/moe_runner/marlin.py:157`,
  `layers/quantization/marlin_utils_fp4.py:497`,
  `kernels/jit/csrc/gemm/marlin_moe/marlin_template.h:1906`.
- **Blindly transplant kernel-track code:** prior isolated MoE port duplicates
  weight layouts and has an unresolved full-model normalization contract. Its
  kernel timings do not establish model throughput. See MOE_PORT_STATUS.md.
- **Treat all public top scores as trusted serving results:**
  [keell0renz's README](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/README.md)
  disclaims a current#1 submission and separately lists historical kernel files,
  noisy repeated scores and a fresh-thread prefill issue. Read README/NOTICE
  and tree only; did not port or audit its device code. Historical ranks are
  neither current serving standings nor proof of correctness.
- [erkin98's README](https://github.com/erkin98/muse-open-kernel-challenge/blob/cf977def7ea72d870151b0610474e54d9c589479/README.md)
  was now read completely. It reports neutral/worse launch/grid/split/softmax
  experiments and only small increments over its credited upstream kernels.
  It supports testing fixed overheads and noise, not wholesale serving adoption.
- **Try official serving immediately:** attempted candidate 01 at 23:07:40 UTC,
  Nov 9. The service rejected it before returning a job handle: track opens
  Nov 12 at 9 am PT. Exact request/response preserved in
  OFFICIAL_CHECK_ATTEMPT_2026-11-09.json. Jobs remained empty. No GPU result,
  consumed-quota count or serving rank can be inferred from the rejection.

## Search/read coverage added in this pass

Six additional queries cover exact serving-check text, the new MoE module,
Marlin tiling, challenge name, H200 and DFlash. Public-filtered result counts
were5/63/75 code matches and5/15/13 repository matches. These are overlapping
search hits, not distinct entrants. No pagination was needed at those counts;
GitHub indexing and private repositories still limit coverage.

Three more exact public trees were inventoried: lee101/openkernels607 files,
sgl-project/sgl-nvidia-mega406 files, keell0renz9 files. Ten selected files were
downloaded/hash-verified. Read lee101's complete README and selected upstream
analysis sections; the Mega introduction/model matrix and selected matching
lines; keell0renz's complete README. Downloaded licenses/NOTICES are provenance,
not performance evidence. Together with the prior six trees this is nine
selected repositories, not a full audit of their1,351 files. The two vLLM files
are an additional upstream source screen, not a tenth entrant.
