# Related public GPU work from the prefill leader

The Nov1 19:03 UTC OKC snapshot identifies `GodlyDonuts` as prefill leader
at2.2815x; our scored result is1.784958x. This document follows the owner's
request to study an entrant's other technical work when the winning source
is not public. It is a bounded public-source review, not an exhaustive audit
or access to a private submission. It does not establish how the winning
kernel was implemented.

| Primary public source | What was inspected | Transfer to this challenge |
|---|---|---|
| [Psi CUDA backend, exact7774221](https://github.com/GodlyDonuts/psi/blob/777422135fc6d8a0e500261ff6ed8d6208608a63/src/step4_cuda/cuda_backend.cu) | Complete CUDA backend; selected CUDA/Metal design notes and README | The inspected backend offers FP32 cuBLAS or16x16 shared-memory SGEMM, with host copies per call. Our implementation already keeps work on the GPU and uses Hopper tensor cores. No NVFP4 grouped-expert replacement found in this file. |
| [Psi GPU design notes](https://github.com/GodlyDonuts/psi/blob/777422135fc6d8a0e500261ff6ed8d6208608a63/docs/GPU_KERNELS.md) | Selected vector-load, register-blocking, padding and rejected-variant sections | Useful design questions, but measurements are Apple Metal. Our native path already has vector loads, swizzled shared operands and staging padding. Ternary conversion would change required weight precision. No code port. |
| [Shohin H100 training baseline, exactdc0cd26](https://github.com/GodlyDonuts/shohin/tree/dc0cd269e7384cbf03c894996a6c3dbcb5b10d1f) | README, architecture map, model attention call and trainer compile/precision setup | The inspected path uses PyTorch SDPA, BF16 autocast, compilation and DDP. Training throughput is not matched single-H200 kernel evidence. Distributed training and data-loader changes do not replace this kernel workload. |
| [Shohin-ettr, exacta7e1880](https://github.com/GodlyDonuts/shohin-ettr/tree/a7e18807ce78c419917858845add62bdf987c380) | Public README and tree metadata only | Learned model revision and sparse-model accuracy transfer concern a different objective. We cannot change the model or routes to manufacture a kernel speedup. No full research-archive audit claimed. |
| [Authored PyTorch global-scratch change42a7078](https://github.com/pytorch/pytorch/commit/42a70784390a343ebd90e9e84b1bf5a5df4901f3) | Primary change summary and selected static-launcher/test diff | Preserves Triton TMA scratch size/alignment through static launcher restoration. Relevant to allocation/ABI correctness; no measured native expert-loop gain. Do not patch loaded runtime modules in a challenge submission. |
| [Author's public projects page](https://saicharanramineni.com/projects) | Professional project descriptions and source links | Supplies provenance for these related projects, not a winning Muse submission or an H200 optimization recipe. |
| [CUDA-Python fork](https://github.com/GodlyDonuts/cuda-python/tree/5078fe6032d2193eb7c502b91a1ad7b623b1db06) | Default tree, branch names and upstream comparison | Default is0ahead/117behind NVIDIA/main with no changed files in that comparison. Other branch contents were not exhaustively reviewed. |

Thirteen selected source/doc files were fetched at exact commits and their
Git-blob/SHA-256 hashes verified. No root license was established for the
personal project repositories. Their source stays in temporary research
storage; Git retains our summaries, URLs/hashes and review scope. No downloaded
module was imported/executed or copied into our candidate. Contact details
and unrelated personal information are not part of this research record.

A public author-attributed CUDA/Triton/kernel/TMA PR metadata query returns
50results, including compiler/runtime correctness and portability changes.
The narrow Muse/openkernel/H200 query returns one torchtitan initialization
PR, not a published challenge submission. This search is not proof that no
unindexed branch, blog or public source exists. Query/results are preserved
in LEADER_PUBLIC_TECHNICAL_SEARCH_2026-11-01.json. Repo/source metadata and
bounded read scope are in LEADER_RELATED_METADATA_2026-11-01.json.

The actionable distinction is between a technique and an additive gain:
GPU residency, vector loads, shared padding and precision checks already
exist in our current code. Applying the same named technique again does
not establish a faster implementation. H200 checkcfc76057d5d641b7
tested a specific unresolved dependency schedule: four wide RS-MMAs
per commit group, distinct stage-parity register banks, original data layout.
Its protocol and results belong in WIDE_BATCH_RESEARCH_2026-11-01.md.
It failed a nearby numeric fixture before timing; the result and concrete
compiler-lifetime repair follow-up are recorded there. No speed gain is claimed.

Additional branch inspection: six named `codex/` branches of the CUDA-Python
fork were compared against NVIDIA/main, rather than stopping at the default
branch. Three concern documentation/binding references, two concern CI,
and the incremental-LTO branch changes linker-cache handling. Its three-file
displayed diff was read: it disables intermediate caching for certain old
incremental-LTO versions. Our compiled kernel does not use that Python linker.
The other branch file inventories were screened, not all source lines.
This is not an audit of the remaining fork branches. Exact heads, comparison
URLs, paths and hashes are in `LEADER_CUDA_BRANCH_SCREEN_2026-11-01.json`.

A related inference project from former decoder/overall leader `mpeng19`
was also screened: [auto-inference at f28ff938](https://github.com/mpeng19/auto-inference/tree/f28ff938446fe214e7e44452a6008314e9db0589).
Its complete README and tree, plus titles/first280 mechanism characters of
27 idea records, were inspected. Our initial scope classification finds five
kernel implementation ideas, eight serving/workload controls, and fourteen
ideas that change the fixed precision, attention/cache semantics, workload
or input reuse when applied as described. The five kernel records concern
query-group reuse, cache layout, asynchronous movement, shape tuning and
attention/projection fusion; several already exist in our implementation.
The reported H100 serving-price improvement is not a Muse H200 result.
Its full methodology, complete idea entries and generated implementation
runs have not been audited. No harness was run or source ported.
`RELATED_INFERENCE_IDEA_SCREEN_2026-11-01.json` stores exact provenance,
record-index classifications, read scope and limits.

The five potentially applicable idea records were subsequently read in full,
including prerequisites and risks. Their benefit claims are hypotheses:
uncited bandwidth percentages, generic tuning multipliers and claimed
reliability do not constitute measurements of this parent. Our assessment:

| Idea family | What still needs to be established here |
|---|---|
| Reuse KV across grouped query heads | Already present in our grouped tensor-core attention; a new tile must outperform the current grouping |
| Wider/coalesced KV movement | Existing vectorized/swizzled accesses and fixed external cache format constrain changes; need measured wasted transactions |
| Async movement/cache residency | Existing staged TMA already overlaps loads; judge flushes L2 before replay, so retained-cache assumptions do not transfer |
| Shape tuning | Existing fixed dispatch and prior sweeps cover many cases; only a new configuration with a causal mechanism warrants another sweep |
| Fuse attention with output projection | Cross-head dependencies, mandatory BF16 rounding and loss of tuned dense GEMM can outweigh saved traffic; no matched implementation or credible measured gain yet |

The saved prefill leader/current ratio is1.2782x: matching it would require
about27.8% more throughput, or21.8% less whole-layer time on the scored mix.
A repeated one-percent component change cannot close that gap by itself.

The corrected V34 H200 experiment is now terminal. All five nearby fixtures
and24 groups pass with exact initial/fresh agreement, but no component or
fixed composition meets the performance gates. The largest fixture is
about2.3% slower. The tested form is rejected; scored source and scores are
unchanged. `WIDE_BATCH_RESEARCH_2026-11-01.md` preserves the failure, repairs,
raw results, limits and next unimplemented possibilities.
