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
not establish a faster implementation. The live H200 checkcfc76057d5d641b7
instead tests a specific unresolved dependency schedule: four wide RS-MMAs
per commit group, distinct stage-parity register banks, original data layout.
Its protocol and results belong in WIDE_BATCH_RESEARCH_2026-11-01.md.
We will accept or reject it from same-GPU correctness and paired timing.
