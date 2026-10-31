# New BF16 lossless-compression sources and transfer decisions

Current public-code refresh returns244paths/13repos, zero new blobs.
Direct HEAD queries confirm the existing Nathen86aff75, alygg776cae49a,
vjeuxf35a41e and lossless-bf164429b1d heads. Current leader arashattar's
public owner-repository query returns0. These are bounded public discovery
results, not access to unpublished/private submissions.

## NWC

[parda21/NWC](https://github.com/parda21/NWC/tree/648b35ff74f1456e9c7912aac4c044ce8cdc47d3),
Apache-2.0: eight selected files179515bytes pinned with verified Git blobs,
SHA-256 hashes and original license. Reviewed format docs, selected native
decoder/layout2 bodies, measurement limitations and the CPU-reference source;
no downloaded module executed or kernel ported.

Its prefix decoder uses per-lane streams and a16KiB pair table; the optional
tensor-core layout feeds BF16 fragments. Layout2 replaces variable decoding
with fixed nibble ranks and exact exception lists. It remains12bits plus
metadata, so it establishes no new byte advantage over our C12 parent.
The authors' H100-SXM figures are projections, not measurements; the library
has a second finalize kernel and its token path is batch1. Published4070/A16
results use trained weights and different baselines. Direct adoption would
need BF16-input adapters, multi-token execution, exception/one-hot checks
and matched H200 measurements. No additive gain is established.

## ZipServ

[HPMLL/ZipServ_ASPLOS26](https://github.com/HPMLL/ZipServ_ASPLOS26/tree/6f8a209bfe46b0905f90d61b0796e352147ff041),
Apache-2.0: nine selected files130752bytes pinned and hash-verified, including
the license. Read the paper's format/decode/workflow sections and selected
compressor, actual SingleRow decoder and launch/tiling paths. The initial
HKUST PDF URL timed out; the [author paper on arXiv](https://arxiv.org/html/2603.17435v1)
was accessible. No downloaded code executed.

The format uses three bitmaps with packed sign/mantissa and exact fallback
values. The selected active decoder receives the exponent base; an older
hardcoded-base helper has no call site in the inspected file. Native kernels
stage compressed operands then reconstruct BF16 register fragments for
mma.sync; alignment padding, two-level offsets and split-K tails count
against nominal storage savings. The repository explicitly targets GDDR
GPUs. It supplies no matched H200 win over our current persistent/TMA/C12
portfolio. A direct large-M replacement would also replace WGMMA/pipeline
and reduction choices, so headline ratios cannot justify a wholesale port.
Preserve the format as a research lead; no new GPU decoder is implemented.

## NVIDIA allocation alternative selected for this cycle

[NVIDIA cudaCompressibleMemory](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/3_CUDA_Features/cudaCompressibleMemory),
BSD-3-Clause: six files18704bytes pinned with original notices and verified
against their Git tree. The sample demonstrates capability/property checks,
granularity rounding, VMM map/access setup and cleanup. Our independent
wrapper adds shared ownership, stored-context guards, transactional owner
cleanup and a fixed-VA normal/compressible comparison. Its native binding
and actual device capability remain untested before the planned check.

[ILC_RESEARCH_2026-10-31.md](ILC_RESEARCH_2026-10-31.md) records the exact
candidate, local corrections, byte-copy proof, non-free opportunity screen
and predeclared H200 protocol. Hardware compression changes allocation
properties while retaining all mathematical kernels/weight bits. Its benefit
is unknown; this is not a software codec port or a novelty claim.

All23selected files328971bytes retain licenses and exact URLs/commits in
their `research/public/*/PROVENANCE.json` files. The earlier untracked
Madhavan research directory is untouched. No emails, issues, stars or
messages were sent to repository owners.
