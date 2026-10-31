# K64 scale reuse while preserving the original activation layouts

This is a new native implementation and CPU/compiler result, not an H200
measurement or a qualifying submission. The corrected ordinary SS probe
dad7baa049c943dc is terminal: numerical/resource gates pass, but every timing
group loses. Its data do not justify the final routine prefill check for
this follow-up. Canonical/scored sources remain unchanged.

The earlier K64 implementation permuted X during gather and permuted the
GEMM1 intermediate after stmatrix. Those timed costs offset scale reuse.
With producer-owned expansion, the shared weight stores can undo the weight
K permutation instead: every matrix operation still consumes standard-order
BF16 X and A. Only the prepare-time lossless NVFP4 repack changes. For lane
t, physical K16 steps j map to original columns16*t+4*j; each step's high-K
pair lands two columns later. Two consecutive steps therefore form eight
consecutive original values and can be emitted by an aligned16-byte store.

`tools/prefill_ss_k64_native.py` constructs one scale pair per logical
consumer fragment instead of four. For each64-thread producer worker stage,
64scalar stores become16aligned vector stores, with the same256output bytes.
No X/gather/GEMM1 permutation or extra global pass is introduced. All packed
code/scale words remain in registers before the existing64-thread overwrite
barrier. The corrected explicit worker bounds and SS readiness protocol
remain unchanged. This combines prior KPERM scale reuse with a new staging
path; it is not a claim to invent KPERM or producer/consumer specialization.

`verify_prefill_ss_k64.py` independently builds the actual parent's packed
bit layout, selecting permuted original K indices and scales. Across all64K
blocks, all524288symbolic BF16 destinations,4096code/scale-byte pairs and
65536aligned vector writes match the original standard-layout operand after
randomized worker order. This is a byte/assignment proof, not floating-point
WGMMA or concurrent GPU validation. The full native source/diff are archived.
It requires the existing exact `_ks_repack` from V27; a complete independent
GPU probe with that actual repack has not yet been built.

Full FINE0 and FINE1 NVCC13/sm90a builds both report168registers,8-byte ABI
stack,zero spill stores/loads and noC751x warning. FINE0 disassembly is
complete (4357returned rows), and its comparison to the corrected ordinary
SS kernel (4469rows) confirms the following static opcode sites:

| Opcode | Ordinary SS | SS + K64 |
|---|---:|---:|
| STS |119|55|
| STS.128 |0|16|
| PRMT |19|7|
| LDS.64 |6|2|
| LDS.U16 |0|4|

FINE1 SASS is clipped at5000rows, so no whole-view count comparison is made
there. These are static sites over whole FINE0 kernels, not dynamic work,
physical memory requests, cycle savings or evidence of a producer bottleneck.
The prototype does not remove the extra SS operand traffic or KPS1 cost on
narrow tiles. Conditional opportunity figures and unchanged numerical/
performance gates remain those in SS_PIPELINE_H200_2026-10-31.md.


Terminal decision: defer before H200. Ordinary SS takes~31%longer at16384
and loses more at753/3012. The same-job phase proxy requires about1.69x
extra improvement over SS for first-turn-only3%board gain. No causal
producer-time measurement or complete integration establishes that recovery.
This is not an impossibility proof or a measured K64 timing failure.
