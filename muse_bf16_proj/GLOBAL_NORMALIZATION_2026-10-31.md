# Global exponent normalization in the current SK2 path

Canonical/scored source remains SHA-256
`290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90`.
This cycle tests additional arithmetic normalization, distinct from the
closed sign-layout-only transfer. No gain or rank improvement is established.
The preceding attention cycle made CPU/compiler/source progress, pushed8397194.

## New public research and applicability

Fresh indexed GitHub search:244 BF16 paths/13 repositories,zero new blobs
versus the earlier Oct31 snapshot. This is indexed coverage,not access to
private submissions or proof every entrant publishes code.

[Cloudflare Unweight](https://github.com/cloudflareresearch/unweight-kernels/tree/db0e6d3d4505cf71379f6b9b4567acf598bc186c)
is newly pinned at db0e6d3:17source/license/build files,113,104bytes,all Git
blobs and SHA-256 hashes verified. BSD-3-Clause notice retained. Its
[technical report](https://research.cloudflare.com/papers/unweight-2026.pdf)
describes separate Huffman and palette representations and reconstructive
Hopper matmul. Its preliminary H100 end-to-end table reports a throughput
penalty,including40.7% at batch1; its compression percentage is not a
kernel speedup. These are author measurements,not our H200 replication.

Our static source review finds that the direct palette path still streams
one sign/mantissa byte and one exponent nibble per weight,plus verbatim
64-value row data and metadata. Runtime Huffman preprocessing would add
another kernel and exponent/palette write/read traffic. The reconstructive
matmul consumes decoded exponents or palette indices; it does not directly
consume the Huffman bitstream. Its128-row M tile and whole-K output-tile
ownership can underfill this problem's small-M/skinny-N grids. It pads M
in the tensor descriptor but does not supply our complete fresh-input padding
adapter. No whole-stack port or untested performance claim is made. The
repository was statically inspected; no downloaded module imported/executed.

`tools/screen_unweight_muse.py` additionally verifies all pinned file hashes,
models78legal Muse/variant geometries,and independently checks exact
sign/mantissa/exponent storage reconstruction on CPU Gaussian BF16 samples
(256rows at each K1024/4096/8192). Direct palette storage is0.46–0.63%
larger than an optimistic C12payload+global8-byte-escape model. Huffman is
0.94–8.22%smaller in these samples including its32KiB table,row offsets,
metadata and verbatim rows;these are distribution estimates,not judged
weight hashes or physical memory traffic. Separate exponent decoding models
about3.327logical bytes/weight across reads/writes before activation/output
traffic,caches or metadata. This does not predict runtime. Complete
histograms,code lengths,raw byte terms and shapes are preserved in
`UNWEIGHT_FORMAT_SCREEN_2026-10-31.json`. Larger matrices amortize table
storage differently. A new fused entropy decoder is not implemented here.

The existing Apache-2.0 MrNirmata/lossless-bf16 fixed-window component
(pinned4429b1d) uses per-row normalization and escapes. This experiment
instead retains our parent's matrix-global base, byte streams, exact
original escape records and correction pipeline. It restores each main
partial to original units before any original-unit corrections are added.
Nathen Fernandes' MIT sign permutation,already embedded in the parent,is
retained with its notices. No novelty claim is made for lossless exponent
normalization or the seven-instruction fixed-window decoder.

## Arithmetic and parameter layout

An ordinary decoded high byte is sign OR(B+offset),where offset is0..7.
Using fixed B=56 shifts every directly decoded BF16 exponent by
`s=2*(56-B)`,while preserving sign and mantissa. The same packed low bytes
and offsets therefore represent normalized BF16 weights. The original base
remains metadata,so the storage transformation is invertible. Main MMA
partials are multiplied by exact FP32 `2**(-s)` after their NACC reduction,
before shared reduction or original-unit escape corrections. No activation
dtype or original weight significand changes. No new correction stream.

The naive C++ fixed-window expression unexpectedly compiles to9 decoder
instructions,not7:mask and OR remain separate because both masks are
immediates. V42 preserves this unsubmitted prototype and compiler evidence.
It is not sent to H200.

V43 stores the original base in otherwise ignored bits of a kernel scalar:
`p=0x38383838 | (B&7) | ((B&0x38)<<5) | ((B&0x40)<<1)`.
Thus `p & ~0x87878787 == 0x38383838`. One PTX ternary selection emits each
normalized high-byte word while ignoring the metadata bits. The decoder
uses one rotate,two LOP3,four PRMT. The escape warp extracts the original
base and retains its original correction expression. Base recovery and
four FP32 restore multiplies occur outside the repeated decode loop.

`tools/bf16_global_normalization.py` supplies both generated versions;
the V1 snapshot matches the first math report's original module hash.
CPU checks prove exact recovery across487,424decoded normal BF16 patterns,
all119supported global bases,243,712ternary-word cases and512bounded
256-term scalar FP32/libc-fmaf dot fixtures. NACC reduction and unchanged
original-unit correction order are checked. The FP32 homogeneity claim is
conditional on no intermediate underflow/overflow/FTZ crossing;GPU MMA
behavior and extreme BF16 inputs remain unproven. Bases outside1..119 are
not admitted to this experimental diagnostic.

Both complete NVCC13/sm90a compiler requests/responses are archived in
`GLOBAL_NORMALIZATION{,_LUT}_COMPILER_2026-10-31.json.gz`.
Old/normalized standalone decoders both use16registers,zero stack/spills.
The final normalized core emits7instructions versus old12. The restore
kernel is separate compiler evidence,not an extra production GPU launch;
restore operations are integrated in the actual SK2 epilogue. Full-native
resource use and runtime still require H200 validation.

## Opportunity, fixed protocol and gates before H200

The nine exact parent routes match the previous sign-transfer route set,
but use new normalization and metadata logic. All geometry/stages,weight
bytes/addresses,MMA issue order,escape records,synchronization and output
precision otherwise remain. The canonical Python/timed entry is an exact
prefix. The experiment runs only in an independent prepare diagnostic.

Same-job b89f4e5d007047d9 load modeling gives a loose1.826841x phase-zero
ceiling. Uniform5/10/15%target-cell cuts model1.015931/1.032719/1.050450x;
3% board gain requires9.2082%uniform target-cell time reduction. These
are selection scenarios,not predictions. The earlier9-instruction transfer
found no eligible gain;memory bandwidth,restore overhead and register
pressure can defeat this additional reduction too. One bounded check tests
the new mechanism;no coefficients,geometry or accuracy-gate sweep is planned.

`GLOBAL_NORMALIZATION_PREFLIGHT_2026-10-31.json` executes source generation,
verifies all nine unchanged specializations and238actual host-call metadata
setups,archives127,010native-source bytes,and hashes the immutable V43file:
`7967d1c4492558893b9c95d605f3193e78067f51f6d8cf27839fe9bf2c095e65`.
The first local preflight omitted a static config containing a named constant;
the assertion failed before GPU use. The corrected restricted AST evaluator
recovers the exact original configurations and passes. No credit spent there.

Reuse V41's corrected protocol:four arms(actual parent,matched old,new,
duplicate parent),identical old/new weight-slot address,copies outside
events,read-only256MiB cache eviction,eight balanced graph rounds,explicit
inside-graph event nodes,all raw durations,selected simultaneous host-event
calibration,two seeds/four retained allocation pads,72groups. Five nearby
M7/9/17/24/31 fixtures,FP64 output/one-hot checks and changed-input graph
replays remain. The complete compressed report must fit8KB;no dropped pads.

Fixed gates remain:FP64 global<1%,row<5%,one-hot<0.1%;parent agreement
global<0.02%,row<0.2%;finite outputs. Each eligible component must beat
both actual and matched controls by>2% in every group,A/A offset<2%,matched
control offset<3%,zero full-native local allocation. Any fixed combination
must model>=3%board gain in its worst group. No gate relaxation. Only a
qualifying path proceeds to real changed-dispatch challenge checks and an
independent repeat before promotion/submission.

Budget before check:BF16 routine3/8,all scored submits0Oct31 UTC. Preserve
the owner's reserve. Log any job immediately in RESULTS.md,poll the same
handle through transient failures,and inspect every passing profile/log.

Check `3aa502d1eb8f4854` launched20:26UTC,cluster sfo3,fourth routine
check. Its current status is running;no terminal accuracy/performance result
yet. Source is immutable. Do not start a duplicate run.
