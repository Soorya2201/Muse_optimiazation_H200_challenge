# Nine-instruction C12 decoding transferred to SK2

Canonical remains exact alygg77 v34, SHA-256
`290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90`.
This is a component-transfer experiment, not a new codec or a gain claim.
It does not reopen the closed TMA/cluster geometry or sub12-bit families.

## Source and mechanism

The current source already embeds Nathen Fernandes' MIT decodediet decoder
and uses it on pooler16. The older SK2 routes retain a12-instruction decoder.
This experiment transfers the embedded nine-instruction layout to SK2's
existing shared-memory/escape pipeline, retaining all geometry, stage counts,
MMA order, corrections, barriers and output precision. Attribution and full
MIT/Apache notices remain in the immutable parent prefix. No newly discovered
unlicensed implementation is ported.

The parent stores two signs at bits3/7 of each code byte. The new layout moves
the first sign to bit7 and the second sign to the next byte's bit3, wrapping
within each32-bit code word. Thus a rotate-right4 exposes the second group's
high bytes. Two masks, two additions and four PRMT instructions complete the
decode. Low bytes, exponent window, escape table and stream dimensions stay
identical. This is lossless repacking of static weights in prepare.

The [NVIDIA PTX PRMT specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)
documents byte selection; actual instruction cost is verified with NVCC13
sm90a, not inferred from PTX alone. Standalone old/new helpers use16registers,
zero frame/spill bytes. Their decode cores are12versus9SASS instructions,
including shifts/masks/adds/PRMT. This is25% fewer decoder instructions,
not25% faster kernels. Full-native resources and timing still need H200.

`C12_SIGN_PERMUTE_CODEGEN_2026-10-31.json` preserves complete requests and
responses plus223904CPU equivalence cases: every code byte in each word
position for every legal exponent base0–120, and100000random32-bit words.
The explicit inverse recovers every code bit; equal high bytes and unchanged
low-byte words/PRMT selectors prove equal decoded BF16 fragments.
An initial per-byte sign swap failed locally: rotate-right4 crosses byte
boundaries. It was corrected to the word permutation before compilation or
GPU use. No credit spent on that failed construction.

The live Nathen repository default is`solution`,head
`86aff75e6aca72511bf9b41b6dc190bf1e763373`,MIT,pushedOct24. Querying`main`
returned422; no missing-source conclusion is drawn. No newer donor was
established. Existing embedded component is the source of this transfer.

## Gain screen and fixed experiment

Nine actual parent routes are targeted:qkv/unpooler8,qkv/o/unpooler16,
qkv/o/pooler/unpooler32. The pooler8/16,O8,router and large-token routes keep
their existing implementations. Preserve qkv's balanced row order, the
M8independent correction path, and alygg77's distinct unpooler32 geometry.

`C12_PERMUTE_PREFLIGHT_2026-10-31.json` validates generated specifications
and hashes the125854-byte native source, archived compressed. Against scored
parent a3d4758c1e174aba,zeroing all nine cells gives a loose1.836550x board
bound. Assuming uniform5/10/15/25% latency cuts gives1.016023/1.032912/
1.050752/1.089710x planning proxies. No decoder phase fraction is measured;
instruction counts do not predict these time cuts. The broad target set
passes the initial >=3% opportunity screen; actual usefulness is uncertain.

V39`submission_v39_c12_permute_probe.py`,SHA-256
`99f9deba704a7bf85e46dfe0e162e2feff49db8a18902f8bf565251ff3ea0a38`,
keeps the exact parent prefix and timed dispatch. It compiles only operator
code at import, then prepares unchanged parent weights and runs an independent
diagnostic. No extra input/weight work is hidden outside prepare. Old/new
matched kernels share the identical weight-slot pointer: appropriate constant
packed bytes are copied outside event timing, followed by a cold-L2 flush
before every arm. These copies occur only in synthetic diagnostics; a
production implementation would repack weights once in prepare.

Four arms:actual parent,matched-old,new,duplicate parent. Four retained pads
and deepcopied states,two random/outlier seeds,five rotated cold-L2 rounds,
72timing groups. Both layouts are invertible and low bytes checked unchanged.
FP64 finite/global/per-row/one-hot checks apply; changed-activation graph
replay must change output and remain correct. Five nearby M7/9/17/24/31
fixtures validate the same native configurations. Full resource getters
cover both variants of all nine specialized kernels. All scratch/graphs/
fixtures/cloned states leave scope; original state is returned untouched.

Gates declared before H200:ordinary output<1% global/<5%per-row and one-hot
<0.1%;matched/parent differences<0.02%global/<0.2%row. Every component must
beat actual parent AND matched-old by>2% across all seeds/pads,duplicate
offset<2%,matched control within3%of actual parent,and zero local allocation.
Any adopted fixed composition must model>=3%board gain in its worst group.
Only qualifiers receive actual timed-path checks and independent repeats.
Do not relax gates after failure or score an unchanged diagnostic parent.
Compressed final report must stay<8KB; preserve every raw event timestamp.

## Public search refresh

`GLOBAL_CODE_SEARCH_2026-10-31.json` records244paths/13repositories,up from
234. Ten additions are from Kushagrabainsla at0778ecd:five BF16 files and
five decoder references left for that track. Owner positions and exact blob
links are archived. This is an index screen, not exhaustive source review or
access to private winners. Detailed BF16 source inspection is separate from
the independently motivated embedded-decoder transfer above.
That inspection is now archived in`KUSHAGRA_SOURCE_SCREEN_2026-10-31.json`:
all five Git blobs/SHA-256 hashes verified,default=E14 exact bytes,primary
experiment notes read. Owner#1025vsours#229atcapture;source uses older
Triton/cluster/C12 families and reports no new faster replacement component.
License remains unestablished;no implementation executed,ported or mirrored.
