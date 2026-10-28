# C12 decode-to-shared cluster GEMM

Status: pack/decode/patch/swizzle contract is CPU-proven; CUDA kernel is not implemented or
measured. Build it only after exact v18, v30, and the v21/v22 uncompressed cluster candidates
establish current-pod baselines.

## Why this is the next architecture

The two strongest public large-m approaches each solve only half of the problem:

- Nathen v18's m128 qkv route, reused by v21/v22 for pooler/unpooler, TMA-loads BF16 weights into
  shared memory, runs shared/shared WGMMA, and reduces split K through cluster DSMEM. It avoids the
  measured 8-9.5 us global workspace/counter tail and is fast, but streams two bytes per weight.
- Madhavan's public `wide_kernel` streams lossless C12 weights at 1.5 bytes per weight, decodes them
  into registers, and uses register/shared WGMMA. Its judged artifact takes 15.24 us qkv, 22.66 us
  pooler, and 19.46 us unpooler at m128. Exact v18 is already faster at 13.99, 20.49, and 19.37 us.
  The compressed path also uses a global partial workspace and counter when split K is required.

The proposed kernel combines the missing halves: stream C12 from HBM, expand it into a swizzled
BF16 shared-memory tile with a dedicated decoder warpgroup, then retain shared/shared WGMMA and
the v18 cluster-DSMEM epilogue. This is materially different from another geometry sweep.

## Pipeline

Each CTA owns 64 output rows. Pooler uses a two-CTA cluster splitting K; unpooler initially uses
one CTA with no reduction. One ring slot contains:

1. 6,144 bytes of C12 weights for a 64x64 K tile;
2. an 8,192-byte decoded BF16 64x64 tile in WGMMA's 128-byte-swizzled layout; and
3. `TM * 128` bytes of activation for the same K tile.

The producer warp issues a bulk copy for the packed weight tile and TMA for x. A decoder
warpgroup waits for those transfers, decodes eight weights from 12 bytes at a time using the
existing `lop3`/`prmt` method, applies rare exact patches, writes the BF16 swizzled tile, and
arrives on a decoded-stage barrier. The consumer warpgroup performs shared/shared WGMMA on the
previous decoded stage. A slot returns to the producer only after its WGMMA group completes.

Use three barrier rings per slot: packed-ready, decoded-ready, and empty. The steady state is
producer(N+1), decoder(N), consumer(N-1). There is no timed host work and no input-derived cache.

## Exactness and exception format

C12 stores each weight's raw low byte and a four-bit sign/exponent-window code. The decoder
reconstructs the exact BF16 value except for the roughly 0.007-0.014% public-weight exceptions.
Prepare groups exceptions by `(row block, K64 stage)` and stores a bounded packed patch list.
Because the low byte is already exact, a patch needs only a 12-bit tile-local index and the XOR
for the reconstructed high byte; one 32-bit record is sufficient. Reserve eight records per
stage and reject overflow during prepare. The decoder applies patches to the decoded shared tile
before the decoded-ready barrier, so there is no output correction tail.

Prepare must reject a route on patch-list overflow, packing mismatch, one-hot mismatch, random
relative error, launch failure, or insufficient timing margin. The original route remains the
fallback. Exact reconstruction should also be exhaustively emulated on CPU before an H200 check.

That first gate is complete in `tools/c12_cluster_emulator.py`. With seed 5151 and one exponent
base calibrated across 64 tiles then held fixed, it passed all 65,536 BF16 bit patterns, 20,480
patch-field boundary cases, 4,096 normal-like 64x64 tiles, adversarial values, and the 128-byte
swizzle roundtrip. The normal-like tiles had median zero, p99 two, and maximum four patches; none
exceeded the eight-record capacity. This proves the proposed 6,144-byte input and 8,192-byte
decoded layout in CPU semantics, not CUDA instruction mapping or H200 performance.

`tools/c12_cluster_decoder.cu` is the matching standalone device contract: one 128-thread
warpgroup decodes four groups per thread directly into the swizzled tile, then applies up to eight
packed high-byte XOR patches in place. It exists to compile and inspect the exact Hopper device
mapping before that code is embedded in the cluster pipeline; it is not a timed kernel candidate.
Compiler Explorer's NVCC 13.0 accepted it with `-O3 -std=c++20 -arch=sm_90a -c -Xptxas=-v`:
22 registers, one barrier resource, zero stack, and zero spill loads/stores. A separate PTX build
confirmed shared-memory loads/stores, all three `bar.sync` points, and retained `lop3.b32` and
`prmt.b32` instructions. This compile proves device syntax and resource allocation only; it does
not prove runtime correctness, barrier ordering inside the future three-role pipeline, or latency.

`tools/c12_cluster_pooler128_source.py` now emits the first complete compile-only pipeline:
six packed/decoded/x stages, bulk packed-weight copies, TMA x loads, the 128-thread decoder,
shared/shared WGMMA, and the two-CTA cluster-DSMEM epilogue for pooler m128. NVCC 13.0 requires
the explicit `-gencode=arch=compute_90a,code=sm_90a` pair for its WGMMA PTX target; with that flag,
PTXAS reports 122 registers, three barrier resources, zero stack, and zero spills. This proves all
device pieces instantiate together, but the source still lacks its Torch wrapper, pack-time tensor
construction, GPU correctness tests, and timing gate.
The audited generated-source SHA-256 is
`7ca033c0bc8860edc84a90c46bda38af921049fa4f6df6cbbbce21954a7d34ce`.

## Static H200 resource bound

For the first pooler route (`NR=64`, cluster split two), the proposed stage counts fit the 227 KiB
opt-in block limit:

| m | stages | bytes/stage | DSMEM reduction | modeled total | headroom |
|---:|---:|---:|---:|---:|---:|
| 128 | 6 | 30,720 | 17,408 | 202,912 | 29,536 |
| 256 | 4 | 47,104 | 33,792 | 223,344 | 9,104 |

The total includes 1,024-byte alignment, three 8-byte barriers per stage, a small fixed header,
and the four-float padding needed by each remote accumulator row. The eight patch records and
one-byte count stay in their small global per-tile table and are consumed by the decoder before
the decoded-ready barrier. Unpooler has no DSMEM reduction and therefore more margin.
The m256 pooler case is tight: source generation must compute the exact dynamic allocation and
reject anything over the device-reported opt-in limit. Do not add another full decoded stage.

Thread count is 288: one 128-thread WGMMA consumer group, one 128-thread decoder group, and one
32-thread producer warp. Large shared memory already limits the kernel to one CTA per SM, so the
extra decoder group does not reduce block residency.

## Score ceiling and rollout

Across all eight non-router m128/m256 shapes, uniform 5/10/15/20/25% time reductions reconstruct
v18 scores of 1.54212/1.56720/1.59399/1.62272/1.65364x. Thus this architecture can cross the
1.6367x leader only near a broad 25% win, or in combination with measured small-m winners from
v30/v31. C12 cuts weight HBM bytes by 25%, but added shared traffic and decode work mean 25% time
reduction is an optimistic ceiling, not an estimate.

Implementation order after the reset measurements:

1. Pooler m128 only, based on the measured winning v21/v22 pipeline and stage policy.
   Preserve `tools/c12_cluster_emulator.py` as the pack/decode contract for the CUDA implementation.
2. Compare decode-to-shared against both v18 and the winning uncompressed cluster route with cold
   L2 graph replays; require the existing 1% and 0.08 us gate.
3. Add unpooler m128 without split K.
4. Add m256 only after exact resource and register reports pass; pooler has only about 10 KiB
   modeled shared-memory headroom.
5. Extend to qkv and o only after the first two layers prove that shared decode overlaps WGMMA.
6. Merge measured winners into the v30 portfolio; never replace a faster route speculatively.

The main failure signal is decoder back-pressure: if packed-ready stages accumulate while the
consumer waits on decoded-ready, the integer/shared expansion cannot hide. In that case test two
decoder warps before a whole warpgroup, and measure stage timestamps; do not restore the rejected
global workspace reduction.
