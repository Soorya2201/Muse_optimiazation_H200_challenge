# Tiny NVFP4 pair lookup: capacity works, this codegen does not justify a check

No OKC check or scored submission was made. Canonical source and its
2.372309331x scored result are unchanged. This is a CPU/compiler screen,
not a full integrated MoE candidate or measured runtime loss. The previous
goal turn only refreshed ranks; this cycle completes the unfinished screen.

## Research and scope correction

[FLUTE](https://arxiv.org/html/2407.10960v2) describes offline packing,
pair lookup and duplicated shared tables. Its A100/A6000 benchmarks with
different quantization settings do not establish a gain over our H200
E2M1-specialized fused parent. The already pinned Apache-2.0 source remains
at `9eb83a12d56949bbe7fe9c836ba97a67bd1e3761`; its config supports
8/16/32 duplicated vectorized tables. Fresh primary metadata confirms
that head. No downloaded module was imported or executed.

The previous capacity rejection concerned the **main WGS** allocation.
V34's complete tiny control/trace reports **214784 dynamic bytes**. Its
failed timing-attribution gates do not invalidate that resource report.
Under our existing 232448-byte block cap:

|Pair table copies|Table bytes|Tiny total|Main WGS total|
|---|---:|---:|---:|
|1|1024|215808, fits|232192, fits|
|8|8192|222976, fits|239360, exceeds cap|
|16|16384|231168, fits|247552, exceeds cap|
|32|32768|247552, exceeds cap|263936, exceeds cap|

Thus a 16-copy table fits the measured tiny build without changing its
shared-memory-limited one-CTA residency. This says nothing about other
M8/MID builds, fused register usage or a whole-kernel speedup.

Fresh public code index `GLOBAL_CODE_SEARCH_LATE_2026-10-31.json` has
217 paths/13 repositories and zero new path/blob entries. A bounded owner
metadata query for current MoE leader `dolanzhao` returns 15 public repos,
most recently pushed in February 2025. Their listed metadata does not
identify a challenge submission. Sources were not exhaustively audited;
private or unlabelled work remains unknown. Exact URLs and metadata are in
`TINY_PAIR_LUT_PUBLIC_METADATA_2026-10-31.json`.

## Exact operands and bank model

A table entry contains two unscaled BF16 operands with raw bits
`((code & 8) << 12) | ((code & 7) << 6)`, including signed zero.
The original BF16x2 scale multiplications remain. A new natural byte pairs
the two nibble codes for each output register. The pure parent packing
map is bijective over all 32 bits. CPU proofs cover all 256 pairs in each
of four byte positions plus 100000 seeded mixed words: **101024 cases**.
All four raw output words match the parent's DQ7 decoder. This is not a
proof of a new full weight repack or GPU execution; both are unimplemented.

Duplicated entries must be interleaved as `table[index*D + lane%D]`.
Copying contiguous 1 KiB slabs does not provide the intended bank layout.
The [NVIDIA shared-memory guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
specifies consecutive four-byte words in successive banks and same-address
broadcasts. Our model deduplicates equal words within each bank. Tight
worst-case unique-word requests per bank for D=1/8/16/32 are **8/4/2/1**;
explicit adversarial indices attain each bound. The original preflight's
`32/D` field is a valid but loose upper bound at D=1; the immutable archive
is preserved and the analysis corrects it. 80000 synthetic random warps
are also checked. Requests are not measured device cycles or real-weight
statistics. No H200 bank-conflict counter was collected.

## Full standalone compiler result

The saved request compiles both diagnostic entry kernels and non-inline
decoder cores with NVCC13, `-O3 -std=c++17`, sm90a. Exit code0; device
PTX453 rows and SASS147 rows are complete, with no truncation marker.
The table kernel initializes all 4096 words before a CTA barrier. The
comparison below excludes that initialization, the caller and call overhead.

|Static core sites|Parent arithmetic|16-copy table|
|---|---:|---:|
|Decode/address integer operations|9|15|
|Dependent shared reads|0|4|
|BF16x2 scale multiplications|4|4|
|Device ABI/return operations|3|3|
|Total through return|16|26|

Three table address operations construct the lane-specific origin and
could be hoisted in a full kernel. Even granting that hypothetical saving,
this compiled table fragment has 12 address operations plus four LDS,
versus nine integer decode operations, before identical scaling. Static
sites do not equal cycles and the scheduling/resource context would change
under inlining. The result fails the proposed fewer-instructions premise;
it does not prove that every lookup implementation must lose.

Standalone parent/table entries use14/17 registers, zero stack/spills;
table entry uses16384 static shared bytes and one barrier. These are not
full fused MoE resource attributes. No numerical GPU execution was made.
Complete source, request, response, stderr, core instructions, hashes,
finite-table witnesses and scope are retained in the adjacent `.cu`,
`CODEGEN*.json.gz`, `PREFLIGHT*.json`, and `ANALYSIS*.json` files. See
`TINY_PAIR_LUT_NOTICE_2026-10-31.md` for original attribution and license.

## Opportunity gate and next action

With five MoE loads unchanged, 3% whole-board gain requires the one-token
case to improve by `1.03**6 = 1.194052x`: a16.2516% time cut, or4.174518us
from its scored25.686852us. Prior tiny traces failed attribution gates;
they provide no validated removable decoder budget. This table form adds
address work, dependent LDS and per-CTA initialization. No practical
4.17us recovery argument is established. **Defer this form before H200**;
do not spend the reserve or automatically run it after reset.

Reproduction is offline:

The first offline replay assertion compared integer histogram keys with
their JSON string-key representation and failed. Canonicalizing that
comparison through JSON fixes the verifier; neither the saved native
source nor the CPU histogram values/compiler result changed. No GPU credit
was consumed by this local verifier error.

```
python3 tools/screen_moe_tiny_pair_lut.py --offline
python3 tools/analyze_moe_tiny_pair_lut.py
```

MoE routine checks stay8/8; attention1/8, BF164/8, decoder1/8, prefill7/8;
all scored submits0 on Oct31 UTC, reserve untouched. Stop this MoE cycle.
Next sequential track: attention. A new opportunity argument is required
before another H200 probe. Rank1 goal remains active and incomplete.
