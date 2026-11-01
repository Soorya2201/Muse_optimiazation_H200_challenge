# Two resident WGS blocks: opportunity, source review and validation protocol

The previous goal turn produced a terminal, correct but slower prefill
batching variant. This cycle moves sequentially to MoE. No agents are used.
Canonical SHA b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281
and best scored2.372309331x remain unchanged. No rank1 claim.

## Public source review

Fresh GitHub code search finds218paths/13repos, four new path/blob entries.
Two are decoder files already reviewed on that track; the new MoE M22 and
current source are at Kushagrabainsla/muse-spark-open ca711441. Exact blobs,
SHA256s, notes and the published069f543bb3604714 result are statically reviewed
in temporary storage; no root license established, no source mirrored or
ported. M22's public1.704x is not our paired measurement. It fuses routing
into GEMM CTAs, publishes metadata to the next launch, and reads final-pass
inputs before a dependent wait. Our persistent parent already includes the
core fusions and register accumulation. The later changes target decoder
integration, while its deeper-ring attempts lost. No additive new gain is
established from these revisions. Metadata/read scope are in the adjacent
PUBLIC_TWO_CTA_SOURCE_SCREEN record.

The new [H200 grouped-MoE project at7d71d5b](https://github.com/harshithkantamneni/moe-kernels/tree/7d71d5bbafae8cebf737c06f7198ccb80c608ef2)
is a measurement study, not a stronger NVFP4 submission. Twelve MIT files,
96,826 bytes, are pinned unchanged with LICENSE and blob/SHA verification.
Complete README, Torch grouped-mm baseline, selected study/integration
sections, partial-retraction marker and corrected summary were read. Other
pinned files are context, not a claim of full line-by-line audit.
Its study retracts the original padding-attack premise and distinguishes
memory movement, compute format, calibration and measured crossover. Its
Torch FP8 isolated-GEMM path caches activation quantization by tensor
identity; that boundary cannot be imported into our fresh-input BF16
challenge. No third-party module or harness was executed. The large CSV
archives were not audited, and none of its speed ratios is a Muse gain.

## Distinct architectural hypothesis

Parent WGS has four consumer warpgroups, two producer warps,576threads and
231168 dynamic shared bytes: one block/SM. The tested REG112 redistribution
kept that residency. This candidate instead uses two consumer warpgroups,
320threads,128-output-row units, three18KiB weight slots, two activation
slots and32-token jobs. A264-block grid aims for two blocks on each of132SMs.
There are still four total consumer warpgroups/SM, with two independent
producer/consumer pipelines and finer output units. It is not a claim that
more blocks or smaller tiles automatically help. Extra job chunks can
reread weights and a shallower activation ring can lose overlap.

Both arms use the exact original packed weight pointers: the new producer
selects the appropriate contiguous half of each old36KiB stage. Global
scratch sizes, xA/xB offsets and fixup-base offsets remain unchanged. Four
new phase-A units times two warpgroups still publish eight credits for each
phase-B256-column stage. A pair-of-stages cost prefix preserves unsigned
32-bit partition products even for384tokens routed entirely locally. Its
conservative product bound is2733637632<2^32. Empty jobs return before
lookup, with Apache2 attribution to trxphxle72db638. Global counter protocol,
NVFP4/BF16 arithmetic and per-route norm/sum semantics remain in the call.
Final FP32 norm reduction grouping changes, so numerical validation is required.

With only M32/128/256 improved,3% six-load board gain needs about5.74% less
time on all three. Conditional5/10/15% cuts model1.025978/1.054093/1.084652x.
These are planning arithmetic, not predicted or measured gains. Current
scored case times are25.686852/92.994285/125.324186/65.064303/173.386921/
210.656709us forM1/16/32/8/128/256. Old phase-deletion traces do not prove a
removable stall budget. This architectural experiment tests pipeline
independence directly rather than retuning a closed ring/fence family.

## CPU/compiler evidence and failures retained

Exact checks cover98304 weight-half addresses,528 complete row-gather maps,
884 cost/owner partitions,4096-column coverage and all readiness buckets.
This is not a GPU concurrency proof. Dynamic shared budget is checked by
NVCC static assertion(110848 or110976bytes depending on struct alignment).

Draft0 full BIGtrue NVCC13/sm90a has16-byte frame/20spillstores/16spillloads.
Splitting16 final accumulators into two8-column output passes does not fix
that. Preprocessing out a fast-build-unreachable LUT loop removes static
code but still spills16/20/24. Late generation rematerialization alone also
fails. Storing the launch-counter pointer in shared memory and reloading it
at use eliminates the long live pointer: complete BIGtrue and final BIGfalse
builds are96registers,16barriers, zero stack/spills, no C751x. PTX is complete;
SASS is clipped at5001rows, so no whole-kernel SASS count is claimed.
Two parent full-body compiler requests fail within Compiler Explorer with
RangeError:Invalid string length, including skipAsm. No parent compile
conclusion is inferred. Actual linked H200 resources remain a mandatory gate.
All requests/responses and diagnostic text are retained.

Three initial exact-anchor count assertions failed locally and were repaired
before compiling. Unsubmitted V36's oracle indexes Float8 before conversion;
CPU reproduces unsupported indexing. V37 converts scales first, but its exact
wire function raises KeyError by popping an event field inside a four-arm
comprehension. V38 fixes that. V39 explicitly requests100% shared carveout
and checks status before occupancy. V40 strengthens bounded failure capture
and wire stress; V41 makes empty fixtures use eight distinct nonlocal IDs.
All drafts remain archived, and no GPU credit was consumed by these errors.
Exact builder stubs,768 event-bit/312-metric roundtrips, six FP64 oracle tests
(maxabsolute8.88e-16) and nine valid unique-route fixtures pass on CPU.
Complete/failure wire stress is7760/11776chars, bound12800.

## Predeclared H200 protocol

Only V41 SHA867444e3f7292f79368556a39ab66c49e7819eabd689d2efae0cd339493055cc
is eligible. Native SHA5090fcdc7944f1d4067ac35232e6c874454b0adf408f9f964a260f59a403502d.
The complete canonical prefix and all kernel_fn definitions remain unchanged.
prepare runs independent synthetic experiments, then returns unchanged state.
The official score always times the old parent; it cannot certify this candidate.

Compile old/new with the same flags, explicit host/device C++20. Query both
BIG instantiations after dynamic-shared and100%carveout setup. Require successful
attribute/occupancy calls, zero local allocation, at least one resident old
block and **at least two resident new blocks/SM before any doubled-grid launch**.
An occupancy failure aborts the experiment rather than risking a grid-barrier
deadlock. H200 capability and132SM grid are checked. Weights/scratch pointers
are shared by the matched arms; no new weight repack or between-arm copy.

Three empty32/128/256 cases run new against literal zero and then replay on
fresh nonempty data. The unguarded canonical empty-WGS path is not launched.
Six nearby fixtures17basis/33mixed/65hot/129mixed/257mixed/384hot compare all
four arms, initial and fresh. The basis case independently checks FP64 whole
expert math against the actual original weights within.015global/.075row.
All parent comparisons require.002global/.01row, finite values, fresh change
>.001. All8local routes at384exercise the maximum per-call route budget.
Fallback scale dispatch stays in the canonical prefix; no new fast-scale
path is claimed to support exceptional scales directly.

Timing uses M32/128/256, seeds3101/7907, pads0/1/3/7MiB,24groups and eight
balanced Williams rounds. Arms are actual A0, matched old, new, actual A1.
Each replay has an identical untimed warmup then256MiB read eviction outside
captured external CUDA events. No overhead subtraction, no discarded samples.
A200s independent diagnostic guard preserves partial evidence. All768original
float32 event bits and312rowmetrics are retained in the binary report.
Comparison arm/phase, finite state and bounded primary error/trace are recorded.

Require every A/A median within2%, actual/matched-control within3%, and both
matched and actual component ratios at least1.02 in every group of an adopted
size. Evaluate all seven fixed subsets; each same-seed/pad policy uses the
smaller actual/matched ratio and must reach1.03 whole-board proxy in the
worst of eight groups. None of this substitutes for full changed-layer checks,
nearby/fallback/cross-variant-state validation and independent repetition before
promotion or a scored submit. Do not loosen gates after seeing results.

First of eight routine MoE checks on Nov1, all scored submits0; owner reserve
remains intact. Push source/protocol before launch, log the handle immediately,
and poll the same handle through terminal state. Read every available profile,
log and complete decoded result. No job is live before this planned launch.
