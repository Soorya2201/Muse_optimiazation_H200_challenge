# Wide register/shared WGMMA commit batching

The previous goal turn made progress: BF16 ILC was measured and rejected,
and the decoder router screen changed the next action. This cycle works
only on prefill. Canonical v20 and scored 1.784958x remain unchanged.

## Mechanism

The parent commits/waits after each K16 MMA on 160/192/256-token tiles.
The new implementation batches four unchanged RS-WGMMA operations per K64
stage into one commit group, with one FP32 accumulator and two banks of
register A fragments. Narrow tiles keep their existing accumulation scheme.
This avoids the extra shared operand traffic, producer expansion and narrow
KPS changes of the measured negative shared/shared experiment.

A naive switch to GROUP4 is invalid for KPS1: kk is always zero, so it
would overwrite the previous stage's pending register operands. The actual
transform alternates A banks by stage iteration on wide KPS1 tiles, while
KPS2/4 retain the existing kk parity. Wide tiles retain one accumulator;
blindly inheriting the narrow two-accumulator policy would exceed the useful
register budget. K order, BF16/NVFP4 precision, stored weight bits, input
layout, shared ring, producer/publisher roles, epilogue and route plan are
unchanged. Slot release follows the existing grouped-path wait and may
occur later than in the old wide path; that is a potential performance cost.

NVIDIA's primary [commit-group and wait-group specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
describes batching and completion. Matrix-A register storage must not be
reused before its pending group completes. This is an independent source
transform of our licensed incumbent, not a new MMA primitive or a claim of
novelty for batching itself.

## Local evidence

`tools/verify_prefill_wide_batch.py` checks source isolation and 1,515
worst-case/random early-completion schedules over KPS1/2/4. The naive
bank rule produces a stage1 overwrite counterexample. These are CPU
ownership models, not a native concurrency/racecheck proof. Complete new
native source/diff and hashes are archived as WIDE_BATCH_NATIVE files.

Four complete fused-body NVCC13/sm90a builds, parent/new FINE0/FINE1, all
succeed: 168 static registers, 8-byte existing ABI stack, zero spill stores
or loads, two barriers, no C751x warning. FINE0 SASS is complete: unchanged
MMA site counts, WARPGROUP.ARRIVE 12→9 and WARPGROUP.DEPBAR.LE 16→13.
FINE1 SASS is clipped at 5,000 displayed lines; its complete PTX reduces
commit/fence sites20→11 and wait sites26→17. Static sites are not dynamic
cycle savings. Linked H200 extension resources remain a mandatory gate.
Exact requests/responses and WIDE_BATCH_CODEGEN summary are retained.

V30–V32 are unsubmitted host-protocol drafts. V30's inherited order balanced
positions but not all adjacent pairs; V31 uses the exact four-order Williams
square, repeated twice, and equal per-arm warm-ups before read-only eviction.
V31's high-entropy complete report stress was10,468chars, above its10,000
budget. V32 moved row errors/fresh errors/change values losslessly into
binary FP32 frames; complete stress7,220chars, but full failure stress12,640
exceeded its12,000 budget. V33 captures primary message head2,500/tail1,000,
original message length, and separate traceback frames without duplicating
the message. Complete/failure stress7,220/11,228chars fit12,000. One extra
parenthesis in the host-only stress script was corrected; candidate unchanged.
All drafts/failures remain documented; no GPU samples were discarded.

V33 SHA-256:
`ced5301e48c264788e1751450740a23086249262ca62dbb45c8bc462ae31b1fb`.
The complete canonical source is an exact prefix; no new timed kernel_fn.
Its CPU preflight proves source isolation, eight schedule shifts and lossless
roundtrip of all768event float32 bit patterns and all error/change metrics.
No experimental graph, synthetic input or output is retained in returned state.

## Opportunity and fixed H200 protocol, before launch

The existing same-job phase model needs a7.5568% expert-duration cut across
all loads, or22.7142% on first-turn alone, to model3%whole-board gain.
These are conditional subtractions including PDL waits, not a prediction or
critical-path bound. Wider grouping reduces synchronization and increases
available in-flight MMA work without adding a memory representation change.
It justifies one bounded test; it does not prove the required cut is attainable.

The judge times the unchanged parent. Its official score will not be treated
as a batching score. Independent actual A0/A1, matched old and matched new
expert-chain+combine graphs share one old/new weight-pointer pair. No repack
or copy occurs between arms. The full FINE0/FINE1 new native modules must
introduce no additional compiler-local allocation compared with matched old.

Five nearby fixtures cover empty work, onehot inputs, concentrated routing,
mixed routing and all remainder boundaries. Initial/fresh results must be
finite, global relative L2<=0.002 and normalized row<=0.01; changed inputs
must change output. Timing covers753/3012/16384tokens, two seeds, four retained
pads0/1/3/7MiB and eight Williams-order rounds. Each replay receives the same
untimed warm-up and read-only256MiB eviction before internal event nodes.
All768original float32 event bit patterns and all native FP32 errors are
retained. The wire uses32u32event bits plus13f32metrics per row; no rounding
or sample dropping. A 200-second independent guard preserves partial reports.

A token route must win every seed/pad by>2% versus both matched and actual
controls, A/A offsets<2%, actual/matched offsets<3%, and retain the numerical
and resource gates. All seven nonempty fixed token-size subsets are modeled
with the smaller actual/matched speedup, using all eight same-job layer cells.
Worst-group conditional whole-board proxy must reach1.03 before promotion.
No relaxed gates or seed/pad-dependent dispatch. A qualifying result still
needs actual changed full-layer dispatch checks and an independent repeat.

Before launch Nov1: prefill0/8routine checks, BF161/8, all other tracks0;
all scored submits0. Owner reserve untouched. Only V33 is eligible to launch.
The live handle and every result will be recorded immediately in RESULTS.md.

## Public-source follow-up

Fresh indexed search remains118paths/13repos; the only changed indexed
entry is Kushagrabainsla's decoder L12 already reviewed in the preceding
cycle. Vjeux/alygg77/Rohan default heads are unchanged. A mistyped, unrelated
MrNirmata repository query failed; this is not evidence about its real
lossless-bf16 repository or a missing submission.

Prefill leader GodlyDonuts has a recently pushed public cuda-python fork.
Its inspected default branch has0commits ahead/117behind NVIDIA/main and
no changed files in that comparison. Branch names were listed; branch
contents were not exhaustively audited. No challenge source is identified
by this inspection. Do not infer a private implementation or method from
an author's rank. Metadata, exact heads and bounded scope are retained in
PUBLIC_HEADS/LEADER_CUDA_FORK files. No downloaded module executed or ported.

## H200 outcome

Pending. No measured batching gain is established before the terminal audit.
