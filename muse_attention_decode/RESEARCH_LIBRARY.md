# Attention research mapped to the current implementation

Updated October 29, 2026. This records mechanisms and transfer limits, rather than importing
published speedup numbers into our score. The verified parent remains vjeux f35a41e,
scored **2.603505x**. Source reviews and experiments are in PUBLIC_RESEARCH.md and root RESULTS.md.

|Primary source|Relevant mechanism|Current-source finding and next action|
|---|---|---|
|[FlashAttention-3](https://arxiv.org/html/2407.08608v2), sections 3.1–3.2|Producer/consumer specialization, asynchronous matrix operations, alternating consumer groups, and overlap between matrix operations and softmax.|The parent already uses specialization, register redistribution, FP8 PV and `wgmma_wait<1>` overlap. Named-barrier turn-taking is absent despite a leftover comment. The c1809247 source defaults TURN off and records a slower per-tile attempt. Prepare an isolated one-time offset, retaining existing per-head transpose fences and barriers. Do not count already-retained overlap as new.|
|[FlashDecoding++](https://arxiv.org/html/2311.01282v4)|A common exponential reference permits partial attention states to combine without repeatedly synchronizing their maximum; the implementation also adapts work distribution to hardware.|Our parent already rescales partial states to a fixed reference before FP32 accumulation, then finalizes through counters/cooperation. Its precision/overflow checks remain necessary. A generic fixed-reference port would duplicate existing work; target an actually measured merge dependency instead.|
|[LeanAttention](https://arxiv.org/html/2405.10480v2), section IV|Associative softmax reduction supports unequal context partitions and balanced Stream-K scheduling.|The current full-context path already flattens ragged requests over a fixed CTA grid. It inserts virtual key costs at request boundaries to compensate extra segment setup. The actionable question is whether measured segment setup/tails justify a different boundary cost or fewer cross-request segments, not whether to add generic Stream-K.|
|[vLLM Flint decode source](https://docs.vllm.ai/en/latest/api/vllm/models/flint/ops/decode_attention/)|Persistent split scratch, self-resetting arrival counters, cluster reduction and pipelined page loads.|The inspected implementation requires BF16, head dimension128 and contiguous16-token pages, while this challenge requires separate FP8 caches, dimension64 and scattered page1. Its bulk page loads and launch table are not transferable unchanged. Parent scratch/counters and clusters already cover the shared concepts. No source copied.|
|[NVIDIA PTX: bulk prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)|Asynchronous cache hints can issue memory requests ahead of demand; supported syntax depends on the target architecture.|Earlier MoE bulk-prefetch tests lost2.9–14.3%; attention PFL2=1 also regressed. Earlier requests are not automatically better: they can add waits, issue pressure or competing traffic. Keep these failures binding when selecting the next experiment.|
|[NVIDIA PTX: WGMMA ordering](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)|The WGMMA `.sync` description synchronizes participating threads within a warp; `.aligned` requires matching execution across the warpgroup. Shared-memory writes also require async-proxy ordering.|Do not treat a WGMMA instruction as a replacement for the parent's cross-warp V-transpose barrier. Our one-time offset retains both the existing proxy fence and per-head warpgroup barrier on every tile. It changes scheduling without removing the producer/consumer visibility protocol.|

## Active bounded experiments

1. **Phase probe, v17 / check13bc760e093343f3, passed.** Exact parent timed path,2.605394x;
   no speedup claim. Debug/control relerr<=1.58e-5,overhead0.93–2.70us. Raw10-shape records
   and interpretation are in PHASE_PROFILE_2026-10-29.md and its linked JSON.
2. **One-time softmax phase offset, v18, checked but not promoted.** Source SHA-256
   `c2d533e3b4b363a4766aed38ca8da328cc0d03585a14acd976357adb62598075`.
   On flat L8 only, head1's128consumer threads wait once; head0's128threads release them
   after issuing the first PV. Named barrier6 expects256consumer threads, excluding producers.
   Later tiles add no new turn-taking barriers. Existing proxy fences, head barriers,
   stage releases, math, precision, context coverage and launch plans remain unchanged.
   Check4bbfb48549864c60 passed at2.593449x,CI[2.554548,2.632351],belowparent;sub1% verify
   observations do not prove a gain. All16tests/12timings passed and profiles/logs were read.
3. **Parallel completion flags, v19 / check5161a53a9b274da4, rejected.** OnlyB1slidingL1,
   <=32splits. Replace contended arrivalRMW with per-split release flags and one warp's
   parallel acquire poll. Keep original FP32 sums/finalizer and CTA publication barrier.
   Reset all32flags before kernel completion;no epoch or cached inputs/outputs. The
   workspace/participant-range/source/syntax checks pass. All16tests/12timings
   passed2.581768x;targeted paired5.13us vsparent4.89us,no improvement. Follow-up
   v20/acf6f11e found0/50eligible workspace/finalizer plans. This family is closed.

Oct30 broader code search found a concrete confound in the legacy hybrid-exp
implementation: EXPPOLY disables the packed-P PK2 path. A new off-board v21
probe2d537621593a41d7 retained PK2 but failed an independent finite check.
The copied polynomial's clamp-127 yieldsNaN for masked logits. Repairedv22
ec545aab86274b3b passes six nearby/16timing cases but loses17-61% with25/50%FMA;
all eight changed compositions reject. V23/0b17352502ca4fca now isolates
bias-basedrange reduction against the identicalfloor50control. CPU66780cases
are bit-identical;standaloneNVCC13sm90aSASS removes FRND/F2I. Fullkernelprobe
completes6nearby/16timing cases: bias50 beats floor50by3.28–6.08%speedup,
but remains26.71–47.07%longer thanparent. All changedcombinations reject.
Both testedfamilies closed;timed/scoredparentunchanged. See PK2_HYBRID_2026-10-30.md.
The old~1.5–2% practical estimate is not an expected gain;the new5.94% idealized
scenario assumes exposed40.5%SFUwork/freeALUand is unverified. No paidGPU assumed.

The phase-offset hypothesis is that a small initial scheduling difference may reduce repeated
softmax contention. It may instead add startup delay or quickly lose the offset. No benefit
was assumed, and the per-tile version's reported regression was counterevidence. Our check
did not demonstrate an aggregate gain. Preserve it as an inconclusive component rather
than continue a blind scheduling sweep.

The builder initially rejected an ambiguous `PROF_TILE` insertion anchor because the source
has three such markers. It was narrowed to the L8 ring-advance block; the successful builder
proves only the two barrier hooks and extension namespace differ. No GPU credit was consumed
by the failed local construction. Python syntax passes. The builder is
`tools/build_attention_phase_offset_candidate.py`.

## Selection constraints

Oct31 new primary-source lead:
[fixed-binade bit exponential](BIT_EXP_RESEARCH_2026-10-31.md) derives IEEE
linear interpolation from Schraudolph1999, with no numeric conversion.
CPU314482grid/random samples validate a2.984% scalar error ceiling; standalone
sm90a emits FMNMX/FADD/LEA plus a high-tail predicate/fallback. V24 tests
50/100% replacement while retaining PK2 and fresh-input/cache checks.
Its timed challenge path remains parent, so its point score is diagnostic.
The dated report records terminal results and adoption decision.

The [Oct31 public source delta](PUBLIC_SOURCE_FOLLOWUP_2026-10-31.md)
adds18indexed paths/one repository,13attention files hash-verified and
statically reviewed,including an entrant below us. Existing fusion/WGMMA/
DSMEM ideas are already in the parent; reported A15 merge regressions give
no replacement. Idle-producer code warming needs a current-parent phase
bound and measured overhead before a new experiment; no license assumed.

Keep every key and the required precisions. Reject layout transfers that require an untimed
rewrite of fresh KV contents. Preserve live WGMMA operand registers until the corresponding
wait; a removed wait is not an optimization without a valid lifetime argument. Require
actual numerical validation and judge measurements, and keep failed or inconclusive results.
