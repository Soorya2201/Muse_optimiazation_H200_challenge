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

## Active bounded experiments

1. **Phase probe, v17 / check13bc760e093343f3.** Exact parent timed path; separate instrumented
   module on independent fixtures. Read its output agreement, raw timing perturbation and phase
   timestamps before inferring bottlenecks. Instrumented intervals are not exact parent timings.
2. **One-time softmax phase offset, v18, prepared but not run.** Source SHA-256
   `c2d533e3b4b363a4766aed38ca8da328cc0d03585a14acd976357adb62598075`.
   On flat L8 only, head1's128consumer threads wait once; head0's128threads release them
   after issuing the first PV. Named barrier6 expects256consumer threads, excluding producers.
   Later tiles add no new turn-taking barriers. Existing proxy fences, head barriers,
   stage releases, math, precision, context coverage and launch plans remain unchanged.

The phase-offset hypothesis is that a small initial scheduling difference may reduce repeated
softmax contention. It may instead add startup delay or quickly lose the offset. No benefit
is assumed, and the per-tile version's reported regression is counterevidence. A full judge
check and confidence gate are required before any promotion.

The builder initially rejected an ambiguous `PROF_TILE` insertion anchor because the source
has three such markers. It was narrowed to the L8 ring-advance block; the successful builder
proves only the two barrier hooks and extension namespace differ. No GPU credit was consumed
by the failed local construction. Python syntax passes. The builder is
`tools/build_attention_phase_offset_candidate.py`.

## Selection constraints

Keep every key and the required precisions. Reject layout transfers that require an untimed
rewrite of fresh KV contents. Preserve live WGMMA operand registers until the corresponding
wait; a removed wait is not an optimization without a valid lifetime argument. Require
actual numerical validation and judge measurements, and keep failed or inconclusive results.
