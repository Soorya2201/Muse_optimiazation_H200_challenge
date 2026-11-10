# Serving research and experiment playbook

Approach-development update: the norm-overlap design now has an isolated
[candidate08 experiment](candidates/08_marlin_norm_overlap.diff), 4,320 paired
CPU coordinator/norm checks, and an optional native whole-MoE experiment.
The native path has only passed source preparation; no measured benefit or
submission selection follows. See [design/evidence](designs/HOPPER_NORM_OVERLAP.md).

This pass answers the owner's request to keep learning/developing while GPU
execution is unavailable. Twelve primary sources and six pinned SGLang files
were reviewed in bounded sections; exact versions/read limits are recorded in
`research/serving/CONCEPT_SOURCES_2026-11-10.json` and the local-source audit.
This is not an exhaustive literature review, a novelty claim or a benchmark.

The strongest approach currently supported by evidence is to validate the
small existing mechanisms, profile the remaining critical path, then spend
the larger implementation effort on whichever measured bottleneck dominates.
We cannot name a fastest composition or guarantee a winning result yet.

## Concepts and how they transfer

| Concept | What matters for this workload | Concrete direction / current status |
|---|---|---|
| Whole-system critical path | A fast kernel matters in proportion to the time it removes from request completion. Overlapping durations cannot simply be summed. | Record per-load phase and slowest-rank timelines before selecting the next large port. |
| Memory traffic versus arithmetic | Decode and verify can have different bottlenecks. Extra layout copies, padded routes and dequantization may cost more than the multiply itself. | Measure bytes, active rows, tile padding, occupancy and native latency. Preserve one canonical weight layout. |
| Lossless weight layouts | Exact encoding may reduce bandwidth without changing BF16 values; decompression and escapes consume instructions and storage. | Defer a port until bandwidth is measured dominant. Existing project negatives and full-model memory duplication rule out automatic adoption of kernel-track layouts. |
| Sparse expert routing | Global token batches do not imply every route belongs to this rank. | Existing01/04/06 address density, mapping and inactive-prefix work. Inspect skew as well as uniform routing. |
| Precision throughout reductions | FP32 MMA does not prove FP32 global partial reduction. | The pinned Humming default Stream-K path is excluded; retain the prototype's explicit precision guard. [Pinned package](https://pypi.org/project/humming-kernels/0.1.12/) |
| Register/shared-memory balance | Deeper pipelines can hide latency while reducing occupancy or spilling registers. | Use compiler resources and same-geometry native timing; do not assume more stages or fusion wins. [FlashAttention-3, §3.1–3.2](https://arxiv.org/html/2407.08608v1) |
| Dataflow overlap | Independent work can execute before a dependent consumer is ready. | New design: move Marlin pre-expert normalization before the existing router join; measure contention. See the overlap design. |
| Programmatic dependent launch | Independent preambles can overlap on Hopper; dependent reads still need readiness/visibility. Concurrency is opportunistic. | Consider only for a profiled launch-bound chain; a blanket launch flag is insufficient. [CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) |
| Collective latency and ordering | Tiny metadata transfers can be launch-limited; all ranks must agree on captured operations and graph launches. | Existing02 combines draft metadata. Validate actual SGLang/NCCL/custom-AR interaction on eight ranks. [NCCL graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html) |
| Expert communication overlap | Dispatch/compute/combine can overlap with explicit ownership and completion events. | DeepEP offers design lessons, but current V2 is not a drop-in replacement for our pinned path. Its BF16 mode avoids an unnecessary precision change. [Pinned README](https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06cd4a0489e7730365d4e0114970a2f3) |
| Draft cost versus accepted yield | Parallel DFlash drafting does not have the same cost curve as serial autoregressive drafting. | Minimize complete cycle time per committed token. Keep the released 8-position drafter. [DFlash, §3](https://arxiv.org/html/2602.06036v2) |
| Hardware-aware verification | A larger speculative budget can improve acceptance and still lose when verifier cost grows. | Measure real graph-bucket costs before considering shorter verified prefixes. Reuse the cost principle, not Sequoia's replacement tree/drafter. [Sequoia, §3.3](https://arxiv.org/html/2402.12374v2) |
| Sampling correctness | Distribution-preserving verification is a specific algorithm, not a consequence of plausible proposals. | Request matching is stricter than a distributional argument: preserve stock RNG and committed outputs. [Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) |
| Request-derived lookup | Repeated code can supply cheap proposals, but hit rate and target verification determine value. | Use only the live conversation; account for search cost and fallback. It is established prior art, not a novel feature by itself. |
| Prefix reuse | Shared prompt KV can avoid repeated prefill; referenced live nodes cannot be evicted. | Separate first-turn and follow-up hit/recompute behavior; existing RadixAttention is already present. [SGLang, §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) |
| Paged KV allocation | Allocation, sharing and fragmentation affect useful batch capacity. | Preserve page ownership and speculative rollback; do not trade hidden stale KV for a faster test. [PagedAttention, §4](https://arxiv.org/html/2309.06180v1) |
| Hierarchical KV transfers | Host cache helps only when the avoided recomputation outweighs transfer/wait cost. | Fixed flags already enable 160 GB HiCache. Measure layer transfer stalls; the pinned implementation already overlaps transfers. [HiCache design](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/docs/docs/advanced_features/hicache_design.mdx) |
| Prefill/decode scheduling | Large prefill batches can improve utilization while stalling ongoing decoding. | Examine queue/TTFT/inter-token latency, preserving fixed flags and fairness. Smaller chunks also add overhead. [Sarathi-Serve, §4](https://arxiv.org/html/2403.02310v3) |

The pinned model's finalize/all-reduce workspace is explicitly gated to
Blackwell, and its early packed-input preparation excludes Marlin. Those
optimizations cannot be enabled on H200 by changing a flag alone. Its router
already uses a side stream alongside the expert-input projection; claiming
that overlap as a new optimization would be incorrect. These are direct
source observations, with file hashes in the local audit.

## Cost model before another policy

For an isolated component occupying fraction f of the measured critical path,
local speedup s and added cost h as a fraction of the original total, the
idealized whole-step speedup is:

`S = 1 / (1 - f + f/s + h)`.

For illustration only, doubling a component occupying 80% of the critical path
gives 1.67x, while doubling a 10% component gives about1.053x before integration
cost. These are mathematical sensitivities, not measured fractions or gains.
Use critical-path attribution rather than sums of overlapping kernel times.

For speculative decoding, let A be the accepted draft prefix length 0..7,
excluding the target bonus token. A full 8-position block contains one anchor
and seven proposals in the pinned worker. When considering k proposals,
estimated committed tokens are `mu(k) = E[min(A,k)+1]` before length clipping.
With remaining budget R, use `E[min(min(A,k)+1,R)]`. Its survival-sum form is
`1 + sum(P(A>=i), i=1..min(k,R-1))`, without independent acceptance assumptions.
The exact source `_commit_accept` adds the
bonus; counting eight proposals here would be an off-by-one error.

Keep the full released drafting cost D8 in the denominator:

`estimated time/output token = (D8 + verifier(k) + other cycle cost) / mu(k)`.

For a fixed active batch b, divide this by b for aggregate token cost. Client
concurrency and actual active batch size are different measurements. Graph
padding can also make two logical prefix lengths cost the same. Compare
complete exclusive cycle costs, including rollback and scheduling, rather
than summing overlapped profiler ranges. The tool assumes a common remaining
budget within a cohort. Mixed stopping budgets need per-request observations,
and the acceptance histogram must precede request-length clipping.

This is a counterfactual screen only. A changed verification cadence can
change the next draft's conditioning, acceptance and RNG consumption. A
histogram collected under the full-block policy does not prove the shortened
policy's steady-state yield. Greedy-only correctness, native graph/KV behavior
and paired whole-conversation trials would be needed before deployment.

`tools/serving_cost_models.py` implements these screens and the normalization
overlap bound. Its 170 math/boundary checks pass using deliberately synthetic
fixtures. The unmeasured input template returns `needs_measurements` and
produces no ranking. Cost bounds are caller inputs, not confidence intervals.
The tool does not select a runtime policy or produce a serving score.

## Development order supported by the evidence

1. **Validate current isolated mechanisms first.** Official checks 01/06/02,
   then map fusion as budget permits. Preserve every failing result. A check
   covers one short rung; the scored run is still needed for all-load results.
2. **Collect one useful profile per load.** Capture actual batch and graph
   buckets, draft/verify/commit time, acceptance histograms, expert-route skew,
   slowest-rank collectives, prefill/queue time and cache-transfer/recompute
   behavior. Use existing metrics or untimed instrumentation; no new `.item()`
   synchronization in the production forward path.
3. **Take the cheapest source-confirmed next experiment.** The pre-norm overlap
   design has a small dataflow change and an explicit ideal bound. Test it only
   if traces show router slack after the projection and normalization is costly.
4. **Evaluate the larger expert-kernel option.** Humming requires native
   precision/numerical/graph checks and representative dimensions. No-Stream-K
   may lose at low concurrency. A component result must include preparation,
   steady-state memory and cache-capacity effects before full-model integration.
5. **Use a measured bottleneck to choose between cache, communication and
   verification work.** A high acceptance ratio alone does not choose a policy.
   Targeted isolated comparisons precede compositions; do not spend three daily
   checks blindly exploring all 16 subsets of four unvalidated mechanisms.

Reject any path that misses stock answers, 220 tokens or the per-load TTFT gate,
even if its isolated throughput improves. Do not assume a serving score
aggregation formula that is absent from the current downloaded rules. Use
same-node paired data and effects beyond observed run variation; keep neutral
and negative results. No method here guarantees rank 1.

New drafters/training, reduced-precision activations/KV, expert dropping and
different node/parallelism flags are excluded by the present task contract.
Broader literature can teach scheduling or kernel design without making its
entire architecture eligible for this submission.
