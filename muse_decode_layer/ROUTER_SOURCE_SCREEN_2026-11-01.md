# Decoder public updates and warp router screen

The preceding BF16 cycle completed a new H200 experiment and rejected all
255 allocation policies. Its failure, repaired build, supported hardware,
exact numerics and negative performance result are preserved separately.
This decoder follow-up finds and screens new public work without spending
an OKC check. The canonical source and scored 2.211494776x result are unchanged.

## Public implementations above and below us

Fresh indexed search at Nov1 15:31:50 UTC returns 94 paths in 10 repositories,
with two changed path/blob entries. Our decoder rank at that capture is
200/1164. This index is incomplete coverage of branches, unindexed and
private sources. It is not a new scan of every ranked owner's repositories.

| Public owner/repository | Rank at capture | What changed and what was checked |
|---|---:|---|
| [rohanarun/Auto-research-agent](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee) | 84 | Exact default HEAD and candidate blobs are unchanged from our earlier evaluation/source screen. The current 2.2547x board job has no established association with one of those public candidates. Do not mistake the author's rank for a published faster source. |
| [Kushagrabainsla/muse-spark-open](https://github.com/Kushagrabainsla/muse-spark-open/tree/ca71144194e993c9769f20654fade8008c6868c7/muse_decode_layer) | 745 | Two indexed code changes, three new commits since 0778ecd. Read the complete L10–L13 experiment notes and builder hooks; parsed current submission AST. Three new published receipts supply all 36 timing cells. |
| [trxphxle/okc-2026-muse-kernels](https://github.com/trxphxle/okc-2026-muse-kernels/tree/72db638fa2718bc0ea5645fdd5c52a5511594579) | not refreshed in this narrow head query | Default HEAD is unchanged; previously pinned measurements remain the current reviewed source. |

Kushagrabainsla's published receipts report 1.439024257x (L10),
1.478359964x (L11) and 1.495936448x (L12), each passing 13 tests and 12
measurements. These are the author's external results, not our paired tests.
The current source equals the L12 family by its builder/experiment history;
we did not perform a complete native synchronization or conformance audit.
Its root license is still unestablished. Selected source remains temporary
research material and was neither executed, ported nor mirrored into Git.
`PUBLIC_ROUTER_DELTA_2026-11-01.json` retains hashes, exact URLs, AST metadata,
all parsed timing cells and review limits.

The useful lessons are specific. L10 collects pooler output squares and
applies the pre-expert norm when staging GEMM1 inputs, including BF16
rounding. L11 adds PDL and fixes stale L1 reads on predecessor-written
inputs. L12 removes a separate device-local expert-ID remap. Our incumbent
already fuses the pre-expert norm and local-ID mapping in `_graft_route`,
`_sl_tile_route` and `_gle_route`, and uses PDL. Its measured route kernels
are 1.8–3.5 microseconds, not the author's standalone 6.2-microsecond norm.
Those savings are not additive to our parent. L13's reverted boundary
experiment also changes routing and expert workload, so a smaller amount of
local expert work must not be mistaken for a faster expert implementation.
We preserve required precision and do not deliberately change routing to
reduce work.

## Newly pinned official CUDA router

[vLLM PR64553](https://github.com/vllm-project/vllm/pull/64553) merged on
Oct30 at `36ed4327d2caeb38543b75b80db2a291e88883bc`. Four selected Apache-2.0
files, 42,990 bytes, are pinned under `research/public/vllm_router_36ed432/`
with their original LICENSE/SPDX headers and exact Git blobs/SHA-256 hashes.
The pinned root tree has no NOTICE entry. No downloaded module was imported
or executed.

Read the complete CUDA routing file and host launch, selected numerical
unit tests, and the existing fused Triton dispatch. This is not a build or
a full-repository audit. The kernel assigns eight expert slots to each of
32 lanes, performs a stable local odd/even sort, then selects only the
32 sorted heads each round. Two group ballots preserve the lowest expert-ID
tie break across the two 128-expert groups. A tapered pop maintains only
slots that can still reach the remaining top-eight positions. Bias affects
selection; output weights use the original sigmoid scores. The implementation
also explicitly matches Triton's exponential/division instructions.

Its FP32 hardware warp maximum instruction is compiled only for the SM100
architecture family. H200 uses the five shuffle steps in the fallback.
PDL support alone is not a new feature for our parent. The standalone
selector does not compute our pre-expert normalization, residual carry,
device-local ID mapping or transformed two-row router input. A real adapter
must preserve each of these and the original precision boundaries.

`tools/model_decoder_warp_router.py` independently models the finite-key
selection and original-score association. It passes 4,142 cases, including
all ties, all eight winners from each possible single lane, tied groups,
adjacent FP32 keys and random values. This is a CPU ordering/layout proof,
not a native GPU port, sigmoid/division rounding validation, NaN contract,
resource measurement or timing result. In particular it does not prove
bitwise equivalence to the parent's ordered-bit treatment of signed zero.

## Why this selector is deferred before H200

The model reads all 12 routing profile durations from our own scored job
`eff0b59dc26c4c74` and recomputes all six 46:16 weighted loads from that same
job's benchmark times. If entire routing-kernel durations were reduced by
25%, 50%, or 100%, the conditional whole-decoder proxies would be
1.004146048x, 1.008332150x and 1.016827272x. These are optimistic subtraction
scenarios: durations include PDL waits and work other than selection. They
are not critical-path bounds or achieved speedups.

A selector-only replacement has no credible 3% decoder gain case in this
screen. Keep the licensed implementation and model in the research library;
do not spend a check merely because its selection algorithm is different.
A future experiment needs an additional mechanism that moves meaningful
work off the critical path, along with a complete BF16/FP32 and publication
protocol. This result does not rule out a joint routing/expert design.

The separate [GB300 specialized-decode PR64077](https://github.com/vllm-project/vllm/pull/64077)
was still unmerged in the retrieved metadata. Its author describes a 1.5x
full-serving result on Blackwell. We reviewed only metadata/file inventory;
no H200 adaptation, performance inference or source port was made.

No decoder OKC check, scored submission or canonical change in this follow-up.
Nov1 routine budgets remain BF16 1/8 and all other tracks 0/8; all scored
submits 0. The owner's reserve is intact. This screen reaches its stop
condition; next sequential track is prefill. The broader rank-one goal is
active and incomplete. Work continues from the internal backup because the
external volume remains disconnected.
