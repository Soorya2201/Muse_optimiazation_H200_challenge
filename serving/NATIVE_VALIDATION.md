# Native validation prepared for an existing Hopper node

Optional `--humming-prototype` now adds the separate development adapter.
See [HUMMING_PROTOTYPE.md](HUMMING_PROTOTYPE.md) for source/CPU evidence and
the no-Stream-K precision guard. That flag has run in source-plan mode only;
no actual library/GPU result is claimed. The default four frozen candidates
and their seven patches are unchanged.

The laptop has no CUDA GPU. All commands below are prepared and unrun on a
GPU. Only the source-preparation mode has run locally; the latest immutable
report is MOE_NATIVE_PLAN_READY_2026-11-09.json. It applies each frozen diff to an isolated git
index, extracts/compiles the exact two MoE methods and hashes them. Stock,
density, map, prefix and combined variants have identical `_forward_marlin`
AST hashes. That supports unchanged finalization source, not runtime quality.

Use the official image with a checkout at 297e54fd and frozen candidate 07
applied, installed editable as directed in the official rules. The harness
checks exact installed source paths/hashes and common native dependencies.
It does not download the model, provision a node or issue an OKC run.

```sh
python tools/benchmark_serving_moe_candidates.py --plan-only --output /tmp/moe-plan.json
python tools/benchmark_serving_moe_candidates.py --benchmark --output /tmp/moe-native-ep8-rank0.json
python tools/benchmark_serving_moe_candidates.py --benchmark --rank 7 --output /tmp/moe-native-ep8-rank7.json
python tools/benchmark_serving_moe_candidates.py --ep 1 --output /tmp/moe-native-ep1.json
```

Each GPU invocation uses one Hopper GPU and simulates that EP rank's ownership.
It does not execute eight-rank communication, target attention, the DFlash head,
checkpoint loading or complete serving. The default 256 hidden / 128 intermediate
dimensions are small correctness fixtures; use the real layer dimensions for
representative performance only after those pass. Each run requires a fresh
output path and saves incremental state, candidate/source hashes, the active
case and errors. A failure preserves prior evidence rather than silently
skipping the case or producing a success report.

The planned native coverage includes:

- Actual SGLang NVFP4 weight preparation, Marlin GEMMs, SiLU and Muse finalizers
  through exact stock/isolated/candidate 07method bodies. Synthetic weights use
  the pinned upstream `make_nvfp4_weight_and_ref` utility, without requantizing
  real checkpoint weights or changing the serving patch.
- Eleven row counts cover 1/4/7/8/16/28/32/112/128/224/256, including seven/eight
  rows per request at concurrency1/4/16/32; random, local, peer, dropped and skewed
  routing; both norm forms and fused/Torch finalization. Default configuration
  plans 220 cells and 1,100 variant evaluations. EP1 and other ranks are separate runs.
- Projection buffers prefilled with NaN/Inf and inactive router weights NaN.
  Only active projections are compared; final outputs must be finite and rows
  with no local routes zero. Stock/candidate output metrics and paired numeric
  assertions use defaults rtol/atol 0.01. Small cases also use dequantized dense
  references at the pinned upstream test's rtol 0.05 / atol 0.25. These fixture
  tolerances do not establish official answer agreement or full-model quality.
- Native CUDA graph capture and fresh hidden states, route weights and IDs at fixed
  addresses; replay local/peer/dropped/random patterns. Compare with eager stock
  and captured stock. The test preserves each invocation's lock allocation.
- Optional paired AB/BA timing over 20 rounds, with 10 forwards inside each graph
  so Python dispatch cannot dominate tiny event intervals. Save raw per-call
  microseconds and median paired ratios; no best-of selection or serving score.

## Why a new regression fixture is needed

The pinned upstream `test_muse_marlin_inactive_routes` has strong dense-reference,
poisoned-buffer and mask checks, but its observer also asserts `is_ep` equals
`use_expert_map` and a one-word count when an expert map exists. Candidate 06
intentionally uses a local consumer and two-word inactive-prefix counts after
mapping. Eight parameter combinations would fail those policy assertions
before comparing projection arithmetic. This is a source-derived expectation,
not a test run or a waived quality requirement. No upstream test was edited.

The new harness retains the actual native operations and output checks, and
adds paired stock comparison and fresh-input graphs. It does not turn the old
test green by discarding its checks. A native failure remains a failure; an
official serving run is still required even if every component case passes.
