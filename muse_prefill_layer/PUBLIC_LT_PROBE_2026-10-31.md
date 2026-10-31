# Public cuBLASLt signature comparison

PIPE3 check `4c855e373fee4973` is terminal, all logs/profiles read and
all40 independent groups archived; its changed combinations were rejected.
This next experiment stays on prefill and keeps canonical v20 unchanged.

## Independent lead and bound before launch

The public ruwwww prefill source at
[`3b15eb8`](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_prefill_layer/submission.py)
differs from our pinned vjeux base by only35 unified-diff lines. It changes
several cuBLASLt algorithm signatures and also disables residual recompute.
This diagnostic tests only its **ten additional signature observations**,
preserving recomputation and every parent timed choice. No native source
from that repository is copied, and no speed follows from its source alone.

Using same-job sfo3 profile `4c855e373fee4973`, summing the visible dense
`nvjet` durations and subtracting those from each weighted benchmark cell
gives a loose **58.548%** board-speedup phase-zero bound. The profile
contains dependency waits/overlap, so this is an optimistic screening bound,
not an achievable estimate. It clears the ~3% gate for investigating the
dense family. It does not establish that these ten particular algorithms
can approach that bound.

## Protocol

`tools/build_prefill_public_lt_probe.py` builds
`submission_v24_public_lt_probe.py`, SHA-256
`3e6606ec33a14142378293d8d80ebb5d3f2620fe85e5f738350897d9d6ae1f9e`.
The canonical prefix and its kernel_fn definitions remain byte-identical.
No new native kernel is compiled; the existing context-safe Lt API queries
the offered algorithms and matches signatures by their returned fields.
Unavailable or already identical parent choices are explicitly recorded
and skipped. This is not a broad algorithm sweep.

For each distinct offered candidate, use identical BF16 input/weight tensors
for incumbent A0, duplicate incumbent A1 and candidate graphs. Two seeds,
four retained input/output allocation pads, nine alternating/rotated
cold-L2 repetitions per group. Raw nanosecond event durations are retained.
Random, outlier and one-hot rows are checked against an FP64 matrix product;
FP32 outputs have5e-5 relative and BF16 outputs0.005 diagnostic gates.
Whole outputs must agree with parent and remain finite. Fresh random inputs
must change parent outputs and pass the same captured-graph checks.

The pads perturb fixture allocations, **not the checkpoint-weight addresses**;
this does not simulate four independently prepared weight layouts. No claim
of broader layout robustness should follow. Router/pooler concurrency is
absent from this component diagnostic; a win must be checked in the full
layer. Returned state and timed choices are asserted unchanged. Synthetic
graphs/inputs/outputs do not persist. A single final compressed
`PREFILL_PUBLIC_LT_Z` report preserves plans, rejections and raw repetitions.

Require recurring >2% component gains beyond A/A offsets plus a conservative
weighted-board model>=3% before selecting a full-layer candidate. An
unchanged timed-path score is not a candidate gain or grounds for a submit.
Second of eight reserved prefill checks Oct31; zero scored submits.
