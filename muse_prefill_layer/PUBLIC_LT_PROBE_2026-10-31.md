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

## First run: challenge passes, diagnostic capture fails

Check `05a710507460440a` completed on sfo3c, all5tests/8timings pass:
1.820144307x,212842.410us,CI1.792842–1.847446,prepare55.592s. All eight
profiles and all available local/remote logs were read and archived.
Timed dispatch remains parent, so this point is not a dense candidate gain.

The service's16000-character log tail begins inside the compressed base64
payload: the report marker and compression prefix were lost. The CLI's
local log has the same incomplete payload; structured remote records contain
only setup/tests/benchmarks/profiles, not the full prepare stdout. The
diagnostic's own completion, plans and timings therefore remain **unknown**.
No missing performance evidence is reconstructed or inferred from the score.
`PUBLIC_LT_REPORT_05a710507460440a.json` records this limitation explicitly.

This was a report-design failure. A lossless binary codec now retains exact
uint32 nanosecond timings, exact GPU binary32 error metrics and binary64
ratios. An80-row randomized CPU stress fixture still exceeded the safe log
budget, so that full-size repair was **not submitted**. The candidate set
is instead split into two immutable five-signature parts, each retaining
both seeds, all four pads and every repetition. A40-row stress fixture is
8312 payload characters plus855 summary characters; every value round-trips
exactly. A separate small summary is always printed after the raw report.
This repeats an unresolved measurement whose entire diagnostic was lost,
not an unchanged-source scored retry or an attempt to seek a favorable score.

- Part A: `submission_v25_public_lt_binary_part_a.py`, SHA
  `93609c40abb929b20c877d2c4eb5ab61cec2fbd59b75fa7ad97634b9b3c1523a`:
  T753 qkv/o/pooler and T3012 qkv/o.
- Part B: `submission_v26_public_lt_binary_part_b.py`, SHA
  `63600c0a34b7260f0c9f91f8a9eaf1245a6186e876fe252f2189c0fab514d3ee`:
  T3012 router/pooler/unpooler and T16384 qkv/router.

Run sequentially, archive/read Part A before launching Part B. These will
be checks3/4 of eight reserved; zero scored submits. GPU computation,
precision, numerical gates and timed dispatch are unchanged. Record any
remaining capture failure instead of silently dropping raw repetitions.

## Part A result

Check `fe20034a4ad04dea`, sfo3: all5tests/8timings pass,
1.792476332x,216127.763us,CI1.765589–1.819363,prepare45.041s.
Independent diagnostic complete in0.476s, all40groups, five distinct offered
algorithms. Payload6344characters; complete raw report and independent
summary both survive. Every raw repetition, plan, numerical gate, all8
profiles and full available local/remote logs were inspected and archived.
Maximum FP64 comparison error0.00172705; parent/candidate output agreement
is exact in these initial/fresh fixtures. Largest A/A offset0.9954%.

| Cell | Paired speedup range |
|---|---:|
| T753 qkv | 0.993939–1.007204 |
| T753 o | 0.992824–1.016578 |
| T753 pooler | 1.007161–1.012717 |
| T3012 qkv | 1.011326–1.019395 |
| T3012 o | 0.997235–1.006202 |

No candidate clears the predeclared2% recurring component gate. All31
nonempty fixed combinations are rejected; best nonempty optimistic
whole-board model worst1.000627/median1.000753/best1.001087. Even adding
these isolated favorable microseconds cannot explain a rank-one jump.
This is a same-job model, not a full-layer measurement. No canonical change
or scored submit. Part B remains the distinct unresolved half.

## Part B result and closure

Check `0848e5e1b2ba40e5`, sfo3: all5tests/8timings pass,
1.798535737x,215399.612us,CI1.771558–1.825514,prepare55.463s.
The0.902s independent diagnostic completes all40groups/five distinct
offered choices. Payload7092characters; raw and independent summary agree.
All repetitions, plans, numerical gates, eight profiles and complete
available local/remote logs were read. FP64 error max0.00171789; router
FP32 error max1.59511e-5. Initial/fresh parent agreement is exact in these
fixtures. Largest duplicate-control offset1.0582%, at T3012 router.

| Cell | Paired speedup range |
|---|---:|
| T3012 router | 0.992664–1.016784 |
| T3012 pooler | 1.005589–1.011620 |
| T3012 unpooler | 0.998267–1.001853 |
| T16384 qkv | 1.011248–1.012992 |
| T16384 router | 0.996181–1.003995 |

No candidate clears2% across groups. All31 nonempty fixed combinations
are rejected; best nonempty optimistic same-job board model
worst1.000571/median1.000718/best1.000814. Do not sum absolute component
timings across separate jobs and call the result a joint measured gain.
No full-layer candidate qualifies and no scored submit is made.

Both halves together cover all ten signatures and80 complete independent
groups. The older v24 report remains unrecoverable; these are new
measurements, not reconstructed old data. Four of eight routine prefill
checks used today, zero submits, all four jobs terminal.

**Lesson for experiment selection:** the58.548% phase-zero bound covered the
entire dense family and was too loose to justify expecting a large gain
from signatures already offered to the parent's tuner. The parent had
already timed these algorithms with a3% selection margin. Future pin-only
leads need evidence of an algorithm absent from that search or a distinct
in-chain interaction; do not repeat these ten choices or broaden a blind
Lt sweep. Their microsecond savings can be revisited only as part of a
larger independently measured architecture change.

Canonical v20 is unchanged. Current score differences between diagnostic
jobs are parent/judge/layout variation, not evidence of an improved
implementation. This closes the public Lt signature cycle.
