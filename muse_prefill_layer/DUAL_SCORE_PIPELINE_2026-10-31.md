# Two-score attention pipeline diagnostic

The previous goal turn completed decoder discovery and changed the next
action to prefill; it counts as research progress, not a score improvement.
This cycle stays on prefill. The canonical v20 hash remains
`8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5`.

## Mechanism and bound before running

The broader public source search uncovered an older prefill module with
additional pipeline dispatches. Its FP8 dense-projection branches violate
the challenge precision rule and are excluded in full. Its external
constant-ratio timing claims are not evidence of performance.

Our exact Apache-2.0 vjeux parent already contains the relevant `PIPE3`
implementation, although no host launch selects it. It uses two score
accumulator sets: while softmax processes tile t, tensor cores can process
QK(t+1) and PV(t-1). Two consumer warpgroups alternate exponential turns and
request 240 registers each. The producer relinquishes registers. This is
the asynchronous overlap direction described in
[FlashAttention-3](https://arxiv.org/abs/2407.08608), not a new lower-precision
weight or activation representation.

This differs from closed v22: that tested the existing single-score PIPE/RRS
body on two warpgroups, plus FMA exponential splits. The new arm retains
hardware exponentials and uses two independent score arrays. No blind
FMA-fraction sweep is reopened. All template/helper bodies remain unchanged.

The pre-existing same-job phase analysis in
`HYBRID_PIPELINE_2026-10-30.md` gives a loose **15.623%** board-speedup bound
if all full-attention forward duration vanished. A 25% cut models **3.487%**.
These optimistic subtractions include possible overlap/waits and are not
predicted or measured gains. They exceed the ~3% experiment-screening gate;
whether additional register state and a smaller query tile erase all overlap
benefit is precisely what this run must resolve.

## Probe

`tools/build_prefill_dual_score_probe.py` builds immutable
`submission_v23_dual_score_probe.py`, SHA-256
`f0b6b986080441848dfec5e62e5fc777e719eec8aea4f757a9efd6869d0b6e17`.
Its complete canonical prefix is unchanged. Only host dispatch/resource
wrappers expose:

- 43: incumbent three-warpgroup PIPE/RRS, hardware exponentials.
- 48: two-warpgroup single-score PIPE/RRS control.
- 49: two-warpgroup dual-score PIPE3, hardware exponentials.

No new upstream implementation is copied: the PIPE3 body is already in our
pinned parent. BF16 dense projections, query/cache precision and complete
causal key coverage are unchanged. Timed full-layer dispatch remains the
incumbent sliding8/full43; its whole-run score is **not** a PIPE3 score.

Prepare-only diagnostics query compiled register/local-memory attributes,
validate against FA3 on three independent nearby fixtures with split1/3,
then measure all five scored full-attention geometries, two independent
seeds and four retained allocation pads (0/1/3/7MiB): 40 groups. Nine graph
replays rotate/reverse arm order, with an untimed 256MiB L2 sweep before
each replay and separate duplicate-parent graphs. Numerically invalid
arms are recorded rather than timed. Changed query, gate and cache values
must change the parent and continue matching on fresh replays. Fixture
graphs/outputs are not stored in returned state.

Require recurring >2% phase gains beyond duplicate-parent offsets and a
conservative weighted whole-board model >=3% before promoting any arm to a
full-layer candidate. Preserve failures, raw repetitions and all profiles.
Compressed diagnostics use one final `PREFILL_DUAL_SCORE_Z` record.

Oct31 budget before launch: zero checks/submits on every track. This is the
first of eight reserved prefill checks; no scored submission is warranted
before the diagnostic establishes a gain.

## Terminal result

Check `4c855e373fee4973` completed on sfo3: all5 challenge tests and8
benchmark cells passed,1.807119390x,214376.483us,CI1.780013–1.834226.
Prepare63.980s; independent probe6.983s, all40groups complete. Timed dispatch
is still parent: this whole-run point is not a PIPE3 gain and is not a reason
to resubmit the unchanged source. Receipt, complete compressed diagnostic,
all eight profile reports and all available local/remote logs were read.

Small FA3 validation accepts43/48/49, worst0.0151. At t1f both2WG arms differ
from the parent by1.4791–1.4856%, exceeding the chosen1% diagnostic agreement
gate; neither is timed there. This is not proof of a challenge failure.
All other geometries pass initial agreement and changed-query/gate/cache
replay checks. Largest duplicate-parent median offset is0.1708%.

| Geometry | 48 parent/candidate range | 49 dual-score parent/candidate range |
|---|---:|---:|
| t4f | 0.637261–0.638334 | 0.621914–0.628892 |
| first0 | 0.661199–0.661851 | 0.651050–0.652331 |
| first32768 | 0.654137–0.654486 | 0.704489–0.706555 |
| first65536 | 0.649240–0.649525 | 0.717458–0.720200 |

The dual-score arm is38.85–60.79% longer than incumbent across these32
groups. It helps the weaker2WG single-score control at the long contexts,
but still loses to the3WG parent. Every one of26 changed fixed class
combinations is rejected; a missing t1f arm is not treated as a measured tie.
`DUAL_SCORE_REPORT_4c855e373fee4973.json` retains every raw repetition,
numerical gate and same-job weighted model.

Resource attributes: parent43 has128 registers/thread;48/49 have168.
All report zero compiler-local bytes and zero static shared bytes. The
dynamic setmaxnreg requests differ (PIPE3 asks240 for consumers); static
attributes do not identify the exact scheduling bottleneck. There is no
evidence of compiler-local spill allocation here. Do not claim fewer
warpgroups or an extra score set solved the remaining issue pressure.

No canonical change or scored submit. Close this PIPE3 configuration;
next distinct lead is independently comparing the public dense-algorithm
signatures while preserving the parent's residual recomputation.
