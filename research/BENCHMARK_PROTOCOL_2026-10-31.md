# Independent H200 timing protocol follow-up

Actual scored runs use the judge's own paired read-flush protocol and remain
authoritative. Independent synthetic diagnostics must match read eviction:
read a buffer larger than L2 and retain its scalar sum before each timed arm.
Writing eviction with`zero_`/`add_` can create a different cache state.
Keep any historical such measurements explicitly scoped;no universal
post-hoc offset or correction is justified.

The new primary external source is
[MrNirmata's H200 notes](https://github.com/MrNirmata/h200-decode-notes/tree/4f195ed71190f5556d48adbf23a0e1a9db5412b1),
CC-BY-4.0,selected unmodified notes/data pinned with license/attribution.
It reports different read/write flush offsets and infers write-back traffic
without DRAM counters. Its multi-run comparisons and GPU/counter limitations
remain attached. Existing alygg77 timing notes and CODEX_NEXT_STEPS already
require judge-faithful read flushing;the new BF16 probe mistakenly used
`add_` and is repaired in V41. This is a diagnostic-method correction.

BF16V39also exposed an uneven arm-order schedule:reversing a cyclic rotation
on odd rounds gave each arm only two possible positions. Eight Williams-square
rounds now give each arm every position twice and each ordered adjacent pair
twice. CPU combinatorial checks preserve exact order lists for13group shifts.
Use balanced schedules,actual-parent and matched-old controls,duplicate A/A,
multiple seeds/layouts and raw observations without dropping difficult rows.

Host-issued event pairs around a graph replay and event nodes captured inside
a graph measure different boundaries. The
[PyTorch2.14 Event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)
supports explicit captured record nodes with`external=True`. BF16V40records
both boundaries simultaneously for selected independent cases and an
event-only calibration. Its event-only inner span~2.9us proves captured event
nodes are not free. Outer-minus-inner~5.7–5.8us on these instrumented graphs
does not isolate a CPU cause or provide a universal subtraction constant.
Preserve raw values and control gates. Numerical/resource correctness and
independent real timed-path checks are still required for any promotion.

Details and terminal results:
`muse_bf16_proj/DEVICE_EVENT_METHOD_2026-10-31.md`. V39/V40's higher unchanged
parent point scores are not improvements. Closed algorithm families remain
closed unless a distinct new mechanism and a meaningful gain screen justify
investigation;this protocol repair is not permission for blind retries.


V41 completes the repair with read-sum eviction, balanced ordering and
inside-graph timestamps. All 72 groups remain nonqualifying: best worst-group
board proxy 1.000443x, maximum A/A offset 2.677%, matched-control offset
3.410%. All raw frames and profiles are preserved and inspected. No universal
timing correction, causal write/read claim, or algorithm promotion follows.
