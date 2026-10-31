# Device timestamp and order calibration

V39is preserved as a correct but nonqualifying experiment. Its controls
occasionally differ by8%,and its nominally rotated/reversed schedule does
not distribute each arm evenly over positions:over eight rounds,each arm
uses just two of four positions. It cannot certify subpercent gains.
Inspection also shows host-issued events bracketing each graph replay;
their interval may include launch/host scheduling beyond GPU kernel work.
This second point is a hypothesis pending simultaneous measurements,not
a measured explanation of V39's variation.

This follow-up is a measurement-method repair/calibration,not an unchanged
scored resubmission or a new decoder optimization. Its purpose is to settle
these limitations before further component ports. Native code generation,
weight repacking,configuration specifications and numerical checks are AST
identical to V39;the actual challenge timed dispatch remains canonical.

The [PyTorch2.14 Event documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)
supports`external=True` for explicit record/wait nodes during graph capture.
The [NVIDIA CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
describes event-record nodes and distinguishes graph definition from launch.
Each synthetic graph now captures start-event -> operator -> stop-event.
No guessed replay cost or event cost is subtracted. Captured event nodes
can themselves perturb scheduling;matched/A-A controls and an event-only
graph are required,not an assumption of zero overhead.

Eight Williams-square rounds give every arm every position twice and every
ordered adjacent pair twice. CPU checks cover13group shifts and prove the
balance. Five nearby cases and the same72timing groups preserve all previous
numerical,resource,fresh-input and performance gates. On pad0/first seed,
simultaneous outer host-bracketed timestamps are preserved alongside the
inner timestamps for all nine routes. Event-only graphs give an additional
calibration. The method captures no harness objects,reads no harness state,
and affects only independent synthetic diagnostics.

Source`submission_v40_device_event_probe.py`,SHA
`8a056167d53d6ee072e9d6d2a8e2a1f48e14e6199dc6023be41b41dece645250`.
`DEVICE_EVENT_PREFLIGHT_2026-10-31.json` proves native/numerics identity and
all position/adjacency counts. The first builder draft used stale AST line
positions after a dependent edit;the local syntax gate caught it before
creating/submitting an artifact. It now reparses between edits. No GPU
credit spent on that construction failure.

The final report stores JSON metadata plus little-endian u32 raw-time frames
under a length-prefixed compressed envelope,retaining all event samples
under8KB. No sample trimming or convenient-row selection. The existing
extractor recognizes both codecs;V39evidence is unchanged. The opportunity
screen remains the nine-route phase-zero1.836550x bound;no timing-method
change is itself claimed to improve the leaderboard. Only a qualifying
changed native path could proceed to actual timed-path checks/repeats.

## V40 result and remaining flush mismatch

Check`1a1ec90bda294d64`,sfo3,passes9tests/30timings at1.572543945x,
CI1.548956–1.596132,3071.965025us,prepare168.587057s. This is the unchanged
parent timed path;the higher point score is not a decoder gain. Complete
1.004s diagnostic:5nearby/72groups,7708characters,all2304device-event
samples and288simultaneous outer samples intact. All30profiles,full available
logs,all frames/resources/errors were inspected. Full native resources match
V39;all outputs agree bit-for-bit,including changed inputs. No eligible route
or fixed policy;all511reject,best worst-group board proxy1.000504x.
Maximum A/A2.927%,matched-control offset5.430%;some attribution gates fail.

Event-only inner medians are about2.88–2.91us. Simultaneous host-minus-inner
paired median excesses span5.696–5.824us across the nine routes/four arms.
This measures the boundary difference on these instrumented graphs;it does
not isolate CPU starvation,prove pure-kernel time,or justify subtracting a
universal constant. Event-node spacing is material. No subtraction is made.
Balanced ordering/inside-graph events alone do not establish a native gain.

V39andV40still evict L2 with a writing`add_`. The judge uses read eviction.
That protocol mismatch was found during the new primary-methodology review;
it prevents promotion from these diagnostics. Existing results are retained
with their scope,not retroactively corrected or blamed on an unmeasured
mechanism. Old/new read identical byte counts at the identical pointer,so
a differential write-back effect remains unproven. One follow-up changes
only synthetic eviction to a read-sum,keeping native code/numerics/events/
balanced order/gates identical.

V41`submission_v41_read_flush_probe.py`,SHA
`53433583563e8a9227a12dfde2786c705714be720419ff016e5ec0d597eaeef0`.
It reads the same256MiB buffer and copies the scalar sum before each graph.
One initial read/synchronize drains initialization before measurements.
The time frames use exact u16nanoseconds when every sample fits,otherwise
u32;no quantization or dropped samples. A CPU codec roundtrip restores all
2304samples exactly. The earlier V40 u32codec remains supported.

## V41 terminal result: read-flush repair complete, no eligible gain

Check `b89f4e5d007047d9`, sfo3, passes all 9 challenge tests and 30 timing
cells: 1.542033518x, CI [1.5189030155, 1.5651640211], 3132.746430us,
prepare 166.691983s. The challenge still times the unchanged parent.
The independent read-flush diagnostic completed in 0.887s with all five
nearby validations and 72 groups, retaining all raw samples in 6852
characters (exact u16 ns frames). Initial and fresh-input outputs agree
exactly; maximum FP64-relative global/row errors are 0.00166718/0.00174107,
with zero one-hot error. All 18 full native resource records have zero local
allocation. Registers remain the V39/V40 pairs; standalone register parity
must not be generalized to full kernels.

| Route | Median matched-old/new ratio | Worst ratio |
|---|---:|---:|
| qkv8 | 1.003738 | 0.993985 |
| unpooler8 | 0.999992 | 0.986008 |
| qkv16 | 1.015004 | 1.003574 |
| O16 | 1.008650 | 1.000000 |
| unpooler16 | 0.998669 | 0.993625 |
| qkv32 | 1.002714 | 0.981644 |
| O32 | 1.004186 | 0.991689 |
| pooler32 | 0.990783 | 0.973742 |
| unpooler32 | 1.000879 | 0.987820 |

No route clears the fixed performance gates. Maximum A/A offset is 2.677%
and matched-control offset 3.410%, so some attribution gates fail too.
All 511 fixed nonempty policies reject. The best worst-group board proxy
is 1.000442840x (median 1.000841123x, routes 0/2/4); all nine routes give
0.998898345x worst and 0.999587229x median. Neither supports promotion.

Event-only inner medians are 2.912–2.976us; simultaneous host-minus-inner
paired medians are 5.824–5.920us. These instrumented boundaries do not provide
a universal correction. Differences from V40 across separate jobs, random
weights and allocations do not identify the causal effect of flush mode.
All 30 profiles, the full available remote log, all resource/numerical
records, all 72 raw device frames and all nine simultaneous host frames
were inspected. Across V39–V41, all 90 profiles and 216 timing groups were
inspected; each run rejects its 511 policies (these are overlapping policy
sets, not 1533 unique algorithms).

The external volume disappeared while the same V41 job was being recovered.
Its structured result was archived from that existing handle in the internal
checkout; no run was restarted. The receipt preserves complete remote
profiles/log data, not an invented original local CLI log. See
`run_receipts/b89f4e5d007047d9.json` and
`C12_PERMUTE_REPORT_b89f4e5d007047d9.json`.

Close the tested C12 sign-layout transfer. No source promotion, scored
submission or rank gain. BF16 Oct31 checks: 3/8 routine allocation; submits 0.
Move sequentially to decoder; preserve its small-T FP32 router contract.
