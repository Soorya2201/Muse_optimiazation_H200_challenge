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
