# Attention phase evidence, October 29

Check `13bc760e093343f3` passed all 16 tests and 12 timings at **2.605394x**,
1445.578 us, CI [2.566313, 2.644475]. The timed path is the parent; no algorithmic
speedup is claimed. Both modules built in 47.4 seconds; the full probe reported 47.6.
Raw extracted observations are in `phase_probe_13bc760e093343f3.json`.

The table uses global phase timestamps relative to the earliest CTA entry, in microseconds.
Raw graph times include replay overhead. These are diagnostic measurements on one independent
fixture per shape, with normal random FP8 keys/values and scattered page-1 addresses. They
are not the scored coherent-key fixtures. Debug output differs from control by at most
1.58e-5 relative L2; measured debug overhead ranges from 0.93 to 2.70 us.

|Batch|Query length|Window|Parent/debug raw us|First K/V median|Last head-0 loop mark|Last partial issue|Last merge-ready mark|Last finalizer-exit mark|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|1|2048|8.784/9.760|2.016|2.752|2.944|4.224|4.640|
|1|8|2048|10.560/11.600|2.240|3.744|4.416|6.048|6.720|
|1|1|0|16.512/17.952|5.632|9.888|10.080|11.872|12.640|
|1|8|0|18.928/20.640|4.544|11.712|12.576|14.528|15.424|
|16|1|0|100.928/102.496|4.832|96.480|96.704|97.856|98.496|
|32|1|0|178.640/181.344|4.576|171.776|172.096|173.408|174.048|
|16|8|2048|15.552/17.200|3.616|8.544|9.472|11.488|12.288|
|32|8|2048|19.456/21.248|3.680|12.096|12.928|14.816|15.680|
|16|8|0|112.320/113.616|6.400|103.328|103.808|105.600|107.072|
|32|8|0|195.552/196.480|7.136|186.848|187.296|189.280|191.200|

Interpretation and limits:

- At B1 sliding decode, the last partial-issue mark is 2.944 us and merge-ready is
  4.224 us: a 1.280 us global-stage interval. For B1 sliding verify it is 4.416 to
  6.048 us, or 1.632 us. Differences between median markers (1.50/2.37 us) also include
  CTA imbalance and must not be called pure counter latency.
- Batched full verify spends most of its elapsed time before the loop marks finish.
  Head-0 loop marks reach 103.328/186.848 us at B16/B32; finalizer-exit marks reach
  107.072/191.200 us. This supports a loop/scheduling experiment, but does not establish
  whether SFU, tensor cores or memory wait dominates that loop.
- The loop marker samples thread0/head0, not every consumer group. The last-to-last
  intervals also include other-head work and coordination. First markers are per-CTA
  first-segment observations; last markers can span later segments.
- Instrumentation perturbs register allocation, scheduling and memory traffic. Differences
  between graph timing and one cold eager trace do not justify subtracting a universal
  overhead from every phase. `output_issued` marks finalizer exit; some CTAs own no rows,
  and this is not a fence proving every output store has reached HBM.

Actions from this evidence:

1. The one-time flat-L8 softmax offset was checked next, passed at 2.593449x and did not
   improve the aggregate. Keep it as negative/inconclusive component evidence.
2. Test B1 sliding-L1 completion flags against the original contended arrival counter.
   Preserve the existing FP32 reductions, CTA publication barrier and finalizer. One warp
   polls at most 32 split flags in parallel; all flags reset before kernel completion.
3. If that protocol wins, validate it on different context lengths/page sizes and graph
   replays before considering a separate distributed-finalizer version for L8. Do not
   assume one CTA finalizing all 128 L8 rows will improve its cooperative parent.
