# Decoder replacement: same-day evidence

All three checks below used cluster `sfo3` on October 30. Each passed 13 tests
and 12 timed shapes; all profiles and logs were inspected. Full receipts are
archived under `run_receipts/`.

|Source|Job|Speedup|95% interval|Prepare seconds|
|---|---|---:|---|---:|
|Historical Madhavan control|be9500fc5b9b4a48|2.137329x|2.105269-2.169389|193.18|
|Exact alygg77|521b039c454045da|2.189801x|2.156954-2.222648|128.70|
|AST-identical alygg77 repeat|5c929cfd9c0b4215|2.221428x|2.188107-2.254749|125.32|

The historical control is **ineligible** at T <= 8 because it rounds the router
activation to BF16. It was checked only to compare same-day behavior; its scored
entry was withdrawn. Restoring FP32 routing is required regardless of speed.

|Batch|Query length|Layer|Historical us|Exact us|Repeat us|
|---:|---:|---|---:|---:|---:|
|1|1|sliding|72.95|71.48|67.92|
|1|1|full|79.15|76.11|75.55|
|16|1|sliding|128.26|132.00|126.96|
|16|1|full|205.45|202.08|206.18|
|32|1|sliding|155.37|154.27|145.89|
|32|1|full|309.14|301.95|292.30|
|1|8|sliding|111.15|111.51|105.07|
|1|8|full|118.40|111.41|117.61|
|16|8|sliding|254.86|243.38|240.29|
|16|8|full|339.79|324.77|330.18|
|32|8|sliding|321.64|310.27|313.98|
|32|8|full|488.92|473.46|475.43|

The candidate's aggregate normalized score is 2.455%/3.935% higher than the
historical control. All 12 per-shape speedup ratios versus SGLang increase in
both checks, although raw times on some shapes do not improve. Baseline timing
variation matters; many individual differences remain inside uncertainty.
The two unchanged-source checks themselves differ by 1.44% in aggregate.

This is a combined public-source improvement: FP32-router/glue fusion, KPERM3,
cuBLASLt selection and context/state handling differ. These checks do not isolate
KPERM3 causally. MoE remains dominant on sliding verify (about61-65% of summed
profile durations), with full attention large on full verify. PDL waits overlap;
do not add kernel durations to estimate attainable wall-time savings.

Decision: promote exact source SHA
`8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.
Scored replacement `eff0b59dc26c4c74` passed all13tests/12timings at
**2.211495x**, CI[2.178322,2.244667],10228.512us,prepare128.57s,cluster sfo3.
All profiles/logs inspected and full receipt archived. Live decoder rank#50/1007,
overall#50/1649 at capture. No rank1; decoder leader2.4178x still needs roughly
8.53% lower normalized time to match. The submit is +3.470% normalized score
versus the same-day historical control; these are separate jobs, not paired
same-input A/B experiments.
Next track is the standalone KPERM3 MoE control, per CODEX_NEXT_STEPS.md.
