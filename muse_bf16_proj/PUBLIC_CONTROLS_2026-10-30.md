# BF16 public controls, October 30

Every source passed9tests and30timed shapes. Every profile/log was inspected;
full receipts are in run_receipts. Current parent is the original Nathen v18.

|Source|Job|Cluster|Score|Prepare seconds|
|---|---|---|---:|---:|
|Chen exact|f5ea143175564684|sfo3c|1.519434x|196.84|
|Chen identical repeat|640d50a9220444e3|sfo3|1.549794x|199.69|
|alygg77 exact|9c6378b77fb94bb8|sfo3|1.544358x|140.18|
|alygg77 identical repeat|9bc0fbb427aa45ac|sfo3|1.547060x|167.17|
|Nathen unchanged control|c200267a5274433e|sfo3|1.510571x|65.83|

Chen's cross-cluster spread includes route-bake and memory-layout differences;
do not attribute it to a code improvement or to cluster alone. The table below
uses only sfo3. It lists raw projection times in microseconds. Normalized gains
must use each job's paired SGLang speedup, not raw time ratios.

|Layer|Tokens|Parent us|Chen us|alygg77 us|alygg77 repeat us|
|---|---:|---:|---:|---:|---:|
|qkv_gate|1|9.53|8.83|8.62|8.74|
|o|1|5.51|4.89|5.18|5.24|
|pooler|1|13.68|13.55|13.24|13.41|
|unpooler|1|3.82|4.25|2.89|3.39|
|router|1|3.97|3.45|3.69|3.63|
|qkv_gate|16|10.21|9.76|9.64|9.57|
|o|16|5.38|5.32|5.53|5.58|
|pooler|16|14.19|13.85|13.98|14.14|
|unpooler|16|14.00|13.74|13.73|13.74|
|router|16|4.26|4.80|4.83|4.80|
|qkv_gate|32|11.96|10.25|11.08|10.74|
|o|32|5.94|6.33|5.95|5.90|
|pooler|32|16.92|15.13|15.40|14.84|
|unpooler|32|15.65|15.01|14.57|14.52|
|router|32|5.20|5.39|5.24|5.47|
|qkv_gate|8|9.66|9.32|9.28|9.10|
|o|8|5.17|5.36|5.49|5.41|
|pooler|8|13.59|13.54|13.74|13.62|
|unpooler|8|4.36|4.09|4.19|3.78|
|router|8|4.13|4.54|4.07|4.24|
|qkv_gate|128|13.94|12.09|13.90|13.82|
|o|128|7.28|7.39|6.93|7.10|
|pooler|128|20.97|17.36|20.42|20.06|
|unpooler|128|18.94|17.43|19.17|19.54|
|router|128|6.37|7.25|6.02|6.36|
|qkv_gate|256|17.35|17.32|17.82|17.93|
|o|256|9.06|9.34|8.72|9.52|
|pooler|256|24.22|24.73|24.60|24.65|
|unpooler|256|23.41|24.48|23.15|23.45|
|router|256|7.91|9.07|6.98|7.36|

|Load|Chen normalized gain|alygg77 normalized gain|alygg77 repeat normalized gain|
|---|---:|---:|---:|
|decode 1|2.79%|5.61%|3.93%|
|decode 16|1.12%|2.15%|1.13%|
|decode 32|7.38%|6.38%|8.97%|
|verify 1|-0.26%|0.21%|1.75%|
|verify 16|8.84%|0.32%|0.80%|
|verify 32|-3.76%|-1.03%|-1.77%|

Decision: promote exact alygg77 SHA290a86ab...7b90 based on recurring gains on
decode1/32 and its two stable same-cluster checks. Aggregate gains vs the
same-day parent are2.237%/2.416%, with overlapping95% intervals. Verify32
regresses1.03-1.77%; it is not a strict per-cell win. Shared O8/pooler8/pooler16
paths vary little. Native kernel rows confirm which implementations actually
ran; printed base choice tables do not describe every later wrapper override.
The parent used2 captured state copies versus4 for the candidates, so this
comparison includes state/layout effects and does not isolate one kernel change.
Scored job a3d4758c1e174aba passed at **1.535843x**,CI1.512805-1.558881,
3145.373us,prepare166.55s,sfo3. All9tests/30timings pass; allprofiles/logs
were inspected and archived. Scored best rises0.918% from1.521867x; this small
point difference is not a precisely established algorithm gain. Live BF16#133/1435,
overall#54/1654 at capture. Leader1.6498x remains about6.91% lower normalized time away.

## Remaining opportunity and why not to mix every winner blindly

Chen's 128-token qkv/pooler/unpooler paths beat alygg77's in these checks by
about1.7/2.7-3.1/1.7-2.1us. Swapping those three into alygg77 projects roughly
+1.7% on the board if the isolated differences survive integration. This is
an estimate, not a measured hybrid gain. Removing those three entire phases
would be a loose~31% board bound; their observed replacement savings are far
smaller. A blend would change packed-weight allocations, graphs and library
plans; these changed performance in previous composition attempts.

Do not claim the projected hybrid as a win. First run an independent-memory-pad
A/B/A comparison with the preserved timing kit on an authorized H200. If its
confidence bound clears the A/A spread, port only that RQ128 kernel/packing
subset, retain all original weight bits, validate one-hot and random/outlier
inputs, then check the integrated30-cell score. Avoid paying to compile both
entire donors inside prepare; keep its300s limit. No new GPU rental is assumed.

Closed items remain closed: sub12-bit coding has little usable headroom,
large-M split-K or integer decoding often loses, and the historical complete
Chen component stacks did not add their isolated gains. Existing v33 is an
M1-only blend; its likely board impact is small, so it is not the next priority.
