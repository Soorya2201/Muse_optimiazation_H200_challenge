# KPERM3 same-day comparison, October 30

All runs used sfo3 and passed five tests/six timed shapes. Full profiles/logs
are archived in run_receipts. Exact-source checks: 802e58c6f9964fc5 at2.404125x
and 6ad96077d56c42d9 at2.376541x. Same-day parent 8ceabe44c7624c93:2.332013x.
Prepare167.36/164.85/169.94s, all below300s.

|Tokens|Parent us|Exact us|Repeat us|Normalized gain exact|Normalized gain repeat|Native path changed?|
|---:|---:|---:|---:|---:|---:|---|
|1|26.21|24.95|24.91|1.70%|1.34%|No|
|16|93.30|90.16|93.24|3.37%|1.39%|No|
|32|126.21|123.96|130.57|2.84%|2.24%|Yes|
|8|63.73|63.22|63.48|0.88%|-0.45%|No|
|128|184.72|171.49|175.05|4.73%|2.65%|Yes|
|256|220.92|206.80|209.16|5.10%|4.35%|Yes|

Normalized gains compare each job's paired speedup against SGLang, as the board
does. They are not ratios of the listed raw microsecond values. In particular,
repeat M32 has higher raw latency than the parent but a higher normalized score
because its SGLang baseline also ran slower. Do not claim an absolute latency
reduction on every shape.

The changed WGS shapes improve by2.24-5.10% in normalized score in both checks.
Unchanged paths show noise/layout shifts, including +3.37% at first-check M16;
this was only +1.39% in the repeat. Both native sources share all literal variant
settings and differ in only two WGS sections; weight preparation and the context
binder also differ. No per-SASS or causal same-input A/B claim is made.
Aggregate gains are3.09% and1.91%; intervals overlap, and identical-source
spread is1.16%. The exact gain remains uncertain.

Decision: promote the lossless public layout based on recurring changed-load
evidence and matching upstream measurements, then score it once. Original
vjeux source remains archived. Scored job ce241acf7ab84c04 passed at
**2.372309x**,CI2.336725-2.407894,5668.949us,prepare164.59s,cluster sfo3.
All five tests/six timed shapes pass; all profiles/logs read and archived.
Scored best rises2.076% from2.324062x; exact gain remains provisional.
Live MoE#79/1150,overall#50/1651 at capture. Leader2.6995x still requires
about12.12% lower normalized time to match.
No rank1 claim. Do not retry closed bs16, deeper-ring, fourth-slot or early
W2-prefetch directions. Next is BF16 exact Chen followed by alygg77.
