# BF16 public-source and component refresh

The fresh account screen covers1526ranked accounts:229above our then-rank230,
1296below,and ourselves. Eleven profiles return404and remain unresolved.
`RANKED_GITHUB_REPO_SCREEN_2026-10-31.json` preserves every account result.
It screens public repository names/descriptions/topics,not all source or
private submissions. The public-code index separately returns244paths/
13repos,ten additions;changing indices are not exhaustive source coverage.

Fourteen repository matches are new relative to Oct30. Their exact default
heads,licenses and bounded follow-ups are in
`NEW_REPO_FOLLOWUP_2026-10-31.json`. Six SGLang forks,including RunFMe#13and
islamborghini#78above us,show0commits ahead and no changed files against
sgl-project/main on their default branches. No all-branches claim is made.
The remaining new NVFP4/group-GEMM utilities are deferred to that track;
pocket-muse is a separate model project,not a projection-kernel lead.

## Newly pinned lossless BF16 library

[MrNirmata/lossless-bf16](https://github.com/MrNirmata/lossless-bf16/tree/4429b1d036a77b8ac8b7d63682ef7dceb32a0ac2),
Apache-2.0,27selected source/docs/data/test files,197689bytes. All Git blobs
and SHA-256 hashes verified in
`research/public/lossless_bf16_4429b1d/PROVENANCE.json`. Owner#510at capture.
Four raw CUDA kernels are supplied:rowpair,full-K direct loads,full-K TMA,
equal-byte split-K with a PDL finalizer. Their SHA-256 hashes match the
author's four-file manifest. No kernel code was ported or GPU-executed.

Its distinct component is a row-wise power-of-two exponent normalization
to a fixed16-binade window,allowing a7-instruction decoder. Per-row scales
and sparse original-bit escape records preserve storage exactly. This goes
beyond V39's9-instruction variable-base helper,but transferring it also
changes scaling/escape arithmetic and epilogues. It is not a free two-op
replacement or evidence that the earlier transfer wins.

After reviewing its lazy imports and CPU-only code,ran its complete
`test_format.py` and`test_emulation.py` on local CPU PyTorch2.8.0:77passing
test indicators,exit0;original log and command archived in
`LOSSLESS_LIBRARY_CPU_TESTS_2026-10-31.{json,txt}`. This includes all65536BF16
patterns,layout/emulation cases and the suite's numerical-limit examples.
These tests do not certify the CUDA wrappers,FP32 behavior on arbitrary
inputs,or H200 performance. The author explicitly says the new package/
bindings/tuner were compile-checked but have not been run on a GPU;the
published R1–R3measurements concern the raw kernel sources and earlier host
packing,not the new package end to end.

Published paired results do not establish a faster current-parent component:
qkv8~11.6–11.8us,pooler16~16.6–16.8us,unpooler16~15.9–16.0us are above our
current-parent controls. These are different processes/inputs and different
reference dispatch,so they are screening evidence,not matched performance
proof. No whole-library GPU check is allocated. The parent already has
lossless C12 and more advanced selected paths;new library value is portable
format/packing/tests and component study,not its cuBLAS headline ratio.

Numerical caveats matter:the full-K kernels combine placeholder cancellation
with FP32 sums;very deep escaped values can lose their contribution. The
author documents subnormal/fast-math limits and distinguishes bit-exact
storage from arithmetic exactness. Task one-hot checks still need independent
validation before any port. Weight generation in both the library benchmark
and challenge is FP32 Gaussian -> BF16;no distribution mismatch is claimed.

## Newly pinned H200 methodology notes

[MrNirmata/h200-decode-notes](https://github.com/MrNirmata/h200-decode-notes/tree/4f195ed71190f5556d48adbf23a0e1a9db5412b1),
CC-BY-4.0. Ten selected unmodified method/note/data files are pinned with
attribution and hashes in`research/public/h200_notes_4f195ed/PROVENANCE.json`.
Their single-author experiments are external evidence,not our replication.
Selected measurements concern read/write cache flushes,bytes in flight,
first-stage latency and graph boundaries. MoE-specific changes are deferred.

The read/write flush distinction is directly relevant:the judge reads its
flush buffer,while V39/V40's independent synthetic protocol uses`add_`,which
writes. The notes report kernel-dependent offsets across different runs;
they infer dirty write-back traffic without DRAM counters. Our old/new
decoder has equal byte counts and the same pointer,so that specific
differential effect is unproven here. Nevertheless matching the judge's
read-flush protocol is required before promotion. No causal conclusion or
retroactive numerical correction is applied to old results. V40separately
repairs timestamp placement/order;the dated method report records follow-up.

## Additional entrant below us

[ScrashOff/okc-muse-kernels](https://github.com/ScrashOff/okc-muse-kernels/tree/d870912e7c5a030c52050bb2445418224c28d9de),
owner#909at metadata capture,license unestablished. Seven BF16 sources
(six history plus shipped),seven profile artifacts and README are fetched
only into a temporary research directory. All15Git blobs/SHA-256 hashes,
210external profile rows and static architecture/dispatch interpretations
are archived in`SCRASHOFF_SOURCE_SCREEN_2026-10-31.json`. No code executed,
ported or whole-source mirrored;other tracks are left for their turns.

The shipped file exactly matches v5,reported1.228593x check/1.214x scored;
v6checks1.222085x. It uses Triton FP32-FMA GEMV at M1,swap-AB BF16 dot for
M8–32,last-CTA split reduction and an allowed hi+lo router. Current-parent
native C12/selected routes already address those costs. Published BF16
cells supply no new large additive lead. These are static screens/external
profiles,not exhaustive per-line conformance audits or reproduced scores.

No public rank-one implementation is established by this refresh. Research
includes useful libraries from lower-ranked entrants while retaining the
earlier above/below coverage and its explicit limits.
