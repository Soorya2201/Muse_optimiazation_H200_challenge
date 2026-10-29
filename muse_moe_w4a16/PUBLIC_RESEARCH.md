# Public MoE research and reproduction record

## Oct29 second cycle: per-shape source audit and exact dequant port

The previous goal turn made verified progress by scoring the new whole decoder and prefill
architectures and completing all five above/below account screens. This cycle resumes MoE.
Its Oct29 budget before the new experiment was2checks/1submit;reserved maxima remain8/2.

Zhebrak source is pinned atff4d55ab669977f15874943044d20312d98a44d1:
https://github.com/zhebrak/muse-spark-kernels/tree/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_moe_w4a16
The Apache-2.0 source SHA is1043a88e5e2cca48f14792f17162044feeccb6b4e0a3baa732734d31e802fc35.
Public scored receipt3d6cadbe6117411d reports2.018186x,6663.659us,CI[1.987913,2.048458].
It uses per-expert readiness and persistent stream-K above64 tokens. Deeper review finds
our parent already has finer readiness:each phase-A epilogue publishes a release-add
credit to `doneAs[job,phase-B-K-stage]`,and the phase-B x producer waits for eight credits
with acquire loads. There is no whole-grid A-to-B barrier;the grid barrier precedes the
final norm/sum and scratch reset. The parent's introductory comment overstates its phase
barriers. Adding generic per-expert readiness would duplicate existing synchronization.

|Tokens|Our scored paired us|Zhebrak receipt paired us|
|---:|---:|---:|
|1|24.88|31.15|
|16|99.41|102.51|
|32|127.40|141.61|
|8|68.30|75.79|
|128|180.10|211.58|
|256|217.50|258.53|

These are cross-job comparisons,not a same-input causal ablation. Nonetheless there is no
per-load receipt win to justify spending a check on the complete slower implementation.
GPU-busy profile snippets can look faster at bs8,but the paired score is75.79us;overlap
duration is not a substitute for the reported paired time. Per-expert readiness remains
an architectural reference,not an adopted improvement.

The next controlled candidate is `submission_v20_lee8_dequant.py`,derived from exactv18
with only lee101's8-op E2M1 conversion and matching lossless bit placement. It preserves
all scale/global-scale/norm precision,activation precision,launch choices,and scheduling.
All packed/natural-K/fat/unit representations share the changed layout and the same dq4.
`tools/verify_moe_dequant_layout.py` compares34848 old/new words,all16 codes,all slot pairs,
multiply-carry combinations,and deterministic random words. Fragment bits are identical;
all other native computation and packing functions are unchanged. Python syntax passes.
Apache-2.0 attribution is retained in upstream_lee101_NOTICE.

Check949c14d4f0b14a9c is running,SHA
d50015ce9261a43dfc107314cb8abae1434bb02c97807545dcf1d2e65060b4e8.
The prefill port gave only0.06–1.58% expert-phase gains. This parent's shift multipliers
are literals,whereas prefill deliberately obscures them to force IMAD. That compiler
difference motivates one direct test;source counts do not establish generated SASS or speed.
Do not promote a sub-CI change. Keep exact v18 in submission.py until a qualifying result.

## vjeux persistent fused MoE

Audited source: https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_moe_w4a16.py

License: Apache-2.0, preserved in `upstream_vjeux_APACHE_2.0.txt`.
Local candidate: `submission_v18_vjeux_f35a41e.py`, byte-identical SHA-256
`10e0690d7fc21244bf29f6df2916b89f0ddcbe8f94d7afe15983478d315883a3`.

Oct 29 live context: our 1.6329x ranks #624/1110; vjeux 2.3267x ranks #135, scored job
`06beba86b8784027`. This repository revision is not cryptographically tied to that job, so
only our reproduction may establish its score. Our check `f2d5efeeb8884bbf` now passes 5/5
tests and six scored cases at **2.339526x**, 5748.386 us, CI [2.304433, 2.374619]. Prepare
took 137.2 seconds. It is promoted locally and qualifies for scoring against the 1.6329x incumbent.

The current incumbent uses three kernels. The public design uses one persistent CTA per SM
and device-side routing, projection A with SwiGLU, projection B with post-norm scale, and final
RMS normalization plus route weighting. Grid barriers publish phase outputs. Large shapes use
register-A WGMMA and stream-K fixups; tiny shapes use helper-warp TMA rings and mma.sync.
It retains packed NVFP4 codes, scale bytes, original FP32 global scales, BF16 intermediates,
and BF16 norm weights. Fast scale conversion is guarded by actual weight-byte validation in
prepare. Fresh-input routing and all activation-dependent work occur in the timed launch.

Static review checked the active Python entry points and feature flags: DIAG=False,
DIAG_BW=False, OKC_NOLOAD=0, OKC_DEBUG=0. The timed Python path selects precompiled variants by
shape, aligns route IDs when needed, allocates output, and launches on the current stream.
Scratch generations coordinate barriers and sum buffers; they are not cached outputs. Runtime
correctness, graph replay execution, and prepare duration are now supported by the passing
judge run. This is benchmark evidence; it does not substitute for a formal concurrency proof.

The mechanism targets all three observed bottlenecks: repeated per-CTA routing, repeated hot
expert weight streams, and serial projection/finalization launch boundaries. Our AP/WP and grid
sweeps did not change those architectural costs. This is therefore a stronger next experiment
than another adjacent prefetch-depth sweep.

## Madhavan extension

Source: https://github.com/Madhavan113/kerneloptimizer/blob/c180924742f5ec30ee7434b1d99cf2c16d2aa777/muse_moe_w4a16/submission.py

Local source: `submission_v19_madhavan_c180924.py`, byte-identical SHA-256
`7b14d4bb3a6b1c580477cd017da9bab1ca178191ab32f9341dc2e3b1c1ec12de`.
The Apache-2.0 original license and derivative notice are preserved. Current live context:
2.3570x, rank #69, scored job `add303d3a3224b15`; the current file is not source-hash-tied
to that receipt. Python syntax passes; check `b601303f873c4a08` is now running after the
successful v18 control. It finished at 2.330435x, 5770.812 us, CI [2.295478, 2.365391],
with 5/5 tests and all six scored cases passing. It did not improve the aggregate. The
16/32-token paired times were 91.80/127.51 us versus the first check's 95.42/131.52 us,
but these are cross-job observations with different inputs and baselines; no isolated causal
gain is established. Scored submission `9835951891204c56` confirms v18 at **2.324062x**,
5786.636 us, CI [2.289201,2.358923], 5/5 tests and all six scored cases passing. New live
rank #146/1112 and overall #447/1599; prepare165.25s. Keep exact v18 promoted.

Compared with the vjeux parent, its timed selection adds a WGMMA FT_T=32 build for up to 32
tokens and moves the helper warp to hardware warp zero for the 16-token build. The latter aims
to issue producer work earlier under warp scheduling; the former avoids padding the WGMMA token
tile to 64. It also carries a separately disabled routing-logits API for whole-layer composition.
Only ordinary top-k MoE is active in this standalone test. Setup compiles the required variants
through concurrent load_inline calls, with no diagnostic builds enabled.

## Lower-ranked source lessons

### Completed whole-board account screen

`RANKED_GITHUB_REPO_SCREEN.json` preserves the 2026-10-29 account-by-account API evidence for
all 1112 ranked teams at capture: 625 above our then-rank #626, 486 below, and ourselves.
109 accounts had repository metadata matches. Five public accounts could not be resolved:
Abarbir12, Pinoeer-kingxi, accountforgoofystuff, cyber199306, and gurrapu21. These are recorded
as unresolved, not as proof of absent public work. Global GitHub code search supplements the
screen. The original pass reached the end but failed when sorting one unranked row; it was
corrected to include integer-ranked rows and rerun to completion. No GPU credit was used.

The strongest explicit challenge repositories in the profile screen were vjeux, ruwwww,
Madhavan's indexed ParadigmKernelChallenge repository, lee101, Ranoobaba, tommyyliu,
Aravind-11, infraax, xharshvardhan, and MadanChaollaPark. Global search also found the
kerneloptimizer repository and other non-ranked source owners. The complete keyword matches,
including SGLang forks and unrelated Muse projects, remain in the JSON.

The below-ranked [infraax campaign](https://github.com/infraax/okc-kernel-challenge) preserves
failed producer-ring and PDL experiments rather than a stronger current MoE kernel. The
[titoatwork FP4 study](https://github.com/titoatwork/qgemm-mx) has useful pack/occupancy/measurement
artifacts but explicitly has not implemented its performant Hopper kernel. Neither justifies
copying an unmeasured performance claim. Aravind-11's public repository likewise explicitly
contains early versions rather than the current board entry.

[lee101/openkernels](https://github.com/lee101/openkernels) reports a historical 1.418x MoE
result, below our old 1.6329x. Its documented heaviest-expert-first queue and release-add
publication are worth checking. Source inspection finds the vjeux WGS path already uses
`red.release.gpu.global.add.s32`, so that mechanism is already present in the stronger parent
and should not be counted as a new additive gain. Its activation-traffic/dequant ablations
support profiling those phases, but do not prove a speedup in our kernel.

The lower-ranked PostTrainLLM fused-normalization experiment reports about 1.224x. Our old
finalize is already only 0.8–6% of profiled time, and the public persistent parent includes it
inside its one launch, so copying only that standalone epilogue cannot plausibly close the
large W13/W2 deficit. Keep the source as reference rather than spend a whole-file check.
