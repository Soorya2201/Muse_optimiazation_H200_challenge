# Public MoE research and reproduction record

## Oct30 exact KPERM3 control (P1)

Public source https://github.com/alygg77/okc-muse-spark-kernels,
Apache-2.0 full commit `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`.
Pinned exact `submission_v24_alygg77_6cae49a.py`, SHA-256
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`.
Byte equality against the public raw source and AST syntax were verified.
Upstream attribution includes vjeux and Matsvei; notices are preserved.

KPERM3 permutes codes within each256-wide K stage, repacks the same scale
bytes, and gathers matching activation columns. Gate/up row ordering makes
the intermediate align with phase B's K permutation. The main WGS consumer
loads eight scale bytes once per stage and reuses them across four load
groups, reducing live registers and repeated scale loads. Tiny/mid builds
stay selected for M<=16. `prepare` uses weights only; `kernel_fn` reads fresh
activations/routes and allocates a fresh output. No input/output cache exists.
The new context binder handles fresh host threads. Standalone PDL is disabled.

Exact check `802e58c6f9964fc5` passed on sfo3 at2.404125x,
CI2.368063-2.440187,all5tests/6timings,prepare167.36s. Every profile/log was
read and archived. Independent repeat `6ad96077d56c42d9` passed on sfo3 at
2.376541x,all5tests/6timings,prepare164.85s. Same-day parent control
`8ceabe44c7624c93` passed on sfo3 at2.332013x,prepare169.94s.
Changed M32/128/256 improve2.24-5.10% in normalized speedup in both checks;
unchanged paths show smaller/mixed shifts. Aggregate gains3.09%/1.91% with
overlapping intervals; exact magnitude remains uncertain. All profiles/logs
were read. Promoted exactv24; scored job `ce241acf7ab84c04` passed at
2.372309x,CI2.336725-2.407894,5668.949us,prepare164.59s,sfo3.
Live MoE#79/1150,overall#50/1651 at capture. All scored profiles/logs archived/read.
See `KPERM3_COMPARISON_2026-10-30.md` for every shape and limitations.

AST comparison confirms every common literal configuration assignment matches
the parent, including tiny/mid/main defines. Native `CUDA_SRC` differs in only
two WGS sections (32 unified-diff lines): input gather permutation and per-stage
scale reuse. This is source isolation, not a claim of identical compiled SASS
on small shapes. Use M1/8/16 as controls for memory-layout or timing shifts.
P1 explicitly requires this published control even though upstream's claimed
board gain is only about1.25%, below the gate for new speculative microchanges.
Check twice and compare affected M32/128/256 against unchanged M1/8/16 on a
same-day/same-cluster parent before deciding. No local speedup claim yet.
Prepare must stay within300s; the public measured range273-312s is a risk.
AST-identical repeat v25 and parent control v26 are archived for fresh runs.

The public timing kit is a future independent-GPU tool, not submission code.
It uses memory-layout variation and an A/A arm; its stated identical-source
spread is another reason not to promote a small single-check score increase.

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

Check949c14d4f0b14a9c passed all5tests/6timings at **2.363739x**,5689.503us,
CI[2.328283,2.399195],prepare136.92s;SHA
d50015ce9261a43dfc107314cb8abae1434bb02c97807545dcf1d2e65060b4e8.
The prefill port gave only0.06–1.58% expert-phase gains. This parent's shift multipliers
are literals,whereas prefill deliberately obscures them to force IMAD. That compiler
difference motivates one direct test;source counts do not establish generated SASS or speed.
The gain is1.71% over the scored parent but only1.03% over the exact parent check,with
overlapping confidence intervals. Preserve as a promising component;do not claim a proven
isolated gain or promote yet. All profiles/logs were read,no failures.

The next isolated test `submission_v21_register_redistribution.py` returns to exactv18,
without the8-op port,and adapts the `OKC_WGSREG` path from the reviewed c1809247 source.
OnlyM>=64 uses640threads:512consumers request112registers after128producer threads
release to32registers. Demand512*(112−96)=8192 equals128*(96−32)=8192 released registers;
the initial640*96=61440 allocation fits65536. The two idle producer warps join the
one CTA-wide start barrier before returning. Consumer thread0 computes stage ranges after
gather,keeping the producer's reduced register region free of that call. Weight layouts,
precision,scratch,and per-stage readiness remain unchanged.

Prepare queries real kernel register/local-memory sizes and occupancy for both variants;
it selects the extra module only if at least one CTA can reside per SM. Build/resource
exceptions retain the parent. No timed host tensor reads or data-value dispatch is added.
Check83c410ffc02141f1 passed5tests/6timings at2.324033x,5786.707us,
CI[2.289173,2.358894],prepare220.34s;SHA
0e21741cd1520251bc0a5a5d754e3da09dc35980393c2002d13d0deec3e04d41.
Native resources (blocksFalse/True,registersFalse/True,localBytesFalse/True,sharedBytes,threads)
were parent[1,1,96,96,0,0,231168,576] and REG112[1,1,96,96,0,0,231168,640].
REG112 was selected forM>=64,but the aggregate tied the scored parent. Both builds already
had zero local-memory allocation and one resident CTA/SM. Redistribution neither reduced
spill traffic nor increased residency;moving range computation onto the consumers also
changes overlap. No causal benefit was established;not promoted. All profiles/logs read.

An NVCC13.0/sm90a Compiler Explorer micro-probe compiled the two conversion expressions.
Both used14registers with zero stack/spill. The device SASS section (not the host asm view)
contains legacy5LOP3+2SHF+2IMAD.SHL versus8-op5LOP3+2SHF+1IMAD. Thus the literal-multiplier
hypothesis does not create extra ALU savings in this probe:the old constants already become
IMAD.SHL. This is minimal-kernel evidence,not a full-kernel SASS/performance claim.
The first API response's top-level asm was host x86 and was not interpreted as GPU code;
the follow-up inspected `devices[SASS (compute_90a, sm_90a)]` explicitly.

Check683c1eb4664a4396 now runs `submission_v22_paired_dequant_gate.py`,SHA
c40cbe4a68fbfb5a13eb4cb37d4b8945a9c5d1fa2f8da3b3868f40b5fa6fa142.
It preserves the exact parent,compiles the verified8-op helper as an alternate module,
and repacks matching alternate weights without changing precision. Three synthetic routed
inputs per token bucket are shared by both layouts. Outputs must agree within0.2% and
two independent alternating-order,cold-L2 groups on each input must all show>2% gain.
Otherwise shape-only dispatch retains the parent. Compile/packing/measurement time caps
retain the parent;all synthetic scratch is zeroed before returning state. No input-derived
graph/output survives in state. Offline source equivalence and34848-word checks pass.
Five of eight checks and one of two submits used. Keep exactv18 in submission.py.

Check683c1eb4664a4396 passed5tests/6timings,2.336220x,5756.521us,
CI[2.301177,2.371263],prepare214.50s. Every token bucket retained the parent.
Observed ranges across the six paired groups:

|Tokens|Old/new time factor|Decision|
|---:|---:|---|
|1|0.97863-1.00250|Reject; usually slower|
|8|0.98924-0.99764|Reject; consistently slower|
|16|0.99782-1.01605|Reject; mixed/subthreshold|
|32|1.00156-1.01814|Reject; subthreshold|
|128|1.00688-1.01119|Retain as small component,not promoted|
|256|1.00439-1.01563|Retain as small component,not promoted|

Maximum output relerr1.428e-5;all log/profile files inspected. The aggregate point is
parent-run variation,not evidence of an adopted8-op improvement. This corrects the tempting
cross-job interpretation of v20's1.71% edge against the older scored run.

### v23: early W2 L2 prefetch (rejected)

The lee101 MoE gap study suggests prefetching the small-batch W2 working set during W13:
https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/docs/moe_gap_analysis.md
The parent already prefetches its first B ring units near the end of A,but does not issue
the entire touched W2 working set at routing completion. This is the missing mechanism
being tested,not a claim that all cross-phase prefetch was absent.

`submission_v23_w2_prefetch.py` keeps the original packing/math and compares two extra
tiny modules:default-policy bulkL2 prefetch,and keep-W2 plus evict-first streaming W13.
OnlyM1 dispatch may select them,and the device hook only issues prefetch for<=4local
jobs (<=36MiB W2). Each CTA thread0 prefetches round-robin36KiB weight units after route;
the existing early W13 fetches are already in flight. No new waits,barriers,or arithmetic.
Potential failure:extra HBM traffic competes with W13,or W2 is evicted before use.

Three independent synthetic inputs,two alternating cold-L2 groups of20replays each,
all six ratios>1.03 plus output relerr<=0.002 are required before selecting a module.
Other shapes retain parent. All synthetic scratch is reset;graphs/activations are not
returned. Compile/validation/time failures retain parent. Source-prefix audit and syntax
pass;builder `tools/build_moe_w2_prefetch_candidate.py` reproduces the self-contained file.
SHAfe3eda4762831116ea4028a26ec00184c0066444f15226ac7c02a2fb69488657,
check40673e1600084298,sixth of eight checks.

It passed all5tests/6timings at2.328927x,5774.546us,CI[2.293994,2.363861],
prepare205.66s. The three synthetic routes touched3/3/2experts. Default-policy factors
were0.87491-0.90746 (10.2-14.3% slower),and keep-W2/evict-first-W13 factors
0.91842-0.97213 (2.9-8.9% slower). Relative output errors1.364e-4/4.033e-8 pass validation.
Both variants fail the3% gate;the timed submission retains parent everywhere. This is
consistent with competing HBM/TMA traffic or eviction,not proof of a specific hardware
counter cause. Logs and all6profiles read. No promotion;stop this prefetch idea and move
to attention. MoE second cycle ends at6/8checks,1/2submits,scored2.324062x unchanged.

Official PTX documentation supports `.L2::cache_hint` on sm90:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch
Use this older form;the newly documented `.L2::evict_last` spelling requires newer hardware.
A public minimal NVCC13.0/sm90a compile probe passes with4registers/no spills. That is
syntax/target evidence,not full-kernel GPU validation or performance evidence.

### Three additional source screens,Oct29

|Source pinned revision|Reviewed MoE content|What it adds versus our parent|
|---|---|---|
|[operatorstack/open-kernel-challenge-solutions](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_moe_w4a16)|Published source SHA281d56e94d22f7ebb440274d25c240697b4728af2586da252fb1befb1f35cc35;MIT with SGLang Apache portions preserved in NOTICE. Public receiptc33ef77ff5f14a25=1.832036x,7340.741us,CI1.804555-1.859516. Reviewed actual decode ASM,packing,direct-M1,dispatch,and GEMM sections.|Tile-major/interleaved copies,register double buffering,direct-route M1 splitK2/4,and separate W13/W2 item sizes. Our parent already has contiguous code/scale units,register decode,TMA overlap,fused routing and phase scheduling. A multi-launch splitK path is an alternative architecture,not a proven additive win. Do not whole-copy it for its lower aggregate score.|
|[tommyyliu/okc](https://github.com/tommyyliu/okc/blob/f134725e27f368588bd9ef27cb5e97d023b415a5/muse_moe_w4a16/submission.py)|SHAcb41995890b9ca069f3615a134d37fccb617bf35838b3fad652786627ce7830d. Public-board snapshot1.8997x,#409 when our old entry was#626;now below our2.324062x. Reviewed all principal GEMM/packing/dispatch paths;no license discovered at repo root.|Four-launch Gluon path with adjacent code/scale loads,one-block register prefetch,lossless positive-E4M3 recoding. Our native parent already overlaps loads/decode and handles scale validation. Its heavier decode and extra launch boundaries supply no measured missing mechanism. Study only;no code copied.|
|[lilaclilac09/open-kernel-challenge](https://github.com/lilaclilac09/open-kernel-challenge/tree/d9bdcc9f01dff1018f2f256a0a1a63911a16557e/muse_moe_w4a16)|Actual submission is plain-PyTorch starter;notes/moe_nvfp4_sketch.py contains only a placeholder.|No optimized implementation to transfer. No GPU check spent.|

These deeper screens are bounded source inspections,not claims to have read every file
in each repository. GitHub HEAD checks confirm vjeux stillf35a41e and Madhavan stillc1809247;
neither has an unseen newer public revision at this capture.

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
