# Deep Dive: `chenwainuo/okc` BF16 Research

Snapshot: upstream commit [`3d418e1625f6de2075b5684a27ddbc3d1442422d`](https://github.com/chenwainuo/okc/tree/3d418e1625f6de2075b5684a27ddbc3d1442422d), audited 2026-10-29 PT. License: Apache-2.0.

## Why this audit extends beyond the pinned submission

`submission_rank78_chen_3d418e1.py` is the clean exported kernel that reports 3133.1 us / 1.542x. The upstream repository also preserves the development history: 188 candidate directories, four clean-room implementations, 74 archived official jobs (20 scored submissions and 54 checks in its frozen history), local H100 studies, source audits, and negative results. Those records explain which mechanisms produced the exported file and which attractive ideas already failed.

We preserve the exact shipped source locally, but do not copy all 188 candidate files. Most are superseded controls or measured regressions. Their pinned commit links, outcomes, and lessons are catalogued here so we can reproduce a specific branch without treating source presence as proof of a gain.

## Judged progression

The repository's preserved official history shows this progression before the final exported 3133.1-us file:

| Version | Job | Score (us) | Main change |
|---|---|---:|---|
| r6-c12 | `23cecdef1bb54c9e` | 3427.96 | Lossless C12, direct GV, tuned cuBLASLt, hi+lo router baseline. |
| pw23 | `17f4484b97644673` | 3399.26 | Pipelined split-K router128. |
| pw31 | `bbb732a78955425b` | 3337.13 | Self-zeroing small paths and revised router128. |
| pw33 | `8eeaca43bf43467f` | 3332.11 | Shorter-dependency QKV geometry. |
| pw34 | `846e25d9df09433f` | 3293.94 | RS128 selected after repairing a skipped prepare bake. |
| pw35 | `5d880cd0aa504209` | 3278.64 | One-node RTK router128/256. |
| rs32 | `fb548b5dc6f04696` | 3200.62 | Register-A WGMMA at m16/m32; retained older router. |
| pw36 | `e22a4b9d693d4902` | 3212.42 | rs32 plus RTK; component wins did not combine into a better total. |
| pw38 | `a3f0ebf25a754c01` | 3198.62 | Optional early-half epilogue plus RTK. |
| pw45 | `d5a125fcdcdc408d` | 3179.29 | Optional O32 HALF52 contenders. |
| rq32-predxor | `7b2da0f017d54e3d` | 3193.82 | RQ32 predicated escape repair; whole score regressed. |
| gv-predxor-v2 | `9916d34357b745cf` | 3157.04 | GV predicated-XOR repair; frozen-history best. |
| keepchoice | `c2b3b2f924e043a8` | 3199.39 | Alternative selector; regressed. |
| GV + RQ predxor | `22d677fa4501409f` | 3234.20 | Combined branch; regressed. |
| GV + private EH | `ed28c762f3274df1` | 3207.57 | Mixed local route gains, whole-score regression. |
| full GV/RQ/EH/O32 | `161bdb8de7354344` | 3187.14 | Complete compatible stack; still lost to GV-only best. |
| full + splitCTA M1 | `ae061d458eaa420a` | 3186.74 | Optional M1 split-CTA; no new best. |
| selected early-load | `40d856e085274f51` | 3175.90 | M1 keyed route selected; M8 kept parent; no new best. |
| final exported source | public README, job not stated | **3133.1** | Judge-faithful bake, conservative proven picks, forced cursor routes, later router/RQ refinements. |

The central lesson is that locally positive components are not additive. Several combinations scored worse even when their individual routes won prepare-time comparisons.

## Complete approach taxonomy

The 188 candidate directories divide into these source families:

| Family | Directories | Purpose |
|---|---:|---|
| Stack progression (`pw*`, `fable-*`, `r1`–`r7`) | 77 | End-to-end versions, controls, and integration trials. |
| GV/C12/small-m | 43 | Direct compressed GEMV, escape repair, geometry, early-load, X-prefetch, and codec work. |
| RQ/RS WGMMA | 20 | RS128/RS32 register-A tensor-core kernels, escape cursors, compact/HALF variants. |
| Operand-reverse/native/TMA | 24 | Alternate operand orientation, packed X, STMatrix, dual producers, and refill schemes. |
| Hardware/library/graph | 13 | Hardware compression probes, cuBLASLt graph/layout work, MX, and direct-O experiments. |
| Policy/integration | 9 | Combined selectors and full-stack composition trials. |
| Router-specific | 2 | Small-router and RTK-early experiments. |

The repository additionally contains four clean-room schools—latency, bandwidth, tensor-core, and contrarian—under [`proj_rsi/cleanroom`](https://github.com/chenwainuo/okc/tree/3d418e1625f6de2075b5684a27ddbc3d1442422d/proj_rsi/cleanroom).

## Exact candidate-directory inventory

All 188 upstream candidate directory names at the audited commit are retained here for searchability:

### Stack progression (`pw*`, `fable-*`, `r1`–`r7`) (77)

```text
fable-e1
fable-e2
fable-e4
fable-e5
pw1
pw10
pw11
pw12
pw13
pw14
pw15
pw16
pw17
pw17r-router-dropped
pw18
pw19
pw2
pw20
pw21
pw22
pw23
pw24
pw24h
pw25
pw26
pw27
pw27h
pw28
pw28h
pw29
pw29h
pw29m
pw3
pw30
pw31
pw32
pw33
pw34
pw35
pw36
pw36_wring_rebase
pw37
pw38
pw38_hotpath_trace_trial
pw38_hotpath_trace_v2_trial
pw38_selected_profile_trial
pw38_trial
pw39
pw39_trial
pw4
pw40
pw40_trial
pw41
pw42
pw42_admission
pw43
pw43_trial
pw44
pw44_trial_v2
pw45
pw45_private_eh_screen_v1
pw45_trial
pw5
pw6
pw7
pw8
pw9
r1-compose
r1-splitk-cuda
r1-tensorcore
r1-triton
r2-deepgemm
r3-fast
r4-fast
r5-fast
r6-c12
r7-bulk
```

### GV/C12/small-m (43)

```text
group8_qkv
group8_qkv_runtime_trial
group8_qkv_trial
gv53_earlyload_v1
gv53_earlyload_wonly_v1
gv53_exactso_counters_v1
gv53_keyed_choice_v1
gv53_keyed_half_v1
gv53_maskgate_trial_v1
gv53_maskgate_v1
gv53_quarter_sparse_header_v1
gv53_quarter_sparse_trial_v1
gv53_quarter_stream_trial_v1
gv53_quarter_stream_v1
gv53_selected_earlyload_trial_v1
gv53_selected_earlyload_v1
gv53_splitcta_m1_selection_v1
gv53_splitcta_trial_v1
gv53_thin_cabi_validation_v1
gv_codec_frontier_v1
gv_lt_unpooler256_balanced_v1
gv_predxor_keepchoice_v1
gv_predxor_pw45_v1
gv_predxor_pw45_v2
gv_predxor_trial
gv_private_eh_o32_policy_trial_v1
gv_private_eh_o32_v1
gv_private_eh_policy_trial_v1
gv_private_eh_screen_v1
gv_qkv_half_xprefetch_v1
gv_qkv_spill_tradeoff_trial_v1
gv_qkv_xprefetch_v1
gv_top8_exponent_stats_v1
gv_unpool16_deltakey_robust_v1
gv_unpool16_deltakey_v1
gv_unpool16_lanebase_v1
gv_unpool16_xprefetch_choice_v1
gv_unpool16_xprefetch_v1
qkv_geometry_followup
smallm_increment
smallm_increment_trial
smallm_increment_trial_v3
smallm_matched_geometry_trial
```

### RQ/RS WGMMA (20)

```text
rq11_actualescape
rq11_codec
rq32_gv_predxor_pw45_v1
rq32_predxor
rq32_predxor_trial
rq_compact256_half_trial_v1
rq_compact256_half_v1
rq_descriptor_offset_trial_v1
rq_footer
rq_pooler32_escape_exposure_v1
rq_pooler32_escape_lookahead_trial_v1
rq_pooler32_exactso_counters_v1
rq_unpooler32_descriptor_offset_v4
rs128
rs128b
rs128p
rs128s
rs256full
rs32
rs32_eh_backport
```

### Operand-reverse/native/TMA (24)

```text
aligned_epilogue
native_dedicated_xwarp
operand_reverse
operand_reverse_dual_producer
operand_reverse_dual_stmatrix
operand_reverse_fence_pair
operand_reverse_group8
operand_reverse_group8_m256
operand_reverse_group8_stmatrix
operand_reverse_packed_x
operand_reverse_stmatrix
operand_reverse_v2
operand_reverse_v3
operand_reverse_v4
reverse_tma_x
reverse_tma_x4
reverse_tma_x4_stmatrix
reverse_tma_x4_stmatrix_trial
reverse_tma_x4_trial
reverse_tma_x_async_refill
reverse_tma_x_async_trial
reverse_tma_x_stmatrix
reverse_tma_x_stmatrix_trial
reverse_tma_x_trial
```

### Hardware/library/graph (13)

```text
hw_compression
hw_compression_byteplanes
hw_compression_byteplanes_inverse
hw_compression_counters
hw_compression_owner
hw_compression_patterns
hw_compression_signlow
hw_compression_signlow_zero
hw_compression_zero_control
hw_native
lt_graph
mx
o_direct_w
```

### Policy/integration (9)

```text
combined_policy_trial_v1
combined_positive_selection_v1
complete_gv_rq_eh_o32_policy_trial_v1
complete_gv_rq_eh_o32_splitcta_m1_v1
complete_gv_rq_eh_o32_v1
eh_same_geometry_trial
policy_integration_trial_v1
policy_selection_trial_v1
policy_selection_trial_v2
```

### Router-specific/other (2)

```text
router_small
rtk_early
```

## Mechanisms that actually survived into the strong lineage

### 1. Lossless C12 packing and direct GV

- Encode BF16 as sign + seven mantissa bits + a four-bit exponent offset.
- Store rare exact XOR repairs in stage-ordered escape lists.
- Decode directly from global registers into FP32 accumulation at small m.
- Retain tuned cuBLASLt for shapes where the custom path loses.

This is the foundational 3427.96-us jump and remains present in every later winner.

### 2. Self-zeroing and graph-node removal

Small router/unpooler paths remove memset and cluster edges. The router m1 path uses a single-block reduction, while m8/16/32 and sliced unpooler retain specialized direct paths. Fixed graph-node cost matters at these microsecond-scale shapes.

### 3. Shorter QKV dependency chains

The pw33 lineage changed small-QKV geometry to expose more instruction-level parallelism. Later RS32 overrides some QKV shapes, but the shorter-chain family remains useful for routes not taken by WGMMA.

### 4. RS128 register-A WGMMA

The m128 kernel decodes C12 weights into WGMMA A registers, sends X through TMA shared-memory B, uses two math warpgroups and a five-slot ring, and reduces split-K partials through bulk tensor reduction. The judged winner selected QKV128 and pooler128; unpooler128 and m256 often remained on the incumbent.

### 5. RS32/RS16

Narrow token tiles extend the same register-A design to m16/m32. The best archived job selected QKV16, QKV32, O32, pooler32, and unpooler32. This produced the largest late-stage score improvement: 3293.94 to 3200.62 us.

### 6. One-node RTK router

RTK accumulates BF16 high and low planes in one WGMMA kernel at m128/256 and self-zeroes its output. It was an authentic route win, but combining it with the rs32 stack produced 3212.42 us versus rs32's 3200.62 us. The final exported source keeps later, conservatively selected router descendants.

### 7. Predicated-XOR and per-thread escape cursors

GV predicated-XOR repair produced the frozen-history best of 3157.04 us. The later exported source adds per-thread staged cursors to RS routes, warp votes to skip empty repair blocks, and a shorter high-byte apply path. Its comments record resident-weight gains of roughly 0.66–1.38 us for selected m16/m32/m128 routes, while streamed deltas were near zero. These routes are forced only after correctness validation because cold-L2 bake noise cannot reliably resolve their fixed-latency benefit.

### 8. Judge-faithful preparation and conservative selection

The exported source replaces a `torch.sum` cache flush that left small weights partly warm with a validated read flush larger than L2. It alternates candidate and baseline graph replays, subtracts fixed graph cost, uses the median paired ratio, and imposes 2–3% replacement margins around proven picks. This solves the repeated failure where a path looked 0.2–2.2 us faster during preparation and then lost officially.

### 9. Final-source refinements beyond the frozen 3157-us history

Comparison against `gv_predxor_pw45_v2` shows the 3133.1-us export is a close descendant: 87.0% line similarity, 5,265 matching lines, and 1,354 inserted lines. The additions include:

- pw55 judge-faithful cold-L2 paired baking;
- pw60 forced per-thread cursor variants at m128;
- pw62 cursor twins for QKV/pooler/unpooler m32;
- pw63 QKV16 cursor and an O32 two-CTA HALF twin;
- conservative proven route tables from later judge sweeps;
- later router variants and selection safeguards;
- dropping O128 and all m256 RQ paths after losses greater than 1.9 us.

These are likely the main mechanisms behind 3157.04 to 3133.1 us, but the public README does not expose the final job ID or a formal ablation, so the exact causal contribution remains unproven.

## Useful unresolved or narrowly positive experiments

These are stored as research leads, not ready-to-submit gains:

| Experiment | Evidence | Decision |
|---|---|---|
| Unpooler16 X prefetch | Local paired +0.576 us, 21 wins/3 losses; official check with the route selected scored 3225.13 us. | Mechanism is real locally; whole-policy transfer failed. Revisit only in an isolated exact-parent experiment. |
| GV53 early W/X issue | M8 local paired +0.192 us; selected production source scored 3175.90 us and retained the parent at M8. | Do not force globally. Selector/context effects dominate. |
| W-only early load | +0.096 us versus parent, but slightly worse than combined W+X arm. | Too small for an official attempt without a cleaner causal test. |
| Private early-half epilogue | Pooler128 locally positive; QKV128 estimator conflict; combined scored branches lost. | Preserve as route-local evidence only. |
| Separate W ring | About 0.38 us local QKV128 observation. | Unconfirmed and near noise; deeper rings were worse. |
| RQ descriptor offset | Unpooler32 +0.048 us paired; pooler32 regressed. | Scope strictly to unpooler if revisited. |
| Compact 256-thread HALF | Resource capacity proven, timing slightly regressed. | Useful infrastructure, not a speed win. |
| Constant-depth entropy packets | Conditional model says a uniform 11.764% cut across selected GV cases could reach 3000 us. | Format still unproven; C11's repair explosion is strong counterevidence. |
| CTA-cooperative quarter-stream decode | Earlier first MMA achieved, but extra header/repair work increased executed warp instructions and regressed. | Needs a different packet/control design, not parameter tuning. |

## Closed directions we should not repeat

| Direction | Why it failed |
|---|---|
| Variable-length codec E / fixed C11 | Exact, but 256–286 warp instructions per 1024 weights versus C12's 49; C11 adds about 1.3% escapes. |
| Decode warps writing BF16 shared memory | Hid decode but lost over 10 us to X delivery and shared-memory traffic. |
| Plain BF16 hand pipelines | Parity against `torch.mm` but behind tuned cuBLASLt. |
| Stock CUTLASS and mixed-input CUTLASS | Best configurations only reached `torch.mm`; remained behind tuned Lt/RS128. |
| DeepGEMM BF16 port | Routes BF16 back to cuBLASLt; no direct advantage. |
| TMA rings for m <= 32 | About 4.6 us overhead versus direct register streaming. |
| Small-m cross-CTA or cluster reduction | Adds 1–3 us; cluster scheduling leaves CTAs underfilled and adds launch cost. |
| QKV tail balancing | The gap came from X/decode/MMA work rather than the tail. |
| RS128 at m256 | Load and epilogue costs erase full tensor utilization; 0–3 us behind cuBLASLt. |
| Router X-sharing cluster | Deadlocked with self-zeroing when the grid required two waves. |
| Reverse-operand/native-X/STMatrix ladder | Improved its own weak parents but remained roughly 7–33 us behind the actual incumbent. |
| Direct O, alternate Lt graph layout, small router | Measured regressions of roughly 0.9–4.1 us on their target cells. |
| Hardware compression representations | Counter reductions on synthetic zeros did not translate to real-weight matmul gains. |
| Four clean-room submissions | Zero strict wins across 120 school/shape comparisons against the faster incumbent bracket. |
| Soak process during measurement | Raised the replay floor to about 12.6 us and corrupted comparisons. |

## What we should implement from the other work

1. **First reproduce the exact 3133.1-us export.** This validates the whole upstream stack under our evaluator and is still the highest-value experiment.
2. **If reproduced, promote it as the new parent.** Do not merge two independent full stacks; our current v18 and this source share only 171 lines and a 0.0248 line-sequence ratio.
3. **Extract the final-source ablation next.** Compare the export against its 3157.04-us `gv_predxor_pw45_v2` ancestor to measure judge-faithful baking plus forced cursor routes as one controlled delta.
4. **Then isolate only plausible route transplants from our incumbent.** Our profile suggests O, large router, and tiny unpooler may be competitive; measure each against the new parent rather than combining cross-run minima.
5. **Keep the upstream closed-track list binding.** New work must change the architecture or causal mechanism, not retune a rejected stage depth, cluster size, or codec width.

## Pinned source references

- [Final exported submission](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/muse_bf16_proj/submission.py), local SHA-256 `bb50079be2ed74d65bb7109f1432fbc7a49bbe70913d247c254717cd0f27f712`.
- [Archived 3157.04-us GV-predxor parent](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/proj_rsi/candidates/gv_predxor_pw45_v2/submission.py), SHA-256 `501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9`.
- [Frozen official history](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/proj_rsi/HISTORY.md).
- [Closed tracks](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/handover_folder/10-closed-tracks.md).
- [RS128 design](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/handover_folder/04-track-rs128-m128.md).
- [RS32 design](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/handover_folder/05-track-rs32.md).
- [Router design](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/handover_folder/06-track-router-pw35.md).
- [Escape-path anatomy](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/handover_folder/07-track-p4-anatomy.md).
- [Large-GV opportunity ranking](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/audits/projection-large-gv-designs-20261027/ranked-summary-v1.md).
- [Candidate tree](https://github.com/chenwainuo/okc/tree/3d418e1625f6de2075b5684a27ddbc3d1442422d/proj_rsi/candidates).
