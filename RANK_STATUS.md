# Verified leaderboard snapshot

Captured **2026-10-31 18:24:13 UTC** from the live OKC boards. Overall **#145/1762**.
No track has reached rank1. Positions change as others submit.

|Track|Our rank|Scored speedup|Leader speedup|Time cut to match leader|
|---|---:|---:|---:|---:|
|BF16 projections|243/1543|1.535843x|1.6523x|7.05%|
|Routed experts|178/1248|2.372309x|2.7578x|13.98%|
|Attention decode|190/1190|2.603505x|3.2221x|19.20%|
|Whole decoder|145/1107|2.211495x|2.4178x|8.53%|
|Prefill|180/752|1.784958x|2.1551x|17.18%|

The final column is 1−our_score/leader_score at a fixed leaderboard baseline;
it is a planning gap, not a prediction. Prefill has a separate board and does
not count toward overall.

Scored receipts: BF16 a3d4758c1e174aba,MoE ce241acf7ab84c04,attention
30c669b9702d434f,decoder eff0b59dc26c4c74,prefill f62a284779b4447d.
All promoted GPU sources passed the full challenge checks first. Decoder,
MoE and BF16 were independently repeated and compared with same-day controls.
Small gains have overlapping uncertainty; see RESULTS.md and each track's
comparison report for limitations and every per-shape time.

The earlier overall#50 position was real at its capture time; other entrants
moved ahead. Decoder d857144e59784ad0 was withdrawn for the small-T FP32-router
precision issue; its old rank/score is ineligible. The compliant replacement
scores2.211495x. Current local decoder/MoE/BF16 files match their pinned
Apache-2.0 public sources, with Nathen's MIT notices retained where used.

Prefill context-hardening check a77a2557e1a24fa7 and submit25a1e0e5df0d4949
both passed. Its native GPU path is unchanged; no speedup claim. The old
eligible1.784958x remains the board best. All five boards' public-owner
metadata screens are refreshed; each dated follow-up records its limits.
The isolated serving MoE load_jit binding passed two H200 kernel checks
at2.369600x/2.384810x,including independent shard-offset/deepcopy/fresh-thread
graph probes. These do not establish a speedup;canonical/scored kernels are
unchanged. The port is not model-integrated. See
serving/MOE_PORT_STATUS.md. The repository was reverified PRIVATE at capture.
Current test budgets and source-discovery scope are in CODEX_HANDOFF.md and
PUBLIC_SOURCE_INDEX.md. Rank1 work remains open.

Two new prefill performance probes passed5tests/8timings each but found no
eligible faster pipeline/exponential combination. Their official timings
use the parent dispatch;they are not new algorithm scores. Independent
diagnostics/resources and losing cases are archived in
muse_prefill_layer/HYBRID_PIPELINE_2026-10-30.md. No scored kernel changed.

BF16 TMA probe a8d2c8d0433a4829 passed9/30 with parent timed dispatch.
Its independent comparisons rejected all15changed fixed compositions;
no new scored submission or rank improvement. One of32diagnostic rows
was truncated by the service log tail;remaining31 and limitation archived
in muse_bf16_proj/TMA_PUBLIC_FOLLOWUP_2026-10-30.md.

Three new attention checks are terminal. First independent probe failed a
masked-logit polynomial edge; corrected floor and bias-range probes each
completed six nearby validations and sixteen timing cases. Both reject all
changed combinations. Bias range helps the slower polynomial control by
3.28-6.08% speedup but remains27-47%longerthanincumbent. No scoredchange.
See muse_attention_decode/PK2_HYBRID_2026-10-30.md andarchivedreceipts.

Oct31 prefill cycle: four H200 checks completed,all5tests/8timings pass.
PIPE3 loses;ten public Lt choices across80fully captured groups fail gates.
One earlier Lt diagnostic was truncated and remains explicitly inconclusive.
Canonical sources/scores unchanged;no new scored submit. Detailed dated
reports and all32profiles are archived.

Oct31 decoder V20/check98ca0484be124a39 passed13tests/12timings while
its independent register/GS4 comparisons found no eligible gain. All16
groups and8changed policies fail gates;GS4 full build has32localbytes.
Canonical scores/ranks unchanged. See
muse_decode_layer/WGS_REGISTER_SPLIT_2026-10-31.md.

Oct31 MoE V27/V28 passed5tests/6timings each with parent timed dispatch.
The first trace perturbed performance;the repaired diagnostic haszero
localallocation andnear-parenttiming across24groups. Exactroute/cutdata
andtwoindividualCTAclockcaptures narrow scheduling hypotheses;no new
candidate score or rankgain. See
muse_moe_w4a16/PHASE_AND_LIBRARY_FOLLOWUP_2026-10-31.md.

Oct31 sampled-unit diagnostic e038ff14796944dc passes5tests/6timings but
fails timing attribution (8localbytes and11–16%traceinflation). Its2.374572x
point is the unchanged timed parent. The simpler flush diagnostic
b22622ebb61946df passes full resource/numerical/perturbation/control gates.
Its2.381024x point also uses unchanged timed dispatch; no scoredchange.
See muse_moe_w4a16/UNIT_FLUSH_FOLLOWUP_2026-10-31.md.

Oct31 main-loop cycle: V31/c364de8b14af4f36 and V32/72dff8caec5047d8
pass challenge checks with unchanged timed parent. Independent four-arm
comparisons each complete3nearby validations+24timinggroups with zero
agreement error and zero full-native local allocation. Both candidates
are consistently slower; all14per-run policy combinations reject. No
canonical/scored change. Expanded branches/gists and four component pins
are documented in muse_moe_w4a16/STAGE_LOOP_AND_BRANCH_RESEARCH_2026-10-31.md.

Oct31 attention V24/check9c335063894c4abc passes16tests/12timings with
unchanged timed parent2.604626x. Independent bit50/100 interpolation paths
pass six nearby coherent/cache checks,but fail the predeclared1%parent
global-error gate on the first random long-context fixture(1.461/2.008%).
No timing groups recorded,so performance remains unknown. No gate relaxation,
canonical change or scoredsubmit. Detailed result,primary sources,CPU/sm90a
evidence and new public-code delta225paths/13repos are in
muse_attention_decode/BIT_EXP_RESEARCH_2026-10-31.md.

Oct31 BF16 V39/V40/V41: all 9 tests/30 timings pass per check, with the
unchanged parent timed path. Independent C12 layout-transfer diagnostics
reject all fixed policies; the final read-flush best worst-group proxy is
1.000443x. No canonical/scored changes. See
muse_bf16_proj/DEVICE_EVENT_METHOD_2026-10-31.md. The external workspace
volume is unavailable; the current pushed work and recovered V41 evidence
are maintained in /Users/soorya/muse_work/continuation_20261030.

Oct31 decoder fresh-source/compiler cycle: no H200 check or scored change.
New licensed trxphxle measurement data and18verified route tuples narrow
arithmetic/reread-only leads. A delayed-activation-wait source transform
compiles in a reduced sm90a build but has no practical>=3%runtime case yet.
See muse_decode_layer/PUBLIC_DELTA_AND_OPERAND_SCREEN_2026-10-31.md. The
peer precision note distinguishes policy-unconfirmed higher-FP32 fusion
from the small-T FP32-to-BF16 router reduction already corrected here.
Passing tests are not an organizer ruling on every fused intermediate.

Oct31 late prefill work: K64 scale-sharing CPU encoding proof passes
65536words/524288weight-scale assignments; no GPU implementation/check yet.
The next obstacle is matched normalized-X gathering and GEMM1 output
staging. No scored improvement or canonical change. See
muse_prefill_layer/SCALE_SHARE_RESEARCH_2026-10-31.md.
