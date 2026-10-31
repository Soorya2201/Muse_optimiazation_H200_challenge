# Public kernel source research across all five tracks

The owner's scope includes useful public implementations from teams ranked both above and below
our entry. Review happens one track at a time. Rankings are discovery context; actual per-shape
latencies, numerical correctness, compliance, and source provenance determine adoption.

## Coverage status (2026-10-30)

Oct31 continuation: indexed public-code search now supplements all five
account screens. It returned234 BF16,203 MoE,207 attention,87 decoder and118
prefill unique paths (different capture times; changing index totals are
recorded in the individual JSONs). This is not full-source review or access
to private/unpublished winners. Latest detailed reviews:

- [Decoder](muse_decode_layer/GLOBAL_SOURCE_FOLLOWUP_2026-10-31.md): six
  additional Rohan sources plus11files/five other repositories screened;
  no stronger replacement established. FP32 small-T router preserved.
- [Prefill](muse_prefill_layer/GLOBAL_SOURCE_FOLLOWUP_2026-10-31.md): seven
  Rohan sources plus four other prefill sources inspected. Gemini's active
  FP8 dense branches violate precision; constant-ratio external timings are
  not usable evidence. Existing dormant PIPE3 tested in check4c855e373fee4973;
  all changed combinations rejected. Public Lt signatures are a separate
  component lead; see the dated probe report for terminal status.
- [Chord](research/public/chord_6199d44/PROVENANCE.json): pinned H200
  scheduling/library references with licenses and exact file hashes.
  Centered INT4 contract is incompatible with directly passing NVFP4 codes;
  a complete compatible pipeline port remains unimplemented/unbenchmarked.
- [Scheduling paper addition](research/SCHEDULING_READINGS_2026-10-31.md):
  Twill's joint pipeline/warp-assignment model, applicability and limits.
  No solver run or performance claim; actual H200 operation costs are needed.
- [MoE](muse_moe_w4a16/GLOBAL_SOURCE_FOLLOWUP_2026-10-30.md),
  [attention](muse_attention_decode/PK2_HYBRID_2026-10-30.md), and
  [BF16](muse_bf16_proj/TMA_PUBLIC_FOLLOWUP_2026-10-30.md) preserve the other
  recent source audits and measured negative controls.

These additions supersede older "unreviewed" labels below within the
documented scope, without changing the historical capture-time ranks.

All five public-owner metadata screens were refreshed this UTC day. Dated
`RANKED_GITHUB_REPO_SCREEN_2026-10-30.json` and
`PUBLIC_REPO_FOLLOWUP_2026-10-30.md` in each track supersede the older metadata
counts below: BF161433accounts(ten unresolved),MoE1151(five),attention1104(five),
decoder1011(five),prefill662(four). Metadata coverage is not full-source review.
Selected updated forks and leader portfolios were examined; alygg77's useful
files have been checked/adopted, and its attention/prefill source matches our
original vjeux base. No additional published rank-one implementation was found
in these inspections. Historical ranks in source notes are capture-time facts;
use RANK_STATUS.md for the latest verified positions.

| Track | Research artifacts | Implementation/testing status |
|---|---|---|
| BF16 projections | `muse_bf16_proj/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN_2026-10-30.json`, `PUBLIC_CONTROLS_2026-10-30.md` | Fresh metadata screen1433accounts:191 above/1241 below/self,ten unresolved. ExactChen checks1.519434x(sfo3c)/1.549794x(sfo3). Ali checks1.544358x/1.547060x vs same-day sfo3 parent1.510571x. Exactv34 scoreda3d4758c1e174aba **1.535843x**,rank#133/1435. All30-cell profiles/logs archived;five checks/one submit. Older source/history audits preserved. |
| MoE | `muse_moe_w4a16/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json`, `KPERM3_COMPARISON_2026-10-30.md` | Public metadata screen covers1112ranked accounts. Oct30 exact KPERM3 checks2.404125x/2.376541x vs same-day sfo3 parent2.332013x; changed32/128/256 loads improve repeatedly. Promoted exactv24 scoredce241acf7ab84c04 **2.372309x**,rank#79/1150 (+2.076% over old scored2.324062x). Cycle3/8checks,1/2submits. Earlier dequant,REG112,prefetch negatives retained. |
| Attention | `muse_attention_decode/PUBLIC_RESEARCH.md`, `RESEARCH_LIBRARY.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Full metadata screen of1064 ranked accounts. Scored exact source **2.603505x**. Oct29 six-check cycle complete: L2 lookahead regressed; phase probe diagnostic only; startup offset and parallel flags did not win; 0/50 workspace plans qualified. All profiles/logs recovered and archived. B1 merge/protocol family closed by CODEX_NEXT_STEPS.md. |
| Whole decoder | `muse_decode_layer/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Metadata screen covers980 ranked accounts. Oct30 precision audit withdrew Madhavan scored d857144e59784ad0: BF16 small-T router input violates baseline FP32 requirement. Exact alygg77 FP32-router replacement passed2.189801x/2.221428x; same-day sfo3 historical control2.137329x. Compliant replacement eff0b59dc26c4c74 scored2.211495x,decoder#50/1007,overall#50/1649. Public leader follow-up documented; no winning source identified. |
| Prefill | `muse_prefill_layer/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Full metadata screen of644 ranked accounts:538 above,105 below,ourselves;four404s. Exact vjeux scoredf62a284779b4447d **1.784958x**,rank#102/644. New lee1018-op port passes34848offline cases and5tests/8timings at1.801190x;not promoted. Identical-route paired gates show only0.06–1.58% expert gains,below2%;v19 retains parent everywhere. Cycle complete at3checks/1submit. |

This is a coverage record, not a claim that every repository or private submission was examined.
The account screen paginates publicly owned repositories and matches name/description/topic
signals. Matches still require source review; private, renamed, or unlabelled source may be missed.
Global code search supplements the profile screen, including accounts absent from the board.

## Source families found so far

- https://github.com/vjeux/okc-2026 — Apache-2.0, complete single-file native kernels for all five tracks.
- https://github.com/Madhavan113/kerneloptimizer — Apache-2.0 derivative with per-track extensions and profiles.
- https://github.com/lee101/openkernels — Apache-2.0, many intermediate versions and diagnostic evidence. Its weaker whole scores do not rule out useful individual mechanisms.
- https://github.com/chenwainuo/okc — Apache-2.0, detailed BF16 source/history and broader experimental work.
- https://github.com/NathenFernandes/muse-bf16-proj — MIT, exact BF16 v18 already reproduced by us.
- https://github.com/operatorstack/open-kernel-challenge-solutions — MoE6d9b776 source sections/1.832036x receipt and attention producer/consumer/plan sections/1.822823x receipt reviewed. Retained source concepts already overlap our stronger parents; no additional measured winner yet.
- https://github.com/lilaclilac09/open-kernel-challenge — MoEd9bdcc9 actual source is plain-PyTorch starter;sketch notes are a placeholder. No optimized MoE source found in this inspection.
- https://github.com/PostTrainLLM/posttrainllm — experiments/openkernel sources and provenance; published MoE score below ours, useful mechanism inventory.
- https://github.com/Ranoobaba/okc-campaign — indexed challenge material; source relevance still needs verification.
- https://github.com/tommyyliu/okc — MoEf134725 GEMM/packing/dispatch and attention split/merge/epilogue/plan sections reviewed. Lossless packing,register prefetch,head packing and hierarchical merges overlap our stronger parents. Later board snapshots are not asserted identical to the inspected source. No code copied.
- https://github.com/xharshvardhan/open-kernel-challenge — MoE source4958d314 reviewed Oct30: published1.3749x, fused local alignment and hot/cold Marlin tile buckets,retaining both library GEMMs. Existing persistent parent already fuses routing/phases/finalization;no additional winning component established. Other tracks remain separate reviews.
- https://github.com/theluckystrike/muse-spark-open-kernels — code search hit, current tree only exposed README; no verified submission yet.
- https://github.com/ruwwww/open-kernel-challenge — current README reports attention 2.6032x, MoE 2.3370x, decoder 2.0751x, and prefill 1.7950x with job IDs. Strong reproduction candidates for the remaining tracks after source/precision review; README mechanism descriptions alone are not proof.
- https://github.com/Aravind-11/muse-moe-w4a16 — historical 1.629x early kernels; README explicitly says current entry is not published.
- https://github.com/infraax/okc-kernel-challenge — lower-ranked archived campaign with measured failures; producer-ring and PDL counterevidence is useful, but no stronger full MoE source than the reproduced parent.
- https://github.com/titoatwork/qgemm-mx — account below us; FP4/Hopper measurement and packing study. README explicitly says performant Hopper WGMMA/TMA kernel is not implemented; use encoding/roofline evidence without claiming speedups.
- https://github.com/zhebrak/muse-spark-kernels — below us on prefill with no published prefill winner, but Apache-2.0 sources/receipts for four other tracks. Describes persistent stream-K expert readiness counters and cluster residual boundaries; possible future component controls, not verified improvements over our promoted sources.
- https://github.com/MadanChaollaPark/open-kernel-experiments — near our old prefill rank; stopped campaign with exact scored sources and627 candidate snapshots. Preserves correct-but-slower PRMT dequant and unfinished Hopper mixed-input proof without a speed claim; useful negative evidence.

- https://github.com/alygg77/okc-muse-spark-kernels — Apache-2.0, commit 6cae49a (published Oct 29, #40 overall). Built on vjeux f35a41e. Exact files pinned as `muse_decode_layer/submission_v17_alygg77_6cae49a.py` (2.2058x, FP32 router at T<=8), `muse_moe_w4a16/submission_v24_alygg77_6cae49a.py` (2.4015x, KPERM3 scale-sharing re-layout), `muse_bf16_proj/submission_v34_alygg77_6cae49a.py` (1.5502x). Measured-facts, negative-results and timing kit in `research/public/alygg77_6cae49a/`. Decoder audited and twice checked at2.189801x/2.221428x on sfo3; replacement scored2.211495x (#50 decoder/overall). MoE checked twice and scored2.372309x (#79); BF16 checked twice and scored1.535843x (#133); see CODEX_NEXT_STEPS.md P1.
- https://github.com/Madhavan113/kerneloptimizer — newer commits through 721c627 (Oct 29 20:47 UTC): attention TMA bulk-reduction merge (-0.3..-0.5 us), MoE MID OKC_WSH0 and WGS32 builds. Design notes in `research/public/madhavan_721c627/`.

## Adoption procedure

Oct30 attention broader code search:207unique public matches/12repos. Seven
Rohan archive candidates inspected/hashchecked; external arithmetic/constant
Gemini2.6537 timings excluded. Legacy EXPPOLY disables PK2 joint probability
packing, so its source-comment negative is a confounded comparison. New
v21off-board probe2d537621593a41d7(sfo3) preserved PK2 for25/50%FP32FMA but
its copied polynomial underflowed toNaN. Repairedv22/ec545aab86274b3b passed
all independent cases but both hybrids lose17-61%;all8changedcompositions
reject. Newv23/0b17352502ca4fca(sfo3c) isolates bias-basedrange reduction
against byte-identicalfloor50control. CPU66780bitidenticalcases andstandalone
sm90aSASSverifyconversionremoval. Completed6nearby/16timing cases: bias50
beatsfloor50by3.28-6.08%speedup,but still27-47%longerthanincumbent. All
changedcompositionsreject;testedfamilyclosed. Timed/scoredparentunchanged. No reopening B1 merge work. Sources,
phase-zero/resource-model assumptions and independent validation plan in
`muse_attention_decode/PK2_HYBRID_2026-10-30.md`. Old terminal receipts also
recovered via read-only status when the external volume could not save profiles.

Oct30 MoE broader code search:203unique public matches/12repositories,
including seven newly reviewed rohanarun candidate files. Its multiply-based
decoder is bit-exact in35136CPUcases but has the same operation classes as
the already measured8-op family; no new large-gain evidence. External Gemini
constant-ratio timings are excluded. Ranoobaba's previously unresolved
receipts are weaker;vLLM's new native NVFP4 GEMV is SM100. Humming's BF16
conversion components checked separately; no supported faster replacement
established,complete alternate schedules not benchmarked. Exact manifests,
licenses,hashes,limits and sources in
`muse_moe_w4a16/GLOBAL_SOURCE_FOLLOWUP_2026-10-30.md`. No GPU run/new score;
canonicalKPERM3 unchanged and routine MoE6/8checks,1/2submits preserved.

Oct30 broader BF16 code-search follow-up: 234 public file matches archived in
`muse_bf16_proj/GLOBAL_CODE_SEARCH_2026-10-30.json`. Newly reviewed
`rohanarun/Auto-research-agent` at018b7a1 (MIT) contains a standalone BF16 TMA
GEMM; isolated source/license pinned under `research/public/rohanarun_018b7a1/`.
Its external constant-ratio Gemini timing rows are not accepted as performance
evidence. Prepare-only v38 paired probe a8d2c8d0433a4829 on sfo3c passed9/30;
timed dispatch remains parent. All15changed compositions rejected: pooler256
44-54%longer; other cells near parity. Service log tail lost one of32rows;
31archived,all seven complete groups modeled. No score promotion.
Ozgur's Apache-2.0 v027 is slower than current
controls; muse-sputter is an explicitly slow exercise. Audit/harness material
is excluded as an optimization source. Scope/provenance/model and pending
result: `muse_bf16_proj/TMA_PUBLIC_FOLLOWUP_2026-10-30.md`.

Pin the commit, source hash, and license; inspect the timed path and precision; establish an
unchanged-parent OKC result; compare all relevant shapes; port only justified mechanisms;
check the resulting source on current inputs; submit only after the project's improvement gate.
Preserve losing runs and partial/invalid evidence with their actual status. All judge invocations
go into root RESULTS.md. No unmeasured gain or paper's unrelated benchmark becomes a claimed score.
