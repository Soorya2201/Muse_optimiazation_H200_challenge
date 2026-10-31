# Decoder public-source delta and operand scheduling screen

The current scored decoder remains 2.211494776x, job `eff0b59dc26c4c74`,
source SHA-256 `8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.
No new H200 job or scored submission was allocated in this cycle. The prior
BF16 turn made progress: three experiments closed the sign-layout transfer
and repaired its diagnostic protocol. This turn moves sequentially to decoder.

## What changed in the public-source inventory

Fresh public GitHub code search returns **94 paths in 10 repositories**,
versus 87/9 previously. Seven new entries are at Kushagrabainsla's `0778ecd`.
This is an indexed search, not exhaustive branch or private-source coverage.
The fresh owner screen covers **1106 ranked accounts**, including 144 above
our then-rank145, 961 below, and ourselves. Five accounts remain unresolved:
Abarbir12, accountforgoofystuff, Pinoeer-kingxi, cyber199306 and gurrapu21.
Only Abarbir12 is above us. Metadata screening uses public repository names,
descriptions and topics; it is not a full source audit of every account.

Nineteen repository matches are new relative to the previous decoder screen.
Eight SGLang forks have zero commits ahead and no changed default-branch
files against sgl-project/main. This says nothing about their other branches.
The adopted alygg77 head is still the pinned source. New NVFP4 utilities
belong to the next MoE review; the pocket-muse model and lysithea utilities
do not supply a decoder replacement. Records:

- `GLOBAL_CODE_SEARCH_2026-10-31.json`
- `RANKED_GITHUB_REPO_SCREEN_2026-10-31.json`
- `NEW_REPO_FOLLOWUP_2026-10-31.json`

### Deferred lower-ranked implementations

[Kushagrabainsla/muse-spark-open](https://github.com/Kushagrabainsla/muse-spark-open/tree/0778ecd8293e375ffe33faa466db0231929aaac3/muse_decode_layer)
is #731 at capture. All 32 selected files have verified Git-blob/SHA-256
hashes; submission ASTs were parsed without executing them. Read the
experiment history, default prepare/dispatch and generated integration.
L9 adds E14 projections to L8; the author reports 1.410x check versus
L8's 1.424701x board. Eight complete external tables (96 timing cells)
were parsed. The native projection/expert components and selected fused
attention are older than current-parent choices; full verify still uses FA3.
The default does retain FP32 small-T router inputs. That observation is not
a complete conformance audit. No stronger additive component established.

[ScrashOff/okc-muse-kernels](https://github.com/ScrashOff/okc-muse-kernels/tree/d870912e7c5a030c52050bb2445418224c28d9de)
is #848 at capture. Fifteen selected files have verified hashes. Seven
external tables (84 timing cells, including the repeated v6/default receipt)
were parsed. The shipped decoder exactly equals v6: 1.280499x check,
README1.300x scored. It combines Triton GEMV/split-KV attention and Marlin
expert tuning. Its default requests no FP32 router activation and routes
from BF16 ffn_input at T<=8. Do not adopt that lower-precision path.

Neither root license was established. Sources remain temporary research
files; none was executed, ported or mirrored into tracked submissions.
`DEFERRED_SOURCE_SCREEN_2026-10-31.json` preserves paths, hashes, URLs,
AST metadata, numerical tables and the bounded review scope.

### Newly found licensed same-family measurements

[trxphxle/okc-2026-muse-kernels](https://github.com/trxphxle/okc-2026-muse-kernels/tree/72db638fa2718bc0ea5645fdd5c52a5511594579)
is #211 at capture. Apache-2.0. Fifteen selected files (1,090,917 bytes)
were fetched and Git-blob/SHA verified. Its decoder's SHA matches its
published BEST manifest (`053dbacc...419c086`), with reported ranked
2.17561776x/check2.19161708x. These are publisher-linked external results,
not our paired comparison. The file is slower in aggregate than ours;
that alone does not exclude useful components.

Nine unmodified documents/data files are pinned with license, upstream
NOTICE, authorship, URLs and hashes under
`research/public/trxphxle_72db638/`. The whole decoder source remains in
temporary research storage and was not imported or executed. Read README,
BEST, full PRECISION note, counter summary, selected scheduling/research
sections, complete attribution/ceiling JSONs and all 18 route tuples.
This is not a complete per-line or whole-repository audit.

The external expert attribution explicitly identifies our same canonical
standalone MoE SHA `b3307089...4281`. Its counters show no local traffic;
reported opcode shares cannot be treated as elapsed-time fractions or
added across different profiler passes. Deleting much of unpack and half
the scale multiplies yields expert-board proxies 1.00554/1.00930x with
controls0.99721/0.99795x. These scoped external deletion experiments weaken
an arithmetic-only gain thesis; they are not universal upper bounds on
new dependency schedules. Wider tiles, generic VMM compression and naive
concurrency also have measured negatives on that author's controls.

## Recomputed route and byte-pool screen

`tools/model_decoder_public_reread.py` independently verifies all 18 raw
histograms: touched experts K, total routes R, sum ceil(n_e/64),
sum ceil(n_e/128), exact planned/unique/duplicated bytes and the reported
single-pass metric flags. Each whole expert stream is 28,311,552 bytes.
D32 has zero repeated 64-token-job weight streams; V16 repeats0–9.09%,
V32 repeats0–17.65%. A broad claim that all V32 has negligible repeats is
incorrect; the third seed has a meaningful byte pool.

For planning only, apply each external fixture's byte fraction to our
scored job's profiled expert duration, recompute 46:16 layer sums, and
hold the other loads unchanged. The jobs/inputs differ, durations contain
PDL waits, and traffic is not proportional to critical-path latency.
These are conditional proxies, not achieved gains or rigorous bounds.

| External weight seed | Remove all repeated streams | Use 128-token jobs |
|---|---:|---:|
| 731283 | 1.000968x | 1.000968x |
| 731284 | 1.014151x | 1.012974x |
| 731285 | 1.024388x | 1.018300x |

No >=3% decoder case is established by removing repeated reads alone in
this screen. A design that also improves memory service or scheduling is
a separate hypothesis. Results and every histogram are in
`PUBLIC_REREAD_SCOPE_2026-10-31.json`. Do not turn this into a native
128-token-job parameter sweep or general impossibility claim.

## New operand-wait transformation: compiled, not GPU-qualified

Current `wgs_seg` waits for the weight slot, waits for X, then decodes the
first register operands. The independent proposed edit waits for weights,
loads/decodes the first GS2 A group, then waits for X immediately before
the unchanged operand fence and first WGMMA. It preserves byte traffic,
geometry, precisions, accumulation order, commits, slot releases and output
logic. It does not borrow any implementation from unlicensed sources.

The sole source delta is implemented in
`tools/decoder_wgs_variants.py:delayed_activation_wait`; the compiler
probe accepts `delayed_x_wait`. CPU source equality proves that removing
the old/new wait placements leaves byte-identical remaining native text.
[NVIDIA PTX WGMMA ordering](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
requires the register fence before asynchronous operand use; it remains.
The activation-completion wait still guards the first shared X read.

NVCC13/sm90a, reduced BIG=true/NTT8/internal-helper build: 96 registers,
zero stack, zero spill stores/loads. Complete PTX verifies BF16 dequant
before the X wait, followed by fence/MMA. SASS is clipped at5000 lines;
no full binary count or complete-switch resource inference is made.
Full request/response and exact source hashes are archived in
`WGS_CODEGEN_delayed_x_wait_single_true_2026-10-31_internal_helpers_fixed_ntt8_explicit_gencode.json.gz`
and `DELAYED_X_WAIT_PREFLIGHT_2026-10-31.json`.

**Do not spend a check on this evidence alone.** The main-path phase-zero
screen exceeds3%, but no current measurement establishes enough exposed
activation wait that this first-group move can save the roughly15% expert
time needed on V16/V32. Source order and zero spills are feasibility only.
A future diagnostic must measure that exposure with matching controls;
new source evidence should justify it before another instrumentation run.
This candidate is deferred, not a measured loser or a leaderboard gain.
The old register/GS4 and uniform-loop experiments remain closed.

## Precision evidence: distinguish two issues

The ScrashOff small-T FP32-to-BF16 router loss is a concrete precision
reduction and remains excluded. Separately, trxphxle's PRECISION review
finds no attributable organizer decision on retaining higher FP32 fused
intermediates across SGLang's BF16 materialization boundaries. Its author
explicitly retracts earlier categorical disqualification claims. This is
peer research, not an organizer ruling. Do not label higher retention
approved, proven cheating, or evidence that the router reduction is legal.
Our passing checks and router audit do not settle that broader policy
question. The pinned note preserves the evidence and its limitations.
No new precision changes or withdrawals were made from this unresolved
interpretation. No messages/registration were sent to third parties.

## Decision and next action

No measured improvement or rank1. Canonical files stay unchanged. Decoder
Oct31 routine checks remain1/8, submits0; all other budgets unchanged.
Local source/AST/compiler/model verification completes this bounded cycle;
no live GPU job is being awaited. Move to prefill in the sequential cycle,
read its AGENTS/TASK and current closed-family reports, and screen the new
licensed source's relevant prefill/dense components before allocating a run.
The broad rank1 goal remains active and incomplete.

Verification: all five canonical SHA-256 hashes remain unchanged; all nine
pinned file/Git-blob hashes reverify. Python source syntax and the model's
18 histogram/byte assertions pass. The original CSV retains CRLF bytes to
preserve upstream hashes; Git's default whitespace check reports those
line endings. Authored edits pass the ordinary check; the complete staged
change passes with CR-at-EOL recognition. No GPU validation was run for
the new compiler-only candidate.
