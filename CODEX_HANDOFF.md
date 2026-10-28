# Codex Handoff: Muse Spark Open Kernel Challenge

You are taking over a GPU-kernel optimization effort. Work through the tracks **one at a time, in the order below**. Do not run parallel agents or sub-agents. Finish a track (or hit its stop condition), record results, then move to the next.

Owner: Soorya (GitHub `Soorya2201`). Project root: `/Volumes/Soorya_1_TB/Projects/Muse_optimiazation_H200_challenge` (an external drive; see "Known hazards").

---

## 1. What we are trying to do

The Muse Spark Open Kernel Challenge (https://openkernelchallenge.dev, rules at `/rules`) asks for faster GPU kernels for Meta's **Muse Spark Open** model, timed against **SGLang's** own kernels on one **NVIDIA H200**. Score = geometric-mean speedup over SGLang across the scored loads.

Two tracks:
- **Kernel track (now, closes Nov 4 2026, 11:59 pm PT).** Five problems, one Python file each. No prize money, but the **top 50 on the overall kernel board get priority** in the serving track, and the kernels we build get reused there.
- **Serving track (opens Nov 5 2026, noon PT).** Patch SGLang (commit `a55ab7d5`) to serve the model faster on 8×H200. **All $100K in prizes are here.** 1 submit + 3 checks/day.

Our strategy: get solid, *legitimate* speedups on every kernel problem, climb into the top 50 overall, and find **one genuinely novel optimization** (not what the other ~1,000 entrants are doing) that carries over to the serving patch.

## 2. Environment

- The laptop is a Mac (arm64), **no NVIDIA GPU**. Nothing can be run or timed locally. All testing goes through the `okc` CLI to a remote H200.
- CLI: `~/.okc/bin/okc` (v0.7.1). Run `export PATH="$HOME/.okc/bin:$PATH"` first. Already signed in as `Soorya2201` (`okc whoami`).
- Judge image: `lmsysorg/sglang:dev-muse-spark-open`. Available there: PyTorch 2.14 (cu130), Triton 3.8, FlashInfer, FA3, DeepGEMM, TileLang, CuTe DSL, CUTLASS, sglang. Native CUDA is allowed via `torch.utils.cpp_extension.load_inline` (nvcc). No network.
- Local Python is only useful for syntax checks (`python3 -m py_compile file.py`). You can't import torch with CUDA here.

### okc commands

```bash
okc problems                  # today's remaining runs/checks per problem (reset 00:00 UTC)
okc check  <file> --json      # full tests + timing, NOT on the leaderboard (20/day/problem)
okc submit <file> --json      # scored, goes on the board (10/day/problem)
okc status <job> --json       # result again; .result.speedup, .result.score_us, .result.log_tail
okc jobs                      # recent runs
okc leaderboard <problem>     # board; `okc leaderboard` = overall
okc withdraw <job>            # take a run off the board
```

- The problem is read from the file's first line, e.g. `# okc: problem=muse_moe_w4a16`. Never remove it.
- A run has a **10-minute limit including compile/autotune** in `prepare()`.
- Passing runs write `profiles/submission.<job>.txt` (time per kernel) and `.log` beside the file. **Read these after every passing run.**
- `print()` in `prepare()` comes back in `log_tail` (last 16K chars). The existing versions use this for self-tests and prepare-time benchmarks. Keep doing that.
- Resubmitting byte-identical code returns the cached earlier result.

## 3. Rules (violations get disqualified on human review)

- `kernel_fn` must be **CUDA-graph capturable**: no `.item()`, `.cpu()`, `torch.cuda.synchronize()`, printing tensors, and no allocation or branching on input **values** (shapes and `config` are fine).
- Every call gets fresh inputs. **Never cache anything derived from inputs.** No work outside the timed call except weight repacking in `prepare()` (untimed, ≤300 s).
- Weights may be repacked or **losslessly** compressed, never changed in precision: BF16 stays BF16, NVFP4 stays NVFP4 (dequant in-kernel is fine). The router's FP32 weights may run as a BF16 hi+lo pair with FP32 accumulation.
- Activations stay BF16 where SGLang keeps BF16. The FP8 KV cache stays FP8. Attention queries may be rounded to FP8 E4M3, nothing coarser.
- Attention must attend to **every** key (the whole 2,048 window on sliding layers, the whole context on full layers). No top-k, sparse or block skipping.
- Don't import the reference, touch the harness or timing, start processes (other than load_inline's nvcc), or use the network.
- If anything you did is reward-hacky, `okc withdraw` it. **Only real speedups.**

## 4. How timing works (shapes your decisions)

- Each scored shape is captured as a CUDA graph and replayed, alternating with SGLang's kernel on the same inputs.
- **L2 is flushed before every replay** (cold L2), so weights always come from HBM.
- About 4 µs of graph overhead is subtracted from both sides.
- Per shape you get the median of paired ratios. Each load sums its shapes × weight, and the problem score is the geometric mean over loads, so **every load counts equally**. A small-m load that's 2× slower hurts as much as a big one that's 2× faster helps.
- Decode shapes (m = 1–32 tokens) are **memory-bandwidth and launch-latency bound**. Measured on this H200: a plain streaming read reaches only **~3.2 TB/s at 8 MB, ~4.3 TB/s at 64 MB**, so the TASK.md floors (4.8 TB/s) are optimistic.
- Prefill is compute-bound. The H200 has **no FP4 tensor cores**, so NVFP4 is dequantized to BF16 for WGMMA/MMA.
- Tune with CUDA-graph timing, not eager timing (eager misleads at these sizes). Several versions already autotune per shape inside `prepare()` using graph replays with an L2 flush. Reuse that pattern.

## 5. Current state (as of Oct 28 2026)

### Continuation update (supersedes the older table below)

#### Oct 28 MoE continuation

The MoE cycle is closed at 8/8 reserved checks and 1/2 submits. v12 increased activation
prefetch depth from AP=2 to AP=3 and retained NT=1 specifically at bs32. Its check scored
1.6392x (`bcf981ec568f4d5c`, 8204.4 us) and its submit established the new live best at
**1.6329x** (`9a02dcaaad274ffb`, 8236.0 us, CI 1.6084-1.6574), improving the old v9
1.6092x submit. `muse_moe_w4a16/submission.py` is restored byte-for-byte to archived
`submission_v12_ap3_nt1_bs32.py`.

The remaining checks rejected global AP=4, AP3/AP4 and AP3/AP2 static hybrids, WP=7, and
a bs256-only 1.5-wave grid. The extra grid wave was particularly harmful because every CTA
repeats routing setup and streams overlapping weights. Every result is in `RESULTS.md` and
the track `IMPROVISATIONS.md`. Do not run more MoE checks on Oct 28 under the reserved-budget
policy.

#### Oct 28 attention continuation

The attention cycle is closed at 8/8 reserved checks and 1/2 submits. v12 compiled four- and
five-stage cp.async specializations and selected NST5 only for B16/B32 full decode. Its check
scored 1.3012x (`50948bd6edc344dc`), and its submit established the live best at **1.2954x**
(`3135d826e3994f11`, 2907.4 us, CI 1.2760-1.3188), up from 1.2491x.

Follow-up isolation showed the main win was C2 cluster geometry rather than the fifth pipeline
stage. v13 used NST4 with C2/NC8 and C2/NC4 for B16/B32 full decode and checked at **1.3116x**
(`ccaa5314524e4001`, 2871.6 us, CI 1.2919-1.3312); its 123.86/232.86 us paired times beat v12,
but the aggregate gain over the scored result was only 1.25% with overlapping intervals, so it
was not submitted. NST3 v14 regressed to 1.3061x. Keep v13 as the best local candidate and v12
as the live scored result. Batched sliding verify one-read experiments v7-v10 all lost to FA3.

Ranks refreshed after the attention submit: overall **#655/1417**; attention **#761/936** at
1.2954x; MoE **#529/978** at 1.6329x; BF16 **#1085/1230** at 1.1158x; decode **#697/854** at
1.2326x; prefill **#455/563** at 1.1184x. Ranks are dynamic.

#### Oct 28 BF16 continuation

The BF16 cycle is closed at 8/8 reserved checks and 0/2 submits. The live and local winner
remains v3 at **1.1158x** (`167d6e5c85f74682`); `submission.py` was never changed. v10's
cold-L2 stage/ring sweep confirmed the existing qkv and pooler pipeline choices
(`cddcfa69cbbf4c5a`, 1.1080x). v11 replaced same-address shared atomics with per-warp
partials and an explicit epilogue sum; it regressed to 1.0986x (`fb91e9bfbf7e43d4`).

v12 instead assigned each qkv consumer warp a disjoint 16-row slice and made it traverse all
four K slices, eliminating the intra-CTA atomic reduction without adding MMA work. It passed
9/9 at 1.1179x (`f73b3d06dcef4761`, 4321.5 us, CI 1.1011-1.1346); qkv m1/m8 profiles were
9.9/11.7 us. This recovered v11 and helped m8, but did not strictly beat the earlier
1.1189-1.122 checks, so it was not promoted or submitted. Next BF16 check should combine the
original qkv mapping at m=1 with row partition only at m=8. Do not run more BF16 checks on
Oct 28 under the reserved-budget policy.

Latest Oct 28 rank refresh after the BF16 research checkpoint: overall **#662/1426**; MoE
**#534/987** at 1.6329x; attention **#768/942** at 1.2954x; BF16 **#1090/1239** at
1.1158x; decode **#702/862** at 1.2326x; prefill **#463/569** at 1.1184x.
Ranks are dynamic.

Post-budget preparation (no additional OKC run): v13 is the planned qkv hybrid that keeps the
original mapping at m=1 and enables row ownership only at m=8. More importantly, public research
found a former-leading Muse BF16 design at 1.343x check / 1.331x submit. Its successful exact
12-bit kernel bypasses the TMA ring: direct non-allocating global vector loads, a pack-time
`prmt`/`lop3` layout, software-prefetched compressed groups, and a dedicated concurrent exception
warp. The guide reports 13-20% qkv/pooler/o gains. No source from current #1 `colinlikescode` was
publicly discoverable, so do not describe this as their private method. Primary pages are linked
in `muse_bf16_proj/IMPROVISATIONS.md`.

`submission_v14_direct_compressed.py` implements an original simplified version for qkv/pooler
m=1/8 only. CPU tests prove its 12-bit packed-fragment layout and sparse corrections reconstruct
the BF16 weights exactly; Python syntax and diff checks pass. It has not compiled or run on CUDA.

The exact newer public v18 source was subsequently found at `NathenFernandes/muse-bf16-proj`
and vendored, with MIT license and result evidence, under `muse_bf16_proj/upstream_nathen_v18/`.
Source SHA-256 `111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44`
matches its manifest. Published judge job `66d8b007f6c641ee` passed 9/9 at **1.518570x**
(CI 1.4958-1.5413); this repository historically reached #1, though v18 did not beat
the current 1.6367x leader. It is much stronger than our simplified v14 and adds stream-ordered
per-warp escape patches, WGMMA/TMA/cluster routes, pinned cuBLASLt plans, and better router paths.

At the next BF16 budget reset, first check
`muse_bf16_proj/upstream_nathen_v18/submission.py` byte-for-byte. If it reproduces a strict win,
promote and submit it with attribution. Then check
`muse_bf16_proj/submission_v30_all_small_qkv32.py`. It contains v29 as an exact prefix; v29
contains v28 as an exact prefix; v28
contains v25 as an exact prefix and first combines the full meaningful v9-v18 route history with
three range-validated cross-small C12 plans: 52 graphs over 23 changed
shapes, fixed v18 elsewhere, and a 1%/0.08 us activation gate, then independently validates and
gates v27's WR=2/4 qkv16 geometry against the v25-selected route. V29 then independently tests
v26's WR=2/4 o geometry at m1/m8 against the v28-selected route. V30 finally validates and gates
WR=2/4 qkv32 geometry against v29. V30 now starts from v25 once and compiles those three independent
experimental extensions concurrently, then retains the original sequential qkv16→o→qkv32 GPU
validation and autotune order. Prefix, syntax, and mocked state/order checks pass, but H200 prepare
time is still unmeasured. Each experimental build is now failure-isolated: if one module does not
compile, v30 prints the bounded error, keeps exact v25 plus the other successful modules, and skips
only that module's validation/autotune. Mock tests cover the success path and each of the three
isolated failures. Descend v30→v29→v28 only on a top-level resource/prepare-time failure;
V25 is the no-new-compilation
fallback; V23 and v24 are isolation diagnostics only; v19 is a narrower fallback. The separately prepared
`submission_v26_o_c12_geometry.py` keeps the exact C12 decoder but tests legal WR=2/4 `o` routes
at m1/m8 (512/1024 threads and 256/128 CTAs), with exhaustive one-hot plus random validation and
the same conservative gate. `submission_v27_qkv16_c12_geometry.py` similarly preserves v18's
qkv16 decoder and four-block reduction while shrinking 576 128-thread CTAs to 288 256-thread or
144 512-thread CTAs; all three geometries execute 2,304 warps. It rejects escape overflow, local
spill, correctness failure, or a non-strict timing result. V26/v27 are now isolation candidates
for surprising combined logs, not default checks. Next, check
`submission_v31_pooler32_geometry.py` as an isolated follow-up if the first two checks leave
prepare budget: it preserves 2,048 warps while testing 256→128/64 CTAs and should only be combined
after a measured win. Next, check
`muse_bf16_proj/submission_v20_cluster_dense_m128.py`. It preserves v19 as an exact prefix and
generalizes v18's judged qkv cluster-DSMEM primitive to pooler m128, while using an independent,
no-reduction schedule for unpooler m128. Both new routes must pass random and one-hot GPU tests
and a conservative cold-L2 timing gate before activation. If exact v18 fails because of
environment drift, diagnose that artifact before trying v29 or any cluster derivative. Then check
`muse_bf16_proj/submission_v21_cluster_dense_m128_m256.py`, which adds independently gated
TM256 pooler/unpooler routes to the same cluster/no-reduction design. Its static generator and
shared-memory checks pass, but it has not compiled on H200. Finally, check
`muse_bf16_proj/submission_v22_cluster_dense_autotune.py` only after v21 establishes that the
generated architecture compiles. V22 directly starts from v19 and compares v21 against v18's
3D-TMA/cache-policy bits plus one adjacent stage depth for each of four shapes; all 12 candidates
are correctness-tested and conservatively gated. Use v28, v27, or v26 only to isolate a v29
compile, prepare-time, or timing anomaly.
Then fall back to v14 and v13.
Keep the current
`submission.py` untouched until a qualifying check passes.

V18's exact shape CSV shows the m128/m256 pooler/unpooler line cannot reach #1 alone: 20% faster
on all four models to 1.5832x, and even substituting the optimistic TASK floors models to only
1.6076x. Treat v21/v22 as bridge candidates; after measuring them, return to the small-m
qkv/pooler compression path needed for the remaining gap.

The converse portfolio bound is now explicit. V30+v31 targets alone reach only 1.58896x even at
their optimistic TASK floors. Combining those five targets with v21/v22's four large targets
reaches 1.68212x at all floors, but a uniform 20% win reconstructs only 1.64014x. The prepared
portfolio can therefore beat 1.6367 only with unusually broad, near-20% gains. Do not prepare
more adjacent geometry variants before measuring this set; combine measured winners, then return
to a broader compression/scheduling design if the aggregate remains below the leader.

Quantitative target: beating the 1.6367x leader from v18 requires about 7.22% lower time per load.
Pooler, qkv, and unpooler dominate; 10% faster router alone adds only about 1.0% aggregate, while
8% across all non-router projections models to about 1.6363x. The public vjeux v141 artifact is
1.5014x (clean equivalent 1.4905x), so it is research material rather than the first whole-file
candidate. A broader authenticated GitHub search also found lee101's Apache v22/diagnostic line:
its live entry is 1.4627x, but its later RS-WGMMA clock profiles prove that global split-K
reduction—not CTA start spread or a single decode phase—creates an 8-9.5 us m128 tail. If v23
does not close the dense-shape gap, design the next kernel around cluster DSMEM/st.async or a
no-split schedule; do not retry deeper decode-ahead or another global workspace/counter tail.
`rohanarun` is live at 1.5503x, but its current public repository publishes no matching kernel or
receipt, so it is score evidence only. No current-leader source was found publicly.

An Oct 28 repeat public search found no newly published source from the current top teams. The
guide's visible “1.67x” refers only to its router m128/256 shape, not an overall result; its shipped
artifact remains 1.343x check / 1.331x scored. The linked report nevertheless supplies the next
architecture if v22 fails: 128-wide K stages to halve ~190 ns handshakes, a decode warpgroup that
writes BF16 tiles into shared memory, shared/shared WGMMA instead of the measured ~50%-peak
register-A path, and cluster/DSMEM reduction instead of the observed 4-6 us global split-K tail.
Treat this as public design evidence, not a reconstruction of private leader code.

The same Oct 28 closed-budget window was used for a deferred-source audit. Python syntax, exact
v18's manifest hash, candidate prefix chains, v30/v31 dispatch replacement, and every v21/v22
generated CUDA case passed. Exact shared-memory accounting puts the largest v22 plan at 221,344 B,
under Hopper's documented 227 KiB per-block maximum; an early manual estimate had confused the
group-count field with `NR`, so no stage depth was changed. V20-v22 now carry a formula-based
source-generation guard against oversized future configurations. This consumed no OKC run and
does not change the reset measurement order.

A subsequent Oct 28 public search found no source for the newly risen #3-#5 BF16 teams. It did
find an independent attributed exact-v18 reproduction at 1.53155x check / 1.52107x submit (job
`0babd4cb0aca4510`), strengthening the first-reset choice. Two reported follow-ups are deliberately
deferred: PDL helped a lower-scoring global producer/reducer design, whereas static decompression
found no PDL or global semaphore/atomic reducer in v18's active native sources; and a row-major
`W @ X.T` o-M8 path models to roughly 5.99 us even at its reported 13% gain, versus v18's measured
5.496 us. Do not spend a pre-v30 check on either technique without stronger per-shape evidence.
Even granting perfect transfer of the upper claims across all stated shapes, PDL models to
1.539329x, row-major o to 1.530820x, and their union to only 1.551944x. They do not change the
rank-one architecture priority.

#### Oct 28 prefill continuation

The prefill cycle has used 8/8 reserved checks and 0/2 submits today. The local promoted
`submission.py` is restored byte-for-byte to v8; the live scored best remains v5 at
1.1184x (`b6fbcde2c4aa4418`). Runs v9-v11 tested W2 decode ordering, a two-wave persistent
grid, and M-group tile ordering; all regressed and were removed. Runs v12-v14 integrated
SGLang's masked BF16 DeepGEMM after losslessly decoding the touched NVFP4 experts. Two
integration checks safely fell back; the completed v14 path was correct but took 2954 us
vs G32's 2192 us at T=3012 and 11603 us vs G64W8's 7327 us at T=16384, so it was rejected
and removed. v15/v16 then tested branch-in-kernel and statically split hot/cold expert
materialization. The split K4 candidate won synthetic timing slightly but regressed to
1.1281x end to end (`bed381d5de1d42d1`), so it was archived and v8 restored. See
`muse_prefill_layer/IMPROVISATIONS.md` and `RESULTS.md` for every job. Do not run more
prefill checks on Oct 28 under the reserved-budget policy.

Ranks refreshed Oct 28 after the MoE cycle: overall **#660/1412**; MoE **#527/974** at
1.6329x (leader 2.6853x); attention **#795/932** at 1.2491x (leader 3.0316x); BF16
**#1082/1227** at 1.1158x (leader 1.6367x); decode **#694/853** at 1.2326x (leader
2.4029x); prefill **#453/560** at 1.1184x (leader 2.0252x). Ranks are dynamic.

The prescribed exact 12-bit BF16 implementation cycle is complete. Five checks were used and
all passed 9/9 tests, but none beat the v3 scored winner, so `muse_bf16_proj/submission.py`
correctly remains unchanged at 1.1158x (`167d6e5c85f74682`). Runs are logged in `RESULTS.md`
and detailed in that track's `IMPROVISATIONS.md`:

- v5 scalar exact BF16-12: 0.9951x (`211b39873ae4497f`).
- v6 PRMT/pairwise exact BF16-12: 1.0316x (`3762b11e7ff64f2e`). The 24% byte saving is real,
  but consumer-side integer reconstruction is on the critical path (qkv m1 18.9 us vs 9.5 us).
- v7 old choice refresh: 1.1120x (`5e56c7e77fb34524`), statistically tied/slightly worse.
- v8 16-warp pooler: 1.1018x (`5f6f3910b4bb4ac9`); m8 regressed.
- v9 generic Triton m16/32 GEMM: 0.9239x (`7897be35f5c548f9`); qkv/pooler were about 45/38 us.

Do not retry the v5/v6 TMA-ring BF16-12 decoder, block-level raw fallback, or generic Triton tile.
The direct-load compressed v14 is a materially different route supported by public H200 data and
is now the first BF16 priority. Do not port compression elsewhere until this direct path passes.

Live ranks at the end of the cycle: overall kernel #661; MoE #535, attention #783, BF16 #1074,
decode #776, prefill #448. Prefill remains a separate board. BF16 has 15/20 checks and 10/10
submits left for Oct 28; every other problem has its full daily budget.

Oct 28 decode continuation: v7 enabled the embedded fused CUDA MoE at T=8 and submitted
1.1780x (`cf2294f258114dc5`). v8 also enabled it at T=16 and submitted **1.2326x**
(`bbb6e0110f24457f`, 18.352 ms, CI 1.2141-1.2511). Both scored slots reserved by this
runbook are used for decode today. The remaining six reserved checks tested T=32 NT1,
1x/2x/3x grids, large-T/T16 grid expansion, and wider attention splitting. Best check is
v12 at **1.2505x** (`ce61a83b4ddd450b`, 18.089 ms): fused MoE at T=8/16/32/128/256,
NT1 and a two-wave grid only at T=32, plus attention targets through 1,584 CTAs. Failed
expansions are logged in `RESULTS.md`. Decode's Oct 28 budget is closed at 8 checks and
2 submits. Last refreshed live ranks were decode **#689** and overall **#664**.
`submission.py` is the restored v12-equivalent check-only candidate; the live scored v8
is archived as `submission_v8_fast_moe_t16.py`.

All five tracks in the original runbook have now been completed once. Live overall rank
was **#631/1,977** after the final refresh; pjmullin3 was #1. Prefill is a separate board.

| Problem | Best score and job | Live rank / leader | Local promoted file |
|---|---|---|---|
| `muse_moe_w4a16` | **1.609x submit**, `63e72333221040c0` | #510; Abarbir12 2.620x | `submission.py` = v9 (WP=6 fused CUDA) |
| `muse_attention_decode` | **1.249x submit**, `b38b60919dfb42e6` | #758; pjmullin3 3.032x | `submission.py` = v6 hybrid |
| `muse_bf16_proj` | **1.116x submit**, `167d6e5c85f74682` | #1047; colinlikescode 1.637x | `submission.py` = v3 cluster |
| `muse_decode_layer` | **1.127x submit**, `e599e9c37ed040a3` | #751; pjmullin3 2.399x | `submission.py` = v6, Track-1 MoE for T>=128 |
| `muse_prefill_layer` | **1.118x submit**, `b6fbcde2c4aa4418` | #430; ericligit05 2.025x | `submission.py` restored to v5 |

Key continuation work:

- MoE v9 deepened the register prefetch pipeline to WP=6.
- Attention v6 uses the custom kernel only for batch-1 sliding verify and FA3 for the
  verify shapes where it wins.
- BF16 diagnostic v4 measured the proposed exact 12-bit exponent encoding: only
  0.0098--0.0103% escapes and about 1.3% blocks with an escape (very promising next track).
- Decode v6 embeds the Track-1 fused CUDA MoE for T=128/256; all 13 tests passed.
- Prefill v5 decodes touched experts to a timed BF16 scratch for custom grouped GEMMs.
  v6 overlap and v7 stage-4 trials lost; v8 decode-tile autotuning checked at 1.146x but
  submitted at 1.116x, so v5 remains the scored/local winner.
- Every folder now has `ARCHITECTURE.md` and `IMPROVISATIONS.md`; every remote run is in
  root `RESULTS.md`. No git commits were made.

For the next optimization cycle, start with BF16's exact 12-bit compressor/kernel, then
port it to the dense projections in decode. Do not redo Step 0 or Tracks 1--5 below; they
are retained as historical context.

Overall kernel board rank: **#780** (counts decode_layer + moe; prefill has its own board).

| Problem | Best measured | Scored? | SGLang → ours | #1 on board | Best code file |
|---|---|---|---|---|---|
| `muse_moe_w4a16` | 1.577× check / **1.554× submit** (`d0805799cf2245be`) | yes | 13.45 → 8.65 ms | ~2.52× | `submission.py` = `submission_v1_clean.py` (verify: check job `3469d3b6396a4309` scored 1.577×) |
| `muse_attention_decode` | **1.195× check** (`de9fbaaf2108416b`) | **no** | 3.77 → 3.15 ms | ~2.91× | `submission_v4.py` (recovered; very likely the 1.195× file) |
| `muse_bf16_proj` | **1.122× check** (`87850e2a50124d20`) | **no** | 4.83 → 4.30 ms | 1.637× | `submission_v3_cluster.py` (recovered; very likely the 1.122× file) |
| `muse_decode_layer` | **1.097× submit** (`6501bf63c08b48dd`) | yes | 22.62 → 20.62 ms | ~2.28–2.38× | `submission.py` = `submission_v5.py` |
| `muse_prefill_layer` | **1.065× submit** (`d3a6811b14844cb6`) | yes | 387.4 → 363.7 ms | ~1.93× | `submission_v5.py` (recovered; matches the submitted run) |

`submission.py` is **not** the best version in bf16_proj (still the starter), prefill (still the starter) or attention (old v0). Fix that in step 0.

Notes files: `muse_decode_layer` has both `ARCHITECTURE.md` and `IMPROVISATIONS.md`. bf16_proj, attention and moe have only `ARCHITECTURE.md`. prefill has neither. Read the existing ones before working on a track.

Leftover: `~/muse_work/muse_decode_layer` is a stale duplicate from when the drive was unplugged. Ignore it (the owner decides whether to delete it).

## 6. Working protocol (every track)

1. Read the track's `TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py`, `checks.py`, any `ARCHITECTURE.md`/`IMPROVISATIONS.md`, and the latest `profiles/*.txt` and `.log`.
2. **Never edit a passing version in place.** Copy it to `submission_vN.py` (next number) and edit that. `submission.py` always holds the best *verified* version.
3. Before each `okc check`, run `python3 -m py_compile` and re-read the diff for CUDA-graph violations.
4. Budget per track per day: **at most 8 checks and 2 submits**, so the owner keeps a reserve. Check `okc problems` first.
5. Only `okc submit` a file that has passed an `okc check` and beats the current best by more than the 95% interval (about ±1.5%).
6. After every run, append a line to `RESULTS.md` (project root): date, problem, file, job id, check/submit, speedup, notes. Create the file if it's missing.
7. If a check fails, read the failing case and `log_tail`, fix, and recheck. Two consecutive failures on the same idea means revert to the last passing version and move on.
8. End of track: update that folder's `IMPROVISATIONS.md` (see section 9). Then go to the next track.

## 7. Step 0: secure the cheap wins (do this first)

1. `okc check muse_attention_decode/submission_v4.py`. If it's around 1.19× and passes, copy it to `submission.py` and `okc submit` it.
2. `okc check muse_bf16_proj/submission_v3_cluster.py`. If it's around 1.12×, copy it to `submission.py` and submit.
3. Copy `muse_prefill_layer/submission_v5.py` to `muse_prefill_layer/submission.py`. It's already submitted, so no new run is needed.
4. Confirm `muse_moe_w4a16/submission.py` is the 1.55×-class code: compare it with `submission_v1_clean.py` and the log of job `3469d3b6396a4309` (its `log_tail` prints `[bench]` lines).

Expected outcome: all five problems on the board and the overall rank improved.

## 8. Track runbooks, in this order

### Track 1: `muse_moe_w4a16` (biggest headroom; counts on the overall board)

- **Now:** 3 launches, custom NVFP4 W4A16 CUDA kernel. Per-shape bench vs the bandwidth floor: bs1 39 vs 12 µs, bs16 76 vs 35, bs32 137 vs 71, bs8 87 vs 47, bs128 340 vs 136, bs256 486 vs 142. So it runs at **30–45% of the HBM floor**, and the gap grows with bs.
- An earlier v3 scored 1.345× (judge bs1 47.9 µs vs 35.9 µs in its own bench). Find out why judge and self-bench disagree (cold L2? launch count?) before trusting self-benchmarks.
- **Try, in order:**
  1. At bs≥128, many tokens share an expert. Make sure each expert's weights are read from HBM **once** per step, with all of its tokens batched into one tile (an M-tile over tokens), instead of per token.
  2. Persistent kernel with work split by **bytes of weight**, not by expert count, so skewed popularity doesn't leave SMs idle.
  3. Deeper async pipeline (TMA or cp.async, 4–6 stages) so HBM stays saturated. Measure achieved TB/s.
  4. Fuse w13 → SwiGLU → w2 → rmsnorm → gate-weighted sum to cut launches; use split-K on w2 for small bs.
  5. Faster NVFP4 → BF16 dequant (LUT in registers / `prmt`) if profiling shows it's ALU-bound.
- Stop when you've used today's budget or reached ≥2.0×.

### Track 2: `muse_attention_decode`

- **Now (v4):** custom CUDA split-KV kernel with clusters (see its `[plan]` log lines). Decode loads beat SGLang by 1.36–1.94×, but **verify (L=8, causal within the block) was much slower**. That's the main thing holding the geomean at 1.195×.
- **Try:**
  1. A verify path that packs all 8 query tokens × 8 query heads per KV head (GQA 16q/2kv per GPU) into one MMA tile, so each K/V page is read **once for all 8 tokens**.
  2. FP8 E4M3 queries with FP8 MMA for QK^T (allowed).
  3. Split-KV sized to fill 132 SMs for the 80K full-context layer.
  4. Keep QK-norm + RoPE + FP8 cache write + output norm + sigmoid gate fused.
  5. Per-shape choice between the own kernel and FA3 in `prepare()`.

### Track 3: `muse_bf16_proj`

- **Now (v3 cluster):** TMA-bulk weight streaming into a shared-memory ring, split-K across a thread-block cluster with the reduction over **distributed shared memory** (no global atomics). It wins at m=1 and on the router (fused hi+lo FP32 split, 1.8–4.5×). It **loses at m=16/32**: qkv 23.5 vs cuBLAS 13.6 µs, pooler 27.8 vs 20.8 at m=32. At m=128/256 it falls back to cuBLAS (about 1.0×).
- Stream-K with a global fix-up was tried and costs 2–4 µs of serial latency per CTA. Don't retry it.
- **Try:**
  1. Fix m=16/32: spread the per-stage x-slice copies across the producer warp's 32 lanes (one thread issuing 32 copies was the suspected bottleneck), and use WGMMA with m padded to 64 instead of mma.sync.
  2. Use plain `__syncthreads` when the cluster size is 1.
  3. Pick the faster of own kernel and cuBLAS for every (m, layer) in `prepare()`.
  4. At m=128/256, try a Hopper WGMMA GEMM (CUTLASS / DeepGEMM-style) tuned per shape to beat cuBLAS heuristics.

### Track 4: `muse_decode_layer`

- **Now (v5, 1.097×):** SGLang's composition plus a Marlin MoE block-size retune (block 16 at T=128/256: 391 vs 534 µs and 536 vs 717 µs), own Triton split-KV attention chosen per shape vs FA3, a fused FP8 KV write, and fused expert-id mapping.
- Bottlenecks:
  1. The MoE takes 71% of verify-32 time (444 µs, ~1.9 TB/s).
  2. About 35 launches ≈ 40 µs of gaps at T=1.
  3. Full attention at B=16/32 with page size 1: FA3 takes 210/395 µs against a 68/136 µs floor.
- **Try:**
  1. **Port the Track 1 MoE kernel** in place of Marlin (biggest lever).
  2. **Port the Track 2 attention** and **Track 3 projections**.
  3. Fuse the norm+residual chains and the router (sigmoid + bias + top-8) into single kernels to cut launches.
  4. Use PDL (programmatic dependent launch) to overlap kernel prologues.

### Track 5: `muse_prefill_layer` (own board, lowest priority)

- **Now (v5, 1.065×):** autotunes in `prepare()` between Marlin and own Triton grouped-GEMM variants for each T. Marlin wins at T=753; own variant C at T=3012 (2259 vs 2480 µs); variant E at T=16384 (9852 vs 13585 µs). Variant G fails on shared memory (needs 294,912 bytes, limit 232,448). Make it fit by reducing stages or block size.
- **Try:**
  1. A WGMMA-based grouped NVFP4 GEMM with warp specialization (producer TMA warps, consumer MMA warpgroups) and in-register dequant.
  2. Fused SwiGLU and norm epilogues.
  3. FA3 FP8 prefill attention tuning.
- Write the missing `ARCHITECTURE.md` and `IMPROVISATIONS.md` here.

## 9. Documentation required per track

Each problem folder must end with:
- **`ARCHITECTURE.md`:**
  - H200 (Hopper) facts that matter for this kernel: HBM3e 4.8 TB/s nominal (~3.2–4.3 measured), 132 SMs, 50 MB L2, 228 KB shared memory per SM, TMA, WGMMA, FP8 tensor cores, no FP4, clusters and distributed shared memory, CUDA-graph launch overhead.
  - How this piece of Muse Spark Open works: shapes, TP8/EP8 sharding, where it sits in a decode step.
- **`IMPROVISATIONS.md`:**
  - (a) Every known optimization for this problem, with estimated gain and tried/not tried.
  - (b) What was done per version, with measured okc results and job ids.
  - (c) Profile analysis: time per phase, the longest phase and **why**, and % of roofline.
  - (d) 2–4 **novel** ideas. For each: what the existing approach does at which stage vs. what ours does instead, why it wins on H200, estimated gain (marked as an estimate), risk, and an implementation plan.

Write for a human studying later: clear, articulate, honest. Mark estimates as estimates.

## 10. The novel-idea agenda (where we try to beat the crowd)

The top ~100 entries cluster tightly (bf16_proj: 1.53–1.64×). The crowd is clearly doing the standard things: split-K GEMV, autotuned tiles, fused routers. Ideas we have that others probably don't:

1. **Lossless 12-bit BF16 weight compression.** The rules allow lossless compression. Store each weight's 8-bit exponent as a 4-bit code relative to a per-block base exponent, with a rare escape list for outliers. That's sign + 7 mantissa + 4 exponent-code = 12 bits, so **~25% fewer weight bytes** on bandwidth-bound decode GEMVs. Rebuild exact BF16 in registers with idle integer ALUs.
   - Estimated +6–10% on decode loads, applying to bf16_proj, decode_layer and the serving track.
   - First step: in `prepare()`, measure the per-block exponent spread of the real weights and print the escape rate. If the escape rate is under ~1%, build the kernel.
2. **Cluster split-K with a DSMEM reduction** (already prototyped in bf16_proj). It removes the global-memory fix-up chain of stream-K. Extend it to the MoE w2 GEMV and the attention split-KV combine.
3. **Byte-balanced persistent MoE scheduling.** Assign work by weight bytes, not expert count, so skewed routing doesn't leave SMs idle.
4. **Cross-kernel prefetch in the whole layer.** As soon as routing is known, start pulling the selected experts' NVFP4 weights toward L2/shared memory while the pooler GEMV runs. That overlaps the two largest HBM streams. The L2 is cold only at the start of the timed call, so prefetch inside the call is legitimate.

Record which idea was tried, the measured result, and whether it should go into the serving patch.

## 11. Known hazards

- **The external drive (`/Volumes/Soorya_1_TB`) has unmounted mid-session several times.** Check `ls /Volumes/Soorya_1_TB` before writing. If it's gone, stop and tell the owner; don't rebuild work elsewhere.
- Anything saved only under `/private/tmp/...` is lost when the session ends. Keep every version in the project folder.
- Judge queues can be slow (several minutes). Use `okc status <job> --json`; don't resubmit.
- Prepare-time autotuning eats into the 10-minute run limit. Prune losing candidates.
- Don't commit to git unless the owner asks.

## 12. Definition of done (per session)

- Step 0 completed.
- Each track attempted in order within its budget.
- `submission.py` in every folder is the best verified version.
- `RESULTS.md` updated, and `ARCHITECTURE.md` + `IMPROVISATIONS.md` present and current in all five folders.
- Final summary to the owner: per-problem speedup before → after, job ids, overall rank, and which novel idea showed the most promise.
