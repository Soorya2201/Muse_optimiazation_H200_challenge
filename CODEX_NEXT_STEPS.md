# Codex next steps (written Oct 29 2026, ~23:30 UTC)

Read this after `CODEX_HANDOFF.md`. It **overrides the handoff's "next" lines**. It is based on:
- a review of the per-shape judge data for every scored file,
- a new competitor audit (alygg77, Madhavan113's newer commits),
- the official serving-track rules.

Work one item at a time in the order below, and log every okc run in `RESULTS.md`.

Kernel submissions close **Nov 4, 11:59 pm PT**. Overall rank was **#108/1627** at 22:45 UTC
(it was #95 earlier). The top 50 get serving priority.

## What the review found

- Every big gain so far came from checking exact public sources (vjeux, Nathen, Madhavan).
  The ~20 own experiments since then were all ±1% micro-variants. That is below the judge's
  noise, and none was promoted.
- The judge is noisier than we assumed. alygg77 measured an unchanged file spreading
  **1.6-2.2%** across runs, and the two judge clusters (`sfo3`, `sfo3c`) differ by a common
  offset of about 1%. A single check-vs-check comparison can't resolve gains under about 2%.
- v20 attention probe (`acf6f11e935049db`): 0/50 alternative workspace/finalizer plans
  qualified. Most regressed; B16L8 nc2/coop1 was mixed (0.998952-1.014768x), and
  larger B1L8 plans lost much more than 35%. No consistent >2% gain. **Stop** the batch-1
  attention merge/protocol work (v17-v20). alygg77 also closed 11 real-kernel arms there:
  one-request full attention is fixed-latency bound.

## P0: compliance risk in our scored decoder (do first)

- **The rule:** for `muse_decode_layer` at T ≤ 8, SGLang feeds the router an **FP32**
  activation. See `muse_decode_layer/baseline.py` lines 13-14 and 316-318 (`router_dtype=torch.float32`
  when not split). Above 8 tokens it uses a BF16 activation with the hi+lo weight pair.
- **What our file does:** our scored decoder (`submission.py` = Madhavan `c180924`, job
  `d857144e59784ad0`) computes the pooler+router from the BF16 `h`. See `_after_attention`,
  comment "pooler + router (BF16 hi + lo rows, FP32 accumulation)", and `rgemv` takes
  `const __nv_bfloat16* x`.
- **Why it matters:** alygg77 and Matsvei (agents.kernarc.com) call this the non-compliant
  case. Organizers review the top entries by hand.
- **Confirm it:** read `_after_attention` (about lines 18647-18760) and every pooler+router
  form (`bfp`, `bfpf`, `rgemv`) at T = 1 and T = 8.
- **Fix:** see P1 item 1. alygg77's decoder already fuses an exact FP32-activation × FP32-weight
  router into the pooler at T ≤ 8, and scores higher than ours.
- **Already checked, fine:**
  - The `muse_bf16_proj` track: its input x is BF16 by definition, and hi+lo weights are
    allowed at any batch size.
  - The MoE and attention tracks don't compute the router.

## P1: check the newly found public files (exact controls first)

alygg77 published `github.com/alygg77/okc-muse-spark-kernels` today (Apache-2.0, commit
`6cae49a`). It is #40 overall, and every file is built on the same vjeux `f35a41e` base we use.
Exact copies are already in the repo; they pass `py_compile` and keep their first-line problem
markers. Their docs, license, NOTICE and timing kit are in `research/public/alygg77_6cae49a/`.

| Order | File | SHA-256 | Their scored | Ours scored | What's new |
|---|---|---|---|---|---|
| 1 | `muse_decode_layer/submission_v17_alygg77_6cae49a.py` | `8f17613b…af4a` | 2.2058× (`a61afb9687f14eff`) | 2.1278× | FP32 router at T ≤ 8 (fixes P0), KPERM3 experts at T ≥ 17, cuBLASLt holder/context bind |
| 2 | `muse_moe_w4a16/submission_v24_alygg77_6cae49a.py` | `b3307089…4281` | 2.4015× (`e1bcad5a55cd4a29`) | 2.3241× | **KPERM3**: lossless NVFP4 K-permutation so one scale pair serves 4 k16 steps; no register spill. Judge per-load: d32 +2.6%, v16 +3.9%, v32 +3.1% |
| 3 | `muse_bf16_proj/submission_v34_alygg77_6cae49a.py` | `290a86ab…7b90` | 1.5502× (`f07235a6ca8a46e2`) | 1.5219× | Custom WGMMA `o` kernels, pooler tuning, Nathen's winning small-m routes only. Their own unchanged-file spread is 1.526-1.550, so check it **after** the queued exact Chen (`submission_rank78_chen_3d418e1.py`) |

Rules for these checks:
- Read each file's top docstring, license/attribution header and timed path first. Confirm
  precision, that `prepare()` stays under 300 s (alygg77 saw 273-312 s cold on MoE builds: watch
  it), and that nothing is cached from inputs.
- Run each exact file twice with `okc check`, then compare **per-shape rows** against our same-day
  parent on the same judge cluster. Promote only if the changed loads move and the unchanged ones
  don't.
- For the decoder, P0 makes this worth doing even at score parity.
- Prefill and attention: alygg77 says both of its files are vjeux's unchanged, so there's
  nothing new there. Their 1.8167× prefill row was a fast run of the same file we have.
- Madhavan's newer attention/decoder commits (`012d92b`, `721c627`: TMA bulk-reduction merge,
  -0.3 to -0.5 µs) are too small to resolve on the judge. Skip them.

## P2: work practices

1. **Judge clusters:** record the cluster name in every `RESULTS.md` line. alygg77 reads it from
   the judge output; check the `okc status --json` / log for `sfo3`/`sfo3c`. Only compare runs
   from the same cluster.
2. **Bound the gain first:** before any experiment, compute from the per-shape table the
   board-level gain if the targeted phase went to zero. Skip anything under ~3%.
3. **Memory layout is a hidden variable:** growing one `prepare()` scratch buffer by 3 MB slowed
   d1/v1 by 0.5-1.3 µs with identical SASS. Reuse existing buffers, and suspect layout before
   claiming a regression on unchanged shapes.
4. **PDL pitfall:** never read producer-written data through `const __restrict__`/`__ldg` in a
   PDL-dependent kernel. ptxas can hoist `ld.global.nc` above `griddepcontrol.wait`, which breaks
   only the first replay. Read producer data with `__ldcg` after the wait. Audit any new decoder
   glue for this.
5. **Fresh-thread context:** `muse_prefill_layer/submission.py` calls raw cuBLASLt handles (2 hits)
   with no CUDA-context bind. alygg77 saw such a file fail when called from a fresh thread until
   it bound the primary context. It passes today; if the organizers' reruns use another thread,
   it could fail. Low-cost hardening: port alygg77's context-bind helper (credit Matsvei).
6. **Owner decision:** a rented judge-faithful H200 (Madhavan pays about $5.40/h) with
   `research/public/alygg77_6cae49a/timing_kit` (paired cold-L2 A/B/A with 4 memory pads)
   resolves 0.5 µs differences. The judge can't. Ask the owner before assuming access.

## P3: optimization ideas, corrected by competitor evidence

- **Withdrawn: the MoE bs16 double-read fix.** Our `MID` build re-streams a >8-token expert,
  but alygg77 measured that removing the second read speeds other CTAs by only ~1%. The d16
  path is co-bound (phase A at the size-matched ceiling, phase B consumer-bound). Do not spend
  a check on `OKC_SKT=16`.
- **MoE (after KPERM3):**
  - The WGS stream is limited by unit ends, not by rate. Each 16-stage unit end stalls a CTA
    1-7 µs, a release `MEMBAR.GPU` costs ~0.9 µs, and fence count is the cost.
  - The one untested direction: fewer releases per CTA (e.g. publish readiness per 2 units)
    without adding fixups. Measured ceiling evidence is weak (their NOFENCE gained +0.23%).
    Treat this as low priority.
- **Attention full verify (v16f/v32f):**
  - The consumer spends 40.5% on exponentials (MUFU `ex2`, 16 ops/clk/SM). Packed f16x2 exp
    lowers to two MUFU ops, so `OKC_EXH8F` won't help. A full degree-5 polynomial and an
    FA3-style one-CTA pipeline both lost.
  - **Untested:** a hybrid split as in FlashAttention-4. Compute a fixed fraction (e.g. 25-50%)
    of the exponentials with a degree-3 Cody-Waite polynomial on the FMA pipe and the rest on
    MUFU, only in `attn_kernel<8,true,0>`. P is rounded to FP8 E4M3 anyway, so ~1e-4 relative
    exp error is far below the 15% bound.
  - Estimate, not measured: if verify-full drops to decode-full time + 50% of the current gap,
    the attention board gains ~+1.5-2%. Build it on a rented GPU first.
- **Decoder:** d32s is MoE 84 µs + dense GEMVs 51.5 µs (34.9 µs floor) + glue 8.8 µs + attention
  9.8 µs (alygg77's breakdown on the same base). Side streams, early pooler starts, cross-kernel
  L2 prefetch, launch skipping and cuBLASLt sweeps are all measured negatives (their
  NEGATIVE_RESULTS.md). The only open big item is the MoE stream itself.
- **Already closed by others' measurements (don't retry):**
  - NVFP4 byte compression (≤5%, weights look random).
  - Sub-12-bit BF16 codes (upper bound +2.4%, realistic ≤1%).
  - In-register C12 decode at m ≥ 128 (integer ALU doesn't overlap HGMMA).
  - Split-K at m ≥ 128 (L2 write bandwidth ~45 GB/s per SM).
  - Page sorting in attention.
  - 4th WGS weight slot.
  - Deeper rings.
  - Fused final MoE pass.
  - Dynamic MoE tails.

## P4: serving track preparation

Follow `serving/SERVING_PLAN.md`. Start with the offline MoE port to SGLang's `load_jit`
infrastructure behind `SGLANG_OKC_MOE=1`. It's the biggest share of a serving step at 16-32
requests, and the KPERM3 MoE from P1 is the version to port.

## Public-source ledger update

Add to `PUBLIC_SOURCE_INDEX.md` (done in this commit):
- alygg77 `6cae49a`: three exact candidates pinned, plus its facts/negative docs.
- Madhavan's `721c627` design notes in `research/public/madhavan_721c627/`. Two of their
  notes contain someone else's VM access details and were not copied.
