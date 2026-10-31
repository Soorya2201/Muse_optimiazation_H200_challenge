# muse_decode_layer: optimizations, measurements, and new ideas


Latest Oct31 decoder cycle: see [rounded-width experiment](ROUNDED_WIDTHS_RESEARCH_2026-10-31.md)
and [research library](RESEARCH_LIBRARY.md). New native four-width code passes
64-count CPU address/guard checks and full-switch NVCC compilation, but
spills16bytes and adds arithmetic/shared writes; no credible>=3%gain case.
No H200 job/submit/promotion. Current leader public-repo count0; licensed
FlashInfer MonoMoE source/transfer limits pinned. Next sequential trackprefill.
Older priorities below are history and do not authorize closed-family retries.

## Oct31 fresh-source and operand scheduling screen

Canonical score remains2.211495x; no new GPU check or scored submit.
The public index grows87->94paths/9->10repos; all1106ranked account metadata
screened,5unresolved. Deferred Kushagra/ScrashOff sources and180external
cells supply no stronger replacement; ScrashOff loses required small-T
router precision. Newly discovered Apache2 trxphxle72db638 measurements and
nine unmodified docs/data are pinned. All18route histograms/byte pools
independently verify; conditional decoder reread-removal proxies1.00097–
1.02439x do not establish a>=3%gain. These are not critical-path bounds.
A new delayed-X-wait native transformation compiles reduced sm90a with
96regs/zero spills; no GPU validation or practical>=3%case yet. Do not
promote or spend a check on compiler success alone. Full sources, models,
status, higher-precision fusion policy uncertainty and next action are in
[the dated review](PUBLIC_DELTA_AND_OPERAND_SCREEN_2026-10-31.md).

## Oct30 precision correction

The Oct29 public decoder result below is historical and was withdrawn.
Madhavan c180924 feeds BF16 activations to the small-T router, whose input the
baseline keeps in FP32. Its vjeux parent has the same issue. Do not promote or
resubmit either unchanged. See `ROUTER_PRECISION_AUDIT_2026-10-30.md`.
Exact alygg77 source `submission_v17_alygg77_6cae49a.py` fixes the router and is
passed checks `521b039c454045da` and `5c929cfd9c0b4215` on sfo3 at
2.189801x/2.221428x, all13tests/12timings each. Local `submission.py` is now
the exact verified v17. Same-day historical-source control is running;
the noncompliant small-T parent is never eligible for resubmission.

## Oct 29 public-source reproduction (supersedes historical baseline claims below)

Exact Apache-2.0 vjeux source at f35a41e is archived in
`submission_v15_vjeux_f35a41e.py`. Check `aabb75632b31456b` passed all13 tests and12
timed cases at **2.089048x**,10828.043us,CI[2.057712,2.120384],prepare101.64s.
This is a substantial gain over the scored incumbent1.232602x/18351.662us.
All980 ranked accounts were screened for public repository metadata, including807 above
and172 below us; `PUBLIC_RESEARCH.md` records actual source review and limitations.

The measured architecture combines losslessly compressed BF16 projections, fused paged
attention, persistent NVFP4 MoE, and residual/norm/routing fusion. These are reproduced
public mechanisms, not a claim of original invention. The normal tuner path restores its
synthetic calibration bias; no tune failure was reported in this passing control.

Representative profile shares: T1 sliding MoE22.6us/28.4%,pooler18.5us/23.2%,
QKV10.2us/12.8%; T128 sliding MoE155.4us/about60%; T256 sliding MoE205.1us/59.9%;
T256 full MoE224us/42.1%,attention187.2us/35.2%. PDL consumer waits can overlap,
so adding native-kernel durations does not estimate wall-time savings.

The higher-ranked Madhavan derivative passed check9f256608c2df4cc8 at **2.119204x**,
10673.962us,CI[2.087416,2.150992],prepare189.92s,all13tests/12timings. Exact source is
promoted; scored reproduction d857144e59784ad0 passed all13tests/12timings at2.127754x,
10631.070us,CI[2.095837,2.159670],prepare190.60s. Rank#95/984,overall#95/1607. It selects CUDA boundary and
routing kernels,agent MoE everywhere,gluev2=False. Experimental tails/in-MoE routing are
unavailable. Its1.44% edge over the control is within uncertainty; both decisively beat the
prior scored incumbent. The derivative's standalone MoE had not improved the aggregate.

The2.119204x candidate needs approximately12.35% less normalized time to match the
then-leader2.4178x. An optimistic phase model projects MoE-20%->2.3381x,MoE-30%->2.4659x,
attention-30%->2.2189x,glue-30%->2.1685x. These are planning estimates that overcount PDL
wait savings, not measured improvements. Exact8-op dequant with a matching lossless repack
from lower-ranked lee101 is queued for a future controlled candidate; source and numerical
requirements are in PUBLIC_RESEARCH.md. First confirm this cycle,then proceed to prefill.

The sections below describe older source and results and are retained as experiment history.

All speedups are okc results: the geometric mean over the 6 loads of SGLang time / mine,
measured on the judge's H200 with CUDA-graph replays and a cold L2. "(est.)" marks my own
estimates. Prepare-time micro-benchmarks ("lab") are CUDA-graph replays with a cold L2,
taken inside `prepare()` on the judge. They include ~3 µs of empty-graph overhead
(measured: `empty_graph=2.9`).

**Bottom line, stated plainly.** My best version (v5) is about **1.10x** over SGLang. The
leader is at about 2.38x, so this entry is nowhere near the top. Section (d) explains
what it would take to get there.

---

## (a) Known optimizations: estimated gain and status

Each gain is a share of the geomean speedup, estimated from the v0/v5 profiles.

| # | Optimization | Where it helps | Est. gain | Status |
|---|---|---|---|---|
| 1 | **Marlin block-size retune** (moe_block_size 16 instead of SGLang's heuristic 48/64 at T=128/256) | verify-16/32 MoE (Marlin 409→~330, 559→444 µs) | +5–8% geomean | **Done (v5). Measured: verify-16 sliding 1.22x, verify-32 sliding 1.16x** |
| 2 | Split-KV Triton decode attention with fused combine + per-head RMSNorm·sigmoid(gate) | sliding layers (all B), full B=1, full decode B=16/32 | +1–3% | **Done (v1→v5), autotuned per shape against FA3** |
| 3 | Fuse FP8 cast of k and v + `store_cache` into one kernel | every shape, −2 launches | +0.5–1% | **Done (v5), byte-exact vs SGLang** |
| 4 | Fuse `(ids − local_start) % 256` + expert_map gather into one kernel | every shape, −2–3 launches | +0.5–1% | **Done (v5)** |
| 5 | T≤8: post-FFN boundary reads the 1024-column unpooler slice directly (no `F.pad`, no 8192-wide zero tensor) | decode-1, verify-1 | +1% | **Done (v5)** |
| 6 | Custom NVFP4 W4A16 grouped GEMM (Triton) | MoE everywhere | potentially +30–60% if at bandwidth | **Tried (v1–v4): correct (rel err 1.7e-3) but 1.3–2x *slower* than Marlin. Not used** |
| 7 | Custom NVFP4 MoE in CUDA (Marlin-style lop3 dequant + wgmma, swap-AB, persistent over experts, split-K for T≤8) | MoE | +40–70% (est.) | Not tried (the biggest remaining lever) |
| 8 | FP32 router as a Triton GEMV at T≤8 (replaces cuBLAS dot / sm80 xmma FP32: 5 µs at T=1, 22 µs at T=8) | decode-1, verify-1 | +1–2% | Not tried (it is on the side stream, so mostly overlapped) |
| 9 | Better BF16 GEMV for qkv / pooler at T≤32 (cuBLAS reaches ~3.2 TB/s: 105 MB in 32 µs) | all small-T loads | +3–5% | Not tried |
| 10 | Fuse QK-norm + RoPE + KV store + query packing into one kernel after the qkv GEMM (or into the GEMM epilogue) | all | +1–2% | Partly (KV store only) |
| 11 | Fuse pre-expert RMSNorm into the pooler GEMM epilogue (needs a full 4096-wide row per CTA, or a two-pass norm) | all | +1–2% | Not tried |
| 12 | PDL between consecutive GEMVs (weights prefetched before the activations are ready) | all, mostly T≤32 | +10–20% at T=1 (est.) | Not tried (needs custom kernels; cuBLAS and sglang kernels are not PDL-chained here) |
| 13 | Whole-layer persistent megakernel | all | +50–100% at T=1 (est.) | Not tried |
| 14 | FA3-quality full-context attention at page_size=1 (wgmma FP8 QK^T, 128 B KV rows read per token for both heads) | full B=16/32 (FA3: 210 / 395 µs against a 68 / 136 µs KV floor) | +5–10% | Partly: mine wins at decode B=16/32 (189 / 345 µs) but loses at verify (L=8, compute-heavier) |
| 15 | Router hi/lo BF16 split also at T≤8 | T≤8 | small | Not tried (FP32 is cheap at T=1) |

---

## (b) What each version did, with measured okc results

| Version | Content | okc job | Speedup (95% CI) |
|---|---|---|---|
| v0 (`submission_v0_working.py`) | SGLang's composition verbatim (baseline.py plus the header line) | 0db4409f9ba84d39 | 1.001x (0.986–1.016) |
| v1 | + Triton split-KV attention (always on) and a Triton NVFP4 MoE (did not compile: a Triton global was not constexpr, so the self-test fell back to Marlin) | b061269de3a84857 | **0.956x**. My attention was 1.3–1.4x slower than FA3 on full layers at B=16/32 (276 vs 194 µs, 507 vs 376 µs) but faster on sliding layers |
| v2 | Attention prefetches the next tile's page ids and loads V early. Splits are tunable, and a prepare-time autotuner picks mine or FA3 per (B, L, layer) | f5913655bfc3494e | 1.024x (1.008–1.039) |
| v3 | Fixed the MoE kernel compile (asm string inlined). MoE self-test passes (1.67e-3) but the autotuner keeps Marlin at every T | 7520dee352494333 | 1.027x (1.012–1.043) |
| v4 | "Lab" build: split-nibble repacked NVFP4 layout (lossless), FMA-GEMV variants, a pure-bandwidth probe, kernel register/spill stats | 73420bcb91b3469c | 1.021x (1.006–1.036) (same choices as v3) |
| **v5** (`submission.py`) | Marlin block-size / atomic autotune per T, self-tested against the SGLang default. Fused KV store, fused local-id mapping, T≤8 sliced FFN boundary. Attention autotune kept | 847f09c9413f4dd7 (check) | **1.100x (1.083–1.116)** |

v5 per-shape results (check 847f09c9413f4dd7), in µs:

| Load | Sliding: SGLang / mine | Full: SGLang / mine |
|---|---|---|
| decode 1 | 170.2 / 156.3 (1.09x) | 181.0 / 171.3 (1.06x) |
| decode 16 | 267.6 / 252.7 (1.06x) | 434.3 / 415.2 (1.05x) |
| decode 32 | 262.0 / 246.1 (1.07x) | 619.3 / 566.4 (1.09x) |
| verify 1 | 268.2 / 252.6 (1.06x) | 278.5 / 270.7 (1.03x) |
| verify 16 | 537.4 / 440.7 (1.22x) | 711.6 / 615.2 (1.16x) |
| verify 32 | 724.8 / 623.0 (1.16x) | 1079.8 / 974.0 (1.11x) |

Scored submit: see the final section.

### What the labs measured (prepare-time, judge H200)

Attention (µs per graph replay, ~3 µs of which is overhead), FA3 vs my best variant:

| Shape | Sliding: FA3 / mine | Full: FA3 / mine |
|---|---|---|
| B=1 L=1 | 18.1 / 11.6 | 37.2 / 28.3 |
| B=1 L=8 | 20.7 / 14.1 | 38.9 / 38.7 |
| B=16 L=1 | 23.4 / 16.7 | 210.0 / 189.3 |
| B=16 L=8 | 26.5 / 21.6 | 210.5 / 280.7 (FA3 kept) |
| B=32 L=1 | 29.1 / 23.8 | 394.5 / 345.0 |
| B=32 L=8 | 31.8 / 32.7 (FA3 kept) | 395.1 / 524.9 (FA3 kept) |

The attention self-test error vs an FP32 torch reference is 1.85e-3 (BF16 queries and P),
which is lower than FA3's FP8 path.

MoE (lab, GEMM1 alone, T=1 with 3 local experts, 57 MB of codes + scales):
Marlin full MoE ≈ 85 µs per graph, **my best Triton GEMM1 103 µs, and the pure-load
bandwidth probe in the same tiling 56.7 µs** (≈ 1.0 TB/s). Registers were fine (106–178)
with zero spills. The conclusion: my Triton tiling (64 rows × 64 B per load, one weight
tile per CTA per stage, grid = 3 experts × 64 n-tiles = 192 CTAs) cannot keep enough
bytes in flight, so even zero-math streaming reaches only about 20% of HBM. The FMA-GEMV
variant was 3–40x slower (the 3D broadcast-reduce compiles badly). At T=256 the best
Triton variant (split layout, BM=32) took 491 µs for GEMM1 alone, against Marlin's ~536
µs for the *whole* MoE at BM=16.

Marlin block size (whole MoE, µs, synthetic Zipf routing, mean of 2 draws):
T=128: BM16 391 vs SGLang's BM48 534. T=256: BM16 536 vs SGLang's BM64 717. At T≤16,
SGLang's default (8) is already best. Atomic-add mode never helped.

---

## (c) Profiling: where the time goes

**decode-1 sliding, v5 (161.7 µs GPU time vs a 40 µs floor, 25% of roofline):**

| Phase | µs | Floor (bytes / 4.8 TB/s) | Notes |
|---|---|---|---|
| MoE (Marlin ×2 + align + silu + post-norm) | ~55 | ~6–18 (1–3 experts × 28 MB) | **longest phase** |
| qkv + pooler GEMV (cuBLAS splitK, 2 calls) + reduces | 32 + 5 | 21.9 | ~3.2 TB/s |
| o-proj + unpooler-slice GEMV | 7 + 5 | 5.3 | |
| router (FP32 dot) + routing (side stream) | ~8 | 1.7 | mostly overlapped with the pooler |
| attention (mine) | 5.1 | 0.1 | latency-bound |
| ~15 small norm / elementwise / boundary kernels | ~40 | ~0 | launch tails and cold-start latency |

**verify-32 sliding, v5 (629 µs vs a 144 µs floor):** Marlin is 444 µs, or 71%. That is the
longest phase on every T≥16 shape.

**Why the MoE is the longest phase.** At T=256 the busiest GPU sees ~400 routes over
~30 experts, which is ~840 MB of NVFP4 (175 µs at 4.8 TB/s) plus ~40–80 GFLOP of BF16
MMA after dequant. Marlin uses `mma.sync` (no wgmma) and a 4-bit dequant per weight per
token block. With skewed routing, hot experts need several M-blocks, and each block
re-reads and re-dequantizes the expert's weights (from L2 if lucky). Measured: 444 µs,
≈ 1.9 TB/s effective, ~2.5x off the bandwidth floor. At T=1 Marlin is ~45 µs for ~3
experts (~85 MB): ~1.9 TB/s, because a few experts' worth of CTAs cannot fill 132 SMs.

**Why full-context attention at B=16/32 is the second longest.** At page_size=1 every key
is a scattered 128 B gather (64 B per head). FA3 gets ~1.7 TB/s (376 µs for 655 MB). My
kernel reaches ~1.9 TB/s at decode, and loses at verify because its P·V and QK^T run as
BF16 `mma.sync`-class MMAs with 64 query rows (it becomes compute-limited; FA3 uses FP8
wgmma).

---

## (d) New ideas (2–4), honest estimates

### Idea 1: entropy-coded BF16 weights ("lossless weight compression in the GEMV")

* **What exists.** SGLang streams the BF16 dense weights (qkv 37.7 MB, pooler 67 MB,
  o 16.8 MB, unpooler 8.4/67 MB) as raw 16-bit values. At T≤32 these GEMVs are purely
  bandwidth-bound: ~130–190 MB of BF16 per step, about 75% of all decode-1 bytes.
* **What's new.** TASK.md explicitly allows lossless compression in `prepare()`. BF16
  weights from a trained (or here Gaussian, std ~0.011) distribution have very
  low-entropy exponents: ~2.6 bits of information in an 8-bit field. Store each weight as
  sign + 7-bit mantissa + a **4-bit exponent code relative to a per-64-weight block
  base**, plus a rare-escape side list for exponents outside the 15-value window: 12 bits
  instead of 16, a 25% cut in bytes. The GEMV kernel reconstructs the exact BF16 values
  in registers (one `prmt` + add of the block base per pair, ~1–2 ops per weight) right
  before the FMA or MMA. This is DFloat11 / ZipNN-style compression, but with a
  fixed-length 12-bit code so the decode is branch-free and SIMD-friendly. Huffman
  decoding cannot keep up at 4.8 TB/s, while a fixed 12-bit format can.
* **Why it wins on H200.** Decode is bandwidth-bound and ALU is idle: at T=1 a GEMV does
  1 FMA per 2 bytes, so the SMs have ~10x more integer throughput than needed to
  reconstruct exponents. 25% fewer bytes means ~25% less time on the 75% of the step that
  is dense GEMV.
* **Estimated gain.** decode-1: ~29 µs of dense-weight floor → ~22 µs, which is
  −7 µs at the floor, or about −10–15 µs vs cuBLAS-class kernels (est.). Roughly +6–10%
  geomean on the small-T loads, more once launch overhead is gone.
* **Risk.** It must be bit-exact. The escape path needs to be rare; I estimate
  P(|w| < base·2^-15) ≈ 3e-5 for Gaussian weights. Real checkpoints could have a heavier
  tail (more escapes, but still correct). Reviewers must agree it is "lossless": it is,
  by construction, and it is verifiable with a byte compare against the original in
  `prepare()`.
* **Plan.** (1) prepare: per 64-weight block, base = max exponent; codes = base − exp,
  clamped to 14, with 15 = escape; escapes go to a CSR side list. (2) Byte-exact
  round-trip check in prepare. (3) A swap-AB GEMV kernel (TMA-fed, 4–6 stages, split-K
  for N=2304) that unpacks 3-byte groups into bf16x2 with `prmt`, adds the block base
  into the exponent field with an integer add, and applies escapes in the epilogue as a
  sparse correction. (4) Use it for qkv, pooler, o, and unpooler.

### Idea 2: route-aware expert-weight streaming ("router-first, experts-early")

* **What exists.** SGLang computes the router on a side stream in parallel with the
  pooler, but the experts cannot start until both the routing *and* the pooler output are
  done. The 28 MB/expert weight stream (the longest phase) begins only after ~60% of the
  step has elapsed.
* **What's new.** The routing decision depends only on `h` (available right after the
  attention boundary), not on the pooler. Run the router first (8 MB, ~2 µs at
  bandwidth), then launch the expert GEMM1 kernel with PDL / as a persistent co-resident
  kernel *concurrently with the pooler*. Its CTAs immediately start prefetching their
  experts' w13 tiles (cp.async.bulk.prefetch.L2 or TMA into shared memory) while
  `griddepcontrol.wait` holds the math until the pooler and pre-expert norm finish. Size
  the prefetch to L2 (50 MB ≈ one hot expert's w13 plus the pooler's streaming
  footprint), ordering experts by route count so the hottest expert's weights are already
  on-chip when the activations land.
* **Why it wins on H200.** The pooler (67 MB) and the experts (28 MB × k) are *both*
  bandwidth-bound and independent in their weight traffic. Overlapping them does not help
  if both saturate HBM, but neither does today: cuBLAS GEMV reaches ~3.2 TB/s and Marlin
  ~1.9 TB/s. Overlap raises the union toward the 4.8 TB/s ceiling and hides the expert
  kernel's ramp-up and launch gap.
* **Estimated gain.** 10–20 µs per decode step at small T (est.), +5–8% geomean.
* **Risk.** L2 thrash if prefetching too much. It requires custom MoE kernels (Marlin is a
  black box), and correctness is unaffected.
* **Plan.** A router GEMV plus top-8 in one Triton/CUDA kernel → a tiny expert-histogram
  kernel → the MoE kernel launched with the PDL attribute and an L2-prefetch prologue →
  pooler and pre-norm on the main stream → MoE `griddepcontrol.wait`.

### Idea 3: verify-block KV reuse across GQA *and* draft tokens in one wgmma tile ("fat-query attention")

* **What exists.** For verify (L=8), FA3 packs GQA (8 heads) × 8 draft tokens = 64 query
  rows per KV head and streams the KV once. That part is good. But it treats
  `page_size = 1` gathers through a generic paged path, and at B=16/32 full layers it
  reaches ~1.7 TB/s (210 / 395 µs vs a 68 / 136 µs floor).
* **What's new.** A dedicated page_size=1 kernel where one CTA owns *both* KV heads of a
  request (a 128 B row per token = one full L2 line, versus 64 B half-lines today), uses
  `cp.async` 16 B gathers 8 tiles deep, and runs QK^T as an FP8 wgmma (M = 2 heads × 64
  rows split over two warpgroups, N = 128 keys), with P×V in BF16 (more accurate than
  FA3's FP8 P). The split-KV combine fuses the per-head RMSNorm·sigmoid(gate) (already
  done in my combine kernel) and writes the BF16 input of the o-projection directly.
* **Why it wins on H200.** Whole-line gathers double the useful bytes per DRAM burst
  pair, and enough in-flight gathers per SM (8 × 16 KB) cover the ~0.8 µs latency.
  wgmma FP8 keeps verify compute at <40% of the time.
* **Estimated gain.** full B=32 decode 345 → ~180 µs and verify 395 → ~220 µs (est.),
  which is +6–9% on the decode-32/verify-32 loads.
* **Risk.** Moderate. It needs CUDA/CuTe (Triton's wgmma path with gathered operands
  was the reason my L=8 variant lost).

### Idea 4 (the biggest, combining the above): a decode-layer megakernel with phase overlap

A persistent kernel (132 CTAs × 1–2 per SM) with a device-side schedule:
qkv GEMV → norm/RoPE/KV-store → split-KV attention → combine+gate → o GEMV → boundary →
{router ∥ pooler} → experts → unpooler → boundary. Phases synchronize through global
flags (per-tile counters) instead of kernel boundaries, and every phase's *weights* start
streaming (TMA prefetch to shared memory or L2) while the previous phase finishes. This
removes ~35 launch tails (~40 µs at T=1, measured), and it is the only way to approach
the 40 µs floor at decode-1. Est.: decode-1 156 → ~70 µs. Risk: high engineering effort;
deadlock-free grid sync under CUDA graphs needs cooperative-launch semantics or careful
co-residency guarantees.

---

## Scored submit

`okc submit submission.py` (= v5): job **6501bf63c08b48dd**, passed 13/13, **1.097x
(95% CI 1.080–1.113)**.

| Load | Sliding: SGLang / mine (µs) | Full: SGLang / mine (µs) |
|---|---|---|
| decode 1 | 174.4 / 158.3 (1.10x) | 180.6 / 171.5 (1.05x) |
| decode 16 | 265.7 / 250.3 (1.06x) | 435.2 / 412.8 (1.05x) |
| decode 32 | 257.4 / 242.6 (1.06x) | 608.2 / 559.9 (1.09x) |
| verify 1 | 262.8 / 245.5 (1.07x) | 272.8 / 269.3 (1.01x) |
| verify 16 | 522.8 / 431.5 (1.21x) | 681.1 / 598.7 (1.14x) |
| verify 32 | 723.1 / 630.4 (1.15x) | 1071.8 / 986.0 (1.09x) |

Leaderboard context at the v5 submit: #1 was about 2.38x. This entry was a correct,
reviewed-clean 1.10x. The rest of the gap was mostly ideas 2–4 above plus a CUDA MoE kernel.

---

## v6: Track-1 fused MoE integrated into the whole layer

v6 implements the highest-confidence item from the v5 profile. It embeds the Track-1
WP=6 CUDA expert kernels and uses them for `T>=128`, while preserving Marlin for small
loads. No host read of the dynamic `local_start` is needed: the layer's existing Triton
mapping produces local IDs (`0..31`, `-1` otherwise), and the CUDA path consumes them with
a constant zero local base. A prepare-time comparison against Marlin measured relative
L2 error `5.901e-3`; all 13 judge tests passed.

| Version | Job | Mode | Speedup | Score time | Result |
|---|---|---|---:|---:|---|
| v5 | 847f09c9413f4dd7 | check | 1.100x | 20.565 ms | prior reference |
| v6 | 64b21e403afd4931 | check | 1.136x | 19.915 ms | 13/13 pass |
| v6 | e599e9c37ed040a3 | submit | 1.127x | 20.078 ms | 13/13 pass, CI 1.110-1.144 |

The check profile reduced the T=128 expert phase from 281 us in v5 to 244 us (two fused
GEMMs; finalization is ~2 us), and the T=256 phase from 445 us to 421 us. End-to-end,
verify-16 sliding/full improved from 1.219x/1.157x to 1.384x/1.253x in the matched checks.
The scored live decode standing after submission was #751 at 1.127x; pjmullin3 led at
2.399x. A small-T expansion was not checked: standalone measurements predict a useful
T=8 gain but neutral/negative T=16/32 performance, too little aggregate value for the
remaining run budget.

## v7: fast MoE expanded to the 8-token verify shape

v7 enables the already embedded, self-tested Track-1 fused CUDA MoE when `T == 8` as well
as `T >= 128`, while retaining Marlin at T=1/16/32. This was the strongest untested choice
left by v6's standalone measurements and required no change to the numerical algorithm.

| Version | Job | Mode | Speedup | Score time | Result |
|---|---|---|---:|---:|---|
| v7 | `02a4f7b9f5d34b7a` | check | 1.1718x | 19.303 ms | 13/13 pass |
| v7 / current `submission.py` | `cf2294f258114dc5` | submit | **1.1780x** | **19.202 ms** | 13/13 pass, CI 1.1603-1.1957 |

On the check run, verify-1 sliding/full improved to 1.411x/1.321x. The old Marlin expert
phase was about 116-117 us; the two fused CUDA GEMMs totalled about 63 us on the check
(40 + 23 us). The scored run varied to roughly 74-90 us depending on the layer/input,
but still produced a clear end-to-end win. The remaining profile bottlenecks are:

- decode T=16/32: Marlin remains about 79-114 us and is the largest non-attention phase;
- verify T=128/256: the fused MoE is still about 250-425 us;
- full-context attention: about 179 us at B=16 decode and 375 us at B=32 verify;
- decode-1: roughly 30 small kernels and two dense split-K GEMVs dominate after the MoE.

Public-source research did not reveal the rank-1 private submission. The most relevant
upstream designs are CUTLASS/CuTe's Hopper persistent grouped GEMM (TMA producer warpgroup,
WGMMA consumer warpgroup, persistent group-aware scheduler), FlashAttention-3's explicit
split-KV sweep and reusable scheduler metadata, and SGLang's own Muse roadmap calling out
MegaMoE plus decode megakernels. These corroborate the next implementation order: (1) tune
or extend the fused MoE to T=16/32, (2) port a warp-specialized WGMMA producer/consumer
mainloop for the expert kernels, and (3) build a page-size-1 FP8 split-KV verify path.

## v8: fast MoE expanded to the 16-token decode shape

v8 also enables the fused CUDA MoE at `T == 16`. This contradicted the earlier noisy
standalone estimate, so it was validated end-to-end before promotion. T=32 deliberately
remains on Marlin because its standalone fused timing was about 137 us versus roughly
113 us for Marlin inside the layer.

| Version | Job | Mode | Speedup | Score time | Result |
|---|---|---|---:|---:|---|
| v8 | `6df54644ee034846` | check | 1.2210x | 18.526 ms | 13/13 pass |
| v8 / current `submission.py` | `bbb6e0110f24457f` | submit | **1.2326x** | **18.352 ms** | 13/13 pass, CI 1.2141-1.2511 |

In the check, decode-16 sliding/full fell to 179.25/346.79 us, corresponding to
1.460x/1.246x. Relative to the v6 scored starting point, the two small-token MoE routing
changes improved the whole-layer score from 20.078 ms (1.127x) to 18.352 ms (1.233x), a
further 8.6% reduction in our execution time without changing output tolerances. Both
reserved decode submission slots for 2026-10-28 are now used; further candidates must be
check-only until the next daily reset.

## v9: single-token-thread fused MoE at T=32

v9 extends the fused path to T=32 but changes its compile-time token grouping from NT=2
to NT=1. This isolates the occupancy choice that made the earlier standalone T=32 result
look uncompetitive. Job `e96574729e4c4561` passed 13/13 at **1.2378x** (18.274 ms,
CI 1.2193-1.2564). Against the matched v8 check, decode-32 sliding/full improved from
246.71/566.50 us to 216.32/539.98 us. The expert kernels totalled about 115 us in the
sliding profile and 97 us in the full profile. Aggregate execution improved 1.36%, which
is useful but overlaps normal run variance; it remains check-only until the next daily
submission reset and another controlled measurement.

## v10: two queued CTA waves for T=32

v10 doubles only the T=32 launch grid. Although the second CTA wave cannot be resident
simultaneously, it shortens each persistent CTA's item loop and reduces load-balance tails.
Job `d29145d81a964459` passed 13/13 at **1.2501x** (18.094 ms, CI 1.2314-1.2689).
The T=32 expert pair fell from about 115/97 us in v9's sliding/full profiles to about
94/89 us. This is the current best check-only candidate; the next controlled test applies
the same grid-depth hypothesis to T=128/256, whose expert kernels dominate verify loads.

v11 tested that expansion and rejected it. Job `d3fb664609ba41e6` passed 13/13 but
regressed to 1.2284x (18.415 ms). At T=128 the two expert kernels rose to about 293 us
in the sliding profile, versus about 261 us with the original one-wave grid in v10.
T=256 also slowed. Large-token work already amortizes the routing setup and needs the
maximum resident grid for bandwidth; the T=32 two-wave special case is retained alone.

## v12: wider split-KV attention search

v12 adds custom attention candidates targeting 1,584 CTAs with BN=64 and BN=128. The
prepare-time graph tuner selected a new path only for B=16, L=1, full context:
BN=128/target=1584 measured 186.9 us versus about 190 us for target=1056. Every other
shape retained its old custom or FA3 choice. Job `ce61a83b4ddd450b` passed 13/13 at
**1.2505x** (18.089 ms, CI 1.2318-1.2693), essentially tied with v10 but marginally best.

v13 increased only the T=32 grid from two to three queued waves. Job
`ef91810898f24de2` passed 13/13 but regressed to 1.2315x (18.368 ms). Together with v9
and v10, this brackets the tested scheduling optimum at two waves; three waves add too
much repeated routing/setup and over-partition the persistent item loop.

v14 tested two queued waves at T=16 while retaining the T=32 two-wave path. Job
`4249c27a81214062` passed 13/13 but regressed to 1.2418x (18.216 ms) versus v12's
18.089 ms. T=16 returned to one wave. The final retained check-only candidate is v12:
T=32/NT1/two-wave scheduling plus the 1,584-CTA attention candidates. The 2026-10-28
decode runbook budget is closed at eight checks and two submits.
