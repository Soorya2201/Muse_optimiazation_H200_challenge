# `muse_attention_decode` optimization record

The current verified local submission is `submission.py` = `submission_v15_vjeux_f35a41e.py`.

## Oct29 public-source reproduction

Exact Apache-2.0 vjeux attention source, identical to ruwwww's public file, passed
`385a2d2e58cd476d` at **2.597212x**,1450.132us,CI[2.558253,2.647694],all16 tests and12
scored shapes passing. The source is promoted byte-for-byte. Scored confirmation
`30c669b9702d434f` passes all16 tests and12timings at **2.603505x**,1446.627us,
CI[2.564452,2.659499],rank#90/1066,overall#354/1602. Both profiles/logs were read.
One of eight checks and one of two submits used today.

The largest architecture changes are complete fused verify execution with Hopper WGMMA,
flat work partitioning across ragged full-context requests, and efficient cooperative or
last-CTA finalization. Every timed shape profiles one launch. B32 full verify182.84us
versus old390.08us; B32 full decode166.16us versus old240.84us. All visible keys are processed,
and the judge verifies both outputs and new FP8 cache entries, including1M-context tests.

The complete above/below account screen covers1064 ranked users; source hashes, licenses,
and specific stronger/weaker repository lessons are in PUBLIC_RESEARCH.md and its JSON ledger.

## Oct29 second-cycle isolated producer prefetch

`submission_v16_l2_lookahead.py` ports only the c1809247 `OKC_PFL2=1` hook into the
current parent's flat L1 producer. Check `5213b424b3eb47f5` passes all16tests/12timings
at **2.482162x**,1517.347us,CI[2.444929,2.519394],below scored2.603505x.
B16/B32 full decode paired times regress86.29/164.98->92.99/217.65us;native profiles
86.5/165.0->94.9/218.2us. Other paths remain close. Reject;parent stays promoted.

All logs/profiles were read. Source isolation/syntax pass;the inherited upstream trailing
whitespace is preserved in the archived candidate. Current budget2/8checks,1/2submits.
Further operatorstack/tommyyliu source sections and receipts were reviewed:head packing,
warp specialization and hierarchical merge overlap our stronger parent. No unmeasured
whole-file transfer is justified. See PUBLIC_RESEARCH.md for exact commits and hashes.

Do not blindly sweep more PFL2 distances after this regression. The next meaningful
experiment should expose actual parent producer/consumer/merge phase costs or reduce a
measured small-request merge dependency while preserving complete attention and precision.

## Historical v12/v13 result and bottleneck

Scored job `3135d826e3994f11` passed at **1.2954x** (SGLang 3,766.3 us, ours
2,907.4 us; 95% CI 1.2760-1.3188). The best local candidate is v13 at
**1.3116x** (`ccaa5314524e4001`, 2,871.6 us), but its 1.25% gain over the scored
point did not clear the strict resubmit threshold. After the submit the live rank
was #761/936; first place was 3.0316x.

The custom fused decode kernel wins on all six decode shapes: 1.39-1.85x. It is one launch
that performs QK norm, optional RoPE, FP8 cache store, split-KV attention, cluster/global
merge, output norm, and gate. Full-context batch-32 decode now uses C2/NC4 geometry and
takes about 233 us in v13, down from v6's 278 us. Its TASK floor is 126 us, so it reaches
about 54% of the ideal roofline; random
128-byte page-1 gathers make the 4.8 TB/s floor optimistic.

Verify is the remaining bottleneck. At B=1 sliding verify, the custom fused kernel takes about
12.7 us and gives 1.40x. At B=16/32, v4's custom mode-18 path took about 31/59 us and lost to
FA3 because it processed verify tokens as separate virtual requests and reread the window.
v6 therefore selects the custom path only for B=1 sliding verify and SGLang/FA3 for all other
verify shapes. Full-context B=32 verify is about 391 us versus a 127 us ideal floor; FA3's
main attention kernel is over 95% of profiled time.

## Optimization inventory

| Optimization | Status | Result |
|---|---|---|
| Fuse QK norm, RoPE, cache write, attention, output norm and gate | Tried, retained for custom shapes | Removes the baseline's multi-launch chain. |
| FP8 E4M3 queries against the existing FP8 KV cache | Tried, retained | Allowed by the rules and passes all 16 tests. |
| Split KV along context to fill 132 SMs | Tried, retained | Essential for batch-1 full attention. |
| Thread-block clusters and on-chip partial merge | Tried, retained | Avoids a separate global combine when one cluster suffices. |
| Global last-arriver merge for multiple clusters | Tried, retained | Covers 80K and 1M contexts with graph-safe counters. |
| Deep `cp.async` gather pipeline | Tried, retained | Keeps scattered page-1 K/V requests in flight. |
| Pair both local KV heads in one CTA | Tried, retained | Uses each 128-byte cache row as a full line. |
| Shape-specific custom/FA3 selection | Tried in v4-v6, retained | v6 improved scored speedup from 1.2023x to 1.2491x. |
| Treat eight verify tokens as separate virtual requests | Tried in v4, rejected for B>=16 | Re-reads the same window; only retained at B=1 where fusion wins. |
| Pack 8 tokens x 8 query heads into one 64-row warp-MMA tile | Tried in v7-v10, rejected | Reads K/V once, but 32-49 us custom timing still lost to FA3's 25-30 us chain. |
| Use C2 plus global merge for batched full decode | Tried in v12-v14, retained in v13 | Reduced B16/B32 from about 146/280 to 124/233 us paired. |
| FP8 WGMMA for QK | Not yet implemented | Estimated 10-25% on verify compute; precision must be rechecked. |
| DSMEM split-KV merge across clusters | Not yet implemented | Clusters cannot directly span GPCs; needs hierarchical design. |

## Version and OKC history

| Version | Job | Result | Notes |
|---|---|---:|---|
| v4 | `de9fbaaf2108416b` check | 1.1952x | Recovered custom cluster kernel. |
| v4 / old `submission.py` | `41a3fc2bdbe84169` submit | 1.2023x | First scored attention result. |
| v5 | `6f6fa502d1df4355` check | 1.2080x | FA3 for every verify shape; lost the B=1 custom win. |
| v6 | `637b6a93b70c4f1a` check | 1.2488x | Hybrid: custom B=1 sliding verify, FA3 otherwise. |
| v6 / old `submission.py` | `b38b60919dfb42e6` submit | **1.2491x** | Prior scored hybrid result. |
| v7 | `4cc63346dc924fdd` check | 1.2091x | Existing L8 kernel correctly reused each K/V tile across 64 query rows, but the all-resident planner forced C2 with 8/4 cross-cluster merges; B16/B32 sliding verify regressed to 38.95/49.26 us. |
| v8 | `f8057bf43de7434a` check | 1.1911x | Natural C16/C8 clusters avoided global combine but synchronization and queued tails worsened B16/B32 sliding verify to 48.67/56.12 us; rejected. |
| v9 | `6c982f5b856040ad` check | 1.2237x | One-wave C8/C4 plans reduced B16 to 35.07 us paired/32.1 us profiled, but B32 remained 49.84/46.9 us; both still lost to v6 FA3 at 29.33/34.61 us. |
| v10 | `956826766cad4fed` check | 1.2177x | Replacing C8/C4 with C2 reduced B32's profile to 38.6 us, but B16 rose to 34.2 us; paired 39.84/48.29 us confirmed merge overhead still leaves both behind FA3. |
| v11 | `ba1b8a40f6bd454c` check | 1.1848x | Five cp.async stages globally forced C2 plans and hurt sliding/B1, but improved B16/B32 full decode paired times by 17.45/42.57 us. The next variant specializes NST5 to those two shapes. |
| v12 | `50948bd6edc344dc` check | **1.3012x** | Hybrid NST4/NST5 retained every v6 dispatch and used five stages only for B16/B32 full decode, cutting them to 129.49/237.95 us paired. CI 1.2816-1.3277 clears v6 and qualifies for submit. |
| v12 / scored submission | `3135d826e3994f11` submit | **1.2954x** | New live best: 2907.4 us, CI 1.2760-1.3188; 3.7% higher speedup than v6's 1.2491x. |
| v13 | `ccaa5314524e4001` check | **1.3116x** | NST4 plus C2 improved B16/B32 full decode to 123.86/232.86 us paired. This isolates C2 geometry as the main gain; the 1.25% aggregate lift over scored v12 does not clear the submit threshold. |
| v14 | `d461fc4f71a74060` check | 1.3061x | NST3 plus C2 regressed B16/B32 full decode to 124.78/236.62 us paired. Four stages are retained; reserved budget closed at 8 checks/1 submit. |

## Novel next ideas

### 1. One-read 64-row verify tile

Existing: mode-18 turns each verify token into a virtual request, so eight tokens reread almost
the same K/V window. Proposed: pack `8 tokens x 8 query heads per KV head = 64` query rows into
one Hopper WGMMA M=64 tile. Load each K/V page once, update 64 online-softmax states, and apply
the intra-block causal mask per row. Estimated gain: **1.5-2.5x on sliding verify and 15-35% on
full verify**. Risk: high register/shared-memory pressure and softmax exponent throughput. Start
with B=1 sliding, then split context across clusters for full attention.

### 2. FP8 QK WGMMA plus BF16/FP16 PV

Existing: the custom kernel converts cache values around warp-level MMA. Proposed: quantize the
normalized query to E4M3 (explicitly allowed), use E4M3 x E4M3 WGMMA for QK at twice BF16 tensor
throughput, and keep probabilities/value accumulation at higher precision. Estimated gain:
**10-25% on verify**; decode is primarily memory-bound. Risk: medium-high because output error is
data-dependent and every test must remain within its query-derived tolerance.

### 3. Persistent page-index producer

Existing: each split independently walks `seq_lens -> page_table -> cache`, paying dependent
latency. Proposed: one producer warp per request gathers a long page-index strip into shared or
cluster DSMEM while consumer warps issue K/V copies; double-buffer indices separately from K/V.
Estimated gain: **5-12% on page-size-1 full attention**. Risk: medium; the producer can become a
serialization point and DSMEM only spans one GPC.

### 4. Cross-layer page-table reuse in the serving patch

Existing: every attention layer rereads identical page-table entries, and the challenge flushes
L2 only before the whole timed layer call. In whole-layer/serving code, build a compact physical-row
schedule once per step and reuse it across the 46 sliding and 16 full layers. Estimated gain:
**3-8% of attention time** plus reduced TLB pressure. Risk: low-medium if the schedule is built as
normal step metadata rather than cached from fresh tensor values inside a kernel submission.
