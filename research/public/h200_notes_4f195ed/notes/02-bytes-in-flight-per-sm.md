# 02. Bytes in flight are necessary, not sufficient: per-SM streaming on H200

## Question

A decode kernel that streams weights is limited by how fast each SM pulls bytes from HBM. Little's law says the rate per SM is the bytes in flight per SM divided by the loaded latency. On an H200, how much does a deeper ring buy? How much do more CTAs per SM buy? And what does the very first request cost?

## Setup

Three streaming kernels, all with weights arriving by `cp.async.bulk` into shared-memory rings guarded by mbarriers.

- **Persistent MoE kernel.** One or two CTAs per SM. Each CTA has one weight-producer warp that issues one 9216-byte bulk copy per stage and never waits on data dependencies. A separate activation-producer warp handles activations.
  - Rings of 8, 12 or 16 weight slots per SM: 72, 108 or 144 KiB.
  - Consumers are either *no-op* (they release each slot as soon as it lands) or *spin-700* (they hold each slot about 700 ns, a stand-in for real math).
- **Two-kernel MoE design** (the two-kernel form of note 03's design). 4 CTAs per SM × 4 stages of 9216 bytes = 144 KiB per SM, with no-op consumers.
- **One-CTA-per-SM split-K projection.** One CTA per SM (132 CTAs), 8 consumer warps plus one producer warp. Each CTA's ring holds 205–215 kB. Each CTA has only 98–393 kB to move in total, so nearly all of its requests are in flight from the start.

## Method

- **Rates.** For the two MoE kernels I timed the CUDA graph (tuner timer, read flush) at four problem sizes and fitted `time = intercept + bytes / rate`. The *marginal rate* is the slope's inverse. It excludes fixed costs (launch, routing, first-stage latency, tail), which land in the intercept. Sizes: 85, 198, 340 and 425 MB for the persistent kernel; 81, 216, 243 and 405 MiB for the two-kernel design.
- **Timelines.** For the split-K projection I read per-CTA `%globaltimer` stamps from one cold-L2 replay: CTA start, first stage ready, loop end. "Per-SM GB/s" is the CTA's bytes divided by (loop end − CTA start).

## Data

**Table 1. Marginal streaming rate vs CTAs per SM and weight bytes in flight per SM.**

| Design | CTAs/SM | Weight ring per SM | Consumers | Marginal rate (TB/s) | Per SM (GB/s, derived = rate/132) | Intercept (µs) |
|---|---|---|---|---|---|---|
| persistent | 1 | 72 KiB | no-op | 3.41 | 25.8 | 7.1 |
| persistent | 1 | 108 KiB | no-op | 3.32 | 25.2 | 7.3 |
| persistent | 1 | 144 KiB | no-op | 3.42 | 25.9 | 8.7 |
| persistent | 2 | 108 KiB | no-op | 4.23 | 32.0 | 11.4 |
| persistent | 2 | 144 KiB | no-op | 4.30 | 32.6 | 12.9 |
| two-kernel | 4 | 144 KiB | no-op | 4.13–4.29 (4 runs) | 31.3–32.5 | 12.6–13.5 |
| persistent | 1 | 72 / 108 / 144 KiB | spin-700 | 2.94 / 2.95 / 3.04 | 22.3–23.0 | 6.8–9.4 |
| persistent | 2 | 108 / 144 KiB | spin-700 | 3.91 / 3.87 | 29.6 / 29.3 | 16.1 / 16.8 |

A single-phase fused MoE kernel (note 03, failed alternatives) with 2 CTAs per SM and, by design, about the same 144 KiB per SM reached 3.24–3.69 TB/s with no-op consumers.

**Table 2. One CTA per SM, split-K projection: per-CTA timeline (one cold replay each).**

| Layer, m | Bytes per CTA (kB) | First stage ready, mean / max (µs) | Loop end, mean / max (µs) | Per-SM GB/s, mean / min | Aggregate (TB/s) | First stage as share of loop time (derived) | GB/s after the first stage (derived) |
|---|---|---|---|---|---|---|---|
| qkv+gate, 8 | 214 | 1.83 / 2.30 | 9.71 / 10.50 | 22.2 / 20.3 | 2.70 | 19% | 27 |
| qkv+gate, 16 | 214 | 1.57 / 2.05 | 9.00 / 9.73 | 23.8 / 21.9 | 2.91 | 17% | 29 |
| qkv+gate, 32 | 221 | 1.94 / 2.30 | 9.97 / 13.06 | 22.2 / 16.9 | 2.17 | 19% | 28 |
| o-proj, 8 | 98 | 2.16 / 2.82 | 4.68 / 5.12 | 21.1 / 19.2 | 2.46 | 46% | 39 |
| o-proj, 16 | 98 | 1.85 / 2.30 | 5.36 / 5.63 | 19.2 / 18.3 | 2.23 | 35% | 28 |
| o-proj, 32 | 98 | 1.83 / 2.05 | 5.80 / 6.14 | 17.0 / 16.0 | 2.05 | 32% | 25 |
| in-proj, 16 | 393 | 1.77 / 2.56 | 14.77 / 15.36 | 26.9 / 25.6 | 3.28 | 12% | 30 |
| in-proj, 32 | 393 | 1.77 / 2.30 | 16.66 / 19.46 | 23.7 / 20.2 | 2.59 | 11% | 26 |
| out-proj, 16 | 393 | 2.27 / 3.07 | 14.13 / 15.62 | 28.3 / 25.6 | 3.22 | 16% | 33 |
| out-proj, 32 | 382 | 1.96 / 2.82 | 17.15 / 20.22 | 22.3 / 18.9 | 2.49 | 11% | 25 |

Derived columns use the means: share = first / loop end; rate after the first stage = bytes / (loop end − first). Both are approximate because they combine means of different CTAs.

**Table 3. Latency of the first data, cold L2, by design.**

| Design | What is measured | Value |
|---|---|---|
| 1 CTA/SM split-K projection (Table 2) | CTA start → first stage ready, mean | 1.57–2.27 µs (max 2.05–3.07) |
| two-kernel MoE, 4 CTAs/SM, 1 token | routing done → first stage ready, mean | 1.23–1.56 µs (4 runs) |
| two-kernel MoE, 256 tokens | same | 2.82–3.01 µs (4 runs) |
| single-phase fused MoE kernel, activations issued *after* the ring's weights | first copy issued → first stage ready, p50 | 3.7–4.6 µs |

Data: [`data/stream_rate_vs_inflight.csv`](../data/stream_rate_vs_inflight.csv), [`data/one_cta_per_sm_timeline.csv`](../data/one_cta_per_sm_timeline.csv). Figure: [`figures/fig2_rate_vs_inflight.png`](../figures/fig2_rate_vs_inflight.png).

![Marginal streaming rate vs weight bytes in flight per SM, by CTAs per SM](../figures/fig2_rate_vs_inflight.png)

## Finding

1. **At one CTA per SM, a deeper ring did nothing.** With no-op consumers the marginal rate was 3.32–3.42 TB/s (25–26 GB/s per SM) at 72, 108 and 144 KiB in flight. If latency stayed fixed, Little's law would have the 144 KiB ring stream at twice the rate of the 72 KiB one. It did not, so something other than bytes in flight capped a single producer per SM.
2. **Two CTAs per SM reached 4.23–4.30 TB/s** (32.0–32.6 GB/s per SM) with 108–144 KiB in flight, and the 4-CTA two-kernel design reached 4.13–4.29 TB/s with 144 KiB. That is 86–90% of the 4.8 TB/s nominal. It is the highest marginal rate I measured in any design. No fit reached the 4.6 TB/s (35 GB/s per SM) that I had used as a planning figure.
3. **At the same 144 KiB per SM, the rate depended on how the bytes were issued.** It was 3.42 TB/s from one producer per SM, 4.30 from two, and 4.13–4.29 from four. The single-phase fused kernel, also at 2 CTAs/SM, managed only 3.24–3.69, so the producer's per-stage work matters as well as the number of producers.
4. **Holding slots costs bandwidth even when the ring is deep.** Consumers that held each slot about 700 ns cut the marginal rate by 7.6–14% and, at 2 CTAs/SM, raised the intercept by 3.9–4.7 µs.
5. **The first request costs 1.2–2.3 µs on a cold start**, and much more if it queues behind a burst. In the short one-CTA-per-SM projection kernels, waiting for the first stage took 11–46% of each CTA's streaming time. That, more than the ring, explains the low end-to-end 17–28 GB/s per SM: after the first stage the same CTAs streamed at about 25–39 GB/s (derived).
6. **Queueing the activations behind a full ring of weights delayed the first usable stage to 3.7–4.6 µs.** This is an inference that fits the numbers: when every CTA issues its whole ring at t = 0, a request issued afterwards waits for roughly that burst to drain at about 4.2 TB/s.

**Limits.**
- The marginal rates are 4-point fits on MoE-shaped traffic: 9216-byte bulk copies of mostly distinct experts, with some activation traffic. The four sizes also differ in routing.
- I did not test 2 CTAs per SM with less than 108 KiB per SM; that configuration did not fit the 2-CTA occupancy target.
- The cause in finding 1 is not isolated. A single producer warp waits for a free slot, arms the barrier and issues the copy, and that per-stage interval would cap the rate at about one 9216-byte stage per 360 ns per SM (derived from 25.8 GB/s). This is consistent with the data but not demonstrated.

## Implications

- **Use at least two independent producers per SM** (two CTAs, or two producer warps with separate rings) before deepening rings. In these kernels the second producer was worth 26%; doubling the ring at one producer was worth 0%.
- **About 108 KiB per SM was enough at 2 CTAs/SM; 144 KiB added under 2%.** By Little's law, 108 KiB at 32 GB/s per SM corresponds to about 3.5 µs of loaded latency (derived: 110,592 B ÷ 32.0 GB/s), well above the 1.2–2.3 µs cold first-request latency. So this depth leaves headroom.
- **For kernels that move under ~400 kB per SM, first-request latency is the budget item.** A 2 µs first stage is 20–45% of a 5–10 µs kernel. Hide it behind the previous kernel, for example by issuing weight loads in a PDL prologue (note 04), rather than by deepening the ring.
- **Issue the first stage's activations with or before the first weights.** Never queue them behind a ring-sized burst.
- **Plan with about 32 GB/s per SM, not the nominal 36 GB/s (4.8 TB/s ÷ 132).** Every SM is capped, so the busiest SM sets the kernel time, and work must be split evenly across all 132 SMs.

## Related work

- V. Volkov, *Understanding Latency Hiding on GPUs*, UC Berkeley, 2016. Little's law applied to GPU memory throughput: [tech report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html).
- W. Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, 2024. Hopper latency and throughput microbenchmarks: [arXiv:2402.13499](https://arxiv.org/abs/2402.13499).
- Chips and Cheese, *Nvidia's H100: Funny L2, and Tons of Bandwidth*, 2023. The two-partition L2 and its latency: [article](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth).
- S. Manjunath and R. Ramachandra, *Concurrency Response of Plain Global Loads on the NVIDIA H100*, 2026. Plain-load bandwidth vs per-thread loads in flight: [arXiv:2608.15764](https://arxiv.org/abs/2608.15764).
- W. Guo, T. Zadouri, T. Dao, *Getting Memory-bound Kernels to Speed-of-Light*. About 90% of peak DRAM bandwidth on H100: [blog](https://sysml.cs.princeton.edu/blogs/memory-bound-kernels.html).
