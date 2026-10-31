# H200 decode kernel notes

Short, measured notes on memory-bound and latency-bound LLM decode kernels on an NVIDIA H200 SXM.

## Abstract

These notes come from building the decode-time kernels of one transformer layer on a single NVIDIA H200 SXM: dense BF16 projections, a 256-expert NVFP4 mixture-of-experts layer, and paged attention over an FP8 KV cache. Each note asks one question and states how it was measured: CUDA-graph replays with a cold L2, medians, and in-kernel `%globaltimer` timelines. It then reports the data, including the designs that did not work. Four themes come out of it:
- **Measurement.** Flushing L2 by writing a buffer inflated cold-cache timings by up to 6 µs over a read flush, in proportion to the bytes the kernel reads, and that was enough to choose the wrong kernel.
- **The memory system.** Per-SM streaming topped out near 32 GB/s. A second producer per SM mattered more than a deeper ring. The first data of a cold kernel takes 1.2–2.3 µs.
- **4-bit MoE consumers.** At 4.2 TB/s on 132 SMs, the dequant + MMA loop must spend at most about 3.9 issue slots per weight. Mine spent 3.0 per weight at 62% issue efficiency, so it could not keep up at large batch.
- **Graph-level costs.** An extra CUDA-graph node costs about 1.1 µs. PDL hides launches, not tails. Running independent weight streaming beside latency-bound attention made the layer slower.

## Hardware and method in brief

- **GPU.** One H200 SXM per run: sm_90a, 132 SMs, 141 GB HBM3e at 4.8 TB/s nominal, 50 MB L2. CUDA 13 toolchain; kernels in CUDA C++ with inline PTX. Clocks were neither locked nor recorded. Where a cycle budget is needed I assume 1.75 GHz and say so.
- **Timing.**
  - Each shape is captured as a CUDA graph and replayed. Before every timed replay, L2 is flushed by reading a buffer larger than L2. I report medians.
  - An independent harness that pairs replays with a reference implementation supplies the ground-truth times. My own in-process timer is used for A/B comparisons inside one run.
  - Per-CTA `%globaltimer` stamps show where the time goes inside a kernel.
- **Labels.** Every number is *measured* (printed by one of those timers), *derived* (arithmetic on measured numbers, shown), or an *estimate/inference* (labelled as such). Cross-run comparisons are flagged.

Details: [notes/00-method.md](notes/00-method.md).

## Notes

| # | Note | Finding |
|---|---|---|
| 00 | [Hardware, workloads and method](notes/00-method.md) | The three timers, the workloads (N×K list, MoE and attention shapes) and the conventions every note uses. |
| 01 | [Write-flush vs read-flush](notes/01-l2-flush-bias.md) | With a `zero_()` flush, my timer's offset from an independent read-flush harness ranged from 3.9 to 9.6 µs, depending on the kernel and growing with the bytes read, and it reversed a kernel choice by ~4 µs. A read flush gave a flat 3.0–3.6 µs offset. |
| 02 | [Bytes in flight per SM](notes/02-bytes-in-flight-per-sm.md) | One producer per SM streamed 3.3–3.4 TB/s whether 72 or 144 KiB were in flight. Two producers reached 4.2–4.3 TB/s (32 GB/s per SM). The first stage lands 1.2–2.3 µs after a cold start. |
| 03 | [Streaming NVFP4 MoE weights](notes/03-nvfp4-moe-streaming.md) | At 1–32 tokens the math adds ~21 µs per call, not per byte. At 128–256 tokens the consumers are issue-bound: 3.0 instructions per weight at 62% issue efficiency, where ~76% is needed. Failed alternatives are listed with numbers. |
| 04 | [Kernel boundaries and PDL](notes/04-kernel-boundaries-and-pdl.md) | One more graph node costs 1.09 µs (median, n = 1,278). PDL cut a boundary to 0.26–0.51 µs but cannot hide the primary's tail, and it slowed one kernel pair by 2 µs. Concurrent weight streaming during latency-bound attention was slower in 7 of 8 layer groups. |
| 05 | [Small measured facts](notes/05-small-facts.md) | Clusters of 4 at 1 CTA/SM ran in two waves (30 of 32 resident). FP8 `mma.sync` gives no FP8 rate on Hopper. A serialized load → atomic chain cost up to 155 µs. CTA end-time spread is the largest identified small-batch MoE loss. L2 prefetch did not help in four tries. |

## Data and figures

- [`data/`](data) holds the tables behind the notes as CSV.
- [`figures/make_figures.py`](figures/make_figures.py) regenerates every figure from them: `python figures/make_figures.py` (needs NumPy and Matplotlib).
- The kernels themselves are not part of this repository. The notes describe their structure where it matters for the result.

## Scope

- One GPU model, measured over three days in October 2026, on the shapes listed in note 00. Nothing here is claimed for other GPUs, clocks, or shapes.
- Most comparisons are inside one run. Where two runs are compared, the text says so.

## How to cite

```bibtex
@misc{mrnirmata2026h200decode,
  author       = {MrNirmata},
  title        = {{H200} Decode Kernel Notes: Measured Notes on Memory- and Latency-Bound {LLM} Decode Kernels},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/MrNirmata/h200-decode-notes}},
  note         = {Research notes}
}
```

## License

Text, data and figures: [CC BY 4.0](LICENSE).
