# Measured data (H200 SXM)

All data come from three runs on one H200 SXM (141 GB, 132 SMs) in October 2026. The three builds contain
byte-identical 12-bit kernels and differ only in code not included here. Each run:
1. packed synthetic N(0, 1/K) BF16 weights, regenerated per run;
2. checked every 12-bit configuration on the GPU (bit-exact one-hot probes, random-activation error < 5·10^-3);
3. timed every configuration against cuBLAS in the same process;
4. ran the paired harness on the configuration it selected.

The README table uses run R2. The CSVs were transcribed by a script from the runs' logs and timing records. The
raw logs are not included because they contain unrelated material.

| file | content |
|---|---|
| `h200_same_process.csv` | per run, shape and m: cuBLAS default (via PyTorch), the fastest 12-bit configuration and how many replays its median is over, the fastest non-default cuBLASLt algorithm and the fastest uncompressed custom kernel when they were among the four fastest candidates, and the kind of the overall fastest candidate |
| `h200_paired.csv` | paired-harness rows where a 12-bit kernel was dispatched: raw medians of 60 replays per side, the subtracted fixed replay cost, the median of paired ratios and its interquartile range |
| `h200_splitk_timeline.csv` | `%globaltimer` timeline of one cold-L2 replay of the split-K kernel per shape and m |
| `escape_counts.csv` | escapes counted at pack time in 15 runs, per shape |
| `speedup.png`, `splitk_timeline.png` | made by `plot_results.py` from the CSVs |

## Method

- **Same process.** One call per CUDA graph. Before every replay, a 64 MiB buffer is read with `sum()`, so the
  weights come from DRAM; then the replay is timed with CUDA events. Candidates are screened with 5 replays, the 8
  fastest (plus cuBLAS) are re-timed with 24, and at m = 1 the top 3 are timed twice more and ranked by the median of
  the three 24-replay medians. Times include the fixed cost of a graph launch.
- **Paired.** cuBLAS and the dispatched kernel take turns on the same fresh inputs, each replay after a read of a
  buffer larger than L2, 60 replays per side. A per-case fixed replay cost (3.6–4.4 µs) is subtracted from both
  sides. The speedup is the median of paired ratios, taken two repetitions at a time with the order swapped.
  `ours_us = cublas_us / median_paired_ratio`. The harness checked every timed output against an fp64 reference.

## Run-to-run variation

Same-process speedups of the 16 (shape, m) points differ by at most 0.035 across R1–R3. Paired ratios differ by at
most 0.13, at 1024×4096 with m = 1: 1.686 in R1 and 1.814 in R2; R3 dispatched an uncompressed kernel there. Their
interquartile ranges are in `h200_paired.csv`.
