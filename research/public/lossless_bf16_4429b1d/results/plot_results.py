"""Plots for results/ (needs matplotlib):  python results/plot_results.py"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "R2"   # the run shown in the README table
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORDER = [("qkv", 2304, 8192), ("attn_out", 8192, 1024), ("moe_in", 4096, 8192), ("moe_out", 8192, 4096),
         ("moe_out_tp8", 1024, 4096)]


def read(name):
    with open(HERE / name) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def speedup_plot():
    same = {(r["shape"], int(r["m"])): r for r in read("h200_same_process.csv") if r["run"] == RUN}
    paired = {(r["shape"], int(r["m"])): r for r in read("h200_paired.csv") if r["run"] == RUN}
    rows = [(s, n, k, m) for s, n, k in ORDER for m in (1, 8, 16, 32) if (s, m) in same]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, (s, n, k, m) in zip(ys, rows):
        a = float(same[(s, m)]["speedup_vs_cublas"])
        ax.plot([1.0, a], [y, y], color=GRID, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if (s, m) in paired:
            b = float(paired[(s, m)]["median_paired_ratio"])
            ax.scatter([b], [y], s=52, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=4)
    ax.axvline(1.0, color=INK2, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}×{k}   m={m:<2d}" for s, n, k, m in rows], fontfamily="monospace", fontsize=8.5,
                       color=INK)
    ax.set_xlim(0.7, 1.9)
    ax.set_xlabel("speedup over cuBLAS (×)", color=INK2, fontsize=9)
    ax.text(0.985, ys[-1] - 0.9, "cuBLAS faster", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(1.015, ys[-1] - 0.9, "12-bit faster", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_ylim(ys[-1] - 1.5, ys[0] + 0.6)
    h1 = ax.scatter([], [], s=46, color=BLUE, label="same process: medians of 24 replays, launch cost included")
    h2 = ax.scatter([], [], s=52, facecolor="none", edgecolor=ORANGE, linewidth=2, label="paired harness: 60 alternating replays, launch cost removed")
    leg = ax.legend(handles=[h1, h2], loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
                    labelcolor=INK)
    leg.set_in_layout(True)
    fig.suptitle("12-bit kernels vs cuBLAS on H200, cold L2, CUDA graphs", x=0.02, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "speedup.png", facecolor=SURFACE)
    plt.close(fig)


def timeline_plot():
    rows = [r for r in read("h200_splitk_timeline.csv") if r["run"] == RUN]
    key = {s: i for i, (s, _, _) in enumerate(ORDER)}
    rows.sort(key=lambda r: (key[r["shape"]], int(r["m"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        start = float(r["first_stage_mean_us"])
        stream = float(r["loop_end_max_us"]) - start
        tail = float(r["finalize_end_us"]) - float(r["loop_end_max_us"])
        left = 0.0
        for width, color in ((start, BLUE), (stream, ORANGE), (tail, AQUA)):
            ax.barh(y, width - 0.06, left=left, height=0.62, color=color, linewidth=0)
            left += width
        ax.text(left + 0.25, y, f"{left:.1f} µs   {float(r['per_sm_mean_GBps']):.0f} GB/s per SM", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['n']}×{r['k']}   m={int(r['m']):<2d}" for r in rows], fontfamily="monospace",
                       fontsize=8.5, color=INK)
    ax.set_xlim(0, 30)
    ax.set_xlabel("µs from the first CTA start (one cold-L2 replay, %globaltimer)", color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["start → first stage in shared memory", "streaming (until the last CTA's loop ends)",
                        "partial stores + PDL hand-off + finalize"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Split-K kernel: where the time goes (H200)", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "splitk_timeline.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    speedup_plot()
    timeline_plot()
