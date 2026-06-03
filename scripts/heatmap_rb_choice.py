#!/usr/bin/env python3
"""Plot 2x2 heatmaps showing the (P, n)-cell-optimal radix r and batch
size b for gata (uniform) and gatav (non-uniform).

Output: plots/heatmap_rb_choice.png  (single figure, 2 rows x 2 cols)
        plots/heatmap_rb_choice_gata_r.png   (individual)
        plots/heatmap_rb_choice_gata_b.png
        plots/heatmap_rb_choice_gatav_r.png
        plots/heatmap_rb_choice_gatav_b.png

For each (P, n) cell we compute the best-of-medians (b, r) over the
swept grid, then render two heatmaps per variant:
  * left:  cell colour = best r
  * right: cell colour = best b

Axes:
  * x-axis: process count P (log scale)
  * y-axis: per-peer message size n (log scale, in 8-byte elements)
"""

import os, re, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR  = os.path.join(HERE, "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Format:  [tag] P, n, b, r, time
LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*([\deE+\-.]+)\s*$")

def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "run_*_merged.sh.o"))):
        with open(path) as f:
            for line in f:
                m = LINE_RE.match(line.strip())
                if not m:
                    continue
                tag, P, n, b, r, t = m.groups()
                if tag not in ("gata", "gatav"):
                    continue
                rows.append((tag, int(P), int(n), int(b), int(r), float(t)))
    return pd.DataFrame(rows, columns=["tag", "P", "n", "b", "r", "t"])

def best_per_cell(df, tag):
    sub = df[df["tag"] == tag].copy()
    # 1. median across reps for each (P, n, b, r)
    med = sub.groupby(["P", "n", "b", "r"], as_index=False)["t"].median()
    # 2. for each (P, n), pick the (b, r) with the smallest median
    idx = med.groupby(["P", "n"])["t"].idxmin()
    best = med.loc[idx].reset_index(drop=True)
    return best  # columns: P, n, b, r, t

def render_heatmap(ax, best_df, value_col, title, cmap, ylabel,
                   n_bytes_label=True):
    """Render a heatmap where rows are message sizes, columns are process
    counts, and cell colour encodes best_df[value_col]."""
    Ps  = sorted(best_df["P"].unique())
    ns  = sorted(best_df["n"].unique())
    grid = np.full((len(ns), len(Ps)), np.nan)
    for _, row in best_df.iterrows():
        i = ns.index(row["n"])
        j = Ps.index(row["P"])
        grid[i, j] = row[value_col]

    # log-color normalisation since r, b can span 2..8192
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    if vmin <= 0:
        vmin = 1
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    # annotate each cell with the best value
    for i in range(len(ns)):
        for j in range(len(Ps)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="gray")
            else:
                rgba = cmap(norm(v))
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "white" if lum < 0.55 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(Ps)))
    ax.set_xticklabels([str(p) for p in Ps], fontsize=13)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([(f"{n*8}" if n_bytes_label else str(n)) for n in ns],
                       fontsize=12)
    ax.set_xlabel("Process Count (P)", fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(value_col, fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    return im

def main():
    df = load()
    if df.empty:
        raise RuntimeError("No gata/gatav data found in " + DATA_DIR)

    best_gata  = best_per_cell(df, "gata")
    best_gatav = best_per_cell(df, "gatav")

    print(f"gata cells:  {len(best_gata)}")
    print(f"gatav cells: {len(best_gatav)}")

    cmap_r = plt.get_cmap("viridis")
    cmap_b = plt.get_cmap("plasma")

    YLABEL_GATA  = "Data-block Size (N) (bytes)"
    YLABEL_GATAV = "Maximum Data-block Size (N) (bytes)"

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    render_heatmap(axes[0, 0], best_gata,  "r",
                   "(a) gata: best radix $r^\\star(P, n)$", cmap_r, YLABEL_GATA)
    render_heatmap(axes[0, 1], best_gata,  "b",
                   "(b) gata: best batch size $b^\\star(P, n)$", cmap_b, YLABEL_GATA)
    render_heatmap(axes[1, 0], best_gatav, "r",
                   "(c) gatav: best radix $r^\\star(P, n)$", cmap_r, YLABEL_GATAV)
    render_heatmap(axes[1, 1], best_gatav, "b",
                   "(d) gatav: best batch size $b^\\star(P, n)$", cmap_b, YLABEL_GATAV)
    fig.suptitle("Per-cell optimal $(r, b)$ on Polaris (Slingshot 11)",
                 fontsize=16, fontweight="bold")
    out_combined = os.path.join(OUT_DIR, "heatmap_rb_choice.png")
    fig.savefig(out_combined, dpi=160, bbox_inches="tight")
    print(f"Saved {out_combined}")

    # Also save four standalone PNGs (handy if a figure caption wants them split)
    for tag, best_df, kind, title, cmap, ylabel in [
        ("gata",  best_gata,  "r", "gata: best radix $r^\\star(P, n)$",      cmap_r, YLABEL_GATA),
        ("gata",  best_gata,  "b", "gata: best batch size $b^\\star(P, n)$", cmap_b, YLABEL_GATA),
        ("gatav", best_gatav, "r", "gatav: best radix $r^\\star(P, n)$",     cmap_r, YLABEL_GATAV),
        ("gatav", best_gatav, "b", "gatav: best batch size $b^\\star(P, n)$",cmap_b, YLABEL_GATAV),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
        render_heatmap(ax, best_df, kind, title, cmap, ylabel)
        out = os.path.join(OUT_DIR, f"heatmap_rb_choice_{tag}_{kind}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
