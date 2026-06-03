#!/usr/bin/env python3
"""Compute best-case (per-cell) speedups of gata/gatav vs vendor and
vs the best non-vendor user-space algorithm.

NOTE on driver disambiguation: the ata (gata_example) and atav
(gatav_example) drivers print the SAME tag string for their shared
baselines (Bruck, MPICH-scattered, Spreadout, OMPI-linear,
OMPI-pairwise). The merged log alternates ata/atav blocks; we
walk the file and maintain a driver-mode state machine keyed on
the unambiguous tags (MPI_Alltoall / gata = ata mode; MPI_Alltoallv
/ gatav = atav mode). All other tags inherit the current mode and
get a `_ata` or `_atav` suffix in the analysis.
"""

import os, re, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

line_re = re.compile(
    r"^\[([^\]]+)\]\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*([\deE+\-.]+)\s*$"
)

ATA_ONLY  = {"MPI_Alltoall", "gata"}
ATAV_ONLY = {"MPI_Alltoallv", "gatav"}

rows = []
for path in sorted(glob.glob(os.path.join(DATA, "run_*_merged.sh.o"))):
    mode = None
    with open(path) as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            tag, P, n, b, r, t = m.groups()
            P, n, b, r, t = int(P), int(n), int(b), int(r), float(t)
            if tag in ATA_ONLY:
                mode = "ata"
            elif tag in ATAV_ONLY:
                mode = "atav"
            elif mode is None:
                continue
            full_tag = tag if tag in ATA_ONLY | ATAV_ONLY else f"{tag}_{mode}"
            rows.append((full_tag, P, n, b, r, t))

df = pd.DataFrame(rows, columns=["tag", "P", "n", "b", "r", "t"])
df = df[df["t"] > 0]

agg = df.groupby(["tag", "P", "n", "b", "r"], as_index=False)["t"].median()
best_per_cell = agg.groupby(["tag", "P", "n"], as_index=False)["t"].min()
pivot = best_per_cell.pivot(index=["P", "n"], columns="tag", values="t")
print("Cells:", len(pivot))
print("Tags :", list(pivot.columns))

def speedup_summary(name, num_tag, den_tag):
    if num_tag not in pivot or den_tag not in pivot:
        print(f"  ({name}: missing {num_tag} or {den_tag})")
        return
    s = (pivot[num_tag] / pivot[den_tag]).dropna()
    cmax = s.idxmax()
    print(f"  {name:<40} med={s.median():>6.2f}x  max={s.max():>8.2f}x  "
          f"@ P={cmax[0]}, n={cmax[1]} ({den_tag}={pivot.loc[cmax,den_tag]*1e6:.0f}us, "
          f"{num_tag}={pivot.loc[cmax,num_tag]*1e6:.0f}us)")

print("\n========== gata (uniform) speedups ==========")
for c in ["MPI_Alltoall", "Bruck_ata", "MPICH-scattered_ata", "Spreadout_ata",
          "OMPI-pairwise_ata", "OMPI-linear_ata"]:
    speedup_summary(f"gata vs {c}", c, "gata")

print("\n  --- gata vs best non-vendor (per cell, ata-only baselines) ---")
ucols = [c for c in ["Bruck_ata", "MPICH-scattered_ata", "Spreadout_ata",
                     "OMPI-pairwise_ata", "OMPI-linear_ata"] if c in pivot]
pivot["best_user_ata"] = pivot[ucols].min(axis=1, skipna=True)
pivot["best_user_ata_who"] = pivot[ucols].idxmin(axis=1, skipna=True)
s = (pivot["best_user_ata"] / pivot["gata"]).dropna()
cmax = s.idxmax()
print(f"  median={s.median():.2f}x  MAX={s.max():.2f}x at P={cmax[0]}, n={cmax[1]}, "
      f"best={pivot.loc[cmax,'best_user_ata_who']}")

print("\n========== gatav (non-uniform) speedups ==========")
for c in ["MPI_Alltoallv", "Bruck_atav", "MPICH-scattered_atav", "Spreadout_atav",
          "OMPI-pairwise_atav", "OMPI-linear_atav"]:
    speedup_summary(f"gatav vs {c}", c, "gatav")

print("\n  --- gatav vs best non-vendor (per cell, atav-only baselines) ---")
ucols = [c for c in ["Bruck_atav", "MPICH-scattered_atav", "Spreadout_atav",
                     "OMPI-pairwise_atav", "OMPI-linear_atav"] if c in pivot]
pivot["best_user_atav"] = pivot[ucols].min(axis=1, skipna=True)
pivot["best_user_atav_who"] = pivot[ucols].idxmin(axis=1, skipna=True)
s = (pivot["best_user_atav"] / pivot["gatav"]).dropna()
cmax = s.idxmax()
print(f"  median={s.median():.2f}x  MAX={s.max():.2f}x at P={cmax[0]}, n={cmax[1]}, "
      f"best={pivot.loc[cmax,'best_user_atav_who']}")

print("\n  --- gatav top 10 vs best non-vendor atav ---")
top = (pivot["best_user_atav"] / pivot["gatav"]).dropna().sort_values(ascending=False).head(10)
for (P, n), v in top.items():
    print(f"    P={P:>5}, n={n:>5} B : gatav={pivot.loc[(P,n),'gatav']*1e6:>9.1f}us, "
          f"best={pivot.loc[(P,n),'best_user_atav_who']:<25}={pivot.loc[(P,n),'best_user_atav']*1e6:>9.1f}us, "
          f"speedup={v:.2f}x")
