#!/usr/bin/env python3
"""Plot Fugaku benchmark results in the same style as the Polaris notebook.

The Fugaku runs only contain the uniform alltoall variant (gata + ata
baselines), so this script produces one PNG per process count:
  plots/fugaku/gata_<P>.png

It re-uses the parsing / aggregation logic from plot_results.ipynb,
overriding only the input glob and the output directory.
"""
import os, re, glob, math, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "fugaku-runs")
OUT_DIR  = os.path.join(HERE, "..", "plots", "fugaku")
os.makedirs(OUT_DIR, exist_ok=True)

# Fugaku files: 'fugaku-<P>' or 'fagaku-<P>' (typo in one file).
DATA_GLOB = os.path.join(DATA_DIR, "*ugaku-*") + " " + os.path.join(DATA_DIR, "*agaku-*")

LINE_RE = re.compile(
    r'\[([^\]]+)\]\s+(\d+),\s+(\d+),\s+(\d+),\s+(\d+),\s+([\d.eE+-]+)'
)

ATA_TAGS = ['MPI_Alltoall', 'OMPI-linear', 'OMPI-pairwise',
            'MPICH-scattered', 'Spreadout', 'Bruck', 'gata']

TAG_LABELS = {
    'OMPI-linear':     'Basic linear',
    'OMPI-pairwise':   'Pairwise',
    'MPICH-scattered': 'scattered',
    'Bruck':           'Classic Bruck',
    'gata':            'GAta',
    'gatav':           'GAtav',
}

MIN_N_PER_TAG = {'OMPI-linear': 64}
MAX_N_PER_TAG = {'Bruck': 512}
BAD_POINTS = set()
AUTO_BAD_TAGS  = {'OMPI-linear'}
AUTO_BAD_REF   = 'min'
AUTO_BAD_RATIO = 2.0
BYTES_PER_ELEM = 8


def parse_file(path):
    """Read one Fugaku log file. Only ata-side tags exist in these
    files, so we treat the whole file as one ata block."""
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nprocs_seen = set()
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            tag, nprocs, n, e1, e2, t = m.groups()
            nprocs_seen.add(int(nprocs))
            store[tag][int(n)][(int(e1), int(e2))].append(float(t))
    return store, nprocs_seen


def robust_stats(times):
    sorted_t = sorted(times)
    med = statistics.median(sorted_t)
    if len(sorted_t) < 4:
        return med, sorted_t[0], sorted_t[-1]
    q1, _, q3 = statistics.quantiles(sorted_t, n=4)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [t for t in sorted_t if lo_fence <= t <= hi_fence]
    if not filtered:
        filtered = sorted_t
    return med, min(filtered), max(filtered)


def best_per_n(data_for_tag):
    result = {}
    for n, params in data_for_tag.items():
        best = None
        for p, times in params.items():
            med, lo, hi = robust_stats(times)
            if best is None or med < best[1]:
                best = (p, med, lo, hi)
        result[n] = best
    return result


def tuned_bruck_per_n(data, nprocs, our_tag='gata'):
    if our_tag not in data:
        return {}
    target = math.sqrt(nprocs)
    rs_with_b1 = set()
    for n, params in data[our_tag].items():
        for (b, r) in params.keys():
            if b == 1:
                rs_with_b1.add(r)
    if not rs_with_b1:
        return {}
    near_r = min(rs_with_b1, key=lambda r: abs(r - target))
    result = {}
    for n, params in data[our_tag].items():
        if (1, near_r) in params:
            med, lo, hi = robust_stats(params[(1, near_r)])
            result[n] = (med, lo, hi)
    return result


def _fmt_xtick(n):
    b = n * BYTES_PER_ELEM
    if b > 512:
        return f'{b // 1024}k'
    return str(b)


def plot(data, nprocs, save_to=None):
    tags = ATA_TAGS
    ours = 'gata'

    fig, ax = plt.subplots(figsize=(14, 9))
    colors  = ['#888888', '#1f77b4', '#2ca02c', '#9467bd',
               '#8c564b', '#17becf', '#d62728']
    TUNED_BRUCK_COLOR = '#ff7f0e'

    bests = {tag: best_per_n(data[tag]) for tag in tags if tag in data}

    # Auto-filter
    reducer = min if AUTO_BAD_REF == 'min' else statistics.median
    auto_bad = set()
    for watched in AUTO_BAD_TAGS:
        if watched not in bests:
            continue
        for n, (_, my_med, _, _) in bests[watched].items():
            others = []
            for t, b in bests.items():
                if t == watched:
                    continue
                if ('ata', t, nprocs, n) in BAD_POINTS:
                    continue
                if n in b:
                    others.append(b[n][1])
            if not others:
                continue
            ref = reducer(others)
            if my_med > AUTO_BAD_RATIO * ref:
                auto_bad.add((watched, n))
    if auto_bad:
        print(f'[ata {nprocs}p] auto-dropped: ' +
              ', '.join(f'{t}@n={n}' for t, n in sorted(auto_bad)))

    all_ns = set()
    for i, tag in enumerate(tags):
        if tag not in data:
            continue
        best = bests[tag]
        min_n = MIN_N_PER_TAG.get(tag, 0)
        max_n = MAX_N_PER_TAG.get(tag, float('inf'))
        ns = sorted(n for n in best.keys()
                    if min_n <= n <= max_n
                    and ('ata', tag, nprocs, n) not in BAD_POINTS
                    and (tag, n) not in auto_bad)
        if not ns:
            continue
        all_ns.update(ns)
        meds = [best[n][1] * 1000 for n in ns]
        los  = [best[n][2] * 1000 for n in ns]
        his  = [best[n][3] * 1000 for n in ns]
        yerr = [[m - l for m, l in zip(meds, los)],
                [h - m for h, m in zip(his, meds)]]
        is_ours = (tag == ours)
        label = TAG_LABELS.get(tag, tag)
        ax.errorbar(ns, meds, yerr=yerr,
                    marker='o',
                    color=colors[i % len(colors)],
                    label=label,
                    linewidth=4 if is_ours else 2,
                    markersize=14 if is_ours else 10,
                    capsize=6,
                    capthick=2 if is_ours else 1.5,
                    elinewidth=2 if is_ours else 1.5,
                    zorder=10 if is_ours else 1)

    # Tuned Bruck overlay
    tb = tuned_bruck_per_n(data, nprocs, ours)
    if tb:
        tb_ns = sorted(n for n in tb.keys() if n in all_ns or all_ns == set())
        if tb_ns:
            tb_meds = [tb[n][0] * 1000 for n in tb_ns]
            tb_los  = [tb[n][1] * 1000 for n in tb_ns]
            tb_his  = [tb[n][2] * 1000 for n in tb_ns]
            tb_yerr = [[m - l for m, l in zip(tb_meds, tb_los)],
                       [h - m for h, m in zip(tb_his, tb_meds)]]
            ax.errorbar(tb_ns, tb_meds, yerr=tb_yerr,
                        marker='o',
                        color=TUNED_BRUCK_COLOR,
                        label='Tuned Bruck',
                        linewidth=2, markersize=10,
                        capsize=6, capthick=1.5, elinewidth=1.5,
                        zorder=2)
            all_ns.update(tb_ns)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    sorted_ns = sorted(all_ns)
    ax.set_xticks(sorted_ns)
    ax.set_xticklabels([_fmt_xtick(n) for n in sorted_ns])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    y_formatter = ScalarFormatter()
    y_formatter.set_scientific(False)
    y_formatter.set_useOffset(False)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2.0, 5.0], numticks=24))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_minor_formatter(y_formatter)

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='y', which='minor', labelsize=22)

    ax.set_xlabel('Data-block size (bytes)', fontsize=32, fontweight='bold')
    ax.set_ylabel('Time (ms, log scale)', fontsize=32, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=24)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_to}')
    plt.close(fig)


def main():
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*ugaku-*")) +
        glob.glob(os.path.join(DATA_DIR, "*agaku-*")),
        key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))
    )
    print(f'Found {len(files)} Fugaku file(s):')
    for f in files:
        print(f'  {os.path.basename(f)}')

    for path in files:
        base = os.path.basename(path)
        print('\n' + '=' * 70)
        print(f'FILE: {base}')
        print('=' * 70)
        data, nprocs_seen = parse_file(path)
        if not nprocs_seen:
            print('(no benchmark lines)')
            continue
        nprocs = next(iter(nprocs_seen))
        out = os.path.join(OUT_DIR, f'gata_{nprocs}.png')
        plot(data, nprocs, save_to=out)


if __name__ == '__main__':
    main()
