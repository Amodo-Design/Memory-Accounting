"""Recreate the main timing figure from a v0.19 verifier run.jsonl.

Usage: python reproduce_timing.py run.jsonl --out figures
Reference environment: NumPy 2.5.5, SciPy 1.18.1, Matplotlib 3.11.2, Arial.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from scipy.stats import gaussian_kde
from memory_accounting.core.wire import CONTROL_POSTURES

SEED = 20261118
DRAWS = 20_000
SAMPLE_SIZE = 5_000
SIZES = [5_000, 50_000, 500_000, 5_000_000]
SETTLE = 2.0
ORDER = ["honest", "encode_otf@0.01", "partial@0.01", "other_gpu@0.01", "dram_copy@0.01", "cpu@0.01"]
LABELS = {
    "honest": "Honest",
    "encode_otf@0.01": "Data encoded on demand",
    "partial@0.01": "Data read from DRAM",
    "other_gpu@0.01": "Other GPU (simulated locally)",
    "dram_copy@0.01": "DRAM data copied to HBM",
    "cpu@0.01": "Matmul computed on the CPU",
}
COLORS = dict(zip(ORDER, ["#0c8c88", "#aa7b2c", "#8452a0", "#d64f42", "#3384b9", "#bd5829"]))
INK, MUTED, GRID, RULE = "#304d4f", "#708184", "#e3e9e7", "#bccbc8"


def read_timing(path, kind="static"):
    postures, rows, claims = [], [], {}
    counts = collections.Counter()
    alert_counts = collections.Counter()
    excluded = collections.Counter()
    for line in path.open():
        event = json.loads(line)
        counts[event["type"]] += 1
        if event["type"] == "posture":
            postures.append((event["t"], event["posture"]))
        elif event["type"] == "meta":
            claims = event.get("claims", {})
        elif event["type"] == "alert":
            alert_counts[event.get("kind", "unknown")] += 1
        elif event["type"] == "verification":
            if event.get("phase") == "calibration" or event.get("warmup"):
                continue
            if event.get("round") == kind:
                if event.get("timestamp_source") != "kernel":
                    excluded["non_kernel_timestamps"] += 1
                    continue
                ns = event.get("rtt_ns") if kind == "static" else event.get("dyn", {}).get("rtt_ns")
                rows.append((event["t"], ns))
    pt = np.array([p[0] for p in postures])
    slots = collections.defaultdict(list)
    for t, ns in rows:
        j = int(np.searchsorted(pt, t, side="right")) - 1
        if j < 0:
            excluded["before_first_posture"] += 1
        elif t - pt[j] <= SETTLE:
            excluded["first_two_seconds_after_posture_event"] += 1
        elif ns is None or not np.isfinite(ns):
            excluded["missing_or_nonfinite_rtt"] += 1
        else:
            slots[j].append(ns)
    blocks = {p: [] for p in ORDER}
    records = []
    for j, values in sorted(slots.items()):
        if len(values) < 5:
            excluded["short_stretch_samples"] += len(values)
            continue
        key = postures[j][1]
        if key not in blocks:
            if key in CONTROL_POSTURES:
                excluded["control_posture_samples"] += len(values)
                continue
            raise ValueError(f"Unexpected posture: {key}")
        block = np.array(values, dtype=np.int64)
        blocks[key].append(block)
        records.append(dict(posture=key, slot=j, event_time=postures[j][0], samples=len(block),
                            mean_us=float(block.mean()/1000), sd_us=float(block.std(ddof=1)/1000)))
    if not all(blocks.values()):
        raise ValueError("Expected all six strategies to have usable stretches")
    return blocks, records, dict(events=dict(counts), alerts=dict(alert_counts), exclusions=dict(excluded),
                                 claims=claims, kind=kind, production_count=len(rows),
                                 first_production_t=rows[0][0], last_production_t=rows[-1][0])


def bootstrap_means(blocks, n, repetitions, rng):
    """Exact whole-stretch resampling, truncating the final stretch at n.

    Multinomial counts aggregate a chunk of draws only when even the longest
    possible blocks in that chunk are guaranteed to fit. Block order cannot
    affect a chunk's total; the final block is sampled separately and its prefix
    retained. This avoids allocating millions of ordered block draws.
    """
    lengths = np.array([len(b) for b in blocks], dtype=np.int64)
    maximum = int(lengths.max())
    prefix = np.zeros((len(blocks), maximum + 1), dtype=np.int64)
    for j, b in enumerate(blocks):
        prefix[j, 1:len(b)+1] = np.cumsum(b)
        prefix[j, len(b)+1:] = prefix[j, len(b)]
    sums = prefix[np.arange(len(blocks)), lengths]
    remaining = np.full(repetitions, n, dtype=np.int64)
    totals = np.zeros(repetitions, dtype=np.int64)
    probabilities = np.full(len(blocks), 1 / len(blocks))
    while np.any(remaining):
        active = np.flatnonzero(remaining)
        whole = (remaining[active] - 1) // maximum
        counts = rng.multinomial(whole, probabilities)
        totals[active] += counts @ sums
        remaining[active] -= counts @ lengths
        chosen = rng.integers(0, len(blocks), len(active))
        take = np.minimum(remaining[active], lengths[chosen])
        totals[active] += prefix[chosen, take]
        remaining[active] -= take
    return totals / n / 1000


def check_bootstrap():
    """Compare the fast sampler with exact finite-distribution moments."""
    blocks = [np.array([1, 7]), np.array([3, 2, 9]), np.array([4])]
    n = 17
    means = np.zeros(n+1)
    seconds = np.zeros(n+1)
    for remaining in range(1, n+1):
        for b in blocks:
            take = min(remaining, len(b))
            subtotal = int(b[:take].sum())
            after = remaining-take
            means[remaining] += (subtotal + means[after]) / len(blocks)
            seconds[remaining] += (subtotal**2 + 2*subtotal*means[after] + seconds[after]) / len(blocks)
    expected = means[n]/n/1000
    variance = (seconds[n]-means[n]**2)/n**2/1e6
    result = bootstrap_means(blocks, n, 150_000, np.random.default_rng(12))
    z = abs(result.mean()-expected)/np.sqrt(variance/len(result))
    ratio = float(result.var()/variance)
    assert z < 5 and abs(ratio-1) < .025, (z, ratio)
    constant = bootstrap_means([np.full(2, 6000), np.full(7, 6000)], 101, 100, np.random.default_rng(1))
    assert np.all(constant == 6)
    return dict(exact_mean_us=expected, monte_carlo_mean_us=float(result.mean()),
                mean_error_in_standard_errors=float(z), variance_ratio=ratio,
                constant_blocks_exact=True)


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": RULE,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "legend.frameon": False, "legend.labelcolor": INK,
    })


def header(fig, title, subtitle):
    fig.text(.075, .955, title, fontsize=24, weight="bold", va="top")
    fig.text(.075, .88, subtitle, fontsize=12, color=MUTED, va="top")


def density_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(0, 1.16)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", color=RULE, length=4, labelsize=10, labelrotation=90)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def curve(samples):
    kde = gaussian_kde(samples, bw_method="scott")
    bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    x = np.linspace(samples.min()-3*bandwidth, samples.max()+3*bandwidth, 800)
    y = kde(x)
    return x, y/y.max()


def save(fig, out, stem):
    for ext in ("png", "svg"):
        fig.savefig(out/f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def all_cheats_plot(samples, out, honest_mean, kind="static"):
    curves = {p: curve(samples[p]) for p in ORDER}
    ranked = sorted(ORDER, key=lambda p: float(np.mean(samples[p])))
    intervals = [(min(float(curves[p][0][0]) for p in group),
                  max(float(curves[p][0][-1]) for p in group))
                 for group in (ranked[:4], ranked[4:5], ranked[5:])]
    merged = []
    for lo, hi in intervals:
        if merged and lo-merged[-1][1] < 15:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    # Every panel uses the same number of pixels per microsecond.
    for interval in merged:
        pad = max(2, (interval[1]-interval[0])*.06)
        interval[0] -= pad
        interval[1] += pad
    fig = plt.figure(figsize=(15, 7.3))
    gs = fig.add_gridspec(1, len(merged), left=.075, right=.975, bottom=.38, top=.79,
                          width_ratios=[hi-lo for lo, hi in merged], wspace=.055)
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(merged))]
    for j, (ax, (lo, hi)) in enumerate(zip(axes, merged)):
        density_axis(ax)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MultipleLocator(25 if max(b-a for a, b in merged) < 400 else 50))
        for p in ORDER:
            x, y = curves[p]
            if x[-1] >= lo and x[0] <= hi:
                ax.fill_between(x, 0, y, color=COLORS[p], alpha=.25, linewidth=0)
                ax.plot(x, y, color=COLORS[p], lw=2)
        if lo <= honest_mean <= hi:
            ax.axvline(honest_mean, color=MUTED, linestyle=(0, (2, 3)), lw=1, zorder=0)
        if j:
            # A single diagonal stroke on each facing edge marks the omitted range.
            ax.plot([-.012, .012], [-.013, .013], color=MUTED, lw=1.3,
                    transform=ax.transAxes, clip_on=False)
        if j < len(axes)-1:
            ax.plot([.988, 1.012], [-.013, .013], color=MUTED, lw=1.3,
                    transform=ax.transAxes, clip_on=False)
    fig.canvas.draw()
    pixels_per_us = [ax.get_window_extent().width/(hi-lo) for ax, (lo, hi) in zip(axes, merged)]
    assert np.allclose(pixels_per_us, pixels_per_us[0], rtol=1e-10)
    axes[0].set_ylabel("Relative density", labelpad=12)
    region = "static memory" if kind == "static" else "dynamic KV memory"
    header(fig, "Distribution of mean response times for honest and cheating provers",
           f"Five strategies at 1% of {region} · 5,000 challenges per mean · 20,000 resampled means per curve")
    fig.text(.525, .23, f"Mean {kind} response time (µs)", ha="center", fontsize=14)
    legend_order = ["honest", "partial@0.01", "dram_copy@0.01", "encode_otf@0.01", "other_gpu@0.01", "cpu@0.01"]
    fig.legend([Line2D([], [], color=COLORS[p], lw=3.2) for p in legend_order],
               [LABELS[p] for p in legend_order], loc="lower center", bbox_to_anchor=(.52, .006),
               ncol=2, labelspacing=.3, columnspacing=2.5, handlelength=2.8, fontsize=15,
               title="Strategies")
    save(fig, out, "timing_all_cheats_1pct")
    return merged


def sizes_plot(means, comparison, out, stem, kind="static"):
    fig = plt.figure(figsize=(15, 7.3))
    ax = fig.add_axes([.075, .32, .9, .47])
    density_axis(ax)
    ax.tick_params(axis="x", labelrotation=0)
    palettes = [["#84c9c4", "#38a69f", "#077975", "#003f40"],
                ["#efa99c", "#e5705c", "#b83929", "#671811"]]
    lo, hi = np.inf, -np.inf
    for row, (posture, shades) in enumerate(zip(["honest", comparison], palettes)):
        for i, (n, shade) in enumerate(zip(SIZES, shades)):
            x, y = curve(means[posture, n])
            lo, hi = min(lo, x.min()), max(hi, x.max())
            ax.fill_between(x, 0, y, color=shade, alpha=.25, linewidth=0)
            ax.plot(x, y, color=shade, lw=2)
            xpos, ypos = .41 + i*.112, .075 - row*.048
            fig.add_artist(Line2D([xpos-.021, xpos+.021], [ypos, ypos], color=shade,
                                  lw=5, transform=fig.transFigure))
            if row == 0:
                fig.text(xpos, .106, f"{n:,}", ha="center", fontsize=15)
        fig.text(.355, .075-row*.048, LABELS[posture], ha="right", va="center", fontsize=15)
    pad = .025*(hi-lo)
    ax.set_xlim(lo-pad, hi+pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.set_ylabel("Relative density", labelpad=12)
    region = "static blocks within weights" if kind == "static" else "dynamic blocks within the KV region"
    header(fig, "Distribution of mean response times with different sample sizes",
           "Honest prover versus " + LABELS[comparison].lower()
           + f" · 1% of challenged {region} · projection from measured stretches")
    fig.text(.525, .215, f"Mean {kind} response time (µs)", ha="center", fontsize=14)
    fig.text(.56, .147, "Challenges per resampled mean", ha="center", fontsize=15)
    save(fig, out, stem)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--kind", choices=("static", "dynamic"), default="static")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    check_bootstrap()
    blocks, _, metadata = read_timing(args.log, args.kind)
    samples = {}
    rows = []
    for index, strategy in enumerate(ORDER):
        measured = np.concatenate(blocks[strategy]) / 1000
        seed = np.random.SeedSequence([SEED, index, SAMPLE_SIZE])
        simulated = bootstrap_means(blocks[strategy], SAMPLE_SIZE, DRAWS, np.random.default_rng(seed))
        samples[strategy] = simulated
        rows.append(dict(strategy=LABELS[strategy], stretches=len(blocks[strategy]),
                         response_count=len(measured), measured_mean_us=measured.mean(),
                         bootstrap_mean_us=simulated.mean(), bootstrap_sd_us=simulated.std(ddof=1)))
    all_cheats_plot(samples, args.out, rows[ORDER.index("honest")]["measured_mean_us"], args.kind)
    measured_means = {ORDER[i]: row["measured_mean_us"] for i, row in enumerate(rows)}
    fastest = min(ORDER[1:], key=lambda posture: measured_means[posture])
    sample_size_rows = []
    for comparison, stem in [("encode_otf@0.01", "timing_weights_rebuilt_sample_sizes"),
                             (fastest, "timing_fastest_cheat_sample_sizes")]:
        means = {}
        for posture in ("honest", comparison):
            for n in SIZES:
                seed = np.random.SeedSequence([SEED, ORDER.index(posture), n])
                simulated = bootstrap_means(blocks[posture], n, DRAWS, np.random.default_rng(seed))
                means[posture, n] = simulated
                sample_size_rows.append(dict(comparison=LABELS[comparison], posture=LABELS[posture],
                                             challenges=n, bootstrap_mean_us=simulated.mean(),
                                             bootstrap_sd_us=simulated.std(ddof=1),
                                             lower_95_us=np.quantile(simulated, .025),
                                             upper_95_us=np.quantile(simulated, .975)))
        sizes_plot(means, comparison, args.out, stem, args.kind)
    with (args.out / "timing_sample_sizes.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sample_size_rows[0])
        writer.writeheader()
        writer.writerows(sample_size_rows)
    with (args.out / "timing_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Excluded {args.kind} responses:", metadata["exclusions"])


if __name__ == "__main__":
    main()
