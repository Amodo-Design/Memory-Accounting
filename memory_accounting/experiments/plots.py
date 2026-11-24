"""Figures in the house style: RTT distributions (mean +- jitter) honest vs cheat, AUC heatmaps,
gap/jitter scaling, honest overhead, and the best-configuration summary."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.ticker

# Bundled IBM Plex fonts render the figures; the OS fallbacks listed after them are looked up
# too for per-glyph fallback and are missing on most Linux hosts, which is harmless noise.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde

from ..core.analysis import (
    MIN_RATIO,
    all_comparisons,
    best_config,
    compare,
    core_samples,
    expected_cheats,
    index,
    load_results,
    network_samples,
)
from . import amodo_graph_style as st

CHEAT_LABEL = {
    "other_gpu": "Copy on another GPU, permute + re-encode on the fly",
    "encode_otf": "Compressible weights, encode on the fly",
    "dram_copy": "Weights in DRAM, copy each block to HBM",
    "dram_mapped": "Weights in DRAM, GPU reads over PCIe",
    "cpu": "Weights in DRAM, matmul + hash on CPU",
}
CHEAT_COLOR = {
    "other_gpu": st.AMODO_ACCENT,
    "encode_otf": st.AMODO_AMBER,
    "dram_copy": st.AMODO_BLUE,
    "dram_mapped": "#7a4f93",
    "cpu": "#b05f2d",
}
HONEST = st.AMODO_DEEP_TEAL
_OTHER_COLORS = [st.AMODO_GREEN, st.AMODO_INK, st.AMODO_TEAL, "#8a8f3c", "#b0577e", "#5c6f91"]


def cheat_color(cheat: str) -> str:
    """The cheat's colour; a strategy without one of its own (partial, lora, ...) gets a stable other one."""
    return CHEAT_COLOR.get(cheat) or _OTHER_COLORS[sum(cheat.encode()) % len(_OTHER_COLORS)]


def cheat_label(cheat: str) -> str:
    return CHEAT_LABEL.get(cheat, cheat.replace("_", " "))


def cheat_order(cheats) -> list[str]:
    """The known cheats in CHEAT_LABEL's order, then any others by name."""
    known = list(CHEAT_LABEL)
    return sorted(cheats, key=lambda c: (known.index(c) if c in known else len(known), c))


def fmt_b(B: int) -> str:
    return f"{B >> 20} MB" if B >= 1 << 20 else f"{B >> 10} KB"


def _unit(max_us: float):
    return (1e3, "ms") if max_us > 2000 else (1.0, "µs")


def _title(fig, title, subtitle):
    """House-style figure title with the subtitle spaced by figure height, not a fixed fraction."""
    height = fig.get_size_inches()[1]
    sub_y = 0.985 - 0.36 / height
    art = fig.suptitle(
        title.upper(),
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        fontfamily=st.AMODO_TITLE_FONT_FAMILY,
    )
    st.register_bbox_extra_artist(fig, art)
    sub = fig.text(0.01, sub_y, subtitle, ha="left", va="top", fontsize=9, color=st.AMODO_MUTED)
    st.register_bbox_extra_artist(fig, sub)
    fig._amodo_content_top = sub_y - 0.30 / height


def _density(ax, x_us, color, label, scale, lw=2.0, fill=True, annotate=True):
    x = x_us / scale
    if x.std() < 1e-9 or len(x) < 3:
        ax.axvline(x.mean(), color=color, lw=lw, label=label)
        return
    # KDE is fitted without scheduler stalls (> 10 MAD from the median) so they do not flatten the curve
    core = core_samples(x)
    if core.std() < 1e-9:
        core = x
    lo, hi = core.min() - 3 * core.std(), core.max() + 3 * core.std()
    grid = np.linspace(lo, hi, 400)
    kde = gaussian_kde(core)(grid)
    ax.plot(grid, kde, color=color, lw=lw, label=label)
    if fill:
        ax.fill_between(grid, kde, color=color, alpha=0.15, lw=0)
    ymax = kde.max()
    mean, std = core.mean(), core.std()
    ax.axvline(mean, color=color, lw=1, ls="--")
    ax.axvspan(mean - std, mean + std, color=color, alpha=0.08, lw=0)
    n_out = len(x) - len(core)
    if n_out:
        ax.text(
            hi,
            ymax * 0.5,
            f"+{n_out} outlier{'s' if n_out > 1 else ''} →",
            color=color,
            fontsize=8,
            ha="right",
            va="center",
        )
    if annotate:
        ax.annotate(
            "",
            xy=(mean - std, ymax * 1.06),
            xytext=(mean + std, ymax * 1.06),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 0.9},
        )
        ax.text(mean, ymax * 1.11, f"±{std * scale:.0f} µs", color=color, ha="center", va="bottom", fontsize=9)


def fig_distribution_ladder(results, cheat, B, r, ns, jitter_us, outdir, deadline_q=0.999, seed=0):
    ix = index(results["records"])
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.4), squeeze=False)
    for ax, n in zip(axes[0], ns):
        h = ix.get(("honest", B, n, r))
        c = ix.get((cheat, B, n, r))
        if h is None or c is None:
            ax.axis("off")
            continue
        hs = network_samples(h, jitter_us, seed=seed) / 1e3
        cs = network_samples(c, jitter_us, seed=seed) / 1e3
        k = compare(hs * 1e3, cs * 1e3, cheat, B, n, r, deadline_q)  # the same numbers as the summary
        scale, unit = _unit(max(hs.max(), cs.max()))
        _density(ax, hs, HONEST, "honest", scale)
        _density(ax, cs, cheat_color(cheat), "cheat", scale)
        hc, cc = core_samples(hs), core_samples(cs)
        ax.set_xlim(min(hc.min(), cc.min()) / scale * 0.9, max(hc.max(), cc.max()) / scale * 1.1)
        dl = k.deadline_us / scale
        ax.axvline(dl, color=st.AMODO_INK, lw=1, ls=":")
        a, far = k.auc, k.far
        st.style_axis(
            ax, f"n = {n} link{'s' if n > 1 else ''}", f"challenge RTT ({unit})", "density" if ax is axes[0][0] else ""
        )
        ax.set_yticks([])
        ax.text(
            0.98,
            0.97,
            f"AUC {a:.3f}\ncheat accepted {far * 100:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=st.AMODO_INK,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    h_, l_ = axes[0][0].get_legend_handles_labels()
    axes[0][0].legend(h_, l_, loc="upper left", fontsize=10)
    net = f"LAN jitter model σ = {jitter_us:.0f} µs added" if jitter_us else "device timing only (no network)"
    _title(
        fig,
        f"RTT distributions — {cheat_label(cheat)}",
        f"B = {fmt_b(B)}, r = {r}, {results['meta']['size_bytes'] >> 30} GiB array on {results['meta']['device']}; "
        f"{net}; dashed = mean, band = ±1σ, dotted = honest p{deadline_q * 100:g} deadline (set on half the honest samples)",
    )
    return st.save_plot(fig, outdir / f"dist_{cheat}_B{B >> 10}K_r{r}{'_lan' if jitter_us else ''}.png")


def fig_auc_heatmaps(comps, r, outdir, jitter_us):
    cheats = cheat_order({c.cheat for c in comps if c.r == r})
    Bs = sorted({c.B for c in comps if c.r == r})
    ns = sorted({c.n for c in comps if c.r == r})
    cmap = LinearSegmentedColormap.from_list("auc", [st.AMODO_ACCENT, "#e7c779", st.AMODO_TEAL, st.AMODO_BLUE])
    fig, axes = plt.subplots(
        1,
        len(cheats) + 1,
        figsize=(4.4 * len(cheats) + 1, 4.8),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * len(cheats) + [0.05]},
    )
    for ax, cheat in zip(axes[0][:-1], cheats):
        grid = np.full((len(ns), len(Bs)), np.nan)
        for c in comps:
            if c.cheat == cheat and c.r == r:
                grid[ns.index(c.n), Bs.index(c.B)] = c.auc
        im = ax.imshow(grid, origin="lower", cmap=cmap, norm=Normalize(0.5, 1.0), aspect="auto")
        for i in range(len(ns)):
            for j in range(len(Bs)):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=st.AMODO_BG if grid[i, j] > 0.93 or grid[i, j] < 0.6 else st.AMODO_INK,
                    )
        ax.set_xticks(range(len(Bs)))
        ax.set_xticklabels([fmt_b(b) for b in Bs])
        ax.set_yticks(range(len(ns)))
        ax.set_yticklabels([str(n) for n in ns])
        st.style_axis(
            ax, cheat.replace("_", " "), "block size B", "chain length n" if ax is axes[0][0] else "", grid_axis="x"
        )
        ax.grid(False)
    cbar = fig.colorbar(im, cax=axes[0][-1])
    cbar.set_label("AUC (1.0 = perfectly separable)")
    st.style_colorbar(cbar)
    _title(
        fig,
        f"Detectability (AUC) by chain length and block size — r = {r}",
        f"red = cheater indistinguishable from honest, blue = trivially separable; "
        f"{f'LAN jitter σ = {jitter_us:.0f} µs' if jitter_us else 'device timing only'}",
    )
    return st.save_plot(fig, outdir / f"auc_heatmap_r{r}{'_lan' if jitter_us else ''}.png")


def fig_gap_vs_n(comps, B, r, outdir, jitter_us):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sel = [c for c in comps if c.B == B and c.r == r]
    cheats = cheat_order({c.cheat for c in sel})
    ns = sorted({c.n for c in sel})
    for cheat in cheats:
        pts = sorted([(c.n, c.cheat_mean_us - c.honest_mean_us) for c in sel if c.cheat == cheat])
        ax.plot(
            [p[0] for p in pts],
            [max(p[1], 1e-3) for p in pts],
            marker="o",
            color=cheat_color(cheat),
            lw=2,
            label=f"gap: {cheat.replace('_', ' ')}",
        )
    hj = sorted([(c.n, c.honest_std_us) for c in sel if c.cheat == cheats[0]])
    ax.plot([p[0] for p in hj], [p[1] for p in hj], marker="s", color=HONEST, lw=2, ls="--", label="honest jitter (1σ)")
    ax.plot([p[0] for p in hj], [3 * p[1] for p in hj], color=HONEST, lw=1, ls=":", label="3σ honest jitter")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    st.style_axis(ax, f"Honest-vs-cheat gap grows with chain length (B = {fmt_b(B)}, r = {r})", "chain length n", "µs")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    _title(
        fig,
        "Mean gap vs jitter",
        f"gap = mean cheat RTT − mean honest RTT; {f'LAN jitter σ = {jitter_us:.0f} µs included' if jitter_us else 'device timing only'}",
    )
    return st.save_plot(fig, outdir / f"gap_vs_n_B{B >> 10}K_r{r}{'_lan' if jitter_us else ''}.png")


def fig_overhead(results, outdir):
    ix = index(results["records"])
    Bs = sorted({k[1] for k in ix})
    ns = sorted({k[2] for k in ix})
    rs = sorted({k[3] for k in ix})
    fig, axes = plt.subplots(1, len(rs), figsize=(4.8 * len(rs), 4.6), squeeze=False, sharey=True)
    for ax, r in zip(axes[0], rs):
        for i, n in enumerate(ns):
            ys = [np.median(ix[("honest", B, n, r)]["samples_ns"]) / 1e3 for B in Bs if ("honest", B, n, r) in ix]
            xs = [B for B in Bs if ("honest", B, n, r) in ix]
            ax.plot(
                xs,
                ys,
                marker=st.AMODO_MARKERS[i % len(st.AMODO_MARKERS)],
                color=st.AMODO_SERIES[i % len(st.AMODO_SERIES)],
                lw=1.6,
                label=f"n = {n}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Bs)
        ax.set_xticklabels([fmt_b(b) for b in Bs])
        st.style_axis(
            ax,
            f"r = {r}" + (" (hash only)" if r == 0 else ""),
            "block size B",
            "honest chain time (µs)" if ax is axes[0][0] else "",
        )
    axes[0][-1].legend(fontsize=9)
    dec = results["meta"].get("decode_gbps")
    _title(
        fig,
        "Honest prover cost per challenge",
        f"median device time; setup encode of {results['meta']['size_bytes'] >> 30} GiB took {results['meta']['setup_seconds']:.1f} s"
        + (f"; VDE decode runs at {dec:.0f} GB/s on this GPU" if dec and not np.isnan(dec) else ""),
    )
    return st.save_plot(fig, outdir / "honest_overhead.png")


def _core_stats(x_us):
    """Mean/std without scheduler stalls plus the number of stalls excluded."""
    core = core_samples(x_us)
    out = len(x_us) - len(core)
    return f"μ {core.mean():.0f} µs  σ {core.std():.0f} µs" + (
        f"  (+{out} outlier{'s' if out > 1 else ''})" if out else ""
    )


def fig_best(
    results,
    comps,
    key,
    outdir,
    jitter_us,
    deadline_q=0.999,
    seed=0,
    only: str | None = None,
    min_ratio: float = MIN_RATIO,
):
    """Honest vs every cheat at the selected configuration; with `only`, honest vs that one cheat.
    Samples, noise draw and deadline are those of the comparisons (`comps`), so the figure and the
    summary agree."""
    B, n, r = key
    ix = index(results["records"])
    hs = network_samples(ix[("honest", B, n, r)], jitter_us, seed=seed) / 1e3
    sel = {c.cheat: c for c in comps if (c.B, c.n, c.r) == key and (only is None or c.cheat == only)}
    cheats = cheat_order(sel)
    css = {c: network_samples(ix[(c, B, n, r)], jitter_us, seed=seed) / 1e3 for c in cheats}
    fig, ax = plt.subplots(figsize=(13, 6.4))
    allmax = max([hs.max()] + [css[c].max() for c in cheats])
    scale, unit = _unit(allmax)
    _density(ax, hs, HONEST, f"honest   {_core_stats(hs)}", scale, lw=2.5, annotate=False)
    for c in cheats:
        cs = css[c]
        k = sel[c]
        _density(
            ax,
            cs,
            cheat_color(c),
            f"{c.replace('_', ' ')}   {_core_stats(cs)}  AUC {k.auc:.3f}  accepted {k.far * 100:.1f}%",
            scale,
            lw=1.8,
            annotate=False,
        )
    dl = sel[cheats[0]].deadline_us / scale
    ax.axvline(dl, color=st.AMODO_INK, lw=1.2, ls=":")
    allq = [hs] + [css[c] for c in cheats]
    ax.set_xlim(
        min(core_samples(a).min() for a in allq) / scale * 0.7, max(core_samples(a).max() for a in allq) / scale * 1.3
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    st.style_axis(
        ax,
        f"Best configuration: B = {fmt_b(B)}, n = {n}, r = {r}" + (f" — {cheat_label(only)}" if only else ""),
        f"challenge RTT ({unit}, log scale)",
        "density",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.text(
        dl,
        ax.get_ylim()[1] * 0.98,
        f" deadline\n (honest p{deadline_q * 100:g})",
        va="top",
        fontsize=9,
        color=st.AMODO_INK,
    )
    leg = ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.5,
    )
    st.register_bbox_extra_artist(fig, leg)
    _title(
        fig,
        ("All cheats" if only is None else "One cheat") + " against the honest prover at the selected configuration",
        f"{f'LAN jitter σ = {jitter_us:.0f} µs added' if jitter_us else 'device timing only'}; dashed = mean, band = ±1σ; "
        f"selected = cheapest honest chain with AUC ≥ 0.999 and ≤ 0.1% cheats accepted against every strategy"
        + (f", mean cheat RTT ≥ {min_ratio:g}× honest" if min_ratio > 1 else ""),
    )
    return st.save_plot(fig, outdir / f"best_config{'_' + only if only else ''}{'_lan' if jitter_us else ''}.png")


def fig_cpu_vs_gpu(results, outdir):
    """Marginal per-link cost = slope of median chain time between the shortest and longest chains,
    which removes the fixed per-challenge launch/sync overhead."""
    ix = index(results["records"])
    rs = sorted({k[3] for k in ix})
    Bs = sorted({k[1] for k in ix})
    ns = sorted({k[2] for k in ix})
    if not any(k[0] == "cpu" for k in ix) or len(ns) < 2:
        return None
    n0, n1 = ns[0], ns[-1]

    def per_link(p, B, r):
        a = np.median(ix[(p, B, n0, r)]["samples_ns"])
        b = np.median(ix[(p, B, n1, r)]["samples_ns"])
        return max((b - a) / (n1 - n0) / 1e3, 1e-3)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, B in enumerate(Bs):
        alpha = 0.35 + 0.65 * i / max(1, len(Bs) - 1)
        ax.plot(
            rs,
            [per_link("honest", B, r) for r in rs],
            marker="o",
            color=HONEST,
            lw=1.6,
            alpha=alpha,
            label=f"GPU honest, B = {fmt_b(B)}",
        )
        ax.plot(
            rs,
            [per_link("cpu", B, r) for r in rs],
            marker="s",
            color=CHEAT_COLOR["cpu"],
            lw=1.6,
            alpha=alpha,
            label=f"CPU cheat ({results['meta'].get('cpu_threads')} threads), B = {fmt_b(B)}",
        )
    ax.set_yscale("log")
    ax.set_xticks(rs)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    st.style_axis(
        ax, "Marginal per-link cost: GPU vs CPU as the matmul widens", "matmul width r (0 = hash only)", "µs per link"
    )
    ax.legend(fontsize=8.5, ncol=2)
    _title(
        fig,
        "Why the matmul is in the challenge",
        f"slope of median chain time between n = {n0} and n = {n1}; the matmul on the block is what separates a GPU next to the data from a CPU reading DRAM",
    )
    return st.save_plot(fig, outdir / "cpu_vs_gpu_matmul.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", default="results/sweep.json", nargs="?")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--jitter-us", type=float, default=45.0, help="LAN jitter model std (doc: 36-54 µs measured)")
    ap.add_argument("--ladder", default="1,4,16,64")
    ap.add_argument("--deadline-q", type=float, default=0.999)
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=MIN_RATIO,
        help="also require mean cheat RTT >= this multiple of honest for every cheat (default 3: the "
        "CPU cheater has ~2x of headroom left, so margins below ~3x are not relied on)",
    )
    args = ap.parse_args(argv)
    st.apply_amodo_style()
    results = load_results(Path(args.results))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    ns_all = sorted({r["n"] for r in results["records"]})
    ladder = [n for n in (int(x) for x in args.ladder.split(",")) if n in ns_all] or ns_all[:4]
    summary = {}
    for jit in (0.0, args.jitter_us):
        comps = all_comparisons(results, jitter_us=jit, deadline_q=args.deadline_q)
        key, table = best_config(comps, min_ratio=args.min_ratio, expected=expected_cheats(results))
        tag = "lan" if jit else "device"
        summary[tag] = {"jitter_us": jit, "min_ratio": args.min_ratio, "best": key, "table": table}
        for r in sorted({c.r for c in comps}):
            written.append(fig_auc_heatmaps(comps, r, outdir, jit))
        if key:
            written.append(fig_best(results, comps, key, outdir, jit, args.deadline_q, min_ratio=args.min_ratio))
            for cheat in cheat_order({c.cheat for c in comps}):
                written.append(
                    fig_best(results, comps, key, outdir, jit, args.deadline_q, only=cheat, min_ratio=args.min_ratio)
                )
            B, _, r = key
        else:
            B, r = table[-1]["B"], table[-1]["r"]
        written.append(fig_gap_vs_n(comps, B, r, outdir, jit))
        for cheat in cheat_order({c.cheat for c in comps}):
            written.append(fig_distribution_ladder(results, cheat, B, r, ladder, jit, outdir, args.deadline_q))
    written.append(fig_overhead(results, outdir))
    written.append(fig_cpu_vs_gpu(results, outdir))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for w in written:
        if w:
            print("wrote", w)
    for tag, s in summary.items():
        print(f"\n== {tag}: best config = {s['best']}")
        ok = [t for t in s["table"] if t["ok"]][:8]
        if not ok:
            print("  no configuration beats every cheat; nearest misses by weakest-cheat AUC:")
            ok = sorted(s["table"], key=lambda t: -t["min_auc"])[:8]
        for t in ok:
            print(
                f"  B={fmt_b(t['B']):>7} n={t['n']:3d} r={t['r']:4d}  honest {t['honest_mean_us']:8.0f}±{t['honest_std_us']:.0f} µs  minAUC {t['min_auc']:.4f} ({t['weakest_cheat']})  maxFAR {t['max_far']:.3f}  min ratio {t['min_ratio']:.1f}x"
                + (f"  not measured against: {', '.join(t['missing'])}" if t.get("missing") else "")
            )


if __name__ == "__main__":
    main()
