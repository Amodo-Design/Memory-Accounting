"""Load curves: how the honest timing envelope and detectability move with GPU load.

  python -m memory_accounting.experiments.loadcurve results/load/idle.json results/load/coarse_f*.json results/load/fine_f*.json \
      --outdir figures/loadcurve

Each input is a sweep JSON; its load fraction and granularity are read from meta.background_load
(absent = idle, fraction 0, drawn in both series).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from ..core.analysis import MIN_RATIO, all_comparisons, best_config, expected_cheats, index, load_results
from . import amodo_graph_style as st
from .plots import CHEAT_COLOR, HONEST, _title, cheat_color, cheat_order, fmt_b

KEY_CONFIGS = [(65536, 16, 32), (1048576, 8, 128), (1048576, 16, 0), (262144, 32, 128)]
_PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def _fraction_axis(ax, log_y=True):
    ax.set_xlim(-0.05, 1.05)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(_PLAIN)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


LINESTYLE = {"coarse": "-", "fine": "--"}


def series_of(res) -> tuple[str, float]:
    bl = res["meta"].get("background_load")
    if not bl:
        return "idle", 0.0
    return ("coarse" if bl.get("matmul_dim", 8192) >= 4096 else "fine"), float(bl.get("fraction", 1.0))


def load_all(paths):
    runs = []
    for p in paths:
        res = load_results(Path(p))
        bl = res["meta"].get("background_load") or {}
        if bl.get("error"):  # its load thread died (memory_accounting.experiments.load): not a point at its fraction
            print(f"warning: {p}: the background load did not run ({bl['error']}): left out", file=sys.stderr)
            continue
        gran, frac = series_of(res)
        runs.append({"path": p, "res": res, "gran": gran, "frac": frac, "ix": index(res["records"])})
    grans = sorted({r["gran"] for r in runs if r["gran"] != "idle"}) or ["coarse"]
    # the idle run anchors every granularity series at fraction 0
    out = []
    for r in runs:
        if r["gran"] == "idle":
            for g in grans:
                out.append({**r, "gran": g})
        else:
            out.append(r)
    return sorted(out, key=lambda r: (r["gran"], r["frac"]))


def fig_honest_vs_load(runs, outdir):
    grans = sorted({r["gran"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for i, key in enumerate(KEY_CONFIGS):
        color = st.AMODO_SERIES[i % len(st.AMODO_SERIES)]
        for g in grans:
            pts = [
                (r["frac"], r["ix"][("honest",) + key]["samples_ns"] / 1e3)
                for r in runs
                if r["gran"] == g and ("honest",) + key in r["ix"]
            ]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            axes[0].plot(
                xs,
                [np.median(p[1]) for p in pts],
                LINESTYLE.get(g, "-"),
                marker=st.AMODO_MARKERS[i],
                color=color,
                lw=1.6,
                label=f"{fmt_b(key[0])} n{key[1]} r{key[2]} ({g})",
            )
            axes[1].plot(
                xs,
                [np.quantile(p[1], 0.999) for p in pts],
                LINESTYLE.get(g, "-"),
                marker=st.AMODO_MARKERS[i],
                color=color,
                lw=1.6,
            )
    for ax, lab in zip(axes, ("honest median", "honest p99.9 = deadline")):
        _fraction_axis(ax)
        st.style_axis(ax, lab, "background load fraction", "µs")
    axes[0].legend(fontsize=8, ncol=1)
    _title(
        fig,
        "Honest chain time vs GPU load",
        "solid = coarse load (8192² matmul + 1 GiB streaming), dashed = fine load (2048² + 64 MiB); fraction 0 = idle",
    )
    return st.save_plot(fig, outdir / "honest_vs_load.png")


def fig_perlink_penalty(runs, outdir):
    """Marginal per-link honest cost vs load, against the per-link cost of the CPU cheat (load-independent)."""
    grans = sorted({r["gran"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (B, r) in zip(axes, ((65536, 32), (1048576, 128))):
        for g in grans:
            for prov, color, lab in (
                ("honest", HONEST, "GPU honest"),
                ("cpu", CHEAT_COLOR["cpu"], "CPU cheat"),
                ("dram_mapped", CHEAT_COLOR["dram_mapped"], "PCIe zero-copy cheat"),
            ):
                pts = []
                for run in runs:
                    if run["gran"] != g:
                        continue
                    ns = sorted({k[2] for k in run["ix"] if k[0] == prov and k[1] == B and k[3] == r})
                    if len(ns) < 2:
                        continue
                    a = np.median(run["ix"][(prov, B, ns[0], r)]["samples_ns"])
                    b = np.median(run["ix"][(prov, B, ns[-1], r)]["samples_ns"])
                    pts.append((run["frac"], max((b - a) / (ns[-1] - ns[0]) / 1e3, 1e-3)))
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        LINESTYLE.get(g, "-"),
                        marker="o",
                        color=color,
                        lw=1.6,
                        label=f"{lab} ({g})",
                    )
        _fraction_axis(ax)
        st.style_axis(ax, f"B = {fmt_b(B)}, r = {r}", "background load fraction", "µs per link")
        ax.legend(fontsize=8)
    _title(
        fig,
        "Per-link cost vs load: where the load penalty overtakes the cheat delta",
        "slope of median chain time over n; a cheat is detectable only while its line stays above the honest line",
    )
    return st.save_plot(fig, outdir / "perlink_vs_load.png")


def fig_detectability(runs, outdir, jitter_us, min_ratio=MIN_RATIO):
    grans = sorted({r["gran"] for r in runs})
    cheats = cheat_order({k[0] for r in runs for k in r["ix"]} - {"honest"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows, any_pass = [], False
    for g in grans:
        xs, best_auc, cheap = [], [], []
        per_cheat = {c: [] for c in cheats}
        for run in [r for r in runs if r["gran"] == g]:
            comps = all_comparisons(run["res"], jitter_us=jitter_us)
            key, table = best_config(comps, min_ratio=min_ratio, expected=expected_cheats(run["res"]))
            # a configuration missing a cheat (its record dropped) has a min-AUC over the others only: left out
            complete = [t for t in table if not t.get("missing")]
            top = max(complete, key=lambda t: t["min_auc"]) if complete else None
            xs.append(run["frac"])
            best_auc.append(top["min_auc"] if top else np.nan)
            cheap.append(next((t["honest_mean_us"] for t in table if t["ok"]), np.nan))
            for cheat in cheats:
                per_cheat[cheat].append(max([x.auc for x in comps if x.cheat == cheat] or [np.nan]))
            # the summary row describes one configuration: the selected one, or with none passing the complete
            # one with the best min-AUC; its min-AUC and weakest cheat are that configuration's
            shown = next((t for t in table if (t["B"], t["n"], t["r"]) == key), None) if key else top
            rows.append(
                (
                    g,
                    run["frac"],
                    key,
                    (shown["B"], shown["n"], shown["r"]) if shown else None,
                    shown["min_auc"] if shown else np.nan,
                    shown["weakest_cheat"] if shown else "—",
                )
            )
        axes[0].plot(
            xs,
            best_auc,
            LINESTYLE.get(g, "-"),
            marker="o",
            color=st.AMODO_INK,
            lw=2,
            label=f"best config, all cheats ({g})",
        )
        for cheat in cheats:
            axes[0].plot(
                xs,
                per_cheat[cheat],
                LINESTYLE.get(g, "-"),
                marker=".",
                color=cheat_color(cheat),
                lw=1.2,
                alpha=0.8,
                label=f"{cheat.replace('_', ' ')} ({g})",
            )
        if not np.all(np.isnan(cheap)):
            any_pass = True
            axes[1].plot(
                xs, cheap, LINESTYLE.get(g, "-"), marker="o", color=HONEST, lw=2, label=f"cheapest passing config ({g})"
            )
    axes[0].axhline(0.999, color=st.AMODO_INK, ls=":", lw=0.8)
    axes[0].set_ylim(0.3, 1.02)
    _fraction_axis(axes[0], log_y=False)
    st.style_axis(axes[0], "Best achievable AUC vs load", "background load fraction", "AUC (max over configs)")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    _fraction_axis(axes[1], log_y=any_pass)  # nothing to put on a log axis when no config passes anywhere
    st.style_axis(
        axes[1],
        "Cost of the cheapest passing config",
        "background load fraction",
        "honest chain time (µs); gap = none passes",
    )
    if any_pass:
        axes[1].legend(fontsize=8)
    else:
        axes[1].set_yticks([])
        axes[1].text(
            0.5,
            0.5,
            "no configuration passes at any load",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=st.AMODO_MUTED,
        )
    _title(
        fig,
        "Detectability vs GPU load",
        f"LAN jitter σ = {jitter_us:.0f} µs added; per-cheat lines = best AUC any config reaches against that cheat; "
        f"passing = AUC ≥ 0.999, ≤ 0.1% cheats accepted and mean cheat RTT ≥ {min_ratio:g}× honest against every cheat",
    )
    return st.save_plot(fig, outdir / "detectability_vs_load.png"), rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--outdir", default="figures/loadcurve")
    ap.add_argument("--jitter-us", type=float, default=45.0)
    ap.add_argument("--min-ratio", type=float, default=MIN_RATIO, help="selection rule, as in memory_accounting.experiments.plots")
    args = ap.parse_args(argv)
    st.apply_amodo_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = load_all(args.results)
    for key in KEY_CONFIGS:
        if not any(("honest",) + key in r["ix"] for r in runs):
            print(
                f"warning: no run has B = {fmt_b(key[0])}, n = {key[1]}, r = {key[2]}: its series is left out",
                file=sys.stderr,
            )
    print("wrote", fig_honest_vs_load(runs, outdir))
    print("wrote", fig_perlink_penalty(runs, outdir))
    f, rows = fig_detectability(runs, outdir, args.jitter_us, args.min_ratio)
    print("wrote", f)
    print(f"\n{'load':>7} {'frac':>5} | selected config (B, n, r), or the closest | its min-AUC | its weakest cheat")
    for g, frac, key, shown, auc, weak in rows:

        def name(k):
            return f"{fmt_b(k[0])} n{k[1]} r{k[2]}"

        cfg = (
            name(key)
            if key
            else (f"none passes (closest: {name(shown)})" if shown else "none passes (no complete config)")
        )
        print(f"{g:>7} {frac:5.2f} | {cfg:<41} | {auc:11.4f} | {weak}")


if __name__ == "__main__":
    main()
