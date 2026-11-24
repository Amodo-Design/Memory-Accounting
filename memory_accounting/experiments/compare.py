"""Compare two sweeps (e.g. idle GPU vs background inference load): honest RTT jitter and the
weakest-cheat margin per configuration.

  python -m memory_accounting.experiments.compare results/sweep_vde.json results/sweep_vde_load.json --outdir figures/load
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.analysis import all_comparisons, expected_cheats, index, load_results
from ..core.analysis import core_samples as _core
from . import amodo_graph_style as st
from .plots import HONEST, _density, _title, _unit, fmt_b


def fig_jitter(a, b, la, lb, outdir):
    ia, ib = index(a["records"]), index(b["records"])
    keys = sorted({k for k in ia if k[0] == "honest" and k in ib}, key=lambda k: (k[3], k[1], k[2]))
    rs = sorted({k[3] for k in keys})
    Bs = sorted({k[1] for k in keys})
    ns = sorted({k[2] for k in keys})
    fig, axes = plt.subplots(1, len(rs), figsize=(5 * len(rs), 4.8), squeeze=False, sharey=True)
    for ax, r in zip(axes[0], rs):
        for i, B in enumerate(Bs):
            for res, ix, lab, ls in ((a, ia, la, "-"), (b, ib, lb, "--")):
                ys = [_core(ix[("honest", B, n, r)]["samples_ns"] / 1e3).std() for n in ns if ("honest", B, n, r) in ix]
                xs = [n for n in ns if ("honest", B, n, r) in ix]
                ax.plot(
                    xs,
                    ys,
                    ls=ls,
                    marker=st.AMODO_MARKERS[i],
                    color=st.AMODO_SERIES[i % len(st.AMODO_SERIES)],
                    lw=1.6,
                    label=f"{fmt_b(B)}, {lab}",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        st.style_axis(ax, f"r = {r}", "chain length n", "honest jitter, 1σ (µs)" if ax is axes[0][0] else "")
    axes[0][-1].legend(fontsize=8, ncol=1)
    _title(
        fig,
        "Honest jitter: idle GPU vs background load",
        "σ of honest chain times with scheduler stalls removed; solid = first sweep, dashed = second sweep",
    )
    return st.save_plot(fig, outdir / "honest_jitter_idle_vs_load.png")


def worst_cheat(results: dict, comps) -> tuple[dict, list]:
    """Per configuration, the comparison against its weakest cheat, over the configurations measured against
    every cheat the sweep ran (analysis.expected_cheats); the others, whose missing cheat (its record dropped
    as incorrect) may be the weakest, are returned apart and not drawn."""
    expected = set(expected_cheats(results))
    w, seen = {}, {}
    for c in comps:
        k = (c.B, c.n, c.r)
        seen.setdefault(k, set()).add(c.cheat)
        if k not in w or c.auc < w[k].auc:
            w[k] = c
    incomplete = sorted(k for k in w if not expected <= seen[k])
    return {k: v for k, v in w.items() if k not in incomplete}, incomplete


def fig_margin(a, b, la, lb, outdir, jitter_us):
    ca = all_comparisons(a, jitter_us=jitter_us)
    cb = all_comparisons(b, jitter_us=jitter_us)
    (wa, ia), (wb, ib) = worst_cheat(a, ca), worst_cheat(b, cb)
    for lab, inc in ((la, ia), (lb, ib)):
        if inc:
            print(
                f"{lab}: {len(inc)} configuration(s) left out of the margin figure, a cheat's record dropped: "
                + ", ".join(f"{fmt_b(k[0])} n{k[1]} r{k[2]}" for k in inc),
                file=sys.stderr,
            )
    keys = sorted(set(wa) & set(wb), key=lambda k: (k[2], k[0], k[1]))
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(keys)), 5))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [wa[k].auc for k in keys], 0.4, color=HONEST, label=f"min AUC, {la}")
    ax.bar(x + 0.2, [wb[k].auc for k in keys], 0.4, color=st.AMODO_ACCENT, label=f"min AUC, {lb}")
    ax.axhline(0.999, color=st.AMODO_INK, lw=0.8, ls=":")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fmt_b(k[0])}\nn{k[1]} r{k[2]}" for k in keys], fontsize=7)
    st.style_axis(ax, "Weakest-cheat AUC per configuration", "", "AUC against the weakest cheat", grid_axis="y")
    ax.legend(fontsize=9, loc="lower right")
    _title(
        fig,
        "Detectability with and without background load",
        f"LAN jitter σ = {jitter_us:.0f} µs added; dotted = 0.999 threshold",
    )
    return st.save_plot(fig, outdir / "min_auc_idle_vs_load.png")


def fig_honest_shapes(a, b, la, lb, outdir, B, r, ns):
    ia, ib = index(a["records"]), index(b["records"])
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.4), squeeze=False)
    for ax, n in zip(axes[0], ns):
        if ("honest", B, n, r) not in ia or ("honest", B, n, r) not in ib:
            ax.axis("off")
            continue
        ha, hb = ia[("honest", B, n, r)]["samples_ns"] / 1e3, ib[("honest", B, n, r)]["samples_ns"] / 1e3
        scale, unit = _unit(max(np.quantile(ha, 0.995), np.quantile(hb, 0.995)))
        _density(ax, ha, HONEST, la, scale)
        _density(ax, hb, st.AMODO_ACCENT, lb, scale)
        st.style_axis(ax, f"n = {n}", f"honest chain time ({unit})", "density" if ax is axes[0][0] else "")
        ax.set_yticks([])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    axes[0][0].legend(loc="upper left", fontsize=9)
    _title(
        fig,
        f"Honest chain time, idle vs loaded — B = {fmt_b(B)}, r = {r}",
        "device timing only; dashed = mean, band = ±1σ; stalls beyond 10 MAD excluded",
    )
    return st.save_plot(fig, outdir / f"honest_shapes_B{B >> 10}K_r{r}.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--labels", default="idle,loaded")
    ap.add_argument("--outdir", default="figures/load")
    ap.add_argument("--jitter-us", type=float, default=45.0)
    args = ap.parse_args(argv)
    st.apply_amodo_style()
    first, second = load_results(Path(args.first)), load_results(Path(args.second))
    la, lb = args.labels.split(",")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("wrote", fig_jitter(first, second, la, lb, outdir))
    print("wrote", fig_margin(first, second, la, lb, outdir, args.jitter_us))
    ib = index(second["records"])
    Bs = sorted({k[1] for k in ib})
    rs = sorted({k[3] for k in ib})
    ns = sorted({k[2] for k in ib})
    for B in (Bs[0], Bs[-1]):
        for r in (rs[0], rs[-1]):
            print("wrote", fig_honest_shapes(first, second, la, lb, outdir, B, r, ns[:4]))
    lr = second["meta"].get("background_load")
    if lr:
        print("background load during second sweep: " + load_line(lr))


def load_line(lr: dict) -> str:
    """The background load's report, or why it is not one (a load thread that died: memory_accounting.experiments.load)."""
    if lr.get("error") or lr.get("tflops") is None:
        return f"NOT RUNNING: {lr.get('error') or 'no throughput measured'} (the sweep did not run under load)"
    return f"{lr['tflops']:.0f} TFLOP/s matmul, {lr['bandwidth_gbps']:.0f} GB/s streaming"


if __name__ == "__main__":
    main()
