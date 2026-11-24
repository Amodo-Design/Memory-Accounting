"""The throughput benchmark as figures, from the JSON files scripts/bench_throughput.py writes (<dir>/<variant>.json).

  python scripts/plot_throughput.py results/throughput/qwen7b [--out docs/figures/throughput]
  python scripts/plot_throughput.py --summary results/throughput/qwen0.5b results/throughput/qwen7b results/throughput/qwen32b

  <tag>_throughput.png  generated tokens per second for each variant at each batch size, and the same relative to stock vLLM
  <tag>_encoding.png    the encoding's cost (Θ' stream-decoded, filler) over each baseline it can run on, as a slowdown, and
                        the cost of holding the dynamic region encoded over holding it plain (v0.17)
  <tag>_settings.png    each PoSM vLLM setting on its own, relative to stock vLLM (appendix)

  summary.png           (--summary) the PoSM settings, + encoded and + challenged as a share of stock vLLM, model by model,
                        at the fewest and the most concurrent requests

<tag> is the directory's name. A missing variant is left out, and a figure with nothing to show is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_accounting.experiments import amodo_graph_style as st
from memory_accounting.experiments.amodo_graph_style import apply_amodo_style, save_plot, style_axis, style_figure_title

LIGHT_AMBER = "#d8c08f"
VARIANTS = [
    ("vanilla", "Stock vLLM", st.AMODO_RULE),
    ("stock+eager", "Stock, eager (no CUDA graphs)", LIGHT_AMBER),
    ("stock+eager+encode", "Stock, eager + encoded", st.AMODO_AMBER),
    ("settings", "PoSM settings", st.AMODO_TEAL),
    ("encoded", "PoSM settings + encoded", st.AMODO_DEEP_TEAL),
    ("dynamic_plain", "+ dynamic region, plain (before v0.17)", st.AMODO_GREEN),
    ("dynamic", "+ dynamic region, encoded (v0.17)", st.AMODO_BLUE),
    ("full", "PoSM settings + encoded + challenged (full)", st.AMODO_ACCENT),
]
SETTINGS = [
    ("batch_invariant", "Batch-invariant kernels"),
    ("eager", "Eager (no CUDA graphs, no torch.compile)"),
    ("triton_attn", "Triton attention"),
    ("token_budget", "Token budget = max length (2048)"),
    ("no_chunked_prefill", "No chunked prefill"),
    ("no_async", "No async scheduling"),
    ("no_prefix_cache", "No prefix caching"),
    ("v1_runner", "V1 model runner"),
]
# one hue, light to dark: fewest to most requests
BATCH_COLOURS = [
    "#b9dcda",
    st.AMODO_TEAL,
    st.AMODO_DEEP_TEAL,
    "#0b4a48",
]


def load(d: Path) -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def by_batch(run: dict) -> dict:
    return {r["batch"]: r for r in run["batches"]}


def header(fig, run: dict, title: str, note: str = ""):
    style_figure_title(fig, f"{title}: {run['model'].split('/')[-1]} on one H200 NVL")
    fig.text(
        0.01,
        0.93,
        f"Each request: {run['prompt_len']} random prompt tokens, {run['gen_len']} generated (greedy), prefill included."
        + (" " + note if note else ""),
        ha="left",
        va="top",
        fontsize=9,
        color=st.AMODO_MUTED,
    )


def bar_labels(ax, xs, ys, fmt):
    for xi, yi in zip(xs, ys):
        if np.isfinite(yi):
            ax.annotate(
                fmt(yi),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color=st.AMODO_INK,
            )


def legend_below(fig, ax, ncol, y=0.06):
    leg = fig.legend(
        *ax.get_legend_handles_labels(), loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y)
    )
    st.register_bbox_extra_artist(fig, leg)


def fig_throughput(runs: dict, tag: str, out: Path):
    shown = [s for s in VARIANTS if s[0] in runs]
    if "vanilla" not in runs or len(shown) < 2:
        return None
    tok = {v: by_batch(runs[v]) for v, _, _ in shown}
    batches = sorted(tok["vanilla"])
    apply_amodo_style()
    fig, (ax, rel) = plt.subplots(1, 2, figsize=(14, 6.2))
    x = np.arange(len(batches))
    w = 0.84 / len(shown)
    others = [s for s in shown if s[0] != "vanilla"]
    wr = 0.84 / len(others)
    for i, (value, label, color) in enumerate(shown):
        ys = [tok[value][b]["tok_s"] if b in tok[value] else np.nan for b in batches]
        es = [tok[value][b]["tok_s_std"] if b in tok[value] else 0 for b in batches]
        ax.bar(
            x + (i - (len(shown) - 1) / 2) * w,
            ys,
            w * 0.92,
            yerr=es,
            color=color,
            label=label,
            edgecolor=st.AMODO_BG,
            linewidth=1,
            error_kw={"ecolor": st.AMODO_INK, "elinewidth": 1, "capsize": 2},
        )
    for i, (value, label, color) in enumerate(others):
        r = [100 * tok[value][b]["tok_s"] / tok["vanilla"][b]["tok_s"] if b in tok[value] else np.nan for b in batches]
        xs = x + (i - (len(others) - 1) / 2) * wr
        rel.bar(xs, r, wr * 0.92, color=color, edgecolor=st.AMODO_BG, linewidth=1)
        bar_labels(rel, xs, r, lambda y: f"{y:.0f}")
    ax.set_xticks(x, [str(b) for b in batches])
    ax.set_yscale("log")
    style_axis(ax, "Throughput", "Concurrent requests", "Generated tokens / s (log scale)", grid_axis="y")
    rel.axhline(100, color=st.AMODO_INK, linewidth=1, linestyle="--")
    rel.annotate(
        "stock vLLM = 100",
        (x[-1] + 0.45, 100),
        textcoords="offset points",
        xytext=(0, 4),
        ha="right",
        fontsize=9,
        color=st.AMODO_MUTED,
    )
    rel.set_xticks(x, [str(b) for b in batches])
    rel.set_ylim(0, 112)
    style_axis(rel, "Relative to stock vLLM", "Concurrent requests", "% of stock vLLM throughput", grid_axis="y")
    legend_below(fig, ax, 3, y=0.03)
    header(fig, runs["vanilla"], "Inference throughput under PoSM", "Error bars: spread across calls.")
    fig.subplots_adjust(top=0.80, bottom=0.2, wspace=0.22)
    return save_plot(fig, out / f"{tag}_throughput.png")


def fig_encoding(runs: dict, tag: str, out: Path):
    pairs = [
        ("stock+eager+encode", "vanilla", "Encoded over stock vLLM (includes going eager)", LIGHT_AMBER),
        ("stock+eager+encode", "stock+eager", "Encoded over stock vLLM, eager", st.AMODO_AMBER),
        ("encoded", "settings", "Encoded over vLLM with the PoSM settings", st.AMODO_DEEP_TEAL),
        (
            "dynamic",
            "dynamic_plain",
            "Dynamic region encoded over plain (KV cache and activations, v0.17)",
            st.AMODO_BLUE,
        ),
    ]
    pairs = [p for p in pairs if p[0] in runs and p[1] in runs]
    if not pairs:
        return None
    batches = sorted(by_batch(runs[pairs[0][1]]))
    apply_amodo_style()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(batches))
    w = 0.8 / len(pairs)
    for i, (a, b, label, c) in enumerate(pairs):
        ta, tb = by_batch(runs[a]), by_batch(runs[b])
        slow = [tb[n]["tok_s"] / ta[n]["tok_s"] if n in ta and n in tb else np.nan for n in batches]
        xs = x + (i - (len(pairs) - 1) / 2) * w
        ax.bar(xs, slow, w * 0.92, color=c, label=label, edgecolor=st.AMODO_BG, linewidth=1)
        bar_labels(ax, xs, slow, lambda y: f"{y:.1f}×")
    ax.axhline(1, color=st.AMODO_INK, linewidth=1, linestyle="--")
    ax.set_xticks(x, [str(b) for b in batches])
    if len(batches) == 1:  # one batch size: a bar of a sensible width, not the whole axis
        ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, None)
    style_axis(
        ax,
        "Slowdown from the encoding",
        "Concurrent requests",
        "Baseline throughput ÷ encoded throughput",
        grid_axis="y",
    )
    legend_below(fig, ax, 1)
    header(
        fig,
        runs[pairs[0][1]],
        "Cost of the incompressible encoding",
        "On stock vLLM it needs eager mode (no CUDA graphs)."
        if any(p[0] == "stock+eager+encode" for p in pairs)
        else "",
    )
    fig.subplots_adjust(top=0.80, bottom=0.26)
    return save_plot(fig, out / f"{tag}_encoding.png")


def fig_settings(runs: dict, tag: str, out: Path):
    singles = [(f"stock+{s}", label) for s, label in SETTINGS if f"stock+{s}" in runs]
    if "vanilla" not in runs or not singles:
        return None
    base = by_batch(runs["vanilla"])
    batches = sorted(base)
    rel = {
        v: [100 * by_batch(runs[v])[b]["tok_s"] / base[b]["tok_s"] for b in batches]
        for v in [v for v, _ in singles] + (["settings"] if "settings" in runs else [])
    }
    order = sorted(singles, key=lambda r: np.mean(rel[r[0]])) + (
        [("settings", "All of them (PoSM settings)")] if "settings" in runs else []
    )
    apply_amodo_style()
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(order) + 2.2))
    y = np.arange(len(order))[::-1]
    h = 0.8 / len(batches)
    for j, b in enumerate(batches):
        vals = [rel[v][j] for v, _ in order]
        ys = y + ((len(batches) - 1) / 2 - j) * h
        ax.barh(
            ys,
            vals,
            h * 0.9,
            color=BATCH_COLOURS[j % len(BATCH_COLOURS)],
            label=f"{b} concurrent",
            edgecolor=st.AMODO_BG,
            linewidth=0.8,
        )
        for yi, vi in zip(ys, vals):
            ax.annotate(
                f"{vi:.0f}",
                (vi, yi),
                textcoords="offset points",
                xytext=(3, 0),
                va="center",
                fontsize=7,
                color=st.AMODO_INK,
            )
    ax.axvline(100, color=st.AMODO_INK, linewidth=1, linestyle="--")
    ax.set_yticks(y, [label for _, label in order])
    ax.set_xlim(0, max(115, max(max(v) for v in rel.values()) + 10))
    style_axis(
        ax,
        "Each setting alone, relative to stock vLLM",
        "% of stock vLLM throughput (100 = no cost)",
        "",
        grid_axis="x",
    )
    legend_below(fig, ax, len(batches), y=0.02)
    header(fig, runs["vanilla"], "Cost of each PoSM vLLM setting")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.32)
    return save_plot(fig, out / f"{tag}_settings.png")


def fig_summary(dirs: list[Path], out: Path):
    models = [(d, load(d)) for d in dirs]
    models = [(d, r) for d, r in models if "vanilla" in r]
    shown = [
        s for s in VARIANTS if s[0] in ("settings", "encoded", "dynamic", "full") and any(s[0] in r for _, r in models)
    ]
    batches = sorted(by_batch(models[0][1]["vanilla"]))
    picks = sorted({batches[0], batches[-1]})  # one panel if the runs had one batch size
    apply_amodo_style()
    fig, axes = plt.subplots(1, len(picks), figsize=(14 if len(picks) > 1 else 9, 5.8), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(models))
    w = 0.8 / len(shown)
    for ax, pick in zip(axes, picks):
        for i, (v, label, c) in enumerate(shown):
            r = [
                100 * by_batch(runs[v])[pick]["tok_s"] / by_batch(runs["vanilla"])[pick]["tok_s"]
                if v in runs
                else np.nan
                for _, runs in models
            ]
            xs = x + (i - (len(shown) - 1) / 2) * w
            ax.bar(xs, r, w * 0.92, color=c, label=label, edgecolor=st.AMODO_BG, linewidth=1)
            bar_labels(ax, xs, r, lambda y: f"{y:.0f}")
        ax.axhline(100, color=st.AMODO_INK, linewidth=1, linestyle="--")
        ax.set_xticks(x, [runs["vanilla"]["model"].split("/")[-1].replace("-Instruct", "") for _, runs in models])
        ax.set_ylim(0, 112)
        style_axis(
            ax,
            f"{pick} concurrent request{'s' if pick > 1 else ''}",
            "Model",
            "% of stock vLLM throughput" if pick == picks[0] else "",
            grid_axis="y",
        )
    legend_below(fig, axes[0], len(shown), y=0.03)
    run = models[0][1]["vanilla"]
    style_figure_title(fig, "Inference throughput under PoSM, by model size: one H200 NVL")
    fig.text(
        0.01,
        0.93,
        f"Each request: {run['prompt_len']} random prompt tokens, {run['gen_len']} generated (greedy), prefill included. "
        "Stock vLLM = 100.",
        ha="left",
        va="top",
        fontsize=9,
        color=st.AMODO_MUTED,
    )
    fig.subplots_adjust(top=0.80, bottom=0.2, wspace=0.08)
    return save_plot(fig, out / "summary.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "throughput")
    ap.add_argument("--summary", action="store_true", help="one figure across the directories (one per model)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary:
        print(fig_summary(args.dir, args.out))
        return
    if len(args.dir) != 1:
        ap.error("one directory, or several with --summary")
    args.dir = args.dir[0]
    runs, tag = load(args.dir), args.dir.name
    for f in (fig_throughput, fig_encoding, fig_settings):
        p = f(runs, tag, args.out)
        if p:
            print(p)


if __name__ == "__main__":
    main()
