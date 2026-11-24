"""The decode profile as figures, from the JSON files scripts/bench_decode.py writes.

  python scripts/plot_decode.py results/decode/kernels_qwen7b.json results/decode/insitu_qwen7b.json [--out docs/figures/decode]

  <tag>_kernel.png  the time to decode Θ' once (one forward step's worth): the kernel in use, the same with coalesced
                    access, with coalesced access and a cheaper tweak (each a scheme change), and the floors
  <tag>_step.png    GPU time per forward step inside vLLM by kind of kernel, with the weights plain and encoded, at
                    each batch size

<tag> is the kernels file's name after "kernels_".
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

# (variant, label, group): group 0 = in use, 1 = same scheme, 2 = scheme change, 3 = floor
KERNEL_ROWS = [
    ("real", "In use: SipHash-2-4 tweak, one thread per 16-word group", 0),
    ("tile", "Same tweak, coalesced loads and stores", 1),
    ("tile-sip24-1w", "Coalesced + SipHash-2-4 of q alone (1-word message)", 2),
    ("tile-sip24-pair", "Coalesced + SipHash-2-4, both halves (one call per 2 words)", 2),
    ("tile-sip13-1w", "Coalesced + SipHash-1-3 of q alone", 2),
    ("tile-chacha12", "Coalesced + ChaCha12 counter mode (8 words a block)", 2),
    ("tile-chacha8", "Coalesced + ChaCha8 counter mode", 2),
    ("tile-square", "Floor: coalesced, squaring and unpacking, no PRF", 3),
    ("dram-copy", "Floor: copying the bytes (HBM bandwidth)", 3),
]
GROUP_COLOURS = [st.AMODO_ACCENT, st.AMODO_AMBER, st.AMODO_DEEP_TEAL, st.AMODO_RULE]
GROUP_LABELS = ["In use", "Same scheme, coalesced", "Coalesced + cheaper tweak (scheme change)", "Floors"]
CATS = [
    ("matmul", "Matmuls", st.AMODO_DEEP_TEAL),
    ("attention", "Attention", st.AMODO_TEAL),
    ("other", "Other kernels", st.AMODO_RULE),
    ("decode", "Decoding Θ'", st.AMODO_ACCENT),
    ("kv", "Decoding and encoding the KV cache and activations", st.AMODO_BLUE),
]


def legend_below(fig, ax, ncol, y=0.02):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), fontsize=9)


def fig_kernel(k: dict, insitu: dict | None, tag: str, out: Path):
    rows = [r for r in KERNEL_ROWS if r[0] in k["variants"]]
    ms = [k["variants"][v]["ms"] for v, _, _ in rows]
    apply_amodo_style()
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2.6))
    y = np.arange(len(rows))[::-1]
    for g in range(4):
        idx = [i for i, r in enumerate(rows) if r[2] == g]
        ax.barh(
            y[idx],
            [ms[i] for i in idx],
            0.7,
            color=GROUP_COLOURS[g],
            label=GROUP_LABELS[g],
            edgecolor=st.AMODO_BG,
            linewidth=0.8,
        )
    base = k["variants"]["real"]["ms"]
    for yi, m in zip(y, ms):
        ax.annotate(
            f"{m:.1f} ms" + (f"  ({base / m:.1f}× faster)" if m < base * 0.95 else ""),
            (m, yi),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
            color=st.AMODO_INK,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": st.AMODO_BG, "edgecolor": "none"},
        )
    if insitu and "1" in insitu.get("plain", {}):
        p = insitu["plain"]["1"]["ms_per_step_wall"]
        ax.axvline(p, color=st.AMODO_INK, linewidth=1, linestyle="--")
        ax.annotate(
            f"a whole plain vLLM step\nat 1 request: {p:.1f} ms",
            (p, y[-1]),
            textcoords="offset points",
            xytext=(30, 0),
            va="center",
            fontsize=8,
            color=st.AMODO_MUTED,
        )
    ax.set_yticks(y, [label for _, label, _ in rows])
    ax.set_xlim(0, max(ms) * 1.3)
    style_axis(ax, "", "ms to decode Θ' once (every forward step decodes all of it)", "", grid_axis="x")
    legend_below(fig, ax, 4, y=-0.01)
    model = (insitu or {}).get("model", tag).split("/")[-1]
    style_figure_title(fig, f"Where the decode kernel's time goes: {model} on one H200 NVL")
    fig.text(
        0.01,
        0.93,
        f"Θ' {k['theta_gb']:.1f} GB ({k['words']:,} words) in {k['launches']} launches, about one layer each; median of "
        f"{len(k['variants']['real']['ms_all'])} passes. Each coalesced variant's output is checked against the one-thread-a-group kernel's.",
        ha="left",
        va="top",
        fontsize=9,
        color=st.AMODO_MUTED,
    )
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.42)
    return save_plot(fig, out / f"{tag}_kernel.png")


def fig_step(insitu: dict, tag: str, out: Path):
    batches = sorted(int(b) for b in insitu["plain"])
    modes = [m for m in (("plain", "plain"), ("encoded", "encoded"), ("dynamic", "+ live")) if m[0] in insitu]
    apply_amodo_style()
    fig, ax = plt.subplots(figsize=(11 if len(modes) < 3 else 13, 5.6))
    x = np.arange(len(batches))
    w = 0.72 / len(modes)
    for j, (mode, label) in enumerate(modes):
        xs = x + (j - (len(modes) - 1) / 2) * w
        bottom = np.zeros(len(batches))
        for c, clabel, colour in CATS:
            v = np.array([insitu[mode][str(b)]["gpu_ms_per_step"].get(c, 0.0) for b in batches])
            if not v.any():
                continue
            ax.bar(xs, v, w * 0.92, bottom=bottom, color=colour, label=clabel, edgecolor=st.AMODO_BG, linewidth=0.8)
            bottom += v
        for xi, b, tot in zip(xs, batches, bottom):
            wall = insitu[mode][str(b)]["ms_per_step_wall"]
            ax.annotate(
                f"{label}\n{wall:.0f} ms",
                (xi, tot),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color=st.AMODO_INK,
            )
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(list(seen.values()), list(seen.keys()), loc="upper left", ncol=2, frameon=False, fontsize=9)
    ax.set_xticks(x, [f"{b} concurrent" for b in batches])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    style_axis(ax, "", "", "GPU ms per forward step", grid_axis="y")
    what = "weights and live memory encoded" if len(modes) > 2 else "weights plain and encoded"
    style_figure_title(fig, f"A vLLM step, {what}: {insitu['model'].split('/')[-1].replace('-Instruct', '')}, H200 NVL")
    fig.text(
        0.01,
        0.93,
        "PoSM vLLM settings in all. Bars: GPU kernel time per forward step (prefill steps included), by kind; "
        "labels: wall time per step."
        + (' "+ live": the KV cache and activation rows held encoded (v0.17).' if len(modes) > 2 else ""),
        ha="left",
        va="top",
        fontsize=9,
        color=st.AMODO_MUTED,
    )
    fig.subplots_adjust(top=0.84, bottom=0.16)
    return save_plot(fig, out / f"{tag}_step.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kernels", type=Path)
    ap.add_argument("insitu", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "decode")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    k = json.loads(args.kernels.read_text())
    ins = json.loads(args.insitu.read_text()) if args.insitu else None
    tag = args.kernels.stem.removeprefix("kernels_")
    print(fig_kernel(k, ins, tag, args.out))
    if ins:
        print(fig_step(ins, tag, args.out))


if __name__ == "__main__":
    main()
