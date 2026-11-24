"""The writeup's timing figures, from saved runs. Writes docs/figures/*.png, which is not tracked: rebuild
the figures with this script.

    python scripts/figures.py extract <live_events.jsonl> <run.npz>     # once per log: the round trips, compactly
    python scripts/figures.py all --runs ../posm_runs                     # every figure whose data is there
    python scripts/figures.py all --live direct --only main_m8_1pct,...  # live figures from the direct-cable run
    python scripts/figures.py all --live onepct --only main_m6_1pct,main_m8_1pct,main_m9,main_m12   # the 1% night run

Each figure makes one point of the timing argument, in plain words (a chain is called a challenge, Δc and F are
spelt out, cheats are named by what they do):

  main_m1  each link served from host memory adds a fixed step (Δc), 1 MiB blocks
  main_m2  one challenge cannot see a 20% cheat; the average of a window can
  main_m3  a labelled run: each cheat fails the windows it overlaps, and only those
  main_m4  windows caught against the share of reads cheated
  main_m5  every known cheat against the three checks (deadline, window test, digest), each on one scale
  main_m6  the honest prover and every cheat at full strength, as challenge-time densities (bench, no network)
  main_m6_1pct, main_m6_10pct  every cheat on 1% / 10% of the weights against honest, as challenge-time densities
  main_m7  one cheat by the share of the weights it cheats on, a density per share on one plot (partial; and
           main_m7_encode_otf, _other_gpu, _dram_copy, _cpu); live if the live run has that cheat's shares, else bench
  main_m8_1pct, main_m8_10pct, main_m8_whole  the window test's mean over N challenges, window to window: honest and
           every cheat at a share, simulated from the live run's challenges drawn in whole stretches (keeps the drift)
  main_m9  windows past the threshold against the window's size, honest and the rebuilt-on-the-fly cheat by share
  main_m10 the smallest share the window test catches at longer challenges (4 to 64 reads), simulated from the live run
  main_m11 the same against a cheater that copies its weights back into HBM as each challenge arrives (4 to 512 reads)
  main_m12 each posture's level through the live run, slot by slot and over a window's time: the drift against the threshold
  appx_a1  Δc by block size
  appx_a2  the share of each cheat's single challenges past the honest p99
  appx_a3  cheat slowdown and honest noise against chain length
  appx_a4  the honest level over 43 hours, and the noise left in an average of n challenges
  appx_a5  READY waits
  appx_a6  challenges a window needs against the cheat to catch, with and without the drift allowance
  appx_a7  median round trip: whole challenge, network (kernel timestamps), network (user space)
  appx_a8  windows that missed a bunched 20% cheat, by calibration

Simulated cheats (fig4, fig5) add Δc to each cheated read of real honest chains, K ~ Binomial(L, F) spread or L
with probability F bunched; the real cheat runs (fig1, fig7) are the check on that.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"
CAPTIONS = True  # the "How to read" caption under each figure


# ------------------------------------------------------------------------------------------------- extraction


def extract(log: Path, out: Path) -> dict:
    """One pass over a live verifier's log: per chain kind, (time s, round trip µs, round id, calibration?) of
    every timed chain (a missing reply as nan), the READY waits, posture changes and window/period verdicts."""
    st, dy, lv, arm_s, arm_d, post, verd, meta = [], [], [], [], [], [], [], {}
    with open(log, "rb") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ty = e.get("type")
            if ty == "verification":
                cal = 1 if e.get("phase") == "calibration" else 0
                if e.get("warmup"):
                    cal = 2
                when, vid = float(e["t"]), int(e["id"])
                if e.get("round", "static") == "static":
                    for c in e.get("challenges", []):
                        r = c.get("rtt_ns")
                        st.append((when, r / 1e3 if r is not None else np.nan, vid, cal))
                    if e.get("arm_ns"):
                        arm_s.append((when, e["arm_ns"] / 1e3, cal))
                else:
                    d = e.get("dyn") or {}
                    r = d.get("rtt_ns")
                    dy.append((when, r / 1e3 if r is not None else np.nan, vid, cal))
                    if "live" in d:
                        r = d["live"].get("rtt_ns")
                        lv.append((when, r / 1e3 if r is not None else np.nan, vid, cal))
                    if d.get("arm_ns"):
                        arm_d.append((when, d["arm_ns"] / 1e3, cal))
            elif ty == "posture":
                post.append((float(e["t"]), str(e.get("posture") or e.get("requested") or "")))
            elif ty in ("claim_verdict", "claim_period_verdict"):
                verd.append(
                    (
                        float(e.get("t", 0)),
                        e.get("chain") or "static",
                        float(e.get("mean_shift", np.nan)),
                        float(e.get("tau", np.nan)),
                        1 if (e.get("pass") if ty == "claim_verdict" else e.get("test") != "fail") else 0,
                        int(e.get("id", -1)),
                    )
                )
            elif ty == "meta" and "claims" in e:
                meta["claims"] = e["claims"]

    def arr(rows, cols):
        return np.array(rows, dtype=np.float64).reshape(-1, cols)

    data = {
        "static": arr(st, 4),
        "dynamic": arr(dy, 4),
        "live": arr(lv, 4),
        "arm_static": arr(arm_s, 3),
        "arm_dynamic": arr(arm_d, 3),
        "posture_t": np.array([p[0] for p in post]),
        "posture": np.array([p[1] for p in post], dtype=object),
        "verdict": np.array([v[:5] for v in verd], dtype=object),
        "verdict_id": np.array([v[5] for v in verd]),
        "meta": np.array(json.dumps(meta)),
    }
    np.savez_compressed(out, **data)
    print(
        f"{out}: static {len(st)}, dynamic {len(dy)}, live {len(lv)} chains; {len(post)} postures; {len(verd)} verdicts"
    )
    return data


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------- shared

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_accounting.experiments import amodo_graph_style as st
from memory_accounting.experiments.amodo_graph_style import apply_amodo_style, save_plot, style_axis, style_figure_title
from memory_accounting.verifier.timing_claim import estimate_baseline, params_from_baseline, plan

L, DC = 8, 17.0  # links per chain; Δc of a 1 MiB block (static, and the dynamic chains at 1 MiB)
F_DEMO, C_DEMO, MARGIN, DRIFT = 0.2, 0.999, 0.1, 4.0
DEADLINE_SIGMAS, DEADLINE_MARGIN = 8.0, 0.05  # the single-chain deadline rule's defaults (memory_accounting.verifier.net baseline)
CHAIN_COLOR = {"static": st.AMODO_DEEP_TEAL, "dynamic": st.AMODO_BLUE, "live": st.AMODO_AMBER}
HONEST, CHEAT, CHEAT2 = st.AMODO_DEEP_TEAL, st.AMODO_ACCENT, st.AMODO_AMBER


def chains(d: dict, kind: str):
    """(calibration µs, run µs, run times s) of one chain kind; warm-up dropped, missing replies kept as nan in the run."""
    a = d[kind]
    cal = a[(a[:, 3] == 1) & np.isfinite(a[:, 1]), 1]
    run = a[a[:, 3] == 0]
    return cal, run[:, 1], run[:, 0]


def claim(cal, drift=DRIFT, F=F_DEMO):
    base = estimate_baseline(cal, L, DC)
    p = params_from_baseline(base, F, C_DEMO, L, DC, sigma_margin=MARGIN, drift=drift)
    return base, p, plan(p)


def deadline(cal) -> float:
    """The single-chain deadline the verifier's sigma rule sets from these calibration chains: median + max(8 σ, 5 %)."""
    from memory_accounting.core.analysis import core_samples

    core = core_samples(cal)
    med = float(np.median(core))
    return med + max(DEADLINE_SIGMAS * float(core.std()), DEADLINE_MARGIN * med)


def windows(x, n):
    """Tumbling windows of n chains (the last partial one dropped)."""
    k = len(x) // n
    return x[: k * n].reshape(k, n)


def cheat_add(rng, shape, F, mode):
    """Extra time of a cheater at F per chain: spread, K ~ Binomial(L, F); bunched, all L reads with probability F."""
    if mode == "spread":
        return DC * rng.binomial(L, F, shape)
    return DC * L * (rng.random(shape) < F)


def window_shift(p, xw):
    """The test's mean shift of each window row: clipped at μ + M, raised to the floor, minus μ."""
    return np.clip(xw, p.floor, p.clip).mean(axis=1) - p.mu_b


def new_fig(w=10, h=5.2, cols=1, rows=1, **kw):
    apply_amodo_style()
    return plt.subplots(rows, cols, figsize=(w, h), **kw)


def titles(fig, title: str, subtitle: str) -> None:
    """The house title, with the subtitle placed for this figure's height (the style's fixed position overlaps
    the title on figures under about 7 inches), and the content kept below both."""
    style_figure_title(fig, title)
    fig._suptitle.set_in_layout(False)  # else tight_layout reserves its height a second time, under the rect
    h = fig.get_size_inches()[1]
    y = 0.985 - 26 / 72 / h
    sub = fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=9.5, color=st.AMODO_MUTED)
    fig._amodo_bbox_extra_artists.append(sub)
    fig._amodo_content_top = y - (14 + 12 * subtitle.count("\n")) / 72 / h


def caption(fig, text: str, y: float = -0.01) -> None:
    """How to read the figure, under it (lower `y` to clear a legend below the axes); none for a live set without them."""
    if not CAPTIONS:
        return
    t = fig.text(0.01, y, text, ha="left", va="top", fontsize=9.5, color=st.AMODO_INK, wrap=True)
    t._get_wrap_line_width = lambda: fig.get_window_extent().width * 0.97
    fig._amodo_bbox_extra_artists.append(t)


def note(ax, x, y, text, color=st.AMODO_INK, **kw):
    """A direct label in data coordinates."""
    kw.setdefault("fontsize", 10)
    return ax.text(x, y, text, color=color, **kw)


def headroom(ax, f=1.35) -> None:
    """Room above the data for direct labels."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * f)


def hist_peak(x, bins, density=False):
    h, e = np.histogram(x, bins, density=density)
    i = int(np.argmax(h))
    return (e[i] + e[i + 1]) / 2, h[i]


def name_of(bb: int) -> str:
    return f"{bb >> 20} MiB" if bb >= 1 << 20 else f"{bb >> 10} KiB"


FIG_RUNS = "figures_20261023"
SETTLE_S = 30  # chains this soon after a posture change are left out of the labelled sets


def run_npz(runs: Path, name: str):
    """A run's extracted arrays (extracted from its .jsonl on first use), or None if the run is not there."""
    d = runs / FIG_RUNS
    npz, log = d / f"{name}.npz", d / f"{name}.jsonl"
    if not npz.exists():
        if not log.exists():
            return None
        extract(log, npz)
    return load(npz)


def label(d: dict, kind: str, settle: float = SETTLE_S):
    """(times, µs, posture) of a run's chains of one kind, each labelled by the posture in force (calibration and
    the first `settle` seconds after a change left out). Before the first posture event the prover is honest."""
    a = d[kind]
    a = a[(a[:, 3] == 0) & np.isfinite(a[:, 1])]
    pt, pp = d["posture_t"].astype(float), d["posture"].astype(str)
    idx = np.searchsorted(pt, a[:, 0], side="right") - 1
    lab = np.where(idx < 0, "honest", pp[np.clip(idx, 0, None)])
    since = a[:, 0] - np.where(idx < 0, -1e18, pt[np.clip(idx, 0, None)])
    keep = since > settle
    return a[keep, 0], a[keep, 1], lab[keep]


# ================================================================================================= plain words

# What each prover posture does, in the words the figures use.
POSTURE = {
    "honest": "honest",
    "evict_dram": "20% of the weights in host memory",
    "dyn_dram": "half the KV cache in host memory",
    "tamper": "weights altered (no slowdown)",
    "encode_otf": "weights rebuilt on the fly",
    "dram_mapped": "all weights read from host memory",
    "dram_copy": "all weights copied in per read",
    "cpu": "computed on the CPU",
    "partial": "10% of the weights in host memory",
    "lora": "5% of weight blocks altered",
}


def pct(v, _=None) -> str:
    return f"{v * 100:.3g}%"


def thousands(v, _=None) -> str:
    return f"{v:,.0f}"


# ================================================================================================= main text


def main_m1(runs: Path, period, rng):
    """M1: what one cheated read costs at 1 MiB blocks (the claim's Δc)."""
    bench = _bench(runs)
    if bench is None or 1 << 20 not in bench:
        return None
    fig, ax = new_fig(9, 5.2)
    sl, da = _delta_c_line(ax, bench[1 << 20], CHEAT)
    note(ax, 0.3, sl * 7.2, f"+{sl:.0f} µs for each link\nserved from host memory", CHEAT, fontsize=11)
    ax.set_xticks(range(L + 1))
    style_axis(ax, "", f"links of the {L}-link chain served from host memory", "extra round-trip time (µs)")
    titles(
        fig,
        "Each link served from host memory adds a fixed step of time",
        "1 MiB blocks on the H200; each dot is the median of 1,000 challenges, the band the middle 80%",
    )
    caption(
        fig,
        f"How to read: the slower a cheater's memory, the longer a challenge takes, one fixed step per link. The claim uses the "
        f"smallest step measured, {da:.0f} µs (the whole KV cache in host memory).",
    )
    return save_plot(fig, OUT / "main_m1_cost_per_read.png")


def main_m2(runs: Path, period, rng):
    """M2: one chain cannot show a partial cheat; the mean of a window can."""
    if period is None:
        return None
    cal, run, _ = chains(period, "static")
    run = run[np.isfinite(run)]
    _base, params, pl = claim(cal)
    N = int(pl["n_required"])
    tau = pl["tau"]
    fig, (ax1, ax2) = new_fig(14, 5.2, 2)
    cheats = ((0.1, CHEAT2), (F_DEMO, CHEAT))  # (share of reads from host memory, colour)
    # a: single chains, honest against each cheat
    one = [run + cheat_add(rng, run.shape, f, "spread") for f, _ in cheats]
    dl = deadline(cal)
    bins = np.linspace(np.percentile(run, 0.2), max(np.percentile(one[-1], 99), dl + 12), 90)
    ax1.hist(run, bins, density=True, color=HONEST, alpha=0.45)
    xh, yh = hist_peak(run, bins, True)
    note(ax1, xh - 3, yh * 1.02, "honest", HONEST, ha="right", fontsize=11)
    # point on the curve, label height (× honest peak)
    for x, (f, col), (px, ty) in zip(one, cheats, ((1322, 0.85), (1345, 0.6))):
        h_, e_ = np.histogram(x, bins, density=True)
        ax1.hist(x, bins, density=True, histtype="step", lw=2, color=col)
        py = float(h_[min(np.searchsorted(e_, px), len(h_)) - 1])
        ax1.annotate(
            f"{f:.0%} of reads cheated",
            xy=(px, py),
            xytext=(1352, ty * yh),
            color=col,
            fontsize=11,
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.8},
        )
    headroom(ax1, 1.1)
    ax1.axvline(dl, color=st.AMODO_INK, lw=1.4, ls="--")
    missed = 1 - np.mean([np.mean(x > dl) for x in one])
    note(
        ax1,
        dl - 2,
        ax1.get_ylim()[1] * 0.97,
        f"deadline\n{dl:,.0f} µs",
        st.AMODO_INK,
        ha="right",
        va="top",
        fontsize=11,
    )
    note(
        ax1,
        dl - 2,
        ax1.get_ylim()[1] * 0.30,
        f"lets {missed:.0%} of\ncheated challenges\nthrough",
        st.AMODO_INK,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax1.set_yticks([])
    style_axis(ax1, "a · one challenge", "round-trip time (µs)", "how often")
    # b: window means, honest against the same cheats
    w = windows(run, N)
    hon = window_shift(params, w)
    avg = [window_shift(params, w + cheat_add(rng, w.shape, f, "spread")) for f, _ in cheats]
    bins = np.linspace(min(hon.min(), -16), avg[-1].max() + 2, 90)
    groups = [(hon, HONEST, f"honest\n{np.mean(hon >= tau):.0%} flagged", 0)]
    groups += [
        (x, col, f"{f:.0%} cheated\n{np.mean(x >= tau):.0%} flagged", dx)
        for x, (f, col), dx in zip(avg, cheats, (3.5, 0))
    ]
    for x, col, lab, dx in groups:
        ax2.hist(x, bins, color=col, alpha=0.55)
        xp, yp = hist_peak(x, bins)
        note(ax2, xp + dx, yp * 1.04, lab, col, ha="center", va="bottom", fontsize=11)
    headroom(ax2, 1.35)
    ax2.axvline(tau, color=st.AMODO_INK, lw=1.4, ls="--")
    note(ax2, tau - 0.5, ax2.get_ylim()[1] * 0.97, "alarm", st.AMODO_INK, ha="right", va="top", fontsize=11)
    ax2.set_yticks([])
    style_axis(ax2, f"b · average of {N} challenges", "average slowdown (µs)", "how often")
    titles(
        fig,
        "One challenge cannot see a partial cheat; the average of a few hundred can",
        f"43 honest hours on the H200; each cheat serves a share of reads from host memory, {DC:g} µs slower each",
    )
    caption(
        fig,
        f"How to read: (a) cheaters' challenges are only slightly slower and overlap the honest ones, so almost all of them beat "
        f"the single-challenge deadline. (b) Averaged over "
        f"{N} challenges, the honest noise shrinks but a cheater's slowdown does not: the 20% cheat is always flagged, "
        f"the 10% one about half the time.",
    )
    return save_plot(fig, OUT / "main_m2_chain_vs_window.png")


def main_m3(runs: Path, period, rng):
    """M3: a labelled run, cheats switched on and off."""
    run = run_npz(runs, "C_vllm_timeline")
    if run is None or not len(run["posture_t"]):
        return None
    fig, axes = new_fig(14, 6.6, 1, 2, sharex=True)
    pt, pp = run["posture_t"].astype(float), run["posture"].astype(str)
    a0 = run["static"]
    t0 = a0[a0[:, 3] == 0, 0].min()
    shade = {"evict_dram": CHEAT, "dyn_dram": st.AMODO_BLUE, "tamper": CHEAT2}
    for ax, kind in zip(axes, ("static", "dynamic")):
        a = run[kind]
        a = a[(a[:, 3] == 0) & np.isfinite(a[:, 1])]
        ax.plot((a[:, 0] - t0) / 60, a[:, 1], ".", ms=1.2, color=st.AMODO_MUTED, alpha=0.3, rasterized=True)
        lo, hi = np.percentile(a[:, 1], 0.5), np.percentile(a[:, 1], 99.5)
        ax.set_ylim(lo - 45, hi + 70)
        for i, (t, pst) in enumerate(zip(pt, pp)):
            end = pt[i + 1] if i + 1 < len(pt) else a[:, 0].max()
            if pst in shade:
                ax.axvspan((t - t0) / 60, (end - t0) / 60, color=shade[pst], alpha=0.10)
                if kind == "static":
                    note(
                        ax,
                        ((t + end) / 2 - t0) / 60,
                        hi + 60,
                        POSTURE[pst],
                        shade[pst],
                        ha="center",
                        va="top",
                        fontsize=10,
                    )
        v, vid_ = run["verdict"], run["verdict_id"]
        ends = [run[kind][run[kind][:, 2] == vid, 0] for row, vid in zip(v, vid_) if row[1] == kind]
        ends = [e[0] for e in ends if len(e)]
        start = ends[0] - np.median(np.diff(ends))  # a window runs from the previous verdict to its own
        for row, vid in zip(v, vid_):
            if row[1] != kind:
                continue
            vt = run[kind][run[kind][:, 2] == vid, 0]
            if len(vt):
                ok = bool(row[4])
                ax.plot(
                    ((start + vt[0]) / 2 - t0) / 60,
                    lo - 28,
                    "o" if ok else "X",
                    color=st.AMODO_GREEN if ok else CHEAT,
                    ms=7 if ok else 10,
                )
                start = vt[0]
        name = {"static": "a · challenges of the weights", "dynamic": "b · challenges of the KV cache"}[kind]
        style_axis(ax, name, "minutes into the run" if kind == "dynamic" else "", "round trip (µs)")
    titles(
        fig,
        "Each cheat is caught while it runs, and only then",
        "vLLM on the H200, cheats switched on and off by hand; grey dots: single challenges; "
        "markers below: each window's verdict (● pass, × fail)",
    )
    caption(
        fig,
        "How to read: moving weights to host memory slows the weight challenges (a); moving the KV cache slows the KV-cache "
        "challenges (b). Every window that overlaps either cheat fails; every other window passes. Altering the weights "
        "costs no time, so timing cannot see it: the replay of the records catches it instead.",
    )
    return save_plot(fig, OUT / "main_m3_labelled_run.png")


def main_m4(runs: Path, period, rng):
    """M4: detection against the true share of cheating."""
    if period is None:
        return None
    from matplotlib.ticker import FuncFormatter

    cal, run, _ = chains(period, "static")
    run = run[np.isfinite(run)]
    _base, params, pl = claim(cal)
    N = int(pl["n_required"])
    tau = pl["tau"]
    w0 = windows(run, N)
    Fs = np.round(np.arange(0.0, 0.305, 0.005), 3)
    det, bun = (
        np.array([np.mean(window_shift(params, w0 + cheat_add(rng, w0.shape, F, mode)) >= tau) for F in Fs])
        for mode in ("spread", "bunched")
    )
    fig, ax = new_fig(9, 5.2)
    ax.plot(Fs, bun, color=CHEAT2, lw=1.6)
    ax.plot(Fs, det, color=CHEAT, lw=2.4)
    ax.annotate(
        "cheated reads spread\nover all challenges",
        xy=(0.085, float(np.interp(0.085, Fs, det))),
        xytext=(0.005, 0.66),
        color=CHEAT,
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": CHEAT, "lw": 0.8},
    )
    ax.annotate(
        "cheated reads bunched into\nwhole challenges (worst case)",
        xy=(0.16, float(np.interp(0.16, Fs, bun))),
        xytext=(0.215, 0.5),
        color=CHEAT2,
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": CHEAT2, "lw": 0.8},
    )
    ax.axvline(F_DEMO, color=st.AMODO_MUTED, lw=1, ls="--")
    note(ax, F_DEMO + 0.004, 0.08, "target", st.AMODO_MUTED, fontsize=11)
    f50 = float(np.interp(0.5, det, Fs))
    ax.plot([f50], [0.5], "o", color=CHEAT, ms=7)
    note(ax, f50 + 0.008, 0.46, f"half caught at {f50:.0%}", st.AMODO_INK, fontsize=11)
    note(
        ax,
        F_DEMO + 0.008,
        0.9,
        f"spread: all caught at the target\nbunched: {1 - float(np.interp(F_DEMO, Fs, bun)):.1%} missed (see A8)",
        st.AMODO_INK,
        va="top",
        fontsize=10,
    )
    note(
        ax,
        0.215,
        0.24,
        f"honest windows flagged:\n{int(np.sum(window_shift(params, w0) >= tau))} of {len(w0)}",
        HONEST,
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    style_axis(ax, "", "share of reads the cheater serves from host memory", "windows caught")
    titles(
        fig,
        "The more a prover cheats, the more surely it is caught",
        f"real honest challenges (43 hours) with a simulated cheat added; windows of {N} challenges; test sized for a 20% cheat",
    )
    caption(
        fig,
        "How to read: a prover cheating on 20% of reads or more fails nearly every window, however it places the cheated reads; "
        "small cheats slip through, which is why the claim names a target share.",
    )
    return save_plot(fig, OUT / "main_m4_detection.png")


def digest_checks(runs: Path, name: str) -> dict:
    """{posture: (checks, wrong digests)} of a run, each check labelled by the posture in force when its verification was sent."""
    log = runs / FIG_RUNS / f"{name}.jsonl"
    post, sent, out = [], {}, {}
    if not log.exists():
        return out
    checks = []
    with open(log, "rb") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ty = e.get("type")
            if ty == "posture":
                post.append((float(e["t"]), str(e.get("posture") or e.get("requested") or "")))
            elif ty == "verification":
                sent[e["id"]] = float(e["t"])
            elif ty == "check":
                checks.append((e["id"], bool(e.get("digest_ok", True))))
    pt = [p[0] for p in post]
    for vid, ok in checks:
        if vid not in sent:
            continue
        i = int(np.searchsorted(pt, sent[vid], side="right")) - 1
        n, bad = out.get(post[i][1] if i >= 0 else "honest", (0, 0))
        out[post[i][1] if i >= 0 else "honest"] = (n + 1, bad + (not ok))
    return out


M5_RUNS = [
    ("vLLM serving", "C_vllm_timeline", [("evict_dram", "static"), ("dyn_dram", "dynamic"), ("tamper", "static")]),
    (
        "simulation prover",
        "D_sim_cheats",
        [
            ("partial", "static"),
            ("dram_mapped", "static"),
            ("encode_otf", "static"),
            ("dram_copy", "static"),
            ("cpu", "static"),
            ("lora", "static"),
        ],
    ),
]
M5_NAME = {**POSTURE, "tamper": "weights altered in place (and the replay)"}
SOME_LATE = 0.05  # a cheat with this share of its challenges late is flagged by the deadline some of the time


def cheat_rows(runs: Path) -> list:
    """Every labelled cheat, and honest running of each run, against the three checks: one challenge's deadline, the
    window test and the digest. Timing cheats come first, the slowest first; cheats that cost no time come last."""
    honest, timed, free = [], [], []
    for run_name, run, posts in M5_RUNS:
        run_data = run_npz(runs, run)
        if run_data is None or not len(run_data["posture_t"]):
            continue
        claims = json.loads(str(run_data["meta"])).get("claims", {})
        dig = digest_checks(runs, run)
        for pst, kind in [("honest", "static")] + posts:
            _, x, lab = label(run_data, kind)
            xs, h = x[lab == pst], x[lab == "honest"]
            if len(xs) < 20:
                continue
            claim_info = claims[kind]
            a = run_data[kind]
            med = np.median(h)
            dl = deadline(a[(a[:, 3] == 1) & np.isfinite(a[:, 1]), 1]) - med
            n, bad = dig.get(pst, (0, 0))
            r = {
                "name": run_name if pst == "honest" else M5_NAME[pst],
                "honest": pst == "honest",
                "sl": xs - med,
                "dl": dl,
                "late": float(np.mean(xs - med > dl)),
                "shift": float(
                    np.clip(xs, claim_info["floor_us"], claim_info["clip_us"]).mean() - claim_info["mu_b_us"]
                ),
                "tau": claim_info["tau_us"],
                "cap": claim_info["clip_us"] - claim_info["mu_b_us"],
                "checks": n,
                "wrong": bad,
            }
            (honest if r["honest"] else timed if r["shift"] >= r["tau"] else free).append(r)
    return (
        honest
        + sorted(timed, key=lambda r: -np.median(r["sl"]))
        + sorted(free, key=lambda r: -r["wrong"] / max(r["checks"], 1))
    )


def main_m5(runs: Path, period, rng):
    """M5: every cheat we know of against the three checks, each on one scale: 1 is where the check raises the alarm."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    rows = cheat_rows(runs)
    if len(rows) < 3:
        return None
    lines = ["honest"] + [r for r in rows if r["honest"]] + ["cheats"] + [r for r in rows if not r["honest"]]
    fig, axes = new_fig(13, 0.36 * len(lines) + 2.1, 3, sharey=True)
    passed = "#b3afa3"
    hi = 300
    ticks = []
    for i, r in enumerate(lines):
        y = len(lines) - 1 - i
        if isinstance(r, str):
            ticks.append((y, r, True))
            continue
        ticks.append((y, r["name"], False))
        ax1, ax2, ax3 = axes
        sl = r["sl"] / r["dl"]
        col = HONEST if r["honest"] else CHEAT if r["late"] >= 0.5 else passed
        ax1.plot(
            [np.percentile(sl, 10), np.percentile(sl, 90)], [y, y], color=col, lw=6, alpha=0.35, solid_capstyle="butt"
        )
        ax1.plot([np.median(sl)], [y], "o", color=col, ms=7)
        if not r["honest"] and SOME_LATE <= r["late"] < 0.5:
            note(
                ax1, np.percentile(sl, 90) * 1.3, y, f"{r['late']:.0%} late", st.AMODO_MUTED, va="center", fontsize=9.5
            )
        w = r["shift"] / r["tau"]
        ax2.plot([w], [y], "o", color=HONEST if r["honest"] else CHEAT if w >= 1 else passed, ms=7)
        ax3.plot([r["wrong"]], [y], "o", color=HONEST if r["honest"] else CHEAT if r["wrong"] else passed, ms=7)
        if r["wrong"]:
            far = r["wrong"] > 30  # label to the left of a dot near the right edge
            note(
                ax3,
                r["wrong"] / 1.5 if far else r["wrong"] * 1.6,
                y,
                f"{r['wrong']:,} of {r['checks']:,}",
                st.AMODO_MUTED,
                va="center",
                ha="right" if far else "left",
                fontsize=9.5,
            )
    top = len(lines) - 1
    for ax, title, xl, line in (
        (axes[0], "a · one challenge", "time over honest, × the deadline", "deadline"),
        (axes[1], "b · window average", "slowdown, × the alarm", "alarm"),
        (axes[2], "c · digest", "wrong digests", "one wrong"),
    ):
        ax.axvspan(1, hi, color=CHEAT, alpha=0.06, lw=0)
        ax.axvline(1, color=st.AMODO_INK, lw=1.2, ls=(0, (2, 1.5)))
        note(ax, 1.15, top, line, st.AMODO_INK, va="center", fontsize=10)
        ax.set_xscale("symlog", linthresh=1, linscale=1.0)
        ax.xaxis.set_major_locator(FixedLocator([0, 1, 10, 100]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlim(-2.5, hi)
        style_axis(ax, title, xl, "", grid_axis="x")
        ax.tick_params(axis="y", length=0)
    ax1 = axes[0]
    ax1.set_yticks([y for y, _, _ in ticks])
    ax1.set_yticklabels([t for _, t, _ in ticks])
    for tick, (_, _, head) in zip(ax1.get_yticklabels(), ticks):
        tick.set_color(st.AMODO_MUTED if head else st.AMODO_INK)
        tick.set_fontstyle("italic" if head else "normal")
    ax1.set_ylim(-0.6, len(lines) - 0.4)
    titles(
        fig,
        "Every cheat we know of is caught",
        "labelled runs on the H200, each check on one scale: past 1 (shaded) it raises the alarm; red: caught by that check; "
        "bar: middle 80% of challenges",
    )
    caption(
        fig,
        "How to read: a cheat is caught when its dot is red in any panel. Cheats that cost time all fail the window (b); "
        "the big ones also miss the deadline (a), and some challenges of the middling ones do. Cheats that cost no time "
        "come back with wrong digests (c). The window counts a challenge at most 136 µs late, so the biggest cheats share "
        "that cap (about 14× the alarm). Honest windows of the simulation run sit near the alarm because it was calibrated "
        "at start-up (A8).",
    )
    return save_plot(fig, OUT / "main_m5_every_cheat.png")


# ------------------------------------------------------------------------------------------------- cheats by fraction
# A bench sweep on the GPU host (memory_accounting.experiments.bench, no network): the honest prover and each cheat answer the same
# challenges, interleaved sample by sample. name@f is the cheat on a share f of the array; every fraction of a cheat
# is answered by one instance of it, and partial@0 (the same table, nothing on the host) is the honest reference.
# The other cheats' shares (SHARE_RUNS) are answered by one honest instance, which also answers honest: their reference.
FRAC_RUNS = "fractions_20261027"
SHARE_RUNS = "shares_20261028"
SHARE_CHEATS = ("partial", "encode_otf", "other_gpu", "dram_copy", "cpu")
SHARE_STEP = {
    "partial": "read from host memory",
    "encode_otf": "rebuilt on the fly",
    "other_gpu": "gathered from the other GPU's copy",
    "dram_copy": "copied in from host memory",
    "cpu": "computed on the CPU",
}
FRACTIONS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
CHEAT_NAME = {
    "partial": "weights read from host memory",
    "dram_mapped": "weights read from host memory",
    "dram_copy": "weights copied in from host memory per read",
    "encode_otf": "weights rebuilt on the fly",
    "cpu": "computed on the CPU",
    "other_gpu": "weights on another GPU (simulated on this one)",
}
CHEAT_COL = {
    "dram_mapped": "#7a4f93",
    "partial": "#7a4f93",
    "encode_otf": st.AMODO_AMBER,
    "dram_copy": st.AMODO_BLUE,
    "cpu": "#b05f2d",
    "other_gpu": st.AMODO_ACCENT,
}
DEADLINE_Q = 0.999


# labelled live runs (the verifier over the LAN, kernel timestamps) cycling the
FRAC_LIVE = (
    "shares_live",
    "fractions_live",
)
# postures: every cheat whole and the others' shares; partial's shares
# v0.17 (dynamic memory encoded), shapes from the sweeps, every cheat at 1%        # over a direct cable, 128 reads a
# challenge: every cheat at 1% only
LIVE_SETS = {
    "lan": FRAC_LIVE,  # `all --live`: which labelled live runs the live figures use
    "direct": ("direct_live",),  # over a direct cable, every cheat and its shares, partial's too
    "onepct": ("onepct_live",),
    "v017": ("v017_live",),
    "v019": ("v019_live",),  # v0.19: the same, without the live chain
}
LIVE_LINK = {
    "lan": "over the LAN",
    "direct": "over a direct cable",
    "onepct": "over a direct cable",
    "v017": "over a direct cable",
    "v019": "over a direct cable",
}
# reads per static challenge (None: the run's own claim's L)
LIVE_READS = {
    "lan": 8,
    "direct": 8,
    "onepct": 128,
    "v017": None,
    "v019": 128,  # its claim records no L: its window test was off
}
LIVE_CYCLE = {
    "onepct": "postures in a fresh random order each round, 40 s each",
    "v017": "postures in a fresh random order each round, 40 s each",
    "v019": "postures in a fresh random order each round, 40 s each",
}
LIVE_NO_CAPTION = {"onepct", "v017", "v019"}  # live sets whose figures go without the "How to read" caption
LIVE_WHEN = {"v019": "the run"}  # what M12 calls the run's span, if not the night: v0.19's ran by day
LINK, READS, CYCLE = LIVE_LINK["lan"], LIVE_READS["lan"], "postures cycled in turn, 40 to 60 s each"
WHEN = "the night"
FRAC_SETTLE_S = 2.0  # the share postures swap a table between challenges: nothing to settle but the switch itself


def fraction_sweep(runs: Path, cheat: str | None = None):
    """(live?, {posture: (round trips µs, share reads per challenge or None)}, deadline µs, its name) with the honest
    reference under "ref": the live run if there is one and it has this cheat's shares (cheat None: any live run;
    honest posture, and the deadline the verifier set from its calibration), else the bench sweeps (the reference
    timed by the cheat's own instance, and its 99.9th percentile). None if neither is there."""
    for name in FRAC_LIVE:
        live = runs / name / "run.npz"
        if not live.exists() and (runs / name / "run.jsonl").exists():
            extract(runs / name / "run.jsonl", live)
        if not live.exists():
            continue
        data = load(live)
        _, x, lab = label(data, "static", settle=FRAC_SETTLE_S)
        if cheat is None or any(str(p).startswith(f"{cheat}@") for p in np.unique(lab)):
            out = {p: (x[lab == p], None) for p in np.unique(lab)}
            out["ref"] = out.pop("honest")
            a = data["static"]
            return (
                True,
                out,
                deadline(a[(a[:, 3] == 1) & np.isfinite(a[:, 1]), 1]),
                "deadline (the verifier's, set at calibration)",
            )
    out = {}
    for run, ref in ((SHARE_RUNS, "honest"), (FRAC_RUNS, "partial@0")):  # the partial sweep's whole cheats win
        f = runs / run / ("share_sweep.json" if run == SHARE_RUNS else "fraction_sweep.json")
        if f.exists():
            data = {
                r["prover"]: (np.array(r["samples_ns"]) / 1e3, np.array(r["host_links"]) if "host_links" in r else None)
                for r in json.loads(f.read_text())["records"]
            }
            out.update(data)
            for c in SHARE_CHEATS:
                if any(k.startswith(f"{c}@") and k != "partial@0" for k in data):
                    out[f"ref:{c}"] = data[ref]
    if "partial@0" not in out and cheat is None:
        return None
    key = f"ref:{cheat}" if cheat else "ref:partial"
    if key not in out:
        return None
    out["ref"] = out[key]
    return False, out, float(np.quantile(out["ref"][0], DEADLINE_Q)), "deadline (honest 99.9th percentile)"


def _source(live: bool) -> str:
    return (
        f"live: the verifier challenging the H200 {LINK}, kernel timestamps; {CYCLE}"
        if live
        else "the H200 bench, no network: the same 5,000 challenges to each, interleaved"
    )


def _kde(x, grid, bw):
    """Gaussian density of all samples (none dropped: a cheat's slow challenges are the point) at a fixed bandwidth."""
    from scipy.stats import gaussian_kde

    return gaussian_kde(x, bw_method=bw / x.std())(grid)


def main_m6_every_cheat(runs: Path, period, rng):
    """M6: the honest prover and every cheat at full strength (the whole array), as challenge-time densities."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    got = fraction_sweep(runs)
    if got is None:
        return None
    live, sw, dl, dl_name = got
    h = sw["ref"][0]
    cheats = [c for c in ("encode_otf", "dram_mapped", "other_gpu", "dram_copy", "cpu") if c in sw]
    fig, ax = new_fig(12, 5.6)
    lg = np.log10
    grid = np.linspace(lg(1000), lg(10000), 4000)
    rows = []
    for name, x, col in [("honest", h, HONEST)] + [(c, sw[c][0], CHEAT_COL[c]) for c in cheats]:
        y = _kde(lg(x), grid, 0.0025)
        y /= y.max()
        # dashed: it lies on encode_otf
        ax.plot(10**grid, y, color=col, lw=2, ls=(0, (4, 2)) if name == "dram_mapped" else "-")
        ax.fill_between(10**grid, y, color=col, alpha=0.15, lw=0)
        ax.plot([x.mean()] * 2, [0, 1], color=col, lw=1, ls=(0, (3, 2)))
        rows.append((name, x, col))
    ax.axvline(dl, color=st.AMODO_INK, lw=1.2, ls=":")
    note(ax, dl * 1.01, 1.08, dl_name, st.AMODO_INK, va="bottom", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks([1000, 2000, 5000, 10000])
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1000, 10000)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    style_axis(
        ax, "", "time to answer one challenge (µs, log scale)", "density (each scaled to its peak)", grid_axis="x"
    )
    handles = [plt.Line2D([], [], color=c, lw=2.5, ls=(0, (4, 2)) if n == "dram_mapped" else "-") for n, _, c in rows]
    labels = [
        ("honest" if n == "honest" else CHEAT_NAME[n])
        + f"   µ {x.mean():,.0f} µs  σ {x.std():.0f} µs"
        + ("" if n == "honest" else f"   {np.mean(x > dl):.0%} past the deadline")
        for n, x, _ in rows
    ]
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    titles(
        fig,
        "Every cheat against the honest prover",
        f"{_source(live)}; 1 MiB blocks, 8 reads a challenge; dashed: mean",
    )
    caption(
        fig,
        "How to read: each cheat keeps the whole array somewhere other than the GPU's own memory, or rebuilds it, and pays for "
        "it on every read. "
        + (
            "Every one is clear of the honest peak. The slow ones (copying in, the CPU) miss the deadline every time; the "
            "two near the honest peak miss it only some of the time, because the deadline leaves room for the network's "
            "jitter, and the window test, which compares means, is what catches them reliably. "
            if live
            else "All of them sit well past the slowest honest challenges. "
        )
        + "Reading from host memory and rebuilding on the fly take about the same time on this GPU (the dashed curve "
        "lies on the amber one)."
        + (
            " The other-GPU copy is simulated on this GPU (the test host has one), so a real one over NVLink would be slower still."
            if "other_gpu" in cheats
            else ""
        ),
    )
    return save_plot(fig, OUT / "main_m6_every_cheat_density.png")


def main_m6_share(runs: Path, period, rng, share: float = 0.01):
    """M6 at a share: the honest prover and every cheat on a share of the weights, as challenge-time densities (log
    density: only the challenges that read the share pay, the rest lie on the honest curve). Each cheat is measured
    against the honest prover and the deadline of its own run (host memory's shares are in an earlier live run)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows, runs_of = [], {}
    for cheat in ("encode_otf", "partial", "other_gpu", "dram_copy", "cpu"):
        got = fraction_sweep(runs, cheat)
        if got is None or f"{cheat}@{share:g}" not in got[1]:
            continue
        live, sw, dl, dl_name = got
        runs_of[cheat] = (live, sw["ref"][0], dl, dl_name)
        rows.append((cheat, sw[f"{cheat}@{share:g}"][0]))
    if len(rows) < 2:
        return None
    base = rows[0][0]
    live, honest_run, dl, dl_name = runs_of[base]

    def same(r):
        return len(r[1]) == len(honest_run) and np.array_equal(r[1], honest_run)  # measured in the first cheat's run

    mixed = not all(same(r) for r in runs_of.values())
    lg = np.log10
    # 8 reads: 1 to 10 ms on a log axis; a longer challenge: honest to the slowest cheats, linear
    logx = np.median(honest_run) < 5000
    if logx:
        grid = np.linspace(lg(1000), lg(10000), 4000)
        bw = 0.004 if live else 0.0015
        kde, xg = (lambda x: _kde(lg(x), grid, bw)), 10**grid
    else:
        lo, hi = np.quantile(honest_run, 0.0005) - 30, max(max(np.quantile(x, 0.9995) for _, x in rows), dl) * 1.02
        grid = np.linspace(lo, hi, 6000)
        bw = 6.0  # µs: under the honest spread, over the network's tick
        kde, xg = (lambda x: _kde(x, grid, bw)), grid
    fig, ax = new_fig(12, 5.6)
    curves = [("honest", honest_run, HONEST)] + [(c, x, CHEAT_COL[c]) for c, x in rows]
    for name, x, col in curves:
        y = kde(x)
        ax.plot(xg, y, color=col, lw=2, ls=(0, (4, 2)) if name == "partial" else "-")
        ax.fill_between(xg, y, color=col, alpha=0.10, lw=0)
    top = max(line.get_ydata().max() for line in ax.lines)
    one = 1 / (min(len(x) for _, x, _ in curves) * math.sqrt(2 * math.pi) * bw)  # the density one challenge makes
    ax.axvline(dl, color=st.AMODO_INK, lw=1.2, ls=":")
    note(ax, dl * 1.01 if logx else dl + 0.006 * (hi - lo), top * 1.3, dl_name, st.AMODO_INK, va="bottom", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(max(2 * one, top * 3e-4), top * 2.5)
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(1000, 10000)
        ax.set_xticks([1000, 2000, 5000, 10000])
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    style_axis(
        ax,
        "",
        "time to answer one challenge (µs" + (", log scale)" if logx else ")"),
        "density (log scale)",
        grid_axis="x",
    )
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    handles = [plt.Line2D([], [], color=c, lw=2.5, ls=(0, (4, 2)) if n == "partial" else "-") for n, _, c in curves]
    labels = ["honest"]
    for cheat, x in rows:
        _, hc, dlc, _ = runs_of[cheat]
        labels.append(
            f"{CHEAT_NAME[cheat]}   mean {x.mean() - hc.mean():+,.0f} µs   {np.mean(x > dlc):.1%} past the deadline"
            + ("" if same(runs_of[cheat]) else "*")
        )
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    touch = 1 - (1 - share) ** READS
    titles(
        fig,
        f"Every cheat on {share:.0%} of the weights against the honest prover",
        f"{_source(live)}; 1 MiB blocks, {READS} reads a challenge"
        + ("; * an earlier live run, against its own honest prover" if mixed else ""),
    )
    other = [r for r in runs_of.values() if not same(r)]
    caption(
        fig,
        f"How to read: each cheat keeps {share:.0%} of the weights somewhere other than the GPU's own memory, or rebuilds "
        f"them. A challenge reads {READS} blocks at random, so the {touch:.0%} of challenges that read that share pay for it and "
        "the rest lie on the honest curve; the density is on a log scale so that the slow ones show. Copying in and the "
        "CPU put each challenge that reads the share well past the honest ones. Rebuilding, the other GPU's copy and "
        "reading host memory directly cost about 20 µs per block read, inside the network's jitter: the window test, "
        "which compares means, is what can see them."
        + (
            f" * Host memory's shares are from an earlier live run (honest mean {other[0][1].mean():,.0f} µs there, "
            f"{honest_run.mean():,.0f} µs here); its mean and share past the deadline are against that run's honest prover and deadline."
            if other
            else ""
        ),
    )
    return save_plot(fig, OUT / f"main_m6_every_cheat_{share * 100:g}pct.png")


# 1% .. 100%, validated for colour blindness
FRACTION_COL = (
    "#2a78d6",
    "#eb6834",
    "#6250d6",
    "#eda100",
    "#e34948",
    "#8a2f5e",
)


def main_m7_fraction(runs: Path, period, rng, cheat: str = "partial"):
    """M7: one cheat on a growing share of the array: a challenge-time density per share, on one plot over honest."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    got = fraction_sweep(runs, cheat)
    if got is None:
        return None
    live, sw, dl, dl_name = got
    whole = "dram_mapped" if cheat == "partial" else cheat  # 100%: the share posture, else the cheat itself
    xs = {f: sw.get(f"{cheat}@{f:g}") or (sw.get(whole) if f == 1 else None) for f in FRACTIONS}
    xs = {f: v for f, v in xs.items() if v is not None}
    fr = list(xs)
    if len(fr) < 2:
        return None
    ref = sw["ref"][0]
    med = np.median(ref)
    # live: past the network's spikes
    lo, hi = (
        np.quantile(ref, 0.001) - 10,
        max(max(np.quantile(x, 0.99 if live else 0.995) for x, _ in xs.values()), dl) + 15,
    )
    wide = hi / lo > 2.5  # the CPU's steps are a millisecond each: a log axis, densities of log time
    if wide:
        hi *= 1.05
        grid = np.linspace(np.log10(lo), np.log10(hi), 6000)
        bw = 0.006 if live else 0.0015
        kde, xg = (lambda x: _kde(np.log10(x), grid, bw)), 10**grid
    else:
        grid = np.linspace(lo, hi, 3000)
        bw = (8.0 if cheat in ("dram_copy", "cpu") else 4.0) if live else 1.2  # live jitter (σ ≈ 17 µs) blurs the
        # per-block steps: a narrow kernel draws noise
        kde, xg = (lambda x: _kde(x, grid, bw)), grid
    fig, ax = new_fig(12, 5.8)
    hy = kde(ref)
    ax.fill_between(xg, hy, color=HONEST, alpha=0.18, lw=0)
    ax.plot(xg, hy, color=HONEST, lw=2.2)
    handles, labels = [plt.Line2D([], [], color=HONEST, lw=2.5)], ["honest"]
    for f, col in zip(fr, FRACTION_COL):
        x, k = xs[f]
        y = kde(x)
        ax.fill_between(xg, y, color=col, alpha=0.07, lw=0)
        ax.plot(xg, y, color=col, lw=1.8)
        handles.append(plt.Line2D([], [], color=col, lw=2.5))
        labels.append(
            f"{f:.0%} of the weights:  mean {x.mean() - ref.mean():+,.0f} µs,  {np.mean(x > dl):.1%} past the deadline"
            + (f" ({np.mean(k > 0):.0%} read the share)" if k is not None and f < 1 else "")
        )
    top = max(line.get_ydata().max() for line in ax.lines)
    ax.axvline(dl, color=st.AMODO_INK, lw=1.2, ls=":")
    logy = wide or cheat in ("dram_copy", "cpu")  # a host round trip per stop spreads the slow challenges thin
    if logy:
        note(ax, dl * 1.01 if wide else dl + 3, top * 1.3, dl_name, st.AMODO_INK, ha="left", va="bottom", fontsize=9.5)
    else:
        note(ax, dl - 1.5, top * 1.02, dl_name, st.AMODO_INK, ha="right", va="bottom", fontsize=9.5)
    x_c, k_c = xs[0.25] if 0.25 in xs else xs[fr[len(fr) // 2]]
    if k_c is not None and (k_c == 1).any():
        per = float(np.median(x_c[k_c == 1]) - np.median(x_c[k_c == 0]))
        note(
            ax,
            med + (1.3 if logy else 2.6) * per,
            top * (0.6 if logy else 0.42),
            f"each peak: one more block\n{SHARE_STEP[cheat]} (+{per:,.0f} µs)",
            st.AMODO_MUTED,
            fontsize=9.5,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top * 1.12)
    ax.set_yticks([])
    if logy:
        one = 1 / (min(len(x) for x, _ in xs.values()) * math.sqrt(2 * math.pi) * bw)  # the density one challenge makes
        ax.set_yscale("log")
        ax.set_ylim(max(2 * one, top * 3e-4), top * 2.5)
        ax.set_yticks([])
        ax.yaxis.set_minor_locator(plt.NullLocator())
    if wide:
        ax.set_xscale("log")
        ax.set_xticks([t for t in (1000, 1500, 2000, 3000, 5000, 7000, 10000) if lo <= t <= hi])
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.xaxis.set_minor_formatter(NullFormatter())
    style_axis(
        ax,
        "",
        "time to answer one challenge (µs" + (", log scale)" if wide else ")"),
        "density" + (" (log scale)" if logy else ""),
        grid_axis="x",
    )
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    order = [0, 2, 4, 6, 1, 3, 5][: len(handles)]  # honest first, then the shares down each column
    leg = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=10,
        handlelength=2.0,
        columnspacing=2.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    name = "weights on another GPU" if cheat == "other_gpu" else CHEAT_NAME[cheat]
    titles(
        fig,
        f"{name.capitalize()}: the cheat by the share of the weights",
        f"{_source(live)}; 1 MiB blocks, 8 reads a challenge"
        + ("; the other GPU simulated on this one" if cheat == "other_gpu" else ""),
    )
    caption(
        fig,
        "How to read: a challenge reads 8 blocks at random, so at a small share most challenges never read the cheated "
        "share and look exactly honest (the 1% curve lies on the honest one); the rest are slower by a fixed step per "
        f"block {SHARE_STEP[cheat]}"
        + (", which the network's jitter blurs into one wider hump" if live and not wide else "")
        + (
            " (on a share, the GPU stops at that block, the host answers it and the GPU goes on: a round trip more "
            "per stop than the whole cheat pays per block)"
            if cheat in ("dram_copy", "cpu")
            else ""
        )
        + ". "
        + "The share past the deadline is the share of challenges a single-challenge check would flag"
        + ("; the mean shift is what the window test sees." if live else "."),
        y=-0.035,
    )
    return save_plot(fig, OUT / f"main_m7_fraction_{cheat}.png")


WM_DRAWS = 20000  # simulated windows per posture
WM_FAR_US = 300  # a posture whose windows all lie this far past the threshold goes in a panel of its own
WM_MIN_SLOT = 5  # a posture's stretches shorter than this (a switch at the end of the run) are left out


def live_claims(runs: Path) -> list:
    """Each labelled live run's window test as the verifier ran it, and its postures' challenges as the test sees them:
    [(run, params, N, {posture: [stretches]})], a stretch the challenges of one posture slot in order, each clipped
    as the test clips it and less the calibration's honest mean (µs)."""
    from memory_accounting.verifier.timing_claim import ClaimParams

    out = []
    for name in FRAC_LIVE:
        f = runs / name / "run.npz"
        if not f.exists():
            continue
        data = load(f)
        claim_meta = json.loads(str(data["meta"])).get("claims", {}).get("static") or {}
        if claim_meta.get("n_window"):
            # scoring recorded by the verifier
            params = ClaimParams(
                F=claim_meta["F"],
                C=claim_meta["C"],
                L=claim_meta["L"],
                delta_c=claim_meta["delta_c_us"],
                sigma_b=claim_meta["sigma_b_us"],
                alpha=claim_meta["alpha"],
                M=claim_meta["M_us"],
                sigma_margin=claim_meta["sigma_margin"],
                drift=claim_meta["drift_us"],
                mu_b=claim_meta["mu_b_us"],
                n_cal=claim_meta["n_cal"],
                x_min=claim_meta["floor_us"] + 3 * claim_meta["sigma_b_us"],
                c=claim_meta.get("c_us"),
            )
            nwin = int(claim_meta["n_window"])
        elif name in POSTHOC_CLAIM:
            params, nwin = posthoc_claim(data, *POSTHOC_CLAIM[name])
        else:
            continue
        t, x, lab = label(data, "static", settle=FRAC_SETTLE_S)
        slot = np.searchsorted(data["posture_t"].astype(float), t, side="right") - 1
        xc = np.clip(x, params.floor, params.clip) - params.mu_b
        posts = {}
        for q in np.unique(lab):
            sel = lab == q
            posts[str(q)] = [s for k in np.unique(slot[sel]) if len(s := xc[sel][slot[sel] == k]) >= WM_MIN_SLOT]
        out.append((name, params, nwin, posts))
    return out


# labelled live runs whose verifier ran no window test (v0.19's fixed-cap test, set up from 200 warmup rounds, could not
# support F = 0.01): the figures run v0.17's on them, set up after the run from the run's own calibration with the
# verifier's settings then: (F, C, reads a challenge, Δ_c µs)
POSTHOC_CLAIM = {"v019_live": (0.01, 0.999, 128, 15.9)}


def posthoc_claim(data: dict, F: float, C: float, L: int, delta_c: float):
    """(params, N) of v0.17's window test from a run's static calibration, as its verifier set it up: clipped at
    μ ± L·Δ_c, σ with a margin of 0.1, no drift allowed."""
    from memory_accounting.verifier.timing_claim import estimate_baseline, params_from_baseline, plan

    a = data["static"]
    base = estimate_baseline(a[(a[:, 3] == 1) & np.isfinite(a[:, 1]), 1], L, delta_c)
    params = params_from_baseline(base, F, C, L, delta_c, sigma_margin=0.1)
    return params, math.ceil(plan(params)["n_required"])


def claim_source(name: str) -> str:
    """How a run's window test came to be, a subtitle's second line: blank if its verifier ran it."""
    return (
        "\nthe window test is v0.17's, set up after the run from its calibration (the verifier's own was off at F = 1%)"
        if name in POSTHOC_CLAIM
        else ""
    )


def window_means(stretches, n: int, k: int, rng, block: bool):
    """k simulated means of a window of n challenges from one posture: challenges drawn one at a time, or (block)
    whole stretches drawn in turn and the window cut at n, which keeps the drift between stretches."""
    if not block:
        x = np.concatenate(stretches)
        return np.concatenate(
            [
                x[rng.integers(0, len(x), (min(k, 2_000_000 // n + 1), n))].mean(1)
                for _ in range(0, k, 2_000_000 // n + 1)
            ]
        )[:k]
    lens = np.array([len(s) for s in stretches])
    pre = np.zeros((len(stretches), lens.max() + 1))  # each stretch's running sums, held at its total
    for i, s in enumerate(stretches):
        pre[i, 1 : len(s) + 1] = np.cumsum(s)
        pre[i, len(s) + 1 :] = pre[i, len(s)]
    ids = rng.integers(0, len(stretches), (k, n // lens.min() + 2))
    cl, tot = np.cumsum(lens[ids], 1), np.cumsum(pre[ids, lens[ids]], 1)
    j, r = (cl < n).sum(1), np.arange(k)  # the stretch the window ends in
    took = np.where(j > 0, cl[r, j - 1], 0)
    return (np.where(j > 0, tot[r, j - 1], 0) + pre[ids[r, j], n - took]) / n


def main_m8_window_means(runs: Path, period, rng, share: float = 0.1):
    """M8: the distribution of the window test's mean over N challenges, honest against every cheat at a share,
    simulated from a labelled live run's challenges drawn in whole stretches (which keeps the drift between them)."""
    got = live_claims(runs)
    if not got:
        return None

    def post(c):
        return ("dram_mapped" if c == "partial" else c) if share == 1 else f"{c}@{share:g}"

    base, p0, n0, posts0 = got[0]
    rows = [("honest", HONEST, posts0["honest"], p0, n0)]  # (label, colour, stretches, params, N)
    for c in ("encode_otf", "partial", "other_gpu", "dram_copy", "cpu"):
        for name, p, n, posts in got:
            if post(c) in posts:
                rows.append((CHEAT_NAME[c], CHEAT_COL[c], posts[post(c)], p, n))  # each against its own run's test
                break
    if len(rows) < 2:
        return None
    sims = [window_means(s, n, WM_DRAWS, rng, True) for _, _, s, _, n in rows]
    mu0 = p0.mu_b  # drawn in this run's µs: each posture's shift over its own run's calibration, plus this
    tau = threshold_tau(p0, n0)  # run's calibration mean
    # far past the threshold: a panel of their own
    far = [i for i, v in enumerate(sims) if np.quantile(v, 0.0005) > tau + WM_FAR_US]
    near = [i for i in range(len(rows)) if i not in far]

    def span(ids):
        return (
            mu0 + min(np.quantile(sims[i], 0.0005) for i in ids) - 4,
            mu0 + max(np.quantile(sims[i], 0.9995) for i in ids) + 6,
        )

    lo, hi = span(near)
    hi = max(hi, mu0 + tau + 4)
    lo = min(lo, mu0 - 0.17 * (hi - lo))  # room left of the calibration line for its label
    if far:
        fig, (ax, axf) = new_fig(12, 5.6, cols=2, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
        flo, fhi = span(far)
        pad = 0.06 * (fhi - flo)
        flo, fhi = flo - pad, fhi + pad
    else:
        fig, ax = new_fig(12, 5.6)

    def draw(ax, ids, x0, x1):
        grid = np.linspace(x0, x1, 3000)
        for i in ids:
            _lab, col, s, _p, n = rows[i]
            v = mu0 + sims[i]
            if v.std() < 0.05 or v.std() < (x1 - x0) / 1500:  # every challenge at the clip, or narrower than a pixel
                ax.plot([v.mean()] * 2, [0, 1], color=col, lw=2)
            else:
                y = _kde(v, grid, max(0.25, v.std() / 6))
                y /= y.max()
                ax.plot(grid, y, color=col, lw=2)
                ax.fill_between(grid, y, color=col, alpha=0.10, lw=0)
            for w in mu0 + windows(np.concatenate(s), n).mean(1):  # the posture's own windows, its challenges in order
                ax.plot([w, w], [-0.11, -0.03], color=col, lw=1.6, solid_capstyle="butt")
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.14, 1.25)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(thousands))
        ax.yaxis.set_major_locator(plt.NullLocator())

    draw(ax, near, lo, hi)
    ax.axvline(mu0 + tau, color=st.AMODO_INK, lw=1.2, ls=":")
    ax.axvline(mu0, color=st.AMODO_MUTED, lw=1, ls=(0, (1, 3)))
    note(
        ax,
        mu0 + tau + 0.006 * (hi - lo),
        1.12,
        f"window test's threshold ({mu0 + tau:,.1f} µs)",
        st.AMODO_INK,
        fontsize=9.5,
        va="center",
    )
    note(
        ax,
        mu0 - 0.006 * (hi - lo),
        1.12,
        f"honest mean at calibration ({mu0:,.0f} µs)",
        st.AMODO_MUTED,
        fontsize=9.5,
        va="center",
        ha="right",
    )
    style_axis(ax, "", "mean response time of a window, each response capped (µs)", "density", grid_axis="x")
    ax.yaxis.set_major_locator(plt.NullLocator())
    if far:
        draw(axf, far, flo, fhi)
        axf.xaxis.set_major_locator(plt.MaxNLocator(3))
        style_axis(axf, "", "", "", grid_axis="x")
        axf.yaxis.set_major_locator(plt.NullLocator())
        axf.spines["left"].set_visible(False)
        axf.tick_params(axis="y", length=0)
        ax.spines["right"].set_visible(False)
        for a, x in ((ax, 1), (axf, 0)):  # the axis break
            a.plot(
                [x - 0.012, x + 0.012],
                [-0.025, 0.025],
                transform=a.transAxes,
                color=st.AMODO_INK,
                lw=1.2,
                clip_on=False,
            )
        note(
            axf,
            fhi - 0.02 * (fhi - flo),
            1.12,
            "far past it (axis broken)\nticks below: the run's own windows",
            st.AMODO_MUTED,
            fontsize=9.5,
            va="center",
            ha="right",
        )
    else:
        note(
            ax,
            hi - 0.01 * (hi - lo),
            1.12,
            "ticks below: the run's own windows",
            st.AMODO_MUTED,
            fontsize=9.5,
            va="center",
            ha="right",
        )
    handles = [plt.Line2D([], [], color=c, lw=2.5) for _, c, *_ in rows]
    labels = [
        f"{lab}   {'false alarms' if lab == 'honest' else 'caught'} {np.mean(sim > threshold_tau(p, n)):.1%}"
        for (lab, _, _, p, n), sim in zip(rows, sims)
    ]
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.675 if far else 0.5, -0.2),
        ncol=2,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.5,
        title="share of windows past the threshold",
        title_fontsize=9.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    what = "every cheat on the whole array" if share == 1 else f"every cheat on {share:.0%} of the weights"
    titles(
        fig,
        f"Window means over {n0} challenges: {what}",
        f"simulated from the live run's challenges (the verifier {LINK}, kernel timestamps), {WM_DRAWS:,} windows each "
        f"drawn in whole 40 s stretches, densities scaled to their peaks; 1 MiB blocks, {p0.L} reads a challenge"
        f"{claim_source(base)}",
    )
    if far:  # the panels' own spacing leaves them out of tight_layout: kept under the subtitle here
        fig.subplots_adjust(top=min(plt.rcParams["figure.subplot.top"], fig._amodo_content_top - 0.005))
    return save_plot(fig, OUT / f"main_m8_window_means_{share * 100:g}pct.png")


def threshold_tau(p, n: int) -> float:
    """The window test's threshold on the mean over n challenges (µs over the calibration's honest mean)."""
    from memory_accounting.verifier.timing_claim import threshold

    return threshold(p, n) / n


WM_SIZES = np.unique(np.round(np.geomspace(25, 3000, 22)).astype(int))


def m9_drift_note(p, posts, rows, caught, sizes=None) -> str:
    """M9's caption on the stretches as they ran, from the run: where the honest level sat against its calibration, and
    so whether its wander flags honest windows (the LAN runs) or hides the small shares (the direct cable's)."""
    shift = float(np.mean(np.concatenate(posts["honest"])))  # clipped, less the calibration's mean
    fa = max(caught["honest", True])
    small = [lab.split(" ")[0] for lab, _, _ in rows[1:] if max(caught[lab, True]) < 0.1]
    small = " and ".join(small) + (" shares'" if len(small) > 1 else " share's")
    allowance = f" as well as the {p.drift:g} µs drift allowance" if p.drift else ""
    if fa < 0.01 and sizes is not None and all(max(caught[lab, True]) >= 0.95 for lab, _, _ in rows[1:]):
        need = max(int(sizes[int(np.argmax(np.array(caught[lab, True]) >= 0.95))]) for lab, _, _ in rows[1:])
        return (
            f"Drawn in stretches, as they ran, the honest level sat {abs(shift):.1f} µs {'below' if shift < 0 else 'above'} "
            f"its calibration mean, so no honest window is flagged, and every cheat makes up that gap{allowance}: each is "
            f"caught in 95% of windows from about {need:,} challenges, inside the verifier's window."
        )
    if fa >= 0.01:
        return (
            "Drawn in stretches, as they ran, the honest level's wander between stretches puts a few percent of honest "
            "windows past the threshold until the window spans many stretches, and slows the cheats' rise: the 1% share's "
            "windows cross the threshold only when the wander happens to push them over."
        )
    return (
        f"Drawn in stretches, as they ran, the honest level sat {abs(shift):.1f} µs {'below' if shift < 0 else 'above'} "
        "its calibration mean, so no honest window is flagged; but a cheat then has to make up that gap"
        f"{allowance or ''} before it shows, and the {small} windows hardly ever cross the threshold, "
        "however long the window."
    )


def m9_far_note(p, posts, f1, sizes, rng) -> str:
    """M9 with every cheat: the slow ones left off the plot, and from what window size they are caught (from the run)."""
    out = []
    for c in ("dram_copy", "cpu"):
        if f"{c}@{f1:g}" in posts:
            y = np.array(
                [np.mean(window_means(posts[f"{c}@{f1:g}"], n, 2000, rng, True) > threshold_tau(p, n)) for n in sizes]
            )
            out.append(
                f"{CHEAT_NAME[c].split(' per read')[0]} "
                + ("at every size" if y.min() >= 0.95 else f"from {int(sizes[int(np.argmax(y >= 0.95))]):,} challenges")
            )
    return (
        (" Left off the plot, caught in 95% of windows: " + "; ".join(out) + ".") if out else ""
    ) + " The stretches are drawn from the whole night, which averages its slow wander away: M12 shows it."


def main_m9_window_size(runs: Path, period, rng, cheat: str = "encode_otf"):
    """M9: the share of windows past the threshold against the window's size, honest and one cheat by share,
    simulated from the live run as in M8."""
    got = [g for g in live_claims(runs) if any(k.startswith(f"{cheat}@") for k in g[3])]
    if not got:
        return None
    name, params, n0, posts = got[0]
    shares = [(f, col) for f, col in zip(FRACTIONS, FRACTION_COL) if f"{cheat}@{f:g}" in posts][:4]
    # the run has one share: every cheat at it, the near ones (the far ones are caught at once)
    every = len(shares) == 1
    if every:
        f1 = shares[0][0]
        rows = [("honest", HONEST, posts["honest"])] + [
            (CHEAT_NAME[c], CHEAT_COL[c], posts[f"{c}@{f1:g}"])
            for c in ("encode_otf", "other_gpu", "partial")
            if f"{c}@{f1:g}" in posts
        ]
    else:
        rows = [("honest", HONEST, posts["honest"])] + [
            (f"{f:.0%} of the weights", col, posts[f"{cheat}@{f:g}"]) for f, col in shares
        ]
    sizes = WM_SIZES if n0 <= WM_SIZES[-1] else np.unique(np.round(np.geomspace(25, 2 * n0, 24)).astype(int))
    fig, ax = new_fig(11, 5.8)
    caught = {}  # (row, block?): share of windows past the threshold, at each size
    for lab, col, s in rows:
        for block, ls in ((True, "-"), (False, (0, (3, 2)))):
            y = caught[lab, block] = [
                np.mean(window_means(s, n, 4000, rng, block) > threshold_tau(params, n)) for n in sizes
            ]
            ax.plot(sizes, y, color=col, lw=2 if block else 1.4, ls=ls)
    ax.axvline(n0, color=st.AMODO_INK, lw=1.2, ls=":")
    whose = "v0.17's" if name in POSTHOC_CLAIM else "the verifier's"
    note(ax, n0 * 1.04, 1.03, f"{whose} window ({n0:,})", st.AMODO_INK, fontsize=9.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(sizes[0], sizes[-1])
    ax.set_ylim(-0.02, 1.12)
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    ax.set_xticks([t for t in (25, 50, 100, 200, 500, 1000, 2000, 5000) if t <= sizes[-1]])
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style_axis(ax, "", "challenges in a window (log scale)", "windows past the threshold", grid_axis="y")
    handles = [plt.Line2D([], [], color=c, lw=2.5) for _, c, _ in rows]
    handles += [
        plt.Line2D([], [], color=st.AMODO_MUTED, lw=2),
        plt.Line2D([], [], color=st.AMODO_MUTED, lw=1.4, ls=(0, (3, 2))),
    ]
    labels = [lab + ("  (false alarms)" if lab == "honest" else "") for lab, _, _ in rows]
    labels += ["whole stretches drawn in turn (keeps the drift)", "challenges drawn one at a time"]
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.0,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    titles(
        fig,
        f"How big a window needs to be: every cheat near honest, on {f1:.0%} of the weights"
        if every
        else f"How big a window needs to be: {CHEAT_NAME[cheat]}, by share",
        f"simulated from the live run's challenges (the verifier {LINK}, kernel timestamps), 4,000 windows at each size; "
        f"1 MiB blocks, {params.L} reads a challenge{claim_source(name)}",
    )
    allowed = f", with {params.drift:g} µs allowed for drift" if params.drift else ", no allowance for drift"
    caption(
        fig,
        "How to read: for each window size the threshold is set the way the verifier sets it (an honest prover past "
        f"it at most 1 time in {1 / params.a:,.0f}{allowed}), so it falls as the window grows "
        "and a smaller shift shows. Drawn one challenge at a time, honest is never flagged and each "
        + ("cheat" if every else "share")
        + " is caught once the window is long enough. "
        + m9_drift_note(params, posts, rows, caught, sizes)
        + (m9_far_note(params, posts, f1, sizes, rng) if every else ""),
    )
    return save_plot(
        fig, OUT / ("main_m9_window_size_every_cheat.png" if every else f"main_m9_window_size_{cheat}.png")
    )


def main_m12_night(runs: Path, period, rng):
    """M12: each posture's level through the run: each 40 s slot's mean over the calibration's honest mean, and the
    posture's mean over a span as long as the verifier's window takes (its challenges in that span only), which is
    what a window of that posture alone, run then, would have seen, drift and all. The slow cheats are only noted."""
    got = live_claims(runs)
    if not got:
        return None
    name, params, n0, _ = got[0]
    d = load(runs / name / "run.npz")
    times, x, lab = label(d, "static", settle=FRAC_SETTLE_S)
    slot = np.searchsorted(d["posture_t"].astype(float), times, side="right") - 1
    xc = np.clip(x, params.floor, params.clip) - params.mu_b  # as the window test sees them
    t0 = times.min()
    a = d["static"]
    rate = np.isfinite(a[a[:, 3] == 0, 1]).sum() / (times.max() - t0)
    span = n0 / rate  # seconds a window of n0 challenges takes
    f1 = FRACTIONS[0]
    show = [("honest", "honest", HONEST)] + [
        (f"{c}@{f1:g}", CHEAT_NAME[c], CHEAT_COL[c])
        for c in ("encode_otf", "other_gpu", "partial")
        if f"{c}@{f1:g}" in set(lab)
    ]
    tau = threshold_tau(params, n0)
    fig, ax = new_fig(12, 5.8)
    handles, labels, ys = [], [], []
    for kind, nm, col in show:
        sel = lab == kind
        tq, xq = times[sel], xc[sel]
        ks = np.unique(slot[sel])
        st_t = np.array([tq[slot[sel] == k].mean() for k in ks])
        st_m = np.array([xq[slot[sel] == k].mean() for k in ks])
        cs = np.concatenate([[0], np.cumsum(xq)])
        lo_i, hi_i = np.searchsorted(tq, st_t - span / 2), np.searchsorted(tq, st_t + span / 2)
        # the posture's mean over a window's time around each slot
        loc = (cs[hi_i] - cs[lo_i]) / np.maximum(hi_i - lo_i, 1)
        full = (st_t - span / 2 >= t0) & (st_t + span / 2 <= times.max())
        ax.plot((st_t - t0) / 3600, st_m, "o", ms=2.5, color=col, alpha=0.35, mew=0)
        ax.plot((st_t - t0) / 3600, np.where(full, loc, np.nan), color=col, lw=2)
        ys.append(st_m)
        past = np.mean(loc[full] > tau)
        handles.append(plt.Line2D([], [], color=col, lw=2.5))
        labels.append(
            f"{nm}   mean {xq.mean():+.1f} µs;  "
            + (f"over the threshold {past:.0%} of {WHEN}" if kind == "honest" else f"under it {1 - past:.0%} of {WHEN}")
        )
    allm = np.concatenate(ys)
    lo, hi = np.quantile(allm, 0.003) - 5, np.quantile(allm, 0.997) + 12
    ax.axhline(0, color=st.AMODO_MUTED, lw=1, ls=(0, (1, 3)))
    ax.axhline(tau, color=st.AMODO_INK, lw=1.2, ls=":")
    xe = (times.max() - t0) / 3600
    handles += [
        plt.Line2D([], [], color=st.AMODO_INK, lw=1.2, ls=":"),
        plt.Line2D([], [], color=st.AMODO_MUTED, lw=1, ls=(0, (1, 3))),
    ]
    # in the legend:
    labels += [
        f"window test's threshold (+{tau:.1f} µs)",
        f"honest mean at calibration ({params.mu_b:,.0f} µs)",
    ]
    # the posture lines cross any label on the plot
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, xe)
    style_axis(ax, "", "hours into the run", "over the calibration's honest mean (µs)", grid_axis="y")
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    far = [
        (CHEAT_NAME[c].split(" per read")[0], xc[lab == f"{c}@{f1:g}"].mean())
        for c in ("dram_copy", "cpu")
        if f"{c}@{f1:g}" in set(lab)
    ]
    titles(
        fig,
        f"Through {WHEN}: the honest level wanders more than the threshold allows",
        f"the live run, {CYCLE}; dots: each slot's mean; lines: each posture's mean over {span / 60:.0f} min, "
        f"the time a window of {n0:,} takes; 1 MiB blocks, {params.L} reads a challenge{claim_source(name)}",
    )
    caption(
        fig,
        "How to read: each line is what a window of that posture alone would have measured, run at that time of the "
        f"night. All the postures rise and fall together by about ±20 µs over the hours, so the wander is in the honest "
        f"work and the round trip, not in the cheats, and the random order spreads it evenly over the postures. But "
        f"the threshold is only {tau:.1f} µs above the calibration: where the level wanders up, an honest window would be "
        "flagged, and where it wanders down, the closest cheats come within a few µs of passing. M8 and M9 draw their "
        "stretches from the whole night, which averages the wander away."
        + (" Off the scale: " + ", ".join(f"{n} {m:+,.0f} µs" for n, m in far) + "." if far else ""),
        y=-0.035,
    )
    return save_plot(fig, OUT / "main_m12_night.png")


CL_LINKS = (4, 8, 16, 32, 64)
CL_MEASURED = 16  # chain lengths up to this have live honest runs (the noise model's fit); longer are extrapolated
CL_COL = ("#a9cbf2", "#6aa5e8", "#2a78d6", "#1a5199", "#0e2f5e")  # one hue, light to dark: more links per challenge
CL_SHARES = np.geomspace(0.002, 0.2, 25)
CL_ALPHA = 1e-3  # honest windows past the threshold: the test's design rate


def cl_sigma(L: float) -> float:
    """Spread of one honest challenge (robust σ, µs) at L links, fitted to the live vLLM runs at 4, 8 and 16 links
    (16.2, 18.3, 21.1 µs): σ² = 205 + 15.5·L, most of it the round trip, paid once a challenge."""
    return math.sqrt(205 + 15.5 * L)


def main_m10_chain_length(runs: Path, period, rng):
    """M10: windows caught against the share of the weights, at longer challenges: the live run's honest stretches at
    8 links, rescaled to L links (each challenge's own spread by the fitted σ(L); each stretch's drift unchanged, as it
    is in the round trip), plus Δc per cheated read, K ~ Binomial(L, share). The threshold holds honest windows to 1 in
    1,000 (set from the honest windows themselves, so it allows for the drift)."""
    got = live_claims(runs)
    if not got:
        return None
    _name, _p, n0, posts = got[0]
    hs = posts["honest"]
    mu = np.concatenate(hs).mean()

    def at(L, f, k):
        g = cl_sigma(L) / cl_sigma(8)
        ss = [s.mean() - mu + g * (s - s.mean()) for s in hs]  # drift kept, the challenge's own spread scaled
        if f > 0:
            ss = [s + DC * rng.binomial(L, f, len(s)) for s in ss]
        # capped at L·Δc over honest, as the test
        return window_means([np.minimum(s, L * DC) for s in ss], n0, k, rng, True)

    fig, ax = new_fig(11, 5.8)
    floors = []
    for L, col in zip(CL_LINKS, CL_COL):
        thr = np.quantile(at(L, 0, 200_000), 1 - CL_ALPHA)
        y = np.array([np.mean(at(L, f, 6000) > thr) for f in CL_SHARES])
        ax.plot(CL_SHARES, y, color=col, lw=2.2, ls="-" if L <= CL_MEASURED else (0, (4, 2)))
        i = int(np.argmax(y >= 0.95)) if (y >= 0.95).any() else None
        floors.append((L, CL_SHARES[i] if i is not None else None))
    # the check at 8 links: the real cheat on another GPU (about Δc a cheated read live), against the same threshold
    thr8 = np.quantile(at(8, 0, 200_000), 1 - CL_ALPHA)
    real = [
        (f, np.mean(window_means(posts[f"other_gpu@{f:g}"], n0, 20000, rng, True) - mu > thr8))
        for f in FRACTIONS
        if f"other_gpu@{f:g}" in posts
    ]
    if real:
        ax.plot(
            [f for f, _ in real], [c for _, c in real], "o", ms=8, color=CL_COL[1], mec=st.AMODO_INK, mew=1.2, zorder=5
        )
    ax.set_xscale("log")
    ax.set_xlim(CL_SHARES[0], CL_SHARES[-1])
    ax.set_ylim(-0.02, 1.08)
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    ax.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style_axis(
        ax, "", "share of the weights cheated on (log scale)", f"windows of {n0} challenges caught", grid_axis="y"
    )
    handles = [
        plt.Line2D([], [], color=c, lw=2.5, ls="-" if L <= CL_MEASURED else (0, (4, 2)))
        for L, c in zip(CL_LINKS, CL_COL)
    ]
    labels = [
        f"{L} reads a challenge"
        + (" (now)" if L == 8 else "")
        + (f": 95% caught from {f:.1%}" if f else ": not 95% caught")
        for L, f in floors
    ]
    if real:
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=8, color=CL_COL[1], mec=st.AMODO_INK, mew=1.2))
        labels.append("measured at 8 reads: weights on another GPU")
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2.5,
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    titles(
        fig,
        "Longer challenges: the smallest share the window test catches",
        f"simulated from the live run's honest challenges at 8 reads, rescaled to each length; {DC:g} µs per cheated read; "
        f"threshold set so 1 honest window in {1 / CL_ALPHA:,.0f} is flagged; dashed: beyond the lengths measured live",
    )
    caption(
        fig,
        "How to read: the slow drift of the honest level is in the round trip, not the GPU's work (in the live run the "
        "static chain and the much shorter dynamic chain drift together, µs for µs), so a longer challenge keeps the "
        "same drift and nearly the same spread while a cheat's extra time grows with every read. Each line is one "
        "challenge length; the dots are the real cheat at 8 reads against the same threshold.",
    )
    return save_plot(fig, OUT / "main_m10_chain_length.png")


RS_LINKS = (4, 8, 16, 32, 64, 128, 256, 512)
RS_SHARES = np.geomspace(5e-4, 0.5, 36)
RS_LINK_US = 136.0  # one static link on an H200 NVL: honest medians 759, 1,289 and 2,388 µs at 4, 8 and 16 links
RS_STATIC_GB = 76.31 * (1 << 30) / 1e9  # the static weights challenged in the vLLM runs (76.31 GiB)
RS_MODELS = (
    ("no restore (as M10)", None, "#2a78d6"),
    ("restores at 57.7 GB/s: an H200 host's PCIe 5 x16, pinned, measured", 57.7, "#eb6834"),
    ("restores at 450 GB/s: an NVLink-C2C class host link, not measured", 450.0, "#6250d6"),
)


def main_m11_restore(runs: Path, period, rng):
    """M11: M10's smallest share caught against the challenge's length, when the cheater starts copying all its
    cheated weights back into HBM as each challenge arrives. Read k comes about k·RS_LINK_US in, by when BW·t of the
    f·S cheated bytes are back, so it lands on cheated data with probability max(0, f − BW·t/S). Same honest
    stretches, rescaling and threshold as M10."""
    got = live_claims(runs)
    if not got:
        return None
    _name, _p, n0, posts = got[0]
    hs = posts["honest"]
    mu = np.concatenate(hs).mean()

    def at(L, q, k):
        g = cl_sigma(L) / cl_sigma(8)
        ss = [s.mean() - mu + g * (s - s.mean()) for s in hs]
        if q is not None:
            ss = [s + DC * (rng.random((len(s), L)) < q).sum(1) for s in ss]
        return window_means([np.minimum(s, L * DC) for s in ss], n0, k, rng, True)

    res = {m: [] for m, _, _ in RS_MODELS}
    for L in RS_LINKS:
        thr = np.quantile(at(L, None, 200_000), 1 - CL_ALPHA)
        t = np.arange(1, L + 1) * RS_LINK_US * 1e-6
        for model, bw, _ in RS_MODELS:
            y = np.array(
                [
                    np.mean(at(L, np.clip(f - (bw or 0) * 1e9 * t / (RS_STATIC_GB * 1e9), 0, None), 3000) > thr)
                    for f in RS_SHARES
                ]
            )
            i = int(np.argmax(y >= 0.95)) if (y >= 0.95).any() else None
            if i is None or i == 0:
                res[model].append(np.nan if i is None else RS_SHARES[0])
                continue
            a, b = np.log(RS_SHARES[i - 1]), np.log(RS_SHARES[i])  # the 95% crossing, interpolated on the log scale
            res[model].append(float(np.exp(a + (b - a) * (0.95 - y[i - 1]) / max(y[i] - y[i - 1], 1e-9))))

    fig, ax = new_fig(11, 5.8)
    Ls = np.array(RS_LINKS, float)
    meas = Ls <= CL_MEASURED
    handles, labels = [], []
    for model, bw, col in RS_MODELS:
        y = np.array(res[model])
        ax.plot(Ls[meas], y[meas], color=col, lw=2.2, marker="o", ms=5)
        ax.plot(Ls[Ls >= CL_MEASURED], y[Ls >= CL_MEASURED], color=col, lw=2.2, ls=(0, (4, 2)), marker="o", ms=5)
        j = int(np.argmax(y <= 1.1 * np.nanmin(y)))  # where it comes within 10% of its least
        handles.append(plt.Line2D([], [], color=col, lw=2.5, marker="o", ms=5))
        labels.append(
            f"{model}: still falling at {RS_LINKS[-1]} reads ({y[j]:.1%})"
            if j == len(Ls) - 1
            else f"{model}: levels off at about {y[j]:.1%} from {RS_LINKS[j]} reads"
        )
    ax.axvline(8, color=st.AMODO_INK, lw=1, alpha=0.35, zorder=0)
    ax.text(
        8 * 1.06, 0.98, "now", transform=ax.get_xaxis_transform(), fontsize=10, color=st.AMODO_INK, alpha=0.7, va="top"
    )
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"{L}\n{L * RS_LINK_US / 1000:.2g} ms" for L in RS_LINKS])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    style_axis(
        ax,
        "",
        "reads per challenge, and how long the challenge's reads take",
        "share caught in 95% of windows",
        grid_axis="y",
    )
    leg = ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, fontsize=10, handlelength=2.2
    )
    fig._amodo_bbox_extra_artists = getattr(fig, "_amodo_bbox_extra_artists", []) + [leg]
    titles(
        fig,
        "Against a restoring cheater, longer challenges level off",
        f"M10's simulation; the cheater copies all {RS_STATIC_GB:.0f} GB × share back to HBM from each challenge's arrival, "
        f"a read every {RS_LINK_US:.0f} µs; dashed: beyond the lengths measured live",
    )
    caption(
        fig,
        "How to read: a read that comes after the copy has finished costs the cheater nothing, so once a challenge "
        "outlasts the copy, more reads add nothing to the cheat's extra time (which grows with the share squared, over "
        "the copy's speed), while the honest spread grows slowly. The smallest share caught stops falling where the "
        "copy of that share fits inside the challenge: the faster the host link, the sooner.",
    )
    return save_plot(fig, OUT / "main_m11_restore.png")


# ================================================================================================= appendix


def _bench(runs: Path):
    files = sorted((runs / FIG_RUNS).glob("bench_dyn_dram_f*.json"))
    if not files:
        return None
    by_block: dict = {}
    for f in files:
        dj = json.loads(f.read_text())
        for bb, postures in dj["samples"].items():
            for post, smp in postures.items():
                by_block.setdefault(int(bb), {}).setdefault(post, []).append(
                    (np.array(smp["chain_ns"]) / 1e3, np.array(smp["host_links"]))
                )
    return by_block


def _delta_c_line(ax, posts, color, band=True):
    """Median extra time against host-memory links (part of the pool evicted), with its fit; returns (slope, whole-pool cost
    per link)."""

    def cat(post):
        return (
            np.concatenate([p[0] for p in posts[post]]),
            np.concatenate([p[1] for p in posts[post]]).astype(int),
        )

    th, _ = cat("honest")
    base = np.median(th)
    t, k = cat("half")
    ks = np.unique(k)
    med = np.array([np.median(t[k == j]) - base for j in ks])
    if band:
        q1 = np.array([np.percentile(t[k == j], 10) - base for j in ks])
        q9 = np.array([np.percentile(t[k == j], 90) - base for j in ks])
        ax.fill_between(ks, q1, q9, color=st.AMODO_TINT)
    sl, ic = np.polyfit(k, t - base, 1)
    ax.plot(ks, sl * ks + ic, "-", color=color, lw=1.4)
    ax.plot(ks, med, "o", color=color, ms=6)
    ta, _ = cat("all")
    return sl, (np.median(ta) - base) / L


def appx_a1(runs: Path, period, rng):
    bench = _bench(runs)
    if bench is None:
        return None
    fig, ax = new_fig(9, 5.2)
    for (bb, posts), col in zip(sorted(bench.items(), reverse=True), (CHEAT, CHEAT2, st.AMODO_BLUE)):
        sl, _ = _delta_c_line(ax, posts, col, band=False)
        note(
            ax,
            L + 0.2,
            sl * L,
            f"{name_of(bb)} blocks: {sl:.0f} µs a link" if sl >= 3 else f"{name_of(bb)} blocks: {sl:.1f} µs a link",
            col,
            va="center",
            fontsize=11,
        )
    ax.set_xticks(range(L + 1))
    ax.set_xlim(-0.3, L + 3.6)
    style_axis(ax, "", f"links of the {L}-link chain served from host memory", "extra round-trip time (µs)")
    titles(
        fig,
        "Bigger blocks make a cheated read cost more",
        "the KV-cache challenge on the H200, at three block sizes; each dot is the median of 1,000 challenges",
    )
    caption(
        fig,
        "How to read: reading a small block from host memory costs a cheater almost nothing, so it would be hard to see. "
        "The KV-cache challenges use 1 MiB blocks.",
    )
    return save_plot(fig, OUT / "appx_a1_cost_per_read_all_blocks.png")


def appx_a2(runs: Path, period, rng):
    c, sim = run_npz(runs, "C_vllm_timeline"), run_npz(runs, "D_sim_cheats")
    if c is None or not len(c["posture_t"]):
        return None
    from matplotlib.ticker import FuncFormatter

    rows = []  # (name, share of its chains slower than that run's honest p99)
    for d, kind, posts in (
        (c, "static", ["evict_dram"]),
        (c, "dynamic", ["dyn_dram"]),
        (sim, "static", ["encode_otf", "dram_mapped", "dram_copy", "cpu"]),
    ):
        if d is None or not len(d["posture_t"]):
            continue
        _, x, lab = label(d, kind)
        cut = np.percentile(x[lab == "honest"], 99)
        if not rows:
            rows.append(("honest", float(np.mean(x[lab == "honest"] > cut))))
        rows += [(POSTURE[pst], float(np.mean(x[lab == pst] > cut))) for pst in posts if np.sum(lab == pst) >= 20]
    fig, ax = new_fig(9, 4.6)
    for i, (name, share) in enumerate(rows):
        y = len(rows) - 1 - i
        col = HONEST if i == 0 else (CHEAT2 if share < 0.5 else CHEAT)
        ax.barh(y, share, height=0.6, color=col, alpha=0.85)
        note(ax, share + 0.015, y, f"{share:.0%}", st.AMODO_INK, va="center", fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows][::-1], color=st.AMODO_INK)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    style_axis(ax, "", "challenges slower than 99% of honest ones", "", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    titles(
        fig,
        "A single challenge flags a big cheat, but not a small one",
        "labelled runs on the H200; each bar: the share of that cheat's challenges past the honest 99th percentile",
    )
    caption(
        fig,
        "How to read: cheats that move or rebuild all the weights, or half the KV cache, are slower on almost every challenge. "
        "Holding 20% of the weights in host memory shows on few, so it is caught by averaging windows instead (M2).",
    )
    return save_plot(fig, OUT / "appx_a2_single_chain_by_cheat.png")


def appx_a3(runs: Path, period, rng):
    pts = []
    for L_, name in ((4, "E_vllm_L4"), (8, "C_vllm_timeline"), (16, "E_vllm_L16"), (32, "E_vllm_L32")):
        d = run_npz(runs, name)
        if d is None or not len(d["posture_t"]):
            continue
        _, x, lab = label(d, "static")
        h, c = x[lab == "honest"], x[lab == "evict_dram"]
        if len(h) < 50 or len(c) < 50:
            continue
        pts.append((L_, np.median(h), np.median(c), (np.percentile(h, 75) - np.percentile(h, 25)) / 1.349))
    if len(pts) < 2:
        return None
    points = np.array(pts)
    fig, ax = new_fig(9, 5.2)
    gap = points[:, 2] - points[:, 1]
    ax.plot(points[:, 0], gap, "o-", color=CHEAT, lw=2.2, ms=7)
    ax.plot(points[:, 0], points[:, 3], "o-", color=HONEST, lw=2.2, ms=7)
    note(
        ax,
        points[-1, 0],
        gap[-1],
        "   cheater's slowdown\n   (20% of weights in host memory)",
        CHEAT,
        va="center",
        fontsize=11,
    )
    note(ax, points[-1, 0], points[-1, 3], "   honest noise", HONEST, va="center", fontsize=11)
    ax.set_xticks(points[:, 0])
    ax.set_xlim(0, points[-1, 0] * 1.45)
    ax.set_ylim(0, None)
    style_axis(ax, "", "links in the chain", "microseconds")
    titles(
        fig,
        "Longer chains make a cheat stand out more",
        "vLLM on the H200, one run per chain length; the cheater holds 20% of the weights in host memory",
    )
    caption(
        fig,
        "How to read: the cheater pays for every cheated link, so its slowdown grows with the chain's length; most honest noise "
        "(network, launch) is paid once per challenge, so it barely grows.",
    )
    return save_plot(fig, OUT / "appx_a3_chain_length.png")


def appx_a4(runs: Path, period, rng):
    if period is None:
        return None
    fig, (ax1, ax2) = new_fig(14, 5.2, 2)
    cal, run, t = chains(period, "static")
    ok = np.isfinite(run)
    run, t = run[ok], t[ok]
    _base, p, pl = claim(cal)
    hours = (t - t[0]) / 3600
    edges = np.arange(0, hours[-1] + 1e-9, 10 / 60)
    idx = np.digitize(hours, edges)
    m = np.array([np.clip(run[idx == i], p.floor, p.clip).mean() - p.mu_b for i in range(1, len(edges))])
    ax1.plot(edges[:-1] + 5 / 120, m, color=HONEST, lw=1.2)
    ax1.axhline(pl["tau"], color=CHEAT, lw=1.4, ls="--")
    note(ax1, 0.3, pl["tau"] + 0.8, "alarm", CHEAT, fontsize=11)
    ax1.set_ylim(-16, 14)
    style_axis(ax1, "a · the honest level over 43 hours", "hours", "10-minute average\nagainst calibration (µs)")
    ns = np.unique(np.logspace(0.5, math.log10(len(run) / 8), 30).astype(int))
    sd = np.array([windows(run, n).mean(axis=1).std(ddof=1) for n in ns])
    sig = run.std(ddof=1)
    ax2.loglog(ns, sd, "o-", color=HONEST, ms=4, lw=1.8)
    ax2.loglog(ns, sig / np.sqrt(ns), "--", color=st.AMODO_MUTED, lw=1.2)
    note(ax2, ns[-1], sd[-1] * 1.9, "measured", HONEST, ha="right", fontsize=11)
    note(
        ax2,
        ns[-1],
        sig / math.sqrt(ns[-1]) * 1.4,
        "if the noise were independent",
        st.AMODO_MUTED,
        ha="right",
        fontsize=11,
    )
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    style_axis(ax2, "b · averaging stops helping", "challenges averaged", "noise left in the average (µs)")
    titles(
        fig,
        "Honest timing drifts slowly, and averaging cannot remove drift",
        "43 honest hours on the H200 under vLLM; weight challenges (the KV-cache challenges move the same way)",
    )
    caption(
        fig,
        "How to read: (a) the honest level wanders by about 10 µs over hours, but never near the alarm. (b) Averaging more "
        "challenges shrinks the noise less and less, because the drift is shared by all of them. The test allows for it.",
    )
    return save_plot(fig, OUT / "appx_a4_honest_drift.png")


def appx_a5(runs: Path, period, rng):
    if period is None:
        return None
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, ax = new_fig(9, 5.2)
    for kind, key, c, lab, (lx, ly) in (
        ("static", "arm_static", HONEST, "weight challenges", (0.16, 0.03)),
        ("dynamic", "arm_dynamic", st.AMODO_BLUE, "KV-cache challenges", (1.25, 0.25)),
    ):
        a = period[key]
        run = np.sort(a[a[:, 2] == 0, 1] / 1e3)
        ccdf = 1 - np.arange(len(run)) / len(run)
        ax.loglog(run, ccdf, color=c, lw=2)
        note(ax, lx, ly, lab, c, fontsize=11)
    ax.yaxis.set_major_locator(FixedLocator([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "all" if v >= 1 else f"1 in {1 / v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style_axis(ax, "", "time to pause and confirm READY (ms)", "rounds that took longer")
    titles(
        fig,
        "Pausing for a challenge usually takes under a millisecond",
        "every round of 43 honest hours on the H200 under vLLM",
    )
    caption(
        fig,
        "How to read: follow a curve to see how many rounds paused for longer than a given time. A very long pause is honest but "
        "rare, and the verifier counts it as a slow round rather than an alarm, so stalling to reload memory gains a cheater nothing.",
    )
    return save_plot(fig, OUT / "appx_a5_ready_waits.png")


def appx_a6(runs: Path, period, rng):
    if period is None:
        return None
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    cal, _, _ = chains(period, "static")
    base = estimate_baseline(cal, L, DC)
    Fs = np.logspace(-3, math.log10(0.5), 80)
    fig, ax = new_fig(9, 5.2)
    for drift, color, lab in (
        (0, st.AMODO_MUTED, "if timing never drifted"),
        (DRIFT, CHEAT, "allowing for this H200's drift"),
    ):
        ns = np.array(
            [
                plan(params_from_baseline(base, F, C_DEMO, L, DC, sigma_margin=MARGIN, drift=drift))["n_required"]
                for F in Fs
            ]
        )
        ax.loglog(Fs, ns, color=color, lw=2.2)
        ok = np.isfinite(ns)
        note(ax, Fs[ok][0] * 1.06, ns[ok][0], lab, color, va="center", fontsize=11)
    n_demo = plan(params_from_baseline(base, F_DEMO, C_DEMO, L, DC, sigma_margin=MARGIN, drift=DRIFT))["n_required"]
    ax.plot([F_DEMO], [n_demo], "o", color=CHEAT, ms=7)
    note(ax, F_DEMO * 1.08, n_demo * 1.3, f"the demo: {n_demo:,.0f} for a 20% cheat", st.AMODO_INK, fontsize=11)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.5]))
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    style_axis(ax, "", "smallest cheat to catch (share of reads)", "challenges per window")
    titles(
        fig,
        "Catching smaller cheats takes many more challenges",
        "challenges a window needs to catch a cheat of a given size 999 times in 1,000, on this H200",
    )
    caption(
        fig,
        "How to read: halving the cheat to catch needs at least four times the challenges. With drift, a single GPU cannot "
        "go below about 10%: the curve ends there. Averaging over many GPUs and a long time is how smaller claims become reachable.",
    )
    return save_plot(fig, OUT / "appx_a6_cost.png")


def appx_a7(runs: Path, period, rng):
    f = runs / FIG_RUNS / "echo_vllm.json"
    c = run_npz(runs, "C_vllm_timeline")
    if not f.exists() or c is None:
        return None
    rows = json.loads(f.read_text())["rows"]
    user = np.median([r[1] for r in rows if r[1] is not None]) / 1e3
    kern = np.median([r[2] for r in rows if r[2] is not None]) / 1e3
    _, x, lab = label(c, "static")
    chain = np.median(x[lab == "honest"])
    fig, ax = new_fig(9, 3.6)
    bars = [
        ("whole challenge", chain, HONEST),
        ("network alone", kern, st.AMODO_BLUE),
        ("network alone, timed\nby an ordinary program", user, st.AMODO_MUTED),
    ]
    for i, (name, v, col) in enumerate(bars):
        y = len(bars) - 1 - i
        ax.barh(y, v, height=0.55, color=col, alpha=0.85 if i < 2 else 0.4)
        note(ax, v + 15, y, f"{v:,.0f} µs", st.AMODO_INK, va="center", fontsize=11)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars][::-1], color=st.AMODO_INK)
    ax.set_xlim(0, user * 1.15)
    ax.xaxis.set_major_formatter(thousands)
    share = kern / chain
    style_axis(ax, "", "median round trip (µs)", "", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    titles(
        fig,
        f"The network is about {share:.0%} of a challenge's round trip",
        "UDP echo between the verifier and prover hosts while vLLM served, beside honest challenges of the same run",
    )
    caption(
        fig,
        "How to read: timed by the network card's kernel timestamps, as the verifier does, the network is a small part of a "
        "challenge. Timed by an ordinary program, the same echo looks slower than a whole challenge, so the verifier must use "
        "kernel or network-card timestamps.",
    )
    return save_plot(fig, OUT / "appx_a7_network.png")


def appx_a8(runs: Path, period, rng):
    if period is None:
        return None
    from matplotlib.ticker import FuncFormatter

    cal, run, _ = chains(period, "static")
    run = run[np.isfinite(run)]
    hour = 7100  # chains in about an hour at this run's pace
    setups = [
        ("calibrated at start-up", cal, run, DRIFT, CHEAT),
        ("calibrated at start-up,\ndouble drift allowance", cal, run, 8.0, CHEAT2),
        ("calibrated on an hour\nof steady running", run[:hour], run[hour:], DRIFT, HONEST),
    ]
    fig, ax = new_fig(9, 4.0)
    allowed = 1 - C_DEMO
    for i, (name, c_, r_, drift, col) in enumerate(setups):
        _base, p, pl = claim(c_, drift)
        N = int(pl["n_required"])
        tau = pl["tau"]
        w0 = windows(r_, N)
        miss = np.mean(
            [
                np.mean(window_shift(p, w0 + cheat_add(np.random.default_rng(sd), w0.shape, F_DEMO, "bunched")) < tau)
                for sd in range(20)
            ]
        )
        y = len(setups) - 1 - i
        ax.barh(y, miss, height=0.55, color=col, alpha=0.85)
        note(
            ax,
            max(miss, allowed) + 0.0001,
            y,
            f" {miss:.2%} missed ({N:,} challenges a window)",
            st.AMODO_INK,
            va="center",
            fontsize=11,
        )
    ax.axvline(allowed, color=st.AMODO_INK, lw=1.2, ls="--")
    note(ax, allowed, len(setups) - 0.45, f" allowed: {allowed:.1%}", st.AMODO_INK, va="center", fontsize=11)
    ax.set_yticks(range(len(setups)))
    ax.set_yticklabels([s[0] for s in setups][::-1], color=st.AMODO_INK)
    ax.set_xlim(0, 0.0086)
    ax.set_ylim(-0.5, len(setups) - 0.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f}%"))
    style_axis(ax, "", "windows that missed a 20% cheat", "", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    titles(
        fig,
        "Calibrate once the prover has settled, not at start-up",
        "the worst-case 20% cheat (all cheated reads in whole challenges), on 43 hours of real honest timing",
    )
    caption(
        fig,
        "How to read: start-up timing sat higher than where the prover settled, which gave a cheater room. Calibrating on an hour of "
        "steady running meets the target at the same window size; a bigger drift allowance nearly does, but needs three times the challenges.",
    )
    return save_plot(fig, OUT / "appx_a8_calibration.png")


FIGURES = {
    "main_m1": main_m1,
    "main_m2": main_m2,
    "main_m3": main_m3,
    "main_m4": main_m4,
    "main_m5": main_m5,
    "main_m6": main_m6_every_cheat,
    "main_m6_1pct": functools.partial(main_m6_share, share=0.01),
    "main_m6_10pct": functools.partial(main_m6_share, share=0.1),
    "main_m7": main_m7_fraction,
    **{f"main_m7_{c}": functools.partial(main_m7_fraction, cheat=c) for c in SHARE_CHEATS[1:]},
    **{
        f"main_m8_{k}": functools.partial(main_m8_window_means, share=f)
        for k, f in (("1pct", 0.01), ("10pct", 0.1), ("whole", 1))
    },
    "main_m9": main_m9_window_size,
    "main_m10": main_m10_chain_length,
    "main_m11": main_m11_restore,
    "main_m12": main_m12_night,
    "appx_a1": appx_a1,
    "appx_a2": appx_a2,
    "appx_a3": appx_a3,
    "appx_a4": appx_a4,
    "appx_a5": appx_a5,
    "appx_a6": appx_a6,
    "appx_a7": appx_a7,
    "appx_a8": appx_a8,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("log", type=Path)
    e.add_argument("out", type=Path)
    a = sub.add_parser("all")
    a.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[2] / "posm_runs")
    a.add_argument("--only", default=None, help="comma-separated figure names (main_m1, appx_a4, ...)")
    a.add_argument(
        "--live",
        choices=LIVE_SETS,
        default="lan",
        help="the labelled live runs the live figures use; other than lan, they go to docs/figures/<live>/",
    )
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        extract(args.log, args.out)
        return
    global OUT, FRAC_LIVE, LINK, READS, CYCLE, CAPTIONS, WHEN
    if args.cmd == "all" and args.live != "lan":
        OUT, FRAC_LIVE, LINK, READS = OUT / args.live, LIVE_SETS[args.live], LIVE_LINK[args.live], LIVE_READS[args.live]
        CYCLE = LIVE_CYCLE.get(args.live, CYCLE)
        WHEN = LIVE_WHEN.get(args.live, WHEN)
        CAPTIONS = args.live not in LIVE_NO_CAPTION
        if READS is None:  # the run's own static claim says how many reads a challenge had
            d = run_npz(args.runs, FRAC_LIVE[0]) if (args.runs / FIG_RUNS / f"{FRAC_LIVE[0]}.jsonl").exists() else None
            npz = args.runs / FRAC_LIVE[0] / "run.npz"
            if d is None and not npz.exists() and (args.runs / FRAC_LIVE[0] / "run.jsonl").exists():
                extract(args.runs / FRAC_LIVE[0] / "run.jsonl", npz)
            d = d if d is not None else (load(npz) if npz.exists() else None)
            READS = int(json.loads(str(d["meta"]))["claims"]["static"]["L"]) if d is not None else 8
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    period_npz = args.runs / "period_20261021_140606" / "run.npz"
    period = load(period_npz) if period_npz.exists() else None
    for name, fn in FIGURES.items():
        if only is None or name in only:
            out = fn(args.runs, period, np.random.default_rng(2026))
            print(out if out else f"{name}: its data is not there yet")


if __name__ == "__main__":
    main()
