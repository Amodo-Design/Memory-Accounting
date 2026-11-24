"""Choose challenge parameters from sweeps by what the timing claim needs: for each (B, n, r), the window test's size
(`memory_accounting.verifier.timing_claim.plan`) and how long a window takes to fill within a GPU-time budget.

  python -m memory_accounting.experiments.select --static sweep_static.json --dynamic dyn_vllm.json [--dynamic dyn_int.json] \\
      --F 0.01 --C 0.999 --jitter-us 15 --budget 0.05 --max-rate 2 --out selection.json [--md selection.md]

Per configuration and chain kind (static; dynamic main; dynamic live):
  * honest: μ and σ of the bench's round trips (in-process, no network), σ widened by the network's jitter in quadrature
    (--jitter-us: the direct cable with kernel timestamps; the bench cannot see it);
  * Δc: each cheat's cost per deviating read, (median cheat − median honest) / (n · deviating share), and the smallest
    over the cheats (the claim must hold for the cheapest). The live chain takes the main chain's at the same shape:
    its reads are of the same words;
  * N: the window test's challenges at F, C (`plan`); the window's wall time N / rate, the rate being the most rounds
    a second the budget allows (budget / (round trip + the round's fixed cost: --static-cost-us, --dyn-cost-us, the
    freeze), at most --max-rate). Ranked by that time; shapes with a wrong digest, too few blocks (--min-blocks) or
    no cheat that costs anything are left out.
Static and dynamic chains share r and k (`memory_accounting.verifier.net._dyn_shape`), so the dynamic choice is made at the static
choice's r.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..verifier.timing_claim import estimate_baseline, params_from_baseline, plan


def _by_shape(records, chain=None):
    out = {}
    for r in records:
        if chain is not None and r.get("chain", "main") != chain:
            continue
        out.setdefault((r["B"], r["n"], r["r"]), {})[r["prover"]] = r
    return out


def _delta_c(recs: dict, n: int, share_of) -> dict:
    """Δc per cheat (µs a deviating read) at one shape."""
    h = np.median(recs["honest"]["samples_ns"]) / 1e3
    out = {}
    for name, rec in recs.items():
        if name == "honest" or not rec.get("correct", True):
            continue
        sh = share_of(name, rec)
        if sh <= 0:
            continue
        d = (np.median(rec["samples_ns"]) / 1e3 - h) / (n * sh)
        out[name] = float(d)
    return out


def score(x_us, n: int, delta_c: float, args, fixed_us: float, blocks: int) -> dict:
    x = np.asarray(x_us, dtype=np.float64)
    base = estimate_baseline(x, n, delta_c)
    sigma = math.hypot(base["sigma_b"], args.jitter_us)
    base = dict(base, sigma_b=sigma)
    base = dict(base, n=args.n_cal)  # the live verifier's calibration, not the bench's sample count
    claim_plan = plan(params_from_baseline(base, args.F, args.C, n, delta_c))
    N = claim_plan["n_required"]
    rtt = float(np.median(x))
    rate = min(args.max_rate, args.budget / ((rtt + fixed_us) / 1e6))
    return {
        "median_us": rtt,
        "sigma_bench_us": float(estimate_baseline(x, n, delta_c)["sigma_b"]),
        "sigma_us": sigma,
        "delta_c_us": delta_c,
        "N": N,
        "tau_us": claim_plan["tau"],
        "rate_per_s": rate,
        "window_s": N / rate if math.isfinite(N) else math.inf,
        "gpu_s_per_window": N * (rtt + fixed_us) / 1e6 if math.isfinite(N) else math.inf,
        "blocks": blocks,
    }


def rank_static(path: Path, args) -> list[dict]:
    d = json.loads(Path(path).read_text())
    words = d["meta"]["size_bytes"] // 8
    rows = []
    for (B, n, r), recs in _by_shape(d["records"]).items():
        if "honest" not in recs or not recs["honest"].get("correct", True):
            continue
        dc = _delta_c(recs, n, lambda name, rec: 1.0 if "@" not in name else float(name.split("@")[1]))
        dc = {k: v for k, v in dc.items() if v > 0}
        if not dc:
            continue
        cheapest = min(dc, key=dc.get)
        s = score(
            np.asarray(recs["honest"]["samples_ns"]) / 1e3, n, dc[cheapest], args, args.static_cost_us, words * 8 // B
        )
        rows.append({"chain": "static", "B": B, "n": n, "r": r, "cheapest": cheapest, "delta_c_by_cheat": dc, **s})
    return sorted(rows, key=lambda x: (x["window_s"], x["gpu_s_per_window"]))


def rank_dynamic(path: Path, args, r_fixed: int | None) -> list[dict]:
    d = json.loads(Path(path).read_text())
    meta = d["meta"]
    main, live = _by_shape(d["records"], "main"), _by_shape(d["records"], "live")
    rows = []
    for (B, n, r), recs in main.items():
        if r_fixed is not None and r != r_fixed:
            continue
        if "honest" not in recs:
            continue
        dc = _delta_c(recs, n, lambda name, rec: float(rec.get("deviating_share", 0.0)))
        dc = {k: v for k, v in dc.items() if v > 0}
        if not dc:
            continue
        cheapest = min(dc, key=dc.get)
        nb = meta["region_words"] * 8 // B
        nbl = meta["live_words"] * 8 // B
        for chain, rec, blocks in (
            ("main", recs["honest"], nb),
            ("live", (live.get((B, n, r)) or {}).get("honest"), nbl),
        ):
            if rec is None or blocks < args.min_blocks:
                continue
            scored = score(np.asarray(rec["samples_ns"]) / 1e3, n, dc[cheapest], args, args.dyn_cost_us, blocks)
            rows.append(
                {
                    "chain": f"dynamic {chain}",
                    "geometry": meta.get("geometry"),
                    "B": B,
                    "n": n,
                    "r": r,
                    "cheapest": cheapest,
                    "delta_c_by_cheat": dc,
                    **scored,
                }
            )
    return rows


def best_dynamic(rows: list[dict]) -> list[dict]:
    """Dynamic shapes ranked by the slower of the two chains' windows (one shape serves both chains of a freeze)."""
    by = {}
    for x in rows:
        by.setdefault((x["geometry"], x["B"], x["n"], x["r"]), {})[x["chain"]] = x
    out = []
    for (geo, B, n, r), cell in by.items():
        if "dynamic main" not in cell or "dynamic live" not in cell:
            continue
        w = max(cell["dynamic main"]["window_s"], cell["dynamic live"]["window_s"])
        out.append(
            {
                "geometry": geo,
                "B": B,
                "n": n,
                "r": r,
                "window_s": w,
                "main": cell["dynamic main"],
                "live": cell["dynamic live"],
            }
        )
    return sorted(out, key=lambda x: (x["window_s"], x["main"]["gpu_s_per_window"] + x["live"]["gpu_s_per_window"]))


def _kib(b):
    return f"{b >> 20}M" if b >= 1 << 20 else f"{b >> 10}K"


def markdown(sel: dict, top: int = 10) -> str:
    a = sel["assumptions"]
    lines = [
        "# Challenge parameters chosen from the sweeps",
        "",
        (
            f"F = {a['F']}, C = {a['C']}, network jitter {a['jitter_us']} µs in quadrature, GPU-time budget {a['budget']:.0%}, "
            f"at most {a['max_rate']} rounds a second; fixed cost a round: static {a['static_cost_us']} µs, dynamic {a['dyn_cost_us']} µs; "
            f"calibration {a['n_cal']} rounds."
        ),
        "",
    ]
    if sel.get("static"):
        lines += [
            "## Static chains",
            "",
            "| B | n | r | median µs | σ µs | cheapest cheat | Δc µs | N | τ µs | rounds/s | window | GPU s/window |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for x in sel["static"][:top]:
            lines.append(
                f"| {_kib(x['B'])} | {x['n']} | {x['r']} | {x['median_us']:.0f} | {x['sigma_us']:.1f} | {x['cheapest']} | {x['delta_c_us']:.2f} | "
                f"{x['N']:.0f} | {x['tau_us']:.2f} | {x['rate_per_s']:.2f} | {x['window_s'] / 60:.1f} min | {x['gpu_s_per_window']:.1f} |"
            )
        lines.append("")
    for geo, rows in sel.get("dynamic", {}).items():
        lines += [
            f"## Dynamic chains ({geo})",
            "",
            "| B | n | r | window (slower chain) | main: median µs, σ, Δc (cheat), N | live: median µs, σ, N |",
            "|---|---|---|---|---|---|",
        ]
        for x in rows[:top]:
            main_chain, live_chain = x["main"], x["live"]
            lines.append(
                f"| {_kib(x['B'])} | {x['n']} | {x['r']} | {x['window_s'] / 60:.1f} min | {main_chain['median_us']:.0f}, {main_chain['sigma_us']:.1f}, "
                f"{main_chain['delta_c_us']:.2f} ({main_chain['cheapest']}), {main_chain['N']:.0f} | {live_chain['median_us']:.0f}, {live_chain['sigma_us']:.1f}, {live_chain['N']:.0f} |"
            )
        lines.append("")
    if sel.get("choice"):
        lines += ["## Choice", "", "```", json.dumps(sel["choice"], indent=1), "```"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", type=Path, default=None)
    ap.add_argument("--dynamic", type=Path, action="append", default=[])
    ap.add_argument("--F", type=float, default=0.01)
    ap.add_argument("--C", type=float, default=0.999)
    ap.add_argument("--jitter-us", type=float, default=15.0)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--max-rate", type=float, default=2.0)
    ap.add_argument(
        "--static-cost-us", type=float, default=300.0, help="a static round's fixed cost (the drain's ARM and READY)"
    )
    ap.add_argument("--dyn-cost-us", type=float, default=1500.0, help="a dynamic round's fixed cost (the freeze)")
    ap.add_argument(
        "--n-cal", type=int, default=3000, help="calibration rounds the live verifier will take (the 1%% run's)"
    )
    ap.add_argument("--min-blocks", type=int, default=4)
    ap.add_argument("--r", type=int, default=None, help="the dynamic r (default: the static choice's)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args(argv)
    sel = {
        "assumptions": {
            k: getattr(args, k)
            for k in (
                "F",
                "C",
                "jitter_us",
                "budget",
                "max_rate",
                "static_cost_us",
                "dyn_cost_us",
                "min_blocks",
                "n_cal",
            )
        },
        "inputs": {"static": str(args.static) if args.static else None, "dynamic": [str(p) for p in args.dynamic]},
    }
    choice = {}
    r_dyn = args.r
    if args.static:
        sel["static"] = rank_static(args.static, args)
        if sel["static"]:
            best = sel["static"][0]
            choice["static"] = {
                k: best[k] for k in ("B", "n", "r", "N", "tau_us", "delta_c_us", "cheapest", "window_s", "rate_per_s")
            }
            r_dyn = r_dyn if r_dyn is not None else best["r"]
    sel["dynamic"] = {}
    for p in args.dynamic:
        rows = best_dynamic(rank_dynamic(p, args, r_dyn))
        if rows:
            geo = rows[0]["geometry"]
            sel["dynamic"][geo] = rows
            best = rows[0]
            choice[f"dynamic_{geo}"] = {
                "B": best["B"],
                "n": best["n"],
                "r": best["r"],
                "window_s": best["window_s"],
                "delta_c_us": best["main"]["delta_c_us"],
                "cheapest": best["main"]["cheapest"],
                "N_main": best["main"]["N"],
                "N_live": best["live"]["N"],
            }
    sel["choice"] = choice
    args.out.write_text(json.dumps(sel, indent=1, default=float))
    md = markdown(sel)
    if args.md:
        args.md.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
