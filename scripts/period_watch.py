"""Watch a live verifier's log in the period model and run the offline timing check after every period.

Runs on the verifier host beside `memory_accounting.verifier.net live --claim-mode period`. Each time a period ends (a
`claim_period_verdict` line appears) it appends to `<out>/report.txt` and `<out>/report.jsonl`:

* the period alone, against the live verifier's calibration (what the live page showed);
* everything since the first period, against the live calibration (the whole run so far);
* split: the run's honest data so far, its first half as calibration and its second half as the claim.
  A calibration as long as the claim is what the document asks for (K_cal = K); the live calibration
  is a few minutes and caps the run's floor however long it continues. Valid for a test run in which the
  prover is honest (or cheats only in known windows, left out with --exclude; the file is read again
  before every analysis, so a window added while the watcher runs counts from the next report on).

Each analysis is the sum test of the design log v0.2 (memory_accounting.verifier.timing_claim: each response clipped at μ + L·Δ_c, a
missing reply at the clip), given for the assumed slow movement (σ_s, conservative) and for each target F: the
threshold does not depend on the target, only whether the period holds enough chains to show it.

When the live verifier stops mid-period it judges the open period on what it has (its verdict carries
`partial`): that period is reported too, marked "partial, judged at stop", over its shorter span.

One chain is checked: the static one by default (--chain); the live verifier publishes a period verdict for
each chain it runs, and only that chain's verdicts end a period here. Only that chain's round trips are kept,
as two float arrays (16 bytes a response), so a run of a week stays in memory.

    python scripts/period_watch.py demo_live.jsonl runs/period_20261016 --F 0.01 0.001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_accounting.verifier.timing_claim import analyze_period, chain_rtts_us


class Rows:
    """(time s, round trip µs) of one chain, in log order, as two float arrays: a list of tuples costs about
    ten times the memory, which a run of days or weeks notices."""

    def __init__(self, pairs=()):
        self.t, self.x = array("d"), array("d")
        for t, x in pairs:
            self.append(t, x)

    def append(self, t: float, x: float) -> None:
        self.t.append(t)
        self.x.append(x)

    def __len__(self) -> int:
        return len(self.t)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.t, dtype=np.float64), np.array(self.x, dtype=np.float64)  # copies: the arrays keep growing


def new_state() -> dict:
    return {"cal": Rows(), "run": Rows(), "verdicts": [], "missing": [], "missing_cal": []}


def _rows(state: dict, key: str) -> Rows:
    if not isinstance(state[key], Rows):  # a caller's plain list of (t, µs)
        state[key] = Rows(state[key])
    return state[key]


# what the report reads of a verdict
VERDICT_KEYS = (
    "t",
    "chain",
    "period",
    "n",
    "mean_shift",
    "tau",
    "F_testable",
    "test",
    "partial",
)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load(log: Path, pos: int, state: dict, chain: str = "static") -> int:
    """Read the log from byte `pos` on; returns the position after the last whole line. A line that does not
    parse (torn: a verifier killed mid-write, and the next run's first line appended to it) or carries no
    usable time is skipped; a line still being written (no newline yet) is read next time."""
    cal, run = _rows(state, "cal"), _rows(state, "run")
    with open(log, "rb") as f:
        f.seek(pos)
        for line in f:
            if not line.endswith(b"\n"):
                break  # a line still being written: read it next time
            pos += len(line)
            try:
                e = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(e, dict) or not _number(e.get("t")):
                continue
            when = float(e["t"])
            if e.get("type") == "verification":
                if e.get("phase") == "calibration":
                    if e.get("warmup"):
                        continue
                    rows, miss = cal, "missing_cal"
                else:
                    rows, miss = run, "missing"
                try:
                    xs = chain_rtts_us(e, chain)
                except (TypeError, AttributeError, KeyError):
                    continue  # a malformed record: nothing to time
                for x in xs:
                    if x is None:  # no reply (net.py logs null): no round trip to time
                        state.setdefault(miss, []).append(when)
                        if rows is run:
                            rows.append(
                                when, math.inf
                            )  # in a claim it counts at the clip, as the live verifier counts it
                    elif _number(x):
                        rows.append(when, float(x))
            elif e.get("type") == "claim_period_verdict" and (e.get("chain") or "static") == chain:
                # another chain's verdict does not end a period here
                state["verdicts"].append({k: e.get(k) for k in VERDICT_KEYS})
    return pos


WINDOWS_S = (5, 10, 20, 30, 60, 120, 300, 600, 1200)


def slow_movement(t_s, x_us, windows_s=None) -> dict:
    """memory_accounting.verifier.timing_claim.slow_movement in O(N) a window length: the window means come from one bincount
    instead of a mask over every response for every window (O(N²/w): seconds at a day of data, minutes at a
    week). Same rows, same estimate."""
    times = np.asarray(t_s, dtype=np.float64)
    x = np.asarray(x_us, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    times, x = times[order], x[order]
    span = times[-1] - times[0]
    if windows_s is None:
        windows_s = [w for w in WINDOWS_S if w <= span / 3]
    s2 = x.var(ddof=1)
    rows = []
    for window in windows_s:
        idx = np.floor((times - times[0]) / window).astype(np.int64)
        counts = np.bincount(idx)
        sums = np.bincount(idx, weights=x)
        keep = counts >= 3
        if keep.sum() < 3:
            continue
        means, cnt = sums[keep] / counts[keep], counts[keep]
        nbar = float(np.mean(cnt))
        obs = float(np.var(means, ddof=1))
        expect = s2 / nbar
        rows.append(
            {
                "window_s": window,
                "windows": int(keep.sum()),
                "mean_n": nbar,
                "sd_of_window_means": math.sqrt(obs),
                "sd_if_independent": math.sqrt(expect),
                "excess_var": obs - expect,
            }
        )
    excess = [r["excess_var"] for r in rows if r["window_s"] >= 30]
    sigma_s = math.sqrt(max(0.0, float(np.median(excess)))) if excess else None
    return {"rows": rows, "sigma_s_estimate": sigma_s}


def section(title, cal, run, F_list, args):
    """One analysis: calibration and claim responses, each a (times s, µs) pair of arrays."""
    (cal_t, cal_x), (ts, xs) = cal, run
    fin = np.isfinite(cal_x)
    cal_t, cal_x = cal_t[fin], cal_x[fin]  # a missing reply is no honest timing: not in a calibration
    if len(cal_x) < 20 or len(xs) < 20:
        return {"title": title, "skipped": "too few responses"}, f"{title}: too few responses\n"
    cal_span = float(cal_t[-1] - cal_t[0])
    fin = np.isfinite(xs)
    sm = slow_movement(ts[fin], xs[fin])
    span = float(ts[-1] - ts[0])
    out = {
        "title": title,
        "calibration": {"n": len(cal_x), "span_s": cal_span},
        "claim": {"n": len(xs), "span_s": span},
        "slow_movement_measured_us": sm["sigma_s_estimate"],
        "results": [],
    }
    lines = [
        f"{title}",
        (
            f"  calibration {len(cal_x)} chains over {cal_span / 60:.1f} min; claim {len(xs)} chains over {span / 60:.1f} min; "
            f"slow movement measured in the claim data {sm['sigma_s_estimate'] if sm['sigma_s_estimate'] is None else round(sm['sigma_s_estimate'], 2)} µs"
        ),
    ]
    for sigma_s in args.sigma_s:
        tests = []
        for F in F_list:
            r = analyze_period(cal_x, cal_span, xs, ts, F, args.C, args.L, args.delta_c, sigma_s, args.t_corr)
            verdict = r["verdict"]
            out["results"].append(
                {
                    "sigma_s": sigma_s,
                    "F": F,
                    "test": verdict["test"],
                    "D_us": verdict["D_us"],
                    "t_us": verdict["t_us"],
                    "mean_shift": verdict["mean_shift"],
                    "tau": verdict["tau"],
                    "F_testable": verdict["F_testable"],
                    "F_floor": verdict["F_floor"],
                    "n_required": r["plan"]["n_required"],
                    "clipped": verdict["clipped"],
                    "baseline": {k: r["baseline"][k] for k in ("mu_b", "sigma_b", "clip")},
                }
            )
            nr = r["plan"]["n_required"]
            tests.append(
                f"F {F:g}: {verdict['test']}"
                + (
                    f" (needs N = {nr:.3g})"
                    if verdict["test"] == "inconclusive" and nr != float("inf")
                    else " (below the slow-movement floor)"
                    if verdict["test"] == "inconclusive"
                    else ""
                )
            )
        lines.append(
            f"  σ_s {sigma_s:g} µs: sum test {'FAILS  <-- CHEATING DETECTED' if verdict['test'] == 'fail' else 'passes'} at false-alarm rate "
            f"{1 - args.C:g} (mean shift {verdict['mean_shift']:+.2f} µs, threshold {verdict['tau']:.2f} µs; {verdict['clipped']} at the clip); "
            f"the least F these chains can show {verdict['F_testable']:.3g}, floor {verdict['F_floor']:.3g}"
        )
        lines.append("      the test at each F: " + "; ".join(tests))
    return out, "\n".join(lines) + "\n"


def read_exclude(path) -> list:
    """The cheating windows noted so far, one 'start_unix end_unix' per line. Read again before every
    analysis, so a window noted (`period_run.sh exclude`) while the watcher runs counts from the next report."""
    if not path:
        return []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            a, b = map(float, line.split()[:2])
        except ValueError:
            continue  # blank or half-written line
        out.append((a, b))
    return out


def excluded(ts: np.ndarray, excl: list) -> np.ndarray:
    """Which times fall in a noted cheat window."""
    out = np.zeros(len(ts), dtype=bool)
    for a, b in excl:
        out |= (ts >= a) & (ts < b)
    return out


def strict(obj):
    """What report.jsonl holds: strict JSON, with an infinite or undefined number (an n_required beyond any
    N) as null, which a browser's JSON.parse accepts where it rejects Infinity and NaN."""
    if isinstance(obj, dict):
        return {k: strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()  # a numpy scalar
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def missing_line(missing: dict) -> str:
    """Chains with no reply (rtt_ns null in the log): counted at the clip in the analyses below, as the live
    verifier counts them (a calibration's are left out of it)."""
    if not any(missing.values()):
        return ""
    return (
        f"  missing replies (no round trip, counted at the clip): {missing['period']} in this period, {missing['so_far']} so far"
        + (f", {missing['calibration']} in the calibration" if missing.get("calibration") else "")
        + "\n"
    )


def report_period(k: int, state: dict, args, out: Path) -> None:
    """The offline analysis of period k (0-based) once its verdict is in the log: appended to report.txt
    and report.jsonl."""
    excl = read_exclude(args.exclude)
    verdict = state["verdicts"][k]
    cal = _rows(state, "cal").arrays()
    rt, rx = _rows(state, "run").arrays()
    t_end = verdict["t"]
    t_start = state["verdicts"][k - 1]["t"] if k > 0 else (rt[0] if len(rt) else t_end)
    period = (rt >= t_start) & (rt < t_end)
    sofar = rt < t_end
    honest = np.flatnonzero(sofar & ~excluded(rt, excl))
    half = len(honest) // 2
    partial = bool(verdict.get("partial"))  # the open period, judged when the verifier stopped
    label = f"period {verdict['period']}" + (" (partial, judged at stop)" if partial else "")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t_end))
    miss = state.get("missing", [])
    missing = {
        "period": sum(1 for t in miss if (k == 0 or t_start <= t) and t < t_end),
        "so_far": sum(1 for t in miss if t < t_end),
        "calibration": len(state.get("missing_cal", [])),
    }
    rec = {
        "period": verdict["period"],
        "partial": partial,
        "time_utc": stamp,
        "live_verdict": {k2: verdict.get(k2) for k2 in ("n", "mean_shift", "tau", "F_testable", "test")},
        "excluded_windows": len(excl),
        "missing_replies": missing,
        "sections": [],
    }
    text = [
        f"===== {label}, {'stopped' if partial else 'ended'} {stamp} UTC"
        + (f"; {len(excl)} noted cheat window(s) left out of the split" if excl else "")
        + " =====\n",
        missing_line(missing),
    ]
    for title, c, r in (
        (f"{label} alone, live calibration", cal, (rt[period], rx[period])),
        ("whole run so far, live calibration", cal, (rt[sofar], rx[sofar])),
        (
            "whole run so far, split: first half calibrates, second half claims",
            (rt[honest[:half]], rx[honest[:half]]),
            (rt[honest[half:]], rx[honest[half:]]),
        ),
    ):
        o, t = section(title, c, r, args.F, args)
        rec["sections"].append(o)
        text.append(t)
    with open(out / "report.txt", "a") as f:
        f.write("".join(text) + "\n")
    with open(out / "report.jsonl", "a") as f:
        f.write(json.dumps(strict(rec), allow_nan=False) + "\n")


def catch_up(log: Path, out: Path, watch: dict, args) -> bool:
    """Read what the log gained and report every period that ended since; True if the log grew."""
    if not log.exists():
        return False
    new = load(log, watch["pos"], watch["state"], getattr(args, "chain", "static"))
    grew = new != watch["pos"]
    watch["pos"] = new
    while watch["done"] < len(watch["state"]["verdicts"]):
        p = watch["state"]["verdicts"][watch["done"]].get("period")
        if p not in watch["reported"]:
            report_period(watch["done"], watch["state"], args, out)
            watch["reported"].add(p)  # once per period, also within this run
        watch["done"] += 1
    return grew


def reported_periods(out: Path) -> set:
    """Periods already in <out>/report.jsonl, so that a restarted watcher does not report them again."""
    try:
        with open(out / "report.jsonl") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return set()
    done = set()
    for line in lines:
        try:
            done.add(json.loads(line).get("period"))  # older reports may hold Infinity; Python reads it
        except (json.JSONDecodeError, AttributeError):
            continue
    return done - {None}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--F", type=float, nargs="+", default=[0.01, 0.001])
    ap.add_argument("--C", type=float, default=0.999)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--delta-c", type=float, default=17.0)
    ap.add_argument(
        "--sigma-s",
        type=float,
        nargs="+",
        default=[8.0, 3.5],
        help="slow movement to assume, µs (8 conservative, 3.5 measured in honest runs on an H200 NVL host)",
    )
    ap.add_argument("--t-corr", type=float, default=600.0)
    ap.add_argument(
        "--exclude",
        default=None,
        help="file of 'start_unix end_unix' lines: cheating windows to leave out of the split analysis (read before every analysis)",
    )
    ap.add_argument("--idle-exit-s", type=float, default=1800, help="stop when the log has not grown for this long")
    ap.add_argument(
        "--now",
        action="store_true",
        help="print the analysis of everything up to now (the period in progress too) and exit; the report is not touched",
    )
    ap.add_argument(
        "--chain",
        default="static",
        choices=["static", "dynamic", "live"],
        help="the chain to check: its period verdicts and its round trips (default static; for dynamic or live, "
        "give that chain's --L (the verifier's --dyn-n) and --delta-c (its --claim-delta-c-dyn))",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log, out = Path(args.log), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = new_state()
    if args.now:
        excl = read_exclude(args.exclude)
        load(log, 0, state, args.chain)
        cal = state["cal"].arrays()
        rt, rx = state["run"].arrays()
        t_last = state["verdicts"][-1]["t"] if state["verdicts"] else (rt[0] if len(rt) else time.time())
        current = rt >= t_last
        honest = np.flatnonzero(~excluded(rt, excl))
        half = len(honest) // 2
        part = sum(1 for v in state["verdicts"] if v.get("partial"))
        print(
            f"===== now: {len(state['verdicts']) - part} period(s) finished"
            + (f", {part} judged partial at a stop" if part else "")
            + f", {int(current.sum())} chains in the period in progress"
            + (f", {len(excl)} noted cheat window(s) left out of the split" if excl else "")
            + " ====="
        )
        print(
            missing_line(
                {
                    "period": sum(1 for t in state["missing"] if t >= t_last or not state["verdicts"]),
                    "so_far": len(state["missing"]),
                    "calibration": len(state["missing_cal"]),
                }
            ),
            end="",
        )
        for title, c, r in (
            ("period in progress, live calibration", cal, (rt[current], rx[current])),
            ("whole run so far, live calibration", cal, (rt, rx)),
            (
                "whole run so far, split: first half calibrates, second half claims",
                (rt[honest[:half]], rx[honest[:half]]),
                (rt[honest[half:]], rx[honest[half:]]),
            ),
        ):
            print(section(title, c, r, args.F, args)[1])
        return
    w = {"state": state, "pos": 0, "done": 0, "reported": reported_periods(out)}
    last_growth = time.time()
    while True:
        if catch_up(log, out, w, args):
            last_growth = time.time()
        if time.time() - last_growth > args.idle_exit_s:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
