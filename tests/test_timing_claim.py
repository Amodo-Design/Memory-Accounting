"""The statistical timing claim (memory_accounting.verifier.timing_claim, the design log v0.2): the log's worked example, the two
Bernstein inequalities at the sized N, clipping, and simulations of honest and cheating provers against the
window and period tests."""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from memory_accounting.verifier.timing_claim import (
    ClaimParams,
    F_floor,
    F_testable,
    MeanShiftTest,
    PeriodAnalysis,
    analyze_period,
    estimate_baseline,
    fn_gap,
    n_required,
    params_from_baseline,
    plan,
    room,
    term_range,
    threshold,
    var_cheat,
    var_honest,
)


def test_design_log_example():
    """The log's example: L = 16, Δ_c = 17 µs, σ = 25 µs, M = 300 µs, α = 1 − C = 10⁻⁹, F = 10⁻³, a baseline as
    long as the test, 10³ GPUs for a week: √N = 1,709, N ≈ 2.9 × 10⁶, λ ≈ 0.0048 per GPU per second. The exact
    solution keeps the (aM/3)² terms the closed form drops: N moves by about 3 × 10⁻⁵."""
    p = ClaimParams(F=1e-3, C=1 - 1e-9, L=16, delta_c=17.0, sigma_b=25.0, M=300.0, n_gpu=1000, T=6.05e5)
    out = plan(p)
    assert abs(math.sqrt(out["n_closed_form"]) - 1709) < 1, out
    assert abs(out["n_required"] - 2.92e6) / 2.92e6 < 0.005, out
    assert abs(out["n_required"] - out["n_closed_form"]) / out["n_closed_form"] < 1e-4, out
    assert abs(out["challenges_per_gpu_per_s"] - 0.0048) < 0.0001, out
    assert (
        var_honest(p, 1000) == 2 * 1000 * 25.0**2 and var_cheat(p, 1000) == 2 * 1000 * 625 + 1000 * 16 * 1e-3 * 17 * 300
    )


def test_both_inequalities_hold_at_the_sized_n():
    """At N the threshold t meets ineq 1 with equality (honest false positives exactly α by Bernstein) and the
    room left below N·L·F·Δ_c meets ineq 2; one chain fewer does not fit both."""
    for kw in ({}, {"n_cal": 500}, {"n_cal": 200}, {"sigma_s": 1.5, "t_corr": 60, "T": 120, "k_cal": 1}):
        p = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=20.0, sigma_margin=0.1, **kw)
        N = n_required(p)
        assert math.isfinite(N), kw
        t = threshold(p, N)
        g = fn_gap(p, N)
        M = term_range(p, N)
        fp = math.exp(-t * t / (2 * var_honest(p, N) + 2 / 3 * M * t))
        fn = math.exp(-g * g / (2 * var_cheat(p, N) + 2 / 3 * M * g))
        assert abs(fp - p.a) / p.a < 1e-9 and abs(fn - (1 - p.C)) / (1 - p.C) < 1e-9, (kw, fp, fn)
        assert room(p, N) >= 0 > room(p, N - 1) and t + g <= N * p.delta_F, kw
    # a calibration shorter than the test: its error and its range M·N/N_cal stay in every test, and set a floor on F
    q = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=20.0, n_cal=30)
    assert term_range(q, 300) == 10 * q.M_ and F_floor(q) > 0
    assert n_required(ClaimParams(**{**q.__dict__, "F": F_floor(q) * 0.9})) == math.inf


def test_drift_allowance():
    """A bound h on the honest level's drift (a margin on σ cannot absorb it: a larger σ makes N larger, which
    shrinks t/N while the drift stays): the threshold rises by N·h, the cheat's shift must clear 2h, and no N tests
    an F with L·F·Δ_c ≤ 2h. An honest prover whose level sits h above the calibration passes; one cheating at F
    whose level sits h below fails."""
    rng = np.random.default_rng(9)
    q0 = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=15.0, mu_b=1300.0, n_cal=1000)
    q = ClaimParams(**{**q0.__dict__, "drift": 4.0})
    N = n_required(q)
    assert N > n_required(q0) and abs(threshold(q, N) - threshold(q0, N) - 4.0 * N) < 1e-6
    assert F_floor(q) >= 8 / (8 * 17.0) and n_required(ClaimParams(**{**q.__dict__, "drift": 14.0})) == math.inf
    t = MeanShiftTest(q, int(N))
    up = sum(not t.judge(1304 + rng.normal(0, 15, int(N)))["pass"] for _ in range(300))
    down = sum(
        t.judge(1296 + rng.normal(0, 15, int(N)) + 17.0 * rng.binomial(8, 0.2, int(N)))["pass"] for _ in range(300)
    )
    assert up <= 1 and down <= 1, (up, down)


def test_clip_is_never_below_L_delta_c():
    try:
        ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=20.0, M=100.0)
        raise AssertionError("M below L·Δ_c accepted")
    except ValueError as e:
        assert "below L·Δ_c" in str(e)
    p = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=20.0, mu_b=1000.0)
    assert p.M_ == 136.0 and p.clip == 1136.0 and p.floor == 864.0


def test_baseline_is_clipped_as_the_test_clips():
    """μ is the mean of the calibration clipped at μ + M (a fixed point): an honest stall counts at the cap, as in
    a test, not at its full length, and not left out."""
    rng = np.random.default_rng(3)
    x = np.concatenate([1300 + rng.normal(0, 15, 5000), [5000.0] * 20])
    b = estimate_baseline(x, 8, 17.0)
    mu = b["mu_b"]
    assert (
        abs(np.clip(x, mu - 136, mu + 136).mean() - mu) < 1e-6
        and abs(b["clip"] - mu - 136) < 1e-9
        and b["clipped"] == 20
    )
    assert 1300 < mu < 1300 + 20 * 136 / 5020 + 1 and b["sigma_b"] < 25, b


def _honest(rng):
    return lambda n: 1300 + rng.normal(0, 15, n) + np.where(rng.random(n) < 0.005, rng.exponential(300, n), 0)


def test_simulated_provers():
    """Gaussian honest timing with rare stalls; a cheater at the claimed F with Δ_c a deviating read, its reads
    spread over chains or bunched in full chains. The sized window flags at most about α of honest windows and
    catches at least C of cheating ones; a prover that drops its replies instead counts at the clip."""
    rng = np.random.default_rng(7)
    L, dc, F, C = 8, 17.0, 0.2, 0.99
    honest = _honest(rng)
    base = estimate_baseline(honest(2000), L, dc)
    params = params_from_baseline(base, F, C, L, dc, sigma_margin=0.1)
    n = int(n_required(params))
    assert 20 < n < 5000, n
    test = MeanShiftTest(params, n)
    fp = sum(not test.judge(honest(n))["pass"] for _ in range(400)) / 400

    def cheat(n):
        return honest(n) + dc * rng.binomial(L, F, n)

    def bunched(n):
        return honest(n) + dc * L * (rng.random(n) < F)

    def dropped(n):
        return np.where(rng.random(n) < F, math.inf, honest(n))  # no reply to F of the chains

    det = [sum(not test.judge(f(n))["pass"] for _ in range(400)) / 400 for f in (cheat, bunched, dropped)]
    assert fp <= 0.03, fp
    assert min(det) >= C - 0.02, det
    v = test.judge([1300.0] * (n - 1) + [math.inf])
    assert v["clipped"] == 1 and abs(v["D_us"] - ((n - 1) * 1300 + params.clip - n * params.mu_b)) < 1e-6, v


def test_responses_below_the_floor_are_raised():
    """A response far below anything honest (a clock step under kernel timestamps) is raised to the floor, the
    higher of μ − M and the calibration's fastest − 3 σ_b, as a stall is lowered to the clip: it cannot drag a
    sum down and hide a slow cheat."""
    p = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17, sigma_b=10.0, mu_b=500.0, x_min=470.0)
    assert p.floor == 440.0 and p.clip == 636.0
    v = MeanShiftTest(p, 4).judge([500.0, 500.0, 500.0, -1e6])
    assert abs(v["mean_shift"] - (440.0 - 500.0) / 4) < 1e-9, v
    pa = PeriodAnalysis(p)
    for x in (500.0, -1e6, math.inf):
        pa.add(x, 0.0)
    assert pa.s1 == 500.0 + 440.0 + 636.0 and pa.clipped == 1


def test_period_model():
    """A period is judged by the same sum test at the N it holds: an honest period does not fail (false
    positives at most α at any N), a cheat at the target fails, a period with too few chains is inconclusive at
    the target and shows a larger F; slow movement of the honest level raises the floor on F."""
    rng = np.random.default_rng(11)
    L, dc, C = 8, 17.0, 0.99
    honest = _honest(rng)
    cal = honest(20000)
    n = 20000
    t = np.arange(n) * 0.5
    hv = [analyze_period(cal, 10000, honest(n), t, 0.05, C, L, dc)["verdict"] for _ in range(50)]
    assert all(v["test"] == "pass" for v in hv) and max(v["F_testable"] for v in hv) <= 0.05, [v["test"] for v in hv]
    cv = analyze_period(cal, 10000, honest(n) + dc * rng.binomial(L, 0.05, n), t, 0.05, C, L, dc)["verdict"]
    assert cv["test"] == "fail" and cv["implied_F"] > 0.03, cv
    short = analyze_period(cal, 10000, honest(200), t[:200], 0.05, C, L, dc)["verdict"]
    assert short["test"] == "inconclusive" and short["F_testable"] > 0.05 and short["n_required"] > 200, short
    wander = np.repeat(rng.normal(0, 8, n // 1200 + 1), 1200)[:n]
    mv = analyze_period(cal, 600, honest(n) + wander, t, 0.05, C, L, dc, sigma_s=8.0, t_corr=600)["verdict"]
    assert mv["F_floor"] > 0.05 and mv["n_required"] == math.inf and mv["test"] in ("pass", "inconclusive"), mv


def test_period_error_rates():
    """Over many periods at the target's N: honest periods fail at most about α, cheats at F (bunched in full
    chains, the most variance, or spread) pass at most about 1 − C. The calibration is drawn again for each."""
    rng = np.random.default_rng(5)
    L, dc, F, C = 8, 17.0, 0.05, 0.99
    honest = _honest(rng)

    def rate(cheat, trials=300):
        bad = 0
        for _ in range(trials):
            base = estimate_baseline(honest(5000), L, dc)
            p = params_from_baseline(base, F, C, L, dc)
            N = int(n_required(p))
            x = honest(N) + (cheat(N) if cheat else 0)
            pa = PeriodAnalysis(p)
            for i, xi in enumerate(x):
                pa.add(float(xi), i * 0.5)
            v = pa.verdict()
            bad += (v["test"] != "fail") if cheat else (v["test"] == "fail")
        return bad / trials

    assert rate(None) <= 0.02
    for cheat in (lambda n: dc * L * (rng.random(n) < F), lambda n: dc * rng.binomial(L, F, n)):
        assert rate(cheat) <= 0.03


def test_F_testable_falls_with_n():
    p = ClaimParams(F=0.2, C=0.999, L=8, delta_c=17.0, sigma_b=20.0, n_cal=2000)
    fs = [F_testable(p, n) for n in (100, 1000, 10000)]
    assert fs[0] > fs[1] > fs[2] > F_floor(p), fs
    assert (
        room(ClaimParams(**{**p.__dict__, "F": fs[1] * 1.001}), 1000)
        >= 0
        > room(ClaimParams(**{**p.__dict__, "F": fs[1] * 0.99}), 1000)
    )


def test_period_watch():
    """scripts/period_watch.py: the exclude file is read again before every report (and by --now), so a
    cheat window noted while the watcher runs counts; report.jsonl is strict JSON (an infinite n_required is
    null); a restarted watcher does not repeat periods already reported."""
    import contextlib
    import importlib.util
    import io
    import json
    import re
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "period_watch", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "period_watch.py"
    )
    pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    rng = np.random.default_rng(3)
    tmp = pathlib.Path(tempfile.mkdtemp())
    log, out, ex = tmp / "live.jsonl", tmp / "out", tmp / "exclude.txt"
    ex.write_text("")

    def ver(t, phase=None):
        return (
            json.dumps(
                {
                    "type": "verification",
                    "t": t,
                    **({"phase": phase} if phase else {}),
                    "challenges": [{"rtt_ns": float(1300 + rng.normal(0, 15)) * 1e3}],
                }
            )
            + "\n"
        )

    with open(log, "w") as f:
        f.writelines(ver(t, "calibration") for t in np.arange(0, 1000, 5.0))
        f.writelines(ver(t) for t in np.arange(1000, 2000, 5.0))
        f.write(
            '{"type": "claim_period_verdict", "t": 2000.0, "period": 1, "n": 200, "test": "inconclusive", "n_required": Infinity}\n'
        )
    args = pw.parse_args([str(log), str(out), "--exclude", str(ex)])
    out.mkdir()
    watch = {"state": {"cal": [], "run": [], "verdicts": []}, "pos": 0, "done": 0, "reported": set()}
    assert pw.catch_up(log, out, watch, args)

    def no_constants(c):
        raise ValueError(f"non-strict JSON token {c}")

    recs = [json.loads(line, parse_constant=no_constants) for line in (out / "report.jsonl").read_text().splitlines()]

    def split(rec):
        return rec["sections"][2]["calibration"]["n"] + rec["sections"][2]["claim"]["n"]

    assert len(recs) == 1 and split(recs[0]) == 200 and recs[0]["excluded_windows"] == 0
    assert any(r["n_required"] is None for r in recs[0]["sections"][0]["results"])  # F = 0.001: beyond any N
    assert "below the slow-movement floor" in (out / "report.txt").read_text()
    # a cheat window noted while the watcher runs: the next report leaves it out of the split
    with open(ex, "a") as f:
        f.write("1500 1800\n\n")  # a blank line is skipped
    with open(log, "a") as f:
        f.writelines(ver(t) for t in np.arange(2000, 3000, 5.0))
        f.write('{"type": "claim_period_verdict", "t": 3000.0, "period": 2, "n": 200, "test": "inconclusive"}\n')
    assert pw.catch_up(log, out, watch, args)
    recs = [json.loads(line, parse_constant=no_constants) for line in (out / "report.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and split(recs[1]) == 400 - 60 and recs[1]["excluded_windows"] == 1, split(recs[1])
    # --now honours it as well
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pw.main([str(log), str(tmp / "now"), "--exclude", str(ex), "--now"])
    text = buf.getvalue()
    m = re.search(
        r"split: first half calibrates, second half claims\n  calibration (\d+) chains .*; claim (\d+) chains", text
    )
    assert m and int(m.group(1)) + int(m.group(2)) == 400 - 60 and "1 noted cheat window(s)" in text, text[:400]
    # a restarted watcher skips the periods already reported
    w2 = {"state": {"cal": [], "run": [], "verdicts": []}, "pos": 0, "done": 0, "reported": pw.reported_periods(out)}
    assert w2["reported"] == {1, 2}
    pw.catch_up(log, out, w2, args)
    assert len((out / "report.jsonl").read_text().splitlines()) == 2
    assert pw.strict({"a": [math.inf, -math.inf, math.nan, np.float64(1.5), np.int64(2), np.bool_(True)]}) == {
        "a": [None, None, None, 1.5, 2, True]
    }


def test_period_cli_counts_missing_replies_at_the_clip():
    """A reply that never came is logged as rtt_ns null (a fault): the period command leaves it out of the
    calibration (honest timing) and counts it at the clip in the period (K_i = L, as the live verifier does),
    instead of raising a TypeError on it. --chain reads the dynamic or the live chain the same way."""
    import contextlib
    import io
    import json
    import tempfile

    from memory_accounting.verifier import timing_claim

    rng = np.random.default_rng(5)

    def rt():
        return float(1300 + rng.normal(0, 15)) * 1e3

    def dyn(t):
        return {"rtt_ns": None if t == 1500 else rt() / 2.5, "live": {"rtt_ns": rt() / 3}}

    def ver(t, phase=None, miss=False):
        return (
            json.dumps(
                {
                    "type": "verification",
                    "t": t,
                    **({"phase": phase} if phase else {}),
                    "challenges": [{"rtt_ns": None if miss else rt()}, {"rtt_ns": rt()}],
                    "dyn": dyn(t),
                }
            )
            + "\n"
        )

    log = pathlib.Path(tempfile.mkdtemp()) / "live.jsonl"
    with open(log, "w") as f:
        f.writelines(ver(t, "calibration", miss=t == 50) for t in np.arange(0, 1000, 5.0))
        f.writelines(ver(t, miss=t in (1200, 1700)) for t in np.arange(1000, 2000, 5.0))
    for chain, cal_missing, period_missing in (("static", 1, 2), ("dynamic", 0, 1), ("live", 0, 0)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            timing_claim.main(["period", str(log), "--F", "0.2", "--chain", chain, "--json"])
        out = json.loads(buf.getvalue())
        assert out["missing_replies"] == {"calibration": cal_missing, "period": period_missing}, (
            chain,
            out["missing_replies"],
        )
        assert out["verdict"]["n"] == {"static": 400, "dynamic": 200, "live": 200}[chain], out["verdict"]["n"]
        assert out["verdict"]["clipped"] >= period_missing, out["verdict"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        timing_claim.main(["period", str(log), "--F", "0.2"])
    assert "1 calibration static challenges left out, 2 period ones counted at the clip" in buf.getvalue(), (
        buf.getvalue()[:300]
    )


def test_miss_count():
    """The miss-count test of a chain: k is sized for a count over a sliding window tested after every round
    trip (`scan_threshold`); a prover late on every chain alarms at the k-th."""
    from math import comb

    from memory_accounting.verifier.timing_claim import MissCountTest, miss_threshold, scan_threshold

    for W, q, alpha in ((100, 0.01, 1e-3), (50, 0.005, 1e-3), (10, 0.01, 1e-3)):
        k = miss_threshold(W, q, alpha)

        def tail(k):
            return sum(comb(W, i) * q**i * (1 - q) ** (W - i) for i in range(k, W + 1))

        assert tail(k) <= alpha < tail(k - 1), (W, q, k)
        ks = scan_threshold(W, q, alpha)

        def scan(k):
            return W * q * sum(comb(W - 1, i) * q**i * (1 - q) ** (W - 1 - i) for i in range(k - 1, W))

        assert scan(ks) <= alpha < scan(ks - 1) and ks >= k, (W, q, ks)
    assert miss_threshold(100, 0.01, 1e-3) == 6 and scan_threshold(100, 0.01, 1e-3) == 7
    t = MissCountTest(100, 0.01, 1e-3)
    assert [t.add(True) for _ in range(7)] == [False] * 6 + [True] and t.progress()["misses"] == 0
    rng = np.random.default_rng(1)
    t = MissCountTest(100, 0.01, 1e-3)
    # honest, at the worst rate measured on an H200 NVL host
    assert sum(t.add(m) for m in rng.random(200_000) < 0.0045) <= 2


def test_miss_count_false_alarms_per_window():
    """Tested after every round trip, the count is `window` tests per window, not one: sized for one test (the
    binomial tail) an honest prover missing at exactly q alarmed 1.13e-2 times per 100 round trips at alpha 1e-2
    (2.3e-3 at 1e-3). Sized for the scan (`scan_threshold`) it stays under alpha."""
    from memory_accounting.verifier.timing_claim import MissCountTest

    W, q, a, N = 100, 0.01, 1e-2, 4_000_000
    t = MissCountTest(W, q, a)
    alarms = sum(t.add(m) for m in (np.random.default_rng(7).random(N) < q).tolist())
    assert alarms / (N / W) <= a, (t.k, alarms / (N / W))


def test_period_cli_skips_a_torn_line():
    """The log of a run still writing (or cut short) can end in a torn line: the period command skips it."""
    import contextlib
    import io
    import json
    import tempfile

    from memory_accounting.verifier import timing_claim

    rng = np.random.default_rng(2)
    log = pathlib.Path(tempfile.mkdtemp()) / "live.jsonl"
    with open(log, "w") as f:
        f.writelines(
            json.dumps(
                {
                    "type": "verification",
                    "t": t,
                    **({"phase": "calibration"} if t < 1000 else {}),
                    "challenges": [{"rtt_ns": float(1300 + rng.normal(0, 15)) * 1e3}],
                }
            )
            + "\n"
            for t in np.arange(0, 2000, 5.0)
        )
        f.write('{"type": "verification", "t": 2000.0, "challe')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        timing_claim.main(["period", str(log), "--F", "0.2", "--json"])
    assert json.loads(buf.getvalue())["verdict"]["n"] == 200


def test_period_cli_calibration_file_without_samples():
    """`period --calibration` with a baseline file that keeps no calibration samples (a baseline-role file from
    before they were kept) or none for the chain asked: a clear message, not a KeyError."""
    import json
    import tempfile

    from memory_accounting.verifier import timing_claim

    tmp = pathlib.Path(tempfile.mkdtemp())
    log = tmp / "live.jsonl"
    log.write_text(
        "".join(
            json.dumps({"type": "verification", "t": float(t), "challenges": [{"rtt_ns": 1.3e6}]}) + "\n"
            for t in range(10)
        )
    )
    for content, why in (
        ({"deadline_ns": 1.0}, "keeps no calibration_samples"),
        ({"calibration_samples": {"static_chain_rtt_ns": [1.3e6] * 30}}, "no live calibration responses"),
    ):
        (tmp / "b.json").write_text(json.dumps(content))
        try:
            timing_claim.main(
                [
                    "period",
                    str(log),
                    "--calibration",
                    str(tmp / "b.json"),
                    "--chain",
                    "live" if "live" in why else "static",
                ]
            )
            raise AssertionError("no exit")
        except SystemExit as e:
            assert why in str(e.code), e.code


if __name__ == "__main__":
    from _runner import run

    run(globals())
