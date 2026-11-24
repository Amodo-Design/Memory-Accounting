"""The analysis tools and scripts: configuration selection (coverage of every cheat, the shared 3x rule),
figures with strategies they have no colour for, the background load's device check, the period watcher
on a missing reply, the load-sweep script's grid and --force, and demo.sh's prover tracking and seed."""

import importlib.util
import inspect
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _runner import skip

from memory_accounting.core.analysis import MIN_RATIO, all_comparisons, best_config, expected_cheats, load_results

rng = np.random.default_rng(11)
META = {"size_bytes": 1 << 30, "device": "cuda", "setup_seconds": 1.0, "decode_gbps": 100.0, "cpu_threads": 8}


def _rec(prover, B, n, r, mean_us, correct=True, k=60):
    return {
        "prover": prover,
        "B": B,
        "n": n,
        "r": r,
        "correct": correct,
        "samples_ns": (rng.normal(mean_us, mean_us * 0.01, k) * 1e3).tolist(),
    }


def _write(records, meta=None) -> pathlib.Path:
    f = pathlib.Path(tempfile.mkdtemp()) / "sweep.json"
    f.write_text(json.dumps({"meta": {**META, **(meta or {})}, "records": records}))
    return f


def test_selection_needs_every_cheat():
    """A configuration whose cheat record was dropped as incorrect is not approved, however well the
    remaining cheats separate: the cheap 64 KB point (whose other_gpu record failed) loses to 256 KB."""
    recs = []
    for B, h in ((65536, 100.0), (262144, 200.0)):
        recs += [_rec("honest", B, 4, 0, h), _rec("cpu", B, 4, 0, 10 * h)]
    recs += [_rec("other_gpu", 65536, 4, 0, 100.0, correct=False), _rec("other_gpu", 262144, 4, 0, 2000.0)]
    res = load_results(_write(recs))
    assert res["dropped"] == [("other_gpu", 65536, 4, 0)] and expected_cheats(res) == ["cpu", "other_gpu"]
    comps = all_comparisons(res)
    for expected in (expected_cheats(res), None):  # explicit, and inferred from the comparisons
        key, table = best_config(comps, expected=expected)
        assert key == (262144, 4, 0), key
        row = next(t for t in table if t["B"] == 65536)
        assert not row["ok"] and row["missing"] == ["other_gpu"]
        assert "missing" not in next(t for t in table if t["B"] == 262144)  # complete rows unchanged
    # a cheat dropped at every configuration still counts: nothing passes
    recs = [
        _rec("honest", 65536, 4, 0, 100.0),
        _rec("cpu", 65536, 4, 0, 1000.0),
        _rec("other_gpu", 65536, 4, 0, 100.0, correct=False),
    ]
    res = load_results(_write(recs))
    key, table = best_config(all_comparisons(res), expected=expected_cheats(res))
    assert key is None and table[0]["missing"] == ["other_gpu"]


def test_selection_rule_is_shared():
    """memory_accounting.experiments.plots, memory_accounting.experiments.finalist and memory_accounting.experiments.loadcurve select with analysis.MIN_RATIO (3x): a cheat that is
    perfectly separable but only 2x slower does not pass, in the load curves either."""
    from memory_accounting.experiments import finalist, loadcurve, plots

    assert MIN_RATIO == 3.0 and inspect.signature(best_config).parameters["min_ratio"].default == MIN_RATIO
    assert inspect.signature(loadcurve.fig_detectability).parameters["min_ratio"].default == MIN_RATIO
    for mod in (plots, finalist):
        assert "MIN_RATIO" in inspect.getsource(mod.main)
    recs = [
        _rec("honest", 65536, 4, 0, 100.0),
        _rec("cpu", 65536, 4, 0, 200.0),
        _rec("honest", 262144, 4, 0, 200.0),
        _rec("cpu", 262144, 4, 0, 800.0),
    ]
    res = load_results(_write(recs))
    assert best_config(all_comparisons(res), min_ratio=1.0)[0] == (65536, 4, 0)
    assert best_config(all_comparisons(res))[0] == (262144, 4, 0)
    out = pathlib.Path(tempfile.mkdtemp())
    runs = loadcurve.load_all([str(_write(recs))])
    _, rows = loadcurve.fig_detectability(runs, out, 45.0)
    assert [r[2] for r in rows] == [(262144, 4, 0)], rows
    # nothing passes anywhere: the figure is still drawn (no log axis over nothing)
    runs = loadcurve.load_all([str(_write(recs[:2]))])
    _, rows = loadcurve.fig_detectability(runs, out, 45.0)
    assert rows[0][2] is None and (out / "detectability_vs_load.png").exists()


def test_finalist_reruns_on_the_sweeps_device_and_seed():
    """memory_accounting.experiments.finalist re-measures with the sweep's device and seed (bench records the seed in its meta)."""
    from memory_accounting.experiments import bench, finalist

    assert '"seed": args.seed' in inspect.getsource(bench.run)
    recs = [_rec("honest", 262144, 4, 0, 200.0), _rec("cpu", 262144, 4, 0, 2000.0)]
    sweep = _write(recs, {"device": "cuda:1", "seed": 7, "encoding": "vde"})
    calls = []

    def fake_main(argv):
        calls.append(argv)
        pathlib.Path(argv[argv.index("--out") + 1]).write_text(sweep.read_text())

    real = finalist.bench.main
    finalist.bench.main = fake_main
    try:
        finalist.main([str(sweep), "--samples", "10"])
    finally:
        finalist.bench.main = real
    a = calls[0]
    assert a[a.index("--device") + 1] == "cuda:1" and a[a.index("--seed") + 1] == "7", a


def test_loadcurve_summary_complete_configs():
    """The load curves' "best min-AUC" leaves out a configuration missing a cheat (its min-AUC is over the
    others only), and each summary row's min-AUC and weakest cheat belong to the configuration it names."""
    from memory_accounting.experiments import loadcurve

    recs = [
        _rec("honest", 65536, 4, 0, 100.0),
        _rec("cpu", 65536, 4, 0, 100.5),
        _rec("other_gpu", 65536, 4, 0, 1000.0),
        _rec("honest", 262144, 4, 0, 200.0),
        _rec("cpu", 262144, 4, 0, 5000.0),
        _rec("other_gpu", 262144, 4, 0, 200.0, correct=False),
    ]
    out = pathlib.Path(tempfile.mkdtemp())
    _, rows = loadcurve.fig_detectability(loadcurve.load_all([str(_write(recs))]), out, 0.0)
    _g, _frac, key, shown, auc, weak = rows[0]
    # not the 256 KB point (other_gpu dropped)
    assert key is None and shown == (65536, 4, 0) and weak == "cpu" and auc < 0.999, rows
    res = load_results(_write(recs))
    table = best_config(all_comparisons(res), expected=expected_cheats(res))[1]
    assert abs(auc - next(t["min_auc"] for t in table if t["B"] == 65536)) < 1e-12


def test_compare_margin_complete_configs():
    """memory_accounting.experiments.compare's weakest-cheat margin leaves out a configuration whose cheat record was dropped, and a
    background load that did not run is reported as such, not as a throughput."""
    from memory_accounting.experiments.compare import load_line, worst_cheat

    recs = [
        _rec("honest", 65536, 4, 0, 100.0),
        _rec("cpu", 65536, 4, 0, 1000.0),
        _rec("other_gpu", 65536, 4, 0, 100.0, correct=False),
        _rec("honest", 262144, 4, 0, 200.0),
        _rec("cpu", 262144, 4, 0, 2000.0),
        _rec("other_gpu", 262144, 4, 0, 900.0),
    ]
    res = load_results(_write(recs))
    w, incomplete = worst_cheat(res, all_comparisons(res))
    assert set(w) == {(262144, 4, 0)} and incomplete == [(65536, 4, 0)], (w, incomplete)
    assert "NOT RUNNING" in load_line({"tflops": None, "error": "RuntimeError: x"}) and "TFLOP/s" in load_line(
        {"tflops": 10.0, "bandwidth_gbps": 5.0}
    )


def test_background_load_dead_thread_reported():
    """A load thread that dies (here: no CUDA context to set) reports its error and no throughput, not
    iterations over a zero time (absurd TFLOP/s)."""
    import threading

    import torch

    from memory_accounting.experiments.load import BackgroundLoad

    bl = BackgroundLoad.__new__(BackgroundLoad)
    bl.dim, bl.inflight, bl.fraction, bl.iters, bl.elapsed, bl.error = 64, 4, 1.0, 0, 0.0, None
    bl.a = torch.zeros(1)
    bl.buf = torch.zeros(8)
    bl._stop = threading.Event()
    bl._thread = None
    assert bl.stop()["error"] == "the load thread was never started"
    bl._stop = threading.Event()
    bl._thread = threading.Thread(target=bl._loop)
    bl._thread.start()
    bl._thread.join(10)
    assert not bl.alive and bl.error, bl.error
    rep = bl.stop()
    assert rep["tflops"] is None and rep["bandwidth_gbps"] is None and rep["error"] == bl.error, rep


def test_plots_with_unknown_strategies():
    """A sweep with strategies the figures have no colour or label for (partial, lora) is drawn with
    default ones instead of raising."""
    from memory_accounting.experiments import plots

    recs = []
    for B in (65536, 1 << 20):
        for n in (1, 4):
            recs += [
                _rec("honest", B, n, 0, 30.0 * n),
                _rec("cpu", B, n, 0, 300.0 * n),
                _rec("partial", B, n, 0, 200.0 * n),
                _rec("lora", B, n, 0, 150.0 * n),
            ]
    out = pathlib.Path(tempfile.mkdtemp())
    plots.main([str(_write(recs)), "--outdir", str(out), "--jitter-us", "0.5"])
    s = json.loads((out / "summary.json").read_text())
    assert s["device"]["best"] is not None
    # 64 KB and 1 MB cost the same here: the noise of the shared generator picks the block size the dist figures use
    assert (out / "best_config_partial_lan.png").exists() and list(out.glob("dist_lora_B*_r0.png")), sorted(
        p.name for p in out.iterdir()
    )
    assert plots.cheat_order({"partial", "cpu", "lora", "other_gpu"}) == ["other_gpu", "cpu", "lora", "partial"]
    assert (
        plots.cheat_color("partial") == plots.cheat_color("partial") and plots.cheat_label("evict_dram") == "evict dram"
    )


def test_background_load_device():
    """--background-load accepts an explicit GPU index, and still refuses a CPU device."""
    from memory_accounting.experiments.load import BackgroundLoad, is_cuda

    assert is_cuda("cuda") and is_cuda("cuda:0") and is_cuda("cuda:1")
    assert not is_cuda("cpu") and not is_cuda("mps") and not is_cuda("nonsense")
    try:
        BackgroundLoad("cpu")
        raise AssertionError("accepted a CPU device")
    except RuntimeError as e:
        assert "CUDA" in str(e)


def test_period_watch_missing_reply():
    """net.py logs rtt_ns null for a chain with no reply: the watcher counts it, at the clip in a claim (as the
    live verifier does) and not in a calibration, and goes on, also when restarted on the same log."""
    spec = importlib.util.spec_from_file_location("period_watch", ROOT / "scripts" / "period_watch.py")
    pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    d = pathlib.Path(tempfile.mkdtemp())
    log, out = d / "live.jsonl", d / "out"

    def ver(t, phase=None, rtt=True):
        return (
            json.dumps(
                {
                    "type": "verification",
                    "t": t,
                    **({"phase": phase} if phase else {}),
                    "challenges": [{"rtt_ns": float(1300 + rng.normal(0, 15)) * 1e3 if rtt else None}],
                }
            )
            + "\n"
        )

    with open(log, "w") as f:
        f.writelines(ver(t, "calibration", rtt=(t != 50)) for t in np.arange(0, 1000, 5.0))
        f.writelines(ver(t, rtt=t not in (1200, 1500)) for t in np.arange(1000, 2000, 5.0))
        f.write('{"type": "claim_period_verdict", "t": 2000.0, "period": 1, "n": 198, "test": "inconclusive"}\n')
        f.writelines(ver(t, rtt=(t != 2100)) for t in np.arange(2000, 3000, 5.0))
        f.write('{"type": "claim_period_verdict", "t": 3000.0, "period": 2, "n": 199, "test": "inconclusive"}\n')
    args = pw.parse_args([str(log), str(out)])
    out.mkdir()
    for attempt in range(2):  # a restart reads the same lines again
        w = {
            "state": {"cal": [], "run": [], "verdicts": [], "missing": [], "missing_cal": []},
            "pos": 0,
            "done": 0,
            "reported": pw.reported_periods(out),
        }
        pw.catch_up(log, out, w, args)
        st = w["state"]
        assert (
            len(st["cal"]) == 199
            and len(st["run"]) == 400
            and st["missing"] == [1200.0, 1500.0, 2100.0]
            and st["missing_cal"] == [50.0]
        )
    recs = [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]
    assert [r["missing_replies"] for r in recs] == [
        {"period": 2, "so_far": 2, "calibration": 1},
        {"period": 1, "so_far": 3, "calibration": 1},
    ]
    assert recs[0]["sections"][0]["claim"]["n"] == 200 and recs[0]["sections"][0]["results"][0]["clipped"] >= 2, recs[
        0
    ]["sections"][0]
    assert "missing replies (no round trip, counted at the clip): 2 in this period" in (out / "report.txt").read_text()


def _period_watch():
    spec = importlib.util.spec_from_file_location("period_watch", ROOT / "scripts" / "period_watch.py")
    pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    return pw


def test_period_watch_one_chain_once():
    """The live verifier publishes a period verdict per chain (static, dynamic, live): the watcher reports
    each period once, on its own chain's verdict and round trips (static unless --chain), also when a
    verdict comes again within the run; torn and malformed lines are skipped."""
    pw = _period_watch()
    r = np.random.default_rng(3)
    tmp = pathlib.Path(tempfile.mkdtemp())
    log = tmp / "live.jsonl"

    def ver(t, phase=None):
        return (
            json.dumps(
                {
                    "type": "verification",
                    "t": t,
                    **({"phase": phase} if phase else {}),
                    "challenges": [{"rtt_ns": float(r.normal(100e3, 2e3))}],
                    "dyn": {"rtt_ns": float(r.normal(300e3, 3e3)), "live": {"rtt_ns": float(r.normal(500e3, 4e3))}},
                }
            )
            + "\n"
        )

    def verdicts(t, p):
        return "".join(
            json.dumps(
                {
                    "type": "claim_period_verdict",
                    "chain": ch,
                    "t": t + dt,
                    "period": p,
                    "n": 400,
                    "mean_shift": ms,
                    "F_testable": 0.1,
                    "tau": 1.0,
                    "test": "pass",
                }
            )
            + "\n"
            for ch, dt, ms in (("static", 0.001, 0.1), ("dynamic", 0.002, 5.0), ("live", 0.003, -3.0))
        )

    now = 1000.0
    with open(log, "w") as f:
        for _ in range(200):
            now += 0.5
            f.write(ver(now, "calibration"))
        for p in (1, 2):
            for i in range(400):
                now += 0.5
                f.write(ver(now))
                if p == 1 and i == 100:  # a verifier killed mid-line, then a restart appending
                    f.write(ver(now)[:40])
                    f.write('[1, 2]\n{"type": "verification", "t": "x", "challenges": []}\n')
                    f.write('{"type": "verification", "t": 1.0, "challenges": [{"rtt_ns": "slow"}]}\n')
            f.write(verdicts(now, p))
        f.write(verdicts(now, 2))  # period 2's verdicts again
    with open(log, "ab") as f:
        f.write(b'\xff\xfe{"type": "verification"}\n')
    for chain, shift, n_live in (("static", 0.1, 400), ("dynamic", 5.0, 400)):
        out = tmp / f"out_{chain}"
        out.mkdir()
        args = pw.parse_args([str(log), str(out), "--F", "0.01", "--chain", chain])
        w = {"state": pw.new_state(), "pos": 0, "done": 0, "reported": pw.reported_periods(out)}
        pw.catch_up(log, out, w, args)
        pw.catch_up(log, out, w, args)
        recs = [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]
        assert [x["period"] for x in recs] == [1, 2], [x["period"] for x in recs]
        assert all(x["live_verdict"]["mean_shift"] == shift for x in recs), recs
        # the bad lines skipped
        assert len(w["state"]["cal"]) == 200 and len(w["state"]["run"]) == 800, (
            len(w["state"]["cal"]),
            len(w["state"]["run"]),
        )
        assert recs[0]["sections"][0]["claim"]["n"] == 400 and recs[1]["sections"][1]["claim"]["n"] == 800, recs[0][
            "sections"
        ][0]
        mu = recs[0]["sections"][0]["results"][0]["baseline"]["mu_b"]
        assert abs(mu - {"static": 100, "dynamic": 300}[chain]) < 5, mu  # that chain's round trips, in µs


def test_period_watch_partial_period_at_stop():
    """The live verifier judges the open period when it stops (its verdict carries "partial"): the watcher reports
    it as such, "period N (partial, judged at stop)", and report.jsonl says so. Before: the field was dropped and
    the stop's verdict read as a normal period end."""
    pw = _period_watch()
    r = np.random.default_rng(5)
    d = pathlib.Path(tempfile.mkdtemp())
    log, out = d / "live.jsonl", d / "out"

    def ver(t, phase=None):
        return (
            json.dumps(
                {
                    "type": "verification",
                    "round": "static",
                    "t": t,
                    **({"phase": phase} if phase else {}),
                    "challenges": [{"rtt_ns": float(r.normal(100e3, 2e3))}],
                }
            )
            + "\n"
        )

    def verdict(t, p, **kw):
        return (
            json.dumps(
                {
                    "type": "claim_period_verdict",
                    "chain": "static",
                    "t": t,
                    "period": p,
                    "n": 100,
                    "mean_shift": 0.1,
                    "F_testable": 0.1,
                    "tau": 1.0,
                    "test": "inconclusive",
                    **kw,
                }
            )
            + "\n"
        )

    now = 1000.0
    with open(log, "w") as f:
        for _ in range(100):
            now += 1.0
            f.write(ver(now, "calibration"))
        for _ in range(200):
            now += 1.0
            f.write(ver(now))
        f.write(verdict(now, 1))
        for _ in range(60):
            now += 1.0
            f.write(ver(now))
        f.write(verdict(now, 2, partial=True))  # the stop, a fifth into period 2
    out.mkdir()
    args = pw.parse_args([str(log), str(out), "--F", "0.01"])
    w = {"state": pw.new_state(), "pos": 0, "done": 0, "reported": pw.reported_periods(out)}
    pw.catch_up(log, out, w, args)
    recs = [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]
    assert [(x["period"], x["partial"]) for x in recs] == [(1, False), (2, True)], recs
    assert recs[1]["sections"][0]["title"] == "period 2 (partial, judged at stop) alone, live calibration", recs[1][
        "sections"
    ][0]["title"]
    assert recs[1]["sections"][0]["claim"]["n"] == 60, recs[1]["sections"][0]["claim"]
    text = (out / "report.txt").read_text()
    assert "===== period 2 (partial, judged at stop), stopped " in text and "===== period 1, ended " in text, text[:400]


def test_period_watch_slow_movement_linear():
    """The watcher's slow-movement estimate is memory_accounting.verifier.timing_claim.slow_movement's, from one bincount a window
    length instead of a mask a window (quadratic in the run's length): the same rows on a few hours, and a
    day of responses in well under a second."""
    pw = _period_watch()
    from memory_accounting.verifier.timing_claim import slow_movement

    r = np.random.default_rng(5)
    times = np.sort(r.uniform(0, 4 * 3600, 20000))
    x = 100 + 2 * np.sin(times / 900) + r.normal(0, 3, len(times))
    a, b = slow_movement(times, x), pw.slow_movement(times, x)
    assert len(a["rows"]) == len(b["rows"]) > 3
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra["window_s"] == rb["window_s"] and ra["windows"] == rb["windows"]
        for k in ("mean_n", "sd_of_window_means", "sd_if_independent", "excess_var"):
            assert abs(ra[k] - rb[k]) <= 1e-9 * max(1.0, abs(ra[k])), (k, ra[k], rb[k])
    assert abs(a["sigma_s_estimate"] - b["sigma_s_estimate"]) < 1e-9
    times = np.arange(0, 86400, 0.2)
    x = 100 + r.normal(0, 3, len(times))  # a day at 5 a second
    t0 = time.perf_counter()
    pw.slow_movement(times, x)
    assert time.perf_counter() - t0 < 1.0, time.perf_counter() - t0


def test_run_load_sweeps_grid_and_force():
    """scripts/run_load_sweeps.sh measures the r values memory_accounting.experiments.loadcurve draws (and results/load/ holds),
    skips sweeps already on disk, and measures them again with --force."""
    from memory_accounting.experiments.loadcurve import KEY_CONFIGS

    tmp = pathlib.Path(tempfile.mkdtemp())
    fake = tmp / "fake_py.py"
    fake.write_text(
        "import sys, pathlib\n"
        "a = sys.argv[1:]\n"
        "open('calls.log', 'a').write(' '.join(a) + '\\n')\n"
        "if a[0] == '-c': sys.exit(1)\n"
        "if a[:2] == ['-m', 'memory_accounting.experiments.bench']: pathlib.Path(a[a.index('--out') + 1]).write_text('{}')\n"
    )
    env = {**os.environ, "PY": f"{sys.executable} {fake}", "FRACS": "0.5"}

    def run(*extra):
        (tmp / "calls.log").write_text("")
        p = subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_load_sweeps.sh"), *extra],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        return [
            line
            for line in (tmp / "calls.log").read_text().splitlines()
            if line.startswith("-m memory_accounting.experiments.bench")
        ]

    bench = run()
    assert len(bench) == 3 and all(
        (tmp / "results" / "load" / f).exists() for f in ("idle.json", "coarse_f0.5.json", "fine_f0.5.json")
    )
    from memory_accounting.core.params import parse_bytes

    grid = {
        k: {parse_bytes(x) for x in re.search(rf"{k} (\S+)", bench[0]).group(1).split(",")}
        for k in ("--blocks", "--links", "--r")
    }
    for B, n, r in KEY_CONFIGS:
        assert B in grid["--blocks"] and n in grid["--links"] and r in grid["--r"], (B, n, r, grid)
    idle = ROOT / "results" / "load" / "idle.json"
    if not idle.exists():  # a checkout without the committed results (e.g. a synced copy)
        skip("results/load not present: the grid was not checked against the measured sweeps")
    committed = {rec["r"] for rec in json.loads(idle.read_text())["records"]}
    assert grid["--r"] == committed, (grid["--r"], committed)
    assert run() == []  # all on disk: nothing measured
    assert len(run("--force")) == 3
    assert "#### " in (tmp / "results" / "load" / "load_sweeps.log").read_text()
    assert (
        subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_load_sweeps.sh"), "--bogus"], cwd=tmp, env=env, capture_output=True
        ).returncode
        != 0
    )


def _demo(fn: str, *args) -> subprocess.CompletedProcess:
    """Run one of demo.sh's functions locally (the script sourced: functions only)."""
    return subprocess.run(
        ["bash", "-c", f'source "{ROOT}/scripts/demo.sh"; {fn} "$@"', "x", *args], capture_output=True, text=True
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_demo_stops_only_its_prover():
    """demo.sh's stop runs prover_stop_script on the GPU host: it stops the pid it recorded at start, and
    that prover's EngineCore (which ignores SIGTERM and outlives its parent), and leaves another memory_accounting.prover.vllm_serve
    of the same user alone. A recorded pid that is no longer a posm process is not touched."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "launch.py").write_text(
        "import os, sys\nos.setsid()\n"
        "os.execvp('bash', ['bash', '-c', 'echo $$ > \"$0\"; exec \"$@\"'] + sys.argv[1:])\n"
    )
    (tmp / "leader.py").write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, 'core.py', 'VLLM::EngineCore'])\ntime.sleep(120)\n"
    )
    (tmp / "core.py").write_text(
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(120)\n"
    )

    def launch(pidfile, *cmd):  # as demo.sh starts it: setsid, detached, pid recorded
        subprocess.run(
            ["bash", "-c", '("$@" > /dev/null 2>&1 < /dev/null &)', "x", sys.executable, "launch.py", pidfile, *cmd],
            cwd=tmp,
            check=True,
        )
        for _ in range(100):
            if (tmp / pidfile).exists() and (tmp / pidfile).read_text().strip():
                return int((tmp / pidfile).read_text())
            time.sleep(0.05)
        raise AssertionError("no pid file")

    def group(pgid):
        ps = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,pgid=,args="], capture_output=True, text=True
        ).stdout
        return {int(line.split()[0]): line for line in ps.splitlines() if int(line.split()[1]) == pgid}

    script = _demo("prover_stop_script").stdout
    assert script.startswith("pidf=demo_prover.pid\n")
    (tmp / "bin").mkdir()
    (tmp / "bin" / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    (tmp / "bin" / "nvidia-smi").chmod(0o755)
    # an empty GPU: the script's wait for one is not this test's
    env = {
        **os.environ,
        "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
    }
    others = []
    try:
        decoy = launch("decoy.pid", sys.executable, "leader.py", "memory_accounting.prover.vllm_serve")
        others.append(decoy)
        prover = launch("demo_prover.pid", sys.executable, "leader.py", "memory_accounting.prover.vllm_serve")
        for _ in range(100):
            if len(group(prover)) == 2 and len(group(decoy)) == 2:
                break
            time.sleep(0.05)
        core = [p for p, line in group(prover).items() if "VLLM::EngineCore" in line]
        others += [p for p in group(decoy)]
        assert len(core) == 1
        result = subprocess.run(["bash", "-c", script], cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
        assert "prover stopped" in result.stdout, result.stdout + result.stderr
        time.sleep(0.2)
        assert not _alive(prover) and not _alive(core[0]) and not group(prover)
        assert all(_alive(x) for x in others) and len(group(decoy)) == 2  # the other vllm_serve and its EngineCore
        assert not (tmp / "demo_prover.pid").exists()
        # no pid file: nothing stopped
        result = subprocess.run(["bash", "-c", script], cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
        assert "nothing stopped" in result.stdout and all(_alive(x) for x in others)
        # a stale pid that now names something else: left alone
        stranger = launch("demo_prover.pid", "sleep", "60")
        others.append(stranger)
        result = subprocess.run(["bash", "-c", script], cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
        assert "not a demo prover" in result.stdout and _alive(stranger), result.stdout
    finally:
        for x in others:
            try:
                os.kill(x, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_demo_stops_only_its_verifier_and_watcher():
    """The live verifier and the period watcher are stopped by the pid each recorded at its start (no pkill -f,
    which stopped any of the user's matching processes on the verifier host): SIGINT first (a clean stop for a
    process that handles it), SIGTERM for one that ignores it. A decoy with the same command line is left
    alone, and so is a recorded pid that is now something else."""
    text = (ROOT / "scripts" / "demo.sh").read_text() + (ROOT / "scripts" / "period_run.sh").read_text()
    assert "pkill -f '^" not in text and 'pkill -f "^' not in text
    assert (
        "echo \\$\\$ > $LIVE_PIDFILE; exec $VERIFIER_PY -m memory_accounting.verifier.net live" in text
        and "echo \\$\\$ > $WATCH_PIDFILE; exec $VERIFIER_PY scripts/period_watch.py" in text
    )
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "fake.py").write_text(
        "import signal, sys, time\n"
        "if 'ignore-int' in sys.argv: signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "open(sys.argv[1], 'w').write('started')\n"
        "try:\n    time.sleep(120)\nexcept KeyboardInterrupt:\n    open(sys.argv[1], 'w').write('interrupted')\n"
    )
    procs = []

    def spawn(*args):
        # SIGINT back to its default in the child, as the live verifier does at start: a test runner started in
        # the background by a non-interactive shell hands its children SIGINT ignored
        p = subprocess.Popen(
            [sys.executable, "fake.py", *args],
            cwd=tmp,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),  # noqa: PLW1509 (no threads hold locks here)
        )
        procs.append(p)
        for _ in range(100):
            if (tmp / args[0]).exists():
                return p
            time.sleep(0.05)
        raise AssertionError("did not start")

    try:
        for pidfile, pat, what, words in (
            ("demo_live.pid", r"memory_accounting\.verifier\.net live", "live verifier", ["-m", "memory_accounting.verifier.net", "live"]),
            ("period_watch.pid", r"scripts/period_watch\.py", "period watcher", ["scripts/period_watch.py"]),
        ):
            script = _demo("pid_stop_script", pidfile, pat, what).stdout
            decoy = spawn(f"decoy_{pidfile}", *words)
            mine = spawn(f"mine_{pidfile}", *words)
            (tmp / pidfile).write_text(f"{mine.pid}\n")
            result = subprocess.run(["bash", "-c", script], cwd=tmp, capture_output=True, text=True, timeout=60)
            assert f"{what} stopped" in result.stdout, result.stdout + result.stderr
            # SIGINT: a clean stop
            assert mine.wait(timeout=10) is not None and (tmp / f"mine_{pidfile}").read_text() == "interrupted"
            assert decoy.poll() is None and not (tmp / pidfile).exists()
            result = subprocess.run(["bash", "-c", script], cwd=tmp, capture_output=True, text=True, timeout=60)
            assert "nothing stopped" in result.stdout and decoy.poll() is None
            (tmp / pidfile).write_text(f"{decoy.pid}\n")  # a pid that is no longer this tool: (here, sleep)
            other = subprocess.Popen(["sleep", "60"])
            procs.append(other)
            (tmp / pidfile).write_text(f"{other.pid}\n")
            result = subprocess.run(["bash", "-c", script], cwd=tmp, capture_output=True, text=True, timeout=60)
            assert f"not the {what} any more" in result.stdout and other.poll() is None, result.stdout
        # one that ignores SIGINT gets SIGTERM
        stubborn = spawn("stubborn", "ignore-int", "-m", "memory_accounting.verifier.net", "live")
        (tmp / "demo_live.pid").write_text(f"{stubborn.pid}\n")
        result = subprocess.run(
            [
                "bash",
                "-c",
                _demo("pid_stop_script", "demo_live.pid", r"memory_accounting\.verifier\.net live", "live verifier").stdout,
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "live verifier stopped" in result.stdout and stubborn.wait(timeout=10) is not None
    finally:
        for result in procs:
            if result.poll() is None:
                result.kill()


def test_demo_scripts_inputs():
    """demo.sh: the vLLM replay copies the verifier's log and checks the records against it (with the pad seed
    and GPU); the usage text states the --max-scratch bounds the commands use (one variable each); the repo
    directories that reach remote cd and rm -rf are refused when empty, / or unusual. period_run.sh's
    exclude takes two plain numbers before anything reaches the remote shell."""
    text = (ROOT / "scripts" / "demo.sh").read_text()
    assert (
        "--evidence evidence_demo --log $LIVE_LOG" in text
        and "rsync -a $VERIFIER_ADDR:$VERIFIER_DIR/$LIVE_LOG ./" in text
    )
    assert 'seed_arg="--filler-seed $seed --pad-seed $seed --gpu 0"' in text and "rm -rf $PROVER_DIR" not in text
    bounds = dict(re.findall(r"^(INT_MAX_SCRATCH|VLLM_MAX_SCRATCH)=(\S+)", text, re.MULTILINE))
    assert set(bounds) == {"INT_MAX_SCRATCH", "VLLM_MAX_SCRATCH"}
    assert (
        "--max-scratch $INT_MAX_SCRATCH" in text
        and "--max-scratch $VLLM_MAX_SCRATCH" in text
        and not re.search(r"--max-scratch \d", text)
    )
    usage = subprocess.run(["bash", str(ROOT / "scripts" / "demo.sh")], capture_output=True, text=True).stdout
    assert (
        f"more than {bounds['INT_MAX_SCRATCH']} of the GPU" in usage
        and f"at most {bounds['VLLM_MAX_SCRATCH']} unchallenged" in usage
    ), usage
    assert "${" not in usage
    for ok in ("LVHEM_MACC_SW", "work/LVHEM_MACC_SW", "~/x.y-z_1"):
        assert _demo("safe_dir", ok).returncode == 0, ok
    for bad in ("", "/", "a b", "x;rm -rf ~", "../x", "a/../b", "$(id)", "a'b"):
        assert _demo("safe_dir", bad).returncode != 0, bad
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "posm_period_run").write_text("20261019_120000\n")
    env = {**os.environ, "TMPDIR": str(d), "POSM_VERIFIER": "unreachable.invalid"}
    for args in (["abc", "1"], ["1", "2; rm -rf ~"], ["1'", "2"], ["1"], ["1", "2", "3"]):
        p = subprocess.run(
            ["bash", str(ROOT / "scripts" / "period_run.sh"), "exclude", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert p.returncode != 0 and "usage" in p.stdout, (args, p.stdout, p.stderr)


def test_demo_filler_seed():
    """demo.sh has no fixed filler seed: each start mints one on the verifier and gives the same value to
    the prover (every profile) and to the live verifier; replay checks the records' pad seed against it.
    Every profile gives its prover the weights' key too: no prover has a default for either."""
    text = (ROOT / "scripts" / "demo.sh").read_text()
    assert "5eed" not in text and "FILLER_SEED" not in text
    assert 'on_verifier "$VERIFIER_PY -m memory_accounting.setup.cli filler-seed"' in text
    prover_cmds = re.findall(r'prover_cmd="([^"]*)"', text)
    assert len(prover_cmds) == 3 and all("--filler-seed $filler_seed" in c for c in prover_cmds)
    assert all("--enc-key $ENC_KEY" in c for c in prover_cmds), prover_cmds
    assert "memory_accounting.verifier.net live --seed $seed --manifest $bundle/manifest.json --filler-seed $filler_seed" in text
    assert "--pad-seed $seed" in text
    assert _demo("valid_seed", "35584a65399af6df").returncode == 0
    for bad in ("", "5eed", "35584a65399af6dfx", "35584A65399AF6DF", "error: no module"):
        assert _demo("valid_seed", bad).returncode != 0, bad
    usage = subprocess.run(["bash", str(ROOT / "scripts" / "demo.sh")], capture_output=True, text=True)
    assert (
        usage.returncode == 1 and "scripts/demo.sh start" in usage.stdout and "mint a fresh filler seed" in usage.stdout
    )


def test_provers_need_the_key_and_seed():
    """The serving provers have no default weights' key or filler seed (vllm_serve had a fixed pair, the others
    seed 0): started without them they stop before loading anything."""
    for mod in ("memory_accounting.prover.intserve", "memory_accounting.prover.serve", "memory_accounting.prover.vllm_serve"):
        p = subprocess.run(
            [sys.executable, "-m", mod, "--model", "x"], capture_output=True, text=True, cwd=ROOT, timeout=300
        )
        assert p.returncode == 2 and "required: --enc-key, --filler-seed" in p.stderr, (mod, p.stderr)


def test_provers_stop_when_their_port_is_taken():
    """A prover whose challenge port another process holds stops at once, before loading its model (under vLLM,
    minutes of setup), naming the port."""
    import socket

    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("0.0.0.0", 0))
    port = holder.getsockname()[1]
    try:
        for mod in ("memory_accounting.prover.intserve", "memory_accounting.prover.serve", "memory_accounting.prover.vllm_serve"):
            cmd = [
                sys.executable,
                "-m",
                mod,
                "--model",
                "x",
                "--enc-key",
                "1",
                "--filler-seed",
                "1",
                "--port",
                str(port),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
            assert p.returncode == 1 and f"UDP port {port}" in p.stderr, (mod, p.returncode, p.stderr[-2000:])
    finally:
        holder.close()


def test_demo_live_verifier_settings():
    """demo.sh's live verifier: no --dyn-jitter-ms (live ignores it: its dynamic rounds come at their own random
    times); records kept up to POSM_EVIDENCE_MAX (default 100G, about 20 hours of the vllm profile; the vllm
    profile had 5G, under two hours), a size only since it reaches a remote command; the replay's cache pinned
    to the int prover's --max-len. Its stop waits 20 s after SIGINT before SIGTERM (its own stop runs up to 8 s
    of last checks); other processes keep 5 s."""
    text = (ROOT / "scripts" / "demo.sh").read_text()
    vargs = re.findall(r'vargs="([^"]*)"', text)
    assert len(vargs) == 3 and not [v for v in vargs if "--dyn-jitter-ms" in v or "--evidence-max-total" in v], vargs
    assert "--evidence-max-total $EVIDENCE_MAX" in text and re.search(
        r"^EVIDENCE_MAX=\$\{POSM_EVIDENCE_MAX:-100G\}", text, re.MULTILINE
    )
    assert "--max-len 512" in re.findall(r'prover_cmd="([^"]*)"', text)[0] and "--replay-max-len 512" in vargs[0]
    assert "pid_stop_script \"$LIVE_PIDFILE\" 'memory_accounting\\.verifier\\.net live' 'live verifier' 20" in text
    script = _demo("pid_stop_script", "x.pid", "pat", "what", "20").stdout
    assert "intwait=20" in script and "INT:$((intwait * 2))" in script
    assert "intwait=5" in _demo("pid_stop_script", "x.pid", "pat", "what").stdout
    d = pathlib.Path(tempfile.mkdtemp())
    env = {
        **os.environ,
        "TMPDIR": str(d),
        "POSM_VERIFIER": "unreachable.invalid",
        "POSM_PROVER": "unreachable.invalid",
        "POSM_EVIDENCE_MAX": "1G; rm -rf ~",
    }
    p = subprocess.run(
        ["bash", str(ROOT / "scripts" / "demo.sh"), "start", "int"], capture_output=True, text=True, env=env, timeout=30
    )
    assert p.returncode != 0 and "is not a size" in p.stderr, (p.stdout, p.stderr)


def test_demo_dynamic_block_and_dyn_dram():
    """POSM_DYN_B (default 1M for vllm, where 64K hides dyn_dram, and 64K for int, whose region is a few MiB) is the prover's region block (--dyn-block) and the verifier's --dyn-B in every
    profile, and the vllm replay's (the block of the last start, from its log); the int and vllm profiles offer
    dyn_dram at POSM_DYN_EVICT_FRACTION (default 0.5). Every flag the demo gives a prover or the verifier exists
    there. Both values reach remote commands: anything but a block size or a fraction in (0, 1] is refused
    before any host is reached."""
    text = (ROOT / "scripts" / "demo.sh").read_text()
    assert re.search(r"^DYN_B=\$\{POSM_DYN_B:-\}", text, re.MULTILINE)
    assert '[ -n "$DYN_B" ] || { [ "$profile" = vllm ] && DYN_B=1M || DYN_B=64K; }' in text
    assert re.search(r"^DYN_EVICT_FRACTION=\$\{POSM_DYN_EVICT_FRACTION:-0\.5\}", text, re.MULTILINE)
    assert "dyn_dram" in re.search(r"^CHEATS=(\S+)", text, re.MULTILINE).group(1).split(",")
    cmds, vargs = re.findall(r'prover_cmd="([^"]*)"', text), re.findall(r'vargs="([^"]*)"', text)
    assert len(cmds) == len(vargs) == 3
    assert all("--dyn-block $DYN_B" in c for c in cmds) and all("--dyn-B $DYN_B" in v for v in vargs), (cmds, vargs)
    assert not re.search(r"--dyn-(B|block) \d", text), "a fixed dynamic block is left"
    int_cmd, sim_cmd, vllm_cmd = cmds
    for prover_cmd in (int_cmd, vllm_cmd):
        assert "--strategies $CHEATS" in prover_cmd and "--dyn-evict-fraction $DYN_EVICT_FRACTION" in prover_cmd, (
            prover_cmd
        )
    assert 'echo "$DYN_B" > "$DYN_B_STATE"' in text and "--log $LIVE_LOG --dyn-B $dyn_b" in text

    def flags(*cmd):
        p = subprocess.run(
            [sys.executable, "-m", *cmd, "--help"], capture_output=True, text=True, cwd=ROOT, timeout=300
        )
        assert p.returncode == 0, p.stderr
        return set(re.findall(r"--[A-Za-z][A-Za-z0-9-]*", p.stdout))

    def used(s):
        return set(re.findall(r"--[A-Za-z][A-Za-z0-9-]*", s))

    intserve, serve, net = flags("memory_accounting.prover.intserve"), flags("memory_accounting.prover.vllm_serve"), flags("memory_accounting.verifier.net")
    for prover_cmd, have in ((int_cmd, intserve), (sim_cmd, intserve), (vllm_cmd, serve)):
        assert used(prover_cmd) <= have, used(prover_cmd) - have
    for v in vargs:
        assert used(v) <= net, used(v) - net
    assert {"--dyn-B"} <= flags("memory_accounting.verifier.vllm_replay", "replay")
    for ok in ("64K", "256K", "1M"):
        assert _demo("valid_block", ok).returncode == 0, ok
    for bad in ("", "64", "0K", "064K", "1G", "64k", "1M; rm -rf ~", "$(id)"):
        assert _demo("valid_block", bad).returncode != 0, bad
    for ok in ("0.5", ".5", "1", "1.0", "0.25"):
        assert _demo("valid_fraction", ok).returncode == 0, ok
    for bad in ("", "0", "0.0", ".", "1.5", "2", "abc", "0.5;x"):
        assert _demo("valid_fraction", bad).returncode != 0, bad
    d = pathlib.Path(tempfile.mkdtemp())
    base = {
        **os.environ,
        "TMPDIR": str(d),
        "POSM_VERIFIER": "unreachable.invalid",
        "POSM_PROVER": "unreachable.invalid",
    }
    for var, val, why in (
        ("POSM_DYN_B", "1M; rm -rf ~", "is not a block size"),
        ("POSM_DYN_EVICT_FRACTION", "0.5 x", "is not a fraction"),
    ):
        p = subprocess.run(
            ["bash", str(ROOT / "scripts" / "demo.sh"), "start", "int"],
            capture_output=True,
            text=True,
            env={**base, var: val},
            timeout=30,
        )
        assert p.returncode != 0 and why in p.stderr and "stopping" not in p.stdout, (var, p.stdout, p.stderr)


def test_runner_reports_each_failure_and_skip():
    """tests/_runner.py runs every test of a file whatever failed before it, reports a skip as skipped (before, a test
    that could not run returned and was reported passed) and exits 1 only for a failure. Under pytest the same skip
    is pytest's, reported at the test's line."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "test_x.py").write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'tests')!r})\nfrom _runner import skip\n"
        "def test_fails():\n    assert False, 'boom'\n"
        "def test_skips():\n    skip('no widget')\n"
        "def test_passes():\n    pass\n"
        "if __name__ == '__main__':\n    from _runner import run\n    run(globals())\n"
    )
    result = subprocess.run([sys.executable, str(tmp / "test_x.py")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "boom" in result.stderr, (result.stdout, result.stderr)
    assert [line.split(" (")[0] for line in result.stdout.splitlines()] == [
        "FAIL test_fails",
        "skip test_skips: no widget",
        "ok test_passes",
        "1 of 3 skipped: test_skips",
        "1 of 3 failed: test_fails",
    ], result.stdout
    result = subprocess.run(
        [sys.executable, str(tmp / "test_x.py"), "test_skips", "test_passes"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and "1 of 2 skipped" in result.stdout, (result.stdout, result.stderr)
    if importlib.util.find_spec("pytest"):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-rs", "-p", "no:cacheprovider", "test_x.py"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            result.returncode == 1
            and "1 failed, 1 passed, 1 skipped" in result.stdout
            and "test_x.py:7: no widget" in result.stdout
        ), result.stdout


if __name__ == "__main__":
    from _runner import run

    run(globals())
