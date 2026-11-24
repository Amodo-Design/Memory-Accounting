"""The verifier and the prover over a real UDP socket on loopback (CPU prover, synthetic array): every
datagram the verifier sends is recorded by a relay in between. Checks what crosses the wire before the
timed window: the ARM token is never the challenge's nonce, and challenges are not reproducible from
the verifier's --seed. The relay also drops, duplicates and delays replies: a lost or stale reply fails
one verification and the run goes on."""

import json
import pathlib
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools

from memory_accounting.core import wire
from memory_accounting.prover import udp as udp_prover
from memory_accounting.verifier import net


def _free_port() -> int:
    """A port free for UDP and TCP both: a relay's port also carries a record proxy (`_tcp_proxy`), and an earlier
    test's proxy still listens on its own; a port taken over from one sent records to the wrong place."""
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        p = sock.getsockname()[1]
        t = socket.socket()
        try:
            t.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            sock.close()
            t.close()


def _relay(listen: int, target: int, log: list, fault=None, times: list | None = None):
    """Forward datagrams verifier <-> prover, keeping (direction, bytes) (and the arrival time in `times`); a
    dropped one is kept as (direction + "-dropped", bytes): it never reached the other side.
    fault(direction, bytes) -> delays in s, one copy forwarded per entry: [] drops it, [0, 0] duplicates it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", listen))
    client = None
    while True:
        data, addr = sock.recvfrom(2048)
        way = "up" if addr[1] == target else "down"
        if way == "down":
            client = addr
        delays = fault(way, data) if fault else [0]
        log.append((way if delays else way + "-dropped", data))
        if times is not None:
            times.append(time.perf_counter())
        dest = client if way == "up" else ("127.0.0.1", target)
        for d in delays:
            if d:
                threading.Timer(d, sock.sendto, (data, dest)).start()
            else:
                sock.sendto(data, dest)


def _nth(way: str, size: int, picks: dict):
    """A fault: the k-th datagram (1-based) of this direction and size gets picks[k] (drop [], duplicate [0, 0], delay [s])."""
    seen = [0]

    def f(w, data):
        if w != way or len(data) != size:
            return [0]
        seen[0] += 1
        return picks.get(seen[0], [0])

    return f


def _both(*faults):
    def f(w, data):
        out = [0]
        for x in faults:
            r = x(w, data)
            if r != [0]:
                out = r
        return out

    return f


_PROVER = {}


def _prover() -> int:
    if "port" not in _PROVER:  # (the dynamic prover may have been started first)
        port = _free_port()
        threading.Thread(
            target=udp_prover.main,
            args=(["--bind", "127.0.0.1", "--port", str(port), "--size", "4M", "--device", "cpu"],),
            daemon=True,
        ).start()
        time.sleep(1.5)
        _PROVER["port"] = port
    return _PROVER["port"]


def _run_verifier(seed: int) -> list:
    """One short verifier run through the relay; returns the datagrams it sent."""
    relay, log = _free_port(), []
    threading.Thread(target=_relay, args=(relay, _prover(), log), daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "udp.json"
        net.main(
            [
                "verifier",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--samples",
                "6",
                "--warmup",
                "1",
                "--drain",
                "--strategies",
                "honest",
                "--seed",
                str(seed),
                "--verify-all",
                "--out",
                str(out),
            ]
        )
        res = json.loads(out.read_text())
    assert all(r["correct"] for r in res["records"]), "digests over the relay do not match the verifier's"
    return [data for way, data in log if way == "down"], res


def test_arm_token_is_not_the_nonce():
    sent, _res = _run_verifier(1)
    arms = [wire.ARM.unpack(d) for d in sent if len(d) == wire.ARM.size and wire.ARM.unpack(d)[0] == wire.ARM_MAGIC]
    chs = [wire.CH.unpack(d) for d in sent if len(d) == wire.CH.size and wire.CH.unpack(d)[0] == wire.MAGIC]
    assert arms and len(arms) == len(chs), (len(arms), len(chs))
    for a, c in zip(arms, chs):
        assert a[1] != c[1], "ARM carries the challenge nonce before the timed window"


def test_challenges_do_not_follow_from_the_seed():
    one = [wire.CH.unpack(d)[1:3] for d in _run_verifier(1)[0] if len(d) == wire.CH.size]
    two = [wire.CH.unpack(d)[1:3] for d in _run_verifier(1)[0] if len(d) == wire.CH.size]
    assert one and two and not set(one) & set(two), "the same --seed gave the same (nonce, start block)"


def _record(**over):
    from memory_accounting.core.evidence import Evidence

    ch = {"block_bytes": 1 << 16, "n": 32, "r": 256, "k": 1024, "g": 1, "j0": 0}
    ev = Evidence(
        nonce=5,
        challenge=dict(ch),
        digest=[1, 2],
        words=4096,
        block_bytes=1 << 16,
        pad_seed=0,
        layout=[],
        stage={"kind": "toy", "step": 0, "request_id": 0, "chunks": [], "act_layer": 0, "in_flight": False},
    )
    for k, v in over.items():
        if k in ch:
            ev.challenge[k] = v
        else:
            setattr(ev, k, v)
    return ev


def test_record_must_describe_the_challenge_sent():
    """A record is checked against the whole challenge, the region and the READY size: replay recomputes
    what the record says, so a record of an easier chain or a smaller region must not pass."""
    from memory_accounting.core.params import Challenge

    ch = Challenge(1 << 16, 32, 256, nonce=5, j0=0)
    assert net._record_mismatch(_record(), ch, (1, 2), 4096, "main") is None
    for over, why in (
        ({"n": 1, "r": 0}, "n 1 (sent 32)"),
        ({"k": 512}, "k 512"),
        ({"block_bytes": 1 << 12}, "block_bytes"),
        ({"words": 18}, "READY reported 4096"),
        ({"region": "live"}, "live region"),
        ({"digest": [1, 3]}, "reply"),
    ):
        msg = net._record_mismatch(_record(**over), ch, (1, 2), 4096, "main")
        assert msg and why in msg, (over, msg)


def test_record_with_another_pad_seed_or_gpu_fails():
    """The pad seed and GPU index a replay builds the filler from are the verifier's, not the record's: a
    record that claims others fails before any replay."""
    from types import SimpleNamespace

    import memory_accounting.core.evidence as E
    from memory_accounting.core.params import Challenge

    ch = Challenge(1 << 16, 32, 256, nonce=5, j0=0)
    real = E.fetch_evidence
    try:
        for over, why in (({"pad_seed": 0x77}, "pad seed"), ({"model": {"gpu": 1}}, "GPU 1")):
            ev = _record(**over)
            E.fetch_evidence = lambda *a, ev=ev, **k: ev
            rep = net._check_record(
                SimpleNamespace(host="x", port=0, _dyn_expected=(0, 0)), None, ch, (1, 2), words=4096
            )
            assert rep["digest_ok"] is False and why in rep["error"], rep
        ev = _record()
        E.fetch_evidence = lambda *a, **k: ev
        assert (
            net._check_record(SimpleNamespace(host="x", port=0, _dyn_expected=(0, 0)), None, ch, (1, 2), words=4096)[
                "digest_ok"
            ]
            is None
        )
    finally:
        E.fetch_evidence = real


def test_malformed_record_is_a_failed_check():
    """A record the prover malformed (here: no request_id in its stage) fails its check; it must not raise
    in the verifier's only checking thread."""
    from types import SimpleNamespace

    import memory_accounting.core.evidence as E
    from memory_accounting.core.params import Challenge

    bad = _record()
    bad.stage = {"kind": "toy", "step": 1}
    real = E.fetch_evidence
    E.fetch_evidence = lambda *a, **k: bad
    try:
        rep = net._check_record(
            SimpleNamespace(host="x", port=0), None, Challenge(1 << 16, 32, 256, nonce=5, j0=0), (1, 2), words=4096
        )
    finally:
        E.fetch_evidence = real
    assert rep["digest_ok"] is False and "malformed record" in rep["error"], rep


def test_memory_claim_rejects_impossible_totals():
    G = 1 << 30
    ok = net._memory_claim(140 * G, 76 * G, 58 * G, 1 * G, 7 * G)
    assert not ok["impossible"] and not ok["over"] and ok["unchecked_bytes"] == 7 * G  # the live 1 GiB is not credited
    assert net._memory_claim(140 * G, 76 * G, 100 * G, 0, None)["impossible"]  # more than the GPU holds
    assert net._memory_claim(140 * G, 76 * G, 58 * G, 59 * G, None)["impossible"]  # live region larger than its pool


def test_memory_credit_is_outside_the_live_region_in_whole_blocks():
    """The dynamic region is credited outside its live region (live KV comes from tokens the prover chooses and
    may compress: checked, never credited) and in whole --dyn-B blocks (the chain reads whole blocks; the words
    past the last one are read by nothing). Before: words x 8, the live region and a partial block included."""
    B, G = 1 << 16, 1 << 30
    c = net._memory_claim(4 * G, G, 10 * B + 100, 3 * B, None, 64, B)
    assert (
        c["dynamic_bytes"] == 7 * B and c["dynamic_region_bytes"] == 10 * B + 100 and c["live_region_bytes"] == 3 * B
    ), c
    assert c["unchecked_bytes"] == 3 * G - 7 * B and not c["over"]
    # a partial live block takes its block
    assert net._memory_claim(4 * G, G, 10 * B, 3 * B + 5, None, 64, B)["dynamic_bytes"] == 6 * B
    # the dtype's bits, as before
    assert net._memory_claim(4 * G, G, 64 * B, 0, None, 59, B)["dynamic_bytes"] == 64 * B * 59 // 64


def test_gaps_between_verifications_are_random():
    g = [net._gap(0.5) for _ in range(4000)]
    assert len(set(g)) == len(g) and 0.45 < sum(g) / len(g) < 0.55, sum(g) / len(g)


def test_synthetic_array_does_not_depend_on_torch():
    import numpy as np

    from memory_accounting.core import memory

    a = memory.random_words_np(1 << 12, 3)
    if memory.torch is not None:
        assert np.array_equal(memory.random_words(1 << 12, "cpu", 3).numpy(), a)


def test_lost_and_duplicate_replies_do_not_stop_the_verifier():
    """Drop one static reply, duplicate a READY and a reply: the run finishes, the dropped one fails its
    verification (inf, counted as missing), every reply that came is checked and correct. Before: the drop
    raised TimeoutError, the stale READY struct.error, and the run ended."""
    relay, log = _free_port(), []
    fault = _both(_nth("up", wire.RE.size, {3: [], 5: [0, 0]}), _nth("up", wire.READY.size, {4: [0, 0]}))
    threading.Thread(target=_relay, args=(relay, _prover(), log, fault), daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "udp.json"
        net.main(
            [
                "verifier",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--samples",
                "6",
                "--warmup",
                "1",
                "--drain",
                "--strategies",
                "honest",
                "--verify-all",
                "--reply-timeout",
                "0.5",
                "--out",
                str(out),
            ]
        )
        rec = json.loads(out.read_text())["records"][0]
    assert rec["missing"] == 1 and sum(t == float("inf") for t in rec["samples_ns"]) == 1, rec
    assert rec["correct"] and rec["incorrect_samples"] == 0, rec


_LIVE = {}


def _live_run() -> dict:
    """One live run under a timing claim through a faulty relay, after calibration: a READY delayed past its
    bound but under 3 x the bound, one over 3 x, a lost reply, a duplicated reply and READY. Returns the log's events."""
    if _LIVE:
        return _LIVE
    W, C = 1, 24
    last_cal = W + C  # the last calibration verification
    relay, log, times = _free_port(), [], []
    fault = _both(
        _nth("up", wire.READY.size, {last_cal + 2: [0.15], last_cal + 7: [0, 0], last_cal + 9: [0.35]}),
        _nth("up", wire.RE.size, {last_cal + 4: [], last_cal + 6: [0, 0]}),
    )
    threading.Thread(target=_relay, args=(relay, _prover(), log, fault, times), daemon=True).start()
    ui = socket.socket()
    ui.bind(("127.0.0.1", 0))
    ui_port = ui.getsockname()[1]
    ui.close()
    import contextlib
    import io

    printed = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(printed):
        out = pathlib.Path(d) / "live.jsonl"
        net.main(
            [
                "live",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--device",
                "cpu",
                "--drain",
                "--strategies",
                "honest",
                "--warmup",
                str(W),
                "--calibrate",
                str(C),
                "--interval",
                "0.2",
                "--arm-max-ms",
                "100",
                "--claim-F",
                "0.5",
                "--claim-mode",
                "period",
                "--reply-timeout",
                "0.5",
                "--max-verifications",
                "12",
                "--ui-port",
                str(ui_port),
                "--out",
                str(out),
            ]
        )
        evs = [json.loads(line) for line in out.read_text().splitlines()]
    print(printed.getvalue(), end="")
    _LIVE.update(events=evs, log=log, times=times, v=last_cal, printed=printed.getvalue())
    return _LIVE


def test_dynamic_delta_c_follows_the_block():
    """The dynamic chains' Δ_c is the measured least cost of a link from host memory for their --dyn-B block (H200,
    bench_dyn_dram.py): 0.3 µs at 64 KiB, 3.5 at 256 KiB, 17 at 1 MiB, linear between and proportional outside.
    Before: 2 µs whatever the block, above what 64 KiB costs."""
    d = net.dyn_delta_c_us

    def near(a, b):
        return abs(a - b) < 1e-9

    assert near(d(64 << 10), 0.3) and near(d(256 << 10), 3.5) and near(d(1 << 20), 17.0)
    assert near(d(32 << 10), 0.15) and near(d(2 << 20), 34.0) and 3.5 < d(512 << 10) < 17.0


def test_live_monitoring_survives_lost_and_stale_replies():
    """A lost reply under a timing claim (--challenge-flags warn, the default): a flag, not an alert (the design log
    v0.2: no single challenge alerts), and the claim counts the chain at the clip."""
    r = _live_run()
    evs, last_cal = r["events"], r["v"]
    vers = [e for e in evs if e.get("type") == "verification"]
    assert len(vers) == last_cal + 12, len(vers)  # it ran to the end
    lost = [e for e in vers if e.get("faults")]
    assert (
        [e["id"] for e in lost] == [last_cal + 4]
        and not lost[0]["within"]
        and lost[0]["challenges"][0]["rtt_ns"] is None
    ), lost
    assert lost[0].get("faults_flagged") and not lost[0]["gross"] and lost[0]["flagged"], lost[0]
    assert not [e for e in evs if e.get("type") == "alert" and e.get("kind") == "missing"]
    flags = [e for e in evs if e.get("type") == "flag" and e.get("kind") == "missing"]
    assert [a["id"] for a in flags] == [last_cal + 4] and flags[0]["outside_claim"] and flags[0]["chain"] == "static", (
        flags
    )
    part = [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("partial")]
    assert part and part[0]["n"] == 12 and part[0]["clipped"] >= 1, part  # the lost chain is in the sum, at the clip
    # the verdict keeps its event time: its threshold is t_us (a field "t" would replace the time, as v0.14.0 did)
    assert part[0]["t"] > 1.7e9 and part[0]["t_us"] > 0 and "D_us" in part[0], part[0]
    checks = {e["id"]: e for e in evs if e.get("type") == "check"}
    assert all(checks[e["id"]]["digest_ok"] for e in vers if not e.get("faults")), (
        "a reply that came was not checked or wrong"
    )


def test_late_ready_under_a_timing_claim():
    """A READY over --ready-ceiling (3) × its bound puts its round's chains at the clip in their claim (whatever the wait
    reloaded, the round counts as cheated); on a chain whose claim runs it is a flag under --challenge-flags warn (the
    default), not an alert: over 21 honest hours under vLLM, 1 READY in 5000 passed 3 × its bound, up to about 100 ms.
    One late but under the ceiling is counted (a miss count of late READYs, k of the last window), not alerted on its own."""
    r = _live_run()
    evs, last_cal = r["events"], r["v"]
    assert "timing claim, period model" in r["printed"], "the timing claim was not active; the test needs it"
    vers = {e["id"]: e for e in evs if e.get("type") == "verification"}
    assert not [e for e in evs if e.get("type") == "alert" and e.get("kind") == "late_ready"], (
        "a late READY alerted under the claim"
    )
    flags = [e for e in evs if e.get("type") == "flag" and e.get("kind") == "late_ready"]
    # 3.5 x the bound
    assert [f["id"] for f in flags] == [last_cal + 9] and flags[0]["chain"] == "static" and flags[0]["outside_claim"], (
        flags
    )
    assert (
        vers[last_cal + 9]["ready_clipped"]
        and vers[last_cal + 9]["late_ready_flagged"]
        and not vers[last_cal + 9]["gross"]
    ), vers[last_cal + 9]
    # 1.5 x: counted
    assert (
        vers[last_cal + 2]["late_ready"]
        and vers[last_cal + 2]["late_ready_weighed"]
        and not vers[last_cal + 2].get("ready_clipped")
    ), vers[last_cal + 2]
    part = [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("partial")]
    assert part and part[0]["clipped"] >= 2, part  # the lost reply (v + 4) and the READY past the ceiling (v + 9)


def test_stats_are_fetched_before_the_gap_not_after_it():
    """The STATS round trip comes right after a verification's last reply, before the random gap; never
    between the gap and the next ARM, where it would warn the prover and give it an untimed reply to hold."""
    r = _live_run()
    log, times = r["log"], r["times"]
    stats = [
        i
        for i, (w, d) in enumerate(log)
        if w == "down" and len(d) == wire.STATS.size and wire.STATS.unpack(d)[0] == wire.STATS_MAGIC
    ]
    stats = stats[1:]  # the first is the strategy negotiation at start
    assert stats, "no STATS request in the run"
    # else the verification ended on a timeout
    after = [times[i] - times[i - 1] for i in stats if log[i - 1][0] == "up"]
    assert after and max(after) < 0.03, f"STATS {1e3 * max(after):.0f} ms after the last reply: after the gap"


def test_negotiation_drops_strategies_the_prover_does_not_serve():
    from types import SimpleNamespace

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def serve():
        data, addr = srv.recvfrom(512)
        m, tok = wire.STATS.unpack(data)
        srv.sendto(
            wire.STATSRE.pack(
                m, tok, *([0] * 9), wire.strategy_mask(["honest", "dram_copy"]), wire.strategy_mask(["honest", "evict"])
            ),
            addr,
        )

    threading.Thread(target=serve, daemon=True).start()
    args = SimpleNamespace(host="127.0.0.1", port=port)
    assert net._negotiate(args, ["honest", "dram_copy", "partial", "lora"]) == ["honest", "dram_copy"]
    t = time.perf_counter()
    assert net._negotiate(args, ["honest", "partial"]) == ["honest", "partial"]  # no answer now: kept as given
    assert time.perf_counter() - t < 3


def _trickle_server(chunks, delay: float):
    """A TCP evidence server that sends `chunks` one at a time, `delay` s apart."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        c, _ = srv.accept()
        c.recv(64)
        try:
            for ch in chunks:
                c.sendall(ch)
                time.sleep(delay)
        except OSError:
            pass
        c.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def test_evidence_fetch_has_a_whole_fetch_deadline():
    """A prover that trickles its record a byte at a time cannot hold the checking thread: the fetch fails
    at its deadline (each recv's timeout alone never fires)."""
    from memory_accounting.core.evidence import HEAD, MAGIC, fetch_evidence

    h = json.dumps(_record().header()).encode()
    port = _trickle_server([HEAD.pack(MAGIC, len(h), 0)] + [bytes([b]) for b in h], 0.05)
    t = time.perf_counter()
    try:
        fetch_evidence("127.0.0.1", port, 5, timeout=1.0)
        raise AssertionError("a trickled record was accepted")
    except TimeoutError:
        pass
    assert time.perf_counter() - t < 2.0


def test_evidence_lengths_are_bounded():
    import numpy as np

    from memory_accounting.core.evidence import HEAD, MAGIC, fetch_evidence

    port = _trickle_server([HEAD.pack(MAGIC, 1 << 40, 0)], 0.0)
    try:
        fetch_evidence("127.0.0.1", port, 5, timeout=2.0)
        raise AssertionError("a 1 TB header was accepted")
    except ValueError as e:
        assert "limit" in str(e)
    ev = _record()
    ev.data = np.arange(4096, dtype=np.uint64)  # 32 kB of debug bytes over a 1 kB cap: not read
    port = _trickle_server([ev.to_wire()], 0.0)
    got = fetch_evidence("127.0.0.1", port, 5, timeout=2.0, max_data=1024)
    assert got.data is None and "bytes_dropped" in got.debug and got.nonce == 5


def test_stalled_record_is_a_failed_check():
    from types import SimpleNamespace

    from memory_accounting.core.evidence import HEAD, MAGIC
    from memory_accounting.core.params import Challenge

    port = _trickle_server([HEAD.pack(MAGIC, 100, 0), b"{"], 5.0)
    t = time.perf_counter()
    rep = net._check_record(
        SimpleNamespace(host="127.0.0.1", port=port, evidence_timeout=0.5),
        None,
        Challenge(1 << 16, 32, 256, nonce=5, j0=0),
        (1, 2),
        words=4096,
    )
    assert rep["digest_ok"] is False and "no record" in rep["error"] and time.perf_counter() - t < 2.0, rep


def test_drift_fallback_compares_the_minimum():
    """With an old baseline (no chain statistics) the CUSUM compares the verification's minimum, the statistic
    the baseline is of; the first chain (one in m slow) drifted falsely."""
    base = {"median_ns": 1000.0, "sigma_core_ns": 10.0}
    assert net._cusum_step(0.0, [1100.0, 1000.0, 1000.0, 1000.0], base, 0.5) == 0.0
    assert net._cusum_step(0.0, [None, 1030.0], base, 0.5) == 2.5
    assert net._cusum_step(1.0, [None], base, 0.5) == 1.0  # nothing came: unchanged


def _args(**kw):
    """The verifier's parsed command line (defaults), with these values set."""
    from types import SimpleNamespace

    ns = SimpleNamespace(
        manifest=None,
        B=None,
        r=None,
        k=None,
        n=8,
        min_r=2048,
        allowed_B="256K,512K,1M",
        allowed_k="1024",
        repeat=1,
        drain=True,
        timing="user",
        dynamic=False,
        dyn_B="64K",
        dyn_n=None,
        dyn_live=False,
    )
    ns.__dict__.update(kw)
    return ns


def test_shape_policy_is_the_verifiers():
    """The manifest's challenge shape is held to the verifier's policy: a hash-only (r = 0) or light setup is
    refused, as is a --B/--r/--k that differs from the manifest's (before: silently replaced by the manifest's)."""
    from types import SimpleNamespace

    def spec(**kw):
        return SimpleNamespace(**{"block_bytes": 1 << 20, "r": 2048, "k": 1024, **kw})

    assert net._shape(_args(manifest="m.json"), spec()) == (1 << 20, 8, 2048, 1024, 1)
    # the same, said again: fine
    assert net._shape(_args(manifest="m.json", r=2048, B="1M"), spec()) == (
        1 << 20,
        8,
        2048,
        1024,
        1,
    )
    for over, args, why in (
        ({"r": 0}, {}, "below the verifier's minimum"),
        ({"r": 256}, {}, "below the verifier's minimum"),
        ({"block_bytes": 64 << 10}, {}, "block size 65536"),
        ({"k": 128}, {}, "k = 128"),
        ({}, {"r": 0}, "--r 0 (the manifest's is 2048)"),
        ({}, {"B": "256K"}, "--B 262144"),
    ):
        try:
            net._shape(_args(manifest="m.json", **args), spec(**over))
            raise AssertionError(f"accepted {over} {args}")
        except SystemExit as e:
            assert why in str(e), (over, args, str(e))
    assert net._shape(_args(manifest="m.json", min_r=0), spec(r=0)) == (1 << 20, 8, 0, 1024, 1)  # relaxed knowingly
    assert net._shape(_args(), None) == (1 << 20, 8, 2048, 1024, 1)  # synthetic: the verifier's own
    assert net._shape(_args(B="64K", r=0, k=512), None) == (1 << 16, 8, 0, 512, 1)
    assert net.SHAPE_DEFAULTS == {"B": "1M", "r": 2048, "k": 1024}


def test_baseline_must_match_the_run():
    """A baseline judges a run only under the conditions it was measured in: shape, --repeat, --drain, the
    timing mode and the dynamic chains (block size, length, live chain). Before: only the static shape and
    --repeat were compared, so an undrained baseline (round trips holding an inference step) judged a
    drained run, and the batch verifier took any dynamic section. A drained run also needs the file's READY
    bounds: without arm_max_ns the READY window went unbounded, silently."""
    good = {
        "B": 1 << 20,
        "n": 8,
        "r": 2048,
        "k": 1024,
        "g": 1,
        "repeat": 1,
        "drain": True,
        "timing": "kernel",
        "arm_max_ns": 2e6,
        "dyn": {"B": 1 << 16, "n": 8, "live": {"deadline_ns": 1.0}, "arm_max_ns": 4e6, "schedule": "independent"},
    }
    args = _args(B="1M", timing="kernel", dynamic=True, dyn_live=True)
    assert net.baseline_problems(good, args, None) == []
    # a baseline role's dynamic rounds used to follow the static drain at once (a freeze at a step's start): such a
    # file (no schedule, no calibration samples) is refused; a live-saved one always scheduled them at random times
    old_role = {**good, "dyn": {k: v for k, v in good["dyn"].items() if k != "schedule"}}
    assert any("right after the static drain" in p for p in net.baseline_problems(old_role, args, None))
    assert net.baseline_problems({**old_role, "calibration_samples": {"dyn_rtt_ns": [1]}}, args, None) == []
    for over, why in (
        ({"drain": False}, "drain False (this run: True)"),
        ({"timing": "user"}, "timing user (this run: kernel)"),
        ({"timing": None}, "timing not recorded"),
        ({"repeat": 4}, "repeat 4"),
        ({"r": 256}, "r 256"),
        ({"dyn": {"B": 1 << 16, "n": 16, "live": {}, "arm_max_ns": 4e6}}, "dynamic chain of 16 links"),
        ({"dyn": {"B": 1 << 12, "n": 8, "live": {"deadline_ns": 1.0}, "arm_max_ns": 4e6}}, "over 4096-byte blocks"),
        ({"dyn": {"B": 1 << 16, "n": 8, "arm_max_ns": 4e6}}, "no live-chain deadline"),
        ({"dyn": None}, "no dynamic section"),
        ({"arm_max_ns": None}, "no READY bound"),
        ({"arm_max_ns": 0.0}, "no READY bound"),
        ({"dyn": {"B": 1 << 16, "n": 8, "live": {"deadline_ns": 1.0}}}, "no dynamic READY bound"),
    ):
        probs = net.baseline_problems({**good, **over}, args, None)
        assert any(why in p for p in probs), (over, probs)
    no_bounds = {**good, "arm_max_ns": None, "dyn": {**good["dyn"], "arm_max_ns": None}}
    # given on the command line
    assert (
        net.baseline_problems(
            no_bounds,
            _args(B="1M", timing="kernel", dynamic=True, dyn_live=True, arm_max_ms=5.0, dyn_arm_max_ms=9.0),
            None,
        )
        == []
    )
    assert any(
        "no dynamic READY bound" in p
        for p in net.baseline_problems(
            no_bounds, _args(B="1M", timing="kernel", dynamic=True, dyn_live=True, arm_max_ms=5.0), None
        )
    ), "the static --arm-max-ms stood in for the dynamic READY bound"
    # no drain: no READY
    assert net.baseline_problems({**no_bounds, "drain": False}, _args(B="1M", timing="kernel", drain=False), None) == []
    assert any(
        "a live-chain deadline" in p
        for p in net.baseline_problems(good, _args(B="1M", timing="kernel", dynamic=True), None)
    )
    # live-saved files
    assert (
        net.baseline_problems({**good, "timing": None, "calibration_samples": {"timing": "kernel"}}, args, None) == []
    )
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "b.json"
        path.write_text(json.dumps({**good, "drain": False}))
        try:
            net._load_baseline(_args(B="1M", timing="kernel", dynamic=True, dyn_live=True, baseline=str(path)), None)
            raise AssertionError("accepted")
        except SystemExit as e:
            assert "drain False" in str(e)


def test_gpu_path_builds_the_seeded_array():
    """The synthetic array on a GPU is built from --seed, as on the CPU: before, the GPU paths used Hbm's
    default data seed (1), so a GPU prover and a CPU verifier at another seed disagreed on every digest."""
    from types import SimpleNamespace

    import memory_accounting.core.backend
    import memory_accounting.core.memory
    from memory_accounting.core.params import Challenge

    seen = []

    class FakeHbm:
        def __init__(self, be, cfg, data_seed=1, **kw):
            seen.append(data_seed)
            self.cfg = cfg
            self.theta_host_np = memory_accounting.core.memory.cpu_synthetic_theta(cfg, data_seed)

    real = (net.pick_device, memory_accounting.core.backend.DeviceBackend, memory_accounting.core.memory.Hbm)
    net.pick_device = lambda d=None: "fakegpu"
    memory_accounting.core.backend.DeviceBackend = lambda dev: SimpleNamespace(device=dev)
    memory_accounting.core.memory.Hbm = FakeHbm
    try:
        args = SimpleNamespace(
            manifest=None,
            size="1M",
            encoding="vde",
            no_perm=False,
            seed=7,
            device=None,
            verify_threads=2,
            filler_seed=None,
            pad_seed=None,
            gpu=0,
        )
        from memory_accounting.core.params import array_from_args

        spec, keys = array_from_args(args)
        gpu_ver, _ = net._make_verifier(args, spec, keys)
    finally:
        net.pick_device, memory_accounting.core.backend.DeviceBackend, memory_accounting.core.memory.Hbm = real
    cpu_ver, _ = net._make_verifier(SimpleNamespace(**{**vars(args), "device": "cpu"}), spec, keys)
    assert seen == [7], seen
    ch = Challenge(1 << 16, 4, 0, 99, 3, 1024)
    assert gpu_ver.expected(ch)[:2] == cpu_ver.expected(ch)[:2]


def _fake_dyn_prover(live_words: int):
    """A prover that answers a dynamic ARM with READY_DYN (this live region size) and any chain with a digest."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))

    def serve():
        while True:
            data, addr = srv.recvfrom(512)
            if len(data) == wire.ARM.size:
                tok = wire.ARM.unpack(data)[1]
                srv.sendto(wire.READY_DYN.pack(wire.ARM_DYN_MAGIC, tok, 1 << 16, 1 << 16, 0, live_words), addr)
            elif len(data) == wire.CH.size:
                srv.sendto(wire.RE.pack(wire.CH.unpack(data)[1], 1, 2), addr)

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def test_missing_live_region_is_a_fault():
    """Asked for a live chain, a prover that reports no live region fails the verification: every honest
    live region holds at least the activation rows (padded to a block). Before, a zero was accepted and the
    live chain silently skipped."""
    from types import SimpleNamespace

    import numpy as np

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    shape = (1 << 16, 4, 256, 1024, 1)
    for live_words, asked, fault in ((0, True, "no live region"), (1 << 13, True, None), (0, False, None)):
        args = SimpleNamespace(host="127.0.0.1", port=_fake_dyn_prover(live_words), dyn_live=asked, reply_timeout=2.0)
        sample, ch, _dg, live = net._dynamic_round(sock, args, np.random.default_rng(0), shape, 0)
        assert ch is not None and (fault in sample.get("fault", "") if fault else "fault" not in sample), (
            live_words,
            asked,
            sample,
        )
        assert (live is not None) == (asked and live_words > 0), sample


def test_prover_drops_datagrams_from_others():
    """--verifier-addr: the prover answers only the verifier's address (resolved once); a datagram from any
    other host is dropped unread (no ARM, posture, counters or answer)."""
    assert wire.resolve_peers(None) is None and wire.resolve_peers([]) is None
    assert wire.resolve_peers("127.0.0.1, localhost") == {"127.0.0.1"} == wire.resolve_peers(["127.0.0.1", "localhost"])
    try:
        wire.resolve_peers("no-such-host.invalid")
        raise AssertionError("an unresolvable verifier address was accepted")
    except ValueError:
        pass
    from types import SimpleNamespace

    for allowed, answered in (("192.0.2.7", False), ("127.0.0.1", True)):
        port = _free_port()
        threading.Thread(
            target=udp_prover.main,
            args=(
                [
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--size",
                    "1M",
                    "--device",
                    "cpu",
                    "--verifier-addr",
                    allowed,
                ],
            ),
            daemon=True,
        ).start()
        time.sleep(1.0)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            got = net._prover_stats(s, SimpleNamespace(host="127.0.0.1", port=port), 1.0)
        finally:
            s.close()
        assert (got is not None) == answered, (allowed, got)


def _dyn_prover() -> int:
    """A CPU prover with the toy model's dynamic region, shared by the dynamic live runs."""
    if "dyn_port" not in _PROVER:
        port = _free_port()
        threading.Thread(
            target=udp_prover.main,
            args=(
                [
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--size",
                    "4M",
                    "--device",
                    "cpu",
                    "--dynamic",
                    "--dyn-block",
                    "64K",
                    "--toy-step-ms",
                    "1",
                ],
            ),
            daemon=True,
        ).start()
        time.sleep(2.0)
        _PROVER["dyn_port"] = port
    return _PROVER["dyn_port"]


def _tcp_proxy(listen: int, target: int, delay):
    """The record side channel through the relay's port: each request forwarded, its reply held `delay` s (or, if
    `delay` is callable, until `delay(request)` returns)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", listen))
    srv.listen(64)

    def one(conn):
        try:
            req = conn.recv(64)
            u = socket.create_connection(("127.0.0.1", target))
            u.sendall(req)
            buf = b""
            while chunk := u.recv(1 << 20):
                buf += chunk
            delay(req) if callable(delay) else time.sleep(delay)
            conn.sendall(buf)
        except OSError:
            pass
        finally:
            conn.close()

    def run():
        while True:
            c, _ = srv.accept()
            threading.Thread(target=one, args=(c,), daemon=True).start()

    threading.Thread(target=run, daemon=True).start()


def _live(prover: int, args: list, fault=None, tcp_delay: float | None = None) -> dict:
    """One live run through a relay (and, with tcp_delay, a record proxy on the same port): the log's events,
    what it printed, the relay's datagrams and their arrival times."""
    relay, log, times = _free_port(), [], []
    threading.Thread(target=_relay, args=(relay, prover, log, fault, times), daemon=True).start()
    if tcp_delay is not None:
        _tcp_proxy(relay, prover, tcp_delay)
    ui = socket.socket()
    ui.bind(("127.0.0.1", 0))
    ui_port = ui.getsockname()[1]
    ui.close()
    import contextlib
    import io

    printed = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(printed):
        out = pathlib.Path(d) / "live.jsonl"
        try:
            net.main(
                [
                    "live",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(relay),
                    "--size",
                    "4M",
                    "--B",
                    "64K",
                    "--n",
                    "4",
                    "--r",
                    "0",
                    "--device",
                    "cpu",
                    "--drain",
                    "--strategies",
                    "honest",
                    "--reply-timeout",
                    "1.0",
                    "--ui-port",
                    str(ui_port),
                    "--out",
                    str(out),
                ]
                + args
            )
        finally:
            evs = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    print(printed.getvalue()[-3000:], end="")
    return {"events": evs, "printed": printed.getvalue(), "log": log, "times": times}


def _after_calibration(nth: int, way: str, magic: int, action):
    """A fault: after the first `nth` replies to challenges of this magic, `action(data)` -> delays (or a new datagram)."""
    state = {"n": 0, "last": None}

    def fault(w, data):
        if w == "down" and len(data) == wire.CH.size:
            state["last"] = wire.CH.unpack(data)[0]
            return [0]
        if w == way and len(data) == wire.RE.size and state["last"] == magic:
            state["n"] += 1
            if state["n"] > nth:
                return action(data)
        return [0]

    return fault


_DYN = {}


def _dyn_live_run() -> dict:
    """A live run with --dynamic and the live chain under a timing claim; after calibration every reply to the chain
    over all dynamic memory held back 0.3 s: late on every dynamic chain, far below the ceiling (1000 x here). The period model: its
    claim is on whatever the calibration (the window model cannot size one from 24 chains)."""
    if not _DYN:
        W, C = 1, 24
        _DYN.update(
            _live(
                _dyn_prover(),
                [
                    "--dynamic",
                    "--dyn-B",
                    "64K",
                    "--dyn-n",
                    "4",
                    "--warmup",
                    str(W),
                    "--calibrate",
                    str(C),
                    "--interval",
                    "0.2",
                    "--claim-F",
                    "0.5",
                    "--claim-mode",
                    "period",
                    "--claim-ceiling",
                    "1000",
                    "--claim-miss-window",
                    "10",
                    "--claim-max-fill-s",
                    "1e9",
                    "--max-verifications",
                    "24",
                    "--with-dyn-live",
                ],
                _after_calibration(W + C, "up", wire.DYN_MAGIC, lambda d: [0.3]),
            )
        )
    return _DYN


def test_dynamic_chain_late_under_a_claim_alerts():
    """With a timing claim on, a prover late on every dynamic chain (here by 0.3 s, under the ceiling) alerts
    within a few verifications: the dynamic chain has its own claim (a miss count, its mean shift being
    untestable at 64 KiB). Before, under a claim its misses were dropped and only 3 x its deadline alerted."""
    r = _dyn_live_run()
    evs = r["events"]
    vers = [
        e
        for e in evs
        if e.get("type") == "verification" and e.get("phase") != "calibration" and e.get("round") == "dynamic"
    ]
    assert len(vers) >= 3 and all(not e["dyn"]["within"] for e in vers), [e["dyn"].get("rtt_ns") for e in vers]
    alerts = [e for e in evs if e.get("type") == "alert"]
    claim = [e for e in alerts if e.get("kind") == "claim" and e.get("chain") == "dynamic"]
    assert claim and claim[0]["rule"] == "misses" and "dynamic chain" in claim[0]["detail"], alerts
    assert claim[0]["id"] <= vers[2]["id"], (claim[0]["id"], [e["id"] for e in vers])  # the third late chain (3 of 10)
    # (the relay passes no TCP: records fail)
    single = [e for e in alerts if e.get("kind") == "dynamic" and "dynamic round trip" in e["detail"]]
    assert not single, "a single dynamic miss alerted under the claim"
    assert not [e for e in alerts if e.get("kind") == "dynamic" and "live chain round trip" in e["detail"]], (
        "a single live miss alerted"
    )
    assert "calibrating on the prover under test" in r["printed"]


def test_dynamic_rounds_have_their_own_random_times():
    """Static and dynamic rounds come at independent random times: a static ARM no longer announces a freeze.
    Before, every verification was static ARM, static chains, then at once the dynamic ARM, so a prover could
    start reloading dynamic memory at the static ARM. Each round is its own event ("round"); dynamic ones keep "dyn"."""
    r = _dyn_live_run()
    vers = [e for e in r["events"] if e.get("type") == "verification"]
    kinds = [e["round"] for e in vers]
    assert set(kinds) == {"static", "dynamic"}, kinds
    assert all(("dyn" in e) == (e["round"] == "dynamic") for e in vers) and all(
        not e["challenges"] for e in vers if e["round"] == "dynamic"
    )
    # two of a kind in a row: independent draws
    assert any(a == b for a, b in itertools.pairwise(kinds)), "rounds strictly alternate"
    log, times = r["log"], r["times"]

    def arm(d):
        return len(d) == wire.ARM.size and wire.ARM.unpack(d)[0]

    follow = []  # a dynamic ARM within 10 ms of the static reply before it
    for i, (w, d) in enumerate(log):
        if w == "down" and arm(d) == wire.ARM_DYN_MAGIC:
            j = max((k for k in range(i) if log[k][0] == "up"), default=None)
            follow.append(j is not None and times[i] - times[j] < 0.01)
    assert len(follow) >= 30 and sum(follow) < 0.4 * len(follow), (sum(follow), len(follow))


_SLOW = {}


def _slow_records_run() -> dict:
    """Static replies with a flipped digest after calibration (an evict-like cheat) while every record is held
    2 s (inside the fetch deadline): the repro of a prover that stalls the checks with slow records."""
    if _SLOW:
        return _SLOW
    W, C = 1, 8
    relay = _free_port()
    prover = _dyn_prover()
    state = {"n": 0, "last": None}

    def udp():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", relay))
        client = None
        while True:
            data, addr = sock.recvfrom(2048)
            if addr[1] == prover:
                if len(data) == wire.RE.size and state["last"] == wire.MAGIC:
                    state["n"] += 1
                    if state["n"] > W + C + 2:
                        nonce, lo, hi = wire.RE.unpack(data)
                        data = wire.RE.pack(nonce, lo ^ 1, hi)
                sock.sendto(data, client)
            else:
                client = addr
                if len(data) == wire.CH.size:
                    state["last"] = wire.CH.unpack(data)[0]
                sock.sendto(data, ("127.0.0.1", prover))

    threading.Thread(target=udp, daemon=True).start()
    _tcp_proxy(relay, prover, 2.0)
    ui = socket.socket()
    ui.bind(("127.0.0.1", 0))
    ui_port = ui.getsockname()[1]
    ui.close()
    import contextlib
    import io

    printed = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(printed):
        out = pathlib.Path(d) / "live.jsonl"
        net.main(
            [
                "live",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--device",
                "cpu",
                "--drain",
                "--dynamic",
                "--dyn-B",
                "64K",
                "--dyn-n",
                "4",
                "--strategies",
                "honest",
                "--warmup",
                str(W),
                "--calibrate",
                str(C),
                "--interval",
                "0.05",
                "--reply-timeout",
                "1.0",
                "--mode",
                "monitor",
                "--check-lag-s",
                "1",
                "--max-verifications",
                "30",
                "--ui-port",
                str(ui_port),
                "--out",
                str(out),
            ]
        )
        evs = [json.loads(line) for line in out.read_text().splitlines()]
    _SLOW.update(events=evs, printed=printed.getvalue())
    return _SLOW


def test_wrong_static_digests_alert_while_records_wait():
    """Records served slowly no longer hold back the static digest checks: those run on their own thread, so
    the wrong static digests alert while the records wait. Before: one queue and one thread for both; in the
    reviewer's run 25 wrong digests, none checked by the end, no alert."""
    evs = _slow_records_run()["events"]
    vers = {e["id"]: e for e in evs if e.get("type") == "verification"}
    static_ids = {i for i, e in vers.items() if e["round"] == "static" and not e.get("faults")}
    checked = {e["id"] for e in evs if e.get("type") == "check" and e.get("challenges")}
    assert static_ids and checked >= static_ids, sorted(static_ids - checked)
    alerts = [e for e in evs if e.get("type") == "alert" and e.get("kind") == "digest"]
    bad = [e["id"] for e in evs if e.get("type") == "check" and e.get("digest_ok") is False and e.get("challenges")]
    assert len(alerts) >= 5 and len(bad) >= 5, (len(alerts), len(bad))


def test_record_checks_falling_behind_alert_and_are_reported_at_stop():
    """A check waiting longer than --check-lag-s alerts; the record checks still waiting at stop are reported
    unchecked (an alert, an `unchecked` event in the log, a line printed): before, the backlog was dropped."""
    r = _slow_records_run()
    evs = r["events"]
    lag = [e for e in evs if e.get("type") == "alert" and e.get("kind") == "check" and "falling behind" in e["detail"]]
    assert lag, "no alert while the record checks fell behind"
    left = [e for e in evs if e.get("type") == "unchecked"]
    assert left and left[0]["dynamic_ids"] and "UNCHECKED" in r["printed"], left
    assert any(e.get("type") == "alert" and e.get("kind") == "check" and "unchecked" in e["detail"] for e in evs)
    # both alerts carry the current round's id (the page places an alert by it), the rounds concerned apart; the lag
    # alert says the prover's slow records are the cause (here each is held 2 s), not the verifier's checking.
    # Before: the id of the oldest waiting round, and a text that blamed the verifier
    ids = [e["id"] for e in evs if e.get("type") == "verification"]
    for a in lag:
        assert (
            a["oldest_id"] < a["id"] and a["id"] in ids and a["cause"] == "slow_fetch" and "slow to come" in a["detail"]
        ), a
    stop = [e for e in evs if e.get("type") == "alert" and e.get("kind") == "check" and "unchecked" in e["detail"]][-1]
    unchecked = sorted(left[0]["static_ids"] + left[0]["dynamic_ids"])
    assert stop["id"] == max(ids) and stop["oldest_id"] == unchecked[0] and stop["ids"] == unchecked, stop


_STATIC_LATE = {}


def test_static_misses_alert_under_a_period_claim():
    """Under a claim, static misses have a backstop: a miss count on the static chains alerts within a few
    round trips (rule "misses", chain "static"), and the open period is judged at stop (partial). Before: a
    prover late on every static chain raised nothing until the hour's period ended, and nothing if it stopped
    first. meta "claims" lists the tests that run on each chain."""
    W, C = 1, 24
    if not _STATIC_LATE:
        _STATIC_LATE.update(
            _live(
                _prover(),
                [
                    "--warmup",
                    str(W),
                    "--calibrate",
                    str(C),
                    "--interval",
                    "0.05",
                    "--claim-F",
                    "0.5",
                    "--claim-mode",
                    "period",
                    "--claim-ceiling",
                    "1000",
                    "--claim-miss-window",
                    "10",
                    "--claim-max-fill-s",
                    "1e9",
                    "--max-verifications",
                    "10",
                    "--mode",
                    "monitor",
                    "--challenge-flags",
                    "alert",
                ],
                _after_calibration(W + C, "up", wire.MAGIC, lambda d: [0.2]),
            )
        )
    evs = _STATIC_LATE["events"]
    alerts = [e for e in evs if e.get("type") == "alert"]
    claim = [e for e in alerts if e.get("kind") == "claim" and e.get("chain") == "static" and e.get("rule") == "misses"]
    assert claim, [(a["kind"], a["detail"][:60]) for a in alerts]
    assert not [a for a in alerts if a.get("kind") == "deadline"], "a single static miss alerted under the claim"
    part = [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("partial")]
    assert part and part[0]["chain"] == "static" and part[0]["n"] > 0, part
    meta = [e for e in evs if e.get("type") == "meta" and "claims" in e]
    assert meta and meta[-1]["claims"]["static"]["tests"] == ["period", "misses"], meta
    assert claim[0]["outside_claim"] and meta[-1]["claims"]["static"]["challenge_flags"] == "alert", claim[0]


def test_single_challenge_flags_do_not_alert_by_default():
    """--challenge-flags warn (the default, the design log v0.2 Issue #1): on a chain whose mean-shift claim runs, no
    single challenge alerts. A prover late on every static chain raises miss-count flags (logged, outside the
    claim), no alert, and the claim weighs the chains; a claim too slow to fill does not fall back to an alert a miss."""
    W, C = 1, 24
    r = _live(
        _prover(),
        [
            "--warmup",
            str(W),
            "--calibrate",
            str(C),
            "--interval",
            "0.05",
            "--claim-F",
            "0.5",
            "--claim-mode",
            "period",
            "--claim-ceiling",
            "1000",
            "--claim-miss-window",
            "10",
            "--claim-max-fill-s",
            "60",
            "--max-verifications",
            "10",
            "--mode",
            "monitor",
        ],
        _after_calibration(W + C, "up", wire.MAGIC, lambda d: [0.2]),
    )
    evs = r["events"]
    alerts = [e for e in evs if e.get("type") == "alert" and e.get("kind") in ("deadline", "gross", "missing")]
    assert not alerts, [(a["kind"], a["detail"][:60]) for a in alerts]
    # the claim may fail the open period when the run stops (how many clipped chains it holds by then varies with
    # timing): that is the claim weighing the chains, not a single challenge alerting
    claims = [e for e in evs if e.get("type") == "alert" and e.get("kind") == "claim"]
    assert all("(judged at stop)" in a["detail"] for a in claims), [a["detail"][:80] for a in claims]
    flags = [e for e in evs if e.get("type") == "flag" and e.get("rule") == "misses" and e.get("chain") == "static"]
    assert flags and flags[0]["outside_claim"], [e for e in evs if e.get("type") == "flag"]
    info = [e for e in evs if e.get("type") == "meta" and "claims" in e][-1]["claims"]["static"]
    assert (
        info["challenge_flags"] == "warn"
        and not info.get("per_miss")
        and "nothing on this chain alerts" in info["reason"]
    ), info
    part = [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("partial")]
    assert part and part[0]["clipped"] == part[0]["n"] > 0, part  # 0.2 s late: every chain at the clip


def test_missing_ready_counts_at_the_clip_under_a_claim():
    """A READY that never comes (here dropped by the relay after calibration) on a chain whose claim runs is a flag
    under --challenge-flags warn, not an alert, and its round counts at the clip in the claim: withholding READY
    would be how a prover dodges a challenge it cannot answer in time, so it costs as much as a cheated chain. An
    honest lost packet costs one clipped chain. Before: an alert at once, outside the claim's alpha."""
    W, C = 1, 24
    r = _live(
        _prover(),
        [
            "--warmup",
            str(W),
            "--calibrate",
            str(C),
            "--interval",
            "0.05",
            "--claim-F",
            "0.5",
            "--claim-mode",
            "period",
            "--claim-max-fill-s",
            "1e9",
            "--reply-timeout",
            "0.5",
            "--max-verifications",
            "8",
            "--mode",
            "monitor",
        ],
        _nth("up", wire.READY.size, {W + C + 3: []}),
    )
    evs = r["events"]
    vers = [e for e in evs if e.get("type") == "verification" and e.get("phase") != "calibration"]
    lost = [e for e in vers if e.get("faults")]
    assert (
        len(lost) == 1 and lost[0]["faults"] == [net.FAULT_NO_READY] and lost[0]["challenges"][0]["missing"] == "READY"
    ), lost
    assert lost[0].get("faults_flagged") and not lost[0]["gross"], lost[0]
    assert not [e for e in evs if e.get("type") == "alert" and e.get("kind") == "missing"]
    flags = [e for e in evs if e.get("type") == "flag" and e.get("kind") == "missing"]
    assert [f["id"] for f in flags] == [lost[0]["id"]] and flags[0]["chain"] == "static", flags
    part = [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("partial")]
    # the lost round is in the sum, at the clip
    assert part and part[0]["n"] == len(vers) and part[0]["clipped"] >= 1, part


def test_claim_too_slow_to_fill_alerts_every_miss():
    """A claim whose period (or window) takes longer than --claim-max-fill-s to fill leaves its chain's misses
    to alert one by one, and says so in meta "claims" (per_miss, reason)."""
    W, C = 1, 24
    r = _live(
        _prover(),
        [
            "--warmup",
            str(W),
            "--calibrate",
            str(C),
            "--interval",
            "0.05",
            "--claim-F",
            "0.5",
            "--claim-mode",
            "period",
            "--claim-ceiling",
            "1000",
            "--claim-max-fill-s",
            "60",
            "--max-verifications",
            "4",
            "--mode",
            "monitor",
            "--challenge-flags",
            "alert",
        ],
        _after_calibration(W + C, "up", wire.MAGIC, lambda d: [0.2]),
    )
    evs = r["events"]
    assert [a for a in evs if a.get("type") == "alert" and a.get("kind") == "deadline"], (
        "the misses did not alert one by one"
    )
    info = [e for e in evs if e.get("type") == "meta" and "claims" in e][-1]["claims"]["static"]
    assert info["per_miss"] and "--claim-max-fill-s" in info["reason"] and "misses" in info["tests"], info
    # --max-verifications counts the rounds after the calibration (before: the calibration's too, so 4 meant none)
    run = [e for e in evs if e.get("type") == "verification" and e.get("phase") != "calibration"]
    assert len(run) == 4, len(run)


def test_a_slow_chain_cannot_hide_behind_a_fast_one():
    """--repeat 2, every second static chain 0.5 s late: the round's minimum is on time, but every chain is held
    to the single-chain deadline (within fails) and the ceiling to the slowest chain (a gross alert under the
    claim). Before: 14 of 15 within the deadline, no alert."""
    W, C = 1, 24
    flip = {"n": 0}

    def every_second(d):
        flip["n"] += 1
        return [0.5] if flip["n"] % 2 == 0 else [0]

    r = _live(
        _prover(),
        [
            "--repeat",
            "2",
            "--warmup",
            str(W),
            "--calibrate",
            str(C),
            "--interval",
            "0.05",
            "--claim-F",
            "0.5",
            "--claim-mode",
            "period",
            "--claim-max-fill-s",
            "1e9",
            "--max-verifications",
            "6",
            "--mode",
            "monitor",
            "--challenge-flags",
            "alert",
        ],
        _after_calibration(2 * (W + C), "up", wire.MAGIC, every_second),
    )
    evs = r["events"]
    vers = [e for e in evs if e.get("type") == "verification" and e.get("phase") != "calibration"]
    slow = [e for e in vers if any(c["rtt_ns"] and c["rtt_ns"] > 4e8 for c in e["challenges"])]
    assert slow and all(not e["within"] and e["chain_deadline_ns"] for e in slow), [
        (e["within"], e.get("chain_deadline_ns")) for e in slow
    ]
    gross = [a for a in evs if a.get("type") == "alert" and a.get("kind") == "gross" and a.get("chain") == "static"]
    assert len(gross) >= len(slow), (len(gross), len(slow))


def test_verifier_checks_every_verification_by_default():
    """The verifier role checks every verification's digests unless told otherwise (--sample-verify N, which
    reports the rest UNCHECKED). Before: only the first, so 9 wrong digests out of 11 passed as correct."""

    def corrupt_after(k):
        seen = [0]

        def f(w, data):
            if w == "up" and len(data) == wire.RE.size:
                seen[0] += 1
                if seen[0] > k:
                    nonce, lo, hi = wire.RE.unpack(data)
                    return ("rewrite", wire.RE.pack(nonce, lo ^ 1, hi))
            return [0]

        return f

    import contextlib
    import io

    for extra, expect_bad, expect_unchecked in (([], True, 0), (["--sample-verify", "1"], False, 9)):
        relay = _free_port()
        threading.Thread(target=_rewriting_relay, args=(relay, _prover(), corrupt_after(2)), daemon=True).start()
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(printed):
            out = pathlib.Path(d) / "udp.json"
            net.main(
                [
                    "verifier",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(relay),
                    "--size",
                    "4M",
                    "--B",
                    "64K",
                    "--n",
                    "4",
                    "--r",
                    "0",
                    "--samples",
                    "10",
                    "--warmup",
                    "1",
                    "--drain",
                    "--strategies",
                    "honest",
                    "--out",
                    str(out),
                ]
                + extra
            )
            rec = json.loads(out.read_text())["records"][0]
        assert (rec["incorrect_samples"] > 0) == expect_bad and rec["unchecked_samples"] == expect_unchecked, (
            extra,
            rec["incorrect_samples"],
            rec["unchecked_samples"],
        )
        assert ("UNCHECKED" in printed.getvalue()) == bool(expect_unchecked)


def _rewriting_relay(listen: int, target: int, fault):
    """A relay whose fault may return ("rewrite", new bytes) to forward another datagram in place of this one."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", listen))
    client = None
    while True:
        data, addr = sock.recvfrom(2048)
        way = "up" if addr[1] == target else "down"
        if way == "down":
            client = addr
        got = fault(way, data)
        if isinstance(got, tuple):
            data = got[1]
        sock.sendto(data, client if way == "up" else ("127.0.0.1", target))


def test_calibration_without_replies_ends_with_an_alert():
    """Every calibration round faulting (no prover) ends the run with a clear message and an alert. Before:
    an IndexError out of the deadline computation over no round trips."""
    import contextlib
    import io

    ui = socket.socket()
    ui.bind(("127.0.0.1", 0))
    ui_port = ui.getsockname()[1]
    ui.close()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        out = pathlib.Path(d) / "live.jsonl"
        try:
            net.main(
                [
                    "live",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(_free_port()),
                    "--size",
                    "4M",
                    "--B",
                    "64K",
                    "--n",
                    "4",
                    "--r",
                    "0",
                    "--device",
                    "cpu",
                    "--drain",
                    "--strategies",
                    "honest",
                    "--warmup",
                    "0",
                    "--calibrate",
                    "2",
                    "--interval",
                    "0.01",
                    "--reply-timeout",
                    "0.2",
                    "--ui-port",
                    str(ui_port),
                    "--out",
                    str(out),
                ]
            )
            raise AssertionError("a calibration without replies went on")
        except SystemExit as e:
            assert "calibration failed" in str(e), e
        evs = [json.loads(line) for line in out.read_text().splitlines()]
    assert any(e.get("type") == "alert" and "calibration failed" in e.get("detail", "") for e in evs)
    try:
        net._deadline_from([None, float("inf")], _args(deadline_rule="sigma", sigmas=8, margin=0.05))
        raise AssertionError("no error")
    except ValueError:
        pass


class _FailingSock:
    """A socket whose sends fail (EHOSTUNREACH from the stack, EPERM from a firewall)."""

    def sendto(self, *a):
        raise OSError(65, "No route to host")


def test_failed_sends_are_missing_replies():
    """A send that fails is a missing READY or reply (a failed verification and an alert), never the end of
    the verifier. Before: the OSError ended it."""
    from types import SimpleNamespace

    from memory_accounting.core.params import Challenge

    args = SimpleNamespace(host="127.0.0.1", port=9, reply_timeout=0.2, dyn_live=True)
    s = _FailingSock()
    assert net._arm(s, args, 1 << 16, 4, 0, 1024, 1, 0) is None
    assert net._timed(s, args, Challenge(1 << 16, 4, 0, 5, 0), wire.MAGIC, 0) is None
    sample, ch, _dg, _lv = net._dynamic_round(s, args, None, (1 << 16, 4, 0, 1024, 1), 0, jitter=False)
    assert ch is None and "could not be sent" in sample["fault"], sample
    assert net._prover_stats(s, args, 0.1) is None


def test_kernel_round_trip_out_of_range_is_missing():
    """Under --timing kernel a round trip at or below zero, or beyond the reply timeout (a clock step between
    the stamps), counts as a missing reply, never as a time."""
    from types import SimpleNamespace

    from memory_accounting.core.params import Challenge

    ch = Challenge(1 << 16, 4, 0, 5, 0)
    real = net._await
    try:
        for bad, missing in ((-5, True), (None, False)):

            def fake(sock, args, want, timeout, kind=None, bad=bad):
                args._bad_ts = bad
                args._last_krtt = 1000 if bad is None else None
                return wire.RE.pack(ch.nonce, 1, 2)

            net._await = fake
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            got = net._timed(
                s, SimpleNamespace(host="127.0.0.1", port=9, timing="kernel", reply_timeout=1.0), ch, wire.MAGIC, 0
            )
            s.close()
            assert (got is None) == missing, (bad, got)
    finally:
        net._await = real
    net._krtt_add("test-kind", 1)
    assert net.KERNEL_RTT["test-kind"].maxlen == net.KERNEL_RTT_KEEP  # bounded in a long live run
    del net.KERNEL_RTT["test-kind"]


def _fake_records(records: dict):
    """Serve records by nonce in place of the prover's TCP side channel."""
    import memory_accounting.core.evidence as E

    real = E.fetch_evidence

    def fetch(host, port, nonce, **kw):
        if nonce not in records:
            raise LookupError(f"no evidence for nonce {nonce:016x}")
        return records[nonce]

    E.fetch_evidence = fetch
    return lambda: setattr(E, "fetch_evidence", real)


def test_live_record_must_share_the_main_stage():
    """The two records of a freeze must describe one stage (JSON-equal) with regions main and live: before, each
    was checked on its own, so a live region built from any stage (an idle one) passed and the live chain
    checked nothing."""
    from types import SimpleNamespace

    from memory_accounting.core.params import Challenge

    main, live = Challenge(1 << 16, 32, 256, nonce=5, j0=0), Challenge(1 << 16, 32, 256, nonce=6, j0=0)
    st = {
        "kind": "toy",
        "step": 3,
        "request_id": 0,
        "chunks": [[[1, 2, 3]]],
        "act_layer": 1,
        "in_flight": False,
        "n_layers": 4,
    }

    def rec(nonce, region, stage):
        return _record(nonce=nonce, region=region, words=4096 if region == "main" else 1024, stage=stage)

    args = SimpleNamespace(host="x", port=0, _dyn_expected=(0, 0))
    sample = {"words": 4096, "live": {"words": 1024}}
    for live_stage, ok in (
        (dict(st), None),
        ({**st, "step": 0, "chunks": []}, False),
        ({**st, "in_flight": True}, False),
    ):
        restore = _fake_records({5: rec(5, "main", dict(st)), 6: rec(6, "live", live_stage)})
        try:
            rep = net._check_dynamic(args, None, main, (1, 2), live=(live, (1, 2)), sample=sample)
        finally:
            restore()
        assert rep["digest_ok"] is ok, (live_stage, rep)
        assert ok is None or "stage is not the main record's" in rep["error"], rep


def test_records_are_published_short_and_saved_only_when_bound():
    """A record that replays exactly can carry junk where the replay does not look (stage fields, debug). The
    verifier publishes only its own short, typed reading of it; it saves a record only if it describes the chain
    sent, under the challenge's nonce, with the prover's debug facts cut to a note when large; and --evidence-dir
    stops growing at --evidence-max-total. Before: 40 MB of junk went into the check event and the directory."""
    import contextlib
    import io
    from types import SimpleNamespace

    from memory_accounting.core.params import Challenge

    ch = Challenge(1 << 16, 32, 256, nonce=5, j0=0)
    junk = _record()
    junk.stage = {**junk.stage, "n_layers": "x" * 500_000, "kind": "k" * 1000}
    junk.debug = {"note": "y" * 500_000}
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(host="x", port=0, _dyn_expected=(0, 0), evidence_dir=tmp)
        restore = _fake_records({5: junk})
        try:
            rep = net._check_record(args, None, ch, (1, 2), words=4096)
            assert (
                rep["digest_ok"] is None
                and rep["n_layers"] is None
                and len(rep["kind"]) < 100
                and len(json.dumps(rep)) < 2000
            ), rep
            saved = json.loads(pathlib.Path(rep["path"]).read_text())
            assert pathlib.Path(rep["path"]).name == f"{ch.nonce:016x}.json" and "dropped" in saved["debug"], saved[
                "debug"
            ]
            for f in pathlib.Path(tmp).iterdir():
                f.unlink()
            args._evidence_used = None
            bad = net._check_record(args, None, ch, (1, 3), words=4096)  # not the reply that came: not saved
            assert bad["digest_ok"] is False and "path" not in bad and not list(pathlib.Path(tmp).iterdir()), bad
            args.evidence_max_total = "1"
            args._evidence_used = None
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                paths = [net._check_record(args, None, ch, (1, 2), words=4096).get("path") for _ in range(3)]
            assert paths[0] and paths[1:] == [None, None] and printed.getvalue().count("no longer saved") == 1, (
                paths,
                printed.getvalue(),
            )
        finally:
            restore()
    from memory_accounting.core import evidence

    assert evidence.MAX_HEADER <= 4 << 20


def test_record_budget_bounds_fetch_and_replay():
    """The fetch and replay of one record must finish within --record-budget-s, or its check fails: a prover
    cannot hold the record checks back with records slow to send or to replay."""
    from types import SimpleNamespace

    import memory_accounting.verifier.consistency as K
    from memory_accounting.core.evidence import HEAD, MAGIC
    from memory_accounting.core.params import Challenge

    port = _trickle_server([HEAD.pack(MAGIC, 100, 0), b"{"], 5.0)
    t = time.perf_counter()
    rep = net._check_record(
        SimpleNamespace(host="127.0.0.1", port=port, record_budget_s=0.5),
        None,
        Challenge(1 << 16, 32, 256, nonce=5, j0=0),
        (1, 2),
        words=4096,
    )
    assert rep["digest_ok"] is False and time.perf_counter() - t < 2.0, rep
    real = (K.verify, net._replayer)
    K.verify = lambda *a, **k: (time.sleep(0.3), {"ok": True, "expected": "00"})[1]
    net._replayer = lambda args: object()
    restore = _fake_records({5: _record()})
    try:
        args = SimpleNamespace(host="x", port=0, _dyn_expected=(0, 0), record_budget_s=0.2)
        rep = net._check_record(args, None, Challenge(1 << 16, 32, 256, nonce=5, j0=0), (1, 2), words=4096)
        assert rep["digest_ok"] is False and "record budget" in rep["error"], rep
        args.record_budget_s = 5.0
        assert (
            net._check_record(args, None, Challenge(1 << 16, 32, 256, nonce=5, j0=0), (1, 2), words=4096)["digest_ok"]
            is True
        )
    finally:
        K.verify, net._replayer = real
        restore()


def test_evidence_store_counts_layouts_and_drops_fetched_records():
    """The prover's store counts what a record holds (layout entries, tokens), counts an unassembled record as
    a large one, and drops a record once the verifier has fetched it (after a grace period). Before: every
    record counted 16 KiB and fetched records stayed until pushed out."""
    from memory_accounting.core.evidence import LAYOUT_ENTRY_BYTES, LAZY_RECORD_BYTES, EvidenceStore, LazyEvidence

    big = _record(nonce=1)
    big.layout = [
        {"name": f"r0.l{layer}.b{j}", "dtype": "uint8", "shape": [32768], "words": 4096}
        for layer in range(28)
        for j in range(80)
    ]
    store = EvidenceStore(fetched_ttl=0.0)
    store.put(big)
    assert store.bytes >= len(big.layout) * LAYOUT_ENTRY_BYTES
    store.put(LazyEvidence(2, lambda: _record(nonce=2)))
    assert store.bytes >= len(big.layout) * LAYOUT_ENTRY_BYTES + LAZY_RECORD_BYTES
    # built: its real size
    assert store.get(2).nonce == 2 and store.bytes < len(big.layout) * LAYOUT_ENTRY_BYTES + LAZY_RECORD_BYTES
    store.fetched(1)
    assert len(store) == 1 and store.get(1) is None and store.get(2) is not None
    keep = EvidenceStore(fetched_ttl=60.0)
    keep.put(_record(nonce=3))
    keep.fetched(3)
    assert keep.get(3) is not None  # a failed fetch can ask again meanwhile


def test_replays_use_the_verifier_threads():
    """Replays run on --verify-threads CPU threads, not on every core of the verifier box."""
    from types import SimpleNamespace

    import torch

    import memory_accounting.verifier.consistency as K

    real, before = K.replayer_from_spec, torch.get_num_threads()
    K.replayer_from_spec = lambda spec, device="cpu": object()
    try:
        net._REPLAYER.pop("fake-spec", None)
        net._replayer(SimpleNamespace(replay="fake-spec", verify_threads=2, replay_device="cpu"))
        assert torch.get_num_threads() == 2
    finally:
        K.replayer_from_spec = real
        torch.set_num_threads(before)
        net._REPLAYER.pop("fake-spec", None)


def test_sigterm_stops_the_live_verifier_cleanly():
    """SIGTERM stops the live verifier as Ctrl-C does: the stop runs (state stopped, the last checks, the
    unchecked report) and the process exits 0. The demo stops it by pidfile with INT, then TERM."""
    import signal
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    ui = socket.socket()
    ui.bind(("127.0.0.1", 0))
    ui_port = ui.getsockname()[1]
    ui.close()
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "live.jsonl"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "memory_accounting.verifier.net",
                "live",
                "--host",
                "127.0.0.1",
                "--port",
                str(_prover()),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--device",
                "cpu",
                "--drain",
                "--strategies",
                "honest",
                "--warmup",
                "1",
                "--calibrate",
                "5",
                "--interval",
                "0.05",
                "--ui-port",
                str(ui_port),
                "--out",
                str(log),
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            t = time.time()
            while (
                time.time() - t < 60
                and proc.poll() is None
                and (not log.exists() or len(log.read_text().splitlines()) < 12)
            ):
                time.sleep(0.2)
            proc.send_signal(signal.SIGTERM)
            out, _ = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
    assert proc.returncode == 0 and "live verifier stopping" in out, (proc.returncode, out[-1500:])


def _log_events(path) -> list:
    return (
        [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()]
        if pathlib.Path(path).exists()
        else []
    )


def test_records_past_the_cap_are_marked_and_the_audit_passes():
    """Past --evidence-max-total the verifier stops keeping records: one alert (kind evidence, the current round's
    id), meta evidence_saving false, and each check whose bound record was not kept says record_saved false.
    Without --replay such a freeze is checked by nothing: its check fails with an alert each (never "stored").
    With --replay the records are still replayed online: no round alerts. The offline audit with the verifier's
    log reports the records not kept apart and passes an honest run; a kept record that goes missing still fails
    it. Before: one printed warning, the rounds past the cap passed as stored, and the audit of the honest run
    failed (every record past the cap counted missing)."""
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    base = [
        "--dynamic",
        "--dyn-B",
        "64K",
        "--dyn-n",
        "4",
        "--warmup",
        "1",
        "--calibrate",
        "6",
        "--interval",
        "0.05",
        "--evidence-max-total",
        "12K",
        "--max-verifications",
        "24",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ev_dir, log = pathlib.Path(tmp) / "ev", pathlib.Path(tmp) / "live.jsonl"
        # the last --out wins
        _live(_dyn_prover(), base + ["--evidence-dir", str(ev_dir), "--out", str(log)], tcp_delay=0.0)
        evs = _log_events(log)
        alerts = [e for e in evs if e.get("type") == "alert"]
        saving = [a for a in alerts if a["kind"] == "evidence"]
        assert (
            len(saving) == 1
            and "no longer saved" in saving[0]["detail"]
            and saving[0]["first_unsaved_id"] <= saving[0]["id"]
        ), saving
        assert any(e.get("type") == "meta" and e.get("evidence_saving") is False for e in evs)
        checks = [e for e in evs if e.get("type") == "check" and "dyn_check" in e]
        unsaved = [e for e in checks if e.get("record_saved") is False]
        kept = [e for e in checks if "record_saved" not in e]
        assert unsaved and kept and all(e["dyn_digest_ok"] is None for e in kept), (len(unsaved), len(kept))
        assert all(e["dyn_digest_ok"] is False and e["dyn_check"]["failed"] == "unchecked" for e in unsaved)
        off = {a["id"] for a in alerts if a["kind"] == "dynamic" and "neither replayed nor kept" in a["detail"]}
        assert off == {e["id"] for e in unsaved}, (off, [e["id"] for e in unsaved])

        def audit():
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "memory_accounting.verifier.consistency",
                    "--evidence",
                    str(ev_dir),
                    "--log",
                    str(log),
                    "--dyn-B",
                    "64K",
                    "--dyn-n",
                    "4",
                    "--r",
                    "0",
                    "--k",
                    "1024",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )

        a = audit()
        assert a.returncode == 0 and "not kept by the verifier" in a.stdout, a.stdout[-1500:] + a.stderr[-1500:]
        next(ev_dir.glob("*.json")).unlink()  # a record the verifier kept, gone: still fails
        a = audit()
        assert a.returncode == 1 and "have a reply but no record" in a.stdout, a.stdout[-800:]
        # with --replay: past the cap the records are still replayed online, so no round alerts (the toy prover's cache:
        # 64 slots)
        log2 = pathlib.Path(tmp) / "live2.jsonl"
        _live(
            _dyn_prover(),
            base
            + [
                "--evidence-dir",
                str(pathlib.Path(tmp) / "ev2"),
                "--replay",
                "toy",
                "--replay-max-len",
                "64",
                "--out",
                str(log2),
            ],
            tcp_delay=0.0,
        )
        evs = _log_events(log2)
        alerts = [e for e in evs if e.get("type") == "alert"]
        saving = [a for a in alerts if a["kind"] == "evidence"]
        assert len(saving) == 1 and "still replayed online" in saving[0]["detail"], saving
        # (timing misses may alert: no claim here)
        bad = [a for a in alerts if a["kind"] in ("dynamic", "check") and "round trip" not in a["detail"]]
        assert not bad, [(a["kind"], a["detail"][:80]) for a in bad]
        checks = [e for e in evs if e.get("type") == "check" and "dyn_check" in e]
        assert any(e.get("record_saved") is False for e in checks) and all(e["dyn_digest_ok"] is True for e in checks)


def test_saved_records_are_compact_and_fetches_bounded():
    """Records are saved as compact JSON (a vLLM live record took 1.6 times the bytes indented); one record's
    fetch is bounded well below the record budget by default (10 s against 60 s)."""
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        text = _record().save(d).read_text()
    assert "\n" not in text and ": " not in text and json.loads(text)["nonce"] == 5
    assert (
        net._fetch_timeout(SimpleNamespace()) == 10.0
        and net._fetch_timeout(SimpleNamespace(evidence_timeout=90.0, record_budget_s=60.0)) == 60.0
    )


def test_replay_cache_length_is_the_verifiers():
    """The replay's KV cache length is the verifier's (--replay-max-len, default 512, the demo's), not the record's:
    a record naming another is refused before anything is allocated. <spec>@<slots> pins it the same way; 0 is
    the record's. Before: the record's, up to the model's positions, so a prover chose the verifier's allocation."""
    from types import SimpleNamespace

    import torch

    import memory_accounting.verifier.consistency as K

    real, threads, got = K.replayer_from_spec, torch.get_num_threads(), []
    K.replayer_from_spec = lambda spec, device="cpu": got.append(spec) or object()
    try:
        for replay, slots, want in (
            ("toy", 512, "toy@512"),
            ("toy@64", 512, "toy@64"),
            ("int:m.int.safetensors", 0, "int:m.int.safetensors"),
            ("hf:org/model", 2048, "hf:org/model@2048"),
        ):
            net._REPLAYER.clear()
            net._replayer(SimpleNamespace(replay=replay, replay_max_len=slots, verify_threads=1, replay_device="cpu"))
            assert got[-1] == want, (replay, slots, got[-1])
    finally:
        K.replayer_from_spec = real
        torch.set_num_threads(threads)
        net._REPLAYER.clear()


def test_the_live_chain_is_off_unless_asked_for():
    """The live chain runs only with --with-dyn-live (v0.19; before, it ran unless --no-dyn-live). Without it a
    dynamic round is one chain over all dynamic memory, and the memory claim credits none of a plain dynamic region:
    only the live chain's READY size says how much of it is live KV (checked, never credited), so without it nothing
    of the region can be told apart, and --max-scratch is refused. With it, a second chain over the live region, and
    the region outside it is credited."""
    common = [
        "--dynamic",
        "--dyn-B",
        "64K",
        "--dyn-n",
        "4",
        "--warmup",
        "1",
        "--calibrate",
        "4",
        "--interval",
        "0.05",
        "--hbm-bytes",
        "64M",
        "--max-verifications",
        "6",
    ]

    def dynamic_rounds(r):
        return [
            e
            for e in r["events"]
            if e.get("type") == "verification" and e.get("round") == "dynamic" and e["dyn"].get("words")
        ]

    dyn = dynamic_rounds(_live(_dyn_prover(), common))
    assert dyn and all("live" not in e["dyn"] for e in dyn), [e["dyn"] for e in dyn]
    assert all(e["memory"]["dynamic_bytes"] == 0 and e["memory"]["dynamic_region_bytes"] > 0 for e in dyn), [
        e["memory"] for e in dyn
    ]
    try:
        _live(_dyn_prover(), common + ["--max-scratch", "8M"])
        raise AssertionError("--max-scratch without the live chain accepted")
    except SystemExit as e:
        assert "--with-dyn-live" in str(e), e
    dyn = dynamic_rounds(_live(_dyn_prover(), common + ["--with-dyn-live"]))
    assert dyn and all(e["dyn"]["live"]["rtt_ns"] for e in dyn), [e["dyn"] for e in dyn]
    # the region outside the live one is credited (the toy's live region is at times the whole of it)
    outside = [e["memory"] for e in dyn if e["memory"]["live_region_bytes"] < e["memory"]["dynamic_region_bytes"]]
    assert outside and all(m["dynamic_bytes"] > 0 for m in outside), [e["memory"] for e in dyn]
    import contextlib
    import io

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            net.main(["live", "--host", "127.0.0.1", "--port", "9", "--no-dyn-live"])
        raise AssertionError("--no-dyn-live accepted")
    except SystemExit as e:
        assert e.code == 2 and "unrecognized arguments: --no-dyn-live" in err.getvalue(), err.getvalue()


def test_setup_manifest_needs_the_verifiers_image():
    """A setup manifest without --state is refused: chains recomputed over the image the prover's own setup wrote
    show only that the prover answers from that file. --trust-prover-image runs anyway, crediting no static
    memory, and refuses --max-scratch. Before: a warning, and the whole array credited."""
    from types import SimpleNamespace

    import numpy as np

    import memory_accounting.setup.verify as SV
    from memory_accounting.core.params import Keys, MemoryConfig

    words = 1 << 13
    with tempfile.TemporaryDirectory() as d:
        np.save(pathlib.Path(d) / "image.npy", np.arange(words, dtype=np.uint64).view(np.int64))
        spec = SimpleNamespace(
            gpu=lambda i: SimpleNamespace(image_file=str(pathlib.Path(d) / "image.npy")),
            is_bundle=lambda: False,
            filler_seed=0x5EED,
            accounted_words=words,
            memory_config=lambda i, image=True: MemoryConfig(words=words, encoding="vde", perm=False, image=image),
        )

        def args(**kw):
            return SimpleNamespace(
                **{
                    "manifest": "manifest.json",
                    "gpu": 0,
                    "state": None,
                    "filler_seed": "5eed",
                    "pad_seed": None,
                    "verify_threads": 1,
                    "replay": None,
                    "max_scratch": None,
                    "trust_prover_image": False,
                    "hbm_bytes": "1G",
                    "gpu_type": None,
                    "dyn_B": None,
                    **kw,
                }
            )

        real = SV.manifest_problems
        SV.manifest_problems = lambda *a, **k: []
        try:
            for kw, why in (({}, "--state"), ({"trust_prover_image": True, "max_scratch": "1G"}, "--max-scratch")):
                try:
                    net._make_verifier(args(**kw), spec, Keys.deterministic(1))
                    raise AssertionError(f"accepted: {kw}")
                except SystemExit as e:
                    assert why in str(e), e
            a = args(trust_prover_image=True)
            _ver, cfg = net._make_verifier(a, spec, Keys.deterministic(1))
        finally:
            SV.manifest_problems = real
    assert net._static_credit(a, cfg) == 0
    m = net._memory_bound(a, cfg, {"honest": {"rtt_ns": [1], "dyn_words": []}}, "honest")
    assert (
        m["static_bytes"] == 0
        and m["static_credited_bytes"] == 0
        and m["static_array_bytes"] == words * 8
        and not m["impossible"]
    ), m


def test_a_second_signal_does_not_cut_the_stop_short():
    """The demo stops the live verifier with SIGINT, then SIGTERM if it is still there: here 1 s apart, while record
    checks are backed up (each record held 3 s). The stop runs to its end anyway, within its bound: every round
    whose check was owed ends in a check or in the unchecked report, and the process exits 0. Before: the second
    signal raised KeyboardInterrupt inside the stop, which then lost its unchecked report (the reviewer's run: 172
    rounds with neither a check nor an unchecked report)."""
    import signal
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    relay = _free_port()
    threading.Thread(target=_relay, args=(relay, _dyn_prover(), []), daemon=True).start()
    _tcp_proxy(relay, _dyn_prover(), 3.0)
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "live.jsonl"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "memory_accounting.verifier.net",
                "live",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--device",
                "cpu",
                "--drain",
                "--dynamic",
                "--dyn-B",
                "64K",
                "--dyn-n",
                "4",
                "--strategies",
                "honest",
                "--warmup",
                "1",
                "--calibrate",
                "6",
                "--interval",
                "0.05",
                "--reply-timeout",
                "1.0",
                "--mode",
                "monitor",
                "--ui-port",
                str(_free_port()),
                "--out",
                str(log),
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            t = time.time()
            while time.time() - t < 60 and proc.poll() is None and len(_log_events(log)) < 120:
                time.sleep(0.2)
            proc.send_signal(signal.SIGINT)
            time.sleep(1.0)
            proc.send_signal(signal.SIGTERM)
            t_stop = time.time()
            out, _ = proc.communicate(timeout=40)
            took = time.time() - t_stop
        finally:
            if proc.poll() is None:
                proc.kill()
        evs = _log_events(log)
    assert proc.returncode == 0 and "UNCHECKED" in out, (proc.returncode, out[-1500:])
    assert took < 15, took
    owed = {
        e["id"]
        for e in evs
        if e.get("type") == "verification"
        and (e["round"] == "static" or (e["dyn"].get("nonce") and e["dyn"].get("received")))
    }
    checked = {e["id"] for e in evs if e.get("type") == "check"}
    left = [e for e in evs if e.get("type") == "unchecked"]
    assert left, "no unchecked report"
    reported = set(left[0]["static_ids"]) | set(left[0]["dynamic_ids"])
    assert owed and not owed - checked - reported, sorted(owed - checked - reported)


def test_baseline_role_schedules_dynamic_rounds_like_live():
    """The baseline (and verifier) role waits a random time (mean --interval / 2, the CSPRNG) before each dynamic
    round, as live does, so its dynamic deadline and READY bound are of a freeze anywhere in a step; the file
    says so (dyn.schedule) and keeps its calibration samples (role "baseline") for a live run's timing claim.
    Before: the freeze came at once after the static drain, and the file kept no samples."""
    import contextlib
    import io

    relay, log, times = _free_port(), [], []
    threading.Thread(target=_relay, args=(relay, _dyn_prover(), log, None, times), daemon=True).start()
    _tcp_proxy(relay, _dyn_prover(), 0.0)
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        out = pathlib.Path(d) / "b.json"
        net.main(
            [
                "baseline",
                "--host",
                "127.0.0.1",
                "--port",
                str(relay),
                "--size",
                "4M",
                "--B",
                "64K",
                "--n",
                "4",
                "--r",
                "0",
                "--device",
                "cpu",
                "--drain",
                "--dynamic",
                "--dyn-B",
                "64K",
                "--dyn-n",
                "4",
                "--strategies",
                "honest",
                "--samples",
                "30",
                "--warmup",
                "1",
                "--interval",
                "0.1",
                "--reply-timeout",
                "1.0",
                "--out",
                str(out),
            ]
        )
        b = json.loads(out.read_text())
    cs = b["calibration_samples"]
    assert (
        cs["role"] == "baseline"
        and len(cs["dyn_rtt_ns"]) >= 25
        and len(cs["static_chain_rtt_ns"]) >= 25
        and cs["span_s"] > 0
    ), cs.keys()
    assert b["dyn"]["schedule"] == "independent" and b["dyn"]["wait_mean_s"] == 0.05, b["dyn"]
    assert net.baseline_problems(b, _args(B="64K", n=4, r=0, dynamic=True, dyn_n=4), None) == []

    def arm(d):
        return len(d) == wire.ARM.size and wire.ARM.unpack(d)[0]

    follow = []  # a dynamic ARM within 10 ms of the reply before it
    for i, (w, dg) in enumerate(log):
        if w == "down" and arm(dg) == wire.ARM_DYN_MAGIC:
            j = max((k for k in range(i) if log[k][0] == "up"), default=None)
            follow.append(j is not None and times[i] - times[j] < 0.01)
    assert len(follow) >= 25 and sum(follow) < 0.5 * len(follow), (sum(follow), len(follow))


def test_period_claim_is_off_below_its_floor_on_dynamic_chains():
    """In the period model a dynamic chain whose slow-movement floor F_floor is over the target F is never testable
    (here 8 against 0.5): its period claim is off with the reason, its miss count runs, as the window model's rule.
    Before: the period ran with "floor F ≥ 8" printed. And a dynamic check that failed says what failed (here no
    record came: the relay passes no TCP), not "replaying ... does not reproduce" when nothing was replayed."""
    evs = _dyn_live_run()["events"]
    claims = [e for e in evs if e.get("type") == "meta" and "claims" in e][-1]["claims"]
    dyn = claims["dynamic"]
    assert dyn["tests"] == ["misses"] and "slow wander" in dyn["reason"] and dyn["F_floor"] > dyn["F"], dyn
    assert claims["static"]["tests"] == ["period", "misses"], claims["static"]
    assert not [e for e in evs if e.get("type") == "claim_period_verdict" and e.get("chain") == "dynamic"]
    failed = [
        a
        for a in evs
        if a.get("type") == "alert"
        and a.get("kind") == "dynamic"
        and "chain" not in a
        and "round trip" not in a["detail"]
    ]
    assert failed and all(a["detail"].startswith("no record of the chain came from the prover") for a in failed), [
        a["detail"][:80] for a in failed
    ]


def test_confirm_misses_never_excuses_a_gross_miss():
    """--confirm-misses: a round trip over the ceiling alerts at once (kind gross), whatever its retry shows; the
    retry comes after a random gap, as any round. Before: a gross miss whose retry passed raised nothing.
    The READY bound is 300 ms here, not twice the p99.9 of 12 calibration READYs (about 0.2 ms): a round whose
    READY is late is not also a timing miss, so a host stall on that READY left the round without a retry and the
    test failed (once in three full runs on a GPU host, where the earlier tests' threads share the process)."""
    W, C = 1, 12
    first = {"n": 0}

    def once(d):
        first["n"] += 1
        return [0.5] if first["n"] == 1 else [0]

    r = _live(
        _prover(),
        [
            "--warmup",
            str(W),
            "--calibrate",
            str(C),
            "--interval",
            "0.05",
            "--confirm-misses",
            "--max-verifications",
            "6",
            "--mode",
            "monitor",
            "--arm-max-ms",
            "300",
        ],
        _after_calibration(W + C, "up", wire.MAGIC, once),
    )
    evs = r["events"]
    late = [
        e
        for e in evs
        if e.get("type") == "verification" and any(c["rtt_ns"] and c["rtt_ns"] > 4e8 for c in e["challenges"])
    ]
    assert len(late) == 1, [e["id"] for e in late]
    gross = [a for a in evs if a.get("type") == "alert" and a.get("kind") == "gross" and a["id"] == late[0]["id"]]
    retry = [e for e in evs if e.get("type") == "retry" and e["id"] == late[0]["id"]]
    assert gross and retry, (
        gross,
        retry,
        {k: late[0].get(k) for k in ("late_ready", "arm_ns", "rtt_ns", "deadline_ns")},
    )


def test_memory_claim_holds_from_the_first_round():
    """The memory claim alerts from the first round, calibration included (here a capacity below the array: every
    round is impossible). Impossibility is judged against the array's size, not its credited size. Before: no
    memory alert while calibrating, and the credited size (62/64 of the array) let 3 % more through."""
    r = _live(
        _prover(),
        ["--warmup", "1", "--calibrate", "4", "--interval", "0.05", "--hbm-bytes", "3M", "--max-verifications", "2"],
    )
    cal = {e["id"] for e in r["events"] if e.get("type") == "verification" and e.get("phase") == "calibration"}
    mem = {a["id"] for a in r["events"] if a.get("type") == "alert" and a.get("kind") == "memory"}
    assert cal and cal <= mem, (sorted(cal), sorted(mem))
    G = 1 << 30
    assert net._memory_claim(100 * G, 62 * G, 37 * G, 0, None, static_array=64 * G)["impossible"]
    assert not net._memory_claim(100 * G, 62 * G, 36 * G, 0, None, static_array=64 * G)["impossible"]


def test_encoded_dynamic_region_is_credited_whole():
    """v0.17, --dyn-encoded: the prover holds its dynamic region encoded (memory_accounting.prover.kvenc: the KV cache and activation
    rows as VDE words, plaintext only in scratch), so the whole region, live region included, is credited at 62/64 in
    whole --dyn-B blocks, with or without the live chain (and --max-scratch is then allowed without it). A record whose
    region is not encoded is refused as unbound: the toy prover's plain records fail every check."""
    from types import SimpleNamespace as NS

    B, G = 1 << 16, 1 << 30
    c = net._memory_claim(4 * G, G, 10 * B + 100, 3 * B, None, 62, B, live_credited=True)
    assert c["dynamic_bytes"] == 10 * B * 62 // 64 and c["live_region_bytes"] == 3 * B, c
    # a live region larger than its pool
    assert net._memory_claim(4 * G, G, 10 * B, 11 * B, None, 62, B, live_credited=True)["impossible"]
    enc = NS(dynamic=True, dyn_encoded=True, dyn_live=False, dyn_dtype="bfloat16", max_scratch="8M")
    assert net._dyn_credit_bits(enc) == 62
    net._no_live_scratch(enc)  # allowed: nothing needs telling apart
    assert net._encoding_mismatch(enc, NS(stage={"kind": "vllm"}))
    assert net._encoding_mismatch(enc, NS(stage={"kind": "vllm", "encoded": True})) is None
    assert net._encoding_mismatch(NS(dyn_encoded=False), NS(stage={"kind": "vllm"})) is None
    common = [
        "--dynamic",
        "--dyn-B",
        "64K",
        "--dyn-n",
        "4",
        "--warmup",
        "1",
        "--calibrate",
        "4",
        "--interval",
        "0.05",
        "--hbm-bytes",
        "64M",
        "--max-verifications",
        "6",
        "--dyn-encoded",
    ]
    r = _live(_dyn_prover(), common, tcp_delay=0.0)
    dyn = [
        e
        for e in r["events"]
        if e.get("type") == "verification" and e.get("round") == "dynamic" and e["dyn"].get("words")
    ]
    assert dyn and all(
        e["memory"]["dynamic_bytes"] == e["memory"]["dynamic_region_bytes"] // B * B * 62 // 64 for e in dyn
    ), [e["memory"] for e in dyn]
    checks = [e["dyn_check"] for e in r["events"] if e.get("type") == "check" and e.get("dyn_check")]
    assert checks and all(c["digest_ok"] is False and "not encoded" in c.get("error", "") for c in checks), checks[:2]
    assert "encoded and credited whole" in r["printed"]


if __name__ == "__main__":
    from _runner import run

    run(globals())
