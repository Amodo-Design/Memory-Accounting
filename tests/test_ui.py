"""The live page's server (memory_accounting.verifier.ui.server) and the page itself: /history does not subscribe, a closed SSE
client is unsubscribed, inf and nan reach the browser as strict JSON, POSTs need the page's origin and JSON,
every request needs an allowed Host (DNS rebinding) and, when one is set, the token; event strings (some shaped
by the prover) are escaped wherever the page builds HTML, and the claim panel and banner follow one timing
claim per chain (static, dynamic, live)."""

import http.client
import json
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _runner import skip

from memory_accounting.verifier.ui import server
from memory_accounting.verifier.ui.server import EventHub

PAGE = pathlib.Path(server.__file__).parent / "index.html"
PAYLOAD = "<img src=x onerror=alert(1)>"


def _start(hub: EventHub) -> int:
    srv = server.start(hub, 0, "127.0.0.1")
    return srv.server_address[1]


def _strict(text: str):
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")

    return json.loads(text, parse_constant=bad)


def _sse(port: int, n_messages: int) -> tuple[socket.socket, list[str]]:
    """Open /events on a raw socket and read the first `n_messages` data lines."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(f"GET /events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
    buf = b""
    while buf.count(b"\ndata: ") + buf.startswith(b"data: ") < n_messages or not buf.endswith(b"\n\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    body = buf.split(b"\r\n\r\n", 1)[1].decode()
    return s, [ln[len("data: ") :] for ln in body.split("\n") if ln.startswith("data: ")]


def _post(port: int, path: str, body, headers: dict) -> tuple[int, str]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_history_does_not_subscribe():
    hub = EventHub()
    port = _start(hub)
    hub.publish({"type": "verification", "id": 1})
    for _ in range(5):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/history", timeout=5) as r:
            assert len(json.loads(r.read())["events"]) == 1
    assert hub.subscribers == 0, hub.subscribers


def test_sse_client_unsubscribed_on_close():
    hub = EventHub()
    port = _start(hub)
    s, msgs = _sse(port, 1)
    assert _strict(msgs[0])["type"] == "snapshot" and hub.subscribers == 1
    s.close()
    deadline = time.time() + 10
    while hub.subscribers and time.time() < deadline:  # the server notices at its next write
        hub.publish({"type": "stats"})
        time.sleep(0.05)
    assert hub.subscribers == 0, hub.subscribers


def test_non_finite_numbers_are_strict_json():
    hub = EventHub()
    port = _start(hub)
    hub.set_meta(claim={"mode": "period", "n_required": float("inf"), "F_floor": float("nan")})
    ev = {
        "type": "claim_period_verdict",
        "n_required": float("inf"),
        "mean_shift": float("nan"),
        "terms_us": {"period": float("inf"), "calibration": 1.5},
        "xs": [float("-inf"), 2.0, (float("nan"),)],
    }
    hub.publish(ev)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/history", timeout=5) as r:
        h = _strict(r.read().decode())
    v = h["events"][-1]
    assert (
        v["n_required"] is None and v["mean_shift"] is None and v["terms_us"] == {"period": None, "calibration": 1.5}
    ), v
    assert v["xs"] == [None, 2.0, [None]] and h["meta"]["claim"]["n_required"] is None, h
    s, msgs = _sse(port, 1)
    snap = _strict(msgs[0])
    assert snap["events"][-1]["n_required"] is None and snap["meta"]["claim"]["F_floor"] is None
    hub.publish(ev)  # a live event, not only the snapshot
    buf = b""
    while not buf.endswith(b"\n\n"):
        buf += s.recv(65536)
    s.close()
    live = _strict(buf.decode().split("data: ", 1)[1])
    assert live["n_required"] is None and live["terms_us"]["period"] is None, live
    assert hub.history[-1]["n_required"] == float("inf")  # the hub's events (and the verifier's log) keep the value


def test_post_needs_same_origin_and_json():
    hub = EventHub()
    port = _start(hub)
    hub.allowed["strategies"] = ["honest", "evict"]
    me = f"http://127.0.0.1:{port}"
    j = {"Content-Type": "application/json"}
    ok = [{}, {"Origin": me}, {"Referer": me + "/"}, {"Origin": me, "Referer": me + "/index.html"}]
    for extra in ok:
        code, _ = _post(port, "/control", {"strategies": ["honest"]}, {**j, **extra})
        assert code == 200, (extra, code)
        code, _ = _post(port, "/posture", {"posture": "evict"}, {**j, **extra})
        assert code == 202, (extra, code)
    assert hub.requests == [{"posture": "evict"}], hub.requests  # every POST accepted; only the latest is kept
    code, _ = _post(port, "/control", {"strategies": ["honest"]}, {"Content-Type": "application/json; charset=utf-8"})
    assert code == 200, code
    hub.requests.clear()
    hub.control.clear()
    bad = [
        ({**j, "Origin": "http://evil.example"}, 403),
        ({**j, "Origin": "null"}, 403),
        ({**j, "Origin": f"http://127.0.0.1:{port + 1}"}, 403),
        ({**j, "Referer": "http://evil.example/x"}, 403),
        ({**j, "Origin": me, "Referer": "http://evil.example/"}, 403),
        ({"Content-Type": "text/plain"}, 415),
        ({"Content-Type": "application/x-www-form-urlencoded"}, 415),
        ({}, 415),
    ]
    for headers, want in bad:
        for path, body in (("/control", {"strategies": ["evict"]}), ("/posture", {"posture": "evict"})):
            code, _ = _post(port, path, body, headers)
            assert code == want, (path, headers, code)
    assert hub.requests == [] and hub.control == {}, (hub.requests, hub.control)
    code, _ = _post(port, "/control", b"x" * (server.MAX_BODY + 1), j)
    assert code == 413, code
    code, _ = _post(port, "/posture", {"posture": "nope"}, j)
    assert code == 400, code


# ---- the page


def _script() -> str:
    return re.search(r"<script>(.*)</script>", PAGE.read_text(), re.DOTALL).group(1)


def test_page_static_no_raw_event_strings():
    """Best effort: no `${x.field}` of a string field that events carry appears outside esc() in the page
    script, except on lines that only set textContent."""
    fields = "detail|kind|received|expected|error|stage|path|name|strategy|host|nonce|posture|evidence_dir|gross|test|act_layer|n_layers"
    raw = re.compile(r"\$\{\s*[\w.]*\.(" + fields + r")\s*\}")
    # inside an esc(`...`) template is fine
    bad = [
        ln.strip()
        for ln in _script().splitlines()
        for m in raw.finditer(ln)
        if "textContent" not in ln and "esc(`" not in ln[: m.start()]
    ]
    assert not bad, bad
    for ln in _script().splitlines():
        if re.search(r"\.innerHTML\s*=|setHTML\(\$", ln) and "${" in ln:
            for m in re.finditer(r"\$\{([^}]*)\}", ln):
                e = m.group(1).strip()
                if "esc(`" not in ln[: m.start()]:
                    assert not re.fullmatch(r"[a-z]\.[A-Za-z_]+", e), (e, ln.strip())  # a bare field straight into HTML


HARNESS = r"""
const writes = [], errors = [];
const ctx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
function el(id) {
  let html = "";
  const e = { id, style: {}, dataset: {}, textContent: "", className: "", width: 0, height: 0, _h: {},
    classList: { add() {}, remove() {}, toggle() {} }, addEventListener(t, f) { this._h[t] = f; },
    getBoundingClientRect: () => ({ width: 800, height: 200 }), getContext: () => ctx,
    parentElement: { previousElementSibling: { firstElementChild: { textContent: "" } } } };
  Object.defineProperty(e, "innerHTML", { get: () => html, set: v => { html = String(v); writes.push([id, html]); } });
  return e;
}
const els = {};
globalThis.document = { getElementById: id => els[id] || (els[id] = el(id)), querySelector: s => document.getElementById(s.slice(1)),
                        documentElement: {}, body: { dataset: {} } };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "", fontFamily: "x" });
globalThis.window = { devicePixelRatio: 1, addEventListener() {} };
let tick = null, es = null;
globalThis.setInterval = f => { tick = f; };
globalThis.setTimeout = () => {};
globalThis.EventSource = class { constructor() { es = this; } close() {} };
globalThis.fetch = () => Promise.resolve({ json: () => ({}) });
//SCRIPT//
const send = ev => es.onmessage({ data: JSON.stringify(ev) });
const run = () => { try { tick(); } catch (e) { errors.push(String(e && e.stack || e)); } };
const click = id => { try { els.rows._h.click({ target: { closest: () => ({ dataset: { id: String(id) } }) } }); } catch (e) { errors.push(String(e.stack || e)); } };
"""
SCENARIO_ESCAPES = r"""
const P = "<img src=x onerror=alert(1)>";
const base = { median_ns: 1e5, sigma_core_ns: 1e3, deadline_ns: 2e5, deadline_rule: "sigma", sigmas: P, margin: 0.1, arm_max_ns: 1e6,
               honest_reject_rate: 0.01, dyn: { median_ns: 1e5, deadline_ns: 2e5, live: { deadline_ns: 3e5 }, arm_max_ns: 1e6 } };
const winClaim = { F: 0.01, C: 0.999, alpha: 0.001, L: P, delta_c_us: 17, mu_b_us: null, sigma_b_us: null, delta_F_us: null, tau_us: null,
                   t_us: null, M_us: P, n_window: null, n_cal: P, clip_us: null };
send({ type: "snapshot", meta: { mode: "research", host: P, manifest: P, evidence_dir: P, dynamic: true, dyn_B: 4096, dyn_n: 4, B: 4096, n: P, r: P, k: P,
       repeat: 2, drain: true, baseline: base, state: "running", claim: winClaim,
       memory: { capacity_bytes: 8 * 2 ** 30, static_bytes: 4 * 2 ** 30, bound_bytes: 2 ** 30, gpu_type: P },  // (8 << 30 is 0 in JS)
       allowed: { strategies: [P, "honest"] }, control: { strategies: [P] } }, events: [] });
const dyn = { nonce: P, j0: P, blocks: P, words: 8, received: P, rtt_ns: 1e5, arm_ns: 10, deadline_ns: 2e5, in_flight: true,
              live: { nonce: P, j0: P, blocks: P, words: 8, received: P, rtt_ns: 1e5, deadline_ns: 2e5 } };
const chs = [{ nonce: P, j0: P, rtt_ns: 1e5, arm_ns: 5, received: P }];
send({ type: "verification", id: 1, t: 1, strategy: P, phase: "run", within: true, rtt_ns: 1e5, gap_sigma: 1, timing_miss: true, gross: [P],
       challenges: chs, dyn, memory: { dynamic_bytes: 2 ** 30, unchecked_bytes: 2 ** 30, over: true }, cusum: 1 });
send({ type: "verification", id: 2, t: 2, strategy: "honest", phase: "run", within: true, rtt_ns: 1e5, challenges: chs, dyn, retried_by: P, miss_confirmed: false });
const check = { type: "check", digest_ok: false, dyn_digest_ok: false, challenges: [{ nonce: P, expected: P, ok: false }],
                dyn_check: { error: P, replayed: true, stage: P, path: P, differing_segments: [{ name: P }], in_flight: true, act_layer: P, n_layers: P,
                             expected: P, live: { digest_ok: false, expected: P, error: P } } };
send({ ...check, id: 1 }); send({ ...check, id: 2, digest_ok: true });
// net.py's separate rounds: a dynamic one whose record did not bind (the verifier's reason quotes the prover), one
// whose record was bound but not kept (cap), one left unchecked at the stop; alerts naming older rounds; evidence saving stopped
send({ type: "verification", round: "dynamic", id: 3, t: 3, strategy: P, phase: "run", within: true, rtt_ns: null, challenges: [], dyn });
send({ type: "check", id: 3, digest_ok: true, dyn_digest_ok: false, challenges: [], dyn_check: { digest_ok: false, replayed: false, error: P } });
send({ type: "verification", round: "dynamic", id: 4, t: 4, strategy: P, phase: "run", within: true, rtt_ns: null, challenges: [], dyn });
send({ type: "check", id: 4, digest_ok: true, dyn_digest_ok: null, challenges: [], record_saved: false, dyn_check: { digest_ok: null, replayed: false, stage: P } });
send({ type: "verification", round: "dynamic", id: 5, t: 5, strategy: P, phase: "run", within: true, rtt_ns: null, challenges: [], dyn });
send({ type: "alert", id: 5, oldest_id: P, ids: [P], t: 5, kind: "check", detail: P });
send({ type: "alert", id: 5, t: 5, kind: "evidence", detail: P });
send({ type: "meta", evidence_saving: false });
send({ type: "unchecked", static_ids: [], dynamic_ids: [5], detail: P });
run(); click(3); click(4); click(5); run();
send({ type: "prover", served: P, fired: P, armed: P, hold_expired: P, svc_p50_ns: 1, svc_p99_ns: 2, dt_p50_ns: 1, dt_p99_ns: 2, posture: P });
send({ type: "alert", id: P, t: 3, kind: P, detail: P });
send({ type: "claim_progress", n: P, mean_shift: null, clipped: P });
send({ type: "claim_verdict", window: P, pass: false, mean_shift: null, D_us: null, t_us: null, tau: null, clipped: P, implied_F: null });
send({ type: "claim_verdict", window: P, pass: true, baseline_moved: true, mean_shift: 1, tau: 3, implied_F: null });
run(); click(1); click(2); run();
send({ type: "meta", mode: "monitor" }); run(); click(1);
send({ type: "meta", claim: { ...winClaim, mode: "period", period_s: 600, sigma_s_us: P, t_corr_s: 600, F_floor: null, n_required: null,
                              rate_required: null, n_expected: null, cal_span_s: 60 } });
send({ type: "claim_period_progress", period: P, elapsed_s: 10, n: P, mean_shift: 1, F_testable: 0.5, F_floor: null,
       tau: null, D_us: null, t_us: null, clipped: P });
send({ type: "claim_period_verdict", id: 2, period: P, n: P, test: P, tau: null, mean_shift: null, F_testable: null, supports_target: false });
run();
send({ type: "claim_period_verdict", id: 2, period: P, n: P, test: "pass", tau: null, mean_shift: null, F_testable: null, supports_target: true });
run();
send({ type: "meta", claim: { ...winClaim, chain: P } });                  // a chain name is a string from the verifier: escaped too
send({ type: "claim_verdict", chain: P, window: P, pass: false, mean_shift: null, tau: null, implied_F: null });
run();
const last = {}; writes.forEach(([id, h]) => { last[id] = h; });
console.log(JSON.stringify({ writes, errors, last, status: els.status.textContent }));
"""


def _node_page(scenario: str):
    """Run the page script in node under the stub DOM, then `scenario`; its JSON output, or None without node."""
    node = shutil.which("node")
    if not node:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(HARNESS.replace("//SCRIPT//", _script()) + scenario)
        path = f.name
    out = subprocess.run([node, path], capture_output=True, text=True, timeout=60)
    pathlib.Path(path).unlink()
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_page_escapes_event_strings():
    """Drives the page script in node with a stub DOM: every string field of every event kind carries an HTML
    payload, and no innerHTML the page writes may contain it as markup. Numbers sent as null (inf and nan
    from the hub) must render without an exception."""
    r = _node_page(SCENARIO_ESCAPES)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    leaks = [(i, h) for i, h in r["writes"] if "<img" in h]
    assert not leaks, leaks[:3]
    esc = "&lt;img src=x onerror=alert(1)&gt;"
    assert any(i == "memlegend" for i, _ in r["writes"])  # the memory panel was drawn
    for id_ in (
        "meta",
        "prover",
        "toggles",
        "postures",
        "alerts",
        "rows",
        "claim-kv",
        "claim-verdicts",
        "claim-chains",
    ):
        assert any(i == id_ and esc in h for i, h in r["writes"]), id_  # the payload went through this path, escaped
    rows = [h for i, h in r["writes"] if i == "rows" and "dynamic: nonce" in h]
    # the expanded detail: nonces, digests, errors, stage, path
    assert rows and all(h.count(esc) >= 10 for h in rows), rows[:1]
    assert "—" in r["last"]["claim-kv"]


SCENARIO_CHAINS = r"""
const out = {};
const snapAt = () => ({ status: els.status.textContent, banner: els.banner.className, chains: els["claim-chains"].innerHTML,
                        rows: els.rows.innerHTML, meta: els.meta.innerHTML, statement: els["claim-statement"].textContent,
                        windows: els["c-windows"].textContent, kv: els["claim-kv"].innerHTML });
const base = { median_ns: 1e5, sigma_core_ns: 1e3, deadline_ns: 2e5, deadline_rule: "sigma", sigmas: 6, margin: 0.1, arm_max_ns: 1e6,
               honest_reject_rate: 0.01, dyn: { median_ns: 1e5, deadline_ns: 2e5, live: { deadline_ns: 3e5 }, arm_max_ns: 1e6 } };
const win = { mode: "window", F: 0.01, C: 0.999, alpha: 0.001, L: 8, delta_c_us: 17, mu_b_us: 100, sigma_b_us: 2, delta_F_us: 3,
              tau_us: 1.5, t_us: 96, M_us: 136, n_window: 64, n_cal: 500, clip_us: null };
const per = { ...win, mode: "period", period_s: 600, sigma_s_us: 0.5, t_corr_s: 600, F_floor: 1e-4, n_required: 1e5, rate_required: 100,
              n_expected: 2e5, cal_span_s: 60, chain: "live" };
const dynWin = { ...win, chain: "dynamic", mu_b_us: 222 };
// A static round that passed, then a dynamic round whose dynamic and live chains both missed their deadlines (and no
// gross miss): each miss feeds its own chain's claim. (net.py publishes static and dynamic rounds as separate events.)
const dyn = { nonce: "n", j0: 1, blocks: 8, words: 8, received: "r", rtt_ns: 2.5e5, arm_ns: 10, deadline_ns: 2e5, within: false, in_flight: false, late_ready: false,
              live: { nonce: "n", j0: 1, blocks: 8, words: 8, received: "r", rtt_ns: 4e5, deadline_ns: 3e5, within: false } };
const sver = { type: "verification", round: "static", id: 6, t: 1, strategy: "honest", phase: "run", within: true, rtt_ns: 1e5, deadline_ns: 2e5,
               timing_miss: false, gross: [], challenges: [], late_ready: false };
const ver = { type: "verification", round: "dynamic", id: 7, t: 1, strategy: "honest", phase: "run", within: false, rtt_ns: null, arm_ns: 10,
              deadline_ns: 2e5, late_ready: false, timing_miss: true, gross: [], challenges: [], dyn };
const checks = [{ type: "check", id: 6, digest_ok: true, challenges: [] },
                { type: "check", id: 7, digest_ok: true, dyn_digest_ok: true, challenges: [], dyn_check: { replayed: true, live: { digest_ok: true } } }];
// The snapshot's meta carries only the last claim the verifier set (live); the history replays the other two.
send({ type: "snapshot", meta: { mode: "monitor", dynamic: true, baseline: base, baseline_self_calibrated: true, state: "running", claim: per,
                                 allowed: { strategies: ["honest"] }, control: { strategies: ["honest"] } },
       events: [{ type: "meta", claim: win }, { type: "meta", claim: dynWin }, { type: "meta", claim: per }, sver, ver, ...checks] });
send({ type: "claim_progress", n: 10, mean_shift: 0.1, clipped: 0 });                                     // no chain: static
send({ type: "claim_verdict", window: 1, pass: true, mean_shift: 0.2, tau: 1.5, implied_F: 0.001 });
send({ type: "claim_window", chain: "dynamic", window: 1, pass: true, mean_shift: 0.4, tau: 1.5, implied_F: 0.002 });
send({ type: "claim_period_progress", chain: "live", period: 1, elapsed_s: 30, n: 50, mean_shift: 4, F_testable: 0.2, F_floor: null, tau: 3 });
run(); out.progress = snapAt();
send({ type: "claim_period_verdict", chain: "live", id: 7, period: 1, n: 90, test: "fail", mean_shift: 5, tau: 2, F_testable: 0.3, supports_target: false });
run(); out.fail = snapAt();
try { els["claim-chains"]._h.click({ target: { closest: () => ({ dataset: { chain: "dynamic" } }) } }); } catch (e) { errors.push(String(e.stack || e)); }
run(); out.dynamic = snapAt();
send({ type: "claim_period_verdict", chain: "live", id: 8, period: 2, n: 90, test: "pass", mean_shift: 0, tau: 2, F_testable: 0.004, supports_target: true });
run(); out.recovered = snapAt();
// A run with a static claim only: a dynamic miss feeds no claim, so it stays a failure.
send({ type: "snapshot", meta: { mode: "monitor", dynamic: true, baseline: base, state: "running", claim: win, allowed: { strategies: ["honest"] } },
       events: [sver, ver, ...checks] });
run(); out.static_only = snapAt();
console.log(JSON.stringify({ errors, out }));
"""


def test_page_claims_per_chain():
    """Three timing claims, keyed by the events' "chain" (none: static): each chain gets its row in the claim
    panel, the banner shows a failure on any chain (here the live one) and recovers when it holds again, a
    dynamic or live miss is weighed by its chain's claim, and a self-calibrated baseline is marked."""
    r = _node_page(SCENARIO_CHAINS)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    out = r["out"]
    progress = out["progress"]
    for ch in ("static", "dynamic", "live"):
        assert f'data-chain="{ch}"' in progress["chains"], (ch, progress["chains"])
    assert progress["banner"] == "banner ok" and progress["status"].startswith("Claim: static holds"), progress[
        "status"
    ]
    assert "period 1 in progress" in progress["status"] and "dynamic holds" in progress["status"], progress["status"]
    assert (
        "weighed by the claim" in progress["rows"]
        and "v warn" in progress["rows"]
        and "v fail" not in progress["rows"].split("dyn-only")[0]
    ), progress["rows"]
    assert (
        "baseline" in progress["meta"]
        and "calibrated on this prover (not independently trusted)" in progress["meta"]
        and "quiet-warn" in progress["meta"]
    )
    assert progress["windows"] == "static 1 of 1 · dynamic 1 of 1 · live 0 of 0 periods", progress["windows"]
    assert progress["statement"].startswith("Static chain. Claim:"), progress[
        "statement"
    ]  # the first chain is shown by default
    f = out["fail"]
    assert f["banner"] == "banner alert" and f["status"].startswith("Live chain · Claim fails over period 1"), f[
        "status"
    ]
    live_row = re.search(r'<tr class="row[^"]*" data-chain="live">.*?</tr>', f["chains"]).group(0)
    assert 'class="v fail">fails<' in live_row and "0 of 1 periods" in live_row, live_row
    static_row = re.search(r'<tr class="row[^"]*" data-chain="static">.*?</tr>', f["chains"]).group(0)
    assert 'class="v pass">holds<' in static_row and "1 of 1 windows held" in static_row, static_row
    d = out["dynamic"]  # a click on a row shows that chain
    assert d["statement"].startswith("Dynamic chain. Claim:") and "222.0" in d["kv"], (d["statement"], d["kv"])
    assert 'class="row sel" data-chain="dynamic"' in d["chains"], d["chains"]
    assert d["banner"] == "banner alert", d["status"]
    rec = out["recovered"]
    assert rec["banner"] == "banner ok" and rec["status"] == "Claim holds on every chain (static, dynamic, live)", rec[
        "status"
    ]
    s = out["static_only"]
    assert 'data-chain="static"' in s["chains"] and 'data-chain="dynamic"' not in s["chains"], s["chains"]
    assert "weighed by the claim" not in s["rows"] and "v fail" in s["rows"], s["rows"]
    assert "not independently trusted" not in s["meta"]


SCENARIO_TESTS = r"""
const out = {};
const get = id => (els[id] ? (els[id].innerHTML || els[id].textContent) : "");
const snapAt = () => ({ status: els.status.textContent, banner: els.banner.className, chains: get("claim-chains"), rows: els.rows.innerHTML,
                        meta: els.meta.innerHTML, statement: get("claim-statement"), pn: get("claim-progress-n"), pw: els["claim-progress"] ? els["claim-progress"].style.width : "",
                        alerts: els["c-alerts"].textContent, pass: els["c-pass"].textContent, misses: els["c-misses"].textContent, kv: get("claim-kv"),
                        memlegend: get("memlegend"), memnote: get("memnote"), state: els.state.textContent, gauges: els["claim-gauges"] ? els["claim-gauges"].style.display : "" });
const base = { median_ns: 1e5, sigma_core_ns: 1e3, deadline_ns: 2e5, deadline_rule: "sigma", sigmas: 6, margin: 0.1, arm_max_ns: 1e6,
               dyn: { median_ns: 1e5, deadline_ns: 2e5, arm_max_ns: 1e6, live: { deadline_ns: 3e5 } } };
const misses = { window: 200, k: 9, miss_rate: 0.01, alpha: 0.001 }, G = 2 ** 30;
// the static chain's mean shift cannot be sized: a miss count only; the dynamic chain likewise; the live chain runs nothing
const sClaim = { chain: "static", tests: ["misses"], F: 0.2, C: 0.999, alpha: 0.001, n_window: null, reason: "F = 0.2 cannot be tested with 30 calibration responses", misses };
const dClaim = { chain: "dynamic", tests: ["misses"], F: 0.2, C: 0.999, n_window: null, reason: "τ = 0.80 µs is below the honest level's slow wander", misses };
const lClaim = { chain: "live", tests: [], F: 0.2, C: 0.999, n_window: null, reason: "τ = 0.50 µs is below the honest level's slow wander" };
const meta0 = { mode: "monitor", dynamic: true, baseline: base, state: "running", claim: sClaim, claims: { static: sClaim, dynamic: dClaim, live: lClaim },
                memory: { capacity_bytes: 16 * G, static_bytes: 4 * G, bound_bytes: null }, allowed: { strategies: ["honest", "evict"] } };
const dyn = (live_within = true) => ({ rtt_ns: 1e5, deadline_ns: 2e5, within: live_within, late_ready: false, words: 8, arm_ns: 1,
                                      live: { rtt_ns: live_within ? 1e5 : 4e5, deadline_ns: 3e5, within: live_within } });
// net.py's rounds: a static one (its chains, a digest check) and a dynamic one (its chains, a record check), apart
const ver = (id, extra) => ({ type: "verification", round: "static", id, t: 5 * id, phase: "run", within: true, rtt_ns: 1e5, deadline_ns: 2e5, late_ready: false,
                              timing_miss: false, gross: [], challenges: [], ...extra });
const dver = (id, extra = {}) => { const d = extra.dyn || dyn();
  return { type: "verification", round: "dynamic", id, t: 5 * id, phase: "run", rtt_ns: null, arm_ns: 1, deadline_ns: 2e5, timing_miss: false, gross: [], challenges: [],
           ...extra, within: d.within, late_ready: !!d.late_ready, dyn: d }; };
const ok = id => ({ type: "check", id, digest_ok: true, challenges: [] });
const dok = id => ({ type: "check", id, digest_ok: true, dyn_digest_ok: true, challenges: [], dyn_check: { replayed: true, live: { digest_ok: true } } });
send({ type: "snapshot", meta: meta0, events: [] });
send(ver(1, { within: false, rtt_ns: 3e5, timing_miss: true })); send(ok(1));                       // a static miss: the miss count weighs it
send(dver(2, { timing_miss: true, dyn: dyn(false) })); send(dok(2));                                  // a live miss: no test runs there
send({ type: "claim_progress", chain: "static", misses: 1, misses_n: 2, misses_window: 200, misses_k: 9 });
send({ type: "claim_progress", chain: "dynamic", misses: 3, misses_n: 50, misses_window: 200, misses_k: 9 });
send({ type: "alert", id: 2, t: 10, kind: "dynamic", detail: "live chain round trip 400 µs over its deadline 300 µs" });
for (let i = 3; i < 20; i++) { send(ver(i)); send(ok(i)); }
send(ver(20)); send({ type: "prover", posture: "evict", served: 1 });
run(); out.counts = snapAt();
try { els["claim-chains"]._h.click({ target: { closest: () => ({ dataset: { chain: "dynamic" } }) } }); } catch (e) { errors.push(String(e.stack || e)); }
run(); out.dynsel = snapAt();
try { els["claim-chains"]._h.click({ target: { closest: () => ({ dataset: { chain: "live" } }) } }); } catch (e) { errors.push(String(e.stack || e)); }
run(); out.livesel = snapAt();
// the dynamic chain's miss count alarms: it stays failed long after ten verifications, until its window has passed
send({ type: "alert", id: 21, t: 105, kind: "claim", chain: "dynamic", rule: "misses", detail: "dynamic chain: 9 of its last 200 round trips missed the deadline" });
send({ type: "claim_progress", chain: "dynamic", misses: 0, misses_n: 0, misses_window: 200, misses_k: 9 });
for (let i = 21; i < 60; i++) { send(ver(i)); send(ok(i)); send({ type: "claim_progress", chain: "dynamic", misses: 0, misses_n: i - 20, misses_window: 200, misses_k: 9 }); }
run(); out.alarm = snapAt();
send({ type: "claim_progress", chain: "dynamic", misses: 0, misses_n: 200, misses_window: 200, misses_k: 9 });
run(); out.cleared = snapAt();
// a reconnect: the snapshot replays the alert, which must be counted once; the old posture and opened rows go
click(5);
const hist = [{ type: "alert", id: 1, t: 1, kind: "x", detail: "a" }, ver(1), ok(1)];
send({ type: "snapshot", meta: { ...meta0 }, events: hist }); send({ type: "snapshot", meta: { ...meta0 }, events: hist });
run(); out.reconnect = { ...snapAt(), posture: els["posture-now"].textContent };
// a verification with a missing reply is not excused by a passing retry, nor weighed; a gross miss neither
send({ type: "snapshot", meta: { ...meta0, confirm_misses: true }, events: [] });
send(ver(50, { within: false, faults: ["no reply to a static challenge"], rtt_ns: 3e5, timing_miss: true, gross: ["no reply to a static challenge"] })); send(ok(50));
send(ver(51, { retry_of: 50 })); send(ok(51)); send({ type: "retry", id: 50, retried_by: 51, miss_confirmed: false });
send(ver(52, { within: false, rtt_ns: 9e5, timing_miss: true, gross: ["round trip 900 µs, over 4 × the deadline 200 µs"] })); send(ok(52));
send(dver(53, { dyn: { ...dyn(), late_ready: true }, timing_miss: true, gross: ["dynamic READY after 9 ms"] })); send(dok(53));
// a check that threw (net.py's failed record check); memory impossible with no bound
send(dver(54, { memory: { dynamic_bytes: 3 * G, dynamic_region_bytes: 4 * G, live_region_bytes: G / 2, unchecked_bytes: 9 * G, impossible: true, over: true } }));
send({ type: "check", id: 54, error: "OSError: <evidence> unreadable", challenges: [], digest_ok: true, dyn_digest_ok: false,
       dyn_check: { digest_ok: false, error: "OSError: <evidence> unreadable" } });
run(); click(54); run(); out.faults = snapAt();
// the run stops, then the connection drops
send({ type: "meta", state: "stopped" }); run(); out.stopped = snapAt();
try { es.onerror(); } catch (e) { errors.push(String(e.stack || e)); }
run(); out.disconnected = snapAt();
// the static chain with both a window test and a miss count (net.py's contract): the window decides while the count is fine
const wm = { chain: "static", tests: ["window", "misses"], mode: "window", F: 0.01, C: 0.999, alpha: 0.001, L: 8, delta_c_us: 17, mu_b_us: 100, sigma_b_us: 2,
             delta_F_us: 3, tau_us: 1.5, t_us: 96, M_us: 136, n_window: 64, n_cal: 500, clip_us: null, misses };
send({ type: "snapshot", meta: { mode: "monitor", dynamic: false, baseline: base, state: "running", claims: { static: wm }, allowed: { strategies: ["honest"] } }, events: [] });
for (let i = 1; i < 15; i++) { send(ver(i)); send(ok(i)); }
send({ type: "claim_verdict", chain: "static", window: 1, pass: true, mean_shift: 0.2, tau: 1.5, implied_F: 0.001 });
send({ type: "claim_progress", chain: "static", n: 5, n_window: 64, mean_shift: 0.1, clipped: 0, misses: 2, misses_n: 69, misses_window: 200, misses_k: 9 });
run(); out.combined = snapAt();
send({ type: "alert", id: 14, t: 70, kind: "claim", chain: "static", rule: "misses", detail: "9 of its last 200 round trips missed the deadline" });
for (let i = 15; i < 30; i++) { send(ver(i)); send(ok(i)); }
run(); out.combined_alarm = snapAt();
console.log(JSON.stringify({ errors, out }));
"""


def test_page_claim_tests_and_failures():
    """The page follows what runs on each chain (meta claims[chain].tests): a chain where nothing runs is
    "claim off" with its reason and weighs no miss; the miss count shows "x of k misses in the last n" and
    its alarm keeps the chain failed until its window has passed. A snapshot replaces alerts and posture;
    a missing reply, a gross miss or a late READY is never softened; a check that threw says so; memory
    impossible is not "over bound"; a stopped run and a lost connection grey the banner."""
    r = _node_page(SCENARIO_TESTS)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    out = r["out"]

    def row(snap, i):
        return re.search(rf'data-id="{i}">.*?</tr>', snap["rows"], re.DOTALL).group(0)

    counts = out["counts"]
    assert "weighed by the claim" in row(counts, 1) and "v warn" in row(counts, 1), row(
        counts, 1
    )  # static: its miss count runs
    assert "v fail" in row(counts, 2) and "weighed" not in row(counts, 2), row(counts, 2)  # live: claim off, a failure
    live_row = re.search(r'<tr class="row[^"]*" data-chain="live">.*?</tr>', counts["chains"]).group(0)
    assert 'class="v off">claim off: τ = 0.50 µs is below' in live_row, live_row
    dyn_row = re.search(r'<tr class="row[^"]*" data-chain="dynamic">.*?</tr>', counts["chains"]).group(0)
    assert (
        "3 of 9 misses in the last 50" in dyn_row and "miss count: 9 of 200" in dyn_row and 'class="v pass"' in dyn_row
    ), dyn_row
    for snap in (counts, out["dynsel"], out["livesel"]):
        for bad in ("undefined", "NaN", "window of —", "first window in progress"):
            assert bad not in json.dumps(snap), (bad, snap)
    assert (
        counts["banner"] == "banner ok"
        and "claim off on live" in counts["status"]
        and "dynamic 3 of 9 misses in the last 50" in counts["status"]
    ), counts["status"]
    # #3 to #19; the weighed miss, the live miss and #20 (unchecked) are not passes
    assert str(counts["pass"]) == "17", counts["pass"]
    assert counts["misses"].startswith("1 "), counts["misses"]  # the weighed one only
    assert (
        "weighed by the timing claim (static, dynamic; a gross miss alerts at once); live: alerts at once"
        in counts["meta"]
    ), counts["meta"]
    assert "dynamic, credited" in counts["memlegend"] and "unchallenged or uncredited" in counts["memlegend"], counts[
        "memlegend"
    ]
    d = out["dynsel"]
    assert (
        "Mean-shift claim off: τ = 0.80" in d["statement"]
        and "3 of 9 misses in the last 50" in d["pn"]
        and d["pw"] == "33.3%"
    ), d
    lv = out["livesel"]
    assert lv["statement"].startswith("Live chain. Claim off: τ = 0.50") and lv["gauges"] == "none", lv
    a = out["alarm"]  # 39 verifications after the alarm: still failed
    assert a["banner"] == "banner alert" and a["status"].startswith("Dynamic chain · Too many deadline misses"), a[
        "status"
    ]
    assert 'class="v fail">too many misses<' in a["chains"], a["chains"]
    cl = out["cleared"]
    assert cl["banner"] == "banner ok", cl["status"]
    rc = out["reconnect"]
    assert (
        str(rc["alerts"]) == "1"
        and rc["posture"].startswith("prover posture unknown")
        and "dynamic: nonce" not in rc["rows"]
    ), rc
    faults = out["faults"]
    assert (
        "retry #51 passed" not in faults["rows"]
        and "no reply to a static challenge" in row(faults, 50)
        and "v fail" in row(faults, 50)
    ), row(faults, 50)
    assert "gross: round trip 900" in row(faults, 52) and "v fail" in row(faults, 52), row(faults, 52)
    assert "dynamic: late READY" in row(faults, 53) and "v fail" in row(faults, 53), row(faults, 53)
    r54 = row(faults, 54)
    assert (
        "check failed: OSError: &lt;evidence&gt; unreadable" in r54
        and "fetching record" not in r54
        and "digest mismatch" not in r54
    ), r54
    assert (
        "check failed:</span> OSError: &lt;evidence&gt;" in faults["rows"]
        and "impossible: more than the GPU holds" in faults["rows"]
    ), faults["rows"]
    assert "over bound" not in faults["rows"] and "live region 512 MiB checked, not credited" in faults["rows"]
    assert (
        "declared more memory than the GPU holds" in faults["memnote"]
        and "live region, 512 MiB, is checked, not credited" in faults["memnote"]
    ), faults["memnote"]
    assert faults["misses"].startswith("0 "), faults["misses"]
    s = out["stopped"]
    assert s["banner"] == "banner idle" and s["status"].startswith("Run stopped · last state: "), s["status"]
    dc = out["disconnected"]
    assert dc["banner"] == "banner idle" and dc["status"].startswith("Disconnected from the verifier"), dc["status"]
    assert dc["state"] == "disconnected", dc["state"]
    cb = out["combined"]
    assert (
        cb["banner"] == "banner ok"
        and cb["status"].startswith("Claim holds: window 1")
        and "miss count: 2 of 9 misses in the last 69" in cb["status"]
    ), cb["status"]
    assert (
        "window of 64 challenges + miss count: 9 of 200" in cb["chains"]
        and "1 of 1 windows held · 2 of 9 misses in the last 69" in cb["chains"]
    ), cb["chains"]
    assert "deadline misses" in cb["kv"] and cb["pn"] == "5 of 64 challenges", (cb["kv"], cb["pn"])
    ca = out["combined_alarm"]
    assert ca["banner"] == "banner alert" and ca["status"].startswith("Too many deadline misses"), ca["status"]


SCENARIO_ROUNDS = r"""
const out = {};
const get = id => (els[id] ? (els[id].innerHTML || els[id].textContent) : "");
const snapAt = () => ({ status: els.status ? els.status.textContent : "", banner: els.banner ? els.banner.className : "", rows: els.rows.innerHTML,
                        meta: els.meta.innerHTML, chains: get("claim-chains"), statement: get("claim-statement"), alerts: get("alerts"),
                        summary: get("summary"), memlegend: get("memlegend"), memnote: get("memnote") });
const G = 2 ** 30;
const base = { median_ns: 1e5, sigma_core_ns: 1e3, deadline_ns: 2e5, deadline_rule: "sigma", sigmas: 6, margin: 0.1, arm_max_ns: 1e6,
               dyn: { median_ns: 1e5, deadline_ns: 2e5, arm_max_ns: 1e6, live: { deadline_ns: 3e5 } } };
const misses = { window: 200, k: 9, miss_rate: 0.01, alpha: 0.001 };
const win = { chain: "static", tests: ["window", "misses"], mode: "window", F: 0.01, C: 0.999, alpha: 0.001, L: 8, delta_c_us: 17, mu_b_us: 100, sigma_b_us: 2,
              delta_F_us: 3, tau_us: 1.5, t_us: 96, M_us: 136, n_window: 64, n_cal: 500, clip_us: null, misses };
// the dynamic chain's period takes an hour to fill: its tests run, but every miss of it alerts at once (net.py: per_miss)
const why = "its period takes about 60 min to fill at this pace, over --claim-max-fill-s 1800 s: every miss of this chain alerts at once";
const per = { chain: "dynamic", tests: ["period", "misses"], per_miss: true, reason: why, mode: "period", F: 0.01, C: 0.999, alpha: 0.001, L: 4, delta_c_us: 2,
              mu_b_us: 100, sigma_b_us: 2, tau_us: 1, period_s: 3600, sigma_s_us: 0.5, t_corr_s: 600, F_floor: 1e-4, n_required: 1e5, rate_required: 30,
              n_expected: 2e4, cal_span_s: 60, n_cal: 100, clip_us: null, misses };
const dyn = (over = false) => ({ nonce: "n", j0: 1, blocks: 8, words: 8, received: "r", rtt_ns: over ? 2.5e5 : 1e5, arm_ns: 1, deadline_ns: 2e5, within: !over,
                                 late_ready: false, in_flight: true, live: { rtt_ns: 1e5, deadline_ns: 3e5, within: true, words: 8 } });
const T0 = 1000;
const ver = (id, t, extra) => ({ type: "verification", round: "static", id, t, strategy: "honest", phase: "run", within: true, rtt_ns: 1e5, deadline_ns: 2e5,
                                 late_ready: false, timing_miss: false, gross: [], challenges: [], ...extra });
const dver = (id, t, over = false, extra = {}) => ({ type: "verification", round: "dynamic", id, t, strategy: "honest", phase: "run", within: !over, rtt_ns: null,
                                 arm_ns: 1, deadline_ns: 2e5, late_ready: false, timing_miss: over, gross: [], challenges: [], dyn: dyn(over), ...extra });
const ok = id => ({ type: "check", id, digest_ok: true, challenges: [] });
const dok = (id, extra = {}) => ({ type: "check", id, digest_ok: true, dyn_digest_ok: true, challenges: [], dyn_check: { replayed: true, live: { digest_ok: true } }, ...extra });
const mem = { capacity_bytes: 16 * G, static_bytes: 4 * G * 62 / 64, static_credited_bytes: 4 * G * 62 / 64, static_array_bytes: 4 * G, bound_bytes: null };
const meta0 = { mode: "monitor", dynamic: true, baseline: base, state: "running", claim: win, claims: { static: win, dynamic: per }, memory: mem,
                evidence_dir: "runs/ev", allowed: { strategies: ["honest"] } };
send({ type: "snapshot", meta: meta0, events: [] });
send(ver(1, T0)); send(ok(1));
send(dver(2, T0 + 0.25));                                                   // its record still being fetched
send(dver(3, T0 + 0.5, true)); send(dok(3));                               // a dynamic miss: per_miss, so not weighed
send(dver(4, T0 + 0.75)); send(dok(4, { dyn_digest_ok: null, record_saved: false, dyn_check: { digest_ok: null, replayed: false } }));
send(dver(5, T0 + 1)); send(dok(5, { dyn_digest_ok: false, dyn_check: { digest_ok: false, replayed: false, error: "record's challenge differs from the one sent: nonce" } }));
run(); out.rounds = snapAt();
// forty rounds later (ids run ahead: static and dynamic rounds share them), the record check of #2 alerts
let t = T0 + 1;
for (let i = 6; i < 46; i++) { t += 0.25; send(ver(i, t)); send(ok(i)); }
send({ type: "alert", id: 2, t, kind: "dynamic", detail: "replaying inference to the recorded stage does not reproduce the timed digest" });
run(); out.late = snapAt();
send({ type: "alert", id: 45, oldest_id: 2, ids: [2], t: t + 1, kind: "check", detail: "checking is falling behind: the record check of #2 has waited 130 s" });
run(); out.lag = snapAt();
for (let i = 46; i < 50; i++) { t += 20; send(ver(i, t)); send(ok(i)); }      // over a minute of the verifier's clock, no new alert
run(); out.expired = snapAt();
// the record cap is reached: saving stops (one evidence alert, meta evidence_saving false)
send({ type: "alert", id: 49, t, kind: "evidence", detail: "records over --evidence-max-total 20G: later records are checked, not kept" });
send({ type: "meta", t, evidence_saving: false });
run(); out.evidence = snapAt();
try { els["claim-chains"]._h.click({ target: { closest: () => ({ dataset: { chain: "dynamic" } }) } }); } catch (e) { errors.push(String(e.stack || e)); }
run(); out.dynsel = snapAt();
// the run stops long after: #2's record never checked
send({ type: "meta", t: t + 300, state: "stopped" });
send({ type: "unchecked", t: t + 300, static_ids: [], dynamic_ids: [2], detail: "stopped with 1 round(s) unchecked (0 digest checks, 1 record checks; first #2): each counts as a failed check" });
send({ type: "alert", id: 49, ids: [2], oldest_id: 2, t: t + 300, kind: "check", detail: "stopped with 1 round(s) unchecked (0 digest checks, 1 record checks; first #2): each counts as a failed check" });
run(); click(2); run(); out.stopped = snapAt();
// research mode: static figures from static rounds only, dynamic rounds apart (a failed replay is no good digest)
send({ type: "snapshot", meta: { ...meta0, mode: "research", claims: {}, claim: null }, events: [] });
for (let i = 1; i < 5; i++) { send(ver(i, T0 + i)); send(ok(i)); }
send(dver(5, T0 + 5)); send(dok(5));
send(dver(6, T0 + 6)); send(dok(6, { dyn_digest_ok: false, dyn_check: { digest_ok: false, replayed: true } }));
send(dver(7, T0 + 7, true)); send(dok(7, { dyn_digest_ok: null, dyn_check: { digest_ok: null, replayed: false } }));
run(); out.research = snapAt();
// no --dynamic: every static round carries the memory claim with a dynamic region of 0
const smem = { dynamic_bytes: 0, dynamic_region_bytes: 0, live_region_bytes: 0, unchecked_bytes: 16 * G - mem.static_bytes, impossible: false, over: false };
send({ type: "snapshot", meta: { ...meta0, dynamic: false, claims: {}, claim: null }, events: [ver(1, T0, { memory: smem }), ok(1)] });
run(); click(1); run(); out.static_mem = snapAt();
// a stream-decode bundle: Θ' not credited, only the filler (here a quarter of the array) at 62/64
const bmem = { ...mem, static_bytes: G * 62 / 64, static_credited_bytes: G * 62 / 64 };
send({ type: "snapshot", meta: { ...meta0, dynamic: false, claims: {}, claim: null, memory: bmem, manifest: "m.json" }, events: [ver(1, T0, { memory: smem }), ok(1)] });
run(); out.bundle_mem = snapAt();
console.log(JSON.stringify({ errors, out }));
"""


SCENARIO_FLAGS = r"""
const out = {};
const get = id => (els[id] ? (els[id].innerHTML || els[id].textContent) : "");
const snapAt = () => ({ status: els.status.textContent, banner: els.banner.className, rows: els.rows.innerHTML, alerts: get("alerts"),
                        n_alerts: els["c-alerts"].textContent, chains: get("claim-chains"), meta: els.meta.innerHTML, statement: get("claim-statement") });
const base = { median_ns: 1e5, sigma_core_ns: 1e3, deadline_ns: 2e5, deadline_rule: "sigma", sigmas: 6, margin: 0.1, arm_max_ns: 1e6 };
const misses = { window: 100, k: 7, miss_rate: 0.01, alpha: 0.001 };
const wm = { chain: "static", tests: ["window", "misses"], mode: "window", F: 0.2, C: 0.999, alpha: 0.001, L: 8, delta_c_us: 17, mu_b_us: 100, sigma_b_us: 2,
             delta_F_us: 27.2, tau_us: 8, t_us: 1600, M_us: 136, n_window: 200, n_cal: 1000, clip_us: 236, challenge_flags: "warn", misses };
const ver = (id, extra) => ({ type: "verification", round: "static", id, t: id, phase: "run", within: true, rtt_ns: 1e5, deadline_ns: 2e5, late_ready: false,
                              timing_miss: false, gross: [], challenges: [], ...extra });
const ok = id => ({ type: "check", id, digest_ok: true, challenges: [] });
send({ type: "snapshot", meta: { mode: "monitor", dynamic: false, baseline: base, state: "running", challenge_flags: "warn", claims: { static: wm },
                                 allowed: { strategies: ["honest"] } }, events: [] });
for (let i = 1; i < 10; i++) { send(ver(i)); send(ok(i)); }
// a lost reply: counted at the clip, a flag; a gross round trip: a flag; neither turns anything red
send(ver(10, { within: false, faults: ["no reply to a static challenge"], faults_flagged: true, flagged: ["no reply to a static challenge"], timing_miss: true })); send(ok(10));
send({ type: "flag", id: 10, t: 10, kind: "missing", chain: "static", outside_claim: true, detail: "no reply to a static challenge" });
send(ver(11, { within: false, rtt_ns: 9e6, timing_miss: true, flagged: ["round trip 9000 µs, over 10200 µs"] })); send(ok(11));
send({ type: "flag", id: 11, t: 11, kind: "gross", chain: "static", outside_claim: true, detail: "round trip 9000 µs, over 10200 µs" });
send({ type: "claim_progress", chain: "static", n: 11, n_window: 200, mean_shift: 1, clipped: 2, misses: 2, misses_n: 11, misses_window: 100, misses_k: 7 });
send(ver(15, { within: false, late_ready: true, late_ready_flagged: true, ready_clipped: true, timing_miss: true, flagged: ["READY after 9 ms, over 3 × its bound 2.5 ms"] })); send(ok(15));
run(); out.flags = snapAt();
send({ type: "flag", id: 12, t: 12, kind: "claim", rule: "misses", chain: "static", outside_claim: true, detail: "7 of its last 100 round trips missed the deadline" });
send({ type: "claim_progress", chain: "static", n: 12, n_window: 200, mean_shift: 1, clipped: 2, misses: 0, misses_n: 0, misses_window: 100, misses_k: 7 });
run(); out.missflag = snapAt();
// under --challenge-flags alert the same gross round trip is an alert, marked as a single challenge
send({ type: "alert", id: 13, t: 13, kind: "gross", chain: "static", outside_claim: true, detail: "round trip 9000 µs, over 10200 µs" });
run(); out.alerted = snapAt();
console.log(JSON.stringify({ errors, out }));
"""


def test_page_single_challenge_flags():
    """--challenge-flags warn: a missing reply or a gross round trip on a chain under a claim is a flag (a warning
    row, listed with the alerts as "flag", not counted as one, never a red banner); a miss-count flag makes the
    chain "miss flag" (warn); an alert outside the claim (--challenge-flags alert) is marked as a single challenge."""
    r = _node_page(SCENARIO_FLAGS)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    out = r["out"]

    def row(snap, i):
        return re.search(rf'data-id="{i}">.*?</tr>', snap["rows"], re.DOTALL).group(0)

    f = out["flags"]
    assert f["banner"] == "banner ok" and str(f["n_alerts"]) == "0", (f["status"], f["n_alerts"])
    assert "v warn" in row(f, 10) and "the claim counts it at the clip" in row(f, 10), row(f, 10)
    assert "v warn" in row(f, 11) and "flag: round trip 9000" in row(f, 11) and "v fail" not in row(f, 11), row(f, 11)
    assert "v warn" in row(f, 15) and "late READY · flag: its chains count at the clip" in row(f, 15), row(f, 15)
    assert f["alerts"].count('class="v warn">flag · ') == 2 and "flag · gross" in f["alerts"], f["alerts"]
    assert "a gross miss or missing reply is a flag, counted at the clip" in f["meta"], f["meta"]
    assert "No single challenge raises an alert" in f["statement"], f["statement"]
    m = out["missflag"]
    assert m["banner"] == "banner warn" and "Many deadline misses" in m["status"] and "miss flag" in m["chains"], m[
        "status"
    ]
    a = out["alerted"]
    assert a["banner"] == "banner alert" and "gross · single challenge" in a["alerts"] and str(a["n_alerts"]) == "1", a[
        "alerts"
    ]


SCENARIO_CALIBRATION_ALERT = r"""
const out = {};
const snap = () => ({ status: els.status ? els.status.textContent : "", banner: els.banner ? els.banner.className : "" });
const T0 = 1000;
for (const [name, claim] of [["plain", null], ["claim", { chain: "static", tests: ["window"], mode: "window", F: 0.2, C: 0.999, n_window: 64 }]]) {
  send({ type: "snapshot", meta: { mode: "monitor", state: "calibrating", claim, claims: claim ? { static: claim } : {} }, events: [] });
  send({ type: "verification", id: 1, t: T0, phase: "calibration", round: "static", rtt_ns: 1e5, within: true, challenges: [] });
  run(); out[name + "_before"] = snap();
  send({ type: "alert", id: 1, t: T0, kind: "memory", detail: "7.20 GiB of the GPU unchallenged or uncredited, over the bound of 7.00 GiB" });
  run(); out[name] = snap();
}
console.log(JSON.stringify({ errors, out }));
"""


def test_page_alert_during_calibration_turns_the_banner_red():
    """An alert raised while the verifier calibrates (memory over its bound, a missing reply) turns the banner red,
    with or without a claim; before, the calibrating/waiting text hid it until it aged out."""
    r = _node_page(SCENARIO_CALIBRATION_ALERT)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    o = r["out"]
    for name in ("plain", "claim"):
        assert o[name + "_before"]["banner"] != "banner alert", o
        assert o[name]["banner"] == "banner alert" and "unchallenged" in o[name]["status"], (name, o[name])


def test_page_separate_rounds_and_late_alerts():
    """net.py publishes static and dynamic rounds as separate events, so round ids run ahead fast and checks finish
    late: an alert keeps the monitor banner red for a minute of the verifier's clock (its "t"), whatever round it
    names, and the rounds left unchecked at the stop keep it red. Before: only an alert within ten ids of the latest
    round counted. A chain whose claim fills too slowly (per_miss) weighs no miss; a dynamic row waits for its record,
    not a digest; a record bound but not kept (cap), a binding failure's reason, evidence saving stopped; research
    figures keep dynamic rounds apart; the memory panel shows the raw static array against its credit."""
    r = _node_page(SCENARIO_ROUNDS)
    if r is None:
        skip("node not found")
    assert not r["errors"], r["errors"]
    out = r["out"]

    def row(snap, i):
        return re.search(rf'data-id="{i}">.*?</tr>', snap["rows"], re.DOTALL).group(0)

    def cells(snap, i):
        return [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row(snap, i), re.DOTALL)]

    rd = out["rounds"]
    c2 = cells(rd, 2)
    # no static digest in a dynamic round
    assert c2[2] == "pass · checking record" and c2[6] == "—" and c2[9].startswith("fetching record"), c2
    assert "v fail" in row(rd, 3) and "weighed" not in row(rd, 3), row(rd, 3)  # per_miss: a failure, not weighed
    c4 = cells(rd, 4)
    assert (
        c4[2] == "pass · record bound, not kept (cap)" and "record not kept (cap)" in c4[9] and "stored" not in c4[9]
    ), c4
    assert "record&#39;s challenge differs from the one sent: nonce" in row(rd, 5) and "no record" not in row(rd, 5), (
        row(rd, 5)
    )
    assert (
        "weighed by the timing claim (static; a gross miss alerts at once); dynamic, live: alerts at once (the claim on dynamic fills too slowly to weigh them)"
        in rd["meta"]
    ), rd["meta"]
    dyn_row = re.search(r'<tr class="row[^"]*" data-chain="dynamic">.*?</tr>', rd["chains"]).group(0)
    assert "every miss alerts" in dyn_row, dyn_row
    late = out["late"]  # 43 ids late, at once: red
    assert (
        late["banner"] == "banner alert"
        and late["status"] == "Alert: replaying inference to the recorded stage does not reproduce the timed digest"
    ), late["status"]
    lag = out["lag"]
    assert (
        lag["banner"] == "banner alert"
        and lag["status"].startswith("Alert: checking is falling behind")
        and "(+1 more in the last 60 s)" in lag["status"]
    ), lag["status"]
    assert "(from #2)" in lag["alerts"], lag["alerts"]
    ex = out["expired"]
    assert ex["banner"] == "banner ok", ex["status"]
    ev = out["evidence"]
    assert "saving stopped: records over --evidence-max-total 20G" in ev["meta"] and "quiet-warn" in ev["meta"], ev[
        "meta"
    ]
    assert ev["banner"] == "banner alert", ev["status"]
    ds = out["dynsel"]
    assert ds["statement"].startswith("Dynamic chain. Every miss alerts: its period takes about 60 min"), ds[
        "statement"
    ]
    st = out["stopped"]  # the stop's report: red for good, the row failed
    assert st["banner"] == "banner alert" and st["status"].startswith(
        "Run stopped · last state: Alert: stopped with 1 round(s) unchecked"
    ), st["status"]
    c2 = cells(st, 2)
    assert (
        c2[2] == "not checked (the run stopped first)" and c2[9].startswith("not checked") and "v fail" in row(st, 2)
    ), c2
    assert (
        "not checked:</span> the run stopped before this round's record check" in st["rows"]
        and "pending" not in st["rows"]
    ), st["rows"]
    rs = out["research"]
    srow = re.search(r"<tr><td>.*?honest.*?</tr>", rs["summary"], re.DOTALL).group(0)
    nums = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r'<td class="num[^"]*">(.*?)</td>', srow, re.DOTALL)]
    assert nums[0] == "4" and nums[1] == "4/4" and nums[2] == "4/4", nums  # the static rounds only
    # dynamic rounds: 3, one over its deadline, one replay failed, one stored
    assert nums[5:] == ["3", "2/3", "1/2 · 1 stored"], nums
    sm = out["static_mem"]
    assert (
        "Dynamic region" not in sm["memnote"]
        and "Static array 4.00 GiB, 3.88 GiB credited; 128 MiB challenged but not credited" in sm["memnote"]
    ), sm["memnote"]
    assert "two bits of every encoded word" in sm["memnote"] and "Θ'" not in sm["memnote"], sm["memnote"]
    assert "static, credited (of a 4.00 GiB array)" in sm["memlegend"] and "weights + filler" not in sm["memlegend"], (
        sm["memlegend"]
    )
    assert "memory: " in sm["rows"] and "dynamic region" not in sm["rows"], sm["rows"]
    bm = out["bundle_mem"]
    assert "the bundle's Θ' (its encoded weights) is not credited at all" in bm["memnote"], bm["memnote"]


def _live_dynamic_run():
    """A real live run, as the reviewer reproduced it: the CPU toy prover with a dynamic region, monitor mode,
    records replayed online (--replay toy). After calibration every dynamic reply's digest is one bit off (the
    record checks fail) and the record of each of those rounds is held by a TCP proxy until LAG more rounds have been
    sent, so the record checks fall behind: their alerts name rounds long past, and the stop (given 0.2 s for the
    checks still waiting, not STOP_CHECKS_S) leaves some unchecked. Counted in rounds, not seconds, so that holds
    on a host of any speed: with every record held 0.35 s and 8 s at the stop, a slow host's checks kept up (alerts
    a few rounds late, nothing left unchecked) and the test failed there. The hub's snapshot (meta and events), as
    the page receives it; None without torch (the toy model)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import test_net as T

    from memory_accounting.core import wire
    from memory_accounting.verifier import net

    hubs, init, stop_s = [], server.EventHub.__init__, net.STOP_CHECKS_S

    def keep(self, *a, **k):
        init(self, *a, **k)
        hubs.append(self)

    server.EventHub.__init__ = keep
    net.STOP_CHECKS_S = 0.2
    W, C, LAG = 1, 8, 12
    seen = {"n": 0, "last": None, "rounds": 0}
    flipped, sent = set(), threading.Condition()

    def flip(way, data):
        if way == "down" and len(data) == wire.CH.size:
            seen["last"] = wire.CH.unpack(data)[0]
            if seen["last"] in (wire.MAGIC, wire.DYN_MAGIC):  # one per round (--repeat 1)
                with sent:
                    seen["rounds"] += 1
                    sent.notify_all()
        elif way == "up" and len(data) == wire.RE.size and seen["last"] == wire.DYN_MAGIC:
            seen["n"] += 1
            if seen["n"] > W + C:  # past the calibration's dynamic rounds (one reply each)
                nonce, lo, hi = wire.RE.unpack(data)
                flipped.add(nonce)
                return ("rewrite", wire.RE.pack(nonce, lo ^ 1, hi))
        return [0]

    def hold(req):  # b"EVID <nonce>\n"; other records go at once
        if int(req.split()[1], 16) in flipped:
            with sent:
                n = seen["rounds"]
                sent.wait_for(lambda: seen["rounds"] >= n + LAG, timeout=60)

    import contextlib
    import io

    try:
        prover, relay = T._dyn_prover(), T._free_port()
        threading.Thread(target=T._rewriting_relay, args=(relay, prover, flip), daemon=True).start()
        T._tcp_proxy(relay, prover, hold)
        ui = socket.socket()
        ui.bind(("127.0.0.1", 0))
        ui_port = ui.getsockname()[1]
        ui.close()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
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
                    "--mode",
                    "monitor",
                    "--replay",
                    "toy",
                    "--warmup",
                    str(W),
                    "--calibrate",
                    str(C),
                    "--interval",
                    "0.05",
                    "--reply-timeout",
                    "1.0",
                    "--max-verifications",
                    "80",
                    "--ui-port",
                    str(ui_port),
                    "--out",
                    str(pathlib.Path(tmp) / "live.jsonl"),
                ]
            )
    finally:
        server.EventHub.__init__ = init
        net.STOP_CHECKS_S = stop_s
    events, meta = hubs[-1].snapshot()
    return json.loads(server.dumps({"meta": meta, "events": events}))


SCENARIO_LIVE = r"""
// the run as the page saw it live: the snapshot of a running verifier, then every event as it came
const out = [];
send({ type: "snapshot", meta: { ...SNAP.meta, state: "running" }, events: [] });
run();
let latest = 0;
for (const ev of SNAP.events) {
  if (ev.type === "meta" && ev.state === "stopped") continue;                 // the stop comes last, below
  send(ev); run();
  if (ev.type === "verification") latest = Math.max(latest, ev.id);
  if (ev.type === "alert") out.push({ id: ev.id, kind: ev.kind, latest, banner: els.banner.className, status: els.status.textContent });
}
send({ type: "meta", state: "stopped" }); run();
const rows = {};
for (const m of els.rows.innerHTML.matchAll(/data-id="(\d+)">(.*?)<\/tr>/gs)) rows[m[1]] = m[2].replace(/<[^>]+>/g, " ");
console.log(JSON.stringify({ errors, out, rows, final: { banner: els.banner.className, status: els.status.textContent } }));
"""


def test_page_live_run_late_alerts_and_unchecked():
    """A real live run (see _live_dynamic_run) fed to the page event by event: every failed record check alerts
    many rounds after its own (the reviewer's run: 33 replay alerts 19 to 54 ids late, the banner never red), and
    each turns the banner red; the stop's unchecked report keeps it red and marks those rows "not checked" (before:
    "All recent verifications passing" and "pass · checking digest")."""
    if not shutil.which("node"):
        skip("node not found")
    snap = _live_dynamic_run()
    if snap is None:
        skip("no torch")
    unchecked = [e for e in snap["events"] if e.get("type") == "unchecked"]
    assert unchecked and unchecked[0]["dynamic_ids"], [e for e in snap["events"] if e.get("type") == "alert"][-3:]
    r = _node_page("const SNAP = " + json.dumps(snap) + ";\n" + SCENARIO_LIVE)
    assert not r["errors"], r["errors"]
    replay = [a for a in r["out"] if a["kind"] == "dynamic"]
    assert replay and max(a["latest"] - a["id"] for a in replay) >= 10, r["out"]  # the alerts name rounds long past
    assert all(a["banner"] == "banner alert" for a in r["out"]), [a for a in r["out"] if a["banner"] != "banner alert"][
        :3
    ]
    f = r["final"]
    assert f["banner"] == "banner alert" and f["status"].startswith("Run stopped · last state: Alert: stopped with"), f
    for i in unchecked[0]["dynamic_ids"] + unchecked[0]["static_ids"]:
        if str(i) in r["rows"]:
            row = r["rows"][str(i)]  # "not checked", or a more specific failure the round already had (a late READY)
            assert (
                ("not checked" in row or "late READY" in row or "gross" in row)
                and "checking" not in row
                and " pass" not in row
            ), row


def test_fonts_served_locally():
    """The page loads nothing from another host: its fonts come from this server, under /fonts/, behind the
    same Host and token checks as the page; only the fixed names are served."""
    html = PAGE.read_text()
    assert not re.search(r"""(?:src|href)\s*=\s*["']?https?://|url\(\s*["']?https?://|@import""", html), (
        "external resource in the page"
    )
    for name in re.findall(r'url\("/fonts/([^"]+)"\)', html):
        assert name in server.FONTS, name
    hub = EventHub()
    tok = "font-token_0123"
    port = server.start(hub, 0, "127.0.0.1", token=tok).server_address[1]
    headers = {"Host": f"127.0.0.1:{port}"}
    code, hdrs, body = _req(port, "GET", "/fonts/plex-sans.ttf", {**headers, "Authorization": "Bearer " + tok})
    assert (
        code == 200
        and hdrs["Content-Type"] == "font/ttf"
        and body[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO")
        and len(body) > 10000
    ), (code, hdrs)
    assert _req(port, "GET", "/fonts/plex-sans.ttf", headers)[0] == 401
    assert (
        _req(port, "GET", "/fonts/plex-sans.ttf", {"Host": "evil.example", "Authorization": "Bearer " + tok})[0] == 403
    )
    for bad in ("/fonts/../server.py", "/fonts/IBMPlexSerif-Bold.ttf", "/fonts/", "/fonts/%2e%2e/server.py"):
        assert _req(port, "GET", bad, {**headers, "Authorization": "Bearer " + tok})[0] == 404, bad


def test_posture_requests_bounded_and_token_alphabet():
    """A burst of posture POSTs keeps only the latest request for the live loop; a token may hold / and =
    (a query string reads + as a space, so + is refused at start)."""
    hub = EventHub()
    hub.allowed["strategies"] = ["honest", "evict", "tamper"]
    port = _start(hub)
    for i in range(50):
        _post(port, "/posture", {"posture": ["honest", "evict", "tamper"][i % 3]}, {"Content-Type": "application/json"})
    assert hub.requests == [{"posture": "evict"}], hub.requests  # 49 % 3 = 1
    tok = "a/b=c-d_e.f~g"
    port = server.start(hub, 0, "127.0.0.1", token=tok).server_address[1]
    assert _req(port, "GET", "/history?token=" + tok, {"Host": "localhost"})[0] == 200
    try:
        server.start(hub, 0, "127.0.0.1", token="with+plus-token")
    except ValueError as e:
        assert "+" in str(e)
    else:
        raise AssertionError("a token with + accepted")


def _req(port: int, method: str, path: str, headers: dict, body: bytes | None = None):
    """One request with exactly these headers (Host included): (status, headers, body). The connection is
    closed after the headers for /events, which never ends."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        c.request(method, path, body=body, headers=headers)
        r = c.getresponse()
        data = b"" if path.startswith("/events") and r.status == 200 else r.read()
        return r.status, dict(r.getheaders()), data
    finally:
        c.close()


def test_host_allowlist():
    """DNS rebinding: a page of the attacker's whose name resolves to this machine sends its own name as Host
    and as Origin. Every endpoint, GET included, answers only the loopback names, the bind address and the
    names given; the port in Host does not matter (an ssh -L tunnel may use another local port)."""
    assert (
        server.hostname("[::1]:8080") == "::1"
        and server.hostname("LocalHost.:9") == "localhost"
        and server.hostname("::1") == "::1"
    )
    assert server.allowlist("0.0.0.0") == {"localhost", "127.0.0.1", "::1"} and "10.0.0.5" in server.allowlist(
        "10.0.0.5"
    )
    assert {"verifier.lan", "192.168.1.9"} <= server.allowlist("::", ["Verifier.LAN:8765", "192.168.1.9"])
    hub = EventHub()
    hub.allowed["strategies"] = ["honest", "evict"]
    port = server.start(hub, 0, "127.0.0.1", allowed_hosts=["verifier.lan"]).server_address[1]
    j = {"Content-Type": "application/json"}
    for host in (f"127.0.0.1:{port}", "localhost:9999", "localhost", f"[::1]:{port}", f"verifier.lan:{port}"):
        for path in ("/", "/history", "/events"):
            assert _req(port, "GET", path, {"Host": host})[0] == 200, (host, path)
        code = _req(
            port,
            "POST",
            "/posture",
            {**j, "Host": host, "Origin": f"http://{host}"},
            json.dumps({"posture": "evict"}).encode(),
        )[0]
        assert code == 202, (host, code)
    hub.requests.clear()
    for host in (
        "evil.example",
        f"evil.example:{port}",
        "127.0.0.1.evil.example",
        "localhost.evil.example",
        "10.0.0.5",
        "",
    ):
        for path in ("/", "/index.html", "/history", "/events", "/nope"):
            code, _, _body = _req(port, "GET", path, {"Host": host})
            assert code == 403, (host, path, code)
        code = _req(
            port,
            "POST",
            "/posture",
            {**j, "Host": host, "Origin": f"http://{host}"},
            json.dumps({"posture": "evict"}).encode(),
        )[0]
        assert code == 403, (host, code)  # the review's case: Host and Origin both evil.example
        code = _req(port, "POST", "/control", {**j, "Host": host}, json.dumps({"strategies": ["evict"]}).encode())[0]
        assert code == 403, (host, code)
    assert hub.requests == [] and "strategies" not in hub.control, (hub.requests, hub.control)
    assert b"--ui-allow-host evil.example" in _req(port, "GET", "/history", {"Host": "evil.example"})[2]
    s = socket.create_connection(("127.0.0.1", port), timeout=5)  # HTTP/1.0 without a Host header
    s.sendall(b"GET /history HTTP/1.0\r\n\r\n")
    resp = b""
    while chunk := s.recv(65536):
        resp += chunk
    s.close()
    assert resp.startswith(b"HTTP/1.0 403"), resp[:80]
    deadline = time.time() + 10  # only the five accepted /events subscribed
    while hub.subscribers and time.time() < deadline:
        hub.publish({"type": "stats"})
        time.sleep(0.05)
    assert hub.subscribers == 0, hub.subscribers


def test_token():
    """With a token (for --ui-bind 0.0.0.0), every request needs it: /?token= once sets a cookie and
    redirects to the bare page, or a Bearer header, or ?token= on a GET. The Host check comes first."""
    hub = EventHub()
    hub.allowed["strategies"] = ["honest", "evict"]
    tok = "s3cret-token_0123"
    port = server.start(hub, 0, "127.0.0.1", token=tok).server_address[1]
    headers = {"Host": f"localhost:{port}"}
    j = {**headers, "Content-Type": "application/json", "Origin": f"http://localhost:{port}"}
    body = json.dumps({"posture": "evict"}).encode()
    for path in ("/", "/history", "/events", "/?token=wrong", "/history?token=" + tok[:-1], "/nope"):
        assert _req(port, "GET", path, headers)[0] == 401, path
    assert _req(port, "GET", "/history", {**headers, "Authorization": "Bearer nope"})[0] == 401
    # another server's cookie
    assert _req(port, "GET", "/history", {**headers, "Cookie": f"posm_ui_token_{port + 1}={tok}"})[0] == 401
    assert _req(port, "POST", "/posture", j, body)[0] == 401
    assert _req(port, "POST", "/posture", j, body + b"")[0] == 401 and hub.requests == []
    code, hdrs, _ = _req(port, "GET", "/?token=" + tok, headers)
    assert code == 303 and hdrs["Location"] == "/", (code, hdrs)
    cookie = hdrs["Set-Cookie"]
    assert cookie.startswith(f"posm_ui_token_{port}={tok};") and "HttpOnly" in cookie and "SameSite=Strict" in cookie, (
        cookie
    )
    jar = {"Cookie": f"other=1; {cookie.split(';')[0]}"}
    for path in ("/", "/history", "/events"):
        assert _req(port, "GET", path, {**headers, **jar})[0] == 200, path
    assert _req(port, "GET", "/events?token=" + tok, headers)[0] == 200
    assert _req(port, "GET", "/history", {**headers, "Authorization": "Bearer " + tok})[0] == 200
    assert _req(port, "POST", "/posture", {**j, **jar}, body)[0] == 202
    assert _req(port, "POST", "/posture", {**j, "Authorization": "Bearer " + tok}, body)[0] == 202
    assert _req(port, "GET", "/history?token=" + tok, {"Host": "evil.example"})[0] == 403
    assert hub.requests == [{"posture": "evict"}]
    for bad in ("short", "has space in it", "semi;colon-token", "plus+sign-token"):
        try:
            server.start(hub, 0, "127.0.0.1", token=bad)
        except ValueError:
            continue
        raise AssertionError(f"token {bad!r} accepted")


if __name__ == "__main__":
    from _runner import run

    run(globals())
