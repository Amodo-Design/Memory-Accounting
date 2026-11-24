"""Event hub + HTTP server for the live page. Standard library only: the verifier may sit on a LAN with
no internet, so the page carries no external dependency either (its fonts are the package's own, served
under /fonts/)."""

from __future__ import annotations

import hmac
import json
import math
import queue
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC = Path(__file__).parent
FONT_DIR = Path(__file__).parents[2] / "assets" / "fonts"
# The page's fonts, by the fixed names it asks for: a fixed map (no path from the request reaches the file
# system). IBM Plex Mono is not bundled; the page falls back to the system's monospace for it.
FONTS = {
    "plex-sans.ttf": "IBMPlexSans-VariableFont_wdth,wght.ttf",
    "plex-sans-italic.ttf": "IBMPlexSans-Italic-VariableFont_wdth,wght.ttf",
    "plex-serif-light.ttf": "IBMPlexSerif-Light.ttf",
    "plex-serif.ttf": "IBMPlexSerif-Regular.ttf",
}
MAX_BODY = 64 * 1024  # a POST from the page is a few hundred bytes
LOOPBACK = ("localhost", "127.0.0.1", "::1")  # always accepted as Host: the local and ssh-tunnel (-L) flow
# a token travels in a cookie, a header and a query string: no "+" (a query string reads it as a space)
TOKEN_RE = re.compile(r"[A-Za-z0-9._~/=-]{8,256}")


def _finite(x):
    """A copy of `x` with every non-finite float as None (JSON null)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def dumps(x) -> str:
    """Strict JSON for the browser. Python writes inf and nan as Infinity and NaN, which JSON.parse rejects
    (one such field loses the event, and in the snapshot every event): they go out as null, which the page
    shows as "—". The events themselves are not changed, so the verifier's own log keeps the values."""
    try:
        return json.dumps(x, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(x), allow_nan=False)


class EventHub:
    """Fan-out of JSON events to any number of SSE subscribers, with a bounded history for late joiners."""

    def __init__(self, history: int = 5000):
        self.history: deque[dict] = deque(maxlen=history)
        self.meta: dict = {}  # run description (host, shape, deadline...) sent first to every client
        self.control: dict = {}  # operator settings from the page (which strategies to run); the live loop reads it
        self.allowed: dict = {}  # what the page may set: {"strategies": [...]}
        self.requests: list[
            dict
        ] = []  # demo requests for the live loop (owner of the socket) to carry out: at most one
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._seq = 0

    def publish(self, ev: dict) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": time.time(), **ev}
            self.history.append(ev)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev

    def set_meta(self, **kw) -> None:
        with self._lock:
            self.meta.update(kw)
        self.publish({"type": "meta", **kw})

    def set_control(self, **kw) -> dict:
        with self._lock:
            if "strategies" in kw:
                allowed = self.allowed.get("strategies")
                kw["strategies"] = [s for s in kw["strategies"] if not allowed or s in allowed]
            self.control.update(kw)
            snap = dict(self.control)
        self.publish({"type": "control", **snap})
        return snap

    def request_posture(self, name: str) -> None:
        """Ask the live loop to switch the prover's posture. Only the latest request is kept: the loop carries
        out one a round, and a page (or a script) posting faster must not queue requests without end. The
        list is replaced in place, never emptied, so the loop's `while hub.requests: hub.requests.pop(0)`
        cannot find it empty between its test and its pop."""
        with self._lock:
            self.requests[:] = [{"posture": name}]

    def _snapshot(self) -> tuple[list[dict], dict]:
        return list(self.history), {**self.meta, "control": dict(self.control), "allowed": dict(self.allowed)}

    def snapshot(self) -> tuple[list[dict], dict]:
        """History and run description, without subscribing (for /history)."""
        with self._lock:
            return self._snapshot()

    def subscribe(self) -> tuple[queue.Queue, list[dict], dict]:
        """A queue of the events published from now on, plus the snapshot up to now (no gap, no repeat).
        The caller must unsubscribe the queue when it is done."""
        q: queue.Queue = queue.Queue(maxsize=10000)
        with self._lock:
            self._subs.append(q)
            return (q, *self._snapshot())

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)


def hostname(value: str) -> str:
    """The host part of a Host header (or an allowlist entry), lower case, without port, brackets or a
    trailing dot: "[::1]:8080" -> "::1", "Localhost.:8080" -> "localhost". A bare IPv6 address stays whole."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        v = v[1:].split("]", 1)[0]
    elif v.count(":") == 1:
        v = v.split(":", 1)[0]
    return v.rstrip(".")


def allowlist(bind: str, extra=None) -> frozenset:
    """The Host names the server answers: the loopback names, the bind address when it is one address (not
    0.0.0.0 or ::), and any extra names (the machine's LAN name or address when the page is reached over the
    network)."""
    names = set(LOOPBACK)
    if hostname(bind) not in ("", "0.0.0.0", "::"):
        names.add(hostname(bind))
    names.update(hostname(h) for h in (extra or ()) if hostname(h))
    return frozenset(names)


def _handler(
    hub: EventHub,
    allowed_hosts: frozenset = frozenset(LOOPBACK),
    token: str | None = None,
    cookie: str = "posm_ui_token",
):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _gate(self) -> bool:
            """Whether the request may go on; when not, the refusal has been sent. Every request, GET included:
            /events and /history carry the run, and a page that reads them is as bad as one that posts.
            Host must name this server: a DNS rebinding page (a name of the attacker's that resolves to this
            machine) sends its own name as Host and its own origin as Origin, so the Origin check of a POST
            alone does not stop it. Then the token, when the server has one: an Authorization: Bearer header,
            the cookie that /?token=... sets, or ?token= on a GET."""
            host = self.headers.get("Host")
            name = hostname(host or "")
            if not name or name not in allowed_hosts:
                msg = f"Host {host!r} is not a name this server answers ({', '.join(sorted(allowed_hosts))}); " + (
                    f"the verifier accepts it with --ui-allow-host {name}" if name else "a Host header is required"
                )
                self._send(403, msg.encode(), "text/plain")
                return False
            if token is None:
                return True
            given = []
            auth = self.headers.get("Authorization") or ""
            if auth[:7].lower() == "bearer ":
                given.append(auth[7:].strip())
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == cookie:
                    given.append(v.strip().strip('"'))
            if self.command == "GET":
                given += parse_qs(urlsplit(self.path).query).get("token", [])
            if any(hmac.compare_digest(g.encode(), token.encode()) for g in given):
                return True
            self._send(
                401,
                b"this page needs its token: open /?token=... once (the page keeps it in a cookie), "
                b"or send Authorization: Bearer ...",
                "text/plain",
            )
            return False

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._gate():
                return
            path = self.path.split("?")[0]
            if token is not None and path in ("/", "/index.html") and "token" in parse_qs(urlsplit(self.path).query):
                # keep the token in a cookie (the page's own fetches and EventSource send it) and take it out
                # of the address bar and the browser history
                self.send_response(303)
                self.send_header("Location", path)
                self.send_header("Set-Cookie", f"{cookie}={token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            elif path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/fonts/") and path[len("/fonts/") :] in FONTS:
                self._send(
                    200, (FONT_DIR / FONTS[path[len("/fonts/") :]]).read_bytes(), "font/ttf", "private, max-age=86400"
                )
            elif path == "/history":
                hist, meta = hub.snapshot()
                self._send(200, dumps({"meta": meta, "events": hist}).encode(), "application/json")
            elif path == "/events":
                updates, hist, meta = hub.subscribe()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    self.wfile.write(f"data: {dumps({'type': 'snapshot', 'meta': meta, 'events': hist})}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            ev = updates.get(timeout=15)
                            self.wfile.write(f"data: {dumps(ev)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except OSError:  # the client went away: noticed at the next write (an event or the keepalive)
                    self.close_connection = True
                finally:
                    hub.unsubscribe(updates)
            else:
                self._send(404, b"not found", "text/plain")

        def _refuse(self) -> tuple[int, str] | None:
            """Why a POST is refused, or None. The page's own fetch sends JSON from the page's origin. A browser
            sends Origin on every cross-origin POST (and Referer, unless told not to): when either is there, its
            host must be the Host the request came to, so another site open in the operator's browser cannot
            drive the run. A request with neither is not from a page (curl, a script on the LAN, which could
            send any header it liked anyway) and is accepted. JSON only: an HTML form cannot send it, and a
            cross-site fetch with it needs a CORS preflight, which this server never grants. Host itself has
            passed the allowlist (_gate) by then, which is what stops a DNS rebinding page."""
            host = (self.headers.get("Host") or "").strip().lower()
            for header in ("Origin", "Referer"):
                v = self.headers.get(header)
                if v is None:
                    continue
                try:
                    u = urlsplit(v.strip())
                    src = u.netloc.lower() if u.scheme in ("http", "https") else None
                except ValueError:
                    src = None
                if not host or src != host:
                    return 403, f"{header} {v!r} is not this server ({host or 'no Host header'})"
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return 415, "Content-Type must be application/json"
            return None

        def _body(self):
            """The request's JSON body, or a (code, message) refusal."""
            why = self._refuse()
            if why:
                return why
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return 400, "bad Content-Length"
            if n < 0 or n > MAX_BODY:
                return 413, "body too large"
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, UnicodeDecodeError) as e:
                return 400, str(e)

        def do_POST(self):
            if not self._gate():
                return
            path = self.path.split("?")[0]
            if path not in ("/posture", "/control"):
                return self._send(404, b"not found", "text/plain")
            body = self._body()
            if isinstance(body, tuple):
                return self._send(body[0], body[1].encode(), "text/plain")
            if path == "/posture":
                try:
                    name = body.get("posture")
                    allowed = hub.allowed.get("strategies") or []
                    if not isinstance(name, str) or (allowed and name not in allowed):
                        raise ValueError("posture must be one of " + ", ".join(allowed))
                except (ValueError, AttributeError) as e:
                    return self._send(400, str(e).encode(), "text/plain")
                hub.request_posture(name)
                return self._send(202, dumps({"requested": name}).encode(), "application/json")
            try:
                if not isinstance(body, dict):
                    raise ValueError("object expected")
                if "strategies" in body and not (
                    isinstance(body["strategies"], list) and all(isinstance(x, str) for x in body["strategies"])
                ):
                    raise ValueError("strategies: list of names")
            except ValueError as e:
                return self._send(400, str(e).encode(), "text/plain")
            snap = hub.set_control(**{k: v for k, v in body.items() if k in ("strategies",)})
            self._send(200, dumps(snap).encode(), "application/json")

    return H


def start(
    hub: EventHub, port: int = 8080, bind: str = "0.0.0.0", allowed_hosts=None, token: str | None = None
) -> ThreadingHTTPServer:
    """Serve the page on bind:port. A request is answered only when its Host names this server: the loopback
    names, `bind` when it is one address, or one of `allowed_hosts` (extra names or addresses; a port is
    ignored). With `token`, every request must also carry it (for --ui-bind 0.0.0.0, where anyone on the LAN
    reaches the port); without one, the local and ssh-tunnel flow (http://localhost:PORT/) needs nothing more."""
    if token is not None and not TOKEN_RE.fullmatch(token):
        raise ValueError(
            "the UI token must be 8 to 256 characters of A-Z a-z 0-9 . _ ~ / = - (no +: a query string reads it as a space)"
        )
    srv = ThreadingHTTPServer((bind, port), _handler(hub))
    srv.RequestHandlerClass = _handler(
        hub, allowlist(bind, allowed_hosts), token, f"posm_ui_token_{srv.server_address[1]}"
    )
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="posm-ui", daemon=True).start()
    return srv
