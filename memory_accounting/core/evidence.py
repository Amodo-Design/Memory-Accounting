"""The record of a dynamic challenge: the stage of inference the region was frozen at, the region's
layout, and the challenge and digest it answered. The prover keeps recent records by nonce and serves
them over a TCP side channel after the timed reply (never inside the timed window); the verifier
fetches them and checks each by replaying inference to the stage, rebuilding the region and
recomputing the chain (`memory_accounting.verifier.consistency`).

The region's bytes are not part of a record: the verifier recomputes them. A prover started with
`--evidence-bytes` attaches them anyway, for debugging a mismatch (which segment differs); the verdict
never uses them.

Wire format (TCP, one request per connection):
  request : b"EVID " + 16 hex digits of the nonce + b"\\n"
  reply   : magic b"EVID", u64 header length, u64 data length, header JSON, data (uint64 words, little endian, usually empty)
            an error is a header {"error": ...} with no data
Standard library plus numpy only: the verifier box needs no torch for this.
"""

from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"EVID"
HEAD = struct.Struct("<4sQQ")
FORMAT = 2  # 2: no region bytes by default (verifier replays); 1 was v0.2 (bytes always)


@dataclass
class Evidence:
    nonce: int
    challenge: dict  # block_bytes, n, r, k, g, j0 (nonce is the key)
    digest: list  # [lo, hi] the prover returned
    words: int  # words of the region (padded to whole blocks)
    block_bytes: int
    pad_seed: int
    layout: list  # [{name, dtype, shape, words}] in array order
    stage: dict  # StageTracker.stage() or the vLLM stage
    data: np.ndarray | None = field(default=None, repr=False)  # uint64[words], debug only
    # which model produced the region: {"kind": "hf", "name": ...} or {"kind": "toy", ...}
    model: dict = field(default_factory=dict)
    # prover-side facts for testing: how it froze; with bytes, whether the region changed during the challenge
    debug: dict = field(default_factory=dict)
    region: str = "main"  # "main": all dynamic memory; "live": the live region of the same freeze (second chain)
    software_version: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    format: int = FORMAT

    def header(self) -> dict:
        d = asdict(self)
        d.pop("data")
        return d

    # ---- files: <dir>/<nonce>.json (header) + <dir>/<nonce>.npy (data)
    def save(self, out_dir, nonce: int | None = None, max_debug: int | None = None) -> Path:
        """<dir>/<nonce>.json, plus <nonce>.npy when the record carries debug bytes. nonce: the name to save
        under (the verifier passes the challenge's, never the record's own claim). max_debug: the most bytes of
        the prover's `debug` facts kept (as JSON); a larger `debug` is replaced by a note (the verdict never
        reads it, and a prover could pad it). Compact JSON: indented, a vLLM live record (5,000 layout entries)
        took 1.6 times the bytes (0.55 MB, 0.35 MB compact), which a verifier keeping every record of a long run pays."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{int(self.nonce if nonce is None else nonce):016x}"
        if self.data is not None:
            np.save(stem.with_suffix(".npy"), self.data.view(np.int64), allow_pickle=False)
        h = self.header()
        if max_debug is not None and len(json.dumps(h.get("debug"))) > max_debug:
            h["debug"] = {"dropped": f"the prover's debug facts were over {max_debug} bytes"}
        stem.with_suffix(".json").write_text(json.dumps(h, separators=(",", ":")))
        return stem.with_suffix(".json")

    @classmethod
    def load(cls, path) -> Evidence:
        path = Path(path)
        if path.suffix == ".npy":
            path = path.with_suffix(".json")
        d = json.loads(path.read_text())
        npy = path.with_suffix(".npy")
        data = np.load(npy, allow_pickle=False).view(np.uint64) if npy.exists() else None
        return cls(data=data, **d)

    # ---- wire
    def to_wire(self) -> bytes:
        h = json.dumps(self.header()).encode()
        body = np.ascontiguousarray(self.data.view(np.uint64)).astype("<u8").tobytes() if self.data is not None else b""
        return HEAD.pack(MAGIC, len(h), len(body)) + h + body

    @classmethod
    def from_wire(cls, h: bytes, body: bytes) -> Evidence:
        d = json.loads(h.decode())
        if "error" in d:
            raise LookupError(d["error"])
        return cls(data=np.frombuffer(body, dtype="<u8").astype(np.uint64) if body else None, **d)

    def segment(self, name: str, data: np.ndarray | None = None) -> np.ndarray:
        """The words of one named segment of a region array laid out as this record says."""
        data = self.data if data is None else data
        off = 0
        for e in self.layout:
            if e["name"] == name:
                return data[off : off + e["words"]]
            off += e["words"]
        raise KeyError(name)

    def describe(self) -> str:
        stage = self.stage
        if stage.get("kind") == "vllm":
            reqs = stage.get("requests", [])
            where = (
                f"layer {stage['act_layer']}/{stage['n_layers']} in flight"
                if stage.get("in_flight")
                else ("step boundary" if reqs else "idle")
            )
            return f"vLLM step {stage.get('step')}: {len(reqs)} live requests, {sum(r['num_tokens'] for r in reqs)} tokens in KV, {where}"
        if stage.get("step", -1) < 0:
            return f"req {stage.get('request_id')} before first step"
        where = (
            f"layer {stage['act_layer']}/{stage['n_layers']} in flight"
            if stage.get("in_flight")
            else f"step complete, act of layer {stage['act_layer']}"
        )
        return f"req {stage['request_id']} step {stage['step']} ({sum(len(c[0]) if c else 0 for c in stage['chunks'])} tokens fed) {where}"


class LazyEvidence:
    """A record whose fields are fixed but not yet assembled: `build()` makes the Evidence when it is first
    fetched. The prover's responder answers the next ARM as soon as the last reply has gone; assembling a
    vLLM live record there (about 2,000 layout entries) delayed that READY by several ms."""

    def __init__(self, nonce: int, build):
        self.nonce, self._build, self.data = nonce, build, None

    def build(self) -> Evidence:
        return self._build()


LAYOUT_ENTRY_BYTES = 120  # one layout entry {name, dtype, shape, words} in memory (about 80 as JSON)
LAZY_RECORD_BYTES = 1 << 20  # a record not yet assembled: a vLLM live record is about 0.2 to 0.7 MB built


class EvidenceStore:
    """Recent evidence by nonce (bounded, thread-safe)."""

    def __init__(self, keep: int = 100_000, max_bytes: int = 1 << 30, fetched_ttl: float = 60.0):
        """Recent records, the oldest dropped first beyond `keep` records or about `max_bytes` (a verifier
        that fetches after its timing loop needs two records per verification; records with debug bytes
        are large). A record the verifier has fetched is dropped `fetched_ttl` s later: a prover serving for
        days keeps only what has not been fetched yet (and the bounds above cap that)."""
        self._d: OrderedDict[int, Evidence] = OrderedDict()
        self._size: dict[int, int] = {}
        self._fetched: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.keep, self.max_bytes, self.fetched_ttl = keep, max_bytes, fetched_ttl
        self.stored = 0
        self.bytes = 0
        self.dropped = 0
        self.expired = 0

    def stats(self) -> dict:
        """A small snapshot for run diagnostics; bytes is an approximate retained size."""
        with self._lock:
            return {"retained_records": len(self._d), "retained_bytes_estimate": self.bytes,
                    "stored": self.stored, "dropped": self.dropped, "expired": self.expired}

    @staticmethod
    def _approx_bytes(ev) -> int:
        """About what a record holds in memory: its layout entries (a vLLM live record has one per request,
        layer and KV block, about 2,000 for 16 requests), its stage's tokens and block ids, debug bytes. A
        record not yet assembled (`LazyEvidence`) counts as LAZY_RECORD_BYTES until it is."""
        if isinstance(ev, LazyEvidence):
            return LAZY_RECORD_BYTES
        n = 4096
        data = getattr(ev, "data", None)
        n += data.nbytes if data is not None else 0
        lay = getattr(ev, "layout", None)
        n += LAYOUT_ENTRY_BYTES * len(lay) if isinstance(lay, list) else 0
        st = getattr(ev, "stage", None)
        if isinstance(st, dict):
            try:
                for r in st.get("requests") or ():
                    n += 8 * (len(r.get("tokens") or ()) + len(r.get("block_ids") or ()))
                for c in st.get("chunks") or ():
                    n += 8 * (len(c[0]) if c and hasattr(c[0], "__len__") else 1)
            except (AttributeError, TypeError, IndexError):
                pass
        return n

    def _drop(self, nonce: int) -> None:
        if self._d.pop(nonce, None) is not None:
            self.bytes -= self._size.pop(nonce, 0)
        self._fetched.pop(nonce, None)

    def _expire(self) -> None:
        cut = time.monotonic() - self.fetched_ttl
        while self._fetched:
            nonce, t = next(iter(self._fetched.items()))
            if t > cut:
                break
            self._drop(nonce)
            self.expired += 1

    def put(self, ev: Evidence) -> None:
        with self._cv:
            if ev.nonce in self._d:
                self.bytes -= self._size.pop(ev.nonce, 0)
            self._d[ev.nonce] = ev
            self.stored += 1
            self._size[ev.nonce] = b = self._approx_bytes(ev)
            self.bytes += b
            self._expire()
            while len(self._d) > 1 and (len(self._d) > self.keep or self.bytes > self.max_bytes):
                k, _ = self._d.popitem(last=False)
                self.bytes -= self._size.pop(k, 0)
                self.dropped += 1
                self._fetched.pop(k, None)
            self._cv.notify_all()

    def fetched(self, nonce: int) -> None:
        """The verifier has this record: drop it after `fetched_ttl` s (a fetch that failed half-way can ask again meanwhile)."""
        with self._cv:
            if nonce in self._d:
                self._fetched[nonce] = time.monotonic()
                self._fetched.move_to_end(nonce)
            self._expire()

    def get(self, nonce: int, timeout: float = 0.0) -> Evidence | None:
        """The record for a nonce; with a timeout, wait for it: the prover stores a record just after the
        timed reply, so a verifier that asks at once can be a moment early."""
        with self._cv:
            if timeout > 0:
                self._cv.wait_for(lambda: nonce in self._d, timeout)
            ev = self._d.get(nonce)
        if isinstance(ev, LazyEvidence):  # built here, in the fetching thread, off the prover's timed path
            built = ev.build()
            with self._cv:
                if self._d.get(nonce) is ev:
                    self._d[nonce] = built
                    b = self._approx_bytes(built)
                    self.bytes += b - self._size.get(nonce, 0)
                    self._size[nonce] = b
            ev = built
        return ev

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


# The largest record header a verifier reads. A vLLM live record carries one layout entry per (request, layer, KV
# block), about 80 bytes of JSON each, plus its tokens and block ids: the demo's live records hold about 2,000 entries
# (design log v0.6.1), and a synthetic record of 16 requests, 2,830 tokens over 28 layers (5,180 entries) is 0.41 MB.
# The main record and static-cache records are a few kB. 4 MiB is ten times that; a verifier of larger live batches
# raises it (memory_accounting.verifier.net --evidence-max-header).
MAX_HEADER = 4 << 20


def _recv_exact(sock: socket.socket, n: int, deadline: float | None = None) -> bytes:
    """n bytes, by `deadline` (time.monotonic()) if given: a socket timeout bounds one recv, not the whole
    read, and a prover that sends a byte a second would otherwise keep the reader for as long as it likes."""
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"record not received in time ({len(buf)} of {n} bytes)")
            sock.settimeout(left)
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("connection closed early")
        buf += chunk
    return bytes(buf)


class EvidenceServer(threading.Thread):
    """Serves the store over TCP; runs beside the UDP responder on the same port number."""

    def __init__(self, store: EvidenceStore, bind: str = "0.0.0.0", port: int = 5555):
        super().__init__(name="posm-evidence", daemon=True)
        self.store = store
        store_ = store

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                line = self.rfile.readline(64).strip()
                try:
                    if not line.startswith(b"EVID "):
                        raise ValueError("bad request")
                    nonce = int(line[5:], 16)
                    ev = store_.get(nonce, timeout=5.0)
                    if ev is None:
                        raise LookupError(f"no evidence for nonce {nonce:016x}")
                    self.wfile.write(ev.to_wire())
                    self.wfile.flush()
                    store_.fetched(nonce)
                except (ValueError, LookupError) as e:
                    h = json.dumps({"error": str(e)}).encode()
                    self.wfile.write(HEAD.pack(MAGIC, len(h), 0) + h)

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.server = socketserver.ThreadingTCPServer((bind, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    def run(self):
        self.server.serve_forever(poll_interval=0.2)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def fetch_evidence(
    host: str, port: int, nonce: int, timeout: float = 30.0, max_header: int = MAX_HEADER, max_data: int = 256 << 20
) -> Evidence:
    """The record for this nonce. timeout bounds the whole fetch (connect to last byte; TimeoutError past
    it); the lengths come from the prover, so a header over max_header is refused (ValueError) and debug
    bytes over max_data are not read: the record is returned without them (the verdict never uses them)."""
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(b"EVID %016x\n" % nonce)
        magic, hlen, dlen = HEAD.unpack(_recv_exact(s, HEAD.size, deadline))
        if magic != MAGIC:
            raise ValueError("bad evidence reply")
        if hlen > max_header:
            raise ValueError(f"record header of {hlen} bytes, over the {max_header}-byte limit")
        h = _recv_exact(s, hlen, deadline)
        body = _recv_exact(s, dlen, deadline) if 0 < dlen <= max_data else b""
    ev = Evidence.from_wire(h, body)
    if dlen > max_data:
        ev.debug = dict(
            ev.debug or {}, bytes_dropped=f"{dlen} bytes of region data over the {max_data}-byte limit, not read"
        )
    return ev
