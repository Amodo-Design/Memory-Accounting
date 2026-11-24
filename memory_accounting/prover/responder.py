"""In-process challenge responder: a thread that answers PoSM challenges over a resident Θ'.

Runs inside the inference process (a vLLM worker, or `memory_accounting.prover.serve`), so the challenged array is
the live Θ' that inference decodes from. Honest strategy only; the cheat simulators live in
`memory_accounting.prover.strategies` for the experiments. Wire format is that of `memory_accounting.core.wire`.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import torch

from ..core.backend import DeviceBackend, to_u64_tensor
from ..core.params import Challenge, Keys, MemoryConfig
from ..core.wire import (
    ARM,
    ARM_DYN_MAGIC,
    ARM_MAGIC,
    CH,
    DYN_LIVE_MAGIC,
    DYN_MAGIC,
    LIVE_FLAG,
    MAGIC,
    POSTURE,
    POSTURE_MAGIC,
    RE,
    READY,
    READY_DYN,
    STATS,
    STATS_MAGIC,
    STATSRE,
    STRATS,
    resolve_peers,
    strategy_mask,
)

PROFILE_KEEP = 500  # in-process timings kept for the counters (the last 500)
# they change HBM (or where a chain reads it), so they act only as the posture (memory_accounting.prover.strategies)
POSTURE_ONLY = (
    "evict",
    "evict_dram",
    "tamper",
    "dyn_dram",
)
# which strategies are pre-armed at ARM: every prover with `armable` set (memory_accounting.prover.strategies), i.e. every one that
# runs
# the one device chain, cheats as well as honest, so that both are timed alike; dram_copy and cpu cannot be


def watch_gc(threshold_s: float = 0.02, freeze: bool = True) -> dict:
    """Observe GC pauses in the prover process. A full collection can also pause
    the responder. Collect and freeze setup objects first so later collections
    scan objects created during serving. Count every collection and log those
    longer than threshold_s; freezing does not prevent later GC pauses."""
    import gc

    if freeze:
        gc.collect()
        gc.freeze()
    st = {"collections": 0, "gen2_collections": 0, "objects_collected": 0,
          "uncollectable_objects": 0, "total_ns": 0,
          "slow": 0, "slow_ns": 0, "max_ns": 0, "t0_ns": 0}
    threshold_ns = int(threshold_s * 1e9)

    def cb(phase, info):
        if phase == "start":
            st["t0_ns"] = time.perf_counter_ns()
        elif phase == "stop" and st["t0_ns"]:
            dt_ns = time.perf_counter_ns() - st["t0_ns"]
            st["t0_ns"] = 0
            st["collections"] += 1
            st["objects_collected"] += info.get("collected", 0)
            st["uncollectable_objects"] += info.get("uncollectable", 0)
            st["total_ns"] += dt_ns
            st["max_ns"] = max(st["max_ns"], dt_ns)
            if info.get("generation") == 2:
                st["gen2_collections"] += 1
            if dt_ns > threshold_ns:
                st["slow"] += 1
                st["slow_ns"] += dt_ns
                print(
                    f"posm: garbage collection (generation {info.get('generation')}) took {dt_ns / 1e6:.0f} ms", flush=True
                )

    gc.callbacks.append(cb)
    return st


def require_free_ports(port: int, bind: str = "0.0.0.0", tcp: bool = True) -> None:
    """Stop the prover at once if another process holds its challenge port (UDP) or, with a dynamic region, the TCP
    port of the same number its records are served on: before any setup, which under vLLM takes minutes. The
    responder binds them again when it starts (`Responder.start`), which is what holds them."""
    kinds = [("UDP", socket.SOCK_DGRAM)] + ([("TCP", socket.SOCK_STREAM)] if tcp else [])
    for name, kind in kinds:
        with socket.socket(socket.AF_INET, kind) as s:
            if kind == socket.SOCK_STREAM:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the record server binds it
            try:
                s.bind((bind, port))
            except OSError as e:
                raise SystemExit(f"posm prover: {name} port {port} on {bind}: {e.strerror} (another process holds it)")


def _bound(what: str, bind: str, port: int, fn):
    """fn(), with the port named in its error."""
    try:
        return fn()
    except OSError as e:
        raise OSError(e.errno, f"posm responder: {what} port {port} on {bind}: {e.strerror}") from e


class ResidentArray:
    """The challenged array as segments (device pointer, words) plus the segment table on the device.
    Optionally builds what the cheat simulators need: a contiguous pinned host copy (DRAM cheats and
    the CPU cheat), the raw payload on the device (encode on the fly) and a canonical copy under
    another key on another GPU (the 'other GPU' cheat)."""

    def __init__(
        self,
        segments: list[tuple[int, int, torch.Tensor]],
        enc_key: int,
        device: str = "cuda",
        strategies: tuple[str, ...] = ("honest",),
        canon_device: str | None = None,
        canon_key: int = 0xC0FFEE,
        segment_keys: tuple[int, ...] | None = None,
        partial_fraction: float = 0.10,
        lora_fraction: float = 0.05,
        evict_fraction: float = 0.10,
        tamper_fraction: float = 0.10,
        share_span: tuple[int, int] | None = None,
    ):
        self.tensors = [t for _, _, t in segments]
        self.segments = [(p, w) for p, w, _ in segments]
        starts = np.cumsum([0] + [w for _, w in self.segments]).astype(np.uint64)
        self.segment_starts = [int(s) for s in starts]
        self.segment_keys = tuple(segment_keys) if segment_keys is not None else (enc_key,) * len(segments)
        if len(self.segment_keys) != len(segments):
            raise ValueError("one encoding key is required per challenged segment")
        ptrs = np.array([p for p, _ in self.segments], dtype=np.uint64)
        self.words = int(starts[-1])
        self.share_span = share_span
        if share_span is not None and not (0 <= share_span[0] < share_span[1] <= self.segment_starts[1]):
            raise ValueError("the timing share must lie entirely in the first encoded weight segment")
        self.table = to_u64_tensor(np.concatenate([starts, ptrs]), device)
        self.nseg = len(self.segments)
        self.device, self.enc_key, self.canon_key = device, enc_key, canon_key
        self.canon_device = canon_device or device
        self.cfg = MemoryConfig(
            words=self.words, encoding="vde", perm=False, seed_t=enc_key, seed_c=canon_key, image=True
        )
        self.backend = DeviceBackend(device)
        # honest reads via the table; kept for the prover API
        self.theta_prime = torch.zeros(1, dtype=torch.int64, device=device)
        self.raw = None
        self.canon = self.theta_prime
        self.theta_host = None
        self.theta_host_np = None
        self.setup_seconds = 0.0
        self.partial_nseg = 0
        self.partial_table = None
        self.partial_fraction = partial_fraction
        self.lora = None
        self.lora_fraction = lora_fraction
        self.lora_blocks: list[int] = []
        if "lora" in strategies:
            # a contiguous copy of the array with a random fraction of its 1 MiB blocks altered in place
            # (every word of a chosen block gets its low bit flipped: a different value, same size)
            self.lora = torch.empty(self.words, dtype=torch.int64, device=device)
            off = 0
            for tensor, (_, words) in zip(self.tensors, self.segments):
                self.lora[off : off + words].copy_(tensor[:words])
                off += words
            block_words = (1 << 20) // 8
            nb = self.words // block_words
            rng = np.random.default_rng(0x10BA)
            self.lora_blocks = sorted(rng.choice(nb, size=max(1, round(nb * lora_fraction)), replace=False).tolist())
            for j in self.lora_blocks:
                blk = self.lora[j * block_words : (j + 1) * block_words]
                blk.bitwise_xor_(1)
            torch.cuda.synchronize(device)
        self._setup_posture_cheats(strategies, evict_fraction, tamper_fraction)
        base = {s.split("@")[0].removesuffix("_control") for s in strategies}
        # A control uses the same source allocations and segment-table shape as its cheat.
        need_host = any(s in base for s in ("dram_mapped", "dram_copy", "cpu", "partial"))
        need_raw = any(s in base for s in ("encode_otf", "other_gpu"))
        if need_host:
            if share_span is not None:
                start, end = share_span
                self.theta_host = torch.empty(end - start, dtype=torch.int64, pin_memory=True)
                self.theta_host.copy_(self.tensors[0][start:end])
                torch.cuda.synchronize(device)
                self.theta_host_np = self.theta_host.numpy().view(np.uint64)
                if "partial" in base:
                    from .strategies import host_span_table
                    self.partial_nseg, self.partial_table = host_span_table(
                        self.backend, self.segments, self.words, self.theta_host, start, end, device
                    )
                need_host = False
        if need_host:
            # one pinned buffer, filled segment by segment (a cat + pin_memory would peak at two host copies)
            self.theta_host = torch.empty(self.words, dtype=torch.int64, pin_memory=True)
            off = 0
            for tensor, (_, words) in zip(self.tensors, self.segments):
                self.theta_host[off : off + words].copy_(tensor[:words])
                off += words
            torch.cuda.synchronize(device)
            self.theta_host_np = self.theta_host.numpy().view(np.uint64)
            if "partial" in base:
                # segment table for the partial cheat: the honest segments up to (1 - f) of the array, then the
                # tail served from the pinned host copy at its device address
                from .strategies import host_tail_table

                self.partial_nseg, self.partial_table, _ = host_tail_table(
                    self.backend, self.segments, self.words, self.theta_host, partial_fraction, device
                )
        if need_raw:
            from ..core.params import setup_params
            from ..core.codec_profile import FIELD_BITS, PACKED128, ROW_WORDS89

            if share_span is not None:
                if FIELD_BITS != 89:
                    raise ValueError("compact timing reconstruction requires 89-bit encoding")
                start, end = share_span
                if start % ROW_WORDS89 or end % ROW_WORDS89:
                    raise ValueError("compact timing share must contain complete codec rows")
                raw = torch.empty(end - start, dtype=torch.int64, device=device)
                if PACKED128:
                    self.backend.decode_at_packed128(self.tensors[0][start:end], raw, enc_key, start)
                else:
                    groups = (end - start) // 64
                    decoded = torch.empty(groups * 55, dtype=torch.int64, device=device)
                    self.backend.decode_unpack(self.tensors[0], decoded, start, groups, enc_key)
                    self.backend.pack(decoded, raw, groups)
                check = torch.empty(ROW_WORDS89, dtype=torch.int64, device=device)
                for q in (start, end - ROW_WORDS89):
                    self.backend.setup_at(raw[q-start:q-start+ROW_WORDS89], check, enc_key, q)
                    if not torch.equal(check, self.tensors[0][q:q+ROW_WORDS89]):
                        raise RuntimeError(f"compact timing reconstruction differs at word {q}")
                if "encode_otf" in base:
                    self.raw = raw
                if "other_gpu" in base:
                    canon = torch.empty_like(raw)
                    self.backend.setup_at(raw, canon, canon_key, start)
                    self.canon = canon
                self.backend.sync()
                return
            raw = torch.empty(self.words, dtype=torch.int64, device=device)
            off = 0
            for tensor, (_, words), segment_key in zip(self.tensors, self.segments, self.segment_keys):
                if FIELD_BITS == 89 and (off % 8 or words % 8):
                    raise ValueError("grouped 89-bit reconstruction requires complete 64-byte rows")
                # decode this segment at its global positions: decode kernel uses index == position, so shift via a view
                tmp = torch.empty(off + words, dtype=torch.int64, device=device)
                tmp[off : off + words] = tensor[:words]
                out = torch.empty_like(tmp)
                self.backend.decode(tmp, out, to_u64_tensor(setup_params(off + words, 1, segment_key, 0, 0, 0), device))
                raw[off : off + words] = out[off : off + words]
                if FIELD_BITS == 89:
                    for start in (off, off + words - 8):
                        check = torch.empty(8, dtype=torch.int64, device=device)
                        self.backend.setup_at(raw[start : start + 8], check, segment_key, start)
                        original = tensor[start - off : start - off + 8]
                        if not torch.equal(check, original):
                            different = torch.nonzero(check != original).flatten().tolist()
                            raise RuntimeError(
                                f"grouped 89-bit reconstruction differs at word {start} "
                                f"(segment {off}+{words}, positions {different})"
                            )
                del tmp, out
                off += words
            self.backend.sync()
            if "encode_otf" in base:
                self.raw = raw
            if "other_gpu" in base:
                canon = torch.empty_like(raw)
                self.backend.setup_at(raw, canon, canon_key, 0)
                self.backend.sync()
                if (torch.device(self.canon_device).index or 0) != (torch.device(device).index or 0):
                    self.backend.enable_peer(
                        torch.device(self.backend.device).index or 0, torch.device(self.canon_device).index
                    )
                    canon = canon.to(self.canon_device)
                    torch.cuda.synchronize(self.canon_device)
                self.canon = canon
            if self.raw is None:
                del raw

    # ---- cheats that change HBM: set up once, applied and undone with the prover's posture
    def _setup_posture_cheats(self, strategies, evict_fraction: float, tamper_fraction: float) -> None:
        """evict / evict_dram: the last `evict_fraction` of the array (in the filler) is what the prover
        reclaims for its own data; its filler is kept in pinned host memory, both for evict_dram to read
        and to put back when the posture returns to honest. tamper: `tamper_fraction` of the blocks of the
        first segment (the weights, Θ') are the ones it modifies; the originals are kept to restore them."""
        self.evict_tail = None
        self.evict_backup = None
        self.evict_nseg = 0
        self.evict_table = None
        self.evicted = False
        self.tamper_blocks: list[int] = []
        self.tamper_backup = None
        self.tampered = False
        bw = (1 << 20) // 8
        if any(x in strategies for x in ("evict", "evict_dram")):
            t, w = self.tensors[-1], self.segments[-1][1]
            e = min(w, int(self.words * evict_fraction) // bw * bw)
            if e < bw:
                raise ValueError("evict: the last segment (the filler) holds less than one block to reclaim")
            self.evict_tail = t.view(-1)[w - e : w]
            self.evict_backup = torch.empty(e, dtype=torch.int64, pin_memory=True)
            self.evict_backup.copy_(self.evict_tail)
            torch.cuda.synchronize(self.device)
            keep = self.words - e
            segs, off = [], 0
            for p, sw in self.segments:
                if off >= keep:
                    break
                take = min(sw, keep - off)
                segs.append((p, take))
                off += take
            segs.append((self.backend.host_device_ptr(self.evict_backup), e))
            st = np.cumsum([0] + [x for _, x in segs]).astype(np.uint64)
            self.evict_table = to_u64_tensor(
                np.concatenate([st, np.array([p for p, _ in segs], dtype=np.uint64)]), self.device
            )
            self.evict_nseg = len(segs)
            self.evict_words = e
        if "tamper" in strategies:
            theta, tw = self.tensors[0], self.segments[0][1]
            nb = tw // bw
            if nb < 1:
                raise ValueError("tamper: the weights segment is smaller than one block")
            rng = np.random.default_rng(0x7A3E)
            k = max(1, round(nb * tamper_fraction))
            self.tamper_blocks = sorted(rng.choice(nb, size=k, replace=False).tolist())
            self.tamper_backup = torch.empty(k * bw, dtype=torch.int64, pin_memory=True)
            flat = theta.view(-1)
            for i, j in enumerate(self.tamper_blocks):
                self.tamper_backup[i * bw : (i + 1) * bw].copy_(flat[j * bw : (j + 1) * bw])
            torch.cuda.synchronize(self.device)

    def set_posture(self, name: str | None) -> None:
        """Make HBM hold what the posture's prover would hold: reclaimed filler overwritten with the prover's
        own data (evict, evict_dram), modified weights (tamper), or everything as committed (any other)."""
        want_evict, want_tamper = name in ("evict", "evict_dram"), name == "tamper"
        if self.evict_tail is not None and want_evict != self.evicted:
            if want_evict:
                self.evict_tail.random_()  # the prover's own data now sits where the filler was
            else:
                self.evict_tail.copy_(self.evict_backup)
            torch.cuda.synchronize(self.device)
            self.evicted = want_evict
        if self.tamper_blocks and want_tamper != self.tampered:
            bw = (1 << 20) // 8
            flat = self.tensors[0].view(-1)
            if want_tamper:
                from ..core.codec_profile import GROUP_RAW, GROUP_ENC

                g = bw // GROUP_ENC
                raw = torch.empty(GROUP_RAW * g, dtype=torch.int64, device=self.device)
                payload = torch.empty(GROUP_ENC * g, dtype=torch.int64, device=self.device)
                for j in self.tamper_blocks:
                    off = j * bw  # decode, change the weights' lowest bits, re-encode in place
                    self.backend.decode_unpack(flat, raw, off, g, self.enc_key)
                    raw.view(torch.int32).bitwise_xor_(1)
                    self.backend.pack(raw, payload, g)
                    self.backend.setup_at(payload, flat[off : off + bw], self.enc_key, off)
            else:
                for i, j in enumerate(self.tamper_blocks):
                    flat[j * bw : (j + 1) * bw].copy_(self.tamper_backup[i * bw : (i + 1) * bw])
            torch.cuda.synchronize(self.device)
            self.tampered = want_tamper

    @property
    def segtab_ptr(self) -> int:
        return self.table.data_ptr()

    def num_blocks(self, block_bytes: int) -> int:
        return self.words * 8 // block_bytes


class Responder(threading.Thread):
    """Answers challenges for the requested strategies (honest by default; cheats for experiments)."""

    # {<cheat>@f: (segment table or None, ShareCheat or None)}; set per instance in __init__, never changed here
    shares: dict = {}  # noqa: RUF012 (the default of an instance built without __init__, as tests do)

    def __init__(
        self,
        resident: ResidentArray,
        keys: Keys,
        bind: str = "0.0.0.0",
        port: int = 5555,
        strategies: tuple[str, ...] = ("honest",),
        drain: bool = False,
        hold_ms: float = 100.0,
        prearm: bool = True,
        dynamic=None,
        verifier_addrs=None,
        dyn_evict_fraction: float = 0.5,
        dynamic_timing_shares: bool = False,
    ):
        """dynamic: a `memory_accounting.prover.dynamic.DynamicResponder` over the KV cache + activation region. A dynamic
        ARM freezes inference at the next layer or step boundary instead of draining the step; the challenge
        runs over the frozen region; the stage record is served over TCP on the same port number after the reply.
        verifier_addrs: the verifier's addresses (IPs or host names, a list or a comma string), resolved here
        once (`memory_accounting.verifier.net.resolve_peers`); datagrams from any other address are dropped. None: any sender.
        dyn_evict_fraction: the dyn_dram posture's share of the dynamic region's evictable memory; when the
        posture is offered its host copy is made here, once, so that switching to it is immediate."""
        super().__init__(name="posm-responder", daemon=True)
        from .strategies import PROVERS

        self.resident, self.keys, self.bind, self.port = resident, keys, bind, port
        self.dynamic = dynamic
        self.peers = resolve_peers(verifier_addrs)
        self.dropped = 0  # datagrams from addresses other than the verifier's
        if resident is None and dynamic is None:
            raise ValueError("nothing to answer: no resident array and no dynamic region")
        # dynamic-only: no static array
        self.dev = str(resident.device) if resident is not None else str(dynamic.region.device)
        self.provers = (
            {s: PROVERS[s](resident, keys) for s in strategies if "@" not in s} if resident is not None else {}
        )
        # <cheat>@f: the honest instance answering as the cheat on a share f of the array, so that honest and every
        # share are timed by one instance (identical instances differ by up to about 17 µs). partial@f reads through
        # a table whose tail is the pinned host copy; the others through memory_accounting.prover.strategies.ShareCheat
        self.shares = {}
        fr = [s for s in strategies if "@" in s]
        if fr:
            if resident is None or "honest" not in self.provers:
                raise ValueError("<cheat>@f needs the honest prover over a resident array")
            from .strategies import ShareCheat, host_tail_table, host_span_table

            for spec in fr:
                cheat, fraction = spec.split("@")[0], float(spec.split("@")[1])
                control = cheat.endswith("_control")
                cheat = cheat.removesuffix("_control")
                if cheat == "partial":
                    if getattr(resident, "theta_host", None) is None:
                        raise ValueError("partial@f needs a pinned host copy")
                    if resident.share_span is not None:
                        if control:
                            start, end = resident.share_span
                            nseg, tab = host_span_table(
                                resident.backend, resident.segments, resident.words,
                                resident.theta_host, start, end, str(resident.device), use_host=False,
                            )
                        else:
                            nseg, tab = resident.partial_nseg, resident.partial_table
                    else:
                        if control:
                            raise ValueError("partial control needs a selected share_span")
                        nseg, tab, _ = host_tail_table(
                            resident.backend,
                            resident.segments,
                            resident.words,
                            resident.theta_host,
                            fraction,
                            str(resident.device),
                        )
                    self.shares[spec] = ((nseg, tab), None)
                else:
                    if cheat == "cpu" and "cpu" not in self.provers:
                        raise ValueError("cpu@f needs the cpu strategy offered too (its thread pool)")
                    self.shares[spec] = (None, ShareCheat(cheat, resident, fraction,
                                                         cpu=self.provers.get("cpu"), control=control))
        if dynamic_timing_shares:
            if dynamic is None or len(fr) not in (5, 10):
                raise ValueError("dynamic timing needs a dynamic region and five shares, optionally with five controls")
            dynamic.prepare_timing_shares(fr)
        if "dyn_dram" in self.provers:
            if dynamic is None or getattr(dynamic, "dram_fn", None) is None:
                raise ValueError(
                    "dyn_dram needs a dynamic region with an evictable set (memory_accounting.prover.intserve, memory_accounting.prover.vllm_serve)"
                )
            t0 = time.perf_counter()
            d = dynamic.prepare_dram(dyn_evict_fraction)
            print(
                f"posm responder: dyn_dram keeps {d.words * 8 / (1 << 20):.1f} MiB of dynamic filler in {'pinned ' if d.region.cuda else ''}host memory "
                f"({dyn_evict_fraction:g} of the evictable set; made in {time.perf_counter() - t0:.1f} s)",
                flush=True,
            )
        for p in self.provers.values():  # every strategy alike: the split costs the same on each timed path
            p.profile = True
        self.prof: dict[str, deque] = {k: deque(maxlen=PROFILE_KEEP) for k in ("launch_ns", "sync_ns", "gpu_ns")}
        self._halt = threading.Event()
        self.served = 0
        # Drain window: the inference engine runs each step under `gate`; an ARM datagram makes the
        # responder take the gate (i.e. wait for the step in flight to finish), synchronise the GPU,
        # and answer READY. The verifier then sends the challenge, which runs on an idle GPU, and the
        # gate is released after the reply (or after hold_ms if no challenge arrives). Only the
        # challenge round trip is timed, so the step wait never enters the honest distribution.
        # prearm: pre-launch the honest chain at ARM time
        self.drain, self.hold_ms, self.prearm = (
            drain,
            hold_ms,
            prearm,
        )
        self.gate = threading.Lock()
        self._holding = False
        self._hold_deadline = 0.0
        self.armed = 0
        self.hold_expired = 0
        self.fired = 0
        # Posture: when set, every challenge is answered with this strategy whatever id the verifier sent.
        # This is how a real cheater behaves (one cheat, unknown to the verifier); the research harness
        # leaves it None and picks the strategy per challenge from the wire.
        self.posture: str | None = None
        # the last PROFILE_KEEP of each (the counters and the report use no more): a prover serving for days must not
        # grow them
        self.svc_ns: deque = deque(maxlen=PROFILE_KEEP)
        self.gap_ns: deque = deque(maxlen=PROFILE_KEEP)  # chain time, READY-sent -> challenge-received
        self.dt_ns: deque = deque(maxlen=PROFILE_KEEP)
        self._last_dt = 0  # the GPU part of it (launch + sync)
        self.static_served = 0
        self._ready_at = 0
        self.trace_enabled = False
        self._timing_trace: list[dict] = []
        self._timing_trace_lock = threading.Lock()

    def take_timing_trace(self) -> list[dict]:
        """Drain optional static timings recorded after replies were sent."""
        with self._timing_trace_lock:
            rows, self._timing_trace = self._timing_trace, []
        return rows

    def step(self, fn, *args, **kwargs):
        """Run one inference step under the gate (what the engine wraps its step methods in)."""
        with self.gate:
            return fn(*args, **kwargs)

    def _armed_provers(self):
        return [p for p in self.provers.values() if getattr(p, "_armed_shape", None) is not None]

    def _disarm_all(self):
        for p in self._armed_provers():  # an armed graph waits on the side stream and blocks every other launch
            try:
                p.disarm()
            except RuntimeError as e:  # timed out: the prover refuses to arm until the old graph is done
                print(f"posm responder: {e}", flush=True)

    def _release(self):
        if self._holding:
            self._holding = False
            self.gate.release()
        self._disarm_all()
        if self.dynamic is not None:
            try:
                self.dynamic.release()  # resume a paused inference thread; drop an armed dynamic graph
            except RuntimeError as e:  # a timed-out disarm of a dynamic graph: logged, the responder goes on
                print(f"posm responder: {e}", flush=True)

    def _prearm(self, eff: str, shape: Challenge | None) -> None:
        """Pre-launch the chain the effective strategy will answer with, if it can be: the timed window then
        has no CUDA call in it, for a cheat as for honest."""
        h = self.provers.get(self._pname(eff))
        if not self.prearm or shape is None or not getattr(h, "armable", False):
            return
        try:
            h.arm(shape)
        except (ValueError, RuntimeError) as e:
            print(f"posm responder: pre-arm of {eff} skipped: {e}", flush=True)

    def _pname(self, strategy: str) -> str:
        """The prover instance that answers a strategy: <cheat>@f is the honest one (set up by the posture)."""
        return "honest" if strategy in self.shares else strategy

    def offered(self) -> list[str]:
        return list(self.provers) + list(self.shares)

    def respond(self, strategy: str, ch: Challenge) -> tuple[int, int]:
        p = self.provers[self._pname(strategy)]
        if self.dynamic is not None:
            self.dynamic.disarm()  # an armed dynamic graph would block the side stream too
        if getattr(p, "_armed_shape", None) is not None and p.armed_for(ch):
            lo, hi, _, dt = p.fire(ch)
            self.fired += 1
            self._last_dt = dt
            return lo, hi
        self._disarm_all()  # every other path launches on the side stream an armed graph is waiting on
        lo, hi, _, dt = p.respond(ch)
        self._last_dt = dt  # launch + stream sync only (no prep, no readback)
        return lo, hi

    def start(self):
        """Bind the challenge port (UDP) and, with a dynamic region, start the record server on the TCP port of the
        same number, here in the caller; then serve on the thread. A port another process holds raises here and
        stops the prover. Before, the bind failed on the thread and the prover went on generating, answering
        nothing."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            _bound("UDP", self.bind, self.port, lambda: sock.bind((self.bind, self.port)))
            if self.dynamic is not None and self.dynamic.capture_evidence:
                _bound("TCP", self.bind, self.port, lambda: self.dynamic.serve_evidence(self.bind, self.port))
        except BaseException:
            sock.close()
            raise
        sock.settimeout(0.5)
        self._sock = sock
        super().start()

    def run(self):
        sock = self._sock
        d = torch.device(self.dev)
        torch.cuda.set_device(d.index if d.index is not None else torch.cuda.current_device())
        try:
            self._serve(sock, d)
        finally:  # whatever ends the loop, inference resumes
            self._release()
            sock.close()

    def _serve(self, sock, device):
        while not self._halt.is_set():
            armed = bool(self._armed_provers())
            dyn_hold = self.dynamic is not None and (self.dynamic.frozen or self.dynamic.armed)
            if self._holding or armed or dyn_hold:
                if time.perf_counter() > self._hold_deadline:
                    self._release()
                    self.hold_expired += 1
                else:
                    sock.settimeout(max(0.0005, self._hold_deadline - time.perf_counter()))
            try:
                data, addr = sock.recvfrom(512)
            except TimeoutError:
                continue
            except OSError as e:  # e.g. an ICMP error queued on the socket: not fatal
                print(f"posm responder: receive failed: {e}", flush=True)
                continue
            finally:
                if not self._holding and not armed and not dyn_hold:
                    sock.settimeout(0.5)
            try:
                self._handle(sock, data, addr, device)
            except OSError as e:
                # sendto can fail for a datagram (EHOSTUNREACH, ENETUNREACH, EPERM from a firewall): that answer is
                # lost, which the verifier sees as a missing reply. It must not end this thread: a frozen region
                # would then never be released and inference would hang
                print(f"posm responder: datagram from {addr} not answered: {e}", flush=True)
                self._release()

    def _handle(self, sock, data, addr, device):
        """One datagram from the socket: answer it, or drop it. Raises OSError when an answer cannot be sent."""
        if self.peers is not None and addr[0] not in self.peers:  # not the verifier: dropped unread
            self.dropped += 1
            return
        if len(data) == POSTURE.size:  # demo control: set the prover's posture
            try:
                magic, token, sid = POSTURE.unpack(data)
            except struct.error:
                return
            if magic == POSTURE_MAGIC:
                if 0 <= sid < len(STRATS) and STRATS[sid] in self.offered():
                    self._release()  # nothing may read HBM while it changes
                    if "honest" in self.provers:
                        h = self.provers["honest"]
                        h.seg_override, h.share = self.shares.get(STRATS[sid], (None, None))
                    if self.resident is not None and hasattr(self.resident, "set_posture"):
                        self.resident.set_posture(STRATS[sid])
                    if self.dynamic is not None and hasattr(self.dynamic, "set_posture"):
                        # dyn_dram: its table from the next freeze; else the own one
                        self.dynamic.set_posture(STRATS[sid])
                    self.posture = STRATS[sid]
                    print(f"posm responder: posture -> {self.posture}", flush=True)
                sock.sendto(POSTURE.pack(POSTURE_MAGIC, token, STRATS.index(self.posture) if self.posture else 0), addr)
            return
        if len(data) == STATS.size:  # prover-side view for the live page
            try:
                magic, token = STATS.unpack(data)
            except struct.error:
                return
            if magic == STATS_MAGIC:
                sock.sendto(self.stats_packet(token), addr)
            return
        if len(data) == ARM.size:
            try:
                magic, token, B_, n_, r_, k_, g_, sid_ = ARM.unpack(data)
            except struct.error:
                return
            if magic == ARM_DYN_MAGIC:
                if self.dynamic is None:
                    return  # not offered: the verifier times out
                self._release()  # no static hold or armed graph may overlap
                try:
                    shape = Challenge(B_, n_, r_, 0, 0, k_, g_) if self.prearm else None
                except ValueError as e:
                    print(f"posm responder: bad dynamic shape: {e}", flush=True)
                    return
                try:
                    words, bb, flags, lw = self.dynamic.arm(
                        shape, self.hold_ms / 1000, want_live=bool(sid_ & LIVE_FLAG)
                    )
                except RuntimeError as e:  # ArmFailed: unfrozen already; no READY (the verifier logs a fault)
                    print(f"posm responder: {e}", flush=True)
                    return
                self._hold_deadline = time.perf_counter() + self.hold_ms / 1000
                self.armed += 1
                sock.sendto(READY_DYN.pack(ARM_DYN_MAGIC, token, words, bb, flags, lw), addr)
                self._ready_at = time.perf_counter_ns()
                return
            if magic != ARM_MAGIC:
                return
            if self.dynamic is not None and self.drain:
                # freeze inference at its next layer boundary (or at once if no forward runs), as for a
                # dynamic challenge: the GPU is idle after a layer instead of after a whole step, so the
                # READY window (what a prover could reload before the challenge) is one layer, not one step
                try:
                    self.dynamic.freeze_static(self.hold_ms / 1000)
                except RuntimeError as e:  # ArmFailed: unfrozen already; answer undrained
                    print(f"posm responder: {e}", flush=True)
                self._hold_deadline = time.perf_counter() + self.hold_ms / 1000
            elif self.dynamic is not None:
                self.dynamic.release()  # a paused thread holds the gate: let it finish the step
            if self.drain and not self._holding and self.dynamic is None:
                self.gate.acquire()
                self._holding = True  # waits for the step in flight
                torch.cuda.synchronize(device)  # and for its kernels to finish
                self._hold_deadline = time.perf_counter() + self.hold_ms / 1000
            self._disarm_all()  # a stale armed graph would block the side stream
            eff = self.posture or (STRATS[sid_] if 0 <= sid_ < len(STRATS) else "")
            if eff in POSTURE_ONLY and self.posture != eff:
                eff = ""  # refused on the wire (below): nothing to arm
            try:
                shape = Challenge(B_, n_, r_, 0, 0, k_, g_)
            except ValueError as e:
                print(f"posm responder: bad ARM shape: {e}", flush=True)
                shape = None
            self._prearm(eff, shape)
            if not self.drain:
                self._hold_deadline = time.perf_counter() + self.hold_ms / 1000
            self.armed += 1
            sock.sendto(READY.pack(ARM_MAGIC, token), addr)
            self._ready_at = time.perf_counter_ns()
            return
        t_in = time.perf_counter_ns()
        try:
            magic, nonce, j0, B, n, r, k, g, sid = CH.unpack(data)
            if magic in (DYN_MAGIC, DYN_LIVE_MAGIC):
                if self.dynamic is None:
                    raise ValueError("dynamic challenge but no dynamic region")
                live = magic == DYN_LIVE_MAGIC
                ch = Challenge(B, n, r, nonce, j0, k, g)
                lo, hi = self.dynamic.answer(ch, live=live)
                sock.sendto(RE.pack(nonce, lo, hi), addr)
                self.served += 1
                self.dynamic.finish(ch, lo, hi, live=live)  # off the timing path: keep the record
                if not self.dynamic.live_pending:  # the live chain may still be to come: stay frozen
                    self._release()
                return
            # <cheat>@f: posture only
            if (
                magic != MAGIC
                or not 0 <= sid < len(STRATS)
                or (self.posture is None and STRATS[sid] not in self.provers)
            ):
                raise ValueError(f"bad magic or strategy {sid}")
            strat = self.posture or STRATS[sid]
            if strat in POSTURE_ONLY and self.posture != strat:
                raise ValueError(f"{strat} changes what is in HBM: it runs only as the prover's posture")
            lo, hi = self.respond(strat, Challenge(B, n, r, nonce, j0, k, g))
        except (struct.error, ValueError, RuntimeError) as e:
            print(f"posm responder: ignored datagram from {addr}: {e}", flush=True)
            self._release()
            return
        t_out = time.perf_counter_ns()
        sock.sendto(RE.pack(nonce, lo, hi), addr)
        t_sent = time.perf_counter_ns()
        self.served += 1
        self._release()
        self._keep(strat, t_in, t_out)  # after the reply: off the timed path
        if self.trace_enabled:
            row = {"nonce": f"{nonce:016x}", "service_ns": t_out - t_in,
                   "gpu_launch_sync_ns": self._last_dt, "send_call_ns": t_sent - t_out}
            with self._timing_trace_lock:
                self._timing_trace.append(row)

    def _keep(self, strat: str, t_in: int, t_out: int):
        """The counters for the live page and the report, for whichever strategy answered (honest and every cheat
        are kept alike, and only after the reply is sent, so none of this is in any round trip)."""
        self.svc_ns.append(t_out - t_in)
        self.dt_ns.append(self._last_dt)
        for k_, v_ in getattr(self.provers[self._pname(strat)], "last_prof", {}).items():
            self.prof[k_].append(v_)
        if self._ready_at:
            self.gap_ns.append(t_in - self._ready_at)
            self._ready_at = 0
        self.static_served += 1
        if self.static_served % PROFILE_KEEP == 0:
            self._report()

    def stats_packet(self, token: int) -> bytes:
        def q(a, p):
            return int(np.percentile(list(a), p)) if a else 0

        return STATSRE.pack(
            STATS_MAGIC,
            token,
            self.served,
            self.fired,
            self.armed,
            self.hold_expired,
            q(self.svc_ns, 50),
            q(self.svc_ns, 99),
            q(self.dt_ns, 50),
            q(self.dt_ns, 99),
            STRATS.index(self.posture) if self.posture else 0,
            strategy_mask(s for s in self.provers if s not in POSTURE_ONLY),
            strategy_mask(self.offered()),
        )

    def _report(self):
        def q(a, p):
            return float(np.percentile(list(a), p)) / 1e3 if a else float("nan")

        print(
            f"posm responder: {self.fired} fired pre-armed; in-process p50 {q(self.svc_ns, 50):.0f} µs p99 {q(self.svc_ns, 99):.0f} µs; "
            f"of which launch+sync p50 {q(self.dt_ns, 50):.0f} µs p99 {q(self.dt_ns, 99):.0f} µs "
            f"[prep+launch p50 {q(self.prof['launch_ns'], 50):.0f} p99 {q(self.prof['launch_ns'], 99):.0f}; "
            f"GPU events p50 {q(self.prof['gpu_ns'], 50):.0f} p99 {q(self.prof['gpu_ns'], 99):.0f}; "
            f"sync wait p50 {q(self.prof['sync_ns'], 50):.0f} p99 {q(self.prof['sync_ns'], 99):.0f}]; "
            f"READY->challenge gap p50 {q(self.gap_ns, 50):.0f} µs p99 {q(self.gap_ns, 99):.0f} µs (last 500)",
            flush=True,
        )

    def stop(self):
        self._halt.set()
        for p in self.provers.values():
            p.close()
        if self.dynamic is not None:
            self.dynamic.close()
