"""Prover strategies. Each returns (digest_lo, digest_hi, index_trace, elapsed_ns) for a challenge.

honest      : reads the challenged block from the GPU's own encoded array (mode 0)
other_gpu   : holds only the canonical copy; gathers permuted positions at challenge time and,
              under VDE, re-encodes each word for the challenged GPU's seed (mode 1)
encode_otf  : holds the raw (compressible) weights in place; VDE-encodes at challenge time (mode 3)
dram_copy   : holds the encoded array in host DRAM; per link copies the block to the GPU
dram_mapped : (CUDA only) GPU kernel reads the encoded array straight from pinned host DRAM
cpu         : holds the encoded array in host DRAM; whole chain (incl. matmul) on the CPU
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..core.backend import CpuLib, default_threads, to_u64_tensor
from ..core.memory import Hbm
from ..core.params import (
    MAX_LINKS,
    MODE_ENCODE_OTF,
    MODE_HONEST,
    MODE_OTHER_GPU,
    P_FIELDS,
    Challenge,
    Keys,
    pack_params,
    table_seed,
)

_HOST_SYNC = None
_NATIVE_WAIT = False  # switched only at diagnostic workload boundaries
STOP_BIT = np.uint64(1 << 63)  # POSM_STOP_BIT: the trace entry where a share cheat's chain stopped


def _host_sync():
    """The pre-armed chain's host barriers (`posm_flag_store_release`, `posm_fence_acquire` in the C library),
    loaded once. The host writes the parameters, then the flag, into host-mapped memory the GPU polls, and
    reads the digests after the GPU's done flag: on x86 those stores and loads keep their order anyway, on a
    weakly ordered host (aarch64) only with a system-scope barrier between them (the GPU side: posm.cu)."""
    global _HOST_SYNC
    if _HOST_SYNC is None:
        _HOST_SYNC = CpuLib().lib
    return _HOST_SYNC


class Prover:
    name = "base"
    armable = False  # can pre-launch its chain at ARM time (DeviceProver.arm); the responder arms every one that can

    def __init__(self, hbm: Hbm, keys: Keys):
        self.hbm, self.keys = hbm, keys

    def respond(self, ch: Challenge):
        raise NotImplementedError

    def close(self):
        pass


class DeviceProver(Prover):
    """Runs the chain kernel on the GPU.
    kernel="tensor"     : per link a grid of blocks runs the int8 tensor-core matmul and hashes with the
                          per-chain power tables (default).
    kernel="multi"      : same launch structure with the scalar matmul (correctness comparison).
    kernel="persistent" : one block loops over all links inside a single launch (comparison).
    graph=True          : the chain is captured once into a CUDA graph and replayed (default).
    Every strategy that runs this one chain (over its own arrays or segment table) is armable: honest and
    the cheats are then timed alike, none paying a launch and a sync inside the timed window."""

    mode = MODE_HONEST
    armable = True

    def __init__(
        self,
        hbm: Hbm,
        keys: Keys,
        kernel: str | None = None,
        groups: int | None = None,
        threads: int = 256,
        graph: bool | None = None,
    ):
        super().__init__(hbm, keys)
        self.dev = hbm.backend.device
        self.kernel = kernel or "tensor"
        self.groups = groups or 2 * torch.cuda.get_device_properties(torch.device(self.dev)).multi_processor_count
        self.threads = threads
        self.graph = True if graph is None else graph
        self.mem, self.canon = self._arrays()
        self.out = torch.zeros(2 + MAX_LINKS, dtype=torch.int64, device=self.dev)
        self.state = torch.zeros(8, dtype=torch.int64, device=self.dev)
        # resized per g (_ensure_buffers)
        self.partials = torch.zeros(max(self.groups, 1), dtype=torch.int64, device=self.dev)
        self.prm = torch.zeros(len(P_FIELDS), dtype=torch.int64, device=self.dev)  # rewritten in place per challenge
        # pinned staging for the parameter upload and the digest readback: pageable copies go through
        # driver staging and page pinning, whose latency jitters by ~100 µs when other processes churn host memory
        self.prm_host = torch.zeros(len(P_FIELDS), dtype=torch.int64, pin_memory=True)
        self.out_host = torch.zeros(2 + MAX_LINKS, dtype=torch.int64, pin_memory=True)
        self.pw = None
        self.mbuf = None
        self.ablk = None
        self.max_groups = self.groups

    def _arrays(self):
        return self.hbm.theta_prime, self.hbm.theta_prime

    def _ensure_buffers(self, ch: Challenge):
        """Per-chain scratch sized for this challenge shape; a new size means a new CUDA graph. The partial
        sums are one per grid block and entry group: the kernels refuse fewer than groups * g words, which
        is what a fixed buffer of `groups` words gave every g > 1."""
        need = max(self.groups, self.max_groups, 1) * ch.g
        if self.partials.numel() != need:
            self.partials = torch.zeros(need, dtype=torch.int64, device=self.dev)
        if self.pw is None or self.pw.numel() != ch.pw_count:
            self.pw = self.hbm.backend.power_buffer(ch.pw_count)
        if self.mbuf is None or self.mbuf.numel() != max(ch.mbuf_bytes, 256):
            self.mbuf = self.hbm.backend.matrix_buffer(ch.mbuf_bytes)
        if self.ablk is None or self.ablk.numel() != ch.block_bytes:
            self.ablk = self.hbm.backend.block_buffer(ch.block_bytes)

    def _launch(self, mem, prm_np, ch: Challenge, n: int):
        """prm_np: packed parameter block (numpy int64). Copies it into the fixed device buffer and runs."""
        backend = self.hbm.backend
        be = ch.be
        self.prm_host.copy_(torch.from_numpy(prm_np))
        self.prm.copy_(self.prm_host, non_blocking=True)  # on the current stream; the side stream waits on it
        canon = self.canon if mem is self.mem else mem
        if self.kernel == "persistent":
            backend.chain(
                mem, canon, self.out, self.prm, self.state, self.pw, self.mbuf, ch.k, ch.r, max(self.threads, 1024)
            )
        elif self.kernel == "tensor":
            # the extension picks the (row tiles x column tiles) decomposition per shape and trims the grid
            backend.chain_multi(
                mem,
                canon,
                self.out,
                self.prm,
                self.state,
                self.partials,
                self.pw,
                self.mbuf,
                self.max_groups,
                self.threads,
                n,
                ch.k,
                ch.r,
                be,
                ch.g,
                self.hbm.cfg.x > 0,
                tensor=ch.r > 0,
                graph=self.graph,
                ablk=self.ablk,
            )
        else:
            backend.chain_multi(
                mem,
                canon,
                self.out,
                self.prm,
                self.state,
                self.partials,
                self.pw,
                self.mbuf,
                self.groups,
                self.threads,
                n,
                ch.k,
                ch.r,
                be,
                ch.g,
                self.hbm.cfg.x > 0,
                tensor=False,
                graph=self.graph,
            )

    uses_segments = False  # honest over a resident (scattered) array reads through the segment table

    seg_override = None  # (nseg, table tensor): read through this table instead (the partial@f postures)
    share = None  # a ShareCheat: answer as that cheat on a share of the array (the <cheat>@f postures)

    def _segments(self) -> tuple[int, int]:
        """(nseg, segment table pointer) the kernel reads through; (0, 0) means one contiguous array."""
        if self.seg_override is not None:
            return self.seg_override[0], self.seg_override[1].data_ptr()
        return (self.hbm.nseg, self.hbm.segtab_ptr) if self.uses_segments and getattr(self.hbm, "nseg", 0) else (0, 0)

    def _pack(self, ch: Challenge, **kw) -> np.ndarray:
        """This prover's parameter block for a challenge (kw: a continuation's d0, link0, n, j0)."""
        nseg, segtab = self._segments()
        if self.share is not None and self.kernel == "persistent":
            raise RuntimeError("a share cheat needs the multi-kernel chain (the persistent kernel has no early stop)")
        extra = self.share.params(ch) if self.share is not None else {}
        mode = extra.pop("mode", self.mode)
        return pack_params(self.hbm.cfg, ch, self.keys, mode, nseg=nseg, segtab=segtab, **extra, **kw)

    def _stopped(self, ch: Challenge, o: np.ndarray) -> bool:
        return self.share is not None and self.share.stops and bool((o[2 : 2 + ch.n] & STOP_BIT).any())

    def _run(self, ch: Challenge, prm: np.ndarray, n: int) -> np.ndarray:
        """One launch, not pre-armed, to its end: the output words (digest, then the trace)."""
        b = self.hbm.backend
        b.sync_stream()
        self._launch(self.mem, prm, ch, n)
        b.sync_stream()
        self.out_host[: 2 + n].copy_(self.out[: 2 + n], non_blocking=True)
        b.sync_stream()
        return self.out_host[: 2 + n].numpy().view(np.uint64).copy()

    def _go_on(self, ch: Challenge, out: np.ndarray) -> tuple[int, int, list[int]]:
        """The rest of a chain the kernel stopped (POSM_STOP_BIT in posm_core.h): the links whose block is in the
        cheated share answered by the host (ShareCheat.host_link), the runs between them on the GPU."""
        n, jstop, nb = ch.n, self.share.jstop(ch), self.hbm.num_blocks(ch.block_bytes)
        trace, link, sub = [], 0, out[2 : 2 + n]
        while True:
            stop = np.flatnonzero(sub & STOP_BIT)
            if not len(stop):
                return int(out[0]), int(out[1]), trace + sub.tolist()
            first_stop = int(stop[0])
            trace += sub[:first_stop].tolist()
            link += first_stop
            d, j = (int(out[0]), int(out[1])), int(sub[first_stop] & ~STOP_BIT)
            while jstop <= j < self.share.end * 8 // ch.block_bytes and link < n:
                d = self.share.host_link(self, ch, link, d, j)
                trace.append(j)
                link += 1
                j = d[0] % nb
            if link == n:
                return d[0], d[1], trace
            out = self._run(ch, self._pack(ch, d0=d, link0=link, n=n - link, j0=j), n - link)
            sub = out[2 : 2 + n - link]

    profile = False  # when set, last_prof holds a split of the last response's time
    _armed_shape = None
    _stale = False  # a disarm timed out: that graph may still run and write done/out; arm waits it out first

    def arm(self, shape: Challenge):
        """Pre-launch the chain for this shape (block_bytes, n, r, k, g; nonce not needed yet). The graph's
        first node waits on a host-mapped flag; fire() then supplies the parameters without any CUDA call."""
        if self.kernel == "persistent":
            raise RuntimeError("arm needs the multi-kernel chain")
        if self._armed_shape is not None:
            self.disarm()
        if self._stale:
            self._clear_stale()
        if getattr(self, "_flag", None) is None:
            self._flag = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            self._prm_src = torch.zeros(len(P_FIELDS), dtype=torch.int64, pin_memory=True)
            self._out_dst = torch.zeros(2 + MAX_LINKS, dtype=torch.int64, pin_memory=True)
            self._done = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            self._done_np = self._done.numpy()
            self._flag_np = self._flag.numpy()
            self._prm_np = self._prm_src.numpy()
        self._ensure_buffers(shape)
        self._done_np[0] = 0
        self._flag_np[0] = 0
        _host_sync().posm_fence_acquire()  # both cleared before the graph that reads them is launched
        b = self.hbm.backend
        groups = self.max_groups if self.kernel == "tensor" else self.groups
        b.chain_multi_armed(
            self.mem,
            self.canon,
            self.out,
            self.prm,
            self.state,
            self.partials,
            self.pw,
            self.mbuf,
            groups,
            self.threads,
            shape.n,
            shape.k,
            shape.r,
            shape.be,
            shape.g,
            self.hbm.cfg.x > 0,
            self.kernel == "tensor" and shape.r > 0,
            self.ablk,
            self._flag,
            self._prm_src,
            self._out_dst,
            self._done,
        )
        self._armed_shape = (shape.block_bytes, shape.n, shape.r, shape.k, shape.g)

    def armed_for(self, ch: Challenge) -> bool:
        return self._armed_shape == (ch.block_bytes, ch.n, ch.r, ch.k, ch.g)

    def fire(self, ch: Challenge):
        """Answer with the pre-armed graph: write the parameters and the flag, spin on done. No CUDA API."""
        if not self.armed_for(ch):
            raise RuntimeError("not armed for this shape")
        prm = self._pack(ch)
        sync = _host_sync()
        store, fence, flag = sync.posm_flag_store_release, sync.posm_fence_acquire, self._flag_np.ctypes.data
        t0 = time.perf_counter_ns()
        self._prm_np[:] = np.asarray(prm).view(np.int64)  # same bytes whatever the packer's dtype
        store(flag, 1)  # release: the GPU sees the parameters before the flag
        t1 = time.perf_counter_ns()
        done = self._done_np
        if _NATIVE_WAIT:
            sync.posm_wait_done(done.ctypes.data)  # waits without the GIL, then acquire fence
        else:
            while done[0] == 0:
                pass
            fence()  # acquire: the digests are read after done was seen
        dt = time.perf_counter_ns() - t0
        self._armed_shape = None
        if self.profile:
            self.last_prof = {"launch_ns": t1 - t0, "sync_ns": time.perf_counter_ns() - t1, "gpu_ns": 0}
        o = self._out_dst[: 2 + ch.n].numpy().view(np.uint64).copy()
        if self._stopped(ch, o):  # a share cheat's link is off the GPU: the host goes on
            t2 = time.perf_counter_ns()
            lo, hi, trace = self._go_on(ch, o)
            return lo, hi, trace, dt + time.perf_counter_ns() - t2
        return int(o[0]), int(o[1]), o[2:].tolist(), dt

    def disarm(self, timeout_s: float = 2.0):
        """Abort a waiting graph: release the wait node with flag 2, which marks the parameter block aborted so
        every chain kernel returns without reading memory; then wait for its done flag so it cannot be
        mistaken for the next fire's. The graph's output is not a digest."""
        if self._armed_shape is None:
            return
        _host_sync().posm_flag_store_release(self._flag_np.ctypes.data, 2)
        t0 = time.perf_counter()
        while self._done_np[0] == 0 and time.perf_counter() - t0 < timeout_s:
            time.sleep(0.0002)
        self._armed_shape = None
        if self._done_np[0] == 0:
            # the graph is still queued or running and will set done and write the output later: a next arm that
            # cleared done and fired would take that for its own answer. Not armed, and no arm until it is seen.
            self._stale = True
            raise RuntimeError(
                f"{self.name}: disarm timed out after {timeout_s:.1f} s; the aborted graph has not finished"
            )

    def _clear_stale(self, timeout_s: float = 2.0):
        """Before arming again after a timed-out disarm: wait for the old graph's done flag (it was released
        with flag 2, so it returns without reading memory once it runs)."""
        t0 = time.perf_counter()
        while self._done_np[0] == 0 and time.perf_counter() - t0 < timeout_s:
            time.sleep(0.0002)
        if self._done_np[0] == 0:
            raise RuntimeError(
                f"{self.name}: the graph of a timed-out disarm has still not finished; not arming over it"
            )
        self._stale = False

    def respond(self, ch: Challenge):
        prm = self._pack(ch)
        self._ensure_buffers(ch)
        backend = self.hbm.backend
        backend.sync_stream()
        if self.profile:
            side = backend.side_stream()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record(side)
        t0 = time.perf_counter_ns()
        self._launch(self.mem, prm, ch, ch.n)
        t1 = time.perf_counter_ns()
        if self.profile:
            ev1.record(side)
        backend.sync_stream()
        dt = time.perf_counter_ns() - t0
        if self.profile:
            ev1.synchronize()
            self.last_prof = {
                "launch_ns": t1 - t0,
                "sync_ns": time.perf_counter_ns() - t1 - (time.perf_counter_ns() - t0 - dt),
                "gpu_ns": int(ev0.elapsed_time(ev1) * 1e6),
            }
        self.out_host[: 2 + ch.n].copy_(self.out[: 2 + ch.n], non_blocking=True)
        backend.sync_stream()
        o = self.out_host[: 2 + ch.n].numpy().view(np.uint64)
        if self._stopped(ch, o):
            t2 = time.perf_counter_ns()
            lo, hi, trace = self._go_on(ch, o.copy())
            return lo, hi, trace, dt + time.perf_counter_ns() - t2
        return int(o[0]), int(o[1]), o[2:].tolist(), dt


class HonestProver(DeviceProver):
    name = "honest"
    uses_segments = True


class OtherGpuProver(DeviceProver):
    name = "other_gpu"
    mode = MODE_OTHER_GPU

    def _arrays(self):
        return self.hbm.canon, self.hbm.canon


class EncodeOtfProver(DeviceProver):
    name = "encode_otf"
    mode = MODE_ENCODE_OTF

    def _arrays(self):
        if self.hbm.raw is None:
            raise RuntimeError("encode_otf needs the raw payload on the device (synthetic Hbm only)")
        return self.hbm.raw, self.hbm.raw


class DramCopyProver(DeviceProver):
    """Data lives in host DRAM. Each link: read back the digest, pick the block, copy it up, run one link.
    Not armable: a host round trip per link is the strategy, so there is no one chain to pre-launch."""

    name = "dram_copy"
    armable = False

    def respond(self, ch: Challenge):
        backend, dev, cfg = self.hbm.backend, self.dev, self.hbm.cfg
        nb = self.hbm.num_blocks(ch.block_bytes)
        if getattr(self, "_staging", None) is None or self._staging.numel() != ch.be:
            # fixed address keeps the CUDA graph valid
            self._staging = torch.empty(ch.be, dtype=torch.int64, device=dev)
        staging = self._staging
        host = self.hbm.theta_host
        trace, digest = [], (0, 0)
        idx = ch.j0
        self._ensure_buffers(ch)
        # the seed of the first link comes from the initial digest, later ones from the chain (host-side copy of
        # seed_from_digest)
        from ..core.reference import siphash24_128

        d0 = siphash24_128(self.keys.k0, self.keys.k1, ch.nonce, 0, 0, 0)
        seed = table_seed(cfg.tseed, d0[1] % cfg.x) if cfg.x else ch.nonce
        backend.sync_stream()
        t0 = time.perf_counter_ns()
        for link in range(ch.n):
            staging.copy_(host[idx * ch.be : (idx + 1) * ch.be], non_blocking=True)
            prm = pack_params(
                cfg, ch, self.keys, MODE_HONEST, nb=1, d0=digest, link0=link, n=1, j0=0, s0=seed, qoff=idx * ch.be
            )
            self._launch(staging, prm, ch, 1)
            o = self.out[:2].cpu().numpy().view(np.uint64)  # forces the sync
            digest = (int(o[0]), int(o[1]))
            trace.append(idx)
            idx = digest[0] % nb
            seed = table_seed(cfg.tseed, digest[1] % cfg.x) if cfg.x else ch.nonce
        dt = time.perf_counter_ns() - t0
        return digest[0], digest[1], trace, dt


class DramMappedProver(DeviceProver):
    """Kernel dereferences pinned host memory over PCIe (zero-copy)."""

    name = "dram_mapped"

    def _arrays(self):
        return self.hbm.theta_host, self.hbm.theta_prime


def tail_start(words: int, fraction: float) -> int:
    """Where a share `fraction` of an array of `words` words begins when it is the array's tail: a whole number of
    1 MiB blocks, at least one (fraction 0: the half, the control's split)."""
    block_words = (1 << 20) // 8
    tail = (words // 2) if fraction == 0 else max(block_words, int(words * fraction))
    return words - tail // block_words * block_words


def host_tail_table(
    backend, segments: list[tuple[int, int]], words: int, host: torch.Tensor, fraction: float, device: str
) -> tuple[int, torch.Tensor, int]:
    """Segment table for partial residency: the array's own segments (device address, words) up to (1 - fraction)
    of it, then the tail served from the pinned host copy `host` at its device address. The tail is a whole number
    of 1 MiB blocks, at least one. fraction 0 is the control: the table split the same way (at the half), the tail
    left on the device, since the split alone costs time (one more segment to look up), host memory or not.
    Returns (nseg, table, keep), keep = the words before the tail."""
    keep = tail_start(words, fraction)
    head, rest, off = [], [], 0
    for ptr, w in segments:
        take = min(w, max(keep - off, 0))
        if take:
            head.append((ptr, take))
        if take < w:
            rest.append((ptr + 8 * take, w - take))
        off += w
    segs = head + (rest if fraction == 0 else [(backend.host_device_ptr(host) + 8 * keep, words - keep)])
    starts = np.cumsum([0] + [w for _, w in segs]).astype(np.uint64)
    return (
        len(segs),
        to_u64_tensor(np.concatenate([starts, np.array([ptr for ptr, _ in segs], dtype=np.uint64)]), device),
        keep,
    )


def host_span_table(backend, segments, words, host, start, end, device, *, use_host=True):
    """Split out a whole-block interval; point it to pinned DRAM or retain its HBM pointers for a control."""
    pieces = []
    offset = 0
    for ptr, length in segments:
        lo, hi = offset, offset + length
        for a, b in ((lo, min(hi, start)), (max(lo, start), min(hi, end)),
                     (max(lo, end), hi)):
            if a < b:
                source = backend.host_device_ptr(host) if use_host and start <= a and b <= end else ptr + 8 * (a - lo)
                pieces.append((source, b - a))
        offset = hi
    if sum(length for _, length in pieces) != words:
        raise ValueError("host interval does not match the segment layout")
    starts = np.cumsum([0] + [length for _, length in pieces]).astype(np.uint64)
    return len(pieces), to_u64_tensor(
        np.concatenate([starts, np.array([ptr for ptr, _ in pieces], dtype=np.uint64)]), device
    )


class PartialProver(DeviceProver):
    """Partial residency: all but a fraction of the array is in HBM; the rest is served from a pinned host
    copy over PCIe. Same honest read path through a segment table whose last segment points into host
    DRAM (the missing fraction is a contiguous tail; with uniform next-block indices its placement does
    not matter). Each link that lands in the missing part costs the PCIe read, the others nothing.
    fraction: build its own table over a contiguous Hbm (the sweeps); without it, the responder's table."""

    name = "partial"
    uses_segments = True

    def __init__(self, hbm: Hbm, keys: Keys, fraction: float | None = None, **kw):
        self.own, self.keep = None, None
        if fraction is not None:
            words = hbm.cfg.words
            *self.own, self.keep = host_tail_table(
                hbm.backend, [(hbm.theta_prime.data_ptr(), words)], words, hbm.theta_host, fraction, hbm.backend.device
            )
        super().__init__(hbm, keys, **kw)

    def use_fraction(self, fraction: float) -> None:
        """Read through the table for this fraction from now on (built once, then kept): one instance, its graph and
        buffers, answers for every fraction, since otherwise identical instances differ by up to about 17 µs
        (where their buffers land), which is the size of the cheat at small fractions."""
        tables = self.__dict__.setdefault("tables", {})
        if fraction not in tables:
            words = self.hbm.cfg.words
            *own, keep = host_tail_table(
                self.hbm.backend,
                [(self.hbm.theta_prime.data_ptr(), words)],
                words,
                self.hbm.theta_host,
                fraction,
                self.hbm.backend.device,
            )
            tables[fraction] = (tuple(own), keep)
        self.own, self.keep = tables[fraction]

    def host_links(self, trace: list[int], block_bytes: int) -> int:
        """How many links of a response's index trace read a block from the host tail (own table only; for the
        control, fraction 0, the links that would have)."""
        first = self.keep * 8 // block_bytes
        return sum(int(j) >= first for j in trace)

    def _arrays(self):
        # refuse at construction, not at the first challenge
        if self.own is None and getattr(self.hbm, "partial_nseg", 0) == 0:
            raise RuntimeError("partial needs the responder's partial segment table (a pinned host copy)")
        return super()._arrays()

    def _segments(self) -> tuple[int, int]:
        if self.own is not None:
            return self.own[0], self.own[1].data_ptr()
        if getattr(self.hbm, "partial_nseg", 0) == 0:
            raise RuntimeError("partial needs the responder's partial segment table (a pinned host copy)")
        return self.hbm.partial_nseg, self.hbm.partial_table.data_ptr()


class DynShareCheat:
    """Encoded dynamic memory rebuilt on demand (`memory_accounting.prover.kvenc`; the dynamic sweep's dyn_encode_otf@f): region words
    [keep, end) are not held; the prover holds their payload (`payload`, int64, end - keep words) and encodes a word when
    a chain reads it, at its region position in the encoded memory's tweak domain (mode 4, the dynamic key `key`); the
    rest is read honestly through the region's table. This measures the selected codec's
    on-demand encoding cost on a share of the dynamic region."""

    stops = False

    def __init__(self, keep: int, end: int, payload, key: int):
        if not 0 <= keep < end or payload.numel() < end - keep:
            raise ValueError("the share is words [keep, end) and its payload holds them all")
        from ..core.codec_profile import FIELD_BITS

        if FIELD_BITS == 89 and (keep % 8 or end % 8):
            raise ValueError("grouped 89-bit dynamic reconstruction needs complete 64-byte rows")
        self.keep, self.end, self.src, self.key = keep, end, payload, key

    def jstop(self, ch: Challenge) -> int:
        return 0

    def params(self, ch: Challenge) -> dict:
        from ..core.params import MODE_DYN_ENCODE_OTF

        return {
            "mode": MODE_DYN_ENCODE_OTF,
            "qcheat": self.keep,
            "qcheat_hi": self.end,
            "canonp": self.src.data_ptr() - 8 * self.keep,
            "ckey": self.key,
        }


class DynOtherGpuCheat:
    """A selected dynamic share encoded for a different GPU key, rekeyed at each challenged read."""

    stops = False

    def __init__(self, keep: int, end: int, source, source_key: int, target_key: int):
        if not 0 <= keep < end or source.numel() < end - keep:
            raise ValueError("invalid other-GPU dynamic share")
        self.keep, self.end, self.src = keep, end, source
        self.source_key, self.target_key = source_key, target_key

    def params(self, ch: Challenge) -> dict:
        from ..core.params import MODE_DYN_OTHER_GPU

        return {
            "mode": MODE_DYN_OTHER_GPU,
            "qcheat": self.keep,
            "qcheat_hi": self.end,
            "canonp": self.src.data_ptr() - 8 * self.keep,
            "ckey": self.source_key,
            "tkey": self.target_key,
        }


class ShareCheat:
    """A cheat on a share of the array, answered by the honest instance with its own graph and buffers (identical
    instances differ by up to about 17 µs, the size of a small share's whole cost). The share is the array's
    tail from word `keep` (tail_start); with uniformly random blocks its placement does not matter.
      encode_otf, other_gpu: the kernel itself reads words from `keep` on as that cheat does (rebuilt from the
                  decoded payload; gathered from the canonical copy and re-encoded), the rest honestly.
      dram_copy, cpu: the chain stops before the first link whose block is in the share, the host answers that
                  link (copies the block up and runs one link on it; runs the link on the CPU) and the GPU goes
                  on from the next. The host round trip per such link is the cheat's cost, as for the whole cheat.
    (dram_mapped on a share is the partial cheat: host_tail_table, PartialProver.)"""

    KINDS = ("encode_otf", "other_gpu", "dram_copy", "cpu")

    def __init__(self, cheat: str, hbm, fraction: float, cpu: CpuProver | None = None,
                 control: bool = False):
        if cheat not in self.KINDS:
            raise ValueError(f"no share version of {cheat!r} (one of {', '.join(self.KINDS)}; dram_mapped is partial)")
        if not 0 < fraction <= 1:
            raise ValueError("the share is in (0, 1]")
        self.cheat, self.hbm, self.fraction = cheat, hbm, fraction
        self.control = control
        self.keep, self.end = getattr(
            hbm, "share_span", (tail_start(hbm.cfg.words, fraction), hbm.cfg.words)
        )
        self.target_key = hbm.cfg.seed_t
        if cheat in ("encode_otf", "other_gpu") and hasattr(hbm, "segment_starts"):
            segment = max(i for i, start in enumerate(hbm.segment_starts[:-1]) if start <= self.keep)
            if self.end > hbm.segment_starts[segment + 1]:
                raise ValueError("the reconstructed share must fit within one encoded segment")
            self.target_key = hbm.segment_keys[segment]
        self.stops = cheat in ("dram_copy", "cpu")
        if self.stops and self.keep == 0:
            raise ValueError(f"{cheat} on the whole array is the cheat itself ({cheat}), not a share of it")
        self.src = None
        if cheat == "encode_otf":
            self.src = getattr(hbm, "raw", None)
        elif cheat == "other_gpu":
            self.src = getattr(hbm, "canon", None)
        if not self.stops and (self.src is None or self.src.numel() < self.end - self.keep):
            raise RuntimeError(
                f"{cheat} on a share needs the {'decoded payload' if cheat == 'encode_otf' else 'canonical copy'} on a GPU"
            )
        if self.stops and getattr(hbm, "theta_host", None) is None:
            raise RuntimeError(f"{cheat} on a share needs the pinned host copy")
        if cheat == "cpu" and cpu is None:
            raise RuntimeError("cpu on a share needs the CPU prover (its thread pool)")
        self.cpu = cpu
        self._staging = None

    def jstop(self, ch: Challenge) -> int:
        return (self.end if self.control else self.keep) * 8 // ch.block_bytes

    def params(self, ch: Challenge) -> dict:
        if self.stops:
            return {"jstop": self.jstop(ch), "jstop_hi": self.end * 8 // ch.block_bytes}
        return {
            "mode": MODE_ENCODE_OTF if self.cheat == "encode_otf" else MODE_OTHER_GPU,
            "qcheat": self.keep,
            "canonp": self.src.data_ptr() - 8 * self.keep,
            "qcheat_hi": self.keep if self.control else self.end,
            "tkey": self.target_key,
        }

    def host_link(
        self, prover: DeviceProver, ch: Challenge, link: int, digest: tuple[int, int], j: int
    ) -> tuple[int, int]:
        """Link `link` of the chain, on block j, from the host copy: the digest after it."""
        cfg, be, host = prover.hbm.cfg, ch.be, prover.hbm.theta_host
        local = j * be - self.keep
        if self.cheat == "cpu":
            out = np.zeros(3, dtype=np.uint64)
            self.cpu.lib.chain(
                prover.hbm.theta_host_np[local : local + be],
                pack_params(cfg, ch, prover.keys, MODE_HONEST, nb=1, d0=digest, link0=link, n=1, j0=0, qoff=j * be),
                out,
                mt=True,
            )
            return int(out[0]), int(out[1])
        if self._staging is None or self._staging.numel() != be:
            self._staging = torch.empty(
                be, dtype=torch.int64, device=prover.dev
            )  # fixed address keeps the CUDA graph valid
        self._staging.copy_(host[local : local + be], non_blocking=True)
        seed = table_seed(cfg.tseed, digest[1] % cfg.x) if cfg.x else ch.nonce
        prm = pack_params(
            cfg, ch, prover.keys, MODE_HONEST, nb=1, d0=digest, link0=link, n=1, j0=0, s0=seed, qoff=j * be
        )
        prover._launch(self._staging, prm, ch, 1)
        o = prover.out[:2].cpu().numpy().view(np.uint64)  # forces the sync
        return int(o[0]), int(o[1])


class LoraProver(DeviceProver):
    """Modified weights: the prover holds a copy of the array in HBM in which a random fraction of the
    blocks differ from the commitment (an adapter merged into those weights). Timing is honest; a link
    that lands on a modified block yields a wrong digest, so this is caught by the digest check, with
    pass probability (1 - f)^(n * m) per verification."""

    name = "lora"
    uses_segments = False

    def _arrays(self):
        if getattr(self.hbm, "lora", None) is None:
            raise RuntimeError("lora needs the responder's modified copy")
        return self.hbm.lora, self.hbm.lora


class EvictProver(DeviceProver):
    """Reclaimed memory, answered from HBM as it is: the prover has overwritten a fraction of the filler with
    its own data (the responder does it when this posture is set) and runs the honest chain over what is
    there. A link that lands in the overwritten part yields a wrong digest: pass probability (1 - f)^n."""

    name = "evict"
    uses_segments = True

    def _arrays(self):
        if not hasattr(self.hbm, "set_posture"):  # without it this would answer as honest under a cheat's name
            raise RuntimeError(f"{self.name} acts only as a posture of the responder's resident array")
        return super()._arrays()


class EvictDramProver(DeviceProver):
    """Reclaimed memory, answered from host DRAM: the overwritten part's filler was kept in pinned host
    memory and the chain reads it from there over PCIe (a mapped segment in the table). Correct digests;
    each link that lands there costs a PCIe read, so it shows as drift over verifications."""

    name = "evict_dram"
    uses_segments = True

    def _arrays(self):
        if getattr(self.hbm, "evict_nseg", 0) == 0:
            raise RuntimeError("evict_dram needs the responder's eviction table (start the prover with it offered)")
        return super()._arrays()

    def _segments(self) -> tuple[int, int]:
        if getattr(self.hbm, "evict_nseg", 0) == 0:
            raise RuntimeError("evict_dram needs the responder's eviction table (start the prover with it offered)")
        return self.hbm.evict_nseg, self.hbm.evict_table.data_ptr()


class TamperProver(DeviceProver):
    """Modified weights, in place: a fraction of the weight blocks re-encoded with every 32-bit element's
    lowest bit flipped (an adapter merged into those weights), the rest untouched, no extra memory.
    Inference runs on the modified weights. The static chain sees it only when a link lands on a modified
    block (rare: weights are a small part of the challenged memory); the dynamic replay sees it in every
    record, because the KV cache came from other weights than the verifier's."""

    name = "tamper"
    uses_segments = True

    def _arrays(self):
        if not hasattr(self.hbm, "set_posture"):
            raise RuntimeError(f"{self.name} acts only as a posture of the responder's resident array")
        return super()._arrays()


class DynDramProver(HonestProver):
    """Free dynamic memory answered from host DRAM (posture only): the filler of a fraction of the dynamic
    region's free memory (the top of the vLLM pool's block ids, the top slots of a static cache) is kept in
    pinned host memory and the dynamic chain reads it from there over PCIe whenever it is free at the frozen
    stage (`memory_accounting.prover.dynamic.DramEvict`, switched by the dynamic responder's posture); the HBM it frees is
    the cheater's. Correct digests; each dynamic link that lands there costs a PCIe read. The static array is
    untouched, so static chains are answered as honest (this class)."""

    name = "dyn_dram"

    def _arrays(self):
        if not hasattr(self.hbm, "set_posture"):  # the dynamic responder makes the cheat; posture only
            raise RuntimeError(f"{self.name} acts only as a posture of the responder's resident array")
        return super()._arrays()


class CpuProver(Prover):
    """Whole chain on the CPU over the host copy, rows fanned out over a spinning thread pool."""

    name = "cpu"

    def __init__(self, hbm: Hbm, keys: Keys, threads: int | None = None):
        super().__init__(hbm, keys)
        self.lib = CpuLib()
        # Leave two logical cores free: with every core spinning, any other runnable thread evicts a
        # worker and the per-link barrier then waits a scheduler timeslice (multi-ms stalls).
        self.nthreads = self.lib.pool_start(threads or default_threads())
        self.out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)

    def respond(self, ch: Challenge):
        prm = pack_params(self.hbm.cfg, ch, self.keys, MODE_HONEST)
        t0 = time.perf_counter_ns()
        self.lib.chain(self.hbm.theta_host_np, prm, self.out, mt=True)
        dt = time.perf_counter_ns() - t0
        return int(self.out[0]), int(self.out[1]), self.out[2 : 2 + ch.n].tolist(), dt

    def close(self):
        self.lib.pool_stop()


PROVERS = {
    c.name: c
    for c in (
        HonestProver,
        OtherGpuProver,
        EncodeOtfProver,
        DramCopyProver,
        DramMappedProver,
        CpuProver,
        PartialProver,
        LoraProver,
        EvictProver,
        EvictDramProver,
        TamperProver,
        DynDramProver,
    )
}
# cheats that change what is in HBM (or, dyn_dram, where the dynamic chain reads it from): they act only as the
# prover's posture, which makes the change (and undoes it)
POSTURE_ONLY = ("evict", "evict_dram", "tamper", "dyn_dram")
CHEATS = ["other_gpu", "encode_otf", "dram_copy", "cpu"]
