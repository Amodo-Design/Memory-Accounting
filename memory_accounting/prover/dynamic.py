"""Dynamic memory: the KV cache and the activation buffer as a challengeable region, a pause point at
layer boundaries, and the stage record that lets the verifier recompute the region exactly.

The static scheme challenges Θ' and the filler, whose contents the verifier knows at setup. Activations
and the KV cache change at every layer of every step, so the verifier cannot hold a reference for
them in advance. Instead (design log, "Dynamic memory"):

  1. the prover keeps the dynamic data in fixed buffers that form a *dynamic region*: the KV cache
     and an activation buffer that a layer boundary writes the residual stream into, plus a pad to whole
     blocks. Every part of the region that holds no live data holds dynamic filler (`dynfill`):
     verifier-seeded, VDE-encoded, position-bound words, restored whenever memory is freed;
  2. a challenge pauses inference at the next layer boundary or step boundary (`PauseController`),
     records the *stage* (request, token chunks fed, step, layer), runs the timed chain over the
     frozen region, replies, keeps the stage as a record keyed by the nonce (`memory_accounting.core.evidence`), and
     resumes;
  3. the verifier replays inference to the recorded stage with the weights it holds, which rebuilds
     the region bit for bit, runs the same chain over it and compares the digest (`memory_accounting.verifier.consistency`).
     There is no tolerance: the replay is exact by construction (`memory_accounting.models.intllm`, CPU or GPU) or by
     identical hardware and batch-invariant kernels (vLLM).

Works on CPU tensors (tests, laptop demo) and CUDA tensors (the responder).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..core.params import MemoryConfig

ACT_NAME = "act"
PAD_NAME = "pad"


@dataclass
class Segment:
    name: str
    dtype: str  # torch dtype name without the "torch." prefix
    shape: tuple
    words: int  # 8-byte words
    tensor: torch.Tensor  # a contiguous tensor (or byte view) whose bytes are the segment
    mask: int = (1 << 64) - 1  # filler mask for this segment's element type (keeps float filler finite)
    offset: int = 0  # first word of the segment in the region (the filler's position)
    ptr: int = 0  # device (or host) address of the first byte
    enc: object = (
        None  # an encoded store (`memory_accounting.prover.kvenc.EncStore`): its words are VDE words, empty = encoded zeros
    )

    def layout(self) -> dict:
        if self.enc is not None:
            return self.enc.layout(self.name)
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "words": self.words}


class PageRef:
    """A segment given by address instead of a tensor: nbytes at `ptr` on `device`. A vLLM live region is
    one segment per (request, layer, block), about 2,000 for a 16-request batch; making a tensor view for
    each took 15 to 18 ms before READY. `make()` builds the uint8 view only when something reads the bytes
    (debug snapshots and checksums)."""

    __slots__ = ("device", "make", "nbytes", "ptr")

    def __init__(self, ptr: int, nbytes: int, device, make):
        self.ptr, self.nbytes, self.device, self.make = ptr, nbytes, device, make

    def numel(self) -> int:  # as the uint8 view would report
        return self.nbytes


class PagedLive:
    """A live region given as arrays rather than a buffer list: `ptrs` (uint64) are the device addresses of
    equal pages of `page_words` words, in region order; `tail` are ordinary (name, tensor) buffers after
    them (the activation rows). `names()` and `make(k)` (a view of page k) are called only when the
    segment list is read, after the chains: building it per page before READY took 2 to 15 ms."""

    def __init__(self, ptrs: np.ndarray, page_words: int, names, make, tail: list, enc=None):
        """enc: the pages are units of an encoded store (`memory_accounting.prover.kvenc.EncStore` of one unit: their layout)."""
        self.ptrs, self.page_words, self.names, self.make, self.tail, self.enc = (
            ptrs,
            page_words,
            names,
            make,
            tail,
            enc,
        )


def _words_of(s: Segment) -> torch.Tensor:
    """A segment's storage as int64 words (a PageRef is turned into its view on first use)."""
    if isinstance(s.tensor, PageRef):
        s.tensor = s.tensor.make()
    return _flat_words(s.tensor)


def _flat_words(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor's storage as int64 words (a view: same bytes, same address)."""
    if not tensor.is_contiguous():
        raise ValueError("dynamic buffers must be contiguous")
    if (tensor.numel() * tensor.element_size()) % 8:
        raise ValueError(
            f"dynamic buffers must be a whole number of 8-byte words, got {tensor.numel() * tensor.element_size()} bytes"
        )
    if tensor.data_ptr() % 8:
        raise ValueError("dynamic buffers must be 8-byte aligned")
    return tensor.reshape(-1).view(torch.uint8).view(torch.int64)


def buffer_words(t) -> int:
    """Words of a region buffer: a tensor, a page by address or an encoded store."""
    if isinstance(t, PageRef):
        return t.nbytes // 8
    if hasattr(t, "unit_words"):
        return t.words
    return t.numel() * t.element_size() // 8


def pad_words(seed: int, count: int, pos0: int = 0, gpu: int = 0, live: bool = False) -> np.ndarray:
    """Dynamic filler words [pos0, pos0 + count) for this seed and GPU (the pad, or any free range; live: a
    live region's pad, which has its own stream)."""
    from .dynfill import dyn_key, filler_np

    return filler_np(count, pos0, seed, dyn_key(seed, gpu), live=live) if count else np.zeros(0, dtype=np.uint64)


def layer_modules(model: torch.nn.Module, module_re: str = r"layers\.(\d+)$") -> list[torch.nn.Module]:
    """The layer modules in index order (the modules whose forward pre-hook is a layer boundary)."""
    found = {}
    for name, mod in model.named_modules():
        m = re.search(module_re, name)
        if m:
            found[int(m.group(1))] = mod
    return [found[i] for i in sorted(found)]


class DynamicRegion:
    """Named buffers presented to the chain kernels as one segmented array, padded with seeded filler to
    a whole number of blocks so that every word is challengeable. `rebuild()` replaces the buffer list
    in place (a vLLM region changes with the live requests at every pause).

    Exposes what `memory_accounting.prover.strategies.HonestProver` reads from a resident array (`backend`, `cfg`,
    `theta_prime`, `nseg`, `segtab_ptr`, `num_blocks`), so the same armed CUDA graph answers over it."""

    def __init__(
        self,
        buffers: list[tuple[str, torch.Tensor]],
        block_bytes: int,
        pad_seed: int = 0,
        device=None,
        gpu: int = 0,
        masks: dict | None = None,
        live: bool = False,
    ):
        """pad_seed: the verifier's seed for the dynamic filler (free memory and pad); gpu: this GPU's index as
        the verifier knows it (the filler key is derived from both). masks: filler mask per segment name, for
        byte views whose element type is not the tensor's (default: from the tensor's dtype). live: a live
        region over buffers of the main region; only its pad is its own, and it has its own stream, so it is
        not the main region's filler at the same positions."""
        from .dynfill import dyn_key

        if not buffers and device is None:
            raise ValueError("a dynamic region needs a buffer or a device")
        self.device = torch.device(device) if device is not None else buffers[0][1].device
        self.cuda = self.device.type == "cuda"
        self.block_bytes, self.pad_seed, self.gpu, self.live = block_bytes, pad_seed, gpu, live
        self.dyn_key = dyn_key(pad_seed, gpu)
        self.masks = dict(masks or {})
        bw = block_bytes // 8
        self._pad = torch.zeros(bw, dtype=torch.int64, device=self.device)
        if self.cuda:
            from ..core.backend import DeviceBackend

            self.backend = DeviceBackend(str(self.device))
            self.theta_prime = torch.zeros(1, dtype=torch.int64, device=self.device)  # reads go through the table
        else:
            self.backend = None
            self.theta_prime = torch.zeros(1, dtype=torch.int64)
        self.canon = self.theta_prime
        self._host = None
        self._segments: list = []
        self._lazy = None
        self.rebuild(buffers)

    @property
    def segments(self) -> list:
        """The segment list; for a region built with `rebuild_paged`, made on first use."""
        if self._lazy is not None:
            live, tail = self._lazy
            pw = live.page_words
            pb = pw * 8
            M = (1 << 64) - 1
            names = live.names()
            segs = [
                Segment(
                    names[k],
                    "uint8",
                    (pb,),
                    pw,
                    PageRef(int(p), pb, self.device, (lambda k=k: live.make(k))),
                    M,
                    k * pw,
                    int(p),
                    enc=live.enc,
                )
                for k, p in enumerate(live.ptrs)
            ]
            self._segments, self._lazy = segs + tail, None
        return self._segments

    @segments.setter
    def segments(self, v: list) -> None:
        self._segments, self._lazy = v, None

    def _segment(self, name: str, tensor, off: int) -> Segment:
        """The segment of one buffer at region word `off`: a tensor, a page by address, or an encoded store."""
        from .dynfill import mask_for
        from .kvenc import EncStore

        M = (1 << 64) - 1
        if isinstance(tensor, EncStore):
            e = tensor.tensor
            if e is not None:
                if e.device != self.device:
                    raise ValueError("all dynamic buffers must live on one device")
                if e.dtype != torch.int64 or not e.is_contiguous() or e.numel() != tensor.words:
                    raise ValueError(f"{name}: an encoded store is {tensor.words} contiguous int64 words")
            return Segment(
                name,
                "enc",
                (tensor.units, tensor.plain),
                tensor.words,
                e,
                M,
                off,
                e.data_ptr() if e is not None else 0,
                enc=tensor,
            )
        if tensor.device != self.device:
            raise ValueError("all dynamic buffers must live on one device")
        if isinstance(tensor, PageRef):
            if tensor.nbytes % 8 or tensor.ptr % 8:
                raise ValueError(f"{name}: pages must be whole, aligned 8-byte words")
            return Segment(
                name, "uint8", (tensor.nbytes,), tensor.nbytes // 8, tensor, self.masks.get(name, M), off, tensor.ptr
            )
        fw = _flat_words(tensor)
        return Segment(
            name,
            str(tensor.dtype).replace("torch.", ""),
            tuple(tensor.shape),
            fw.numel(),
            tensor,
            self.masks.get(name, mask_for(tensor.dtype)),
            off,
            fw.data_ptr(),
        )

    def rebuild(self, buffers: list[tuple[str, torch.Tensor]]) -> None:
        from .dynfill import fill_

        bw = self.block_bytes // 8
        self.segments: list[Segment] = []
        off = 0
        for name, t in buffers:
            seg = self._segment(name, t, off)
            self.segments.append(seg)
            off += seg.words
        live = off
        total = max(bw, -(-live // bw) * bw)  # whole blocks; at least one
        if total > live:
            pad = self._pad[: total - live]
            fill_(pad, live, self.pad_seed, self.dyn_key, live=self.live)  # the pad is filler at its own positions
            self.segments.append(
                Segment(PAD_NAME, "int64", (total - live,), total - live, pad, (1 << 64) - 1, live, pad.data_ptr())
            )
        self.words = total
        starts = np.cumsum([0] + [s.words for s in self.segments]).astype(np.uint64)
        ptrs = np.array([s.ptr for s in self.segments], dtype=np.uint64)
        self._set_table(starts, ptrs)

    def rebuild_paged(self, live: PagedLive) -> None:
        """`rebuild` for a region of many equal pages: the table from arrays, the segment list lazily."""
        from .dynfill import fill_

        bw = self.block_bytes // 8
        n, pw = len(live.ptrs), live.page_words
        off, tail = n * pw, []
        for name, t in live.tail:
            seg = self._segment(name, t, off)
            tail.append(seg)
            off += seg.words
        live_words = off
        total = max(bw, -(-live_words // bw) * bw)
        if total > live_words:
            pad = self._pad[: total - live_words]
            fill_(pad, live_words, self.pad_seed, self.dyn_key, live=self.live)
            tail.append(
                Segment(
                    PAD_NAME,
                    "int64",
                    (total - live_words,),
                    total - live_words,
                    pad,
                    (1 << 64) - 1,
                    live_words,
                    pad.data_ptr(),
                )
            )
        self.words = total
        starts = np.concatenate(
            [
                np.arange(n, dtype=np.uint64) * np.uint64(pw),
                np.array([t.offset for t in tail] + [total], dtype=np.uint64),
            ]
        )
        ptrs = np.concatenate(
            [np.asarray(live.ptrs, dtype=np.uint64), np.array([t.ptr for t in tail], dtype=np.uint64)]
        )
        self._segments, self._lazy = None, (live, tail)
        self._set_table(starts, ptrs)

    def segtab(self, starts: np.ndarray, ptrs: np.ndarray) -> tuple:
        """A segment table over this region's words, on its device: (host array, table tensor, segments)."""
        if len(starts) != len(ptrs) + 1 or int(starts[-1]) != self.words:
            raise ValueError(
                f"segment table of {int(starts[-1]) if len(starts) else 0} words for a region of {self.words}"
            )
        tab = np.ascontiguousarray(
            np.concatenate([np.asarray(starts, dtype=np.uint64), np.asarray(ptrs, dtype=np.uint64)]).view(np.int64)
        )
        if self.cuda:
            from ..core.backend import to_u64_tensor

            return tab, to_u64_tensor(tab, str(self.device)), len(ptrs)
        return tab, torch.from_numpy(tab), len(ptrs)

    def _set_table(self, starts: np.ndarray, ptrs: np.ndarray) -> None:
        self._own = self.segtab(starts, ptrs)
        self._segtab_np, self.table, self.nseg = self._own
        # the chain reads the words as they are (VDE words where the region is encoded); no one-way image: the verifier
        # rebuilds the region by replay
        self.cfg = MemoryConfig(words=self.words, encoding="none", perm=False, image=False)

    def use_table(self, tab: tuple) -> None:
        """Read the region through another segment table over the same words (`segtab`), until `restore_table`:
        the dyn_dram cheat, which reads part of the free memory from host memory. The caller keeps the table
        alive while a chain may read it."""
        self._segtab_np, self.table, self.nseg = tab

    def restore_table(self) -> None:
        """Back to the region's own segment table (every word at its address)."""
        self._segtab_np, self.table, self.nseg = self._own

    @property
    def own_table(self) -> bool:
        return self.table is self._own[1]

    @property
    def segtab_ptr(self) -> int:
        return int(self.table.data_ptr())

    @property
    def nbytes(self) -> int:
        return self.words * 8

    def num_blocks(self, block_bytes: int) -> int:
        return self.nbytes // block_bytes

    def layout(self) -> list[dict]:
        return [s.layout() for s in self.segments]

    def layout_fn(self):
        """A function giving this region's layout later, valid after the region is rebuilt; for a paged
        region it writes the page entries from the names alone (no segment objects)."""
        if self._lazy is not None:
            live, tail = self._lazy
            pw = live.page_words
            tl = [s.layout() for s in tail]
            if live.enc is not None:
                return lambda: [live.enc.layout(n) for n in live.names()] + tl
            return lambda: [{"name": n, "dtype": "uint8", "shape": [pw * 8], "words": pw} for n in live.names()] + tl
        lay = self.layout()
        return lambda: lay

    def find(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    # ---- free memory holds dynamic filler
    def fill(self, name: str, lo: int = 0, hi: int | None = None) -> None:
        """Write filler into words [lo, hi) of a segment (memory just freed): computed, not copied. An encoded store's
        free words are the encoding of zeros."""
        s = self.find(name)
        hi = s.words if hi is None else hi
        self.free_(s, lo, _words_of(s)[lo:hi])

    def free_(self, s: Segment, lo: int, out: torch.Tensor) -> None:
        """What words [lo, lo + n) of segment s hold when free, into `out` (int64, n words, on any device)."""
        from .dynfill import fill_

        if s.enc is not None:
            from .kvenc import empty_

            empty_(out, s.offset + lo, self.dyn_key)
        else:
            fill_(out, s.offset + lo, self.pad_seed, self.dyn_key, s.mask)

    def free_np(self, s: Segment, lo: int, n: int) -> np.ndarray:
        """`free_` on the host (uint64): the verifier's reference."""
        from .dynfill import filler_np

        if s.enc is not None:
            from .kvenc import empty_np

            return empty_np(n, s.offset + lo, self.dyn_key)
        return filler_np(n, s.offset + lo, self.pad_seed, self.dyn_key, s.mask)

    @property
    def codec(self):
        """The encoder of this region's stores (the dynamic key, this device)."""
        if getattr(self, "_codec", None) is None:
            from .kvenc import Codec

            self._codec = Codec(self.dyn_key, self.device)
        return self._codec

    def fill_pages(self, pages: list[tuple[str, int]], pw: int) -> None:
        """Write filler into many pages at once: each (segment name, first word) gets words
        [lo, lo + pw). On CUDA one kernel launch per element mask (all of a vLLM pool's freed blocks in all
        layers), instead of one fill per page per layer under the step lock."""
        if not pages:
            return
        if not self.cuda:
            for name, lo in pages:
                self.fill(name, lo, lo + pw)
            return
        from .dynfill import _backend, stream_seed
        from .kvenc import GROUP_ENC

        by_mask: dict[int, tuple[list[int], list[int]]] = {}
        by_store: dict[str, tuple[Segment, list]] = {}
        base: dict[str, tuple[int, Segment]] = {}
        for name, lo in pages:
            if name not in base:
                seg = self.find(name)
                base[name] = (seg.ptr, seg)
            ptr, seg = base[name]
            if lo < 0 or lo + pw > seg.words:
                raise ValueError(f"page [{lo}, {lo + pw}) outside segment {name} of {seg.words} words")
            if seg.enc is not None:  # an encoded store's unit: encoded zeros, one launch per store
                if pw != seg.enc.unit_words or lo % pw:
                    raise ValueError(
                        f"{name}: a free page of an encoded store is one of its units ({seg.enc.unit_words} words)"
                    )
                by_store.setdefault(name, (seg, []))[1].append((0, lo, 0, pw // GROUP_ENC))
                continue
            d, positions = by_mask.setdefault(seg.mask, ([], []))
            d.append(ptr + 8 * lo)
            positions.append(seg.offset + lo)
        be = _backend(self.device)

        def s64(v):
            return v - (1 << 64) if v >= 1 << 63 else v

        for mask, (d, positions) in by_mask.items():
            dst = torch.tensor([s64(x) for x in d], dtype=torch.int64, device=self.device)
            pos = torch.tensor(positions, dtype=torch.int64, device=self.device)
            be.fill_pages(dst, pos, pw, stream_seed(self.pad_seed), self.dyn_key, mask)
        for seg, rows in by_store.values():
            self.codec.code(0, seg.ptr, seg.offset, self.codec.ent(rows), 0, False)

    def clear(self, name: str) -> None:
        """Restore a whole segment to filler from a cached template (a copy: cheap enough per layer)."""
        from .dynfill import template

        s = self.find(name)
        if s.enc is not None:
            from .kvenc import empty_template

            _words_of(s).copy_(empty_template(s.words, s.offset, self.dyn_key, self.device))
            return
        _words_of(s).copy_(template(s.words, s.offset, self.pad_seed, self.dyn_key, s.mask, self.device))

    def reset_to_filler(self) -> None:
        """Every segment back to filler: the region of an engine with no live data."""
        for s in self.segments:
            if s.name != PAD_NAME:
                self.clear(s.name)

    def filler_view(self) -> np.ndarray:
        """The region as it is with no live data (uint64 words, host): what the verifier starts from."""
        return np.concatenate([self._seg_filler_np(s) for s in self.segments])

    def _seg_filler_np(self, s: Segment) -> np.ndarray:
        if s.name == PAD_NAME:
            from .dynfill import filler_np

            return filler_np(s.words, s.offset, self.pad_seed, self.dyn_key, live=self.live)
        return self.free_np(s, 0, s.words)

    def checksum(self) -> tuple[int, int]:
        """Two wrapping sums over every word (debug: did the region change?). No host copy of the region."""
        a = b = 0
        for s in self.segments:
            w = _words_of(s)
            a = (a + int(w.sum())) & ((1 << 64) - 1)
            b = (b + int((w * w).sum())) & ((1 << 64) - 1)
        return a, b

    def snapshot(self) -> np.ndarray:
        """The region as the logical array the chain reads (uint64 words, a host copy)."""
        if self.cuda:
            if self._host is None or self._host.numel() < self.words:  # pinned only when a snapshot is taken (debug)
                self._host = torch.empty(self.words, dtype=torch.int64, pin_memory=True)
            off = 0
            for s in self.segments:
                self._host[off : off + s.words].copy_(_words_of(s), non_blocking=True)
                off += s.words
            torch.cuda.synchronize(self.device)
            return self._host[: self.words].numpy().view(np.uint64).copy()
        return np.concatenate([_words_of(s).numpy().view(np.uint64) for s in self.segments])


class PauseController:
    """Where the inference thread can be frozen.

    * `checkpoint()` is passed at every layer boundary and at every step boundary. When a pause has been
      `request()`ed, the thread runs `on_pause` (sync the device), parks, and `resume()` lets it go.
    * `lock` is held by the inference thread for the whole of a forward and around anything else that
      writes the region (a cache reset at a new request). A responder that finds no boundary within its
      wait (inference idle or between requests) takes the lock instead, which freezes the region too.

    `cancel()` withdraws a request atomically: it returns False if the thread has already taken the
    pause, so the responder can never wait on the lock of a paused thread."""

    def __init__(self):
        self._cv = threading.Condition()
        self._want = False
        self._paused = False
        self.pauses = 0
        self.lock = threading.Lock()
        self.held_off = 0  # times the inference side found the lock taken (a frozen region made it wait)

    def acquire_inference(self) -> None:
        """Inference side: take the lock, counting the times a frozen region made us wait."""
        if not self.lock.acquire(blocking=False):
            self.held_off += 1
            self.lock.acquire()

    def request(self) -> None:
        with self._cv:
            self._want = True

    def cancel(self) -> bool:
        with self._cv:
            if self._paused:
                return False
            self._want = False
            return True

    @property
    def paused(self) -> bool:
        with self._cv:
            return self._paused

    def checkpoint(self, on_pause=None) -> None:
        """Inference thread: pause here if a pause was requested."""
        with self._cv:
            if not self._want:
                return
            self._want = False
            if on_pause is not None:
                on_pause()
            self._paused = True
            self.pauses += 1
            self._cv.notify_all()
            while self._paused:
                self._cv.wait()

    def wait_paused(self, timeout: float) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._paused, timeout)

    def resume(self) -> None:
        with self._cv:
            self._want = False
            self._paused = False
            self._cv.notify_all()


class StageTracker:
    """What the dynamic region holds, in terms the verifier can replay: the request, every token chunk
    fed to the model so far (prefill, then one token per decode step), the current step, and where in
    the step the last layer boundary was. Convention:

      * `act_layer` L: the activation buffer holds the hidden state entering layer L at `step`;
      * `kv_layers_done` = L while the step is in flight (layers < L have written this step's K/V),
        or n_layers once the step has completed (then act_layer = n_layers - 1, the last write);
      * `step` = -1 before the first forward of a request (region freshly reset)."""

    def __init__(self, n_layers: int, act: torch.Tensor | None = None, encoded: bool = False):
        """encoded: the region holds the cache and the activation buffer encoded (`memory_accounting.prover.kvenc`); the stage says so."""
        self.n_layers, self.act, self.encoded = n_layers, act, encoded
        self.request_id = 0
        self.new_request(0)

    def new_request(self, request_id: int) -> None:
        """Call with the pause lock held, after resetting the cache and the region to filler."""
        self.request_id = request_id
        self.chunks: list = []
        self.step = -1
        self.act_layer = 0
        self.kv_layers_done = self.n_layers
        self.in_flight = False

    def on_forward(self, input_ids: torch.Tensor) -> None:
        self.chunks.append(input_ids.detach().cpu().tolist())
        self.step = len(self.chunks) - 1
        self.in_flight = True
        self.kv_layers_done = 0

    def on_layer(self, index: int) -> None:
        self.act_layer = index
        self.kv_layers_done = index

    def on_forward_end(self) -> None:
        self.in_flight = False
        self.kv_layers_done = self.n_layers
        self.act_layer = self.n_layers - 1

    def stage(self) -> dict:
        st = {
            "kind": "static-cache",
            "request_id": self.request_id,
            "chunks": [list(map(list, c)) for c in self.chunks],
            "step": self.step,
            "act_layer": self.act_layer,
            "kv_layers_done": self.kv_layers_done,
            "in_flight": self.in_flight,
            "n_layers": self.n_layers,
        }
        if self.encoded:
            st["encoded"] = True
        return st


def live_tokens(stage: dict) -> int:
    """Cache slots a static-cache stage has fed, the in-flight step's included."""
    if stage.get("step", -1) < 0:
        return 0
    return sum(len(c[0]) for c in stage["chunks"][: stage["step"] + 1])


def write_act(act: torch.Tensor, h: torch.Tensor, clear=None) -> None:
    """The activation buffer's contents after a layer boundary: the hidden state, flattened, then filler
    (`clear` restores the filler; zeros if no region is given)."""
    flat = h.reshape(-1)
    if flat.numel() > act.numel():
        raise ValueError(f"activation buffer too small: {act.numel()} < {flat.numel()} elements")
    if clear is not None:
        clear()
    else:
        act.zero_()
    act[: flat.numel()].copy_(flat.to(act.dtype))


def free_all(
    model, region: DynamicRegion, tracker: StageTracker, pause: PauseController, request_id: int, reset=None
) -> None:
    """A request ended: empty the cache, put filler back in every free word, start a new stage. Under
    the pause lock, so a challenge sees either the old state or the new one, never half of it."""
    pause.acquire_inference()
    try:
        (reset or model.reset)()
        region.reset_to_filler()
        enc = getattr(region, "encoded_cache", None)
        if enc is not None:
            enc.reset()
        tracker.new_request(request_id)
    finally:
        pause.lock.release()


class Hooks:
    """Installs the boundary hooks on a model with a static cache:

    * model pre-hook: take the pause lock (held for the whole forward), pass a step-boundary
      checkpoint (the stage there is the previous step, complete), then record the input tokens;
    * layer pre-hook: write the activation buffer, note the layer, pass a layer-boundary checkpoint;
    * model post-hook (also on an exception): close the step, release the lock.
    `on_pause` runs on the inference thread when a pause is taken."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[torch.nn.Module],
        act: torch.Tensor,
        tracker: StageTracker,
        pause: PauseController,
        on_pause=None,
        region: DynamicRegion | None = None,
        after_act=None,
    ):
        """after_act(n): the activation buffer now holds n elements, zeros after (an encoded region encodes it then;
        `region` is then not given: the buffer is plaintext scratch, zeroed before each write)."""
        self.tracker, self.pause, self.act, self.on_pause, self.after_act = tracker, pause, act, on_pause, after_act
        self.clear_act = (lambda: region.clear(ACT_NAME)) if region is not None else None
        self._handles = [
            model.register_forward_pre_hook(self._model_pre, with_kwargs=True),
            model.register_forward_hook(self._model_post, always_call=True),
        ]
        for i, mod in enumerate(layers):
            self._handles.append(mod.register_forward_pre_hook(self._layer_pre(i), with_kwargs=True))

    def _model_pre(self, module, args, kwargs):
        self.pause.acquire_inference()
        self.pause.checkpoint(self.on_pause)
        ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
        if ids is not None:
            self.tracker.on_forward(ids)

    def _model_post(self, module, args, out):
        self.tracker.on_forward_end()
        self.pause.lock.release()

    def _layer_pre(self, index: int):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            write_act(self.act, h, self.clear_act)
            if self.after_act is not None:
                self.after_act(h.numel())
            self.tracker.on_layer(index)
            self.pause.checkpoint(self.on_pause)

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# ---- HF transformers helpers (static cache)


def hf_cache_buffers(cache) -> list[tuple[str, torch.Tensor]]:
    """(name, tensor) for every layer's K and V of a transformers StaticCache, in layer order.
    Handles the `layers[i].keys/values` (>= 4.56) and `key_cache[i]/value_cache[i]` layouts."""
    out = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if k is None or v is None:
                raise RuntimeError("static cache not initialised yet: run one forward (or early_initialization) first")
            out += [(f"kv.{i}.k", k), (f"kv.{i}.v", v)]
        return out
    for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        out += [(f"kv.{i}.k", k), (f"kv.{i}.v", v)]
    return out


def make_region(
    kv_buffers: list[tuple[str, torch.Tensor]], act: torch.Tensor, block_bytes: int, pad_seed: int = 0, gpu: int = 0
) -> DynamicRegion:
    """The region in canonical order: K/V per layer, then the activation buffer, then the pad."""
    return DynamicRegion(kv_buffers + [(ACT_NAME, act)], block_bytes, pad_seed, gpu=gpu)


def act_buffer(tokens: int, hidden: int, dtype: torch.dtype, device) -> torch.Tensor:
    """An activation buffer for up to `tokens` positions of a `hidden`-wide residual stream."""
    per_word = max(1, 8 // torch.empty(0, dtype=dtype).element_size())  # elements per 8-byte word
    numel = -(-tokens * hidden // per_word) * per_word
    return torch.zeros(numel, dtype=dtype, device=device)


class DeviceSync:
    """on_pause callback: wait for the inference thread's queued kernels so the region is frozen."""

    def __init__(self, device):
        self.device = torch.device(device)

    def __call__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


# ---- the dyn_dram cheat: free dynamic memory read from host memory


class DramEvict:
    """The dyn_dram cheat (a posture of `DynamicResponder`): a set of the region's units (KV pages, cache slots)
    whose filler is kept in pinned host memory, and read from there over PCIe (mapped, zero-copy) whenever the
    unit is free at the frozen stage. A cheater would use the HBM this frees for its own data and serve the
    chain's reads of free memory from the host copy on demand. This one leaves HBM as it is (vLLM may allocate
    those blocks at any step: overwriting them would change inference and confound the measurement), so its
    effect on a chain is exactly the attack's: the same words, read at PCIe latency.

    ranges: the set, [(segment name, first unit, units, unit words)]: units [first, first + units) of the
    segment, each `unit words` long. Their filler depends only on the seed, the GPU and the position, so it is
    computed into the host copy once and stays valid while a unit is free (a freed unit gets the same filler
    back). free_fn(stage): per range, a boolean array over its units, True where the unit is free at that stage.
    `apply(stage)` reads the region through a table in which the free units of the set point into the host
    copy and every other word (live units, units outside the set, the activation rows, the pad) at its HBM
    address."""

    def __init__(self, region: DynamicRegion, ranges, free_fn, chunk_words: int = 1 << 22):
        if region.live:
            raise ValueError("dyn_dram reads the main region's free memory from host memory, not a live region's")
        self.region, self.free_fn = region, free_fn
        segs = {s.name: s for s in region.segments}
        self.ranges, off = [], 0
        for name, first, units, uw in ranges:
            if name not in segs:
                raise ValueError(f"dyn_dram: no segment {name} in the region")
            first, units, uw = int(first), int(units), int(uw)
            if first < 0 or units < 1 or uw < 1 or (first + units) * uw > segs[name].words:
                raise ValueError(
                    f"dyn_dram: units [{first}, {first + units}) of {uw} words outside {name} ({segs[name].words} words)"
                )
            self.ranges.append((name, first, units, uw, off))
            off += units * uw
        if not self.ranges:
            raise ValueError("dyn_dram: nothing to keep in host memory")
        self.words = off
        self.host = torch.empty(off, dtype=torch.int64, pin_memory=region.cuda)
        for name, first, units, uw, hoff in self.ranges:
            seg, lo, n = segs[name], first * uw, units * uw
            dst = self.host[hoff : hoff + n]
            if region.cuda:  # computed on the GPU in chunks, then copied out
                tmp = torch.empty(min(chunk_words, n), dtype=torch.int64, device=region.device)
                for a in range(0, n, chunk_words):
                    b = min(n, a + chunk_words)
                    region.free_(seg, lo + a, tmp[: b - a])
                    dst[a:b].copy_(tmp[: b - a])
                del tmp
            else:
                dst.copy_(torch.from_numpy(region.free_np(seg, lo, n).view(np.int64)))
        self.host_ptr = region.backend.host_device_ptr(self.host) if region.cuda else int(self.host.data_ptr())
        self._key = None
        self._tab = None
        self.fraction = None  # the share of the evictable set this is (`DynamicResponder.prepare_dram`)
        self.host_words_read = 0  # words of the region the last table reads from host memory

    def table(self, stage) -> tuple[np.ndarray, np.ndarray]:
        """(starts, ptrs) of the region's segment table for this stage: the set's free units in the host copy,
        everything else where the region's own table has it. Adjacent pieces that continue each other merge."""
        return self._build(self._free(stage))

    def _free(self, stage) -> list[np.ndarray]:
        free = [np.asarray(f, dtype=bool) for f in self.free_fn(stage)]
        if len(free) != len(self.ranges) or any(len(f) != r[2] for f, r in zip(free, self.ranges)):
            raise ValueError("dyn_dram: the free units do not match the set")
        return free

    def _build(self, free: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # vectorised: under vLLM a rebuild (the live blocks in the set changed) is ~100 runs in each of ~24 layers,
        # and it runs in the READY window
        u64 = np.uint64
        by = {r[0]: (r, f) for r, f in zip(self.ranges, free)}
        W, ptrs, runs, hw = [], [], {}, 0
        for seg in self.region.segments:
            got = by.get(seg.name)
            if got is None:
                W.append(np.array([seg.words], u64))
                ptrs.append(np.array([seg.ptr], u64))
                continue
            (_, first, units, uw, hoff), f = got
            if id(f) not in runs:  # (a vLLM pool gives every layer the same array: its runs once)
                a = np.concatenate([[0], np.flatnonzero(f[1:] != f[:-1]) + 1]).astype(u64)
                b = np.append(a[1:], u64(units))
                runs[id(f)] = (a, b - a, f[a.astype(np.int64)])
            a, n, host = runs[id(f)]
            W += [np.array([first * uw], u64), n * u64(uw), np.array([seg.words - (first + units) * uw], u64)]
            ptrs += [
                np.array([seg.ptr], u64),
                np.where(
                    host,
                    u64(self.host_ptr) + u64(8) * (u64(hoff) + a * u64(uw)),
                    u64(seg.ptr) + u64(8) * (u64(first) + a) * u64(uw),
                ),
                np.array([seg.ptr + 8 * (first + units) * uw], u64),
            ]
            hw += int(n[host].sum()) * uw
        w, p = np.concatenate(W), np.concatenate(ptrs)
        keep = w > 0
        w, p = w[keep], p[keep]
        new = np.ones(len(w), dtype=bool)  # a piece that continues the one before it merges into it
        new[1:] = p[:-1] + u64(8) * w[:-1] != p[1:]
        first_of = np.flatnonzero(new)
        self.host_words_read = hw
        return np.concatenate([np.zeros(1, u64), np.cumsum(np.add.reduceat(w, first_of), dtype=u64)]), p[first_of]

    def apply(self, stage) -> None:
        """Read the region through this stage's cheat table (built again only when the free units changed)."""
        free = self._free(stage)
        uniq, idx = {}, []  # the free units as a key, each distinct array once
        for f in free:
            if id(f) not in uniq:
                uniq[id(f)] = (len(uniq), np.packbits(f).tobytes() + len(f).to_bytes(8, "little"))
            idx.append(uniq[id(f)][0])
        key = (tuple(idx), tuple(v[1] for v in uniq.values()))
        if key != self._key or self._tab is None:
            self._tab = self.region.segtab(*self._build(free))
            self._key = key
        self.region.use_table(self._tab)

    def reads_host(self, ptrs) -> np.ndarray:
        """For each entry of a segment table's pointers: does it point into the host copy?"""
        p = np.asarray(ptrs, dtype=np.uint64)
        return (p >= np.uint64(self.host_ptr)) & (p < np.uint64(self.host_ptr + 8 * self.words))


# ---- answering dynamic challenges (shared by the in-process responder and the harness prover)


class DynCpuProver:
    """The chain over the region on the CPU through the segment table (host pointers): the CPU demo
    prover and the tests. Same answer as the CUDA path."""

    def __init__(self, region: DynamicRegion, keys):
        from ..core.backend import CpuLib
        from ..core.params import MAX_LINKS

        self.region, self.keys, self.lib = region, keys, CpuLib()
        self.out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)

    def respond(self, ch):
        from ..core.params import MODE_HONEST, pack_params

        prm = pack_params(
            self.region.cfg, ch, self.keys, MODE_HONEST, nseg=self.region.nseg, segtab=self.region.segtab_ptr
        )
        t0 = time.perf_counter_ns()
        self.lib.chain(self.region.theta_prime.numpy().view(np.uint64), prm, self.out)
        return int(self.out[0]), int(self.out[1]), self.out[2 : 2 + ch.n].tolist(), time.perf_counter_ns() - t0

    def close(self):
        pass


class ArmFailed(RuntimeError):
    """A dynamic ARM (or a static freeze) failed after the region was frozen. The responder has already
    unfrozen it, withdrawn any pause request and dropped what it had armed or built; no READY is sent, so the
    verifier records the challenge as a fault (no dynamic READY)."""


class DynamicResponder:
    """Freeze, chain, record, resume.

    `arm()` requests a pause and waits for the inference thread to reach a boundary; if none comes within
    the wait (idle, or between requests) it takes the pause lock instead. Either way the region is
    frozen, the stage is final, and the chain is pre-launched. `answer()` runs the timed chain;
    `finish()` keeps the stage record (and, with `include_bytes`, the region's bytes for debugging) and
    resumes. `on_freeze`, if given, runs once the region is frozen.

    Two chains per freeze: when the verifier asks for it and `live_fn` is given, `arm()` also builds a
    *live region* from the frozen stage (the live KV and the activation buffer only, no free memory), and
    a second challenge runs over it (`answer(..., live=True)`). The first chain covers all of the dynamic
    memory, mostly free filler under vLLM; the second checks the live data at every verification. The
    region stays frozen until both are answered (or the hold expires).

    On CUDA the chain is the honest prover's armed graph over the region's segment table; on the CPU it
    is the C chain.

    The dyn_dram posture (`set_posture`, a cheat for the demo and the measurements): at every freeze the
    main region is read through a table that takes the free part of an evictable set from a pinned host
    copy (`DramEvict`), built before the chains are armed; any other posture restores the region's own."""

    def __init__(
        self,
        region: DynamicRegion,
        stage_fn,
        pause: PauseController,
        keys,
        prearm: bool = True,
        model_info: dict | None = None,
        include_bytes: bool = False,
        on_freeze=None,
        debug_fn=None,
        live_fn=None,
        dram_fn=None,
    ):
        """include_bytes: attach the region's bytes to each record and compare them before and after the
        challenge (small regions). debug_fn: for regions too large to copy, a function returning a
        comparable summary (checksums) and extra debug fields, called at the freeze and after the reply.
        live_fn(stage): the live region's buffers [(name, tensor)] for a frozen stage, or None if this
        prover has no separate live region. dram_fn(fraction): the dyn_dram cheat's set for this region,
        (ranges, free_fn) as `DramEvict` takes them, or None if the prover cannot take that posture."""
        from ..core.evidence import EvidenceStore

        self.region, self.pause, self.keys, self.prearm = region, pause, keys, prearm
        self.stage_fn = stage_fn.stage if hasattr(stage_fn, "stage") else stage_fn
        self.model_info, self.include_bytes, self.on_freeze, self.debug_fn = (
            dict(model_info or {}),
            include_bytes,
            on_freeze,
            debug_fn,
        )
        # the record names the GPU whose key made its filler (the verifier checks it)
        self.model_info.setdefault("gpu", region.gpu)
        if region.cuda:
            from .strategies import HonestProver

            self.prover = HonestProver(region, keys)
        else:
            self.prover = DynCpuProver(region, keys)
        self.live_fn, self.live_region, self.live_prover, self._live_pending = live_fn, None, None, False
        self._deferred = None  # the main record, kept until the live chain is answered (its debug work is slow)
        self.store = EvidenceStore()
        self.capture_evidence = True
        self.server = None
        self.in_flight = False
        self.holding = False  # a paused thread is parked in a checkpoint
        self.locked = False  # we hold the pause lock (no boundary reached)
        self.stage: dict = {}
        self.served = 0
        self.paused_in_flight = 0
        self.paused_between = 0
        self.idle = 0
        self.static_freezes = 0
        self.slow_freezes = 0
        self.slow_arms = 0
        self.free_min: int | None = None  # least free device memory seen at a freeze (the prover's scratch use)
        self.dram_fn, self.dram, self.dram_on, self.dram_fraction = dram_fn, None, False, 0.5
        self.timing_shares = {}
        self.timing_span = None
        self._timing_active = None
        self._timing_seg = self._timing_raw = self._timing_other = None
        self._timing_other_key = None
        self.timing_live_arms = 0
        self.timing_live_pages = 0
        self.timing_live_arms_by_posture = {}
        self.timing_write_refreshes = 0

    def prepare_timing_shares(self, specs: list[str], block_bytes: int = 1 << 20) -> dict:
        """Prepare one block-aligned KV span for all five timing postures.

        Use the low page IDs that inference can allocate. A posture refreshes its hidden source
        when selected and after each write or reset to one of these pages.
        """
        from types import SimpleNamespace

        from ..core.backend import CpuLib
        from ..core.codec_profile import FIELD_BITS
        from .dynfill import dyn_key
        from .kvenc import empty_
        from .strategies import DynOtherGpuCheat, DynShareCheat, ShareCheat

        if not self.region.cuda or FIELD_BITS != 89:
            raise ValueError("dynamic timing shares require the CUDA 89-bit codec")
        fractions = {float(s.split("@", 1)[1]) for s in specs}
        if len(fractions) != 1 or not fractions or not 0 < next(iter(fractions)) <= 1:
            raise ValueError("dynamic timing postures must use one common fraction")
        fraction = fractions.pop()
        r = self.region
        bw = block_bytes // 8
        blocks = max(1, int(r.num_blocks(block_bytes) * fraction))
        kv = [s for s in r.segments if s.name.startswith("kv.") and s.enc is not None]
        if not kv:
            raise ValueError("dynamic timing shares need encoded KV stores")
        seg = kv[-1]
        keep = -(-seg.offset // bw) * bw
        end = keep + blocks * bw
        uw = seg.enc.unit_words
        if end > seg.offset + seg.words:
            raise ValueError("the last KV store cannot hold the requested aligned dynamic share")
        local = keep - seg.offset
        words = end - keep
        host = torch.empty(words, dtype=torch.int64, pin_memory=True)
        host.copy_(seg.tensor[local : local + words])
        r.theta_host = host
        r.theta_host_np = host.numpy().view(np.uint64)
        r.share_span = (keep, end)
        host_ptr = r.backend.host_device_ptr(host)
        pieces = []
        control_pieces = []
        for s in r.segments:
            a, b = s.offset, s.offset + s.words
            cuts = [a, b]
            if a < keep < b:
                cuts.append(keep)
            if a < end < b:
                cuts.append(end)
            cuts.sort()
            for lo, hi in zip(cuts, cuts[1:]):
                device_ptr = s.ptr + 8 * (lo - a)
                ptr = host_ptr + 8 * (lo - keep) if keep <= lo and hi <= end else device_ptr
                pieces.append((hi - lo, ptr))
                control_pieces.append((hi - lo, device_ptr))
        starts = np.cumsum([0] + [w for w, _ in pieces]).astype(np.uint64)
        partial_table = r.segtab(starts, np.array([p for _, p in pieces], dtype=np.uint64))
        partial_control_table = r.segtab(starts, np.array([p for _, p in control_pieces], dtype=np.uint64))
        raw = torch.zeros(words, dtype=torch.int64, device=r.device)
        other_key = dyn_key(r.pad_seed, r.gpu + 1)
        other = torch.empty_like(raw)
        empty_(other, keep, other_key)
        torch.cuda.synchronize(r.device)
        cpu = SimpleNamespace(lib=CpuLib())
        shares = {}
        for spec in specs:
            name = spec.split("@", 1)[0]
            control = name.endswith("_control")
            name = name.removesuffix("_control")
            if name == "partial":
                shares[spec] = (partial_control_table if control else partial_table, None)
            elif name == "encode_otf":
                share = DynShareCheat(keep, end, raw, r.dyn_key)
                if control:
                    share.end = keep
                shares[spec] = (None, share)
            elif name == "other_gpu":
                share = DynOtherGpuCheat(keep, end, other, other_key, r.dyn_key)
                if control:
                    share.end = keep
                shares[spec] = (None, share)
            elif name in ("dram_copy", "cpu"):
                shares[spec] = (None, ShareCheat(name, r, fraction, cpu=cpu, control=control))
            else:
                raise ValueError(f"unsupported dynamic timing posture: {spec}")
        self.timing_shares = shares
        self.timing_span = (seg.name, (keep - seg.offset) // uw, -(-(end - seg.offset) // uw))
        self._timing_seg, self._timing_raw, self._timing_other = seg, raw, other
        self._timing_other_key = other_key
        print(
            f"posm dynamic timing share: {seg.name} words [{keep}, {end}), "
            f"{blocks}/{r.num_blocks(block_bytes)} challenge blocks ({blocks / r.num_blocks(block_bytes):.5%}); "
            f"pages [{self.timing_span[1]}, {self.timing_span[2]})",
            flush=True,
        )
        return shares

    def _refresh_timing_source(self, name: str, pages=None, *, scratch=None, zeroed: bool = False) -> None:
        """Bring the active hidden representation up to date with the encoded KV store.

        Called under the pause lock at a posture switch, or by the inference thread while it
        holds that lock after a layer writes or pages are reset. Decode from the encoded store,
        including after a live write: scratch is not a reliable source for every byte of a page.
        CUDA work stays on the current stream; a challenge freeze synchronizes it before READY.
        """
        if self.timing_span is None or name not in ("partial", "dram_copy", "cpu", "encode_otf", "other_gpu"):
            return
        from .kvenc import KV_POS

        seg = self._timing_seg
        keep, end = self.region.share_span
        if pages is None:
            intervals = [(keep, end, None)]
        else:
            intervals = []
            uw = seg.enc.unit_words
            for page in sorted(set(int(b) for b in pages)):
                lo = max(keep, seg.offset + page * uw)
                hi = min(end, seg.offset + (page + 1) * uw)
                if lo < hi:
                    intervals.append((lo, hi, page))
        for lo, hi, page in intervals:
            source = seg.tensor[lo - seg.offset : hi - seg.offset]
            target = slice(lo - keep, hi - keep)
            if name in ("partial", "dram_copy", "cpu"):
                self.region.theta_host[target].copy_(source)
            else:
                payload = self._timing_raw[target]
                if zeroed:
                    payload.zero_()
                else:
                    self.region.backend.decode_at_packed128(source, payload, self.region.dyn_key, KV_POS | lo)
                if name == "other_gpu":
                    self.region.backend.setup_at(
                        payload, self._timing_other[target], self._timing_other_key, KV_POS | lo
                    )
        if pages is not None and intervals:
            self.timing_write_refreshes += len(intervals)

    def update_timing_pages(self, pages, *, scratch=None, zeroed: bool = False) -> None:
        """Track live KV writes and fills for the currently selected timing posture."""
        if self._timing_active is not None:
            self._refresh_timing_source(self._timing_active, pages, scratch=scratch, zeroed=zeroed)

    def serve_evidence(self, bind: str, port: int):
        from ..core.evidence import EvidenceServer

        self.server = EvidenceServer(self.store, bind, port)
        self.server.start()
        return self.server

    @property
    def armed(self) -> bool:
        return getattr(self.prover, "_armed_shape", None) is not None

    @property
    def live_armed(self) -> bool:
        return self.live_prover is not None and getattr(self.live_prover, "_armed_shape", None) is not None

    @property
    def frozen(self) -> bool:
        return self.holding or self.locked

    def prepare_dram(self, fraction: float | None = None) -> DramEvict:
        """The dyn_dram cheat's host copy of `fraction` of the evictable memory (made once per fraction: tens
        of GiB under vLLM, so the responder makes it at start when the posture is offered)."""
        fraction = self.dram_fraction if fraction is None else float(fraction)
        if self.dram_fn is None:
            raise ValueError("dyn_dram: this prover's dynamic region has no evictable set")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"dyn_dram: fraction {fraction} outside (0, 1]")
        if self.dram is None or self.dram.fraction != fraction:
            self.region.restore_table()  # nothing may read through the old copy's table
            self.dram = None  # the old host copy goes before the new one is made
            ranges, free_fn = self.dram_fn(fraction)
            self.dram = DramEvict(self.region, ranges, free_fn)
            self.dram.fraction = fraction
        self.dram_fraction = fraction
        return self.dram

    def set_posture(self, name: str | None) -> None:
        """dyn_dram: from the next freeze on, the main chain reads the free part of the evictable set from host
        memory; any other posture: the region's own table again. Not while frozen (the responder releases first)."""
        if self.frozen:
            raise RuntimeError("posture change while the region is frozen")
        with self.pause.lock:
            self.prover.share = None
            self._timing_active = None
            if name == "dyn_dram":
                if self.dram is None:
                    self.prepare_dram()
                self.dram_on = True
            elif name in self.timing_shares:
                self.dram_on = False
                tab, share = self.timing_shares[name]
                base = name.split("@", 1)[0]
                active = None if base.endswith("_control") else base
                if active is not None:
                    torch.cuda.synchronize(self.region.device)
                    self._refresh_timing_source(active)
                    torch.cuda.synchronize(self.region.device)
                self.region.use_table(tab) if tab is not None else self.region.restore_table()
                self.prover.share = share
                self._timing_active = active
            else:
                self.dram_on = False
                self.region.restore_table()

    def _count_timing_share_live(self) -> None:
        if self.timing_span is None or self._timing_active is None:
            return
        _, first, last = self.timing_span
        occupied = {int(b) for req in self.stage.get("requests", []) for b in req.get("block_ids", [])}
        count = sum(first <= b < last for b in occupied)
        if count:
            self.timing_live_arms += 1
            self.timing_live_pages += count
            old = self.timing_live_arms_by_posture.get(self._timing_active, 0)
            self.timing_live_arms_by_posture[self._timing_active] = old + 1
            if old == 0:
                print(
                    f"posm dynamic timing live share: posture={self._timing_active} "
                    f"live_pages={count}",
                    flush=True,
                )

    def disarm(self):
        """Release armed graphs in queue order: the pool graph first, then the live graph behind it."""
        if self.armed:
            self.prover.disarm()
        if self.live_armed:
            self.live_prover.disarm()

    def freeze(self, wait_s: float) -> bool:
        """Stop inference writing the region, as soon as possible. Returns True if the inference thread is
        parked at a boundary (holding), False if we hold the lock.

        No forward running (the lock is free): take the lock at once. A forward running: ask for a pause
        at its next layer boundary, and take the lock instead if the forward ends first. Every checkpoint
        runs with the lock held, so holding the lock means no thread is parked. The wait is therefore at
        most one layer (or the rest of a forward's non-layer work), not a step and not `wait_s`, which is
        only the fallback. This keeps the READY window, and so what a prover could reload in it, small."""
        self.release()
        t0 = time.perf_counter()
        path = "lock free"
        if self.pause.lock.acquire(blocking=False):
            self.locked = True
            reached = False
        else:
            self.pause.request()
            t_end = time.perf_counter() + wait_s
            reached = None
            while reached is None:
                if self.pause.wait_paused(0.0002):
                    reached = True
                    path = "boundary"
                elif self.pause.lock.acquire(blocking=False):  # the forward ended without a boundary
                    path = "forward ended"
                    if self.pause.cancel():
                        self.locked = True
                        reached = False
                    else:  # cannot happen (checkpoints hold the lock); be safe
                        self.pause.lock.release()
                        reached = True
                elif time.perf_counter() > t_end:
                    path = "fallback"
                    if self.pause.cancel():
                        self.pause.lock.acquire()
                        self.locked = True
                        reached = False
                    else:
                        reached = True  # taken just as we gave up: it is parked now
            if reached:
                self.holding = True
        if self.region.cuda:
            torch.cuda.synchronize(self.region.device)
            free = torch.cuda.mem_get_info(self.region.device)[0]  # inference is stopped here, often mid-forward
            self.free_min = free if self.free_min is None else min(self.free_min, free)
        wait = time.perf_counter() - t0
        self.slow_freezes += wait > 0.02
        if wait > 0.02:  # diagnostic: what made the READY window long
            st = self.stage_fn()
            print(
                f"posm dynamic: slow freeze {wait * 1e3:.1f} ms ({path}); stage step {st.get('step')} layer {st.get('act_layer', st.get('layer'))} "
                f"in flight {st.get('in_flight')}",
                flush=True,
            )
        return reached

    def freeze_static(self, wait_s: float) -> None:
        """Freeze for a static challenge (the drain window): the GPU is idle at a boundary, nothing recorded.
        Raises ArmFailed (with the region unfrozen) if the freeze fails."""
        try:
            self.freeze(wait_s)
        except BaseException as e:
            self._abort()
            if isinstance(e, Exception):
                raise ArmFailed(f"static freeze failed: {type(e).__name__}: {e}") from e
            raise
        self.static_freezes += 1

    def arm(self, shape, wait_s: float, want_live: bool = False) -> tuple[int, int, int, int]:
        """Freeze the region, then pre-launch the chain. Returns (words, block_bytes, flags, live_words):
        flags bit 0 = paused inside a step, bit 1 = paused at a step boundary, neither = took the lock
        (no forward running); live_words = size of the live region (0: no live chain).

        Exception-safe: if anything fails once the freeze has begun (the stage, a freeze hook, the live
        region, the write ordering), the region is unfrozen, a pending pause request withdrawn and the
        partial state dropped before ArmFailed is raised; inference is never left blocked."""
        try:
            return self._arm(shape, wait_s, want_live)
        except BaseException as e:
            self._abort()
            if isinstance(e, Exception):
                raise ArmFailed(f"dynamic ARM failed: {type(e).__name__}: {e}") from e
            raise

    def _abort(self) -> None:
        """Undo a freeze or ARM that failed part way: withdraw a pause request nobody will wait for (or, if the
        inference thread took it just now, count it as parked so it is resumed), drop the half-built live
        state and stage, and release."""
        try:
            if not self.frozen and not self.pause.cancel():  # the thread parked as we failed: resume it below
                self.holding = True
        except Exception as e:  # (cannot happen; never let the cleanup itself raise)
            print(f"posm dynamic: abort: {e}", flush=True)
        self._live_pending = False
        self._deferred = None
        self.stage = {}
        self.release()

    def _arm(self, shape, wait_s: float, want_live: bool) -> tuple[int, int, int, int]:
        if want_live and not self.capture_evidence:
            raise ValueError("the live-only chain requires stage capture")
        t0 = time.perf_counter()
        self.freeze(wait_s)
        t1 = time.perf_counter()
        self.stage = self.stage_fn()
        self._count_timing_share_live()
        t2 = time.perf_counter()
        self.in_flight = bool(self.stage.get("in_flight"))
        if self.holding and self.in_flight:
            self.paused_in_flight += 1
            flags = 1
        elif self.holding:
            self.paused_between += 1
            flags = 2
        else:
            self.idle += 1
            flags = 0
        if self.on_freeze is not None:
            self.on_freeze(self.stage)
        if self.dram_on:  # dyn_dram: this stage's table, before anything is armed over it
            self.dram.apply(self.stage)
        self.frozen_by = {1: "mid-step", 2: "step-boundary", 0: "idle-lock"}[flags]
        self._snap0 = self.region.snapshot() if self.include_bytes else None  # debug: the region before the chain
        self._dbg0 = self.debug_fn() if self.debug_fn is not None else None
        t3 = time.perf_counter()
        live_words = self._build_live() if want_live and self.live_fn is not None else 0
        self._order_writes()
        t4 = time.perf_counter()
        self._live_pending = live_words > 0
        if self.prearm and shape is not None and hasattr(self.prover, "arm"):
            try:
                self.prover.arm(shape)
                if self._live_pending:  # queued behind the pool graph on the same stream: it starts
                    self.live_prover.arm(shape)  # waiting for its flag only once the pool chain has run
            except (ValueError, RuntimeError) as e:
                print(f"posm dynamic: pre-arm skipped: {e}", flush=True)
        t5 = time.perf_counter()
        if t5 - t0 > 0.003:  # diagnostic: where a long READY window went
            self.slow_arms += 1
            print(
                f"posm dynamic: slow ARM {1e3 * (t5 - t0):.1f} ms: freeze {1e3 * (t1 - t0):.1f}, stage {1e3 * (t2 - t1):.1f}, "
                f"freeze hooks {1e3 * (t3 - t2):.1f}, live region {1e3 * (t4 - t3):.1f}, pre-arm {1e3 * (t5 - t4):.1f} ms; "
                f"{len(self.stage.get('requests', []))} live requests",
                flush=True,
            )
        return self.region.words, self.region.block_bytes, flags, live_words

    def _order_writes(self) -> None:
        """Order every write made since the freeze before the chains. The freeze synchronises the device,
        which covers what the parked inference thread queued (vLLM's capture of the activation rows), but
        the freeze hooks run after it on this thread's stream: the activation rows put back to filler, the
        live region's pad and segment table. The armed chains run on the chain kernels' private stream with
        no ordering against torch's, so without this a chain fired right after READY could read the region
        before those writes land. An event on this stream that the private stream waits on orders them
        without blocking the host (a few µs before READY; the GPU finishes the writes, tens of µs at most,
        while the challenge is on the wire); the fallback synchronises this stream."""
        if not self.region.cuda:
            return
        dev = self.region.device
        if getattr(self, "_order", None) is None:
            side = None
            be = getattr(self.region, "backend", None)
            if be is not None and hasattr(be, "side_stream"):
                try:
                    with torch.cuda.device(dev):
                        side = be.side_stream()
                except (RuntimeError, AttributeError):
                    side = None
            self._order = (side, torch.cuda.Event() if side is not None else None)
        side, ev = self._order
        if side is None:
            torch.cuda.current_stream(dev).synchronize()
            return
        ev.record(torch.cuda.current_stream(dev))
        side.wait_event(ev)  # enqueued ahead of the armed graphs on the private stream

    def _build_live(self) -> int:
        """The live region of the frozen stage, laid out over the same buffers (no copy)."""
        live = self.live_fn(self.stage)
        if isinstance(live, PagedLive):
            if self.live_region is None:
                self.live_region = DynamicRegion(
                    [],
                    self.region.block_bytes,
                    self.region.pad_seed,
                    device=self.region.device,
                    gpu=self.region.gpu,
                    live=True,
                )
                if self.live_region.cuda:
                    from .strategies import HonestProver

                    self.live_prover = HonestProver(self.live_region, self.keys)
                else:
                    self.live_prover = DynCpuProver(self.live_region, self.keys)
            self.live_region.rebuild_paged(live)
            return self.live_region.words
        segs = [(n, t) for n, t in live if buffer_words(t)]
        if not segs:
            return 0
        if self.live_region is None:
            self.live_region = DynamicRegion(
                segs,
                self.region.block_bytes,
                self.region.pad_seed,
                device=self.region.device,
                gpu=self.region.gpu,
                live=True,
            )
            if self.live_region.cuda:
                from .strategies import HonestProver

                self.live_prover = HonestProver(self.live_region, self.keys)
            else:
                self.live_prover = DynCpuProver(self.live_region, self.keys)
        else:
            self.live_region.rebuild(segs)
        return self.live_region.words

    def answer(self, ch, live: bool = False) -> tuple[int, int]:
        if not self.frozen:
            raise RuntimeError("dynamic challenge without a dynamic ARM")
        if live:
            if not self._live_pending:
                raise RuntimeError("live challenge without a live region")
            if self.armed:
                self.prover.disarm()  # an unused pool graph ahead of it would block the stream
            if self.live_armed and self.live_prover.armed_for(ch):
                lo, hi, _, _ = self.live_prover.fire(ch)
            else:
                self.disarm()
                lo, hi, _, _ = self.live_prover.respond(ch)
            return lo, hi
        if self.armed and self.prover.armed_for(ch):
            lo, hi, _, _ = self.prover.fire(ch)
        else:
            self.disarm()
            lo, hi, _, _ = self.prover.respond(ch)
        return lo, hi

    def finish(self, ch, lo: int, hi: int, live: bool = False):
        """After a timed reply, region still frozen: keep the record; resume unless the live chain is
        still to come. The main record is completed after the live chain, so that its debug work (a
        checksum of the whole region under vLLM) never delays the live challenge."""
        if not self.capture_evidence:
            self.served += 1
            self.release()
            return
        from .. import __version__
        from ..core.evidence import Evidence

        if not live and self._live_pending:
            self._deferred = (ch, lo, hi)
            self.served += 1
            return
        if live:
            from ..core.evidence import LazyEvidence

            r = self.live_region  # everything the record needs, captured now (the region is reused)
            chal = {"block_bytes": ch.block_bytes, "n": ch.n, "r": ch.r, "k": ch.k, "g": ch.g, "j0": ch.j0}
            words, bb, ps, lay, st, mi, fb = (
                r.words,
                r.block_bytes,
                r.pad_seed,
                r.layout_fn(),
                self.stage,
                dict(self.model_info),
                getattr(self, "frozen_by", ""),
            )
            self.store.put(
                LazyEvidence(
                    ch.nonce,
                    lambda: Evidence(
                        nonce=ch.nonce,
                        challenge=chal,
                        digest=[lo, hi],
                        words=words,
                        block_bytes=bb,
                        pad_seed=ps,
                        layout=lay(),
                        stage=st,
                        software_version=__version__,
                        model=mi,
                        debug={"frozen_by": fb},
                        region="live",
                    ),
                )
            )
            self._live_pending = False
            self.release()
            return
        self._store_main(ch, lo, hi)
        self.served += 1
        self.release()

    def _store_main(self, ch, lo: int, hi: int):
        from .. import __version__
        from ..core.evidence import Evidence

        data = self.region.snapshot() if self.include_bytes else None
        debug = {"frozen_by": getattr(self, "frozen_by", "")}
        if data is not None and self._snap0 is not None:
            # nothing wrote the region between freeze and reply
            debug["region_unchanged"] = bool(np.array_equal(self._snap0, data))
        if self._dbg0 is not None:
            dbg1 = self.debug_fn()
            debug["region_unchanged"] = dbg1["checksum"] == self._dbg0["checksum"]
            debug.update({k: v for k, v in dbg1.items() if k != "checksum"})
        self.store.put(
            Evidence(
                nonce=ch.nonce,
                challenge={"block_bytes": ch.block_bytes, "n": ch.n, "r": ch.r, "k": ch.k, "g": ch.g, "j0": ch.j0},
                digest=[lo, hi],
                words=self.region.words,
                block_bytes=self.region.block_bytes,
                pad_seed=self.region.pad_seed,
                layout=self.region.layout(),
                stage=self.stage,
                software_version=__version__,
                model=dict(self.model_info),
                data=data,
                debug=debug,
            )
        )

    @property
    def live_pending(self) -> bool:
        return self._live_pending

    def release(self):
        """Unfreeze (idempotent) and drop any armed graph; a main record still waiting is completed first,
        while the region is frozen (the live chain may never have come). The region is unfrozen even when
        completing the record or disarming raises (a timed-out disarm); the error is raised after."""
        self._live_pending = False
        try:
            if self._deferred is not None:
                d, self._deferred = self._deferred, None
                self._store_main(*d)
            self.disarm()
        finally:
            if self.holding:
                self.holding = False
                self.pause.resume()
            if self.locked:
                self.locked = False
                self.pause.lock.release()

    def close(self):
        self.release()
        if self.server is not None:
            self.server.stop()
        self.prover.close()
