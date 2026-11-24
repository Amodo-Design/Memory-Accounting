"""Dynamic memory under vLLM, prover side: the KV pool and activation buffers as a challengeable region, and the
serving loop (stream-decoded weights, filler, the responder) under generation load.

Prover (inside the vLLM worker, via `memory_accounting.prover.vllm_ext.PosmWorkerExtension.posm_dynamic_setup`):

  * the dynamic region is the whole KV pool, every block of every layer in physical order (`PagedPool`),
    then the activation buffers `act.h` and `act.r` (the hidden state and the residual entering the paused
    layer, one row per scheduled token in batch order, filler after), then the pad. Its layout is fixed for
    the life of the engine; the chain kernels read the pages in place through the segment table;
  * every word that holds no live KV holds dynamic filler (`memory_accounting.prover.dynfill`): the whole pool at setup,
    a request's blocks again as soon as the worker learns it finished or was preempted, and the unwritten
    slots of a live block, because a block keeps (or, if it was given out since its last fill, gets) its
    filler when it is allocated. Zeros (v0.4 to v0.11) let a prover report a request whose slots were all
    "unwritten" and hold nothing there;
  * a second, *live* region per freeze holds only the live requests' pages (stage order, layer-major,
    block-table order) and the activation rows, so every verification reads live KV (v0.5);
  * the pause points are every decoder layer (inside a step) and the start of `execute_model` (between
    steps); an idle engine is frozen by the step lock;
  * the stage record lists every live request, in batch order when it is scheduled in the frozen step, with
    its tokens, how many were in the cache before this step (`num_computed`), how many this step scheduled,
    its physical block ids; plus the paused layer.

Settings both sides need: memory_accounting.models.vllm. The verifier's replay: memory_accounting.verifier.vllm_replay.

  python -m memory_accounting.prover.vllm_serve --model <dir> --enc-key <hex> --filler-seed <hex> --port 5557 [--strategies honest,evict_dram,dyn_dram] --verifier-addr <verifier>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ..models.vllm import apply_env, page_bytes, settings
from .dynamic import DynamicRegion, DynamicResponder, PauseController, layer_modules

# ---- the KV pool as part of the dynamic region (vLLM-independent: tested on a simulated pool)


class PagedPool:
    """The whole KV pool, every layer and every block, in the dynamic region, plus the request bookkeeping.

    * Region: segment `kv.<l>` is layer l's pool (blocks in id order, each page in physical byte order),
      then `act.h`, `act.r` (the paused layer's hidden state and residual, one row per scheduled token in
      batch order, filler after), then the pad. The layout is fixed for the life of the engine.
    * Free memory holds dynamic filler: the whole pool at setup, and a request's blocks again as soon as
      the worker learns it finished or was preempted. An allocated block's unwritten slots hold filler
      too: a block that has been free since its last fill keeps it, and one given out since (freed and
      reallocated in the same step, so its refill was skipped) is filled again when it is allocated
      (`alloc_blocks`). A refill never touches a block a live request of the step owns: vLLM keeps a
      preempted request's old block list on the worker until it resumes or ends, and the scheduler may
      already have given those blocks to another request.
    * `kv_tokens`: for each live request, how many of its tokens are in the cache after the last step."""

    def __init__(
        self,
        kv: list[torch.Tensor],
        act_h: torch.Tensor,
        act_r: torch.Tensor,
        block_bytes: int,
        pad_seed: int,
        gpu: int = 0,
        encoded: bool = False,
    ):
        """encoded (v0.17, `memory_accounting.prover.kvenc`): the region holds every layer's pool encoded, page by page, and the
        activation rows encoded; layer 0's pool becomes the plaintext scratch every layer's attention runs on
        (`EncodedPool`), and the encoded pool is laid out in the memory of the other layers' pools (plus what does not
        fit there)."""
        from .dynfill import mask_for

        self.kv, self.act_h, self.act_r, self.encoded = kv, act_h, act_r, encoded
        B = kv[0].shape[0]
        for layer, k in enumerate(kv):
            page = k[0]
            if k.stride(0) * k.element_size() != page.numel() * page.element_size():
                raise ValueError(
                    f"layer {layer}: blocks are not adjacent pages (layout not layer-major); set VLLM_KV_CACHE_LAYOUT=LBNHC"
                )
            page_bytes(k)  # raises if the layer's pool is not dense
        self.n_blocks, self.block_size = B, kv[0].shape[2]
        self.page_bytes = kv[0][0].numel() * kv[0].element_size()
        if self.page_bytes % 8:
            raise ValueError("KV page is not a whole number of 8-byte words")
        self.page_words = self.page_bytes // 8
        self._base = [k.data_ptr() for k in kv]  # page b of layer l: _base[l] + b * page_bytes
        for layer, k in enumerate(kv):
            if page_bytes(k[B - 1]).data_ptr() != self._base[layer] + (B - 1) * self.page_bytes:
                raise ValueError(f"layer {layer}: page addresses are not base + index * page size")
        if encoded:
            self.enc = EncodedPool(self, block_bytes)
            segs = self.enc.buffers()
            self.page_stride_words = self.enc.epw  # a page of the region: an encoded page
        else:
            self.enc = None
            segs = [(f"kv.{layer}", page_bytes(k)) for layer, k in enumerate(kv)] + [("act.h", act_h), ("act.r", act_r)]
            self.page_stride_words = self.page_words
        masks = {f"kv.{layer}": mask_for(k.dtype) for layer, k in enumerate(kv)}
        self.region = DynamicRegion(segs, block_bytes, pad_seed, device=kv[0].device, gpu=gpu, masks=masks)
        if encoded:
            self.enc.attach(self.region)
        self.kv_tokens: dict[str, int] = {}
        self.released: set[str] = set()  # preempted requests whose blocks were refilled (their worker list is stale)
        self.dirty: set[int] = set()  # blocks given out since their last fill (anything else holds filler)
        self.freed_blocks = 0
        self.free_s_max = 0.0
        self.refill_skipped = 0
        self.alloc_filled = 0

    def init_filler(self) -> None:
        """At setup, nothing is live: every block of every layer and the activation rows hold filler (encoded zeros)."""
        for layer in range(len(self.kv)):
            self.region.fill(f"kv.{layer}")
        self.dirty.clear()
        if self.enc is not None:
            self.enc.init_act(self.region)
        else:
            self.clear_act()

    def clear_act(self) -> None:
        if self.enc is not None:
            self.enc.clear_act()
            return
        self.region.clear("act.h")
        self.region.clear("act.r")

    def free_blocks(self, blocks) -> None:
        """Freed blocks get their filler back, in every layer: one kernel for all of them. (One fill per
        block per layer was about 2,000 launches when a 16-request batch finished, 25 to 30 ms under the
        step lock, which a freeze had to wait out.)"""
        if not blocks:
            return
        t0 = time.perf_counter()
        self._fill_blocks(blocks)
        self.freed_blocks += len(blocks)
        self.free_s_max = max(self.free_s_max, time.perf_counter() - t0)

    def alloc_blocks(self, blocks) -> None:
        """Blocks the scheduler handed out this step: their slots hold filler until the forward writes them,
        and the verifier rebuilds every slot the replay does not write as filler. A block given out since its
        last fill (freed and reallocated in one step, its refill skipped because it is live) is filled now;
        one free since its last fill already holds it."""
        blocks = [int(b) for b in blocks]
        stale = sorted(set(blocks) & self.dirty)
        if stale:
            t0 = time.perf_counter()
            self._fill_blocks(stale)
            self.alloc_filled += len(stale)
            self.free_s_max = max(self.free_s_max, time.perf_counter() - t0)
        self.dirty.update(blocks)

    def _fill_blocks(self, blocks) -> None:
        pw = self.page_stride_words
        self.region.fill_pages([(f"kv.{layer}", int(b) * pw) for layer in range(len(self.kv)) for b in blocks], pw)
        if self.enc is not None and self.enc.on_kv_write is not None:
            self.enc.on_kv_write(blocks, zeroed=True)
        self.dirty.difference_update(int(b) for b in blocks)

    def begin_step(self, so, requests) -> None:
        """Before vLLM updates its request state for a step: free what the scheduler freed, then give what it
        allocated its filler where it lacks it (in that order: a freed block can be reallocated in the same
        step, and its refill is then skipped as live).

        The blocks to refill come from the worker's copy of each ended or preempted request's block list,
        which can be stale: vLLM V1 keeps a preempted request's state, old list included, until it resumes
        (the list is replaced) or ends, and the scheduler hands the freed blocks to other requests at once.
        So the live block set of this step is computed first (every request holding KV, every request
        scheduled, with a resumed request's new list instead of its old one) and nothing in it is refilled,
        and a preempted request's blocks are refilled once, at the preemption, not again when it ends.

        A streaming-input session (a resumable request) keeps its KV between turns and its next turn arrives as
        a new request with the same id and its whole block list; only the blocks it did not already hold are
        newly allocated (`_owned`). Refusing such sessions instead is not possible at setup (resumable is per
        request, set by the API caller) and would mean raising inside a step, which ends the engine."""
        finished = set(getattr(so, "finished_req_ids", ()) or ())
        preempted = set(getattr(so, "preempted_req_ids", None) or ())
        cached = so.scheduled_cached_reqs
        resumed = set(getattr(cached, "resumed_req_ids", None) or ())
        if not resumed and getattr(cached, "resumed_from_preemption", None) is not None:  # older V1: a list of flags
            resumed = {r for r, f in zip(cached.req_ids, cached.resumed_from_preemption) if f}
        gone = finished | preempted
        for rid in gone:
            self.kv_tokens.pop(rid, None)
        live: set[int] = set()
        for r in so.scheduled_new_reqs:
            live.update(int(b) for b in r.block_ids[0])
        for rid, nb in zip(cached.req_ids, cached.new_block_ids):
            st = requests.get(rid)
            if st is not None and st.block_ids and rid not in resumed and rid not in gone:
                live.update(int(b) for b in st.block_ids[0])
            if nb is not None:
                live.update(int(b) for b in nb[0])
        for rid in self.kv_tokens:  # holding KV, scheduled this step or not
            st = requests.get(rid)
            if st is not None and st.block_ids and rid not in resumed:
                live.update(int(b) for b in st.block_ids[0])
        freed = set()
        for rid in gone:
            st = requests.get(rid)
            if rid in self.released:  # refilled at its preemption; the worker's list is stale
                if rid in finished:
                    self.released.discard(rid)
                continue
            if st is not None and st.block_ids:
                freed.update(int(b) for b in st.block_ids[0])
            if rid in preempted and rid not in finished:
                self.released.add(rid)
        self.released -= resumed  # a resumed request's list is replaced: it owns blocks again
        self.refill_skipped += len(freed & live)
        self.free_blocks(sorted(freed - live))
        new = []
        for r in so.scheduled_new_reqs:
            owned = self._owned(r.req_id, requests, gone)
            new += [b for b in r.block_ids[0] if int(b) not in owned]
            self.released.discard(r.req_id)  # its list is replaced by this one
        for nb in cached.new_block_ids:
            if nb is not None:
                new += list(nb[0])
        self.alloc_blocks(new)

    def _owned(self, rid, requests, gone) -> set[int]:
        """Blocks a request scheduled as new already holds its KV in: a streaming-input session's next turn
        (vLLM 0.30 schedules it as a new request with its whole block list: the blocks of its earlier turns,
        whose KV it keeps, then any new ones). Only the others are newly allocated; filling the old ones would
        overwrite the session's KV. None for a request the worker does not know, or whose list is stale (a
        preempted request whose blocks were refilled and may be someone else's now)."""
        st = requests.get(rid)
        if st is None or not st.block_ids or rid in gone or rid in self.released:
            return set()
        return {int(b) for b in st.block_ids[0]}

    def end_step(self, so, requests) -> None:
        """After a step: the scheduled requests' tokens are in the cache."""
        for rid, n in so.num_scheduled_tokens.items():
            st = requests.get(rid)
            if st is not None:
                self.kv_tokens[rid] = int(st.num_computed_tokens) + int(n)

    def stage(self, so, requests, batch_order, in_flight: bool, layer: int, step: int, n_layers: int) -> dict:
        """The live requests: in a step, the scheduled ones (batch order) with this step's tokens, then the
        others complete; between steps or idle, every live request complete."""

        def entry(rid, c, scheduled):
            st = requests[rid]
            total = c + scheduled
            tokens = list(st.prompt_token_ids or []) + [int(t) for t in st.output_token_ids]
            if len(tokens) < total or any(t < 0 for t in tokens[:total]):
                raise RuntimeError(f"request {rid}: tokens not known on the worker (async scheduling?)")
            nb = -(-total // self.block_size)
            return {
                "tokens": tokens[:total],
                "num_computed": c,
                "scheduled": scheduled,
                "num_tokens": total,
                "nblocks": nb,
                "block_ids": [int(b) for b in st.block_ids[0][:nb]],
            }

        reqs, seen = [], set()
        if in_flight and so is not None:
            for rid in batch_order:
                if rid in so.num_scheduled_tokens and rid in requests:
                    reqs.append(entry(rid, int(requests[rid].num_computed_tokens), int(so.num_scheduled_tokens[rid])))
                    seen.add(rid)
        for rid, kt in self.kv_tokens.items():
            if rid not in seen and rid in requests and kt > 0:
                reqs.append(entry(rid, kt, 0))
        st = {
            "kind": "vllm",
            "step": step,
            "in_flight": in_flight,
            "act_layer": layer if in_flight else -1,
            "n_layers": n_layers,
            "block_size": self.block_size,
            "page_words": self.page_words,
            "n_blocks": self.n_blocks,
            "requests": reqs,
        }
        if self.enc is not None:
            st["encoded"] = True
        return st

    def live_segments(self, stage: dict) -> list:
        """The live region of a frozen stage: each live request's pages (stage order, layer-major, block-table
        order), then the activation rows. In place, no copy."""
        from .dynamic import PageRef

        if self.enc is not None:
            return self.enc.live_segments(stage)
        pb, segs = self.page_bytes, []
        for i, r in enumerate(stage.get("requests", [])):
            for layer, k in enumerate(self.kv):
                base = self._base[layer]
                for j, b in enumerate(r["block_ids"]):  # blocks are adjacent equal pages (checked at setup)
                    segs.append(
                        (f"r{i}.l{layer}.b{j}", PageRef(base + b * pb, pb, k.device, lambda k=k, b=b: page_bytes(k[b])))
                    )
        return segs + [("act.h", self.act_h), ("act.r", self.act_r)]

    def live_pages(self, stage: dict):
        """`live_segments` as arrays (what the prover uses before READY): page addresses computed per
        request with numpy, names and views made only if the segment list is read."""
        from .dynamic import PagedLive

        if self.enc is not None:
            return self.enc.live_pages(stage)
        reqs = stage.get("requests", [])
        base = np.array(self._base, dtype=np.uint64)[:, None]
        pb = np.uint64(self.page_bytes)
        parts = [(base + np.asarray(r["block_ids"], dtype=np.uint64)[None, :] * pb).ravel() for r in reqs]
        ptrs = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
        L = len(self.kv)

        def names():
            return [
                f"r{i}.l{layer}.b{j}"
                for i, r in enumerate(reqs)
                for layer in range(L)
                for j in range(len(r["block_ids"]))
            ]

        lb: list = []

        def make(k):
            if not lb:
                lb.extend((layer, b) for r in reqs for layer in range(L) for b in r["block_ids"])
            layer, b = lb[k]
            return page_bytes(self.kv[layer][b])

        return PagedLive(ptrs, self.page_words, names, make, [("act.h", self.act_h), ("act.r", self.act_r)])

    def dram_plan(self, fraction: float):
        """The dyn_dram cheat's set (`memory_accounting.prover.dynamic.DramEvict`): the top `fraction` of the block ids, the
        same ids in every layer (vLLM hands out low ids first, so under a light load they are free); a block of it
        is read from host memory when no live request of the frozen stage lists it, which is when the verifier
        rebuilds it as filler. A region block may span many pages (a 1 MiB block is 128 pages of 8 KiB): the table
        splits at pages, so a block mixes live pages from HBM and free ones from host memory."""
        nb, pw, L = self.n_blocks, self.page_stride_words, len(self.kv)
        k = min(nb, max(1, int(np.ceil(nb * fraction))))
        b0 = nb - k

        def free_fn(stage):
            f = np.ones(k, dtype=bool)
            ids = np.array([int(b) for r in stage.get("requests", []) for b in r["block_ids"]], dtype=np.int64)
            f[ids[ids >= b0] - b0] = False
            return [f] * L

        return [(f"kv.{layer}", b0, k, pw) for layer in range(L)], free_fn

    def capture(self, hidden: torch.Tensor, residual: torch.Tensor | None, n: int) -> None:
        """At a pause inside a step: the paused layer's inputs, one row per scheduled token, filler after (plain pool;
        an encoded one encodes them at every layer boundary, `EncodedPool.act_rows`)."""
        self.clear_act()
        self.act_h[:n].copy_(hidden[:n])
        if residual is not None:
            self.act_r[:n].copy_(residual[:n])

    def debug(self, stage: dict) -> dict:
        """A summary to compare before and after a challenge, and a sum per live page for locating a mismatch."""
        pw = self.page_stride_words
        sums = {}
        for r in stage.get("requests", []):
            idx = torch.tensor(r["block_ids"], dtype=torch.long, device=self.kv[0].device)
            for layer, k in enumerate(self.kv):
                pool = self.enc.stores[layer].tensor if self.enc is not None else page_bytes(k).view(torch.int64)
                w = pool.view(self.n_blocks, pw)[idx].sum(dim=1)
                for b, v in zip(r["block_ids"], w.tolist()):
                    sums[f"{layer}.{b}"] = v & ((1 << 64) - 1)
        return {"checksum": self.region.checksum(), "page_sums": sums}


class EncodedPool:
    """The KV pool and the activation rows held encoded (`memory_accounting.prover.kvenc`; `PagedPool(encoded=True)`).

    * Each layer's pool is an encoded store of n_blocks units, one per page: page b of layer l (its plaintext the
      page's bytes in physical order, `pw` words) is `epw` = 16 * ceil(pw / 15) words at b * epw of the layer's store.
      The stores are laid out in the memory of layers 1..L-1's pools (vLLM allocates every layer in one buffer,
      layer-major) and, for what does not fit there, one more allocation (about 1/8 of the pool).
    * Layer 0's pool is the plaintext scratch: every layer's attention reads and writes it (`rebind`). Before layer l
      runs, the pages of the step's requests are decoded from layer l's store into it (`decode_layer`); after, the
      groups holding the slots the step wrote are encoded back (`encode_layer`). A page not in the step is never read
      by attention; a page of another layer left in the scratch is never read either.
    * The activation rows ("act.h", "act.r": the hidden state and residual entering a layer, one row per scheduled
      token) are encoded at every layer boundary of a step (`act_rows`) and are encoded zeros between steps.
    * A free page and a page's unwritten slots are encoded zeros (a block is allocated as it was freed)."""

    def __init__(self, pool: PagedPool, block_bytes: int):
        from ..verifier.vllm_replay import slot_index
        from .kvenc import EncStore, enc_words, groups_of, DYNAMIC_RAW_BYTES

        self.pool = pool
        kv = pool.kv
        L, nb, pb = len(kv), pool.n_blocks, pool.page_bytes
        self.L, self.nb, self.bs = L, nb, pool.block_size
        self.pw = pool.page_words
        self.epw, self.gpp = enc_words(self.pw), groups_of(self.pw)
        self.dtype = str(kv[0].dtype).replace("torch.", "")
        self.page_shape = tuple(kv[0].shape[1:])
        self.s_base = pool._base[0]  # the plaintext scratch: layer 0's pool
        self.scratch = kv[0]
        dev = kv[0].device
        need = nb * self.epw  # words of one layer's store
        span = []  # memory of layers 1..L-1, if layer-major in one buffer
        if L > 1 and all(pool._base[layer] == pool._base[0] + layer * nb * pb for layer in range(L)):
            st = kv[0].untyped_storage()
            off = pool._base[1] - st.data_ptr()
            if 0 <= off and off + (L - 1) * nb * pb <= st.nbytes():
                u8 = torch.empty(0, dtype=torch.uint8, device=dev)
                u8.set_(st, off, ((L - 1) * nb * pb,), (1,))
                span = [u8.view(torch.int64)]
        self.stores, used, fit = [], 0, 0
        free = span[0].numel() if span else 0
        for layer in range(L):
            if used + need <= free:
                self.stores.append(EncStore(span[0][used : used + need], self.pw, nb, self.dtype, self.page_shape))
                used += need
                fit += 1
            else:
                break
        rest = L - fit
        self.extra = torch.empty(rest * need, dtype=torch.int64, device=dev) if rest else None
        for i in range(rest):
            self.stores.append(
                EncStore(self.extra[i * need : (i + 1) * need], self.pw, nb, self.dtype, self.page_shape)
            )
        self.extra_bytes = rest * need * 8
        rows, hid = pool.act_h.shape
        self.act_esize = pool.act_h.element_size()
        self.row_words = hid * self.act_esize // 8
        if hid * self.act_esize % 8:
            raise ValueError("activation rows are not whole 8-byte words")
        aw = enc_words(rows * self.row_words)
        # act.h then act.r, adjacent as in the region
        self._act_mem = torch.empty(2 * aw, dtype=torch.int64, device=dev)
        self.act_stores = [
            EncStore(
                self._act_mem[i * aw : (i + 1) * aw],
                rows * self.row_words,
                1,
                str(pool.act_h.dtype).replace("torch.", ""),
                (rows, hid),
            )
            for i in range(2)
        ]
        # the groups each slot's elements are in (a page's bytes in physical order)
        slots = slot_index(self.page_shape, kv[0][0].stride())
        element_bytes = kv[0].element_size()
        starts = np.arange(len(slots)) * element_bytes
        grp = starts // DYNAMIC_RAW_BYTES
        self.slot_groups = np.zeros((self.bs, self.gpp), dtype=bool)
        self.slot_groups[slots, grp] = True
        self.slot_groups[slots, (starts + element_bytes - 1) // DYNAMIC_RAW_BYTES] = True
        self._runs: dict = {}
        self.codec = None
        self._dec = self._enc = None
        self._act_extent = [0, 0]
        self._ents: dict = {}
        self._written_pages = ()
        self.on_kv_write = None

    def buffers(self) -> list:
        return [(f"kv.{layer}", st) for layer, st in enumerate(self.stores)] + [
            ("act.h", self.act_stores[0]),
            ("act.r", self.act_stores[1]),
        ]

    def attach(self, region) -> None:
        self.codec = region.codec
        pos = {s.name: s.offset for s in region.segments}
        self.q = [pos[f"kv.{layer}"] for layer in range(self.L)]
        self.act_q = [pos["act.h"], pos["act.r"]]

    def runs(self, s0: int, s1: int) -> list:
        """(first group, groups) runs of a page holding slots [s0, s1)."""
        k = (s0, s1)
        r = self._runs.get(k)
        if r is None:
            m = self.slot_groups[s0:s1].any(axis=0)
            idx = np.flatnonzero(m)
            if not len(idx):
                r = []
            else:
                cut = np.flatnonzero(np.diff(idx) != 1) + 1
                r = [(int(a[0]), len(a)) for a in np.split(idx, cut)]
            self._runs[k] = r
        return r

    def plan(self, reqs) -> None:
        """A step's work: reqs is [(block ids, num_computed, scheduled)] of every scheduled request. Decoded before each
        layer: every page of those requests' first num_computed + scheduled tokens (what attention reads, and every
        group it will write in); encoded after: the groups holding slots [num_computed, num_computed + scheduled)."""
        pw, epw, bs = self.pw, self.epw, self.bs
        dec, enc, written = [], [], set()
        for blocks, c, s in reqs:
            total = c + s
            nbk = -(-total // bs)
            dec.append(np.asarray(blocks[:nbk], dtype=np.uint64))
            for j in range(c // bs, nbk):
                b = int(blocks[j])
                for g0, ng in self.runs(max(c, j * bs) - j * bs, min(total, (j + 1) * bs) - j * bs):
                    enc.append((b * pw, b * epw, g0, ng))
                    written.add(b)
        b = np.concatenate(dec) if dec else np.zeros(0, dtype=np.uint64)
        rows = np.stack([b * np.uint64(pw), b * np.uint64(epw), np.zeros_like(b), np.full_like(b, self.gpp)], axis=1)
        self._dec, self._enc = self.codec.ent(rows, "dec"), self.codec.ent(enc, "enc")
        self._written_pages = tuple(sorted(written))

    def end_plan(self) -> None:
        self._dec = self._enc = None
        self._written_pages = ()

    @property
    def planned(self) -> bool:
        return self._dec is not None

    def decode_layer(self, layer: int) -> None:
        self.codec.code(self.s_base, self.stores[layer].tensor.data_ptr(), self.q[layer], self._dec, self.pw, True)

    def encode_layer(self, layer: int) -> None:
        self.codec.code(self.s_base, self.stores[layer].tensor.data_ptr(), self.q[layer], self._enc, self.pw, False)
        if layer == self.L - 1 and self.on_kv_write is not None and self._written_pages:
            self.on_kv_write(self._written_pages, scratch=self.scratch)

    def _ent_groups(self, groups: int):
        e = self._ents.get(groups)
        if e is None:
            e = self._ents[groups] = self.codec.ent([(0, 0, 0, groups)])
        return e

    def act_rows(self, hidden: torch.Tensor, residual: torch.Tensor | None, n: int) -> None:
        """The inputs of the layer about to run, n rows each: encoded into the activation stores, every group that
        holds rows now or held rows at the last boundary (zeros past n rows; a missing residual: zeros)."""
        from .kvenc import groups_of

        words = n * self.row_words
        if (
            residual is not None
            and self._act_extent[0] <= words
            and self._act_extent[1] <= words
            and words
            and self.act_q[1] == self.act_q[0] + self.act_stores[0].words
        ):
            # both stores in one launch: raw base 0 and each entry's raw offset its rows' address in words
            h, r = hidden[:n], residual[:n]
            if h.is_contiguous() and r.is_contiguous() and h.element_size() == r.element_size() == self.act_esize:
                g = groups_of(words)
                ent = self.codec.ent(
                    [(h.data_ptr() // 8, 0, 0, g), (r.data_ptr() // 8, self.act_stores[0].words, 0, g)], "act"
                )
                self.codec.code(0, self.act_stores[0].tensor.data_ptr(), self.act_q[0], ent, words, False)
                self._act_extent = [words, words]
                return
        for i, tensor in enumerate((hidden, residual)):
            words = n * self.row_words if tensor is not None else 0
            hi = max(words, self._act_extent[i])
            if hi:
                if tensor is not None:
                    tensor = tensor[:n]
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    if tensor.element_size() != self.act_esize:
                        raise ValueError(
                            f"activation rows of {tensor.element_size()}-byte elements, the store's are {self.act_esize}"
                        )
                self.codec.code(
                    tensor.data_ptr() if words else 0,
                    self.act_stores[i].tensor.data_ptr(),
                    self.act_q[i],
                    self._ent_groups(groups_of(hi)),
                    words,
                    False,
                )
            self._act_extent[i] = words

    def clear_act(self) -> None:
        """No step in flight: the activation stores are encoded zeros (only the groups that held rows are rewritten)."""
        from .kvenc import groups_of

        for i in range(2):
            if self._act_extent[i]:
                self.codec.code(
                    0,
                    self.act_stores[i].tensor.data_ptr(),
                    self.act_q[i],
                    self._ent_groups(groups_of(self._act_extent[i])),
                    0,
                    False,
                )
                self._act_extent[i] = 0

    def init_act(self, region) -> None:
        region.clear("act.h")
        region.clear("act.r")
        self._act_extent = [0, 0]

    def rebind(self, layers) -> int:
        """Point every attention layer that holds one of the pool's layers at the scratch (layer 0's pool); returns how
        many it rebound. From here on only the stores hold the other layers' KV."""
        bases = {k.data_ptr() for k in self.pool.kv}
        n = 0
        for layer in layers:
            kvc = getattr(layer, "kv_cache", None)
            if isinstance(kvc, torch.Tensor) and kvc.numel() and kvc.data_ptr() in bases:
                layer.kv_cache = self.scratch
                n += 1
        return n

    def _page_ptrs(self, stage: dict) -> np.ndarray:
        reqs = stage.get("requests", [])
        base = np.array([st.tensor.data_ptr() for st in self.stores], dtype=np.uint64)[:, None]
        eb = np.uint64(self.epw * 8)
        parts = [(base + np.asarray(r["block_ids"], dtype=np.uint64)[None, :] * eb).ravel() for r in reqs]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)

    def live_pages(self, stage: dict):
        """The live region: each live request's encoded pages (stage order, layer-major, block-table order), then the
        activation stores."""
        from .dynamic import PagedLive
        from .kvenc import EncStore

        reqs = stage.get("requests", [])
        L, epw = self.L, self.epw

        def names():
            return [
                f"r{i}.l{layer}.b{j}"
                for i, r in enumerate(reqs)
                for layer in range(L)
                for j in range(len(r["block_ids"]))
            ]

        lb: list = []

        def make(k):
            if not lb:
                lb.extend((layer, b) for r in reqs for layer in range(L) for b in r["block_ids"])
            layer, b = lb[k]
            return self.stores[layer].tensor[b * epw : (b + 1) * epw].view(torch.uint8)

        return PagedLive(
            self._page_ptrs(stage),
            epw,
            names,
            make,
            [("act.h", self.act_stores[0]), ("act.r", self.act_stores[1])],
            enc=EncStore(None, self.pw, 1, self.dtype, self.page_shape),
        )

    def live_segments(self, stage: dict) -> list:
        from .kvenc import EncStore

        epw, segs = self.epw, []
        for i, r in enumerate(stage.get("requests", [])):
            for layer in range(self.L):
                for j, b in enumerate(r["block_ids"]):
                    segs.append(
                        (
                            f"r{i}.l{layer}.b{j}",
                            EncStore(
                                self.stores[layer].tensor[b * epw : (b + 1) * epw],
                                self.pw,
                                1,
                                self.dtype,
                                self.page_shape,
                            ),
                        )
                    )
        return segs + [("act.h", self.act_stores[0]), ("act.r", self.act_stores[1])]


# ---- the region inside the vLLM worker


class VllmDynamic:
    """Prover-side state, hooks and region, created inside the vLLM worker."""

    def __init__(
        self,
        worker,
        keys,
        block_bytes: int = 1 << 16,
        pad_seed: int = 0,
        include_bytes: bool = False,
        model_name: str = "",
        gpu: int | None = None,
        encoded: bool = True,
    ):
        """gpu: this GPU's index as the verifier knows it (the bundle's), which keys the dynamic filler. The
        worker's device index is not it: with one GPU per engine it is 0 on every GPU of the host. encoded (v0.17,
        the default): the pool and the activation rows are held encoded (`EncodedPool`); False: plain, as before."""
        from vllm import envs

        self.worker = worker
        self.runner = runner = worker.model_runner
        problems = []
        if not envs.VLLM_BATCH_INVARIANT:
            problems.append("VLLM_BATCH_INVARIANT is not set")
        if not hasattr(runner, "input_batch") or not hasattr(runner, "requests"):
            problems.append("not the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=0)")
        vc = worker.vllm_config
        if vc.cache_config.enable_prefix_caching:
            problems.append("prefix caching is on")
        if not vc.model_config.enforce_eager:
            problems.append("enforce_eager is off (CUDA graphs cannot pause inside a step)")
        if vc.scheduler_config.enable_chunked_prefill:
            problems.append("chunked prefill is on (chunk invariance is not covered by vLLM's tests)")
        if getattr(vc.scheduler_config, "async_scheduling", False):
            problems.append("async scheduling is on")
        if vc.parallel_config.tensor_parallel_size != 1 or vc.speculative_config is not None:
            problems.append("tensor parallelism or speculative decoding")
        if problems:
            raise RuntimeError("dynamic memory under vLLM needs: " + "; ".join(problems))
        self.model = runner.get_model()
        self.layers = layer_modules(self.model)
        self.kv = list(runner.kv_caches)
        if len(self.kv) != len(self.layers):
            raise RuntimeError(
                f"{len(self.kv)} KV caches for {len(self.layers)} layers (hybrid or shared-KV models are not supported)"
            )
        from .dynfill import MASKS

        kcc = getattr(runner, "kv_cache_config", None)
        if getattr(kcc, "needs_kv_cache_zeroing", False):
            raise RuntimeError(
                "vLLM zeroes new KV blocks for this model (Mamba layers or a mixed-precision KV cache): "
                "unwritten slots must keep their filler"
            )
        kv_dtypes = {str(k.dtype).replace("torch.", "") for k in self.kv}
        if len(kv_dtypes) != 1 or not kv_dtypes <= set(MASKS):
            raise RuntimeError(
                f"KV cache dtype {sorted(kv_dtypes)}: one of {sorted(MASKS)} is needed (filler in unwritten slots is kept finite)"
            )
        self.device = self.kv[0].device
        hidden = vc.model_config.get_hidden_size()
        rows = vc.scheduler_config.max_num_batched_tokens
        dtype = vc.model_config.dtype
        self.act_h = torch.zeros(rows, hidden, dtype=dtype, device=self.device)
        self.act_r = torch.zeros(rows, hidden, dtype=dtype, device=self.device)
        gpu = int(gpu) if gpu is not None else (self.device.index or 0)
        self.pool = PagedPool(self.kv, self.act_h, self.act_r, block_bytes, pad_seed, gpu, encoded=encoded)
        self.region = self.pool.region
        self.enc = self.pool.enc
        if self.enc is not None:  # every layer's attention on the scratch (layer 0's pool) from here on
            n = self.enc.rebind(vc.compilation_config.static_forward_context.values())
            if n != len(self.kv):
                raise RuntimeError(
                    f"rebound {n} attention layers to the plaintext scratch, the pool has {len(self.kv)} layers"
                )
            runner.kv_caches[:] = [self.enc.scratch] * len(self.kv)
        t0 = time.perf_counter()
        self.pool.init_filler()
        torch.cuda.synchronize(self.device)
        self.init_seconds = time.perf_counter() - t0
        self.pause = PauseController()
        self.so = None
        self.step = 0
        self.in_flight = False
        self.layer = 0
        layout = getattr(vc.cache_config, "kv_cache_layout", None)
        self.info = {
            "kind": "vllm",
            "name": model_name or vc.model_config.model,
            "dtype": str(dtype).replace("torch.", ""),
            "kv_dtype": kv_dtypes.pop(),
            "block_size": self.pool.block_size,
            "kv_layout": str(layout),
            "page_shape": list(self.kv[0].shape[1:]),
            "page_stride": list(self.kv[0][0].stride()),
            "gpu_name": torch.cuda.get_device_name(self.device),
            "gpu": gpu,
            "n_blocks": self.pool.n_blocks,
            "page_bytes": self.pool.page_bytes,
            "pool_bytes": self.pool.page_bytes * self.pool.n_blocks * len(self.kv),
            "vllm": str(__import__("vllm").__version__),
            "torch": str(torch.__version__),
            "act_rows": rows,
            "hidden": hidden,
            "max_model_len": int(vc.model_config.max_model_len),
            "encoded": self.enc is not None,
            "store_extra_bytes": self.enc.extra_bytes if self.enc is not None else 0,
        }
        self._stage: dict = {}
        self.dyn = DynamicResponder(
            self.region,
            self.stage,
            self.pause,
            keys,
            model_info=self.info,
            on_freeze=self.on_freeze,
            debug_fn=(lambda: self.pool.debug(self._stage)) if include_bytes else None,
            live_fn=self.pool.live_pages,
            dram_fn=self.pool.dram_plan,
        )
        if self.enc is not None:
            self.enc.on_kv_write = self.dyn.update_timing_pages
        self._hooks = [layer.register_forward_pre_hook(self._layer_pre(i)) for i, layer in enumerate(self.layers)]
        if self.enc is not None:
            self._hooks += [layer.register_forward_hook(self._layer_post(i)) for i, layer in enumerate(self.layers)]
        self._plan_step = -1
        orig = worker.execute_model

        def execute_model(scheduler_output, *args, **kw):
            self.pause.acquire_inference()  # waits while an idle challenge holds the region
            try:
                self.in_flight = False
                self.pause.checkpoint(self._sync)  # between steps: the previous step is complete
                self.pool.begin_step(scheduler_output, self.runner.requests)
                self.so = scheduler_output
                self.step += 1
                self.in_flight = True
                try:
                    return orig(scheduler_output, *args, **kw)
                finally:
                    self.in_flight = False
                    self.pool.end_step(scheduler_output, self.runner.requests)
                    if self.enc is not None:  # between steps the activation rows are encoded zeros
                        self.enc.end_plan()
                        self.enc.clear_act()
            finally:
                self.pause.lock.release()

        worker.execute_model = execute_model

        # vLLM calls sample_tokens after execute_model releases the pause lock.
        # Sampling launches GPU work; it must also wait while a challenge owns
        # the region, and a freeze must wait for sampling before sending READY.
        orig_sample = worker.sample_tokens

        def sample_tokens(*args, **kw):
            if self.in_flight:
                raise RuntimeError("sample_tokens ran before execute_model completed")
            self.pause.acquire_inference()
            try:
                return orig_sample(*args, **kw)
            finally:
                self.pause.lock.release()

        worker.sample_tokens = sample_tokens
        print("posm: sampling protected by the inference pause lock", flush=True)

    def _sync(self):
        torch.cuda.synchronize(self.device)

    def _layer_pre(self, index: int):
        if self.enc is not None:
            return self._layer_pre_encoded(index)

        def hook(module, args):
            self.layer = index
            self.pause.checkpoint(lambda: self._capture(args))

        return hook

    def _layer_pre_encoded(self, index: int):
        """Encoded pool: the layer's inputs into the activation stores, then the pause point (the region is complete
        there: nothing is left to write at a freeze), then the layer's pages into the plaintext scratch. Only in a
        step (`execute_model`); a dummy or profiling forward runs on the scratch as it is."""

        def hook(module, args):
            self.layer = index
            enc = self.enc
            step = self.in_flight and self.so is not None
            if step:
                if self._plan_step != self.step:  # first layer of the step: vLLM has updated its requests
                    self._plan()
                enc.act_rows(args[1], args[2] if len(args) > 2 else None, int(self.so.total_num_scheduled_tokens))
            self.pause.checkpoint(self._sync)
            if step and enc.planned:
                enc.decode_layer(index)

        return hook

    def _layer_post(self, index: int):
        def hook(module, args, out):
            if self.in_flight and self.enc.planned:
                self.enc.encode_layer(index)

        return hook

    def _plan(self) -> None:
        so, reqs = self.so, self.runner.requests
        work = []
        for rid, n in so.num_scheduled_tokens.items():
            st = reqs.get(rid)
            if st is None or not st.block_ids:
                raise RuntimeError(f"request {rid} is scheduled but the worker has no block list for it")
            work.append((list(st.block_ids[0]), int(st.num_computed_tokens), int(n)))
        self.enc.plan(work)
        self._plan_step = self.step

    def _capture(self, args) -> None:
        torch.cuda.synchronize(self.device)
        self.pool.capture(args[1], args[2] if len(args) > 2 else None, int(self.so.total_num_scheduled_tokens))

    def stage(self) -> dict:
        """Called frozen. In a step: this step's batch and the other live requests; between steps or idle:
        every live request, complete."""
        order = [r for r in self.runner.input_batch.req_ids if r is not None]
        self._stage = self.pool.stage(
            self.so, self.runner.requests, order, self.in_flight, self.layer, self.step, len(self.layers)
        )
        return self._stage

    def on_freeze(self, stage: dict) -> None:
        if not stage["in_flight"] and self.enc is None:
            self.pool.clear_act()  # no step in flight: the activation rows are free (encoded: already)

    def remove(self):
        for h in self._hooks:
            h.remove()


def _serve_main(args):
    """Prover: vLLM with stream-decoded weights (static) and the dynamic region; generates under load."""
    from .responder import require_free_ports

    require_free_ports(args.port)  # before vLLM starts: its setup takes minutes, and the responder binds last
    apply_env()
    from vllm import LLM, SamplingParams

    phases = []
    if args.timing_phases:
        if not args.timing_workload or not args.timing_phase_start_file:
            raise ValueError("--timing-phases needs --timing-workload and --timing-phase-start-file")
        for item in args.timing_phases.split(","):
            parts = item.split(":")
            if len(parts) not in (3, 4) or parts[2] not in ("on", "off") or (len(parts) == 4 and parts[3] not in ("python", "native")):
                raise ValueError("timing phases must be concurrency:seconds:on|off[:python|native]")
            concurrency, seconds = int(parts[0]), float(parts[1])
            if concurrency < 1 or seconds <= 0:
                raise ValueError("timing phase concurrency and seconds must be positive")
            phases.append((concurrency, seconds, parts[2] == "on", len(parts) == 4 and parts[3] == "native"))

    model_settings = settings(args.model, args.max_model_len, args.gpu_memory_utilization)
    max_concurrency = max([args.timing_concurrency, *(p[0] for p in phases)])
    if args.timing_workload and max_concurrency >= 256:
        model_settings["max_num_seqs"] = max_concurrency
    if args.model_revision:
        model_settings["revision"] = args.model_revision
    llm = LLM(**model_settings)
    prompts = ["The quick brown fox", "In 1969, the first", "Memory accounting checks", "def fibonacci(n):"]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    vanilla = [o.outputs[0].text for o in llm.generate(prompts[:2], sp, use_tqdm=False)]
    print("vanilla:", vanilla, flush=True)
    cap = int(args.filler_gb * (1 << 30)) if args.filler_gb > 0 else None
    # the dynamic region first: its encoded pool takes memory beyond vLLM's pool, which the static filler must leave
    print(
        "dynamic:",
        llm.collective_rpc(
            "posm_dynamic_setup",
            args=(args.dyn_block, args.pad_seed or args.filler_seed, args.evidence_bytes),
            kwargs={"gpu": args.gpu, "encoded": not args.plain_dynamic, "filler_seed_hex": args.filler_seed},
        )[0],
        flush=True,
    )
    print(
        "converted:",
        llm.collective_rpc(
            "posm_convert",
            args=(args.enc_key, args.filler_seed, args.scratch_fraction),
            kwargs={
                "filler_max_bytes": cap,
                "reserve_bytes": int(args.scratch_gb * (1 << 30)) if args.scratch_gb > 0 else -1,
                "gpu": args.gpu,
            },
        )[0],
        flush=True,
    )
    after = [o.outputs[0].text for o in llm.generate(prompts[:2], sp, use_tqdm=False)]
    print("output after setup:", after, flush=True)
    if after != vanilla:  # as intserve: a weight decoded at the wrong time changes the text, not the replay
        raise SystemExit("posm setup (stream-decode, the dynamic region) changed vLLM's greedy output")
    if args.reuse_bundle:
        if not (Path(args.bundle) / "manifest.json").is_file():
            raise RuntimeError("--reuse-bundle requires an existing bundle manifest")
        print("verifier bundle: reused", args.bundle, flush=True)
    else:
        print(
            "verifier bundle:",
            llm.collective_rpc("posm_export", args=(args.bundle, args.filler_seed), kwargs={"gpu": args.gpu})[0],
            flush=True,
        )
    print(
        llm.collective_rpc(
            "posm_serve",
            args=(args.port, None, args.strategies, None, True, 100.0),
            kwargs={
                "evict_fraction": args.evict_fraction,
                "tamper_fraction": args.tamper_fraction,
                "dyn_evict_fraction": args.dyn_evict_fraction,
                "verifier_addrs": args.verifier_addr,
                "compact_share_fraction": args.compact_share_fraction,
                "dynamic_timing_shares": args.dynamic_timing_shares,
            },
        )[0],
        flush=True,
    )
    print(llm.collective_rpc("posm_hbm")[0], flush=True)
    trace_path = os.environ.get("POSM_TIMING_TRACE_OUT")
    if trace_path:
        llm.collective_rpc("posm_enable_timing_trace")
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    if args.stress_tokens:  # one full-size step under the filled memory: the engine's peak must fit
        long = list(range(1000, 1000 + args.stress_tokens))
        o = llm.generate([{"prompt_token_ids": long}], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
        print(
            f"stress: a {args.stress_tokens}-token prefill ran under the filled memory ({len(o[0].outputs[0].token_ids)} tokens out); "
            + llm.collective_rpc("posm_hbm")[0],
            flush=True,
        )
    t0 = time.perf_counter()
    toks = 0
    timing_batches = 0
    phase_index = -1
    phase_started = None
    phases_complete = False
    try:
        while True:
            concurrency = args.timing_concurrency
            if phases and Path(args.timing_phase_start_file).exists():
                advance = phase_index < 0 or (phase_index + 1 < len(phases)
                    and time.monotonic() - phase_started >= phases[phase_index][1])
                if advance:
                    phase_index += 1
                    concurrency, _, gc_enabled, native_wait = phases[phase_index]
                    actual_gc = llm.collective_rpc("posm_set_cyclic_gc", args=(gc_enabled,))[0]
                    if actual_gc != gc_enabled:
                        raise RuntimeError("worker did not apply cyclic GC phase setting")
                    actual_wait = llm.collective_rpc("posm_set_native_wait", args=(native_wait,))[0]
                    if actual_wait != native_wait:
                        raise RuntimeError("worker did not apply response wait phase setting")
                    phase_started = time.monotonic()
                    print(f"timing diagnostic phase: index={phase_index} concurrency={concurrency} "
                          f"cyclic_gc={'on' if gc_enabled else 'off'} wait={'native' if native_wait else 'python'} "
                          f"unix={time.time():.6f}", flush=True)
                concurrency = phases[phase_index][0]
                if (phase_index == len(phases) - 1 and not phases_complete
                    and time.monotonic() - phase_started >= phases[phase_index][1]):
                    phases_complete = True
                    print(f"timing diagnostic phases complete: unix={time.time():.6f}", flush=True)
            if args.request_gap_s:
                time.sleep(args.request_gap_s)  # the engine idles: exercises the idle freeze
            if args.timing_workload:
                requests = [{"prompt_token_ids": [1000 + (j * 7919 + k * 31) % 50000 for k in range(256)]}
                            for j in range(concurrency)]
                outs = llm.generate(requests, SamplingParams(temperature=0.8, max_tokens=48,
                                                            ignore_eos=True, seed=1), use_tqdm=False)
                if timing_batches == 0:
                    print(f"timing workload: completed {len(outs)} requests in one batch; "
                          f"{sum(len(o.outputs[0].token_ids) for o in outs)} generated tokens", flush=True)
                timing_batches += 1
                if trace_path:
                    rows = llm.collective_rpc("posm_take_timing_trace")[0]
                    with Path(trace_path).open("a") as stream:
                        for row in rows:
                            stream.write(json.dumps(row) + "\n")
            else:
                outs = llm.generate(prompts * 4, SamplingParams(temperature=0.8, max_tokens=64, seed=1),
                                    use_tqdm=False)
            toks += sum(len(o.outputs[0].token_ids) for o in outs)
            if time.perf_counter() - t0 > 10:
                info = llm.collective_rpc("posm_info")[0]
                print(
                    f"{toks / (time.perf_counter() - t0):.0f} tok/s; {info.get('served')} challenges served; dynamic {info.get('dynamic')}; {info.get('hbm')}",
                    flush=True,
                )
                t0 = time.perf_counter()
                toks = 0
    except KeyboardInterrupt:
        print("\nstopping", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="dynamic memory under vLLM, prover side: serve challenges under generation load"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--compact-share-fraction", type=float, default=None,
                    help="share of static blocks in a selected weight interval for compact timing simulations")
    ap.add_argument("--timing-workload", action="store_true",
                    help="repeat batches of 256-prompt, 48-generated-token requests")
    ap.add_argument("--timing-concurrency", type=int, default=16,
                    help="requests per timing batch (default 16; 256 for the high-load experiment)")
    ap.add_argument("--timing-phases", default=None,
                    help="diagnostic workload phases: concurrency:seconds:cyclic-GC-on|off, comma-separated")
    ap.add_argument("--timing-phase-start-file", default=None,
                    help="diagnostic phases begin at the next batch boundary after this file appears")
    ap.add_argument("--enc-key", required=True)
    ap.add_argument(
        "--filler-seed", required=True, help="hex seed supplied by the verifier (memory_accounting.setup.cli filler-seed)"
    )
    ap.add_argument("--pad-seed", default=None)
    ap.add_argument(
        "--scratch-fraction", type=float, default=0.0, help="unfilled share of total HBM (on top of --scratch-gb)"
    )
    ap.add_argument(
        "--scratch-gb",
        type=float,
        default=0.0,
        help="memory left unfilled for the engine's own use (activations, workspaces): "
        "the prover's scratch; 0 = vLLM's profiled activation peak + 1 GiB",
    )
    ap.add_argument(
        "--strategies",
        default="honest",
        help="cheats the prover offers as postures: evict, evict_dram, tamper, dyn_dram work with the GPU full",
    )
    ap.add_argument(
        "--verifier-addr", action="append", default=None, help="answer only datagrams from this address (repeatable)"
    )
    ap.add_argument("--evict-fraction", type=float, default=0.10)
    ap.add_argument("--tamper-fraction", type=float, default=0.10)
    ap.add_argument("--dynamic-timing-shares", action="store_true",
                    help="apply all five 1%% timing postures to the dynamic KV region as well")
    ap.add_argument(
        "--dyn-evict-fraction",
        type=float,
        default=0.5,
        help="dyn_dram: share of the KV pool's block ids (the top ones, every layer) "
        "whose filler is kept in pinned host memory and read from there while free (that much host memory, pinned at start)",
    )
    ap.add_argument(
        "--stress-tokens",
        type=int,
        default=0,
        help="after setup, run one prefill of this many tokens to show the engine's peak fits",
    )
    ap.add_argument(
        "--filler-gb",
        type=float,
        default=0.0,
        help="cap the static filler (GiB); 0 = fill all free memory but the scratch",
    )
    ap.add_argument(
        "--dyn-block",
        default="64K",
        help="the dynamic region's block (its pad to whole blocks); the verifier's --dyn-B",
    )
    ap.add_argument("--evidence-bytes", action="store_true")
    ap.add_argument("--bundle", default="verifier_bundle_vllm_dyn")
    ap.add_argument("--reuse-bundle", action="store_true", help="diagnostic: use an existing verifier bundle")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--request-gap-s", type=float, default=0.0, help="idle time between batches of requests")
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="this GPU's index as the verifier knows it (keys the static and dynamic filler; the bundle's GPU entry)",
    )
    ap.add_argument(
        "--plain-dynamic",
        action="store_true",
        help="hold the KV pool and activation rows plain, as before v0.17 (the live region is then not credited)",
    )
    _serve_main(ap.parse_args(argv))


if __name__ == "__main__":
    main()
