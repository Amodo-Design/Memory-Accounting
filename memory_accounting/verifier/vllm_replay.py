"""Dynamic memory under vLLM, verifier side: replay a stage record on the same GPU type (batch-invariant kernels),
rebuild the region it describes and recompute the chain.

Verifier (`python -m memory_accounting.verifier.vllm_replay replay`, a vLLM instance with the same model, dtype, block size and
KV layout, on the same GPU type, batch-invariant): for each request of a record, one prefill of its
tokens up to the stage (`num_computed + scheduled`), capturing the paused layer's input rows for the
step's tokens and, after the forward, the request's pages. Every slot the replay wrote and the prover had
written (this step's tokens only in layers before the paused one) is the replayed KV; every other word of
the pool, unwritten slots of live blocks included, is filler recomputed at the prover's position
(`VllmRegionView`, built only where a chain reads). So no slot the prover holds is credited unless the
model recomputes it from the tokens or it is filler. The chain runs over the view; the digest must match.
What the view is made of is the verifier's: the region's layout is rebuilt from the stage and the verifier's
own engine (block size, layers, page size, activation rows of its max_num_batched_tokens, context length;
`check_engine`, `pool_layout`, `live_layout`) and a record must carry exactly that layout. A live record is
replayed from the stage of the main record of the same freeze when the caller pairs them (`verify(main=)`;
`replay --log` pairs them through the verifier's log), and fails if its own stage is another.

That replay recomputes a request's history as one prefill, while the prover computed it as a prefill
plus decode steps, in a batch with other requests. It is exact only if vLLM's batch-invariant mode
makes a token's results independent of the batch and of prefill-versus-decode. vLLM's own tests check
batch invariance and decode-versus-prefill logprobs bitwise, but only with prompts under 128 tokens,
and with FLASH_ATTN the property fails past one KV tile (see below); chunked prefill is not covered, so
the prover runs with it off. The `selftest` command measures the property on your GPU first.

Settings both sides need: memory_accounting.models.vllm. The prover side: memory_accounting.prover.vllm_serve.

  python -m memory_accounting.verifier.vllm_replay selftest --model <dir> --device-prover 0 --device-verifier 1
  python -m memory_accounting.verifier.vllm_replay replay --evidence evidence/ --model <dir> [--manifest bundle/manifest.json] [--log live.jsonl --pad-seed <hex>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..models.vllm import ENV, apply_env, physical_order, settings
from ..prover.dynamic import layer_modules


def masked_page_bytes(page: torch.Tensor, valid: int, slot_dim: int = 1) -> np.ndarray:
    """A page's bytes in physical order with slots >= valid zeroed. Verifier side: the replay engine's
    unwritten slots hold whatever its pool held; zeroing them makes the bytes it returns deterministic, and
    the driver replaces them with the filler at the prover's position (`merge_page`)."""
    t = page.clone()
    if valid < t.shape[slot_dim]:
        t.narrow(slot_dim, valid, t.shape[slot_dim] - valid).zero_()
    return t.permute(*physical_order(page)).contiguous().reshape(-1).view(torch.uint8).cpu().numpy()


def written_slots(num_computed: int, scheduled: int, act_layer: int, in_flight: bool, layer: int) -> int:
    """Slots of a request the prover had written in this layer: the step's tokens only in the layers before
    the paused one."""
    return num_computed + (scheduled if (not in_flight or layer < act_layer) else 0)


def page_valid(written: int, j: int, bs: int) -> int:
    """Written slots of the request's j-th block."""
    return max(0, min(bs, written - j * bs))


def replay_pages(
    kv: list[torch.Tensor], blocks: list[int], num_computed: int, scheduled: int, act_layer: int, in_flight: bool
) -> list[np.ndarray]:
    """Verifier: one replayed request's pages, layer-major. The replay ran every layer over all
    num_computed + scheduled tokens; the prover, frozen before layer act_layer, had written this step's
    tokens only in the layers before it. The slots it had not written are zeroed here and become the
    filler at the prover's position in `VllmRegionView` (the prover's block kept its filler there)."""
    bs = kv[0].shape[2]
    pages = []
    for layer, k in enumerate(kv):
        written = written_slots(num_computed, scheduled, act_layer, in_flight, layer)
        for j, b in enumerate(blocks):
            pages.append(masked_page_bytes(k[b], page_valid(written, j, bs)))
    return pages


def slot_index(page_shape, page_stride, slot_dim: int = 1) -> np.ndarray:
    """For each element of a page in physical byte order (`page_bytes`, `masked_page_bytes`), the slot it
    belongs to (the page's dim `slot_dim`, as in `masked_page_bytes`)."""
    shape, stride = [int(x) for x in page_shape], [int(x) for x in page_stride]
    order = sorted(range(len(shape)), key=lambda d: -stride[d])  # `physical_order`
    return np.ascontiguousarray(np.indices(shape)[slot_dim].transpose(order)).reshape(-1)


def merge_page(replayed: np.ndarray, filler: np.ndarray, valid: int, slots: np.ndarray) -> np.ndarray:
    """A live page as the prover holds it: the replayed elements of slots < valid, the filler elsewhere
    (uint64 words in physical order; slots from `slot_index`)."""
    n = len(slots)
    if valid >= int(slots.max(initial=-1)) + 1:
        return replayed
    if valid <= 0:
        return filler
    esize = replayed.nbytes // n
    dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(esize)
    if dt is None or esize * n != replayed.nbytes:
        raise ValueError(f"page of {replayed.nbytes} bytes is not {n} elements of 1, 2, 4 or 8 bytes")
    return np.where(slots < valid, replayed.view(dt), filler.view(dt)).astype(dt, copy=False).view(np.uint64)


def check_stage(st: dict) -> None:
    """The fields of a vLLM stage the replay trusts, checked before anything is replayed (the prover writes
    them): every request has exactly num_computed + scheduled tokens, the blocks those tokens need, distinct
    block ids inside the pool, and a paused layer inside the model. A stage that fails cannot be replayed
    (ValueError): the record fails."""
    bs, L, nb = int(st["block_size"]), int(st["n_layers"]), int(st["n_blocks"])
    if bs < 1 or L < 1 or nb < 1 or int(st["page_words"]) < 1:
        raise ValueError("stage has no pool (block size, layers, blocks or page size below 1)")
    if st["in_flight"] and not 0 <= int(st["act_layer"]) < L:
        raise ValueError(f"paused layer {st['act_layer']} outside the model's {L} layers")
    owner: dict[int, int] = {}
    for i, r in enumerate(st["requests"]):
        c, s_, toks = int(r["num_computed"]), int(r["scheduled"]), r["tokens"]
        total = c + s_
        if c < 0 or s_ < 0 or total < 1 or len(toks) != total or int(r["num_tokens"]) != total:
            raise ValueError(
                f"request {i}: {len(toks)} tokens for num_computed {c} + scheduled {s_} (num_tokens {r['num_tokens']})"
            )
        if not all(isinstance(t, int) and t >= 0 for t in toks):
            raise ValueError(f"request {i}: tokens are not token ids")
        need = -(-total // bs)
        if int(r["nblocks"]) != need or len(r["block_ids"]) != need:
            raise ValueError(
                f"request {i}: {len(r['block_ids'])} blocks (nblocks {r['nblocks']}) for {total} tokens, {need} needed"
            )
        for b in r["block_ids"]:
            b = int(b)
            if not 0 <= b < nb:
                raise ValueError(f"request {i}: block {b} outside the pool of {nb}")
            if b in owner:
                raise ValueError(f"block {b} listed by requests {owner[b]} and {i}")
            owner[b] = i


ACT_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def check_engine(st: dict, mi: dict, engine: dict) -> None:
    """The stage and the record's model facts against the verifier's own engine (`VllmReplayer.engine`): block
    size (the replay's block list is cut to the stage's block count, which holds only with equal blocks),
    layers, page size, and the context length, which sets the activation rows on both sides (records before
    v0.12 do not carry it; their activation rows are checked through the layout). ValueError if one differs."""
    for k, mine, theirs in (
        ("block size", engine["block_size"], st.get("block_size")),
        ("block size", engine["block_size"], mi.get("block_size")),
        ("layers", engine["n_layers"], st.get("n_layers")),
        ("page words", engine["page_words"], st.get("page_words")),
    ):
        if theirs != mine:
            raise ValueError(f"the record's {k} is {theirs!r}, the verifier's engine has {mine}")
    for k in ("max_model_len", "act_rows", "hidden"):
        if k in mi and mi[k] != engine[k]:
            raise ValueError(
                f"the record's {k} is {mi[k]!r}, the verifier's engine has {engine[k]} (prover and verifier must use the same settings)"
            )


def engine_facts(vc, info: dict) -> dict:
    """What the verifier's own engine says a record's region is made of (its vLLM config and the replay
    worker's KV facts, `posm_replay_setup`): block size, layers, page size, activation rows (the engine's
    max_num_batched_tokens, which `settings` sets to the context length on both sides), hidden size, model
    dtype, context length."""
    mc = vc.model_config
    esize = torch.empty(0, dtype=getattr(torch, info["kv_dtype"])).element_size()
    return {
        "block_size": int(info["block_size"]),
        "n_layers": int(mc.get_num_layers(vc.parallel_config)),
        "page_words": int(np.prod(info["page_shape"])) * esize // 8,
        "act_rows": int(vc.scheduler_config.max_num_batched_tokens),
        "hidden": int(mc.get_hidden_size()),
        "dtype": str(info["dtype"]),
        "max_model_len": int(mc.max_model_len),
        "kv_dtype": str(info["kv_dtype"]),
        "page_shape": [int(x) for x in info["page_shape"]],
    }


def act_layout(engine: dict, encoded: bool = False) -> list[dict]:
    """The activation buffers as the verifier's engine sizes them: max_num_batched_tokens rows of the hidden size
    (encoded: their encoded stores, `memory_accounting.prover.vllm_serve.EncodedPool`)."""
    rows, hid, dt = int(engine["act_rows"]), int(engine["hidden"]), engine["dtype"]
    nbytes = rows * hid * ACT_BYTES[dt]
    if nbytes % 8:
        raise ValueError("activation rows are not a whole number of words")
    if encoded:
        from ..prover.kvenc import EncStore

        return [EncStore(None, nbytes // 8, 1, dt, (rows, hid)).layout(n) for n in ("act.h", "act.r")]
    return [{"name": n, "dtype": dt, "shape": [rows, hid], "words": nbytes // 8} for n in ("act.h", "act.r")]


def page_store(engine: dict, units: int):
    """The encoded store of `units` KV pages as the verifier's engine shapes them (geometry only)."""
    from ..prover.kvenc import EncStore

    if "page_shape" not in engine or "kv_dtype" not in engine:
        raise ValueError(
            "the verifier's engine facts have no page shape or KV dtype: an encoded pool cannot be laid out"
        )
    return EncStore(None, int(engine["page_words"]), units, engine["kv_dtype"], tuple(engine["page_shape"]))


def _padded(layout: list[dict], block_bytes: int) -> list[dict]:
    """A region's layout with its one trailing pad to whole blocks (at least one), as `DynamicRegion` adds it."""
    live, bw = sum(e["words"] for e in layout), block_bytes // 8
    total = max(bw, -(-live // bw) * bw)
    return layout + (
        [{"name": "pad", "dtype": "int64", "shape": [total - live], "words": total - live}] if total > live else []
    )


def pool_layout(st: dict, engine: dict, block_bytes: int) -> list[dict]:
    """The main region's layout from the stage and the verifier's engine (`PagedPool`): every layer's pool of
    n_blocks pages, the activation rows, one pad. A record must carry exactly this."""
    L, nb, pw = int(st["n_layers"]), int(st["n_blocks"]), int(st["page_words"])
    if st.get("encoded"):
        return _padded(
            [page_store(engine, nb).layout(f"kv.{layer}") for layer in range(L)] + act_layout(engine, True), block_bytes
        )
    kv = [{"name": f"kv.{layer}", "dtype": "uint8", "shape": [nb * pw * 8], "words": nb * pw} for layer in range(L)]
    return _padded(kv + act_layout(engine), block_bytes)


def live_layout(st: dict, engine: dict, block_bytes: int) -> list[dict]:
    """The live region's layout from the stage and the verifier's engine: each live request's pages (stage
    order, layer-major, block-table order), the activation rows, one pad."""
    pw = int(st["page_words"])
    names = [
        f"r{i}.l{layer}.b{j}"
        for i, r in enumerate(st["requests"])
        for layer in range(int(st["n_layers"]))
        for j in range(int(r["nblocks"]))
    ]
    if st.get("encoded"):
        ps = page_store(engine, 1)
        return _padded([ps.layout(n) for n in names] + act_layout(engine, True), block_bytes)
    pages = [{"name": n, "dtype": "uint8", "shape": [pw * 8], "words": pw} for n in names]
    return _padded(pages + act_layout(engine), block_bytes)


class ReplayCapture:
    """Inside the verifier's vLLM worker. `arm()` before generating one request; the layer hook takes the
    paused layer's input rows for the step's tokens; a model post-hook takes the request's pages, with the
    slots the prover had not written yet zeroed (the driver puts the prover's filler there)."""

    def __init__(self, worker):
        self.runner = worker.model_runner
        self.model = self.runner.get_model()
        self.layers = layer_modules(self.model)
        self.kv = list(self.runner.kv_caches)
        self.req: dict | None = None
        self.out: dict = {}
        self._hooks = [layer.register_forward_pre_hook(self._layer_pre(i)) for i, layer in enumerate(self.layers)]
        self._hooks.append(self.model.register_forward_hook(self._post))

    def arm(self, num_computed: int, scheduled: int, act_layer: int, in_flight: bool, nblocks: int) -> None:
        self.req = {"c": num_computed, "s": scheduled, "L": act_layer, "in_flight": in_flight, "nblocks": nblocks}
        self.out = {}

    def _layer_pre(self, index: int):
        def hook(module, args):
            r = self.req
            if r is None or not r["in_flight"] or not r["s"] or index != r["L"] or "act_h" in self.out:
                return
            c, s = r["c"], r["s"]
            hidden, residual = args[1], (args[2] if len(args) > 2 else None)
            self.out["act_h"] = (
                hidden[c : c + s].detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy().copy()
            )
            self.out["act_r"] = (
                residual[c : c + s].detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy().copy()
                if residual is not None
                else None
            )

        return hook

    def _post(self, module, args, out):
        r = self.req
        if r is None or "pages" in self.out:
            return
        self.req = None  # (an error here is reported, never raised: it would end the engine)
        rids = [x for x in self.runner.input_batch.req_ids if x is not None]
        if len(rids) != 1:
            self.out["error"] = f"replay expects one request per forward, found {len(rids)}"
            return
        blocks = list(self.runner.requests[rids[0]].block_ids[0][: r["nblocks"]])
        if len(blocks) != r["nblocks"]:  # (equal block sizes on both sides make this hold)
            self.out["error"] = f"the replay holds the request in {len(blocks)} blocks, the stage lists {r['nblocks']}"
            return
        torch.cuda.synchronize(self.kv[0].device)
        self.out["pages"] = replay_pages(self.kv, blocks, r["c"], r["s"], r["L"], r["in_flight"])

    def take(self) -> dict:
        """Plain bytes: vLLM's RPC serialiser passes them through unchanged (numpy arrays come back encoded)."""
        return {
            k: (
                v
                if k == "error"
                else None
                if v is None
                else [bytes(memoryview(np.ascontiguousarray(p))) for p in v]
                if k == "pages"
                else bytes(memoryview(np.ascontiguousarray(v)))
            )
            for k, v in self.out.items()
        }


def _flat(x) -> np.ndarray:
    return (
        np.frombuffer(x, dtype=np.uint8)
        if isinstance(x, (bytes, bytearray, memoryview))
        else np.asarray(x, dtype=np.uint8).reshape(-1)
    )


class VllmRegionView:
    """The prover's region as the verifier knows it, without materialising the pool: block j is assembled
    on demand from live pages (the replayed slots, filler in the slots the prover had not written),
    recomputed filler (free blocks), the activation rows and the pad. `words` and `block(j, bw)` are what
    the lazy chain needs; `live_page(l, b)` is a live page as the prover holds it."""

    def __init__(
        self,
        ev,
        live_pages: dict,
        act_rows_h: list,
        act_rows_r: list,
        expected=None,
        valid: dict | None = None,
        *,
        engine: dict,
    ):
        """live_pages: {(layer, prover block id): replayed page bytes}; valid: {(layer, block): slots the
        prover had written} (default: all of them); expected: the verifier's pad seed and GPU index
        (`memory_accounting.verifier.consistency.Expected`); without it the record's own. engine: the verifier's engine facts
        (`VllmReplayer.engine`): the layout is rebuilt from them and the stage, and the record must carry it
        exactly (a pad anywhere but at the end, of any other size, or activation buffers of another size, would
        move the pool's segments to positions the verifier never expects them at)."""
        from ..prover.dynfill import dyn_key, filler_np, mask_for
        from ..prover.kvenc import enc_words
        from .consistency import region_block_bytes, seed_and_gpu

        st, mi = ev.stage, ev.model
        self.ev, self.words = ev, ev.words
        seed, gpu = seed_and_gpu(ev, expected)
        self.seed, self.key = seed, dyn_key(seed, gpu)
        self.pw = int(st["page_words"])
        self.encoded = bool(st.get("encoded"))
        self.epw = enc_words(self.pw) if self.encoded else self.pw  # a page's words in the region
        # the KV pool's element type (records before v0.12: the model's)
        self.kv_mask = mask_for(mi.get("kv_dtype") or mi.get("dtype", "bfloat16"))
        self.act_mask = mask_for(engine["dtype"])
        self._replayed = {k: _flat(v).view(np.uint64) for k, v in live_pages.items()}
        self._valid = dict(valid or {})
        self.live: dict = {}  # merged pages, made on first use
        self.bs = int(st["block_size"])
        self._slots = None
        layout = pool_layout(st, engine, region_block_bytes(ev))
        if ev.layout != layout:
            raise ValueError(
                "record layout is not the pool region its stage and the verifier's engine make (layers of n_blocks pages, "
                "the engine's activation rows, one pad)"
            )
        self.segs, off = [], 0
        for e in layout:
            self.segs.append((e["name"], off, e["words"]))
            off += e["words"]
        if off != ev.words:
            raise ValueError("record layout does not add up to its region size")
        self._kv_off = {}
        for name, offset, words in self.segs:
            if name.startswith("kv."):
                if words != int(st["n_blocks"]) * self.epw:
                    raise ValueError(f"{name} is {words} words, not {st['n_blocks']} pages of {self.epw}")
                self._kv_off[int(name[3:])] = offset
        for (layer, block), page in self._replayed.items():
            if len(page) != self.pw or layer not in self._kv_off:
                raise ValueError(f"replayed page (layer {layer}, block {block}) does not fit the record's pool")
        self.act = {}
        for name, rows in (("act.h", act_rows_h), ("act.r", act_rows_r)):
            offset, words = next((o, w) for n, o, w in self.segs if n == name)
            if self.encoded:  # the rows' plaintext, zeros after; encoded where read
                words = next(e for e in layout if e["name"] == name)["shape"][1]
            fill = (
                np.zeros(words, dtype=np.uint64)
                if self.encoded
                else filler_np(words, offset, self.seed, self.key, self.act_mask)
            )
            pos = 0  # bytes; a row given as an int is that many bytes left as filler
            for x in rows:  # (the prover copies no residual when the layer has none)
                if isinstance(x, (int, np.integer)):
                    pos += int(x)
                    continue
                block = _flat(x)
                if len(block) % 8 or pos % 8 or pos + len(block) > words * 8:
                    raise ValueError(f"{name}: {pos + len(block)} bytes of rows do not fit in {words * 8}")
                fill[pos // 8 : (pos + len(block)) // 8] = block.view(np.uint64)
                pos += len(block)
            if pos > words * 8:
                raise ValueError(f"{name}: {pos} bytes of rows do not fit in {words * 8}")
            self.act[name] = (
                fill[: -(-pos // 8)] if self.encoded else fill
            )  # encoded: the rows (zeros after are implied)
        self._act_off = {n: o for n, o, _ in self.segs if n in self.act}
        self._filler_np = filler_np

    def live_page(self, layer: int, block: int) -> np.ndarray:
        """Page b of layer l as the prover holds it: replayed where it had written, filler at its own pool
        position elsewhere (a fresh block keeps its filler until the forward writes it)."""
        page = self.live.get((layer, block))
        if page is None and self.encoded:  # the replayed page (zeros past the written slots), encoded
            from ..prover.kvenc import encode_np

            page = self.live[(layer, block)] = encode_np(
                self._replayed[(layer, block)], self._kv_off[layer] + block * self.epw, self.key
            )
        if page is None:
            page = self._replayed[(layer, block)]
            valid = self._valid.get((layer, block), self.bs)
            if valid < self.bs:
                if self._slots is None:
                    mi = self.ev.model
                    if "page_shape" not in mi or "page_stride" not in mi:
                        raise ValueError("record has no page shape and stride: its unwritten slots cannot be placed")
                    self._slots = slot_index(mi["page_shape"], mi["page_stride"])
                    if int(self._slots.max(initial=-1)) + 1 != self.bs:
                        raise ValueError(f"page shape {mi['page_shape']} has no {self.bs} slots in dim 1")
                p0 = self._kv_off[layer] + block * self.pw
                page = merge_page(
                    page, self._filler_np(self.pw, p0, self.seed, self.key, self.kv_mask), valid, self._slots
                )
            self.live[(layer, block)] = page
        return page

    def _empty(self, n: int, pos: int) -> np.ndarray:
        from ..prover.kvenc import empty_np

        return empty_np(n, pos, self.key)

    def act_words(self, name: str, wa: int, wb: int) -> np.ndarray:
        """Words [wa, wb) of an activation buffer as the prover holds it (encoded: the rows' encoding at its position)."""
        if not self.encoded:
            return self.act[name][wa:wb]
        from ..prover.kvenc import encode_words_np

        return encode_words_np(self.act[name], self._act_off[name], self.key, wa, wb)

    def block(self, j: int, bw: int) -> np.ndarray:
        lo, hi = j * bw, (j + 1) * bw
        out = np.empty(bw, dtype=np.uint64)
        for name, offset, words in self.segs:
            start, end = max(lo, offset), min(hi, offset + words)
            if start >= end:
                continue
            if name.startswith("kv."):
                layer = int(name[3:])
                pos = start
                while pos < end:
                    pg = (pos - offset) // self.epw
                    p0 = offset + pg * self.epw
                    p1 = min(end, p0 + self.epw)
                    if (layer, pg) in self._replayed:
                        out[pos - lo : p1 - lo] = self.live_page(layer, pg)[pos - p0 : p1 - p0]
                    elif self.encoded:  # a free page: encoded zeros (just the words read)
                        out[pos - lo : p1 - lo] = self._empty(p1 - pos, pos)
                    else:
                        out[pos - lo : p1 - lo] = self._filler_np(self.pw, p0, self.seed, self.key, self.kv_mask)[
                            pos - p0 : p1 - p0
                        ]
                    pos = p1
            elif name in self.act:
                out[start - lo : end - lo] = self.act_words(name, start - offset, end - offset)
            else:  # the pad
                out[start - lo : end - lo] = self._filler_np(end - start, start, self.seed, self.key)
        return out


class LiveRegionView:
    """The live region of a record (second chain) as the chain reads it, block by block: the live pages in
    stage order as the pool view has them, the activation rows (the same buffers on the prover, so their filler
    is at their pool positions), then the pad at the live region's own positions (its own stream). The region
    is the live part of the stage the pool view was built for; its layout is checked against
    `live_layout` before this is built."""

    def __init__(self, ev, view: VllmRegionView):
        from .consistency import region_block_bytes

        st = ev.stage
        self.view, self.pw = view, view.epw  # a page's words in the region
        self.pages = [
            (layer, int(b)) for r in st["requests"] for layer in range(int(st["n_layers"])) for b in r["block_ids"]
        ]
        self.act_names = ["act.h", "act.r"]
        self.act_len = [w for n in self.act_names for nn, _, w in view.segs if nn == n]
        self.kv_words = len(self.pages) * self.pw
        self.live_words = self.kv_words + sum(self.act_len)
        bw = region_block_bytes(ev) // 8
        self.words = max(bw, -(-self.live_words // bw) * bw)

    def block(self, j: int, bw: int) -> np.ndarray:
        from ..prover.dynfill import filler_np

        lo, hi = j * bw, (j + 1) * bw
        out = np.empty(bw, dtype=np.uint64)
        pos, pw = lo, self.pw
        while pos < hi:
            if pos < self.kv_words:  # a live page
                k = pos // pw
                p0 = k * pw
                p1 = min(hi, p0 + pw)
                out[pos - lo : p1 - lo] = self.view.live_page(*self.pages[k])[pos - p0 : p1 - p0]
            elif pos < self.live_words:  # the activation rows
                o = self.kv_words
                for name, n in zip(self.act_names, self.act_len):
                    if pos < o + n:
                        break
                    o += n
                p1 = min(hi, o + n)
                out[pos - lo : p1 - lo] = self.view.act_words(name, pos - o, p1 - o)
            else:  # the live pad's own stream
                p1 = hi
                out[pos - lo :] = filler_np(hi - pos, pos, self.view.seed, self.view.key, live=True)
            pos = p1
        return out

    def array(self) -> np.ndarray:
        bw = 1 << 12
        return np.concatenate([self.block(j, bw) for j in range(-(-self.words // bw))])[: self.words]


def live_words(ev, view: VllmRegionView) -> np.ndarray:
    """The live region of a record, materialised (tests, debugging): `LiveRegionView` over the pool view."""
    return LiveRegionView(ev, view).array()


def pool_record_of(ev, engine: dict):
    """For a live record: the pool region of the same freeze, as far as the live chain needs it (the pool
    positions of the activation buffers, whose filler was written at those positions), from the stage and the
    verifier's engine, never from the live record's own layout."""
    from types import SimpleNamespace

    from .consistency import region_block_bytes

    layout = pool_layout(ev.stage, engine, region_block_bytes(ev))
    return SimpleNamespace(
        stage=ev.stage,
        model=ev.model,
        pad_seed=ev.pad_seed,
        block_bytes=ev.block_bytes,
        layout=layout,
        words=sum(e["words"] for e in layout),
    )


def live_layout_names(ev) -> list[str]:
    st = ev.stage
    return [
        f"r{i}.l{layer}.b{j}"
        for i, r in enumerate(st["requests"])
        for layer in range(st["n_layers"])
        for j in range(r["nblocks"])
    ] + ["act.h", "act.r"]


def region_view_from_replay(ev, per_request: list[dict], expected=None, *, engine: dict) -> VllmRegionView:
    """Place each request's replayed pages at the prover's block ids, with the slots the prover had written
    (the rest is its filler); activation rows in batch order. engine: the verifier's engine facts."""
    live, valid, rows_h, rows_r = {}, {}, [], []
    st = ev.stage
    check_stage(st)
    check_engine(st, ev.model, engine)
    if len(per_request) != len(st["requests"]):
        raise ValueError(f"{len(per_request)} requests replayed, the stage has {len(st['requests'])}")
    bs = int(st["block_size"])
    for r, got in zip(st["requests"], per_request):
        pages = got["pages"]
        nb = r["nblocks"]
        if len(pages) != st["n_layers"] * nb:
            raise ValueError(f"{len(pages)} pages replayed, {st['n_layers'] * nb} expected")
        in_flight = bool(st["in_flight"]) and r["scheduled"] > 0
        for layer in range(st["n_layers"]):
            written = written_slots(r["num_computed"], r["scheduled"], st["act_layer"], in_flight, layer)
            for j, b in enumerate(r["block_ids"]):
                live[(layer, b)] = pages[layer * nb + j]
                valid[(layer, b)] = page_valid(written, j, bs)
        if st["in_flight"] and r["scheduled"]:
            rows_h.append(got["act_h"])
            # no residual: the prover's rows keep their filler
            rows_r.append(got["act_r"] if got.get("act_r") is not None else len(_flat(got["act_h"])))
    return VllmRegionView(ev, live, rows_h, rows_r, expected, valid, engine=engine)


class VllmReplayer:
    """Replays vLLM records with a vLLM engine on this GPU (same model, dtype, block size, layout)."""

    def __init__(self, model: str, max_model_len: int = 2048, gpu_memory_utilization: float = 0.4):
        apply_env()
        from vllm import LLM

        self.llm = LLM(**settings(model, max_model_len, gpu_memory_utilization, verifier=True))
        self.info = self.llm.collective_rpc("posm_replay_setup")[0]
        self.engine = engine_facts(self.llm.llm_engine.vllm_config, self.info)
        self._last: dict = {}

    def region_words(self, ev, expected=None):
        """The record's region as a lazy view (main: `VllmRegionView`; live: `LiveRegionView`). Everything the
        replay sizes or trusts is checked first against the verifier's engine and the stage: block size, layers,
        page size, context length, the layout (rebuilt, never read from the record), the stage's requests and
        blocks. A live record's stage is the main record's when `verify` is given it."""
        from vllm import SamplingParams

        from .consistency import ReplayError, region_block_bytes

        st = ev.stage
        if st.get("kind") != "vllm":
            raise ReplayError("not a vLLM record")
        mi = ev.model
        for k in ("block_size", "page_shape", "page_stride", "dtype") + (("kv_dtype",) if "kv_dtype" in mi else ()):
            if mi.get(k) != self.info.get(k):
                raise ReplayError(f"verifier engine differs from the prover's: {k} {self.info.get(k)} != {mi.get(k)}")
        if mi.get("gpu_name") != self.info.get("gpu_name"):
            print(
                f"warning: prover GPU {mi.get('gpu_name')!r}, verifier GPU {self.info.get('gpu_name')!r}: bit-exactness is only expected on the same GPU type",
                flush=True,
            )
        live = getattr(ev, "region", "main") == "live"
        try:
            check_stage(st)
            check_engine(st, mi, self.engine)
            bb = region_block_bytes(ev)
            want = live_layout(st, self.engine, bb) if live else pool_layout(st, self.engine, bb)
        except (KeyError, TypeError, ValueError) as e:
            raise ReplayError(f"stage cannot be replayed: {e}")
        if ev.layout != want:
            raise ReplayError(
                f"record layout is not the {'live' if live else 'pool'} region its stage and the verifier's engine make"
            )
        key = json.dumps(st, sort_keys=True)
        if self._last.get("key") != key:  # the pool and live records of one freeze share a replay
            per = []
            sp = SamplingParams(max_tokens=1, temperature=0.0)
            for r in st["requests"]:
                in_flight = bool(st["in_flight"]) and r["scheduled"] > 0
                self.llm.collective_rpc(
                    "posm_replay_arm",
                    args=(r["num_computed"], r["scheduled"], st["act_layer"], in_flight, r["nblocks"]),
                )
                self.llm.generate([{"prompt_token_ids": r["tokens"]}], sp, use_tqdm=False)
                got = self.llm.collective_rpc("posm_replay_take")[0]
                if got.get("error") or "pages" not in got:
                    self._last = {}
                    raise ReplayError(f"replay of request {len(per)}: {got.get('error') or 'no pages captured'}")
                per.append(got)
            self._last = {"key": key, "per": per}
        if live:
            view = region_view_from_replay(
                pool_record_of(ev, self.engine), self._last["per"], expected, engine=self.engine
            )
            self._last["view"] = view
            return LiveRegionView(ev, view)
        view = region_view_from_replay(ev, self._last["per"], expected, engine=self.engine)
        self._last["view"] = view
        return view

    def diagnose(self, ev) -> list[str]:
        """With the prover's per-page sums (records from a prover with debug on): the live pages that differ."""
        sums = (ev.debug or {}).get("page_sums")
        view = self._last.get("view")
        if not sums or view is None:
            return []
        bad = []
        for k, v in sums.items():
            layer, b = (int(x) for x in k.split("."))
            page = view.live_page(layer, b) if (layer, b) in view._replayed else None
            if page is not None and int(page.view(np.int64).sum()) & ((1 << 64) - 1) != v:
                bad.append(f"layer {layer} block {b}")
        return bad


def _replay_main(args):
    from ..core.evidence import Evidence
    from ..core.params import parse_bytes
    from .consistency import (
        Expected,
        audit_order,
        audit_record,
        checker_keys,
        coverage_lines,
        log_entries,
        record_loader,
        verify,
    )

    man = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    # derived, never the manifest's
    keys = checker_keys(args.filler_seed, args.pad_seed, man["filler_seed"] if man else None, args.hash_seed)
    expected = Expected(int(args.pad_seed, 16), args.gpu) if args.pad_seed is not None else None
    entries = log_entries(args.log) if args.log else None
    shape = (parse_bytes(args.dyn_B), args.dyn_n, man["r"] if man else 2048, man["k"] if man else 1024, 1)
    p = Path(args.evidence)
    files = sorted(p.glob("*.json")) if p.is_dir() else ([p] if p.exists() else [])
    if not files and entries is None:
        raise SystemExit(f"no records under {p}")
    ok = True
    rp = VllmReplayer(args.model, args.max_model_len, args.gpu_memory_utilization) if files else None
    reports, seen = [], set()
    main_of = record_loader(p)  # with --log, a live record is replayed from its main record's stage
    if entries is None and files:
        print(
            "note: without --log a live record is not paired with its main record (its own stage is replayed)",
            flush=True,
        )
    for f in audit_order(files, entries):
        ev = Evidence.load(f)
        seen.add(int(ev.nonce))
        rep = audit_record(
            ev, entries, shape, lambda ev, main: verify(ev, rp, keys, expected=expected, main=main), main_of
        )
        reports.append(rep)
        ok &= rep["ok"] or bool(rep.get("fault"))
        dbg = ev.debug or {}
        tag = (
            (
                f" [frozen by {dbg['frozen_by']}"
                + (
                    f", region {'unchanged' if dbg['region_unchanged'] else 'CHANGED'}"
                    if "region_unchanged" in dbg
                    else ""
                )
                + "]"
            )
            if dbg.get("frozen_by")
            else ""
        )
        print(
            f"{rep['nonce']}: {rep['stage']}{tag}: "
            + (
                "digest reproduced"
                if rep["ok"]
                else f"FAULT ({rep['error']})"
                if rep.get("fault")
                else f"MISMATCH ({rep.get('error') or rep.get('expected')})"
            )
            + (
                "; differing: " + ", ".join(d["name"] for d in rep.get("differing_segments", [])[:8])
                if rep.get("differing_segments")
                else ""
            ),
            flush=True,
        )
    if entries is not None:  # every logged challenge with a reply needs its record
        lines, covered = coverage_lines(entries, seen, args.partial)
        for line in lines:
            print(line, flush=True)
        ok &= covered
    checked = [r for r in reports if not r.get("fault")]
    print(
        f"{sum(r['ok'] for r in checked)}/{len(checked)} records reproduced exactly"
        + (" and match the verifier's log" if entries is not None else "")
        + ("" if ok else "; AUDIT FAILED")
    )
    sys.exit(0 if ok else 1)


def _selftest_main(args):
    """Measure the property the replay needs, on this host: a prover engine on one GPU generates a batch
    and records stages at random layers; a replay engine on another GPU (same type) reproduces each
    region and digest. Each engine runs in its own process (CUDA_VISIBLE_DEVICES)."""
    import subprocess

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device_prover), **ENV)
    print(f"prover on GPU {args.device_prover}: recording {args.records} stages", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "memory_accounting.verifier.vllm_replay",
            "_record",
            "--model",
            args.model,
            "--out",
            str(out_dir),
            "--records",
            str(args.records),
            "--max-model-len",
            str(args.max_model_len),
            "--dyn-B",
            args.dyn_B,
        ],
        env=env,
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device_verifier), **ENV)
    print(f"verifier on GPU {args.device_verifier}: replaying", flush=True)
    # live records paired
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "memory_accounting.verifier.vllm_replay",
            "replay",
            "--model",
            args.model,
            "--evidence",
            str(out_dir),
            "--max-model-len",
            str(args.max_model_len),
            "--log",
            str(out_dir / SELFTEST_LOG),
            "--pad-seed",
            "5eed",
            "--dyn-B",
            args.dyn_B,
        ],
        env=env,
    )
    sys.exit(rc)


SELFTEST_LOG = "selftest.jsonl"


def _record_main(args):
    """selftest prover half: generate ragged batches while a client in this process sends dynamic
    challenges to the worker's UDP responder (freezing it at whatever layer or step boundary it is at)
    and saves the records, with debug bytes, for the replay half."""
    import random
    import socket
    import threading
    from types import SimpleNamespace

    from vllm import LLM, SamplingParams

    from ..core.evidence import fetch_evidence
    from ..core.params import parse_bytes
    from .net import _dynamic_round

    llm = LLM(**settings(args.model, args.max_model_len, 0.45))
    print("dynamic:", llm.collective_rpc("posm_dynamic_setup", args=(args.dyn_B, "5eed", True))[0], flush=True)
    print(llm.collective_rpc("posm_serve", args=(args.port, None, "honest", None, False, 200.0))[0], flush=True)
    rng = random.Random(7)
    words = ["alpha", "beta", "gamma", "delta", "memory", "hash", "chain", "matrix", "verifier", "prover"]
    # up to ~4 KV tiles of 128
    prompts = [" ".join(rng.choice(words) for _ in range(rng.randint(5, 400))) for _ in range(16)]
    done = threading.Event()
    saved = []

    def client():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10.0)
        ns = SimpleNamespace(host="127.0.0.1", port=args.port)
        nrng = np.random.default_rng(11)
        time.sleep(2.0)  # let the load start
        while len(saved) < args.records:
            sample, ch, _dg, lv = _dynamic_round(sock, ns, nrng, (parse_bytes(args.dyn_B), 8, 256, 1024, 1), 0)
            with open(
                Path(args.out) / SELFTEST_LOG, "a"
            ) as f:  # the verifier's log: the replay pairs each live record with its main one
                f.write(json.dumps({"type": "verification", "id": len(saved), "dyn": sample}) + "\n")
            ev = fetch_evidence("127.0.0.1", args.port, ch.nonce)
            saved.append(ev.save(args.out))
            msg = f"recorded {ev.describe()} (rtt {sample['rtt_ns'] / 1e3:.0f} µs"
            if lv is not None:
                evl = fetch_evidence("127.0.0.1", args.port, lv[0].nonce)
                evl.save(args.out)
                msg += f"; live chain over {sample['live']['words'] * 8 >> 20} MiB, rtt {sample['live']['rtt_ns'] / 1e3:.0f} µs"
            print(msg + ")", flush=True)
            time.sleep(0.2)
        done.set()

    ct = threading.Thread(target=client, daemon=True)
    ct.start()
    while not done.is_set() and ct.is_alive():
        llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=48, seed=3), use_tqdm=False)
    llm.collective_rpc("posm_stop")
    if not done.is_set():
        raise SystemExit("the challenge client failed; see above")


# ---- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description="dynamic memory under vLLM, verifier side: replay records, self-test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--evidence", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--manifest", default=None, help="the prover's bundle manifest: supplies r, k")
    r.add_argument(
        "--hash-seed",
        type=int,
        default=None,
        help="Keys.deterministic seed (a test's keys); overrides the keys derived from the filler seed",
    )
    r.add_argument(
        "--filler-seed",
        default=None,
        help="hex: the filler seed the verifier minted; the challenge-hash keys are derived from it (default: --pad-seed)",
    )
    r.add_argument("--max-model-len", type=int, default=2048)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    r.add_argument(
        "--pad-seed", default=None, help="hex: the pad seed the verifier minted (records claiming another fail)"
    )
    r.add_argument("--gpu", type=int, default=0, help="with --pad-seed: the index of the GPU the verifier challenged")
    r.add_argument(
        "--log",
        default=None,
        help="the verifier's log (memory_accounting.verifier.net live --out): records must match the chains it sent",
    )
    r.add_argument("--dyn-B", default="64K")
    r.add_argument("--dyn-n", type=int, default=8)
    r.add_argument(
        "--partial",
        action="store_true",
        help="with --log: the records are a subset; do not fail on logged challenges without one",
    )
    t = sub.add_parser("selftest")
    t.add_argument("--model", required=True)
    t.add_argument("--device-prover", default="0")
    t.add_argument("--device-verifier", default="1")
    t.add_argument("--records", type=int, default=8)
    t.add_argument("--out", default="evidence_vllm_selftest")
    t.add_argument("--max-model-len", type=int, default=2048)
    t.add_argument(
        "--dyn-B",
        default="64K",
        help="the dynamic block (the prover's region block and the chains'; 1M spans 128 KV pages of Qwen2.5-0.5B)",
    )
    q = sub.add_parser("_record")
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--records", type=int, default=8)
    q.add_argument("--max-model-len", type=int, default=2048)
    q.add_argument("--port", type=int, default=5571)
    q.add_argument("--dyn-B", default="64K")
    args = ap.parse_args(argv)
    {"replay": _replay_main, "selftest": _selftest_main, "_record": _record_main}[args.cmd](args)


if __name__ == "__main__":
    main()
