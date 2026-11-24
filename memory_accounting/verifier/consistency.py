"""Verifier: check a dynamic challenge by recomputation.

For each record (`memory_accounting.core.evidence.Evidence`) the verifier

  1. replays inference with the weights it holds on the recorded token chunks, stopping exactly where
     the prover was frozen (the recorded step and layer), which rebuilds the dynamic region: KV cache,
     activation buffer, pad;
  2. checks the rebuilt region has the layout the prover declared;
  3. runs the challenge's chain (same nonce, start block, shape, keys) over the rebuilt bytes and
     compares the digest with the one the prover returned inside the deadline.

Equality is exact: there is no tolerance anywhere. That needs a bit-exact replay: the integer-exact
model (`memory_accounting.models.intllm`) gives it on any CPU or GPU; float models give it only on the same hardware and
kernels (HF on the same device here; vLLM in batch-invariant mode on the same GPU type, in
`memory_accounting.verifier.vllm_replay`). What a match proves, and what it does not (the chain reads n blocks of the region per
challenge), is in the design log.

A record is the prover's account of a freeze. Two of its fields choose what the replay builds and are
not the prover's to choose: the pad seed (the verifier mints it) and the GPU index (the verifier knows
which GPU it challenged). With `expected` the verifier supplies both; a record that claims others is
rejected, and the replay uses the verifier's values. Offline, `--log` checks each record against the
verifier's own log of the challenge sent and the reply received (the live verifier's matching rule), not
only against itself, and every logged challenge that got a reply must have its record (a prover could
otherwise withhold the records it would fail); a challenge that got no reply failed online and needs none.

The live record of a freeze (the second chain, over the live region) is not replayed from a stage of its
own: the live region is the live part of the main stage. `verify(..., main=)` refuses a live record whose
stage or model is not its main record's and rebuilds the live region from the main record's stage; offline,
`--log` pairs each live record with the main record of the same verification (a live record without its
main record fails). Without `--log` there is nothing to pair by, and a live record is checked against its
own stage.

What the replay allocates and computes is the verifier's choice, not the record's: the model is the
verifier's (`--int-model`, `--model`, the toy's fixed shape), its KV cache length is the verifier's
(`--max-len`; without it, any length the model allows, with one model and one cache kept at a time), and a
record that describes another cache or activation buffer is refused before anything is built. The rebuilt
region is not materialised: the chain reads it block by block (`StaticRegionView`), the replay's buffers where
the prover had written and recomputed filler elsewhere.

  python -m memory_accounting.verifier.consistency --evidence evidence/ [--int-model models/x.int.safetensors] [--model <hf dir>] [--manifest bundle/manifest.json]
      [--pad-seed <hex> --gpu <i>] [--log demo_live.jsonl [--dyn-B 64K --dyn-n 8]] [--max-len <slots>]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.backend import CpuLib
from ..core.evidence import Evidence
from ..core.params import MAX_LINKS, MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params

ACT_NAME, PAD_NAME = "act", "pad"


class ReplayError(Exception):
    """The replay could not rebuild the region the record describes (model, layout or stage mismatch)."""


@dataclass(frozen=True)
class Expected:
    """What the verifier, not the record, decides about a replayed region: the pad seed it minted for the
    dynamic filler and the index of the GPU it challenged (the filler key is derived from both)."""

    pad_seed: int
    gpu: int = 0


def record_gpu(ev) -> int:
    """The GPU index a record claims (records that do not say were made with GPU 0's key)."""
    return int((getattr(ev, "model", None) or {}).get("gpu", 0))


def claim_mismatch(ev, expected: Expected | None) -> str | None:
    """Why a record's pad seed or GPU is not the verifier's, or None."""
    if expected is None:
        return None
    if int(ev.pad_seed) != int(expected.pad_seed):
        return f"record's pad seed {int(ev.pad_seed):x} is not the verifier's {int(expected.pad_seed):x}"
    if record_gpu(ev) != int(expected.gpu):
        return f"record names GPU {record_gpu(ev)}, the verifier challenged GPU {int(expected.gpu)}"
    return None


def seed_and_gpu(ev, expected: Expected | None) -> tuple[int, int]:
    """The pad seed and GPU a replay builds the region with: the verifier's when it gives them."""
    return (int(expected.pad_seed), int(expected.gpu)) if expected is not None else (int(ev.pad_seed), record_gpu(ev))


def challenge_of(ev: Evidence) -> Challenge:
    c = ev.challenge
    return Challenge(c["block_bytes"], c["n"], c["r"], ev.nonce, c["j0"], c.get("k", 1024), c.get("g", 1))


def chain_digest(words: np.ndarray, ev: Evidence, keys: Keys, lib: CpuLib | None = None) -> tuple[int, int]:
    """The challenge's chain over a region array (uint64 words, the logical order the prover's segment
    table presented)."""
    lib = lib or CpuLib()
    cfg = MemoryConfig(words=len(words), encoding="none", perm=False, image=False)
    out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)
    lib.chain(
        np.ascontiguousarray(words, dtype=np.uint64),
        pack_params(cfg, challenge_of(ev), keys, MODE_HONEST),
        out,
        mt=True,
    )
    return int(out[0]), int(out[1])


def chain_digest_lazy(view, ev: Evidence, keys: Keys, lib: CpuLib | None = None) -> tuple[int, int]:
    """The same chain over a region too large to hold (a vLLM KV pool): one link at a time, asking the view
    for the block each link reads. Same digest as `chain_digest` over the materialised array."""
    lib = lib or CpuLib()
    ch = challenge_of(ev)
    bw = ch.be
    nb = view.words // bw
    cfg = MemoryConfig(words=bw, encoding="none", perm=False, image=False)
    out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)
    digest, idx = (0, 0), ch.j0
    for link in range(ch.n):
        blk = np.ascontiguousarray(view.block(idx, bw), dtype=np.uint64)
        lib.chain(
            blk,
            pack_params(cfg, ch, keys, MODE_HONEST, nb=1, d0=digest, link0=link, n=1, j0=0, s0=ch.nonce, qoff=idx * bw),
            out,
        )
        digest = (int(out[0]), int(out[1]))
        idx = digest[0] % nb
    return digest


def hexd(d) -> str:
    return f"{int(d[1]):016x}{int(d[0]):016x}"


MAX_REGION_BLOCK = 1 << 24  # a record's region block (the pad's unit): the verifier allocates one


def region_block_bytes(ev) -> int:
    """The record's region block size, checked before the replay allocates its pad: whole words, at most 16 MiB."""
    bb = ev.block_bytes
    if not isinstance(bb, int) or isinstance(bb, bool) or not 8 <= bb <= MAX_REGION_BLOCK or bb % 8:
        raise ReplayError(f"region block of {bb!r} bytes: whole 8-byte words up to {MAX_REGION_BLOCK} bytes")
    return bb


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True)


def live_mismatch(ev, main) -> str | None:
    """Why a live record cannot be the second chain of this main record's freeze, or None. The live region is
    the live part of the main region at one freeze, so it has the main record's stage and model (JSON-equal),
    pad seed and region block; a live record of another stage (fewer tokens, another step) would let the live
    chain check data the main stage does not describe."""
    if getattr(main, "region", "main") != "main":
        return "the record paired with this live record is not a main record"
    if getattr(ev, "region", "main") != "live":
        return "not a live record"
    try:
        if _canon(ev.stage) != _canon(main.stage):
            return "the live record's stage is not its main record's (the same freeze has one stage)"
        if _canon(ev.model) != _canon(main.model):
            return "the live record's model is not its main record's"
        if _canon([ev.pad_seed, ev.block_bytes]) != _canon([main.pad_seed, main.block_bytes]):
            return "the live record's pad seed or region block is not its main record's"
    except (TypeError, ValueError) as e:
        return f"records not comparable: {e}"
    return None


def from_main(ev, main):
    """The live record as the replay sees it: the main record's stage and model (a copy; the record is not changed)."""
    out = copy.copy(ev)
    out.stage, out.model = main.stage, main.model
    return out


def verify(
    ev: Evidence, replayer, keys: Keys, lib: CpuLib | None = None, expected: Expected | None = None, main=None
) -> dict:
    """Replay, rebuild, chain, compare. The verdict is the digest equality alone. expected: the verifier's
    pad seed and GPU index; a record claiming others fails, and the replay uses the verifier's values
    (without it the record's own are used: a check of the record against itself). main: for a live record,
    the main record of the same freeze; the live region is rebuilt from its stage, and a live record whose
    stage or model differs fails (without it, the live record's own stage is replayed)."""
    rep = {"nonce": f"{ev.nonce:016x}", "stage": ev.describe(), "received": hexd(ev.digest), "ok": False}
    if (why := claim_mismatch(ev, expected)) is not None:
        rep["error"] = why
        return rep
    if main is not None and getattr(ev, "region", "main") == "live":
        if (why := live_mismatch(ev, main)) is not None:
            rep["error"] = why
            return rep
        ev = from_main(ev, main)
    try:
        words = replayer.region_words(ev, expected) if expected is not None else replayer.region_words(ev)
    except (
        ReplayError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        RuntimeError,
    ) as e:  # the record is the prover's: anything in it may be wrong
        rep["error"] = (
            str(e)
            if isinstance(e, (ReplayError, ValueError))
            else f"record cannot be replayed: {type(e).__name__}: {e}"
        )
        return rep
    lazy = hasattr(words, "block")
    n = words.words if lazy else len(words)
    if n != ev.words:
        rep["error"] = f"rebuilt region has {n} words, the record says {ev.words}"
        return rep
    exp = chain_digest_lazy(words, ev, keys, lib) if lazy else chain_digest(words, ev, keys, lib)
    rep["expected"] = hexd(exp)
    rep["ok"] = exp == tuple(ev.digest)
    # debug bytes from the prover: say where they differ
    if ev.data is not None and (not lazy or hasattr(words, "array")):
        rep["differing_segments"] = diff_segments(ev, words.array() if lazy else words)
    if not rep["ok"] and hasattr(replayer, "diagnose"):
        bad = replayer.diagnose(ev)
        if bad:
            rep["differing_segments"] = [{"name": b, "words_differing": -1, "of": -1} for b in bad]
    return rep


def diff_segments(ev: Evidence, words: np.ndarray) -> list[dict]:
    out, off = [], 0
    for e in ev.layout:
        a, b = ev.data[off : off + e["words"]], words[off : off + e["words"]]
        n = int(np.count_nonzero(a != b))
        if n:
            out.append({"name": e["name"], "words_differing": n, "of": e["words"]})
        off += e["words"]
    return out


def check_layout(ev: Evidence, layout: list[dict]) -> None:
    _check_layout(ev, layout)


def _check_layout(ev: Evidence, layout: list[dict]) -> None:
    if layout != ev.layout:
        mine = {e["name"]: e for e in layout}
        theirs = {e["name"]: e for e in ev.layout}
        bad = [n for n in theirs if mine.get(n) != theirs[n]] + [n for n in mine if n not in theirs]
        raise ReplayError(f"region layout differs from the record (first: {bad[:3]})")


def check_static_stage(st: dict, n_layers: int, max_len: int, vocab: int) -> None:
    """The fields of a static-cache stage the replay trusts, checked before anything is replayed (the prover
    writes them; `StageTracker.stage` is the honest form). ValueError: the stage cannot be replayed.

      * n_layers is the verifier's model's; 0 <= act_layer < n_layers;
      * step -1 (nothing fed since the cache was emptied): no chunks, not in flight;
      * step >= 0: one chunk per step, each one row of token ids of the verifier's vocabulary, no more tokens
        than the cache holds. The replay and the live region read row 0 of a chunk while a model consumes
        every row, so a chunk of two rows would feed tokens the live region does not count;
      * in flight, kv_layers_done = act_layer; a completed step: act_layer = n_layers - 1 (the last boundary
        written), kv_layers_done = n_layers. A completed step with act_layer 0 would make the activation
        buffer hold the embeddings, which anyone can regenerate from the tokens, so a prover need not keep it."""
    step, act, fl, chunks = st.get("step"), st.get("act_layer"), st.get("in_flight"), st.get("chunks")

    def ints(*xs):
        return all(isinstance(x, int) and not isinstance(x, bool) for x in xs)

    if not ints(step, act) or not isinstance(fl, bool) or not isinstance(chunks, list):
        raise ValueError("stage fields missing or of the wrong type (step, act_layer, in_flight, chunks)")
    if st.get("n_layers") != n_layers:
        raise ValueError(f"stage of a {st.get('n_layers')}-layer model, the verifier's has {n_layers}")
    if not 0 <= act < n_layers:
        raise ValueError(f"paused layer {act} outside the model's {n_layers} layers")
    if step < 0:
        if step != -1 or chunks or fl:
            raise ValueError("a stage before the first step has step -1, no chunks and nothing in flight")
        return
    if len(chunks) != step + 1:
        raise ValueError(f"{len(chunks)} chunks for step {step}")
    done = act if fl else n_layers
    if not fl and act != n_layers - 1:
        raise ValueError(
            f"a completed step's activation buffer holds layer {n_layers - 1}'s input, the stage says layer {act}"
        )
    if "kv_layers_done" in st and st["kv_layers_done"] != done:
        raise ValueError(f"kv_layers_done {st['kv_layers_done']}, {done} for this stage")
    total = 0
    for i, c in enumerate(chunks):
        if not (isinstance(c, list) and len(c) == 1 and isinstance(c[0], list) and c[0]):
            raise ValueError(f"chunk {i} is not one row of tokens")
        if not all(ints(t) and 0 <= t < vocab for t in c[0]):
            raise ValueError(f"chunk {i} has tokens outside the vocabulary of {vocab}")
        total += len(c[0])
    if total > max_len:
        raise ValueError(f"{total} tokens fed, the cache holds {max_len}")


def written_slots(stage: dict, layer: int) -> int:
    """Cache slots of this layer the prover had written at the stage (a checked one): every fed token's, except
    the in-flight step's in the paused layer and the layers after it."""
    step = stage["step"]
    if step < 0:
        return 0
    before = sum(len(c[0]) for c in stage["chunks"][:step])
    last = len(stage["chunks"][step][0])
    return before + (last if not stage["in_flight"] or layer < stage["act_layer"] else 0)


def replay(feed, layers: list, stage: dict, act, reset, coalesce: bool = False, clear_act=None) -> int:
    """Run the recorded forwards from an empty cache and stop where the prover was frozen. The hook on the
    recorded layer writes the activation buffer exactly as the prover's hook did; a stage in flight
    stops the forward there, so later layers keep the previous step's K/V, as on the prover. `reset`
    empties the cache (and, for a region read as it is, puts the filler back in every free word); `clear_act`
    restores the activation buffer's filler before a write, as the prover's hook does (None: zeros, for a
    region whose unwritten words the view supplies). Returns the activation elements written.

    coalesce=True feeds all steps before the last as one prefill: valid only for a model whose results do
    not depend on the chunking (the integer-exact model), and much faster than replaying every decode step."""
    import torch

    from ..prover.dynamic import write_act

    reset()
    if clear_act is None:
        act.zero_()
    if stage["step"] < 0:
        return 0
    target, stop = stage["act_layer"], bool(stage["in_flight"])
    wrote = [0]

    class _Stop(Exception):
        pass

    def hook(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        write_act(act, h, clear_act)
        wrote[0] = h.numel()
        if stop:
            raise _Stop()

    chunks = stage["chunks"][: stage["step"] + 1]
    if coalesce and len(chunks) > 2:
        history = [t for c in chunks[:-1] for t in c[0]]
        chunks = [[history], chunks[-1]]
    with torch.no_grad():
        for s, chunk in enumerate(chunks):
            handle = layers[target].register_forward_pre_hook(hook, with_kwargs=True) if s == len(chunks) - 1 else None
            try:
                feed(s, chunk)
            except _Stop:
                pass
            finally:
                if handle is not None:
                    handle.remove()
    return wrote[0]


_NP_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


class StaticRegionView:
    """A replayed static-cache region (main or live) as the chain reads it, one block at a time, never
    materialised. Each part is a buffer of the replay read in place where the prover had written it, and the
    dynamic filler, recomputed for just the words read, where it had not: the cache slots past the tokens fed
    (and this step's slots in the paused layer and after), the activation buffer past the hidden state, the pad.

    parts: (offset, words, tensor or None (pad), filler position of its word 0, mask, written[, encoded]) with written
    None (the buffer as it is) or (element bytes, inner, slots, n): element e is written iff
    (e // inner) % slots < n. encoded: the part is an encoded store of the tensor's plaintext (the replay's buffer, zeros
    where nothing was written), its words the encoding at the prover's positions (`memory_accounting.prover.kvenc`)."""

    def __init__(self, parts: list, words: int, seed: int, key: int, live: bool = False, threads: int = 1):
        self.parts, self.words, self.seed, self.key, self.live, self.threads = parts, words, seed, key, live, threads
        self._plain: dict = {}

    def _read(self, part, wa: int, wb: int) -> np.ndarray:
        from ..prover.dynfill import filler_np

        _, _, tensor, fpos, mask, wr = part[:6]
        if tensor is None:  # the pad: filler at the region's own positions
            return filler_np(wb - wa, fpos + wa, self.seed, self.key, live=self.live)
        if len(part) > 6 and part[6]:  # an encoded store: encode the replayed plaintext
            from ..prover.kvenc import encode_words_np

            plain = self._plain.get(id(tensor))
            if plain is None:
                plain = self._plain[id(tensor)] = tensor.detach().cpu().contiguous().numpy().view(np.uint64)
            return encode_words_np(plain, fpos, self.key, wa, wb, self.threads)
        raw = tensor[wa:wb].cpu().numpy().view(np.uint64)
        if wr is None:
            return raw
        esize, inner, slots, n = wr
        if n >= slots:
            return raw
        fill = filler_np(wb - wa, fpos + wa, self.seed, self.key, mask)
        if n <= 0:
            return fill
        k, dt = 8 // esize, _NP_UINT[esize]
        e = np.arange(wa * k, wb * k, dtype=np.int64)
        keep = (e // inner) % slots < n
        return np.where(keep, raw.view(dt), fill.view(dt)).astype(dt, copy=False).view(np.uint64)

    def block(self, j: int, bw: int) -> np.ndarray:
        lo, hi = j * bw, (j + 1) * bw
        out = np.empty(bw, dtype=np.uint64)
        for p in self.parts:
            off, w = p[0], p[1]
            a, b = max(lo, off), min(hi, off + w)
            if a < b:
                out[a - lo : b - lo] = self._read(p, a - off, b - off)
        return out

    def array(self) -> np.ndarray:
        """The whole region (debug: locating a mismatch against the bytes a prover attached)."""
        return (
            np.concatenate([self._read(p, 0, p[1]) for p in self.parts]) if self.parts else np.zeros(0, dtype=np.uint64)
        )


@dataclass
class Prepared:
    """What a replayer needs of the verifier's model for one record: the forward driver, the layers, the KV
    buffers (names "kv.<layer>.<part>"), the cache reset, the live slices (None: no live region), the
    activation buffer it allocated, the cache length and vocabulary the stage is checked against, and the KV
    buffers' slot dimension (None: the replay puts the filler in the buffers and the view reads them as they are)."""

    feed: object
    layers: list
    kv_buffers: object
    reset: object
    live_kv: object
    act: object
    max_len: int
    vocab: int
    slot_dim: int | None = None


class StaticCacheReplayer:
    """Replays a model whose KV cache is a fixed set of buffers (toy, integer model, HF StaticCache).
    `prepare(ev)` returns a `Prepared` for the record: the verifier's own model, cache and activation buffer
    (a record describing another is refused before anything is built). The region is returned as a
    `StaticRegionView`."""

    def __init__(self, prepare, device="cpu", coalesce: bool = False):
        self.prepare, self.device, self.coalesce = prepare, device, coalesce

    def region_words(self, ev: Evidence, expected: Expected | None = None):
        from ..prover.dynamic import DynamicRegion, _flat_words, live_tokens, make_region

        seed, gpu = seed_and_gpu(ev, expected)
        st = ev.stage
        if st.get("kind", "static-cache") != "static-cache":
            raise ReplayError(f"a {st.get('kind')} record needs its own replayer")
        bb = region_block_bytes(ev)
        prepared = self.prepare(ev)
        try:
            check_static_stage(st, len(prepared.layers), prepared.max_len, prepared.vocab)
        except (KeyError, TypeError, ValueError) as e:
            raise ReplayError(f"stage cannot be replayed: {e}")
        encoded = bool(st.get("encoded"))
        if encoded:  # the prover's region: each buffer's encoded store
            return self._encoded(ev, prepared, st, bb, seed, gpu)
        region = make_region(prepared.kv_buffers(), prepared.act, bb, seed, gpu=gpu)
        if getattr(ev, "region", "main") == "live":
            if prepared.live_kv is None:
                raise ReplayError("this model has no live region")
            segs = [(n, t) for n, t in prepared.live_kv(live_tokens(st)) + [(ACT_NAME, prepared.act)] if t.numel()]
            out = DynamicRegion(segs, bb, seed, gpu=gpu, live=True)
        else:
            out = region
        _check_layout(ev, out.layout())  # the verifier's own buffers against the record's claim
        lazy = prepared.slot_dim is not None
        if lazy:  # zeros in the buffers; the view supplies the filler
            n_act = replay(prepared.feed, prepared.layers, st, prepared.act, prepared.reset, self.coalesce)
        else:

            def reset_to_filler():
                prepared.reset()
                region.reset_to_filler()

            n_act = replay(
                prepared.feed,
                prepared.layers,
                st,
                prepared.act,
                reset_to_filler,
                self.coalesce,
                clear_act=lambda: region.clear(ACT_NAME),
            )
        main_pos = {s.name: s.offset for s in region.segments}

        def written(seg):
            if not lazy:
                return None
            esize = seg.tensor.element_size()
            if seg.name == ACT_NAME:
                return (esize, 1, seg.tensor.numel(), n_act)
            shape = tuple(seg.tensor.shape)
            return (
                esize,
                int(np.prod(shape[prepared.slot_dim + 1 :], dtype=np.int64)),
                shape[prepared.slot_dim],
                written_slots(st, int(seg.name.split(".")[1])),
            )

        parts = [
            (
                s.offset,
                s.words,
                None if s.name == PAD_NAME else _flat_words(s.tensor),
                s.offset if s.name == PAD_NAME else main_pos[s.name],
                s.mask,
                None if s.name == PAD_NAME else written(s),
            )
            for s in out.segments
        ]
        return StaticRegionView(parts, out.words, seed, region.dyn_key, live=out.live)

    def _encoded(self, ev: Evidence, prepared: Prepared, st: dict, bb: int, seed: int, gpu: int):
        """An encoded record (v0.17, `memory_accounting.prover.kvenc`): the region is each cache buffer's encoded store, then the
        activation buffer's, then the pad; a store's words are the encoding, at the prover's positions, of the replay's
        buffer as it is (zeros where the prover had written nothing: the replay starts from an emptied cache and stops
        where the prover was frozen). The live region is each store's groups holding the fed slots, then the
        activation store, at their main-region positions."""
        from ..prover.dynamic import DynamicRegion, _flat_words, live_tokens
        from ..prover.kvenc import groups_of, store_for

        if prepared.slot_dim is None:
            raise ReplayError("this model's cache cannot be held encoded (no slot dimension)")
        kv = list(prepared.kv_buffers())
        stores = [(n, store_for(t, alloc=False)) for n, t in kv]
        region = DynamicRegion(
            stores + [(ACT_NAME, store_for(prepared.act, alloc=False))], bb, seed, device=prepared.act.device, gpu=gpu
        )
        plain = {n: t for n, t in kv}
        plain[ACT_NAME] = prepared.act
        if getattr(ev, "region", "main") == "live":
            n_live = live_tokens(st)

            def groups(t):
                rows = int(t.shape[prepared.slot_dim])
                return groups_of(-(-n_live * (t.numel() * t.element_size() // rows) // 8))

            segs = [(n, s.prefix(groups(plain[n]))) for n, s in stores] + [
                (ACT_NAME, store_for(prepared.act, alloc=False))
            ]
            out = DynamicRegion(
                [(n, s) for n, s in segs if s.words], bb, seed, device=prepared.act.device, gpu=gpu, live=True
            )
        else:
            out = region
        _check_layout(ev, out.layout())
        replay(
            prepared.feed, prepared.layers, st, prepared.act, prepared.reset, self.coalesce
        )  # zeros where nothing is written, as on the prover
        main_pos = {s.name: s.offset for s in region.segments}
        parts = [
            (s.offset, s.words, None, s.offset, s.mask, None)
            if s.name == PAD_NAME
            else (s.offset, s.words, _flat_words(plain[s.name]), main_pos[s.name], s.mask, None, True)
            for s in out.segments
        ]
        import os

        return StaticRegionView(
            parts, out.words, seed, region.dyn_key, live=out.live, threads=min(8, os.cpu_count() or 1)
        )


# ---- replayers per model kind: the verifier's model, cache and buffers

TOY_MAX_LEN = 4096  # the longest toy cache a verifier replays when it does not name one


def _int_field(m: dict, k: str, lo: int, hi: int) -> int:
    v = m.get(k)
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ReplayError(f"record's model {k} {v!r}: an integer from {lo} to {hi} is needed")
    return v


def toy_replayer(
    device="cpu", max_len: int | None = None, d: int = 64, n_layers: int = 4, vocab: int = 97, seed: int | None = None
) -> StaticCacheReplayer:
    """The demo prover's toy model (`memory_accounting.models.toy.toy_dynamic`), built from the seed the record names on the
    verifier's shape (d, n_layers, vocab) and cache length (max_len; None: the record's, up to TOY_MAX_LEN).
    seed pins the weights too (None: the record's). One model is kept at a time."""
    import torch

    from ..models.toy import ToyLM, toy_feed
    from ..prover.dynamic import act_buffer

    state: dict = {}

    def prepare(ev: Evidence):
        m = ev.model
        if m.get("kind", "toy") != "toy":
            raise ReplayError(f"record is from a {m.get('kind')} model, not the toy")
        for k, v in (("d", d), ("n_layers", n_layers), ("vocab", vocab)):
            if m.get(k, 97 if k == "vocab" else None) != v:
                raise ReplayError(f"record's toy model has {k} {m.get(k)!r}, the verifier's has {v}")
        if max_len is not None and m.get("max_len") != max_len:
            raise ReplayError(f"record's toy cache has {m.get('max_len')!r} slots, the verifier's {max_len}")
        ml = _int_field(m, "max_len", 1, TOY_MAX_LEN)
        s = _int_field(dict(m, seed=m.get("seed", 0)), "seed", 0, (1 << 63) - 1)
        if seed is not None and s != seed:
            raise ReplayError(f"record's toy model has seed {s}, the verifier's {seed}")
        if state.get("key") != (s, ml):
            state.clear()  # one model at a time
            torch.manual_seed(s)
            model = ToyLM(vocab=vocab, d=d, n_layers=n_layers, max_len=ml).to(device).eval()
            state.update(key=(s, ml), model=model, act=act_buffer(ml, d, torch.float32, device))
        model = state["model"]
        return Prepared(
            toy_feed(model),
            list(model.layers),
            model.kv_buffers,
            model.reset,
            model.live_kv_buffers,
            state["act"],
            ml,
            vocab,
            slot_dim=1,
        )

    return StaticCacheReplayer(prepare, device)


def int_replayer(path, device="cpu", max_len: int | None = None) -> StaticCacheReplayer:
    """The integer-exact model from its file; the record must name the same file hash. max_len: the verifier's
    cache length (a record with another is refused); None: the record's, up to the file's rope table, with one
    model whose cache is resized."""
    import torch

    from ..models.intllm import IntLLM, file_sha256, int_feed
    from ..prover.dynamic import act_buffer

    sha = file_sha256(path)
    state: dict = {}

    def prepare(ev: Evidence):
        model_info = ev.model
        if model_info.get("kind") != "int":
            raise ReplayError(f"record is from a {model_info.get('kind')} model, not the integer model")
        if model_info.get("sha256") != sha:
            raise ReplayError(
                f"record names model {str(model_info.get('sha256'))[:16]}…, the verifier holds {sha[:16]}…"
            )
        if max_len is not None and model_info.get("max_len") != max_len:
            raise ReplayError(f"record's cache has {model_info.get('max_len')!r} slots, the verifier's {max_len}")
        if "model" not in state:
            state["model"] = IntLLM.from_file(path, max_len or 1, device)
        model = state["model"]
        ml = _int_field(model_info, "max_len", 1, model.cfg["max_pos"])
        if model.max_len != ml or "act" not in state:
            state.pop("act", None)
            model.resize(ml)
            state["act"] = act_buffer(ml, model.d, torch.float64, device)
        return Prepared(
            int_feed(model),
            list(model.layers),
            model.kv_buffers,
            model.reset,
            model.live_kv_buffers,
            state["act"],
            ml,
            int(model.cfg["vocab"]),
            slot_dim=0,
        )

    return StaticCacheReplayer(prepare, device, coalesce=True)  # exact whatever the chunking


def hf_replayer(
    model, device="cpu", max_cache_len: int | None = None, module_re: str = r"layers\.(\d+)$"
) -> StaticCacheReplayer:
    """A HF model on a StaticCache of batch 1, the model's KV heads, head size and dtype, and the verifier's
    cache length (max_cache_len; None: the record's, up to the model's positions). Bit-exact only on the device
    type and kernels the prover used (a float model). The replay puts the filler in the cache (HF attends over
    every slot, masked), so the view reads the buffers as they are."""
    import torch
    from transformers import StaticCache

    from ..prover.dynamic import act_buffer, hf_cache_buffers, layer_modules

    layers = layer_modules(model, module_re)
    cfg = model.config
    heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    dtype = str(model.dtype).replace("torch.", "")
    limit = int(getattr(cfg, "max_position_embeddings", None) or 1 << 20)
    state: dict = {}

    def prepare(ev: Evidence):
        k0 = next((e for e in ev.layout if e.get("name") == "kv.0.k"), None)
        if k0 is None:
            raise ReplayError("record has no kv.0.k segment")
        shape = k0.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or shape[0] != 1
            or shape[1] != heads
            or shape[3] != head_dim
            or k0.get("dtype") != dtype
        ):
            raise ReplayError(
                f"record's cache {shape} {k0.get('dtype')} is not this model's (1, {heads}, slots, {head_dim}) {dtype}"
            )
        if max_cache_len is not None and shape[2] != max_cache_len:
            raise ReplayError(f"record's cache has {shape[2]!r} slots, the verifier's {max_cache_len}")
        ml = _int_field({"slots": shape[2]}, "slots", 1, limit)
        if state.get("ml") != ml:
            state.clear()
            c = StaticCache(cfg, max_cache_len=ml)
            c.early_initialization(
                batch_size=1, num_heads=heads, head_dim=head_dim, dtype=model.dtype, device=torch.device(device)
            )
            state.update(ml=ml, cache=c, act=act_buffer(ml, cfg.hidden_size, model.dtype, device))
        cache = state["cache"]
        pos = {"p": 0}

        def reset():
            cache.reset()
            pos["p"] = 0

        def feed(step: int, chunk: list) -> None:
            ids = torch.tensor(chunk, device=device)
            model(
                input_ids=ids,
                past_key_values=cache,
                cache_position=torch.arange(pos["p"], pos["p"] + ids.shape[1], device=device),
                use_cache=True,
            )
            pos["p"] += ids.shape[1]

        return Prepared(
            feed, layers, lambda: hf_cache_buffers(cache), reset, None, state["act"], ml, int(cfg.vocab_size)
        )

    return StaticCacheReplayer(prepare, device)


def _spec_len(spec: str) -> tuple[str, int | None]:
    """'<spec>@<slots>': the verifier's cache length, if given."""
    head, at, tail = spec.rpartition("@")
    if at and tail.isdigit():
        return head, int(tail)
    return spec, None


def replayer_from_spec(spec: str, device="cpu"):
    """'toy' | 'int:<file>' | 'hf:<model dir or name>', each optionally '@<slots>' (the verifier's KV cache
    length: records of another are refused); the verifier's choice of replay."""
    spec, ml = _spec_len(spec)
    if spec == "toy":
        return toy_replayer(device, max_len=ml)
    if spec.startswith("int:"):
        return int_replayer(spec[4:], device, max_len=ml)
    if spec.startswith("hf:"):
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(spec[3:], dtype=torch.bfloat16, device_map=device).eval()
        return hf_replayer(model, device, max_cache_len=ml)
    raise ValueError(f"unknown replay spec {spec!r} (toy, int:<file>, hf:<model>, each optionally @<cache slots>)")


# ---- the verifier's log: what was sent and received, which a record must describe exactly


def log_entries(path) -> dict:
    """The dynamic chains a verifier log (`memory_accounting.verifier.net live --out`, JSONL) says were sent: nonce -> {region,
    j0, received (lo, hi), words (READY's size), shape (B, n, r, k, g) if the log records it, fault, main}.
    main: for a live chain, the nonce of the main chain of the same verification (the same freeze), whose
    record's stage the live region is rebuilt from.
    A chain sent with no reply logged (the verification failed online: "no reply to the dynamic chain", or
    to the live chain) has received None and its fault; a sample with no nonce (no READY, or a region no
    chain could cover) sent no chain and is not listed.
    not_kept: the verifier's check of the chain says it bound the record but did not keep it ("record_saved":
    false, its --evidence-max-total reached): the record is absent through no act of the prover."""
    out: dict = {}
    unsaved: set = set()  # nonces whose bound record the verifier did not keep
    for line in Path(path).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        dc = e.get("dyn_check") if isinstance(e, dict) and e.get("type") == "check" else None
        if isinstance(dc, dict):
            for smp in (dc, dc.get("live")):
                if isinstance(smp, dict) and smp.get("record_saved") is False and isinstance(smp.get("nonce"), str):
                    try:
                        unsaved.add(int(smp["nonce"], 16))
                    except ValueError:
                        pass
            continue
        dyn = e.get("dyn") if isinstance(e, dict) and e.get("type") == "verification" else None
        if not isinstance(dyn, dict):
            continue
        main = int(dyn["nonce"], 16) if dyn.get("nonce") else None
        for region, smp in (("main", dyn), ("live", dyn.get("live"))):
            if not isinstance(smp, dict) or not smp.get("nonce"):
                continue
            r = smp.get("received")
            out[int(smp["nonce"], 16)] = {
                "region": region,
                "j0": int(smp["j0"]),
                "received": (int(r[16:], 16), int(r[:16], 16)) if r else None,
                "words": int(smp.get("words") or 0),
                "shape": tuple(smp.get("shape") or dyn.get("shape") or ()) or None,
                "id": e.get("id"),
                "fault": None if r else (dyn.get("fault") or "no reply logged"),
                "main": main if region == "live" else None,
            }
    for n, x in out.items():  # a check comes after its round in the log
        x["not_kept"] = n in unsaved
    return out


def log_mismatch(ev, entries: dict, shape: tuple) -> str | None:
    """Why a record does not stand for a chain the verifier's log says it sent, or None: the live
    verifier's rule (`memory_accounting.verifier.net._record_mismatch`), so the record must describe exactly the chain sent
    (block size, n, r, k, g, j0), over the region it was sent for and the size READY reported, with the
    digest that came back. shape: (B, n, r, k, g) for a log that does not record it."""
    from .net import _record_mismatch

    e = entries.get(int(ev.nonce))
    if e is None:
        return "the verifier's log has no dynamic challenge with this nonce"
    if e["received"] is None:
        return f"the verifier logged no reply to this challenge ({e['fault']}): it failed online, there is no digest to check"
    B, n, r, k, g = e["shape"] or shape
    ch = Challenge(int(B), int(n), int(r), int(ev.nonce), e["j0"], int(k), int(g))
    return _record_mismatch(ev, ch, e["received"], e["words"], e["region"])


def log_coverage(entries: dict, seen) -> tuple[list[int], list[int], list[int]]:
    """(missing, faulted, not_kept) nonces of the verifier's log: challenges that got a reply and have no
    record (the audit fails on them, whatever the reason the record is absent: a prover omitting the records
    it would fail must not pass); challenges with no reply (failed online; nothing to check, reported apart);
    and challenges whose record the verifier bound but did not keep (its log says so, `log_entries`: its
    --evidence-max-total was reached; online they were replayed, or alerted as unchecked; reported apart, not
    held against the prover)."""
    seen = {int(x) for x in seen}
    absent = [n for n, e in entries.items() if e["received"] is not None and n not in seen]
    missing = sorted(n for n in absent if not entries[n].get("not_kept"))
    not_kept = sorted(n for n in absent if entries[n].get("not_kept"))
    faulted = sorted(n for n, e in entries.items() if e["received"] is None)
    return missing, faulted, not_kept


def audit_record(ev, entries: dict | None, shape: tuple, check, main_of=None) -> dict:
    """One record of an offline audit: first against the verifier's log (entries None: the record is only
    checked against itself), then `check(ev, main)` (replay and chain: `verify`'s report). A record of a
    challenge the log says got no reply is not checked: it is reported with "fault" (it failed online) and
    counts neither as reproduced nor as a failed record.

    A live record is paired, through the log, with the main record of the same verification (`main_of(nonce)`
    loads it, None if there is none): it fails without one or with a stage that is not that record's, and is
    replayed from that record's stage. Without a log, main is None (the record's own stage is replayed)."""

    def fail(why):
        return {
            "nonce": f"{ev.nonce:016x}",
            "stage": ev.describe(),
            "received": hexd(ev.digest),
            "ok": False,
            "error": why,
        }

    e = entries.get(int(ev.nonce)) if entries is not None else None
    if e is not None and e["received"] is None:
        return dict(fail(f"no reply logged ({e['fault']}): failed online, not checked"), fault=e["fault"])
    why = log_mismatch(ev, entries, shape) if entries is not None else None
    if why is not None:
        return fail(why)
    main = None
    if e is not None and e["region"] == "live":
        mn = e.get("main")
        main = main_of(mn) if (mn is not None and main_of is not None) else None
        if main is None and mn is not None and (entries.get(mn) or {}).get("not_kept"):
            why = "the main record of its freeze was not kept by the verifier (--evidence-max-total): not checked"
            return dict(fail(why), fault=why)
        if main is None:
            return fail(
                f"no record of the main chain of the same freeze ({'none logged' if mn is None else f'{mn:016x}'}): "
                "the live region is the live part of that record's stage"
            )
        if int(main.nonce) != int(mn):
            return fail("the main record found for this live record has another nonce")
        if (why := live_mismatch(ev, main)) is not None:
            return fail(why)
    return check(ev, main)


def record_loader(evidence):
    """main_of for `audit_record`: the record <nonce>.json beside the evidence (a directory, or a file's
    directory), its header only, or None."""
    p = Path(evidence)
    d = p if p.is_dir() else p.parent

    def load(nonce: int):
        f = d / f"{int(nonce):016x}.json"
        if not f.exists():
            return None
        try:
            return Evidence(data=None, **json.loads(f.read_text()))
        except (OSError, ValueError, TypeError):
            return None

    return load


def audit_order(files: list, entries: dict | None) -> list:
    """Records in an order that puts each live record right after its main record (one replay serves both
    when the replayer keeps the last stage's)."""
    if entries is None:
        return files

    def key(f):
        try:
            n = int(Path(f).stem, 16)
        except ValueError:
            return (1 << 64, 0, str(f))
        e = entries.get(n) or {}
        return (e.get("main") or n, 1 if e.get("region") == "live" else 0, str(f))

    return sorted(files, key=key)


def coverage_lines(entries: dict, seen, partial: bool = False) -> tuple[list[str], bool]:
    """What to print about the log's coverage, and whether it passes (missing records fail unless partial;
    records the verifier's log says it did not keep never do)."""
    missing, faulted, not_kept = log_coverage(entries, seen)
    lines = []
    if faulted:
        lines.append(
            f"{len(faulted)} dynamic challenges in the log got no reply (failed online; no record to check), first {faulted[0]:016x}"
        )
    if not_kept:
        lines.append(
            f"{len(not_kept)} dynamic challenges in the log have a record not kept by the verifier (--evidence-max-total reached; "
            f"not counted against the prover), first {not_kept[0]:016x}"
        )
    if missing:
        lines.append(
            f"{len(missing)} dynamic challenges in the log have a reply but no record, first {missing[0]:016x}"
            + (" (--partial)" if partial else ": the audit fails")
        )
    return lines, (not missing) or partial


def checker_keys(
    filler_seed: str | None, pad_seed: str | None, manifest_seed: int | None, hash_seed: int | None
) -> Keys:
    """The challenge-hash keys a record checker recomputes chains with. They are derived from the verifier's
    filler seed (`Keys.from_seed`), never read from the prover's manifest: the seed given (hex), else the pad
    seed (hex; the filler seed unless the prover was given another), else the manifest's copy of the filler
    seed, with a warning (the prover's word). hash_seed, if given, overrides all of them with
    `Keys.deterministic` (a synthetic prover's or a test's keys); with nothing at all, the synthetic provers'
    default (--seed 1)."""
    if hash_seed is not None:
        return Keys.deterministic(hash_seed)
    seed = filler_seed if filler_seed is not None else pad_seed
    if seed is not None:
        if filler_seed is None:
            print(
                "hash keys: derived from --pad-seed (no --filler-seed). Records of a synthetic prover (memory_accounting.prover.udp) need --hash-seed <its --seed>.",
                flush=True,
            )
        return Keys.from_seed(int(seed, 16))
    if manifest_seed is not None:
        print(
            "WARNING: no --filler-seed (or --pad-seed): deriving the hash keys from the manifest's filler seed. Pass the seed you minted.",
            flush=True,
        )
        return Keys.from_seed(int(manifest_seed))
    return Keys.deterministic(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="check dynamic-challenge records by replay and chain recomputation")
    ap.add_argument("--evidence", required=True, help="record directory (every <nonce>.json) or one file")
    ap.add_argument(
        "--int-model",
        default=None,
        help="integer-exact model file (records from memory_accounting.prover.intserve or memory_accounting.prover.udp --dyn-model int:...)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="HF model dir or name (records from memory_accounting.prover.serve; same device type as the prover)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="verifier bundle manifest: supplies r and k (and, with neither --filler-seed nor --pad-seed, "
        "the filler seed the hash keys are derived from, with a warning)",
    )
    ap.add_argument(
        "--filler-seed",
        default=None,
        help="hex: the filler seed the verifier minted; the challenge-hash keys are derived from it "
        "(default: --pad-seed, which is the filler seed unless the prover was given another)",
    )
    ap.add_argument(
        "--hash-seed",
        type=int,
        default=None,
        help="Keys.deterministic seed: records of a synthetic prover (memory_accounting.prover.udp --seed) or of a "
        "test, whose keys are not derived from a filler seed. Overrides the derived keys",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the reports as JSON here")
    ap.add_argument(
        "--pad-seed",
        default=None,
        help="hex: the pad seed the verifier minted; a record claiming another fails, and the replay uses this one",
    )
    ap.add_argument("--gpu", type=int, default=0, help="with --pad-seed: the index of the GPU the verifier challenged")
    ap.add_argument(
        "--log",
        default=None,
        help="the verifier's log (memory_accounting.verifier.net live --out, JSONL): each record must describe exactly a dynamic chain "
        "the log says was sent (shape, j0, region, READY's size) and the reply received",
    )
    ap.add_argument(
        "--partial",
        action="store_true",
        help="with --log: the records are a subset; do not fail on logged challenges without one "
        "(challenges with no reply logged never need one)",
    )
    ap.add_argument(
        "--dyn-B",
        default="64K",
        help="with --log, for logs without the shape: the dynamic block size the verifier used",
    )
    ap.add_argument(
        "--dyn-n", type=int, default=8, help="with --log, for logs without the shape: the dynamic chain length"
    )
    ap.add_argument("--r", type=int, default=None, help="with --log: r (default: the manifest's, else 2048)")
    ap.add_argument("--k", type=int, default=None, help="with --log: k (default: the manifest's, else 1024)")
    ap.add_argument("--g", type=int, default=1)
    ap.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="the prover's KV cache length (slots) as the verifier configured it: records of "
        "another are refused before anything is built (default: any length the model allows, up to 4096 for the toy)",
    )
    args = ap.parse_args(argv)
    r_, k_ = 2048, 1024
    man = None
    if args.manifest:
        from ..setup.manifest import Manifest

        man = Manifest.load(args.manifest)
        r_, k_ = man.r, man.k
    keys = checker_keys(args.filler_seed, args.pad_seed, man.filler_seed if man is not None else None, args.hash_seed)
    from ..core.params import parse_bytes

    shape = (
        parse_bytes(args.dyn_B),
        args.dyn_n,
        args.r if args.r is not None else r_,
        args.k if args.k is not None else k_,
        args.g,
    )
    expected = Expected(int(args.pad_seed, 16), args.gpu) if args.pad_seed is not None else None
    entries = log_entries(args.log) if args.log else None
    notes = [
        x
        for x in (
            "without --pad-seed each record's own pad seed and GPU are used" if expected is None else "",
            "without --log each record is checked against itself, not against what was sent, and a live record is "
            "not paired with its main record (its own stage is replayed)"
            if entries is None
            else "",
        )
        if x
    ]
    if notes:
        print("note: " + "; ".join(notes), flush=True)
    path = Path(args.evidence)
    files = sorted(path.glob("*.json")) if path.is_dir() else ([path] if path.exists() else [])
    if not files and entries is None:
        raise SystemExit(f"no records under {path}")
    replayers: dict = {}

    def replayer_for(ev: Evidence):
        kind = ev.model.get("kind")
        if kind not in replayers:
            if kind == "toy":
                replayers[kind] = toy_replayer(args.device, max_len=args.max_len)
            elif kind == "int":
                if not args.int_model:
                    raise SystemExit("integer-model records: pass --int-model")
                replayers[kind] = int_replayer(args.int_model, args.device, max_len=args.max_len)
            elif kind == "hf":
                if not args.model:  # never the model the record names: the verifier's
                    raise SystemExit("HF records: pass --model (the verifier's copy of the declared model)")
                replayers[kind] = replayer_from_spec(
                    "hf:" + args.model + (f"@{args.max_len}" if args.max_len else ""), args.device
                )
            elif kind == "vllm":
                raise SystemExit(
                    "vLLM records: replay them with `python -m memory_accounting.verifier.vllm_replay replay` on the prover's GPU type"
                )
            else:
                raise SystemExit(f"record from an unknown model kind {kind!r}")
        return replayers[kind]

    lib = CpuLib()
    reports, ok = [], True
    seen = set()
    main_of = record_loader(path)
    for f in audit_order(files, entries):
        ev = Evidence.load(f)
        seen.add(int(ev.nonce))
        rep = audit_record(
            ev,
            entries,
            shape,
            lambda ev, main: verify(ev, replayer_for(main or ev), keys, lib, expected, main=main),
            main_of,
        )
        reports.append(rep)
        ok &= rep["ok"] or bool(rep.get("fault"))
        dbg = ev.debug or {}
        tag = (
            (
                f" [frozen by {dbg['frozen_by']}"
                + (
                    f", region {'unchanged' if dbg['region_unchanged'] else 'CHANGED'} during the challenge"
                    if "region_unchanged" in dbg
                    else ""
                )
                + "]"
            )
            if dbg.get("frozen_by")
            else ""
        )
        line = f"{rep['nonce']}: {rep['stage']}{tag}: " + (
            "digest reproduced"
            if rep["ok"]
            else f"FAULT ({rep['error']})"
            if rep.get("fault")
            else f"MISMATCH ({rep.get('error') or 'expected ' + rep.get('expected', '?') + ', received ' + rep['received']})"
        )
        if rep.get("differing_segments"):
            line += "; differing: " + ", ".join(
                f"{d['name']} ({d['words_differing']}/{d['of']} words)" for d in rep["differing_segments"][:6]
            )
        print(line)
    if entries is not None:
        lines, covered = coverage_lines(entries, seen, args.partial)
        for line in lines:
            print(line)
        ok &= covered
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))
    checked = [r for r in reports if not r.get("fault")]
    print(
        f"{sum(r['ok'] for r in checked)}/{len(checked)} records reproduced exactly"
        + (" and match the verifier's log" if entries is not None else "")
        + (
            f" ({len(reports) - len(checked)} of challenges that failed online, not checked)"
            if len(checked) < len(reports)
            else ""
        )
        + ("" if ok else "; AUDIT FAILED")
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
