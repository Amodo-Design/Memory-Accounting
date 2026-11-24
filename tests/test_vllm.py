"""The vLLM pool region on a simulated paged KV cache (CPU, no vLLM), with the real PagedPool and
VllmRegionView: the whole pool is in the region; free blocks hold dynamic filler (at setup and again as
soon as a request finishes); an allocated block's unwritten slots hold filler too; the verifier assembles
only the blocks a chain reads, from replayed pages on its own block ids (filler where the prover had not
written) plus recomputed filler. Every block of the verifier's view must equal the prover's region,
mid-step, between steps and idle, in two layer-major layouts."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS

import numpy as np
import torch

from memory_accounting.core.params import Challenge, Keys
from memory_accounting.models.vllm import page_bytes, physical_order
from memory_accounting.prover.dynamic import DynamicRegion
from memory_accounting.prover.vllm_serve import PagedPool
from memory_accounting.verifier.consistency import chain_digest, chain_digest_lazy
from memory_accounting.verifier.vllm_replay import (
    LiveRegionView,
    live_layout,
    live_layout_names,
    live_words,
    pool_record_of,
    region_view_from_replay,
    replay_pages,
)

L_, B_, H_, N_, D_ = 3, 16, 2, 4, 8  # layers, blocks in the pool, kv heads, block size (slots), head dim
C_ = 2 * D_  # K and V side by side, as vLLM's FlashAttention cache
HID, ROWS = 8, 16
BLOCK = 2048
KEYS = Keys.deterministic(5)
PW = H_ * N_ * C_ * 2 // 8  # page words (bf16)
ENGINE = {
    "block_size": N_,
    "n_layers": L_,
    "page_words": PW,
    "act_rows": ROWS,
    "hidden": HID,
    "dtype": "bfloat16",
    "max_model_len": 64,
    "kv_dtype": "bfloat16",
    "page_shape": [H_, N_, C_],
}


def make_kv(layout: str):
    if layout == "LBHNC":
        return [torch.zeros(B_, H_, N_, C_, dtype=torch.bfloat16) for _ in range(L_)]
    if layout == "LBNHC":  # pages dense but not contiguous in logical [H, N, C] order
        return [torch.zeros(B_, N_, H_, C_, dtype=torch.bfloat16).permute(0, 2, 1, 3) for _ in range(L_)]
    if layout == "BLHNC":  # block-major across layers: not a layer-major pool
        raw = torch.zeros(B_, L_, H_, N_, C_, dtype=torch.bfloat16)
        return [raw[:, layer] for layer in range(L_)]
    raise ValueError(layout)


def kv_value(layer, tok, pos):
    g = torch.Generator().manual_seed(layer * 1_000_003 + tok * 7919 + pos)
    return torch.randn(H_, C_, generator=g).to(torch.bfloat16)


def hidden(tok, pos, which):
    g = torch.Generator().manual_seed(tok * 31 + pos * 1009 + which)
    return torch.randn(HID, generator=g).to(torch.bfloat16)


def write(kv, blocks, layer, tok, pos):
    kv[layer][blocks[pos // N_]][:, pos % N_, :] = kv_value(layer, tok, pos)


def write_to(pool, blocks, layer, tok, pos):
    """Layer `layer`'s KV of a token into a pool tensor (the layer's own, or the encoded pool's plaintext scratch)."""
    pool[blocks[pos // N_]][:, pos % N_, :] = kv_value(layer, tok, pos)


TOKENS = {
    "A": [5, 9, 2, 7, 7, 1, 3, 8],
    "B": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "C": [21, 22, 23, 24, 25, 26],
    "D": [31, 32, 33, 34, 35, 36, 37],
}


def so(new=(), cached=(), sched=None, finished=(), preempted=None, resumed=()):
    """A fake SchedulerOutput (vLLM 0.30's V1 fields): new = [(rid, blocks)], cached = [(rid, new_blocks or None)];
    a resumed request's new_block_ids are its whole new block list."""
    sched = sched or {}
    return NS(
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted) if preempted is not None else None,
        scheduled_new_reqs=[NS(req_id=r, block_ids=(list(b),)) for r, b in new],
        scheduled_cached_reqs=NS(
            req_ids=[r for r, _ in cached],
            new_block_ids=[(list(b),) if b else None for _, b in cached],
            resumed_req_ids=set(resumed),
        ),
        num_scheduled_tokens=dict(sched),
        total_num_scheduled_tokens=sum(sched.values()),
    )


class Engine:
    """The prover side: a fake runner around a PagedPool, stepping like vLLM's V1 worker."""

    def __init__(self, layout, encoded=False):
        self.kv = make_kv(layout)
        for k in self.kv:  # stale data from before setup
            k.copy_(torch.randn(k.shape).to(torch.bfloat16))
        self.act_h = torch.zeros(ROWS, HID, dtype=torch.bfloat16)
        self.act_r = torch.zeros_like(self.act_h)
        self.pool = PagedPool(self.kv, self.act_h, self.act_r, BLOCK, pad_seed=0x77, encoded=encoded)
        self.enc = self.pool.enc
        self.pool.init_filler()
        self.req = {}
        self.step = 0
        self.so = None

    def forward_layer(self, layer, order, s):
        """Layer l of a step: an encoded pool decodes the step's pages into the scratch, the layer writes there, the
        written groups are encoded back (`VllmDynamic`'s hooks); a plain pool is written in place."""
        if self.enc is not None:
            self.enc.decode_layer(layer)
        target = self.kv[0] if self.enc is not None else self.kv[layer]
        for rid in order:
            st = self.req[rid]
            for p in range(st.num_computed_tokens, st.num_computed_tokens + s.num_scheduled_tokens[rid]):
                write_to(target, st.block_ids[0], layer, TOKENS[rid][p], p)
        if self.enc is not None:
            self.enc.encode_layer(layer)

    def act_rows(self, order, s, residual=True):
        rows = [
            (TOKENS[rid][p], p)
            for rid in order
            for p in range(
                self.req[rid].num_computed_tokens, self.req[rid].num_computed_tokens + s.num_scheduled_tokens[rid]
            )
        ]
        return (
            torch.stack([hidden(t, p, 0) for t, p in rows]),
            torch.stack([hidden(t, p, 1) for t, p in rows]) if residual else None,
            len(rows),
        )

    def run_step(self, sched, stop_at=None, residual=True):
        """begin (free, zero) -> vLLM's state update -> forward (all layers, or freeze before stop_at) -> end."""
        self.pool.begin_step(sched, self.req)
        for rid in sched.finished_req_ids:
            self.req.pop(rid, None)
        for n in sched.scheduled_new_reqs:  # (a known id: a streaming session's next turn keeps its KV)
            old = self.req.get(n.req_id)
            self.req[n.req_id] = NS(
                prompt_token_ids=list(TOKENS[n.req_id]),
                output_token_ids=[],
                num_computed_tokens=old.num_computed_tokens if old else 0,
                block_ids=(list(n.block_ids[0]),),
            )
        # the worker keeps a preempted request's state, stale block list included, until it resumes or ends
        for rid in sched.preempted_req_ids or ():
            self.req[rid].num_computed_tokens = 0
        for rid, nb in zip(sched.scheduled_cached_reqs.req_ids, sched.scheduled_cached_reqs.new_block_ids):
            if rid in sched.scheduled_cached_reqs.resumed_req_ids:  # resumed: the new list replaces the old one
                self.req[rid].block_ids = (list(nb[0]),)
                self.req[rid].num_computed_tokens = 0
            elif nb:
                self.req[rid].block_ids[0].extend(nb[0])
        self.so = sched
        self.step += 1
        order = list(sched.num_scheduled_tokens)
        if self.enc is not None and order:
            self.enc.plan(
                [
                    (self.req[r].block_ids[0], self.req[r].num_computed_tokens, sched.num_scheduled_tokens[r])
                    for r in order
                ]
            )
        layers = range(L_) if stop_at is None else range(stop_at)
        for layer in layers:
            if self.enc is not None and order:  # every layer boundary: its inputs, encoded
                self.enc.act_rows(*self.act_rows(order, sched, residual or layer > 0))
            self.forward_layer(layer, order, sched)
        if stop_at is not None:  # frozen before layer stop_at: its inputs
            if self.enc is not None:
                self.enc.act_rows(*self.act_rows(order, sched, residual))
            else:
                self.pool.capture(*self.act_rows(order, sched, residual))
            return order
        self.finish_step(sched)

    def finish_step(self, s):
        self.pool.end_step(s, self.req)
        if self.enc is not None:  # between steps: no activation rows
            self.enc.end_plan()
            self.enc.clear_act()
        for rid, n in s.num_scheduled_tokens.items():
            self.req[rid].num_computed_tokens += n

    def info(self):
        """What VllmDynamic's model info says about the pool (the verifier places unwritten slots with it)."""
        return {
            "dtype": "bfloat16",
            "kv_dtype": "bfloat16",
            "gpu": 0,
            "page_shape": list(self.kv[0].shape[1:]),
            "page_stride": list(self.kv[0][0].stride()),
            "block_size": N_,
            "act_rows": ROWS,
            "hidden": HID,
            "max_model_len": 64,
        }

    def record(self, stage):
        ev = NS(
            stage=stage,
            layout=self.pool.region.layout(),
            words=self.pool.region.words,
            block_bytes=BLOCK,
            pad_seed=0x77,
            model=self.info(),
            nonce=0x1234,
            challenge={"block_bytes": BLOCK, "n": 32, "r": 0, "k": 1024, "g": 1, "j0": 1},
        )
        return ev


def assert_live_same(engine, ev, layout):
    """The second chain: the prover's live region (its pages in place, the activation rows) against the
    verifier's, built from the same replay as the pool view."""
    st = ev.stage
    live = DynamicRegion(engine.pool.live_segments(st), BLOCK, 0x77, device=engine.kv[0].device, live=True)
    lev = NS(
        stage=st,
        layout=live.layout(),
        words=live.words,
        block_bytes=BLOCK,
        pad_seed=0x77,
        model=engine.info(),
        region="live",
        nonce=0x4321,
        challenge={"block_bytes": BLOCK, "n": 32, "r": 0, "k": 1024, "g": 1, "j0": 0},
    )
    assert [e["name"] for e in lev.layout if e["name"] != "pad"] == live_layout_names(lev)
    assert lev.layout == live_layout(st, ENGINE, BLOCK), (
        "the live layout the verifier rebuilds from the stage and its engine"
    )
    view = verifier_view(layout, pool_record_of(lev, ENGINE))
    mine = live_words(lev, view)
    assert np.array_equal(mine, live.snapshot()), "live region differs"
    from memory_accounting.prover.dynamic import DynamicResponder, PauseController, _words_of

    # list of buffers, and the array form the prover uses
    for fn in (
        engine.pool.live_segments,
        engine.pool.live_pages,
    ):
        dyn = DynamicResponder(engine.pool.region, lambda: st, PauseController(), KEYS, live_fn=fn)
        dyn.stage = st
        assert dyn._build_live() == live.words
        a, b, n = dyn.live_region._segtab_np, live._segtab_np, live.nseg
        assert dyn.live_region.nseg == n and np.array_equal(a[: n + 1], b[: n + 1]), "segment starts differ"
        own = np.array([seg.name == "pad" for seg in live.segments])  # each region has its own pad buffer
        assert np.array_equal(a[n + 1 :][~own], b[n + 1 :][~own]), "segment addresses differ"
        lazy_layout = dyn.live_region.layout_fn()  # what the live record will carry
        assert lazy_layout() == live.layout() == dyn.live_region.layout()
        assert np.array_equal(dyn.live_region.snapshot(), mine)
    for seg in live.segments:  # the chain reads through the table: its addresses
        assert seg.ptr == _words_of(seg).data_ptr(), seg.name  # (page arithmetic) must be the views' addresses
    ch_ev = NS(**{k: getattr(lev, k) for k in ("challenge", "nonce", "words")})
    assert chain_digest(live.snapshot(), ch_ev, KEYS) == chain_digest(mine, ch_ev, KEYS)
    assert chain_digest_lazy(LiveRegionView(lev, view), ch_ev, KEYS) == chain_digest(mine, ch_ev, KEYS), (
        "the lazy live view"
    )
    return live.words


def verifier_view(layout, ev, residual=True):
    """The verifier: replay each live request on its own pool (other block ids, never zeroed, stale data)."""
    per = []
    for i, r in enumerate(ev.stage["requests"]):
        kv = make_kv(layout)
        for k in kv:
            k.copy_(torch.randn(k.shape).to(torch.bfloat16))
        toks, computed, scheduled = r["tokens"], r["num_computed"], r["scheduled"]
        bl = [B_ - 1 - (i * 5 + j) % B_ for j in range(r["nblocks"])]
        for layer in range(L_):
            for p in range(computed + scheduled):
                write(kv, bl, layer, toks[p], p)
        in_flight = ev.stage["in_flight"] and scheduled > 0
        got = {"pages": replay_pages(kv, bl, computed, scheduled, ev.stage["act_layer"], in_flight)}
        if in_flight:
            got["act_h"] = (
                torch.stack([hidden(toks[p], p, 0) for p in range(computed, computed + scheduled)])
                .view(torch.uint8)
                .numpy()
                .reshape(-1)
            )
            got["act_r"] = (
                torch.stack([hidden(toks[p], p, 1) for p in range(computed, computed + scheduled)])
                .view(torch.uint8)
                .numpy()
                .reshape(-1)
                if residual
                else None
            )
        per.append(got)
    return region_view_from_replay(ev, per, engine=ENGINE)


def assert_same(engine, ev, view):
    prover = engine.pool.region.snapshot()
    bw = BLOCK // 8
    for j in range(ev.words // bw):
        assert np.array_equal(view.block(j, bw), prover[j * bw : (j + 1) * bw]), f"block {j} differs"
    ch_ev = NS(**{k: getattr(ev, k) for k in ("challenge", "nonce", "words")})
    assert chain_digest(prover, ch_ev, KEYS) == chain_digest_lazy(view, ch_ev, KEYS)


def scenario(layout, encoded=False):
    e = Engine(layout, encoded)
    filler = e.pool.region.filler_view()
    assert np.array_equal(e.pool.region.snapshot(), filler), "after setup the whole pool is filler"
    assert all(torch.isfinite(k).all() for k in e.kv)
    # step 1: prefill A (8 tokens, blocks 3, 9) and B (prompt 9, blocks 0, 14, 6)
    e.run_step(so(new=[("A", [3, 9]), ("B", [0, 14, 6])], sched={"A": 6, "B": 9}))
    # step 2: one token each
    e.run_step(so(cached=[("A", None), ("B", None)], sched={"A": 1, "B": 1}))
    # step 3: A has finished; C is new and is given A's freed block 3 (not refilled as freed, C owns it: filled as
    # allocated); B decodes; freeze before layer 1
    order = e.run_step(
        so(new=[("C", [3, 11])], cached=[("B", None)], sched={"C": 5, "B": 1}, finished=["A"]), stop_at=1
    )
    st = e.pool.stage(e.so, e.req, order, True, 1, e.step, L_)
    assert [r["num_tokens"] for r in st["requests"]] == [5, 11] and [r["scheduled"] for r in st["requests"]] == [5, 1]
    ev = e.record(st)
    view = verifier_view(layout, ev)
    assert_same(e, ev, view)
    lw = assert_live_same(e, ev, layout)
    assert lw < ev.words // 2, "the live region is a small part of the pool region"
    pw = e.pool.page_stride_words
    snap = e.pool.region.snapshot()
    o9 = 0 * B_ * pw + 9 * pw  # layer 0, block 9: A's other block, freed, filler again
    assert np.array_equal(snap[o9 : o9 + pw], filler[o9 : o9 + pw])
    # the step completes; between steps
    for layer in range(1, L_):
        if e.enc is not None:
            e.enc.act_rows(*e.act_rows(order, e.so))
        e.forward_layer(layer, order, e.so)
    e.finish_step(e.so)
    e.pool.clear_act()
    st = e.pool.stage(e.so, e.req, [], False, 0, e.step, L_)
    assert sorted(r["num_tokens"] for r in st["requests"]) == [5, 11]
    ev = e.record(st)
    assert_same(e, ev, verifier_view(layout, ev))
    assert_live_same(e, ev, layout)
    # a cheater hides data in a free block: the verifier's view (filler there) no longer matches
    hide = e.enc.stores[2].tensor[13 * pw : (13 + 1) * pw] if encoded else e.kv[2][13]
    keep = hide.clone()
    if encoded:
        hide[5] ^= 1
    else:
        hide[0, 1, 3] = 1.0
    prover = e.pool.region.snapshot()
    bw = BLOCK // 8
    o = 2 * B_ * pw + 13 * pw
    j = o // bw
    assert not np.array_equal(verifier_view(layout, ev).block(j, bw), prover[j * bw : (j + 1) * bw])
    hide.copy_(keep)
    # B and C finish; the scheduler tells the worker in the next step, which runs nothing: idle, all filler
    e.run_step(so(finished=["B", "C"]))
    e.pool.clear_act()
    st = e.pool.stage(e.so, e.req, [], False, 0, e.step, L_)
    assert st["requests"] == []
    assert np.array_equal(e.pool.region.snapshot(), filler), (
        "every freed block and the activation rows are filler again"
    )
    assert e.pool.freed_blocks == 1 + 5 and e.pool.refill_skipped == 1 and e.pool.alloc_filled == 1, (
        "a block reallocated in the step it is freed gets its filler as allocated, not as freed"
    )


def test_pool_region_matches_verifier_view():
    for layout in ("LBHNC", "LBNHC"):
        scenario(layout)


def test_encoded_pool_region_matches_verifier_view():
    """v0.17: the pool and the activation rows held encoded, attention on a plaintext scratch of one layer (layer 0's
    pool): the same scenario, every block of the verifier's view (the replayed pages and rows, encoded at the prover's
    positions, encoded zeros elsewhere) equal to the prover's region, and the live region too."""
    for layout in ("LBHNC", "LBNHC"):
        scenario(layout, encoded=True)


def test_layer_zero_freeze_without_residual():
    """Frozen before layer 0, vLLM has no residual yet: the prover leaves those activation rows as filler,
    and the verifier's view must too (it used to put zeros there, failing an honest freeze)."""
    for layout, encoded in (("LBHNC", False), ("LBNHC", False), ("LBHNC", True)):
        e = Engine(layout, encoded)
        e.run_step(so(new=[("A", [3, 9]), ("B", [0, 14, 6])], sched={"A": 6, "B": 9}))
        order = e.run_step(so(cached=[("A", None), ("B", None)], sched={"A": 1, "B": 1}), stop_at=0, residual=False)
        ev = e.record(e.pool.stage(e.so, e.req, order, True, 0, e.step, L_))
        assert_same(e, ev, verifier_view(layout, ev, residual=False))


def test_preemption_never_refills_live_blocks():
    """vLLM V1 keeps a preempted request's state on the worker, its old block list included, until the
    request resumes (the list is replaced) or ends. The scheduler frees those blocks at the preemption and
    may hand them to another request in the same step. A request aborted while preempted then ended with a
    stale list, and the refill wrote filler over the other request's live KV. The pool now refills only
    blocks no live request of the step owns, and a preempted request's blocks once."""
    TOKENS.update(
        {"P": list(range(40, 52)), "Q": list(range(60, 76)), "R": list(range(80, 90)), "S": list(range(90, 100))}
    )

    def check(e, layout):
        ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))
        assert_same(e, ev, verifier_view(layout, ev))
        assert_live_same(e, ev, layout)
        return sorted(r["num_tokens"] for r in ev.stage["requests"])

    for layout in ("LBHNC", "LBNHC"):
        e = Engine(layout)
        e.run_step(so(new=[("P", [3, 9]), ("Q", [0, 14, 6])], sched={"P": 6, "Q": 9}))
        # P is preempted to make room: its blocks are freed and block 3 goes to R in the same step
        e.run_step(so(new=[("R", [3, 11])], cached=[("Q", None)], sched={"R": 5, "Q": 1}, preempted=["P"]))
        assert check(e, layout) == [5, 10], "the preempted request is not live"
        # P is aborted while waiting: it ends with its stale block list [3, 9], and block 3 is R's
        e.run_step(so(cached=[("R", None), ("Q", None)], sched={"R": 1, "Q": 1}, finished=["P"]))
        assert check(e, layout) == [6, 11]
        # S is preempted and resumed on other blocks; R takes S's old block 9
        e.run_step(so(new=[("S", [9, 12])], cached=[("R", None), ("Q", None)], sched={"S": 7, "R": 1, "Q": 1}))
        e.run_step(so(cached=[("R", None), ("Q", [13])], sched={"R": 1, "Q": 1}, preempted=["S"]))
        assert check(e, layout) == [8, 13]
        e.run_step(so(cached=[("S", [5, 7]), ("R", [9]), ("Q", None)], sched={"S": 7, "R": 1, "Q": 1}, resumed=["S"]))
        assert check(e, layout) == [7, 9, 14]
        # S and Q finish: S's new blocks are freed, its old ones (R's block 9) are not freed again
        e.run_step(so(cached=[("R", None)], sched={"R": 1}, finished=["S", "Q"]))
        assert check(e, layout) == [10]


def test_unwritten_slots_are_filler_so_a_fake_request_gains_nothing():
    """Review finding: fresh blocks were zeroed, and the stage the prover writes says which slots it had not
    written yet. A fabricated request (num_computed 0, all its tokens scheduled, frozen before layer 0) then
    had every page of every layer "unwritten": expected zero, so the prover could hold nothing there (zeros
    cost nothing to fake) and be credited for it. Unwritten slots now hold filler and the verifier rebuilds
    them as filler: zeros fail, and a prover that keeps the filler holds what free memory would."""
    for layout in ("LBHNC", "LBNHC"):
        e = Engine(layout)
        e.run_step(so(new=[("A", [3, 9])], sched={"A": 6}))
        TOKENS["F"] = list(range(200, 208))
        fake = {
            "tokens": TOKENS["F"],
            "num_computed": 0,
            "scheduled": 8,
            "num_tokens": 8,
            "nblocks": 2,
            "block_ids": [5, 6],
        }
        st = e.pool.stage(e.so, e.req, [], False, 0, e.step, L_)
        st = dict(st, in_flight=True, act_layer=0, requests=[fake] + st["requests"])
        ev = e.record(st)
        pw, bw = e.pool.page_words, BLOCK // 8
        pages = [layer * B_ * pw + b * pw for layer in range(L_) for b in (5, 6)]

        def expected():
            v = verifier_view(layout, ev)
            full = np.concatenate([v.block(j, bw) for j in range(ev.words // bw)])
            return np.concatenate([full[o : o + pw] for o in pages])

        def held():
            snap = e.pool.region.snapshot()
            return np.concatenate([snap[o : o + pw] for o in pages])

        assert np.array_equal(expected(), held()), "the fake request's pages must be exactly the filler they hold"
        assert np.array_equal(held(), np.concatenate([e.pool.region.filler_view()[o : o + pw] for o in pages]))
        for k in e.kv:  # the cheat: drop the memory, keep zeros
            k[5].zero_()
            k[6].zero_()
        assert not np.array_equal(expected(), held()), "zeros passed for a fabricated request's pages"


def test_stage_fields_are_checked_before_replay():
    """The replay trusts the stage's token counts and block ids: a request with tokens missing, blocks it
    does not need, a block outside the pool or shared with another request, or a paused layer outside the
    model, is refused before anything is replayed (and again when the view is built)."""
    from memory_accounting.verifier.vllm_replay import check_stage

    e = Engine("LBHNC")
    e.run_step(so(new=[("A", [3, 9]), ("B", [0, 14, 6])], sched={"A": 6, "B": 9}))
    ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))
    assert_same(e, ev, verifier_view("LBHNC", ev))
    import copy

    def refused(mutate, why):
        bad = copy.deepcopy(ev)
        mutate(bad.stage)
        try:
            check_stage(bad.stage)
            return False  # VllmReplayer.region_words, before it replays
        except ValueError as err:
            assert why in str(err), err
            return True

    assert refused(lambda st: st["requests"][0].update(tokens=st["requests"][0]["tokens"][:-1]), "tokens for")
    assert refused(lambda st: st["requests"][0].update(num_computed=2), "tokens for")
    assert refused(lambda st: st["requests"][0].update(block_ids=[3, 9, 10], nblocks=3), "needed")
    assert refused(lambda st: st["requests"][0].update(block_ids=[3, B_]), "outside the pool")
    assert refused(lambda st: st["requests"][1].update(block_ids=[0, 14, 3]), "listed by requests")
    assert refused(lambda st: st.update(in_flight=True, act_layer=L_), "outside the model")
    bad = copy.deepcopy(ev)
    bad.stage["requests"][1]["block_ids"] = [0, 14, 3]
    try:
        region_view_from_replay(
            bad,
            [{"pages": [np.zeros(e.pool.page_words, np.uint64)] * (L_ * r["nblocks"])} for r in bad.stage["requests"]],
            engine=ENGINE,
        )
        raise AssertionError("a block listed twice was accepted")
    except ValueError as err:
        assert "listed by requests" in str(err)


def test_merge_matches_the_page_layout():
    """The verifier places a page's unwritten slots by the page's shape and strides; in physical order a
    slot's elements are not contiguous (heads outermost), so this must match masked_page_bytes exactly."""
    from memory_accounting.verifier.vllm_replay import masked_page_bytes, merge_page, slot_index

    for layout in ("LBHNC", "LBNHC"):
        kv = make_kv(layout)
        page = kv[0][2]
        page.copy_(torch.randn(page.shape).to(torch.bfloat16))
        slots = slot_index(page.shape, page.stride())
        for valid in range(N_ + 1):
            full = masked_page_bytes(page, N_).view(np.uint64)
            masked = masked_page_bytes(page, valid).view(np.uint64)
            other = np.full_like(full, 0x0101010101010101)
            got = merge_page(full, other, valid, slots)
            assert (
                np.array_equal(got.view(np.uint16) == other.view(np.uint16), masked.view(np.uint16) == 0) or valid == N_
            ), (layout, valid)
            assert np.array_equal(merge_page(full, np.zeros_like(full), valid, slots), masked), (layout, valid)
        assert physical_order(page) == sorted(range(3), key=lambda d: -page.stride(d))


def test_replay_audit_needs_a_record_for_every_logged_reply():
    """`python -m memory_accounting.verifier.vllm_replay replay --log` checked only the records present: an empty evidence directory
    passed (exit 0) against a log with a challenge in it, and so did one missing the records a prover chose
    to withhold. Every logged challenge that got a reply now needs its record; one with no reply (failed
    online) needs none. (No vLLM here: the replay engine is a stub; no record is replayed.)"""
    import json
    import tempfile

    import memory_accounting.verifier.vllm_replay as vd

    class NoEngine:
        def __init__(self, *a, **kw):
            raise AssertionError("nothing to replay: the engine must not be started")

    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "ev").mkdir()
    sent = {
        "type": "verification",
        "id": 1,
        "dyn": {"nonce": f"{0x5151:016x}", "j0": 0, "received": "0" * 32, "words": 8192, "blocks": 1},
    }
    lost = {
        "type": "verification",
        "id": 2,
        "dyn": {
            "nonce": f"{0x5252:016x}",
            "j0": 0,
            "words": 8192,
            "blocks": 1,
            "fault": "no reply to the dynamic chain",
        },
    }

    def run(*lines):
        log = tmp / "live.jsonl"
        log.write_text("".join(json.dumps(e) + "\n" for e in lines))
        orig, vd.VllmReplayer = vd.VllmReplayer, NoEngine
        try:
            vd.main(["replay", "--evidence", str(tmp / "ev"), "--model", "unused", "--log", str(log)])
            return 0
        except SystemExit as e:
            return e.code
        finally:
            vd.VllmReplayer = orig

    assert run(sent) == 1, "an empty evidence directory passed against a logged challenge"
    assert run(lost) == 0, "a challenge that got no reply was required to have a record"
    assert run(sent, lost) == 1
    try:
        vd.main(["replay", "--evidence", str(tmp / "ev"), "--model", "unused"])
        raise AssertionError("no exit")
    except SystemExit as e:
        assert e.code not in (0, None), "no records and no log passed"


def test_non_layer_major_layout_is_refused():
    kv = make_kv("BLHNC")
    act = torch.zeros(ROWS, HID, dtype=torch.bfloat16)
    try:
        PagedPool(kv, act, act.clone(), BLOCK, 0x77)
        raise AssertionError("accepted a block-major layout")
    except ValueError as e:
        assert "LBNHC" in str(e)


def test_pages_are_dense_views():
    for layout in ("LBHNC", "LBNHC", "BLHNC"):
        kv = make_kv(layout)
        pb = page_bytes(kv[1][5])
        assert pb.numel() == H_ * N_ * C_ * 2 and pb.data_ptr() == kv[1][5].data_ptr()
        kv[1][5][1, 2, 3] = 2.0
        assert pb.view(torch.bfloat16).abs().max() == 2.0, layout


def test_streaming_session_keeps_its_kv():
    """Review finding: vLLM 0.30 schedules a streaming-input session's next turn as a new request with the same
    id and its whole block list (its earlier turns' blocks, whose KV it keeps, then new ones). The pool took
    every block of a new request as freshly allocated and refilled those given out since their last fill, so
    the session's KV became filler under it. Only the blocks it did not hold are new now."""
    TOKENS["T"] = list(range(300, 312))
    for layout in ("LBHNC", "LBNHC"):
        e = Engine(layout)
        e.run_step(so(new=[("T", [3, 9])], sched={"T": 6}))  # turn 1: 6 tokens of KV in blocks 3, 9
        pw = e.pool.page_words
        held = e.pool.region.snapshot()[3 * pw : 4 * pw].copy()  # layer 0, block 3: turn 1's KV
        assert not np.array_equal(held, e.pool.region.filler_view()[3 * pw : 4 * pw])
        ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))  # between turns the session is live
        assert [r["num_tokens"] for r in ev.stage["requests"]] == [6]
        assert_same(e, ev, verifier_view(layout, ev))
        e.run_step(so(new=[("T", [3, 9, 10])], sched={"T": 3}))  # turn 2: the same id, as a new request
        assert np.array_equal(e.pool.region.snapshot()[3 * pw : 4 * pw], held), "the session's KV was refilled"
        assert e.pool.alloc_filled == 0
        ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))
        assert [r["num_tokens"] for r in ev.stage["requests"]] == [9]
        assert_same(e, ev, verifier_view(layout, ev))
        assert_live_same(e, ev, layout)


def test_pool_layout_is_rebuilt_from_the_engine():
    """Review finding: the verifier's pool view took the segment offsets from the record's layout, checking only
    the names other than "pad": a negative pad after layer 0 and a longer one at the end kept the names and the
    total, and the verifier then expected layer 1's live pages at layer 0's positions and layer 0's nowhere.
    The layout is now rebuilt from the stage and the verifier's engine and must be the record's exactly."""
    import copy

    e = Engine("LBHNC")
    e.run_step(so(new=[("A", [3, 9]), ("B", [0, 14, 6])], sched={"A": 6, "B": 9}))
    ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))
    assert_same(e, ev, verifier_view("LBHNC", ev))
    W, pad = ev.layout[0]["words"], ev.layout[-1]

    def refused(layout):
        bad = copy.copy(ev)
        bad.layout = layout
        try:
            verifier_view("LBHNC", bad)
            return False
        except ValueError as err:
            assert "layout" in str(err), err
            return True

    assert refused(
        [ev.layout[0], {"name": "pad", "dtype": "int64", "shape": [-W], "words": -W}]
        + ev.layout[1:-1]
        + [dict(pad, words=pad["words"] + W, shape=[pad["words"] + W])]
    ), "the reviewer's overlay"
    h = ev.layout[L_]
    assert refused(
        ev.layout[:L_]
        + [dict(h, words=h["words"] + 8, shape=[ROWS + 4, HID])]
        + ev.layout[L_ + 1 : -1]
        + [dict(pad, words=pad["words"] - 8, shape=[pad["words"] - 8])]
    ), "activation rows of another size"
    assert refused(ev.layout[:L_] + [pad] + ev.layout[L_:-1]), "a pad before the activation rows"
    assert refused(ev.layout[:-1] + [pad, dict(pad, words=0, shape=[0])]), "two pads"


def test_engine_settings_are_checked():
    """The stage's and the record's block size, layers, page size and context length must be the verifier's
    engine's (the replay cuts its block list to the stage's count, which holds only with equal blocks; the
    context length sets the activation rows on both sides)."""
    import copy

    e = Engine("LBHNC")
    e.run_step(so(new=[("A", [3, 9])], sched={"A": 6}))
    ev = e.record(e.pool.stage(e.so, e.req, [], False, 0, e.step, L_))
    assert_same(e, ev, verifier_view("LBHNC", ev))
    for where, key, value in (
        ("model", "block_size", 2 * N_),
        ("model", "max_model_len", 128),
        ("model", "act_rows", ROWS + 1),
        ("stage", "page_words", PW // 2),
        ("stage", "n_layers", L_ + 1),
    ):
        bad = copy.deepcopy(ev)
        getattr(bad, where)[key] = value
        try:
            verifier_view("LBHNC", bad)
            raise AssertionError(f"{where} {key} {value} accepted")
        except (ValueError, IndexError, KeyError):
            pass


def test_replay_capture_reports_a_short_block_list():
    """ReplayCapture cut the replayed request's block list to the stage's count without checking it: with
    another block size the pages of missing blocks were silently dropped. A short list is now an error the
    driver reports for the record (raised in the hook, it would end the verifier's engine)."""
    from memory_accounting.verifier.vllm_replay import ReplayCapture

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(L_)])

    model = M()
    runner = NS(
        get_model=lambda: model,
        kv_caches=make_kv("LBHNC"),
        input_batch=NS(req_ids=["r", None]),
        requests={"r": NS(block_ids=([3],))},
    )
    cap = ReplayCapture(NS(model_runner=runner))
    cap.arm(0, 6, 0, False, 2)  # 6 tokens: 2 blocks of 4; the engine holds them in 1
    cap._post(model, (), None)
    out = cap.take()
    assert "pages" not in out and "1 blocks" in out["error"], out


def _pieces(region):
    """(first word, words, address) of each entry of the segment table the region is read through now."""
    n, t = region.nseg, region._segtab_np.view(np.uint64)
    return [(int(t[i]), int(t[i + 1] - t[i]), int(t[n + 1 + i])) for i in range(n)]


def _own_address(region, q: int) -> int:
    """Where the region's own table puts word q."""
    own, n = region._own[0].view(np.uint64), region._own[2]
    i = int(np.searchsorted(own[: n + 1], q, side="right")) - 1
    return int(own[n + 1 + i]) + 8 * (q - int(own[i]))


def _through_table(region) -> np.ndarray:
    """The region as a chain reads it through the current table: the words at each entry's address (CPU)."""
    import ctypes

    return np.concatenate(
        [np.ctypeslib.as_array((ctypes.c_uint64 * w).from_address(p)).copy() for _, w, p in _pieces(region)]
    )


def _check_dram_table(pool, stage, dram):
    """The dyn_dram table of a frozen stage: the free blocks of the set (the top of every layer's pool) read from
    the host copy; live blocks, blocks below the set, the activation rows and the pad at their own addresses;
    every word the chain reads is the region's own."""
    region, pw = pool.region, pool.page_words
    b0 = pool.n_blocks - dram.ranges[0][2]
    live = {int(b) for r in stage["requests"] for b in r["block_ids"]}
    want = np.zeros(region.words, dtype=bool)
    for layer in range(len(pool.kv)):
        o = region.find(f"kv.{layer}").offset
        for b in range(b0, pool.n_blocks):
            if b not in live:
                want[o + b * pw : o + (b + 1) * pw] = True
    got = np.zeros(region.words, dtype=bool)
    for q, w, p in _pieces(region):
        if dram.reads_host([p])[0]:
            got[q : q + w] = True
        else:
            assert p == _own_address(region, q), f"word {q}: an HBM piece at another address"
    assert np.array_equal(got, want), "host pieces are not the set's free blocks"
    assert dram.host_words_read == int(want.sum())
    assert np.array_equal(_through_table(region), region.snapshot()), (
        "the cheat reads other words than the region holds"
    )
    return int(want.sum())


def test_dyn_dram_reads_free_pool_blocks_from_host():
    """The dyn_dram posture: the top half of every layer's block ids keeps its filler in a host copy; at each freeze
    the blocks of it no live request lists are read from there, everything else from the pool. The digest is the
    honest one (and the verifier's), the table follows the stage (a block freed since is read from host memory),
    the live chain is untouched, and switching back restores the region's own table."""
    from memory_accounting.prover.dynamic import DynamicResponder, PauseController

    for layout in ("LBHNC", "LBNHC"):
        e = Engine(layout)
        region = e.pool.region
        e.run_step(so(new=[("A", [3, 9]), ("B", [0, 14, 6])], sched={"A": 6, "B": 9}))
        order = e.run_step(so(cached=[("A", None), ("B", None)], sched={"A": 1, "B": 1}), stop_at=1)
        stage = {"st": e.pool.stage(e.so, e.req, order, True, 1, e.step, L_)}
        dyn = DynamicResponder(
            region, lambda: stage["st"], PauseController(), KEYS, live_fn=e.pool.live_pages, dram_fn=e.pool.dram_plan
        )
        ch = Challenge(BLOCK, 32, 0, 0x1234, 1, 1024)

        def round_(live=False):
            dyn.arm(None, 1.0, want_live=live)
            got = dyn.answer(ch)
            lv = dyn.answer(Challenge(BLOCK, 32, 0, 0x4321, 0, 1024), live=True) if live else None
            return got, lv

        honest, honest_live = round_(live=True)
        dyn.release()
        dyn.set_posture("dyn_dram")
        dram = dyn.dram
        assert dram.fraction == 0.5 and dram.words == L_ * (B_ // 2) * e.pool.page_words and region.own_table, (
            "made once; applied at the freeze"
        )
        cheat, cheat_live = round_(live=True)
        assert not region.own_table
        # blocks 8..15 but 9 and 14 (A's, B's)
        assert _check_dram_table(e.pool, stage["st"], dram) == L_ * 6 * e.pool.page_words
        assert cheat == honest and cheat_live == honest_live
        ev = e.record(stage["st"])
        assert (
            chain_digest_lazy(
                verifier_view(layout, ev), NS(challenge=ev.challenge, nonce=ev.nonce, words=ev.words), KEYS
            )
            == cheat
        )
        dyn.release()
        # the step completes; A finishes: its block 9 gets its filler back and is read from host memory from the next
        # freeze on
        for layer in range(1, L_):
            for rid in order:
                stt = e.req[rid]
                for p in range(stt.num_computed_tokens, stt.num_computed_tokens + e.so.num_scheduled_tokens[rid]):
                    write(e.kv, stt.block_ids[0], layer, TOKENS[rid][p], p)
        e.finish_step(e.so)
        e.run_step(so(cached=[("B", None)], sched={"B": 1}, finished=["A"]))
        e.pool.clear_act()
        stage["st"] = e.pool.stage(e.so, e.req, [], False, 0, e.step, L_)
        dyn.set_posture("honest")
        honest, _ = round_()
        dyn.release()
        dyn.set_posture("dyn_dram")
        cheat, _ = round_()
        assert _check_dram_table(e.pool, stage["st"], dram) == L_ * 7 * e.pool.page_words and cheat == honest
        ev = e.record(stage["st"])
        assert (
            chain_digest_lazy(
                verifier_view(layout, ev), NS(challenge=ev.challenge, nonce=ev.nonce, words=ev.words), KEYS
            )
            == cheat
        )
        dyn.release()
        dyn.set_posture("honest")
        assert region.own_table and not dyn.dram_on and dyn.dram is dram, (
            "honest: the region's own table; the host copy is kept"
        )
        assert round_()[0] == honest and region.own_table
        dyn.release()
        try:
            dyn.arm(None, 1.0)
            dyn.set_posture("dyn_dram")
            raise AssertionError("posture changed while frozen")
        except RuntimeError:
            pass
        finally:
            dyn.release()


import contextlib


@contextlib.contextmanager
def _pool_dims(**dims):
    """This module's pool dimensions changed for one test (the helpers read them when called)."""
    g = globals()
    names = ("L_", "B_", "H_", "N_", "D_", "C_", "BLOCK", "PW", "ENGINE")
    old = {k: g[k] for k in names}
    g.update(dims)
    g["C_"] = 2 * g["D_"]
    g["PW"] = g["H_"] * g["N_"] * g["C_"] * 2 // 8
    g["ENGINE"] = dict(old["ENGINE"], block_size=g["N_"], n_layers=g["L_"], page_words=g["PW"])
    try:
        yield
    finally:
        g.update(old)


def test_megabyte_blocks_span_pages_and_segments():
    """--dyn-B 1M over a vLLM pool: a region block is many KV pages (here 512 pages of 2 KiB; Qwen2.5-0.5B's are
    8 KiB, 128 to a block) and crosses layer segments (800 KiB a layer). The prover's chain through the segment
    table, the verifier's lazy view of the pool and of the live region, and the dyn_dram table all agree, at the
    demo's chain shape (8 links, r > 0) as well as at r = 0."""
    from memory_accounting.prover.dynamic import DynamicResponder, PauseController

    TOKENS.update({"M": list(range(500, 508)), "N": list(range(520, 531))})
    with _pool_dims(L_=3, B_=400, D_=64, BLOCK=1 << 20):
        e = Engine("LBHNC")
        region, bw = e.pool.region, BLOCK // 8
        assert e.pool.page_bytes == 2048 and region.words == 3 * bw, "three 1 MiB blocks"
        assert region.find("kv.1").offset % bw and region.find("kv.2").offset % bw, "blocks cross layer segments"
        e.run_step(so(new=[("M", [3, 390]), ("N", [0, 214, 399])], sched={"M": 6, "N": 9}))
        order = e.run_step(so(cached=[("M", None), ("N", None)], sched={"M": 1, "N": 1}), stop_at=1)
        st = e.pool.stage(e.so, e.req, order, True, 1, e.step, L_)
        ev = e.record(st)
        view = verifier_view("LBHNC", ev)
        assert_same(e, ev, view)  # every block, and a 32-link r = 0 chain
        assert assert_live_same(e, ev, "LBHNC") == bw, "the live region pads to one 1 MiB block"
        dyn = DynamicResponder(region, lambda: st, PauseController(), KEYS, dram_fn=e.pool.dram_plan)
        for j0 in range(3):
            ch = Challenge(BLOCK, 8, 16, 0x5EC0 + j0, j0, 1024)  # the dynamic chain's shape, a small r
            cev = NS(
                challenge={"block_bytes": BLOCK, "n": 8, "r": 16, "k": 1024, "g": 1, "j0": j0},
                nonce=ch.nonce,
                words=ev.words,
            )
            want = chain_digest(region.snapshot(), cev, KEYS)
            assert chain_digest_lazy(view, cev, KEYS) == want
            dyn.set_posture("honest")
            dyn.arm(None, 1.0)
            assert dyn.answer(ch) == want
            dyn.release()
            dyn.set_posture("dyn_dram")
            dyn.arm(None, 1.0)
            assert dyn.answer(ch) == want, "dyn_dram with blocks spanning live pages and host pages"
            _check_dram_table(e.pool, st, dyn.dram)
            dyn.release()


if __name__ == "__main__":
    from _runner import run

    run(globals())
