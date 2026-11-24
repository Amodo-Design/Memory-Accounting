"""A toy causal language model with a static KV cache, for tests and the CPU demo prover.

Each layer keeps its K and V in fixed buffers (one slot per position up to `max_len`) and writes the
new positions in place, like a transformers StaticCache. Layers are called with the hidden state as
the first positional argument, so `memory_accounting.prover.dynamic.Hooks` sees the same call shape as a HF
decoder layer. The model tracks its own position: `reset()` then feed token chunks in order.
"""

from __future__ import annotations

import threading
import time

import torch
from torch import nn


class ToyLayer(nn.Module):
    def __init__(self, d: int, max_len: int):
        super().__init__()
        self.d, self.max_len = d, max_len
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("k_cache", torch.zeros(1, max_len, d))
        self.register_buffer("v_cache", torch.zeros(1, max_len, d))

    def forward(self, h: torch.Tensor, pos: int) -> torch.Tensor:
        _b, s, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).split(d, dim=-1)
        self.k_cache[:, pos : pos + s] = k
        self.v_cache[:, pos : pos + s] = v
        K, V = self.k_cache[:, : pos + s], self.v_cache[:, : pos + s]
        att = (q @ K.transpose(1, 2)) / d**0.5  # (b, s, pos+s)
        qi = torch.arange(pos, pos + s, device=h.device)[:, None]
        kj = torch.arange(pos + s, device=h.device)[None, :]
        att = att.masked_fill(kj > qi, float("-inf")).softmax(-1)
        h = h + self.out(att @ V)
        return h + self.mlp(self.norm2(h))


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 97, d: int = 64, n_layers: int = 4, max_len: int = 64):
        super().__init__()
        self.vocab, self.d, self.max_len = vocab, d, max_len
        self.embed = nn.Embedding(vocab, d)
        self.pos_embed = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([ToyLayer(d, max_len) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.pos = 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.v_cache.zero_()
        self.pos = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _b, s = ids.shape
        if self.pos + s > self.max_len:
            raise ValueError("toy cache full")
        h = self.embed(ids) + self.pos_embed(torch.arange(self.pos, self.pos + s, device=ids.device))[None]
        for layer in self.layers:
            h = layer(h, self.pos)
        self.pos += s
        return self.head(self.norm(h))

    def kv_buffers(self) -> list[tuple[str, torch.Tensor]]:
        out = []
        for i, layer in enumerate(self.layers):
            out += [(f"kv.{i}.k", layer.k_cache), (f"kv.{i}.v", layer.v_cache)]
        return out

    def live_kv_buffers(self, n: int) -> list[tuple[str, torch.Tensor]]:
        """The first n slots of every K and V buffer (contiguous views: the live part of the cache)."""
        out = []
        for i, layer in enumerate(self.layers):
            out += [(f"kv.{i}.k", layer.k_cache[:, :n]), (f"kv.{i}.v", layer.v_cache[:, :n])]
        return out

    @torch.no_grad()
    def generate(self, prompt: list[int], new_tokens: int, on_step=None, reset: bool = True) -> list[int]:
        """Greedy generation: prefill the prompt, then one token per step. `on_step(step)` is called
        between forwards. reset=False when the caller has reset the cache under the pause lock."""
        if reset:
            self.reset()
        dev = self.embed.weight.device
        ids = torch.tensor([prompt], device=dev)
        out = list(prompt)
        logits = self(ids)
        for step in range(new_tokens):
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            if on_step is not None:
                on_step(step)
            if self.pos >= self.max_len:
                break
            logits = self(torch.tensor([[nxt]], device=dev))
        return out


def toy_feed(model: ToyLM):
    """The verifier's replay driver for the toy: feed chunk `s` (a 2-D token list) in order, after the
    replay has reset the cache and put the filler back."""

    def feed(step: int, chunk: list) -> None:
        with torch.no_grad():
            model(torch.tensor(chunk, device=model.embed.weight.device))

    return feed


class InferenceLoop(threading.Thread):
    """The demo's inference load: a model generating from a cycle of prompts on a background thread,
    with the dynamic-memory hooks installed. When a request ends its memory is freed at once: the cache
    is emptied and every free word of the region gets its filler back, under the pause lock, so a
    challenge never sees it half done and an idle engine's region is all filler. `step_sleep` paces
    the demo, spread over the layers so that pauses land at every layer index over time."""

    def __init__(
        self,
        model,
        tracker,
        pause,
        step_sleep: float = 0.002,
        prompts=None,
        new_tokens: int | None = None,
        stop_ids=(),
        request_gap: float = 0.0,
        region=None,
    ):
        super().__init__(name="posm-inference", daemon=True)
        self.model, self.tracker, self.pause, self.step_sleep, self.region = model, tracker, pause, step_sleep, region
        self.request_gap = request_gap  # seconds idle between requests (the engine has no work)
        self.prompts = prompts or [[1, 2, 3, 4, 5], [7, 11, 13], [2, 4, 6, 8, 10, 12]]
        self.new_tokens, self.stop_ids = new_tokens, stop_ids
        self._halt = threading.Event()
        self.requests = 0
        self.tokens = 0
        self.last_output: list[int] = []
        if step_sleep:
            per_layer = step_sleep / max(1, len(model.layers))
            for layer in model.layers:
                layer.register_forward_pre_hook(lambda m, a: time.sleep(per_layer))

    def _free(self, next_id: int) -> None:
        if self.region is not None:
            from ..prover.dynamic import free_all

            free_all(self.model, self.region, self.tracker, self.pause, next_id)
            return
        self.pause.acquire_inference()
        try:
            self.model.reset()
            self.tracker.new_request(next_id)
        finally:
            self.pause.lock.release()

    def run(self):
        self._free(1)
        while not self._halt.is_set():
            prompt = self.prompts[self.requests % len(self.prompts)]
            self.requests += 1
            n = self.new_tokens or (self.model.max_len - len(prompt) - 1)
            kw = {"stop_ids": self.stop_ids} if self.stop_ids else {}
            out = self.model.generate(prompt, n, reset=False, **kw)
            self._free(self.requests + 1)  # the request ended: its memory is free again
            self.last_output = out
            self.tokens += len(out) - len(prompt)
            if self.request_gap:
                self._halt.wait(self.request_gap)

    def stop(self):
        self._halt.set()


ToyInference = InferenceLoop


def static_live_fn(model, region):
    """live_fn for a static-cache model: the cache slots the stage has fed (this step's included: the layers
    after the paused one still hold filler there, on both sides) and the activation buffer. Encoded: the groups of
    each store holding those slots, and the activation store (`EncodedStatic.live`)."""
    from ..prover.dynamic import ACT_NAME, live_tokens

    enc = getattr(region, "encoded_cache", None)
    if enc is not None:
        return lambda stage: enc.live(live_tokens(stage))
    act = region.find(ACT_NAME).tensor
    return lambda stage: model.live_kv_buffers(live_tokens(stage)) + [(ACT_NAME, act)]


def static_dram_fn(model, region):
    """dram_fn for a static-cache model (the dyn_dram cheat, `memory_accounting.prover.dynamic.DramEvict`): the top `fraction`
    of every KV buffer's slots (the cache's free end); a slot of it is read from host memory when the stage has
    not fed it (`live_tokens`), which is when it holds filler on both sides. The buffers must keep a slot's
    words together (slots outermost: the toy, the integer model; not a HF StaticCache)."""
    import numpy as np

    from ..prover.dynamic import live_tokens

    enc = getattr(region, "encoded_cache", None)
    if enc is not None:
        return _encoded_dram_fn(enc)
    names, S = [n for n, _ in model.kv_buffers()], int(model.max_len)

    def plan(fraction: float):
        k = min(S, max(1, int(np.ceil(S * fraction))))
        s0, ranges = S - k, []
        for n in names:
            w = region.find(n).words
            if w % S:
                raise ValueError(f"dyn_dram: {n} is {w} words, not {S} slots of whole words")
            ranges.append((n, s0, k, w // S))

        def free_fn(stage):
            f = np.arange(s0, S) >= live_tokens(stage)
            return [f] * len(ranges)

        return ranges, free_fn

    return plan


def _encoded_dram_fn(enc):
    """dram_fn of an encoded static cache: the top `fraction` of every store's groups; a group is read from host
    memory when it holds no slot the stage has fed (all its plaintext words past the fed slots: encoded zeros on both
    sides)."""
    import numpy as np

    from ..prover.dynamic import live_tokens
    from ..prover.kvenc import GROUP_ENC, groups_of

    def plan(fraction: float):
        ranges, firsts = [], []
        for k, (name, st) in enumerate(enc.stores):
            G = groups_of(st.plain)
            n = min(G, max(1, int(np.ceil(G * fraction))))
            ranges.append((name, G - n, n, GROUP_ENC))
            firsts.append((k % enc.per, G - n, n))

        def free_fn(stage):
            t = live_tokens(stage)
            return [np.arange(g0, g0 + n) >= groups_of(enc.words_of(p, t)) for p, g0, n in firsts]

        return ranges, free_fn

    return plan


def attach_dynamic(
    model,
    device,
    block_bytes: int,
    pad_seed: int,
    act_tokens: int,
    act_dtype=torch.float32,
    gpu: int = 0,
    encoded: bool = False,
):
    """Dynamic-memory plumbing for a model with fixed KV buffers (`kv_buffers()`, `layers`, `reset()`):
    activation buffer, region, stage tracker, pause controller, hooks. gpu: this GPU's index as the
    verifier knows it (the manifest's), which keys the dynamic filler; the model's `info` records it.
    encoded (the integer model, `share_cache()`): the region holds every layer's cache and the activation buffer
    encoded (`memory_accounting.prover.kvenc.EncodedStatic`, `region.encoded_cache`), the plaintext cache is shared scratch, and
    the stage says so ("encoded": the verifier encodes its replay). Returns (region, tracker, pause, hooks)."""
    from ..prover.dynamic import (
        DeviceSync,
        DynamicRegion,
        Hooks,
        PauseController,
        StageTracker,
        act_buffer,
        make_region,
    )

    act = act_buffer(act_tokens, model.d, act_dtype, device)
    if encoded:
        from ..prover.dynfill import dyn_key
        from ..prover.kvenc import EncodedStatic

        enc = EncodedStatic(model, act, dyn_key(pad_seed, gpu), model.share_cache)
        region = DynamicRegion(enc.buffers(), block_bytes, pad_seed, device=act.device, gpu=gpu)
        region.encoded_cache = enc
    else:
        region = make_region(model.kv_buffers(), act, block_bytes, pad_seed, gpu=gpu)
    if isinstance(getattr(model, "info", None), dict):
        model.info["gpu"] = int(gpu)
    tracker = StageTracker(len(model.layers), act, encoded=encoded)
    pause = PauseController()
    hooks = Hooks(
        model,
        list(model.layers),
        act,
        tracker,
        pause,
        on_pause=DeviceSync(device),
        region=None if encoded else region,
        after_act=enc.act_written if encoded else None,
    )
    if encoded:
        enc.attach(region, list(model.layers))  # after the pause hooks: a pause comes before a layer's decode
    model.reset()
    region.reset_to_filler()  # nothing is live yet: all free memory holds filler (encoded zeros)
    return region, tracker, pause, hooks


def toy_dynamic(
    device="cpu",
    block_bytes: int = 65536,
    pad_seed: int = 0,
    d: int = 64,
    n_layers: int = 4,
    max_len: int = 64,
    seed: int = 0,
    gpu: int = 0,
):
    """A toy model with its dynamic region, tracker, pause point and hooks on `device` (the demo prover
    and the laptop loop). Returns (model, region, tracker, pause, hooks)."""
    torch.manual_seed(seed)
    model = ToyLM(d=d, n_layers=n_layers, max_len=max_len).to(device).eval()
    model.info = {"kind": "toy", "seed": seed, "d": d, "n_layers": n_layers, "max_len": max_len, "vocab": model.vocab}
    region, tracker, pause, hooks = attach_dynamic(
        model, device, block_bytes, pad_seed, max_len, torch.float32, gpu=gpu
    )
    return model, region, tracker, pause, hooks


def int_dynamic(
    path,
    device="cpu",
    block_bytes: int = 65536,
    pad_seed: int = 0,
    max_len: int = 256,
    gpu: int = 0,
    encoded: bool = True,
):
    """The integer-exact model with its dynamic region (activation buffer float64, max_len rows), held encoded
    (v0.17; encoded=False: plain, as before). Returns (model, region, tracker, pause, hooks)."""
    from .intllm import IntLLM

    model = IntLLM.from_file(path, max_len, device)
    region, tracker, pause, hooks = attach_dynamic(
        model, device, block_bytes, pad_seed, max_len, torch.float64, gpu=gpu, encoded=encoded
    )
    return model, region, tracker, pause, hooks
