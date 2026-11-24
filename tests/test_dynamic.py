"""Dynamic memory on the CPU: freeze a model at a layer or step boundary, chain over the frozen KV cache +
activation region through the segment table, keep the stage record, and have the verifier replay
inference to the stage, rebuild the region and reproduce the digest exactly. Negative cases: a
tampered region, a false stage, other tokens, a wrong model."""

import copy
import pathlib
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from memory_accounting.core.evidence import Evidence, EvidenceServer, fetch_evidence
from memory_accounting.core.params import Challenge, Keys
from memory_accounting.models.toy import InferenceLoop, static_live_fn, toy_dynamic
from memory_accounting.prover.dynamic import ACT_NAME, PAD_NAME, DynamicResponder, free_all, pad_words
from memory_accounting.verifier.consistency import chain_digest, hf_replayer, int_replayer, toy_replayer, verify

BLOCK = 16384
KEYS = Keys.deterministic(3)


def _toy(seed=0, include_bytes=False):
    model, region, tracker, pause, hooks = toy_dynamic(
        "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=seed
    )
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info, include_bytes=include_bytes)
    return model, region, tracker, pause, hooks, dyn


def _challenge(region, nonce, n=48, r=16):
    return Challenge(BLOCK, n, r, nonce, nonce % region.num_blocks(BLOCK), 1024)


def _run_until_paused(model, region, tracker, pause, trigger_step, trigger_layer=None, prompt=(1, 2, 3, 4, 5), new=12):
    """Start a request on a thread and request a pause when it reaches (step, layer): deterministic.
    Without a layer, the pause is requested after step trigger_step - 1 and taken at the next step boundary."""
    fired = []

    def at_layer(m, a):
        if tracker.step == trigger_step and not fired:
            fired.append(1)
            pause.request()

    def after_step(m, a, o):
        if tracker.step == trigger_step - 1 and not fired:
            fired.append(1)
            pause.request()

    if trigger_layer is None:
        h = model.register_forward_hook(after_step)
    else:  # registered after Hooks: runs after layer L-1's boundary hook
        h = model.layers[trigger_layer - 1].register_forward_pre_hook(at_layer)
    free_all(model, region, tracker, pause, 7)
    th = threading.Thread(target=lambda: model.generate(list(prompt), new, reset=False), daemon=True)
    th.start()
    assert pause.wait_paused(5.0), "the inference thread never reached the pause"
    h.remove()
    return th


def test_region_layout_and_pad():
    model, region, *_ = _toy()
    names = [s.name for s in region.segments]
    assert names[:2] == ["kv.0.k", "kv.0.v"] and names[-2:] == [ACT_NAME, PAD_NAME]
    assert region.words * 8 % BLOCK == 0
    live = sum(s.words for s in region.segments if s.name != PAD_NAME)
    assert 0 < region.words - live < BLOCK // 8
    # nothing live yet: the whole region is dynamic filler, finite where it is float
    assert np.array_equal(region.snapshot(), region.filler_view())
    assert torch.isfinite(model.layers[0].k_cache).all() and torch.isfinite(region.find(ACT_NAME).tensor).all()
    assert np.array_equal(
        region.find(PAD_NAME).tensor.numpy().view(np.uint64), pad_words(0x5EED, region.words - live, pos0=live)
    )
    model.layers[0].k_cache.view(-1)[0] = 1.5
    snap = region.snapshot()
    assert snap[:1].view(np.float32)[0] == 1.5  # word 0 is k_cache[0] of layer 0


def test_pause_mid_layer_replay_reproduces_digest():
    model, region, tracker, pause, hooks, dyn = _toy(include_bytes=True)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=2)
    words, _bb, flags, _ = dyn.arm(None, 1.0)
    assert flags == 1 and words == region.words
    assert (
        dyn.stage["step"] == 3
        and dyn.stage["act_layer"] == 2
        and dyn.stage["kv_layers_done"] == 2
        and dyn.stage["in_flight"]
    )
    ch = _challenge(region, 0xABCDEF)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    assert not th.is_alive()
    ev = dyn.store.get(ch.nonce)
    pos = 5 + 3 - 1  # step 3 writes position 7
    fill = region.filler_view()
    kv1 = ev.segment("kv.1.k").view(np.float32).reshape(1, model.max_len, model.d)
    kv2 = ev.segment("kv.2.k").view(np.float32).reshape(1, model.max_len, model.d)
    fill2 = ev.segment("kv.2.k", fill).view(np.float32).reshape(1, model.max_len, model.d)
    assert not np.array_equal(
        kv1[0, pos], ev.segment("kv.1.k", fill).view(np.float32).reshape(1, model.max_len, model.d)[0, pos]
    )
    assert np.array_equal(kv2[0, pos], fill2[0, pos]), (
        "frozen mid-step: layers < 2 hold step 3, layers >= 2 still hold filler there"
    )
    assert np.array_equal(kv2[0, pos + 1 :], fill2[0, pos + 1 :]), "slots past the request hold filler"
    rep = verify(ev, toy_replayer(), KEYS)
    assert rep["ok"] and not rep["differing_segments"], rep
    # ---- a false stage, other tokens, a different digest, another model: all fail, with no tolerance to hide in
    lie = copy.deepcopy(ev)
    lie.stage["act_layer"] = 1
    lie.stage["kv_layers_done"] = 1
    assert not verify(lie, toy_replayer(), KEYS)["ok"]
    other = copy.deepcopy(ev)
    other.stage["chunks"][2] = [[9]]
    assert not verify(other, toy_replayer(), KEYS)["ok"]
    flipped = copy.deepcopy(ev)
    flipped.digest = [ev.digest[0] ^ 1, ev.digest[1]]
    assert not verify(flipped, toy_replayer(), KEYS)["ok"]
    wrong_model = copy.deepcopy(ev)
    wrong_model.model["seed"] = 99
    assert not verify(wrong_model, toy_replayer(), KEYS)["ok"]
    hooks.remove()


def test_tampered_region_is_caught_and_located():
    """A prover whose HBM holds something other than honest KV: the digest it returns is over the
    tampered bytes, the verifier's replay is honest, so the digests differ; debug bytes name the segment."""
    model, region, tracker, pause, hooks, dyn = _toy(include_bytes=True)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=4, trigger_layer=3)
    dyn.arm(None, 1.0)
    model.layers[1].v_cache[0, 2, 5] += 1e-6  # one value, a tiny change
    ch = _challenge(region, 0x1234, n=96)  # long chain: every block visited w.h.p.
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    rep = verify(dyn.store.get(ch.nonce), toy_replayer(), KEYS)
    assert not rep["ok"]
    assert [d["name"] for d in rep["differing_segments"]] == ["kv.1.v"] and rep["differing_segments"][0][
        "words_differing"
    ] == 1
    hooks.remove()


def test_step_boundary_and_idle_freeze():
    model, region, tracker, pause, hooks, dyn = _toy(seed=1)
    # a pause taken at the model boundary: the previous step is complete
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3)
    _, _, flags, _ = dyn.arm(None, 1.0)
    assert flags == 2 and not dyn.stage["in_flight"] and dyn.stage["step"] == 2 and dyn.stage["kv_layers_done"] == 4
    ch = _challenge(region, 77)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    assert verify(dyn.store.get(77), toy_replayer(), KEYS)["ok"]
    # idle: no boundary reached within the wait, so the responder takes the lock; a forward cannot start
    _words, _bb, flags, _ = dyn.arm(None, 0.05)
    assert flags == 0 and dyn.locked
    started = threading.Event()
    done = threading.Event()
    th = threading.Thread(
        target=lambda: (started.set(), model.generate([3, 1], 2, reset=False), done.set()), daemon=True
    )
    th.start()
    started.wait(1.0)
    assert not done.wait(0.2), "a forward ran while the region was frozen"
    ch = _challenge(region, 78)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    assert done.wait(5.0)
    assert verify(dyn.store.get(78), toy_replayer(), KEYS)["ok"]
    # a pause request that is cancelled does not fire later
    pause.request()
    assert pause.cancel()
    free_all(model, region, tracker, pause, 9)
    model.generate([1], 3, reset=False)
    assert not pause.paused
    hooks.remove()


def test_records_files_and_wire():
    model, region, tracker, pause, hooks, dyn = _toy(seed=2)
    loop = InferenceLoop(model, tracker, pause, step_sleep=0.001, region=region)
    loop.start()
    nonces = []
    for i in range(6):
        dyn.arm(None, 0.5)
        ch = _challenge(region, 1000 + i)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        nonces.append(ch.nonce)
    loop.stop()
    loop.join(5.0)
    assert dyn.paused_in_flight + dyn.paused_between + dyn.idle == 6
    d = pathlib.Path(tempfile.mkdtemp())
    srv = EvidenceServer(dyn.store, "127.0.0.1", 0)
    srv.start()
    try:
        for nonce in nonces:
            ev = fetch_evidence("127.0.0.1", srv.port, nonce)
            assert ev.data is None and ev.header() == dyn.store.get(nonce).header()
            back = Evidence.load(ev.save(d))
            assert back.header() == ev.header()
            rep = verify(back, toy_replayer(), KEYS)
            assert rep["ok"], rep
        try:
            fetch_evidence("127.0.0.1", srv.port, 42)
            raise AssertionError("unknown nonce accepted")
        except LookupError:
            pass
    finally:
        srv.stop()
    hooks.remove()


def test_hf_static_cache_path():
    """A tiny random HF Llama on a StaticCache (CPU): the float model replays bit-exactly on the same
    device type, so the digest reproduces."""
    from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

    from memory_accounting.prover.dynamic import (
        Hooks,
        PauseController,
        StageTracker,
        act_buffer,
        hf_cache_buffers,
        layer_modules,
        make_region,
    )

    cfg = LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(5)
    model = LlamaForCausalLM(cfg).eval()
    cache = StaticCache(cfg, max_cache_len=32)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        # initialises the lazy cache buffers
        model.generate(ids, max_new_tokens=2, do_sample=False, past_key_values=cache)
    layers = layer_modules(model)
    act = act_buffer(32, cfg.hidden_size, torch.float32, "cpu")
    region = make_region(hf_cache_buffers(cache), act, BLOCK, pad_seed=1)
    tracker = StageTracker(len(layers), act)
    pause = PauseController()
    Hooks(model, layers, act, tracker, pause, region=region)
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info={"kind": "hf", "name": "tiny-llama-test"})
    fired = []

    def trigger(m, a):
        if tracker.step == 2 and not fired:
            fired.append(1)
            pause.request()

    layers[1].register_forward_pre_hook(trigger)
    free_all(model, region, tracker, pause, 3, reset=cache.reset)
    th = threading.Thread(
        target=lambda: model.generate(ids, max_new_tokens=6, do_sample=False, past_key_values=cache), daemon=True
    )
    with torch.no_grad():
        th.start()
        assert pause.wait_paused(5.0)
        _words, _bb, flags, _ = dyn.arm(None, 0.5)
        assert flags == 1
        ch = Challenge(BLOCK, 16, 16, 0x5151, 1, 1024)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        th.join(5.0)
    ev = dyn.store.get(0x5151)
    assert ev.stage["step"] == 2 and ev.stage["act_layer"] == 2 and ev.stage["chunks"][0] == [[1, 2, 3, 4, 5]]
    rep = verify(ev, hf_replayer(model), KEYS)
    assert rep["ok"], rep
    lie = copy.deepcopy(ev)
    lie.stage["chunks"][1] = [[42]]
    assert not verify(lie, hf_replayer(model), KEYS)["ok"]
    dyn.close()


def test_integer_model_path():
    """The integer-exact model: generation on a thread, challenges at whatever boundary it is at, every
    record reproduced by an independent replay (a second model instance loaded from the same file)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from memory_accounting.models.intllm import convert_hf
    from memory_accounting.models.toy import int_dynamic

    tmp = pathlib.Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    LlamaForCausalLM(cfg).eval().save_pretrained(tmp / "tiny")
    path = tmp / "tiny.int.safetensors"
    convert_hf(str(tmp / "tiny"), str(path), max_pos=256)
    model, region, tracker, pause, hooks = int_dynamic(path, "cpu", BLOCK, pad_seed=0xF00D, max_len=48)
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info)
    loop = InferenceLoop(
        model, tracker, pause, step_sleep=0.003, prompts=[[5, 17, 3, 99], [42, 7, 7, 1, 2, 3]], region=region
    )
    loop.start()
    for i in range(8):
        dyn.arm(None, 0.5)
        ch = _challenge(region, 500 + i)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
    loop.stop()
    loop.join(10.0)
    rp = int_replayer(path)
    stages = []
    for i in range(8):
        ev = dyn.store.get(500 + i)
        rep = verify(ev, rp, KEYS)
        assert rep["ok"], rep
        stages.append((ev.stage["step"], ev.stage["act_layer"], ev.stage["in_flight"]))
    assert dyn.paused_in_flight >= 1, stages
    # another model file (same architecture, other weights) is refused, and with the hash forged its replay does not
    # match
    torch.manual_seed(1)
    LlamaForCausalLM(cfg).eval().save_pretrained(tmp / "other")
    other = tmp / "other.int.safetensors"
    convert_hf(str(tmp / "other"), str(other), max_pos=256)
    rep = verify(dyn.store.get(500), int_replayer(other), KEYS)
    assert not rep["ok"] and "verifier holds" in rep["error"]
    from memory_accounting.models.intllm import file_sha256

    forged = copy.deepcopy(dyn.store.get(501))
    forged.model["sha256"] = file_sha256(other)
    assert not verify(forged, int_replayer(other), KEYS)["ok"]
    hooks.remove()


def test_freed_memory_is_filler_and_hidden_data_is_caught():
    """When a request ends its memory holds filler again, so an idle engine's region is exactly the
    filler; data a cheater keeps in a free slot changes the digest, and debug bytes locate it."""
    model, region, tracker, pause, hooks, dyn = _toy(seed=4, include_bytes=True)
    loop = InferenceLoop(
        model, tracker, pause, step_sleep=0.0, prompts=[[1, 2, 3]], new_tokens=6, region=region, request_gap=5.0
    )
    loop.start()
    t0 = time.time()
    while loop.requests < 1 or loop.tokens == 0:  # first request done, now idle
        assert time.time() - t0 < 10
        time.sleep(0.01)
    _words, _bb, flags, _ = dyn.arm(None, 0.05)
    assert flags == 0  # idle: frozen by the lock
    assert np.array_equal(region.snapshot(), region.filler_view()), "the freed cache and activations must hold filler"
    ch = _challenge(region, 4242)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    ev = dyn.store.get(4242)
    assert ev.stage["step"] == -1 and verify(ev, toy_replayer(), KEYS)["ok"]
    # a cheater hides one value in a free KV slot during an idle period
    dyn.arm(None, 0.05)
    model.layers[2].v_cache[0, 30, 7] = 0.25
    ch = _challenge(region, 4243, n=96)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    rep = verify(dyn.store.get(4243), toy_replayer(), KEYS)
    assert not rep["ok"] and [d["name"] for d in rep["differing_segments"]] == ["kv.2.v"]
    loop.stop()
    loop.join(10.0)
    hooks.remove()


def test_second_chain_over_the_live_region():
    """Two chains per freeze: one over all of the dynamic memory, one over the live region only (the cache
    slots fed so far and the activation buffer). Both records replay exactly; data hidden among the live
    slots is caught by the live chain."""
    for kind in ("toy", "int"):
        if kind == "toy":
            model, region, tracker, pause, hooks = toy_dynamic(
                "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=6
            )
            rp = toy_replayer()
        else:
            from transformers import LlamaConfig, LlamaForCausalLM

            from memory_accounting.models.intllm import convert_hf
            from memory_accounting.models.toy import int_dynamic

            tmp = pathlib.Path(tempfile.mkdtemp())
            torch.manual_seed(0)
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=101,
                    hidden_size=64,
                    intermediate_size=160,
                    num_hidden_layers=3,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=256,
                )
            ).eval().save_pretrained(tmp / "tiny")
            convert_hf(str(tmp / "tiny"), str(tmp / "tiny.int.safetensors"), max_pos=256)
            model, region, tracker, pause, hooks = int_dynamic(
                tmp / "tiny.int.safetensors", "cpu", BLOCK, pad_seed=0xF00D, max_len=48
            )
            rp = int_replayer(tmp / "tiny.int.safetensors")
        dyn = DynamicResponder(
            region, tracker, pause, KEYS, model_info=model.info, live_fn=static_live_fn(model, region)
        )
        th = _run_until_paused(model, region, tracker, pause, trigger_step=4, trigger_layer=2)
        words, _, flags, live_words = dyn.arm(None, 1.0, want_live=True)
        assert flags == 1 and 0 < live_words < words, (live_words, words)
        ch = _challenge(region, 0x7000)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        assert dyn.frozen and dyn.live_pending, "frozen until the live chain is answered"
        nbl = live_words * 8 // BLOCK
        chl = Challenge(BLOCK, 16, 16, 0x7001, 0x7001 % nbl, 1024)
        lo, hi = dyn.answer(chl, live=True)
        dyn.finish(chl, lo, hi, live=True)
        assert not dyn.frozen
        th.join(5.0)
        main, live = dyn.store.get(0x7000), dyn.store.get(0x7001)
        assert main.region == "main" and live.region == "live" and live.stage == main.stage
        assert verify(main, rp, KEYS)["ok"] and verify(live, rp, KEYS)["ok"], kind
        # data hidden in a live slot: the live chain over its few blocks catches it
        th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=1)
        words, _, flags, live_words = dyn.arm(None, 1.0, want_live=True)
        if kind == "toy":
            model.layers[0].v_cache.view(-1)[1] += 1
        else:  # encoded (v0.17): the store, not the plaintext scratch
            region.find("kv.0.vs").tensor.view(-1)[1] ^= 1
        ch = _challenge(region, 0x7100)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        nbl = live_words * 8 // BLOCK
        chl = Challenge(BLOCK, 64, 16, 0x7101, 0, 1024)
        lo, hi = dyn.answer(chl, live=True)
        dyn.finish(chl, lo, hi, live=True)
        th.join(5.0)
        assert not verify(dyn.store.get(0x7101), rp, KEYS)["ok"], kind
        # a verifier that asks for one chain gets one
        words, _bb, flags, live_words = dyn.arm(None, 0.05)
        assert live_words == 0 and not dyn.live_pending
        ch = _challenge(region, 0x7200)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        assert not dyn.frozen
        hooks.remove()


def test_dynamic_filler_is_not_the_static_filler():
    """Free dynamic memory must not be computable from the static filler. Both use the verifier's payload
    stream; a prover that takes the dynamic key as its static filler key (`--enc-key dyn_key(seed, 0)`, the
    filler key of a stream-decode bundle) used to make every free KV word equal a static filler word with
    the bf16 mask applied, so the pool could be dropped and its reads answered from the static filler."""
    from memory_accounting.core.backend import CpuLib
    from memory_accounting.prover.dynfill import dyn_key, filler_np, mask_for

    seed, lib = 0x5EED5EED5EED5EED, CpuLib()
    key = dyn_key(seed, 0)
    pos0, n = 1 << 20, 4096  # a static filler range past Θ', and the same positions of the pool
    pay = np.empty(n, np.uint64)
    lib.filler(pay, seed, pos0)
    static = np.empty(n, np.uint64)
    lib.setup_at(pay, static, key, pos0)  # Enc(filler key, filler(seed)), as allocate_filler writes it
    for mask in (mask_for("bfloat16"), mask_for("int8")):
        dyn = filler_np(n, pos0, seed, key, mask)
        assert not np.any((static & np.uint64(mask)) == dyn), "free dynamic words are static filler words"
    # the live region's pad is not the main region's filler at the same positions either
    assert not np.any(filler_np(n, pos0, seed, key, live=True) == filler_np(n, pos0, seed, key))


def test_replay_takes_the_pad_seed_and_gpu_from_the_verifier():
    """The pad seed and the GPU index choose the filler the replay rebuilds; they are the verifier's, not the
    record's. A prover on GPU 1 keyed with GPU 0's filler (one copy answering for both GPUs), or with a seed
    of its own choosing, fails whatever its record claims."""
    from memory_accounting.verifier.consistency import Expected

    for gpu in (0, 1):
        model, region, tracker, pause, hooks = toy_dynamic(
            "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=7, gpu=gpu
        )
        assert region.gpu == gpu and model.info["gpu"] == gpu
        dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info)
        th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=2)
        dyn.arm(None, 1.0)
        ch = _challenge(region, 0x9000 + gpu)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        th.join(5.0)
        ev = dyn.store.get(ch.nonce)
        assert ev.model["gpu"] == gpu
        assert verify(ev, toy_replayer(), KEYS, expected=Expected(0x5EED, gpu))["ok"]
        rep = verify(ev, toy_replayer(), KEYS, expected=Expected(0x5EED, 1 - gpu))
        assert not rep["ok"] and "GPU" in rep["error"], rep
        rep = verify(ev, toy_replayer(), KEYS, expected=Expected(0x5EEE, gpu))
        assert not rep["ok"] and "pad seed" in rep["error"], rep
        forged = copy.deepcopy(ev)
        forged.model["gpu"] = 1 - gpu  # claims the GPU the verifier expects: the replay's filler differs
        assert not verify(forged, toy_replayer(), KEYS, expected=Expected(0x5EED, 1 - gpu))["ok"]
        forged = copy.deepcopy(ev)
        forged.pad_seed = 0x5EEE
        assert not verify(forged, toy_replayer(), KEYS, expected=Expected(0x5EEE, gpu))["ok"]
        hooks.remove()


def test_offline_check_against_the_verifier_log():
    """`python -m memory_accounting.verifier.consistency --log` checks each record against the verifier's log of the chain it sent
    and the reply it received (the live verifier's rule), not only against itself: a record of another chain
    (here another start block, with a digest consistent with it), a reply that was not the one received, a
    size READY did not report, or a challenge with no record, all fail."""
    import json

    from memory_accounting.verifier.consistency import main as consistency_main

    model, region, tracker, pause, hooks = toy_dynamic(
        "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=8
    )
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info, live_fn=static_live_fn(model, region))
    tmp = pathlib.Path(tempfile.mkdtemp())
    ev_dir = tmp / "evidence"
    lines = []
    for i, (step, layer) in enumerate(((3, 2), (4, 1))):
        th = _run_until_paused(model, region, tracker, pause, trigger_step=step, trigger_layer=layer)
        words, _bb, _flags, live_words = dyn.arm(None, 1.0, want_live=True)
        ch = Challenge(BLOCK, 8, 16, 0x8000 + 2 * i, (3 * i + 1) % (words * 8 // BLOCK), 1024)
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        chl = Challenge(BLOCK, 8, 16, 0x8001 + 2 * i, 0, 1024)
        llo, lhi = dyn.answer(chl, live=True)
        dyn.finish(chl, llo, lhi, live=True)
        th.join(5.0)
        for c in (ch, chl):
            dyn.store.get(c.nonce).save(ev_dir)
        lines.append(
            {
                "type": "verification",
                "id": i,
                "rtt_ns": 1,
                "dyn": {
                    "nonce": f"{ch.nonce:016x}",
                    "j0": ch.j0,
                    "rtt_ns": 1,
                    "received": f"{hi:016x}{lo:016x}",
                    "words": words,
                    "blocks": words * 8 // BLOCK,
                    "live": {
                        "nonce": f"{chl.nonce:016x}",
                        "j0": chl.j0,
                        "rtt_ns": 1,
                        "received": f"{lhi:016x}{llo:016x}",
                        "words": live_words,
                    },
                },
            }
        )
    hooks.remove()

    def run(log_lines, *extra) -> int:
        log = tmp / "live.jsonl"
        log.write_text("".join(json.dumps(e) + "\n" for e in log_lines))
        try:
            consistency_main(
                [
                    "--evidence",
                    str(ev_dir),
                    "--log",
                    str(log),
                    "--pad-seed",
                    "5eed",
                    "--hash-seed",
                    "3",
                    "--dyn-B",
                    "16K",
                    "--dyn-n",
                    "8",
                    "--r",
                    "16",
                    "--k",
                    "1024",
                    *extra,
                ]
            )
        except SystemExit as e:
            return e.code

    assert run(lines) == 0
    other = json.loads(json.dumps(lines))
    other[0]["dyn"]["j0"] += 1  # the verifier sent another start block
    assert run(other) == 1
    other = json.loads(json.dumps(lines))
    other[1]["dyn"]["live"]["received"] = "0" * 32
    assert run(other) == 1
    other = json.loads(json.dumps(lines))
    other[0]["dyn"]["words"] += BLOCK // 8
    assert run(other) == 1
    assert run(lines, "--dyn-n", "9") == 1, "a record of a chain of another length was accepted"
    assert run(lines, "--pad-seed", "5eee") == 1
    extra = lines + [
        {
            "type": "verification",
            "id": 9,
            "dyn": {"nonce": f"{0x9999:016x}", "j0": 0, "received": "0" * 32, "words": 4096},
        }
    ]
    assert run(extra) == 1 and run(extra, "--partial") == 0
    # failed challenges, as memory_accounting.verifier.net logs them: no READY (no chain sent), no reply to the main chain, no reply
    # to
    # the live chain. They used to raise KeyError (no "received"); a challenge with no reply needs no record
    faults = [
        {
            "type": "verification",
            "id": 10,
            "dyn": {"rtt_ns": None, "arm_ns": None, "words": 0, "blocks": 0, "fault": "no dynamic READY"},
        },
        {
            "type": "verification",
            "id": 11,
            "dyn": {
                "rtt_ns": None,
                "words": 4096,
                "blocks": 2,
                "nonce": f"{0x7777:016x}",
                "j0": 1,
                "fault": "no reply to the dynamic chain",
            },
        },
    ]
    no_live = json.loads(json.dumps(lines))
    no_live[1]["dyn"]["live"].pop("received")
    no_live[1]["dyn"]["fault"] = "no reply to the live chain"
    assert run(lines + faults) == 0
    assert run(no_live) == 0, "the live record of a challenge that got no reply was held against it"
    live_nonce = int(no_live[1]["dyn"]["live"]["nonce"], 16)
    (ev_dir / f"{live_nonce:016x}.json").rename(tmp / "aside.json")  # its record absent too: still nothing missing
    assert run(no_live) == 0
    assert run(lines) == 1, "a record with a reply logged went missing and the audit passed"
    (tmp / "aside.json").rename(ev_dir / f"{live_nonce:016x}.json")
    main_nonce = int(lines[0]["dyn"]["nonce"], 16)  # a selectively omitted record fails the audit
    (ev_dir / f"{main_nonce:016x}.json").rename(tmp / "aside.json")
    assert run(lines) == 1
    (tmp / "aside.json").rename(ev_dir / f"{main_nonce:016x}.json")
    empty = tmp / "empty"
    empty.mkdir()
    log = tmp / "live.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in lines))
    try:
        consistency_main(
            [
                "--evidence",
                str(empty),
                "--log",
                str(log),
                "--pad-seed",
                "5eed",
                "--hash-seed",
                "3",
                "--dyn-B",
                "16K",
                "--dyn-n",
                "8",
                "--r",
                "16",
                "--k",
                "1024",
            ]
        )
        raise AssertionError("no exit")
    except SystemExit as e:
        assert e.code not in (0, None), "an empty evidence directory passed the audit of a log with challenges"


def test_failed_arm_unfreezes_the_region():
    """An exception in DynamicResponder.arm() after the freeze (here the stage, then the live region) left the
    region frozen: the inference thread parked at its boundary, or the pause lock held, for ever. arm now
    unfreezes, withdraws the pause request and drops its partial state, then raises ArmFailed."""
    from memory_accounting.prover.dynamic import ArmFailed

    model, region, tracker, pause, hooks = toy_dynamic(
        "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=9
    )

    def bad_stage():
        raise RuntimeError("tokens not known on the worker")

    dyn = DynamicResponder(region, bad_stage, pause, KEYS, model_info=model.info)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=2, trigger_layer=2)  # parked mid-step
    try:
        dyn.arm(None, 1.0)
        raise AssertionError("arm succeeded without a stage")
    except ArmFailed as e:
        assert "tokens not known" in str(e)
    th.join(5.0)
    assert not th.is_alive(), "the inference thread is still parked after a failed ARM"
    assert not dyn.frozen and not pause.paused
    try:  # idle: the lock was taken, then the stage failed
        dyn.arm(None, 0.05)
        raise AssertionError("arm succeeded without a stage")
    except ArmFailed:
        pass
    assert pause.lock.acquire(blocking=False), "the pause lock is still held after a failed ARM"
    pause.lock.release()

    def bad_live(stage):
        raise ValueError("live region")

    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info, live_fn=bad_live)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=1)
    try:
        dyn.arm(None, 1.0, want_live=True)
        raise AssertionError("arm succeeded without its live region")
    except ArmFailed:
        pass
    th.join(5.0)
    assert not th.is_alive() and not dyn.frozen and not dyn.live_pending
    _words, _bb, _flags, _live_words = dyn.arm(None, 0.05)  # the responder still works
    ch = _challenge(region, 91)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    assert verify(dyn.store.get(91), toy_replayer(), KEYS)["ok"]
    hooks.remove()


def _live_pair(seed=10):
    """A main record and the live record of the same freeze (toy, frozen mid-step)."""
    model, region, tracker, pause, hooks = toy_dynamic(
        "cpu", BLOCK, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=seed
    )
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info, live_fn=static_live_fn(model, region))
    th = _run_until_paused(model, region, tracker, pause, trigger_step=4, trigger_layer=2)
    _words, _bb, _flags, live_words = dyn.arm(None, 1.0, want_live=True)
    ch = _challenge(region, 0x7000)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    chl = Challenge(BLOCK, 16, 16, 0x7001, 0x7001 % (live_words * 8 // BLOCK), 1024)
    llo, lhi = dyn.answer(chl, live=True)
    dyn.finish(chl, llo, lhi, live=True)
    th.join(5.0)
    hooks.remove()
    return model, region, dyn.store.get(0x7000), dyn.store.get(0x7001)


def _fake_live(model, region, live, rp):
    """The reviewer's cheat: a live record of another, shorter stage (the first step only), whose region the
    prover can rebuild exactly (it can replay too), with the digest over it."""
    from memory_accounting.prover.dynamic import DynamicRegion, live_tokens
    from memory_accounting.verifier.consistency import chain_digest_lazy

    fake = copy.deepcopy(live)
    fake.stage = dict(
        live.stage, chunks=live.stage["chunks"][:1], step=0, in_flight=False, act_layer=3, kv_layers_done=4
    )
    segs = [
        (n, t)
        for n, t in model.live_kv_buffers(live_tokens(fake.stage)) + [(ACT_NAME, region.find(ACT_NAME).tensor)]
        if t.numel()
    ]
    lr = DynamicRegion(segs, BLOCK, region.pad_seed, live=True)
    fake.layout, fake.words = lr.layout(), lr.words
    words = rp.region_words(fake)
    fake.digest = list(
        chain_digest_lazy(words, fake, KEYS) if hasattr(words, "block") else chain_digest(words, fake, KEYS)
    )
    return fake


def _audit(ev_dir, log, *extra) -> int:
    from memory_accounting.verifier.consistency import main as consistency_main

    try:
        consistency_main(
            [
                "--evidence",
                str(ev_dir),
                "--log",
                str(log),
                "--pad-seed",
                "5eed",
                "--hash-seed",
                "3",
                "--dyn-B",
                "16K",
                "--dyn-n",
                "16",
                "--r",
                "16",
                "--k",
                "1024",
                *extra,
            ]
        )
    except SystemExit as e:
        return e.code


def test_live_record_is_replayed_from_its_main_stage():
    """Review finding: nothing tied the live record to the main record of the same freeze. A prover answered
    the live chain over the live region of another stage (fewer tokens) and both records verified, each
    replayed from its own stage. The live region is now the main stage's: a live record whose stage is not its
    main record's fails, online (`verify(..., main=)`) and in the offline audit, which pairs them through the
    verifier's log."""
    import json

    model, region, main, live = _live_pair()
    rp = toy_replayer()
    assert verify(main, rp, KEYS)["ok"] and verify(live, rp, KEYS)["ok"]
    fake = _fake_live(model, region, live, rp)
    assert fake.stage != main.stage
    try:
        rep = verify(fake, rp, KEYS, main=main)
    except TypeError:  # (no pairing at all)
        rep = verify(fake, rp, KEYS)
    assert not rep["ok"] and "stage" in rep.get("error", ""), rep
    assert verify(live, rp, KEYS, main=main)["ok"]
    other = copy.deepcopy(live)
    other.model = dict(live.model, seed=99)
    assert not verify(other, rp, KEYS, main=main)["ok"]
    # offline: the log pairs the live chain with the main chain of the same verification
    d = pathlib.Path(tempfile.mkdtemp())

    def setup(live_rec, with_main=True):
        ev_dir = d / f"ev{len(list(d.iterdir()))}"
        for r in ((main,) if with_main else ()) + (live_rec,):
            r.save(ev_dir)

        def shape(r):
            return [r.challenge[k] for k in ("block_bytes", "n", "r", "k", "g")]

        line = {
            "type": "verification",
            "id": 1,
            "dyn": {
                "nonce": f"{main.nonce:016x}",
                "j0": main.challenge["j0"],
                "words": main.words,
                "received": f"{main.digest[1]:016x}{main.digest[0]:016x}",
                "shape": shape(main),
                "live": {
                    "nonce": f"{live_rec.nonce:016x}",
                    "j0": live_rec.challenge["j0"],
                    "words": live_rec.words,
                    "shape": shape(live_rec),
                    "received": f"{live_rec.digest[1]:016x}{live_rec.digest[0]:016x}",
                },
            },
        }
        (ev_dir.parent / "live.jsonl").write_text(json.dumps(line) + "\n")
        return ev_dir, ev_dir.parent / "live.jsonl"

    assert _audit(*setup(live)) == 0
    assert _audit(*setup(fake)) == 1, "a live record of another stage passed the audit"
    assert _audit(*setup(live, with_main=False), "--partial") == 1, "a live record without its main record passed"


def test_replay_shapes_are_the_verifiers():
    """Review finding: the replay built whatever the record described: a toy model of the record's size, an
    integer model per record cache length (kept for ever), an activation buffer of the record's shape and dtype,
    the whole region as one array (5 records: RSS 192 to 1707 MiB). Shapes now come from the verifier's model and
    configuration, a record claiming others is refused before anything is built, one model is kept at a time,
    and the region is read block by block."""
    import memory_accounting.models.toy
    from memory_accounting.core.evidence import Evidence

    built = []
    orig = memory_accounting.models.toy.ToyLM

    class Counting(orig):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

    memory_accounting.models.toy.ToyLM = Counting
    try:
        rp = toy_replayer(max_len=48)
        for d, L, ml in ((1024, 8, 2048), (64, 16, 48), (64, 4, 4096), (64, 4, 49)):
            ev = Evidence(
                nonce=1,
                challenge={},
                digest=[0, 0],
                words=1,
                block_bytes=1 << 16,
                pad_seed=0x5EED,
                layout=[{"name": "act", "dtype": "float32", "shape": [1 << 30], "words": 1 << 29}],
                stage={
                    "kind": "static-cache",
                    "step": -1,
                    "chunks": [],
                    "act_layer": 0,
                    "in_flight": False,
                    "n_layers": L,
                },
                model={"kind": "toy", "seed": 0, "d": d, "n_layers": L, "max_len": ml, "vocab": 97},
            )
            rep = verify(ev, rp, KEYS)
            assert not rep["ok"], rep
        assert not built, f"models built for refused records: {built}"
        rp = toy_replayer()  # no length named: the record's, up to TOY_MAX_LEN
        ev.model = {"kind": "toy", "seed": 0, "d": 64, "n_layers": 4, "max_len": 1 << 20, "vocab": 97}
        assert not verify(ev, rp, KEYS)["ok"] and not built
    finally:
        memory_accounting.models.toy.ToyLM = orig
    # an honest record: the region is a view (blocks on demand), the activation buffer the verifier's own shape
    _model, _region, main, live = _live_pair(seed=11)
    words = toy_replayer(max_len=48).region_words(main)
    assert hasattr(words, "block") and not isinstance(words, np.ndarray)
    assert (
        verify(main, toy_replayer(max_len=48), KEYS)["ok"]
        and verify(live, toy_replayer(max_len=48), KEYS, main=main)["ok"]
    )
    big = copy.deepcopy(main)
    big.layout = [dict(e, shape=[1 << 28], words=1 << 27) if e["name"] == ACT_NAME else e for e in main.layout]
    rep = verify(big, toy_replayer(max_len=48), KEYS)
    assert not rep["ok"] and "layout" in rep["error"], rep
    other = copy.deepcopy(main)
    other.model["max_len"] = 64
    assert "verifier" in verify(other, toy_replayer(max_len=48), KEYS)["error"]
    big = copy.deepcopy(main)
    big.block_bytes = 1 << 40
    assert "region block" in verify(big, toy_replayer(max_len=48), KEYS)["error"]


def test_int_replayer_keeps_one_model():
    """The integer replayer kept one full model per record cache length. It now keeps one model and resizes its
    cache; with a length named, a record of another is refused before the file is read."""
    from transformers import LlamaConfig, LlamaForCausalLM

    import memory_accounting.models.intllm
    from memory_accounting.models.intllm import convert_hf
    from memory_accounting.models.toy import int_dynamic

    tmp = pathlib.Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=101,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    ).eval().save_pretrained(tmp / "tiny")
    path = tmp / "tiny.int.safetensors"
    convert_hf(str(tmp / "tiny"), str(path), max_pos=256)
    recs = []
    for ml in (48, 32, 48):
        model, region, tracker, pause, hooks = int_dynamic(path, "cpu", BLOCK, pad_seed=0xF00D, max_len=ml)
        dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info)
        th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=1, new=6)
        dyn.arm(None, 1.0)
        ch = _challenge(region, 0x600 + ml + len(recs))
        lo, hi = dyn.answer(ch)
        dyn.finish(ch, lo, hi)
        th.join(5.0)
        hooks.remove()
        recs.append(dyn.store.get(ch.nonce))
    loads = []
    orig = memory_accounting.models.intllm.IntLLM.from_file.__func__

    def counting(cls, *a, **kw):
        loads.append(a)
        return orig(cls, *a, **kw)

    memory_accounting.models.intllm.IntLLM.from_file = classmethod(counting)
    try:
        rp = int_replayer(path)
        for ev in recs:
            assert verify(ev, rp, KEYS)["ok"]
        assert len(loads) == 1, loads
        rep = verify(recs[1], int_replayer(path, max_len=48), KEYS)
        assert not rep["ok"] and "verifier's 48" in rep["error"] and len(loads) == 1, rep
        assert verify(recs[0], int_replayer(path, max_len=48), KEYS)["ok"]
    finally:
        memory_accounting.models.intllm.IntLLM.from_file = classmethod(orig)


def test_static_stage_is_checked_before_replay():
    """Review finding: a completed step's act_layer was free, so a prover could declare act_layer 0 and keep the
    embeddings (regenerable from the tokens) in the activation buffer; and a chunk of several rows fed tokens
    the live region did not count. Now a completed step has act_layer = n_layers - 1, every chunk is one row of
    token ids, and the stage fits the verifier's model, before anything is replayed."""
    from memory_accounting.verifier.consistency import ReplayError, chain_digest_lazy

    model, region, tracker, pause, hooks, dyn = _toy(seed=12)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3)  # a step boundary: step 2 complete
    dyn.arm(None, 1.0)
    ch = _challenge(region, 0x6100)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    hooks.remove()
    ev = dyn.store.get(ch.nonce)
    rp = toy_replayer()
    assert verify(ev, rp, KEYS)["ok"] and not ev.stage["in_flight"] and ev.stage["act_layer"] == 3

    def cheat(**stage):
        """A record of this stage whose digest is what the verifier rebuilds for it (if it rebuilds anything)."""
        bad = copy.deepcopy(ev)
        bad.stage.update(stage)
        try:
            w = rp.region_words(bad)
            bad.digest = list(chain_digest_lazy(w, bad, KEYS) if hasattr(w, "block") else chain_digest(w, bad, KEYS))
        except (ReplayError, ValueError, RuntimeError, IndexError):
            pass
        return verify(bad, rp, KEYS)

    rep = cheat(act_layer=0, kv_layers_done=4)  # embeddings in the activation buffer
    assert not rep["ok"] and "layer" in rep.get("error", ""), rep
    rep = cheat(chunks=[[c[0], c[0]] for c in ev.stage["chunks"]])  # two rows per chunk
    assert not rep["ok"] and "one row" in rep.get("error", ""), rep
    for bad, why in (
        ({"act_layer": 4}, "outside"),
        ({"act_layer": -1}, "outside"),
        ({"step": 5}, "chunks"),
        ({"n_layers": 5}, "layer"),
        ({"chunks": [[[200]]] + ev.stage["chunks"][1:]}, "vocabulary"),
        ({"chunks": [[[1] * 49]] + ev.stage["chunks"][1:]}, "holds 48"),
        ({"step": -1}, "step -1"),
        ({"in_flight": True, "kv_layers_done": 1}, "kv_layers_done"),
        ({"chunks": [[[]]] + ev.stage["chunks"][1:]}, "one row"),
    ):
        rep = cheat(**bad)
        assert not rep["ok"] and why in rep.get("error", ""), (bad, rep)


def test_hf_records_need_the_verifiers_model():
    """Offline, an HF record was replayed with the model the record named when --model was absent: the
    prover's choice. The verifier must name it."""
    import os

    from memory_accounting.verifier.consistency import main as consistency_main

    os.environ["HF_HUB_OFFLINE"] = "1"
    tmp = pathlib.Path(tempfile.mkdtemp())
    Evidence(
        nonce=0x77,
        challenge={"block_bytes": BLOCK, "n": 8, "r": 16, "k": 1024, "g": 1, "j0": 0},
        digest=[0, 0],
        words=BLOCK // 8,
        block_bytes=BLOCK,
        pad_seed=1,
        layout=[],
        stage={"kind": "static-cache", "step": -1, "chunks": [], "act_layer": 0, "in_flight": False, "n_layers": 1},
        model={"kind": "hf", "name": "some/prover-chosen-model"},
    ).save(tmp)
    try:
        consistency_main(["--evidence", str(tmp)])
        raise AssertionError("no exit")
    except SystemExit as e:
        assert "--model" in str(e.code), e.code


def _host_rows(region, dram) -> dict:
    """Per segment, the words the current table reads from the host copy (as [lo, hi) within the segment)."""
    n, t = region.nseg, region._segtab_np.view(np.uint64)
    out = {}
    for i in np.flatnonzero(dram.reads_host(t[n + 1 :])):
        lo, hi = int(t[i]), int(t[i + 1])
        for s in region.segments:
            a, b = max(lo, s.offset), min(hi, s.offset + s.words)
            if a < b:
                out.setdefault(s.name, []).append((a - s.offset, b - s.offset))
    return out


def test_dyn_dram_static_cache():
    """dyn_dram on a static cache (the toy here; the integer model's buffers are laid out the same, slots
    outermost): the top slots of every K/V buffer keep their filler in a host copy, and those the stage has not
    fed are read from there; the fed ones, the activation buffer and the pad stay in place. Same digest as
    honest, and the verifier's replay reproduces it."""
    from memory_accounting.models.toy import static_dram_fn
    from memory_accounting.prover.dynamic import live_tokens

    model, region, tracker, pause, hooks, dyn = _toy(include_bytes=True)
    dyn.dram_fn = static_dram_fn(model, region)
    S = model.max_len
    dram = dyn.prepare_dram(0.9)  # slots 4..47: some of them live
    s0 = S - dram.ranges[0][2]
    assert s0 == 4 and len(dram.ranges) == 2 * len(model.layers)
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=2)
    dyn.set_posture("dyn_dram")
    dyn.arm(None, 1.0)
    live = live_tokens(dyn.stage)
    assert live == 8 and not region.own_table
    rows = _host_rows(region, dram)
    for name, t in model.kv_buffers():
        uw = region.find(name).words // S
        assert rows.pop(name) == [(live * uw, S * uw)], name  # slots live..S-1 from host memory, 4..7 in place
    assert not rows, f"read from host memory: {sorted(rows)} (only the K/V buffers' free slots may be)"
    ch = _challenge(region, 0xD7A, n=96)
    lo, hi = dyn.answer(ch)
    region.restore_table()
    assert dyn.answer(ch) == (lo, hi), "the cheat's digest is the honest one"
    dram.apply(dyn.stage)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    rep = verify(dyn.store.get(ch.nonce), toy_replayer(), KEYS)
    assert rep["ok"], rep
    dyn.set_posture("honest")
    assert region.own_table
    hooks.remove()


def test_dyn_dram_posture_through_the_responder():
    """The responder offers dyn_dram only with a dynamic region that has an evictable set, makes its host copy at
    start, and switches it with the posture datagram (the static array's posture as well); on the wire it is
    refused like the other posture-only cheats. Its dynamic answers are the honest ones."""
    from types import SimpleNamespace

    from memory_accounting.core import wire
    from memory_accounting.models.toy import static_dram_fn
    from memory_accounting.prover import responder as R
    from memory_accounting.prover import strategies as P

    model, region, _tracker, _pause, hooks, dyn = _toy()
    calls = []
    resident = SimpleNamespace(set_posture=calls.append, device="cpu")

    class Stub:
        armable = False

        def __init__(self, hbm, keys):
            pass

        def close(self):
            pass

    saved = dict(P.PROVERS)
    P.PROVERS.update({"honest": Stub, "dyn_dram": Stub})  # the static chains are not what this tests
    try:
        try:
            R.Responder(resident, KEYS, strategies=("honest", "dyn_dram"), dynamic=dyn)
            raise AssertionError("dyn_dram without an evictable set")
        except ValueError as e:
            assert "evictable" in str(e)
        dyn.dram_fn = static_dram_fn(model, region)
        r = R.Responder(resident, KEYS, strategies=("honest", "dyn_dram"), dynamic=dyn, dyn_evict_fraction=0.25)
    finally:
        P.PROVERS.clear()
        P.PROVERS.update(saved)
    assert dyn.dram is not None and dyn.dram.fraction == 0.25 and not dyn.dram_on, (
        "made at start, off until the posture"
    )
    sent, addr = [], ("127.0.0.1", 9)
    sock = SimpleNamespace(sendto=lambda data, a: sent.append(data))

    def round_(nonce=0xAB):
        r._handle(sock, wire.ARM.pack(wire.ARM_DYN_MAGIC, 5, BLOCK, 16, 16, 1024, 1, 0), addr, None)
        assert len(sent[-1]) == wire.READY_DYN.size
        r._handle(sock, wire.CH.pack(wire.DYN_MAGIC, nonce, 2, BLOCK, 16, 16, 1024, 1, 0), addr, None)
        return wire.RE.unpack(sent[-1])[1:]

    def posture(name):
        r._handle(sock, wire.POSTURE.pack(wire.POSTURE_MAGIC, 1, wire.STRATS.index(name)), addr, None)
        return wire.STRATS[wire.POSTURE.unpack(sent[-1])[2]]

    honest = round_()
    assert posture("dyn_dram") == "dyn_dram" and calls[-1] == "dyn_dram" and dyn.dram_on
    assert round_() == honest and dyn.dram.host_words_read > 0 and not dyn.frozen
    f = wire.STATSRE.unpack(r.stats_packet(3))
    bit = 1 << wire.STRATS.index("dyn_dram")
    assert not f[11] & bit and f[12] & bit, "a posture, never a wire strategy"
    assert posture("honest") == "honest" and calls[-1] == "honest" and not dyn.dram_on and region.own_table
    assert round_() == honest
    hooks.remove()


def _tiny_int(seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    from memory_accounting.models.intllm import convert_hf

    tmp = pathlib.Path(tempfile.mkdtemp())
    torch.manual_seed(seed)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=101,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    ).eval().save_pretrained(tmp / "tiny")
    convert_hf(str(tmp / "tiny"), str(tmp / "tiny.int.safetensors"), max_pos=256)
    return tmp / "tiny.int.safetensors"


def test_integer_model_region_is_encoded():
    """v0.17: the integer model's region is every layer's cache buffers and the activation buffer as encoded stores
    (memory_accounting.prover.kvenc), plaintext only in one shared scratch cache. The model's output does not change; an idle
    region is the encoding of zeros (its filler view); the live region is encoded stores too; the plaintext scratch is
    outside the region (writing it changes nothing a chain reads), and a changed word of a store fails the replay."""
    from memory_accounting.models.intllm import IntLLM
    from memory_accounting.models.toy import int_dynamic
    from memory_accounting.prover.kvenc import decode_np

    path = _tiny_int()
    ref = IntLLM.from_file(path, 48, "cpu").generate([5, 17, 3, 99], 12)
    model, region, tracker, pause, hooks = int_dynamic(path, "cpu", BLOCK, pad_seed=0xF00D, max_len=48)
    assert all(s.enc is not None for s in region.segments if s.name != "pad"), [s.layout() for s in region.segments]
    assert all(layer.k_cache is model.layers[0].k_cache for layer in model.layers), (
        "one plaintext cache for every layer"
    )
    assert model.generate([5, 17, 3, 99], 12) == ref, "holding the cache encoded changed the output"
    s = region.find("kv.1.k")  # after generating: layer 1's store decodes to layer 1's cache
    ref_model = IntLLM.from_file(path, 48, "cpu")
    ref_model.generate([5, 17, 3, 99], 12)
    got = decode_np(s.tensor.numpy().view(np.uint64), s.offset, region.dyn_key, s.enc.plain)
    assert np.array_equal(got, ref_model.layers[1].k_cache.reshape(-1).view(torch.int64).numpy().view(np.uint64))
    model.reset()
    region.reset_to_filler()
    region.encoded_cache.reset()
    tracker.new_request(1)
    assert np.array_equal(region.snapshot(), region.filler_view()), "an idle encoded region is the encoding of zeros"
    dyn = DynamicResponder(region, tracker, pause, KEYS, model_info=model.info, live_fn=static_live_fn(model, region))
    th = _run_until_paused(model, region, tracker, pause, trigger_step=3, trigger_layer=2)
    _words, _bb, flags, _live_words = dyn.arm(None, 1.0, want_live=True)
    assert flags == 1 and all(s.enc is not None for s in dyn.live_region.segments if s.name != "pad")
    model.layers[2].v_cache.view(-1)[0] += 1  # the scratch: no chain reads it
    ch = _challenge(region, 0x7300)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    chl = Challenge(BLOCK, 16, 16, 0x7301, 0, 1024)
    lo, hi = dyn.answer(chl, live=True)
    dyn.finish(chl, lo, hi, live=True)
    th.join(5.0)
    rp = int_replayer(path)
    main, live = dyn.store.get(0x7300), dyn.store.get(0x7301)
    assert main.stage["encoded"] is True
    assert verify(main, rp, KEYS)["ok"] and verify(live, rp, KEYS, main=main)["ok"]
    th = _run_until_paused(model, region, tracker, pause, trigger_step=2, trigger_layer=1)
    dyn.arm(None, 1.0)
    region.find("kv.0.k").tensor.view(-1)[3] ^= 1 << 7  # a word of a live store
    ch = Challenge(BLOCK, 96, 16, 0x7302, 0, 1024)
    lo, hi = dyn.answer(ch)
    dyn.finish(ch, lo, hi)
    th.join(5.0)
    assert not verify(dyn.store.get(0x7302), rp, KEYS)["ok"]
    plain = copy.deepcopy(main)
    plain.stage.pop("encoded")  # the same record claimed plain: the verifier's plain region differs
    assert not verify(plain, rp, KEYS)["ok"]
    hooks.remove()


if __name__ == "__main__":
    from _runner import run

    run(globals())
