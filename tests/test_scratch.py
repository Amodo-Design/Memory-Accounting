"""Bounding the prover's scratch: filler the verifier regenerates instead of storing, the per-verification
memory bound, and the fast freeze that keeps the READY window to about one layer."""

import pathlib
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from memory_accounting.core.backend import CpuLib
from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params
from memory_accounting.verifier.recompute import ImageWithFiller, Verifier

SEED, KEY = 0x5EED5EED5EED5EED, 0x0123456789ABCDEF0123456789ABCDEF


def _encoded_filler(lib, q0, n):
    pay = np.empty(n, np.uint64)
    lib.filler(pay, SEED, q0)
    enc = np.empty(n, np.uint64)
    lib.setup_at(pay, enc, KEY, q0)
    return enc


def test_filler_image_matches_encoded_filler():
    lib = CpuLib()
    n = 1 << 14
    img = np.empty(n, np.uint64)
    lib.image(_encoded_filler(lib, 0, n), img)
    for q0, c, th in [(0, n, 1), (64, 4096, 3), (n - 1000, 1000, 8)]:
        out = np.empty(c, np.uint64)
        lib.filler_image(out, SEED, KEY, q0, th)
        assert np.array_equal(out, img[q0 : q0 + c]), (q0, c)
    try:
        lib.filler_image(np.empty(6, np.uint64), SEED, KEY, 2)
        raise AssertionError("unaligned range accepted")
    except ValueError:
        pass


def test_lazy_chain_equals_full_chain():
    lib = CpuLib()
    B = 1 << 20
    bw = B // 8
    nb = 12
    N = nb * bw
    P = 3 * bw + 512  # the weights end inside a block
    rng = np.random.default_rng(3)
    arr = np.concatenate([rng.integers(0, 1 << 60, P, dtype=np.uint64), _encoded_filler(lib, P, N - P)])
    img = np.empty(N, np.uint64)
    lib.image(arr, img)
    cfg = MemoryConfig(words=N, encoding="vde", perm=False, seed_t=KEY, image=False)
    keys = Keys.deterministic(2)
    full = Verifier(img, cfg, keys, threads=2)
    lazy = Verifier(ImageWithFiller(img[:P].copy(), N, SEED, KEY, lib, threads=2), cfg, keys, threads=2)
    for i in range(6):
        ch = Challenge.random(B, 6, 64 * (i % 2), nb, rng, k=128, g=1)
        assert full.expected(ch) == lazy.expected(ch), i


def test_bundle_with_regenerated_filler():
    """The bundle holds Θ''s image only; the verifier regenerates the filler and reproduces the prover's
    chain over Θ' ‖ filler (prover side: image on the fly, as on the GPU)."""
    from memory_accounting.prover.filler import FillerRegion
    from memory_accounting.prover.stream_decode import convert
    from memory_accounting.setup.manifest import Manifest
    from tests.test_stream_decode import Toy

    torch.manual_seed(1)
    sd = convert(Toy(d=128, n=2), enc_key=KEY)
    lib = CpuLib()
    block = 16384
    bw = block // 8
    fw = 7 * bw
    from memory_accounting.setup.filler import static_filler_key

    fk = static_filler_key(SEED, 0)  # GPU 0's filler key, from the verifier's seed
    pay = np.empty(fw, np.uint64)
    lib.filler(pay, SEED, sd.payload_words)
    enc = np.empty(fw, np.uint64)
    lib.setup_at(pay, enc, fk, sd.payload_words)
    fill = FillerRegion(torch.from_numpy(enc.view(np.int64)), fw, fk)  # the filler under its own key, not the weights'
    d = pathlib.Path(tempfile.mkdtemp())
    keys = Keys.deterministic(1)
    sd.export_bundle(d, filler=fill, filler_seed=SEED, block_bytes=block, r=16, k=1024, hbm_bytes=1 << 30)
    manifest = Manifest.load(d / "manifest.json")
    T = np.load(manifest.gpus[0].image_file).view(np.uint64)
    assert len(T) == sd.payload_words and int(manifest.filler_key, 16) == fk != KEY
    assert manifest.accounted_words == (sd.payload_words + fw) // bw * bw
    arr = np.concatenate([sd.theta.numpy().view(np.uint64), fill.tensor.numpy().view(np.uint64)])[
        : manifest.accounted_words
    ]
    prover_cfg = MemoryConfig(words=manifest.accounted_words, encoding="vde", perm=False, seed_t=KEY, image=True)
    ver = Verifier(
        ImageWithFiller(T, manifest.accounted_words, SEED, fk, lib, threads=2),
        manifest.memory_config(0, image=False),
        keys,
        threads=2,
    )
    nb = manifest.accounted_words // bw
    for r, j0 in ((0, 0), (16, nb - 1), (16, nb // 2)):
        ch = Challenge(block, 6, r, 777 + j0, j0, 1024)
        a = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(np.ascontiguousarray(arr), pack_params(prover_cfg, ch, keys, MODE_HONEST), a)
        lo, hi, visited = ver.expected(ch)
        assert (lo, hi) == (int(a[0]), int(a[1])) and visited == a[2 : 2 + ch.n].tolist(), (r, j0)


def test_memory_bound():
    from memory_accounting.verifier.net import _memory_bound

    G = 1 << 30
    cfg = MemoryConfig(words=128 * G // 8, encoding="vde", perm=False, image=False)
    res = {"honest": {"rtt_ns": [1, 2, 3], "dyn_words": [2 * G // 8, 6 * G // 8, 8 * G // 8]}}
    args = types.SimpleNamespace(gpu_type=None, hbm_bytes=str(140 * G), max_scratch="8.5G", dyn_dtype="float64")
    bound = _memory_bound(args, cfg, res, "honest", ready_ns=2e6, dyn_ready_ns=1e6)
    # 62/64 credited
    assert (
        bound["capacity_bytes"] == 140 * G
        and bound["static_array_bytes"] == 128 * G
        and bound["static_bytes"] == 124 * G
    )
    # float64 filler: 62/64
    assert bound["dynamic_credit_bits"] == 62 and bound["dynamic_credited_bytes_max"] == 8 * G * 62 // 64
    assert (
        bound["unchecked_bytes_max"] == 16 * G - 2 * G * 62 // 64
        and bound["over"] == [True, True, False]
        and bound["over_bound"] == 2
    )
    assert abs(bound["reload_static_bytes"] - 0.1e9) < 1 and abs(bound["reload_dynamic_bytes"] - 0.05e9) < 1
    bound = _memory_bound(
        types.SimpleNamespace(gpu_type=None, hbm_bytes=str(140 * G), max_scratch="8.5G"), cfg, res, "honest"
    )
    # default bfloat16: 59/64
    assert bound["dynamic_credit_bits"] == 59 and bound["unchecked_bytes_max"] == 16 * G - 2 * G * 59 // 64
    assert bound["over"] == [True, True, True]  # 8 GiB of bf16 filler leaves 8.625 GiB unchecked
    args = types.SimpleNamespace(gpu_type="NVIDIA H200 NVL", hbm_bytes=None, max_scratch=None)
    assert (
        _memory_bound(args, cfg, {"honest": {"rtt_ns": [1], "dyn_words": []}}, "honest")["unchecked_bytes_max"]
        == 150_111_977_472 - 124 * G
    )
    try:
        _memory_bound(types.SimpleNamespace(gpu_type=None, hbm_bytes=None, max_scratch="4G"), cfg, res, "honest")
        raise AssertionError("no capacity accepted")
    except SystemExit:
        pass


def test_batch_memory_bound_rejects_impossible_totals():
    """0.5 GiB static and 2 GiB dynamic declared on a 1 GiB GPU: impossible, so over whatever the bound (the
    live verifier's check). Before, the negative remainder was under every bound and passed."""
    from memory_accounting.verifier.net import _memory_bound

    G = 1 << 30
    cfg = MemoryConfig(words=G // 2 // 8, encoding="vde", perm=False, image=False)
    res = {"honest": {"rtt_ns": [1, 2], "dyn_words": [2 * G // 8, G // 8 // 4], "dyn_live_words": [0, G // 8 // 2]}}
    for bound in ("64M", None):
        m = _memory_bound(types.SimpleNamespace(gpu_type=None, hbm_bytes=str(G), max_scratch=bound), cfg, res, "honest")
        # the second: live > pool
        assert m["over"] == [True, True] and m["over_bound"] == 2 and m["impossible"] == 2, (
            bound,
            m,
        )


def test_two_bits_of_every_encoded_word_are_zero():
    """What the memory claim credits the static array with rests on this: VDE words hold a root below
    2^61 and a flag in bit 61, so bits 62 and 63 carry nothing and only 62/64 of the array is forced."""
    from memory_accounting.verifier.net import VDE_WORD_BITS, credited_static_bytes

    lib = CpuLib()
    enc = _encoded_filler(lib, 0, 1 << 14)
    assert int((enc >> np.uint64(VDE_WORD_BITS)).max()) == 0 and int((enc >> np.uint64(61)).max()) == 1
    assert credited_static_bytes(MemoryConfig(words=1 << 20, encoding="vde", perm=False)) == (1 << 23) * 62 // 64
    assert credited_static_bytes(MemoryConfig(words=1 << 20, encoding="none", perm=False)) == 1 << 23


def test_dynamic_filler_is_credited_at_its_incompressible_bits():
    """Free dynamic memory holds VDE filler masked for its buffer's dtype: bits 62 and 63 are always zero, and
    the bf16/fp16 mask also clears bits 14, 30 and 46, so a bf16 pool packs losslessly into 59 bits a word. The
    memory claim credits the dynamic region at exactly the bits that vary (before: 64/64, which on the ~58 GiB
    vLLM pool left about 4.5 GiB a prover could keep for itself unaccounted). Checked on real filler."""
    from memory_accounting.prover.dynfill import dyn_key, filler_np, mask_for
    from memory_accounting.verifier.net import DYN_DTYPES, _memory_claim, dyn_word_bits

    expect = {"bfloat16": 59, "float16": 59, "float32": 61, "float64": 62, "int8": 62, "int64": 62}
    for dt in DYN_DTYPES:
        words = filler_np(1 << 13, 4096, 0x5EED, dyn_key(0x5EED, 0), mask_for(dt))
        ones = int(np.bitwise_or.reduce(words))  # the bits ever set over 8192 words
        assert int(np.bitwise_and.reduce(words)) == 0  # and none always set: they all vary
        assert (ones).bit_count() == dyn_word_bits(dt) == expect.get(dt, 62), (dt, hex(ones))
    assert dyn_word_bits(None) == 59  # the default: the least
    G = 1 << 30
    claim = _memory_claim(140 * G, 76 * G, 64 * G, 0, None, 59)
    assert (
        claim["dynamic_bytes"] == 59 * G
        and claim["dynamic_region_bytes"] == 64 * G
        and claim["unchecked_bytes"] == 5 * G
        and not claim["impossible"]
    )
    claim = _memory_claim(140 * G, 76 * G, 64 * G, G, None, 59, 1 << 16)  # the live region is checked, not credited
    assert (
        claim["dynamic_bytes"] == 63 * G * 59 // 64
        and claim["live_region_bytes"] == G
        and claim["unchecked_bytes"] == 64 * G - 63 * G * 59 // 64
    )
    # the region itself must fit, whatever its credit
    assert _memory_claim(140 * G, 76 * G, 65 * G, G, None, 59)["impossible"]


def _toy_dyn():
    from memory_accounting.models.toy import toy_dynamic
    from memory_accounting.prover.dynamic import DynamicResponder

    model, region, tracker, pause, _hooks = toy_dynamic(
        "cpu", 16384, pad_seed=0x5EED, d=64, n_layers=4, max_len=48, seed=0
    )
    return model, tracker, pause, DynamicResponder(region, tracker, pause, Keys.deterministic(3), model_info=model.info)


def test_freeze_is_immediate_without_a_forward():
    _model, _tracker, pause, dyn = _toy_dyn()
    t = time.perf_counter()
    reached = dyn.freeze(5.0)
    dt = time.perf_counter() - t
    assert reached is False and dyn.locked and dt < 0.05, dt
    dyn.release()
    assert pause.lock.acquire(blocking=False)
    pause.lock.release()


def test_freeze_mid_forward_waits_one_layer():
    model, tracker, _pause, dyn = _toy_dyn()
    for layer in model.layers:  # slow layers: a forward takes 4 x 30 ms
        layer.register_forward_pre_hook(lambda m, a: time.sleep(0.03))
    started = threading.Event()
    model.layers[0].register_forward_pre_hook(lambda m, a: started.set())
    th = threading.Thread(target=lambda: model.generate([1, 2, 3], 3), daemon=True)
    th.start()
    assert started.wait(5.0)
    t = time.perf_counter()
    reached = dyn.freeze(5.0)
    dt = time.perf_counter() - t
    assert reached is True and dyn.holding and tracker.stage()["in_flight"] and dt < 0.1, dt
    dyn.release()
    th.join(10.0)
    assert not th.is_alive()


def test_freeze_takes_the_lock_when_the_forward_ends_without_a_boundary():
    _model, _tracker, pause, dyn = _toy_dyn()
    pause.acquire_inference()  # a forward with no checkpoint, 80 ms long
    threading.Timer(0.08, pause.lock.release).start()
    t = time.perf_counter()
    reached = dyn.freeze(5.0)
    dt = time.perf_counter() - t
    assert reached is False and dyn.locked and 0.07 < dt < 0.3, dt
    assert not pause._want  # the pause request was withdrawn
    dyn.release()


def test_fill_pages_equals_per_page_fill():
    """Many pages in one call give the same words as one fill per page (CPU path here; the CUDA kernel is
    checked the same way where a GPU is present)."""
    from memory_accounting.prover.dynamic import DynamicRegion

    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        bufs = [(f"kv.{layer}", torch.zeros(64, 128, dtype=torch.bfloat16, device=dev)) for layer in range(3)] + [
            ("x", torch.zeros(512, dtype=torch.int64, device=dev))
        ]
        region = DynamicRegion(bufs, 16384, 0x5EED, gpu=0)
        b = DynamicRegion([(n, t.clone()) for n, t in bufs], 16384, 0x5EED, gpu=0)
        pw = 128 * 2 // 8 * 4  # 4 rows of a bf16 [64, 128] buffer per page
        pages = [(f"kv.{layer}", r * pw) for layer in (0, 2) for r in (0, 3, 7)] + [("x", 256), ("x", 0)]
        region.fill_pages(pages, pw)
        for name, lo in pages:
            b.fill(name, lo, lo + pw)
        for (n1, t1), (n2, t2) in zip(
            region.buffers if hasattr(region, "buffers") else [(s.name, s.tensor) for s in region.segments],
            b.buffers if hasattr(b, "buffers") else [(s.name, s.tensor) for s in b.segments],
        ):
            assert torch.equal(t1.reshape(-1).view(torch.uint8), t2.reshape(-1).view(torch.uint8)), (dev, n1)
        assert region.find("kv.1").tensor.abs().sum() == 0  # untouched pages stay as they were


if __name__ == "__main__":
    from _runner import run

    run(globals())
