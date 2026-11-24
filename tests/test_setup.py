"""Setup phase: packing, filler, commitments, end-to-end prover/verifier setup on a small weight file."""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pathlib import Path

import numpy as np

from memory_accounting.core import reference as R
from memory_accounting.core.backend import CpuLib
from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params
from memory_accounting.setup.commit import block_leaves, merkle_root
from memory_accounting.setup.encode import decode_array, decode_tensor, load_array
from memory_accounting.setup.filler import filler_words
from memory_accounting.setup.layout import GROUP_RAW_BYTES, padded_stream, plan_layout
from memory_accounting.setup.manifest import Manifest
from memory_accounting.setup.pack import pack60, packed_words_for, unpack60
from memory_accounting.setup.prover import run_setup
from memory_accounting.setup.verify import verify_setup
from memory_accounting.setup.weights import flatten, load_tensors

rng = np.random.default_rng(42)


def test_pack60_roundtrip():
    for nbytes in (0, 1, 7, 8, 119, 120, 121, 1000, 12345, 65536):
        raw = rng.integers(0, 256, nbytes, dtype=np.uint8)
        p = pack60(raw)
        assert len(p) == packed_words_for(nbytes) and len(p) % 16 == 0
        assert int(p.max(initial=0)) < (1 << 60)
        assert np.array_equal(unpack60(p, nbytes), raw), nbytes
    # bit-exact structure: the first payload word is the low 60 bits of the first raw word
    raw = rng.integers(0, 256, 120, dtype=np.uint8)
    assert int(pack60(raw)[0]) == int(raw.view("<u8")[0]) & ((1 << 60) - 1)


def test_c_pack_and_decode_unpack_match_reference():
    lib = CpuLib()
    groups = 37
    raw = rng.integers(0, 1 << 64, 15 * groups, dtype=np.uint64)
    payload = np.zeros(16 * groups, dtype=np.uint64)
    lib.pack(raw, payload, groups)
    assert np.array_equal(payload, pack60(raw.view(np.uint8)))
    key = (rng.integers(0, 1 << 63) << 64) | rng.integers(0, 1 << 63)
    key = int(key)
    theta = np.array([R.vde_encode(int(p), R.tweak_at(key, 1000 + i)) for i, p in enumerate(payload)], dtype=np.uint64)
    big = np.concatenate([np.zeros(1000, dtype=np.uint64), theta])  # the tensor starts at payload word 1000
    back = np.zeros(15 * groups, dtype=np.uint64)
    lib.decode_unpack(big, back, 1000, groups, key)
    assert np.array_equal(back, raw)


def test_layout_is_group_aligned():
    tensors = [
        ("a", rng.integers(0, 256, 1000, dtype=np.uint8), "uint8"),
        ("b", rng.standard_normal(77).astype(np.float32), "float32"),
        ("c", rng.integers(0, 256, 120, dtype=np.uint8), "uint8"),
    ]
    layout = plan_layout(tensors)
    for e in layout:
        assert e.byte_offset % GROUP_RAW_BYTES == 0 and e.payload_offset % 16 == 0 and e.payload_words == 16 * e.groups
        assert e.groups * GROUP_RAW_BYTES >= e.nbytes > (e.groups - 1) * GROUP_RAW_BYTES
    assert layout[1].byte_offset == 9 * GROUP_RAW_BYTES and layout[2].groups == 1
    stream = padded_stream(tensors, layout)
    assert len(pack60(stream)) == layout[-1].payload_offset + layout[-1].payload_words


def test_setup_at_matches_whole_array_encode():
    from memory_accounting.core.params import setup_params

    lib = CpuLib()
    key = (0xAB << 64) | 0xCD
    payload = rng.integers(0, 1 << 60, 4096, dtype=np.uint64)
    whole = np.zeros(4096, dtype=np.uint64)
    lib.setup(payload, whole, setup_params(4096, 1, key, 0, 0, 0))
    part = np.zeros(1024, dtype=np.uint64)
    lib.setup_at(np.ascontiguousarray(payload[3072:]), part, key, 3072)
    assert np.array_equal(part, whole[3072:]), "chunked encode must use global positions"


def test_filler_is_deterministic_and_bounded():
    a = filler_words(0xABC, 1000, 5000)
    b = filler_words(0xABC, 1000, 5000)
    assert np.array_equal(a, b) and int(a.max()) < (1 << 60)
    assert np.array_equal(filler_words(0xABC, 0, 6000)[1000:], a)  # position-indexed, not stream-indexed
    assert not np.array_equal(filler_words(0xABD, 1000, 5000), a)


def test_commitment_detects_tampering():
    theta = rng.integers(0, 1 << 61, 4096, dtype=np.uint64)
    key = b"k" * 16
    leaves = block_leaves(theta, 4096, key)
    root = merkle_root(leaves)
    assert leaves.shape == (8, 32)
    t2 = theta.copy()
    t2[2049] ^= np.uint64(1)
    l2 = block_leaves(t2, 4096, key)
    # word 2049 lies in block 4 (512 words/block)
    assert np.array_equal(l2[:4], leaves[:4]) and not np.array_equal(l2[4], leaves[4])
    assert merkle_root(l2) != root
    assert merkle_root(block_leaves(theta, 4096, b"j" * 16)) != root


def _small_weights(path: Path, nbytes: int = 300_000) -> Path:
    w = {
        "layer.0.weight": rng.standard_normal((nbytes // 4 // 2,)).astype(np.float32),
        "layer.0.bias": rng.standard_normal((nbytes // 4 - nbytes // 4 // 2,)).astype(np.float32),
    }
    np.savez(path, **w)
    return Path(str(path) + ".npz") if not str(path).endswith(".npz") else path


def test_setup_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    wpath = _small_weights(tmp / "w.npz")
    block = 16384
    manifest = run_setup(
        wpath,
        tmp / "prover",
        [0, 1],
        filler_seed=0x5EED,
        block_bytes=block,
        hbm_bytes=4 << 20,
        accounted_bytes=1 << 20,
        device=None,
        threads=4,
        r=16,
        k=1024,
    )
    assert manifest.accounted_bytes == 1 << 20 and manifest.num_blocks == 64 and len(manifest.gpus) == 2
    assert (
        manifest.payload_words == packed_words_for(manifest.raw_bytes)
        and manifest.payload_words < manifest.accounted_words
    )
    m2 = Manifest.load(tmp / "prover" / "manifest.json")
    assert m2.raw_hash == manifest.raw_hash and [g.enc_seed for g in m2.gpus] == [g.enc_seed for g in manifest.gpus]
    # the two GPUs' arrays differ (per-GPU seed) but decode to the same payload
    t0 = np.asarray(load_array(manifest.gpus[0].theta_file))
    t1 = np.asarray(load_array(manifest.gpus[1].theta_file))
    assert not np.array_equal(t0, t1)
    p0, p1 = (
        decode_array(t0, manifest.gpus[0].enc_seed, threads=4),
        decode_array(t1, manifest.gpus[1].enc_seed, threads=4),
    )
    assert np.array_equal(p0, p1)
    raw, _ = flatten(load_tensors(wpath))
    assert np.array_equal(unpack60(p0[: manifest.payload_words], manifest.raw_bytes), raw)
    src = {n: a for n, a, _ in load_tensors(wpath)}
    for e in manifest.layout:  # per-tensor decode straight from Θ'
        assert np.array_equal(decode_tensor(t1, e, manifest.gpus[1].enc_seed), src[e.name]), e.name
    assert all(g.enc_seed >> 64 for g in manifest.gpus), "keys should be 128-bit"
    assert np.array_equal(
        p0[manifest.payload_words :],
        filler_words(manifest.filler_seed, manifest.payload_words, manifest.accounted_words - manifest.payload_words),
    )
    # verifier recomputes and agrees
    checks = verify_setup(m2, wpath, tmp / "verifier", device=None, threads=4, filler_seed=0x5EED)
    assert all(c.root_ok and c.table_ok for c in checks)
    assert np.array_equal(np.asarray(load_array(tmp / "verifier" / "theta_gpu1.npy")), t1)
    # the verifier makes its own image from the recomputed array; the prover's copy only has to match it
    assert all(c.image_ok is True for c in checks)
    assert np.array_equal(
        np.asarray(load_array(tmp / "verifier" / "image_gpu1.npy")), np.asarray(load_array(manifest.gpus[1].image_file))
    )
    np.save(
        manifest.gpus[1].image_file,
        np.zeros_like(np.asarray(load_array(manifest.gpus[1].image_file))),
        allow_pickle=False,
    )
    checks_z = verify_setup(
        m2, wpath, tmp / "verifier_z", device=None, threads=4, keep_arrays=False, filler_seed=0x5EED
    )
    assert checks_z[1].image_ok is False and checks_z[0].image_ok is True, "a zeroed prover image passed"
    assert np.any(np.asarray(load_array(tmp / "verifier_z" / "image_gpu1.npy"))), (
        "the verifier's image must come from the array"
    )
    # a tampered commitment table is caught, with the block named
    leaves = np.load(manifest.gpus[0].commit_file)
    leaves[5, 0] ^= 1
    np.save(manifest.gpus[0].commit_file, leaves)
    checks = verify_setup(m2, wpath, tmp / "verifier2", device=None, threads=4, keep_arrays=False, filler_seed=0x5EED)
    assert not checks[0].table_ok and checks[0].first_bad_block == 5 and checks[0].root_ok
    # wrong weights are rejected outright
    _small_weights(tmp / "other.npz")
    try:
        verify_setup(m2, tmp / "other.npz", tmp / "verifier3", threads=4, filler_seed=0x5EED)
        raise AssertionError("accepted wrong weights")
    except ValueError:
        pass
    # the real array works with the challenge machinery (non power-of-two length, perm off)
    # prover: chain over Theta' imaging on the fly; verifier: chain over the stored image T with image=False
    from memory_accounting.setup.encode import image_array

    keys = Keys(*manifest.hash_key)
    lib = CpuLib()
    T0 = np.asarray(load_array(manifest.gpus[0].image_file))
    assert T0.shape == t0.shape and not np.array_equal(T0, t0)
    assert np.array_equal(T0, image_array(t0, threads=4))
    assert np.array_equal(T0.tolist(), R.image_array(t0.tolist()))
    ch = manifest.challenge(5, 99, 3)
    prover_cfg, verifier_cfg = manifest.memory_config(0, image=True), manifest.memory_config(0, image=False)
    out_p = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(t0, pack_params(prover_cfg, ch, keys, MODE_HONEST), out_p)
    out_v = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(T0, pack_params(verifier_cfg, ch, keys, MODE_HONEST), out_v)
    assert out_p.tolist() == out_v.tolist(), "prover over Theta' (imaged) != verifier over T"
    lo, hi, trace = R.chain(t0.tolist(), pack_params(prover_cfg, ch, keys, MODE_HONEST).view(np.uint64).tolist())
    assert (int(out_p[0]), int(out_p[1])) == (lo, hi) and out_p[2:].tolist() == trace
    # a chain over raw Theta' without the image does NOT match (the image is really applied)
    out_raw = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(t0, pack_params(verifier_cfg, ch, keys, MODE_HONEST), out_raw)
    assert out_raw.tolist() != out_v.tolist()
    # staged single-link continuation (dram_copy) reproduces the imaged chain: positions carried via qoff
    nb = manifest.num_blocks
    d_, idx = (0, 0), ch.j0
    for link in range(ch.n):
        staging = np.ascontiguousarray(t0[idx * ch.be : (idx + 1) * ch.be])
        prm = pack_params(
            prover_cfg, ch, keys, MODE_HONEST, nb=1, d0=d_, link0=link, n=1, j0=0, s0=ch.nonce, qoff=idx * ch.be
        )
        o = np.zeros(3, dtype=np.uint64)
        lib.chain(staging, prm, o)
        d_ = (int(o[0]), int(o[1]))
        idx = d_[0] % nb
    assert d_ == (int(out_v[0]), int(out_v[1])), "staged continuation with image != verifier"
    # the image is not the data: the verifier cannot decode weights from it
    T1 = np.asarray(load_array(manifest.gpus[1].image_file))
    assert not np.array_equal(T0, T1)


def test_verify_rejects_what_the_prover_chose_for_itself():
    """The manifest is the prover's. verify used to accept identical per-GPU keys (one copy answering for
    both GPUs) and a filler seed the prover picked (filler it could prepare in advance); now it takes the
    seed the verifier minted, and rejects repeated keys and a static key that is a dynamic filler key."""
    import copy

    from memory_accounting.prover.dynfill import dyn_key
    from memory_accounting.setup.verify import SetupRejected

    tmp = Path(tempfile.mkdtemp())
    wpath = _small_weights(tmp / "w.npz")
    manifest = run_setup(
        wpath, tmp / "prover", [0, 1], filler_seed=0x5EED, block_bytes=16384, accounted_bytes=1 << 20, threads=4, r=16
    )

    def rejected(man, seed=0x5EED, why=""):
        try:
            verify_setup(man, wpath, tmp / "v", threads=4, keep_arrays=False, filler_seed=seed)
        except SetupRejected as e:
            assert why in str(e), e
            return True
        return False

    assert not rejected(manifest)
    assert rejected(manifest, seed=0x5EEE, why="filler seed"), "a filler seed the verifier did not mint was accepted"
    same = copy.deepcopy(manifest)
    same.gpus[1].enc_seed = same.gpus[0].enc_seed
    assert rejected(same, why="share an encoding key")
    dup = copy.deepcopy(manifest)
    dup.gpus[1].index = 0
    assert rejected(dup, why="repeat")
    # a key of 0 was skipped by the distinct-key check: two GPUs keyed 0 (byte-identical arrays) passed
    zero = copy.deepcopy(manifest)
    zero.gpus[0].enc_seed = zero.gpus[1].enc_seed = 0
    assert rejected(zero, why="no encoding key"), "two GPUs keyed 0 were accepted"
    one_zero = copy.deepcopy(manifest)
    one_zero.gpus[1].enc_seed = 0
    assert rejected(one_zero, why="no encoding key")
    none = copy.deepcopy(manifest)
    none.gpus = []
    assert rejected(none, why="no GPU entries")
    bundle = copy.deepcopy(manifest)
    bundle.kind = "bundle"
    for g in bundle.gpus:
        g.enc_seed = None
    assert rejected(bundle, why="stream-decode bundle"), (
        "verify recomputed nothing for a manifest calling itself a bundle"
    )
    for seeds in ([0, 5], [5, 5]):
        try:
            run_setup(
                wpath,
                tmp / "p0",
                [0, 1],
                filler_seed=0x5EED,
                block_bytes=16384,
                accounted_bytes=1 << 20,
                threads=4,
                enc_seeds=seeds,
            )
            raise AssertionError(f"the prover accepted keys {seeds}")
        except ValueError:
            pass
    try:
        run_setup(
            wpath,
            tmp / "p2",
            [0, 1],
            filler_seed=0x5EED,
            block_bytes=16384,
            accounted_bytes=1 << 20,
            threads=4,
            enc_seeds=[7, 7],
        )
        raise AssertionError("the prover accepted one key for two GPUs")
    except ValueError:
        pass
    m3 = run_setup(
        wpath,
        tmp / "p3",
        [0, 1],
        filler_seed=0x5EED,
        block_bytes=16384,
        accounted_bytes=1 << 20,
        threads=4,
        enc_seeds=[dyn_key(0x5EED, 1), 12345 << 64 | 1],
    )
    assert rejected(m3, why="dynamic filler key of GPU 1")


def test_scratch_is_never_negative():
    """Without --hbm-bytes on a CPU setup the capacity was taken as the packed weights, so the array (the
    weights in whole blocks) came out larger than HBM and the scratch printed negative."""
    tmp = Path(tempfile.mkdtemp())
    wpath = _small_weights(tmp / "w.npz")
    m = run_setup(wpath, tmp / "a", [0], filler_seed=1, block_bytes=16384, threads=2)
    assert m.scratch_bytes == 0 and m.hbm_bytes == m.accounted_bytes and m.accounted_words * 8 >= m.payload_words * 8
    m = run_setup(wpath, tmp / "b", [0], filler_seed=1, block_bytes=16384, threads=2, accounted_bytes=1 << 20)
    assert m.scratch_bytes == 0 and m.accounted_bytes == 1 << 20
    m = run_setup(wpath, tmp / "c", [0], filler_seed=1, block_bytes=16384, threads=2, hbm_bytes=4 << 20)
    assert m.scratch_bytes > 0 and m.accounted_bytes + m.scratch_bytes == 4 << 20
    for kw in ({"hbm_bytes": 1 << 17}, {"hbm_bytes": 1 << 20, "accounted_bytes": 2 << 20}):
        try:
            run_setup(wpath, tmp / "x", [0], filler_seed=1, block_bytes=16384, threads=2, **kw)
            raise AssertionError(f"accepted {kw}")
        except ValueError as e:
            assert "does not fit" in str(e)


def _torch_free(code: str, cwd) -> str:
    """Run code in a fresh interpreter where `import torch` fails (the verifier box)."""
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    pre = f"import sys; sys.modules['torch'] = None; sys.path.insert(0, {str(root)!r})\n"
    r = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_verifier_setup_needs_no_torch():
    """filler-seed, verify and check-bundle on a host without torch: the CLI used to import torch through
    `memory_accounting.experiments.bench` for parse_bytes, and safetensors weights were read through torch."""
    import torch
    from safetensors.torch import save_file

    tmp = Path(tempfile.mkdtemp())
    g = torch.Generator().manual_seed(9)
    weights = {
        "a.bf16": torch.randn(33, 7, generator=g).to(torch.bfloat16),
        "b.f32": torch.randn(100, generator=g),
        "c.i8": torch.randint(-128, 127, (13, 3), dtype=torch.int8, generator=g),
        "d.f16": torch.randn(5, 4, generator=g).half(),
        "e.i64": torch.arange(9),
        "f.bool": torch.tensor([True, False, True]),
    }
    save_file(weights, str(tmp / "w.safetensors"))
    # the direct reader gives what the torch reader gave: names, dtype names, bytes (so model hashes are unchanged)
    got = {n: (a, dt) for n, a, dt in load_tensors(tmp / "w.safetensors")}
    for n, t in weights.items():
        a, dt = got[n]
        assert dt == ("bfloat16" if t.dtype == torch.bfloat16 else str(t.numpy().dtype)), (n, dt)
        ref = t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy()
        assert a.shape == ref.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(ref).tobytes(), n
    out = _torch_free("from memory_accounting.setup.cli import main; main(['filler-seed'])", tmp)
    seed = out.strip()
    assert len(seed) == 16 and int(seed, 16) >= 0
    _torch_free(
        f"from memory_accounting.setup.cli import main; main(['prover', '--weights', 'w.safetensors', '--out', 'p', '--gpus', '0,1', '--filler-seed', {seed!r}, "
        f"'--block-bytes', '16K', '--accounted-bytes', '256K', '--threads', '2', '--r', '16'])",
        tmp,
    )
    out = _torch_free(
        f"from memory_accounting.setup.cli import main\ntry:\n    main(['verify', '--allow-unregistered', '--manifest', 'p/manifest.json', '--weights', 'w.safetensors', '--out', 'v', "
        f"'--filler-seed', {seed!r}, '--threads', '2'])\nexcept SystemExit as e:\n    assert e.code == 0, e.code",
        tmp,
    )
    assert "commitments verified" in out, out
    out = _torch_free(
        "from memory_accounting.setup.cli import main\ntry:\n    main(['verify', '--allow-unregistered', '--manifest', 'p/manifest.json', '--weights', 'w.safetensors', '--out', 'v2', "
        "'--filler-seed', '1', '--threads', '2'])\nexcept SystemExit as e:\n    assert e.code == 1, e.code",
        tmp,
    )
    assert "REJECTED" in out and "filler seed" in out, out


def test_bin_weights_are_a_torch_state_dict():
    """`.bin` weights (transformers' pytorch_model.bin) were read as one raw uint8 blob named after the file,
    not as the state dict they are; they now load like .pt, and without torch the error says so."""
    import torch

    tmp = Path(tempfile.mkdtemp())
    g = torch.Generator().manual_seed(4)
    sd = {"model.w": torch.randn(6, 5, generator=g), "model.b": torch.randn(5, generator=g).to(torch.bfloat16)}
    torch.save(sd, tmp / "pytorch_model.bin")
    torch.save(sd, tmp / "w.pt")
    got = load_tensors(tmp / "pytorch_model.bin")
    assert [n for n, _, _ in got] == ["model.b", "model.w"] and [dt for _, _, dt in got] == ["bfloat16", "float32"]
    assert [(n, a.tobytes(), dt) for n, a, dt in got] == [
        (n, a.tobytes(), dt) for n, a, dt in load_tensors(tmp / "w.pt")
    ]
    (tmp / "blob.bin").write_bytes(b"not a pickle" * 10)
    try:
        load_tensors(tmp / "blob.bin")
        raise AssertionError("a raw blob was read as weights")
    except ValueError as e:
        assert "not a torch state dict" in str(e)
    out = _torch_free(
        "from memory_accounting.setup.weights import load_tensors\ntry:\n    load_tensors('pytorch_model.bin')\nexcept ValueError as e:\n    print(e)",
        tmp,
    )
    assert "needs torch" in out, out


def test_segmented_array_matches_contiguous():
    """The honest array may be scattered tensors + a filler allocation; the kernels read it through a
    segment table [starts..., total, ptrs...]. On the CPU the pointers are host addresses."""
    parts = [np.ascontiguousarray(rng.integers(0, 1 << 61, n, dtype=np.uint64)) for n in (300, 1000, 748, 2048)]
    total = sum(len(p) for p in parts)  # 4096 words = 32 KB
    starts = np.cumsum([0] + [len(p) for p in parts]).astype(np.uint64)
    ptrs = np.array([p.ctypes.data for p in parts], dtype=np.uint64)
    segtab = np.concatenate([starts, ptrs])
    contiguous = np.concatenate(parts)
    cfg = MemoryConfig(words=total, encoding="vde", perm=False)
    keys = Keys.deterministic(8)
    lib = CpuLib()
    for r in (0, 16):
        ch = Challenge(block_bytes=2048, n=6, r=r, nonce=5, j0=1, k=128)
        a = np.zeros(2 + ch.n, dtype=np.uint64)
        b = np.zeros_like(a)
        lib.chain(contiguous, pack_params(cfg, ch, keys, MODE_HONEST), a)
        lib.chain(
            np.zeros(1, dtype=np.uint64),
            pack_params(cfg, ch, keys, MODE_HONEST, nseg=len(parts), segtab=segtab.ctypes.data),
            b,
        )
        assert a.tolist() == b.tolist(), f"segmented != contiguous at r={r}"


def test_merkle_root_separates_leaves_from_nodes_and_binds_the_count():
    """The tree hashed inner nodes as plain BLAKE2b(left || right) and promoted an odd node, with nothing
    binding the leaf count: the leaves (a, b, c) and the 'leaves' (H(a || b), c) had one root, and a one-leaf
    tree's root was the leaf itself. Leaves and nodes now hash in their own domains and the root binds the count."""
    import hashlib

    from memory_accounting.setup.commit import NODE_TAG

    theta = rng.integers(0, 1 << 61, 3 * 512, dtype=np.uint64)
    leaves = block_leaves(theta, 4096, b"k" * 16)
    leaf0, leaf1, leaf2 = (bytes(x) for x in leaves)

    def as_leaves(*xs):
        return np.stack([np.frombuffer(x, np.uint8) for x in xs])

    root = merkle_root(leaves)
    assert root == merkle_root(as_leaves(leaf0, leaf1, leaf2))
    for node in (
        hashlib.blake2b(leaf0 + leaf1, digest_size=32).digest(),
        hashlib.blake2b(leaf0 + leaf1, digest_size=32, person=NODE_TAG).digest(),
    ):
        assert merkle_root(as_leaves(node, leaf2)) != root, "a tree of inner nodes taken as leaves has the same root"
    assert merkle_root(leaves[:1]) != leaf0, "a one-leaf root is the leaf itself"
    assert (
        merkle_root(leaves[:2]) != merkle_root(leaves[:2][::-1])
        and len({merkle_root(leaves[:i]) for i in range(4)}) == 4
    )
    assert leaves[0].tobytes() != hashlib.blake2b(theta[:512].tobytes(), key=b"k" * 16, digest_size=32).digest(), (
        "leaves are not in their own domain"
    )


def test_manifest_fields_and_files_are_checked_on_load():
    """The manifest is the prover's, and the verifier read it unchecked: a block of 0 bytes raised
    ZeroDivisionError, a GPU index could be a string that went into file names, a key could be negative or
    written '0x…' (aliasing a real key while the collision checks compared the spelled value), and the verifier
    opened whatever paths the manifest named. Load now checks types and ranges, keys are canonical, and the
    files must lie in the manifest's directory (written relative to it, so a moved directory still loads)."""
    import copy
    import shutil

    from memory_accounting.prover.dynfill import dyn_key
    from memory_accounting.setup.verify import manifest_problems

    tmp = Path(tempfile.mkdtemp())
    wpath = _small_weights(tmp / "w.npz")
    run_setup(wpath, tmp / "p", [0, 1], filler_seed=0x5EED, block_bytes=16384, accounted_bytes=1 << 20, threads=2, r=16)
    raw = json.loads((tmp / "p" / "manifest.json").read_text())
    assert [g["image_file"] for g in raw["gpus"]] == ["image_gpu0.npy", "image_gpu1.npy"], raw["gpus"]
    shutil.move(str(tmp / "p"), str(tmp / "moved"))
    manifest = Manifest.load(tmp / "moved" / "manifest.json")
    assert all(
        Path(f).is_absolute() and Path(f).exists()
        for g in manifest.gpus
        for f in (g.theta_file, g.commit_file, g.image_file)
    )
    assert manifest_problems(manifest, 0x5EED) == []
    alias = dyn_key(0x5EED, 1) - (1 << 128)  # the dynamic filler key of GPU 1, written negative

    def refused(why, change):
        bad = copy.deepcopy(raw)
        change(bad)
        (tmp / "moved" / "bad.json").write_text(json.dumps(bad))
        try:
            Manifest.load(tmp / "moved" / "bad.json")
        except ValueError as e:
            assert why in str(e), (why, str(e))
            return True
        return False

    def gpu0(**kw):
        return lambda b: b["gpus"][0].update(kw)

    assert not refused("", lambda b: None)
    assert refused("block_bytes", lambda b: b.update(block_bytes=0)), "a block of 0 bytes was accepted"
    assert refused("block_bytes", lambda b: b.update(block_bytes=100))
    assert refused("blocks", lambda b: b.update(accounted_words=b["accounted_words"] - 8))
    assert refused("GPU index", gpu0(index="0")), "a string GPU index was accepted"
    assert refused("GPU index", gpu0(index=-1))
    assert refused("raw_bytes", lambda b: b.update(raw_bytes="300000"))
    assert refused("hash_key", lambda b: b.update(hash_key=[1, 2]))
    for key in (-5, 1 << 128, "12", True):
        assert refused("encoding key", gpu0(enc_seed=key)), key
    for fk in (f"-{-alias:x}", "0x" + "ab" * 16, "AB" * 16, " ab", "ab_cd", "0", "1" * 33):
        assert refused("filler_key", lambda b, fk=fk: b.update(filler_key=fk)), fk
    for path in ("/etc/passwd", "../w.npz", str(tmp / "w.npz")):
        assert refused("outside", gpu0(image_file=path)), path
        assert refused("outside", gpu0(commit_file=path)), path
    (tmp / "moved" / "link.npy").symlink_to(tmp / "w.npz")
    assert refused("outside", gpu0(theta_file="link.npy")), "a link out of the directory was followed"
    assert refused("version 2", lambda b: b.update(version=2))
    # a manifest made in memory: manifest_problems reports what load refuses, before comparing any key
    neg = copy.deepcopy(manifest)
    neg.filler_key = f"-{-alias:x}"
    assert any("filler_key" in p for p in manifest_problems(neg, 0x5EED)), (
        "a negative filler key aliasing a dynamic key passed"
    )
    try:
        (tmp / "other").mkdir()
        manifest.save(tmp / "other" / "manifest.json")
        raise AssertionError("a manifest was saved with its files outside its directory")
    except ValueError as e:
        assert "outside" in str(e)


def test_hash_keys_come_from_the_verifiers_seed():
    """The challenge-hash keys are derived by both sides from the verifier's filler seed (`Keys.from_seed`); they
    used to be the prover's choice, carried by the manifest. The verifier never reads them from a manifest: a
    manifest whose keys are not the derived ones is refused, and the keys a command uses are the derived ones
    whatever the manifest says."""
    import copy
    from types import SimpleNamespace

    from memory_accounting.core.params import PM61, array_from_args
    from memory_accounting.prover.dynfill import dyn_key
    from memory_accounting.setup.filler import static_filler_key
    from memory_accounting.setup.verify import manifest_problems
    from memory_accounting.verifier.consistency import checker_keys

    a, b = Keys.from_seed(0x5EED), Keys.from_seed(0x5EED)
    assert a == b and a != Keys.from_seed(0x5EEE) and a != Keys.from_seed(0)
    for seed in (0, 1, 0x5EED, (1 << 64) - 1):
        k = Keys.from_seed(seed)
        assert 0 <= k.k0 < 1 << 64 and 0 <= k.k1 < 1 << 64 and 0 < k.polyk < PM61, seed
        assert k.as_list() == [k.k0, k.k1, k.polyk]
        assert (k.k1 << 64 | k.k0) not in (static_filler_key(seed, 0), dyn_key(seed, 0)), (
            "the hash keys share a domain with a filler key"
        )
    for bad in (-1, 1 << 64):
        try:
            Keys.from_seed(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    tmp = Path(tempfile.mkdtemp())
    wpath = _small_weights(tmp / "w.npz")
    m = run_setup(
        wpath, tmp / "p", [0], filler_seed=0x5EED, block_bytes=16384, accounted_bytes=1 << 20, threads=2, r=16
    )
    assert m.hash_key == Keys.from_seed(0x5EED).as_list(), "the setup's hash keys are not the derived ones"
    assert manifest_problems(m, 0x5EED) == []
    own = copy.deepcopy(m)
    own.hash_key = Keys.deterministic(1).as_list()  # a prover that chose its own (as before)
    assert any("challenge-hash keys" in p for p in manifest_problems(own, 0x5EED)), (
        "a manifest with the prover's own hash keys passed"
    )
    assert any("challenge-hash keys" in p for p in manifest_problems(m, 0x5EEE)), (
        "keys derived from another seed passed"
    )
    own.save(tmp / "p" / "manifest.json")  # the keys in use never come from the manifest
    # the verifier's seed; without one, the manifest's
    for fs, want in (
        ("5eed", 0x5EED),
        ("5eee", 0x5EEE),
        (None, 0x5EED),
    ):
        _spec, keys = array_from_args(SimpleNamespace(manifest=str(tmp / "p" / "manifest.json"), filler_seed=fs))
        assert keys == Keys.from_seed(want), fs
    # (a command without the flag)
    assert array_from_args(SimpleNamespace(manifest=str(tmp / "p" / "manifest.json")))[1] == Keys.from_seed(0x5EED)
    # the record checkers: the filler seed, else the pad seed, else the manifest's seed; --hash-seed overrides
    # (synthetic provers, tests)
    assert checker_keys("5eed", "abc", 7, None) == Keys.from_seed(0x5EED)
    assert checker_keys(None, "abc", 7, None) == Keys.from_seed(0xABC)
    assert checker_keys(None, None, 7, None) == Keys.from_seed(7)
    assert checker_keys("5eed", "abc", 7, 3) == Keys.deterministic(3)
    assert checker_keys(None, None, None, None) == Keys.deterministic(1)


if __name__ == "__main__":
    from _runner import run

    run(globals())
