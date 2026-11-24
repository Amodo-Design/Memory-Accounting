"""Stream-decode conversion on a toy transformer on the CPU: same outputs, weights resident only as Θ'."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from memory_accounting.prover.stream_decode import convert
from memory_accounting.setup.encode import decode_tensor
from memory_accounting.setup.layout import TensorEntry


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = torch.nn.Linear(d, d, bias=False)
        self.mlp = torch.nn.Linear(d, 4 * d)
        self.proj = torch.nn.Linear(4 * d, d)
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, x):
        return x + self.proj(torch.relu(self.mlp(self.norm(x + self.attn(x)))))


class Toy(torch.nn.Module):
    def __init__(self, vocab=97, d=64, n=4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def test_stream_decode_preserves_outputs():
    torch.manual_seed(0)
    model = Toy().to(torch.float32)
    ids = torch.randint(0, 97, (3, 11))
    with torch.no_grad():
        ref = model(ids).clone()
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    sd = convert(model, enc_key=(0xABCDEF << 64) | 0x123456789)
    assert set(sd.units) == {"rest", "layers.0", "layers.1", "layers.2", "layers.3"}
    # weights now live only in Θ' plus one layer of scratch: embedding and head are decoded into it when used
    layer_bytes = sd.units["layers.0"].scratch_bytes
    assert sd.scratch.numel() == layer_bytes and sd.resident is None and sd.plain_bytes == layer_bytes
    padded = sum(sl.groups * 120 for u in sd.units.values() for sl in u.slots)  # slots padded to 3840-byte multiples
    assert padded >= total_bytes and sd.theta.numel() * 8 == padded * 16 // 15  # 60-bit packing: 15 raw words -> 16
    for unit in sd.units.values():
        for sl in unit.slots:
            assert sl.scratch_offset % 256 == 0 and sl.payload_offset % 512 == 0 and (sl.view_offset or 0) % 256 == 0, (
                sl.name
            )
    for n, p in model.named_parameters():  # every parameter is a view into the scratch
        assert sd.scratch.data_ptr() <= p.data_ptr() < sd.scratch.data_ptr() + sd.scratch.numel(), n
    # forward: hooks decode each layer in turn; outputs identical
    with torch.no_grad():
        out = model(ids)
    assert torch.equal(out, ref), "stream-decoded forward differs from the original"
    with torch.no_grad():
        out2 = model(ids)
    assert torch.equal(out2, ref)
    # every tensor decodes from Θ' to its original bytes via the setup-phase decoder too
    theta_np = sd.theta.numpy().view(np.uint64)
    for unit in sd.units.values():
        for sl in unit.slots:
            e = TensorEntry(
                sl.name,
                str(sl.dtype).replace("torch.", ""),
                list(sl.shape),
                sl.nbytes,
                0,
                sl.payload_offset,
                sl.groups * 16,
            )
            got = decode_tensor(theta_np, e, sd.enc_key)
            assert np.array_equal(
                got.view(np.uint8).ravel(), originals[sl.name].contiguous().view(torch.uint8).numpy().ravel()
            ), sl.name
    # the challenged array is one contiguous allocation
    ((ptr, words, _),) = sd.segments()
    assert ptr == sd.theta.data_ptr() and words == sd.theta.numel()
    assert sd.check_parameters(originals) == []


class TwoStack(torch.nn.Module):
    """An encoder-decoder shape: two stacks whose layers share indices."""

    def __init__(self, vocab=97, d=32, n=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.encoder.layers:
            x = layer(x)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.head(x)


def test_stacks_sharing_layer_indices_stay_distinct():
    """encoder.layers.0 and decoder.layers.0 used to be one unit with one hook (the last module found), so
    the encoder ran on the decoder's weights: outputs off by up to 2.0 and check_parameters silent."""
    torch.manual_seed(0)
    model = TwoStack()
    ids = torch.randint(0, 97, (2, 7))
    with torch.no_grad():
        ref = model(ids).clone()
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    sd = convert(model, enc_key=0x77)
    assert set(sd.units) == {"rest", "encoder.layers.0", "encoder.layers.1", "decoder.layers.0", "decoder.layers.1"}
    assert [k for k in sd.units] == [
        "rest",
        "encoder.layers.0",
        "encoder.layers.1",
        "decoder.layers.0",
        "decoder.layers.1",
    ]
    with torch.no_grad():
        assert torch.equal(model(ids), ref), "a stack ran on another stack's weights"
    assert sd.check_parameters(originals) == []
    # what decoding would never reach is reported: a parameter in no unit, a unit decoded by a module it is not in
    model.extra = torch.nn.Parameter(torch.ones(3))
    assert any(b.startswith("extra (in no decode unit)") for b in sd.check_parameters(originals))
    del model.extra
    u = sd.units["encoder.layers.0"]
    mod = u.module
    u.module = model.decoder.layers[0]
    assert all(
        "outside decoder.layers.0" in b for b in sd.check_parameters(originals) if b.startswith("encoder.layers.0.")
    )
    assert any(b.startswith("encoder.layers.0.") for b in sd.check_parameters(originals))
    u.module = mod
    # parameters of a layer whose module the module pattern does not find: refused, not silently undecoded
    torch.manual_seed(0)
    try:
        convert(TwoStack(), enc_key=0x77, module_re=r"encoder\.layers\.(\d+)$")
        raise AssertionError("accepted a unit without a hook")
    except ValueError as e:
        assert "decoder.layers" in str(e)


def test_rest_is_decoded_into_the_scratch_when_used():
    """The parameters outside the layers were kept decoded for good, whatever their size (the review: "kept in
    plaintext permanently, with no size limit"). The embedding is now decoded into the scratch before the layers
    and the head after them, over the same bytes, so the scratch is the larger of a layer and either, not their
    sum; a head tied to the embedding is decoded at both. Outputs stay identical, call after call."""
    torch.manual_seed(0)
    model = Toy(vocab=4000, d=32, n=2)  # embedding and head (4000 x 32 float32, 500 KiB) larger than a layer
    ids = torch.randint(0, 4000, (2, 9))
    with torch.no_grad():
        ref = model(ids).clone()
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    sd = convert(model, enc_key=0x55)
    assert sd.resident is None and sd.plain_bytes == sd.scratch.numel()
    assert [sl.name for sl in sd.rest_groups["before"]] == ["embed.weight"]
    assert [sl.name for sl in sd.rest_groups["after"]] == ["head.weight", "head.bias"]
    emb = sd.rest_groups["before"][0].groups * 120
    head = sum(sl.groups * 120 for sl in sd.rest_groups["after"])
    assert sd.scratch.numel() == max(emb, head) > sd.units["layers.0"].scratch_bytes and sd.scratch.numel() < emb + head
    with torch.no_grad():
        for _ in range(3):
            assert torch.equal(model(ids), ref)
    assert sd.check_parameters(originals) == []

    torch.manual_seed(0)
    tied = Toy(vocab=4000, d=32, n=2)
    tied.head.weight = tied.embed.weight
    with torch.no_grad():
        ref = tied(ids).clone()
    originals = {n: p.detach().clone() for n, p in tied.named_parameters()}
    sd = convert(tied, enc_key=0x55)
    assert [sl.name for sl in sd.rest_groups["before"]] == ["embed.weight"]
    assert [sl.name for sl in sd.rest_groups["after"]] == ["embed.weight", "head.bias"]
    assert sd.rest_groups["before"][0].view_offset == 0  # one place for the tied weight, used at both phases
    with torch.no_grad():
        for _ in range(3):
            assert torch.equal(tied(ids), ref), "the tied head ran on what the layers left in the scratch"
    assert sd.check_parameters(originals) == []


class Bridged(torch.nn.Module):
    """Two stacks with a projector between them, and parameters of the model's own (one used, one large and not)."""

    def __init__(self, vocab=97, d=32, n=2, spare=0):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.bridge = torch.nn.Linear(d, d)
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.head = torch.nn.Linear(d, vocab)
        self.scale = torch.nn.Parameter(torch.rand(d) + 0.5)
        if spare:
            self.spare = torch.nn.Parameter(torch.zeros(spare))

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.encoder.layers:
            x = layer(x)
        x = self.bridge(x)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.head(x * self.scale)


def test_rest_without_a_phase_stays_resident_within_a_bound():
    """A parameter outside the layers that no module before or after them holds (a projector between two stacks,
    the model's own parameter) has no phase: it stays decoded, at most one layer's scratch of them; past that the
    conversion is refused, unless resident= names them or max_resident_bytes= allows them."""
    torch.manual_seed(0)
    model = Bridged()
    ids = torch.randint(0, 97, (2, 7))
    with torch.no_grad():
        ref = model(ids).clone()
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    sd = convert(model, enc_key=0x77)
    assert {sl.name for sl in sd.resident_slots} == {"bridge.weight", "bridge.bias", "scale"}
    assert sd.plain_bytes == sd.scratch.numel() + sd.resident.numel()
    with torch.no_grad():
        assert torch.equal(model(ids), ref) and torch.equal(model(ids), ref)
    assert sd.check_parameters(originals) == []
    big = 1 << 16  # 256 KiB of float32, over one layer's scratch here
    try:
        convert(Bridged(spare=big), enc_key=0x77)
        raise AssertionError("kept a large parameter decoded")
    except ValueError as e:
        assert "spare" in str(e) and "stream_phases" in str(e), e
    assert {sl.name for sl in convert(Bridged(spare=big), enc_key=0x77, resident=r"^spare$").resident_slots} >= {
        "spare"
    }
    assert convert(Bridged(spare=big), enc_key=0x77, max_resident_bytes=1 << 20).resident.numel() >= 4 * big


class TLinear(torch.nn.Module):
    """A layer whose weight is stored transposed, as vLLM's fp8 linear keeps `w.t()` for its kernel; the
    kernel here insists on that layout."""

    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(d, 2 * d).t())  # shape (2d, d), stride (1, 2d)
        self.b = torch.nn.Parameter(torch.randn(2 * d))

    def forward(self, x):
        assert self.w.stride() == (1, self.w.shape[0]), "the weight lost its layout"
        return torch.tanh(x @ self.w.t() + self.b)


class Transposed(torch.nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([TLinear(d) for _ in range(n)])
        self.out = torch.nn.Parameter(torch.randn(2 * d, d))

    def forward(self, x):
        for m in self.layers:
            x = m(x) @ self.out
        return x


def test_parameters_keep_their_memory_layout():
    """convert used to make every parameter contiguous, which breaks a kernel that relies on a parameter's
    strides. Θ' now holds a parameter's elements in memory order and its scratch view has its strides."""
    from memory_accounting.setup.encode import decode_tensor

    torch.manual_seed(2)
    model = Transposed()
    x = torch.randn(3, 16)
    with torch.no_grad():
        ref = model(x).clone()
    strides = {n: p.stride() for n, p in model.named_parameters()}
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    sd = convert(model, enc_key=0x55)
    assert {n: p.stride() for n, p in model.named_parameters()} == strides
    with torch.no_grad():
        assert torch.equal(model(x), ref)
    assert sd.check_parameters(originals) == []
    # the bundle's layout describes the bytes: shape in memory order
    sl = next(sl for u in sd.units.values() for sl in u.slots if sl.name == "layers.0.w")
    assert sl.memory_shape == [16, 32] and list(sl.shape) == [32, 16]
    e = TensorEntry(sl.name, "float32", sl.memory_shape, sl.nbytes, 0, sl.payload_offset, sl.groups * 16)
    got = decode_tensor(sd.theta.numpy().view(np.uint64), e, sd.enc_key)
    assert np.array_equal(got, originals["layers.0.w"].t().contiguous().numpy())
    # a parameter that is not dense (a slice with gaps) cannot keep its layout over a slot: refused
    bad = Transposed()
    bad.layers[0].w = torch.nn.Parameter(torch.randn(32, 32)[:, ::2])
    try:
        convert(bad, enc_key=0x55)
        raise AssertionError("accepted a non-dense parameter")
    except ValueError as err:
        assert "not dense" in str(err)


def test_bundle_matches_prover_chain():
    """The verifier's image (from export_bundle) reproduces the chain the prover computes over resident Θ'."""
    import tempfile
    from pathlib import Path

    from memory_accounting.core.backend import CpuLib
    from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params
    from memory_accounting.setup.manifest import Manifest

    torch.manual_seed(1)
    model = Toy(d=128, n=2)
    sd = convert(model, enc_key=0x1234)
    keys = Keys.deterministic(1)
    d = Path(tempfile.mkdtemp())
    block = 16384
    sd.export_bundle(d, block_bytes=block, r=16, k=1024, hbm_bytes=1 << 30)
    manifest = Manifest.load(d / "manifest.json")
    # no weight key (withheld, not 0), and no filler here
    assert manifest.is_bundle() and manifest.gpus[0].enc_seed is None and manifest.filler_key == ""
    T = np.load(manifest.gpus[0].image_file).view(np.uint64)
    theta = sd.theta.numpy().view(np.uint64)[: manifest.accounted_words]
    lib = CpuLib()
    prover_cfg = MemoryConfig(words=manifest.accounted_words, encoding="vde", perm=False, seed_t=sd.enc_key, image=True)
    verifier_cfg = manifest.memory_config(0, image=False)
    for r in (0, 16):
        ch = Challenge(block, 4, r, 4242, 1, 1024)
        a = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(np.ascontiguousarray(theta), pack_params(prover_cfg, ch, keys, MODE_HONEST), a)
        b = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(
            np.ascontiguousarray(T[: manifest.accounted_words]), pack_params(verifier_cfg, ch, keys, MODE_HONEST), b
        )
        assert a.tolist() == b.tolist(), f"r={r}"


def test_bundle_commitment_and_filler_key():
    """A stream-decode bundle carries a commitment the verifier checks its image against, and the filler key both
    sides derive from the verifier's seed and the GPU index, which is not the weights' key. It says nothing about
    the weights (no hash, layout or size: Θ''s size is the image's length), and the verifier needs no copy of
    them to check it. The roots are reproducible by an audit that holds the weights and the key, which the
    verifier does not."""
    import copy
    import tempfile
    from pathlib import Path

    from memory_accounting.core.backend import CpuLib
    from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params
    from memory_accounting.prover.filler import FillerRegion
    from memory_accounting.setup.commit import leaves_of, merkle_root
    from memory_accounting.setup.filler import static_filler_key
    from memory_accounting.setup.manifest import Manifest
    from memory_accounting.setup.verify import bundle_theta_words, verify_bundle
    from memory_accounting.verifier.recompute import ImageWithFiller, Verifier

    torch.manual_seed(3)
    model = Toy(d=64, n=2)
    twin = copy.deepcopy(model)  # the auditor's own copy of the model
    key, seed, block = (0xABC << 64) | 0xDEF, 0x5EED, 16384
    sd = convert(model, enc_key=key)
    fk = static_filler_key(seed, 1)
    assert fk != key and fk != static_filler_key(seed, 0) and fk != static_filler_key(seed + 1, 1)
    lib = CpuLib()
    bw = block // 8
    fw = 3 * bw  # a small filler continuing Θ', under GPU 1's filler key
    pay = np.empty(fw, np.uint64)
    lib.filler(pay, seed, sd.payload_words)
    enc = np.empty(fw, np.uint64)
    lib.setup_at(pay, enc, fk, sd.payload_words)
    fill = FillerRegion(torch.from_numpy(enc.view(np.int64)), fw, fk)
    keys = Keys.deterministic(2)
    tmp = Path(tempfile.mkdtemp())
    sd.export_bundle(tmp, filler=fill, filler_seed=seed, block_bytes=block, r=16, k=1024, hbm_bytes=1 << 30, gpu=1)
    manifest = Manifest.load(tmp / "manifest.json")
    g = manifest.gpu(1)
    assert (
        g.enc_seed is None
        and int(manifest.filler_key, 16) == fk
        and f"{key:032x}" not in (tmp / "manifest.json").read_text()
    )
    assert g.image_root and g.commit_root
    assert (manifest.raw_hash, manifest.layout, manifest.raw_bytes, manifest.payload_words) == ("", [], 0, 0), (
        "the bundle says something about the weights"
    )
    assert bundle_theta_words(manifest, 1) == sd.payload_words  # Θ''s size: the image's length
    assert verify_bundle(manifest, 1, seed) == []
    assert any("filler seed" in p for p in verify_bundle(manifest, 1, seed + 1)), (
        "a filler seed the verifier did not mint was accepted"
    )
    odd = np.load(g.image_file)[:-1]
    np.save(tmp / "odd.npy", odd, allow_pickle=False)
    assert any("16-word groups" in p for p in verify_bundle(manifest, 1, seed, image_path=tmp / "odd.npy")), (
        "an image of part of a group was accepted"
    )
    # the filler key in the manifest is only a record: one that is not the derived key is refused
    m_fk = copy.deepcopy(manifest)
    m_fk.filler_key = f"{static_filler_key(seed, 0):032x}"
    assert any("derived from the verifier's filler seed" in p for p in verify_bundle(m_fk, 1, seed))
    # an image changed after the bundle was made fails its commitment
    T = np.load(g.image_file).view(np.uint64).copy()
    T2 = T.copy()
    T2[5] ^= np.uint64(1)
    np.save(tmp / "forged.npy", T2.view(np.int64), allow_pickle=False)
    assert any("image" in p for p in verify_bundle(manifest, 1, seed, image_path=tmp / "forged.npy"))
    # the audit: with the weights and the key, Θ' and its image are recomputed and both roots reproduced
    audit = convert(copy.deepcopy(twin), enc_key=key)
    ck = bytes.fromhex(manifest.commit_key)
    assert merkle_root(audit.theta_leaves(block, ck)).hex() == g.commit_root
    assert merkle_root(leaves_of(audit.image(), block, ck)).hex() == g.image_root
    # the verifier regenerates the filler from the filler key: the same chain as the prover over Θ' ‖ filler
    resident = np.concatenate([sd.theta.numpy().view(np.uint64), enc])[: manifest.accounted_words]
    ver = Verifier(
        ImageWithFiller(T, manifest.accounted_words, seed, static_filler_key(seed, 1), lib, threads=2),
        manifest.memory_config(1, image=False),
        keys,
        threads=2,
    )
    pcfg = MemoryConfig(words=manifest.accounted_words, encoding="vde", perm=False, seed_t=key, image=True)
    for r in (0, 16):
        ch = Challenge(block, 6, r, 777 + r, 1, 1024)
        a = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(np.ascontiguousarray(resident), pack_params(pcfg, ch, keys, MODE_HONEST), a)
        assert (int(a[0]), int(a[1])) == ver.expected(ch)[:2], f"r={r}"


def _bundle(out_dir, model, key, seed=0x5EED, block=16384, gpu=0, filler_blocks=2, keys=None):
    """A stream-decode bundle with a small filler (encoded under GPU `gpu`'s filler key), as the in-process provers make."""
    from memory_accounting.core.backend import CpuLib
    from memory_accounting.core.params import Keys
    from memory_accounting.prover.filler import FillerRegion
    from memory_accounting.setup.filler import static_filler_key

    keys = keys or Keys.deterministic(2)
    sd = convert(model, enc_key=key)
    lib = CpuLib()
    bw = block // 8
    fw = (
        sd.payload_words + filler_blocks * bw
    ) // bw * bw - sd.payload_words  # Θ' ‖ filler ends on a block, as allocate_filler sizes it
    pay = np.empty(fw, np.uint64)
    lib.filler(pay, seed, sd.payload_words)
    fk = static_filler_key(seed, gpu)
    enc = np.empty(fw, np.uint64)
    lib.setup_at(pay, enc, fk, sd.payload_words)
    fill = FillerRegion(torch.from_numpy(enc.view(np.int64)), fw, fk)
    sd.export_bundle(
        out_dir, filler=fill, filler_seed=seed, block_bytes=block, r=16, k=1024, hbm_bytes=1 << 30, gpu=gpu
    )
    return sd, out_dir / "manifest.json"


def test_check_bundle_requires_distinct_bundles():
    """Two bundles made with one --enc-key (byte-identical images on two GPUs) passed one at a time: check-bundle
    checks several bundles together. It takes no model (--model and --raw-hash are gone): the verifier holds no
    copy of the weights at setup."""
    import copy
    import tempfile
    from pathlib import Path

    from memory_accounting.setup.cli import main
    from memory_accounting.setup.manifest import Manifest

    torch.manual_seed(5)
    model = Toy(d=64, n=2)
    d = Path(tempfile.mkdtemp())
    _, m0 = _bundle(d / "g0", copy.deepcopy(model), key=(7 << 64) | 1, gpu=0)
    _, m1 = _bundle(d / "g1", copy.deepcopy(model), key=(9 << 64) | 3, gpu=1)
    _, same = _bundle(d / "g1same", copy.deepcopy(model), key=(7 << 64) | 1, gpu=1)  # GPU 1 with GPU 0's key
    assert Manifest.load(m0).filler_seed == 0x5EED and Manifest.load(m0).accounted_words * 8 % 16384 == 0

    def code(*argv):
        try:
            main(["check-bundle", "--filler-seed", "5eed", *argv])
            return 0
        except SystemExit as e:
            return e.code

    assert code("--manifest", str(m0)) == 0
    for gone in ("--raw-hash", "--model"):
        # argparse: unknown option
        assert code("--manifest", str(m0), gone, f"{'ab' * 32}:16") == 2, f"check-bundle still takes {gone}"
    assert code("--manifest", str(m0), str(m1)) == 0
    assert code("--manifest", str(m0), str(same)) == 1, "two bundles with one key passed together"
    assert code("--manifest", str(m0), str(m0)) == 1, "one bundle passed for two GPUs"
    from memory_accounting.setup.verify import bundle_set_problems

    probs = bundle_set_problems([(Manifest.load(m0), 0, None), (Manifest.load(same), 1, None)])
    assert any("byte-identical" in p for p in probs), probs


def test_bundle_kind_and_old_bundles():
    """A bundle says it is one (kind "bundle", key None); a setup manifest with a zero key is not a bundle and
    is refused. Manifests of version 2 (their Merkle roots are of the old tree), and so the bundles written
    before manifests had a kind, are refused on load, with a note to set up again; so is one without a kind."""
    import json
    import tempfile
    from pathlib import Path

    from memory_accounting.setup.manifest import Manifest
    from memory_accounting.setup.verify import manifest_problems, verify_bundle

    torch.manual_seed(6)
    model = Toy(d=64, n=2)
    tmp = Path(tempfile.mkdtemp())
    _, mp = _bundle(tmp, model, key=(5 << 64) | 5)
    raw = json.loads(mp.read_text())
    assert raw["kind"] == "bundle" and raw["gpus"][0]["enc_seed"] is None
    assert verify_bundle(Manifest.load(mp), 0, 0x5EED) == []
    old = dict(raw, version=2)
    old.pop("kind")
    old["gpus"] = [dict(raw["gpus"][0], enc_seed=0)]  # as v0.11 wrote it
    (tmp / "old.json").write_text(json.dumps(old))
    try:
        Manifest.load(tmp / "old.json")
        raise AssertionError("a version 2 manifest was loaded")
    except ValueError as e:
        assert "version 2" in str(e) and "again" in str(e), e
    kindless = dict(raw)
    kindless.pop("kind")
    (tmp / "kindless.json").write_text(json.dumps(kindless))
    try:
        Manifest.load(tmp / "kindless.json")
        raise AssertionError("a manifest without a kind was loaded")
    except ValueError as e:
        assert "kind" in str(e)
    # a setup manifest keyed 0 claims to be nothing in particular: it is a setup manifest, and refused
    setup0 = dict(raw, kind="setup")
    setup0["gpus"] = [dict(raw["gpus"][0], enc_seed=0, image_root="")]
    (tmp / "setup0.json").write_text(json.dumps(setup0))
    m0 = Manifest.load(tmp / "setup0.json")
    assert not m0.is_bundle() and any("no encoding key" in p for p in manifest_problems(m0, 0x5EED))
    assert any("not a stream-decode bundle" in p for p in verify_bundle(m0, 0, 0x5EED))
    # a bundle that carries a key, and one with no GPU entries, are refused
    keyed = dict(raw)
    keyed["gpus"] = [dict(raw["gpus"][0], enc_seed=1234)]
    (tmp / "keyed.json").write_text(json.dumps(keyed))
    assert any("carries a weights key" in p for p in verify_bundle(Manifest.load(tmp / "keyed.json"), 0, 0x5EED))
    empty = dict(raw)
    empty["gpus"] = []
    (tmp / "empty.json").write_text(json.dumps(empty))
    assert any("no GPU entries" in p for p in manifest_problems(Manifest.load(tmp / "empty.json"), 0x5EED))


def test_small_bundle_is_refused_and_tail_warned():
    """Without filler the bundle's array was Θ' rounded down to whole blocks, silently, and 0 words (a crash
    later) when Θ' was smaller than one block."""
    import tempfile
    import warnings
    from pathlib import Path

    torch.manual_seed(7)
    sd = convert(Toy(d=32, n=3, vocab=11), enc_key=0x77)
    tmp = Path(tempfile.mkdtemp())
    try:
        sd.export_bundle(tmp, block_bytes=1 << 24, hbm_bytes=1 << 30)
        raise AssertionError("a bundle smaller than a block")
    except ValueError as e:
        assert "less than one" in str(e)
    low = sd.payload_words & -sd.payload_words  # a block that does not divide Θ'
    assert 2 * low <= sd.payload_words
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sd.export_bundle(tmp, block_bytes=16 * low, hbm_bytes=1 << 30)
    assert any(f"last {low} words" in str(x.message) for x in w), [str(x.message) for x in w]


def test_bundle_image_is_thetas_and_the_verifier_regenerates_the_filler():
    """A bundle used to fix its own image length (anywhere from the payload size it declared to the whole array),
    and the verifier regenerated filler only past the image: a prover that exported an image of zeros over the
    whole array, declaring Θ' that large, was accepted, and answered every chain holding nothing. The image must
    now be exactly Θ''s, the size the bundle declares, and the verifier regenerates every filler block past it
    from its own seed under the key it derives for the GPU. The bundle's Θ' region is not credited in the memory
    claim, so an image as long as the array is accepted but credited nothing: the verifier does not need the
    model's size, only the line between what it regenerates and what it does not credit, and the image's length
    draws it."""
    import copy
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from memory_accounting.core.backend import CpuLib
    from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params
    from memory_accounting.setup.commit import leaves_of, merkle_root
    from memory_accounting.setup.encode import image_array
    from memory_accounting.setup.filler import static_filler_key
    from memory_accounting.setup.manifest import Manifest
    from memory_accounting.setup.verify import bundle_theta_words, credited_static_words, verify_bundle
    from memory_accounting.verifier import net
    from memory_accounting.verifier.recompute import ImageWithFiller

    torch.manual_seed(11)
    model = Toy(d=64, n=2)
    tmp = Path(tempfile.mkdtemp())
    seed, block, key = 0x5EED, 16384, (3 << 64) | 9
    sd, mp = _bundle(tmp, model, key=key, seed=seed, block=block, gpu=1, filler_blocks=4)
    manifest = Manifest.load(mp)
    assert verify_bundle(manifest, 1, seed) == []
    tw = bundle_theta_words(manifest, 1)
    assert tw == sd.payload_words and credited_static_words(manifest, tw) == manifest.accounted_words - tw > 0, (
        "a bundle's Θ' region is credited"
    )
    try:
        credited_static_words(manifest)
        raise AssertionError("a bundle was credited without its image's length")
    except ValueError:
        pass
    # the cheat: an image of zeros over the whole array, committed (Θ' as large as the array): accepted, credited
    # nothing
    words, ck = manifest.accounted_words, bytes.fromhex(manifest.commit_key)
    zimg = image_array(np.zeros(words, np.uint64), threads=2)
    np.save(tmp / "zero.npy", zimg.view(np.int64), allow_pickle=False)
    forged = copy.deepcopy(manifest)
    forged.gpus[0].image_file = str(tmp / "zero.npy")
    forged.gpus[0].image_root = merkle_root(leaves_of(zimg, block, ck)).hex()
    assert verify_bundle(forged, 1, seed) == [] and credited_static_words(forged, bundle_theta_words(forged, 1)) == 0, (
        "an image as long as the array was credited"
    )
    forged.save(tmp / "forged.json")
    # online: _make_verifier takes it and credits nothing; it builds the honest bundle's array from Θ''s image and its
    # own filler
    args = SimpleNamespace(
        manifest=None,
        gpu=1,
        state=None,
        replay=None,
        filler_seed=f"{seed:x}",
        pad_seed=None,
        verify_threads=2,
        device="cpu",
    )
    spec = Manifest.load(tmp / "forged.json")
    fargs = SimpleNamespace(**{**vars(args), "manifest": str(tmp / "forged.json")})
    _, fcfg = net._make_verifier(fargs, spec, Keys(*spec.hash_key))
    assert fargs._theta_words == words and net._static_credit(fargs, fcfg) == 0, (
        "an image of zeros over the whole array was credited"
    )
    spec = Manifest.load(mp)
    spec.filler_key = ""  # the verifier derives the key; the manifest's is only a record
    keys = Keys(*spec.hash_key)
    hargs = SimpleNamespace(**{**vars(args), "manifest": str(mp)})
    ver, cfg = net._make_verifier(hargs, spec, keys)
    assert isinstance(ver.mem, ImageWithFiller) and len(ver.mem.image) == sd.payload_words == hargs._theta_words
    assert net._static_credit(hargs, cfg) == (manifest.accounted_words - sd.payload_words) * 8 * net.VDE_WORD_BITS // 64
    assert (
        ver.mem.key == static_filler_key(seed, 1) and ver.mem.seed == seed and ver.mem.words == manifest.accounted_words
    )
    lib, fw = CpuLib(), manifest.accounted_words - sd.payload_words
    pay = np.empty(fw, np.uint64)
    lib.filler(pay, seed, sd.payload_words)
    enc = np.empty(fw, np.uint64)
    lib.setup_at(pay, enc, static_filler_key(seed, 1), sd.payload_words)
    resident = np.concatenate([sd.theta.numpy().view(np.uint64), enc])
    pcfg = MemoryConfig(words=manifest.accounted_words, encoding="vde", perm=False, seed_t=key, image=True)
    nb = manifest.accounted_words // (block // 8)
    for r, j0 in ((0, 0), (16, nb - 1), (16, nb // 2)):
        ch = Challenge(block, 6, r, 1000 + j0, j0, 1024)
        a = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(resident, pack_params(pcfg, ch, keys, MODE_HONEST), a)
        assert (int(a[0]), int(a[1])) == ver.expected(ch)[:2], (r, j0)


def test_bundle_filler_key_is_derived_from_the_seed_and_the_gpu():
    """The filler key was the prover's choice (one-way from its weights key) and the bundle carried it, so the
    verifier could not tell whether two GPUs' filler differed, and the set check compared keys as strings
    ('0x00<hex>' and '<hex>' were two keys). Both sides now derive it from the verifier's filler seed and the GPU
    index; the export refuses a filler under any other key, and keys compare as integers."""
    import copy
    import tempfile
    from pathlib import Path

    from memory_accounting.prover.filler import FillerRegion
    from memory_accounting.setup.filler import static_filler_key
    from memory_accounting.setup.manifest import Manifest
    from memory_accounting.setup.verify import bundle_set_problems

    seed = 0x5EED
    keys_ = {static_filler_key(seed, g) for g in range(8)} | {static_filler_key(seed + 1, 0)}
    assert len(keys_) == 9 and all(0 < k < 1 << 128 for k in keys_)
    from memory_accounting.prover.dynfill import dyn_key

    assert static_filler_key(seed, 0) != dyn_key(seed, 0)
    torch.manual_seed(12)
    sd = convert(Toy(d=32, n=2), enc_key=0x1234)
    fw = 2048
    for gpu, fk in ((1, static_filler_key(seed, 0)), (0, static_filler_key(seed + 1, 0)), (0, None)):
        fill = FillerRegion(torch.zeros(fw, dtype=torch.int64), fw, fk)
        try:
            sd.export_bundle(
                tempfile.mkdtemp(), filler=fill, filler_seed=seed, block_bytes=16384, hbm_bytes=1 << 30, gpu=gpu
            )
            raise AssertionError(f"a filler under key {fk} was exported for GPU {gpu}")
        except ValueError as e:
            assert "derives" in str(e)
    d = Path(tempfile.mkdtemp())
    _, m0 = _bundle(d / "g0", Toy(d=32, n=2), key=(7 << 64) | 1, gpu=0)
    a = Manifest.load(m0)
    b = copy.deepcopy(a)
    b.gpus[0].index = 1
    b.gpus[0].image_root = "cd" * 32
    b.gpus[0].image_file = str(d / "none.npy")
    b.filler_key = "0x00" + a.filler_key  # the same key, written differently
    probs = bundle_set_problems([(a, 0, None), (b, 1, None)])
    assert any("filler key" in p for p in probs), probs
    b.filler_key = a.filler_key.upper()
    assert any("filler key" in p for p in bundle_set_problems([(a, 0, None), (b, 1, None)]))


def test_decode_backend_is_built_once():
    """The CUDA decode hook built a DeviceBackend (load_inline, and torch's build-directory file lock) on every
    layer of every forward. The converted model now asks for one backend per device, cached. (No GPU here:
    the backend class is replaced by a counter.)"""
    import memory_accounting.core.backend as B
    from memory_accounting.prover import dynfill

    made = []

    class Counting:
        def __init__(self, device):
            made.append(device)

    torch.manual_seed(8)
    sd = convert(Toy(d=16, n=2), enc_key=0x99)
    orig, saved = B.DeviceBackend, dict(dynfill._BACKENDS)
    B.DeviceBackend = Counting
    dynfill._BACKENDS.clear()
    try:
        sd.device = torch.device("cuda", 3)  # what a CUDA conversion records; nothing runs on it here
        assert all(sd._be is sd._be for _ in range(10)) and made == ["cuda:3"], made
    finally:
        B.DeviceBackend = orig
        dynfill._BACKENDS.clear()
        dynfill._BACKENDS.update(saved)


if __name__ == "__main__":
    from _runner import run

    run(globals())


def test_throughput_only_export_omits_the_image_and_commitments():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from memory_accounting.setup.manifest import BENCHMARK, Manifest

    sd = convert(Toy(d=128, n=2), enc_key=0x1234)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        with patch.object(sd, "image", side_effect=AssertionError("unexpected image export")), \
             patch.object(sd, "theta_leaves", side_effect=AssertionError("unexpected commitment")):
            filename = sd.export_bundle(folder, block_bytes=16384, benchmark_only=True)
        layout = Manifest.load(filename, allow_benchmark=True)
        assert layout.kind == BENCHMARK
        assert layout.payload_words == sd.theta.numel()
        assert layout.accounted_words == (sd.theta.numel() // 2048) * 2048
        assert [file.name for file in folder.iterdir()] == ["manifest.json"]
