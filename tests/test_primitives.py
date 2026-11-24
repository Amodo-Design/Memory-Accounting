"""C library and (on a CUDA host) the GPU kernels against the pure-Python reference."""

import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools

import numpy as np
import torch
from _runner import skip

from memory_accounting.core import reference as R
from memory_accounting.core.backend import CpuLib, DeviceBackend, pick_device, to_u64_tensor
from memory_accounting.core.params import (
    MAX_LINKS,
    MODE_ENCODE_OTF,
    MODE_HONEST,
    MODE_OTHER_GPU,
    PM61,
    Challenge,
    Keys,
    MemoryConfig,
    pack_params,
    setup_params,
)

lib = CpuLib()
rng = np.random.default_rng(7)


def _struct_members(header: str, name: str) -> list[str]:
    """The u64 members of `typedef struct name { ... }` in posm_core.h, in order."""
    import re

    body = header[header.index(f"typedef struct {name} {{") :]
    body = re.sub(r"/\*.*?\*/", "", body[body.index("{") + 1 : body.index("}")], flags=re.DOTALL)
    return [n.strip() for decl in body.split(";") if decl.strip() for n in decl.strip().removeprefix("u64").split(",")]


def test_param_block_matches_core_header():
    """params.P_FIELDS and SETUP_FIELDS are the layouts of the u64 blocks the kernels read as posm_params and
    posm_setup_params: the same names in the same order, and the C library reports the same sizes."""
    from memory_accounting.core.params import P_FIELDS, SETUP_FIELDS

    h = (Path(__file__).resolve().parents[1] / "memory_accounting/kernels/posm_core.h").read_text()
    assert _struct_members(h, "posm_params") == P_FIELDS
    assert _struct_members(h, "posm_setup_params") == SETUP_FIELDS
    assert (lib.lib.posm_param_words(), lib.lib.posm_setup_param_words()) == (len(P_FIELDS), len(SETUP_FIELDS))
    cfg = MemoryConfig(words=1 << 12, perm=False)
    prm = pack_params(cfg, Challenge(1 << 12, 3, 0, nonce=11, j0=2, k=128), Keys.deterministic(1), MODE_HONEST)
    assert len(prm) == len(P_FIELDS)
    assert len(setup_params(8, 1, 3, 0, 0, 3)) == len(SETUP_FIELDS)


def test_field_and_vde():
    for _ in range(200):
        d, t = int(rng.integers(0, 1 << 60)), int(rng.integers(0, PM61))
        w = lib.lib.posm_vde_encode(d, t)
        assert w == R.vde_encode(d, t)
        assert lib.lib.posm_vde_decode(w, t) == d == R.vde_decode(w, t)
    assert lib.lib.posm_vde_decode(lib.lib.posm_vde_encode(0, 0), 0) == 0


def test_feistel_is_permutation():
    for bits in (1, 2, 5, 12):
        seed = int(rng.integers(0, 2**63))
        imgs = [lib.lib.posm_feistel(q, seed, bits) for q in range(1 << bits)]
        assert sorted(imgs) == list(range(1 << bits))
        assert imgs == [R.feistel_perm(q, seed, bits) for q in range(1 << bits)]


def test_siphash_and_tweak():
    for _ in range(50):
        a = [int(x) for x in rng.integers(0, 2**63, 6)]
        assert lib.siphash(*a) == R.siphash24_128(*a)
        key = (a[1] << 64) | a[0]
        assert lib.lib.posm_tweak(a[0], a[1], a[2]) == R.tweak_at(key, a[2])


# The image T = H(g, w): BLAKE3 keyed_hash(IMG_KEY, u64le(g) || u64le(w0..w3)) read as 4 little-endian u64.
# Known answers from the official `blake3` package (1.0.11), key = derive_key("posm 2026-10 image v1", b""):
#   blake3.blake3(struct.pack("<5Q", g, *w), key=key).digest()
IMAGE_VECTORS = [
    (0x0, (0x0, 0x0, 0x0, 0x0), (0x7A19AD0E633DE21E, 0x4F0D60130B353496, 0x45A04897DE0E71E2, 0x85C907ABEF4A4E57)),
    (0x1, (0x0, 0x0, 0x0, 0x0), (0xAF07D20E37963C90, 0x1680BC49FF0DF7BB, 0x498698E7B1DC3E1B, 0x13B18C6AC6168B68)),
    (
        0x0,
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0x697D5B009CCC7FEE, 0x70043711E89AA84D, 0x6388E5AB2BAEC61C, 0x193AD073F272F4BB),
    ),
    (
        0xFFFFFFFFFFFFFFFF,
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0xC7B44D0E5EF4588B, 0xC4915FB643E8033B, 0x5007A4177F4B5196, 0xF05D5BADE0917AF9),
    ),
    (
        0xFFFFFFFFFFFFFFFF,
        (0x0, 0x0, 0x0, 0x0),
        (0xD3488F96FF3DEC1A, 0xEF4CF4682B05A5B4, 0x9F2A2847D62B6813, 0xD79F40481D656CBE),
    ),
    (
        0x7,
        (0x4000000000000000, 0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
        (0x3A37E6DC52DBAEAE, 0xE8C8E1FB672BDDC9, 0x511BCFE2A0090E5D, 0xA53EB6005681BC51),
    ),
    (
        0x7,
        (0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000),
        (0x0145E6D54CC95303, 0x3C133BEDA33A98AD, 0x6DC9DEFF123C5745, 0x6E657BCC26AB3CA0),
    ),
    (
        0x7,
        (0xC000000000000000, 0x0, 0xC000000000000000, 0x0),
        (0x0A466A6E26A4A32E, 0xF41745CB933D5029, 0xAF49C03415047EE8, 0x3D83AF5E7C1D7ABA),
    ),
    (
        0x100000000,
        (0x1, 0x2, 0x3, 0x4),
        (0x1FF6359B35D4D49D, 0x587924B8F64865CF, 0xC1A95075133FBEC6, 0x7FA2A7FFF6743CEF),
    ),
    (
        0xFFFFFFFF,
        (0x1, 0x2, 0x3, 0x4),
        (0x4E458F060527F676, 0x3268D719FD79B2D5, 0xB4300622566391E8, 0x088A8CFC998D4476),
    ),
    (
        0x10000000000,
        (0x123456789ABCDEF, 0xFEDCBA9876543210, 0x0, 0xFFFFFFFFFFFFFFFF),
        (0x00E89559F5AB3188, 0x8F36801586E06813, 0xE52DDA7B25EF8267, 0xAFAAD9D1C97241BC),
    ),
    (
        0x8000000000000000,
        (0x1FFFFFFFFFFFFFFF, 0x2000000000000000, 0x4000000000000005, 0x8000000000000009),
        (0xF4C0A27922BAF833, 0x4146CF9BBC640548, 0x2203471FB9FDD7B5, 0x61E9C385593E2E2C),
    ),
    (
        0xB75B0124E,
        (0x7B3B7105366EB15E, 0xDBFC54D4A697E485, 0x14F4DDC07EC5FF12, 0xD2921F4A083212DF),
        (0x5B2A2FEAA89E0C1D, 0xDD3679A092428A7E, 0xD3805EF7CC62EA63, 0x6182E3C47849EFB6),
    ),
    (
        0x171D86446D535171,
        (0xA13CAD1E38F0F413, 0x36B302698710EC44, 0x9FB9AFF0CEC0F95, 0xF7A54A22F737909F),
        (0x64635511DF24E160, 0x6E4EDED599EAED5B, 0x8A0366A3776E6DFD, 0xDDBEDC4E04D0FFF7),
    ),
    (
        0x4BFF505E2,
        (0xC736AFA8FEF6A2C3, 0xF4CF2B1AC9A06C29, 0xF1D4A464AF89BF78, 0xBA4E2F5BB6C257D4),
        (0x0CF793788CA43796, 0x3B69644EC49C309F, 0xD826FF6D0094D835, 0xC2566B9A1D0B935E),
    ),
    (
        0xFD2C0,
        (0xB6D257B417D4BE87, 0x8CA0E06A95583970, 0xF4F8F0C6FF987321, 0x7734A5E37536887F),
        (0x42C078518AC7F994, 0x8E15C6C8160C320D, 0xE55BD77C3C73537B, 0xAF8D2D3C758A348A),
    ),
    (
        0x91A1E3864FC5EF45,
        (0x23784D7AF28BE309, 0x4BCC1920EC5329F1, 0xFECB725B184FE7C5, 0x57824711ADB7640E),
        (0x78BCDE0C9BD6649D, 0x9C3D53C8435C1A3B, 0x04E5D154EB0688F8, 0x1581F1618095F01B),
    ),
    (
        0x8D30733301AFA181,
        (0xE563C44B66F07B4F, 0xC115A8F4A2FCE135, 0xA12C23519D288C6C, 0x665AA2C62832DA55),
        (0xD526F0A6C9FD0A7E, 0x6C46825BEB0A2EFD, 0x9DEB0CA34378DFE9, 0xB9F1F3D06E68D108),
    ),
    (
        0xABFF90DF6,
        (0x744EDED55B53B581, 0xD46355DA75AF2B01, 0x376F42A93D1245E0, 0x84930310DBCE0575),
        (0xB362608CB682768B, 0x4755265031FA8134, 0x12E9315D18B3B0AC, 0xEA40FE28D7655DE5),
    ),
    (
        0xA3D7B3DEC6DA74D8,
        (0x8CAEF9458BEE9114, 0x58E4C875DDD9A43C, 0x9DAE528B0CB42DE5, 0xFE53ECD2EB3929BB),
        (0x82AB80885563143D, 0x044E9F5460472D27, 0x0439116AA6C8C0F3, 0xEA6F8BF5F0EB63CF),
    ),
]


def test_image_known_answers():
    """The reference and the C image (the header every kernel shares) both give the official BLAKE3 keyed_hash."""
    import re

    assert R.IMG_KEY.hex() == "f57be4c5137fa6167fa7a2f648754a9cefb52f5122991c15705741611b86e000"
    assert len(IMAGE_VECTORS) >= 20
    for g, w, t in IMAGE_VECTORS:
        assert R.image_group(g, list(w)) == list(t), f"reference g={g:#x} w={w}"
        assert lib.image_group(g, w) == list(t), f"C g={g:#x} w={w}"
    # the header's key words are the derivation in its comment
    h = (Path(__file__).resolve().parents[1] / "memory_accounting/kernels/posm_core.h").read_text()
    words = [int(re.search(rf"#define IMG_KEY{i} (0x[0-9A-Fa-f]+)u", h).group(1), 16) for i in range(8)]
    assert b"".join(x.to_bytes(4, "little") for x in words) == R.IMG_KEY


def test_image_matches_official_blake3():
    """Many random groups against the official package, where it is installed (`uv run --with blake3`)."""
    try:
        import blake3
    except ImportError:
        skip("blake3 package not installed (uv run --with blake3); the known answers above cover the image")
    import struct

    key = blake3.blake3(b"", derive_key_context="posm 2026-10 image v1").digest()
    assert key == R.IMG_KEY
    for _ in range(500):
        g = int(rng.integers(0, 2**64, dtype=np.uint64))
        w = [int(x) for x in rng.integers(0, 2**64, 4, dtype=np.uint64)]
        want = list(struct.unpack("<4Q", blake3.blake3(struct.pack("<5Q", g, *w), key=key).digest()))
        assert R.image_group(g, w) == want and lib.image_group(g, w) == want, f"g={g:#x} w={w}"


def test_cpu_image_matches_reference():
    """C image == reference: single groups at random 64-bit positions, and whole arrays (the setup's image,
    serial and threaded, and the CPU filler image) where the position is the global group index."""
    for _ in range(300):
        g = int(rng.integers(0, 2**64, dtype=np.uint64))
        w = [int(x) for x in rng.integers(0, 2**64, 4, dtype=np.uint64)]
        assert lib.image_group(g, w) == R.image_group(g, w)
    theta = rng.integers(0, 2**64, 4 * 257, dtype=np.uint64)
    want = R.image_array(theta.tolist())
    for th in (1, 3):
        out = np.empty_like(theta)
        lib.image(theta, out, th)
        assert out.tolist() == want, f"array image threads={th}"
    from memory_accounting.setup.filler import filler_words

    seed, key, q0, n = 0x5EED, (0x1234 << 64) | 0x5678, 4 * 1000, 64  # filler image at global words q0..
    enc = [R.vde_encode(int(d), R.tweak_at(key, q0 + i)) for i, d in enumerate(filler_words(seed, q0, n))]
    out = np.empty(n, np.uint64)
    lib.filler_image(out, seed, key, q0, 2)
    assert out.tolist() == [v for g in range(n // 4) for v in R.image_group(q0 // 4 + g, enc[4 * g : 4 * g + 4])]


def _small_setup(words=1024, enc=1, perm=1):
    cfg = MemoryConfig(words=words, encoding="vde" if enc else "none", perm=bool(perm))
    raw = rng.integers(0, 1 << 60, words, dtype=np.int64).view(np.uint64)
    out = np.zeros(words, dtype=np.uint64)
    lib.setup(raw, out, setup_params(words, cfg.enc, cfg.seed_t, int(cfg.perm), cfg.perm_seed, cfg.bits))
    exp = [
        R.vde_encode(int(raw[R.feistel_perm(q, cfg.perm_seed, cfg.bits) if perm else q]), R.tweak_at(cfg.seed_t, q))
        if enc
        else int(raw[R.feistel_perm(q, cfg.perm_seed, cfg.bits) if perm else q])
        for q in range(words)
    ]
    assert out.tolist() == exp
    return cfg, raw, out


def test_pool_serves_one_caller_at_a_time():
    """The CPU pool is process-wide: two threads driving it at once (a CPU prover and the verifier's
    checker in one process) must each get the serial digest, at small blocks (few chunks: most threads
    sit a job out) and large ones."""
    import threading

    cfg = MemoryConfig(words=1 << 16, encoding="none", perm=False)
    mem = rng.integers(0, 1 << 63, 1 << 16, dtype=np.int64).view(np.uint64)
    keys = Keys.deterministic(4)
    jobs = [
        Challenge(B, 3, r, nonce=77 + i, j0=1, k=128)
        for i, (B, r) in enumerate(((1 << 12, 0), (1 << 15, 0), (1 << 12, 16), (1 << 15, 64)))
    ]
    want = []
    for ch in jobs:
        o = np.zeros(2 + ch.n, np.uint64)
        lib.chain(mem, pack_params(cfg, ch, keys, MODE_HONEST), o)
        want.append(o.tolist())
    lib.pool_start(6)
    bad = []

    def run(seed):
        for it in range(40):
            j = (seed + it) % len(jobs)
            o = np.zeros(2 + jobs[j].n, np.uint64)
            lib.chain(mem, pack_params(cfg, jobs[j], keys, MODE_HONEST), o, mt=True)
            if o.tolist() != want[j]:
                bad.append(j)

    th = [threading.Thread(target=run, args=(s,)) for s in range(3)]
    [t.start() for t in th]
    [t.join(60) for t in th]
    alive = any(t.is_alive() for t in th)
    lib.pool_stop()
    assert not alive, "pool hung with concurrent callers"
    assert not bad, f"concurrent callers got wrong digests: {bad[:5]}"


def test_cpu_chain_matches_reference():
    cfg, _raw, mem = _small_setup(words=4096)
    keys = Keys.deterministic(3)
    for k, r in ((128, 0), (128, 16), (256, 16), (128, 32)):
        ch = Challenge(block_bytes=16 * k, n=4, r=r, nonce=12345, j0=3, k=k)  # one 16-row tile per block
        prm = pack_params(cfg, ch, keys, MODE_HONEST)
        out = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(mem, prm, out)
        lo, hi, trace = R.chain(mem.tolist(), prm.view(np.uint64).tolist())
        assert (int(out[0]), int(out[1])) == (lo, hi), f"k={k} r={r}"
        assert out[2:].tolist() == trace
        lib.pool_start(4)
        out2 = np.zeros_like(out)
        lib.chain(mem, prm, out2, mt=True)
        lib.pool_stop()
        assert out2.tolist() == out.tolist()


def test_linear_shortcut_is_blocked():
    """Without the output mixer, hash(A*M) collapses to one multiply per byte: fold M into the hash
    weights once (v[j] = sum_c M[j][c]*K^(c+1)) and evaluate sum_rho K^(rho*r) * sum_j A[rho][j]*v[j].
    Check the shortcut reproduces the UNMIXED hash (so the attack is real) and that the real digest
    (with the nonce-keyed mixer on every output) does not match it."""
    cfg, _raw, mem = _small_setup(words=1024)
    keys = Keys.deterministic(11)
    ch = Challenge(block_bytes=2048, n=1, r=16, nonce=777, j0=1, k=128)
    prm = pack_params(cfg, ch, keys, MODE_HONEST).view(np.uint64).tolist()
    p = dict(zip(__import__("memory_accounting.core.params", fromlist=["P_FIELDS"]).P_FIELDS, prm))
    kn = R.chain_polyk(p["k0"], p["k1"], p["nonce"], p["polyk"])
    mixkey = R.chain_mixkey(p["k0"], p["k1"], p["nonce"], p["polyk"])
    K, r, be = ch.k, ch.r, ch.be
    M = R.matrix(ch.nonce, K, r)
    words = mem[ch.j0 * be : (ch.j0 + 1) * be].tolist()
    rows = [
        [(w >> (8 * q)) & 0xFF for w in words[row * (K // 8) : (row + 1) * (K // 8)] for q in range(8)]
        for row in range(be // (K // 8))
    ]
    outs = [[sum(rows[rho][j] * M[j][c] for j in range(K)) for c in range(r)] for rho in range(len(rows))]
    T1 = [pow(kn, c + 1, PM61) for c in range(r)]
    # (a) the linear shortcut equals the unmixed hash
    unmixed = (
        sum(outs[rho][c] * pow(kn, rho * r, PM61) % PM61 * T1[c] for rho in range(len(rows)) for c in range(r)) % PM61
    )
    v = [sum(M[j][c] * T1[c] for c in range(r)) % PM61 for j in range(K)]
    shortcut = (
        sum(pow(kn, rho * r, PM61) * sum(rows[rho][j] * v[j] for j in range(K)) for rho in range(len(rows))) % PM61
    )
    assert shortcut == unmixed, "factoring should reproduce the unmixed hash"
    # (b) the real per-link value mixes every output first and does not equal the shortcut
    mixed = (
        sum(
            R.mix_output(outs[rho][c], mixkey) * pow(kn, rho * r, PM61) % PM61 * T1[c]
            for rho in range(len(rows))
            for c in range(r)
        )
        % PM61
    )
    assert mixed != shortcut and mixed != unmixed
    # (c) and the C reference agrees with the mixed Python value end to end
    out = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(mem, pack_params(cfg, ch, keys, MODE_HONEST), out)
    lo, hi, _ = R.chain(mem.tolist(), prm)
    assert (int(out[0]), int(out[1])) == (lo, hi)


def test_challenge_validation():
    # k below KMIN
    for bad in (
        {"block_bytes": 16384, "n": 1, "r": 16, "nonce": 1, "j0": 0, "k": 2048},  # 8 rows: not a whole 16-row tile
        {"block_bytes": 65536, "n": MAX_LINKS + 1, "r": 0, "nonce": 1, "j0": 0},  # output buffer holds MAX_LINKS
        {"block_bytes": 65536, "n": 1, "r": 24, "nonce": 1, "j0": 0},  # r not a multiple of 16
        {"block_bytes": 65536, "n": 1, "r": 16, "nonce": 1, "j0": 0, "k": 96},
    ):
        try:
            Challenge(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid challenge {bad}")
    Challenge(block_bytes=16384, n=MAX_LINKS, r=16, nonce=1, j0=0, k=1024)  # 16 rows of 1 KB: valid


def test_public_seeds_need_x_times_r_at_least_k():
    """With public link seeds (x > 0) and x * r < k the public matrices share a kernel: a block changed by a
    vector in it keeps every digest. Such parameters are refused."""
    ch = Challenge(1 << 12, 3, 32, nonce=1, j0=0, k=256)
    pack_params(MemoryConfig(words=1 << 12, perm=False, x=8, tseed=1), ch, Keys.deterministic(1), MODE_HONEST)
    try:
        pack_params(MemoryConfig(words=1 << 12, perm=False, x=4, tseed=1), ch, Keys.deterministic(1), MODE_HONEST)
        raise AssertionError("accepted x * r < k")
    except ValueError:
        pass


def test_challenge_rejects_bad_sizes_before_arithmetic():
    """Challenge fields arrive from the network: zero or out-of-range sizes are a ValueError (which the
    responder catches), never a ZeroDivisionError."""
    for kw in ({"block_bytes": 0}, {"k": 0}, {"k": 0, "r": 16}, {"block_bytes": -32}, {"r": -16}, {"g": 0}, {"n": 0}):
        args = {"block_bytes": 1 << 16, "n": 4, "r": 0, "nonce": 1, "j0": 0, "k": 1024, "g": 1}
        args.update(kw)
        try:
            Challenge(**args)
            raise AssertionError(f"accepted {kw}")
        except ValueError:
            pass


def test_analysis_drops_incorrect_records(tmp_path=None):
    import json
    import tempfile

    from memory_accounting.core.analysis import all_comparisons, best_config, load_results

    recs = []
    for prover, correct, base in (("honest", True, 100.0), ("cpu", False, 900.0)):
        recs.append(
            {
                "prover": prover,
                "B": 65536,
                "n": 4,
                "r": 0,
                "correct": correct,
                "samples_ns": (rng.normal(base, 5, 50) * 1e3).tolist(),
            }
        )
    f = Path(tempfile.mkdtemp()) / "r.json"
    f.write_text(json.dumps({"meta": {}, "records": recs}))
    res = load_results(f)
    assert [r["prover"] for r in res["records"]] == ["honest"]
    key, table = best_config(all_comparisons(res))
    assert key is None and table == []


def test_single_link_continuation_matches_full_chain():
    """The dram_copy prover runs one link at a time over a one-block staging buffer, carrying the digest
    state through d0/link0 and picking the next block on the host.  Check that path against one full chain."""
    cfg, _raw, mem = _small_setup(words=4096)
    keys = Keys.deterministic(21)
    for r in (0, 16):
        ch = Challenge(block_bytes=2048, n=6, r=r, nonce=4242, j0=2, k=128)
        full = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(mem, pack_params(cfg, ch, keys, MODE_HONEST), full)
        nb = len(mem) // ch.be
        d, idx, trace = (0, 0), ch.j0, []
        for link in range(ch.n):
            staging = np.ascontiguousarray(mem[idx * ch.be : (idx + 1) * ch.be])
            prm = pack_params(cfg, ch, keys, MODE_HONEST, nb=1, d0=d, link0=link, n=1, j0=0)
            out = np.zeros(3, dtype=np.uint64)
            lib.chain(staging, prm, out)
            d = (int(out[0]), int(out[1]))
            trace.append(idx)
            idx = d[0] % nb
        assert d == (int(full[0]), int(full[1])), f"r={r}"
        assert trace == full[2:].tolist()


def test_hash_only_is_injective_over_full_words():
    """r = 0 hashes each stored word as two 32-bit halves.  It used to reduce the word mod p = 2^61 - 1
    first, so words differing by a multiple of p gave identical digests and traces under every nonce
    (dynamic memory, with no image hash in front, could be altered or carry 3 spare bits per word)."""
    words = 4096
    cfg = MemoryConfig(words=words, encoding="none", perm=False)
    keys = Keys.deterministic(17)
    base = rng.integers(0, PM61, words, dtype=np.int64).view(np.uint64)

    def alias(a, k):
        return (a + np.uint64(k * PM61)).astype(np.uint64)  # a + k p < 2^64 for a < p, k <= 7

    # one word changed by 3p
    pairs = [
        (np.zeros(words, np.uint64), np.full(words, PM61, np.uint64)),  # the reported pair
        (np.zeros(words, np.uint64), np.full(words, 7 * PM61, np.uint64)),
        (base, alias(base, 1)),
        (base, alias(base, 7)),
        (base, np.where(np.arange(words) == 1500, alias(base, 3), base)),
    ]
    for left, right in pairs:
        assert ((left.astype(object) - right.astype(object)) % PM61 == 0).all() and not np.array_equal(
            left, right
        )  # aliases mod p
        for nonce in (5, 6):
            # block 1 (words 1024..2047) holds word 1500
            ch = Challenge(block_bytes=8192, n=4, r=0, nonce=nonce, j0=1, k=128)
            prm = pack_params(cfg, ch, keys, MODE_HONEST)
            oa, ob = np.zeros(2 + ch.n, np.uint64), np.zeros(2 + ch.n, np.uint64)
            lib.chain(left, prm, oa)
            lib.chain(right, prm, ob)
            assert (oa[0], oa[1]) != (ob[0], ob[1]), f"r=0 digest blind to a multiple of p (nonce {nonce})"
            assert R.chain(left.tolist(), prm.view(np.uint64).tolist())[:2] == (int(oa[0]), int(oa[1]))
            assert R.chain(right.tolist(), prm.view(np.uint64).tolist())[:2] == (int(ob[0]), int(ob[1]))
    # every bit of a word reaches the digest, the top three (above p) included
    left = base.copy()
    prm = pack_params(cfg, Challenge(block_bytes=8192, n=1, r=0, nonce=9, j0=0, k=128), keys, MODE_HONEST)
    o0 = np.zeros(3, np.uint64)
    lib.chain(left, prm, o0)
    for bit in (0, 31, 32, 60, 61, 62, 63):
        a2 = left.copy()
        a2[17] ^= np.uint64(1 << bit)
        o = np.zeros(3, np.uint64)
        lib.chain(a2, prm, o)
        assert o[:2].tolist() != o0[:2].tolist(), f"bit {bit} of a word does not reach the r=0 digest"


def test_grouped_fold_is_injective():
    """With g > 1 a link folds its g entry words into one value.  The fold used K_n^(g+1), and with x = 0 the
    entry key K_s is K_n too, so word loc of group g and word loc-1 of group g+1 landed on the same power of K_n:
    two blocks differing by that shift gave the same digest under every nonce (r = 0).  The fold now uses its own
    key, K_n * K_f^g.  Checked for r = 0 (the word shift, and a half-word shift) and r > 0 (a row moved to the
    next group), with x = 0 and x > 0."""
    words = 4096
    keys = Keys.deterministic(31)
    for r, g, x in ((0, 2, 0), (0, 3, 0), (0, 8, 0), (0, 2, 3), (16, 2, 0), (16, 4, 0), (16, 2, 8)):
        k = 128
        cfg = MemoryConfig(words=words, encoding="none", perm=False, x=x, tseed=0xF00D if x else 0)
        bb = (8160 if g == 3 else 8192) if r == 0 else 16 * k * 2
        span = Challenge(block_bytes=bb, n=1, r=r, nonce=1, j0=0, k=k, g=g).span
        pairs = []
        if r == 0:
            left, right = np.zeros(words, np.uint64), np.zeros(words, np.uint64)
            left[1] = 1
            right[span] = 1  # word 1 of group 0 / word 0 of group 1
            pairs.append((left, right))
            left, right = np.zeros(words, np.uint64), np.zeros(words, np.uint64)
            left[span - 1] = 5
            right[2 * span - 2] = 5  # last word of group 0 / group 1's one before
            pairs.append((left, right))
            left, right = np.zeros(words, np.uint64), np.zeros(words, np.uint64)
            left[1] = 7 << 32
            right[span] = 7 << 32  # high halves: word 1 of group 0 / word 0 of group 1
            pairs.append((left, right))
        else:
            row = rng.integers(0, 1 << 63, k // 8, dtype=np.int64).view(np.uint64)
            left, right = np.zeros(words, np.uint64), np.zeros(words, np.uint64)
            left[: k // 8] = row
            right[span * (k // 8) : (span + 1) * (k // 8)] = row  # row 0 of group 0 / of group 1
            pairs.append((left, right))
        for left, right in pairs:
            for nonce in range(12):
                ch = Challenge(block_bytes=bb, n=1, r=r, nonce=nonce, j0=0, k=k, g=g)
                prm = pack_params(cfg, ch, keys, MODE_HONEST)
                oa, ob = np.zeros(3, np.uint64), np.zeros(3, np.uint64)
                lib.chain(left, prm, oa)
                lib.chain(right, prm, ob)
                assert oa[:2].tolist() != ob[:2].tolist(), (
                    f"g={g} r={r} x={x} nonce={nonce}: shifted blocks share a digest"
                )
                if r == 0 or nonce < 2:
                    assert R.chain(left.tolist(), prm.view(np.uint64).tolist())[:2] == (int(oa[0]), int(oa[1]))


def test_single_group_digests_unchanged():
    """g = 1 (what net and the setup bundles use) must give the same digests as before the fold key got its own
    domain: known answers computed with v0.11.0, through the C chain and the Python reference."""
    words = 2048
    data = np.array([(0x9E3779B97F4A7C15 * (i + 1)) & ((1 << 64) - 1) for i in range(words)], dtype=np.uint64)
    keys = Keys(k0=0x0123456789ABCDEF, k1=0xFEDCBA9876543210, polyk=0x123456789ABCDE)
    known = {
        (0, 0): (0x22E885FAEE69B82D, 0x9D359F98C598D216, [1, 2, 2]),
        (16, 0): (0xE1EB6B39E3A329B0, 0x42EFF30325D2F197, [1, 1, 0]),
        (0, 3): (0xA7B365917294097B, 0x8388BB49531E4EBD, [1, 1, 1]),
        (16, 8): (0x3E0BC855FC30CDF5, 0x32902AF9278DF044, [1, 1, 3]),
    }
    for (r, x), (lo, hi, trace) in known.items():
        cfg = MemoryConfig(words=words, encoding="none", perm=False, x=x, tseed=0xF00D if x else 0)
        prm = pack_params(cfg, Challenge(block_bytes=4096, n=3, r=r, nonce=0xC0FFEE, j0=1, k=128), keys, MODE_HONEST)
        out = np.zeros(5, np.uint64)
        lib.chain(data, prm, out)
        assert out.tolist() == [lo, hi] + trace, f"g=1 digest changed (r={r}, x={x})"
        assert R.chain(data.tolist(), prm.view(np.uint64).tolist()) == (lo, hi, trace)


def test_cpu_paths_match_reference_full_words():
    """The serial chain, the pool at several sizes (worker ranges starting mid-group and mid-stream), the
    segmented and imaged block reads, and per-link seeds (x > 0) all agree with the Python reference over
    full 64-bit words (VDE words have bits 62/63 clear, which hid half-word mistakes)."""
    words = 8192
    raw = rng.integers(0, 1 << 63, words, dtype=np.int64).view(np.uint64) * np.uint64(2) + rng.integers(
        0, 2, words
    ).astype(np.uint64)
    keys = Keys.deterministic(23)
    cuts = [0, 1000, 1004, 5000, words]  # segments of odd sizes (multiples of 4)
    parts = [np.ascontiguousarray(raw[a:b]) for a, b in itertools.pairwise(cuts)]
    segtab = np.array(cuts + [p.ctypes.data for p in parts], dtype=np.uint64)
    # (r, g, k); g=3: span 340 words
    shapes = (
        (0, 1, 128),
        (0, 3, 128),
        (0, 8, 128),
        (0, 1, 128),
        (16, 1, 128),
        (32, 2, 256),
    )
    for (r, g, k), x, img in zip(shapes, (0, 0, 5, 3, 0, 8), (0, 1, 0, 1, 1, 0)):  # x * r >= k where r > 0 and x > 0
        cfg = MemoryConfig(words=words, encoding="none", perm=False, x=x, tseed=0xF00D if x else 0, image=bool(img))
        ch = Challenge(
            block_bytes=8160 if g == 3 else 8192, n=5, r=r, nonce=int(rng.integers(0, 2**63)), j0=1, k=k, g=g
        )
        if r and ch.block_bytes > 4096:
            ch = Challenge(block_bytes=16 * k, n=5, r=r, nonce=ch.nonce, j0=1, k=k, g=g)
        prm = pack_params(cfg, ch, keys, MODE_HONEST)
        lo, hi, trace = R.chain(raw.tolist(), prm.view(np.uint64).tolist())
        exp = [lo, hi] + trace
        out = np.zeros(2 + ch.n, np.uint64)
        lib.chain(raw, prm, out)
        assert out.tolist() == exp, f"serial r={r} g={g} x={x} img={img}"
        seg = pack_params(cfg, ch, keys, MODE_HONEST, nseg=len(parts), segtab=segtab.ctypes.data)
        out = np.zeros(2 + ch.n, np.uint64)
        lib.chain(np.zeros(1, np.uint64), seg, out)
        assert out.tolist() == exp, f"segmented r={r} g={g} x={x} img={img}"
        for nt in (3, 7):
            lib.pool_stop()
            lib.pool_start(nt)
            for m, p in ((raw, prm), (np.zeros(1, np.uint64), seg)):
                out = np.zeros(2 + ch.n, np.uint64)
                lib.chain(m, p, out, mt=True)
                assert out.tolist() == exp, f"pool {nt} r={r} g={g} x={x} img={img} seg={m is not raw}"
        lib.pool_stop()
        # entries under alternating seeds and shapes: the cached per-seed tables must follow the seed
        for seed in (ch.nonce, 12345, ch.nonce):
            for j in (0, 2):
                t = np.zeros(g, np.uint64)
                lib.entry(raw, prm, seed, j, t)
                assert t.tolist() == R.entry(
                    raw.tolist(),
                    dict(
                        zip(
                            __import__("memory_accounting.core.params", fromlist=["P_FIELDS"]).P_FIELDS, prm.view(np.uint64).tolist()
                        )
                    ),
                    seed,
                    j,
                ), f"entry r={r} g={g} seed={seed}"


def test_device_scalar_path_on_cpu():
    """The scalar decomposition the CUDA kernels share through posm_core.h (power-table and matrix fills,
    posm_partial_slots over strided threads, the cheat fetches) gives the C chain's digests, so a change
    to the shared device code is checked without a GPU."""
    cfg0, raw, theta = _small_setup(words=4096)
    canon = np.zeros_like(raw)
    lib.setup(raw, canon, setup_params(len(raw), 1, cfg0.seed_c, 0, 0, cfg0.bits))
    # bits 62/63 set too
    full = rng.integers(0, 1 << 63, 4096, dtype=np.int64).view(np.uint64) * np.uint64(2) + np.uint64(1)
    keys = Keys.deterministic(29)
    for r, g, k, x, img in (
        (0, 1, 128, 0, 0),
        (0, 4, 128, 3, 0),
        (0, 2, 128, 0, 1),
        (16, 1, 128, 0, 0),
        (32, 2, 128, 4, 1),
    ):
        cfg = MemoryConfig(words=4096, encoding="vde", perm=True, x=x, tseed=0xABC if x else 0, image=bool(img))
        ch = Challenge(
            block_bytes=4096 if r == 0 else 2048, n=4, r=r, nonce=int(rng.integers(0, 2**63)), j0=1, k=k, g=g
        )
        exp = np.zeros(2 + ch.n, np.uint64)
        lib.chain(theta, pack_params(cfg, ch, keys, MODE_HONEST), exp)
        for lanes in (1, 3, 32):
            for mode, mem, cn in (
                (MODE_HONEST, theta, theta),
                (MODE_OTHER_GPU, canon, canon),
                (MODE_ENCODE_OTF, raw, raw),
            ):
                out = np.zeros_like(exp)
                lib.chain_scalar(mem, cn, pack_params(cfg, ch, keys, mode), out, lanes)
                assert out.tolist() == exp.tolist(), f"scalar mode={mode} r={r} g={g} x={x} img={img} lanes={lanes}"
        plain = MemoryConfig(words=4096, encoding="none", perm=False, x=x, tseed=0xABC if x else 0, image=bool(img))
        exp = np.zeros(2 + ch.n, np.uint64)
        lib.chain(full, pack_params(plain, ch, keys, MODE_HONEST), exp)
        out = np.zeros_like(exp)
        lib.chain_scalar(full, full, pack_params(plain, ch, keys, MODE_HONEST), out, 7)
        assert out.tolist() == exp.tolist(), f"scalar full words r={r} g={g} x={x} img={img}"


def test_device_kernels_match_cpu():
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device")
    be = DeviceBackend(dev)
    words = 1 << 16
    lib.pool_start(4)
    for enc, perm in ((1, 1), (0, 1), (1, 0)):
        cfg = MemoryConfig(words=words, encoding="vde" if enc else "none", perm=bool(perm))
        raw_np = rng.integers(0, 1 << 60, words, dtype=np.int64).view(np.uint64)
        raw = to_u64_tensor(raw_np, dev)
        theta = torch.empty_like(raw)
        be.setup(
            raw, theta, to_u64_tensor(setup_params(words, cfg.enc, cfg.seed_t, perm, cfg.perm_seed, cfg.bits), dev)
        )
        canon = torch.empty_like(raw)
        be.setup(
            raw, canon, to_u64_tensor(setup_params(words, 1, cfg.seed_c, 0, 0, cfg.bits), dev)
        ) if enc else canon.copy_(raw)
        be.sync()
        theta_np = theta.cpu().numpy().view(np.uint64)
        ref = np.zeros(words, dtype=np.uint64)
        lib.setup(raw_np, ref, setup_params(words, cfg.enc, cfg.seed_t, perm, cfg.perm_seed, cfg.bits))
        assert np.array_equal(theta_np, ref), "device setup != cpu setup"
        dec = torch.empty_like(raw)
        be.decode(theta, dec, to_u64_tensor(setup_params(words, cfg.enc, cfg.seed_t, 0, 0, 0), dev))
        be.sync()
        if perm:
            assert dec.cpu().numpy().view(np.uint64).tolist() == [
                int(raw_np[R.feistel_perm(q, cfg.perm_seed, cfg.bits)]) for q in range(words)
            ]
        else:
            assert torch.equal(dec, raw)
        keys = Keys.deterministic(5)
        for k, r, g, x, img in (
            (1024, 0, 1, 0, 0),
            (128, 16, 1, 0, 0),
            (1024, 256, 4, 8, 0),
            (2048, 64, 2, 32, 0),
            (1024, 0, 16, 32, 0),
            (1024, 0, 1, 0, 1),
            (1024, 256, 1, 0, 1),
        ):
            cfg = MemoryConfig(
                words=words,
                encoding="vde" if enc else "none",
                perm=bool(perm),
                x=x,
                tseed=0xC0DE if x else 0,
                image=bool(img),
            )
            for T in (256, 1024):
                ch = Challenge(block_bytes=32768, n=5, r=r, nonce=int(rng.integers(0, 2**63)), j0=2, k=k, g=g)
                exp = np.zeros(2 + ch.n, dtype=np.uint64)
                lib.chain(theta_np, pack_params(cfg, ch, keys, MODE_HONEST), exp, mt=True)
                pw = be.power_buffer(ch.pw_count)
                mbuf = be.matrix_buffer(ch.mbuf_bytes)
                ablk = be.block_buffer(ch.block_bytes)
                for mode, mem, cn in ((MODE_HONEST, theta, theta), (MODE_OTHER_GPU, canon, canon)) + (
                    ((MODE_ENCODE_OTF, raw, raw),) if enc else ()
                ):
                    prm = to_u64_tensor(pack_params(cfg, ch, keys, mode), dev)
                    out = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
                    state = torch.zeros(8, dtype=torch.int64, device=dev)
                    be.chain(mem, cn, out, prm, state, pw, mbuf, k, r, T)
                    be.sync()
                    assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), (
                        f"persistent mode={mode} enc={enc} perm={perm} k={k} r={r} g={g} x={x} T={T}"
                    )
                    for G in (1, 7, 32):
                        out.zero_()
                        parts = torch.zeros(G * g, dtype=torch.int64, device=dev)
                        be.chain_multi(mem, cn, out, prm, state, parts, pw, mbuf, G, T, ch.n, k, r, ch.be, g, x > 0)
                        be.sync()
                        assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), (
                            f"multi G={G} mode={mode} enc={enc} perm={perm} k={k} r={r} g={g} x={x} img={img} T={T}"
                        )
                        if dev == "cuda":
                            for graph in (False, True):
                                out.zero_()
                                be.chain_multi(
                                    mem,
                                    cn,
                                    out,
                                    prm,
                                    state,
                                    parts,
                                    pw,
                                    mbuf,
                                    G,
                                    T,
                                    ch.n,
                                    k,
                                    r,
                                    ch.be,
                                    g,
                                    x > 0,
                                    tensor=r > 0,
                                    graph=graph,
                                    ablk=ablk,
                                )
                                be.sync()
                                assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), (
                                    f"tensor graph={graph} G={G} mode={mode} enc={enc} perm={perm} k={k} r={r} g={g} x={x} img={img}"
                                )
                                if graph:  # replay with a different challenge written into the same prm buffer
                                    ch2 = Challenge(
                                        block_bytes=32768, n=5, r=r, nonce=int(rng.integers(0, 2**63)), j0=5, k=k, g=g
                                    )
                                    exp2 = np.zeros(2 + ch2.n, dtype=np.uint64)
                                    lib.chain(theta_np, pack_params(cfg, ch2, keys, MODE_HONEST), exp2, mt=True)
                                    prm.copy_(torch.from_numpy(pack_params(cfg, ch2, keys, mode)))
                                    out.zero_()
                                    be.chain_multi(
                                        mem,
                                        cn,
                                        out,
                                        prm,
                                        state,
                                        parts,
                                        pw,
                                        mbuf,
                                        G,
                                        T,
                                        ch2.n,
                                        k,
                                        r,
                                        ch2.be,
                                        g,
                                        x > 0,
                                        tensor=r > 0,
                                        graph=True,
                                        ablk=ablk,
                                    )
                                    be.sync()
                                    assert out.cpu().numpy().view(np.uint64).tolist() == exp2.tolist(), (
                                        f"graph replay G={G} mode={mode} k={k} r={r} g={g}"
                                    )
                                    prm.copy_(torch.from_numpy(pack_params(cfg, ch, keys, mode)))
                    if dev == "cuda" and mode == MODE_HONEST and x:
                        # setup table pass on the device == C entries
                        from memory_accounting.core.params import table_seed

                        nbk = words * 8 // ch.block_bytes
                        trow = torch.zeros(nbk * g, dtype=torch.int64, device=dev)
                        parts = torch.zeros(32 * g, dtype=torch.int64, device=dev)
                        seed = table_seed(cfg.tseed, 3)
                        be.entry_table(
                            mem,
                            cn,
                            prm,
                            state,
                            parts,
                            pw,
                            mbuf,
                            ablk,
                            trow,
                            seed,
                            0,
                            nbk,
                            32,
                            256,
                            k,
                            r,
                            ch.be,
                            g,
                            tensor=r > 0,
                        )
                        be.sync()
                        got = trow.cpu().numpy().view(np.uint64).reshape(nbk, g)
                        for j in (0, nbk // 2, nbk - 1):
                            ref = np.zeros(g, dtype=np.uint64)
                            lib.entry(theta_np, pack_params(cfg, ch, keys, MODE_HONEST), seed, j, ref)
                            assert got[j].tolist() == ref.tolist(), f"entry_table j={j} k={k} r={r} g={g}"
    # multi-tile decompositions: 1 MB / r=2048 -> RT=4,CT=4 ; 1 MB / r=256 -> CT=2 with a 2-way k-split (H100/H200 SM
    # counts)
    words = 1 << 18
    cfg = MemoryConfig(words=words, encoding="vde", perm=True)
    raw = to_u64_tensor(rng.integers(0, 1 << 60, words, dtype=np.int64).view(np.uint64), dev)
    theta = torch.empty_like(raw)
    be.setup(raw, theta, to_u64_tensor(setup_params(words, 1, cfg.seed_t, 1, cfg.perm_seed, cfg.bits), dev))
    be.sync()
    theta_np = theta.cpu().numpy().view(np.uint64)
    keys = Keys.deterministic(9)
    for r in (2048, 256):
        ch = Challenge(block_bytes=1 << 20, n=2, r=r, nonce=int(rng.integers(0, 2**63)), j0=1, k=1024)
        exp = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(theta_np, pack_params(cfg, ch, keys, MODE_HONEST), exp, mt=True)
        prm = to_u64_tensor(pack_params(cfg, ch, keys, MODE_HONEST), dev)
        out = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
        state = torch.zeros(8, dtype=torch.int64, device=dev)
        parts = torch.zeros(264 * 1, dtype=torch.int64, device=dev)
        pw = be.power_buffer(ch.pw_count)
        mbuf = be.matrix_buffer(ch.mbuf_bytes)
        ablk = be.block_buffer(ch.block_bytes)
        be.chain_multi(
            theta,
            theta,
            out,
            prm,
            state,
            parts,
            pw,
            mbuf,
            264,
            256,
            ch.n,
            1024,
            r,
            ch.be,
            1,
            False,
            tensor=True,
            graph=False,
            ablk=ablk,
        )
        be.sync()
        assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), f"tensor large shape r={r}"
    # grouped entries at the real shape (1 MB, r=2048, g=512 -> x=256)
    cfg = MemoryConfig(words=words, encoding="vde", perm=True, x=256, tseed=0xBEEF)
    ch = Challenge(block_bytes=1 << 20, n=2, r=2048, nonce=int(rng.integers(0, 2**63)), j0=1, k=1024, g=512)
    exp = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(theta_np, pack_params(cfg, ch, keys, MODE_HONEST), exp, mt=True)
    prm = to_u64_tensor(pack_params(cfg, ch, keys, MODE_HONEST), dev)
    out = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
    state = torch.zeros(8, dtype=torch.int64, device=dev)
    parts = torch.zeros(264 * 512, dtype=torch.int64, device=dev)
    pw = be.power_buffer(ch.pw_count)
    mbuf = be.matrix_buffer(ch.mbuf_bytes)
    ablk = be.block_buffer(ch.block_bytes)
    be.chain_multi(
        theta,
        theta,
        out,
        prm,
        state,
        parts,
        pw,
        mbuf,
        264,
        256,
        ch.n,
        1024,
        2048,
        ch.be,
        512,
        True,
        tensor=True,
        graph=True,
        ablk=ablk,
    )
    be.sync()
    assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), "tensor grouped entries at 1 MB / r=2048 / g=512"
    # device image == C image, and prover-over-Theta' (imaged) == verifier-over-T at 1 MB / r=2048
    T_dev = torch.empty_like(theta)
    be.image(theta, T_dev, 0)
    be.sync()
    T_np = np.empty(words, dtype=np.uint64)
    lib.image(theta_np, T_np, 4)
    assert np.array_equal(T_dev.cpu().numpy().view(np.uint64), T_np), "device image != C image"
    cfg_p = MemoryConfig(words=words, encoding="vde", perm=True, image=True)
    cfg_v = MemoryConfig(words=words, encoding="vde", perm=True, image=False)
    # 2 MB array: blocks 0 and 1
    ch = Challenge(block_bytes=1 << 20, n=3, r=2048, nonce=int(rng.integers(0, 2**63)), j0=1, k=1024)
    exp = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(T_np, pack_params(cfg_v, ch, keys, MODE_HONEST), exp, mt=True)
    prm = to_u64_tensor(pack_params(cfg_p, ch, keys, MODE_HONEST), dev)
    out = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
    parts = torch.zeros(264, dtype=torch.int64, device=dev)
    pw = be.power_buffer(ch.pw_count)
    mbuf = be.matrix_buffer(ch.mbuf_bytes)
    be.chain_multi(
        theta,
        theta,
        out,
        prm,
        state,
        parts,
        pw,
        mbuf,
        264,
        256,
        ch.n,
        1024,
        2048,
        ch.be,
        1,
        False,
        tensor=True,
        graph=True,
        ablk=ablk,
    )
    be.sync()
    assert out.cpu().numpy().view(np.uint64).tolist() == exp.tolist(), (
        "prover imaging Theta' on the GPU != verifier over T"
    )
    lib.pool_stop()


def test_armed_chain_matches_launch():
    """The pre-armed graph (wait node + chain + publish) must give the same digests as the direct launch."""
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device")
    from memory_accounting.core.memory import Hbm
    from memory_accounting.prover.strategies import HonestProver

    be = DeviceBackend(dev)
    cfg = MemoryConfig(words=1 << 17, encoding="vde", perm=False)
    hbm = Hbm(be, cfg)
    keys = Keys.deterministic(3)
    prover = HonestProver(hbm, keys)
    for n, r in ((4, 0), (3, 256)):
        ch = Challenge(1 << 16, n, r, nonce=0xABCDEF + n, j0=1, k=1024)
        lo, hi, trace, _ = prover.respond(ch)
        prover.arm(Challenge(ch.block_bytes, ch.n, ch.r, 0, 0, ch.k, ch.g))
        lo2, hi2, trace2, _ = prover.fire(ch)
        assert (lo, hi, trace) == (lo2, hi2, trace2), (n, r)
        # abort path: arm, disarm, then a normal response still works
        prover.arm(Challenge(ch.block_bytes, ch.n, ch.r, 0, 0, ch.k, ch.g))
        prover.disarm()
        assert prover.respond(ch)[:2] == (lo, hi)
        # abort with a poisoned parameter block (a freed segment table, an array far larger than the buffer):
        # the aborted graph must not read through it (before the fix it ran the chain on these parameters)
        prover.arm(Challenge(ch.block_bytes, ch.n, ch.r, 0, 0, ch.k, ch.g))
        bad = pack_params(cfg, ch, keys, MODE_HONEST, nb=1 << 40, nseg=3, segtab=0x10)
        prover._prm_np[:] = np.asarray(bad).view(np.int64)
        prover.disarm()
        torch.cuda.synchronize()  # an illegal address would surface here
        assert prover.respond(ch)[:2] == (lo, hi)


def test_tensor_path_needs_a_matmul():
    """The tensor-core kernel is the matmul: at r = 0 it has no columns and would write zero entries without an
    error.  entry_table used to default to tensor=True; now None picks the tensor path exactly when r > 0, and
    tensor=True at r = 0 is refused by every entry point (checked here without a GPU, on a recording stub)."""
    calls = []

    class Rec:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, a[-1] if name == "entry_table" else a[17]))

    be = object.__new__(DeviceBackend)
    be._lib = Rec()
    be.device = "cuda"
    empty = torch.empty(0, dtype=torch.int8)
    for r, expect in ((0, False), (16, True)):
        be.entry_table(*[None] * 9, seed=1, j0=0, nblocks=1, groups=1, threads=32, k=128, r=r, be=256, g=1)
        assert calls[-1] == ("entry_table", expect), (r, calls[-1])
    for call in (
        lambda: be.entry_table(*[None] * 9, 1, 0, 1, 1, 32, 128, 0, 256, 1, tensor=True),
        lambda: be.chain_multi(*[None] * 8, 1, 32, 1, 128, 0, 256, 1, tensor=True, ablk=empty),
        lambda: be.chain_multi_armed(*[None] * 8, 1, 32, 1, 128, 0, 256, 1, False, True, empty, *[empty] * 4),
    ):
        n = len(calls)
        try:
            call()
            raise AssertionError("tensor=True accepted at r = 0")
        except ValueError:
            assert len(calls) == n, "the kernel was called anyway"
    be.chain_multi(*[None] * 8, 1, 32, 1, 128, 0, 256, 1, ablk=empty)  # default: scalar path
    assert calls[-1] == ("chain_multi", False)
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device: the backend's checks passed, the extension's did not run")
    be = DeviceBackend(dev)

    def z(n, dt=torch.int64):
        return torch.zeros(n, dtype=dt, device=dev)

    try:  # the extension refuses it too
        be._lib.entry_table(
            z(1024),
            z(1024),
            z(30),
            z(4),
            z(32),
            z(128),
            z(256, torch.int8),
            z(4096, torch.int8),
            z(4),
            1,
            0,
            1,
            32,
            256,
            128,
            0,
            512,
            1,
            True,
        )
        raise AssertionError("extension accepted the tensor path at r = 0")
    except RuntimeError as e:
        assert "r > 0" in str(e), e


def test_failed_capture_leaves_stream_usable():
    """A chain whose graph capture fails (here an invalid launch shape: 2048 threads per block) must raise and
    leave the side stream out of capture mode, so the next challenge works.  Before, nothing ended the capture
    and every later challenge on that device failed until the prover restarted.  Plain and pre-armed graphs."""
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device")
    from memory_accounting.core.memory import Hbm
    from memory_accounting.prover.strategies import HonestProver

    be = DeviceBackend(dev)
    cfg = MemoryConfig(words=1 << 17, encoding="vde", perm=False)
    hbm = Hbm(be, cfg)
    keys = Keys.deterministic(13)
    for kernel in ("tensor", "multi"):
        prover = HonestProver(hbm, keys, kernel=kernel)
        for r in (0, 256):
            ch = Challenge(1 << 16, 3, r, nonce=0x5151 + r, j0=1, k=1024)
            good = prover.respond(ch)[:3]
            prover.threads = 2048
            for attempt in (
                lambda: prover.respond(ch),
                lambda: prover.arm(Challenge(ch.block_bytes, ch.n, ch.r, 0, 0, ch.k, ch.g)),
            ):
                try:
                    attempt()
                    raise AssertionError(f"{kernel} r={r}: a 2048-thread launch was captured")
                except RuntimeError:
                    pass
                prover._armed_shape = None
            prover.threads = 256
            assert prover.respond(ch)[:3] == good, f"{kernel} r={r}: chain after a failed capture"
            prover.arm(Challenge(ch.block_bytes, ch.n, ch.r, 0, 0, ch.k, ch.g))
            assert prover.fire(ch)[:3] == good, f"{kernel} r={r}: armed chain after a failed capture"
        prover.close()


def test_partial_residency_matches_honest():
    """The partial cheat reads the tail of the array from a pinned host copy through its own segment
    table; its digests must equal the honest prover's (same data, different placement)."""
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device")
    from memory_accounting.core.memory import Hbm
    from memory_accounting.prover.responder import ResidentArray
    from memory_accounting.prover.strategies import HonestProver, PartialProver

    be = DeviceBackend(dev)
    cfg = MemoryConfig(words=(4 << 20) // 8, encoding="vde", perm=False)  # 4 MiB: four 1 MiB blocks
    hbm = Hbm(be, cfg)
    keys = Keys.deterministic(5)
    res = ResidentArray(
        [(hbm.theta_prime.data_ptr(), hbm.theta_prime.numel(), hbm.theta_prime)],
        cfg.seed_t,
        "cuda",
        strategies=("honest", "partial"),
        partial_fraction=0.25,
    )
    assert res.partial_nseg == 2 and res.partial_table is not None
    h, p = HonestProver(res, keys), PartialProver(res, keys)
    for nonce in (11, 12, 13):
        ch = Challenge(1 << 20, 6, 0, nonce=nonce, j0=0, k=1024)
        assert h.respond(ch)[:2] == p.respond(ch)[:2]
    h.close()
    p.close()


def test_lora_modified_blocks_change_digests():
    """A prover with a fraction of blocks altered gives wrong digests exactly for chains that touch them."""
    dev = pick_device()
    if dev != "cuda":
        skip("no CUDA device")
    from memory_accounting.core.memory import Hbm
    from memory_accounting.prover.responder import ResidentArray
    from memory_accounting.prover.strategies import HonestProver, LoraProver

    be = DeviceBackend(dev)
    cfg = MemoryConfig(words=(8 << 20) // 8, encoding="vde", perm=False)  # eight 1 MiB blocks
    hbm = Hbm(be, cfg)
    keys = Keys.deterministic(7)
    res = ResidentArray(
        [(hbm.theta_prime.data_ptr(), hbm.theta_prime.numel(), hbm.theta_prime)],
        cfg.seed_t,
        "cuda",
        strategies=("honest", "lora"),
        lora_fraction=0.25,
    )
    assert len(res.lora_blocks) == 2
    honest, lora = HonestProver(res, keys), LoraProver(res, keys)
    # a chain starting on a modified block must differ; one that starts elsewhere differs only if it lands
    # on one later, so over many nonces some agree and some do not (expected mismatch rate 1 - 0.75^3 = 58 %)
    diffs = 0
    for nonce in range(20, 44):
        ch = Challenge(1 << 20, 3, 0, nonce=nonce, j0=nonce % 8, k=1024)
        same = lora.respond(ch)[:2] == honest.respond(ch)[:2]
        if ch.j0 in res.lora_blocks:
            assert not same, nonce
        diffs += not same
    assert 4 <= diffs <= 22, diffs
    honest.close()
    lora.close()


def test_armed_flag_protocol_has_barriers_on_both_sides():
    """The pre-armed chain's flag protocol assumed a TSO host: fire() wrote the parameters and then the flag with
    plain numpy stores and read the digests right after the done spin, and the GPU polled the flag with a
    volatile load. On a weakly ordered host (aarch64) the GPU could see the flag before the parameters. Now the
    flag is stored behind a system-scope barrier (C), the host fences after done, and the wait kernel reads the
    flag with ld.acquire.sys. (Here without a GPU: the C helpers, fire() through them over a graph that has
    already finished, and the kernel source.)"""
    from types import SimpleNamespace

    from memory_accounting.core.params import P_FIELDS
    from memory_accounting.prover.strategies import DeviceProver

    flag = np.zeros(1, np.int64)
    lib.flag_store_release(flag, 2)
    lib.fence_acquire()
    assert flag[0] == 2
    prover = DeviceProver.__new__(DeviceProver)
    prover.hbm, prover.keys, prover.name, prover.kernel = (
        SimpleNamespace(cfg=MemoryConfig(words=1 << 14, perm=False)),
        Keys.deterministic(3),
        "honest",
        "tensor",
    )
    ch = Challenge(1 << 12, 3, 0, nonce=77, j0=1, k=128)
    prover._armed_shape = (ch.block_bytes, ch.n, ch.r, ch.k, ch.g)
    prover._flag_np, prover._done_np, prover._prm_np = (
        np.zeros(1, np.int64),
        np.ones(1, np.int64),
        np.zeros(len(P_FIELDS), np.int64),
    )
    prover._out_dst = torch.arange(2 + MAX_LINKS, dtype=torch.int64)
    lo, hi, trace, _ = prover.fire(ch)
    assert prover._flag_np[0] == 1 and (lo, hi, trace[: ch.n]) == (0, 1, [2, 3, 4])
    assert np.array_equal(
        prover._prm_np, np.asarray(pack_params(prover.hbm.cfg, ch, prover.keys, MODE_HONEST)).view(np.int64)
    )
    src = (Path(__file__).resolve().parents[1] / "memory_accounting/kernels/posm.cu").read_text()
    wait = src[src.index("__global__ void posm_wait_kernel") :]
    assert "ld.acquire.sys" in src and "ld_acquire_sys(flag)" in wait[: wait.index("__syncthreads")]


def test_encoded_memory_codec():
    """Encoded dynamic memory (memory_accounting.prover.kvenc, v0.17): a unit is its plaintext packed 15 words to 16 and each payload
    word VDE-encoded at its region position in a tweak domain of its own (KV_POS); decoding gives the plaintext back,
    any group range or word range agrees with the whole unit's encoding, empty memory is the encoding of zeros word by
    word, and no word equals the same payload encoded at the same position without the domain (as Θ' and the filler
    are). The CUDA kernel (entries, per-unit offsets) agrees with the C library."""
    from memory_accounting.prover import kvenc as K

    key = (0x1234 << 64) | 0xABCDEF
    plain = rng.integers(0, 2**63, 100, dtype=np.uint64)
    q0 = 777
    e = K.encode_np(plain, q0, key)
    assert len(e) == K.enc_words(100) == 112
    pad = np.zeros(105, dtype=np.uint64)
    pad[:100] = plain
    pay = np.zeros(112, dtype=np.uint64)
    lib.pack(pad, pay, 7)
    ref = np.empty(112, dtype=np.uint64)
    lib.setup_at(pay, ref, key, K.KV_POS | q0)
    assert np.array_equal(e, ref)
    plain_pos = np.empty(112, dtype=np.uint64)
    lib.setup_at(pay, plain_pos, key, q0)
    assert not np.any(e == plain_pos), "an encoded word equals the same payload at the same position outside the domain"
    assert np.array_equal(K.decode_np(e, q0, key, 100), plain)
    assert np.array_equal(K.encode_np(plain, q0, key, 2, 3), e[32:80])
    assert np.array_equal(K.encode_words_np(plain, q0, key, 5, 50), e[5:50])
    assert np.array_equal(K.encode_np(plain, q0, key, threads=4), e)
    zeros = K.empty_np(112, q0, key)
    assert np.array_equal(K.encode_np(np.zeros(100, dtype=np.uint64), q0, key), zeros)
    assert np.array_equal(K.empty_np(40, q0 + 30, key), zeros[30:70]), "empty words do not depend on their group"
    codec = K.Codec(key, "cpu")
    raw = torch.from_numpy(plain.view(np.int64).copy())
    enc = torch.zeros(112, dtype=torch.int64)
    codec.encode_range(raw, enc, q0, 0, 100)
    assert np.array_equal(enc.numpy().view(np.uint64), e)
    out = torch.zeros(100, dtype=torch.int64)
    codec.decode_range(enc, q0, out, 31, 47)
    assert torch.equal(out[30:60], raw[30:60]) and int(out[:30].abs().sum()) == 0, (
        "a decoded range is its groups, nothing else"
    )
    codec.empty_(enc, q0)
    assert np.array_equal(enc.numpy().view(np.uint64), zeros)
    if not torch.cuda.is_available():
        skip("no CUDA device: the CPU codec passed, the GPU's did not run")
    g = K.Codec(key, "cuda")
    units = 5
    rawd = torch.from_numpy(rng.integers(0, 2**63, units * 100, dtype=np.uint64).view(np.int64)).cuda()
    encd = torch.zeros(units * 112, dtype=torch.int64, device="cuda")
    rows = [(u * 100, u * 112, 0, 7) for u in range(units)]
    g.code(rawd.data_ptr(), encd.data_ptr(), q0, g.ent(rows), 100, False)
    rawh = rawd.cpu().numpy().view(np.uint64)
    for u in range(units):
        assert np.array_equal(
            encd[u * 112 : (u + 1) * 112].cpu().numpy().view(np.uint64),
            K.encode_np(rawh[u * 100 : (u + 1) * 100], q0 + u * 112, key),
        )
    back = torch.zeros_like(rawd)
    g.code(back.data_ptr(), encd.data_ptr(), q0, g.ent(rows), 100, True)
    assert torch.equal(back, rawd)
    z = torch.zeros(1000, dtype=torch.int64, device="cuda")
    K.empty_(z, q0 + 3, key)  # whole groups through the kernel, a tail through setup_at
    assert np.array_equal(z.cpu().numpy().view(np.uint64), K.empty_np(1000, q0 + 3, key))


def test_encoded_memory_rebuilt_on_demand():
    """Kernel mode 4 (the dynamic sweep's dyn_encode_otf, memory_accounting.prover.strategies.DynShareCheat): words [qcheat, qcheat_hi)
    of an encoded dynamic region are rebuilt from their payload at their region position in the encoded memory's domain,
    the rest read honestly; the digest is honest's, and a wrong payload changes it."""
    from memory_accounting.core.params import MODE_DYN_ENCODE_OTF
    from memory_accounting.prover.kvenc import KV_POS

    key, W = (77 << 64) | 12345, 1 << 14
    pay = rng.integers(0, 1 << 60, W, dtype=np.int64).view(np.uint64)
    enc = np.empty(W, dtype=np.uint64)
    lib.setup_at(pay, enc, key, KV_POS)
    cfg = MemoryConfig(words=W, encoding="none", perm=False, image=False)
    keys, ch = Keys.deterministic(1), Challenge(4096, 8, 0, 99, 3, 1024)
    out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)
    lib.chain(enc, pack_params(cfg, ch, keys, MODE_HONEST), out)
    honest = (int(out[0]), int(out[1]))
    prm = pack_params(cfg, ch, keys, MODE_DYN_ENCODE_OTF, qcheat=W // 4, qcheat_hi=3 * W // 4, ckey=key)
    lib.chain_scalar(enc, pay.copy(), prm, out, 1)
    assert (int(out[0]), int(out[1])) == honest
    bad = pay.copy()
    bad[W // 4 : 3 * W // 4] ^= np.uint64(1)
    lib.chain_scalar(enc, bad, prm, out, 1)
    assert (int(out[0]), int(out[1])) != honest


if __name__ == "__main__":
    from _runner import run

    run(globals())
