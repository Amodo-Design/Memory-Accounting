"""Parameter blocks shared between Python, C and CUDA: u64 arrays whose fields, in order, are the members of
posm_params and posm_setup_params in posm_core.h."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import numpy as np

from .codec_profile import GROUP_ENC, FIELD_BITS

P_FIELDS = [
    "nb",
    "be",
    "r",
    "n",
    "nonce",
    "j0",
    "k0",
    "k1",
    "polyk",
    "mode",
    "enc",
    "seed_t",
    "seed_c",
    "perm_on",
    "perm_seed",
    "perm_bits",
    "d0_lo",
    "d0_hi",
    "link0",
    "k",
    "nseg",
    "segtab",
    "seed_t_hi",
    "seed_c_hi",
    "x",
    "g",
    "tseed",
    "s0",
    "img",
    "qoff",
    "qcheat",
    "canonp",
    "jstop",
    "qcheat_hi",
    "jstop_hi",
]
SETUP_FIELDS = ["count", "enc", "key_lo", "perm_on", "perm_seed", "perm_bits", "key_hi"]
M64 = (1 << 64) - 1
PM61 = (1 << 61) - 1
MODE_HONEST, MODE_OTHER_GPU, MODE_ENCODE_OTF = 0, 1, 3
MODE_DYN_ENCODE_OTF = (
    4  # encoded dynamic memory (memory_accounting.prover.kvenc) rebuilt from its payload on demand (the dynamic sweep's cheat)
)
MODE_DYN_OTHER_GPU = 5  # rekey a dynamic row held for another GPU, using the same packed codec
MODE_ABORT = 0xAB  # an aborted pre-armed chain (POSM_MODE_ABORT in posm_core.h): every chain kernel returns at once
KMIN, KMAX, RMAX, CPT, HW = 128, 2048, 4096, 8, 4
GMAX = 1024
MAX_LINKS = 4096


def T1LEN(r):
    return r if r else 64


def _rand61() -> int:
    return secrets.randbits(61) % PM61 or 1


HASH_KEY_DOMAIN = 0x484153484B455931  # "HASHKEY1": the challenge-hash keys, from the verifier's seed


@dataclass(frozen=True)
class Keys:
    """The challenge-hash keys (SipHash key + polynomial key), known to the prover and the verifier.

    In the scheme they are derived by both sides from the verifier's seed (`Keys.from_seed`): the prover has no
    say in them and nothing is sent. The random default and `deterministic` are for tests, benchmarks and the
    synthetic provers, where one process or one --seed stands for both sides."""

    k0: int = field(default_factory=lambda: secrets.randbits(64))
    k1: int = field(default_factory=lambda: secrets.randbits(64))
    polyk: int = field(default_factory=_rand61)

    @classmethod
    def from_seed(cls, seed: int) -> Keys:
        """The keys both sides derive from the verifier's seed (the filler seed it mints at setup): a PRF of the
        seed (SipHash-2-4-128, its own domain, like `memory_accounting.setup.filler.static_filler_key` and
        `memory_accounting.prover.dynfill.dyn_key`). One set for all of a prover's GPUs, as before; the seed is the
        verifier's, so the verifier never takes the keys from a manifest (it used to: the prover chose them)."""
        from .reference import siphash24_128

        s = int(seed)
        if not 0 <= s <= M64:
            raise ValueError(f"seed {seed!r} is not a 64-bit non-negative integer")
        k0, k1 = siphash24_128(s, HASH_KEY_DOMAIN, 0, 0, 0, 0)
        lo, _ = siphash24_128(s, HASH_KEY_DOMAIN, 1, 0, 0, 0)
        return cls(k0, k1, lo % (PM61 - 1) + 1)  # 0 < polyk < p

    def as_list(self) -> list[int]:
        """[k0, k1, polyk], as a manifest records them."""
        return [self.k0, self.k1, self.polyk]

    @classmethod
    def deterministic(cls, seed: int) -> Keys:
        rng = np.random.default_rng(seed)
        return cls(int(rng.integers(0, 2**63)), int(rng.integers(0, 2**63)), int(rng.integers(1, PM61)))


@dataclass(frozen=True)
class MemoryConfig:
    """What the challenged GPU stores and how it was derived from the canonical weights."""

    words: int  # 8-byte words in the challenged array
    encoding: str = "vde"  # "vde" (incompressible) or "none" (MVP-PoSM: trusted weights)
    perm: bool = True  # per-GPU Feistel permutation of element positions (needs a power-of-two length);
    # the per-GPU encoding seed already makes copies distinct, so real deployments use perm=False
    seed_t: int = 0xA11CE  # 128-bit encoding key of the challenged GPU (prover secret in the confidential setting)
    seed_c: int = 0xC0FFEE  # 128-bit encoding key of the cheater's canonical (other-GPU) copy
    perm_seed: int = 0x5EED
    x: int = 0  # per-link public seeds (0 = link seed is the challenge nonce; the default and recommended)
    tseed: int = 0  # seed of the public seed list when x > 0
    image: bool = False  # prover side: hash each 32-byte group of the array before use (T = H(Theta')).
    # The verifier holds T itself and runs with image=False.

    def __post_init__(self):
        if FIELD_BITS == 89 and (self.perm or self.words % GROUP_ENC):
            raise ValueError("grouped 89-bit memory needs unpermuted, complete codec groups")
        if self.perm and self.words & (self.words - 1):
            raise ValueError("words must be a power of two when the Feistel permutation is on")
        if self.words <= 0:
            raise ValueError("words must be positive")
        if self.encoding not in ("vde", "none"):
            raise ValueError(self.encoding)

    @property
    def bits(self) -> int:
        return max(self.words - 1, 1).bit_length()

    @property
    def enc(self) -> int:
        return 1 if self.encoding == "vde" else 0

    @property
    def nbytes(self) -> int:
        return self.words * 8


@dataclass(frozen=True)
class Challenge:
    """One verifier challenge: a chain of n links over B-byte blocks.  Each block is viewed as rows of
    k bytes (pieces) multiplied by a nonce-derived k x r int8 matrix; r = 0 hashes the words directly.
    Multiply-adds per byte = r, hashed output = bytes * r / k values."""

    block_bytes: int
    n: int
    r: int
    nonce: int
    j0: int
    k: int = 1024
    g: int = 1  # entry groups per block: T[j] has g words (d = 8g bytes)

    def __post_init__(self):
        # sizes first, before any arithmetic on them: these arrive from the network
        if self.k % 16 or not KMIN <= self.k <= KMAX:
            raise ValueError(f"k must be a multiple of 16 in [{KMIN}, {KMAX}]")
        if self.block_bytes <= 0 or self.block_bytes % 32:
            raise ValueError("block_bytes must be a positive multiple of 32")
        if self.r < 0:
            raise ValueError("r must not be negative")
        if not 1 <= self.g <= GMAX:
            raise ValueError(f"g must be in [1, {GMAX}]")
        units = (self.block_bytes // self.k) if self.r else (self.block_bytes // 8)
        if units % self.g:
            raise ValueError("g must divide the number of rows (r > 0) or words (r = 0) per block")
        if self.r == 0 and (units // self.g) % HW:
            raise ValueError("words per group must be a multiple of 4 when r = 0")
        if self.r and (self.r % 16 or self.r > RMAX):
            raise ValueError(f"r must be 0 or a multiple of 16 up to {RMAX}")
        if self.r and self.block_bytes % (16 * self.k):
            raise ValueError("block_bytes must be a multiple of 16k (whole 16-row tiles of k bytes)")
        if not 1 <= self.n <= MAX_LINKS:
            raise ValueError(f"n must be in [1, {MAX_LINKS}] (output buffer size)")

    @property
    def be(self) -> int:
        return self.block_bytes // 8

    @property
    def rows(self) -> int:
        return self.block_bytes // self.k

    @property
    def span(self) -> int:
        """Rows (r > 0) or words (r = 0) per entry group."""
        return (self.rows if self.r else self.be) // self.g

    @property
    def pw_count(self) -> int:
        """Entries in the power table: seed part T1 | T2, then the nonce fold powers (g)."""
        t2 = self.span if self.r else -(-self.span // 64)
        return T1LEN(self.r) + t2 + self.g

    @property
    def mbuf_bytes(self) -> int:
        """Bytes of the per-chain matrix buffer (tiled int8 matrix, pad, int32 column constants)."""
        return (self.k * self.r + 255) // 256 * 256 + 4 * self.r if self.r else 256

    @classmethod
    def random(cls, block_bytes: int, n: int, r: int, nb: int, rng=None, k: int = 1024, g: int = 1) -> Challenge:
        """Reproducible challenges for local sweeps and tests. Not for a verifier facing a prover: numpy's
        generator can be predicted from its outputs or its seed; use `fresh`."""
        rng = rng or np.random.default_rng()
        return cls(block_bytes, n, r, int(rng.integers(0, 2**63)), int(rng.integers(0, nb)), k, g)

    @classmethod
    def fresh(cls, block_bytes: int, n: int, r: int, nb: int, k: int = 1024, g: int = 1) -> Challenge:
        """A verifier's challenge: nonce and start block from the OS's CSPRNG, so a prover that has seen
        every earlier challenge cannot predict the next one and compute it ahead of the timed window."""
        return cls(block_bytes, n, r, secrets.randbits(63), secrets.randbelow(nb), k, g)


def pack_params(
    mem: MemoryConfig,
    ch: Challenge,
    keys: Keys,
    mode: int,
    *,
    nb: int | None = None,
    d0: tuple[int, int] = (0, 0),
    link0: int = 0,
    n: int | None = None,
    j0: int | None = None,
    nseg: int = 0,
    segtab: int = 0,
    s0: int = 0,
    qoff: int = 0,
    qcheat: int = 0,
    canonp: int = 0,
    jstop: int = 0,
    qcheat_hi: int = 0,
    jstop_hi: int = 0,
    ckey: int | None = None,
    tkey: int | None = None,
) -> np.ndarray:
    """Return the u64 parameter block as an int64 numpy array (bit-identical view).
    nseg/segtab: the honest array as segments (segtab = address of the segment table as an integer).
    qcheat/canonp/jstop: a cheat on a share of the array (memory_accounting.prover.strategies.ShareCheat; posm_core.h); qcheat_hi
    ends the share before the array does; ckey: the cheat's key in place of the canonical copy's (mode 4)."""
    nb = nb if nb is not None else mem.words // ch.be
    if mem.x > 0 and ch.r > 0 and mem.x * ch.r < ch.k:
        # public per-link seeds: with x*r < k the x public k x r matrices have a common kernel, and a block
        # changed by a vector in it gives the same digests forever
        raise ValueError(f"x * r must be at least k with public link seeds (x={mem.x}, r={ch.r}, k={ch.k})")
    if nb < 1:
        raise ValueError("array smaller than one block")
    start = ch.j0 if j0 is None else j0
    if not 0 <= start < nb:
        raise ValueError(f"start block {start} out of range for {nb} blocks")
    vals = {
        "nb": nb,
        "be": ch.be,
        "r": ch.r,
        "n": ch.n if n is None else n,
        "nonce": ch.nonce,
        "j0": ch.j0 if j0 is None else j0,
        "k0": keys.k0,
        "k1": keys.k1,
        "polyk": keys.polyk,
        "mode": mode,
        "enc": mem.enc,
        "seed_t": (mem.seed_t if tkey is None else tkey) & M64,
        "seed_c": mem.seed_c & M64,
        "perm_on": int(mem.perm),
        "perm_seed": mem.perm_seed,
        "perm_bits": mem.bits,
        "d0_lo": d0[0],
        "d0_hi": d0[1],
        "link0": link0,
        "k": ch.k,
        "nseg": nseg,
        "segtab": segtab,
        "seed_t_hi": (mem.seed_t if tkey is None else tkey) >> 64,
        "seed_c_hi": mem.seed_c >> 64,
        "x": mem.x,
        "g": ch.g,
        "tseed": mem.tseed,
        "s0": s0,
        "img": int(mem.image),
        "qoff": qoff,
        "qcheat": qcheat,
        "canonp": canonp,
        "jstop": jstop,
        "qcheat_hi": qcheat_hi,
        "jstop_hi": jstop_hi,
    }
    if ckey is not None:
        vals["seed_c"], vals["seed_c_hi"] = ckey & M64, ckey >> 64
    arr = np.array([vals[k] & 0xFFFFFFFFFFFFFFFF for k in P_FIELDS], dtype=np.uint64)
    return arr.view(np.int64)


def setup_params(count: int, enc: int, key: int, perm_on: int, perm_seed: int, bits: int) -> np.ndarray:
    """Setup/decode parameter block (posm_setup_params). key may be 128-bit."""
    vals = {
        "count": count,
        "enc": enc,
        "key_lo": key & M64,
        "perm_on": perm_on,
        "perm_seed": perm_seed,
        "perm_bits": bits,
        "key_hi": key >> 64,
    }
    arr = np.array([vals[k] for k in SETUP_FIELDS], dtype=np.uint64)
    return arr.view(np.int64)


def table_seed(tseed: int, i: int) -> int:
    """Public link seed i of the table scheme (matches table_seed_i in posm_core.h)."""
    from .reference import splitmix64

    return splitmix64(tseed ^ i)


def parse_bytes(text: str) -> int:
    """'64K', '1M', '2G', '512' -> bytes."""
    t = str(text).strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if t and t[-1] in mult:
        return int(float(t[:-1]) * mult[t[-1]])
    return int(t)


def array_from_args(args):
    """The challenged array and hash keys a prover or verifier command names: a synthetic array from --size/--seed, or the
    setup-phase manifest (--manifest). With a manifest the keys are derived from the filler seed (`Keys.from_seed`):
    the command's own --filler-seed (the verifier's: the seed it minted), else the manifest's (the prover's copy of
    it; a verifier without --filler-seed is warned that it trusts the manifest, `memory_accounting.verifier.net`). The manifest's
    `hash_key` field is never read for them."""
    if args.manifest:
        from ..setup.manifest import Manifest

        m = Manifest.load(args.manifest, allow_benchmark=getattr(args, "skip_response_checks", False))
        fs = getattr(args, "filler_seed", None)
        return m, Keys.from_seed(int(fs, 16) if fs else int(m.filler_seed))
    cfg = MemoryConfig(words=parse_bytes(args.size) // 8, encoding=args.encoding,
                       perm=not args.no_perm and FIELD_BITS != 89)
    return cfg, Keys.deterministic(args.seed)
