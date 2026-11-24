"""Pure-Python reference of every primitive in posm_core.h (slow; for tests only)."""

from __future__ import annotations

from .params import P_FIELDS, PM61

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15


def splitmix64(z: int) -> int:
    z = (z + GOLD) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def red61(x: int) -> int:
    x = (x & PM61) + (x >> 61)
    return x - PM61 if x >= PM61 else x


def tweak_at(key: int, q: int) -> int:
    """Keyed PRF tweak: SipHash-2-4-128(key_lo, key_hi; q, domain).lo reduced into F_p. key may be 128-bit."""
    from .codec_profile import TWEAK
    offset = q % 8 if TWEAK == "group64" else 0
    lo, _ = siphash24_128(key & M64, key >> 64, q-offset, 0x5457454B00000001, 0, 0)
    return (red61(lo & PM61) + offset) % PM61


def vde_encode(d: int, t: int) -> int:
    x = (d + t) % PM61
    r = pow(x, 1 << 59, PM61)
    f = 1 if (r * r) % PM61 != x else 0
    return r | (f << 61)


def vde_decode(w: int, t: int) -> int:
    r = w & PM61
    x = (r * r) % PM61
    if (w >> 61) & 1:
        x = (PM61 - x) % PM61
    return (x - t) % PM61


def feistel_perm(q: int, seed: int, bits: int) -> int:
    if bits == 0:
        return 0
    lb, rb = bits // 2, bits - bits // 2
    lmask, rmask = (1 << lb) - 1, (1 << rb) - 1
    L, R = (q >> rb) & lmask, q & rmask
    for i in range(4):
        R ^= splitmix64(L ^ ((seed + 2 * i * GOLD) & M64)) & rmask
        L ^= splitmix64(R ^ ((seed + (2 * i + 1) * GOLD) & M64)) & lmask
    return (L << rb) | R


def _rotl(x, b):
    return ((x << b) | (x >> (64 - b))) & M64


def _sipround(v):
    v0, v1, v2, v3 = v
    v0 = (v0 + v1) & M64
    v1 = _rotl(v1, 13)
    v1 ^= v0
    v0 = _rotl(v0, 32)
    v2 = (v2 + v3) & M64
    v3 = _rotl(v3, 16)
    v3 ^= v2
    v0 = (v0 + v3) & M64
    v3 = _rotl(v3, 21)
    v3 ^= v0
    v2 = (v2 + v1) & M64
    v1 = _rotl(v1, 17)
    v1 ^= v2
    v2 = _rotl(v2, 32)
    return [v0, v1, v2, v3]


def siphash24_128(k0, k1, m0, m1, m2, m3):
    v = [k0 ^ 0x736F6D6570736575, k1 ^ 0x646F72616E646F6D, k0 ^ 0x6C7967656E657261, k1 ^ 0x7465646279746573]
    v[1] ^= 0xEE
    for m in (m0, m1, m2, m3):
        v[3] ^= m
        v = _sipround(_sipround(v))
        v[0] ^= m
    b = 32 << 56
    v[3] ^= b
    v = _sipround(_sipround(v))
    v[0] ^= b
    v[2] ^= 0xEE
    for _ in range(4):
        v = _sipround(v)
    lo = v[0] ^ v[1] ^ v[2] ^ v[3]
    v[1] ^= 0xDD
    for _ in range(4):
        v = _sipround(v)
    return lo, v[0] ^ v[1] ^ v[2] ^ v[3]


def chain_polyk(k0, k1, nonce, polyk_secret):
    lo, _ = siphash24_128(k0, k1, nonce, polyk_secret, 0x504F4C594B455900, 0)
    k = red61(lo & PM61)
    return k or 1


def chain_mixkey(k0, k1, nonce, polyk_secret):
    _, hi = siphash24_128(k0, k1, nonce, polyk_secret, 0x504F4C594B455900, 0)
    return hi


FOLD_DOMAIN = 0x464F4C444B455900


def chain_foldk(k0, k1, nonce, polyk_secret):
    """Key of the group fold, own domain: H = sum_g T[g] * K_n * K_f^g (K_f^g, not K_n^g, so that at x = 0, where
    the entry key K_s is K_n, group g's word loc and group g+1's word loc-1 are different monomials)."""
    lo, _ = siphash24_128(k0, k1, nonce, polyk_secret, FOLD_DOMAIN, 0)
    return red61(lo & PM61) or 1


def mix_output(v: int, mixkey: int) -> int:
    return red61(splitmix64((v & 0xFFFFFFFF) ^ mixkey) & PM61)


def matrix(nonce: int, K: int, r: int) -> list[list[int]]:
    """M[j][c] for j < K, c < r: signed byte (j*r + c) % 8 of splitmix64(seed + (j*r + c) // 8)."""
    ms = splitmix64(nonce ^ 0x4D41545249580001)
    flat = []
    for i in range(0, K * r, 8):
        g = splitmix64((ms + i // 8) & M64)
        for j in range(8):
            b = (g >> (8 * j)) & 0xFF
            flat.append(b - 256 if b >= 128 else b)
    return [flat[j * r : (j + 1) * r] for j in range(K)]


M32 = (1 << 32) - 1
B3_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A)
B3_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
B3_CHUNK_START, B3_CHUNK_END, B3_ROOT, B3_KEYED_HASH = 1, 2, 8, 16
# BLAKE3 derive_key(context="posm 2026-10 image v1", key_material=b""), as 8 little-endian u32 words
IMG_KEY = bytes.fromhex("f57be4c5137fa6167fa7a2f648754a9cefb52f5122991c15705741611b86e000")


def _rotr32(x, b):
    return ((x >> b) | (x << (32 - b))) & M32


def _b3g(v, a, b, c, d, x, y):
    v[a] = (v[a] + v[b] + x) & M32
    v[d] = _rotr32(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & M32
    v[b] = _rotr32(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + y) & M32
    v[d] = _rotr32(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & M32
    v[b] = _rotr32(v[b] ^ v[c], 7)


def blake3_compress(cv: list[int], block: bytes, counter: int, block_len: int, flags: int) -> list[int]:
    """BLAKE3 compression (the reference implementation's round-then-permute form): 16 output words."""
    m = [int.from_bytes(block[4 * i : 4 * i + 4], "little") for i in range(16)]
    v = list(cv) + list(B3_IV) + [counter & M32, counter >> 32, block_len, flags]
    for rnd in range(7):
        _b3g(v, 0, 4, 8, 12, m[0], m[1])
        _b3g(v, 1, 5, 9, 13, m[2], m[3])
        _b3g(v, 2, 6, 10, 14, m[4], m[5])
        _b3g(v, 3, 7, 11, 15, m[6], m[7])
        _b3g(v, 0, 5, 10, 15, m[8], m[9])
        _b3g(v, 1, 6, 11, 12, m[10], m[11])
        _b3g(v, 2, 7, 8, 13, m[12], m[13])
        _b3g(v, 3, 4, 9, 14, m[14], m[15])
        if rnd < 6:
            m = [m[B3_PERM[i]] for i in range(16)]
    return [v[i] ^ v[i + 8] for i in range(8)] + [v[i + 8] ^ cv[i] for i in range(8)]


def blake3_keyed_hash_block(key: bytes, msg: bytes) -> bytes:
    """BLAKE3 keyed_hash(key, msg)[:32] for a message of at most one block (64 bytes): one compression."""
    assert len(key) == 32 and len(msg) <= 64
    cv = [int.from_bytes(key[4 * i : 4 * i + 4], "little") for i in range(8)]
    out = blake3_compress(
        cv, msg.ljust(64, b"\0"), 0, len(msg), B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED_HASH
    )
    return b"".join(x.to_bytes(4, "little") for x in out[:8])


def image_group(g: int, w: list[int]) -> list[int]:
    """T = BLAKE3 keyed_hash(IMG_KEY, u64le(g) || u64le(w0..w3)), read back as 4 little-endian u64 words."""
    msg = b"".join((int(x) & M64).to_bytes(8, "little") for x in (g, *w))
    h = blake3_keyed_hash_block(IMG_KEY, msg)
    return [int.from_bytes(h[8 * i : 8 * i + 8], "little") for i in range(4)]


def image_array(theta: list[int]) -> list[int]:
    out = []
    for g in range(len(theta) // 4):
        out += image_group(g, theta[4 * g : 4 * g + 4])
    return out


def entry(mem: list[int], p: dict, seed: int, idx: int) -> list[int]:
    """T_seed[idx]: G entry words of block idx under link seed `seed` (slow reference)."""
    be, r, K, G = p["be"], p["r"], p["k"], p["g"]
    ks = chain_polyk(p["k0"], p["k1"], seed, p["polyk"])
    mixkey = chain_mixkey(p["k0"], p["k1"], seed, p["polyk"])
    words = mem[idx * be : (idx + 1) * be]
    if p.get("img"):
        words = [
            v
            for g in range(be // 4)
            for v in image_group((p.get("qoff", 0) + idx * be) // 4 + g, words[4 * g : 4 * g + 4])
        ]
    T = []
    if r == 0:
        # the group's 2*span 32-bit half-words as one polynomial: low halves at K^1..K^span, high at K^(span+1)..K^(2
        # span)
        span = be // G
        for g in range(G):
            H = 0
            for loc, w in enumerate(words[g * span : (g + 1) * span]):
                H = (H + (w & 0xFFFFFFFF) * pow(ks, loc + 1, PM61) + (w >> 32) * pow(ks, span + loc + 1, PM61)) % PM61
            T.append(H)
        return T
    M = matrix(seed, K, r)
    wpr = K // 8
    rows = be // wpr
    span = rows // G
    for g in range(G):
        H = 0
        for rl in range(span):
            row = g * span + rl
            pieces = []
            for w in words[row * wpr : (row + 1) * wpr]:
                pieces += [(w >> (8 * q)) & 0xFF for q in range(8)]
            for c in range(r):
                v = mix_output(sum(pieces[j] * M[j][c] for j in range(K)), mixkey)
                H = (H + v * pow(ks, rl * r + c + 1, PM61)) % PM61
        T.append(H)
    return T


def link_seed(p: dict, digest_hi: int) -> int:
    return splitmix64(p["tseed"] ^ (digest_hi % p["x"])) if p["x"] else p["nonce"]


def chain(mem: list[int], prm_list: list[int]) -> tuple[int, int, list[int]]:
    """Honest chain over a python list of stored words. prm_list is the u64 param block."""
    p = dict(zip(P_FIELDS, [int(v) & M64 for v in prm_list]))
    nb, n, nonce = p["nb"], p["n"], p["nonce"]
    kn = chain_polyk(p["k0"], p["k1"], nonce, p["polyk"])
    kf = chain_foldk(p["k0"], p["k1"], nonce, p["polyk"])
    if p["d0_lo"] == 0 and p["d0_hi"] == 0:
        dlo, dhi = siphash24_128(p["k0"], p["k1"], nonce, 0, 0, 0)
    else:
        dlo, dhi = p["d0_lo"], p["d0_hi"]
    idx, trace = p["j0"], []
    seed = p["s0"] or link_seed(p, dhi)
    for link in range(n):
        T = entry(mem, p, seed, idx)
        H = sum(t * kn * pow(kf, g, PM61) for g, t in enumerate(T)) % PM61
        trace.append(idx)
        dlo, dhi = siphash24_128(p["k0"], p["k1"], (nonce + p["link0"] + link) & M64, dlo, dhi, H)
        idx = dlo % nb
        seed = link_seed(p, dhi)
    return dlo, dhi, trace
