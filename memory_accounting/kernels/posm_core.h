/*
 * posm_core.h -- shared arithmetic for the PoSM de-risking prototype.
 *
 * Compiles as plain C (CPU reference / CPU cheater) and CUDA (H100/H200).
 * Everything here is a pure function on values so it can be shared; the
 * address-space macros are no-ops on both targets.
 *
 * Mechanisms implemented (mapping to the PoSM v8 doc):
 *   - Incompressible encoding  : Basic-VDE-PoRep (Fisch 2018, s3.1.1) over
 *                                F_p, p = 2^61-1.  Encode = modular square
 *                                root (59 squarings, serial).  Decode = one
 *                                squaring.  Position-dependent tweak.
 *   - Per-GPU distinct copy    : Feistel PRP permutation of element indices
 *                                and/or a per-GPU encoding seed.
 *   - Matmul on the block      : int8 matrix M derived from the nonce, applied
 *                                to the bytes (8-bit pieces) of the stored words.
 *   - Keyed nonce-prefixed hash: polynomial hash mod 2^61-1 (r = 0: over the
 *                                32-bit halves of the stored words) folded with a
 *                                SipHash-2-4-128 finalizer (construction "D").
 *   - Hash chain               : next block index = digest mod num_blocks.
 */
#ifndef POSM_CORE_H
#define POSM_CORE_H

#if defined(__CUDACC__)
  #include <stdint.h>
  typedef unsigned long long u64; typedef unsigned int u32; typedef unsigned char u8; typedef signed char i8;
  typedef int i32; typedef long long i64;
  #define INLINE __host__ __device__ __forceinline__   /* pure helpers are usable from host code too */
  #define DEVFN  __device__ __forceinline__            /* bodies that use barriers: device only */
  #define U64C(x) x##ULL
  #define DEVPTR
  #define CONSTPTR
  #define SHPTR
  #define THRPTR
  #define SYNC() __syncthreads()
  INLINE u64 umulhi64(u64 a, u64 b) {
  #ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
  #else
    return (u64)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
  #endif
  }
#else
  #include <stdint.h>
  typedef uint64_t u64; typedef uint32_t u32; typedef uint8_t u8; typedef int8_t i8;
  typedef int32_t i32; typedef int64_t i64;
  #define INLINE static inline
  #define DEVFN  static inline
  #define U64C(x) x##ULL
  #define DEVPTR
  #define CONSTPTR
  #define SHPTR
  #define THRPTR
  #define SYNC() ((void)0)
  INLINE u64 umulhi64(u64 a, u64 b) { return (u64)(((unsigned __int128)a * (unsigned __int128)b) >> 64); }
#endif

#ifndef POSM_VDE_BITS
#define POSM_VDE_BITS 61
#endif
#ifndef POSM_PACKED128
#define POSM_PACKED128 0
#endif
#ifndef POSM_GROUP64
#define POSM_GROUP64 0
#endif
#define PM61            U64C(0x1FFFFFFFFFFFFFFF)
#define GOLD            U64C(0x9E3779B97F4A7C15)
#define VDE_SQUARINGS   59            /* (p+1)/4 = 2^59 */
/* Matmul geometry.  A row is K pieces of 8 bits (K/8 stored words); it is multiplied by the
 * nonce-derived int8 matrix M (K x r).  Multiply-adds per input byte = r, independent of K, while
 * the hashed output is (bytes * r / K) values, so a long inner dimension keeps the honest hash
 * cost flat as r grows.  K and r are challenge parameters (prm->k, prm->r). */
#define POSM_KMIN       128
#define POSM_KMAX       2048          /* 16-row tile of K bytes must fit shared memory per warp */
#define POSM_RMAX       4096
#define POSM_CPT        8             /* matmul columns per scalar slot */
#define POSM_HW         4             /* words per hash-only slot */
#define POSM_MAX_LINKS  4096          /* output buffer holds 2 + n words */
#define POSM_GMAX       1024          /* max entry groups per block */
#define POSM_T1LEN(r)   ((r) ? (r) : 64)   /* power table 1: K_n^(c+1); for r = 0 positions are split i = 64a + b */

/* ---- field arithmetic mod 2^61-1 --------------------------------------- */
INLINE u64 red61(u64 x) {              /* x < 2^63  ->  [0, p) */
    x = (x & PM61) + (x >> 61);
    return x >= PM61 ? x - PM61 : x;
}
INLINE u64 addmod(u64 a, u64 b) { u64 s = a + b; return s >= PM61 ? s - PM61 : s; }
INLINE u64 submod(u64 a, u64 b) { return a >= b ? a - b : a + PM61 - b; }
INLINE u64 mulmod(u64 a, u64 b) {
    u64 lo = a * b, hi = umulhi64(a, b);          /* hi < 2^58 since a,b < 2^61 */
    u64 s = (lo & PM61) + (lo >> 61) + (hi << 3); /* 2^64 == 8 (mod p) */
    s = (s & PM61) + (s >> 61);
    return s >= PM61 ? s - PM61 : s;
}
INLINE u64 powmod(u64 b, u64 e) {
    u64 r = 1;
    while (e) { if (e & 1) r = mulmod(r, b); b = mulmod(b, b); e >>= 1; }
    return r;
}

/* ---- PRNG / tweaks ------------------------------------------------------ */
INLINE u64 splitmix64(u64 z) {
    z += GOLD;
    z = (z ^ (z >> 30)) * U64C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * U64C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}
/* tweak_at (per-element additive tweak keyed by the 128-bit encoding seed) is defined after SipHash below. */

/* Encoded dynamic memory's tweak positions (memory_accounting.prover.kvenc): a domain of their own, bit 62 set. */
#define KV_POS (U64C(1) << 62)

/* ---- VDE (slow sqrt encode / fast square decode) ------------------------ */
/* word layout: bits 0..60 = root r, bit 61 = non-residue flag, bits 62,63 = 0 */
INLINE u64 vde_encode(u64 d, u64 t) {
    u64 x = addmod(d, t);
    u64 r = x;
    for (int i = 0; i < VDE_SQUARINGS; i++) r = mulmod(r, r);   /* r = x^(2^59) */
    u64 f = (mulmod(r, r) != x) ? U64C(1) : U64C(0);          /* r^2 == -x  <=> x is a non-residue */
    return r | (f << 61);
}
INLINE u64 vde_decode(u64 w, u64 t) {
    u64 r = w & PM61;
    u64 x = mulmod(r, r);
    if ((w >> 61) & 1) x = x ? PM61 - x : 0;
    return submod(x, t);
}

/* ---- Feistel PRP on [0, 2^bits) ----------------------------------------- */
INLINE u64 feistel_perm(u64 q, u64 seed, u32 bits) {
    if (bits == 0) return 0;
    u32 lb = bits / 2, rb = bits - lb;
    u64 lmask = lb ? ((U64C(1) << lb) - 1) : 0;
    u64 rmask = (U64C(1) << rb) - 1;
    u64 L = (q >> rb) & lmask, R = q & rmask;
    for (u32 i = 0; i < 4; i++) {
        R ^= splitmix64(L ^ (seed + (u64)(2 * i) * GOLD)) & rmask;
        L ^= splitmix64(R ^ (seed + (u64)(2 * i + 1) * GOLD)) & lmask;
    }
    return (L << rb) | R;
}

/* ---- SipHash-2-4, 128-bit output, fixed 4-word (32 byte) message -------- */
struct u128 { u64 lo, hi; };
INLINE u64 rotl64(u64 x, u32 b) { return (x << b) | (x >> (64 - b)); }
#define SIPROUND(v0, v1, v2, v3) do {                     \
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32); \
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;                     \
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;                     \
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32); } while (0)
INLINE struct u128 siphash24_128(u64 k0, u64 k1, u64 m0, u64 m1, u64 m2, u64 m3) {
    u64 v0 = k0 ^ U64C(0x736f6d6570736575), v1 = k1 ^ U64C(0x646f72616e646f6d);
    u64 v2 = k0 ^ U64C(0x6c7967656e657261), v3 = k1 ^ U64C(0x7465646279746573);
    v1 ^= U64C(0xee);
    u64 m[4]; m[0] = m0; m[1] = m1; m[2] = m2; m[3] = m3;
    for (int i = 0; i < 4; i++) { v3 ^= m[i]; SIPROUND(v0, v1, v2, v3); SIPROUND(v0, v1, v2, v3); v0 ^= m[i]; }
    u64 b = U64C(32) << 56;
    v3 ^= b; SIPROUND(v0, v1, v2, v3); SIPROUND(v0, v1, v2, v3); v0 ^= b;
    v2 ^= U64C(0xee);
    for (int i = 0; i < 4; i++) SIPROUND(v0, v1, v2, v3);
    struct u128 o; o.lo = v0 ^ v1 ^ v2 ^ v3;
    v1 ^= U64C(0xdd);
    for (int i = 0; i < 4; i++) SIPROUND(v0, v1, v2, v3);
    o.hi = v0 ^ v1 ^ v2 ^ v3;
    return o;
}

/* Per-element additive tweak: ties an encoded word to (128-bit encoding key, position).  A keyed PRF,
 * so that with a prover-secret key the encoded array is a ciphertext of the payload (each word is a
 * bijection of payload + PRF(key, q) in F_p) and may be shown to a verifier without revealing weights. */
#define TWEAK_DOMAIN U64C(0x5457454B00000001)
INLINE u64 tweak_prf(u64 key_lo, u64 key_hi, u64 q) {
    struct u128 d = siphash24_128(key_lo, key_hi, q, TWEAK_DOMAIN, 0, 0);
    return red61(d.lo & PM61);
}

INLINE u64 tweak_at(u64 kl,u64 kh,u64 q) {
#if POSM_GROUP64
    return addmod(tweak_prf(kl,kh,q & ~U64C(7)), q & 7);
#else
    return tweak_prf(kl,kh,q);
#endif
}
#if POSM_VDE_BITS == 89
#include "codec89.h"
#if POSM_PACKED128

/* Eleven consecutive 90-bit records in 32 little-endian 32-bit words.
 * q is the word address of the row, including its region domain. */
INLINE field89 packed_tweak(u64 kl, u64 kh, u64 q, u32 slot) {
    return f89_key(siphash24_128(kl, kh, (q / 16) * 11 + slot, TWEAK_DOMAIN, 0, 0));
}

INLINE field89 packed_decode(const u32 *row, u32 slot, field89 tweak) {
    u32 bit = slot * 90, word = bit / 32, shift = bit % 32;
    u32 value[3];
    for (u32 i = 0; i < 3; ++i) {
        value[i] = row[word + i] >> shift;
        if (shift) value[i] |= row[word + i + 1] << (32 - shift);
    }
    return f89_read_root(value, tweak);
}

/* Each output word has a single writer. Unused high bits are zero. */
INLINE u32 packed_word(const u32 *records, u32 word) {
    u32 bit = word * 32, slot = bit / 90, off = bit % 90;
    if (slot >= 11) return 0;
    const u32 *value = records + 3 * slot;
    u32 limb = off / 32, shift = off % 32;
    u32 out = value[limb] >> shift;
    if (shift && limb < 2) out |= value[limb + 1] << (32 - shift);
    if (off > 58 && slot < 10) out |= value[3] << (90 - off);
    return out;
}

INLINE void packed_encode_row(const u8 *raw, u64 bytes, u64 off, u32 *out, u64 q, u64 kl, u64 kh) {
    u32 records[33];
    for (u32 s = 0; s < 11; ++s)
        f89_write_root(records + 3*s, f89_read_plain(raw, bytes, off + 11*s), packed_tweak(kl, kh, q, s));
    for (u32 i = 0; i < 32; ++i) out[i] = packed_word(records, i);
}

INLINE void packed_decode_row(const u32 *in, u8 *raw, u64 bytes, u64 off, u64 q, u64 kl, u64 kh) {
    for (u32 s = 0; s < 11; ++s)
        f89_plain_bytes(raw, bytes, off + 11*s, packed_decode(in, s, packed_tweak(kl, kh, q, s)));
}

#endif
#endif

/* ---- 60-bit packing: 15 raw 64-bit words <-> 16 payload words (memory_accounting.setup.pack) --------------- */
INLINE void pack_group(THRPTR const u64* raw15, THRPTR u64* out16) {
    for (u32 k = 0; k < 16; k++) {
        u32 start = 60 * k, a = start >> 6, off = start & 63;
        u64 v = raw15[a] >> off;
        if (off > 4 && a + 1 < 15) v |= raw15[a + 1] << (64 - off);
        out16[k] = v & ((U64C(1) << 60) - 1);
    }
}
INLINE void unpack_group(THRPTR const u64* in16, THRPTR u64* raw15) {
    for (u32 a = 0; a < 15; a++) {
        u32 start = 64 * a, k = start / 60, off = start % 60;
        u64 v = in16[k] >> off;
        if (k + 1 < 16) v |= in16[k + 1] << (60 - off);
        if (k + 2 < 16 && 120 - off < 64) v |= in16[k + 2] << (120 - off);
        raw15[a] = v;
    }
}

/* ---- challenge parameter block --------------------------------------------
 * One u64 per field, in this order: Python builds it as a u64 array from the same names
 * (posm.params.P_FIELDS; tests/test_params.py checks the two agree), and the pre-armed
 * wait node copies it word for word (POSM_PARAM_WORDS). */
typedef struct posm_params {
    u64 nb;               /* number of blocks in the challenged array           */
    u64 be;               /* words per block (block_bytes / 8)                  */
    u64 r;                /* matmul width r (0 = hash stored words directly)    */
    u64 n;                /* chain length                                       */
    u64 nonce;            /* verifier nonce                                     */
    u64 j0;               /* first block index                                  */
    u64 k0, k1;           /* SipHash key                                        */
    u64 polyk;            /* polynomial hash key                                */
    u64 mode;             /* 0 honest, 1 other-GPU gather (+re-encode), 3 encode-on-the-fly, 4 the same for encoded dynamic memory */
    u64 enc;              /* 0 none, 1 VDE                                      */
    u64 seed_t;           /* encoding seed of the challenged (target) GPU       */
    u64 seed_c;           /* encoding seed of the cheater's canonical copy      */
    u64 perm_on;          /* 1 if target copy is a Feistel permutation of canon */
    u64 perm_seed;
    u64 perm_bits;        /* log2(total words)                                  */
    u64 d0_lo, d0_hi;     /* initial digest state (0,0 => derive from nonce)    */
    u64 link0;            /* link counter offset (for single-link launches)     */
    u64 k;                /* matmul inner dimension: pieces (bytes) per row     */
    u64 nseg;             /* honest array as segments: number of segments (0 = contiguous `mem`) */
    u64 segtab;           /* pointer (as u64) to [nseg+1] cumulative word starts followed by [nseg] base pointers */
    u64 seed_t_hi;        /* high halves of the 128-bit encoding keys                                    */
    u64 seed_c_hi;
    u64 x;                /* table scheme: number of public seeds (0 = legacy, link seed = nonce)         */
    u64 g;                /* entry groups per block: T[j] has G words (d = 8G bytes)                      */
    u64 tseed;            /* table seed: s_i = splitmix64(tseed ^ i)                                      */
    u64 s0;               /* explicit seed of the first link (single-link continuation); 0 = derive       */
    u64 img;              /* 1: the prover hashes each 32-byte group of the fetched array before use (one-way image T = H(Theta')); the verifier holds T and runs with 0 */
    u64 qoff;             /* global word offset of mem[0] (a staged block): image groups are position-bound          */
    /* a cheat on a share of the array (memory_accounting.prover.strategies.ShareCheat); all 0 for the whole-array strategies */
    u64 qcheat;           /* modes 1 and 3: words below qcheat are read honestly (segment table or mem), the rest by the cheat */
    u64 canonp;           /* modes 1 and 3: the cheat's source array as a device address (0 = the kernel's `canon`)   */
    u64 jstop;            /* stop the chain at the first link whose block is at or past jstop (0 = never): POSM_STOP_BIT */
    u64 qcheat_hi;        /* modes 1, 3 and 4: words at or past it are read honestly again (0 = the share runs to the end)    */
    u64 jstop_hi;         /* host share stops only before this block (0 = the share runs to the end) */
} posm_params;
#define POSM_PARAM_WORDS (sizeof(posm_params) / sizeof(u64))

/* The setup/decode parameter block (posm.params.SETUP_FIELDS): encode or decode `count` words. */
typedef struct posm_setup_params {
    u64 count;            /* words to encode or decode                          */
    u64 enc;              /* 0 none, 1 VDE                                      */
    u64 key_lo;           /* 128-bit encoding key, low half                     */
    u64 perm_on;          /* 1 if the array is a Feistel permutation of canon   */
    u64 perm_seed;
    u64 perm_bits;        /* log2(total words)                                  */
    u64 key_hi;           /* 128-bit encoding key, high half                    */
} posm_setup_params;

/* An aborted pre-armed chain: the wait node writes this mode instead of copying the parameters, and
 * every chain kernel returns at once, so no kernel reads memory through a stale parameter block (a
 * segment table freed or a region resized since the graph was armed). */
#define POSM_MODE_ABORT   U64C(0xAB)
#define POSM_ABORTED(prm) ((prm)->mode == POSM_MODE_ABORT)

/* Per-challenge polynomial key.  The block-dependent part of a link is linear in the block, so with
 * a FIXED key a cheater could precompute a 128-element table per block (8 bytes at r = 0) and
 * discard the data (midstate caching).  Evaluating at a point derived from the fresh nonce makes
 * every link a random-point evaluation of the block's polynomial, which no store smaller than
 * the block can answer.  prm->polyk is the long-term secret mixed into the derivation. */
INLINE u64 chain_polyk(u64 k0, u64 k1, u64 nonce, u64 polyk_secret) {
    struct u128 d = siphash24_128(k0, k1, nonce, polyk_secret, U64C(0x504F4C594B455900), 0);
    u64 k = red61(d.lo & PM61);
    return k ? k : U64C(1);
}
/* Per-challenge key of the output mixer (the other half of the same derivation). */
INLINE u64 chain_mixkey(u64 k0, u64 k1, u64 nonce, u64 polyk_secret) {
    struct u128 d = siphash24_128(k0, k1, nonce, polyk_secret, U64C(0x504F4C594B455900), 0);
    return d.hi;
}
/* Per-challenge key of the group fold, in its own domain.  A link folds its G entry words as
 *   H = sum_g T[g] * K_n * K_f^g,
 * where T[g] is a polynomial in the link-seed key K_s.  With x == 0 the link seed is the nonce, so K_s == K_n;
 * folding with K_n^(g+1) (v0.11 and earlier) put word loc of group g at K_n^(g + loc + 2), which group g + 1's word
 * loc - 1 shares, and two blocks differing by such a shift gave the same digest under every nonce.  With K_f
 * independent of K_n every (g, loc) is a distinct monomial K_n^(loc+2) K_f^g.  g = 0 keeps the factor K_n,
 * so G = 1 digests are unchanged. */
#define FOLD_DOMAIN U64C(0x464F4C444B455900)   /* "FOLDKEY" */
INLINE u64 chain_foldk(u64 k0, u64 k1, u64 nonce, u64 polyk_secret) {
    struct u128 d = siphash24_128(k0, k1, nonce, polyk_secret, FOLD_DOMAIN, 0);
    u64 k = red61(d.lo & PM61);
    return k ? k : U64C(1);
}
/* Table scheme.  The link seed s selects the matrix, the output mixer and the entry-hash key; with
 * x public seeds the verifier can precompute T_s[j] for every (s, j) at setup and verify a link by
 * lookup.  x == 0 is the legacy mode where the seed is the challenge nonce itself. */
INLINE u64 table_seed_i(u64 tseed, u64 i) { return splitmix64(tseed ^ i); }
INLINE u64 seed_from_digest(CONSTPTR const posm_params* prm, u64 digest_hi) {
    u64 x = prm->x;
    return x ? table_seed_i(prm->tseed, digest_hi % x) : prm->nonce;
}
/* per-block entry layout: G groups; r > 0 -> rows per group; r == 0 -> words per group */
INLINE u32 posm_group_span(CONSTPTR const posm_params* prm) {
    const u32 G = (u32)prm->g, r = (u32)prm->r, K = (u32)prm->k;
    const u64 be = prm->be;
    return r ? (u32)(be / (K / 8)) / G : (u32)be / G;
}
/* pw layout: [T1: K_s^(c+1), T1LEN] [T2: K_s^(row_local * r) or K_s^(64 a), T2LEN] [PWN: K_n * K_f^g, G] */
INLINE u32 posm_t2len(CONSTPTR const posm_params* prm) {
    const u32 span = posm_group_span(prm);
    return prm->r ? span : (span + 63) / 64;
}

/* Non-linear, nonce-keyed mixing of one matmul output before the polynomial hash.  Without it the
 * hash of A*M is linear in A and collapses to one multiply per byte (fold M into the hash weights
 * once per challenge); with it every output must be computed, so a CPU pays r multiply-adds per
 * byte.  splitmix64 is a bijection on 64 bits, so distinct outputs stay distinct before reduction. */
INLINE u64 mix_output(i32 v, u64 mixkey) {
    return red61(splitmix64(((u64)(u32)v) ^ mixkey) & PM61);
}

/* Matrix entry generator: M[j][c] as int8, nonce-dependent, shared by all links of a chain.
 * Stored in a global buffer in a tiled layout: tile (kt, ct) of 16x16 bytes at ((ct*K/16)+kt)*256,
 * row-major inside the tile, so every 16x16 tile is 256-byte aligned for tensor-core loads. */
INLINE u64 matrix_seed(u64 nonce) { return splitmix64(nonce ^ U64C(0x4D41545249580001)); }
INLINE u32 mtile_index(u32 j, u32 c, u32 K) {
    return ((c >> 4) * (K >> 4) + (j >> 4)) * 256 + (j & 15) * 16 + (c & 15);
}
INLINE u32 mbuf_colconst_offset(u32 K, u32 r) { return ((K * r + 255) / 256) * 256; }

/* mbuf = [K*r tiled int8][pad to 256][r x int32 colconst], colconst[c] = 128 * sum_j M[j][c]
 * (pieces are fed to the tensor cores as p - 128, and the constant restores sum_j p*M). */
INLINE void posm_matrix_fill(DEVPTR i8* mbuf, CONSTPTR const posm_params* prm, u64 seed, u64 i0, u64 istride) {
    const u32 K = (u32)prm->k, r = (u32)prm->r;
    const u64 ms = matrix_seed(seed);
    const u64 total = (u64)K * r;
    for (u64 i = i0 * 8; i < total; i += istride * 8) {
        u64 g = splitmix64(ms + (i >> 3));
        for (u32 q = 0; q < 8; q++) { u32 j = (u32)((i + q) / r), c = (u32)((i + q) % r); mbuf[mtile_index(j, c, K)] = (i8)(u8)(g >> (8 * q)); }
    }
}

/* colconst[c] = 128 * sum_j M[j][c] (tensor path); strided over threads */
INLINE void posm_colconst_fill(DEVPTR i8* mbuf, CONSTPTR const posm_params* prm, u32 c0, u32 cstride) {
    const u32 K = (u32)prm->k, r = (u32)prm->r;
    DEVPTR i32* cc = (DEVPTR i32*)(mbuf + mbuf_colconst_offset(K, r));
    for (u32 c = c0; c < r; c += cstride) {
        i32 sum = 0;
        for (u32 j = 0; j < K; j++) sum += (i32)mbuf[mtile_index(j, c, K)];
        cc[c] = 128 * sum;
    }
}

/* Power tables.  Seed part (refilled per link for the link seed `seed`):
 *   r > 0: output (row, c) of group g has local position (row % rpg) * r + c; power = T2[row % rpg] * T1[c]
 *   r = 0: word i of group g has local position loc = i % wpg = 64 a + b;      power = T2[a] * T1[b]
 * Nonce part PWN[g] = K_n * K_f^g (filled once per chain) folds the G entry words into the link digest. */
INLINE void posm_powtable_fill(DEVPTR u64* pw, CONSTPTR const posm_params* prm, u64 seed, u64 i, u64 count) {
    if (i >= count) return;
    const u32 r = (u32)prm->r;
    const u64 t1 = POSM_T1LEN(r), t2 = posm_t2len(prm);
    if (i < t1 + t2) {
        const u64 ks = chain_polyk(prm->k0, prm->k1, seed, prm->polyk);
        pw[i] = i < t1 ? powmod(ks, i + 1) : powmod(ks, (i - t1) * (r ? (u64)r : U64C(64)));
    } else {
        const u64 kn = chain_polyk(prm->k0, prm->k1, prm->nonce, prm->polyk);
        const u64 kf = chain_foldk(prm->k0, prm->k1, prm->nonce, prm->polyk);
        pw[i] = mulmod(kn, powmod(kf, i - t1 - t2));
    }
}
INLINE u64 posm_power(DEVPTR const u64* pw, u32 r, u64 row_local, u64 c) {
    return mulmod(pw[POSM_T1LEN(r) + row_local], pw[c]);
}
/* r = 0 hashes a stored word as its two 32-bit halves: group g's entry is the polynomial over the group's
 * 2 * span half-words, low halves at K^1 .. K^span and high halves at K^(span+1) .. K^(2 span), i.e.
 *   T[g] = sum_loc (lo(w) + hi(w) * K^span) * K^(loc+1).
 * Every coefficient is below 2^32 < p, so distinct blocks are distinct polynomials and collide only at a
 * root of their difference (probability <= 2 span / p over the key).  Reducing the word mod p first was
 * not injective: w and w + p (0 and 2^61 - 1, ...) gave the same digest under every nonce, so a prover
 * could change words, or keep 3 bits of its own per word, without changing any digest. */
INLINE u64 posm_hash_word(u64 w, u64 khi) { return addmod(w & U64C(0xFFFFFFFF), mulmod(w >> 32, khi)); }
/* K^span from the r = 0 tables: K^(64 a) * K^(b + 1) with 64 a + b = span - 1 */
INLINE u64 posm_khi(DEVPTR const u64* pw, u32 span) { return mulmod(pw[POSM_T1LEN(0) + ((span - 1) >> 6)], pw[(span - 1) & 63]); }

/* Verifier-seeded filler payload word (matches memory_accounting.setup.filler.filler_words). */
INLINE u64 filler_word(u64 seed, u64 idx) { return splitmix64(seed ^ (idx * GOLD)) & ((U64C(1) << 60) - 1); }

/* The honest prover's array may be a set of resident tensors (vLLM's weights) plus a filler
 * allocation, presented as one virtual array: segment table = starts[0..nseg] (cumulative word
 * offsets, starts[nseg] = total) then ptrs[0..nseg) (device addresses as u64). */
INLINE u64 segmented_read(DEVPTR const u64* segtab, u32 nseg, u64 q) {
    u32 lo = 0, hi = nseg;                       /* find the last start <= q */
    while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (segtab[mid] <= q) lo = mid; else hi = mid; }
    DEVPTR const u64* base = (DEVPTR const u64*)(uintptr_t)segtab[nseg + 1 + lo];
    return base[q - segtab[lo]];
}
/* Tensor-chain state[4..6] caches the segment containing the first word of the
 * current block. A block crossing its end still uses segmented_read for the
 * outlying groups. state[7] is 0 for an honest block, 1 for a fully cheated
 * block, 2 for a block crossing the selected interval. */
INLINE void posm_cache_segment(CONSTPTR const posm_params* prm, DEVPTR u64* state) {
    u64 q = state[0] * prm->be;
    if (prm->mode == 0 || q + prm->be <= prm->qcheat ||
        (prm->qcheat_hi && q >= prm->qcheat_hi)) state[7] = 0;
    else if (q >= prm->qcheat && (!prm->qcheat_hi || q + prm->be <= prm->qcheat_hi)) state[7] = 1;
    else state[7] = 2;
    u32 nseg = (u32)prm->nseg;
    if (!nseg) { state[4] = state[5] = state[6] = 0; return; }
    DEVPTR const u64* tab = (DEVPTR const u64*)(uintptr_t)prm->segtab;
    u32 lo = 0, hi = nseg;
    while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (tab[mid] <= q) lo = mid; else hi = mid; }
    state[4] = tab[lo];
    state[5] = tab[lo + 1];
    state[6] = tab[nseg + 1 + lo];
}

/* ---- BLAKE3 compression of one block (32-bit ARX), for the one-way image ------------------------
 * Message words are u32 m[16]; each round takes them in the order of the BLAKE3 permutation applied
 * round-many times (the official MSG_SCHEDULE), written out so every index is a constant. */
#define B3_IV0 0x6A09E667u
#define B3_IV1 0xBB67AE85u
#define B3_IV2 0x3C6EF372u
#define B3_IV3 0xA54FF53Au
#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_ROOT        8u
#define B3_KEYED_HASH 16u
INLINE u32 rotr32(u32 x, u32 b) { return (x >> b) | (x << (32 - b)); }
#define B3G(a, b, c, d, x, y) do {                                                                    \
    v[a] += v[b] + (x); v[d] = rotr32(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = rotr32(v[b] ^ v[c], 12); \
    v[a] += v[b] + (y); v[d] = rotr32(v[d] ^ v[a], 8);  v[c] += v[d]; v[b] = rotr32(v[b] ^ v[c], 7); } while (0)
#define B3ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) do {          \
    B3G(0, 4, 8, 12, m[s0], m[s1]);   B3G(1, 5, 9, 13, m[s2], m[s3]);                                \
    B3G(2, 6, 10, 14, m[s4], m[s5]);  B3G(3, 7, 11, 15, m[s6], m[s7]);                               \
    B3G(0, 5, 10, 15, m[s8], m[s9]);  B3G(1, 6, 11, 12, m[s10], m[s11]);                             \
    B3G(2, 7, 8, 13, m[s12], m[s13]); B3G(3, 4, 9, 14, m[s14], m[s15]); } while (0)
#define B3ROUNDS() do {                                                     \
    B3ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);          \
    B3ROUND(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);          \
    B3ROUND(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);          \
    B3ROUND(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);          \
    B3ROUND(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);          \
    B3ROUND(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);          \
    B3ROUND(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13); } while (0)

/* One-way image of one 32-byte group of the encoded array: T[4g..4g+3] = H(g, Theta'[4g..4g+3]).
 * Same size as Theta', not invertible (so the verifier holding T cannot recover Theta' even with the
 * encoding key), position-bound, and every link still consumes a full block of it.
 * H = BLAKE3 keyed_hash(IMG_KEY, u64le(g) || the 4 words as 32 little-endian bytes)[0..32): the 40-byte
 * message is one block (block_len 40, zero-padded), counter 0, flags CHUNK_START | CHUNK_END | ROOT |
 * KEYED_HASH, so it is a single compression.  The key is public and fixed:
 *   IMG_KEY = BLAKE3 derive_key(context = "posm 2026-10 image v1", key_material = b"")
 *           = f57be4c5137fa6167fa7a2f648754a9cefb52f5122991c15705741611b86e000  (8 u32 words, little-endian). */
#define IMG_KEY0 0xC5E47BF5u
#define IMG_KEY1 0x16A67F13u
#define IMG_KEY2 0xF6A2A77Fu
#define IMG_KEY3 0x9C4A7548u
#define IMG_KEY4 0x512FB5EFu
#define IMG_KEY5 0x151C9922u
#define IMG_KEY6 0x61415770u
#define IMG_KEY7 0x00E0861Bu
INLINE void image_group(u64 g, THRPTR const u64* w, THRPTR u64* t) {
    u32 m[16] = { (u32)g, (u32)(g >> 32), (u32)w[0], (u32)(w[0] >> 32), (u32)w[1], (u32)(w[1] >> 32),
                  (u32)w[2], (u32)(w[2] >> 32), (u32)w[3], (u32)(w[3] >> 32), 0, 0, 0, 0, 0, 0 };
    u32 v[16] = { IMG_KEY0, IMG_KEY1, IMG_KEY2, IMG_KEY3, IMG_KEY4, IMG_KEY5, IMG_KEY6, IMG_KEY7,
                  B3_IV0, B3_IV1, B3_IV2, B3_IV3, 0, 0, 40, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED_HASH };
    B3ROUNDS();
    /* written out (no loop) so v[] stays in registers on the device */
    t[0] = (u64)(v[0] ^ v[8]) | ((u64)(v[1] ^ v[9]) << 32);
    t[1] = (u64)(v[2] ^ v[10]) | ((u64)(v[3] ^ v[11]) << 32);
    t[2] = (u64)(v[4] ^ v[12]) | ((u64)(v[5] ^ v[13]) << 32);
    t[3] = (u64)(v[6] ^ v[14]) | ((u64)(v[7] ^ v[15]) << 32);
}

#if POSM_VDE_BITS == 89
INLINE void posm_reconstruct89_row(DEVPTR const u64* canon, CONSTPTR const posm_params* prm, u64 row, THRPTR u64* encoded) {
#if POSM_PACKED128
    u64 payload[16];
    if (prm->mode == 1 && prm->enc) {
        packed_decode_row((const u32*)(canon + row), (u8*)payload, 121, 0,
                          row, prm->seed_c, prm->seed_c_hi);
    } else {
        for (u32 i = 0; i < 16; i++) payload[i] = canon[row + i];
    }
    if (!prm->enc) {
        for (u32 i = 0; i < 16; i++) encoded[i] = payload[i];
    } else {
        packed_encode_row((const u8*)payload, 121, 0, (u32*)encoded,
                          row, prm->seed_t, prm->seed_t_hi);
    }
#else
    u64 payload[8];
    if (prm->mode == 1 && prm->enc) {
        const u32 *source = (const u32*)(canon + row);
        u8 *bytes = (u8*)payload;
        field89 base = f89_base(prm->seed_c, prm->seed_c_hi, row);
        for (u32 lane = 0; lane < 5; lane++) {
            field89 tweak = f89_slot_tweak(prm->seed_c, prm->seed_c_hi, row, lane, base);
            f89_plain_bytes(bytes, 64, 12*lane, f89_read_root(source + 3*lane, tweak));
            bytes[12*lane + 11] = 0;
        }
        ((u32*)payload)[15] = 0;
    } else {
        for (u32 i = 0; i < 8; i++) payload[i] = canon[row + i];
    }
    if (!prm->enc) {
        for (u32 i = 0; i < 8; i++) encoded[i] = payload[i];
    } else {
        f89_setup_row(payload, (u32*)encoded, prm->seed_t, prm->seed_t_hi, row);
    }
#endif
}
#endif
/* Fetch one stored word of the target array under a given prover strategy (raw, before the image). */
INLINE u64 posm_fetch_raw_word(DEVPTR const u64* mem, DEVPTR const u64* canon, CONSTPTR const posm_params* prm, u64 q) {
    u64 mode = prm->mode;
    if (mode == 0 || q < prm->qcheat || (prm->qcheat_hi && q >= prm->qcheat_hi)) {
        u32 nseg = (u32)prm->nseg;
        if (nseg) return segmented_read((DEVPTR const u64*)(uintptr_t)prm->segtab, nseg, q);
        return mem[q];
    }
    if (prm->canonp) canon = (DEVPTR const u64*)(uintptr_t)prm->canonp;
#if POSM_VDE_BITS == 89
    if (mode == 1 || mode == 3) {
#if POSM_PACKED128
        u64 row = q & ~U64C(15);
        u64 encoded[16];
#else
        u64 row = q & ~U64C(7);
        u64 encoded[8];
#endif
        posm_reconstruct89_row(canon, prm, row, encoded);
        return encoded[q - row];
    }
    if (mode == 4 || mode == 5) {
#if POSM_PACKED128
        u64 row = q & ~U64C(15);
        u64 encoded[16];
        if (mode == 4) {
            packed_encode_row((const u8*)(canon + row), 121, 0, (u32*)encoded,
                              KV_POS | row, prm->seed_c, prm->seed_c_hi);
        } else {
            u8 payload[128];
            packed_decode_row((const u32*)(canon + row), payload, 121, 0,
                              KV_POS | row, prm->seed_c, prm->seed_c_hi);
            packed_encode_row(payload, 121, 0, (u32*)encoded,
                              KV_POS | row, prm->seed_t, prm->seed_t_hi);
        }
#else
        u64 row = q & ~U64C(7);
        u64 encoded[8];
        if (mode == 4) {
            f89_setup_row(canon + row, (u32*)encoded,
                          prm->seed_c, prm->seed_c_hi, KV_POS | row);
        } else {
            u8 payload[64];
            field89 base = f89_base(prm->seed_c, prm->seed_c_hi, KV_POS | row);
            for (u32 lane = 0; lane < 5; lane++) {
                field89 tweak = f89_slot_tweak(prm->seed_c, prm->seed_c_hi, KV_POS | row, lane, base);
                f89_plain_bytes(payload, 64, 12*lane,
                                f89_read_root((const u32*)(canon + row) + 3*lane, tweak));
                payload[12*lane + 11] = 0;
            }
            ((u32*)payload)[15] = 0;
            f89_setup_row((const u64*)payload, (u32*)encoded,
                          prm->seed_t, prm->seed_t_hi, KV_POS | row);
        }
#endif
        return encoded[q - row];
    }
#endif
#if POSM_VDE_BITS == 61
    if (mode == 4)                                       /* encoded dynamic memory held as its payload, encoded on demand at */
        return vde_encode(canon[q], tweak_at(prm->seed_c, prm->seed_c_hi, KV_POS | q));   /* region position q (key in seed_c) */
#endif
    if (mode == 1) {                                     /* cheater holds the canonical copy only */
        u64 src = prm->perm_on ? feistel_perm(q, prm->perm_seed, (u32)prm->perm_bits) : q;
        u64 wc = canon[src];
        if (!prm->enc) return wc;
        u64 d = vde_decode(wc, tweak_at(prm->seed_c, prm->seed_c_hi, src));
        return vde_encode(d, tweak_at(prm->seed_t, prm->seed_t_hi, q));
    }
    /* mode 3: cheater holds the raw (compressible) canonical data and regenerates the
       challenged GPU's permuted, encoded word at challenge time */
    u64 src = prm->perm_on ? feistel_perm(q, prm->perm_seed, (u32)prm->perm_bits) : q;
    return vde_encode(canon[src], tweak_at(prm->seed_t, prm->seed_t_hi, q));
}
/* The 32-byte group containing q, imaged if prm->img (the prover's view of T); the verifier runs with img = 0 over T itself. */
INLINE void posm_fetch_group(DEVPTR const u64* mem, DEVPTR const u64* canon, CONSTPTR const posm_params* prm, u64 g, THRPTR u64* out4) {
    u64 w[4];
#if POSM_VDE_BITS == 89
    u64 first = 4 * g;
    if ((prm->mode == 1 || prm->mode == 3 || prm->mode == 4 || prm->mode == 5) && first >= prm->qcheat &&
        (!prm->qcheat_hi || first + 4 <= prm->qcheat_hi)) {
#if POSM_PACKED128
        u64 row[16];
        u64 row_start = first & ~U64C(15);
#else
        u64 row[8];
        u64 row_start = first & ~U64C(7);
#endif
        if (prm->canonp) canon = (DEVPTR const u64*)(uintptr_t)prm->canonp;
        if (prm->mode == 4 || prm->mode == 5) {
#if POSM_PACKED128
            if (prm->mode == 4) {
                packed_encode_row((const u8*)(canon + row_start), 121, 0, (u32*)row,
                                  KV_POS | row_start, prm->seed_c, prm->seed_c_hi);
            } else {
                u8 payload[128];
                packed_decode_row((const u32*)(canon + row_start), payload, 121, 0,
                                  KV_POS | row_start, prm->seed_c, prm->seed_c_hi);
                packed_encode_row(payload, 121, 0, (u32*)row,
                                  KV_POS | row_start, prm->seed_t, prm->seed_t_hi);
            }
#else
            if (prm->mode == 4) {
                f89_setup_row(canon + row_start, (u32*)row,
                              prm->seed_c, prm->seed_c_hi, KV_POS | row_start);
            } else {
                u8 payload[64];
                field89 base = f89_base(prm->seed_c, prm->seed_c_hi, KV_POS | row_start);
                for (u32 lane = 0; lane < 5; lane++) {
                    field89 tweak = f89_slot_tweak(prm->seed_c, prm->seed_c_hi, KV_POS | row_start, lane, base);
                    f89_plain_bytes(payload, 64, 12*lane,
                                    f89_read_root((const u32*)(canon + row_start) + 3*lane, tweak));
                    payload[12*lane + 11] = 0;
                }
                ((u32*)payload)[15] = 0;
                f89_setup_row((const u64*)payload, (u32*)row,
                              prm->seed_t, prm->seed_t_hi, KV_POS | row_start);
            }
#endif
        } else {
            posm_reconstruct89_row(canon, prm, row_start, row);
        }
        for (u32 i = 0; i < 4; i++) w[i] = row[first - row_start + i];
    } else
#endif
    for (u32 i = 0; i < 4; i++) w[i] = posm_fetch_raw_word(mem, canon, prm, 4 * g + i);
    if (prm->img) image_group(prm->qoff / 4 + g, w, out4); else { for (u32 i = 0; i < 4; i++) out4[i] = w[i]; }
}
INLINE u64 posm_fetch_word(DEVPTR const u64* mem, DEVPTR const u64* canon, CONSTPTR const posm_params* prm, u64 q) {
    if (!prm->img) return posm_fetch_raw_word(mem, canon, prm, q);
    u64 t[4]; posm_fetch_group(mem, canon, prm, q >> 2, t);
    return t[q & 3];
}

/* ---- scalar per-link work decomposition (CUDA `multi` and persistent kernels; correctness checks)
 * Slots: r == 0 -> POSM_HW consecutive words; r > 0 -> (row, group of POSM_CPT columns).  Thread
 * `slot0` handles slots congruent to slot0 mod sstride.  NB: with r > 0 every slot of a row
 * re-fetches the row's K/8 words, so this path over-charges the regeneration cheats at large r;
 * the CUDA tensor path fetches each word once and is the one used for timing. */
INLINE u64 posm_partial_slots(DEVPTR const u64* mem, DEVPTR const u64* canon, CONSTPTR const posm_params* prm,
                              DEVPTR const i8* mbuf, DEVPTR const u64* pw, u64 seed, u64 idx, u32 g,
                              u32 slot0, u32 sstride) {
    /* partial entry word g of block idx over this thread's slots (slot0 mod sstride) */
    const u64 be = prm->be;
    const u32 r = (u32)prm->r, K = (u32)prm->k, span = posm_group_span(prm);
    u64 partial = 0;
    if (r == 0) {
        const u32 nslots = span / POSM_HW;                        /* words of group g in slots of POSM_HW */
        const u64 khi = posm_khi(pw, span), k1 = pw[0];
        for (u32 s = slot0; s < nslots; s += sstride) {
            const u64 loc0 = (u64)s * POSM_HW, q0 = idx * be + (u64)g * span + loc0;
            u64 w[POSM_HW];
            for (u32 e = 0; e < POSM_HW; e++) w[e] = posm_fetch_word(mem, canon, prm, q0 + e);
            u64 h = posm_hash_word(w[POSM_HW - 1], khi);          /* Horner over the slot: sum_e v_e K^e */
            for (u32 e = POSM_HW - 1; e-- > 0;) h = addmod(mulmod(h, k1), posm_hash_word(w[e], khi));
            partial = addmod(partial, mulmod(h, posm_power(pw, 0, loc0 >> 6, loc0 & 63)));
        }
        return partial;
    }
    const u32 wpr = K / 8, tpr = r / POSM_CPT, nslots = span * tpr;
    const u64 mixkey = chain_mixkey(prm->k0, prm->k1, seed, prm->polyk);
    for (u32 s = slot0; s < nslots; s += sstride) {
        const u32 rl = s / tpr, c0 = (s % tpr) * POSM_CPT, row = g * span + rl;
        const u64 q0 = idx * be + (u64)row * wpr;
        i32 acc[POSM_CPT];
        for (u32 c = 0; c < POSM_CPT; c++) acc[c] = 0;
        for (u32 e = 0; e < wpr; e++) {
            u64 w = posm_fetch_word(mem, canon, prm, q0 + e);
            for (u32 bq = 0; bq < 8; bq++) {
                const i32 pc = (i32)((w >> (8 * bq)) & 0xFF);
                const u32 j = 8 * e + bq;
                for (u32 c = 0; c < POSM_CPT; c++) acc[c] += pc * (i32)mbuf[mtile_index(j, c0 + c, K)];
            }
        }
        for (u32 c = 0; c < POSM_CPT; c++)
            partial = addmod(partial, mulmod(mix_output(acc[c], mixkey), posm_power(pw, r, rl, c0 + c)));
    }
    return partial;
}

/* tree-reduce red[0..T) mod p into red[0]; T power of two; all threads call */
DEVFN void posm_reduce(SHPTR u64* red, u32 tid, u32 T) {
    SYNC();
    for (u32 s = T >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = addmod(red[tid], red[tid + s]);
        SYNC();
    }
}

/* chain state: st[0] = block index, st[1..2] = digest, st[3] = link seed */
INLINE void posm_init_state(CONSTPTR const posm_params* prm, THRPTR u64* st) {
    st[0] = prm->j0;
    if (prm->d0_lo == 0 && prm->d0_hi == 0) {
        struct u128 d0 = siphash24_128(prm->k0, prm->k1, prm->nonce, 0, 0, 0);
        st[1] = d0.lo; st[2] = d0.hi;
    } else { st[1] = prm->d0_lo; st[2] = prm->d0_hi; }
    st[3] = prm->s0 ? prm->s0 : seed_from_digest(prm, st[2]);
}
/* fold the G entry words: H = sum_g T[g] * K_n * K_f^g */
INLINE u64 posm_fold_entry(CONSTPTR const posm_params* prm, DEVPTR const u64* pw, SHPTR const u64* T) {
    const u32 G = (u32)prm->g;
    const u64 off = POSM_T1LEN((u32)prm->r) + posm_t2len(prm);
    u64 H = 0;
    for (u32 g = 0; g < G; g++) H = addmod(H, mulmod(T[g], pw[off + g]));
    return H;
}
/* Early stop, for a cheat that answers some links off the GPU (memory_accounting.prover.strategies.ShareCheat): when the
 * block of launch-relative link `link` is at or past jstop, the chain stops before it. The digest so far goes to
 * out[0..1] and the block to out[2 + link] with POSM_STOP_BIT set (the links before it wrote theirs plainly, so
 * the first entry with the bit is this one); the caller then marks the parameter block aborted, so every later
 * kernel of the chain returns at once, and the host answers that link and goes on. */
#define POSM_STOP_BIT (U64C(1) << 63)
INLINE int posm_stop_at(CONSTPTR const posm_params* prm, THRPTR const u64* st, u32 link, DEVPTR u64* out) {
    if (!prm->jstop || link >= (u32)prm->n || st[0] < prm->jstop ||
        (prm->jstop_hi && st[0] >= prm->jstop_hi)) return 0;
    out[0] = st[1]; out[1] = st[2]; out[2 + link] = st[0] | POSM_STOP_BIT;
    return 1;
}
INLINE void posm_finish_link(CONSTPTR const posm_params* prm, THRPTR u64* st, u64 H, u32 link, DEVPTR u64* out) {
    struct u128 d = siphash24_128(prm->k0, prm->k1, prm->nonce + prm->link0 + link, st[1], st[2], H);
    out[2 + link] = st[0];
    st[1] = d.lo; st[2] = d.hi; st[0] = d.lo % prm->nb; st[3] = seed_from_digest(prm, d.hi);
}

/* ---- kernel A: one persistent threadgroup runs the whole chain, refilling the seed tables itself --- */
DEVFN void posm_chain_body(DEVPTR const u64* mem, DEVPTR const u64* canon, DEVPTR u64* out,
                            CONSTPTR const posm_params* prm, DEVPTR i8* mbuf, DEVPTR u64* pw,
                            SHPTR u64* red, SHPTR u64* st, SHPTR u64* T, u32 tid, u32 Tn) {
    const u32 n = (u32)prm->n, G = (u32)prm->g, r = (u32)prm->r;
    const u64 pwcount = POSM_T1LEN(r) + posm_t2len(prm) + G;
    if (tid == 0) { u64 s[4]; posm_init_state(prm, s); st[0] = s[0]; st[1] = s[1]; st[2] = s[2]; st[3] = s[3]; }
    SYNC();
    for (u32 link = 0; link < n; link++) {
        const u64 seed = st[3];
        for (u64 i = tid; i < pwcount; i += Tn) posm_powtable_fill(pw, prm, seed, i, pwcount);
        if (r) { posm_matrix_fill(mbuf, prm, seed, tid, Tn); SYNC(); posm_colconst_fill(mbuf, prm, tid, Tn); }
        SYNC();
        for (u32 g = 0; g < G; g++) {
            red[tid] = posm_partial_slots(mem, canon, prm, mbuf, pw, seed, st[0], g, tid, Tn);
            posm_reduce(red, tid, Tn);
            if (tid == 0) T[g] = red[0];
            SYNC();
        }
        if (tid == 0) {
            u64 s[4]; s[0] = st[0]; s[1] = st[1]; s[2] = st[2]; s[3] = st[3];
            posm_finish_link(prm, s, posm_fold_entry(prm, pw, T), link, out);
            st[0] = s[0]; st[1] = s[1]; st[2] = s[2]; st[3] = s[3];
        }
        SYNC();
    }
    if (tid == 0) { out[0] = st[1]; out[1] = st[2]; }
}

/* ---- kernel B: whole-GPU link = grid of partial blocks (per group) + a finish block ------------ */
DEVFN void posm_link_partial_body(DEVPTR const u64* mem, DEVPTR const u64* canon, CONSTPTR const posm_params* prm,
                                   DEVPTR const u64* state, DEVPTR u64* partials, DEVPTR const i8* mbuf,
                                   DEVPTR const u64* pw, SHPTR u64* red, u32 gid, u32 Gr, u32 tid, u32 Tn) {
    const u32 G = (u32)prm->g;
    for (u32 g = 0; g < G; g++) {
        red[tid] = posm_partial_slots(mem, canon, prm, mbuf, pw, state[3], state[0], g, gid * Tn + tid, Gr * Tn);
        posm_reduce(red, tid, Tn);
        if (tid == 0) partials[(u64)gid * G + g] = red[0];
        SYNC();
    }
}
/* reduce partials[grid][G] -> T[G]; if `table` != 0 write T there (setup table pass), else fold into the chain */
DEVFN void posm_link_finish_body(CONSTPTR const posm_params* prm, DEVPTR u64* state, DEVPTR const u64* partials,
                                  DEVPTR u64* out, DEVPTR const u64* pw, DEVPTR u64* table, SHPTR u64* red,
                                  SHPTR u64* T, u32 Gr, u32 link, u32 tid, u32 Tn) {
    const u32 G = (u32)prm->g;
    for (u32 g = tid; g < G; g += Tn) {
        u64 a = 0;
        for (u32 i = 0; i < Gr; i++) a = addmod(a, partials[(u64)i * G + g]);
        T[g] = a;
        if (table) table[g] = a;
    }
    SYNC();
    if (table) return;
    if (tid == 0) {
        u64 s[4]; s[0] = state[0]; s[1] = state[1]; s[2] = state[2]; s[3] = state[3];
        posm_finish_link(prm, s, posm_fold_entry(prm, pw, T), link, out);
        state[0] = s[0]; state[1] = s[1]; state[2] = s[2]; state[3] = s[3];
        if (link + 1 == (u32)prm->n) { out[0] = s[1]; out[1] = s[2]; }
        else if (posm_stop_at(prm, s, link + 1, out)) ((posm_params*)prm)->mode = POSM_MODE_ABORT;
    }
}
INLINE void posm_init_body(CONSTPTR const posm_params* prm, DEVPTR u64* state) {
    u64 s[4]; posm_init_state(prm, s); state[0] = s[0]; state[1] = s[1]; state[2] = s[2]; state[3] = s[3];
}

/* setup: out[q] = Enc(key, raw[perm(q)]) ; prm: 0 count, 1 enc, 2 key_lo, 3 perm_on, 4 perm_seed, 5 bits, 6 key_hi */
INLINE void posm_setup_one(DEVPTR const u64* raw, DEVPTR u64* out, CONSTPTR const posm_setup_params* sp, u64 q) {
    if (q >= sp->count) return;
    u64 src = sp->perm_on ? feistel_perm(q, sp->perm_seed, (u32)sp->perm_bits) : q;
    u64 d = raw[src];
    out[q] = sp->enc ? vde_encode(d, tweak_at(sp->key_lo, sp->key_hi, q)) : d;
}
/* setup at an offset: out[q] = Enc(key, raw[q]) with the tweak of global position q0 + q (no permutation) */
INLINE void posm_setup_at_one(DEVPTR const u64* raw, DEVPTR u64* out, u64 key_lo, u64 key_hi, u64 q0, u64 q, u64 count) {
    if (q >= count) return;
    out[q] = vde_encode(raw[q], tweak_at(key_lo, key_hi, q0 + q));
}
/* decode: out[q] = Dec(mem[q]) ; prm: 0 count, 1 enc, 2 key_lo, 6 key_hi */
INLINE void posm_decode_one(DEVPTR const u64* mem, DEVPTR u64* out, CONSTPTR const posm_setup_params* sp, u64 q) {
    if (q >= sp->count) return;
#if POSM_PACKED128
    if (!sp->enc) { out[q] = mem[q]; return; }
    u64 row = q & ~U64C(15);
    u32 slot = q % 16;
    u8 *dst = (u8*)(out + row);
    if (slot < 11)
        f89_plain_bytes(dst, 128, 11*slot, packed_decode((const u32*)(mem+row), slot, packed_tweak(sp->key_lo, sp->key_hi, row, slot)));
    else if (slot == 11) for (u32 i=121; i<128; ++i) dst[i]=0;
#elif POSM_VDE_BITS == 89
    if (!sp->enc) { out[q] = mem[q]; return; }
    u64 row = q & ~U64C(7);
    u32 lane = (u32)(q - row);
    u8 *dst = (u8*)(out + row);
    if (lane < 5) {
        field89 base = f89_base(sp->key_lo, sp->key_hi, row);
        field89 tweak = f89_slot_tweak(sp->key_lo, sp->key_hi, row, lane, base);
        field89 plain = f89_read_root((const u32*)(mem + row) + 3*lane, tweak);
        f89_plain_bytes(dst, 64, 12*lane, plain);
        dst[12*lane + 11] = 0;
    } else if (lane == 5) {
        ((u32*)dst)[15] = 0;
    }
#else
    out[q] = sp->enc ? vde_decode(mem[q], tweak_at(sp->key_lo, sp->key_hi, q)) : mem[q];
#endif
}

/* One packing group of the decoded stream: 16 encoded words at payload index q0 + 16 g -> 15 raw words.
 * (What inference reads: a tensor's bytes are whole groups, see memory_accounting.setup.layout.) */
INLINE void posm_decode_unpack_group(DEVPTR const u64* theta, DEVPTR u64* raw, u64 q0, u64 g, u64 key_lo, u64 key_hi) {
#if POSM_PACKED128
    for (u32 i=0; i<8; ++i) {
        u64 w=g*128 + i*16;
        packed_decode_row((const u32*)(theta+q0+w), (u8*)(raw+g*121), 968, 121*i, q0+w, key_lo, key_hi);
    }
#elif POSM_VDE_BITS == 89
    for(u32 row=0;row<8;row++) {
        u64 q=q0+g*64+8*row;
        f89_transform_row((u8*)(raw+55*g),440,55*row,(u64*)theta+q,q,key_lo,key_hi,1);
    }
#else
    u64 p[16], w[15];
#if POSM_GROUP64
    for (u32 j=0;j<16;j+=8) {
        u64 q=q0+16*g+j, base=tweak_prf(key_lo,key_hi,q);
        for (u32 lane=0;lane<8;lane++) p[j+lane]=vde_decode(theta[q+lane],addmod(base,lane));
    }
#else
    for (u32 i = 0; i < 16; i++) { u64 q = q0 + 16 * g + i; p[i] = vde_decode(theta[q], tweak_at(key_lo, key_hi, q)); }
 #endif
    unpack_group(p, w);
    for (u32 a = 0; a < 15; a++) raw[15 * g + a] = w[a];
#endif
}
/* One group of the setup direction: 15 raw words at group g -> 16 payload words (not yet encoded). */
INLINE void posm_pack_group_at(DEVPTR const u64* raw, DEVPTR u64* payload, u64 g) {
#if POSM_PACKED128
    const u8 *src = (const u8*)(raw + 121*g);
    u8 *dst = (u8*)(payload + 128*g);
    for (u32 r=0; r<8; ++r) for (u32 i=0; i<128; ++i)
        dst[128*r+i] = i<121 ? src[121*r+i] : 0;
#elif POSM_VDE_BITS == 89
    u8 *dst=(u8*)(payload + g*64);
    const u8 *src=(const u8*)(raw + g*55);
    for(u32 row=0;row<8;row++) {
        for(u32 lane=0;lane<5;lane++) {
            for(u32 j=0;j<11;j++) dst[64*row+12*lane+j]=src[55*row+11*lane+j];
            dst[64*row+12*lane+11]=0;
        }
        ((u32*)(dst+64*row))[15]=0;
    }
#else
    u64 w[15], p[16];
    for (u32 a = 0; a < 15; a++) w[a] = raw[15 * g + a];
    pack_group(w, p);
    for (u32 i = 0; i < 16; i++) payload[16 * g + i] = p[i];
#endif
}

/* ---- encoded dynamic memory (memory_accounting.prover.kvenc): a unit of rw plaintext words (a KV page, a cache buffer, the
 * activation rows) held as 16 * ceil(rw / 15) VDE words under the dynamic key, word i at region position q0 + i.
 * Its tweak positions are in a domain of their own (KV_POS set): Θ', the filler and the dynamic filler are
 * encoded at positions below 2^62, so no word of theirs equals one of these under any key. Plaintext words
 * past rw are zero (rw = 0: a unit of zeros, raw is not read). */
INLINE void posm_kv_encode_group(DEVPTR const u64* raw, u64 rw, DEVPTR u64* enc, u64 q0, u64 g, u64 key_lo, u64 key_hi) {
    u64 w[15], p[16];
    for (u32 a = 0; a < 15; a++) { u64 i = 15 * g + a; w[a] = i < rw ? raw[i] : 0; }
    pack_group(w, p);
    for (u32 i = 0; i < 16; i++) enc[16 * g + i] = vde_encode(p[i], tweak_at(key_lo, key_hi, KV_POS | (q0 + 16 * g + i)));
}
INLINE void posm_kv_decode_group(DEVPTR const u64* enc, u64 q0, DEVPTR u64* raw, u64 rw, u64 g, u64 key_lo, u64 key_hi) {
    u64 p[16], w[15];
    for (u32 i = 0; i < 16; i++) p[i] = vde_decode(enc[16 * g + i], tweak_at(key_lo, key_hi, KV_POS | (q0 + 16 * g + i)));
    unpack_group(p, w);
    for (u32 a = 0; a < 15; a++) { u64 i = 15 * g + a; if (i < rw) raw[i] = w[a]; }
}
/* One word of a group's encoding, k of 16: payload word k is bits [60k, 60k + 60) of the group's 15 plaintext words, so
 * a thread per word (the CUDA kernel) has 59 serial squarings of latency, not 16 * 59 (a decode step encodes a few groups
 * per layer, and that latency is on the inference stream). */
INLINE void posm_kv_encode_word(DEVPTR const u64* raw, u64 rw, DEVPTR u64* enc, u64 q0, u64 g, u32 k, u64 tweak) {
    u32 start = 60 * k, a = start >> 6, off = start & 63;
    u64 i = 15 * g + a;
    u64 lo = i < rw ? raw[i] : 0;
    u64 v = lo >> off;
    if (off > 4 && a + 1 < 15) { u64 hi = i + 1 < rw ? raw[i + 1] : 0; v |= hi << (64 - off); }
    v &= (U64C(1) << 60) - 1;
    enc[16 * g + k] = vde_encode(v, tweak);
}
/* Many units in one call: entry j is 4 words (raw word offset, encoded word offset, first group, groups); its
 * plaintext is at raw_base + 8 * e[0], its encoded words at enc_base + 8 * e[1], at region position q_base + e[1]. */
INLINE void posm_kv_entry_group(u64 raw_base, u64 enc_base, u64 q_base, CONSTPTR const u64* e, u64 rw, u64 g, u64 key_lo, u64 key_hi, int decode) {
    const u64* raw = rw ? (const u64*)(uintptr_t)raw_base + e[0] : 0;
    u64* enc = (u64*)(uintptr_t)enc_base + e[1];
#if POSM_PACKED128
    u64 w = 16*g, pos = KV_POS | (q_base+e[1]+w);
    if (decode) packed_decode_row((const u32*)(enc+w), (u8*)raw, 8*rw, 121*g, pos, key_lo, key_hi);
    else packed_encode_row((const u8*)raw, 8*rw, 121*g, (u32*)(enc+w), pos, key_lo, key_hi);
#elif POSM_VDE_BITS == 89
    for (u32 row=0;row<8;row++) {
        u64 w=g*64+row*8;
        f89_transform_row((u8*)raw,rw*8,g*440+row*55,enc+w,KV_POS | (q_base+e[1]+w),key_lo,key_hi,decode);
    }
#else
    if (decode) posm_kv_decode_group(enc, q_base + e[1], (u64*)raw, rw, g, key_lo, key_hi);
    else        posm_kv_encode_group(raw, rw, enc, q_base + e[1], g, key_lo, key_hi);
#endif
}

/* setup: the one-way image of a whole array, one group per call */
INLINE void posm_image_one(DEVPTR const u64* theta, DEVPTR u64* out, u64 g, u64 groups, u64 g0) {
    if (g >= groups) return;
    u64 w[4], t[4];
    for (u32 i = 0; i < 4; i++) w[i] = theta[4 * g + i];
    image_group(g0 + g, w, t);                 /* g0: global group index of this chunk's first group */
    for (u32 i = 0; i < 4; i++) out[4 * g + i] = t[i];
}

#endif /* POSM_CORE_H */
