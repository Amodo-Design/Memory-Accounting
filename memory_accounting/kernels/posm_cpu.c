/*
 * posm_cpu.c -- CPU side of the prototype.
 *   posm_chain_cpu     : serial reference (the verifier's expected answer, tests)
 *   posm_chain_cpu_mt  : row-parallel chain on a thread pool -- the "weights in DRAM, matmul on
 *                        the CPU" cheater, and the verifier's fast path
 *   posm_setup_cpu / posm_decode_cpu : small-array reference for tests
 */
#include "posm_core.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
#include <immintrin.h>
#define POSM_VNNI 1     /* u8 x s8 dot product: 4 multiply-adds per int32 lane per instruction */
#endif

#define MAXT 64

/* Seed-dependent tables of a link: K_s powers, mixer key, matrix M (row-major K x r; Mp = VNNI packing
 * [j/4][c][j%4]).  Built only when the link seed (or the shape) changes, so with x == 0 -- seed = nonce
 * for every link -- once per chain, like the GPU without per_link_prep; with x > 0 per link.  Kept per
 * calling thread across calls, so single-link continuations of one challenge (verifier streaming, the
 * setup table pass over every block) also build them once. */
enum { CK_SEED, CK_K0, CK_K1, CK_POLYK, CK_K, CK_R, CK_BE, CK_G, CK_N };
typedef struct {
    int valid; u64 key[CK_N];
    u64 seed, ks, mixkey, khi; u32 K, r, wpr, rows, G, span, t1len, t2len;
    i8* M; i8* Mp; size_t mcap; u64* T1; u64* T2; u32 t1cap, t2cap;
} ctx_t;

static void ctx_destroy(void* p) { ctx_t* c = (ctx_t*)p; free(c->M); free(c->Mp); free(c->T1); free(c->T2); free(c); }
static pthread_key_t ctx_key; static pthread_once_t ctx_once = PTHREAD_ONCE_INIT;
static void ctx_key_init(void) { pthread_key_create(&ctx_key, ctx_destroy); }
static ctx_t* thread_ctx(void) {
    pthread_once(&ctx_once, ctx_key_init);
    ctx_t* c = (ctx_t*)pthread_getspecific(ctx_key);
    if (!c) { c = (ctx_t*)calloc(1, sizeof *c); pthread_setspecific(ctx_key, c); }
    return c;
}

/* Bring c up to date for `seed`.  Power tables by repeated multiplication (r + span multiplies, where a
 * powmod per entry cost ~100); returns 1 when the matrix must be (re)filled by ctx_fill_rows over [0, K). */
static int ctx_prepare(ctx_t* c, const posm_params* prm, u64 seed) {
    const u64 key[CK_N] = { seed, prm->k0, prm->k1, prm->polyk, prm->k, prm->r, prm->be, prm->g };
    if (c->valid && !memcmp(c->key, key, sizeof key)) return 0;
    memcpy(c->key, key, sizeof key); c->valid = 1;
    c->seed = seed;
    c->ks = chain_polyk(prm->k0, prm->k1, seed, prm->polyk);
    c->mixkey = chain_mixkey(prm->k0, prm->k1, seed, prm->polyk);
    c->K = (u32)prm->k; c->r = (u32)prm->r; c->wpr = c->K / 8; c->G = (u32)prm->g; c->span = posm_group_span(prm);
    c->rows = c->r ? (u32)(prm->be / c->wpr) : (u32)(prm->be / POSM_HW);      /* work units: rows, or HW-word slots */
    c->t1len = POSM_T1LEN(c->r); c->t2len = c->r ? c->span : (c->span + 63) / 64;
    if (c->t1cap < c->t1len) { free(c->T1); c->T1 = (u64*)malloc(sizeof(u64) * c->t1len); c->t1cap = c->t1len; }
    if (c->t2cap < c->t2len) { free(c->T2); c->T2 = (u64*)malloc(sizeof(u64) * c->t2len); c->t2cap = c->t2len; }
    c->T1[0] = c->ks;                                             /* T1[c] = K^(c+1) */
    for (u32 i = 1; i < c->t1len; i++) c->T1[i] = mulmod(c->T1[i - 1], c->ks);
    const u64 step = c->T1[c->t1len - 1];                         /* T2[a] = K^(a * T1LEN): T1LEN = r, or 64 at r = 0 */
    c->T2[0] = 1;
    for (u32 a = 1; a < c->t2len; a++) c->T2[a] = mulmod(c->T2[a - 1], step);
    c->khi = c->r ? 0 : mulmod(c->T2[(c->span - 1) >> 6], c->T1[(c->span - 1) & 63]);   /* = posm_khi */
    if (!c->r) return 0;
    const size_t mbytes = (size_t)c->K * c->r;
    if (c->mcap < mbytes) {
        free(c->M); c->M = (i8*)malloc(mbytes);
#ifdef POSM_VNNI
        free(c->Mp); c->Mp = (i8*)aligned_alloc(64, mbytes);
#endif
        c->mcap = mbytes;
    }
    return 1;
}
/* matrix rows [j0, j1) (multiples of 4, so the VNNI packing of these rows reads only these rows): pool
 * workers fill disjoint slices in parallel */
static void ctx_fill_rows(const ctx_t* c, u32 j0, u32 j1) {
    const u32 r = c->r;
    const u64 ms = matrix_seed(c->seed);
    for (u64 i = (u64)j0 * r; i < (u64)j1 * r; i += 8) {           /* entry index i = j*r + c; r % 8 == 0 */
        u64 g = splitmix64(ms + (i >> 3));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        memcpy(c->M + i, &g, 8);                                  /* byte q = (g >> 8q) & 0xFF, one store */
#else
        for (u32 q = 0; q < 8; q++) c->M[i + q] = (i8)(u8)(g >> (8 * q));
#endif
    }
#ifdef POSM_VNNI
    for (u32 j = j0; j < j1; j++) for (u32 cc = 0; cc < r; cc++) c->Mp[((size_t)(j >> 2) * r + cc) * 4 + (j & 3)] = c->M[(size_t)j * r + cc];
#endif
}

/* Words [w0, w1) of block idx, as p with p[0] = word w0: contiguous `mem`, or gathered through the segment
 * table and/or imaged into a thread-local scratch.  Each worker gathers and images only its own range. */
static __thread u64* tls_blk = NULL; static __thread size_t tls_blk_n = 0;
static const u64* block_words(const u64* mem, const posm_params* prm, u64 idx, u64 w0, u64 w1) {
    const u64 q0 = idx * prm->be + w0, n = w1 - w0;
    const u32 nseg = (u32)prm->nseg;
    const int img = prm->img != 0;
    if (!nseg && !img) return mem + q0;
    if (tls_blk_n < n) { free(tls_blk); tls_blk = (u64*)malloc(sizeof(u64) * n); tls_blk_n = n; }
    if (nseg) {                                                   /* segment by segment: one lookup per run, not per word */
        const u64* segtab = (const u64*)(uintptr_t)prm->segtab;
        for (u64 e = 0; e < n;) {
            const u64 q = q0 + e;
            u32 lo = 0, hi = nseg;
            while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (segtab[mid] <= q) lo = mid; else hi = mid; }
            const u64* base = (const u64*)(uintptr_t)segtab[nseg + 1 + lo];
            u64 run = segtab[lo + 1] - q; if (run > n - e) run = n - e;
            memcpy(tls_blk + e, base + (q - segtab[lo]), sizeof(u64) * run);
            e += run;
        }
    } else {
        memcpy(tls_blk, mem + q0, sizeof(u64) * n);
    }
    if (img) for (u64 g = 0; g < n / 4; g++) { u64 t[4]; image_group((prm->qoff + q0) / 4 + g, tls_blk + 4 * g, t); memcpy(tls_blk + 4 * g, t, sizeof t); }
    return tls_blk;
}

/* K^(L+1) for local position L */
static inline u64 kpow(const ctx_t* c, u64 L) { return mulmod(c->T2[L >> 6], c->T1[L & 63]); }
/* r = 0: sum over i < cnt of v(w[i]) K^(L0+i+1), v = posm_hash_word, as 8 interleaved Horner streams:
 * stream s holds positions L0 + s + 8m, evaluated in K^8 from the top down, so a word costs one multiply
 * into its stream (plus the half-word combine) and the 8 streams pipeline instead of forming one
 * latency-bound dependency chain. */
static u64 hash_run(const ctx_t* c, const u64* w, u64 L0, u64 cnt) {
    const u64 khi = c->khi, k8 = c->T1[7];
    u64 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const u64 m8 = cnt / 8;
    for (u64 m = m8; m-- > 0;) {
        const u64* p = w + 8 * m;
        for (u32 s = 0; s < 8; s++) acc[s] = addmod(mulmod(acc[s], k8), posm_hash_word(p[s], khi));
    }
    u64 partial = 0;
    if (m8) for (u32 s = 0; s < 8; s++) partial = addmod(partial, mulmod(acc[s], kpow(c, L0 + s)));
    for (u64 i = 8 * m8; i < cnt; i++) partial = addmod(partial, mulmod(posm_hash_word(w[i], khi), kpow(c, L0 + i)));
    return partial;
}

/* Accumulate the entry words of block idx over work units [ub, ue) into T[G] (mod p).
 *   r == 0 : unit = POSM_HW consecutive words; 8 interleaved Horner streams per group
 *   r >  0 : unit = one row; register-tiled GEMM, 4 rows x CTILE columns per tile (VNNI or vector ext) */
#define ROWBLK 4
#define CTILE  16
typedef i32 v16i __attribute__((vector_size(64)));
typedef i8  v16b __attribute__((vector_size(16)));
static void rows_partial(const u64* mem, const posm_params* prm, const ctx_t* c, u64 idx, u32 ub, u32 ue, u64* T) {
    if (ue <= ub) return;
    if (c->r == 0) {
        /* words [w0, w1); group of word i = i / span, local position = i % span */
        const u64 w0 = (u64)ub * POSM_HW, w1 = (u64)ue * POSM_HW;
        const u64* blk = block_words(mem, prm, idx, w0, w1);
        for (u64 i = w0; i < w1;) {
            const u32 g = (u32)(i / c->span);
            const u64 gend = (u64)(g + 1) * c->span, stop = gend < w1 ? gend : w1;
            T[g] = addmod(T[g], hash_run(c, blk + (i - w0), i - (u64)g * c->span, stop - i));
            i = stop;
        }
        return;
    }
    const u32 r = c->r;
    const u64* blk = block_words(mem, prm, idx, (u64)ub * c->wpr, (u64)ue * c->wpr);
    for (u32 row0 = ub; row0 < ue; row0 += ROWBLK) {
        const u32 nrow = (ue - row0 < ROWBLK) ? (ue - row0) : ROWBLK;
        const u8* rows[ROWBLK];
        for (u32 t = 0; t < ROWBLK; t++) rows[t] = (const u8*)(blk + (u64)(row0 - ub + (t < nrow ? t : 0)) * c->wpr);
        for (u32 c0 = 0; c0 < r; c0 += CTILE) {
            i32 out[ROWBLK][CTILE];
#ifdef POSM_VNNI
            {
                __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
                const i8* mp = c->Mp + (size_t)c0 * 4;
                for (u32 j4 = 0; j4 < c->K / 4; j4++, mp += (size_t)r * 4) {
                    const __m512i m = _mm512_load_si512((const void*)mp);
                    a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(*(const i32*)(rows[0] + 4 * j4)), m);
                    a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(*(const i32*)(rows[1] + 4 * j4)), m);
                    a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(*(const i32*)(rows[2] + 4 * j4)), m);
                    a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(*(const i32*)(rows[3] + 4 * j4)), m);
                }
                _mm512_storeu_si512((void*)out[0], a0); _mm512_storeu_si512((void*)out[1], a1);
                _mm512_storeu_si512((void*)out[2], a2); _mm512_storeu_si512((void*)out[3], a3);
            }
#else
            {
                v16i acc0 = {0}, acc1 = {0}, acc2 = {0}, acc3 = {0};
                for (u32 j = 0; j < c->K; j++) {
                    const i32 p0 = rows[0][j], p1 = rows[1][j], p2 = rows[2][j], p3 = rows[3][j];
                    v16b mb; memcpy(&mb, c->M + (size_t)j * r + c0, sizeof mb);
                    const v16i m = __builtin_convertvector(mb, v16i);
                    acc0 += p0 * m; acc1 += p1 * m; acc2 += p2 * m; acc3 += p3 * m;
                }
                memcpy(out[0], &acc0, sizeof acc0); memcpy(out[1], &acc1, sizeof acc1);
                memcpy(out[2], &acc2, sizeof acc2); memcpy(out[3], &acc3, sizeof acc3);
            }
#endif
            for (u32 t = 0; t < nrow; t++) {
                const u32 row = row0 + t, g = row / c->span, rl = row % c->span;
                const u64 t2 = c->T2[rl];
                u64 partial = 0;
                for (u32 q = 0; q < CTILE; q++) partial = addmod(partial, mulmod(mix_output(out[t][q], c->mixkey), mulmod(t2, c->T1[c0 + q])));
                T[g] = addmod(T[g], partial);
            }
        }
    }
}

/* H = sum_g T[g] * K_n * K_f^g (posm_fold_entry over the PWN table) */
static u64 fold_entry(const posm_params* prm, u64 kn, u64 kf, const u64* T) {
    u64 H = 0, kp = kn;
    for (u32 g = 0; g < (u32)prm->g; g++) { H = addmod(H, mulmod(T[g], kp)); kp = mulmod(kp, kf); }
    return H;
}
static void init_state(const posm_params* prm, u64* st) { posm_init_state(prm, st); }
static void finish_link(const posm_params* prm, u64* st, u64 H, u32 link, u64* out) { posm_finish_link(prm, st, H, link, out); }
static ctx_t* ctx_serial(const posm_params* prm, u64 seed) {
    ctx_t* c = thread_ctx();
    if (ctx_prepare(c, prm, seed)) ctx_fill_rows(c, 0, c->K);
    return c;
}

/* entry T_seed[idx] of one block (setup table pass, verifier reference) */
void posm_entry_cpu(const u64* mem, const posm_params* prm, u64 seed, u64 idx, u64* T) {
    const ctx_t* c = ctx_serial(prm, seed);
    memset(T, 0, sizeof(u64) * c->G);
    rows_partial(mem, prm, c, idx, 0, c->rows, T);
}

void posm_chain_cpu(const u64* mem, const posm_params* prm, u64* out) {
    const u32 n = (u32)prm->n, G = (u32)prm->g;
    const u64 kn = chain_polyk(prm->k0, prm->k1, prm->nonce, prm->polyk);
    const u64 kf = chain_foldk(prm->k0, prm->k1, prm->nonce, prm->polyk);
    u64 T[POSM_GMAX];
    u64 st[4]; init_state(prm, st);
    for (u32 link = 0; link < n; link++) {
        const ctx_t* c = ctx_serial(prm, st[3]);
        memset(T, 0, sizeof(u64) * G);
        rows_partial(mem, prm, c, st[0], 0, c->rows, T);
        finish_link(prm, st, fold_entry(prm, kn, kf, T), link, out);
    }
    out[0] = st[1]; out[1] = st[2];
}

/* ---- thread pool ---------------------------------------------------------
 * Workers block on a condition variable while no chain is in flight (so they do not steal CPU
 * from the honest prover's driver thread between challenges) and spin on the generation counter
 * between jobs while a chain is active (a cheater's dedicated cores would do the same).  A job runs
 * on every pool thread, the caller as thread 0: per link the matrix of a new seed (row slices), then
 * the block.  The block is handed out in chunks from an atomic counter rather than as equal shares,
 * so on a host with fast and slow cores (Apple P/E) nobody waits for the slowest share; the entry
 * words are sums mod p, so the digest does not depend on who took which chunk.  A job runs on only as
 * many threads as it has chunks (`pool.p`; the others skip it, but still acknowledge it, so the caller
 * never writes the next job while a worker could read the current one: no worker runs a job twice).  One caller at a time (`call_mu`): the pool is process-wide, and two
 * threads driving it at once (a CPU prover and the verifier's checker in one process) would corrupt it. */
typedef void (*job_fn)(int i, int n);
static struct {
    pthread_t th[MAXT]; int n; atomic_int stop; atomic_int active;
    pthread_mutex_t mu; pthread_cond_t cv; pthread_mutex_t call_mu;
    atomic_uint gen; atomic_uint done; atomic_uint next; int p;
    job_fn job; const u64* mem; const posm_params* prm; ctx_t* ctx; u64 idx; u32 chunk;
    u64 partial[MAXT][POSM_GMAX];
} pool;

static void job_matrix(int i, int n) {                           /* chunks of 16 matrix rows */
    (void)i; (void)n;
    const u32 K = pool.ctx->K;
    for (;;) {
        const u32 j0 = 16 * atomic_fetch_add_explicit(&pool.next, 1, memory_order_relaxed);
        if (j0 >= K) break;
        ctx_fill_rows(pool.ctx, j0, j0 + 16);                     /* K % 16 == 0 */
    }
}
static void job_rows(int i, int n) {
    (void)n;
    const u32 rows = pool.ctx->rows, chunk = pool.chunk;
    memset(pool.partial[i], 0, sizeof(u64) * pool.ctx->G);
    for (;;) {
        const u64 rb = (u64)atomic_fetch_add_explicit(&pool.next, 1, memory_order_relaxed) * chunk;
        if (rb >= rows) break;
        rows_partial(pool.mem, pool.prm, pool.ctx, pool.idx, (u32)rb, (u32)(rb + chunk < rows ? rb + chunk : rows), pool.partial[i]);
    }
}
/* ~8 chunks per thread, whole ROWBLK row tiles (r > 0) or whole 8-word Horner rounds (r = 0), and at
 * least a few microseconds of work per chunk: below that a thread costs more to wake and gather than its
 * share saves.  A matmul row is k x r multiply-adds (any one row is enough); a hash-only unit is 4 words. */
#define POOL_MIN_UNITS_R0 256
static u32 pool_chunk(const ctx_t* c, int n) {
    u32 ch = c->rows / (8 * (u32)n);
    if (!c->r && ch < POOL_MIN_UNITS_R0) ch = POOL_MIN_UNITS_R0;
    ch = (ch + 3) & ~3u;
    return ch ? ch : 4;                                           /* never 0: a 0-row chunk never advances */
}
/* threads that take part in a job of `units` chunks: at most one per chunk, the caller included */
static int pool_parts(u64 units) {
    return units >= (u64)pool.n ? pool.n : (units ? (int)units : 1);
}
static void pool_run(job_fn f, int p) {
    pool.job = f; pool.p = p;
    atomic_store_explicit(&pool.next, 0, memory_order_relaxed);
    atomic_store_explicit(&pool.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool.gen, 1, memory_order_release);
    f(0, pool.n);
    while (atomic_load_explicit(&pool.done, memory_order_acquire) < (unsigned)(pool.n - 1)) {}   /* every worker, run or skip */
}
static void* worker(void* arg) {
    int i = (int)(intptr_t)arg; unsigned local = 0;
    for (;;) {
        pthread_mutex_lock(&pool.mu);
        while (!atomic_load(&pool.active) && !atomic_load(&pool.stop)) pthread_cond_wait(&pool.cv, &pool.mu);
        pthread_mutex_unlock(&pool.mu);
        if (atomic_load(&pool.stop)) return NULL;
        while (atomic_load(&pool.active) && !atomic_load(&pool.stop)) {
            const unsigned g = atomic_load_explicit(&pool.gen, memory_order_acquire);
            if (g == local) continue;
            /* the caller waits for every worker's acknowledgement before it writes the next job, so the job and
             * its thread count read here (after the acquire of g) are generation g's and stay so until we ack */
            local = g;
            if (i < pool.p) pool.job(i, pool.n);                  /* not needed for this job: acknowledge only */
            atomic_fetch_add_explicit(&pool.done, 1, memory_order_release);
        }
    }
}
int posm_pool_start(int n) {
    if (n < 1) n = 1; if (n > MAXT) n = MAXT;
    if (pool.n) return pool.n;
    pool.n = n; atomic_store(&pool.stop, 0); atomic_store(&pool.active, 0); atomic_store(&pool.gen, 0);
    pthread_mutex_init(&pool.mu, NULL); pthread_cond_init(&pool.cv, NULL); pthread_mutex_init(&pool.call_mu, NULL);
    for (int i = 1; i < n; i++) pthread_create(&pool.th[i], NULL, worker, (void*)(intptr_t)i);
    return n;
}
void posm_pool_stop(void) {
    if (!pool.n) return;
    pthread_mutex_lock(&pool.mu); atomic_store(&pool.stop, 1); pthread_cond_broadcast(&pool.cv); pthread_mutex_unlock(&pool.mu);
    for (int i = 1; i < pool.n; i++) pthread_join(pool.th[i], NULL);
    pool.n = 0;
}
void posm_chain_cpu_mt(const u64* mem, const posm_params* prm, u64* out) {
    if (pool.n <= 1) { posm_chain_cpu(mem, prm, out); return; }
    const u32 n = (u32)prm->n, G = (u32)prm->g;
    const u64 kn = chain_polyk(prm->k0, prm->k1, prm->nonce, prm->polyk);
    const u64 kf = chain_foldk(prm->k0, prm->k1, prm->nonce, prm->polyk);
    u64 T[POSM_GMAX];
    pthread_mutex_lock(&pool.call_mu);
    pool.mem = mem; pool.prm = prm; pool.ctx = thread_ctx();
    pthread_mutex_lock(&pool.mu); atomic_store(&pool.active, 1); pthread_cond_broadcast(&pool.cv); pthread_mutex_unlock(&pool.mu);
    u64 st[4]; init_state(prm, st);
    for (u32 link = 0; link < n; link++) {
        if (ctx_prepare(pool.ctx, prm, st[3])) pool_run(job_matrix, pool_parts(pool.ctx->K / 16));   /* new seed: every link only when x > 0 */
        pool.idx = st[0]; pool.chunk = pool_chunk(pool.ctx, pool.n);
        const int p = pool_parts(((u64)pool.ctx->rows + pool.chunk - 1) / pool.chunk);
        pool_run(job_rows, p);
        memset(T, 0, sizeof(u64) * G);
        for (int i = 0; i < p; i++) for (u32 g = 0; g < G; g++) T[g] = addmod(T[g], pool.partial[i][g]);
        finish_link(prm, st, fold_entry(prm, kn, kf, T), link, out);
    }
    atomic_store(&pool.active, 0);
    pthread_mutex_unlock(&pool.call_mu);
    out[0] = st[1]; out[1] = st[2];
}


/* The device's scalar decomposition run serially: tables by posm_powtable_fill / posm_matrix_fill /
 * posm_colconst_fill and each entry word as the sum of posm_partial_slots over `lanes` strided threads,
 * the cheat fetches (modes 1, 3) included.  Lets the tests check the code posm.cu shares from posm_core.h
 * on a host without a GPU. */
void posm_chain_scalar_cpu(const u64* mem, const u64* canon, const posm_params* prm, u64* out, u32 lanes) {
    const u32 n = (u32)prm->n, G = (u32)prm->g, r = (u32)prm->r, K = (u32)prm->k;
    const u64 pwcount = POSM_T1LEN(r) + posm_t2len(prm) + G;
    u64* pw = (u64*)malloc(sizeof(u64) * pwcount);
    i8* mbuf = r ? (i8*)malloc(mbuf_colconst_offset(K, r) + 4 * (size_t)r) : NULL;
    u64 st[4], T[POSM_GMAX]; posm_init_state(prm, st);
    if (lanes < 1) lanes = 1;
    for (u32 link = 0; link < n; link++) {
        for (u64 i = 0; i < pwcount; i++) posm_powtable_fill(pw, prm, st[3], i, pwcount);
        if (r) { posm_matrix_fill(mbuf, prm, st[3], 0, 1); posm_colconst_fill(mbuf, prm, 0, 1); }
        for (u32 g = 0; g < G; g++) {
            T[g] = 0;
            for (u32 t = 0; t < lanes; t++) T[g] = addmod(T[g], posm_partial_slots(mem, canon, prm, mbuf, pw, st[3], st[0], g, t, lanes));
        }
        posm_finish_link(prm, st, posm_fold_entry(prm, pw, T), link, out);
    }
    out[0] = st[1]; out[1] = st[2];
    free(pw); free(mbuf);
}

void posm_setup_cpu(const u64* raw, u64* out, const posm_setup_params* sp) { for (u64 q = 0; q < sp->count; q++) posm_setup_one(raw, out, sp, q); }
void posm_decode_cpu(const u64* mem, u64* out, const posm_setup_params* sp) { for (u64 q = 0; q < sp->count; q++) posm_decode_one(mem, out, sp, q); }

/* ---- offline phase: range-parallel encode / decode (setup on a host without a GPU) ---------- */
typedef struct { const u64* in; u64* out; const posm_setup_params* sp; u64 lo, hi; int decode; } range_job_t;
static void* range_worker(void* a) {
    range_job_t* j = (range_job_t*)a;
    if (j->decode) for (u64 q = j->lo; q < j->hi; q++) posm_decode_one(j->in, j->out, j->sp, q);
    else           for (u64 q = j->lo; q < j->hi; q++) posm_setup_one(j->in, j->out, j->sp, q);
    return NULL;
}
static void range_run(const u64* in, u64* out, const posm_setup_params* sp, int nthreads, int decode) {
    if (nthreads < 1) nthreads = 1; if (nthreads > MAXT) nthreads = MAXT;
    pthread_t th[MAXT]; range_job_t jobs[MAXT];
    const u64 count = sp->count;
    for (int i = 0; i < nthreads; i++) {
        jobs[i] = (range_job_t){ in, out, sp, count * i / nthreads, count * (i + 1) / nthreads, decode };
        pthread_create(&th[i], NULL, range_worker, &jobs[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}
void posm_setup_at_cpu(const u64* raw, u64* out, u64 key_lo, u64 key_hi, u64 q0, u64 count) {
#if POSM_PACKED128
    for (u64 q=0; q<count; q+=16)
        packed_encode_row((const u8*)(raw+q),121,0,(u32*)(out+q),q0+q,key_lo,key_hi);
#elif POSM_VDE_BITS == 89
    for (u64 q=0;q<count;q+=8) f89_setup_row(raw+q,(u32*)(out+q),key_lo,key_hi,q0+q);
#else
    for (u64 q = 0; q < count; q++) posm_setup_at_one(raw, out, key_lo, key_hi, q0, q, count);
#endif
}
void posm_image_cpu(const u64* theta, u64* out, u64 groups) { for (u64 g = 0; g < groups; g++) posm_image_one(theta, out, g, groups, 0); }
void posm_image_at_cpu(const u64* theta, u64* out, u64 groups, u64 g0) {
    for (u64 g = 0; g < groups; g++) posm_image_one(theta, out, g, groups, g0);
}
typedef struct { const u64* in; u64* out; u64 lo, hi; } img_job_t;
static void* img_worker(void* a) { img_job_t* j = (img_job_t*)a; for (u64 g = j->lo; g < j->hi; g++) posm_image_one(j->in, j->out, g, j->hi, 0); return NULL; }
void posm_image_cpu_mt(const u64* theta, u64* out, u64 groups, int nthreads) {
    if (nthreads < 1) nthreads = 1; if (nthreads > MAXT) nthreads = MAXT;
    pthread_t th[MAXT]; img_job_t jobs[MAXT];
    for (int i = 0; i < nthreads; i++) { jobs[i] = (img_job_t){ theta, out, groups * i / nthreads, groups * (i + 1) / nthreads }; pthread_create(&th[i], NULL, img_worker, &jobs[i]); }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}
void posm_filler_cpu(u64* out, u64 seed, u64 start, u64 count) { for (u64 i = 0; i < count; i++) out[i] = filler_word(seed, start + i); }
/* The verifier's view of filler it never stored: the one-way image of Enc(key, filler(seed)) over global
 * word positions [q0, q0 + count), i.e. what posm_image_one would give over the prover's filler words.
 * q0 and count are multiples of 4 (whole 32-byte groups; the group index is q / 4). */
typedef struct { u64* out; u64 seed, klo, khi, q0, lo, hi; } fimg_job_t;
static void* fimg_worker(void* a) {
    fimg_job_t* j = (fimg_job_t*)a;
    for (u64 g = j->lo; g < j->hi; g++) {
        u64 w[4], t[4];
#if POSM_PACKED128
        u64 start=j->q0+4*g, row=start & ~U64C(15), raw[16];
        u32 enc[32];
        for (u32 i=0;i<16;++i) raw[i]=filler_word(j->seed,row+i);
        packed_encode_row((const u8*)raw,121,0,enc,row,j->klo,j->khi);
        memcpy(w, enc+2*(start-row), sizeof w);
#elif POSM_VDE_BITS == 89
        u64 start=j->q0+4*g, row=start & ~U64C(7), raw[8];
        u32 enc[16];
        for (u32 k=0;k<8;k++) raw[k]=filler_word(j->seed,row+k);
        f89_setup_row(raw,enc,j->klo,j->khi,row);
        memcpy(w, enc + 2*(start-row), sizeof w);
#else
        for (u32 i = 0; i < 4; i++) { u64 q = j->q0 + 4 * g + i; w[i] = vde_encode(filler_word(j->seed, q), tweak_at(j->klo, j->khi, q)); }
#endif
        image_group((j->q0 >> 2) + g, w, t);
        for (u32 i = 0; i < 4; i++) j->out[4 * g + i] = t[i];
    }
    return NULL;
}
int posm_filler_image_cpu_mt(u64* out, u64 seed, u64 key_lo, u64 key_hi, u64 q0, u64 count, int nthreads) {
    if ((q0 & 3) || (count & 3)) return -1;
    u64 groups = count / 4;
    if (nthreads < 1) nthreads = 1; if (nthreads > MAXT) nthreads = MAXT;
    if ((u64)nthreads > groups) nthreads = groups ? (int)groups : 1;
    pthread_t th[MAXT]; fimg_job_t jobs[MAXT];
    for (int i = 0; i < nthreads; i++) { jobs[i] = (fimg_job_t){ out, seed, key_lo, key_hi, q0, groups * i / nthreads, groups * (i + 1) / nthreads }; pthread_create(&th[i], NULL, fimg_worker, &jobs[i]); }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    return 0;
}
void posm_setup_cpu_mt(const u64* raw, u64* out, const posm_setup_params* sp, int nthreads) { range_run(raw, out, sp, nthreads, 0); }
void posm_decode_cpu_mt(const u64* mem, u64* out, const posm_setup_params* sp, int nthreads) { range_run(mem, out, sp, nthreads, 1); }
u64 posm_vde_encode(u64 d, u64 t) { return vde_encode(d, t); }
u64 posm_vde_decode(u64 w, u64 t) { return vde_decode(w, t); }
u64 posm_feistel(u64 q, u64 seed, u32 bits) { return feistel_perm(q, seed, bits); }
u64 posm_tweak(u64 key_lo, u64 key_hi, u64 q) { return tweak_at(key_lo, key_hi, q); }
void posm_pack_cpu(const u64* raw, u64* payload, u64 groups) { for (u64 g = 0; g < groups; g++) posm_pack_group_at(raw, payload, g); }
void posm_decode_unpack_cpu(const u64* theta, u64* raw, u64 q0, u64 groups, u64 key_lo, u64 key_hi) {
    for (u64 g = 0; g < groups; g++) posm_decode_unpack_group(theta, raw, q0, g, key_lo, key_hi);
}
/* Encoded dynamic memory (memory_accounting.prover.kvenc): the CUDA kv kernel's entries, on nthreads threads (group i of the
 * entries in order goes to thread i % nthreads). */
typedef struct { u64 raw_base, enc_base, q_base; const u64* ent; u64 nent, rw, key_lo, key_hi; int decode, t, nt; } kv_job_t;
static void* kv_worker(void* a) {
    kv_job_t* j = (kv_job_t*)a;
    u64 i = 0;
    for (u64 k = 0; k < j->nent; k++) {
        const u64* e = j->ent + 4 * k;
        for (u64 g = e[2]; g < e[2] + e[3]; g++, i++)
            if ((int)(i % (u64)j->nt) == j->t) posm_kv_entry_group(j->raw_base, j->enc_base, j->q_base, e, j->rw, g, j->key_lo, j->key_hi, j->decode);
    }
    return NULL;
}
void posm_kv_code_cpu(u64 raw_base, u64 enc_base, u64 q_base, const u64* ent, u64 nent, u64 rw, u64 key_lo, u64 key_hi, int decode, int nthreads) {
    if (nthreads < 1) nthreads = 1; if (nthreads > MAXT) nthreads = MAXT;
    kv_job_t jobs[MAXT]; pthread_t th[MAXT];
    for (int t = 0; t < nthreads; t++) jobs[t] = (kv_job_t){ raw_base, enc_base, q_base, ent, nent, rw, key_lo, key_hi, decode, t, nthreads };
    if (nthreads == 1) { kv_worker(&jobs[0]); return; }
    for (int t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, kv_worker, &jobs[t]);
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
}
void posm_siphash(u64 k0, u64 k1, u64 m0, u64 m1, u64 m2, u64 m3, u64* out) { struct u128 d = siphash24_128(k0, k1, m0, m1, m2, m3); out[0] = d.lo; out[1] = d.hi; }
void posm_image_group(u64 g, const u64* w, u64* t) { image_group(g, w, t); }

/* Host side of the pre-armed chain's flag protocol (`posm.provers.DeviceProver` arm/fire/disarm; the wait and
 * publish nodes in posm.cu).  The host writes the parameter block into host-mapped memory, then the flag; the
 * GPU's wait node spins on the flag (acquire) and then reads the parameters.  The two host stores must become
 * visible to the GPU in that order: on x86 (TSO) they do anyway, on a weakly ordered host (aarch64, e.g.
 * Grace) only with a barrier between them, at system scope since the reader is the GPU.  Likewise, after the
 * host sees the done flag the GPU raised, a barrier before it reads the digests the GPU wrote. */
static inline void posm_sys_fence(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#else
    atomic_thread_fence(memory_order_seq_cst);
#endif
}
void posm_flag_store_release(volatile u64* flag, u64 v) { posm_sys_fence(); *flag = v; }
void posm_fence_acquire(void) { posm_sys_fence(); }
/* Wait outside the Python interpreter so unrelated Python threads cannot interrupt
 * the response after the GPU has published its done flag. */
void posm_wait_done(volatile u64* done) {
    while (*done == 0) {
#if defined(__x86_64__)
        __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#endif
    }
    posm_sys_fence();
}
/* The parameter blocks' sizes in words, so Python can check its field lists against these structs at load. */
u64 posm_param_words(void) { return POSM_PARAM_WORDS; }
u64 posm_setup_param_words(void) { return sizeof(posm_setup_params) / sizeof(u64); }
