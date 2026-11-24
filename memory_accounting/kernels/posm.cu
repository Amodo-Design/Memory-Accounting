// CUDA entry points (H100/H200).  Built via torch.utils.cpp_extension.load_inline; a
// "#include <torch/extension.h>" line is prepended by backend.py.
//
// Per link (seed s from the chain state): matrix M_s + column constants + power tables for K_s, then
//   prepare  : fetch the challenged block once into a tiled int8 scratch (honest read or cheat regeneration)
//   partial  : int8 tensor-core matmul (WMMA m16n16k16), outputs mixed and polynomial-hashed per entry
//              group into partials[grid][G]
//   finish   : T[g] = sum over grid; either written to a table row (setup pass) or folded with the nonce
//              powers into the link digest, which picks the next block and the next seed.
// With graph=1 the whole chain is captured once per buffer set into a CUDA graph and replayed on a
// private highest-priority stream; a new challenge only rewrites the parameter block in place.
//
// Devices: every entry point runs on the device of its device-resident output (out / state / table),
// under a device guard, not on whatever device happens to be current (`--device cuda:1` must not launch
// on GPU 0).  `mem` / `canon` are exempt: they may be pinned host memory (zero-copy cheat) or a peer
// GPU's array (other-GPU cheat).  The side stream, its events, the shared-memory opt-in and the graph
// caches are per device.  Host-side state is not locked: pybind entry points hold the GIL.
#include "posm_core.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <tuple>
#include <string>
#include <stdexcept>
using namespace nvcuda;

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

// ---------------------------------------------------------------- scalar / setup kernels
__global__ void __launch_bounds__(1024) posm_chain_kernel(const u64* mem, const u64* canon, u64* out, const posm_params* prm, i8* mbuf, u64* pw) {
    __shared__ u64 red[1024]; __shared__ u64 st[4]; __shared__ u64 T[POSM_GMAX];
    posm_chain_body(mem, canon, out, prm, mbuf, pw, red, st, T, threadIdx.x, blockDim.x);
}
__global__ void posm_init_kernel(const posm_params* prm, u64* state, u64* out) {
    if (POSM_ABORTED(prm)) return;
    posm_init_body(prm, state);
    posm_cache_segment(prm, state);
    u64 s[4] = {state[0], state[1], state[2], state[3]};
    if (posm_stop_at(prm, s, 0, out)) const_cast<posm_params*>(prm)->mode = POSM_MODE_ABORT;   // the first block is off the GPU
}
__global__ void posm_setstate_kernel(const posm_params* prm, u64* state, u64 j, u64 seed) {
    state[0] = j; state[3] = seed; posm_cache_segment(prm, state);
}
__global__ void posm_matrix_kernel(const posm_params* prm, i8* mbuf, const u64* state) {
    if (POSM_ABORTED(prm)) return;
    posm_matrix_fill(mbuf, prm, state[3], (u64)blockIdx.x * blockDim.x + threadIdx.x, (u64)gridDim.x * blockDim.x);
}
__global__ void posm_powtable_kernel(const posm_params* prm, u64* pw, const u64* state, u64 count) {
    if (POSM_ABORTED(prm)) return;
    posm_powtable_fill(pw, prm, state[3], (u64)blockIdx.x * blockDim.x + threadIdx.x, count);
}
// colconst[c] = 128 * sum_j M[j][c]: block per 16 columns, 256 threads = 16 columns x 16 j-partitions
__global__ void posm_colconst_fast_kernel(const posm_params* prm, i8* mbuf) {
    if (POSM_ABORTED(prm)) return;
    __shared__ i32 part[16][16];
    const u32 K = (u32)prm->k, r = (u32)prm->r;
    const u32 c = blockIdx.x * 16 + (threadIdx.x & 15), jp = threadIdx.x >> 4;
    i32 sum = 0;
    if (c < r) for (u32 j = jp; j < K; j += 16) sum += (i32)mbuf[mtile_index(j, c, K)];
    part[jp][threadIdx.x & 15] = sum;
    __syncthreads();
    if (threadIdx.x < 16 && c < r) {
        i32 tot = 0;
        for (u32 p = 0; p < 16; p++) tot += part[p][threadIdx.x];
        reinterpret_cast<i32*>(mbuf + mbuf_colconst_offset(K, r))[blockIdx.x * 16 + threadIdx.x] = 128 * tot;
    }
}
__global__ void __launch_bounds__(1024) posm_link_partial_kernel(const u64* mem, const u64* canon, const posm_params* prm, const u64* state, u64* partials,
                                                                 const i8* mbuf, const u64* pw) {
    if (POSM_ABORTED(prm)) return;
    __shared__ u64 red[1024];
    posm_link_partial_body(mem, canon, prm, state, partials, mbuf, pw, red, blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}
__global__ void __launch_bounds__(1024) posm_link_finish_kernel(const posm_params* prm, u64* state, const u64* partials, u64* out, const u64* pw,
                                                                u64* table, u32 Gr, u32 link) {
    if (POSM_ABORTED(prm)) return;
    __shared__ u64 red[1024]; __shared__ u64 T[POSM_GMAX];
    posm_link_finish_body(prm, state, partials, out, pw, table, red, T, Gr, link, threadIdx.x, blockDim.x);
    if (!table && threadIdx.x == 0) posm_cache_segment(prm, state);
}
// Whole eight-lane groups participate before invalid tail lanes return.
__device__ __forceinline__ u64 grouped_tweak61(u64 kl,u64 kh,u64 q) {
#if POSM_GROUP64
    u32 lane=threadIdx.x & 7;
    // An unaligned slice can straddle group boundaries.
    if ((q & 7) != lane) return tweak_at(kl,kh,q);
    u64 base=0;
    if (!lane) base=tweak_prf(kl,kh,q);
    base=__shfl_sync(__activemask(),base,0,8);
    return addmod(base,lane);
#else
    return tweak_at(kl,kh,q);
#endif
}
#if POSM_VDE_BITS == 89
__device__ __forceinline__ field89 warp_tweak89(u64 kl,u64 kh,u64 q,u32 lane) {
    field89 base={0,0,0};
#if POSM_GROUP64
    if (!lane) base=f89_base(kl,kh,q);
    unsigned mask=__activemask();
    base.a=__shfl_sync(mask,base.a,0,8);
    base.b=__shfl_sync(mask,base.b,0,8);
    base.c=__shfl_sync(mask,base.c,0,8);
#endif
    return f89_slot_tweak(kl,kh,q,lane,base);
}
#endif
#if POSM_PACKED128
// Sixteen adjacent lanes own a row. Each stored 32-bit word has one writer.
__device__ __forceinline__ void packed_encode_lanes(const u8 *raw, u64 bytes,
    u64 offset, u64 *out, u64 q, u64 kl, u64 kh, u32 *records) {
    const u32 lane = threadIdx.x % 16;
    const unsigned mask = __activemask();
    if (lane < 11)
        f89_write_root(records+3*lane, f89_read_plain(raw, bytes, offset+11*lane),
                       packed_tweak(kl, kh, q, lane));
    __syncwarp(mask);
    u32 *words = (u32*)out;
    words[lane] = packed_word(records, lane);
    words[lane+16] = packed_word(records, lane+16);
    __syncwarp(mask);
}
#endif
__global__ void posm_filler_kernel(u64* out, u64 seed, u64 start, u64 count) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = filler_word(seed, start + i);
}
// Dynamic filler for many pages in one launch (freed KV blocks in every layer): page j is pw words at
// device address dst[j], holding Enc(key, filler(seed)) at positions pos0[j] .. pos0[j] + pw, masked.
__global__ void posm_fill_pages_kernel(const u64* dst, const u64* pos0, u64 npages, u64 pw, u64 seed, u64 key_lo, u64 key_hi, u64 mask) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npages * pw) return;
    u64 j = i / pw, w = i - j * pw, q = pos0[j] + w;
    u64* out = (u64*)(uintptr_t)dst[j];
#if POSM_PACKED128
    u64 row=q & ~U64C(15), raw[16];
    u32 enc[32];
    for (u32 k=0;k<16;++k) raw[k]=filler_word(seed,row+k);
    packed_encode_row((const u8*)raw,121,0,enc,row,key_lo,key_hi);
    u32 k=2*(q-row);
    out[w]=(enc[k] | ((u64)enc[k+1]<<32)) & mask;
#elif POSM_VDE_BITS == 89
    u64 group=q & ~U64C(7);
    u64 raw[8];
    u32 encoded[16];
    for (u32 k=0;k<8;k++) raw[k]=filler_word(seed,group+k);
    f89_setup_row(raw,encoded,key_lo,key_hi,group);
    u32 slot = 2*(q-group);
    out[w]=((u64)encoded[slot] | ((u64)encoded[slot+1] << 32)) & mask;
#else
    out[w]=vde_encode(filler_word(seed,q),grouped_tweak61(key_lo,key_hi,q)) & mask;
#endif
}
// Encoded dynamic memory (memory_accounting.prover.kvenc): encode or decode many units in one launch, one CUDA block per entry
// (4 words: raw word offset, encoded word offset, first group, groups), its threads over the entry's groups.
__global__ void posm_kv_kernel(u64 raw_base, u64 enc_base, u64 q_base, const u64* ent, u64 nent, u64 rw, u64 key_lo, u64 key_hi, int decode) {
    for (u64 j = blockIdx.x; j < nent; j += gridDim.x) {
        const u64* e = ent + 4 * j;
#if POSM_PACKED128
        __shared__ u32 staging[16][33];
        u8 *raw=rw ? (u8*)((u64*)(uintptr_t)raw_base+e[0]) : nullptr;
        u64 *enc=(u64*)(uintptr_t)enc_base+e[1];
        for (u64 t=threadIdx.x; t<16*e[3]; t+=blockDim.x) {
            u64 row=e[2]+t/16, q=KV_POS | (q_base+e[1]+row*16);
            u32 lane=t%16;
            if (decode) {
                if (lane<11) f89_plain_bytes(raw,rw*8,row*121+lane*11,
                    packed_decode((const u32*)(enc+row*16),lane,packed_tweak(key_lo,key_hi,q,lane)));
            } else packed_encode_lanes(raw,rw*8,row*121,enc+row*16,q,key_lo,key_hi,staging[threadIdx.x/16]);
        }
#elif POSM_VDE_BITS == 89
        u8 *raw=rw ? (u8*)((u64*)(uintptr_t)raw_base+e[0]) : nullptr;
        u64 *enc=(u64*)(uintptr_t)enc_base+e[1];
        for (u64 t=threadIdx.x; t<64*e[3]; t+=blockDim.x) {
            u64 row=e[2]*8+t/8, q=q_base+e[1]+row*8;
            u32 lane=t%8;
            field89 tweak=warp_tweak89(key_lo,key_hi,KV_POS | q,lane);
            u32 *slots=(u32*)(enc+row*8);
            if (lane<5) {
                u64 off=row*55+lane*11;
                if (decode) f89_plain_bytes(raw,8*rw,off,f89_read_root(slots+3*lane,tweak));
                else f89_write_root(slots+3*lane,f89_read_plain(raw,8*rw,off),tweak);
            } else if (!decode && lane==5) slots[15]=0;
        }
#else
        if (decode) {                                       // a thread per word into shared memory, then 15 per group unpack
            __shared__ u64 p[256];
            u64* raw = (u64*)(uintptr_t)raw_base + e[0];
            const u64* enc = (const u64*)(uintptr_t)enc_base + e[1];
            const u64 q0 = q_base + e[1], per = blockDim.x / 16;
            for (u64 gb = e[2]; gb < e[2] + e[3]; gb += per) {
                u64 g = gb + threadIdx.x / 16; u32 k = threadIdx.x % 16;
                bool on = g < e[2] + e[3];
                if (on) p[threadIdx.x] = vde_decode(enc[16 * g + k], grouped_tweak61(key_lo, key_hi, KV_POS | (q0 + 16 * g + k)));
                __syncthreads();
                if (on && k < 15) {
                    const u64* in = p + (threadIdx.x / 16) * 16;
                    u32 start = 64 * k, kk = start / 60, off = start % 60;
                    u64 v = in[kk] >> off;
                    if (kk + 1 < 16) v |= in[kk + 1] << (60 - off);
                    if (kk + 2 < 16 && 120 - off < 64) v |= in[kk + 2] << (120 - off);
                    u64 i = 15 * g + k;
                    if (i < rw) raw[i] = v;
                }
                __syncthreads();
            }
        } else {                                            // 59 squarings a word: a thread per word (posm_kv_encode_word)
            const u64* raw = rw ? (const u64*)(uintptr_t)raw_base + e[0] : 0;
            u64* enc = (u64*)(uintptr_t)enc_base + e[1];
            for (u64 t = threadIdx.x; t < 16 * e[3]; t += blockDim.x)
                posm_kv_encode_word(raw, rw, enc, q_base + e[1], e[2] + t / 16, (u32)(t % 16), grouped_tweak61(key_lo,key_hi,KV_POS | (q_base+e[1]+16*e[2]+t)));
        }
#endif
    }
}
__global__ void posm_pack_kernel(const u64* raw, u64* payload, u64 groups) {
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) posm_pack_group_at(raw, payload, g);
}
__global__ void posm_decode_unpack_kernel(const u64* theta, u64* raw, u64 q0, u64 groups, u64 key_lo, u64 key_hi) {
    u64 g=(u64)blockIdx.x*blockDim.x+threadIdx.x;
#if POSM_PACKED128
    if (g>=groups*128) return;
    u64 row=g/16, w=row*16;
    u32 lane=g%16;
    if (lane<11) f89_plain_bytes((u8*)raw,groups*968,row*121+11*lane,
        packed_decode((const u32*)(theta+q0+w),lane,packed_tweak(key_lo,key_hi,q0+w,lane)));
#elif POSM_VDE_BITS == 89
    if (g >= groups*64) return;
    u64 row=g/8, q=q0+row*8;
    u32 lane=g%8;
    field89 t=warp_tweak89(key_lo,key_hi,q,lane);
    if (lane<5) f89_plain_bytes((u8*)raw,groups*440,row*55+11*lane,
                                f89_read_root((const u32*)(theta+q)+3*lane,t));
#else
    if (g<groups) posm_decode_unpack_group(theta,raw,q0,g,key_lo,key_hi);
#endif
}
__global__ void posm_setup_at_kernel(const u64* raw, u64* out, u64 key_lo, u64 key_hi, u64 q0, u64 count) {
    u64 i=(u64)blockIdx.x*blockDim.x+threadIdx.x;
#if POSM_PACKED128
    __shared__ u32 staging[16][33];
    if (i>=count) return;
    u64 row=i & ~U64C(15);
    packed_encode_lanes((const u8*)(raw+row),121,0,out+row,q0+row,key_lo,key_hi,staging[threadIdx.x/16]);
#elif POSM_VDE_BITS == 89
    if (i>=count) return;
    u32 lane=i%8;
    u64 row=i-lane;
    field89 t=warp_tweak89(key_lo,key_hi,q0+row,lane);
    u32 *dst=(u32*)(out+row);
    if (lane<5) f89_write_root(dst+lane*3,f89_read_plain((const u8*)(raw+row),64,lane*12),t);
    else if(lane==5) dst[15]=0;
#else
    if (i<count) out[i]=vde_encode(raw[i],grouped_tweak61(key_lo,key_hi,q0+i));
#endif
}
__global__ void posm_setup_kernel(const u64* raw, u64* out, const posm_setup_params* sp) {
    posm_setup_one(raw, out, sp, (u64)blockIdx.x * blockDim.x + threadIdx.x);
}
__global__ void posm_decode_kernel(const u64* mem, u64* out, const posm_setup_params* sp) {
    posm_decode_one(mem, out, sp, (u64)blockIdx.x * blockDim.x + threadIdx.x);
}
#if POSM_PACKED128
/* Decode a compact selected interval while retaining global row positions. */
__global__ void posm_decode_at_packed128_kernel(const u64* mem, u64* out,
                                                u64 key_lo, u64 key_hi, u64 q0, u64 count) {
    u64 q = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    u64 row = q & ~U64C(15);
    u32 slot = (u32)(q - row);
    u8 *dst = (u8*)(out + row);
    if (slot < 11)
        f89_plain_bytes(dst, 128, 11*slot,
                        packed_decode((const u32*)(mem + row), slot,
                                      packed_tweak(key_lo, key_hi, q0 + row, slot)));
    else if (slot == 11)
        for (u32 i = 121; i < 128; ++i) dst[i] = 0;
}
#endif

// ---------------------------------------------------------------- tensor-core path
// prepare: fetch every stored word of the challenged block exactly once and write it as p ^ 0x80
// (== p - 128 as int8) into a tiled A scratch: row tile rt at rt*16*K, tile kt at +kt*256, row-major 16x16.
#define TC_THREADS 128
#define TC_WARPS   4
#define PREP_THREADS 256

__device__ __forceinline__ void ablk_store(i8* ablk, u32 K, u64 i, u64 word) {
    const u32 wpr = K / 8, row = (u32)(i / wpr), j0 = 8 * (u32)(i % wpr);
    const size_t off = (size_t)(row >> 4) * 16 * K + (size_t)(j0 >> 4) * 256 + (row & 15) * 16 + (j0 & 15);
    *reinterpret_cast<u64*>(ablk + off) = word ^ U64C(0x8080808080808080);
}
// thread per 32-byte group: fetch the 4 raw words once, image them if prm->img, store the 4 words
__global__ void posm_prepare_kernel(const u64* mem, const u64* canon, const posm_params* prm, const u64* state, i8* ablk) {
    if (POSM_ABORTED(prm)) return;
    const u64 be = prm->be; const u32 K = (u32)prm->k;
    const u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= be / 4) return;
    const u64 absolute_g = (state[0] * be) / 4 + g;
    const u64 first = 4 * absolute_g;
    u64 t[4];
#if POSM_VDE_BITS == 89 && POSM_PACKED128
    /* Four adjacent threads own one 128-byte row. Each computes only its
     * three (or two) roots; all four then pack different output words. */
    __shared__ u32 records[PREP_THREADS / 4][33];
    const u64 mode = prm->mode;
    /* The final, incomplete 128 B row uses the scalar fallback: a masked
     * warp barrier below requires all four lanes of a row to participate. */
    const u64 row_group = g & ~U64C(3);
    if (row_group + 4 <= be / 4 && state[7] == 1 &&
        ((mode == 1 || mode == 3) ? prm->enc : (mode == 4 || mode == 5))) {
        const u64 row = first & ~U64C(15);
        const u64 domain_row = (mode == 4 || mode == 5) ? (KV_POS | row) : row;
        const u32 lane = (u32)g & 3;
        u32* rec = records[threadIdx.x >> 2];
        if (prm->canonp) canon = (const u64*)(uintptr_t)prm->canonp;
        const u32* source = (const u32*)(canon + row);
        const u8* raw = (const u8*)(canon + row);
        const u64 target_lo = mode == 4 ? prm->seed_c : prm->seed_t;
        const u64 target_hi = mode == 4 ? prm->seed_c_hi : prm->seed_t_hi;
        for (u32 slot = lane; slot < 11; slot += 4) {
            field89 plain = (mode == 1 || mode == 5)
                ? packed_decode(source, slot, packed_tweak(prm->seed_c, prm->seed_c_hi, domain_row, slot))
                : f89_read_plain(raw, 121, 11 * slot);
            f89_write_root(rec + 3 * slot, plain, packed_tweak(target_lo, target_hi, domain_row, slot));
        }
        /* Every named lane writes this row's records before any lane packs it.
         * __activemask() is unsafe here: lane 3 has two roots, lanes 0–2 three. */
        const unsigned row_mask = 0xFu << (threadIdx.x & 28);
        __syncwarp(row_mask);
        u64 w[4];
        for (u32 i = 0; i < 4; i++) {
            const u32 word = 8 * lane + 2 * i;
            w[i] = (u64)packed_word(rec, word) | ((u64)packed_word(rec, word + 1) << 32);
        }
        if (prm->img) image_group(prm->qoff / 4 + absolute_g, w, t);
        else for (u32 i = 0; i < 4; i++) t[i] = w[i];
    } else
#endif
    if (state[7] == 0 && prm->nseg && first >= state[4] && first + 4 <= state[5]) {
        const u64* base = (const u64*)(uintptr_t)state[6];
        u64 w[4];
        for (u32 i = 0; i < 4; i++) w[i] = base[first - state[4] + i];
        if (prm->img) image_group(prm->qoff / 4 + absolute_g, w, t);
        else for (u32 i = 0; i < 4; i++) t[i] = w[i];
    } else {
        posm_fetch_group(mem, canon, prm, absolute_g, t);
    }
    for (u32 i = 0; i < 4; i++) ablk_store(ablk, K, 4 * g + i, t[i]);
}
__global__ void posm_image_kernel(const u64* theta, u64* out, u64 groups, u64 g0) {
    posm_image_one(theta, out, (u64)blockIdx.x * blockDim.x + threadIdx.x, groups, g0);
}

__device__ __forceinline__ void atomic_addmod(u64* p, u64 v) {
    unsigned long long old = *p, assumed;
    do { assumed = old; old = atomicCAS((unsigned long long*)p, assumed, addmod(assumed, v)); } while (old != assumed);
}

// 2D decomposition per block: RT row tiles x CT column tiles; the 4 warps are CT columns x KS = 4/CT
// k-splits.  A is staged once per block (16*K*RT bytes) and reused by CT columns; each matrix tile read
// from L2 is reused by RT row tiles.  Outputs are mixed and accumulated per entry group into shared
// red[G] (mod p), written to partials[blockIdx][G] at the end.
// Shared memory = RT*16*K (A) + 4*RT*1 KB (partial C tiles per warp) + 8*GMAX (red).
static size_t tc_smem_bytes(u32 K, int RT) { return (size_t)RT * 16 * K + (size_t)TC_WARPS * RT * 256 * 4 + 8 * POSM_GMAX; }
static int tc_rt_max(u32 K) { return K <= 1024 ? 4 : 2; }
static void tc_pick(u32 K, u32 rowtiles, u32 coltiles, int sms, int* RT, int* CT) {
    int rt = tc_rt_max(K), ct = 4;
    auto units = [&](int a, int c) { return (long)((rowtiles + a - 1) / a) * ((coltiles + c - 1) / c); };
    while (units(rt, ct) < 2L * sms && (rt > 1 || ct > 1)) { if (rt > 1) rt >>= 1; else ct >>= 1; }
    *RT = rt; *CT = ct;
}

__global__ void __launch_bounds__(TC_THREADS)
posm_link_partial_tc_kernel(const posm_params* prm, const u64* state, const i8* ablk, const i8* mbuf, const u64* pw, u64* partials, u32 RT, u32 CT) {
    if (POSM_ABORTED(prm)) return;
    extern __shared__ __align__(128) unsigned char smem[];
    const u64 be = prm->be;
    const u32 K = (u32)prm->k, r = (u32)prm->r, G = (u32)prm->g, span = posm_group_span(prm);
    const u32 wpr = K / 8, rows = (u32)(be / wpr), rowtiles = rows >> 4, coltiles = r >> 4, ktiles = K >> 4;
    const u32 KS = TC_WARPS / CT;
    const u32 cgroups = (coltiles + CT - 1) / CT, rgroups = (rowtiles + RT - 1) / RT, units = rgroups * cgroups;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 wc = warp % CT, ks = warp / CT;
    const u32 kt0 = ks * ktiles / KS, kt1 = (ks + 1) * ktiles / KS;
    i8*  A   = reinterpret_cast<i8*>(smem);
    i32* Cs  = reinterpret_cast<i32*>(A + (size_t)RT * 16 * K);
    u64* red = reinterpret_cast<u64*>(Cs + (size_t)TC_WARPS * RT * 256);
    const i32* colconst = reinterpret_cast<const i32*>(mbuf + mbuf_colconst_offset(K, r));
    const u64* T2 = pw + POSM_T1LEN(r);
    const u64 mixkey = chain_mixkey(prm->k0, prm->k1, state[3], prm->polyk);
    const u32 abytes = 16 * K;

    for (u32 g = threadIdx.x; g < G; g += TC_THREADS) red[g] = 0;
    __syncthreads();
    for (u32 u = blockIdx.x; u < units; u += gridDim.x) {
        const u32 rg = u / cgroups, cg = u % cgroups;
        const u32 rt0 = rg * RT, nrt = min(RT, rowtiles - rt0);
        const u32 ct = cg * CT + wc;
        __syncthreads();
        {   // stage A: 8 x 16-byte loads in flight per thread
            const u32 total = nrt * abytes;
            for (u32 base = 0; base < total; base += TC_THREADS * 16 * 8) {
                int4 v[8];
                #pragma unroll
                for (u32 i = 0; i < 8; i++) { const u32 o = base + (i * TC_THREADS + threadIdx.x) * 16; if (o < total) v[i] = *reinterpret_cast<const int4*>(ablk + (size_t)rt0 * abytes + o); }
                #pragma unroll
                for (u32 i = 0; i < 8; i++) { const u32 o = base + (i * TC_THREADS + threadIdx.x) * 16; if (o < total) *reinterpret_cast<int4*>(A + o) = v[i]; }
            }
        }
        __syncthreads();
        const bool active = ct < coltiles;
        if (active) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[4];
            #pragma unroll
            for (u32 t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0);
            const i8* Bcol = mbuf + (size_t)ct * ktiles * 256;
            for (u32 kt = kt0; kt < kt1; kt += 4) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b[4];
                #pragma unroll
                for (u32 i = 0; i < 4; i++) if (kt + i < kt1) wmma::load_matrix_sync(b[i], Bcol + (kt + i) * 256, 16);
                #pragma unroll
                for (u32 i = 0; i < 4; i++) {
                    if (kt + i < kt1) {
                        #pragma unroll
                        for (u32 t = 0; t < 4; t++) {
                            if (t < nrt) {
                                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
                                wmma::load_matrix_sync(a, A + (size_t)t * abytes + (kt + i) * 256, 16);
                                wmma::mma_sync(acc[t], a, b[i], acc[t]);
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (u32 t = 0; t < 4; t++) if (t < nrt) wmma::store_matrix_sync(Cs + ((size_t)warp * RT + t) * 256, acc[t], 16, wmma::mem_row_major);
        }
        __syncthreads();
        // reduce the KS k-splits of each (column, row tile), mix, hash into the row's entry group.
        // Each thread accumulates in a register and flushes to shared memory only when its group changes
        // (with g = 1 that is once per unit), so the shared atomics are rare.
        {
            u64 acc = 0; u32 cur = 0xFFFFFFFFu;
            for (u32 o = threadIdx.x; o < CT * nrt * 256; o += TC_THREADS) {
                const u32 col_slot = o / (nrt * 256), rem = o % (nrt * 256), t = rem >> 8, e = rem & 255;
                const u32 ctile = cg * CT + col_slot;
                if (ctile >= coltiles) continue;
                i32 v = 0;
                for (u32 k = 0; k < KS; k++) v += Cs[(((size_t)(k * CT + col_slot)) * RT + t) * 256 + e];
                const u32 row = e >> 4, cit = e & 15, c = ctile * 16 + cit;
                const u32 grow = (rt0 + t) * 16 + row, g = grow / span, rl = grow % span;
                v += colconst[c];
                if (g != cur) { if (cur != 0xFFFFFFFFu) atomic_addmod(&red[cur], acc); acc = 0; cur = g; }
                acc = addmod(acc, mulmod(mix_output(v, mixkey), mulmod(T2[rl], pw[c])));
            }
            // flush: if the whole warp is on one group (always true with g = 1) reduce with shuffles and
            // issue one atomic per warp; otherwise one per lane
            const unsigned active = __activemask();
            const bool same = __all_sync(active, cur == __shfl_sync(active, cur, 0));
            if (same) {
                for (int o = 16; o > 0; o >>= 1) acc = addmod(acc, __shfl_down_sync(active, acc, o));
                if (lane == 0 && cur != 0xFFFFFFFFu) atomic_addmod(&red[cur], acc);
            } else if (cur != 0xFFFFFFFFu) {
                atomic_addmod(&red[cur], acc);
            }
        }
    }
    __syncthreads();
    for (u32 g = threadIdx.x; g < G; g += TC_THREADS) partials[(u64)blockIdx.x * G + g] = red[g];
}

// ---------------------------------------------------------------- host side
// The device a call runs on: that of a tensor the call writes, which must be a CUDA tensor.
static c10::Device run_device(const torch::Tensor& t, const char* what) {
    if (!t.is_cuda()) throw std::runtime_error(std::string(what) + " must be a CUDA tensor (it selects the device the kernels run on)");
    return t.device();
}
static void same_device(const torch::Tensor& t, const c10::Device& d, const char* what) {
    if (t.numel() && t.device() != d) throw std::runtime_error(std::string(what) + " is on " + t.device().str() + ", the call runs on " + d.str());
}
static int cur_device() { int d; CUDA_OK(cudaGetDevice(&d)); return d; }
static const u64* p64(const torch::Tensor& t) { return reinterpret_cast<const u64*>(t.data_ptr<int64_t>()); }
static u64* p64m(torch::Tensor& t) { return reinterpret_cast<u64*>(t.data_ptr<int64_t>()); }
/* A parameter block, built by Python as a u64 array (posm.params), viewed as its struct. */
static const posm_params* params_ptr(const torch::Tensor& t) { return reinterpret_cast<const posm_params*>(t.data_ptr<int64_t>()); }
static const posm_setup_params* setup_ptr(const torch::Tensor& t) { return reinterpret_cast<const posm_setup_params*>(t.data_ptr<int64_t>()); }
static const u64* mem_ptr(const torch::Tensor& mem) {
    const u64* p = p64(mem);
    if (!mem.is_cuda()) { void* dp; CUDA_OK(cudaHostGetDevicePointer(&dp, (void*)p, 0)); p = (const u64*)dp; }  // pinned host (zero-copy) cheat
    return p;
}

struct Bufs { const u64* mem; const u64* canon; u64* out; const posm_params* prm; u64* state; u64* partials; u64* pw; u64 pwcount; i8* mbuf; i8* ablk; u64 be; u32 K, r, G; bool per_link_prep; };

// seed-dependent tables for the current state[3]: matrix, column constants, power tables
static void enqueue_link_prep(cudaStream_t s, const Bufs& b) {
    if (b.r) {
        const u64 entries = ((u64)b.K * b.r) / 8;
        posm_matrix_kernel<<<(unsigned)((entries + 255) / 256), 256, 0, s>>>(b.prm, b.mbuf, b.state);
        posm_colconst_fast_kernel<<<(b.r + 15) / 16, 256, 0, s>>>(b.prm, b.mbuf);
    }
    posm_powtable_kernel<<<(unsigned)((b.pwcount + 255) / 256), 256, 0, s>>>(b.prm, b.pw, b.state, b.pwcount);
}
static void enqueue_link_work(cudaStream_t s, const Bufs& b, int groups, int threads, bool tensor, int RT, int CT) {
    if (tensor) {
        posm_prepare_kernel<<<(unsigned)((b.be / 4 + PREP_THREADS - 1) / PREP_THREADS), PREP_THREADS, 0, s>>>(
            b.mem, b.canon, b.prm, b.state, b.ablk);
        posm_link_partial_tc_kernel<<<groups, TC_THREADS, tc_smem_bytes(b.K, RT), s>>>(b.prm, b.state, b.ablk, b.mbuf, b.pw, b.partials, (u32)RT, (u32)CT);
    } else {
        posm_link_partial_kernel<<<groups, threads, 0, s>>>(b.mem, b.canon, b.prm, b.state, b.partials, b.mbuf, b.pw);
    }
}
static void tc_shape(const Bufs& b, bool tensor, int& groups, int& RT, int& CT) {
    RT = 1; CT = 1;
    if (tensor && b.r) {
        int dev; CUDA_OK(cudaGetDevice(&dev)); int sms; CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        const u32 rowtiles = (u32)(b.be * 8 / b.K) >> 4, coltiles = b.r >> 4;
        tc_pick(b.K, rowtiles, coltiles, sms, &RT, &CT);
        const long units = (long)((rowtiles + RT - 1) / RT) * ((coltiles + CT - 1) / CT);
        if (units < groups) groups = (int)units;
    }
}
static void enqueue_chain(cudaStream_t s, const Bufs& b, int groups, int threads, int n, bool tensor) {
    int RT, CT; tc_shape(b, tensor, groups, RT, CT);
    posm_init_kernel<<<1, 1, 0, s>>>(b.prm, b.state, b.out);
    if (!b.per_link_prep) enqueue_link_prep(s, b);                 // constant seed (x == 0): tables once per chain
    for (int link = 0; link < n; link++) {
        if (b.per_link_prep) enqueue_link_prep(s, b);
        enqueue_link_work(s, b, groups, threads, tensor, RT, CT);
        posm_link_finish_kernel<<<1, threads, 0, s>>>(b.prm, b.state, b.partials, b.out, b.pw, nullptr, (u32)groups, (u32)link);
    }
}

// Torch's current stream is normally the legacy NULL stream, which cannot be captured and has default
// priority.  Everything runs on a private non-blocking, highest-priority stream, ordered against torch's
// stream with events, so the challenge outranks background work.  Streams, events and the kernel's
// shared-memory opt-in belong to one device, so they are kept per device (the current one, set by the
// caller's guard).
struct DevState { cudaStream_t side = nullptr; cudaEvent_t ev_in = nullptr, ev_out = nullptr; size_t smem_set = 0; };
static std::map<int, DevState> g_dev;                 // node-based: references stay valid as devices are added
static DevState& dev_state() { return g_dev[cur_device()]; }
static void ensure_smem(u32 K) {
    DevState& d = dev_state();
    const size_t need = tc_smem_bytes(K, tc_rt_max(K));
    if (need > d.smem_set) { CUDA_OK(cudaFuncSetAttribute(posm_link_partial_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need)); d.smem_set = need; }
}
static DevState& ensure_side_stream() {
    DevState& d = dev_state();
    if (d.side) return d;
    int least, greatest;
    CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_OK(cudaStreamCreateWithPriority(&d.side, cudaStreamNonBlocking, greatest));
    CUDA_OK(cudaEventCreateWithFlags(&d.ev_in, cudaEventDisableTiming));
    CUDA_OK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
    return d;
}
int64_t side_stream(int64_t dev) {                    // device `dev`'s side stream, for timing events from Python
    const c10::cuda::CUDAGuard guard((c10::DeviceIndex)dev);
    return (int64_t)(uintptr_t)ensure_side_stream().side;
}

static Bufs make_bufs(torch::Tensor& mem, torch::Tensor& canon, torch::Tensor& out, torch::Tensor& prm, torch::Tensor& state,
                      torch::Tensor& partials, torch::Tensor& pw, torch::Tensor& mbuf, torch::Tensor& ablk, int64_t k, int64_t r, int64_t be, int64_t g, bool tensor, bool per_link_prep) {
    const c10::Device dev = run_device(state, "state");
    same_device(out, dev, "out"); same_device(prm, dev, "prm"); same_device(partials, dev, "partials");
    same_device(pw, dev, "pw"); same_device(mbuf, dev, "mbuf"); same_device(ablk, dev, "ablk");
    Bufs b; b.per_link_prep = per_link_prep;
    b.mem = mem_ptr(mem); b.canon = p64(canon); b.out = p64m(out); b.prm = params_ptr(prm); b.state = p64m(state); b.partials = p64m(partials);
    b.pw = p64m(pw); b.pwcount = (u64)pw.numel(); b.mbuf = reinterpret_cast<i8*>(mbuf.data_ptr<int8_t>());
    b.ablk = ablk.numel() ? reinterpret_cast<i8*>(ablk.data_ptr<int8_t>()) : nullptr;
    b.K = (u32)k; b.r = (u32)r; b.be = (u64)be; b.G = (u32)g;
    if (state.numel() < 8) throw std::runtime_error("tensor chain state needs 8 words");
    if (b.G < 1 || b.G > POSM_GMAX) throw std::runtime_error("g out of range");
    // the tensor-core kernel is the matmul: at r = 0 it has no columns and would hash nothing (zero entries)
    if (tensor && !b.r) throw std::runtime_error("the tensor-core path needs r > 0 (r = 0 runs on the scalar kernel)");
    if (tensor && b.r) {
        if (b.ablk == nullptr || (u64)ablk.numel() < b.be * 8) throw std::runtime_error("tensor path needs a block scratch of block_bytes");
        if ((b.be * 8 / b.K) % 16 != 0) throw std::runtime_error("tensor path needs block_bytes to be a multiple of 16*k (whole 16-row tiles)");
        if ((u64)mbuf.numel() < mbuf_colconst_offset(b.K, b.r) + 4 * (u64)b.r) throw std::runtime_error("matrix buffer too small for k x r");
        ensure_smem(b.K);
    }
    return b;
}

// persistent single-block chain (comparison only)
void chain(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state,
           torch::Tensor pw, torch::Tensor mbuf, int64_t k, int64_t r, int64_t threads) {
    const c10::Device dev = run_device(out, "out");
    same_device(prm, dev, "prm"); same_device(pw, dev, "pw"); same_device(mbuf, dev, "mbuf");
    const c10::cuda::CUDAGuard guard(dev);
    auto s = at::cuda::getCurrentCUDAStream();
    posm_chain_kernel<<<1, threads, 0, s>>>(mem_ptr(mem), p64(canon), p64m(out), params_ptr(prm), reinterpret_cast<i8*>(mbuf.data_ptr<int8_t>()), p64m(pw));
    CUDA_OK(cudaGetLastError());
}

// Pre-armed chain. The graph begins with a node that spins on a host-mapped flag and then copies the
// parameter block from host-mapped memory into the device parameter buffer; it ends with a node that
// publishes the digests to host-mapped memory and raises a done flag. Launched before the nonce exists
// (at ARM time), so the timed window contains no CUDA API call at all: the host writes 34 words and one
// flag, then spins on the done flag.  Protocol (one armed graph per prover at a time):
//   host: done = 0, flag = 0, launch;  fire: prm_src = params, flag = 1;  abort: flag = 2
//   wait node: while (flag == 0) sleep; flag 1: copy prm_src -> prm; flag 2: prm->mode = abort, so
//   every chain kernel returns at once;   publish node: out -> out_dst; done = 1
// Ordering: the host stores prm_src, then the flag, behind a system-scope barrier (posm_flag_store_release in
// posm_cpu.c); the flag is read here with acquire semantics at system scope, so the parameter reads after it
// (the other threads' too, through the barrier) see the host's parameters, also on a weakly ordered host. The
// host reads out_dst after seeing done behind a barrier of its own (posm_fence_acquire).
__device__ __forceinline__ u64 ld_acquire_sys(volatile u64* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    u64 v;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#else
    u64 v = *p; __threadfence_system(); return v;
#endif
}
__global__ void posm_wait_kernel(volatile u64* flag, volatile const u64* prm_src, u64* prm_dst) {
    __shared__ u64 f;
    if (threadIdx.x == 0) { u64 v; while ((v = ld_acquire_sys(flag)) == 0) __nanosleep(200); f = v; }
    __syncthreads();
    if (f == 1) {
        for (u32 i = threadIdx.x; i < (u32)POSM_PARAM_WORDS; i += blockDim.x) prm_dst[i] = prm_src[i];
    } else if (threadIdx.x == 0) {
        reinterpret_cast<posm_params*>(prm_dst)->mode = POSM_MODE_ABORT;
    }
    __threadfence();
}
__global__ void posm_publish_kernel(const u64* out, volatile u64* dst, volatile u64* done, u32 count) {
    for (u32 i = threadIdx.x; i < count; i += blockDim.x) dst[i] = out[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { *done = 1; __threadfence_system(); }
}
int64_t host_device_ptr(torch::Tensor t) {          // device address of a pinned host tensor (zero-copy reads)
    void* dp; CUDA_OK(cudaHostGetDevicePointer(&dp, t.data_ptr(), 0)); return (int64_t)(uintptr_t)dp;
}

// ---------------------------------------------------------------- reserved address ranges (stream-decode's Θ')
// Θ' is one contiguous array, but allocating it while the plain weights are still resident needs room for both
// (about twice the model). Instead a virtual address range is reserved for all of it and backed with physical
// memory chunk by chunk as the encoding reaches it, while each layer's plain weights are freed behind it
// (memory_accounting.prover.stream_decode). The driver calls come through cudaGetDriverEntryPointByVersion: nothing links libcuda.
#define CU_OK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) throw std::runtime_error(std::string(#x) + ": CUresult " + std::to_string((int)r_)); } while (0)
struct Vmm {
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*free_range)(CUdeviceptr, size_t);
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    CUresult (*release)(CUmemGenericAllocationHandle);
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*unmap)(CUdeviceptr, size_t);
    CUresult (*set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult (*granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
};
static const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm v{};
        auto get = [](const char* sym, void** fn) {
            cudaDriverEntryPointQueryResult q;
            CUDA_OK(cudaGetDriverEntryPointByVersion(sym, fn, 12000, cudaEnableDefault, &q));
            if (q != cudaDriverEntryPointSuccess || !*fn) throw std::runtime_error(std::string("driver entry point ") + sym + " not found");
        };
        get("cuMemAddressReserve", (void**)&v.reserve); get("cuMemAddressFree", (void**)&v.free_range);
        get("cuMemCreate", (void**)&v.create); get("cuMemRelease", (void**)&v.release);
        get("cuMemMap", (void**)&v.map); get("cuMemUnmap", (void**)&v.unmap);
        get("cuMemSetAccess", (void**)&v.set_access); get("cuMemGetAllocationGranularity", (void**)&v.granularity);
        return v;
    }();
    return v;
}
static CUmemAllocationProp vmm_prop(int dev) {
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev;
    return p;
}
struct Reserved { size_t bytes, chunk, mapped; int dev; };
static std::map<CUdeviceptr, Reserved>& reserved() { static std::map<CUdeviceptr, Reserved> r; return r; }   // under the GIL

int64_t vmm_granularity(int64_t dev) {
    auto p = vmm_prop((int)dev); size_t g; CU_OK(vmm().granularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); return (int64_t)g;
}
// Back the reserved range at va with device memory up to at least `upto` bytes from its start (whole chunks).
static void vmm_map_range(CUdeviceptr va, size_t upto) {
    Reserved& r = reserved().at(va);
    const size_t want = std::min(r.bytes, (upto + r.chunk - 1) / r.chunk * r.chunk);
    auto prop = vmm_prop(r.dev);
    CUmemAccessDesc acc{}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    while (r.mapped < want) {
        CUmemGenericAllocationHandle h;
        CU_OK(vmm().create(&h, r.chunk, &prop, 0));
        CUresult m = vmm().map(va + r.mapped, r.chunk, 0, h, 0);
        vmm().release(h);                                     // the mapping keeps the memory until it is unmapped
        CU_OK(m);
        CU_OK(vmm().set_access(va + r.mapped, r.chunk, &acc, 1));
        r.mapped += r.chunk;
    }
}
static void vmm_release(void* p) {
    auto it = reserved().find((CUdeviceptr)(uintptr_t)p);
    if (it == reserved().end()) return;
    const Reserved& r = it->second;
    for (size_t off = 0; off < r.mapped; off += r.chunk) vmm().unmap(it->first + off, r.chunk);
    vmm().free_range(it->first, r.bytes);
    reserved().erase(it);
}
// A tensor of `words` int64 over a reserved range of `bytes` (a multiple of `chunk`, itself of the granularity) with only
// its first chunk mapped (torch checks that the pointer is device memory): vmm_map backs the rest. Freeing the tensor
// unmaps every chunk mapped and frees the range.
torch::Tensor vmm_reserve(int64_t words, int64_t bytes, int64_t chunk, int64_t dev) {
    if (words * 8 > bytes || bytes % chunk) throw std::runtime_error("vmm_reserve: the range must hold the words and be whole chunks");
    const c10::cuda::CUDAGuard guard((c10::DeviceIndex)dev);
    CUdeviceptr va; CU_OK(vmm().reserve(&va, (size_t)bytes, (size_t)chunk, 0, 0));
    reserved()[va] = Reserved{(size_t)bytes, (size_t)chunk, 0, (int)dev};
    try { vmm_map_range(va, 1); } catch (...) { vmm_release((void*)(uintptr_t)va); throw; }
    return torch::from_blob((void*)(uintptr_t)va, {words}, vmm_release, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, (c10::DeviceIndex)dev));
}
// Back the reserved tensor t with device memory up to at least `upto` bytes from its start.
void vmm_map(torch::Tensor t, int64_t upto) {
    const CUdeviceptr va = (CUdeviceptr)(uintptr_t)t.data_ptr();
    if (!reserved().count(va)) throw std::runtime_error("vmm_map: not a tensor from vmm_reserve (or a view that does not start it)");
    vmm_map_range(va, (size_t)upto);
}
static u64* mapped_ptr(int64_t host_ptr) {
    void* dp; CUDA_OK(cudaHostGetDevicePointer(&dp, (void*)(uintptr_t)host_ptr, 0)); return (u64*)dp;
}

// Capture what `enqueue` puts on stream s into an instantiated graph.  Exception-safe: if anything fails
// between begin and end (a launch error, a CUDA_OK inside `enqueue`, end capture, instantiate), the capture is
// ended so the stream leaves capture mode, the partial graph is destroyed, the error state is cleared and the
// error is rethrown.  Before, a failed capture left the side stream capturing, and every later challenge on
// that device failed until the process restarted.
static void end_capture_quietly(cudaStream_t s) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) {
        cudaGraph_t partial = nullptr;
        cudaStreamEndCapture(s, &partial);            // also ends an invalidated capture (partial is then null)
        if (partial) cudaGraphDestroy(partial);
    }
    cudaGetLastError();                               // the failure must not surface again in the next call's check
}
template <class F>
static cudaGraphExec_t capture_graph(cudaStream_t s, F&& enqueue) {
    CUDA_OK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    cudaGraph_t g = nullptr;
    try {
        enqueue();
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaStreamEndCapture(s, &g));
    } catch (...) {
        if (g) cudaGraphDestroy(g);
        end_capture_quietly(s);
        throw;
    }
    cudaGraphExec_t exec = nullptr;
    const cudaError_t e = cudaGraphInstantiate(&exec, g, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error(std::string("cudaGraphInstantiate: ") + cudaGetErrorString(e));
    }
    return exec;
}

using GraphKey = std::tuple<int, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*,
                            int, int, int, bool, u32, u32, u64, u64, int64_t, int64_t, u32>;   // device first: graphs belong to one device
static std::map<GraphKey, cudaGraphExec_t> g_graphs;

// All shape arguments come from the host (they are known in Python); nothing here reads device memory.
void chain_multi(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state,
                 torch::Tensor partials, torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk,
                 int64_t groups, int64_t threads, int64_t n, int64_t k_, int64_t r_, int64_t be_, int64_t g_, bool per_link_prep, bool tensor, bool graph) {
    const c10::cuda::CUDAGuard guard(run_device(state, "state"));
    Bufs b = make_bufs(mem, canon, out, prm, state, partials, pw, mbuf, ablk, k_, r_, be_, g_, tensor, per_link_prep);
    if (n > POSM_MAX_LINKS || (int64_t)out.numel() < 2 + n) throw std::runtime_error("chain too long for the output buffer");
    if ((u64)partials.numel() < (u64)groups * b.G) throw std::runtime_error("partials buffer needs groups * g words");
    cudaStream_t cur = at::cuda::getCurrentCUDAStream();
    DevState& d = ensure_side_stream();
    const int dev = cur_device();
    CUDA_OK(cudaEventRecord(d.ev_in, cur));
    CUDA_OK(cudaStreamWaitEvent(d.side, d.ev_in, 0));
    if (!graph) {
        enqueue_chain(d.side, b, (int)groups, (int)threads, (int)n, tensor);
        CUDA_OK(cudaGetLastError());
    } else {
        GraphKey key{dev, b.mem, b.canon, b.out, b.prm, b.state, b.partials, b.pw, b.mbuf, b.ablk, (int)groups, (int)threads, (int)n, tensor, b.K, b.r,
                     b.be, b.pwcount, mbuf.numel(), ablk.numel(), b.G + (per_link_prep ? 0x10000u : 0u)};
        auto it = g_graphs.find(key);
        if (it == g_graphs.end()) {
            cudaGraphExec_t exec = capture_graph(d.side, [&] { enqueue_chain(d.side, b, (int)groups, (int)threads, (int)n, tensor); });
            it = g_graphs.emplace(key, exec).first;
        }
        CUDA_OK(cudaGraphLaunch(it->second, d.side));
    }
    CUDA_OK(cudaEventRecord(d.ev_out, d.side));
    CUDA_OK(cudaStreamWaitEvent(cur, d.ev_out, 0));
}

using ArmedKey = std::tuple<GraphKey, int64_t, int64_t, int64_t, int64_t>;
static std::map<ArmedKey, cudaGraphExec_t> g_armed_graphs;

// Launch the pre-armed chain (see posm_wait_kernel). No ordering against torch's stream: the host side
// is entirely through host-mapped memory, and torch's stream must not block on an armed graph.
void chain_multi_armed(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state,
                       torch::Tensor partials, torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk,
                       int64_t groups, int64_t threads, int64_t n, int64_t k_, int64_t r_, int64_t be_, int64_t g_, bool per_link_prep, bool tensor,
                       int64_t flag_host, int64_t prm_host, int64_t out_host, int64_t done_host) {
    const c10::cuda::CUDAGuard guard(run_device(state, "state"));
    Bufs b = make_bufs(mem, canon, out, prm, state, partials, pw, mbuf, ablk, k_, r_, be_, g_, tensor, per_link_prep);
    if (n > POSM_MAX_LINKS || (int64_t)out.numel() < 2 + n) throw std::runtime_error("chain too long for the output buffer");
    if ((u64)partials.numel() < (u64)groups * b.G) throw std::runtime_error("partials buffer needs groups * g words");
    DevState& d = ensure_side_stream();
    GraphKey key{cur_device(), b.mem, b.canon, b.out, b.prm, b.state, b.partials, b.pw, b.mbuf, b.ablk, (int)groups, (int)threads, (int)n, tensor, b.K, b.r,
                 b.be, b.pwcount, mbuf.numel(), ablk.numel(), b.G + (per_link_prep ? 0x10000u : 0u) + 0x20000u};
    ArmedKey akey{key, flag_host, prm_host, out_host, done_host};
    auto it = g_armed_graphs.find(akey);
    if (it == g_armed_graphs.end()) {
        u64* flag = mapped_ptr(flag_host); u64* psrc = mapped_ptr(prm_host); u64* odst = mapped_ptr(out_host); u64* done = mapped_ptr(done_host);
        cudaGraphExec_t exec = capture_graph(d.side, [&] {
            posm_wait_kernel<<<1, 64, 0, d.side>>>(flag, psrc, reinterpret_cast<u64*>(const_cast<posm_params*>(b.prm)));
            enqueue_chain(d.side, b, (int)groups, (int)threads, (int)n, tensor);
            posm_publish_kernel<<<1, 64, 0, d.side>>>(b.out, odst, done, (u32)(2 + n));
        });
        it = g_armed_graphs.emplace(akey, exec).first;
    }
    CUDA_OK(cudaGraphLaunch(it->second, d.side));
}

// Setup table pass: T_seed[j] for j in [j0, j0 + nblocks) written to table[(j - j0) * G ...].
// The seed-dependent tables are built once; per block: set state, prepare, partial, finish(table row).
void entry_table(torch::Tensor mem, torch::Tensor canon, torch::Tensor prm, torch::Tensor state, torch::Tensor partials,
                 torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk, torch::Tensor table,
                 int64_t seed, int64_t j0, int64_t nblocks, int64_t groups, int64_t threads, int64_t k_, int64_t r_, int64_t be_, int64_t g_, bool tensor) {
    const c10::cuda::CUDAGuard guard(run_device(state, "state"));
    same_device(table, run_device(state, "state"), "table");
    torch::Tensor dummy_out = state;   // finish writes only the table row in table mode
    Bufs b = make_bufs(mem, canon, dummy_out, prm, state, partials, pw, mbuf, ablk, k_, r_, be_, g_, tensor, true);
    if ((u64)table.numel() < (u64)nblocks * b.G) throw std::runtime_error("table buffer too small");
    int gr = (int)groups, RT, CT; tc_shape(b, tensor, gr, RT, CT);
    auto s = at::cuda::getCurrentCUDAStream();
    u64* tab = p64m(table);
    posm_setstate_kernel<<<1, 1, 0, s>>>(b.prm, b.state, (u64)j0, (u64)seed);
    enqueue_link_prep(s, b);
    for (int64_t jb = 0; jb < nblocks; jb++) {
        posm_setstate_kernel<<<1, 1, 0, s>>>(b.prm, b.state, (u64)(j0 + jb), (u64)seed);
        enqueue_link_work(s, b, gr, (int)threads, tensor, RT, CT);
        posm_link_finish_kernel<<<1, threads, 0, s>>>(b.prm, b.state, b.partials, b.out, b.pw, tab + (u64)jb * b.G, (u32)gr, 0);
    }
    CUDA_OK(cudaGetLastError());
}

// one-way image T = H(Theta') of a whole array (setup): what the verifier keeps
void image(torch::Tensor theta, torch::Tensor out, int64_t g0) {
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t groups = theta.numel() / 4;
    posm_image_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(p64(theta), p64m(out), (u64)groups, (u64)g0);
    CUDA_OK(cudaGetLastError());
}
void filler(torch::Tensor out, int64_t seed, int64_t start) {
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = out.numel(); posm_filler_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p64m(out), (u64)seed, (u64)start, (u64)n);
    CUDA_OK(cudaGetLastError());
}
void fill_pages(torch::Tensor dst, torch::Tensor pos0, int64_t pw, int64_t seed, int64_t key_lo, int64_t key_hi, int64_t mask) {
    const c10::cuda::CUDAGuard guard(run_device(dst, "dst"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = dst.numel() * pw;
    if (n == 0) return;
    posm_fill_pages_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p64(dst), p64(pos0), (u64)dst.numel(), (u64)pw, (u64)seed, (u64)key_lo, (u64)key_hi, (u64)mask);
    CUDA_OK(cudaGetLastError());
}
void kv_code(int64_t raw_base, int64_t enc_base, int64_t q_base, torch::Tensor ent, int64_t rw, int64_t key_lo, int64_t key_hi, bool decode) {
    const c10::cuda::CUDAGuard guard(run_device(ent, "ent"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = ent.numel() / 4;
    if (n == 0) return;
    unsigned grid = (unsigned)(n < (1 << 20) ? n : (1 << 20));
    posm_kv_kernel<<<grid, 256, 0, s>>>((u64)raw_base, (u64)enc_base, (u64)q_base, p64(ent), (u64)n, (u64)rw, (u64)key_lo, (u64)key_hi, decode ? 1 : 0);
    CUDA_OK(cudaGetLastError());
}
void pack(torch::Tensor raw, torch::Tensor payload, int64_t groups) {
    const c10::cuda::CUDAGuard guard(run_device(payload, "payload"));
    auto s = at::cuda::getCurrentCUDAStream();
    posm_pack_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(p64(raw), p64m(payload), (u64)groups);
    CUDA_OK(cudaGetLastError());
}
void decode_unpack(torch::Tensor theta, torch::Tensor raw, int64_t q0, int64_t groups, int64_t key_lo, int64_t key_hi) {
    const c10::cuda::CUDAGuard guard(run_device(raw, "raw"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t workers = POSM_PACKED128 ? groups * 128 : (POSM_VDE_BITS == 89 ? groups * 64 : groups);
    posm_decode_unpack_kernel<<<(unsigned)((workers + 255) / 256), 256, 0, s>>>(p64(theta), p64m(raw), (u64)q0, (u64)groups, (u64)key_lo, (u64)key_hi);
    CUDA_OK(cudaGetLastError());
}
void setup_at(torch::Tensor raw, torch::Tensor out, int64_t key_lo, int64_t key_hi, int64_t q0) {
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n=raw.numel();
#if POSM_PACKED128
    TORCH_CHECK(!(q0%16 || n%16), "packed128 setup requires complete rows");
#elif POSM_VDE_BITS == 89
    TORCH_CHECK(!(q0%8 || n%8), "89-bit setup requires complete 64-byte rows");
#endif
    posm_setup_at_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p64(raw), p64m(out), (u64)key_lo, (u64)key_hi, (u64)q0, (u64)n);
    CUDA_OK(cudaGetLastError());
}
void setup(torch::Tensor raw, torch::Tensor out, torch::Tensor prm) {
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = raw.numel(); posm_setup_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p64(raw), p64m(out), setup_ptr(prm));
    CUDA_OK(cudaGetLastError());
}
void decode(torch::Tensor mem, torch::Tensor out, torch::Tensor prm) {
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = mem.numel(); posm_decode_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p64(mem), p64m(out), setup_ptr(prm));
    CUDA_OK(cudaGetLastError());
}
void decode_at_packed128(torch::Tensor mem, torch::Tensor out, int64_t key_lo,
                         int64_t key_hi, int64_t q0) {
#if POSM_PACKED128
    const c10::cuda::CUDAGuard guard(run_device(out, "out"));
    TORCH_CHECK(mem.numel() == out.numel() && !(mem.numel() % 16) && !(q0 % 16),
                "packed128 interval requires equal, complete 128-byte rows");
    auto s = at::cuda::getCurrentCUDAStream();
    int64_t n = mem.numel();
    posm_decode_at_packed128_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(
        p64(mem), p64m(out), (u64)key_lo, (u64)key_hi, (u64)q0, (u64)n);
    CUDA_OK(cudaGetLastError());
#else
    TORCH_CHECK(false, "decode_at_packed128 requires POSM_PACKED128");
#endif
}
// Let kernels on `dev` dereference memory that lives on `peer` (NVLink / PCIe peer access).
void enable_peer(int64_t dev, int64_t peer) {
    int cur; CUDA_OK(cudaGetDevice(&cur));
    CUDA_OK(cudaSetDevice((int)dev));
    cudaError_t e = cudaDeviceEnablePeerAccess((int)peer, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { cudaSetDevice(cur); CUDA_OK(e); }
    cudaGetLastError();
    CUDA_OK(cudaSetDevice(cur));
}
