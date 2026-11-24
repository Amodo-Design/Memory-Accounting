"""Where stream-decode's time goes: the decode kernel taken apart, and its share of a vLLM step.

  python scripts/bench_decode.py kernels --words 2030884864 --unit-words 72531602 --out results/decode/kernels_qwen7b.json
  python scripts/bench_decode.py insitu --model Qwen/Qwen2.5-7B-Instruct --out results/decode/insitu_qwen7b.json

kernels: decodes an array the size of a model's Θ' (--words, its payload words; --unit-words per launch, about one
layer) with the real kernel (posm_cuda decode_unpack) and with variants that drop or swap one part of it, timed with
CUDA events (median of --reps passes over the whole array). Every variant reads Θ' and writes the raw words, so the
differences are the parts' costs:
  real            posm_cuda's decode_unpack (one thread per 16-word group)
  same            this file's copy of it, to check the copy times the same
  copy            load + unpack + store, no decode: the memory traffic and the unpacking
  square          + the squaring (VDE decode) with a free tweak: everything but the PRF
  hash            the PRF alone (SipHash-2-4-128 of a 4-word message, the tweak_at in use), no loads
  coalesced       the real decode with one thread per word and the unpacking through shared memory (same result)
  sip24-1w        SipHash-2-4 of a 1-word message (q alone): the same PRF, 8 rounds a word instead of 14
  sip24-pair      SipHash-2-4-128 of a 1-word message, both output halves used: one call per two words
  sip13-1w        SipHash-1-3 of a 1-word message
  chacha12/8      ChaCha12 / ChaCha8 in counter mode: one 512-bit block per 8 words
  tile-<v>        <v> with coalesced loads and stores: 2048 words a block of 256 threads, consecutive threads on
                  consecutive words, staged in shared memory (tile = the real tweak)
The sip* and chacha* variants change the tweak, so the encoding (a scheme change); they show what each would buy. Each coalesced
variant is checked against the real kernel's output (or its one-thread-a-group twin's).

insitu: a vLLM engine with the PoSM settings, converted (posm_convert), profiled with torch.profiler over decode steps
at each --batches size: CUDA time per forward step (prefill steps included) in the decode kernel, in matmuls, in attention and in the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CUDA_SRC = r"""
#include "posm_core.h"
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ u64 sip_small(u64 k0, u64 k1, u64 m0, int nmsg, int c, int d, u64* hi) {
    u64 v0 = k0 ^ U64C(0x736f6d6570736575), v1 = k1 ^ U64C(0x646f72616e646f6d);
    u64 v2 = k0 ^ U64C(0x6c7967656e657261), v3 = k1 ^ U64C(0x7465646279746573);
    if (hi) v1 ^= U64C(0xee);
    if (nmsg) { v3 ^= m0; for (int i = 0; i < c; i++) SIPROUND(v0, v1, v2, v3); v0 ^= m0; }
    u64 b = (u64)(8 * nmsg) << 56;
    v3 ^= b; for (int i = 0; i < c; i++) SIPROUND(v0, v1, v2, v3); v0 ^= b;
    v2 ^= hi ? U64C(0xee) : U64C(0xff);
    for (int i = 0; i < d; i++) SIPROUND(v0, v1, v2, v3);
    u64 lo = v0 ^ v1 ^ v2 ^ v3;
    if (hi) { v1 ^= U64C(0xdd); for (int i = 0; i < d; i++) SIPROUND(v0, v1, v2, v3); *hi = v0 ^ v1 ^ v2 ^ v3; }
    return lo;
}

__device__ __forceinline__ u32 rotl32(u32 x, int b) { return __funnelshift_l(x, x, b); }
#define QR(a, b, c, d) a += b; d ^= a; d = rotl32(d, 16); c += d; b ^= c; b = rotl32(b, 12); \
                       a += b; d ^= a; d = rotl32(d, 8);  c += d; b ^= c; b = rotl32(b, 7);
template <int ROUNDS>
__device__ __forceinline__ void chacha_tweaks(u64 klo, u64 khi, u64 ctr, u64* t8) {
    u32 s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, (u32)klo, (u32)(klo >> 32), (u32)khi, (u32)(khi >> 32),
                 (u32)klo ^ 0x5a5a5a5au, (u32)(klo >> 32), (u32)khi, (u32)(khi >> 32), (u32)ctr, (u32)(ctr >> 32), 0x5457454Bu, 1u};
    u32 x[16];
#pragma unroll
    for (int i = 0; i < 16; i++) x[i] = s[i];
#pragma unroll
    for (int r = 0; r < ROUNDS; r += 2) {
        QR(x[0], x[4], x[8], x[12]) QR(x[1], x[5], x[9], x[13]) QR(x[2], x[6], x[10], x[14]) QR(x[3], x[7], x[11], x[15])
        QR(x[0], x[5], x[10], x[15]) QR(x[1], x[6], x[11], x[12]) QR(x[2], x[7], x[8], x[13]) QR(x[3], x[4], x[9], x[14])
    }
#pragma unroll
    for (int i = 0; i < 8; i++) t8[i] = red61((((u64)(x[2 * i + 1] + s[2 * i + 1]) << 32) | (x[2 * i] + s[2 * i])) & PM61);
}

// One thread per 16-word group, as posm_decode_unpack_kernel. MODE: 0 copy, 1 same, 2 square, 3 hash, 4 sip24-1w,
// 5 sip24-pair, 6 sip13-1w, 7 chacha12, 8 chacha8
template <int MODE>
__global__ void k_group(const u64* theta, u64* raw, u64 q0, u64 groups, u64 klo, u64 khi) {
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    u64 p[16], w[15];
    if (MODE == 3) {
        u64 acc = 0;
#pragma unroll
        for (u32 i = 0; i < 16; i++) acc ^= tweak_at(klo, khi, q0 + 16 * g + i);
        raw[g] = acc;
        return;
    }
    u64 t[16];
    if (MODE == 7 || MODE == 8) {
        u64 ctr = (q0 >> 3) + 2 * g;
        if (MODE == 7) { chacha_tweaks<12>(klo, khi, ctr, t); chacha_tweaks<12>(klo, khi, ctr + 1, t + 8); }
        else           { chacha_tweaks<8>(klo, khi, ctr, t);  chacha_tweaks<8>(klo, khi, ctr + 1, t + 8); }
    }
#pragma unroll
    for (u32 i = 0; i < 16; i++) {
        u64 q = q0 + 16 * g + i, x = theta[q];
        if (MODE == 0) p[i] = x;
        else if (MODE == 1) p[i] = vde_decode(x, tweak_at(klo, khi, q));
        else if (MODE == 2) p[i] = vde_decode(x, q & PM61);
        else if (MODE == 4) p[i] = vde_decode(x, red61(sip_small(klo, khi, q, 1, 2, 4, nullptr) & PM61));
        else if (MODE == 5) {
            if ((i & 1) == 0) {
                u64 h, l = sip_small(klo, khi, q >> 1, 1, 2, 4, &h);
                t[i] = red61(l & PM61); t[i + 1] = red61(h & PM61);
            }
            p[i] = vde_decode(x, t[i]);
        }
        else if (MODE == 6) p[i] = vde_decode(x, red61(sip_small(klo, khi, q, 1, 1, 3, nullptr) & PM61));
        else p[i] = vde_decode(x, t[i]);
    }
    unpack_group(p, w);
#pragma unroll
    for (u32 a = 0; a < 15; a++) raw[15 * g + a] = w[a];
}

// One thread per word: 256 words (16 groups) a block, decoded into shared memory, then 240 threads write the 240 raw
// words contiguously. DECODE 0: copy only.
template <int DECODE>
__global__ void k_coalesced(const u64* theta, u64* raw, u64 q0, u64 groups, u64 klo, u64 khi) {
    __shared__ u64 sp[256];
    u64 base = (u64)blockIdx.x * 256, q = q0 + base + threadIdx.x;
    bool in = base + threadIdx.x < 16 * groups;
    u64 x = in ? theta[q] : 0;
    sp[threadIdx.x] = DECODE ? vde_decode(x, tweak_at(klo, khi, q)) : x;
    __syncthreads();
    if (threadIdx.x >= 240) return;
    u32 gl = threadIdx.x / 15, a = threadIdx.x % 15;
    if ((u64)blockIdx.x * 16 + gl >= groups) return;
    const u64* in16 = sp + 16 * gl;
    u32 start = 64 * a, k = start / 60, off = start % 60;
    u64 v = in16[k] >> off;
    if (k + 1 < 16) v |= in16[k + 1] << (60 - off);
    if (k + 2 < 16 && 120 - off < 64) v |= in16[k + 2] << (120 - off);
    raw[(u64)blockIdx.x * 240 + threadIdx.x] = v;
}

// Coalesced: 2048 words (128 groups) a block of 256 threads. The tweaks of a stream PRF (ChaCha: 8 a block, SipHash
// pair: 2 a call) first go to shared memory; then each thread decodes words tid, tid + 256, ... (consecutive threads,
// consecutive words) into shared memory, and the 1920 raw words are written the same way. TW as k_group's MODE
// (0 copy, 1 real, 2 square, 4 sip24-1w, 5 sip24-pair, 6 sip13-1w, 7 chacha12, 8 chacha8).
template <int TW>
__global__ void __launch_bounds__(256) k_tile(const u64* theta, u64* raw, u64 q0, u64 groups, u64 klo, u64 khi) {
    __shared__ u64 sp[2048];
    const u32 tid = threadIdx.x;
    const u64 base = (u64)blockIdx.x * 2048, nw = 16 * groups;
    if (TW == 7 || TW == 8) {
        u64 t8[8];
        if (TW == 7) chacha_tweaks<12>(klo, khi, ((q0 + base) >> 3) + tid, t8); else chacha_tweaks<8>(klo, khi, ((q0 + base) >> 3) + tid, t8);
#pragma unroll
        for (int j = 0; j < 8; j++) sp[8 * tid + j] = t8[j];
        __syncthreads();
    } else if (TW == 5) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 pr = i * 256 + tid; u64 h, l = sip_small(klo, khi, (q0 + base) / 2 + pr, 1, 2, 4, &h);
            sp[2 * pr] = red61(l & PM61); sp[2 * pr + 1] = red61(h & PM61);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 l = i * 256 + tid; u64 w = base + l;
        if (w < nw) {
            u64 q = q0 + w, x = theta[q], t = 0;
            if (TW == 1) t = tweak_at(klo, khi, q);
            else if (TW == 2) t = q & PM61;
            else if (TW == 4) t = red61(sip_small(klo, khi, q, 1, 2, 4, nullptr) & PM61);
            else if (TW == 6) t = red61(sip_small(klo, khi, q, 1, 1, 3, nullptr) & PM61);
            else if (TW >= 5) t = sp[l];
            sp[l] = TW == 0 ? x : vde_decode(x, t);
        }
    }
    __syncthreads();
    const u64 g0 = (u64)blockIdx.x * 128;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 l = i * 256 + tid;
        if (l >= 1920) break;
        u32 gl = l / 15, a = l % 15;
        if (g0 + gl >= groups) break;
        const u64* in16 = sp + 16 * gl;
        u32 start = 64 * a, k = start / 60, off = start % 60;
        u64 v = in16[k] >> off;
        if (k + 1 < 16) v |= in16[k + 1] << (60 - off);
        if (k + 2 < 16 && 120 - off < 64) v |= in16[k + 2] << (120 - off);
        raw[15 * g0 + l] = v;
    }
}

void run(int64_t mode, torch::Tensor theta, torch::Tensor raw, int64_t q0, int64_t groups, int64_t klo, int64_t khi) {
    auto s = at::cuda::getCurrentCUDAStream();
    const u64* t = (const u64*)theta.data_ptr<int64_t>(); u64* r = (u64*)raw.data_ptr<int64_t>();
    unsigned gb = (unsigned)((groups + 255) / 256);
    switch (mode) {
        case 0: k_group<0><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 1: k_group<1><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 2: k_group<2><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 3: k_group<3><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 4: k_group<4><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 5: k_group<5><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 6: k_group<6><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 7: k_group<7><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 8: k_group<8><<<gb, 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 9: k_coalesced<1><<<(unsigned)((groups + 15) / 16), 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        case 10: k_coalesced<0><<<(unsigned)((groups + 15) / 16), 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
#define TILE(tw) case 20 + tw: k_tile<tw><<<(unsigned)((groups + 127) / 128), 256, 0, s>>>(t, r, q0, groups, klo, khi); break;
        TILE(0) TILE(1) TILE(2) TILE(4) TILE(5) TILE(6) TILE(7) TILE(8)
    }
}
"""

VARIANTS = {
    "copy": 0,
    "same": 1,
    "square": 2,
    "hash": 3,
    "sip24-1w": 4,
    "sip24-pair": 5,
    "sip13-1w": 6,
    "chacha12": 7,
    "chacha8": 8,
    "coalesced": 9,
    "coalesced-copy": 10,
    "tile-copy": 20,
    "tile": 21,
    "tile-square": 22,
    "tile-sip24-1w": 24,
    "tile-sip24-pair": 25,
    "tile-sip13-1w": 26,
    "tile-chacha12": 27,
    "tile-chacha8": 28,
}
CHECKED = {
    "same": "real",
    "coalesced": "real",
    "tile": "real",
    "tile-copy": "copy",
    "tile-square": "square",
    "tile-sip24-1w": "sip24-1w",
    "tile-sip24-pair": "sip24-pair",
    "tile-sip13-1w": "sip13-1w",
    "tile-chacha12": "chacha12",
    "tile-chacha8": "chacha8",
}
M64 = (1 << 64) - 1


def _s64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def kernels(args):
    import torch
    from torch.utils.cpp_extension import load_inline

    from memory_accounting.core.backend import DeviceBackend

    mod = load_inline(
        "posm_decode_bench",
        cpp_sources="void run(int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources="#include <torch/extension.h>\n" + CUDA_SRC,
        functions=["run"],
        extra_include_paths=[str(ROOT / "memory_accounting" / "kernels")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    be = DeviceBackend("cuda")
    key = int(args.key, 16)
    klo, khi = _s64(key & M64), _s64(key >> 64)
    words = args.words // 16 * 16
    unit = max(16, args.unit_words // 16 * 16)
    theta = torch.randint(0, (1 << 61) - 1, (words,), dtype=torch.int64, device="cuda")
    theta |= torch.randint(0, 2, (words,), dtype=torch.int64, device="cuda") << 61  # the non-residue flag
    raw = torch.empty(unit // 16 * 15, dtype=torch.int64, device="cuda")
    units = [(q, min(unit, words - q)) for q in range(0, words, unit)]

    def one_pass(name):
        for q, n in units:
            if name == "real":
                be.decode_unpack(theta, raw, q, n // 16, key)
            else:
                mod.run(VARIANTS[name], theta, raw, q, n // 16, klo, khi)

    def check(name):  # the first unit, against the real kernel or the one-thread-a-group variant
        q, n = units[0]
        if CHECKED[name] == "real":
            be.decode_unpack(theta, raw, q, n // 16, key)
        else:
            mod.run(VARIANTS[CHECKED[name]], theta, raw, q, n // 16, klo, khi)
        ref = raw[: n // 16 * 15].clone()
        mod.run(VARIANTS[name], theta, raw, q, n // 16, klo, khi)
        return bool(torch.equal(ref, raw[: n // 16 * 15]))

    names = ["real"] + [v for v in (args.variants.split(",") if args.variants else VARIANTS)]
    gb = words * 8 / 1e9
    out = {
        "words": words,
        "unit_words": unit,
        "launches": len(units),
        "theta_gb": gb,
        "raw_gb": words // 16 * 15 * 8 / 1e9,
        "device": torch.cuda.get_device_name(),
        "variants": {},
    }
    print(f"Θ' {gb:.2f} GB ({words} words) in {len(units)} launches of {unit} words; {out['device']}", flush=True)
    for name in names:
        ok = check(name) if name in CHECKED else None
        one_pass(name)
        torch.cuda.synchronize()  # warm-up
        times = []
        for _ in range(args.reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            one_pass(name)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        ms = sorted(times)[len(times) // 2]
        moved = out["theta_gb"] + (0 if name == "hash" else out["raw_gb"])
        out["variants"][name] = {
            "ms": ms,
            "ms_all": times,
            "gb_s": moved / ms * 1e3,
            "words_per_ns": words / ms / 1e6,
            "matches": ok,
        }
        print(
            f"  {name:15s} {ms:8.2f} ms  {moved / ms * 1e3:7.0f} GB/s moved  {words / ms / 1e6:6.2f} words/ns"
            + ("" if ok is None else f"  matches {CHECKED[name]}: {ok}"),
            flush=True,
        )
    start = torch.empty(words // 16 * 15, dtype=torch.int64, device="cuda")
    times = []
    for _ in range(args.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        start.copy_(theta[: words // 16 * 15])
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    ms = sorted(times)[len(times) // 2]
    out["variants"]["dram-copy"] = {"ms": ms, "ms_all": times, "gb_s": 2 * out["raw_gb"] / ms * 1e3}
    print(
        f"  {'dram-copy':15s} {ms:8.2f} ms  {2 * out['raw_gb'] / ms * 1e3:7.0f} GB/s moved (torch copy of the raw size: the floor)",
        flush=True,
    )
    return out


def insitu(args):
    # the engine in this process, so the profiler sees its kernels
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile

    from memory_accounting.models.vllm import apply_env, settings

    apply_env()
    from vllm import LLM, SamplingParams

    llm = LLM(**settings(args.model, args.max_model_len, args.gpu_memory_utilization))
    out = {"model": args.model, "batches": {}}
    rng = np.random.default_rng(1)

    def reqs(b):
        return [{"prompt_token_ids": rng.integers(1000, 30000, args.prompt_len).tolist()} for _ in range(b)]

    def classify(name: str) -> str:
        n = name.lower()
        if "decode_unpack" in n:
            return "decode"
        if any(s in n for s in ("posm_kv", "fill_pages", "setup_at")):  # the dynamic region held encoded (v0.17)
            return "kv"
        if any(
            s in n for s in ("gemm", "gemv", "matmul", "cutlass", "sm90_xmma", "nvjet", "mm_kernel", "_mm_", "splitk")
        ):
            return "matmul"
        if any(s in n for s in ("attn", "attention", "paged", "flash", "unified")):
            return "attention"
        return "other"

    def measure(tag, batches):
        res = {}
        for batch in batches:
            sp = SamplingParams(temperature=0.0, max_tokens=args.gen_len, ignore_eos=True)
            llm.generate(reqs(batch), sp, use_tqdm=False)  # warm-up
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                llm.generate(reqs(batch), sp, use_tqdm=False)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            cats, top, decodes = {}, {}, 0
            for e in prof.key_averages():
                t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
                if not t or e.device_type.name != "CUDA":
                    continue
                c = classify(e.key)
                decodes += e.count if c == "decode" else 0
                cats[c] = cats.get(c, 0.0) + t / 1e3
                top[e.key] = top.get(e.key, 0.0) + t / 1e3
            # forward steps: the prompts at most max_model_len tokens a step (no chunked prefill; the PoSM settings'
            # token
            # budget), which also gives every request its first token, then gen_len - 1 decode steps
            steps = -(-batch * args.prompt_len // args.max_model_len) + args.gen_len - 1
            res[str(batch)] = {
                "wall_s": wall,
                "tok_s": batch * args.gen_len / wall,
                "steps": steps,
                "decode_launches": decodes,
                "ms_per_step_wall": wall / steps * 1e3,
                "gpu_ms_per_step": {c: v / steps for c, v in cats.items()},
                "top": sorted(((k, v / steps) for k, v in top.items()), key=lambda kv: -kv[1])[:12],
            }
            r = res[str(batch)]
            print(
                f"{tag} batch {batch}: {steps} steps ({decodes / steps:.1f} decode launches a step); {r['ms_per_step_wall']:.1f} ms a step wall; GPU ms a step "
                + ", ".join(f"{c} {v:.1f}" for c, v in sorted(r["gpu_ms_per_step"].items(), key=lambda kv: -kv[1])),
                flush=True,
            )
        return res

    batches = [int(x) for x in args.batches.split(",")]
    out["plain"] = measure("plain", batches)
    cap = {"filler_max_bytes": 8 << 30} if args.dynamic else {}  # leave memory for the encoded pool set up after
    out["converted"] = llm.collective_rpc(
        "posm_convert", args=(args.key, args.filler_seed, 0.0), kwargs={"reserve_bytes": -1, **cap}
    )[0]
    out["encoded"] = measure("encoded", batches)
    if args.dynamic:  # + the dynamic region, its KV pool and activation rows encoded
        out["dynamic_setup"] = llm.collective_rpc("posm_dynamic_setup", args=("1M", args.filler_seed, False))[0]
        out["dynamic"] = measure("dynamic", batches)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("--words", type=int, default=2030884864, help="Θ' payload words (default: Qwen2.5-7B's)")
    k.add_argument(
        "--unit-words", type=int, default=72531602, help="words a launch (default: Qwen2.5-7B's Θ' over its 28 layers)"
    )
    k.add_argument("--variants", default=None, help="comma-separated (default: all)")
    k.add_argument("--reps", type=int, default=5)
    i = sub.add_parser("insitu")
    i.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    i.add_argument("--batches", default="1,16,64,256")
    i.add_argument("--prompt-len", type=int, default=256)
    i.add_argument("--gen-len", type=int, default=64)
    i.add_argument("--max-model-len", type=int, default=2048)
    i.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    i.add_argument("--filler-seed", default="5eed5eed5eed5eed")
    i.add_argument(
        "--dynamic",
        action="store_true",
        help="then set up the dynamic region (encoded, v0.17) and profile again; the "
        "static filler is capped at 8 GiB to leave its memory",
    )
    for p in (k, i):
        p.add_argument("--key", default="0123456789abcdef0123456789abcdef")
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = kernels(args) if args.cmd == "kernels" else insitu(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
