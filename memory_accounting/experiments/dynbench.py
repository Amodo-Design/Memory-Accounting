"""Dynamic-chain sweep: round-trip distributions of the dynamic chains (all of the dynamic memory, and the live region)
for the honest prover and the dynamic cheats over (B, n, r), on an encoded dynamic region (v0.17) laid out as a real
prover lays it out, on one GPU with no model running. The dynamic counterpart of `memory_accounting.experiments.bench`.

  python -m memory_accounting.experiments.dynbench --geometry vllm --blocks 64K,256K,1M,4M --links 4,8,16,32,64 --r 0,2048 --out dyn.json
  python -m memory_accounting.experiments.dynbench --geometry int --int-model models/smollm2-135m.int.safetensors ...

Geometries (what the region is made of; its contents are random payloads, encoded, which the timing does not see):
  vllm  `memory_accounting.prover.vllm_serve.EncodedPool`'s: --layers stores of KV pages (Qwen2.5-0.5B's 8 KiB pages by default),
        the two activation stores, the pad; --pool-gib of pool (the real one is 58 GiB; the chain's cost per link
        does not depend on the region's size past the GPU's caches). The live region: --requests requests of
        --live-tokens tokens each, their pages in stage order (one segment per page, as the prover's table has them),
        then the activation stores.
  int   `memory_accounting.prover.kvenc.EncodedStatic`'s for the integer model in --int-model: per layer its four cache buffers'
        stores (--max-len slots), the activation store, the pad. The live region: each store's groups holding the
        first --live-tokens slots, then the activation store.

Postures, each answering the same challenges, in a random order per sample (drift falls on all alike):
  honest            the region's own table (the main chain) and the live region's (the live chain)
  dyn_encode_otf@f  (main chain) the last share f of the KV and activation stores is held as its payload and encoded
                    when a chain reads it (`memory_accounting.prover.strategies.DynShareCheat`, kernel mode 4)
  dyn_dram@f        (main chain) the top share f of every KV store's units, free in the stage, is read from pinned
                    host memory over PCIe (`memory_accounting.prover.dynamic.DramEvict`, the live dyn_dram posture's cheat)
Each cheat's answers are checked against honest's for the same challenge (`correct`); `deviating_share` is the share
of the region's words the cheat reads its way, so a chain of n links has about n * share deviating reads.
The live chain has no cheat here: its reads are of the same encoded words, so a cheat's cost per read is the main
chain's (its Δc).
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from .bench import parse_bytes, parse_list
from ..core.codec_profile import FIELD_BITS


def _gen_payload(n: int, dev, gen) -> torch.Tensor:
    """Random payload words in the selected codec's packing layout."""
    if FIELD_BITS == 89:
        if n % 8:
            raise ValueError("grouped 89-bit payload needs whole 64-byte rows")
        raw = torch.randint(0, 256, (n // 8, 55), dtype=torch.uint8, device=dev, generator=gen)
        slots = torch.zeros((n // 8, 64), dtype=torch.uint8, device=dev)
        for lane in range(5):
            slots[:, 12 * lane : 12 * lane + 11] = raw[:, 11 * lane : 11 * lane + 11]
        return slots.view(torch.int64).reshape(-1)
    hi = torch.randint(0, 1 << 28, (n,), dtype=torch.int64, device=dev, generator=gen)
    lo = torch.randint(0, 1 << 32, (n,), dtype=torch.int64, device=dev, generator=gen)
    return (hi << 32) | lo


def fill_encoded(big: torch.Tensor, key: int, seed: int, keep: int, chunk: int = 1 << 26) -> torch.Tensor:
    """Fill `big` (the region's stores, region positions from 0) with random payloads encoded at their positions; return
    the payload of words [keep, len) (the encode-on-the-fly cheat's source)."""
    from ..prover.dynfill import _backend
    from ..prover.kvenc import KV_POS

    be = _backend(big.device)
    gen = torch.Generator(device=big.device)
    gen.manual_seed(seed)
    n = big.numel()
    tail = torch.empty(max(0, n - keep), dtype=torch.int64, device=big.device)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        p = _gen_payload(hi - lo, big.device, gen)
        be.setup_at(p, big[lo:hi], key, KV_POS | lo)
        a, b = max(lo, keep), hi
        if a < b:
            tail[a - keep : b - keep].copy_(p[a - lo : b - lo])
    torch.cuda.synchronize(big.device)
    return tail


class Geometry:
    """A synthetic encoded region: `stores` [(name, EncStore over `big`)], `kv_units` [(name, units, unit words)] for the
    dram cheat, the live region's buffers (`live_buffers`) and the stage's free test for each store's units."""


def vllm_geometry(args, dev) -> Geometry:
    from ..prover.kvenc import EncStore, enc_words

    g = Geometry()
    L, pw = args.layers, args.page_words
    epw = enc_words(pw)
    nb = int(args.pool_gib * (1 << 30)) // (L * pw * 8)
    rows, hid = args.act_rows, args.hidden
    aplain = rows * hid * 2 // 8
    aw = enc_words(aplain)
    g.kv_words = L * nb * epw
    g.big = torch.empty(g.kv_words + 2 * aw, dtype=torch.int64, device=dev)
    g.stores = [
        (
            f"kv.{layer}",
            EncStore(g.big[layer * nb * epw : (layer + 1) * nb * epw], pw, nb, "bfloat16", (2, 16, pw * 8 // 64)),
        )
        for layer in range(L)
    ]
    g.stores += [
        (n, EncStore(g.big[g.kv_words + i * aw : g.kv_words + (i + 1) * aw], aplain, 1, "bfloat16", (rows, hid)))
        for i, n in enumerate(("act.h", "act.r"))
    ]
    rng = np.random.default_rng(args.seed)
    per = -(-args.live_tokens // 16)
    used = rng.permutation(int(nb * 0.25))[: args.requests * per]  # vLLM hands out low block ids first
    g.stage = {
        "requests": [{"block_ids": [int(b) for b in used[i * per : (i + 1) * per]]} for i in range(args.requests)]
    }
    live_blocks = np.zeros(nb, dtype=bool)
    live_blocks[used] = True
    g.kv_units = [(f"kv.{layer}", nb, epw) for layer in range(L)]
    g.free = lambda name, first, units: ~live_blocks[first : first + units]
    base = g.big.data_ptr()

    def live(region):
        from ..prover.dynamic import PagedLive

        ptrs = np.array(
            [
                base + 8 * (layer * nb * epw + b * epw)
                for r in g.stage["requests"]
                for layer in range(L)
                for b in r["block_ids"]
            ],
            dtype=np.uint64,
        )

        def names():
            return [f"p{k}" for k in range(len(ptrs))]

        def make(k):
            return g.big[(int(ptrs[k]) - base) // 8 : (int(ptrs[k]) - base) // 8 + epw].view(torch.uint8)

        return PagedLive(
            ptrs, epw, names, make, g.stores[-2:], enc=EncStore(None, pw, 1, "bfloat16", (2, 16, pw * 8 // 64))
        )

    g.live = live
    g.desc = {
        "geometry": "vllm",
        "layers": L,
        "page_words": pw,
        "blocks": nb,
        "act_rows": rows,
        "hidden": hid,
        "requests": args.requests,
        "live_tokens": args.live_tokens,
        "live_pages": args.requests * per * L,
    }
    return g


def int_geometry(args, dev) -> Geometry:
    from ..models.intllm import IntLLM
    from ..prover.kvenc import EncStore, enc_words, groups_of

    model = IntLLM.from_file(args.int_model, args.max_len, "cpu")
    named = model.kv_buffers()
    act_plain = args.max_len * model.d  # float64 rows
    sizes = [
        (n, t.numel() * t.element_size() // 8, str(t.dtype).replace("torch.", ""), tuple(t.shape)) for n, t in named
    ]
    sizes.append(("act", act_plain, "float64", (args.max_len, model.d)))
    total = sum(enc_words(p) for _, p, _, _ in sizes)
    g = Geometry()
    g.big = torch.empty(total, dtype=torch.int64, device=dev)
    g.stores, off = [], 0
    for n, p, dt, shp in sizes:
        w = enc_words(p)
        g.stores.append((n, EncStore(g.big[off : off + w], p, 1, dt, shp)))
        off += w
    g.kv_words = off - enc_words(act_plain)
    S, t = args.max_len, args.live_tokens
    rows = {n: p * 8 // S for n, p, _, _ in sizes[:-1]}  # bytes a slot
    live_groups = {n: groups_of(-(-t * rows[n] // 8)) for n in rows}
    g.kv_units = [(n, groups_of(st.plain), 16) for n, st in g.stores[:-1]]
    g.free = lambda name, first, units: np.arange(first, first + units) >= live_groups[name]
    g.stage = {}
    g.live = lambda region: [(n, st.prefix(live_groups[n])) for n, st in g.stores[:-1]] + [g.stores[-1]]
    g.desc = {
        "geometry": "int",
        "model": str(args.int_model),
        "layers": len(model.layers),
        "max_len": S,
        "live_tokens": t,
    }
    return g


def run(args) -> dict:
    from ..core.params import Challenge, Keys
    from ..prover.dynamic import DramEvict, DynamicRegion, PagedLive
    from ..prover.dynfill import dyn_key
    from ..prover.strategies import DynShareCheat, HonestProver

    dev = torch.device(args.device or "cuda")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())  # the region compares devices with their index
    Bs, ns, rs = [parse_bytes(b) for b in args.blocks.split(",")], parse_list(args.links), parse_list(args.r)
    pad_block = max(Bs)
    key = dyn_key(args.pad_seed, 0)
    t0 = time.perf_counter()
    g = vllm_geometry(args, dev) if args.geometry == "vllm" else int_geometry(args, dev)
    region = DynamicRegion(g.stores, pad_block, args.pad_seed, device=dev, gpu=0)
    kv_end = g.big.numel()  # the stores; the pad (dynamic filler) after
    if FIELD_BITS == 89 and kv_end % 64:
        raise ValueError("grouped 89-bit dynamic stores need whole 512-byte groups")
    share_words = int(kv_end * args.encode_share)
    if FIELD_BITS == 89:
        share_words = min(kv_end, max(64, -(-share_words // 64) * 64))
    keep = kv_end - share_words
    payload = fill_encoded(g.big, key, args.seed, keep)
    live_bufs = g.live(region)
    if isinstance(live_bufs, PagedLive):
        live = DynamicRegion([], pad_block, args.pad_seed, device=dev, gpu=0, live=True)
        live.rebuild_paged(live_bufs)
    else:
        live = DynamicRegion(live_bufs, pad_block, args.pad_seed, device=dev, gpu=0, live=True)
    setup_s = time.perf_counter() - t0
    keys = Keys.deterministic(args.seed)
    honest, honest_live = HonestProver(region, keys), HonestProver(live, keys)
    enc_share = DynShareCheat(keep, kv_end, payload, key)
    ranges = []
    for name, units, uw in g.kv_units:
        k = min(units, max(1, int(np.ceil(units * args.dram_share))))
        ranges.append((name, units - k, k, uw))
    dram = DramEvict(region, ranges, lambda stage: [g.free(n, f, u) for n, f, u, _ in ranges])
    # its host copy: what the region holds there (random encoded payloads here, not the empty memory it assumes)
    segs = {s.name: s for s in region.segments}
    for name, first, units, uw, hoff in dram.ranges:
        dram.host[hoff : hoff + units * uw].copy_(segs[name].tensor[first * uw : (first + units) * uw])
    dram_tab = region.segtab(*dram.table(g.stage))
    shares = {
        "honest": 0.0,
        f"dyn_encode_otf@{args.encode_share:g}": (kv_end - keep) / region.words,
        f"dyn_dram@{args.dram_share:g}": dram.host_words_read / region.words,
    }
    names = list(shares)
    print(
        f"region {region.words * 8 / 2**30:.2f} GiB in {region.nseg} segments; live region {live.words * 8 / 2**20:.1f} MiB in {live.nseg} "
        f"segments; deviating shares {', '.join(f'{k} {v:.3f}' for k, v in shares.items())}; set up in {setup_s:.0f} s",
        flush=True,
    )

    def answer(name, ch):
        if name.startswith("dyn_dram"):
            region.use_table(dram_tab)
        try:
            honest.share = enc_share if name.startswith("dyn_encode_otf") else None
            return honest.respond(ch)
        finally:
            honest.share = None
            region.restore_table()

    rng = np.random.default_rng(args.seed)
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < args.warm_seconds:
        for B in Bs:
            ch = Challenge.random(B, 4, rs[0], region.num_blocks(B), rng, k=args.k)
            for name in names:
                answer(name, ch)
            honest_live.respond(Challenge.random(B, 4, rs[0], live.num_blocks(B), rng, k=args.k))
    records, done, total = [], 0, len(Bs) * len(rs) * len(ns)
    t_start = time.perf_counter()
    for B in Bs:
        nb, nbl = region.num_blocks(B), live.num_blocks(B)
        for r in rs:
            for n in ns:
                samples = {name: [] for name in names}
                samples["live"] = []
                wrong = {name: 0 for name in names}
                for sample in range(args.warmup + args.samples):
                    ch = Challenge.random(B, n, r, nb, rng, k=args.k)
                    chl = Challenge.random(B, n, r, nbl, rng, k=args.k) if nbl else None
                    order = names + (["live"] if chl is not None else [])
                    rng.shuffle(order)
                    ans = {}
                    for name in order:
                        if name == "live":
                            _, _, _, dt = honest_live.respond(chl)
                        else:
                            lo, hi, _, dt = answer(name, ch)
                            ans[name] = (lo, hi)
                        if sample >= args.warmup:
                            samples[name].append(dt)
                    for name in names[1:]:
                        wrong[name] += ans[name] != ans["honest"]
                for name in names:
                    records.append(
                        {
                            "chain": "main",
                            "prover": name,
                            "B": B,
                            "n": n,
                            "r": r,
                            "correct": wrong[name] == 0,
                            "incorrect_samples": wrong[name],
                            "deviating_share": shares[name],
                            "samples_ns": samples[name],
                        }
                    )
                if samples["live"]:
                    records.append(
                        {
                            "chain": "live",
                            "prover": "honest",
                            "B": B,
                            "n": n,
                            "r": r,
                            "correct": True,
                            "deviating_share": 0.0,
                            "samples_ns": samples["live"],
                        }
                    )
                done += 1
                med = {k: np.median(v) / 1e3 for k, v in samples.items() if v}
                print(
                    f"[{done}/{total} {time.perf_counter() - t_start:5.0f}s] B={B >> 10}K n={n:3d} r={r:4d} | "
                    + " ".join(f"{k}={v:8.1f}us" for k, v in med.items())
                    + ("" if not any(wrong.values()) else f"  WRONG:{[k for k, v in wrong.items() if v]}"),
                    flush=True,
                )
    honest.close()
    honest_live.close()
    return {
        "meta": {
            "kind": "dynamic",
            **g.desc,
            "region_words": region.words,
            "region_segments": region.nseg,
            "live_words": live.words,
            "live_segments": live.nseg,
            "pad_block": pad_block,
            "encode_share": args.encode_share,
            "dram_share": args.dram_share,
            "deviating_shares": shares,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
            "k": args.k,
            "setup_seconds": setup_s,
            "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kernel": honest.kernel,
            "graph": honest.graph,
        },
        "records": records,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--geometry", choices=("vllm", "int"), default="vllm")
    ap.add_argument("--blocks", default="64K,256K,1M,4M")
    ap.add_argument("--links", default="4,8,16,32,64")
    ap.add_argument("--r", default="0,2048")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--warm-seconds", type=float, default=2.0)
    ap.add_argument(
        "--encode-share", type=float, default=0.25, help="dyn_encode_otf: share of the stores held as payload"
    )
    ap.add_argument(
        "--dram-share", type=float, default=0.25, help="dyn_dram: share of every KV store's units read from host memory"
    )
    ap.add_argument("--pool-gib", type=float, default=16.0, help="vllm: pool size (GiB of plaintext pages)")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--page-words", type=int, default=1024)
    ap.add_argument("--act-rows", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=896)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--live-tokens", type=int, default=384, help="vllm: tokens of each live request; int: fed slots")
    ap.add_argument("--int-model", default="models/smollm2-135m.int.safetensors")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--pad-seed", type=lambda s: int(s, 16), default=0x5EED)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = run(args)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
