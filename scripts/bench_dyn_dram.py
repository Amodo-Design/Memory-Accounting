"""What the dyn_dram cheat costs a dynamic chain, per block size (GPU host).

  python scripts/bench_dyn_dram.py [--blocks 64K,256K,1M] [--region 4G] [--layers 24] [--samples 2000] [--live-pages 0]
                                   [--out results/bench_dyn_dram.json]

A dynamic region like the vLLM pool's (--layers KV segments in HBM, the two activation buffers, the pad) is
challenged with the dynamic chain's shape (n = --n links of each block size, r and k as `memory_accounting.verifier.net._dyn_shape`
gives them for the demo's bundle: the manifest's 2048 and 1024, g = 1), through the path the dynamic responder
uses: `DynamicResponder.arm` (freeze, the cheat's segment table for the stage, the honest prover's pre-armed
graph over the region's table) and `answer`, timed as the responder's fire is (parameters and flag written,
spin on the done flag). Three postures, interleaved chain by chain in random order:

  honest  every block read from HBM;
  half    dyn_dram at --fraction (default 0.5): the top half of every KV segment's pages read from pinned host
          memory (mapped, zero-copy through PCIe), the rest, the activation rows and the pad from HBM;
  all     every word of the region read from pinned host memory.

Every chain's digest is checked against the honest one (the cheat's words are the same; only where they are
read from differs). Printed per block size: per-chain and per-link medians and p99, the extra time per link
whose block touches host memory (Δc: from "all", (median - honest median) / n; from "half", the slope of the
chain time over the number of such links, by least squares), and how long one freeze holds inference for one
chain (ARM, --gap-us standing in for the READY's and the challenge's way over the network, the chain, the
release; the reply's way back comes on top, and the live chain adds a second chain). --live-pages takes some of
the evicted pages as live (a busy pool), which splits the table into more segments.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def build(region_bytes: int, layers: int, page_bytes: int, act_bytes: int, max_block: int, device):
    """The region: `layers` KV segments of whole pages, act.h and act.r, the pad to whole blocks of the largest
    size (so every size reads whole blocks of it); every word holds its dynamic filler, as a free pool does."""
    import torch

    from memory_accounting.prover.dynamic import DynamicRegion

    pages = max(1, (region_bytes - 2 * act_bytes) // layers // page_bytes)
    segs = [
        (f"kv.{layer}", torch.empty(pages * page_bytes // 8, dtype=torch.int64, device=device))
        for layer in range(layers)
    ]
    segs += [(n, torch.empty(act_bytes // 8, dtype=torch.int64, device=device)) for n in ("act.h", "act.r")]
    region = DynamicRegion(segs, max_block, pad_seed=0x5EED, device=device)
    for n, _ in segs:
        region.fill(n)
    if region.cuda:
        torch.cuda.synchronize(device)
    return region, pages


def plans(region, layers: int, pages: int, page_words: int, live_pages: int = 0):
    """dram_fn for the responder: the top `fraction` of every KV segment's pages, free but for `live_pages` random
    ones (the same ids in every layer, as live vLLM blocks: read from HBM, each splitting the table); 1.0 means
    every segment of the region, activation rows and pad included."""

    def plan(fraction):
        if fraction >= 1.0:
            ranges = [(s.name, 0, 1, s.words) for s in region.segments]
            return ranges, lambda stage: [np.ones(1, dtype=bool)] * len(ranges)
        k = max(1, int(np.ceil(pages * fraction)))
        free = np.ones(k, dtype=bool)
        free[np.random.default_rng(3).choice(k, size=min(live_pages, k), replace=False)] = False
        ranges = [(f"kv.{layer}", pages - k, k, page_words) for layer in range(layers)]
        return ranges, lambda stage: [free] * layers

    return plan


def host_blocks(region, dram, block_bytes: int) -> np.ndarray:
    """For each block of this size: does any of its words come from the host copy under this table?"""
    tab = region._segtab_np.view(np.uint64)
    n = region.nseg
    starts, ptrs = tab[: n + 1], tab[n + 1 :]
    host = dram.reads_host(ptrs)
    bw = block_bytes // 8
    out = np.zeros(region.words // bw, dtype=bool)
    for i in np.flatnonzero(host):
        lo, hi = int(starts[i]), int(starts[i + 1])
        out[lo // bw : (hi - 1) // bw + 1] = True
    return out


def main():
    ap = argparse.ArgumentParser(
        description="dyn_dram microbenchmark: the dynamic chain from HBM vs part or all of it from host memory"
    )
    ap.add_argument("--blocks", default="64K,256K,1M")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--r", type=int, default=2048)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument(
        "--region",
        default="4G",
        help="dynamic region size (the demo's vLLM pool is tens of GiB; anything well past L2 reads like it)",
    )
    ap.add_argument("--layers", type=int, default=24, help="KV segments (Qwen2.5-0.5B has 24 layers)")
    ap.add_argument("--page", default="8K", help="KV page size (Qwen2.5-0.5B, 16-token blocks, bf16: 8 KiB)")
    ap.add_argument(
        "--act",
        default="2560K",
        help="each activation buffer (2048 rows x 896 x bf16 for Qwen2.5-0.5B at 2048 tokens: 3.5 MiB)",
    )
    ap.add_argument(
        "--fraction", type=float, default=0.5, help="dyn_dram's share of the KV pages (--dyn-evict-fraction)"
    )
    ap.add_argument(
        "--live-pages",
        type=int,
        default=0,
        help="pages of the evicted set taken as live (read from HBM), the same in every layer: "
        "a busy pool, whose table has about 2 x this many segments a layer (each word's table lookup is a binary search)",
    )
    ap.add_argument(
        "--gap-us",
        type=float,
        default=150.0,
        help="wait between the ARM and the chain (the READY to challenge time of a LAN "
        "round trip): the pre-armed graph is waiting on its flag when the chain fires, as in the responder",
    )
    ap.add_argument("--samples", type=int, default=2000, help="chains of each posture per block size")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument(
        "--device",
        default="cuda",
        help="cuda; cpu runs the same steps as a functional check (small --region; the times mean nothing)",
    )
    ap.add_argument("--out", default=None, help="write the samples and the summary here (JSON)")
    args = ap.parse_args()
    import torch

    from memory_accounting.core.params import Challenge, Keys, parse_bytes
    from memory_accounting.prover.dynamic import DynamicResponder, PauseController

    dev = torch.device(args.device)
    if dev.type == "cuda":
        dev = torch.device("cuda", dev.index or 0)  # set_device needs an index
        torch.cuda.set_device(dev)
    else:
        print("cpu: a functional check of the benchmark only (no PCIe: the times mean nothing)", flush=True)
    Bs = [parse_bytes(b) for b in args.blocks.split(",") if b]
    page = parse_bytes(args.page)
    t0 = time.perf_counter()
    region, pages = build(parse_bytes(args.region), args.layers, page, parse_bytes(args.act), max(Bs), dev)
    print(
        f"region {region.nbytes / (1 << 30):.2f} GiB: {args.layers} KV segments of {pages} pages of {page >> 10} KiB, act 2 x {parse_bytes(args.act) >> 10} KiB, "
        f"pad; filler written in {time.perf_counter() - t0:.1f} s",
        flush=True,
    )
    keys = Keys.deterministic(1)
    stage = {"kind": "bench", "requests": [], "in_flight": False}
    dyn = DynamicResponder(
        region,
        lambda: stage,
        PauseController(),
        keys,
        dram_fn=plans(region, args.layers, pages, page // 8, args.live_pages),
    )
    t0 = time.perf_counter()
    half = dyn.prepare_dram(args.fraction)
    full = dyn.prepare_dram(1.0)
    print(
        f"host copies{' (pinned)' if region.cuda else ''}: half {half.words * 8 / (1 << 30):.2f} GiB, all {full.words * 8 / (1 << 30):.2f} GiB, made in {time.perf_counter() - t0:.1f} s",
        flush=True,
    )
    postures = {"honest": None, "half": half, "all": full}
    rng = np.random.default_rng(7)
    summary, raw = {}, {}
    gap_ns = int(args.gap_us * 1e3)

    def one(name, ch, shape):
        """ARM (freeze, table, pre-arm), the timed chain, release: as the responder runs a dynamic verification."""
        dyn.dram, dyn.dram_on = postures[name], postures[name] is not None
        if not dyn.dram_on:
            region.restore_table()
        ta = time.perf_counter_ns()
        dyn.arm(shape, 1.0)
        tb = time.perf_counter_ns()
        while time.perf_counter_ns() - tb < gap_ns:
            pass
        if region.cuda:
            assert dyn.prover.armed_for(ch), "the responder did not pre-arm the chain"
            lo, hi, trace, dt = dyn.prover.fire(ch)
        else:
            lo, hi, trace, dt = dyn.prover.respond(ch)
        dyn.release()
        tc = time.perf_counter_ns()
        return (lo, hi), trace, dt, tb - ta, tc - ta

    for B in Bs:
        nb = region.words * 8 // B
        shape = Challenge(B, args.n, args.r, 0, 0, args.k, 1)
        hb = {}
        for name, d in postures.items():
            if d is not None:
                dyn.dram, dyn.dram_on = d, True
                d.apply(stage)
                hb[name] = host_blocks(region, d, B)
                print(
                    f"   {name}: {region.nseg} segments in the table, {d.host_words_read * 8 / (1 << 30):.2f} GiB read from host memory",
                    flush=True,
                )
        region.restore_table()
        res = {name: {"chain_ns": [], "arm_ns": [], "hold_ns": [], "host_links": []} for name in postures}
        bad = 0
        for s in range(args.warmup + args.samples):
            ch = Challenge(B, args.n, args.r, int(rng.integers(0, 2**63)), int(rng.integers(0, nb)), args.k, 1)
            order = list(postures)
            rng.shuffle(order)
            got = {}
            for name in order:
                dg, trace, dt, arm, hold = one(name, ch, shape)
                got[name] = dg
                if s >= args.warmup:
                    r = res[name]
                    r["chain_ns"].append(dt)
                    r["arm_ns"].append(arm)
                    r["hold_ns"].append(hold)
                    r["host_links"].append(
                        int(hb[name][np.asarray(trace[: args.n], dtype=np.int64) % nb].sum()) if name in hb else 0
                    )
            bad += sum(got[n] != got["honest"] for n in postures)
        region.restore_table()
        dyn.dram, dyn.dram_on = None, False
        honest_ns = np.array(res["honest"]["chain_ns"], dtype=np.float64)
        out = {
            "block_bytes": B,
            "blocks": nb,
            "n": args.n,
            "r": args.r,
            "k": args.k,
            "digest_mismatches": bad,
            "host_blocks": {k: int(v.sum()) for k, v in hb.items()},
        }
        print(
            f"\n== {B >> 10} KiB blocks ({nb} in the region), chains of {args.n} links, r {args.r}, k {args.k}; "
            f"{args.samples} chains each; digests differing from honest: {bad}",
            flush=True,
        )
        print(
            f"   {'posture':8} {'chain p50':>10} {'chain p99':>10} {'link p50':>9} {'link p99':>9} {'host links':>10} {'freeze hold p50':>16} {'p99':>8}   (µs)"
        )
        for name in postures:
            r = res[name]
            c = np.array(r["chain_ns"], dtype=np.float64) / 1e3
            hl = np.array(r["host_links"], dtype=np.float64)
            hold = np.array(r["hold_ns"], dtype=np.float64) / 1e3
            out[name] = {
                "chain_p50_us": _q(c, 50),
                "chain_p99_us": _q(c, 99),
                "link_p50_us": _q(c / args.n, 50),
                "link_p99_us": _q(c / args.n, 99),
                "host_links_mean": float(hl.mean()) if len(hl) else 0.0,
                "arm_p50_us": _q(np.array(r["arm_ns"]) / 1e3, 50),
                "hold_p50_us": _q(hold, 50),
                "hold_p99_us": _q(hold, 99),
            }
            o = out[name]
            print(
                f"   {name:8} {o['chain_p50_us']:10.1f} {o['chain_p99_us']:10.1f} {o['link_p50_us']:9.2f} {o['link_p99_us']:9.2f} "
                f"{o['host_links_mean']:10.2f} {o['hold_p50_us']:16.1f} {o['hold_p99_us']:8.1f}"
            )
        a = np.array(res["all"]["chain_ns"], dtype=np.float64)
        dc_all = (np.median(a) - np.median(honest_ns)) / args.n / 1e3
        x = np.array(res["half"]["host_links"], dtype=np.float64)
        y = np.array(res["half"]["chain_ns"], dtype=np.float64) / 1e3
        dc_half = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 and x.std() > 0 else float("nan")
        shift = np.median(y) - np.median(honest_ns) / 1e3
        out.update(
            {"delta_c_all_us": float(dc_all), "delta_c_half_fit_us": dc_half, "half_median_shift_us": float(shift)}
        )
        print(
            f"   Δc per host-read link: {dc_all:.2f} µs (all, (p50 - honest p50) / n), {dc_half:.2f} µs (half, least-squares slope over "
            f"host links); half's median shift {shift:.1f} µs over a mean {x.mean():.2f} host links a chain",
            flush=True,
        )
        summary[str(B)] = out
        raw[str(B)] = res
    dyn.close()
    if args.out:
        info = {
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "host": platform.node(),
            "args": vars(args),
            "summary": summary,
            "samples": raw,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(info))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
