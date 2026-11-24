"""Sweep runner: measures RTT distributions for the honest prover and each cheat over (B, n, r).

Provers are interleaved sample by sample so thermal/clock drift affects all strategies equally.
Every config is verified once against the verifier's reference digest.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..core.backend import DeviceBackend, pick_device
from ..core.codec_profile import FIELD_BITS
from ..core.memory import Hbm
from ..core.params import Challenge, Keys, MemoryConfig
from ..prover.strategies import PROVERS
from ..verifier.recompute import Verifier
from .load import BackgroundLoad


def parse_list(s: str, cast=int) -> list:
    return [cast(x) for x in s.split(",") if x]


def parse_bytes(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def run(args) -> dict:
    dev = pick_device(args.device)
    be = DeviceBackend(dev)
    cfg = MemoryConfig(words=parse_bytes(args.size) // 8, encoding=args.encoding,
                       perm=not args.no_perm and FIELD_BITS != 89)
    hbm = Hbm(be, cfg, pattern=args.pattern, canon_device=args.canon_device)
    keys = Keys.deterministic(args.seed)
    ver = Verifier(hbm.theta_host_np, cfg, keys)
    decode_gbps = hbm.decode_benchmark() if cfg.enc else float("nan")
    names = ["honest"] + [p for p in args.provers.split(",") if p and p != "honest"]
    popts = {} if args.kernel == "auto" else {"kernel": args.kernel}
    if args.no_graph:
        popts["graph"] = False

    shared = {}

    class AtFraction:
        """'name@f': the cheat on a share f of the array, answered by one instance of the cheat shared by every f
        (partial; see PartialProver.use_fraction; the other cheats: AtShare)."""

        def __init__(self, n: str, f: float):
            if n not in shared:
                shared[n] = PROVERS[n](hbm, keys, fraction=f, **popts)
            self.p, self.f = shared[n], f
            self.p.use_fraction(f)
            self.keep = self.p.keep

        def respond(self, ch):
            self.p.use_fraction(self.f)
            return self.p.respond(ch)

        def host_links(self, trace, block_bytes):
            self.p.use_fraction(self.f)
            return self.p.host_links(trace, block_bytes)

        def close(self):
            pass

    class AtShare:
        """'name@f' for the other cheats (encode_otf, other_gpu, dram_copy, cpu; memory_accounting.prover.strategies.ShareCheat),
        and 'honest' when any of them is asked for: one honest instance answers all of them, its share set per
        challenge."""

        def __init__(self, n: str, f: float):
            from ..prover.strategies import ShareCheat

            if "share" not in shared:
                shared["share"] = PROVERS["honest"](hbm, keys, **popts)
            if n == "cpu" and "cpu" not in shared:
                shared["cpu"] = PROVERS["cpu"](hbm, keys)
            self.p = shared["share"]
            self.sh = None if n == "honest" else ShareCheat(n, hbm, f, cpu=shared.get("cpu"))
            self.keep = self.sh.keep if self.sh else None

        def __getattr__(self, name):  # the run's record reads honest's kernel, graph and groups
            return getattr(self.p, name)

        def respond(self, ch):
            self.p.share = self.sh
            return self.p.respond(ch)

        def host_links(self, trace, block_bytes):
            first = self.keep * 8 // block_bytes
            return sum(int(j) >= first for j in trace)

        def close(self):
            pass

    from ..prover.strategies import ShareCheat

    any_share = any(s.partition("@")[0] in ShareCheat.KINDS and s.partition("@")[2] for s in names)

    def make(spec: str):
        n, _, f = spec.partition("@")
        if f:
            return AtShare(n, float(f)) if n in ShareCheat.KINDS else AtFraction(n, float(f))
        if n == "honest" and any_share:
            return AtShare(n, 0)
        return PROVERS[n](hbm, keys) if n == "cpu" else PROVERS[n](hbm, keys, **popts)

    provers = {n: make(n) for n in names}
    load = None
    if args.background_load:
        load = BackgroundLoad(
            dev, matmul_dim=args.load_dim, bandwidth_bytes=args.load_buffer_mb << 20, fraction=args.load_fraction
        )
        load.start()
    rng = np.random.default_rng(args.seed)
    # warm every prover for ~2 s (kernel compile, graph capture, pool wake-up, CPU frequency ramp, page
    # touch of the host copy); the first sampled configuration otherwise carries a start-up transient
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < args.warm_seconds:
        for name in names:
            for Bw in {parse_bytes(b) for b in args.blocks.split(",")}:
                provers[name].respond(Challenge.random(Bw, 4, parse_list(args.r)[0], hbm.num_blocks(Bw), rng, k=args.k))
    records = []
    Bs, ns, rs = [parse_bytes(b) for b in args.blocks.split(",")], parse_list(args.links), parse_list(args.r)
    total = len(Bs) * len(ns) * len(rs)
    done = 0
    partial = False
    t_start = time.perf_counter()
    try:
        for B in Bs:
            nb = hbm.num_blocks(B)
            for r in rs:
                for n in ns:
                    samples = {name: [] for name in names}
                    hosted = {name: [] for name in names if getattr(provers[name], "keep", None) is not None}
                    wrong = {name: 0 for name in names}
                    for sample in range(args.warmup + args.samples):
                        ch = Challenge.random(B, n, r, nb, rng, k=args.k)
                        order = list(names)
                        rng.shuffle(order)
                        answers = {}
                        for name in order:
                            lo, hi, trace, dt = provers[name].respond(ch)
                            answers[name] = (lo, hi)
                            if sample >= args.warmup:
                                samples[name].append(dt)
                                if name in hosted:
                                    hosted[name].append(provers[name].host_links(trace[:n], B))
                        # verify every sample, after all provers answered (outside every timed window and
                        # without warming the CPU cheat's caches with the verifier's own read of the blocks)
                        if sample >= args.warmup or args.verify_warmup:
                            exp = ver.expected(ch)[:2]
                            for name in names:
                                if answers[name] != exp:
                                    wrong[name] += 1
                    for name in names:
                        records.append(
                            {
                                "prover": name,
                                "B": B,
                                "n": n,
                                "r": r,
                                "correct": wrong[name] == 0,
                                "incorrect_samples": wrong[name],
                                "samples_ns": samples[name],
                                **({"host_links": hosted[name]} if name in hosted else {}),
                            }
                        )
                    ok = {name: wrong[name] == 0 for name in names}
                    done += 1
                    med = {name: np.median(samples[name]) / 1e3 for name in names}
                    el = time.perf_counter() - t_start
                    print(
                        f"[{done}/{total} {el:5.0f}s] B={B >> 10}K n={n:3d} r={r:3d} | "
                        + " ".join(f"{k}={v:8.0f}us" for k, v in med.items())
                        + ("" if all(ok.values()) else f"  WRONG:{[k for k, v in ok.items() if not v]}")
                        + ("  LOAD NOT RUNNING" if load is not None and not load.alive else ""),
                        flush=True,
                    )
    except KeyboardInterrupt:
        partial = True
        print(
            f"\ninterrupted after {done}/{total} configs; stopping background threads and saving what completed",
            flush=True,
        )
    finally:
        # Stop the pool and the load thread before the interpreter tears CUDA down under them.
        for p in list(provers.values()) + list(shared.values()):
            p.close()
        load_report = load.stop() if load else None
    if load_report and load_report.get("error"):
        print(
            f"WARNING: the background load did not run through the sweep ({load_report['error']}); "
            f"meta.background_load.error says so",
            file=sys.stderr,
            flush=True,
        )
    elif load_report:
        print(
            f"background load: {load_report['tflops']:.0f} TFLOP/s matmul + {load_report['bandwidth_gbps']:.0f} GB/s streaming over {load_report['seconds']:.0f} s"
        )
    return {
        "meta": {
            "background_load": load_report,
            "canon_device": hbm.canon_device,
            "partial": partial,
            "device": dev,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "size_bytes": cfg.nbytes,
            "encoding": cfg.encoding,
            "perm": cfg.perm,
            "pattern": args.pattern,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
            "setup_seconds": hbm.setup_seconds,
            "decode_gbps": decode_gbps,
            "cpu_threads": getattr(provers.get("cpu"), "nthreads", None),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kernel": provers["honest"].kernel,
            "graph": provers["honest"].graph,
            "groups": provers["honest"].groups,
            "k": args.k,
        },
        "records": records,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="PoSM prototype sweep")
    ap.add_argument("--size", default="1G", help="challenged array size, e.g. 1G, 512M (power of two)")
    ap.add_argument("--encoding", default="vde", choices=["vde", "none"])
    ap.add_argument("--no-perm", action="store_true", help="disable the per-GPU Feistel permutation")
    ap.add_argument("--pattern", default="random", choices=["random", "zeros"])
    ap.add_argument("--blocks", default="16K,64K,256K,1M")
    ap.add_argument("--links", default="1,2,4,8,16,32,64")
    ap.add_argument(
        "--r",
        default="0,256,2048",
        help="matmul widths, multiples of 16 up to 4096 (0 = hash only); CPU cost per byte = r",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=1024,
        help="matmul inner dimension in bytes per row (128..2048); hashed output = bytes*r/k",
    )
    ap.add_argument(
        "--provers",
        default="other_gpu,encode_otf,dram_copy,cpu",
        help="comma list; name@f runs the cheat on a share f of the array (partial@0.01 = 1%% read from host memory; "
        "encode_otf, other_gpu, dram_copy, cpu: one honest instance answers them and honest)",
    )
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify-warmup", action="store_true", help="also verify the warm-up samples")
    ap.add_argument(
        "--warm-seconds", type=float, default=2.0, help="exercise every prover for this long before sampling"
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--kernel",
        default="auto",
        choices=["auto", "tensor", "multi", "persistent"],
        help="chain kernel (auto = tensor)",
    )
    ap.add_argument("--no-graph", action="store_true", help="launch kernels directly instead of replaying a CUDA graph")
    ap.add_argument(
        "--background-load",
        action="store_true",
        help="keep the GPU busy with large matmuls and streaming passes on a low-priority stream during the sweep",
    )
    ap.add_argument("--load-fraction", type=float, default=1.0, help="background load duty cycle, 0-1 (1 = saturated)")
    ap.add_argument(
        "--load-dim", type=int, default=8192, help="background matmul size (smaller = finer preemption granularity)"
    )
    ap.add_argument("--load-buffer-mb", type=int, default=1024, help="background streaming buffer in MiB")
    ap.add_argument(
        "--canon-device",
        default=None,
        help="device holding the cheater's canonical copy, e.g. cuda:1 for a real second-GPU cheat (default: same GPU)",
    )
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args(argv)
    res = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
