"""The synthetic UDP prover: answers memory_accounting.core.wire challenges over an array built from seeds (or a setup manifest),
with every cheat strategy on request, for the verifier roles in memory_accounting.verifier.net.

  python -m memory_accounting.prover.udp --port 5555 [--strategies honest,other_gpu,encode_otf,dram_copy,cpu] [--manifest setup/manifest.json --gpu 0]
  python -m memory_accounting.prover.udp --device cpu --size 64M --dynamic [--dyn-model int:<file>]      # no GPU: protocol and UI tests

The prover process can switch strategy on request from the verifier, so one prover process serves a whole
honest-vs-cheat comparison. With --dynamic a model generates on a background thread and its KV cache and
activation buffer are the dynamic region.
"""

from __future__ import annotations

import argparse
import socket
import struct

from ..core.backend import pick_device
from ..core.params import Challenge, array_from_args, parse_bytes
from ..core.wire import (
    ARM,
    ARM_DYN_MAGIC,
    ARM_MAGIC,
    CH,
    DYN_LIVE_MAGIC,
    DYN_MAGIC,
    LIVE_FLAG,
    MAGIC,
    POSTURE,
    POSTURE_MAGIC,
    RE,
    READY,
    READY_DYN,
    STATS,
    STATS_MAGIC,
    STATSRE,
    STRATS,
    resolve_peers,
    strategy_mask,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="synthetic UDP prover: every strategy, on request")
    _add_array_args(ap.add_argument_group("the challenged array"))
    _add_serving_args(ap.add_argument_group("serving"))
    _add_dynamic_args(ap.add_argument_group("dynamic memory (--dynamic)"))
    return ap


def _add_array_args(g) -> None:
    g.add_argument("--manifest", default=None, help="setup-phase manifest (real weights); omit for a synthetic array")
    g.add_argument("--gpu", type=int, default=0, help="which GPU's copy this prover serves")
    g.add_argument("--size", default="1G")
    g.add_argument("--encoding", default="vde", choices=["vde", "none"])
    g.add_argument("--no-perm", action="store_true")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", default=None)
    g.add_argument("--canon-device", default=None, help="device holding the other GPU's copy for the other_gpu cheat")


def _add_serving_args(g) -> None:
    g.add_argument("--bind", default="0.0.0.0")
    g.add_argument("--port", type=int, default=5555)
    g.add_argument(
        "--verifier-addr",
        action="append",
        default=None,
        help="accept datagrams only from this address (IP or host "
        "name, resolved once at start; repeat it or give a comma list). Default: from anyone who reaches the port",
    )
    g.add_argument("--strategies", default=",".join(STRATS))


def _add_dynamic_args(g) -> None:
    g.add_argument(
        "--dynamic", action="store_true", help="serve a dynamic region (a model generating on the prover's device)"
    )
    g.add_argument("--dyn-model", default="toy", help="toy | int:<integer model file>")
    g.add_argument("--dyn-block", default="64K", help="block size the region is padded to")
    g.add_argument(
        "--dyn-evict-fraction",
        type=float,
        default=0.5,
        help="posture dyn_dram: share of every KV buffer's slots (the top ones) read from a host copy while free",
    )
    g.add_argument("--pad-seed", default=None, help="hex seed of the region's filler pad (default 5eed)")
    g.add_argument(
        "--toy-len",
        type=int,
        default=None,
        help="KV cache length (slots) of the toy (default 64) or integer model (default 512, "
        "as intserve); the verifier's --replay assumes the same defaults",
    )
    g.add_argument(
        "--toy-step-ms",
        type=float,
        default=2.0,
        help="pause between decode steps (keeps a step in flight most of the time)",
    )
    g.add_argument(
        "--evidence-bytes",
        action="store_true",
        help="attach the region's bytes to each record (debugging a mismatch; the verdict never uses them)",
    )


def _int_prompts(model):
    """Prompts for the integer model's demo load: tokenised with the source model's tokenizer when it is
    available locally, else a few fixed token sequences."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model.cfg["source"])
        texts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Memory accounting checks what sits in a GPU's memory by",
            "Once upon a time",
        ]
        return [tok(t)["input_ids"] for t in texts]
    except Exception:
        return [[1, 2, 3, 4, 5], [7, 11, 13], [2, 4, 6, 8, 10, 12]]


def prover_main(args):
    spec, keys = array_from_args(args)
    from .strategies import PROVERS

    dev = pick_device(args.device)
    if dev == "cpu":
        # no GPU: a CPU reference prover over the synthetic array (protocol and UI tests on a laptop); every
        # strategy id is answered by the CPU chain, so timings mean nothing here
        if args.manifest:
            raise SystemExit("the CPU-only prover serves the synthetic array only")
        from types import SimpleNamespace

        from ..core.memory import cpu_synthetic_theta

        hbm = SimpleNamespace(cfg=spec, theta_host_np=cpu_synthetic_theta(spec, args.seed), nseg=0)
        provers = {s: PROVERS["cpu"](hbm, keys) for s in STRATS}
        devname = "cpu reference"
    else:
        import torch

        from ..core.backend import DeviceBackend
        from ..core.memory import Hbm

        be = DeviceBackend(dev)
        if str(be.device).startswith("cuda"):  # kernels launch on the current device: make it this one
            torch.cuda.set_device(torch.device(be.device).index or 0)
        if args.manifest:
            hbm = Hbm.from_manifest(be, spec, args.gpu, canon_device=args.canon_device)
            avail = [s for s in STRATS if s != "encode_otf"]  # the prover holds no raw payload
        else:
            hbm = Hbm(be, spec, data_seed=args.seed, canon_device=args.canon_device)
            avail = STRATS  # the seed the CPU paths use
        wanted = [s for s in args.strategies.split(",") if s in avail]
        if "honest" not in wanted:
            wanted.insert(0, "honest")
        provers = {}
        for s in wanted:
            try:
                provers[s] = PROVERS[s](hbm, keys)
            except (RuntimeError, ValueError) as e:  # e.g. lora and the posture cheats need the responder's copies
                if s == "honest":
                    raise
                print(f"prover: {s} not available here: {e}", flush=True)
        devname = str(be.device)
    served = 0
    posture = None
    # dynamic memory demo: a model generates on a background thread (the toy, or the integer-exact model);
    # its KV cache + activation buffer is the dynamic region (on the prover's device), answered through
    # the segment table
    dyn = None
    toy = None
    if args.dynamic:
        from ..models.toy import InferenceLoop, int_dynamic, static_dram_fn, static_live_fn, toy_dynamic
        from .dynamic import DynamicResponder

        if args.dyn_model == "toy":
            model, region, tracker, pause, _ = toy_dynamic(
                dev,
                parse_bytes(args.dyn_block),
                int(args.pad_seed or "5eed", 16),
                max_len=args.toy_len or 64,
                gpu=args.gpu,
            )
            prompts, stop = None, ()
        elif args.dyn_model.startswith("int:"):
            model, region, tracker, pause, _ = int_dynamic(
                args.dyn_model[4:],
                dev,
                parse_bytes(args.dyn_block),
                int(args.pad_seed or "5eed", 16),
                max_len=args.toy_len or 512,
                gpu=args.gpu,
            )
            prompts, stop = _int_prompts(model), ()
        else:
            raise SystemExit(f"--dyn-model {args.dyn_model}: toy or int:<file>")
        dyn = DynamicResponder(
            region,
            tracker,
            pause,
            keys,
            model_info=model.info,
            include_bytes=args.evidence_bytes,
            live_fn=static_live_fn(model, region),
            dram_fn=static_dram_fn(model, region),
        )
        # the dyn_dram posture's share (its host copy is made at the switch)
        dyn.dram_fraction = args.dyn_evict_fraction
        dyn.serve_evidence(args.bind, args.port)
        toy = InferenceLoop(
            model, tracker, pause, step_sleep=args.toy_step_ms / 1000, prompts=prompts, stop_ids=stop, region=region
        )
        toy.start()
    try:
        peers = resolve_peers(args.verifier_addr)
    except ValueError as e:
        raise SystemExit(f"--verifier-addr: {e}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    if peers:
        print(
            f"prover: datagrams only from {', '.join(sorted(peers))} (--verifier-addr); others are dropped", flush=True
        )
    print(
        f"prover ready on {args.bind}:{args.port} ({devname}, {hbm.cfg.nbytes >> 20} MiB, {hbm.cfg.encoding}"
        f"{', manifest ' + args.manifest if args.manifest else ''}"
        + (
            f"; dynamic region {dyn.region.nbytes >> 10} KiB in {dyn.region.nseg} segments, {args.dyn_model} model generating, records over TCP"
            if dyn
            else ""
        )
        + ")",
        flush=True,
    )
    while True:
        try:
            data, addr = sock.recvfrom(512)
        except KeyboardInterrupt:
            print("\nprover stopping", flush=True)
            for p in provers.values():
                p.close()
            if dyn is not None:
                toy.stop()
                dyn.close()
            return
        if peers is not None and addr[0] not in peers:  # not the verifier: no ARM, posture, counters or answer
            continue
        if len(data) == ARM.size:  # nothing to drain here: ready at once
            magic, token, B_, n_, r_, k_, g_, _sid = ARM.unpack(data)
            if magic == ARM_MAGIC:
                if dyn is not None:
                    dyn.release()
                sock.sendto(READY.pack(ARM_MAGIC, token), addr)
            elif magic == ARM_DYN_MAGIC and dyn is not None:  # pause the toy at its next layer boundary
                try:
                    shape = Challenge(B_, n_, r_, 0, 0, k_, g_)
                except ValueError as e:
                    print(f"bad dynamic shape: {e}", flush=True)
                    continue
                try:
                    words, bb, flags, lw = dyn.arm(shape, 0.5, want_live=bool(_sid & LIVE_FLAG))
                except (
                    RuntimeError
                ) as e:  # ArmFailed: the region is unfrozen again; no READY, the verifier logs a fault
                    print(f"dynamic ARM failed: {e}", flush=True)
                    continue
                sock.sendto(READY_DYN.pack(ARM_DYN_MAGIC, token, words, bb, flags, lw), addr)
            continue
        if len(data) == STATS.size:
            magic, token = STATS.unpack(data)
            if magic == STATS_MAGIC:
                sock.sendto(
                    STATSRE.pack(
                        STATS_MAGIC,
                        token,
                        served,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        STRATS.index(posture) if posture else 0,
                        strategy_mask(provers),
                        strategy_mask(provers),
                    ),
                    addr,
                )
            continue
        if len(data) == POSTURE.size:
            magic, token, sid = POSTURE.unpack(data)
            if magic == POSTURE_MAGIC:
                if 0 <= sid < len(STRATS) and STRATS[sid] in provers:
                    if dyn is not None:
                        dyn.release()
                        try:
                            # dyn_dram: the dynamic chain reads free memory from host memory
                            dyn.set_posture(STRATS[sid])
                        except ValueError as e:
                            print(f"prover: {e}", flush=True)
                    posture = STRATS[sid]
                sock.sendto(POSTURE.pack(POSTURE_MAGIC, token, STRATS.index(posture) if posture else 0), addr)
            continue
        try:
            magic, nonce, j0, B, n, r, k, g, sid = CH.unpack(data)
            if magic in (DYN_MAGIC, DYN_LIVE_MAGIC) and dyn is not None:
                live = magic == DYN_LIVE_MAGIC
                ch = Challenge(B, n, r, nonce, j0, k, g)
                lo, hi = dyn.answer(ch, live=live)
                sock.sendto(RE.pack(nonce, lo, hi), addr)
                served += 1
                dyn.finish(ch, lo, hi, live=live)  # off the timing path: record; resume after the last chain
                continue
            if posture is not None and 0 <= sid < len(STRATS):
                sid = STRATS.index(posture)
            if magic != MAGIC or not 0 <= sid < len(STRATS) or STRATS[sid] not in provers:
                raise ValueError("bad magic or strategy id")
            ch = Challenge(B, n, r, nonce, j0, k, g)
            lo, hi, _, _ = provers[STRATS[sid]].respond(ch)
        except (struct.error, ValueError, KeyError, RuntimeError) as e:  # malformed datagram: log, keep serving
            print(f"ignored datagram from {addr}: {e}", flush=True)
            if dyn is not None:
                dyn.release()
            continue
        sock.sendto(RE.pack(nonce, lo, hi), addr)
        served += 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        prover_main(args)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
