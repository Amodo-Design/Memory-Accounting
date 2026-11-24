"""The verifier over UDP: end-to-end round trips over a real socket, against any prover that speaks memory_accounting.core.wire.

  python -m memory_accounting.verifier.net verifier --host 127.0.0.1 --port 5555 --B 64K --n 32 --r 128 --samples 200 --out results/udp.json
  python -m memory_accounting.verifier.net baseline --host <prover> ...      # an honest prover's round trips -> baseline.json
  python -m memory_accounting.verifier.net live     --host <prover> ...      # verify continuously, with the live page

With a synthetic array (no --manifest) the verifier builds the same array from the same seeds as the synthetic
prover (memory_accounting.prover.udp), so it can recompute the expected digests without a setup phase.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import secrets
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..core.backend import default_threads, pick_device
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
)
from .recompute import Verifier

_SYSRAND = secrets.SystemRandom()


def _gap(mean_s: float) -> float:
    """The wait before the next verification: exponential with this mean, from the CSPRNG. Memoryless, so
    the time since the last challenge tells the prover nothing about when the next one comes."""
    return _SYSRAND.expovariate(1.0 / mean_s) if mean_s > 0 else 0.0


# torch, the CUDA backend, Hbm and the provers are imported inside the GPU paths only: a verifier box needs none of them

from ..core.codec_profile import FIELD_BITS, PACKED128
# Five 90-bit encoded values occupy eight physical words in the p89 layout.
# Round down for the verifier's integer credit-per-word interface.
VDE_WORD_BITS = (11 * 90 // 16) if PACKED128 else (5 * 90 // 8) if FIELD_BITS == 89 else 62


def credited_static_bytes(cfg, spec=None, theta_words: int | None = None) -> int:
    """What the memory claim credits the static array with: its incompressible content, not its size. Two
    bits of every VDE word are always zero, so a prover can pack Θ' at 62 bits a word for free (shifts, no
    re-encoding) and keep 2/64 of the array for itself; only the remaining 62/64 is memory the chains force.
    A stream-decode bundle's Θ' region is not credited at all: without the weights' key the verifier cannot
    check that it encodes the weights (only an audit can), so only the filler it regenerates itself counts
    (theta_words: Θ''s size, the bundle image's length).
    Under --encoding none (trusted weights, no incompressibility claim) the array is credited as it stands."""
    from ..setup.manifest import BENCHMARK, Manifest

    words = cfg.words
    if isinstance(spec, Manifest):
        from ..setup.verify import credited_static_words

        if spec.kind == BENCHMARK:
            # Keep the usual size accounting diagnostics for throughput runs. These
            # sizes are declared by the prover and do not constitute verified memory.
            words = max(0, spec.accounted_words - spec.payload_words)
        else:
            words = credited_static_words(spec, theta_words)
    return words * 8 * VDE_WORD_BITS // 64 if cfg.encoding == "vde" else words * 8


def _static_credit(args, cfg) -> int:
    """The static array's credit in this run: `credited_static_bytes`, or 0 when the verifier recomputes chains
    over the image the prover's own setup wrote (--trust-prover-image): nothing ties that file to the weights
    and filler the setup was to hold."""
    if getattr(args, "_prover_image", False):
        return 0
    if getattr(args, "_registered_stream", False) and not getattr(args, "skip_response_checks", False):
        return credited_static_bytes(cfg)
    return credited_static_bytes(cfg, getattr(args, "_spec", None), getattr(args, "_theta_words", None))


def _is_bundle(args) -> bool:
    spec = getattr(args, "_spec", None)
    return bool(spec is not None and hasattr(spec, "is_bundle") and spec.is_bundle()
                and (not getattr(args, "_registered_stream", False) or getattr(args, "skip_response_checks", False)))


def _static_credit_note(bundle: bool) -> str:
    """Why the static array is credited less than its size (`credited_static_bytes`)."""
    return (
        "the bundle's Θ' is challenged but not credited, since without the weights' key the verifier cannot check that it encodes "
        f"the weights; its filler at {VDE_WORD_BITS}/64"
        if bundle
        else f"{VDE_WORD_BITS}/64 after accounting for the fixed bits in the encoding"
    )


def _verifier_seeds(args, spec) -> tuple[int | None, int | None]:
    """The filler seed and the dynamic pad seed as the verifier knows them, never as the prover's manifest or
    records say. Without --filler-seed the manifest's is used, with a warning (the check then binds nothing)."""
    fs = int(args.filler_seed, 16) if getattr(args, "filler_seed", None) else None
    if fs is None and args.manifest:
        fs = int(spec.filler_seed)
        print(
            "verifier: WARNING: no --filler-seed; trusting the manifest's filler seed. Pass the seed you minted.",
            flush=True,
        )
    pad = int(args.pad_seed, 16) if getattr(args, "pad_seed", None) else (fs if fs is not None else 0x5EED)
    args._dyn_expected = (pad, args.gpu)
    return fs, pad


def _make_verifier(args, spec, keys):
    """With a manifest: the verifier holds the one-way image T = H(Theta') and recomputes chains over it
    (image=False: T is the array). Without: the synthetic array is rebuilt on the device."""
    filler_seed, pad_seed = _verifier_seeds(args, spec)
    args._spec = spec  # the memory claim credits a bundle's array differently
    registered = tuple(getattr(args, name, None) for name in
                       ("weight_reference", "weight_registration", "weight_seed_receipt"))
    if any(registered) and (not all(registered) or not args.manifest or not spec.is_bundle() or args.state
                            or not args.filler_seed):
        raise SystemExit("registered stream verification requires a bundle, all three weight options and the verifier's filler seed")
    if getattr(args, "skip_response_checks", False) and not any(registered):
        from ..setup.manifest import BENCHMARK

        if not args.manifest or spec.kind != BENCHMARK:
            raise SystemExit("--skip-response-checks requires a throughput-only layout from the benchmark")
        if spec.filler_seed != filler_seed or spec.hash_key != keys.as_list():
            raise SystemExit("benchmark layout and verifier disagree on the filler seed or challenge keys")
        return None, spec.memory_config(args.gpu, image=False)
    if args.manifest:
        from ..setup.encode import load_array
        from ..setup.verify import manifest_problems, verify_bundle

        g = spec.gpu(args.gpu)
        path = Path(args.state) / f"image_gpu{args.gpu}.npy" if args.state else Path(g.image_file)
        if not path.exists():
            raise SystemExit(
                f"no image for GPU {args.gpu} at {path}; run `memory_accounting.setup.cli verify-registered --out <dir>` and pass --state <dir>"
            )
        bundle = spec.is_bundle()  # stream-decode bundle: no weights key, a committed image
        if bundle and args.state:
            raise SystemExit(
                "--state is for setup manifests (the verifier's own recomputed image); a bundle's image is checked against its commitment"
            )
        if not args.state and not bundle:  # a bundle's image is checked against its commitment below
            # the image the prover's own setup wrote: chains over it show only that the prover answers from that
            # file, not that the file encodes the weights and filler the setup was to hold, so nothing of it is credited
            if not getattr(args, "trust_prover_image", False):
                raise SystemExit(
                    f"verifier: a setup manifest needs the verifier's own image: run `memory_accounting.setup.cli verify-registered --out <dir>` and pass "
                    f"--state <dir>. The image at {path} is the one the prover's setup wrote (--trust-prover-image to use it "
                    "anyway; no static memory is then credited)"
                )
            if getattr(args, "max_scratch", None):
                raise SystemExit(
                    "verifier: --max-scratch with --trust-prover-image: the static array is not credited, so every "
                    "verification would leave all of it uncredited (pass --state with the verifier's own image)"
                )
            args._prover_image = True
            print(
                f"verifier: WARNING: using the image the prover's setup wrote ({path}) (--trust-prover-image): the static array is "
                "credited 0 bytes; for a registered trusted image run `memory_accounting.setup.cli verify-registered` and pass --state",
                flush=True,
            )
        if bundle:
            problems = verify_bundle(spec, args.gpu, filler_seed, image_path=path, pad_seed=pad_seed)
        else:
            problems = manifest_problems(spec, filler_seed, pad_seed)
        if problems:
            raise SystemExit("verifier: the setup is not acceptable:\n  " + "\n  ".join(problems))
        cfg = spec.memory_config(args.gpu, image=False)
        img = np.asarray(load_array(path)).view(np.uint64)
        if all(registered):
            from ..setup.stream_reference import verify_stream_weights

            try:
                verify_stream_weights(spec, args.gpu, img, *registered, args.verify_threads or default_threads())
            except (OSError, ValueError) as exc:
                raise SystemExit(f"verifier: registered stream setup rejected: {exc}") from exc
            args._registered_stream = True
            print("verifier: recomputed registered stream weights and verified the image and encoded commitment", flush=True)
            if getattr(args, "skip_response_checks", False):
                args._theta_words = len(img)
                return None, cfg
        if bundle:
            # the bundle's image is Θ''s, and its length is Θ''s size (verify_bundle); every word past it is filler the
            # verifier regenerates from its own seed under the key it derives for this GPU, never from the prover's
            # image or manifest. An image as long as the array is accepted, and credited nothing (credited_static_words)
            from ..core.backend import CpuLib
            from ..setup.filler import static_filler_key
            from .recompute import ImageWithFiller

            args._theta_words = len(img)
            img = ImageWithFiller(
                img[: min(len(img), spec.accounted_words)],
                spec.accounted_words,
                filler_seed,
                static_filler_key(filler_seed, args.gpu),
                CpuLib(),
                threads=args.verify_threads or default_threads(),
            )
        elif len(img) < spec.accounted_words:
            raise SystemExit(f"image covers {len(img)} of {spec.accounted_words} words")
        return Verifier(img, cfg, keys, threads=args.verify_threads), cfg
    if pick_device(args.device) == "cpu":  # no GPU on the verifier: synthetic array via the C library
        from ..core.memory import cpu_synthetic_theta

        return Verifier(cpu_synthetic_theta(spec, args.seed), spec, keys, threads=args.verify_threads), spec
    import torch

    from ..core.backend import DeviceBackend
    from ..core.memory import Hbm

    be = DeviceBackend(pick_device(args.device))
    if str(be.device).startswith("cuda"):  # kernels launch on the current device: make it this one
        torch.cuda.set_device(torch.device(be.device).index or 0)
    hbm = Hbm(be, spec, data_seed=args.seed)  # the prover's array: the same seed on every path
    return Verifier(hbm.theta_host_np, hbm.cfg, keys, threads=args.verify_threads), hbm.cfg


SHAPE_DEFAULTS = {"B": "1M", "r": 2048, "k": 1024}  # synthetic arrays: the verifier's shape when no flag sets it


def shape_problems(B: int, r: int, k: int, min_r: int, allowed_B, allowed_k) -> list[str]:
    """Why a challenge shape falls short of the verifier's policy (empty: it meets it). The shape decides
    what a chain can tell apart: at small r the matmul is too light for a host CPU or a PCIe read to fall
    behind (r = 0 is hash-only, which cannot tell HBM from host memory), so the policy is the verifier's,
    never the prover's manifest. Defaults: the measured recommendation, 256 KB to 1 MB blocks at r = 2048,
    k = 1024."""
    out = []
    if r < min_r:
        out.append(
            f"r = {r} is below the verifier's minimum {min_r} (--min-r): a light matmul does not tell HBM from host memory"
        )
    if allowed_B and B not in allowed_B:
        out.append(
            f"block size {B} is not one the verifier allows ({', '.join(str(b) for b in sorted(allowed_B))}; --allowed-B)"
        )
    if allowed_k and k not in allowed_k:
        out.append(
            f"k = {k} is not one the verifier allows ({', '.join(str(x) for x in sorted(allowed_k))}; --allowed-k)"
        )
    return out


def _shape(args, spec):
    """Challenge shape. With a manifest, the manifest's shape (its image and commitment are per block), held
    to the verifier's policy (`shape_problems`): a shape below it, or a --B/--r/--k on the command line that
    differs from the manifest's, is refused, never silently replaced. Without one (synthetic arrays, the
    research harness) the command line's, the verifier's own choice."""
    if args.manifest:
        B, r, k = spec.block_bytes, spec.r, spec.k
        given = {
            "--B": parse_bytes(args.B) if getattr(args, "B", None) else None,
            "--r": getattr(args, "r", None),
            "--k": getattr(args, "k", None),
        }
        mine = {"--B": B, "--r": r, "--k": k}
        diff = [f"{f} {v} (the manifest's is {mine[f]})" for f, v in given.items() if v is not None and v != mine[f]]
        if diff:
            raise SystemExit(
                "verifier: the challenge shape on the command line differs from the manifest's: "
                + "; ".join(diff)
                + ". Drop the flag to use the manifest's shape, or set the prover up again at this shape"
            )
        problems = shape_problems(
            B,
            r,
            k,
            getattr(args, "min_r", SHAPE_DEFAULTS["r"]),
            _byte_set(getattr(args, "allowed_B", None)),
            _int_set(getattr(args, "allowed_k", None)),
        )
        if problems:
            raise SystemExit(
                f"verifier: the manifest's challenge shape (B {B}, r {r}, k {k}) is below the verifier's policy:\n  "
                + "\n  ".join(problems)
                + "\nSet the prover up again at a stronger shape (or relax the policy flags knowingly)"
            )
        return B, args.n, r, k, 1

    def pick(name):
        return getattr(args, name, None) if getattr(args, name, None) is not None else SHAPE_DEFAULTS[name]

    return parse_bytes(str(pick("B"))), args.n, int(pick("r")), int(pick("k")), 1


def _byte_set(s) -> set[int]:
    return {parse_bytes(x.strip()) for x in str(s).split(",") if x.strip()} if s else set()


def _int_set(s) -> set[int]:
    return {int(x) for x in str(s).split(",") if x.strip()} if s else set()


SO_TIMESTAMPING = getattr(socket, "SO_TIMESTAMPING", 37)
_TS_FLAGS = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 11)  # TX_SOFTWARE | RX_SOFTWARE | SOFTWARE | OPT_TSONLY
KERNEL_RTT: dict[str, deque] = {}  # --kernel-ts: kind -> [ns], driver TX to driver RX (the last KERNEL_RTT_KEEP)
KERNEL_RTT_KEEP = 200_000  # a long live run must not grow them without end


def _krtt_add(kind: str, v: int) -> None:
    KERNEL_RTT.setdefault(kind, deque(maxlen=KERNEL_RTT_KEEP)).append(v)


def _enable_kernel_ts(sock) -> None:
    if not sys.platform.startswith("linux"):
        raise SystemExit("--kernel-ts needs Linux (SO_TIMESTAMPING)")
    sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPING, _TS_FLAGS)


def _ts_of(anc) -> int | None:
    for _lvl, typ, val in anc:
        if typ == SO_TIMESTAMPING and len(val) >= 16:
            sec, ns = struct.unpack("qq", val[:16])
            return sec * 1_000_000_000 + ns
    return None


def _recv_raw(sock, args, timeout: float):
    """(data, addr, kernel RX ns or None)."""
    if getattr(args, "kernel_ts", False):
        data, anc, _flags, addr = sock.recvmsg(512, 256)
        return data, addr, _ts_of(anc)
    data, addr = sock.recvfrom(512)
    return data, addr, None


def _rtt(args, t0_ns: int) -> int:
    """The timed round trip of the reply just received. --timing kernel: the kernel's view (driver TX of
    the challenge to driver RX of the reply), which leaves out this host's own system-call, scheduling and
    wake-up delays, the main source of honest misses on a busy verifier host. Else user space."""
    k = getattr(args, "_last_krtt", None)
    if getattr(args, "timing", "user") == "kernel" and k is not None:
        return k
    return getattr(args, "_t_recv", time.perf_counter_ns()) - t0_ns


def _recv_timed(sock, args, kind: str = "static", timeout: float = 10.0):
    """Receive a timed reply. With --kernel-ts also record the kernel's view of the round trip (software
    timestamps where the driver sent the challenge and received the reply), which leaves out this host's
    system-call, scheduling and wake-up time: a diagnostic of where verifier-side latency comes from."""
    t_in = time.time_ns() if getattr(args, "kernel_ts", False) else 0
    data, addr, rx = _recv_raw(sock, args, timeout)
    args._t_recv = time.perf_counter_ns()  # user-space receive time, before the TX-stamp bookkeeping
    args._last_krtt = None
    args._bad_ts = None
    t_out = time.time_ns() if t_in else 0
    if rx is not None:
        # TX stamps queue on the error queue, one per send (untimed sends too): the newest is the challenge's; a
        # stray datagram received first in the same wait has drained it already
        tx = getattr(args, "_pending_tx", None)
        sock.setblocking(False)
        try:
            while True:
                try:
                    _d, anc, _f, _a = sock.recvmsg(1, 256, socket.MSG_ERRQUEUE)
                    tx = _ts_of(anc) or tx
                except (BlockingIOError, InterruptedError):
                    break
        finally:
            sock.settimeout(timeout)
        args._pending_tx = tx
        if tx is not None:
            krtt = rx - tx
            if 0 < krtt <= _reply_timeout(args) * 1e9:
                args._last_krtt = krtt
                _krtt_add(kind, krtt)
                _krtt_add(kind + ":tx->send-return", t_in - tx)
                _krtt_add(kind + ":rx->user", t_out - rx)
            else:
                # a clock step between the two stamps (or a stamp of another datagram): not a round trip. Under
                # --timing kernel the reply counts as missing (`_timed`), never as a time
                args._bad_ts = krtt
    return data, addr


def _reply_timeout(args) -> float:
    return float(getattr(args, "reply_timeout", None) or 10.0)


def _await(sock, args, want, timeout: float, kind: str | None = None):
    """The first datagram `want` accepts within `timeout` s, or None. Anything else that arrives meanwhile
    (a reply that came after its own wait gave up, a duplicate, a datagram of the wrong size or token) is
    dropped and the wait goes on: one lost, late or forged reply costs one verification, never the run.
    kind: the wait for a timed reply (`_recv_timed`); `_rtt` then reads the accepted datagram's round trip."""
    end = time.perf_counter() + timeout
    args._pending_tx = None
    while (left := end - time.perf_counter()) > 0:
        try:
            sock.settimeout(left)
            data, _ = _recv_timed(sock, args, kind, left) if kind else sock.recvfrom(512)
        except (TimeoutError, ConnectionRefusedError):  # refused: ICMP for an earlier send; the wait goes on
            continue
        if want(data):
            return data
        args._stray = getattr(args, "_stray", 0) + 1
    return None


def _send(sock, args, pkt: bytes) -> bool:
    """Send one datagram to the prover. False if it could not be sent (EHOSTUNREACH, ENETUNREACH, EPERM from a
    firewall, a full buffer): that challenge is a missing reply, which fails its verification and alerts, and
    the run goes on (before, the exception ended the verifier)."""
    try:
        sock.sendto(pkt, (args.host, args.port))
        return True
    except OSError as e:
        print(f"verifier: send to the prover failed: {e}", flush=True)
        args._send_failed = getattr(args, "_send_failed", 0) + 1
        return False


def _arm(sock, args, B, n, r, k, g, sid: int) -> int | None:
    """ARM -> READY (the drain window). The READY wait in ns, or None if no READY for this ARM came in time."""
    token = secrets.randbits(62) | 1  # never the nonce: ARM is outside the timed window
    ta = time.perf_counter_ns()
    if not _send(sock, args, ARM.pack(ARM_MAGIC, token, B, n, r, k, g, sid)):
        return None
    ok = _await(
        sock, args, lambda d: len(d) == READY.size and READY.unpack(d) == (ARM_MAGIC, token), _reply_timeout(args)
    )
    return time.perf_counter_ns() - ta if ok is not None else None


def _timed(sock, args, ch: Challenge, magic: int, sid: int, kind: str = "static"):
    """One timed challenge: (rtt ns, digest lo, digest hi), or None if no reply with its nonce came in time."""
    pkt = CH.pack(magic, ch.nonce, ch.j0, ch.block_bytes, ch.n, ch.r, ch.k, ch.g, sid)
    t0 = time.perf_counter_ns()
    if not _send(sock, args, pkt):
        return None
    data = _await(sock, args, lambda d: len(d) == RE.size and RE.unpack(d)[0] == ch.nonce, _reply_timeout(args), kind)
    if data is None:
        return None
    if getattr(args, "timing", "user") == "kernel" and (
        getattr(args, "_bad_ts", None) is not None or getattr(args, "_last_krtt", None) is None
    ):
        print(
            f"verifier: unusable kernel timestamp for nonce {ch.nonce:016x} "
            f"(delta {getattr(args, '_bad_ts', None)} ns): counted as a missing reply",
            flush=True,
        )
        return None
    rtt = _rtt(args, t0)
    _, lo, hi = RE.unpack(data)
    return rtt, lo, hi


def _prover_stats(sock, args, timeout: float):
    """The prover's STATS reply as a tuple, or None."""
    tok = secrets.randbits(62) | 1
    if not _send(sock, args, STATS.pack(STATS_MAGIC, tok)):
        return None
    data = _await(sock, args, lambda d: len(d) == STATSRE.size and STATSRE.unpack(d)[:2] == (STATS_MAGIC, tok), timeout)
    return STATSRE.unpack(data) if data is not None else None


def _negotiate(args, strategies: list[str], posture: bool = False) -> list[str]:
    """Keep the strategies the prover serves (the wire mask of its STATS reply; with posture, the postures it
    offers) and drop the rest with a message: a request for a strategy the prover could not build (the
    standalone CUDA prover has no partial segment table, no lora copy, no posture cheats) is ignored, and
    every one of its challenges would only time out. A prover that does not answer keeps the list as given."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        f = _prover_stats(sock, args, 2.0)
    finally:
        sock.close()
    if f is None:
        print(
            f"verifier: the prover did not report the strategies it serves; requesting {','.join(strategies)} as given",
            flush=True,
        )
        return strategies
    mask = f[12 if posture else 11]
    served = [s for s in strategies if mask >> STRATS.index(s) & 1]
    if len(served) < len(strategies):
        print(
            f"verifier: the prover does not serve {','.join(s for s in strategies if s not in served)}; dropped "
            f"(it serves {','.join(s for i, s in enumerate(STRATS) if mask >> i & 1) or 'nothing'})",
            flush=True,
        )
    if "honest" in strategies and "honest" not in served:
        raise SystemExit("the prover does not serve the honest strategy")
    if not served:
        raise SystemExit(f"the prover serves none of {','.join(strategies)}")
    return served


def _dyn_shape(args, spec):
    """The dynamic challenge's shape: its own block size and chain length, the static r and k."""
    _, _, r, k, g = _shape(args, spec)
    return parse_bytes(args.dyn_B), args.dyn_n or args.n, r, k, g


def _dyn_wait_s(args) -> float:
    """The batch roles' wait before a dynamic round: exponential from the CSPRNG, of the mean live waits before
    any round with --dynamic (mean aggregate wait --interval / 2). Right after
    the static drain a freeze lands at the start of a step (the drain has just let one go); after a random wait
    it lands anywhere in a step, as in live, so a dynamic deadline and READY bound mean the same in every role."""
    return _gap(float(getattr(args, "interval", 1.0) or 0.0) / 2)


DYN_SCHEDULE = "independent"  # a baseline's dynamic rounds came at their own random times (every role, since review 5)


def _dyn_schedule(args) -> dict:
    """How the dynamic rounds of a calibration were scheduled, kept in its baseline's dyn section."""
    return {
        "schedule": DYN_SCHEDULE,
        "wait_mean_s": float(getattr(args, "interval", 1.0) or 0.0) / 2,
        "jitter_ms": float(getattr(args, "dyn_jitter_ms", 0.0) or 0.0)
        if getattr(args, "role", "live") != "live"
        else 0.0,
    }


def _dyn_ready_bound(args, darms) -> float:
    """The dynamic READY bound (ns): --dyn-arm-max-ms, else twice the calibration's p99.9 pause wait. Its own
    flag: a freeze waits for the next layer or step boundary, not for a step to drain, so the static
    --arm-max-ms says nothing about it."""
    ms = getattr(args, "dyn_arm_max_ms", None)
    return ms * 1e6 if ms else 2.0 * float(np.quantile(darms, 0.999))


# A reply that did not come: the timing claim counts it at the clip; under --challenge-flags warn it is a flag, not an
# alert
FAULT_NO_READY = "no READY"  # a static ARM's READY that did not come: no challenge went
FAULT_NO_DYN_READY = "no dynamic READY"  # (a prefix: "... (the ARM could not be sent)" too)
FAULT_NO_STATIC_REPLY = "no reply to a static challenge"
FAULT_NO_DYN_REPLY = "no reply to the dynamic chain"
FAULT_NO_LIVE_REPLY = "no reply to the live chain"


def _dynamic_round(sock, args, rng, shape, wire_sid: int, jitter: bool = True, times: dict | None = None):
    """One dynamic verification: ARM_DYN (the prover freezes at its next boundary and reports the sizes of
    its region and of its live region) -> a chain over the region -> reply -> (if the prover has a live
    region) a second chain over it -> reply. Returns (sample, challenge, digest, live) with live =
    (challenge, digest) or None; each chain's round trip is timed on its own. A READY or reply that does not
    come, or a READY whose region no chain can cover, sets sample["fault"] (the verification fails) and
    leaves out what it would have checked: (sample, None, None, None) when the main chain has no reply.
    jitter: wait up to --dyn-jitter-ms first (the batch roles, on top of their random wait, `_dyn_wait_s`; the
    live verifier schedules dynamic rounds at their own random times and passes False). times: filled with
    the perf_counter_ns() at which the ARM ("arm"), the main challenge ("ch") and the live challenge ("live") went."""
    Bd, nd, r, k, g = shape
    token = secrets.randbits(62) | 1
    jit = getattr(args, "dyn_jitter_ms", 0.0) if jitter else 0.0
    if jit:  # land the freeze anywhere in a step, not just after the static drain
        time.sleep(_SYSRAND.uniform(0.0, jit) / 1000)  # the CSPRNG too: a predictable freeze time is one to prepare for
    ta = time.perf_counter_ns()
    if times is not None:
        times["arm"] = ta
    want_live = getattr(args, "dyn_live", False)
    if not _send(
        sock, args, ARM.pack(ARM_DYN_MAGIC, token, Bd, nd, r, k, g, wire_sid | (LIVE_FLAG if want_live else 0))
    ):
        return (
            {
                "rtt_ns": None,
                "arm_ns": None,
                "words": 0,
                "blocks": 0,
                "in_flight": False,
                "prover_block_bytes": 0,
                "fault": FAULT_NO_DYN_READY + " (the ARM could not be sent)",
            },
            None,
            None,
            None,
        )
    data = _await(
        sock,
        args,
        lambda d: len(d) == READY_DYN.size and READY_DYN.unpack(d)[:2] == (ARM_DYN_MAGIC, token),
        _reply_timeout(args),
    )
    arm_ns = time.perf_counter_ns() - ta
    if data is None:
        return (
            {
                "rtt_ns": None,
                "arm_ns": None,
                "words": 0,
                "blocks": 0,
                "in_flight": False,
                "prover_block_bytes": 0,
                "fault": FAULT_NO_DYN_READY,
            },
            None,
            None,
            None,
        )
    _, _, words, bb, flags, live_words = READY_DYN.unpack(data)
    sample = {
        "rtt_ns": None,
        "arm_ns": arm_ns,
        "words": words,
        "blocks": words * 8 // Bd,
        "in_flight": bool(flags & 1),
        "prover_block_bytes": bb,
        "shape": [Bd, nd, r, k, g],
    }
    if sample["blocks"] < 1:  # the prover's figure: a crash here would let it end monitoring
        sample["fault"] = (
            f"dynamic READY reports {words * 8} bytes, less than one {Bd}-byte block (lower --dyn-B if honest)"
        )
        return sample, None, None, None
    ch = Challenge.fresh(Bd, nd, r, sample["blocks"], k=k, g=g)
    sample.update({"nonce": f"{ch.nonce:016x}", "j0": ch.j0})
    if times is not None:
        times["ch"] = time.perf_counter_ns()
    got = _timed(sock, args, ch, DYN_MAGIC, wire_sid, "dynamic")
    if got is None:
        sample["fault"] = FAULT_NO_DYN_REPLY
        return sample, None, None, None
    sample["rtt_ns"], lo, hi = got
    sample["received"] = f"{hi:016x}{lo:016x}"
    live = None
    if want_live:
        # asked for: a live region is never empty (it holds at least the activation rows, padded to a block), so
        # none is a prover dropping its live chain
        nbl = live_words * 8 // Bd
        sample["live"] = {"rtt_ns": None, "words": live_words, "blocks": nbl}
        if nbl < 1:
            sample["fault"] = (
                f"dynamic READY reports a live region of {live_words * 8} bytes, less than one {Bd}-byte block"
                if live_words
                else "dynamic READY reports no live region, though the verifier asked for one (--with-dyn-live; leave it out for a prover without one)"
            )
            return sample, ch, (lo, hi), None
        chl = Challenge.fresh(Bd, nd, r, nbl, k=k, g=g)
        sample["live"].update({"nonce": f"{chl.nonce:016x}", "j0": chl.j0})
        if times is not None:
            times["live"] = time.perf_counter_ns()
        got = _timed(sock, args, chl, DYN_LIVE_MAGIC, wire_sid, "live")
        if got is None:
            sample["fault"] = FAULT_NO_LIVE_REPLY
            return sample, ch, (lo, hi), None
        sample["live"]["rtt_ns"], lol, hil = got
        sample["live"]["received"] = f"{hil:016x}{lol:016x}"
        live = (chl, (lol, hil))
    return sample, ch, (lo, hi), live


_REPLAYER = {}


def _replayer(args):
    """The verifier's replay engine (--replay), built once; None when records are only stored. Replays run on
    at most --verify-threads CPU threads (torch would take every core of a shared verifier box)."""
    spec = getattr(args, "replay", None)
    if not spec:
        return None
    from .consistency import _spec_len

    slots = getattr(args, "replay_max_len", None)
    if slots is None:  # the length each prover is started with by default
        slots = (
            (getattr(args, "toy_len", None) or 64)
            if spec.split("@")[0] == "toy"
            else (getattr(args, "toy_len", None) or 512)
        )
    if _spec_len(spec)[1] is None and slots:
        # the verifier's cache length, never the record's: a record naming another is refused before anything is built,
        # so a prover cannot have the verifier allocate a cache of the size it chooses (up to the model's positions)
        spec = f"{spec}@{int(slots)}"
    if spec not in _REPLAYER:
        from .consistency import replayer_from_spec

        try:
            import torch

            torch.set_num_threads(max(1, getattr(args, "verify_threads", None) or default_threads()))
        except ImportError:
            pass
        _REPLAYER[spec] = replayer_from_spec(spec, getattr(args, "replay_device", "cpu"))
    return _REPLAYER[spec]


TEXT_MAX = 300  # the most characters of any text the verifier publishes that a record could influence


def _short(v, n: int = TEXT_MAX) -> str:
    t = str(v)
    return t if len(t) <= n else t[:n] + f"… ({len(t)} characters)"


def _int_or_none(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _record_fields(ev) -> dict:
    """What the verifier publishes about a record: its own short, typed reading of the stage, never the
    record's fields as sent (a record that replays exactly can still carry megabytes of junk in fields the
    replay does not read; before, they went into the check event, the page's history and the log)."""
    st = ev.stage if isinstance(ev.stage, dict) else {}
    desc = _short(ev.describe(), 200)  # a stage it cannot describe is malformed: a failed check
    kind = st.get("kind")
    return {
        "stage": desc,
        "kind": _short(kind, 32) if isinstance(kind, str) else None,
        "step": _int_or_none(st.get("step")),
        "act_layer": _int_or_none(st.get("act_layer")),
        "in_flight": st.get("in_flight") if isinstance(st.get("in_flight"), bool) else None,
        "n_layers": _int_or_none(st.get("n_layers")),
    }


def _segments(diff) -> list:
    out = []
    for d in (diff or [])[:8]:
        if isinstance(d, dict):
            out.append(
                {
                    "name": _short(d.get("name"), 64),
                    "words_differing": _int_or_none(d.get("words_differing")),
                    "of": _int_or_none(d.get("of")),
                }
            )
    return out


def _canonical(stage) -> str:
    return json.dumps(stage, sort_keys=True, separators=(",", ":"))


def _record_budget(args) -> float:
    return float(getattr(args, "record_budget_s", None) or 60.0)


def _fetch_bound(args, ch: Challenge, digest, words: int | None, region: str, timeout: float):
    """Fetch the record of one timed chain and bind it to what the verifier sent and saw: (record or None,
    why it cannot stand for the chain or None, seconds the fetch took)."""
    from ..core.evidence import MAX_HEADER, fetch_evidence
    from .consistency import Expected, claim_mismatch

    t0 = time.monotonic()
    args._fetch_t0 = t0  # a fetch in progress (the live lag alert reads it)
    try:
        ev = fetch_evidence(
            args.host,
            args.port,
            ch.nonce,
            timeout=max(0.001, timeout),
            max_header=parse_bytes(str(getattr(args, "evidence_max_header", None) or MAX_HEADER)),
            max_data=parse_bytes(str(getattr(args, "evidence_max_bytes", None) or "256M")),
        )
    except (OSError, LookupError, ValueError) as e:
        return None, _short(f"no record: {e}"), time.monotonic() - t0
    finally:
        args._fetch_t0 = None
    exp = Expected(*args._dyn_expected) if getattr(args, "_dyn_expected", None) else None
    why = _record_mismatch(ev, ch, digest, words, region) or claim_mismatch(ev, exp) or _encoding_mismatch(args, ev)
    return ev, (_short(why) if why else None), time.monotonic() - t0


def _replay_one(args, keys, ev, lib, fetch_s: float, main=None) -> dict:
    """Replay one bound record (with --replay) within the record budget: fetch and replay together over it
    fail the check, so a record that takes too long to check cannot hold the checking thread unseen. main: for
    a live record, the main record of the same freeze, already bound and checked (its stage is the one the live
    region is rebuilt from once the replay takes it)."""
    from .consistency import Expected, live_mismatch, verify

    budget = _record_budget(args)
    rep = {"digest_ok": None, "replayed": False}
    rp = _replayer(args)
    if rp is None:
        return rep
    if fetch_s >= budget:
        return {
            "digest_ok": False,
            "replayed": False,
            "failed": "budget",
            "error": f"the fetch took {fetch_s:.1f} s, the whole record budget (--record-budget-s {budget:g})",
        }
    t0 = time.monotonic()
    exp = Expected(*args._dyn_expected) if getattr(args, "_dyn_expected", None) else None
    if main is not None and (why := live_mismatch(ev, main)) is not None:
        return {"digest_ok": False, "replayed": False, "failed": "binding", "error": _short(why)}
    verdict = verify(
        ev, rp, keys, lib, expected=exp, main=main
    )  # a live region is rebuilt from its main record's stage
    rep.update(
        {
            "digest_ok": bool(verdict["ok"]),
            "replayed": True,
            "expected": verdict.get("expected") if isinstance(verdict.get("expected"), str) else None,
        }
    )
    if not verdict["ok"]:
        rep["failed"] = "replay"
    if verdict.get("error"):
        rep["error"] = _short(verdict["error"])
    if verdict.get("differing_segments"):
        rep["differing_segments"] = _segments(verdict["differing_segments"])
    took = fetch_s + time.monotonic() - t0
    if took > budget:
        rep["digest_ok"] = False
        rep["failed"] = "budget"
        rep["error"] = f"fetch and replay took {took:.1f} s, over the record budget (--record-budget-s {budget:g})"
    return rep


def _save_record(args, ev, nonce: int):
    """Keep a record that passed its binding checks in --evidence-dir, under the challenge's nonce, while the
    directory holds less than --evidence-max-total. The path saved to; None without --evidence-dir; False when
    it was not kept: past the cap (or once a write failed) nothing more is saved, with one warning, and each
    record not kept says so (`record_saved` false in its check), so an offline audit with the verifier's log
    tells a record the verifier did not keep from one the prover did not give."""
    evidence_dir = getattr(args, "evidence_dir", None)
    if not evidence_dir:
        return None
    cap = parse_bytes(str(getattr(args, "evidence_max_total", None) or "20G"))
    if getattr(args, "_evidence_used", None) is None:  # what the directory already holds
        p = Path(evidence_dir)
        args._evidence_used = sum(f.stat().st_size for f in p.iterdir() if f.is_file()) if p.is_dir() else 0
    if args._evidence_used >= cap and not getattr(args, "_evidence_full", False):
        args._evidence_full = (
            f"--evidence-dir {evidence_dir} holds {args._evidence_used / 2**30:.2f} GiB, the --evidence-max-total "
            f"{cap / 2**30:.2f} GiB"
        )
        print(f"verifier: WARNING: {args._evidence_full}: records are checked but no longer saved", flush=True)
    if getattr(args, "_evidence_full", False):
        return False
    try:
        path = ev.save(evidence_dir, nonce=nonce, max_debug=4096)
        args._evidence_used += sum(f.stat().st_size for f in (path, path.with_suffix(".npy")) if f.exists())
    except OSError as e:  # a full disk: the verifier's failure, never the record's
        args._evidence_full = _short(f"a record could not be written to --evidence-dir {evidence_dir}: {e}")
        print(f"verifier: WARNING: {args._evidence_full}: records are checked but no longer saved", flush=True)
        return False
    return str(path)


def _not_kept(rep: dict) -> bool:
    """A record check one of whose bound records was not kept (`_save_record` False)."""
    return rep.get("record_saved") is False or (rep.get("live") or {}).get("record_saved") is False


DYN_FAILURE = {
    "fetch": "no record of the chain came from the prover",
    "binding": "the record does not stand for the chain sent",
    "budget": "the record took longer to check than the record budget",
    "replay": "replaying inference to the recorded stage does not reproduce the timed digest",
    "malformed": "the record is malformed",
    "unchecked": "the record was neither replayed nor kept",
}


def _dynamic_failure(rep: dict) -> str:
    """What failed in a dynamic round's record check, in words: the main record's failure first, else the live one's."""
    why = rep.get("failed") or (rep.get("live") or {}).get("failed")
    text = DYN_FAILURE.get(why, "the record check failed")
    if not rep.get("failed") and why:
        text = "live chain: " + text
    return text + (f" ({rep['error']})" if rep.get("error") else "")


def _check_dynamic(args, keys, ch: Challenge, digest, lib=None, live=None, sample=None) -> dict:
    """Off the timing path: check the record of the chain over all dynamic memory and, if there was one,
    the record of the live chain of the same freeze. digest_ok is both exact (None: stored, not replayed).
    sample: the dynamic sample of the verification; its READY sizes must be the sizes the records cover.
    The two records must describe one freeze: the same stage (JSON-equal), regions "main" and "live"; else a
    prover could build its live region from any stage (an idle one) and the live chain would check nothing.
    A record that fails a binding check is not saved. A malformed record is a failed check, never an exception."""
    try:
        return _check_dynamic_inner(args, keys, ch, digest, lib, live, sample)
    except Exception as e:  # the record comes from the prover: anything in it may be wrong
        return {
            "digest_ok": False,
            "failed": "malformed",
            "error": _short(f"malformed record: {type(e).__name__}: {e}"),
        }


def _fetch_timeout(args) -> float:
    """The most one record's fetch may take: --evidence-timeout, never more than the record budget. Well below
    the budget by default (10 s against 60 s), so a prover that trickles its records only slowly is caught on
    the fetch and its records are the ones that fail, not the replays behind them."""
    return min(float(getattr(args, "evidence_timeout", None) or 10.0), _record_budget(args))


def _check_dynamic_inner(args, keys, ch, digest, lib, live, sample) -> dict:
    words = sample.get("words") if sample else None
    live_words = (sample.get("live") or {}).get("words") if sample else None
    timeout = _fetch_timeout(args)
    ev, why, fs = _fetch_bound(args, ch, digest, words, "main", timeout)
    lev = lwhy = None
    lfs = 0.0
    if live is not None:
        lev, lwhy, lfs = _fetch_bound(args, live[0], live[1], live_words, "live", timeout)
        if (
            ev is not None
            and lev is not None
            and why is None
            and lwhy is None
            and _canonical(ev.stage) != _canonical(lev.stage)
        ):
            why = lwhy = (
                "the live record's stage is not the main record's: the two chains of a freeze must cover one stage"
            )
    rep = _one_report(args, keys, ev, why, fs, lib, ch)
    rep["fetch_s"] = round(fs + lfs, 3)  # the verifier's own measure: the lag alert names slow fetches
    rep["nonce"] = f"{ch.nonce:016x}"  # the challenge's (an offline audit pairs record_saved with it)
    if live is not None:  # after the main record, with it at hand
        lrep = _one_report(args, keys, lev, lwhy, lfs, lib, live[0], main=ev)
        rep["live"] = {
            k: lrep.get(k)
            for k in ("digest_ok", "expected", "error", "differing_segments", "path", "failed", "record_saved")
            if lrep.get(k) is not None
        }
        rep["live"]["nonce"] = f"{live[0].nonce:016x}"
        if rep["digest_ok"] is not None and lrep["digest_ok"] is not None:
            rep["digest_ok"] = bool(rep["digest_ok"] and lrep["digest_ok"])
        elif lrep["digest_ok"] is False:
            rep["digest_ok"] = False
        if lrep.get("error") and not rep.get("error"):
            rep["error"] = "live chain: " + lrep["error"]
    return rep


def _one_report(args, keys, ev, why, fetch_s, lib, ch, main=None) -> dict:
    """The check of one record: bound (else failed and not saved), replayed within the budget, saved. main: a
    live record's main record (`_replay_one`)."""
    if ev is None:
        return {"digest_ok": False, "failed": "fetch", "error": why}
    rep = {"digest_ok": None, "replayed": False, **_record_fields(ev)}
    if why is not None:  # it does not stand for the chain that was sent
        rep["digest_ok"] = False
        rep["error"] = why
        rep["failed"] = "binding"
        return rep
    rep.update(_replay_one(args, keys, ev, lib, fetch_s, main=main))
    path = _save_record(args, ev, ch.nonce)
    if path is False:  # bound, but not kept: --evidence-max-total reached
        rep["record_saved"] = False
    elif path is not None:
        rep["path"] = path
    return rep


def _encoding_mismatch(args, ev) -> str | None:
    """With --dyn-encoded the memory claim credits the whole dynamic region, live part included, as VDE words: a
    record of a plain region (the live KV and activation rows as they are) cannot stand for that."""
    if getattr(args, "dyn_encoded", False) and not (isinstance(ev.stage, dict) and ev.stage.get("encoded") is True):
        return "record's region is not encoded (--dyn-encoded credits the whole region, live data included, as encoded)"
    return None


def _record_mismatch(ev, ch: Challenge, digest, words: int | None, region: str) -> str | None:
    """Why a record cannot stand for this challenge, or None. The replay recomputes the chain the record
    describes, so the record must describe exactly the chain that was sent (a prover could otherwise
    record one hash-only link for a 32-link matmul challenge) over exactly the region size READY reported
    (that size is what the memory accounting credits)."""
    if ev.nonce != ch.nonce or tuple(ev.digest) != tuple(digest):
        return "record does not match the challenge and reply"
    sent = {"block_bytes": ch.block_bytes, "n": ch.n, "r": ch.r, "k": ch.k, "g": ch.g, "j0": ch.j0}
    got = {key: ev.challenge.get(key) for key in sent}
    if got != sent:
        diff = ", ".join(f"{key} {_short(got[key], 40)} (sent {sent[key]})" for key in sent if got[key] != sent[key])
        return f"record's challenge differs from the one sent: {diff}"
    if ev.region != region:
        return f"record is of the {_short(ev.region, 40)} region, the challenge was for the {region} region"
    if words is not None and ev.words != words:
        return f"record covers {_short(ev.words, 40)} words, READY reported {words}"
    return None


def _check_record(args, keys, ch: Challenge, digest, lib=None, words: int | None = None, region: str = "main") -> dict:
    """One record on its own: fetch it, bind it to this chain, (with --replay) replay inference to it, rebuild
    the region and recompute the chain: digest_ok is exact equality with the timed reply. Without --replay
    the record is only stored (digest_ok None) for memory_accounting.verifier.consistency. A malformed record is a failed check,
    never an exception."""
    try:
        ev, why, fs = _fetch_bound(args, ch, digest, words, region, _fetch_timeout(args))
        return _one_report(args, keys, ev, why, fs, lib, ch)
    except Exception as e:
        return {
            "digest_ok": False,
            "failed": "malformed",
            "error": _short(f"malformed record: {type(e).__name__}: {e}"),
        }


def _challenge_loop(args, spec, keys, strategies, samples, warmup, check_first: int | None = None):
    """The batch roles' timing loop. check_first: check the digests and records of the first this many
    verifications only (--sample-verify; None: every one); the rest are counted UNCHECKED."""
    fixed_claim = _fixed_claim_options(args)
    ver, cfg = _make_verifier(args, spec, keys)
    B, n, r, k, g = _shape(args, spec)
    nb = cfg.nbytes // B
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10.0)
    if getattr(args, "kernel_ts", False):
        _enable_kernel_ts(sock)
        KERNEL_RTT.clear()
    rng = np.random.default_rng(args.seed + 1)
    results = {
        s: {
            "rtt_ns": [],
            "chains_ns": [],
            "arm_ns": [],
            "correct": 0,
            "checked": 0,
            "unchecked": 0,
            "dyn_rtt_ns": [],
            "dyn_arm_ns": [],
            "dyn_correct": 0,
            "dyn_checked": 0,
            "dyn_unchecked": 0,
            "dyn_in_flight": 0,
            "dyn_live_rtt_ns": [],
            "dyn_words": [],
            "dyn_live_words": [],
            "missing": 0,
            "dyn_faults": 0,
            "dyn_not_kept": 0,
            "span_s": 0.0,
        }
        for s in strategies
    }
    if fixed_claim:
        pilots = {s: _new_claim_samples(args) for s in strategies}
        for result in results.values():
            result["claim_calibration_samples"] = _new_claim_samples(args)
    t_measured = None  # when the first verification past the warm-up began
    drain = getattr(args, "drain", False)
    repeat = max(1, getattr(args, "repeat", 1))
    dynamic = bool(getattr(args, "dynamic", False))
    if fixed_claim and dynamic:
        dyn_B, dyn_n = _dyn_shape(args, spec)[:2]
        dyn_dc = args.claim_delta_c_dyn if args.claim_delta_c_dyn is not None else dyn_delta_c_us(dyn_B)
    inf = float("inf")
    to_check: list[tuple[str, Challenge, int, int]] = []  # digests are verified after the timing loop: the CPU
    to_check_dyn: list[tuple[str, Challenge, tuple]] = []  # evidence is fetched after it too (TCP transfers)
    for i in range(warmup + samples):  # recomputation is a host-wide burst that must not sit between challenges
        checked = i >= warmup and (check_first is None or i - warmup < check_first)
        if i == warmup:
            t_measured = time.time()
            if fixed_claim:
                for strategy in strategies:
                    saved = results[strategy]["claim_calibration_samples"]
                    _freeze_claim_setup(
                        saved, pilots[strategy], args, "static", n, args.claim_delta_c, samples * repeat
                    )
                    if dynamic:
                        for chain in ["dynamic"] + (["live"] if getattr(args, "dyn_live", False) else []):
                            _freeze_claim_setup(saved, pilots[strategy], args, chain, dyn_n, dyn_dc, samples)
        for strategy in strategies:
            # one verification = min RTT over `repeat` challenges with fresh nonces. A cheater's delay is per
            # link and one-sided, so its minimum stays above the honest minimum by the full gap, while a
            # one-sided honest tail (host launch jitter) is removed. Each challenge's digest is checked, and
            # each chain's round trip is kept too (judged against the single-chain deadline: a cheater slow on
            # some chains only must not hide behind its fastest one).
            # A READY or reply that does not come fails the whole verification (inf): were the minimum taken
            # over the replies that did come, a cheater could drop its slow chains.
            best = None
            arm_worst = 0
            missing = 0
            chains = []
            for _ in range(repeat):
                ch = Challenge.fresh(B, n, r, nb, k=k, g=g)
                arm_ns = 0
                if drain:  # arm: prover drains its step, replies READY
                    arm_ns = _arm(sock, args, B, n, r, k, g, STRATS.index(strategy))
                    if arm_ns is None:
                        missing += 1
                        arm_worst = inf
                        chains.append(inf)
                        continue
                got = _timed(sock, args, ch, MAGIC, STRATS.index(strategy))
                arm_worst = max(arm_worst, arm_ns)
                if got is None:
                    missing += 1
                    chains.append(inf)
                    continue
                rtt, lo, hi = got
                chains.append(rtt)
                best = rtt if best is None else min(best, rtt)
                if checked:
                    to_check.append((strategy, ch, lo, hi))
                elif i >= warmup:
                    results[strategy]["unchecked"] += 1
            if missing:
                print(
                    f"verification {i} ({strategy}): {missing} of {repeat} challenges without a READY or reply",
                    flush=True,
                )
            if fixed_claim:
                target = pilots[strategy] if i < warmup else results[strategy]["claim_calibration_samples"]
                _append_claim_samples(target, {"challenges": [{"rtt_ns": t} for t in chains]}, ["static"])
            if i >= warmup:
                results[strategy]["rtt_ns"].append(inf if missing or best is None else best)
                results[strategy]["arm_ns"].append(arm_worst)
                results[strategy]["chains_ns"].append(chains)
                results[strategy]["missing"] += int(bool(missing))
            if dynamic:  # one dynamic round per verification, at a random time after the static ones
                time.sleep(_dyn_wait_s(args))
                sample, chd, dg, lv = _dynamic_round(sock, args, rng, _dyn_shape(args, spec), STRATS.index(strategy))
                if sample.get("fault"):
                    print(f"verification {i} ({strategy}): {sample['fault']}", flush=True)
                if fixed_claim:
                    target = pilots[strategy] if i < warmup else results[strategy]["claim_calibration_samples"]
                    _append_claim_samples(
                        target, {"dyn": sample}, ["dynamic"] + (["live"] if getattr(args, "dyn_live", False) else [])
                    )
                if i >= warmup:
                    bad = bool(sample.get("fault"))  # a fault fails the dynamic verification (inf)
                    results[strategy]["dyn_faults"] += int(bad)
                    results[strategy]["dyn_rtt_ns"].append(inf if bad else sample["rtt_ns"])
                    results[strategy]["dyn_arm_ns"].append(inf if sample["arm_ns"] is None else sample["arm_ns"])
                    results[strategy]["dyn_words"].append(sample["words"])
                    results[strategy]["dyn_live_words"].append(sample.get("live", {}).get("words", 0))
                    results[strategy]["dyn_in_flight"] += int(sample["in_flight"])
                    if "live" in sample:
                        results[strategy]["dyn_live_rtt_ns"].append(
                            inf if sample["live"]["rtt_ns"] is None else sample["live"]["rtt_ns"]
                        )
                    if chd is not None:
                        if checked:
                            to_check_dyn.append((strategy, chd, dg, lv, sample))
                        else:
                            results[strategy]["dyn_unchecked"] += 1
    for strategy in strategies:  # how long the measured verifications took (a baseline's calibration span)
        results[strategy]["span_s"] = time.time() - t_measured if t_measured is not None else 0.0
    if KERNEL_RTT:
        user = {
            kind: [t for s in strategies for t in results[s][key] if math.isfinite(t)]
            for kind, key in (("static", "rtt_ns"), ("dynamic", "dyn_rtt_ns"), ("live", "dyn_live_rtt_ns"))
        }
        for kind, v in KERNEL_RTT.items():
            if ":" in kind:
                print(f"  {kind}: median {np.median(v) / 1e3:.0f} µs, p90 {np.quantile(v, 0.9) / 1e3:.0f} µs")
                continue
            user_us = np.median(user[kind]) / 1e3 if user.get(kind) else float("nan")
            up = np.quantile(user[kind], 0.99) / 1e3 if user.get(kind) else float("nan")
            print(
                f"kernel-timestamped {kind} RTT (driver TX to driver RX on this host): median {np.median(v) / 1e3:.0f} µs, "
                f"p99 {np.quantile(v, 0.99) / 1e3:.0f} µs, min {np.min(v) / 1e3:.0f} µs over {len(v)}; user-space median {user_us:.0f} µs, p99 {up:.0f} µs"
            )
    for strategy, ch, lo, hi in to_check:
        results[strategy]["checked"] += 1
        results[strategy]["correct"] += int((lo, hi) == ver.expected(ch)[:2])
    for strategy, chd, dg, lv, sample in to_check_dyn:
        rep = _check_dynamic(args, keys, chd, dg, ver.lib, live=lv, sample=sample)
        if rep["digest_ok"] is not None:
            results[strategy]["dyn_checked"] += 1
            results[strategy]["dyn_correct"] += int(rep["digest_ok"])
        elif _not_kept(rep):  # neither replayed nor kept: nothing checks it, now or later
            results[strategy]["dyn_not_kept"] += 1
        else:
            results[strategy]["dyn_stored"] = results[strategy].get("dyn_stored", 0) + 1
        if "error" in rep:
            print(f"dynamic record {chd.nonce:016x}: {rep['error']}", flush=True)
    for strategy in strategies:
        user_us, du, nk = (
            results[strategy]["unchecked"],
            results[strategy]["dyn_unchecked"],
            results[strategy]["dyn_not_kept"],
        )
        if user_us or du:
            print(
                f"verifier: UNCHECKED ({strategy}, --sample-verify {check_first}): {user_us} static digests"
                + (f" and {du} dynamic records" if dynamic else "")
                + " were never checked; a wrong answer among them is not seen",
                flush=True,
            )
        if nk:
            print(
                f"verifier: UNCHECKED ({strategy}): {nk} dynamic records were neither replayed (no --replay) nor kept (--evidence-max-total "
                "reached): those freezes are checked by nothing",
                flush=True,
            )
    return results, cfg


def _sample_verify(args, default: int | None = None) -> int | None:
    """--sample-verify N: check only the first N verifications (None: all, the default)."""
    n = getattr(args, "sample_verify", None)
    return default if n is None else max(0, int(n))


def baseline_main(args):
    """Setup-time baselining: many honest challenges at the chosen configuration -> deadline."""
    fixed_claim = _fixed_claim_options(args)
    spec, keys = array_from_args(args)
    _shape(args, spec)  # the shape policy, before anything is loaded or sent
    _no_live_scratch(args)
    results, cfg = _challenge_loop(
        args, spec, keys, ["honest"], args.samples, args.warmup, _sample_verify(args, 1 if args.no_verify_all else None)
    )
    from ..core.analysis import core_samples

    honest_ns = np.array(results["honest"]["rtt_ns"], dtype=np.float64)
    if not np.isfinite(honest_ns).all():  # no READY or reply: not a round trip to set a deadline from
        print(
            f"baseline: {int((~np.isfinite(honest_ns)).sum())} of {len(honest_ns)} verifications had a READY or reply missing; left out",
            flush=True,
        )
        honest_ns = honest_ns[np.isfinite(honest_ns)]
    if len(honest_ns) == 0:
        raise SystemExit(
            "baseline: no verification had all its READYs and replies; no deadline can be set (is the prover up?)"
        )
    arms = np.array(results["honest"]["arm_ns"], dtype=np.float64)
    arms = arms[np.isfinite(arms)]
    quantiles = {p: float(np.quantile(honest_ns, p / 100)) for p in (50, 90, 99, 99.9)}
    core = core_samples(honest_ns)
    sigma = float(core.std())
    if args.deadline_rule == "sigma":
        # robust: a stall in the tail does not set it; the fractional floor keeps a very tight σ from
        # producing a window narrower than ordinary clock/thermal drift
        deadline = float(np.median(core)) + max(args.sigmas * sigma, args.margin * float(np.median(core)))
    else:
        deadline = quantiles[99.9] * (1 + args.margin)
    frr = float(np.mean(honest_ns > deadline))
    B_, _, r_, k_, g_ = _shape(args, spec)
    base = {
        "B": B_,
        "n": args.n,
        "r": r_,
        "k": k_,
        "g": g_,
        "repeat": max(1, args.repeat),
        "samples": len(honest_ns),
        "median_ns": quantiles[50],
        "p99_ns": quantiles[99],
        "sigma_core_ns": sigma,
        "deadline_rule": args.deadline_rule,
        "sigmas": args.sigmas,
        "margin": args.margin,
        "deadline_ns": deadline,
        "honest_reject_rate": frr,
        "quantiles_ns": quantiles,
        "correct": results["honest"]["correct"],
        "host": args.host,
        "drain": bool(args.drain),
        "timing": args.timing,
        "dynamic": bool(args.dynamic),
        "arm_median_ns": float(np.median(arms)) if args.drain else 0.0,
        "arm_p999_ns": float(np.quantile(arms, 0.999)) if args.drain else 0.0,
        # READY bound: the prover learns a challenge is coming, not which blocks; what it can do with
        # the warning is reload the challenged region, bound × PCIe bandwidth bytes of it. So the
        # bound is the honest step drain plus margin, and a late READY fails the challenge.
        "arm_max_ns": (args.arm_max_ms * 1e6 if args.arm_max_ms else 2.0 * float(np.quantile(arms, 0.999)))
        if args.drain
        else 0.0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    base.update(_chain_stats([t for v in results["honest"]["chains_ns"] for t in v], base, args))
    print(
        (f"min-of-{args.repeat} " if args.repeat > 1 else "")
        + f"honest RTT: median {quantiles[50] / 1e3:.0f} µs  core σ {sigma / 1e3:.0f} µs  p99 {quantiles[99] / 1e3:.0f} µs  p99.9 {quantiles[99.9] / 1e3:.0f} µs  "
        f"-> deadline {deadline / 1e3:.0f} µs ({args.deadline_rule}); honest rejected {frr * 100:.1f}% "
        f"({results['honest']['correct']}/{results['honest']['checked']} digests correct)"
        + (f"; each chain against {base['chain_deadline_ns'] / 1e3:.0f} µs" if args.repeat > 1 else "")
        + (
            f"; arm wait median {base['arm_median_ns'] / 1e3:.0f} µs p99.9 {base['arm_p999_ns'] / 1e3:.0f} µs -> READY bound {base['arm_max_ns'] / 1e6:.1f} ms "
            f"(reloadable over PCIe at 50 GB/s: {base['arm_max_ns'] / 1e9 * 50:.2f} GB)"
            if args.drain
            else ""
        )
    )
    if args.dynamic:  # the dynamic chain has its own size and shape, so its own deadline
        honest = results["honest"]
        Bd, nd, _, _, _ = _dyn_shape(args, spec)
        try:
            dyn = _deadline_from(honest["dyn_rtt_ns"], args)
        except ValueError:
            raise SystemExit(
                "baseline: no dynamic round had its READY and replies; no dynamic deadline can be set"
            ) from None
        darms = [t for t in honest["dyn_arm_ns"] if math.isfinite(t)]
        dyn.update(
            {
                "B": Bd,
                "n": nd,
                "arm_median_ns": float(np.median(darms)),
                "arm_p999_ns": float(np.quantile(darms, 0.999)),
                "arm_max_ns": _dyn_ready_bound(args, darms),
                "in_flight": honest["dyn_in_flight"],
                "correct": honest["dyn_correct"],
                "checked": honest["dyn_checked"],
                **_dyn_schedule(args),
            }
        )
        if any(
            math.isfinite(t) for t in honest["dyn_live_rtt_ns"]
        ):  # the live chain: its own region size, its own deadline
            dyn["live"] = _deadline_from(honest["dyn_live_rtt_ns"], args)
            lv = dyn["live"]
            print(
                f"live chain RTT: median {lv['median_ns'] / 1e3:.0f} µs  core σ {lv['sigma_core_ns'] / 1e3:.0f} µs  -> deadline {lv['deadline_ns'] / 1e3:.0f} µs "
                f"({len(honest['dyn_live_rtt_ns'])} of {len(honest['dyn_rtt_ns'])} verifications had a live region)"
            )
        base["dyn"] = dyn
        print(
            f"dynamic RTT: median {dyn['median_ns'] / 1e3:.0f} µs  core σ {dyn['sigma_core_ns'] / 1e3:.0f} µs  -> deadline {dyn['deadline_ns'] / 1e3:.0f} µs; "
            f"pause wait median {dyn['arm_median_ns'] / 1e3:.0f} µs p99.9 {dyn['arm_p999_ns'] / 1e3:.0f} µs -> READY bound {dyn['arm_max_ns'] / 1e6:.1f} ms; "
            f"{dyn['in_flight']}/{len(honest['dyn_rtt_ns'])} paused mid-step; "
            + (
                f"{dyn['correct']}/{dyn['checked']} replayed exactly"
                if dyn["checked"]
                else "records stored, not replayed (no --replay)"
            )
        )
    honest = results["honest"]

    def fin(v):
        return [t for t in v if t is not None and math.isfinite(t)]

    # the responses the deadlines were set from, as live --save-baseline keeps them: a live run's timing claim is
    # sized from them (role "baseline": the static rounds ran back to back, so their pace is not live's)
    base["calibration_samples"] = {
        "static_rtt_ns": fin(honest["rtt_ns"]),
        "static_chain_rtt_ns": fin([t for v in honest["chains_ns"] for t in v]),
        "static_arm_ns": fin(honest["arm_ns"]) if args.drain else [],
        "dyn_rtt_ns": fin(honest["dyn_rtt_ns"]),
        "dyn_arm_ns": fin(honest["dyn_arm_ns"]),
        "live_rtt_ns": fin(honest["dyn_live_rtt_ns"]),
        "timing": args.timing,
        "span_s": honest["span_s"],
        "role": "baseline",
    }
    if fixed_claim:
        base["claim_calibration_samples"] = honest["claim_calibration_samples"]
    mem = _memory_bound(
        args, cfg, results, "honest", base.get("arm_max_ns", 0.0), base.get("dyn", {}).get("arm_max_ns", 0.0)
    )
    if mem:
        _print_memory(mem)
        base["memory"] = {k: v for k, v in mem.items() if k != "over"}
    Path(args.out or "baseline.json").write_text(json.dumps(base, indent=1))
    print("wrote", args.out or "baseline.json")


def _chain_stats(chains_ns, base: dict, args) -> dict:
    """Single-chain statistics beside the min-of-m ones: the drift detector's per-chain z, and the single-chain
    deadline every chain of a verification is held to (the same rule over single chains, which are slower and
    wider than their minimum). With --repeat 1 the chain is the verification: its deadline is the deadline."""
    from ..core.analysis import core_samples

    a = np.array([t for t in chains_ns if t is not None and math.isfinite(t)], dtype=np.float64)
    if len(a) == 0:
        return {}
    cc = core_samples(a)
    out = {"chain_median_ns": float(np.median(cc)), "chain_sigma_ns": float(cc.std())}
    out["chain_deadline_ns"] = (
        base["deadline_ns"] if max(1, getattr(args, "repeat", 1)) == 1 else _deadline_from(a, args)["deadline_ns"]
    )
    return out


def verifier_main(args):
    spec, keys = array_from_args(args)
    _shape(args, spec)  # the shape policy, before anything is loaded or sent
    _no_live_scratch(args)
    strategies = [s for s in args.strategies.split(",") if s]
    unknown = [s for s in strategies if s not in STRATS]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; known: {STRATS}")
    strategies = _negotiate(args, strategies)
    deadline = None
    chain_deadline = None
    if args.baseline:
        baseline = _load_baseline(args, spec)
        deadline = baseline["deadline_ns"]
        chain_deadline = baseline.get("chain_deadline_ns") or (deadline if max(1, args.repeat) == 1 else None)
        if chain_deadline is None:
            print(
                "verifier: WARNING: the baseline has no single-chain deadline (an older file); with --repeat > 1 only the minimum "
                "of each verification is judged against a deadline. Recalibrate to judge every chain",
                flush=True,
            )
    arm_max = (
        (args.arm_max_ms * 1e6 if args.arm_max_ms else (baseline.get("arm_max_ns", 0.0) if args.baseline else 0.0))
        if args.drain
        else 0.0
    )
    dyn_base = baseline.get("dyn") if args.baseline and args.dynamic else None
    if dyn_base and args.dyn_arm_max_ms:
        dyn_base = {**dyn_base, "arm_max_ns": args.dyn_arm_max_ms * 1e6}
    results, cfg = _challenge_loop(args, spec, keys, strategies, args.samples, args.warmup, _sample_verify(args))
    if arm_max:
        for strategy in strategies:  # late READY: the challenge fails whatever its RTT
            late = np.array(results[strategy]["arm_ns"]) > arm_max
            results[strategy]["late_ready"] = int(late.sum())
            results[strategy]["rtt_ns"] = [
                float("inf") if is_late else t for t, is_late in zip(results[strategy]["rtt_ns"], late)
            ]
    if dyn_base and dyn_base.get("arm_max_ns"):
        for strategy in strategies:  # the dynamic READY bound holds the same way
            late = np.array(results[strategy]["dyn_arm_ns"]) > dyn_base["arm_max_ns"]
            results[strategy]["dyn_late_ready"] = int(late.sum())
            results[strategy]["dyn_rtt_ns"] = [
                float("inf") if is_late else t for t, is_late in zip(results[strategy]["dyn_rtt_ns"], late)
            ]
    mems = {
        s: _memory_bound(args, cfg, results, s, arm_max, dyn_base.get("arm_max_ns", 0.0) if dyn_base else 0.0)
        for s in strategies
    }
    for strategy in strategies:  # over the memory bound: the verification fails
        m = mems[strategy]
        if m and m["over_bound"]:
            results[strategy]["over_bound"] = m["over_bound"]
    from ..core.analysis import core_samples

    h = np.array(results["honest"]["rtt_ns"], dtype=np.float64) if "honest" in results else np.array([])
    h = h[np.isfinite(h)]
    honest_median = float(np.median(h)) if len(h) else 0.0
    sigma_core = float(core_samples(h).std()) if len(h) else 0.0  # in-run σ
    for strategy in strategies:
        rtts_us = np.array(results[strategy]["rtt_ns"]) / 1e3
        a_fin = rtts_us[np.isfinite(rtts_us)] if np.isfinite(rtts_us).any() else rtts_us
        line = (
            f"{strategy:11s} median {np.median(a_fin):9.0f} µs  mean {a_fin.mean():9.0f}  std {a_fin.std():7.0f}  p1 {np.quantile(a_fin, 0.01):9.0f}  "
            f"p99 {np.quantile(a_fin, 0.99):9.0f}  digest ok: {results[strategy]['correct']}/{results[strategy]['checked']}"
        )
        if results[strategy]["unchecked"]:
            line += f" (UNCHECKED: {results[strategy]['unchecked']})"
        if deadline is not None:
            ok_mem = ~np.array(mems[strategy]["over"]) if mems[strategy] else np.ones(len(rtts_us), dtype=bool)
            # within: the minimum against the deadline (the statistic it was calibrated on) and every chain against the
            # single-chain deadline (a chain over it is slow beyond the honest single-chain tail, whatever the minimum)
            chains_ok = np.array(
                [chain_deadline is None or max(c, default=0) <= chain_deadline for c in results[strategy]["chains_ns"]],
                dtype=bool,
            )
            passed = int(np.sum((np.array(results[strategy]["rtt_ns"]) <= deadline) & ok_mem & chains_ok))
            line += f"  within deadline: {passed}/{len(rtts_us)}"
            if results[strategy].get("late_ready"):
                line += f"  (late READY: {results[strategy]['late_ready']})"
            if results[strategy].get("missing"):
                line += f"  (READY or reply missing: {results[strategy]['missing']})"
            if results[strategy].get("over_bound"):
                line += f"  (over the memory bound: {results[strategy]['over_bound']})"
            if strategy != "honest" and sigma_core:
                line += f"  gap {(np.median(a_fin) * 1e3 - honest_median) / sigma_core:+.0f}σ"
        print(line)
        if args.dynamic:
            d_all = np.array(results[strategy]["dyn_rtt_ns"]) / 1e3
            d = d_all[np.isfinite(d_all)] if np.isfinite(d_all).any() else d_all
            dl = (
                f"{'':11s} dynamic median {np.median(d):9.0f} µs  std {d.std():7.0f}  p99 {np.quantile(d, 0.99):9.0f}  "
                + (
                    f"replayed exactly: {results[strategy]['dyn_correct']}/{results[strategy]['dyn_checked']}"
                    if results[strategy]["dyn_checked"]
                    else f"stored for replay: {results[strategy].get('dyn_stored', 0)}"
                )
                + (f" (UNCHECKED: {results[strategy]['dyn_unchecked']})" if results[strategy]["dyn_unchecked"] else "")
                + (
                    f" (neither replayed nor kept: {results[strategy]['dyn_not_kept']})"
                    if results[strategy]["dyn_not_kept"]
                    else ""
                )
                + f"  paused mid-step: {results[strategy]['dyn_in_flight']}/{len(d_all)}"
            )
            if dyn_base:
                dl += f"  within deadline: {int(np.sum(np.array(results[strategy]['dyn_rtt_ns']) <= dyn_base['deadline_ns']))}/{len(d_all)}"
            if results[strategy].get("dyn_late_ready"):
                dl += f"  (late READY: {results[strategy]['dyn_late_ready']})"
            if results[strategy].get("dyn_faults"):
                dl += f"  (READY or reply missing, or a region no chain covers: {results[strategy]['dyn_faults']})"
            print(dl)
            if results[strategy]["dyn_live_rtt_ns"]:
                lv_all = np.array(results[strategy]["dyn_live_rtt_ns"]) / 1e3
                lv = lv_all[np.isfinite(lv_all)] if np.isfinite(lv_all).any() else lv_all
                ll = f"{'':11s} live chain median {np.median(lv):7.0f} µs  std {lv.std():7.0f}  p99 {np.quantile(lv, 0.99):9.0f}  chains: {len(lv_all)}/{len(d_all)}"
                if dyn_base and dyn_base.get("live"):
                    ll += f"  within deadline: {int(np.sum(np.array(results[strategy]['dyn_live_rtt_ns']) <= dyn_base['live']['deadline_ns']))}/{len(lv_all)}"
                print(ll)
        _print_memory(mems[strategy])
    if args.out:
        B_, _, r_, k_, _ = _shape(args, spec)
        recs = [
            {
                "prover": s,
                "B": B_,
                "n": args.n,
                "r": r_,
                "correct": results[s]["correct"] == results[s]["checked"],
                "incorrect_samples": results[s]["checked"] - results[s]["correct"],
                "unchecked_samples": results[s]["unchecked"],
                "samples_ns": results[s]["rtt_ns"],
                "chains_ns": results[s]["chains_ns"],
                "arm_ns": results[s]["arm_ns"],
                "missing": results[s]["missing"],
                **(
                    {
                        "dyn_samples_ns": results[s]["dyn_rtt_ns"],
                        "dyn_live_samples_ns": results[s]["dyn_live_rtt_ns"],
                        "dyn_arm_ns": results[s]["dyn_arm_ns"],
                        "dyn_correct": results[s]["dyn_correct"],
                        "dyn_checked": results[s]["dyn_checked"],
                        "dyn_unchecked": results[s]["dyn_unchecked"],
                        "dyn_in_flight": results[s]["dyn_in_flight"],
                        "dyn_words": results[s]["dyn_words"],
                    }
                    if args.dynamic
                    else {}
                ),
                **({"memory": {k: v for k, v in mems[s].items() if k != "over"}} if mems[s] else {}),
            }
            for s in strategies
        ]
        with open(args.out, "w") as f:
            json.dump(
                {
                    "meta": {
                        "device": "udp-verifier",
                        "size_bytes": cfg.nbytes,
                        "encoding": cfg.encoding,
                        "perm": cfg.perm,
                        "transport": f"udp {args.host}:{args.port}",
                        "samples": args.samples,
                        "manifest": args.manifest,
                        "deadline_ns": deadline,
                        "chain_deadline_ns": chain_deadline,
                        "k": k_,
                        "drain": bool(args.drain),
                        "repeat": max(1, args.repeat),
                        "setup_seconds": 0.0,
                        "decode_gbps": float("nan"),
                        "sample_verify": _sample_verify(args),
                        "dynamic": bool(args.dynamic),
                        "dyn_B": parse_bytes(args.dyn_B) if args.dynamic else None,
                        "dyn_n": (args.dyn_n or args.n) if args.dynamic else None,
                        "dyn_deadline_ns": dyn_base["deadline_ns"] if dyn_base else None,
                    },
                    "records": recs,
                },
                f,
            )
        print("wrote", args.out)


def baseline_problems(baseline: dict, args, spec) -> list[str]:
    """Why a baseline file cannot judge this run (empty: it can). A deadline is only as good as the
    conditions it was measured under: the challenge shape, --repeat, the drain (without it an honest round
    trip can hold an inference step, milliseconds, so its deadline is far too loose for a drained run), the
    timing mode (user space or kernel timestamps), and for a dynamic run the dynamic chain's block size and
    length and whether a live chain is timed. Each is recorded in the file and must match; a field an older
    file does not record is a mismatch (recalibrate). A drained run also needs the file's READY bound (or
    --arm-max-ms): without one the READY window, what a prover can reload before a challenge, is unbounded."""
    B, n, r, k, g = _shape(args, spec)
    want = {
        "B": B,
        "n": n,
        "r": r,
        "k": k,
        "g": g,
        "repeat": max(1, getattr(args, "repeat", 1)),
        "drain": bool(getattr(args, "drain", False)),
        "timing": getattr(args, "timing", "user"),
    }
    have = {
        **{key: baseline.get(key) for key in want},
        "g": baseline.get("g", 1),
        "repeat": baseline.get("repeat", 1),
        "timing": baseline.get("timing") or (baseline.get("calibration_samples") or {}).get("timing"),
    }
    out = [
        f"{key} {'not recorded' if have[key] is None else have[key]} (this run: {want[key]})"
        for key in want
        if have[key] != want[key]
    ]
    explicit = bool(getattr(args, "arm_max_ms", None))
    if want["drain"] and not baseline.get("arm_max_ns") and not explicit:
        out.append(
            "no READY bound (arm_max_ns): a drained run's READY window would be unbounded (or pass --arm-max-ms)"
        )
    if getattr(args, "dynamic", False):
        dyn = baseline.get("dyn")
        if not dyn:
            out.append("no dynamic section (this run challenges dynamic memory; calibrate with --dynamic)")
        else:
            Bd, nd = _dyn_shape(args, spec)[:2]
            if (dyn.get("B"), dyn.get("n")) != (Bd, nd):
                out.append(
                    f"dynamic chain of {dyn.get('n')} links over {dyn.get('B')}-byte blocks (this run: --dyn-n {nd}, --dyn-B {Bd})"
                )
            live = bool(getattr(args, "dyn_live", False))
            if bool(dyn.get("live")) != live:
                out.append(
                    "no live-chain deadline (this run times a live chain)"
                    if live
                    else "a live-chain deadline (this run has no live chain: --with-dyn-live times one)"
                )
            if not dyn.get("arm_max_ns") and not getattr(args, "dyn_arm_max_ms", None):
                out.append(
                    "no dynamic READY bound (dyn.arm_max_ns): the freeze's READY window would be unbounded (or pass --dyn-arm-max-ms)"
                )
            # a baseline role's dynamic rounds used to follow the static drain at once: frozen at the start of a step,
            # their pause waits and round trips are not those of a freeze anywhere in a step. A live --save-baseline
            # (it keeps its calibration samples) always scheduled them at their own random times
            if dyn.get("schedule", DYN_SCHEDULE if baseline.get("calibration_samples") else None) != DYN_SCHEDULE:
                out.append(
                    "dynamic rounds calibrated right after the static drain (an older baseline-role file): its dynamic deadline "
                    "and READY bound are not those of a freeze at a random time"
                )
    return out


def _load_baseline(args, spec) -> dict:
    """The --baseline file, refused unless it was measured under this run's conditions (`baseline_problems`)."""
    b = json.loads(Path(args.baseline).read_text())
    problems = baseline_problems(b, args, spec)
    if problems:
        raise SystemExit(
            f"baseline {args.baseline} was measured under other conditions:\n  "
            + "\n  ".join(problems)
            + "\nRecalibrate at this configuration (the baseline role, or live --save-baseline)"
        )
    return b


RELOAD_BYTES_PER_S = 50e9  # what a PCIe Gen5 x16 link moves at best: the reload slack is printed at this rate


def _memory_bound(
    args, cfg, results: dict, strategy: str, ready_ns: float = 0.0, dyn_ready_ns: float = 0.0
) -> dict | None:
    """The memory claim of each verification (`_memory_claim`, the check the live verifier makes): the GPU's
    capacity (the verifier's own figure, `--gpu-type` or `--hbm-bytes`) minus the static array's credited
    size (the verifier's own manifest; `credited_static_bytes`) minus the dynamic region the prover reported
    at the freeze, outside its live region. What is left is unchallenged or uncredited: the prover's scratch,
    which no chain covers, and memory the chains cover but the claim does not credit (a bundle's Θ', the live
    region, the bits of each word a prover can pack away for free). A total the GPU cannot hold fails whatever the bound; with `--max-scratch` a verification whose remainder is
    over the bound fails. Every word counted is challenged (filler is VDE-encoded under distinct keys, live
    data is replayed), so declaring memory that is not held fails challenges; the READY windows give the slack
    a prover could reload just in time (the dynamic round comes a random time after the static ones, as in live:
    no static ARM announces it)."""
    from .hardware import capacity_bytes

    cap = capacity_bytes(
        getattr(args, "gpu_type", None), parse_bytes(args.hbm_bytes) if getattr(args, "hbm_bytes", None) else None
    )
    if cap is None:
        if getattr(args, "max_scratch", None):
            raise SystemExit("--max-scratch needs the verifier's capacity figure: --gpu-type or --hbm-bytes")
        return None
    _no_live_scratch(args)
    static = _static_credit(args, cfg)
    bits = _dyn_credit_bits(args)
    Bd = parse_bytes(args.dyn_B) if getattr(args, "dyn_B", None) else None
    r = results[strategy]
    dyn = [w * 8 for w in r["dyn_words"]] if r["dyn_words"] else [0] * len(r["rtt_ns"])
    live = [w * 8 for w in r.get("dyn_live_words") or []] or [0] * len(dyn)
    bound = parse_bytes(args.max_scratch) if getattr(args, "max_scratch", None) else None
    enc = _dyn_encoded(args)
    claims = [
        _memory_claim(cap, static, dyn_bytes, live_bytes, bound, bits, Bd, static_array=cfg.nbytes, live_credited=enc)
        for dyn_bytes, live_bytes in zip(dyn, live)
    ]
    unchecked = [c["unchecked_bytes"] for c in claims]

    def rel(ns):
        return ns / 1e9 * RELOAD_BYTES_PER_S  # bytes a 50 GB/s link moves in a READY window

    return {
        "capacity_bytes": cap,
        "static_bytes": static,
        "static_credited_bytes": static,
        "static_array_bytes": cfg.nbytes,
        "static_bundle": _is_bundle(args),
        "dynamic_bytes_max": max(dyn),
        "dynamic_bytes_min": min(dyn),
        "dynamic_credited_bytes_max": max(c["dynamic_bytes"] for c in claims),
        "dynamic_credit_bits": bits,
        "dynamic_encoded": enc,
        "live_bytes_max": max(live),
        "unchecked_bytes_max": max(unchecked),
        "unchecked_bytes_min": min(unchecked),
        "bound_bytes": bound,
        "over_bound": sum(c["over"] for c in claims),
        "impossible": sum(c["impossible"] for c in claims),
        "verifications": len(claims),
        "over": [c["over"] for c in claims],
        "reload_static_bytes": rel(ready_ns),
        "reload_dynamic_bytes": rel(dyn_ready_ns),
    }


def _dyn_encoded(args) -> bool:
    return bool(getattr(args, "dynamic", False) and getattr(args, "dyn_encoded", False))


def _dyn_credit_bits(args) -> int:
    """The bits of each dynamic word the memory claim credits: its filler's incompressible bits (`dyn_word_bits`),
    or none without a live chain (the default; --with-dyn-live adds it). Live KV is checked (replayed) but never credited, and only the
    live chain's READY size says how much of the region is live: without it the live KV cannot be told apart
    from the filler around it, so none of the region is credited. An encoded region (--dyn-encoded, v0.17) is VDE
    words throughout, live data included: 62/64 of every word, with or without the live chain."""
    if _dyn_encoded(args):
        return VDE_WORD_BITS
    if getattr(args, "dynamic", False) and not getattr(args, "dyn_live", False):
        return 0
    return dyn_word_bits(getattr(args, "dyn_dtype", None))


def _no_live_scratch(args) -> None:
    """--max-scratch needs a credited dynamic region: refused without the live chain (`_dyn_credit_bits`), unless the
    region is encoded (--dyn-encoded: credited without telling its live part apart)."""
    if (
        getattr(args, "dynamic", False)
        and not getattr(args, "dyn_live", False)
        and getattr(args, "max_scratch", None)
        and not _dyn_encoded(args)
    ):
        raise SystemExit(
            "--max-scratch with a plain dynamic region and no live chain: without the live chain the live KV cannot be told "
            "apart from the rest of the region, so none of it is credited and every verification would be over the bound "
            "(--with-dyn-live credits the region outside the live one; a prover holding its region encoded, with "
            "--dyn-encoded, is credited whole)"
        )


DYN_DTYPES = ("bfloat16", "float16", "float32", "float64", "int8", "int16", "int32", "int64")


def dyn_word_bits(dtype: str | None = None) -> int:
    """Bits of a dynamic filler word a prover cannot pack away, for buffers of this dtype (the verifier's
    --dyn-dtype, never the prover's report). Free dynamic memory holds VDE words (bits 62 and 63 always
    zero) masked for their buffer's dtype (`memory_accounting.prover.dynfill.mask_for`: the top exponent bit of every
    floating-point element cleared, so free memory never holds NaN or Inf). A bf16 or fp16 buffer's mask also
    clears bits 14, 30 and 46, so its filler packs losslessly into 59 bits a word; float32 into 61; float64
    and integer buffers into 62. Default bfloat16, the least of these: crediting more than the incompressible
    bits would let a prover keep the difference (about 4.5 GiB of a 58 GiB vLLM pool) for itself."""
    from ..prover.dynfill import mask_for

    return (((1 << VDE_WORD_BITS) - 1) & mask_for(dtype or "bfloat16")).bit_count()


def _memory_claim(
    cap: int,
    static: int,
    dyn_bytes: int,
    live_bytes: int,
    bound: int | None,
    dyn_bits: int = 64,
    block_bytes: int | None = None,
    static_array: int | None = None,
    live_credited: bool = False,
) -> dict:
    """One verification's memory claim. The sizes are the prover's READY figures; the record check binds
    them to what the replay covered. Credited ("dynamic_bytes", so capacity − static − dynamic = unchecked adds
    up): the dynamic region outside the live region, in whole --dyn-B blocks (the chain reads whole blocks: the
    words past the last one are read by nothing), at dyn_bits/64 of its size (`dyn_word_bits`). The live region
    is checked (its chain, its replay) but never credited: its KV comes from tokens the prover chooses, and
    fabricated or repeated KV may compress. "dynamic_region_bytes" and "live_region_bytes" are the READY
    sizes. A region the GPU cannot hold beside the static array (its size, static_array; default the credited
    size), or a live region larger than the pool it lies in, is impossible: it fails whatever the bound.
    live_credited (an encoded region, --dyn-encoded): the live region is VDE words like the rest and is credited too."""
    read = dyn_bytes // block_bytes * block_bytes if block_bytes else dyn_bytes
    outside = read if live_credited else max(0, read - live_bytes)
    if block_bytes:
        outside = outside // block_bytes * block_bytes
    credited = outside * dyn_bits // 64
    unchecked = cap - static - credited
    impossible = cap - (static if static_array is None else static_array) - dyn_bytes < 0 or live_bytes > dyn_bytes
    return {
        "dynamic_bytes": credited,
        "dynamic_region_bytes": dyn_bytes,
        "live_region_bytes": live_bytes,
        "unchecked_bytes": unchecked,
        "impossible": impossible,
        "over": bool(impossible or (bound is not None and unchecked > bound)),
    }


def _print_memory(memory: dict | None) -> None:
    if not memory:
        return
    G = 1 << 30
    dyn = (
        f"{memory['dynamic_bytes_max'] / G:.2f}"
        if memory["dynamic_bytes_max"] == memory["dynamic_bytes_min"]
        else f"{memory['dynamic_bytes_min'] / G:.2f} to {memory['dynamic_bytes_max'] / G:.2f}"
    )
    sa = memory.get("static_array_bytes", memory["static_bytes"])
    line = (
        f"memory: capacity {memory['capacity_bytes'] / G:.2f} GiB (verifier's figure); challenged: static {memory['static_bytes'] / G:.2f} GiB"
        + (
            (
                f" (none of the {sa / G:.2f} GiB array credited: its image is the prover's own, --trust-prover-image)"
                if not memory["static_bytes"]
                else f" (the credited part of the {sa / G:.2f} GiB array: {_static_credit_note(memory.get('static_bundle', False))})"
            )
            if memory["static_bytes"] != sa
            else ""
        )
        + f" + dynamic region {dyn} GiB"
        + (
            (
                " (none credited: no live chain, so its live KV cannot be told apart; --with-dyn-live adds one)"
                if not memory.get("dynamic_credit_bits", 64)
                else f" (credited {memory['dynamic_credited_bytes_max'] / G:.2f} GiB at most: encoded throughout, the live region"
                + (f" (up to {memory.get('live_bytes_max', 0) / G:.3f} GiB)" if memory.get("live_bytes_max") else "")
                + " included"
                + (f", at {memory['dynamic_credit_bits']}/64 (two bits of every VDE word are zero)")
                + ")"
                if memory.get("dynamic_encoded")
                else f" (credited {memory['dynamic_credited_bytes_max'] / G:.2f} GiB at most: outside the live region"
                + (
                    f", up to {memory.get('live_bytes_max', 0) / G:.3f} GiB, checked but not credited"
                    if memory.get("live_bytes_max")
                    else ""
                )
                + (
                    f", at {memory['dynamic_credit_bits']}/64 (the dtype mask zeroes the rest)"
                    if memory.get("dynamic_credit_bits", 64) != 64
                    else ""
                )
                + ")"
            )
            if memory["dynamic_bytes_max"]
            else ""
        )
        + f"; unchallenged or uncredited at most {memory['unchecked_bytes_max'] / G:.2f} GiB "
        f"({100 * memory['unchecked_bytes_max'] / memory['capacity_bytes']:.1f}%)"
    )
    if memory.get("impossible"):
        line += f"; IMPOSSIBLE totals (more than the GPU holds, or a live region larger than its pool): {memory['impossible']}/{memory['verifications']}"
    if memory["bound_bytes"] is not None:
        line += f"; bound {memory['bound_bytes'] / G:.2f} GiB: {memory['verifications'] - memory['over_bound']}/{memory['verifications']} within"
    if memory["reload_static_bytes"] or memory["reload_dynamic_bytes"]:
        line += (
            f"; READY windows allow reloading up to {memory['reload_static_bytes'] / 1e9:.2f} GB (static) and "
            f"{memory['reload_dynamic_bytes'] / 1e9:.2f} GB (dynamic) at 50 GB/s"
        )
    print(line)


def _deadline_from(rtt_ns, args) -> dict:
    """The baseline statistics and deadline (same rule as the baseline role), over the round trips that came.
    ValueError when none came: there is nothing to set a deadline from."""
    from ..core.analysis import core_samples

    rtts = np.array([t for t in rtt_ns if t is not None], dtype=np.float64)
    rtts = rtts[np.isfinite(rtts)]
    if len(rtts) == 0:
        raise ValueError("no round trip to set a deadline from")
    core = core_samples(rtts)
    sigma = float(core.std())
    med = float(np.median(core))
    if args.deadline_rule == "sigma":
        deadline = med + max(args.sigmas * sigma, args.margin * med)
    else:
        deadline = float(np.quantile(rtts, 0.999)) * (1 + args.margin)
    return {
        "median_ns": med,
        "sigma_core_ns": sigma,
        "p99_ns": float(np.quantile(rtts, 0.99)),
        "deadline_ns": deadline,
        "deadline_rule": args.deadline_rule,
        "sigmas": args.sigmas,
        "margin": args.margin,
        "samples": len(rtts),
        "honest_reject_rate": float(np.mean(rtts > deadline)),
    }


# chain -> its calibration responses
CLAIM_SAMPLES = {
    "static": "static_chain_rtt_ns",
    "dynamic": "dyn_rtt_ns",
    "live": "live_rtt_ns",
}
FIXED_CLAIM_SAMPLE_POLICY = "fixed-cap-attempts-v1"


def _fixed_claim_options(args) -> bool:
    """Window claims use fixed settings, supplied explicitly or selected from warmup."""
    cap = getattr(args, "claim_cap_us", None)
    window = getattr(args, "claim_window", None)
    if cap is not None and (
        isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0
    ):
        raise SystemExit("--claim-cap-us must be a finite positive number")
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1):
        raise SystemExit("--claim-window must be a positive integer")
    if getattr(args, "claim_mode", "window") != "window":
        if cap is not None or window is not None:
            raise SystemExit("--claim-cap-us and --claim-window require --claim-mode window")
        return False
    enabled = (
        cap is not None
        or window is not None
        or getattr(args, "claim_F", None) is not None
        or getattr(args, "role", None) == "baseline"
    )
    if not enabled:
        return False
    factor = getattr(args, "claim_M_factor", 1.0)
    if cap is None and (not math.isfinite(factor) or factor < 1):
        raise SystemExit("automatic cap selection requires finite --claim-M-factor >= 1")
    if getattr(args, "claim_drift_us", 0.0) != 0:
        raise SystemExit("the fixed-cap test requires --claim-drift-us 0")
    return True


def _claim_values(ev: dict, chain: str, *, penalize_ready: bool = True) -> list:
    """Per-chain RTTs in ns, retaining missing READY/reply attempts as infinity.

    The fixed-cap mean test uses received RTTs; READY deadline penalties remain separate.
    The default preserves the legacy late-READY scoring convention.
    """
    if chain == "static":
        if not penalize_ready and any(
            c.get("rtt_ns") is None and c.get("missing") not in ("reply", "READY") for c in ev.get("challenges", [])
        ):
            raise ValueError("a static claim outcome has neither an RTT nor a missing-reply marker")
        clip = penalize_ready and bool(ev.get("ready_clipped"))
        values = [
            c["rtt_ns"] if c["rtt_ns"] is not None and not clip else math.inf
            for c in ev.get("challenges", [])
            if c["rtt_ns"] is not None or c.get("missing") in ("reply", "READY")
        ]
    else:
        dyn = ev.get("dyn") or {}
        fault = dyn.get("fault") or ""
        if fault.startswith(FAULT_NO_DYN_READY):
            return [math.inf]
        clip = penalize_ready and bool(dyn.get("ready_clipped"))
        if chain == "live":
            if fault in (FAULT_NO_LIVE_REPLY, FAULT_NO_DYN_REPLY):
                return [math.inf]
            dyn = dyn.get("live") or {}
        elif fault == FAULT_NO_DYN_REPLY:
            return [math.inf]
        values = ([math.inf] if clip else [dyn["rtt_ns"]]) if dyn.get("rtt_ns") is not None else []
    if not penalize_ready:
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                raise ValueError("a claim RTT must be nonnegative, or infinity for a missing reply")
    return values


def _new_claim_samples(args) -> dict:
    """Raw attempt scores, stored separately from the finite-only deadline calibration."""
    return {
        "policy": FIXED_CLAIM_SAMPLE_POLICY,
        "cap_us": getattr(args, "claim_cap_us", None),
        "window": getattr(args, "claim_window", None),
        "setups": {},
        "reply_timeout_s": _reply_timeout(args),
        **{key: [] for key in CLAIM_SAMPLES.values()},
    }


def _append_claim_samples(samples: dict, ev: dict, chains) -> None:
    for chain in chains:
        values = _claim_values(ev, chain, penalize_ready=False)
        samples[CLAIM_SAMPLES[chain]].extend(None if math.isinf(x) else x for x in values)


def _freeze_claim_setup(samples: dict, pilot: dict, args, chain: str, L: int, delta_c: float, n_cal: int) -> None:
    from .claim_setup import automatic_claim_setup

    setup = automatic_claim_setup(pilot[CLAIM_SAMPLES[chain]], args, L, delta_c, n_cal)
    samples["setups"][chain] = setup
    chosen = (
        f"c = {setup['cap_us']:g} µs, N = {setup['window']}"
        if setup.get("cap_us") is not None and setup.get("window") is not None
        else setup["reason"]
    )
    print(f"timing claim {chain} setup ({setup['pilot_count']} warmup attempts): {chosen}", flush=True)


def _fixed_claim_setup(b: dict, args, chain: str) -> dict:
    samples = b.get("claim_calibration_samples") or {}
    if samples.get("policy") != FIXED_CLAIM_SAMPLE_POLICY:
        raise ValueError(
            "fixed-cap claims need a baseline retaining all challenge attempts and independent settings; recalibrate"
        )
    setup = (samples.get("setups") or {}).get(chain)
    if setup is None and samples.get("cap_us") is not None and samples.get("window") is not None:
        setup = {"cap_us": samples["cap_us"], "window": samples["window"], "pilot_count": 0}
    if setup is None:
        raise ValueError("baseline has no independently selected cap and window for this chain; recalibrate")
    if samples.get("reply_timeout_s") != _reply_timeout(args):
        raise ValueError(
            "fixed-cap baseline reply_timeout_s differs from this run; recalibrate at the selected configuration"
        )
    for key, expected in (
        ("cap_us", getattr(args, "claim_cap_us", None)),
        ("window", getattr(args, "claim_window", None)),
    ):
        if expected is not None and setup.get(key) != expected:
            raise ValueError(
                f"fixed-cap baseline {key} differs from this run; recalibrate at the selected configuration"
            )
    return setup


def _fixed_claim_samples(b: dict, args, chain: str) -> list:
    _fixed_claim_setup(b, args, chain)
    samples = b["claim_calibration_samples"]
    values = [math.inf if value is None else value for value in samples.get(CLAIM_SAMPLES[chain], [])]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x) or x < 0 for x in values):
        raise ValueError("fixed-cap baseline contains an invalid timing outcome")
    return values


# Δ_c of a dynamic or live chain: the least extra time of one link whose --dyn-B block sits in host memory and is
# read over PCIe, measured per link on the H200 (scripts/bench_dyn_dram.py, the dyn_dram cheat's own path, under
# vLLM): 64 KiB 0.3 to 1.9 µs, 256 KiB 3.5 to 5.1 µs, 1 MiB 17 to 18.5 µs. The least of each, interpolated
# linearly in the block size between them and in proportion outside. Erring small is the safe side: a smaller
# Δ_c claims more (it covers slower-to-catch cheats) and needs more chains. (Before v0.14: 2 µs whatever the
# block, above the 64 KiB minimum.)
DYN_DELTA_C_US = ((64 << 10, 0.3), (256 << 10, 3.5), (1 << 20, 17.0))


def dyn_delta_c_us(block_bytes: int) -> float:
    """The measured least Δ_c (µs) of a dynamic link reading a block of this size from host memory."""
    pts = DYN_DELTA_C_US
    if block_bytes <= pts[0][0]:
        return pts[0][1] * block_bytes / pts[0][0]
    if block_bytes >= pts[-1][0]:
        return pts[-1][1] * block_bytes / pts[-1][0]
    for (b0, d0), (b1, d1) in itertools.pairwise(pts):
        if b0 <= block_bytes <= b1:
            return d0 + (d1 - d0) * (block_bytes - b0) / (b1 - b0)
    raise AssertionError("unreachable")


def _cusum_step(S: float, rtts_ns: list, base: dict, k: float) -> float | None:
    """The drift CUSUM after one verification's chains: S = max(0, S + Σ(z_i - k)), z per chain against the
    single-chain calibration, capped at 3 so a stall is the deadline's business. A baseline without chain
    statistics (an old file) is of the min-of-m statistic, so the verification's minimum is what is compared
    with it: any one chain (the first was used) is above the minimum by the min-of-m gain and drifts falsely.
    Missing replies (None) are the fault alert's business. None when the baseline has no σ."""
    rtts_ns = [t for t in rtts_ns if t is not None]
    if not rtts_ns:
        return S
    med, sig = base.get("chain_median_ns"), base.get("chain_sigma_ns")
    if not sig:
        med, sig, rtts_ns = base.get("median_ns"), base.get("sigma_core_ns"), [min(rtts_ns)]
        if not sig:
            return None
    for rtt in rtts_ns:
        S = max(0.0, S + min((rtt - med) / sig, 3.0) - k)
    return S


CHECK_BACKLOG = 10_000  # static digest checks that may wait (a few ms of CPU each)
RECORD_BACKLOG = 1_000  # dynamic record checks that may wait (a fetch and a replay each)
STOP_CHECKS_S = 8.0  # at stop, how long the checks still waiting may run before the rest is reported unchecked
# (all of the stop well inside the 20 s scripts/demo.sh gives the live verifier after SIGINT)


def _pool_round_weights(static_bytes, dynamic_bytes, dynamic_block_bytes, repeat):
    """Integer round weights giving S:D main challenge counts even with repeated static chains."""
    if dynamic_bytes is None:
        raise ValueError("pool-bytes scheduling needs a throughput manifest with the dynamic pool size")
    challengeable = dynamic_bytes // dynamic_block_bytes * dynamic_block_bytes
    if min(static_bytes, challengeable, repeat) <= 0:
        raise ValueError("pool-bytes scheduling needs a full challenge block in each pool and a positive repeat")
    return static_bytes, challengeable * repeat


def _pick_pool_round(weights):
    static, dynamic = weights
    return "static" if _SYSRAND.randrange(static + dynamic) < static else "dynamic"


def live_main(args):
    """Continuous verification with a live page: one event per round (what was sent, what came back, what the
    verifier expected, timing against the deadline), prover-side counters, and an optional JSONL log.

    Two kinds of round, at independent random times: static rounds (the static chains) and, with --dynamic,
    dynamic rounds (a freeze, then the chain over all dynamic memory and optionally the live chain). The CSPRNG
    draws an exponential wait and independently selects the kind according to --round-mix. (Before, every verification was static ARM, static chains, then the freeze: the static ARM
    announced the freeze, and a prover could start reloading dynamic memory there.) Each round is its own
    `verification` event: "round" says which; a dynamic one carries "dyn" (as before) and no static challenges."""
    import queue as _queue
    import signal
    import threading
    from collections import OrderedDict

    from .ui.server import EventHub
    from .ui.server import start as start_ui

    skip_checks = getattr(args, "skip_response_checks", False)
    fixed_claim = _fixed_claim_options(args)
    if not 0 < args.check_fraction <= 1:
        raise SystemExit("--check-fraction must be in (0, 1]")
    dynamic_check_fraction = getattr(args, "dynamic_check_fraction", 1.0)
    if not 0 < dynamic_check_fraction <= 1:
        raise SystemExit("--dynamic-check-fraction must be in (0, 1]")
    if (
        ((not fixed_claim or args.claim_cap_us is None) and args.claim_M_factor < 1)
        or args.claim_sigma_margin < 0
        or args.claim_drift_us < 0
    ):
        raise SystemExit(
            "--claim-M-factor must be at least 1 (a clip below L·Δ_c cuts a chain cheated on every read short), "
            "--claim-sigma-margin and --claim-drift-us at least 0"
        )
    spec, keys = array_from_args(args)
    strategies = [s for s in args.strategies.split(",") if s]
    unknown = [s for s in strategies if s not in STRATS]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; known: {STRATS}")
    monitor = args.mode == "monitor"
    strategies = _negotiate(args, strategies, posture=monitor)  # monitor: the postures the prover's simulator offers
    B, n, r, k, g = _shape(args, spec)  # the shape policy first: nothing is loaded or sent below it
    ver, cfg = _make_verifier(args, spec, keys)
    nb = cfg.nbytes // B
    repeat = max(1, args.repeat)
    drain = bool(args.drain)
    # monitor mode: the verifier behaves as deployed. Every challenge is a plain challenge (wire id honest,
    # which the prover's posture overrides), the verifier never learns which strategy answered, and it
    # alerts on a failed digest, a missed deadline, a late READY or a timing drift. The page's posture
    # control is a demo channel to the prover's simulator, not part of the protocol.
    hub = EventHub()
    hub.allowed["strategies"] = list(strategies)  # research: toggles; monitor: postures the simulator offers
    hub.control["strategies"] = list(strategies) if not monitor else ["honest"]
    try:
        start_ui(hub, args.ui_port, args.ui_bind, allowed_hosts=args.ui_allow_host, token=args.ui_token)
    except ValueError as e:
        raise SystemExit(f"--ui-token: {e}")
    print(f"live page on http://{args.ui_bind}:{args.ui_port}/  (prover {args.host}:{args.port})", flush=True)
    log = open(args.out, "a") if args.out else None  # noqa: SIM115 (open for the whole run, closed at its end)
    log_lock = threading.Lock()
    log_open = [log is not None]

    def write_log(e: dict) -> None:
        """One line of the run's log. The timing loop and both checking threads write here: one lock, so lines
        never interleave, and nothing is written once the log is closed."""
        if log is None:
            return
        line = json.dumps(e) + "\n"
        with log_lock:
            if log_open[0]:
                log.write(line)
                log.flush()

    def publish(e: dict, logged: bool = True) -> dict:
        e = hub.publish(e)  # the published copy carries the time and sequence
        if logged:
            write_log(e)
        return e

    def alert(a: dict) -> None:
        """Publish an alert and keep it in the run's log."""
        publish(a)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10.0)
    if args.kernel_ts:
        _enable_kernel_ts(sock)
    rng = np.random.default_rng(args.seed + 1)
    dynamic = bool(args.dynamic)
    dyn_shape = _dyn_shape(args, spec) if dynamic else None
    mix = getattr(args, "round_mix", "equal")
    weights = None
    schedule = {
        "mix": mix, "static_interval_s": args.interval,
        "dynamic_interval_s": args.interval if dynamic else None, "independent": True,
    }
    if mix == "pool-bytes":
        if not dynamic or not args.manifest:
            raise SystemExit("--round-mix pool-bytes requires --dynamic and a throughput manifest")
        weights = _pool_round_weights(nb * B, spec.benchmark_dynamic_bytes, dyn_shape[0], repeat)
        static_bytes, dynamic_weight = weights
        dynamic_bytes = dynamic_weight // repeat
        total_weight = sum(weights)
        gap = args.interval / 2
        schedule.update(
            static_bytes=static_bytes, dynamic_bytes=dynamic_bytes,
            aggregate_wait_mean_s=gap,
            static_challenge_fraction=static_bytes / (static_bytes + dynamic_bytes),
            dynamic_challenge_fraction=dynamic_bytes / (static_bytes + dynamic_bytes),
            static_round_probability=static_bytes / total_weight,
            dynamic_round_probability=dynamic_weight / total_weight,
            static_interval_s=gap * total_weight / static_bytes,
            dynamic_interval_s=gap * total_weight / dynamic_weight,
        )
        print(f"Challenge mix: {static_bytes:,} static bytes, {dynamic_bytes:,} dynamic bytes; "
              f"target {schedule['static_challenge_fraction']:.1%} static, "
              f"{schedule['dynamic_challenge_fraction']:.1%} dynamic main challenges", flush=True)
    if dynamic and not skip_checks and not args.replay and not args.evidence_dir:
        msg = (
            "dynamic records are neither replayed (--replay) nor kept (--evidence-dir): only the dynamic chains' timing and the "
            "records' binding are checked, never the region's content"
        )
        print(f"verifier: WARNING: {msg}", flush=True)
        alert({"type": "alert", "kind": "evidence", "id": 0, "detail": msg})
    if dynamic and args.dyn_jitter_ms:
        print(
            "live: dynamic rounds come at their own random times, independent of the static rounds, so a freeze lands anywhere "
            "in a step without --dyn-jitter-ms; it is not used here (it still applies to the baseline and verifier roles)",
            flush=True,
        )
    from .hardware import capacity_bytes  # the memory claim: the verifier's own capacity figure

    cap = capacity_bytes(args.gpu_type, parse_bytes(args.hbm_bytes) if args.hbm_bytes else None)
    mem_bound = parse_bytes(args.max_scratch) if args.max_scratch else None
    if mem_bound is not None and cap is None:
        raise SystemExit("--max-scratch needs the verifier's capacity figure: --gpu-type or --hbm-bytes")
    _no_live_scratch(args)
    if skip_checks:
        print("Performance benchmark: response correctness checks disabled; static and dynamic rounds are still timed.", flush=True)
    static_credit = _static_credit(args, cfg)  # the page's "static" is the credited size, so its sum adds up
    # the dynamic region's credit: its filler's incompressible bits (none without a live chain)
    dyn_bits = _dyn_credit_bits(args)
    dyn_enc = _dyn_encoded(args)  # encoded (v0.17): every word a VDE word, the live region credited too
    memory = (
        {
            "capacity_bytes": cap,
            "static_bytes": static_credit,
            "static_credited_bytes": static_credit,
            "static_array_bytes": cfg.nbytes,
            "timing_only": skip_checks,
            "bound_bytes": mem_bound,
            "gpu_type": args.gpu_type,
            "dyn_dtype": args.dyn_dtype,
            "dyn_credit_bits": dyn_bits,
            "dyn_encoded": dyn_enc,
        }
        if cap
        else None
    )
    if cap:
        print(
            f"memory claim: static array {cfg.nbytes / (1 << 30):.2f} GiB credited as {static_credit / (1 << 30):.2f} GiB "
            + (
                "(sizes from the unchecked benchmark layout)"
                if skip_checks
                else "(none: the image is the prover's own, --trust-prover-image)"
                if not static_credit
                else (
                    f"({_static_credit_note(_is_bundle(args))})"
                    if static_credit != cfg.nbytes
                    else "(64/64: --encoding none)"
                )
            )
            + (
                (
                    "; dynamic region not credited (no live chain, so its live KV cannot be told apart; --with-dyn-live adds one)"
                    if not dyn_bits
                    else f"; dynamic region encoded and credited whole, live region included, in whole {dyn_shape[0]}-byte blocks, at {dyn_bits}/64"
                    if dyn_enc
                    else f"; dynamic region credited outside its live region, in whole {dyn_shape[0]}-byte blocks, at {dyn_bits}/64 "
                    f"({args.dyn_dtype} filler); the live region is checked, not credited"
                )
                if dynamic
                else ""
            ),
            flush=True,
        )
    hub.set_meta(
        memory=memory,
        host=f"{args.host}:{args.port}",
        B=B,
        n=n,
        r=r,
        k=k,
        g=g,
        repeat=repeat,
        drain=drain,
        strategies=strategies,
        mode=args.mode,
        cusum_k=args.cusum_k,
        cusum_h=args.cusum_h,
        dynamic=dynamic,
        dyn_B=dyn_shape[0] if dynamic else None,
        dyn_n=dyn_shape[1] if dynamic else None,
        evidence_dir=None if skip_checks else args.evidence_dir,
        manifest=args.manifest,
        size_bytes=cfg.nbytes,
        interval_s=args.interval,
        response_checks=not skip_checks,
        state="calibrating" if not args.baseline else "running",
        timing=args.timing,
        confirm_misses=bool(args.confirm_misses),
        rounds=schedule,
        check_lag_s=args.check_lag_s,
        record_budget_s=_record_budget(args),
        evidence_saving=True if args.evidence_dir and not skip_checks else None,
        evidence_max_total=parse_bytes(str(args.evidence_max_total)) if args.evidence_dir else None,
        replay=bool(args.replay),
    )

    # Checks run on two worker threads, off the timing path: the static digests (a CPU recomputation each) on
    # one, the dynamic records (a TCP fetch and a replay each) on the other, so a slow record never holds the
    # digest checks back. Each queue is bounded; what waits is tracked (`pending`, oldest first) so the timing
    # loop can alert when checking falls behind, and whatever is still unchecked at the end is reported.
    pending = {"static": OrderedDict(), "dynamic": OrderedDict()}  # verification id -> time queued
    pend_lock = threading.Lock()
    queues = {"static": _queue.Queue(maxsize=CHECK_BACKLOG), "dynamic": _queue.Queue(maxsize=RECORD_BACKLOG)}
    check_rng = np.random.default_rng(args.seed + 7)  # which static rounds are digest-checked (--check-fraction)
    dynamic_check_rng = np.random.default_rng(args.seed + 8)
    stop_checks = threading.Event()
    WHAT = {"static": "digest check", "dynamic": "record check"}
    current = {"id": 0}  # the latest round's id: alerts about the run as a whole carry it

    def static_check(vid, chs):
        """The static digests of one round over the verifier's own array."""
        res = []
        for ch, lo, hi in chs:
            elo, ehi, _ = ver.expected(ch)
            res.append({"nonce": f"{ch.nonce:016x}", "expected": f"{ehi:016x}{elo:016x}", "ok": (lo, hi) == (elo, ehi)})
        ok = all(c["ok"] for c in res)
        alerts = (
            []
            if ok
            else [
                {
                    "type": "alert",
                    "kind": "digest",
                    "id": vid,
                    "detail": "digest mismatch: the prover's answer is not the expected chain",
                }
            ]
        )
        return {"type": "check", "id": vid, "challenges": res, "digest_ok": ok}, alerts

    fetches = deque(maxlen=50)  # (fetch s, whole check s) of the last record checks: the lag alert names slow fetches
    saving = {"stopped": False}

    def record_check(vid, dyn_item):
        """The records of one dynamic round: fetched, bound to the chains, replayed (`_check_dynamic`), kept. A
        record neither replayed (no --replay) nor kept (--evidence-max-total reached) is checked by nothing, now
        or later: that round's dynamic check fails (an alert each), it never passes as stored."""
        chd, dg, lv, sample = dyn_item
        t0 = time.monotonic()
        rep = _check_dynamic(args, keys, chd, dg, ver.lib, live=lv, sample=sample)
        fetches.append((float(rep.get("fetch_s") or 0.0), time.monotonic() - t0))
        unsaved = _not_kept(rep)
        # neither replayed nor kept, by configuration
        nowhere = rep["digest_ok"] is None and not getattr(args, "evidence_dir", None)
        if unsaved and rep["digest_ok"] is None:
            rep.update(
                digest_ok=False,
                failed="unchecked",
                error="dynamic checking is off: no --replay, and --evidence-max-total is "
                "reached, so this freeze's records are checked by nothing",
            )
        ev = {
            "type": "check",
            "id": vid,
            "challenges": [],
            "digest_ok": True,
            "dyn_digest_ok": rep["digest_ok"],
            "dyn_check": rep,
            **({"record_saved": False} if unsaved else {}),
            **({"record_saved": False, "not_kept": "no --evidence-dir, no --replay"} if nowhere else {}),
        }
        alerts = []
        if getattr(args, "_evidence_full", False) and not saving["stopped"]:
            saving["stopped"] = True  # once: the page's meta says so from here on
            hub.set_meta(evidence_saving=False)
            write_log({"type": "meta", "t": time.time(), "evidence_saving": False})
            alerts.append(
                {
                    "type": "alert",
                    "kind": "evidence",
                    "id": current["id"],
                    "first_unsaved_id": vid,
                    "detail": f"records are no longer saved: {args._evidence_full}. "
                    + (
                        "They are still replayed online; an offline audit with this log skips the records not kept"
                        if args.replay
                        else "Without --replay nothing checks dynamic memory from here on: every dynamic round alerts"
                    ),
                }
            )
        if rep["digest_ok"] is False:
            alerts.append({"type": "alert", "kind": "dynamic", "id": vid, "detail": _dynamic_failure(rep)})
        return ev, alerts

    CHECKS = {"static": static_check, "dynamic": record_check}

    def take(kind: str, vid: int) -> bool:
        """This verification's check is done: False if it was already reported otherwise (unchecked at the end)."""
        with pend_lock:
            return pending[kind].pop(vid, None) is not None

    def failed_check(kind: str, vid: int, detail: str) -> None:
        ev = {"type": "check", "id": vid, "challenges": [], "error": detail}
        ev.update(
            {"digest_ok": False}
            if kind == "static"
            else {"digest_ok": True, "dyn_digest_ok": False, "dyn_check": {"digest_ok": False, "error": detail}}
        )
        publish(ev)
        alert({"type": "alert", "kind": "check", "id": vid, "detail": detail})

    def worker(kind: str) -> None:
        """One checking thread: one bad verification must never stop it. A failure inside a check is that
        verification's failed check and an alert; the loop goes on."""
        q = queues[kind]
        while not stop_checks.is_set():
            try:
                item = q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if item is None:
                return
            vid = item[0]
            try:
                ev, alerts = CHECKS[kind](*item)
            except Exception as e:
                if take(kind, vid):
                    failed_check(
                        kind, vid, f"the {WHAT[kind]} of this verification failed: {type(e).__name__}: {_short(e)}"
                    )
                continue
            if take(kind, vid):
                publish(ev)
                for a in alerts:
                    alert(a)

    threads = {
        kind: threading.Thread(target=worker, args=(kind,), name=f"posm-check-{kind}", daemon=True)
        for kind in queues if not skip_checks
    }
    for thread in threads.values():
        thread.start()

    def reserve(kind: str, vid: int) -> None:
        """This round's check is owed: from here on it ends in a check or in the unchecked report at stop (it is
        reserved before the round is published, so a stop in between still reports it)."""
        with pend_lock:
            pending[kind][vid] = time.monotonic()

    def enqueue(kind: str, item) -> None:
        vid = item[0]
        reserve(kind, vid)
        try:
            queues[kind].put_nowait(item)
        except _queue.Full:  # never block the timing loop: this one is a failed check
            if take(kind, vid):
                failed_check(
                    kind,
                    vid,
                    f"not checked: {queues[kind].maxsize} {WHAT[kind]}s were already waiting (checking cannot keep up)",
                )

    def slow_fetches() -> str | None:
        """Whether fetching the prover's records is what holds the record checks back (most of the last checks'
        time, or a fetch in progress for over half of --check-lag-s): then the lag alert says so, rather than
        reading as the verifier's own slowness."""
        got = list(fetches)
        f, tot = sum(x for x, _ in got), sum(y for _, y in got)
        t0 = getattr(args, "_fetch_t0", None)
        cur = time.monotonic() - t0 if t0 else 0.0
        bound = f"--evidence-timeout {_fetch_timeout(args):g} s a record"
        if got and tot > 0 and f > 0.5 * tot:
            return (
                f"the prover's records are slow to come: fetching took {f / tot:.0%} of the last {len(got)} record checks "
                f"({f / len(got):.1f} s a round; {bound})"
            )
        if cur > 0.5 * args.check_lag_s:
            return f"the prover's records are slow to come: the record being fetched has taken {cur:.1f} s so far ({bound})"
        return None

    lagging = {}

    def check_lag(vid: int) -> None:
        """An alert when the oldest unchecked verification has waited longer than --check-lag-s (once per episode).
        It carries the current round's id; the oldest waiting one is oldest_id."""
        now_ = time.monotonic()
        for kind, queue in pending.items():
            with pend_lock:
                head = next(iter(queue.items()), None)
                waiting = len(queue)
            late = head is not None and now_ - head[1] > args.check_lag_s
            if late and not lagging.get(kind):
                cause = slow_fetches() if kind == "dynamic" else None
                alert(
                    {
                        "type": "alert",
                        "kind": "check",
                        "id": vid,
                        "oldest_id": head[0],
                        "waiting": waiting,
                        "cause": "slow_fetch" if cause else "checking",
                        "detail": f"checking is falling behind: the {WHAT[kind]} of #{head[0]} has waited {now_ - head[1]:.0f} s (over --check-lag-s "
                        f"{args.check_lag_s:g}); {waiting} waiting" + (f"; {cause}" if cause else ""),
                    }
                )
            lagging[kind] = late

    def static_round(vid: int, strategy: str):
        """The static chains of one round (--repeat of them). A READY or reply that does not come is a fault: the
        round fails (`faults`), its challenge is kept with rtt_ns None and nothing of it is checked or timed. Its
        minimum is over the chains that came, but it cannot pass: else a cheater could drop its slow chains."""
        best = None
        worst_arm = 0
        chs = []
        subs = []
        faults = []
        wire = "honest" if monitor else strategy  # monitor: a plain challenge; the prover's posture decides
        for _ in range(repeat):
            ch = Challenge.fresh(B, n, r, nb, k=k, g=g)
            sub = {"nonce": f"{ch.nonce:016x}", "j0": ch.j0, "rtt_ns": None, "arm_ns": 0}
            subs.append(sub)
            if drain:
                sub["arm_ns"] = _arm(sock, args, B, n, r, k, g, STRATS.index(wire))
                if sub["arm_ns"] is None:
                    sub["missing"] = "READY"
                    faults.append(FAULT_NO_READY)
                    continue
                worst_arm = max(worst_arm, sub["arm_ns"])
            got = _timed(sock, args, ch, MAGIC, STRATS.index(wire))
            if got is None:
                sub["missing"] = "reply"
                faults.append(FAULT_NO_STATIC_REPLY)
                continue
            rtt, lo, hi = got
            chs.append((ch, lo, hi))
            best = rtt if best is None else min(best, rtt)
            sub.update({"rtt_ns": rtt, "received": f"{hi:016x}{lo:016x}"})
        ev = {
            "type": "verification",
            "round": "static",
            "id": vid,
            "rtt_ns": best,
            "arm_ns": worst_arm,
            "challenges": subs,
        }
        if not monitor:
            ev["strategy"] = strategy
        if faults:
            ev["faults"] = faults
        if memory and not dynamic:  # no dynamic region: the claim is static only, every round
            ev["memory"] = _memory_claim(cap, static_credit, 0, 0, mem_bound, dyn_bits, static_array=cfg.nbytes)
        return ev, (vid, chs)

    def dynamic_round(vid: int, strategy: str):
        """A freeze and its chains: the chain over all dynamic memory, then the live chain. The memory claim is
        made here, from the sizes READY reported at this freeze."""
        wire = "honest" if monitor else strategy
        sample, chd, dg, lv = _dynamic_round(sock, args, rng, dyn_shape, STRATS.index(wire), jitter=False)
        if weights and not sample.get("fault") and sample["blocks"] * dyn_shape[0] != schedule["dynamic_bytes"]:
            raise ValueError("dynamic READY size differs from the benchmark layout used to schedule challenges")
        ev = {
            "type": "verification",
            "round": "dynamic",
            "id": vid,
            "rtt_ns": None,
            "arm_ns": sample.get("arm_ns"),
            "challenges": [],
            "dyn": sample,
        }
        if not monitor:
            ev["strategy"] = strategy
        if sample.get("fault"):
            ev["faults"] = ["dynamic: " + sample["fault"]]
        if memory:  # what no chain covered at this freeze (no READY: all of it)
            ev["memory"] = _memory_claim(
                cap,
                static_credit,
                sample["words"] * 8,
                sample.get("live", {}).get("words", 0) * 8,
                mem_bound,
                dyn_bits,
                dyn_shape[0],
                static_array=cfg.nbytes,
                live_credited=dyn_enc,
            )
        return ev, ((vid, (chd, dg, lv, sample)) if chd is not None else None)

    ROUNDS = {"static": static_round, "dynamic": dynamic_round}
    kinds = ["static", "dynamic"] if dynamic else ["static"]

    def pick(allowed: list) -> tuple[str, float]:
        """Keep the aggregate exponential wait independent of the selected round kind."""
        if weights and len(allowed) == 2:
            return _pick_pool_round(weights), _gap(args.interval / 2)
        return allowed[_SYSRAND.randrange(len(allowed))], _gap(args.interval / len(allowed))

    def set_posture(name: str) -> str | None:
        """Demo relay: tell the prover's simulator which posture to take. Returns the posture in force."""
        if name not in STRATS:
            return None
        tok = secrets.randbits(62) | 1  # the CSPRNG: a token from the seeded generator is predictable
        if not _send(sock, args, POSTURE.pack(POSTURE_MAGIC, tok, STRATS.index(name))):
            return None
        # HBM changes first
        data = _await(
            sock, args, lambda d: len(d) == POSTURE.size and POSTURE.unpack(d)[:2] == (POSTURE_MAGIC, tok), 5.0
        )
        sid = POSTURE.unpack(data)[2] if data is not None else -1
        return STRATS[sid] if 0 <= sid < len(STRATS) else None

    # CUSUM for small persistent shifts (partial residency) that no single verification shows. It consumes
    # every chain of a verification, not the minimum: the minimum is the statistic that hides a partial
    # prover (the fastest of m chains is the one that touched the missing region least), while each chain
    # carries ~+17 µs per missing link. z per chain against the single-chain calibration, capped so a
    # stall is the deadline's business; S = max(0, S + Σ(z_i - k)); alarm when S > h, then reset.
    cusum: dict = {}

    def drift(key: str, vid: int, rtts_ns: list, base: dict):
        S = _cusum_step(cusum.get(key, 0.0), rtts_ns, base, args.cusum_k)
        if S is None:
            return None
        if S > args.cusum_h:
            last = cusum.get(("alerted", key), -(10**9))
            if vid - last >= 25:  # one alert per episode, not one per reset
                if not claim_on("static"):  # with a timing claim, it replaces the drift alarm
                    alert(
                        {
                            "type": "alert",
                            "kind": "drift",
                            "id": vid,
                            "detail": f"timing drift: CUSUM {S:.1f} over {args.cusum_h} (round trips persistently above the honest median)",
                            "key": key,
                        }
                    )
                cusum[("alerted", key)] = vid
            S = 0.0
        cusum[key] = S
        return S

    def prover_stats():
        # short: a held reply only delays the gap that follows; a late one is dropped
        stats = _prover_stats(sock, args, 0.25)
        if stats is not None:
            hub.publish(
                {
                    "type": "prover",
                    "served": stats[2],
                    "fired": stats[3],
                    "armed": stats[4],
                    "hold_expired": stats[5],
                    "svc_p50_ns": stats[6],
                    "svc_p99_ns": stats[7],
                    "dt_p50_ns": stats[8],
                    "dt_p99_ns": stats[9],
                    "posture": STRATS[stats[10]] if 0 <= stats[10] < len(STRATS) else None,
                }
            )

    last_stats = [0.0]

    def between():
        """Round trips to the prover that are not challenges (the page's posture requests, the prover's
        counters) go right after a round, before the random gap, never after it: after the gap they would warn
        the prover that a challenge is coming, and their untimed replies, which it can hold, would put a reload
        window between the gap and the challenge, the window the random gaps are there to deny. Of the posture
        requests waiting only the latest is carried out (each can wait 5 s for the prover)."""
        if hub.requests:
            reqs = []
            while hub.requests:
                try:
                    reqs.append(hub.requests.pop(0))
                except IndexError:
                    break
            if reqs:
                req = reqs[-1]
                got = set_posture(req.get("posture", ""))
                # label rounds by posture
                publish(
                    {
                        "type": "posture",
                        "requested": req.get("posture"),
                        "posture": got,  # logged: an analysis can
                        **({"superseded": len(reqs) - 1} if len(reqs) > 1 else {}),
                    }
                )
        if time.time() - last_stats[0] > 2.0:
            prover_stats()
            last_stats[0] = time.time()
        check_lag(current["id"])

    # The timing claim runs on every timed chain, each its own instance on its own calibration: the static
    # chains, and with --dynamic the chain over all dynamic memory and the live chain. Every chain under a claim
    # also runs a miss-count test (memory_accounting.verifier.timing_claim.MissCountTest), alerting within a few round trips of a
    # prover late on every chain: the dynamic and live chains' mean-shift claim can rarely be sized (a 64 KiB
    # block read from host memory costs about 2 µs, below the few µs the honest level of these chains wanders
    # by), and a static window or period can take long to fill. A chain whose misses its claim weighs alerts on
    # a single miss only past the ceiling; one whose window or period would take longer than --claim-max-fill-s
    # to fill alerts on every miss again, as without a claim.
    claims: dict = {}  # chain -> {"test": MeanShiftTest} / the period model's state, and "misses"

    def claim_on(chain: str) -> bool:
        return bool(claims.get(chain))

    def weighs(chain: str) -> bool:
        """The chain's single misses are weighed by its claim (not alerts on their own)."""
        return claim_on(chain) and not claims[chain].get("per_miss")

    def chain_values(ev: dict, chain: str) -> list:
        """The round trips (ns) of one verification on this chain, for the claim. A chain the prover did not answer
        counts as inf (at the clip: K_i = L, the design log's Issue #3), so it cannot leave its slow chains out of the
        sum: a reply that did not come, and a READY that did not come (no challenge went, but withholding READY is how
        a prover would dodge a challenge it cannot answer in time). A READY past the ceiling (ready_clipped) puts every
        chain of its round at the clip too in legacy mode. Fixed-cap mode keeps READY penalties separate."""
        return _claim_values(ev, chain, penalize_ready=not fixed_claim)

    def chain_misses(ev: dict, chain: str) -> list:
        """Whether each round trip of this round on this chain missed its deadline (empty: none to judge). Static
        chains are each held to the single-chain deadline; without one (an older baseline with --repeat > 1) the
        round's minimum is held to the deadline."""
        if chain == "static":
            if not ev.get("challenges") or ev.get("deadline_ns") is None:
                return []
            cd = ev.get("chain_deadline_ns")
            if cd:
                return [t > cd for t in _claim_values(ev, "static")]  # deadline/READY policy is unchanged
            return [ev["rtt_ns"] > ev["deadline_ns"]] if ev.get("rtt_ns") is not None else []
        d = ev.get("dyn") or {}
        if chain == "live":
            d = d.get("live") or {}
        if d.get("rtt_ns") is None or not d.get("deadline_ns"):
            return []
        return [d["rtt_ns"] > d["deadline_ns"]]

    def chain_rate(b: dict, chain: str) -> float | None:
        """Round trips per second on this chain at the calibration's pace (None: not known). A baseline role's
        calibration ran its static rounds back to back, not at this run's pace: --interval sets it then."""
        cs = b.get("calibration_samples") or {}
        span = float(cs.get("span_s") or 0.0)
        scores = (b.get("claim_calibration_samples") or {}) if fixed_claim else cs
        got = len(scores.get(CLAIM_SAMPLES[chain]) or [])
        if span > 0 and got and cs.get("role") != "baseline":
            return got / span
        return (repeat if chain == "static" else 1) / args.interval if args.interval > 0 else None

    def flags_only(chain) -> bool:
        """Under --challenge-flags warn, a chain with a mean-shift claim (window or period) raises no alert on one
        challenge (the design log v0.2, Issue #1): its gross round trips, missing replies and miss count are flags,
        logged and shown, and the claim weighs the round trips (a missing reply at the clip)."""
        return chain is not None and bool((claims.get(chain) or {}).get("flags_only"))

    def flag(ev: dict, chain: str, kind_: str, detail: str, **extra) -> None:
        """A single-challenge flag: an alert under --challenge-flags alert, else a logged warning. Either way it is
        outside the claim and its α."""
        rec = {
            "kind": kind_,
            "chain": chain,
            "id": ev["id"],
            "detail": detail if chain == "static" else f"{chain} chain: {detail}",
            "outside_claim": True,
            **extra,
        }
        if flags_only(chain):
            publish({"type": "flag", **rec})
        else:
            alert({"type": "alert", **rec})

    def setup_claims(baseline: dict) -> None:
        """Check each timed chain's claim against its honest calibration; publish them as meta
        `claims` (chain -> parameters, with "tests": the tests that run on it, from "window", "period" and
        "misses", and "reason" when one is off) and the static one as `claim` too."""
        if args.claim_F is None:
            return
        from .timing_claim import MissCountTest

        chains = ["static"] + (["dynamic"] + (["live"] if args.dyn_live and live_deadline else []) if dynamic else [])
        alpha = args.claim_alpha if args.claim_alpha is not None else 1 - args.claim_C
        infos = {}
        for chain in chains:
            info = setup_claim(baseline, chain)
            st_ = claims.setdefault(chain, {})
            measured = "test" in st_ or "period" in st_
            st_["flags_only"] = measured and args.challenge_flags == "warn"
            info["challenge_flags"] = "warn" if st_["flags_only"] else "alert"
            tests = (["window"] if "test" in st_ else []) + (["period"] if "period" in st_ else [])
            mt = MissCountTest(args.claim_miss_window, args.claim_miss_rate, alpha)
            st_["misses"] = mt
            tests.append("misses")
            info["misses"] = {"window": mt.window, "k": mt.k, "miss_rate": mt.q, "alpha": mt.alpha}
            what = "timing claim" if chain == "static" else f"timing claim on the {chain} chain"
            print(
                f"{what}: {'flag' if st_['flags_only'] else 'alert'} if {mt.k} of its last {mt.window} round trips miss "
                f"{'the single-chain' if chain == 'static' else 'its'} deadline (honest allowance {mt.q:g} a round trip, false alarm "
                f"≤ {mt.alpha:g} per {mt.window})"
                + (
                    "; gross round trips and missing replies are flags too, outside the claim"
                    if st_["flags_only"]
                    else ""
                ),
                flush=True,
            )
            rate = chain_rate(baseline, chain)
            fill = (
                args.claim_period_s
                if "period" in st_
                else (st_["test"].n_window / rate if "test" in st_ and rate else None)
            )
            if fill is not None and fill > args.claim_max_fill_s:
                if st_["flags_only"]:
                    why = (
                        f"its {'period' if 'period' in st_ else 'window'} takes about {fill / 60:.0f} min to fill at this pace: "
                        "nothing on this chain alerts before then (--challenge-flags warn)"
                    )
                else:
                    # a window or period this long leaves a slow prover unjudged for that long: its misses alert one by
                    # one
                    st_["per_miss"] = True
                    info["per_miss"] = True
                    why = (
                        f"its {'period' if 'period' in st_ else 'window'} takes about {fill / 60:.0f} min to fill at this pace, over "
                        f"--claim-max-fill-s {args.claim_max_fill_s:g} s: every miss of this chain alerts at once"
                    )
                info["reason"] = (info["reason"] + "; " if info.get("reason") else "") + why
                print(f"{what}: {why}", flush=True)
            info["tests"] = tests
            infos[chain] = info
        hub.set_meta(claim=infos.get("static"), claims=infos, challenge_flags=args.challenge_flags)
        # the log says which claims ran
        write_log({"type": "meta", "t": time.time(), "claim": infos.get("static"), "claims": infos})

    def setup_claim(baseline: dict, chain: str) -> dict:
        """Set up one chain's timing claim using its frozen window settings or the legacy period model.
        Static chains read 1 MiB blocks (Δ_c --claim-delta-c); the dynamic and live
        chains read --dyn-B blocks (Δ_c --claim-delta-c-dyn) over --dyn-n links. Returns the parameters ("reason"
        when the test is off)."""
        from dataclasses import replace

        from .timing_claim import MeanShiftTest, estimate_baseline, params_from_baseline, plan

        L = n if chain == "static" else dyn_shape[1]
        dc = (
            args.claim_delta_c
            if chain == "static"
            else (args.claim_delta_c_dyn if args.claim_delta_c_dyn is not None else dyn_delta_c_us(dyn_shape[0]))
        )
        what = "timing claim" if chain == "static" else f"timing claim on the {chain} chain"
        if fixed_claim:
            setup = {}
            try:
                setup = _fixed_claim_setup(baseline, args, chain)
                if setup.get("cap_us") is None or setup.get("window") is None:
                    raise ValueError(
                        setup.get("reason", "warmup did not provide a cap and supported window; recalibrate")
                    )
                samples = _fixed_claim_samples(baseline, args, chain)
            except ValueError as e:
                print(f"{what} (fixed-cap test) off: {e}", flush=True)
                return {"chain": chain, "c_us": setup.get("cap_us"), "n_window": None, "setup": setup, "reason": str(e)}
        else:
            samples = (baseline.get("calibration_samples") or {}).get(CLAIM_SAMPLES[chain])
        minimum = 2 if fixed_claim else 20
        if not samples or len(samples) < minimum:
            reason = f"the mean-shift test needs at least {minimum} calibration {'attempts' if fixed_claim else 'responses'} (calibrate in this run, or a baseline saved with its samples)"
            print(f"{what} ({args.claim_mode} test) off: {reason}", flush=True)
            return {"chain": chain, "reason": reason}
        xs = np.array(samples, dtype=np.float64) / 1e3
        base = (
            estimate_baseline(xs, L, dc, c=setup["cap_us"])
            if fixed_claim
            else estimate_baseline(xs, L, dc, args.claim_M_factor * L * dc)
        )
        params = params_from_baseline(
            base, args.claim_F, args.claim_C, L, dc, alpha=args.claim_alpha, sigma_margin=args.claim_sigma_margin
        )
        common = {
            "chain": chain,
            "F": params.F,
            "C": params.C,
            "alpha": params.a,
            "L": params.L,
            "delta_c_us": params.delta_c,
            "mu_b_us": params.mu_b,
            "sigma_b_us": params.sigma_b,
            "sigma_margin": params.sigma_margin,
            "sigma_us": params.sigma,
            "M_us": params.M_,
            "clip_us": params.clip,
            "floor_us": params.floor,
            "delta_F_us": params.delta_F,
            "V_c_us2": params.delta_F * params.M_,
            "n_cal": base["n"],
            "cal_clipped": base["clipped"],
            "ceiling": args.claim_ceiling,
        }
        if fixed_claim:
            common.update(
                c_us=params.c, M_us=None, V_c_us2=params.c**2 / 4, requested_window=setup["window"], setup=setup
            )
        if args.claim_mode == "period":
            return setup_period(baseline, params, chain, what, common)
        params = replace(params, drift=args.claim_drift_us)  # the period model has σ_s for the drift instead
        common["drift_us"] = params.drift
        pl = plan(params, setup["window"]) if fixed_claim else plan(params)
        nwin = (
            (pl["n_window"] if pl["supported"] else None)
            if fixed_claim
            else (math.ceil(pl["n_required"]) if math.isfinite(pl["n_required"]) else None)
        )
        info = {
            **common,
            "n_window": nwin,
            "t_us": pl["t_us"] if nwin else None,
            "tau_us": pl["tau"] if nwin else None,
            "F_floor": pl["F_floor"],
            "honest_miss_rate": baseline.get("honest_reject_rate")
            if chain == "static"
            else (
                (baseline.get("dyn") or {}) if chain == "dynamic" else ((baseline.get("dyn") or {}).get("live") or {})
            ).get("honest_reject_rate"),
        }
        if nwin is None:
            info["reason"] = (
                f"the bounds do not support F = {params.F} with N = {setup['window']} and {base['n']} calibration attempts"
                if fixed_claim
                else f"F = {params.F} cannot be tested with {base['n']} calibration responses (no N reaches below F = {pl['F_floor']:.3g})"
            )
            print(f"{what} (mean shift) off: {info['reason']} (more calibration, or a larger F)", flush=True)
            return info
        if not fixed_claim and chain != "static" and pl["tau"] < args.claim_sigma_s:
            # the window test has no term for the slow wander of the honest level; a dynamic chain's is not run below it
            info["n_window"] = None
            info["reason"] = (
                f"τ = {pl['tau']:.2f} µs is below the honest level's slow wander ({args.claim_sigma_s:g} µs, --claim-sigma-s)"
            )
            print(
                f"{what} (mean shift) off: {info['reason']}; a larger F or --claim-delta-c-dyn (larger --dyn-B) raises it",
                flush=True,
            )
            return info
        clip_note = (
            f"each clipped at c = {params.c:g} µs"
            if fixed_claim
            else f"each clipped at μ + M = {params.mu_b:.1f} + {params.M_:.0f} µs"
        )
        print(
            f"{what}: at most F = {params.F} of reads deviate (each ≥ {params.delta_c:.3g} µs) at confidence {params.C}: windows of {nwin} chains, "
            f"{clip_note}; pass if their sum − N·μ < t = {pl['t_us']:.0f} µs "
            f"(mean shift < {pl['tau']:.2f} µs, of it {params.drift:g} µs for drift)",
            flush=True,
        )
        claims.setdefault(chain, {}).update(test=MeanShiftTest(params, nwin), delta_c=params.delta_c)
        return info

    def setup_period(baseline: dict, params, chain: str, what: str, common: dict):
        """The period model: every chain of a period, judged when the period ends."""
        from dataclasses import replace

        from .timing_claim import PeriodAnalysis, plan

        cs = baseline.get("calibration_samples") or {}
        span = float(cs.get("span_s") or 0.0)
        params = replace(
            params,
            sigma_s=args.claim_sigma_s,
            t_corr=args.claim_t_corr,
            T=args.claim_period_s,
            k_cal=max(1.0, span / args.claim_t_corr) if args.claim_t_corr > 0 else None,
        )
        pl = plan(params)
        rate = chain_rate(baseline, chain)  # chains per second at this pace
        nr = pl["n_required"]
        info = {
            **common,
            "mode": "period",
            "cal_span_s": span,
            "period_s": args.claim_period_s,
            "sigma_s_us": args.claim_sigma_s,
            "t_corr_s": args.claim_t_corr,
            "n_required": nr if math.isfinite(nr) else None,
            "F_floor": pl["F_floor"],
            "tau_us": pl["tau"] if math.isfinite(nr) else None,
            "chains_per_s": rate,
            "n_expected": rate * args.claim_period_s if rate else None,
            "rate_required": nr / args.claim_period_s if math.isfinite(nr) else None,
            "honest_miss_rate": baseline.get("honest_reject_rate") if chain == "static" else None,
        }
        if chain != "static" and pl["F_floor"] > params.F:
            # the window model's rule (no claim below the honest level's slow wander), in the period model's terms: the
            # wander and the calibration's error alone allow no claim below F_floor. Off; the miss count runs
            info["reason"] = (
                f"the honest level's slow wander ({args.claim_sigma_s:g} µs, --claim-sigma-s) and the calibration allow no claim "
                f"below F = {pl['F_floor']:.3g}, over the target F = {params.F}"
            )
            print(
                f"{what} (period model) off: {info['reason']}; a larger F, --claim-delta-c-dyn (larger --dyn-B) or a longer calibration lowers it",
                flush=True,
            )
            return info

        def mk():
            return PeriodAnalysis(params)

        claims.setdefault(chain, {}).update(
            period=mk(), make_period=mk, period_no=1, period_start=time.time(), p=params, delta_c=params.delta_c
        )
        print(
            f"{what}, period model: periods of {args.claim_period_s:.0f} s, each chain clipped at μ + M = {params.mu_b:.1f} + {params.M_:.0f} µs; "
            f"target F = {params.F} at confidence {params.C} needs "
            + (
                f"N = {nr:.3g} chains ({nr / args.claim_period_s:.2g} per second)"
                if math.isfinite(nr)
                else "more than any N: below the slow-movement floor"
            )
            + (f", this pace gives about {rate * args.claim_period_s:.3g}" if rate else "")
            + f"; floor from slow movement and the calibration F ≥ {pl['F_floor']:.3g}",
            flush=True,
        )
        return info

    def claim_alert(ev: dict, chain: str, detail: str, **extra) -> None:
        alert(
            {
                "type": "alert",
                "kind": "claim",
                "chain": chain,
                "id": ev["id"],
                "detail": detail if chain == "static" else f"{chain} chain: {detail}",
                **extra,
            }
        )

    def period_verdict(chain: str, vid: int, partial: bool = False) -> None:
        """Judge the chain's current period: at its end, or (partial) the open one when the run stops."""
        st_ = claims[chain]
        verdict = st_["period"].verdict()
        verdict.update(
            {
                "type": "claim_period_verdict",
                "chain": chain,
                "id": vid,
                "period": st_["period_no"],
                **({"partial": True} if partial else {}),
            }
        )
        publish(verdict)
        if verdict["test"] == "fail":
            claim_alert(
                {"id": vid},
                chain,
                f"timing claim failed over {'the open ' if partial else ''}period {verdict['period']}"
                + (" (judged at stop)" if partial else "")
                + f": sum {verdict['D_us']:+.0f} µs over the honest mean, over the threshold "
                f"t = {verdict['t_us']:.0f} µs (mean {verdict['mean_shift']:+.2f} µs, τ {verdict['tau']:.3f} µs); implied deviating share {verdict['implied_F']:.3g}",
            )
        return verdict

    def period_feed(ev: dict, chain: str, mt=None):
        st_ = claims[chain]
        pa = st_["period"]
        for x in chain_values(ev, chain):
            pa.add(x / 1e3, ev["t"])
        if time.time() - st_["period_start"] >= args.claim_period_s:
            period_verdict(chain, ev["id"])
            st_.update(period=st_["make_period"](), period_no=st_["period_no"] + 1, period_start=time.time())
        else:
            st = pa.stats()
            hub.publish(
                {
                    "type": "claim_period_progress",
                    "chain": chain,
                    "period": st_["period_no"],
                    "elapsed_s": time.time() - st_["period_start"],
                    **{
                        k_: st[k_]
                        for k_ in ("n", "mean_shift", "D_us", "t_us", "tau", "clipped", "F_testable", "F_floor")
                    },
                    **(mt.progress() if mt is not None else {}),
                }
            )

    def judge_open_periods(vid: int) -> None:
        """At stop: the period in progress is judged on what it holds (marked partial), so a cheat that began in
        it is not left unjudged."""
        for chain, st_ in list(claims.items()):
            if "period" in st_ and st_["period"].n > 0:
                v = period_verdict(chain, vid, partial=True)
                print(
                    f"timing claim ({chain} chain): the open period {v['period']} judged at stop: {v['n']} chains, mean shift "
                    f"{v['mean_shift']:+.2f} µs (τ {v['tau']:.3f}), test {v['test']}, the least F they can show {v['F_testable']:.3g}",
                    flush=True,
                )

    def claim_feed(ev: dict):
        """Every chain of a round goes into its chain's current window (a full window is judged) or, in the
        period model, into its current period (judged when it ends), and into its chain's miss count."""
        for chain in list(claims):
            st_ = claims[chain]
            vals = chain_values(ev, chain)
            mt = st_.get("misses")
            missed = chain_misses(ev, chain)
            if mt is not None:
                for miss in missed:
                    if mt.add(miss):
                        detail = (
                            f"{mt.k} of its last {mt.window} round trips missed the deadline, more than an honest "
                            f"prover's {mt.q:g} a round trip explains (confidence {1 - mt.alpha:g})"
                        )
                        if "test" in st_ or "period" in st_:
                            flag(ev, chain, "claim", detail, rule="misses")
                        else:
                            claim_alert(ev, chain, detail, rule="misses")  # the chain's only timing test
            if not vals and not missed:
                continue  # this round has no chain of this kind
            if "period" in st_:  # its progress carries the miss count too
                period_feed(ev, chain, mt)
                continue
            test = st_.get("test")
            if test is None:
                if mt is not None:
                    hub.publish({"type": "claim_progress", "chain": chain, **mt.progress()})
                continue
            p_delta_c = st_["delta_c"]
            for x in vals:
                verdict = test.add(x / 1e3)
                if verdict is not None:
                    verdict.update(
                        {"type": "claim_verdict", "chain": chain, "id": ev["id"], "window": len(test.verdicts)}
                    )
                    publish(verdict)
                    if not verdict["pass"]:
                        claim_alert(
                            ev,
                            chain,
                            f"timing claim failed: mean {verdict['mean_shift']:+.1f} µs over the honest mean (τ {verdict['tau']:.2f} µs; sum "
                            f"{verdict['D_us']:+.0f} µs over t = {verdict['t_us']:.0f} µs), "
                            + (
                                f"{verdict['clipped']} of {verdict['n']} responses at the fixed cap c = {test.p.c:g} µs"
                                if fixed_claim
                                else f"{verdict['clipped']} of {verdict['n']} responses at the clip: slower per read than the claim's {p_delta_c:.3g} µs"
                                if verdict.get("saturated")
                                else f"implied deviating share {verdict['implied_F']:.3f}"
                            ),
                        )
            hub.publish(
                {
                    "type": "claim_progress",
                    "chain": chain,
                    **test.progress(),
                    **(mt.progress() if mt is not None else {}),
                }
            )

    # A late READY within --ready-ceiling × its bound is counted, not alerted on its own: the bound is twice the
    # calibration's p99.9 READY wait, and under vLLM an honest freeze sometimes waits longer for a layer boundary
    # than a calibration of 150 rounds saw (about 1 in 1000 static rounds on the H200: an honest alert every 10 to 20
    # minutes). Each kind of round has a miss count of its late READYs (k of the last --claim-miss-window, as a
    # chain's misses); one over the ceiling alerts at once. A prover can so reload at most ceiling × bound's worth
    # (printed as the reload slack) on fewer than k rounds of each window without an alert.
    from .timing_claim import MissCountTest

    _ready_alpha = args.claim_alpha if args.claim_alpha is not None else 1 - args.claim_C
    ready_tests = {
        k_: MissCountTest(args.claim_miss_window, args.claim_miss_rate, _ready_alpha) for k_ in ("static", "dynamic")
    }

    def ready_verdict(kind: str, arm_ns, bound) -> tuple[bool, str | None, bool, bool]:
        """(late, detail or None, weighed, over) for one READY wait against its bound. `over`: past --ready-ceiling ×
        the bound. Such a round's chains count at the clip in their claims (K_i = L: whatever a prover reloaded in the
        wait, the round counts as cheated, so a delayed READY buys it nothing), and on a chain whose mean-shift claim
        runs it is a single-challenge flag (--challenge-flags); elsewhere it alerts at once. Over 21 honest hours under
        vLLM, 1 READY in 5000 passed 3 × a bound set from a 15-minute calibration, up to about 100 ms: no fixed
        ceiling is free of honest alerts."""
        late = bool(bound and arm_ns is not None and arm_ns > bound)
        what = "READY" if kind == "static" else "dynamic READY"
        if late and arm_ns > args.ready_ceiling * bound:
            return (
                True,
                f"{what} after {arm_ns / 1e6:.1f} ms, over {args.ready_ceiling:g} × its bound {bound / 1e6:.1f} ms",
                False,
                True,
            )
        if bound and arm_ns is not None:
            mt = ready_tests[kind]
            if mt.add(late):
                return (
                    late,
                    (
                        f"{what}s late: {mt.k} of the last {mt.window} over the bound {bound / 1e6:.1f} ms "
                        f"(this one {arm_ns / 1e6:.1f} ms)"
                    ),
                    late,
                    False,
                )
        return late, None, late, False

    def run_and_assess(vid: int, kind: str, strategy: str, retry_of: int | None = None):
        """One round against the baseline: publish it, queue its checks, raise the alerts that do not depend on
        time (memory). Returns (event, misses, now, gross): `now` are the failures that alert at once in every
        mode (a READY or reply that did not come, a late READY: these bound what a prover can reload, not
        timing noise, so neither the claim nor a retry weighs them); `misses` the timing misses, (kind, detail,
        chain); `gross` the round trips over the ceiling, (detail, chain) (alerts on a chain whose misses its
        claim weighs). Static rounds: the minimum is held to the deadline it was calibrated on, and every chain
        to the single-chain deadline (a cheater slow on some chains must not hide behind its fastest); the
        ceiling applies to the slowest chain."""
        ev, item = ROUNDS[kind](vid, strategy)
        if retry_of is not None:
            ev["retry_of"] = retry_of

        def fault_chain(f: str):
            """The chain a fault's round counts against in its claim (None: a fault no claim counts; it alerts)."""
            if f in (FAULT_NO_STATIC_REPLY, FAULT_NO_READY):
                return "static"
            if f == "dynamic: " + FAULT_NO_LIVE_REPLY:
                return "live"
            if f == "dynamic: " + FAULT_NO_DYN_REPLY or f.startswith("dynamic: " + FAULT_NO_DYN_READY):
                return "dynamic"
            return None

        now, misses, gross = [("missing", f, fault_chain(f)) for f in ev.get("faults", [])], [], []
        ceil = args.claim_ceiling
        floor_ns = args.gross_floor_ms * 1e6

        # gross: slow beyond doubt, over the ceiling × the deadline AND the deadline plus --gross-floor-ms. A short
        # chain's 3 × is only a couple of milliseconds, which an honest host stall exceeds now and then (a 4.9 ms
        # dynamic round against a 663 µs deadline, once in 1200 honest rounds under vLLM); between the deadline
        # and this a slow round is a miss, which the miss count judges
        def gross_at(ref):
            return max(ceil * ref, ref + floor_ns)

        if kind == "static":
            rtt = ev["rtt_ns"]
            chains = [c["rtt_ns"] for c in ev["challenges"] if c["rtt_ns"] is not None]
            cd = chain_deadline
            slow = [t for t in chains if cd and t > cd]
            late, ready_alert, weighed, over = ready_verdict("static", ev.get("arm_ns"), arm_max)
            if over:
                ev["ready_clipped"] = True  # its chains count at the clip (chain_values)
                ev["late_ready_flagged"] = flags_only("static")
            ev.update(
                {
                    "deadline_ns": deadline,
                    "chain_deadline_ns": cd,
                    "late_ready": late,
                    "late_ready_weighed": weighed and not ready_alert,
                    "within": rtt is not None and rtt <= deadline and not slow and not late and not ev.get("faults"),
                    "gap_sigma": (rtt - baseline["median_ns"]) / baseline["sigma_core_ns"]
                    if baseline.get("sigma_core_ns") and rtt is not None
                    else None,
                }
            )
            ev["cusum"] = drift("all" if monitor else strategy, vid, chains or [rtt], baseline)
            if ready_alert:
                # the count of late READYs: an alert
                now.append(("late_ready", ready_alert, "static" if over else None))
            if late:
                pass  # a late READY is not also a timing miss
            elif rtt is not None and rtt > deadline:
                misses.append(
                    ("deadline", f"round trip {rtt / 1e3:.0f} µs over the deadline {deadline / 1e3:.0f} µs", "static")
                )
            elif slow:
                misses.append(
                    (
                        "deadline",
                        f"a chain's round trip {max(slow) / 1e3:.0f} µs over the single-chain deadline {cd / 1e3:.0f} µs",
                        "static",
                    )
                )
            ref = cd or deadline
            if ref and chains and max(chains) > gross_at(ref):
                gross.append(
                    (
                        f"round trip {max(chains) / 1e3:.0f} µs, over {gross_at(ref) / 1e3:.0f} µs (the gross limit on the {'single-chain ' if cd else ''}deadline {ref / 1e3:.0f} µs)",
                        "static",
                    )
                )
        else:
            dyn = ev["dyn"]
            dyn["late_ready"], d_ready_alert, d_weighed, d_over = (
                ready_verdict("dynamic", dyn.get("arm_ns"), dyn_arm_max)
                if not dyn.get("fault")
                else (False, None, False, False)
            )
            if d_over:
                dyn["ready_clipped"] = True
                dyn["late_ready_flagged"] = ev["late_ready_flagged"] = flags_only("dynamic")
            dyn["late_ready_weighed"] = d_weighed and not d_ready_alert
            dyn["deadline_ns"] = dyn_deadline
            dyn["within"] = (
                dyn["rtt_ns"] is not None
                and dyn["rtt_ns"] <= dyn_deadline
                and not dyn["late_ready"]
                and not dyn.get("fault")
            )
            if "live" in dyn and live_deadline:
                dyn["live"]["deadline_ns"] = live_deadline
                dyn["live"]["within"] = dyn["live"]["rtt_ns"] is not None and dyn["live"]["rtt_ns"] <= live_deadline
                dyn["within"] = dyn["within"] and dyn["live"]["within"]
            ev.update(
                {
                    "deadline_ns": dyn_deadline,
                    "within": dyn["within"],
                    "late_ready": dyn["late_ready"],
                    "late_ready_weighed": dyn["late_ready_weighed"],
                }
            )
            if not dyn.get("fault"):
                if d_ready_alert:
                    now.append(("late_ready", d_ready_alert, "dynamic" if d_over else None))
                if not dyn["late_ready"]:
                    if dyn["rtt_ns"] > dyn_deadline:
                        misses.append(
                            (
                                "dynamic",
                                f"dynamic round trip {dyn['rtt_ns'] / 1e3:.0f} µs over its deadline {dyn_deadline / 1e3:.0f} µs",
                                "dynamic",
                            )
                        )
                    if "live" in dyn and live_deadline and not dyn["live"]["within"]:
                        misses.append(
                            (
                                "dynamic",
                                f"live chain round trip {dyn['live']['rtt_ns'] / 1e3:.0f} µs over its deadline {live_deadline / 1e3:.0f} µs",
                                "live",
                            )
                        )
                if dyn_deadline and dyn["rtt_ns"] > gross_at(dyn_deadline):
                    gross.append(
                        (
                            f"dynamic round trip {dyn['rtt_ns'] / 1e3:.0f} µs, over {gross_at(dyn_deadline) / 1e3:.0f} µs (the gross limit on its deadline {dyn_deadline / 1e3:.0f} µs)",
                            "dynamic",
                        )
                    )
                if (
                    live_deadline
                    and "live" in dyn
                    and dyn["live"]["rtt_ns"] is not None
                    and dyn["live"]["rtt_ns"] > gross_at(live_deadline)
                ):
                    gross.append(
                        (
                            f"live chain round trip {dyn['live']['rtt_ns'] / 1e3:.0f} µs, over {gross_at(live_deadline) / 1e3:.0f} µs (the gross limit on its deadline {live_deadline / 1e3:.0f} µs)",
                            "live",
                        )
                    )
        ev["timing_miss"] = bool(misses or now)
        # the page fails a gross row; one only flagged (--challenge-flags warn on a chain with a mean-shift claim) is a
        # warning
        ev["gross"] = [detail for _, detail, ch in now if not flags_only(ch)] + [
            detail for detail, ch in gross if not flags_only(ch)
        ]
        flagged = [detail for _, detail, ch in now if flags_only(ch)] + [
            detail for detail, ch in gross if flags_only(ch)
        ]
        if flagged:
            ev["flagged"] = flagged
        if ev.get("faults") and all(flags_only(fault_chain(f)) for f in ev["faults"]):
            ev["faults_flagged"] = True  # every fault a missing reply the claim weighs: the page warns
        ev = publish_round(kind, ev, item)
        claim_feed(ev)
        memory_alert(ev)
        return ev, misses, now, gross

    def publish_round(kind: str, ev: dict, item) -> dict:
        """Publish a round and queue its check: the check is owed (`reserve`) before the round is published, so
        a stop between the two still reports it unchecked. Checks are independently sampled for each kind.
        The other replies are timed and marked `check_sampled_out`."""
        current["id"] = ev["id"]
        if args.timing == "kernel":
            rtt = ev.get("rtt_ns") if kind == "static" else (ev.get("dyn") or {}).get("rtt_ns")
            ev["timestamp_source"] = "kernel" if rtt is not None else "missing"
        if skip_checks:
            ev["response_checks_skipped"] = True
            return publish(ev)
        if (
            item is not None
            and kind == "static"
            and args.check_fraction < 1
            and check_rng.random() >= args.check_fraction
        ):
            item = None
            ev["check_sampled_out"] = True
        if item is not None and kind == "dynamic" and dynamic_check_fraction < 1:
            if dynamic_check_rng.random() >= dynamic_check_fraction:
                item = None
                ev["check_sampled_out"] = True
        if item is not None:
            reserve(kind, item[0])
        ev = publish(ev)
        if item is not None:
            enqueue(kind, item)
        return ev

    def memory_alert(ev: dict) -> None:
        """A round's memory claim over the bound, or impossible: an alert from the first round on, calibration included."""
        memory = ev.get("memory") or {}
        if memory.get("impossible"):
            alert(
                {
                    "type": "alert",
                    "kind": "memory",
                    "id": ev["id"],
                    "detail": "the prover declared more memory than the GPU has "
                    f"({memory['dynamic_region_bytes'] / (1 << 30):.2f} GiB dynamic beside the static array), or a live region larger than its pool",
                }
            )
        elif memory.get("over"):
            alert(
                {
                    "type": "alert",
                    "kind": "memory",
                    "id": ev["id"],
                    "detail": f"{memory['unchecked_bytes'] / (1 << 30):.2f} GiB of the GPU "
                    f"unchallenged or uncredited, over the bound of {mem_bound / (1 << 30):.2f} GiB",
                }
            )

    def report(ev, misses, now, gross, confirmed_by=None):
        for kind_, detail, chain in now:  # a missing READY or reply, a late READY: at once
            if chain is not None and claim_on(chain) and ("test" in claims[chain] or "period" in claims[chain]):
                # a missing READY or reply, a READY past the ceiling, counted at the clip: a flag
                flag(ev, chain, kind_, detail)
            else:
                alert({"type": "alert", "kind": kind_, "id": ev["id"], "detail": detail})
        for detail, chain in gross:  # a chain the claim weighs: only a gross miss alerts now
            if weighs(chain) and ("test" in claims[chain] or "period" in claims[chain]):
                flag(ev, chain, "gross", detail)  # single-challenge flag: the claim weighs it at the clip
            elif weighs(chain) or args.confirm_misses:  # under --confirm-misses too: no retry excuses a gross miss
                alert({"type": "alert", "kind": "gross", "chain": chain, "id": ev["id"], "detail": detail})
        return [(kind_, detail) for kind_, detail, chain in misses if not weighs(chain)]  # the others alert as misses

    def reload_slack() -> dict:
        """What a prover can move at 50 GB/s once it knows a challenge is coming, per chain: a static chain from its
        ARM (the READY window); the chain over dynamic memory from the dynamic ARM (its READY window); the live
        chain from the same ARM, through the main chain's round trip, sent only after the main reply. Static and
        dynamic rounds are independent, so neither announces the other."""
        out = {"bytes_per_s": RELOAD_BYTES_PER_S, "static_ns": arm_max if drain else 0.0}
        if dynamic:
            out["dynamic_ns"] = dyn_arm_max
            if live_deadline:
                out["live_ns"] = dyn_arm_max + dyn_deadline

        def GB(ns):
            return ns / 1e9 * RELOAD_BYTES_PER_S / 1e9

        print(
            "reload slack at 50 GB/s once a prover knows a chain is coming: static "
            + (
                f"{out['static_ns'] / 1e6:.1f} ms READY window, {GB(out['static_ns']):.2f} GB"
                if drain
                else "none (no ARM: the challenge is the first it hears)"
            )
            + (f"; dynamic {dyn_arm_max / 1e6:.1f} ms READY window, {GB(dyn_arm_max):.2f} GB" if dynamic else "")
            + (
                f"; live {out['live_ns'] / 1e6:.1f} ms (READY window and the main chain before it), {GB(out['live_ns']):.2f} GB"
                if out.get("live_ns")
                else ""
            )
            + ("; static and dynamic rounds come at independent random times" if dynamic else ""),
            flush=True,
        )
        return out

    vid = 0
    deadline = None
    chain_deadline = None
    arm_max = 0.0
    dyn_deadline = None
    dyn_arm_max = 0.0
    live_deadline = None
    baseline = {}
    handlers = None
    # what the run imports later (scipy through .analysis at the end of calibration, the claim code), imported now:
    # a stop signal landing inside a C extension's initialisation surfaces as an ImportError, not KeyboardInterrupt,
    # and the stop below would not run
    from ..core import analysis as _analysis  # noqa: F401
    from . import timing_claim as _timing_claim  # noqa: F401

    if threading.current_thread() is threading.main_thread():
        # SIGTERM stops the run as Ctrl-C does, so the stop below runs (state, the last checks, what was left
        # unchecked); SIGINT back to Python's own handler: a background process started from a non-interactive
        # shell inherits it ignored
        def _terminate(signum, frame):
            raise KeyboardInterrupt

        handlers = (signal.signal(signal.SIGTERM, _terminate), signal.signal(signal.SIGINT, signal.default_int_handler))
    try:
        if args.baseline:
            baseline = _load_baseline(args, spec)  # measured under this run's conditions, or refused
            deadline = baseline["deadline_ns"]
            arm_max = (args.arm_max_ms * 1e6 if args.arm_max_ms else baseline.get("arm_max_ns", 0.0)) if drain else 0.0
            if dynamic:
                dyn_deadline = baseline["dyn"]["deadline_ns"]
                dyn_arm_max = (
                    args.dyn_arm_max_ms * 1e6 if args.dyn_arm_max_ms else baseline["dyn"].get("arm_max_ns", 0.0)
                )
                live_deadline = (baseline["dyn"].get("live") or {}).get("deadline_ns")
            hub.set_meta(baseline=baseline, baseline_self_calibrated=False, state="running")
        else:
            # the baseline is measured on the prover under test: a prover that cheats from the start sets its own
            # deadlines and claim levels. The page marks it; a deployment loads one measured on a trusted prover
            if args.calibrate < 1000:
                print(
                    f"verifier: WARNING: --calibrate {args.calibrate}: under 1000 rounds the p99.9 the READY bounds (and a p99.9 deadline) "
                    "come from is about the calibration's maximum, which an honest tail exceeds now and then",
                    flush=True,
                )
            print(
                "verifier: WARNING: calibrating on the prover under test; its deadlines and timing claim are only as honest as it is "
                "during calibration. For a trusted baseline, calibrate on a known-good prover and pass --baseline",
                flush=True,
            )
            hub.set_meta(baseline_self_calibrated=True)
            cal = []
            arms = []
            chain_rtts = []
            dcal = []
            darms = []
            dflight = 0
            lcal = []
            claim_samples = _new_claim_samples(args) if fixed_claim else None
            claim_pilot = _new_claim_samples(args) if fixed_claim else None
            cal_t0 = None
            need = args.warmup + args.calibrate
            done_n = {kind: 0 for kind in kinds}
            kind = pick(kinds)[0]
            while True:  # rounds of each kind until each has its calibration
                i = done_n[kind]
                done_n[kind] += 1
                if i == args.warmup and cal_t0 is None:
                    cal_t0 = time.time()
                if fixed_claim and i == args.warmup:
                    if kind == "static":
                        _freeze_claim_setup(
                            claim_samples, claim_pilot, args, "static", n, args.claim_delta_c, args.calibrate * repeat
                        )
                    else:
                        dc = (
                            args.claim_delta_c_dyn
                            if args.claim_delta_c_dyn is not None
                            else dyn_delta_c_us(dyn_shape[0])
                        )
                        for chain in ["dynamic"] + (["live"] if args.dyn_live else []):
                            _freeze_claim_setup(
                                claim_samples, claim_pilot, args, chain, dyn_shape[1], dc, args.calibrate
                            )
                vid += 1
                ev, item = ROUNDS[kind](vid, "honest")
                ev["phase"] = "calibration"
                if i < args.warmup:
                    ev["warmup"] = True  # logged, but not part of the baseline
                if fixed_claim:
                    _append_claim_samples(
                        claim_pilot if i < args.warmup else claim_samples,
                        ev,
                        ["static"] if kind == "static" else ["dynamic"] + (["live"] if args.dyn_live else []),
                    )
                if ev.get("faults"):  # no round trip to calibrate on; logged and alerted
                    alert(
                        {
                            "type": "alert",
                            "kind": "missing",
                            "id": vid,
                            "detail": "calibration: " + "; ".join(ev["faults"]),
                        }
                    )
                elif i >= args.warmup and kind == "static":
                    cal.append(ev["rtt_ns"])
                    arms.append(ev["arm_ns"])
                    chain_rtts += [c["rtt_ns"] for c in ev["challenges"]]
                elif i >= args.warmup:
                    dcal.append(ev["dyn"]["rtt_ns"])
                    darms.append(ev["dyn"]["arm_ns"])
                    dflight += int(ev["dyn"]["in_flight"])
                    if "live" in ev["dyn"]:
                        lcal.append(ev["dyn"]["live"]["rtt_ns"])
                ev = publish_round(kind, ev, item)  # the calibration is part of the record
                memory_alert(ev)  # the memory claim holds from the first round
                left = [x for x in kinds if done_n[x] < need]
                if not left:
                    break
                kind, gap = pick(left)
                time.sleep(gap)
            for what, got in (("static", cal), ("dynamic", dcal if dynamic else [None])):
                if not got:
                    alert(
                        {
                            "type": "alert",
                            "kind": "missing",
                            "id": vid,
                            "detail": f"calibration failed: no {what} round had all its READYs and replies",
                        }
                    )
                    raise SystemExit(
                        f"posm live: calibration failed: none of the {args.calibrate} {what} calibration rounds had all its READYs "
                        "and replies; no deadline can be set (is the prover up and serving?)"
                    )
            baseline = _deadline_from(cal, args)
            baseline.update(
                {
                    "B": B,
                    "n": n,
                    "r": r,
                    "k": k,
                    "g": g,
                    "repeat": repeat,
                    "drain": drain,
                    "timing": args.timing,
                    "dynamic": dynamic,
                    "arm_median_ns": float(np.median(arms)) if drain else 0.0,
                    "arm_max_ns": (args.arm_max_ms * 1e6 if args.arm_max_ms else 2.0 * float(np.quantile(arms, 0.999)))
                    if drain
                    else 0.0,
                }
            )
            # single-chain statistics: the drift detector, the chain deadline
            baseline.update(_chain_stats(chain_rtts, baseline, args))
            deadline = baseline["deadline_ns"]
            arm_max = baseline["arm_max_ns"]
            if dynamic:  # the dynamic chain: its own deadline and pause bound
                bd = _deadline_from(dcal, args)
                bd.update(
                    {
                        "B": dyn_shape[0],
                        "n": dyn_shape[1],
                        "arm_median_ns": float(np.median(darms)),
                        "in_flight": dflight,
                        "arm_max_ns": _dyn_ready_bound(args, darms),
                        **_dyn_schedule(args),
                    }
                )
                if lcal:
                    bd["live"] = _deadline_from(lcal, args)
                    live_deadline = bd["live"]["deadline_ns"]
                baseline["dyn"] = bd
                dyn_deadline = bd["deadline_ns"]
                dyn_arm_max = bd["arm_max_ns"]
            baseline["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            baseline["host"] = args.host
            baseline["calibration_samples"] = {
                "static_rtt_ns": cal,
                "static_chain_rtt_ns": chain_rtts,
                "static_arm_ns": arms,
                "dyn_rtt_ns": dcal,
                "dyn_arm_ns": darms,
                "live_rtt_ns": lcal,
                "timing": args.timing,
                "span_s": time.time() - (cal_t0 or time.time()),
            }
            if fixed_claim:
                baseline["claim_calibration_samples"] = claim_samples
            hub.set_meta(baseline=baseline, state="running")
            print(
                f"calibrated: honest median {baseline['median_ns'] / 1e3:.0f} µs  core σ {baseline['sigma_core_ns'] / 1e3:.0f} µs  -> deadline {deadline / 1e3:.0f} µs"
                + (
                    f" (each chain: {baseline['chain_deadline_ns'] / 1e3:.0f} µs)"
                    if repeat > 1 and baseline.get("chain_deadline_ns")
                    else ""
                )
                + (
                    f"; dynamic median {baseline['dyn']['median_ns'] / 1e3:.0f} µs -> deadline {dyn_deadline / 1e3:.0f} µs, pause bound {dyn_arm_max / 1e6:.1f} ms, "
                    f"{dflight}/{len(dcal)} paused mid-step"
                    if dynamic
                    else ""
                ),
                flush=True,
            )
            if args.save_baseline:
                Path(args.save_baseline).write_text(json.dumps(baseline, indent=1))
                print(f"baseline saved to {args.save_baseline} (reuse with --baseline)", flush=True)
        chain_deadline = baseline.get("chain_deadline_ns") or (deadline if repeat == 1 else None)
        if chain_deadline is None:
            print(
                "verifier: WARNING: the baseline has no single-chain deadline (an older file); with --repeat > 1 only the minimum "
                "of each round is held to a deadline. Recalibrate to hold every chain to one",
                flush=True,
            )
        hub.set_meta(reload_slack=reload_slack(), chain_deadline_ns=chain_deadline)
        setup_claims(baseline)
        if getattr(args, "benchmark_ready_file", None):
            from ..core.benchmark import file_digest, provenance

            ready = Path(args.benchmark_ready_file)
            options = {k: v for k, v in vars(args).items() if not k.startswith("_") and k != "ui_token"}
            config = {
                "schema": "posm-throughput-verifier-v1",
                "response_checks": not skip_checks,
                "options": options,
                "runtime": provenance(),
                "static_check_threads": 0 if skip_checks else args.verify_threads or default_threads(),
                "manifest_sha256": file_digest(args.manifest),
                "static_shape": {"block_bytes": B, "links": n, "matrix_columns": r, "row_bytes": k, "g": g},
                "dynamic_shape": dyn_shape,
                "round_schedule": schedule,
                "static_array_bytes": cfg.nbytes,
                "baseline": baseline,
                "randomness": "independent CSPRNG challenge nonces, starting blocks and exponential waits",
            }
            temporary = ready.with_suffix(ready.suffix + ".tmp")
            temporary.write_text(json.dumps(config, indent=1))
            temporary.replace(ready)
        kind = pick(kinds)[0]
        cursor = {x: 0 for x in kinds}
        run_from = vid  # --max-verifications counts the rounds after the calibration
        while not (args.max_verifications and vid - run_from >= args.max_verifications):
            for name, th in threads.items():
                if not th.is_alive():  # timing without checks proves nothing: stop
                    alert(
                        {
                            "type": "alert",
                            "kind": "check",
                            "id": vid,
                            "detail": f"the {WHAT[name]} thread stopped; the verifier stops",
                        }
                    )
                    raise SystemExit("posm live: a checking thread stopped")
            active = (
                ["honest"]
                if monitor
                else ([s for s in hub.control.get("strategies", strategies) if s in strategies] or ["honest"])
            )
            strategy = active[cursor[kind] % len(active)]
            cursor[kind] += 1
            vid += 1
            ev, misses, now, gross = run_and_assess(vid, kind, strategy)
            misses = report(ev, misses, now, gross)
            if misses and args.confirm_misses:
                # an isolated honest stall (verifier or prover host) passes a fresh challenge; a cheater that is
                # slow by construction misses again. Only a confirmed miss is an alert; drift sees every chain. The
                # retry comes after a random gap like any round (at once, it was a round a prover could tell apart
                # and be ready for); a gross miss has already alerted (`report`), whatever the retry shows.
                between()
                time.sleep(pick(kinds)[1])
                vid += 1
                ev2, misses2, now2, gross2 = run_and_assess(vid, kind, strategy, retry_of=ev["id"])
                misses2 = report(ev2, misses2, now2, gross2)
                publish({"type": "retry", "id": ev["id"], "retried_by": ev2["id"], "miss_confirmed": bool(misses2)})
                for kind_, detail in misses2:
                    alert(
                        {
                            "type": "alert",
                            "kind": kind_,
                            "id": ev2["id"],
                            "detail": detail + f" (confirmed: #{ev['id']} missed too)",
                        }
                    )
            else:
                for kind_, detail in misses:
                    alert({"type": "alert", "kind": kind_, "id": ev["id"], "detail": detail})
            between()
            kind, gap = pick(kinds)
            time.sleep(gap)
    except KeyboardInterrupt:
        if handlers is not None:  # before anything else: a second signal must not cut the stop short
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nlive verifier stopping", flush=True)
    finally:
        if handlers is not None:
            # the stop runs to its end: a second Ctrl-C, or the SIGTERM a stop script sends after its SIGINT, raised
            # KeyboardInterrupt here and cut the last checks and the unchecked report short. Restored below
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        hub.set_meta(state="stopped")
        try:
            judge_open_periods(vid)
        except Exception as e:  # the stop must go on whatever this hits
            print(f"live verifier: the open period could not be judged: {type(e).__name__}: {e}", flush=True)
        for q in queues.values():
            try:
                q.put_nowait(None)
            except _queue.Full:
                pass
        stop_by = time.monotonic() + STOP_CHECKS_S  # their last checks belong in the log, within a bounded stop
        for thread in threads.values():
            thread.join(timeout=max(0.0, stop_by - time.monotonic()))
        stop_checks.set()
        with pend_lock:
            left = {kind_: list(d) for kind_, d in pending.items()}
            for queued in pending.values():
                queued.clear()
        if left["static"] or left["dynamic"]:
            ids = sorted(left["static"] + left["dynamic"])
            detail = (
                f"stopped with {len(ids)} round(s) unchecked ({len(left['static'])} digest checks, {len(left['dynamic'])} record checks; "
                f"first #{ids[0]}): each counts as a failed check"
            )
            publish(
                {"type": "unchecked", "static_ids": left["static"], "dynamic_ids": left["dynamic"], "detail": detail}
            )
            alert({"type": "alert", "kind": "check", "id": vid, "oldest_id": ids[0], "ids": ids, "detail": detail})
            print(f"live verifier: UNCHECKED: {detail}", flush=True)
        with log_lock:
            log_open[0] = False
            if log:
                log.close()
        if handlers is not None:
            signal.signal(signal.SIGTERM, handlers[0])
            signal.signal(signal.SIGINT, handlers[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["verifier", "baseline", "live"])
    ap.add_argument("--manifest", default=None, help="setup-phase manifest (real weights); omit for a synthetic array")
    ap.add_argument("--weight-reference", help="verifier-owned plaintext stream reference")
    ap.add_argument("--weight-registration", help="verifier's pre-seed registration of that reference")
    ap.add_argument("--weight-seed-receipt", help="original verifier-issued seed receipt for that registration")
    ap.add_argument("--gpu", type=int, default=0, help="which GPU's copy the verifier checks")
    ap.add_argument(
        "--state",
        default=None,
        help="verifier: directory with the image the verifier recomputed at setup (`memory_accounting.setup.cli "
        "verify --out`); a setup manifest needs it (or --trust-prover-image)",
    )
    ap.add_argument(
        "--trust-prover-image",
        action="store_true",
        help="verifier, setup manifest without --state: recompute chains over the "
        "image the prover's setup wrote. Nothing ties that file to the weights and filler, so the static array is credited 0 "
        "bytes and --max-scratch is refused",
    )
    ap.add_argument("--size", default="1G")
    ap.add_argument("--encoding", default="vde", choices=["vde", "none"])
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument(
        "--B",
        default=None,
        help="block size (synthetic arrays; default 1M). With --manifest the manifest's; a different value is refused",
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument(
        "--r",
        type=int,
        default=None,
        help="matmul width r (synthetic arrays; default 2048). With --manifest the manifest's; a different value is refused",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="matmul row bytes k (synthetic arrays; default 1024). With --manifest the manifest's; a different value is refused",
    )
    ap.add_argument(
        "--min-r",
        type=int,
        default=2048,
        help="verifier policy: refuse a manifest whose r is below this (r = 0 is hash-only and cannot "
        "tell HBM from host memory; 2048 is the measured recommendation)",
    )
    ap.add_argument(
        "--allowed-B", default="256K,512K,1M", help="verifier policy: block sizes a manifest may use (comma list)"
    )
    ap.add_argument(
        "--allowed-k", default="1024", help="verifier policy: matmul row sizes k a manifest may use (comma list)"
    )
    ap.add_argument("--strategies", default=",".join(STRATS))
    ap.add_argument(
        "--drain",
        action="store_true",
        help="verifier/baseline: arm the prover before each challenge (drain window); only the challenge round trip is timed",
    )
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="after the static chains, pause inference and chain over the KV cache + "
        "activation region, then fetch the evidence",
    )
    ap.add_argument(
        "--dyn-B",
        default="64K",
        help="dynamic challenge block size (the prover pads its region to its own block size; match it)",
    )
    ap.add_argument("--dyn-n", type=int, default=None, help="dynamic chain length (default: --n)")
    ap.add_argument(
        "--gpu-type",
        default=None,
        help="verifier/baseline: the prover's GPU type as the verifier knows it (memory_accounting.verifier.hardware.CAPACITY); "
        "with it the run reports how much of the GPU's memory no chain covered",
    )
    ap.add_argument(
        "--hbm-bytes", default=None, help="verifier/baseline: the GPU's capacity in bytes, instead of --gpu-type"
    )
    ap.add_argument(
        "--dyn-dtype",
        default="bfloat16",
        choices=DYN_DTYPES,
        help="verifier: dtype of the prover's dynamic buffers as the "
        "verifier knows it (the model it declared), which sets how much of the dynamic region is credited: its filler's "
        "incompressible bits, 59/64 for bfloat16 and float16 (the default, the least), 61/64 float32, 62/64 float64 and integer "
        "buffers (the integer model: int8 KV, float64 scales and activations)",
    )
    ap.add_argument(
        "--dyn-encoded",
        action="store_true",
        help="verifier --dynamic: the prover holds its dynamic region encoded (v0.17 "
        "integer and vLLM provers: the KV cache and activation rows as VDE words, plaintext only in scratch), so the whole "
        "region, live region included, is credited at 62/64 (--dyn-dtype is then unused); every record must say its region "
        "is encoded",
    )
    ap.add_argument(
        "--max-scratch",
        default=None,
        help="verifier: fail verifications that leave more than this much memory unchallenged "
        "or uncredited (scratch, plus what the chains cover but the claim does not credit; e.g. 6G); not with --trust-prover-image, "
        "which credits no static memory, nor with a plain dynamic region (no --dyn-encoded) without --with-dyn-live, which credits none of it",
    )
    ap.add_argument(
        "--claim-F",
        type=float,
        default=None,
        help="live/baseline: test the claim 'at most this fraction of reads deviate' "
        "from the mean response time over windows of challenges (memory_accounting.verifier.timing_claim); off when unset",
    )
    ap.add_argument(
        "--claim-C",
        type=float,
        default=0.999,
        help="live/baseline: confidence of the timing claim (missed-cheat rate 1 - C)",
    )
    ap.add_argument(
        "--claim-alpha",
        type=float,
        default=None,
        help="live/baseline: honest false-alarm rate of the claim test (default 1 - C)",
    )
    ap.add_argument(
        "--claim-cap-us",
        type=float,
        default=None,
        help="live/baseline window test: override the absolute response cap, µs "
        "(default: selected per chain from warmup before calibration, or reused from --baseline)",
    )
    ap.add_argument(
        "--claim-window",
        type=int,
        default=None,
        help="live/baseline window test: override the fixed number of challenge "
        "attempts per test (default: selected per chain from warmup before calibration, or reused from --baseline)",
    )
    ap.add_argument(
        "--claim-delta-c",
        type=float,
        default=17.0,
        help="live: smallest extra time of one deviating read, µs (PCIe read of a 1 MiB block)",
    )
    ap.add_argument(
        "--claim-delta-c-dyn",
        type=float,
        default=None,
        help="live --dynamic: smallest extra time of one deviating read on "
        "the dynamic and live chains, µs (default: measured for the --dyn-B block read over PCIe, DYN_DELTA_C_US: 0.3 for "
        "64 KiB, 3.5 for 256 KiB, 17 for 1 MiB)",
    )
    ap.add_argument(
        "--claim-miss-rate",
        type=float,
        default=0.01,
        help="live, with a timing claim: honest deadline misses a chain "
        "(static, dynamic or live) may have (0.09 to 0.45 %% measured on an H200 NVL host with the demo's deadlines; twice the most)",
    )
    ap.add_argument(
        "--claim-miss-window",
        type=int,
        default=100,
        help="live, with a timing claim: the miss count of each chain is over "
        "its last this many round trips (alarm at the count an honest prover reaches with probability at most the claim's alpha "
        "per window, testing after every round trip)",
    )
    ap.add_argument(
        "--claim-max-fill-s",
        type=float,
        default=1800.0,
        help="live, with a timing claim: a chain whose window (at the "
        "calibration's pace) or period takes longer than this to fill alerts on every miss of it again, as without a claim "
        "(its miss count still runs); raise it for long periods (scripts/period_run.sh's hour periods)",
    )
    ap.add_argument(
        "--claim-M-factor",
        type=float,
        default=1.0,
        help="live/baseline: automatic cap uses the warmup's robust mean + M, M = this × L·Δ_c (default 1; at least 1)",
    )
    ap.add_argument(
        "--claim-sigma-margin",
        type=float,
        default=0.1,
        help="live: the claim sizes its test on the calibration's σ × (1 + this): "
        "the design log's margin for dependence between round trips (v0.2, Issue #2)",
    )
    ap.add_argument(
        "--claim-drift-us",
        type=float,
        default=0.0,
        help="live, window model: how far the honest level may drift from the "
        "calibration over a window, µs: the threshold rises by it and the cheat's shift must clear it twice (a margin on σ cannot "
        "absorb a drift, which does not average away as N grows). 0 (the default) is the design log v0.2 as written; 4 µs kept "
        "every honest window of 3.4 h of LAN data from an H200 NVL host under the threshold, where window means wandered with sd 4.7 µs",
    )
    ap.add_argument(
        "--challenge-flags",
        choices=["warn", "alert"],
        default="warn",
        help="live, with a timing claim: what a single challenge "
        "does on a chain whose mean-shift claim runs (a gross round trip, a missing reply, the miss count): warn (the design log "
        "v0.2: no single challenge raises an alert; the claim weighs it, a missing reply at the clip; it is logged and shown as a "
        "flag) or alert (an alert at once, outside the claim's α). A READY that is missing or past --ready-ceiling × its bound "
        "counts its round's chains at the clip, so withholding or delaying READY gains a prover nothing",
    )
    ap.add_argument(
        "--claim-mode",
        choices=["window", "period"],
        default="window",
        help="live: judge the claim over rolling windows of the "
        "N chains a window needs (window), or over all chains of a fixed period, analysed when it ends (period)",
    )
    ap.add_argument(
        "--claim-period-s", type=float, default=3600.0, help="live, period model: length of a claim period, s"
    )
    ap.add_argument(
        "--claim-sigma-s",
        type=float,
        default=8.0,
        help="live: slow movement of the honest level, µs (5 to 10 measured on an H200 NVL host): the period model's σ_s; "
        "in the window model a dynamic or live chain's mean-shift claim runs only if its τ is at least this",
    )
    ap.add_argument(
        "--claim-t-corr", type=float, default=600.0, help="live, period model: how long the honest level stays put, s"
    )
    ap.add_argument(
        "--gross-floor-ms",
        type=float,
        default=10.0,
        help="live: a round trip is gross (an alert at once) only over both "
        "--claim-ceiling × its deadline and its deadline plus this; below, a slow round is a miss the miss count judges",
    )
    ap.add_argument(
        "--ready-ceiling",
        type=float,
        default=3.0,
        help="live: a READY over this many times its bound alerts at once; "
        "a late READY under it is counted (k of the last --claim-miss-window late alerts)",
    )
    ap.add_argument(
        "--claim-ceiling",
        type=float,
        default=3.0,
        help="live, with a timing claim: a single round trip over this many times "
        "its deadline is gross: a flag or an alert at once (--challenge-flags); other misses are weighed by the claim",
    )
    ap.add_argument(
        "--timing",
        choices=["user", "kernel"],
        default="user",
        help="what a timed round trip is: user-space send to "
        "receive, or the kernel's driver TX to driver RX (Linux; leaves out this host's own scheduling jitter)",
    )
    ap.add_argument(
        "--confirm-misses",
        action="store_true",
        help="live: a verification that fails only on time is re-run, after a random "
        "gap like any round, with a fresh challenge; it alerts only if that misses too (an honest host stall passes the retry, a "
        "slow cheater does not). A round trip over --claim-ceiling times its deadline alerts at once, whatever the retry",
    )
    ap.add_argument(
        "--kernel-ts",
        action="store_true",
        help="verifier/baseline: also report each timed round trip from kernel "
        "software timestamps (Linux), to separate this host's own latency from the network and the prover",
    )
    ap.add_argument(
        "--with-dyn-live",
        dest="dyn_live",
        action="store_true",
        help="a second dynamic chain, over the live region only (the live KV "
        "and activations), after the chain over all dynamic memory: every dynamic round then reads the live data. Off by "
        "default. A plain dynamic region (no --dyn-encoded) needs it to be credited at all, since without it the live KV "
        "cannot be told apart; an encoded one is credited whole either way",
    )
    ap.add_argument(
        "--dyn-jitter-ms",
        type=float,
        default=0.0,
        help="baseline/verifier: an extra random delay (uniform, up to this) before "
        "each dynamic ARM, on top of the random wait (mean --interval / 2) every dynamic round of these roles already has, as in "
        "live. Not used by live",
    )
    ap.add_argument(
        "--evidence-dir",
        default=None,
        help="verifier/live: keep the fetched stage records here for memory_accounting.verifier.consistency (those "
        "that describe the chain sent; saved under the challenge's nonce, compact JSON)",
    )
    ap.add_argument(
        "--evidence-max-total",
        default="20G",
        help="verifier/live: stop saving records once --evidence-dir holds this much "
        "(one alert, kind evidence; each check whose record was not kept says record_saved false, and an offline audit with "
        "the log skips those). With --replay the records are still replayed online; without, every dynamic round alerts from "
        "then on (nothing checks it). Size it to the run: a vLLM freeze's two records are about 0.2 to 0.5 MB",
    )
    ap.add_argument(
        "--evidence-max-header",
        default="4M",
        help="verifier/live: largest record header read (a vLLM live record is about 80 "
        "bytes per request, layer and KV block plus its tokens: 0.4 MB for 16 requests of ~200 tokens over 28 layers)",
    )
    ap.add_argument(
        "--record-budget-s",
        type=float,
        default=60.0,
        help="verifier/live: the fetch and replay of one dynamic record must "
        "finish within this, or its check fails (a prover cannot hold the record checks back with records slow to check)",
    )
    ap.add_argument(
        "--check-fraction",
        type=float,
        default=1.0,
        help="live: digest-check this share of static rounds, at random (1: "
        "every one). For a verifier whose CPU cannot recompute every chain at the challenge rate (long chains, no "
        "AVX-512): every round is still timed; a prover answering wrongly is then caught at this rate per round",
    )
    ap.add_argument(
        "--check-lag-s",
        type=float,
        default=120.0,
        help="live: alert when the oldest unchecked round (digests or records) has waited longer than this",
    )
    ap.add_argument("--dynamic-check-fraction", type=float, default=1.0,
                    help="live: fetch and check this random fraction of dynamic replies (default: every reply)")
    ap.add_argument(
        "--evidence-timeout",
        type=float,
        default=10.0,
        help="verifier/live: the whole fetch of one stage record must finish within "
        "this many seconds (at most --record-budget-s), or its check fails: a prover that trickles its records cannot stall the "
        "checking thread for long, and the lag alert says when slow fetches are what holds the checks back",
    )
    ap.add_argument(
        "--evidence-max-bytes",
        default="256M",
        help="verifier/live: largest region bytes (--evidence-bytes, debugging only) a "
        "record may carry; larger ones are not read and the record is checked without them",
    )
    ap.add_argument(
        "--reply-timeout",
        type=float,
        default=10.0,
        help="verifier/baseline/live: how long to wait for a READY or a reply; "
        "one that does not come fails its verification (an alert in live) and the run goes on",
    )
    ap.add_argument(
        "--replay",
        default=None,
        help="verifier/baseline/live --dynamic: replay each record online and require the exact digest: "
        "toy | int:<integer model file> | hf:<model> (same device type as the prover). Default: store records only",
    )
    ap.add_argument("--replay-device", default="cpu", help="device the verifier replays on")
    ap.add_argument(
        "--replay-max-len",
        type=int,
        default=None,
        help="verifier/baseline/live --replay: the KV cache length (slots) the verifier "
        "replays with, the prover's as the verifier configured it (default: --toy-len for the toy model, 512 otherwise, as "
        "intserve's --max-len); a record of another is refused before anything is allocated. --replay <spec>@<slots> pins it "
        "the same way; 0: the record's (up to the model's)",
    )
    ap.add_argument(
        "--pad-seed",
        default=None,
        help="the hex pad seed the provers were given for their dynamic regions "
        "(default: --filler-seed; records claiming another fail)",
    )
    ap.add_argument(
        "--filler-seed",
        default=None,
        help="verifier: the hex filler seed the verifier minted for the setup; the manifest must carry it",
    )
    ap.add_argument(
        "--toy-len",
        type=int,
        default=None,
        help="--replay: the KV cache length (slots) the prover's toy (default 64) "
        "or integer model (default 512, as intserve) was given",
    )
    ap.add_argument(
        "--verify-threads",
        type=int,
        default=None,
        help="verifier: CPU threads for digest recomputation and replays (default: cores - 2, at most 32); cap it on a shared box",
    )
    ap.add_argument(
        "--mode",
        choices=["research", "monitor"],
        default="research",
        help="live: research = choose strategies per challenge and show them; monitor = plain challenges, verdicts and alerts only (as deployed)",
    )
    ap.add_argument(
        "--cusum-k",
        type=float,
        default=0.5,
        help="live: CUSUM slack per chain in single-chain σ (above the honest level's slow drift, ~0.4σ with locked clocks)",
    )
    ap.add_argument("--round-mix", choices=("equal", "pool-bytes"), default="equal",
                    help="live: select static vs whole-pool dynamic challenges equally or by challenged bytes")
    ap.add_argument("--cusum-h", type=float, default=12.0, help="live: CUSUM alarm threshold")
    ap.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="live: mean random wait with one kind, half this with both kinds; --round-mix chooses their shares; "
        "baseline/verifier --dynamic: a dynamic round waits a random time of mean half of this after the static ones, as in live",
    )
    ap.add_argument(
        "--calibrate",
        type=int,
        default=1000,
        help="live: rounds of each kind for the baseline when --baseline is not given (on the "
        "prover under test, which must be honest meanwhile; the page marks such a baseline self-calibrated; prefer a trusted "
        "--baseline). The READY bounds and a p99.9 deadline are tail estimates: under 1000 rounds the p99.9 is about the "
        "maximum, and a vLLM prover's honest tail runs past it",
    )
    ap.add_argument(
        "--save-baseline",
        default=None,
        help="live: write the calibrated baseline here (reuse with --baseline on restart)",
    )
    ap.add_argument(
        "--max-verifications",
        type=int,
        default=0,
        help="live: stop after this many rounds past the calibration (retries included; 0: run until interrupted)",
    )
    ap.add_argument("--ui-port", type=int, default=8080)
    ap.add_argument("--ui-bind", default="127.0.0.1")
    ap.add_argument(
        "--ui-allow-host",
        action="append",
        default=[],
        help="extra Host name or address the page answers (repeatable), "
        "e.g. the verifier's LAN name with --ui-bind 0.0.0.0; localhost names are always allowed",
    )
    ap.add_argument(
        "--ui-token",
        default=os.environ.get("POSM_UI_TOKEN"),
        help="require this token on every page request (open /?token=... once); default $POSM_UI_TOKEN",
    )
    ap.add_argument(
        "--no-verify-all",
        action="store_true",
        help="baseline: check only the first sample's digests (slow CPU verifiers; the same as --sample-verify 1)",
    )
    ap.add_argument(
        "--sample-verify",
        type=int,
        default=None,
        help="verifier/baseline: check the digests and records of the first N "
        "verifications only (slow CPU verifiers); the rest are reported UNCHECKED. Default: every one",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="verifier/baseline: one verification = min RTT over this many challenges (fresh nonces)",
    )
    ap.add_argument(
        "--arm-max-ms",
        type=float,
        default=None,
        help="drain: READY must arrive within this after a static ARM (default: baseline's 2 x honest p99.9); late READY fails the challenge and, in live, alerts at once (with or without a timing claim)",
    )
    ap.add_argument(
        "--dyn-arm-max-ms",
        type=float,
        default=None,
        help="--dynamic: the dynamic READY must arrive within this after the "
        "freeze's ARM (default: the baseline's, 2 x the honest pause wait's p99.9); a late one fails the round and, in live, alerts at once",
    )
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--verify-all",
        action="store_true",
        help="verifier: check every verification (now the default; the flag is kept for "
        "old command lines; --sample-verify N opts out)",
    )
    ap.add_argument("--skip-response-checks", action="store_true",
                    help="live throughput benchmarking only: time all challenges without checking response contents")
    ap.add_argument("--benchmark-ready-file", help="live: atomically write the calibrated settings JSON for the prover")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--baseline",
        default=None,
        help="verifier/live: baseline.json from the baseline role (or live --save-baseline); reports "
        "pass/fail per challenge. It must match this run's shape, --repeat, --drain, --timing and dynamic chains. Measure it on a "
        "prover you trust (at setup, before it serves) and keep it with the verifier: live without it calibrates on the prover under test",
    )
    ap.add_argument(
        "--deadline-rule",
        choices=["sigma", "p999"],
        default="sigma",
        help="baseline: median + sigmas x core σ (robust to stalls), or honest p99.9 x (1 + margin)",
    )
    ap.add_argument(
        "--sigmas", type=float, default=8.0, help="baseline: deadline distance above the honest median, in core σ"
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="baseline: floor of the deadline window as a fraction of the median (sigma rule), or the p99.9 multiplier (p999 rule)",
    )
    args = ap.parse_args(argv)
    if args.round_mix == "pool-bytes" and args.role != "live":
        ap.error("--round-mix pool-bytes applies to the live verifier")
    if (args.skip_response_checks or args.benchmark_ready_file) and (args.role != "live" or not args.manifest):
        ap.error("benchmark options require the live verifier and a manifest")
    if args.timing == "kernel":
        args.kernel_ts = True
    if args.host:
        # resolve once: sendto() with a host name runs the resolver on every call, and that time (about 1 ms
        # on a host whose resolver goes through systemd-resolved) lands inside every timed round trip
        args.host = socket.getaddrinfo(args.host, args.port, socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]
    try:
        {"verifier": verifier_main, "baseline": baseline_main, "live": live_main}[args.role](args)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
