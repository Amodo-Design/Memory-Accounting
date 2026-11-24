"""The wire format between verifier and prover: datagram layouts, magics and the strategy ids both sides agree on.

Datagram: challenge = 8 x u64 (magic, nonce, j0, B, n, r, k, g) + strategy id; reply = 3 x u64 (nonce, digest lo,
digest hi). An ARM drains (or, for dynamic memory, freezes) inference before the timed challenge; READY answers it.
"""

from __future__ import annotations

import socket
import struct

MAGIC = 0x504F534D
# a cheat on a share of the array (posture only): partial@f has f of the array in host memory; the share
# versions of the other cheats run them on f of it (memory_accounting.prover.strategies.ShareCheat); at 100% they are the cheat
FRACTION_POSTURES = ["partial@0.01", "partial@0.05", "partial@0.1", "partial@0.25", "partial@0.5", "partial@1"]
SHARE_POSTURES = [
    f"{c}@{f}" for c in ("encode_otf", "other_gpu", "dram_copy", "cpu") for f in ("0.01", "0.05", "0.1", "0.25", "0.5")
]
CONTROL_POSTURES = [f"{c}_control@0.01" for c in ("partial", "encode_otf", "other_gpu", "dram_copy", "cpu")]
# wire ids; both sides must agree (new ones only at the end)
STRATS = (
    [
        "honest",
        "other_gpu",
        "encode_otf",
        "dram_copy",
        "dram_mapped",
        "cpu",
        "partial",
        "lora",
        "evict",
        "evict_dram",
        "tamper",
        "dyn_dram",
    ]
    + FRACTION_POSTURES
    + SHARE_POSTURES
    + CONTROL_POSTURES
)
CH = struct.Struct("<8QI")
RE = struct.Struct("<3Q")
ARM_MAGIC = 0x504F5341  # "arm": drain the inference step in flight, then READY
# (magic, token, B, n, r, k, g, sid): shape + strategy let the prover pre-arm the honest chain
ARM = struct.Struct("<8Q")
READY = struct.Struct("<2Q")  # (magic, token)
STATS_MAGIC = 0x504F5353  # prover-side counters for the live page
STATS = struct.Struct("<2Q")  # request (magic, token); only ever received by a prover, so no clash with READY
# (magic, token, served, fired, armed, hold_expired, svc_p50, svc_p99, dt_p50, dt_p99, posture sid, wire mask, posture
# mask): bit i = STRATS[i] answered on the wire / offered as a posture
STATSRE = struct.Struct("<13Q")
POSTURE_MAGIC = 0x504F5350  # demo control: set the prover's posture (which strategy answers every challenge)
# (magic, token, sid); reply echoes the posture in force. A real verifier has no such channel.
POSTURE = struct.Struct("<3Q")
# Dynamic memory: a challenge over the KV cache + activation region. ARM_DYN freezes inference at the next
# layer or step boundary (instead of draining the step) and pre-arms over the region; READY_DYN tells the verifier
# the region's size so it can pick a start block; the challenge itself is a CH datagram under DYN_MAGIC. After
# the reply the prover serves the stage record over TCP (memory_accounting.core.evidence); the verifier replays inference to it,
# rebuilds the region and recomputes the chain (memory_accounting.verifier.consistency).
ARM_DYN_MAGIC = 0x504F5344  # "POSD"
DYN_MAGIC = 0x504F5359  # "POSY": the chain over all dynamic memory (under vLLM, mostly free pool)
DYN_LIVE_MAGIC = 0x504F534C  # "POSL": the second chain, over the live region of the same freeze
LIVE_FLAG = 1 << 16  # in a dynamic ARM's strategy field: also build the live region
# (magic, token, region words, block bytes, flags: bit0 mid-step, bit1 step boundary, 0 idle; live region words, 0 =
# none)
READY_DYN = struct.Struct("<6Q")


def strategy_mask(names) -> int:
    """The STATS reply's strategy bit mask."""
    return sum(1 << STRATS.index(s) for s in set(names) if s in STRATS)


def resolve_peers(addrs) -> set[str] | None:
    """The prover's --verifier-addr: the IPv4 addresses it accepts datagrams from (IP addresses or host
    names, a comma list or repeated), resolved once at start. None or empty: any sender. Without it anyone
    who reaches the port can ARM the prover (a freeze of inference each time), set its posture or read its
    counters."""
    if not addrs:
        return None
    names = [x.strip() for a in ([addrs] if isinstance(addrs, str) else addrs) for x in str(a).split(",") if x.strip()]
    out: set[str] = set()
    for name in names:
        try:
            out |= {info[4][0] for info in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_DGRAM)}
        except socket.gaierror as e:
            raise ValueError(f"verifier address {name!r} does not resolve: {e}") from None
    return out or None
