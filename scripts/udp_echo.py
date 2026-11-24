"""The network's own share of a challenge's round trip: a plain UDP echo between the verifier and the prover host,
with no GPU work (for the writeup's timing analysis: how much of the honest jitter is the network).

    python scripts/udp_echo.py serve --port 5599                       # on the prover host
    python scripts/udp_echo.py probe --host <prover> --port 5599 \\
        --count 20000 --rate 20 --out echo.json                        # on the verifier host

The probe sends a 48-byte datagram (the size of a challenge) at random times averaging --rate a second, times each
round trip in user space (perf_counter_ns, as the live verifier's --timing user does) and, with --kernel (Linux),
also as the live verifier's --timing kernel does (driver TX to driver RX, through memory_accounting.verifier.net's own helpers, which
leaves out this host's scheduling and wake-up delays), and writes them with their send times. A reply that does
not come within --timeout is recorded as null.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PKT = struct.Struct("<QQ32x")  # (magic, sequence), padded to 48 bytes like a challenge
MAGIC = 0x4543484F50524F42  # "ECHOPROB"


def serve(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    print(f"udp_echo: serving on :{port}", flush=True)
    while True:
        data, addr = s.recvfrom(64)
        if len(data) == PKT.size and PKT.unpack(data)[0] == MAGIC:
            s.sendto(data, addr)


def probe(host: str, port: int, count: int, rate: float, timeout: float, out: str, kernel: bool) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from memory_accounting.verifier import net  # the live verifier's send, receive and kernel-stamp helpers

    args = SimpleNamespace(
        host=host, port=port, kernel_ts=kernel, timing="kernel" if kernel else "user", reply_timeout=timeout
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    if kernel:
        net._enable_kernel_ts(sock)
    rows = []  # (send time, user-space rtt ns, kernel rtt ns)
    for seq in range(count):
        time.sleep(random.expovariate(rate))
        pkt = PKT.pack(MAGIC, seq)
        t_send = time.time()
        t0 = time.perf_counter_ns()
        net._send(sock, args, pkt)
        user = krtt = None
        while True:
            try:
                data, _addr = net._recv_timed(sock, args, "echo", timeout)
            except (TimeoutError, BlockingIOError):
                break
            if data == pkt:  # a late reply to an earlier probe is not this one's
                user, krtt = args._t_recv - t0, (args._last_krtt if kernel else None)
                break
        rows.append([t_send, user, krtt])
        if seq % 1000 == 0:
            print(f"udp_echo: {seq}/{count}", flush=True)
    with open(out, "w") as f:
        json.dump(
            {
                "host": host,
                "port": port,
                "rate": rate,
                "size": PKT.size,
                "kernel": kernel,
                "columns": ["t", "user_rtt_ns", "kernel_rtt_ns"],
                "rows": rows,
            },
            f,
        )
    for col, name in ((1, "user"), (2, "kernel")):
        got = sorted(r[col] for r in rows if r[col] is not None)
        if got:
            print(
                f"udp_echo: {name}: {len(got)}/{count} replies; median {got[len(got) // 2] / 1e3:.1f} us, "
                f"p99 {got[int(len(got) * 0.99)] / 1e3:.1f} us",
                flush=True,
            )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=5599)
    p = sub.add_parser("probe")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=5599)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--timeout", type=float, default=0.5)
    p.add_argument("--out", default="echo.json")
    p.add_argument(
        "--kernel", action="store_true", help="also time with kernel software timestamps (Linux), as --timing kernel"
    )
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.port)
    else:
        probe(a.host, a.port, a.count, a.rate, a.timeout, a.out, a.kernel)


if __name__ == "__main__":
    main()
