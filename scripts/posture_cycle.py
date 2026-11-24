"""Labelled live run: switch the demo prover through postures, SLOT seconds each, until a deadline.

Run on the verifier host, beside a running `scripts/demo.sh start sim` (whose page takes the switches on
localhost:8765). Waits for the calibration to end (the claim's meta event in demo_live.jsonl), then plays
rounds: each round is every posture once, in a fresh random order, so that the honest level's slow drift
falls on no posture more than another, and every posture ends with the same number of slots. A round is
started only if it can end by the deadline. At the end the prover goes back to honest and the run log is
copied to runs/<name>_<date>.jsonl.

The default postures are honest and each cheat on 1% of the array. The demo must offer them (cpu@f also
needs cpu itself, for its thread pool):

    POSM_SIM_CHEATS=honest,cpu,encode_otf@0.01,partial@0.01,other_gpu@0.01,dram_copy@0.01,cpu@0.01 \
        scripts/demo.sh start sim

    nohup python3 scripts/posture_cycle.py --until 08:00 > cycle.log 2>&1 &     # on the verifier host

Standard library only, so it runs with any python3 on the verifier host.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shutil
import time
import urllib.request
from pathlib import Path

POSTURES = ["honest", "encode_otf@0.01", "partial@0.01", "other_gpu@0.01", "dram_copy@0.01", "cpu@0.01"]


def deadline(spec: str) -> float:
    """HH:MM: the next such local time (tomorrow if it has passed today); or a number of hours from now."""
    if re.fullmatch(r"\d{1,2}:\d{2}", spec):
        h, m = map(int, spec.split(":"))
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= now:
            t += dt.timedelta(days=1)
        return t.timestamp()
    return time.time() + float(spec) * 3600


def rounds(postures: list[str], n: int, rng: random.Random) -> list[list[str]]:
    """n rounds, each a random order of every posture, the first of a round never the last of the one before
    (that would make one stretch of two slots)."""
    out: list[list[str]] = []
    for _ in range(n):
        r = postures[:]
        rng.shuffle(r)
        while out and len(r) > 1 and r[0] == out[-1][-1]:
            rng.shuffle(r)
        out.append(r)
    return out


def post(url: str, posture: str) -> str:
    req = urllib.request.Request(
        url, data=json.dumps({"posture": posture}).encode(), method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode().strip()
    except OSError as e:  # logged; the run goes on (a missed switch shows in the log's posture events)
        return f"failed: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--postures", default=",".join(POSTURES), help="comma-separated postures, each once per round")
    ap.add_argument("--slot", type=float, default=40, help="seconds per posture")
    ap.add_argument(
        "--until", default="08:00", help="end by this local time (HH:MM, the next one) or after this many hours"
    )
    ap.add_argument("--rounds", type=int, default=0, help="play this many rounds instead (no deadline)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the orders (default: random; printed)")
    ap.add_argument("--dir", default="~/LVHEM_MACC_SW", help="the verifier's checkout (demo_live.jsonl, runs/)")
    ap.add_argument("--url", default="http://localhost:8765/posture")
    ap.add_argument("--name", default="onepct_live", help="the copied log is runs/<name>_<date>.jsonl")
    ap.add_argument("--no-wait", action="store_true", help="start at once, without waiting for the calibration")
    args = ap.parse_args()
    postures = [p for p in args.postures.split(",") if p]
    if len(set(postures)) != len(postures):
        ap.error("each posture once: a round is every posture once")
    out_dir = Path(args.dir).expanduser()
    log = out_dir / "demo_live.jsonl"
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 32)
    rng = random.Random(seed)
    if not args.no_wait:
        pat = re.compile(r'"type": "meta".*"claim"')
        while not (log.exists() and any(pat.search(line) for line in log.open(errors="replace"))):
            time.sleep(10)
    t0 = time.time()
    per_round = len(postures) * args.slot
    n = args.rounds or int((deadline(args.until) - t0) // per_round)
    if n < 1:
        raise SystemExit(f"no whole round ({per_round:.0f} s) fits before {args.until}")
    plan = rounds(postures, n, rng)
    end = dt.datetime.fromtimestamp(t0 + n * per_round).strftime("%Y-%m-%d %H:%M")
    print(
        f"{t0:.0f} calibrated; {n} rounds of {len(postures)} postures x {args.slot:g} s, ending about {end}; seed {seed}",
        flush=True,
    )
    print(f"{t0:.0f} postures: {' '.join(postures)}", flush=True)
    for i, r in enumerate(plan, 1):
        for p in r:
            reply = post(args.url, p)
            print(f"{time.time():.0f} round {i} posture {p}: {reply}", flush=True)
            if reply.startswith("failed:"):
                raise RuntimeError(f"round {i} posture {p} was not accepted: {reply}")
            time.sleep(args.slot)
    reply = post(args.url, "honest")
    print(f"{time.time():.0f} back to honest: {reply}", flush=True)
    if reply.startswith("failed:"):
        raise RuntimeError(f"could not restore honest posture: {reply}")
    (out_dir / "runs").mkdir(exist_ok=True)
    out = out_dir / "runs" / f"{args.name}_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    shutil.copyfile(log, out)
    print(f"{time.time():.0f} done: {out}", flush=True)


if __name__ == "__main__":
    main()
