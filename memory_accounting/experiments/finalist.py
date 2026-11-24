"""Re-measure the configuration a sweep selected, with many more samples, so the false-accept and
false-reject rates have real resolution instead of "0 of 100".

  python -m memory_accounting.experiments.finalist results/sweep_vde.json --samples 1000 [--jitter-us 45] [bench options]

The re-run takes the sweep's conditions from its meta (size, encoding, permutation, pattern, k, kernel,
graph replay, device, canonical-copy device, seed, background load), so it measures what the sweep
measured; bench options given after the sweep are appended and override them. A sweep from before the
seed was recorded is re-run with bench's default seed (said so).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.analysis import MIN_RATIO, all_comparisons, best_config, expected_cheats, load_results
from . import bench


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--jitter-us", type=float, default=45.0)
    ap.add_argument("--min-ratio", type=float, default=MIN_RATIO, help="selection rule, as in memory_accounting.experiments.plots")
    ap.add_argument("--out", default=None)
    args, rest = ap.parse_known_args(argv)
    res = load_results(Path(args.sweep))
    key, _table = best_config(
        all_comparisons(res, jitter_us=args.jitter_us), min_ratio=args.min_ratio, expected=expected_cheats(res)
    )
    if key is None:
        print("no passing configuration in", args.sweep, file=sys.stderr)
        sys.exit(1)
    B, n, r = key
    meta = res["meta"]
    out = args.out or str(Path(args.sweep).with_name(Path(args.sweep).stem + f"_finalist_B{B >> 10}K_n{n}_r{r}.json"))
    provers = expected_cheats(res)
    bargs = [
        "--size",
        str(meta["size_bytes"]),
        "--encoding",
        meta["encoding"],
        "--blocks",
        str(B),
        "--links",
        str(n),
        "--r",
        str(r),
        "--k",
        str(meta.get("k", 1024)),
        "--provers",
        ",".join(provers),
        "--samples",
        str(args.samples),
        "--out",
        out,
    ]
    if meta.get("device"):
        bargs += ["--device", str(meta["device"])]
    if meta.get("seed") is not None:
        bargs += ["--seed", str(meta["seed"])]
    else:
        print("finalist: the sweep did not record its seed: bench's default is used", file=sys.stderr)
    if meta.get("canon_device"):
        bargs += ["--canon-device", meta["canon_device"]]
    if not meta.get("perm", True):
        bargs += ["--no-perm"]
    bargs += ["--pattern", meta.get("pattern", "random"), "--kernel", meta.get("kernel") or "auto"]
    if meta.get("graph") is False:
        bargs += ["--no-graph"]
    load = meta.get("background_load")
    if load:
        bargs += [
            "--background-load",
            "--load-fraction",
            str(load["fraction"]),
            "--load-dim",
            str(load["matmul_dim"]),
            "--load-buffer-mb",
            str(load["buffer_bytes"] >> 20),
        ]
    print("finalist:", " ".join(bargs + rest))
    bench.main(bargs + rest)
    fin = load_results(Path(out))
    comps = all_comparisons(fin, jitter_us=args.jitter_us)
    for c in comps:
        print(
            f"{c.cheat:12s} AUC {c.auc:.4f}  accepted {c.far_count}/{c.cheat_n} under honest p99.9 (split-half)  FRR {c.frr:.4f}"
        )
    for cheat in sorted(set(expected_cheats(fin)) - {c.cheat for c in comps}):
        print(f"{cheat:12s} not measured (record dropped): the configuration is not confirmed")


if __name__ == "__main__":
    main()
