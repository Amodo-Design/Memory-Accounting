"""Statistics on RTT samples: AUC, separation, deadline-based false-accept rate, config selection."""

from __future__ import annotations

import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# The selection rule's margin: mean cheat RTT >= MIN_RATIO x mean honest RTT against every cheat, on top
# of the AUC and false-accept limits. 3 because the CPU cheater has about 2x of headroom left, so margins
# below ~3x are not relied on. memory_accounting.experiments.plots, memory_accounting.experiments.finalist and memory_accounting.experiments.loadcurve all
# select with it.
MIN_RATIO = 3.0


def auc(honest: np.ndarray, cheat: np.ndarray) -> float:
    """P(cheat RTT > honest RTT) -- area under the ROC of a deadline detector."""
    if len(honest) == 0 or len(cheat) == 0:
        return float("nan")
    u = mannwhitneyu(cheat, honest, alternative="greater").statistic
    return float(u / (len(honest) * len(cheat)))


def core_samples(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Samples within k median-absolute-deviations of the median. Gaussian data loses nothing
    (10 MAD ~ 6.7 sigma); multi-ms scheduler stalls on a 50 us distribution are dropped."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    tol = k * max(mad, 1e-6 * max(abs(med), 1.0))
    core = x[np.abs(x - med) <= tol]
    return core if len(core) >= 3 else x


def with_network(
    samples_ns: np.ndarray, jitter_us: float, rng: np.random.Generator, latency_us: float = 0.0
) -> np.ndarray:
    """Add a network model: fixed one-way latency x2 plus Gaussian jitter with the given std (in us)."""
    if jitter_us <= 0 and latency_us <= 0:
        return samples_ns
    noise = rng.normal(2 * latency_us, jitter_us, size=len(samples_ns)) * 1e3
    return samples_ns + noise


def network_samples(rec: dict, jitter_us: float, latency_us: float = 0.0, seed: int = 0) -> np.ndarray:
    """A record's samples (ns) with the network model added. The draw is seeded by `seed` and the
    record's (prover, B, n, r), so the analysis, the figures and the finalist see the same noise for
    the same record, whatever order they visit the records in."""
    key = [seed, zlib.crc32(rec["prover"].encode()), rec["B"], rec["n"], rec["r"]]
    return with_network(rec["samples_ns"], jitter_us, np.random.default_rng(key), latency_us)


def deadline_of(honest: np.ndarray, deadline_q: float = 0.999) -> float:
    """The accept deadline: the honest quantile of the calibration half (odd-indexed samples), in the
    samples' unit. Every consumer (comparisons, figures) uses this one rule."""
    cal = honest[1::2]
    return float(np.quantile(cal if len(cal) >= 2 else honest, deadline_q))


@dataclass
class Comparison:
    cheat: str
    B: int
    n: int
    r: int
    honest_mean_us: float
    honest_std_us: float
    cheat_mean_us: float
    cheat_std_us: float
    auc: float
    deadline_us: float  # honest quantile of the CALIBRATION half, used as the accept deadline
    far: float  # cheat samples accepted (under the deadline)
    frr: float  # honest samples of the EVALUATION half rejected
    gap_sigma: float  # (cheat_mean - honest_mean) / sqrt(std_h^2 + std_c^2)
    far_count: int = 0  # cheat samples under the deadline
    cheat_n: int = 0  # cheat samples
    honest_n: int = 0  # honest samples (both halves)


def compare(
    honest_ns: np.ndarray, cheat_ns: np.ndarray, cheat: str, B: int, n: int, r: int, deadline_q: float = 0.999
) -> Comparison:
    """The deadline is set on the odd-indexed honest samples and evaluated on the even-indexed ones,
    so a configuration is not scored on the samples that set its own threshold.  With 100 samples the
    p99.9 of 50 is the calibration maximum; far_count/cheat_n says how much resolution that has."""
    honest_us, cheat_us = honest_ns / 1e3, cheat_ns / 1e3
    ev = honest_us[0::2] if len(honest_us[1::2]) >= 2 else honest_us
    deadline = deadline_of(honest_us, deadline_q)
    gap = (cheat_us.mean() - honest_us.mean()) / max(np.sqrt(honest_us.std() ** 2 + cheat_us.std() ** 2), 1e-9)
    far_count = int(np.sum(cheat_us <= deadline))
    return Comparison(
        cheat,
        B,
        n,
        r,
        float(honest_us.mean()),
        float(honest_us.std()),
        float(cheat_us.mean()),
        float(cheat_us.std()),
        auc(honest_us, cheat_us),
        deadline,
        far_count / len(cheat_us),
        float(np.mean(ev > deadline)),
        float(gap),
        far_count,
        len(cheat_us),
        len(honest_us),
    )


def load_results(path: Path, drop_incorrect: bool = True) -> dict:
    """Load a sweep. Records whose digest did not match the verifier are dropped (with a warning):
    a wrong answer is a failed challenge whatever its timing, and a broken kernel must not look fast.
    The dropped records' keys go to d["dropped"], so that the selection still expects their strategies
    (a configuration with a dropped cheat record is not approved; see best_config)."""
    data = json.loads(Path(path).read_text())
    kept, dropped = [], []
    for rec in data["records"]:
        if drop_incorrect and not rec.get("correct", True):
            print(
                f"warning: dropping {rec['prover']} B={rec['B']} n={rec['n']} r={rec['r']} from {path}: digest mismatch",
                file=sys.stderr,
            )
            dropped.append((rec["prover"], rec["B"], rec["n"], rec["r"]))
            continue
        rec["samples_ns"] = np.asarray(rec["samples_ns"], dtype=np.float64)
        kept.append(rec)
    data["records"] = kept
    data["dropped"] = dropped
    return data


def expected_cheats(results: dict) -> list[str]:
    """Every cheat the sweep ran, whether its records were kept or dropped as incorrect."""
    provers = {rec["prover"] for rec in results["records"]} | {k[0] for k in results.get("dropped", [])}
    return sorted(provers - {"honest"})


def index(records: list[dict]) -> dict:
    return {(rec["prover"], rec["B"], rec["n"], rec["r"]): rec for rec in records}


def all_comparisons(
    results: dict, jitter_us: float = 0.0, latency_us: float = 0.0, seed: int = 0, deadline_q: float = 0.999
) -> list[Comparison]:
    ix = index(results["records"])
    cheats = sorted({k[0] for k in ix if k[0] != "honest"})
    out = []
    for (prover, B, n, r), rec in ix.items():
        if prover != "honest":
            continue
        h = network_samples(rec, jitter_us, latency_us, seed)
        for cheat in cheats:
            crec = ix.get((cheat, B, n, r))
            if crec is None:
                continue
            c = network_samples(crec, jitter_us, latency_us, seed)
            out.append(compare(h, c, cheat, B, n, r, deadline_q))
    return out


def best_config(
    comps: list[Comparison], min_auc: float = 0.999, max_far: float = 0.001, min_ratio: float = MIN_RATIO, expected=None
) -> tuple[tuple | None, list[dict]]:
    """Smallest honest cost that beats every cheat. Returns (key, table of all configs).
    min_ratio: additionally require mean cheat RTT >= min_ratio * mean honest RTT for every cheat, so
    the selection prefers robust margins over the cheapest point that passes against *this* cheater.
    expected: the cheats every configuration must have been measured against (expected_cheats(results);
    default: every cheat in `comps`). A configuration missing one, e.g. because its record was dropped as
    incorrect, is not approved; its row lists them under "missing"."""
    expected = set(expected) if expected is not None else {c.cheat for c in comps}
    by_cfg: dict[tuple, list[Comparison]] = {}
    for c in comps:
        by_cfg.setdefault((c.B, c.n, c.r), []).append(c)
    table = []
    for key, lst in by_cfg.items():
        missing = sorted(expected - {x.cheat for x in lst})
        row = {
            "B": key[0],
            "n": key[1],
            "r": key[2],
            "honest_mean_us": lst[0].honest_mean_us,
            "honest_std_us": lst[0].honest_std_us,
            "min_auc": min(x.auc for x in lst),
            "max_far": max(x.far for x in lst),
            "weakest_cheat": min(lst, key=lambda x: x.auc).cheat,
            "min_gap_sigma": min(x.gap_sigma for x in lst),
            "min_ratio": min(x.cheat_mean_us / max(x.honest_mean_us, 1e-9) for x in lst),
            "ok": not missing
            and all(
                x.auc >= min_auc and x.far <= max_far and x.cheat_mean_us >= min_ratio * x.honest_mean_us for x in lst
            ),
        }
        if missing:
            row["missing"] = missing  # only when incomplete, so complete sweeps' tables are unchanged
        table.append(row)
    table.sort(key=lambda t: t["honest_mean_us"])
    ok = [t for t in table if t["ok"]]
    return ((ok[0]["B"], ok[0]["n"], ok[0]["r"]) if ok else None), table
