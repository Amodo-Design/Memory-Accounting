"""Statistical timing claim: test the sum of many challenges' response times, not each one alone.

Implements the "Statistical test design log" v0.2 (after "MAcc - Number of challenges for k = δ"):

* claim: at most F of the reads deviate from the declared data (a deviating read costs at least Δ_c extra),
  detected with confidence C; an honest prover is flagged with probability at most α (default 1 − C);
* model: X_i = H_i + Δ_c · K_i, H_i honest (baseline mean μ, sd σ), K_i the cheated reads among the chain's L;
* no single response raises a flag (Issue #1): each is recorded as min(X_i, μ + M), M = L·Δ_c by default
  (never less: a clipped response counts as K_i = L, Issue #3), and raised to at least μ − M, so every term of
  the sum lies within M of μ, as Bernstein's inequality needs; a missing reply counts at the clip;
* test: D = Σ min(X_i, μ + M) − N·μ; the prover passes if D < t;
* Bernstein's inequality on each side. Honest: Var(D) = N·σ²·(1 + N/N_cal), the test's own noise and the
  baseline mean's error, which is in every test (2·N·σ² with a baseline as long as the test). Cheating at F:
  E[D] = N·L·F·Δ_c and Var(D) adds N·L·F·Δ_c·M (a value in [0, M] with mean L·F·Δ_c), however the cheater
  spreads its cheating. t is the least threshold that keeps false positives at α; the test meets the claim
  when what is left below N·L·F·Δ_c also keeps false negatives at 1 − C. N is the least N for which it does
  (both inequalities as equalities: the log's closed form, without its dropped terms);
* dependence (Issue #2): a margin on σ (`sigma_margin`); a bound h on how far the honest level drifts from the
  calibration over a test (`drift`, µs: the honest sum may sit N·h above N·μ and a cheater's N·h below its
  expected shift, so t gains N·h and the room N·(L·F·Δ_c − 2h)), since a margin on σ cannot absorb a drift that
  does not average away as N grows; and in the period model a term for the slow movement of the honest level
  between the calibration and the period, σ_s²·(1/K + 1/K_cal) on Var(D)/N².

Times are in microseconds throughout. `plan()` sizes a test; `estimate_baseline()` and `slow_movement()`
measure its inputs from honest data; `MeanShiftTest` (windows of N) and `PeriodAnalysis` (fixed periods) run it.

With an independently fixed cap `c`, baseline and test use min(X, c), including missing replies as inf.
`plan(p, n)` checks the supplied window size without sizing it. The arbitrary-prover variance bound is
n·c²/4 + n²·σ²/N_cal and the range is c·max(1, n/N_cal). The sum statistic and Bernstein helper are unchanged.
This mode assumes independent samples, stable honest timing and a valid inflated honest variance bound;
L·F·Δ_c must lower-bound the increase in the clipped mean. Deadline alarms are separate from its α.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np


@dataclass
class ClaimParams:
    """What is claimed and what is known about honest timing (µs)."""

    F: float  # claimed bound on the fraction of deviating reads
    C: float  # detection confidence (false negatives at most 1 − C)
    L: int  # reads per challenge chain (links)
    delta_c: float  # smallest extra time per deviating read (µs)
    sigma_b: float  # honest standard deviation of a clipped response (µs), as measured
    alpha: float | None = None  # false-positive rate; default 1 − C
    M: float | None = None  # clip above μ (µs); default L·Δ_c, never below it
    sigma_margin: float = 0.0  # σ used = σ_b · (1 + sigma_margin): the log's margin for dependence
    drift: float = 0.0  # bound on the honest level's drift from the calibration over a test (µs)
    mu_b: float = 0.0  # honest (clipped) mean μ (µs)
    x_min: float | None = None  # fastest calibration response (µs): with σ_b, a floor for bad timestamps
    n_cal: float | None = None  # calibration responses; None: as many as the test (the log's 2·N·σ²)
    sigma_s: float = 0.0  # slow movement of the honest level (µs); 0 leaves it out
    t_corr: float = 0.0  # time over which it stays the same (s)
    k_cal: float | None = None  # independent stretches in the calibration; default K
    n_gpu: int = 1  # GPUs in the claim
    T: float = 0.0  # claim window or period (s)
    c: float | None = None  # absolute cap fixed before baseline/test (µs); None retains the legacy test

    def __post_init__(self):
        if not 0 < self.F < 1 or not 0 < self.C < 1 or self.L < 1 or self.delta_c <= 0:
            raise ValueError("need 0 < F < 1, 0 < C < 1, L ≥ 1, Δ_c > 0")
        if self.alpha is not None and not 0 < self.alpha < 1:
            raise ValueError("need 0 < α < 1")
        if self.c is not None:
            if not math.isfinite(self.c) or self.c <= 0:
                raise ValueError("need a finite cap c > 0")
            if not math.isfinite(self.delta_c) or not math.isfinite(self.L) or int(self.L) != self.L:
                raise ValueError("need finite Δ_c and a positive integer chain length")
            if self.n_cal is None or not math.isfinite(self.n_cal) or self.n_cal < 2 or int(self.n_cal) != self.n_cal:
                raise ValueError("fixed-cap test needs at least two baseline attempts")
            if not 0 <= self.mu_b <= self.c or not math.isfinite(self.sigma_b) or self.sigma_b < 0:
                raise ValueError("need a clipped baseline mean in [0, c] and finite nonnegative SD")
            if not math.isfinite(self.sigma_margin) or self.sigma_margin < 0:
                raise ValueError("need a finite nonnegative SD margin")
            if self.drift or self.sigma_s:
                raise ValueError("fixed-cap test requires drift = sigma_s = 0")
        if self.M is not None and self.M < self.L * self.delta_c * (1 - 1e-9):
            # below L·Δ_c a chain cheated on every read is clipped short of its whole cost, and F would no longer
            # bound the cheating the sum sees (the log, v0.2 "Value of M")
            raise ValueError(f"M = {self.M} µs is below L·Δ_c = {self.L * self.delta_c} µs")

    @property
    def a(self) -> float:
        return self.alpha if self.alpha is not None else 1 - self.C

    @property
    def M_(self) -> float:
        return self.M if self.M is not None else self.L * self.delta_c

    @property
    def sigma(self) -> float:
        return self.sigma_b * (1 + self.sigma_margin)

    @property
    def delta_F(self) -> float:
        return self.L * self.F * self.delta_c

    @property
    def clip(self) -> float:
        return self.c if self.c is not None else self.mu_b + self.M_

    @property
    def floor(self) -> float:
        """The lowest response the tests take: μ − M, so a term of the sum is within M of μ, or the calibration's
        fastest minus 3 σ_b if that is higher (a clock step under kernel timestamps, a bad timestamp, would
        otherwise pull the sum down and hide a slow cheat)."""
        if self.c is not None:
            return 0.0
        lo = self.mu_b - self.M_
        return max(lo, self.x_min - 3 * self.sigma_b) if self.x_min is not None else lo

    @property
    def K(self) -> float:
        return self.n_gpu * self.T / self.t_corr if self.t_corr > 0 and self.T > 0 else math.inf


def bernstein_gap(var: float, rng: float, lg: float) -> float:
    """The least g with exp(−g² / (2·var + ⅔·rng·g)) ≤ e^−lg: how far a sum of independent terms, each within
    rng of its mean, with variance var, strays above (or below) its mean with probability at most e^−lg."""
    h = rng * lg / 3
    return h + math.sqrt(h * h + 2 * lg * var)


def slow_var(params: ClaimParams) -> float:
    """The variance slow movement of the honest level adds to the mean of a test against the calibration:
    σ_s²·(1/K + 1/K_cal), K the independent stretches of the test, K_cal of the calibration (each at least 1)."""
    if params.sigma_s <= 0:
        return 0.0
    K = params.K
    K_cal = params.k_cal if params.k_cal is not None else K

    def inv(k):
        return 1 / max(1.0, k) if math.isfinite(k) else 0.0

    return params.sigma_s**2 * (inv(K) + inv(K_cal))


def var_honest(p: ClaimParams, n: float) -> float:
    """Var(D) for an honest prover over n responses: n·σ² of the test, n²·σ²/N_cal of the baseline mean, n²
    times the slow movement."""
    s2 = p.sigma**2
    n_cal = p.n_cal if p.n_cal is not None else n
    return n * s2 + n * n * s2 / n_cal + n * n * slow_var(p)


def var_cheat(p: ClaimParams, n: float) -> float:
    """Var(D) bound: arbitrary clipped prover with fixed c; legacy extra-cost model otherwise."""
    if p.c is not None:
        return n * p.c**2 / 4 + n * n * p.sigma**2 / p.n_cal
    return var_honest(p, n) + n * p.delta_F * p.M_


def term_range(p: ClaimParams, n: float) -> float:
    """How far one term of D can stray from its mean: M for a test response; the baseline mean enters D as
    n/N_cal times each calibration response, so M·n/N_cal when the test is longer than the calibration."""
    rng = p.c if p.c is not None else p.M_
    return rng * max(1.0, n / p.n_cal) if p.n_cal else rng


def threshold(p: ClaimParams, n: float) -> float:
    """t: the least threshold on D at which an honest prover over n responses is flagged with probability at
    most α (ineq 1), its level up to `drift` above the calibration's."""
    return bernstein_gap(var_honest(p, n), term_range(p, n), math.log(1 / p.a)) + n * p.drift


def fn_gap(p: ClaimParams, n: float) -> float:
    """How far below N·L·F·Δ_c the sum of a prover cheating at F falls with probability at most 1 − C (ineq 2),
    its level up to `drift` below the calibration's."""
    return bernstein_gap(var_cheat(p, n), term_range(p, n), math.log(1 / (1 - p.C))) + n * p.drift


def room(p: ClaimParams, n: float) -> float:
    """N·L·F·Δ_c − t − the false-negative gap: at least 0 when n responses test the claim at F."""
    return n * p.delta_F - threshold(p, n) - fn_gap(p, n)


def n_required(params: ClaimParams, n_max: float = 1e15) -> float:
    """The least N whose test meets both α and C (inf: none up to n_max, e.g. under the floor on F that the
    calibration's error and the slow movement set)."""
    if room(params, 1) >= 0:
        return 1.0
    hi = 2.0
    while room(params, hi) < 0:
        hi *= 2
        if hi > n_max:
            return math.inf
    lo = hi / 2
    while hi - lo > 1:
        mid = math.floor((lo + hi) / 2)
        if room(params, mid) >= 0:
            hi = mid
        else:
            lo = mid
    return float(hi)


def n_closed_form(p: ClaimParams) -> float:
    """The log's closed form, √N = (√2·B + √(2B² + ⁴⁄₃·L·F·Δ_c·M·(a + b))) / (2·L·F·Δ_c), B = √2·σ·√a +
    √(2σ² + L·F·Δ_c·M)·√b: the baseline as long as the test, no slow movement, the (aM/3)² terms dropped."""
    if p.c is not None:
        raise ValueError("the legacy closed form does not apply to a fixed cap")
    a, b = math.log(1 / p.a), math.log(1 / (1 - p.C))
    s, LFD, M = p.sigma, p.delta_F, p.M_
    B = math.sqrt(2) * s * math.sqrt(a) + math.sqrt(2 * s * s + LFD * M) * math.sqrt(b)
    return ((math.sqrt(2) * B + math.sqrt(2 * B * B + 4 / 3 * LFD * M * (a + b))) / (2 * LFD)) ** 2


def F_floor(p: ClaimParams) -> float:
    """The least F any N can test: as N grows the calibration's error (σ²/N_cal, its range M·N/N_cal) and the
    slow movement stay in every test while the test's own noise averages away. 0 with neither."""
    v = (p.sigma**2 / p.n_cal if p.n_cal else 0.0) + slow_var(p)
    c = (p.c if p.c is not None else p.M_) / p.n_cal if p.n_cal else 0.0

    def g(lg):
        return c * lg / 3 + math.sqrt((c * lg / 3) ** 2 + 2 * lg * v)

    return (g(math.log(1 / p.a)) + g(math.log(1 / (1 - p.C))) + 2 * p.drift) / (p.L * p.delta_c)


def F_testable(p: ClaimParams, n: float, hi: float = 0.999) -> float:
    """The least F that n responses can test (nan: not even F = hi): what a passing test over n shows."""

    def at(F):
        return room(replace(p, F=F), n)

    if at(hi) < 0:
        return math.nan
    lo = 1e-12
    if at(lo) >= 0:
        return lo
    for _ in range(200):  # room grows with F where it crosses 0: bisect on log F
        mid = math.sqrt(lo * hi)
        if at(mid) >= 0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0001:
            break
    return hi


def plan(params: ClaimParams, n: int | None = None) -> dict:
    """Check a supplied N for fixed c; otherwise size the legacy claim. τ = t/N is the mean threshold."""
    if params.c is not None:
        if n is None or not math.isfinite(n) or n < 1 or int(n) != n:
            raise ValueError("fixed-cap test needs a positive integer window size")
        thresh = threshold(params, n)
        return {
            "delta_F": params.delta_F,
            "c": params.c,
            "sigma": params.sigma,
            "V_c": params.c**2 / 4,
            "n_window": int(n),
            "t_us": thresh,
            "tau": thresh / n,
            "F_floor": F_floor(params),
            "supported": n * params.delta_F >= thresh + fn_gap(params, n),
        }
    if n is not None:
        raise ValueError("a supplied window size requires a fixed cap c")
    n = n_required(params)
    thresh = threshold(params, n) if math.isfinite(n) else math.nan
    out = {
        "delta_F": params.delta_F,
        "M": params.M_,
        "clip_above_mu": params.M_,
        "sigma": params.sigma,
        "V_c": params.delta_F * params.M_,
        "n_required": n,
        "t_us": thresh,
        "tau": thresh / n if math.isfinite(n) else math.nan,
        "F_floor": F_floor(params),
    }
    if params.n_cal is None and params.sigma_s <= 0:
        out["n_closed_form"] = n_closed_form(params)
    if params.T > 0 and math.isfinite(n):
        out["challenges_per_gpu_per_s"] = n / (params.n_gpu * params.T)
    return out


# ---- measuring the inputs from honest data


def estimate_baseline(x_us, L: int, delta_c: float, M: float | None = None, *, c: float | None = None) -> dict:
    """μ and σ of the calibration clipped as a test clips it, at μ + M (M = L·Δ_c by default) and up to μ − M.
    μ is the mean of the clipped values, a fixed point: clipping lowers the mean, which lowers the cap; the map is
    monotone with slope below 1, so it converges from the median in a few steps. Honest stalls in the
    calibration count at the cap, as they will in a test. Also x_min and, for comparison, the median, the
    central-sample σ and the upper quantiles of the raw responses. With fixed c, use only upper clipping;
    all attempts (including inf for no reply) enter the sample mean and SD."""
    raw = np.asarray(x_us, dtype=np.float64)
    if c is not None:
        if not math.isfinite(c) or c <= 0 or raw.ndim != 1 or len(raw) < 2:
            raise ValueError("need a finite cap c > 0 and at least two baseline attempts")
        if np.any(np.isnan(raw) | (raw < 0)):
            raise ValueError("invalid RTT: expected nonnegative times or inf for no reply")
        x = np.minimum(raw, c)
        finite = raw[np.isfinite(raw)]
        q1, q3 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (math.nan, math.nan)
        return {
            "n": len(raw),
            "mu_b": float(x.mean()),
            "sigma_b": float(x.std(ddof=1)),
            "c": c,
            "M": None,
            "clip": c,
            "clipped": int(np.sum(raw > c)),
            "x_min": float(finite.min()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "core_sigma_iqr": float((q3 - q1) / 1.349) if len(finite) else None,
            "p99": float(np.quantile(finite, 0.99)) if len(finite) else None,
            "p999": float(np.quantile(finite, 0.999)) if len(finite) >= 1000 else None,
        }
    M = M if M is not None else L * delta_c
    mu = float(np.median(raw))
    for _ in range(200):
        new = float(np.clip(raw, mu - M, mu + M).mean())
        if abs(new - mu) <= 1e-9 * max(1.0, abs(mu)):
            mu = new
            break
        mu = new
    x = np.clip(raw, mu - M, mu + M)
    q1, q3 = np.quantile(raw, [0.25, 0.75])
    return {
        "n": len(raw),
        "mu_b": mu,
        "sigma_b": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "M": M,
        "clip": mu + M,
        "clipped": int(np.sum(raw > mu + M)),
        "x_min": float(raw.min()),
        "median": float(np.median(raw)),
        "core_sigma_iqr": float((q3 - q1) / 1.349),
        "p99": float(np.quantile(raw, 0.99)),
        "p999": float(np.quantile(raw, 0.999)) if len(raw) >= 1000 else None,
    }


def params_from_baseline(base: dict, F: float, C: float, L: int, delta_c: float, **kw) -> ClaimParams:
    """The claim's parameters from `estimate_baseline` of its calibration."""
    return ClaimParams(
        F=F,
        C=C,
        L=L,
        delta_c=delta_c,
        sigma_b=base["sigma_b"],
        mu_b=base["mu_b"],
        x_min=base["x_min"],
        M=base["M"],
        n_cal=base["n"],
        c=base.get("c"),
        **kw,
    )


def slow_movement(t_s, x_us, windows_s=None) -> dict:
    """How the standard deviation of a window mean falls as the window grows. With independent responses
    it falls as σ_b/√n; slow movement of the honest mean shows as a floor. For each window length the
    observed variance of window means is compared with σ_b²/n̄ and the excess is an estimate of σ_s²."""
    times = np.asarray(t_s, dtype=np.float64)
    x = np.asarray(x_us, dtype=np.float64)
    order = np.argsort(times)
    times, x = times[order], x[order]
    span = times[-1] - times[0]
    if windows_s is None:
        windows_s = [w for w in (5, 10, 20, 30, 60, 120, 300, 600, 1200) if w <= span / 3]
    s2 = x.var(ddof=1)
    rows = []
    for window in windows_s:
        idx = np.floor((times - times[0]) / window).astype(int)
        means, counts = [], []
        for k in np.unique(idx):
            sel = idx == k
            if sel.sum() >= 3:
                means.append(x[sel].mean())
                counts.append(sel.sum())
        if len(means) < 3:
            continue
        nbar = float(np.mean(counts))
        obs = float(np.var(means, ddof=1))
        expect = s2 / nbar
        rows.append(
            {
                "window_s": window,
                "windows": len(means),
                "mean_n": nbar,
                "sd_of_window_means": math.sqrt(obs),
                "sd_if_independent": math.sqrt(expect),
                "excess_var": obs - expect,
            }
        )
    excess = [r["excess_var"] for r in rows if r["window_s"] >= 30]
    sigma_s = math.sqrt(max(0.0, float(np.median(excess)))) if excess else None
    return {"rows": rows, "sigma_s_estimate": sigma_s}


# ---- running the test


def _clipped(p: ClaimParams, xs) -> tuple[np.ndarray, int]:
    x = np.asarray(xs, dtype=np.float64)
    if p.c is not None:
        if x.ndim != 1 or np.any(np.isnan(x) | (x < 0)):
            raise ValueError("invalid RTT: expected nonnegative times or inf for no reply")
        return np.minimum(x, p.c), int(np.sum(x > p.c))
    over = int(np.sum(x > p.clip))
    return np.clip(x, p.floor, p.clip), over


@dataclass
class MeanShiftTest:
    """Collects response times (a missing reply as inf: it counts at the clip) and judges each window of
    n_window of them: pass if D = Σ min(X_i, μ + M) − n·μ < t, t the threshold for n responses."""

    p: ClaimParams
    n_window: int
    xs: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)

    def __post_init__(self):
        if self.p.c is not None and not plan(self.p, self.n_window)["supported"]:
            raise ValueError("the supplied window does not support the timing claim")
        self.t = threshold(self.p, self.n_window)

    def add(self, x_us: float) -> dict | None:
        if self.p.c is not None and (math.isnan(x_us) or x_us < 0):
            raise ValueError("invalid RTT: expected nonnegative times or inf for no reply")
        self.xs.append(float(x_us))
        if len(self.xs) < self.n_window:
            return None
        v = self.judge(self.xs)
        self.verdicts.append(v)
        self.xs = []
        return v

    def judge(self, xs) -> dict:
        if self.p.c is not None and len(xs) != self.n_window:
            raise ValueError("fixed-cap test requires exactly the configured window size")
        x, clipped = _clipped(self.p, xs)
        n = len(x)
        thresh = self.t if n == self.n_window else threshold(self.p, n)
        D = float(x.sum() - n * self.p.mu_b)
        shift, tau = D / n, thresh / n
        # D_us and t_us, not D and t: a published verdict's "t" is the event's time (memory_accounting.verifier.ui's hub, the log)
        return {
            "n": n,
            "D_us": D,
            "t_us": thresh,
            "mean_shift": shift,
            "tau": tau,
            "pass": D < thresh,
            # the share of deviating reads this shift would mean under the claim's model (capped at 1); a window
            # mostly at the clip is a cheat slower per read than Δ_c, beyond what the model describes
            "implied_F": min(1.0, max(0.0, shift) / (self.p.L * self.p.delta_c)),
            "clipped": clipped,
            "saturated": clipped * 2 >= n,
            # the honest level ran well below the calibration: a cheater has that much extra slack, so the
            # claim is not assured until the baseline is measured again (a warning, not a failure)
            "baseline_moved": shift < -tau / 2,
        }

    def progress(self) -> dict:
        x, clipped = _clipped(self.p, self.xs)
        return {
            "n": len(x),
            "n_window": self.n_window,
            "tau": self.t / self.n_window,
            "mean_shift": float(x.mean() - self.p.mu_b) if len(x) else None,
            "clipped": clipped,
        }


# ---- deadline misses of a chain (single-challenge flags: outside the claim, see memory_accounting.verifier.net --challenge-flags)


def miss_threshold(window: int, q: float, alpha: float) -> int:
    """The least k with P(Binomial(window, q) ≥ k) ≤ alpha: a count of misses an honest prover missing at
    rate q reaches in a window with probability at most alpha."""
    if q <= 0:
        return 1
    tail, pmf = 1.0, (1 - q) ** window
    for k in range(window + 1):
        if tail <= alpha:
            return k
        tail -= pmf
        pmf *= (window - k) / (k + 1) * q / (1 - q)
    return window + 1


def _binom_tail(n: int, q: float, k: int) -> float:
    """P(Binomial(n, q) ≥ k)."""
    if k <= 0:
        return 1.0
    if k > n or q <= 0:
        return 0.0
    return sum(math.comb(n, i) * q**i * (1 - q) ** (n - i) for i in range(k, n + 1))


def scan_threshold(window: int, q: float, alpha: float) -> int:
    """The least k for which a sliding count of misses over the last `window` round trips, tested after every
    round trip, alarms on an honest prover missing at rate q with probability at most alpha per `window` round
    trips. Testing at every step is `window` tests per window, not one: `miss_threshold` (one test) let 2.3e-3
    alarms through per 100 chains at alpha 1e-3. The count first reaches k at a step whose own round trip
    missed, with k − 1 misses among the window − 1 before it, so a window's steps alarm with probability at
    most window · q · P(Binomial(window − 1, q) ≥ k − 1) (a union bound; k is the least that keeps it ≤ alpha)."""
    if q <= 0:
        return 1
    for k in range(1, window + 1):
        if window * q * _binom_tail(window - 1, q, k - 1) <= alpha:
            return k
    return window + 1


@dataclass
class MissCountTest:
    """The deadline misses of one chain over its last `window` round trips: alarm when they reach k, then start
    over. k is sized for a sliding test (`scan_threshold`): an honest prover missing at rate q alarms with
    probability at most alpha per `window` round trips. For chains whose mean shift is not testable (the
    dynamic and live chains: a 64 KiB block from host memory costs ~2 µs, below the honest level's wander),
    and as a backstop on the static chains under a claim, so that a prover late on every chain, below the
    ceiling, alerts after k chains instead of at the end of a window or period. It assumes misses independent;
    honest stalls cluster a little, which q's margin over the measured honest rate absorbs."""

    window: int
    q: float
    alpha: float
    k: int = 0
    hist: deque = field(default_factory=deque)
    alarms: int = 0
    count: int = 0

    def __post_init__(self):
        if self.window < 1 or not 0 <= self.q < 1 or not 0 < self.alpha < 1:
            raise ValueError("need window ≥ 1, 0 ≤ q < 1, 0 < alpha < 1")
        self.k = scan_threshold(self.window, self.q, self.alpha)
        self.hist = deque(maxlen=self.window)

    def add(self, miss: bool) -> bool:
        if len(self.hist) == self.window:
            self.count -= self.hist[0]  # the oldest leaves the window as the newest enters
        self.hist.append(int(bool(miss)))
        self.count += int(bool(miss))
        if self.count >= self.k:
            self.alarms += 1
            self.hist.clear()
            self.count = 0
            return True
        return False

    def progress(self) -> dict:
        return {"misses": self.count, "misses_n": len(self.hist), "misses_window": self.window, "misses_k": self.k}


# ---- the period model: all responses of a period, judged at its end


@dataclass
class PeriodAnalysis:
    """All the chains of a period, judged at its end with the same test at the N the period holds. Keeps a
    running sum of the clipped responses, so a period of millions costs nothing to hold.

    The prover fails if D ≥ t(n): an honest prover does with probability at most α, whatever n. A period that
    passes shows the claim at every F its n can test (F_testable): at the target F when n reaches the N the
    target needs, otherwise the test is inconclusive at the target. The slow movement of the honest level over
    K = elapsed / t_corr independent stretches (K_cal for the calibration) is in Var(D)."""

    p: ClaimParams  # target claim; mu_b, M, n_cal, sigma_s, t_corr, k_cal from the calibration
    n: int = 0
    s1: float = 0.0
    clipped: int = 0
    t0: float | None = None
    t_last: float | None = None

    def __post_init__(self):
        if self.p.c is not None:
            raise ValueError("fixed-cap tests use MeanShiftTest with a supplied window size")

    def add(self, x_us: float, t: float) -> None:
        if self.t0 is None:
            self.t0 = t
        self.t_last = t
        if x_us > self.p.clip:
            self.clipped += 1
        self.n += 1
        self.s1 += min(max(x_us, self.p.floor), self.p.clip)

    def _p_now(self) -> ClaimParams:
        elapsed = (self.t_last - self.t0) if self.n > 1 else 0.0
        return replace(self.p, T=max(elapsed, 1e-9))

    def stats(self) -> dict:
        n = self.n
        elapsed = (self.t_last - self.t0) if n > 1 else 0.0
        q = self._p_now()
        D = self.s1 - n * self.p.mu_b
        t = threshold(q, n) if n else math.nan
        shift = D / n if n else math.nan
        return {
            "n": n,
            "elapsed_s": elapsed,
            "D_us": D,
            "t_us": t,
            "mean_shift": shift,
            "tau": t / n if n else math.nan,
            "clipped": self.clipped,
            "implied_F": min(1.0, max(0.0, shift) / (self.p.L * self.p.delta_c)) if n else math.nan,
            # the least F this many chains can show if the period passes, and what the slow movement allows at all
            "F_testable": F_testable(q, n) if n else math.nan,
            "F_floor": F_floor(q),
        }

    def n_required(self) -> float:
        """The N the test at the target F needs, with the slow movement over the full period."""
        return n_required(self.p)

    def verdict(self) -> dict:
        """End of the period (or the open period at stop): fail if D ≥ t; otherwise pass when the period held
        enough chains for the target F, else inconclusive."""
        st = self.stats()
        q = self._p_now()
        enough = st["n"] > 0 and room(q, st["n"]) >= 0
        test = "fail" if st["n"] and st["D_us"] >= st["t_us"] else ("pass" if enough else "inconclusive")
        return {
            **st,
            "target_F": self.p.F,
            "n_required": self.n_required(),
            "test": test,
            "supports_target": test == "pass",
            "V_c": self.p.delta_F * self.p.M_,
        }


def analyze_period(
    cal_us, cal_span_s, period_us, period_t, F, C, L, delta_c, sigma_s=0.0, t_corr=600.0, M=None, sigma_margin=0.0
):
    """The period model over recorded responses (µs): calibration `cal_us` taken over `cal_span_s`, then the
    claim period `period_us` at times `period_t` (s)."""
    base = estimate_baseline(cal_us, L, delta_c, M)
    T = max(1.0, (period_t[-1] - period_t[0]) if len(period_t) > 1 else 1.0)
    params = params_from_baseline(
        base,
        F,
        C,
        L,
        delta_c,
        sigma_margin=sigma_margin,
        sigma_s=sigma_s,
        t_corr=t_corr,
        T=T,
        k_cal=max(1.0, cal_span_s / t_corr) if t_corr > 0 else None,
    )
    pa = PeriodAnalysis(params)
    for x, t in zip(period_us, period_t):
        pa.add(float(x), float(t))
    pl = plan(params)
    return {
        "baseline": base,
        "plan": {
            "n_required": pl["n_required"],
            "F_floor": pl["F_floor"],
            "delta_F": params.delta_F,
            "tau": pl["tau"],
            "t_us": pl["t_us"],
            "M": params.M_,
        },
        "verdict": pa.verdict(),
    }


# ---- command line

# a saved baseline's calibration, per chain
CAL_SAMPLES = {
    "static": "static_chain_rtt_ns",
    "dynamic": "dyn_rtt_ns",
    "live": "live_rtt_ns",
}


def chain_rtts_us(e: dict, chain: str = "static") -> list:
    """The round trips (µs) of one logged verification on a chain (static: each static challenge; dynamic:
    the chain over all dynamic memory; live: the live chain). A missing reply is logged as rtt_ns null: it
    is a fault, not a round trip, so it is None here and the readers count and drop it."""
    if chain == "static":
        return [c["rtt_ns"] / 1e3 if c.get("rtt_ns") is not None else None for c in e.get("challenges", [])]
    d = e.get("dyn") or {}
    if chain == "live":
        d = d.get("live")
        if d is None:
            return []
    elif not d:
        return []
    return [d["rtt_ns"] / 1e3 if d.get("rtt_ns") is not None else None]


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv=None):
    import argparse
    import csv
    import json

    ap = argparse.ArgumentParser(description="statistical timing claim: plan a test, or measure its inputs from a run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "analyze"):
        sub_parser = sub.add_parser(name)
        sub_parser.add_argument("--F", type=float, default=1e-3)
        sub_parser.add_argument("--C", type=float, default=1 - 1e-9)
        sub_parser.add_argument("--L", type=int, default=16)
        sub_parser.add_argument("--delta-c", type=float, default=17.0, help="µs per deviating read")
        sub_parser.add_argument("--alpha", type=float, default=None)
        sub_parser.add_argument("--M", type=float, default=None, help="clip above μ, µs (default L·Δ_c)")
        sub_parser.add_argument("--sigma-margin", type=float, default=0.0, help="σ used = σ_b × (1 + this)")
        sub_parser.add_argument(
            "--drift",
            type=float,
            default=0.0,
            help="bound on the honest level's drift from the calibration over a test, µs",
        )
        sub_parser.add_argument("--sigma-s", type=float, default=0.0)
        sub_parser.add_argument("--t-corr", type=float, default=0.0)
        sub_parser.add_argument("--n-gpu", type=int, default=1)
        sub_parser.add_argument("--T", type=float, default=0.0, help="claim window, s")
        sub_parser.add_argument("--k-cal", type=float, default=None)
        sub_parser.add_argument("--json", action="store_true")
        if name == "plan":
            sub_parser.add_argument("--sigma-b", type=float, default=25.0)
            sub_parser.add_argument("--mu-b", type=float, default=0.0)
            sub_parser.add_argument(
                "--n-cal", type=float, default=None, help="calibration responses (default: as many as the test)"
            )
        else:
            sub_parser.add_argument("csv", help="verifications.csv from a run (honest rows are used)")
            sub_parser.add_argument("--column", default="static_rtt_us")
    sub_parser = sub.add_parser("period", help="the period model over a live verifier's log (live_events.jsonl)")
    sub_parser.add_argument("log")
    sub_parser.add_argument("--F", type=float, default=1e-3)
    sub_parser.add_argument("--C", type=float, default=0.999)
    sub_parser.add_argument(
        "--chain",
        choices=list(CAL_SAMPLES),
        default="static",
        help="which timed chain: the static chains, the dynamic "
        "chain or the live chain (for those, --delta-c of their block: memory_accounting.verifier.net DYN_DELTA_C_US, 17 µs for 1 MiB)",
    )
    sub_parser.add_argument("--L", type=int, default=8)
    sub_parser.add_argument("--delta-c", type=float, default=17.0)
    sub_parser.add_argument("--M", type=float, default=None, help="clip above μ, µs (default L·Δ_c)")
    sub_parser.add_argument("--sigma-margin", type=float, default=0.1, help="σ used = σ_b × (1 + this)")
    sub_parser.add_argument(
        "--sigma-s",
        default="8",
        help="slow movement of the honest level, µs (5 to 10 measured on an H200 NVL host), or 'auto' to estimate it from the period",
    )
    sub_parser.add_argument("--t-corr", type=float, default=600.0, help="how long the honest level stays put, s")
    sub_parser.add_argument(
        "--calibration",
        default=None,
        help="baseline.json with calibration_samples (default: the log's calibration rows)",
    )
    sub_parser.add_argument("--from-time", type=float, default=None)
    sub_parser.add_argument("--to-time", type=float, default=None)
    sub_parser.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "period":
        ev = []
        with open(args.log) as f:
            lines = f.readlines()
        for line in lines:
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:  # a torn line (the log of a run still writing, or cut short)
                continue
        verifications = [e for e in ev if e.get("type") == "verification" and "t" in e]
        missing = {"calibration": 0, "period": 0}  # replies that never came (rtt_ns null): faults
        if args.calibration:
            with open(args.calibration) as f:
                b = json.load(f)
            cs = b.get("calibration_samples") if isinstance(b, dict) else None
            if not isinstance(cs, dict):
                raise SystemExit(
                    f"{args.calibration} keeps no calibration_samples (a baseline-role file from before they were kept): "
                    "recalibrate, use a live --save-baseline file, or leave out --calibration to use the log's calibration rows"
                )
            raw = cs.get(CAL_SAMPLES[args.chain]) or []
            if not raw:
                raise SystemExit(
                    f"{args.calibration} has no {args.chain} calibration responses ({CAL_SAMPLES[args.chain]})"
                )
            cal = [x / 1e3 for x in raw if x is not None]
            missing["calibration"] = len(raw) - len(cal)
            span = float(cs.get("span_s") or len(cal) * 0.6)
        else:
            cv = [e for e in verifications if e.get("phase") == "calibration" and not e.get("warmup")]
            raw = [x for e in cv for x in chain_rtts_us(e, args.chain)]
            cal = [x for x in raw if x is not None]
            missing["calibration"] = len(raw) - len(cal)
            span = cv[-1]["t"] - cv[0]["t"] if len(cv) > 1 else 0.0
        run = [
            e
            for e in verifications
            if e.get("phase") != "calibration"
            and (args.from_time is None or e["t"] >= args.from_time)
            and (args.to_time is None or e["t"] <= args.to_time)
        ]
        pairs = [(x, e["t"]) for e in run for x in chain_rtts_us(e, args.chain)]
        missing["period"] = sum(x is None for x, _ in pairs)
        # a calibration reply that never came is left out (the baseline is honest timing); one in the period
        # counts at the clip, as the live verifier counts it (K_i = L)
        xs = [x if x is not None else math.inf for x, _ in pairs]
        ts = [t for _, t in pairs]
        if len(cal) < 2 or len(xs) < 2:
            raise SystemExit(f"{args.log}: too few {args.chain} round trips (calibration {len(cal)}, period {len(xs)})")
        fin = [(x, t) for x, t in pairs if x is not None]
        sm = slow_movement([t for _, t in fin], [x for x, _ in fin])
        sigma_s = (sm["sigma_s_estimate"] or 0.0) if str(args.sigma_s) == "auto" else float(args.sigma_s)
        out = analyze_period(
            cal, span, xs, ts, args.F, args.C, args.L, args.delta_c, sigma_s, args.t_corr, args.M, args.sigma_margin
        )
        out["slow_movement"] = sm
        out["missing_replies"] = missing
        out["chain"] = args.chain
        if args.json:
            print(json.dumps(out, indent=1, default=float))
            return
        vd = out["verdict"]
        est = out["slow_movement"]["sigma_s_estimate"]
        print(
            f"slow movement measured in the period: σ_s ≈ {est:.2f} µs (windows ≥ 30 s); used: {sigma_s:.2f} µs per {args.t_corr:.0f} s"
            if est is not None
            else f"slow movement: period too short to measure; used: {sigma_s:.2f} µs per {args.t_corr:.0f} s"
        )
        if missing["calibration"] or missing["period"]:
            print(
                f"without a reply (rtt_ns null; each was a fault): {missing['calibration']} calibration {args.chain} challenges left out, "
                f"{missing['period']} period ones counted at the clip"
            )
        bl = out["baseline"]
        print(
            f"calibration: {len(cal)} responses over {span:.0f} s; μ {bl['mu_b']:.1f} µs, σ {bl['sigma_b']:.1f} µs (clipped at μ + M = {bl['clip']:.1f} µs)"
        )
        print(
            f"period: {vd['n']} responses over {vd['elapsed_s']:.0f} s; D = {vd['D_us']:+.1f} µs (mean {vd['mean_shift']:+.2f} µs), clipped {vd['clipped']}"
        )
        print(
            f"threshold t = {vd['t_us']:.1f} µs (mean {vd['tau']:.3f} µs): an honest prover exceeds it with probability at most "
            f"{1 - args.C:g}: test {vd['test']}" + (" -- the claim fails" if vd["test"] == "fail" else "")
        )
        print(
            f"target F = {args.F}: needs N = {vd['n_required']:.3g}, has {vd['n']}; the least F these chains can show: {vd['F_testable']:.4g}; "
            f"floor from the calibration and slow movement: F ≥ {vd['F_floor']:.3g}"
            if math.isfinite(vd["n_required"])
            else f"target F = {args.F}: no N can test it, below the slow-movement floor F ≥ {vd['F_floor']:.3g}; "
            f"the least F these chains can show: {vd['F_testable']:.4g}"
        )
        return
    common = {
        "F": args.F,
        "C": args.C,
        "L": args.L,
        "delta_c": args.delta_c,
        "alpha": args.alpha,
        "M": args.M,
        "sigma_margin": args.sigma_margin,
        "drift": args.drift,
        "sigma_s": args.sigma_s,
        "t_corr": args.t_corr,
        "n_gpu": args.n_gpu,
        "T": args.T,
        "k_cal": args.k_cal,
    }
    if args.cmd == "plan":
        params = ClaimParams(sigma_b=args.sigma_b, mu_b=args.mu_b, n_cal=args.n_cal, **common)
        out = {"params": asdict(params), "plan": plan(params)}
    else:
        with open(args.csv, newline="") as f:
            honest = [r for r in csv.DictReader(f) if r.get("true_posture", "honest") == "honest"]
        rows = [r for r in honest if _finite(r.get(args.column))]  # no reply: an empty, null or nan cell, left out
        if len(rows) < len(honest):
            print(f"left out: {len(honest) - len(rows)} honest rows without a {args.column} (no reply)")
        x = [float(r[args.column]) for r in rows]
        t = [float(r["time_unix"]) for r in rows]
        base = estimate_baseline(x, args.L, args.delta_c, args.M)
        slow = slow_movement(t, x)
        common.pop("M")
        params = params_from_baseline(
            base, **{**common, "sigma_s": common["sigma_s"] or (slow["sigma_s_estimate"] or 0.0)}
        )
        out = {"baseline": base, "slow_movement": slow, "params": asdict(params), "plan": plan(params)}
    if args.json:
        print(json.dumps(out, indent=1, default=float))
        return
    for section, d in out.items():
        print(f"[{section}]")
        if isinstance(d, dict):
            for k, verifications in d.items():
                if k == "rows":
                    for r in verifications:
                        print("   ", "  ".join(f"{a}={_fmt(b)}" for a, b in r.items()))
                elif isinstance(verifications, dict):
                    print(f"  {k}: " + ", ".join(f"{a}={_fmt(b)}" for a, b in verifications.items()))
                else:
                    print(f"  {k}: {_fmt(verifications)}")


if __name__ == "__main__":
    main()
