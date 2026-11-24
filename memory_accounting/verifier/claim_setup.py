"""Convenience defaults from discarded warmup; the statistical test itself stays fixed."""

import math
from dataclasses import replace

import numpy as np

from .timing_claim import estimate_baseline, n_required, params_from_baseline


def automatic_claim_setup(pilot_ns, args, L, delta_c, n_cal):
    """Freeze c and N using only warmup, before the separate scored baseline.

    The old cap estimator is only a setup heuristic. Planning uses the new bounds;
    the actual baseline must subsequently support this same N, without resizing.
    """
    out = {
        "cap_us": args.claim_cap_us,
        "window": args.claim_window,
        "pilot_count": len(pilot_ns),
        "planned_baseline_count": n_cal,
    }
    if out["cap_us"] is not None and out["window"] is not None:
        return out
    raw = np.array([math.inf if x is None else x for x in pilot_ns], dtype=float) / 1e3
    if raw.ndim != 1 or np.any(np.isnan(raw) | (raw < 0)):
        raise ValueError("invalid warmup RTT")
    if out["cap_us"] is None:
        finite = raw[np.isfinite(raw)]
        if not len(finite):
            return {
                **out,
                "reason": "automatic cap selection needs a received warmup reply; use --warmup or supply --claim-cap-us",
            }
        # Reuse the previous robust mean + M rule, on warmup only.
        out["cap_us"] = estimate_baseline(finite, L, delta_c, args.claim_M_factor * L * delta_c)["clip"]
    if out["window"] is not None:
        return out
    if len(raw) < 2 or n_cal < 2:
        return {
            **out,
            "reason": "automatic window sizing needs at least two warmup attempts and two planned baseline attempts",
        }
    if args.claim_F is None:
        return {**out, "reason": "set --claim-F when collecting the baseline to choose a window automatically"}
    base = estimate_baseline(raw, L, delta_c, c=out["cap_us"])
    p = params_from_baseline(
        base, args.claim_F, args.claim_C, L, delta_c, alpha=args.claim_alpha, sigma_margin=args.claim_sigma_margin
    )
    count = n_required(replace(p, n_cal=n_cal))
    if not math.isfinite(count):
        return {
            **out,
            "reason": f"warmup planning cannot support F = {args.claim_F} with {n_cal} baseline attempts; increase calibration or choose a larger F",
        }
    out["window"] = 2 * int(count)  # headroom for pilot uncertainty; the scored baseline still checks it
    return out
