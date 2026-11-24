"""Deterministic CPU checks for the independently fixed cap and window timing claim."""

import json
import math
import pathlib
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from memory_accounting.verifier import timing_claim as tc


def _params(**changes):
    values = {
        "F": 0.5,
        "C": 0.99,
        "L": 8,
        "delta_c": 10.0,
        "sigma_b": 5.0,
        "sigma_margin": 0.2,
        "mu_b": 20.0,
        "n_cal": 400,
        "c": 100.0,
    }
    values.update(changes)
    return tc.ClaimParams(**values)


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def _close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-12), (actual, expected)


def _mean_oracle(params, n):
    """Solve the Bernstein quadratic for means, independently of the sum helpers."""
    variance = (params.sigma_b * (1 + params.sigma_margin)) ** 2
    honest_variance = variance / n + variance / params.n_cal
    prover_variance = params.c**2 / (4 * n) + variance / params.n_cal
    weight_range = params.c / min(n, params.n_cal)

    def gap(v, probability):
        log_inverse = -math.log(probability)
        linear = (2 / 3) * weight_range * log_inverse
        return (linear + math.sqrt(linear**2 + 8 * v * log_inverse)) / 2

    alpha = params.alpha if params.alpha is not None else 1 - params.C
    return (
        honest_variance,
        prover_variance,
        weight_range,
        gap(honest_variance, alpha),
        gap(prover_variance, 1 - params.C),
    )


def test_sum_and_mean_formulations_agree_for_short_and_long_baselines():
    """N/B weights the baseline terms, including when the test outlasts its baseline."""
    for b in (32, 400, 2048):
        for n in (8, 64, 512, 4096):
            params = _params(n_cal=b, alpha=0.025)
            vh, vc, r, tau, lower_gap = _mean_oracle(params, n)
            out = tc.plan(params, n)
            _close(tc.var_honest(params, n), n**2 * vh)
            _close(tc.var_cheat(params, n), n**2 * vc)
            _close(tc.term_range(params, n), n * r)
            _close(out["t_us"], n * tau)
            _close(out["tau"], tau)
            _close(tc.fn_gap(params, n), n * lower_gap)
            _close(tc.room(params, n), n * (params.L * params.F * params.delta_c - tau - lower_gap))
            assert out["supported"] == (params.L * params.F * params.delta_c >= tau + lower_gap)
            assert out["n_window"] == n and "n_required" not in out and "n_closed_form" not in out
            assert out["V_c"] == params.c**2 / 4


def test_bernstein_bounds_and_first_supported_integer_window():
    """The first feasible supplied N satisfies both error bounds; its predecessor does not."""
    params = _params(alpha=0.02)
    first = next(n for n in range(1, 1000) if sum(_mean_oracle(params, n)[3:]) <= params.L * params.F * params.delta_c)
    assert first > 1
    assert tc.plan(params, first)["supported"] and not tc.plan(params, first - 1)["supported"]
    _raises_value_error(lambda: tc.MeanShiftTest(params, first - 1))
    tc.MeanShiftTest(params, first)
    for n in (first, 400, 1600):
        vh, vc, r, tau, lower_gap = _mean_oracle(params, n)
        fp_bound = math.exp(-(tau**2) / (2 * vh + (2 / 3) * r * tau))
        fn_bound = math.exp(-(lower_gap**2) / (2 * vc + (2 / 3) * r * lower_gap))
        _close(fp_bound, params.alpha)
        _close(fn_bound, 1 - params.C)
        remaining = params.L * params.F * params.delta_c - tau
        assert remaining >= lower_gap
        assert math.exp(-(remaining**2) / (2 * vc + (2 / 3) * r * remaining)) <= 1 - params.C + 1e-14


def test_support_changes_at_the_independently_calculated_shift_boundary():
    p, n = _params(), 64
    *_, tau, lower_gap = _mean_oracle(p, n)
    boundary_f = (tau + lower_gap) / (p.L * p.delta_c)
    assert 0 < boundary_f < 1
    assert not tc.plan(replace(p, F=boundary_f * (1 - 1e-9)), n)["supported"]
    assert tc.plan(replace(p, F=boundary_f * (1 + 1e-9)), n)["supported"]


def test_exact_support_equality_is_accepted_without_subtraction_cancellation():
    p = _params(c=10, n_cal=100, sigma_b=0, mu_b=0, alpha=0.9, C=0.1, L=1, delta_c=100)
    n = 2
    boundary_f = (tc.threshold(p, n) + tc.fn_gap(p, n)) / (n * p.L * p.delta_c)
    boundary = replace(p, F=boundary_f)
    assert n * boundary.delta_F == tc.threshold(boundary, n) + tc.fn_gap(boundary, n)
    assert tc.plan(boundary, n)["supported"]
    tc.MeanShiftTest(boundary, n)


def test_fixed_cap_planning_never_automatically_sizes_the_window():
    originals = tc.n_required, tc.n_closed_form

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed-cap planning called a legacy sizing function")

    tc.n_required = tc.n_closed_form = forbidden
    try:
        p = _params()
        assert tc.plan(p, 64)["n_window"] == 64
        assert tc.MeanShiftTest(p, 64).n_window == 64
        assert not tc.plan(p, 1)["supported"]
        _raises_value_error(lambda: tc.plan(p))
        _raises_value_error(lambda: tc.MeanShiftTest(p, 1))
    finally:
        tc.n_required, tc.n_closed_form = originals
    _raises_value_error(lambda: tc.n_closed_form(_params()))


def test_baseline_includes_all_attempts_and_uses_sample_standard_deviation():
    raw = [0.0, 0.125, 5.0, 70.0, 100.0, 101.0, math.inf]
    clipped = [0.0, 0.125, 5.0, 70.0, 100.0, 100.0, 100.0]
    mean = sum(clipped) / len(clipped)
    sd = math.sqrt(sum((x - mean) ** 2 for x in clipped) / (len(clipped) - 1))
    base = tc.estimate_baseline(raw, L=8, delta_c=10, c=100)
    assert base["n"] == len(raw) and base["clipped"] == 2
    _close(base["mu_b"], mean)
    _close(base["sigma_b"], sd)
    assert base["clip"] == base["c"] == 100 and base["M"] is None and base["x_min"] == 0
    p = tc.params_from_baseline(base, F=0.5, C=0.99, L=8, delta_c=10)
    assert p.c == 100 and p.n_cal == len(raw) and p.mu_b == base["mu_b"] and p.sigma_b == base["sigma_b"]
    # Changing L and delta_c does not move the independently selected cap or the baseline moments.
    other = tc.estimate_baseline(raw, L=1, delta_c=0.001, c=100)
    assert (other["mu_b"], other["sigma_b"], other["clip"]) == (base["mu_b"], base["sigma_b"], base["clip"])


def test_all_missing_baseline_retains_every_attempt_at_the_cap():
    base = tc.estimate_baseline([math.inf] * 9, L=8, delta_c=10, c=100)
    assert base["n"] == base["clipped"] == 9 and base["mu_b"] == 100 and base["sigma_b"] == 0
    assert all(base[key] is None for key in ("x_min", "median", "core_sigma_iqr", "p99", "p999"))


def test_window_uses_upper_clipping_without_raising_valid_low_rtts():
    p = _params(x_min=99.0)
    raw = [0.0, 0.125, 20.0, p.c, p.c + 1, math.inf] + [p.mu_b] * 58
    clipped = [min(value, p.c) for value in raw]
    expected_sum = sum(clipped) - len(raw) * p.mu_b
    test = tc.MeanShiftTest(p, len(raw))
    verdict = test.judge(raw)
    assert p.floor == 0 and verdict["clipped"] == 2
    _close(verdict["D_us"], expected_sum)
    _close(verdict["mean_shift"], sum(clipped) / len(raw) - p.mu_b)
    _close(verdict["t_us"] / len(raw), verdict["tau"])
    assert verdict["pass"] == (expected_sum < test.t)
    for value in raw[:-1]:
        assert test.add(value) is None
    assert test.progress()["n"] == len(raw) - 1
    assert test.add(raw[-1]) == verdict
    assert test.xs == [] and test.verdicts == [verdict]


def test_equal_to_threshold_flags_and_just_below_passes():
    """N=1 and zero baseline give an exact floating-point equality without a sum-rounding ambiguity."""
    p = _params(F=0.75, C=0.1, alpha=0.9, L=1, delta_c=100, sigma_b=0, mu_b=0, n_cal=64)
    test = tc.MeanShiftTest(p, 1)
    assert 0 < test.t < p.c
    assert test.judge([math.nextafter(test.t, -math.inf)])["pass"]
    assert not test.judge([test.t])["pass"]
    assert not test.judge([math.nextafter(test.t, math.inf)])["pass"]


def test_invalid_fixed_cap_parameters_and_window_sizes_are_rejected():
    for key, values in (
        ("c", (0, -1, math.inf, math.nan)),
        ("n_cal", (None, 0, 1, 2.5, math.inf, math.nan)),
        ("mu_b", (-1, 101, math.inf, math.nan)),
        ("sigma_b", (-1, math.inf, math.nan)),
        ("sigma_margin", (-0.1, math.inf, math.nan)),
        ("delta_c", (0, -1, math.inf, math.nan)),
        ("L", (0, -1, 1.5, math.inf, math.nan)),
        ("drift", (0.1,)),
        ("sigma_s", (0.1,)),
    ):
        for value in values:
            _raises_value_error(lambda key=key, value=value: _params(**{key: value}))
    for n in (None, 0, -1, 2.5, math.inf, math.nan):
        _raises_value_error(lambda n=n: tc.plan(_params(), n))
        _raises_value_error(lambda n=n: tc.MeanShiftTest(_params(), n))


def test_invalid_baseline_shapes_caps_and_rtts_are_rejected():
    for raw in ([], [1.0], 1.0, [[1.0], [2.0]], [0.0, -1.0], [0.0, -math.inf], [0.0, math.nan]):
        _raises_value_error(lambda raw=raw: tc.estimate_baseline(raw, L=8, delta_c=10, c=100))
    for c in (0, -1, math.inf, math.nan):
        _raises_value_error(lambda c=c: tc.estimate_baseline([1.0, 2.0], L=8, delta_c=10, c=c))


def test_invalid_rtts_and_wrong_window_counts_do_not_pollute_state():
    params, n = _params(), 64
    test = tc.MeanShiftTest(params, n)
    for value in [params.mu_b] * (n - 1):
        test.add(value)
    before_xs, before_verdicts = list(test.xs), list(test.verdicts)
    for bad in (-1.0, -math.inf, math.nan):
        _raises_value_error(lambda bad=bad: test.add(bad))
        _raises_value_error(lambda bad=bad: test.judge([params.mu_b] * (n - 1) + [bad]))
        assert test.xs == before_xs and test.verdicts == before_verdicts
    for count in (0, n - 1, n + 1):
        _raises_value_error(lambda count=count: test.judge([params.mu_b] * count))
        assert test.xs == before_xs and test.verdicts == before_verdicts
    _raises_value_error(lambda: test.judge(np.full((n, 2), params.mu_b)))
    assert test.xs == before_xs and test.verdicts == before_verdicts
    verdict = test.add(math.inf)
    assert verdict["n"] == n and verdict["clipped"] == 1
    _close(verdict["D_us"], params.c - params.mu_b)
    assert test.xs == [] and test.verdicts == [verdict]


def _net_args(**changes):
    values = {
        "claim_cap_us": 100.0,
        "claim_window": 64,
        "claim_mode": "window",
        "claim_drift_us": 0.0,
        "reply_timeout": 2.0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_net_fixed_options_allow_automatic_defaults_and_manual_overrides():
    from memory_accounting.verifier import net

    assert net._fixed_claim_options(_net_args())
    assert not net._fixed_claim_options(SimpleNamespace())
    assert net._fixed_claim_options(_net_args(claim_cap_us=None))
    assert net._fixed_claim_options(_net_args(claim_window=None))
    assert net._fixed_claim_options(_net_args(claim_cap_us=None, claim_window=None, claim_F=0.01))
    assert net._fixed_claim_options(_net_args(claim_cap_us=None, claim_window=None, role="baseline"))
    assert not net._fixed_claim_options(
        _net_args(claim_cap_us=None, claim_window=None, claim_mode="period", claim_F=0.01)
    )
    for changes in (
        *({"claim_cap_us": value} for value in (0, -1, math.inf, math.nan, True, "100")),
        *({"claim_window": value} for value in (0, -1, 2.5, math.inf, math.nan, True, "64")),
        {"claim_mode": "period"},
        {"claim_drift_us": 0.1},
    ):
        try:
            net._fixed_claim_options(_net_args(**changes))
        except SystemExit:
            continue
        raise AssertionError(("invalid fixed-cap options accepted", changes))


def test_net_ready_lateness_does_not_change_received_fixed_cap_rtts():
    from memory_accounting.verifier import net

    event = {
        "ready_clipped": True,
        "challenges": [{"rtt_ns": 0}, {"rtt_ns": 125000}],
        "dyn": {"ready_clipped": True, "rtt_ns": 1000, "live": {"rtt_ns": 2000}},
    }
    expected = {"static": [0, 125000], "dynamic": [1000], "live": [2000]}
    for chain, values in expected.items():
        assert net._claim_values(event, chain, penalize_ready=False) == values
        assert net._claim_values(event, chain) == [math.inf] * len(values)


def test_net_missing_attempts_remain_in_each_chain():
    from memory_accounting.verifier import net

    event = {
        "challenges": [{"rtt_ns": None, "missing": "reply"}, {"rtt_ns": None, "missing": "READY"}],
        "dyn": {"fault": net.FAULT_NO_DYN_READY + " (the ARM could not be sent)"},
    }
    assert net._claim_values(event, "static", penalize_ready=False) == [math.inf, math.inf]
    for chain in ("dynamic", "live"):
        assert net._claim_values(event, chain, penalize_ready=False) == [math.inf]
        assert net._claim_values({"dyn": {"fault": net.FAULT_NO_DYN_REPLY}}, chain, penalize_ready=False) == [math.inf]
    event = {"dyn": {"rtt_ns": 123, "fault": net.FAULT_NO_LIVE_REPLY}}
    assert net._claim_values(event, "dynamic", penalize_ready=False) == [123]
    assert net._claim_values(event, "live", penalize_ready=False) == [math.inf]


def test_net_rejects_invalid_received_rtts_before_calibration():
    from memory_accounting.verifier import net

    for bad in (-1, -math.inf, math.nan, True, "12"):
        event = {"challenges": [{"rtt_ns": bad}], "dyn": {"rtt_ns": bad, "live": {"rtt_ns": bad}}}
        for chain in ("static", "dynamic", "live"):
            _raises_value_error(lambda chain=chain: net._claim_values(event, chain, penalize_ready=False))


def test_net_calibration_round_trip_preserves_all_attempts_as_json():
    from memory_accounting.verifier import net

    args = _net_args()
    samples = net._new_claim_samples(args)
    chains = tuple(net.CLAIM_SAMPLES)
    net._append_claim_samples(
        samples,
        {
            "ready_clipped": True,
            "challenges": [{"rtt_ns": 0}, {"rtt_ns": 125000}],
            "dyn": {"ready_clipped": True, "rtt_ns": 1000, "live": {"rtt_ns": 2000}},
        },
        chains,
    )
    net._append_claim_samples(
        samples,
        {
            "challenges": [{"rtt_ns": None, "missing": "reply"}],
            "dyn": {"fault": net.FAULT_NO_DYN_READY},
        },
        chains,
    )
    saved = json.loads(json.dumps({"claim_calibration_samples": samples}, allow_nan=False))
    expected = {"static": [0, 125000, math.inf], "dynamic": [1000, math.inf], "live": [2000, math.inf]}
    for chain, values in expected.items():
        assert samples[net.CLAIM_SAMPLES[chain]][-1] is None
        assert net._fixed_claim_samples(saved, args, chain) == values
    raw_us = [value / 1000 for value in net._fixed_claim_samples(saved, args, "static")]
    base = tc.estimate_baseline(raw_us, L=8, delta_c=10, c=args.claim_cap_us)
    assert base["n"] == 3 and base["clipped"] == 2
    _close(base["mu_b"], 200 / 3)


def test_net_rejects_incompatible_or_incomplete_attempt_calibration():
    from memory_accounting.verifier import net

    args = _net_args()
    samples = net._new_claim_samples(args)
    _raises_value_error(lambda: net._fixed_claim_samples({}, args, "static"))
    _raises_value_error(
        lambda: net._fixed_claim_samples({"calibration_samples": {"static_chain_rtt_ns": [1, 2]}}, args, "static")
    )
    for key, value in (("policy", "finite-only"), ("cap_us", 101), ("window", 65), ("reply_timeout_s", 3)):
        bad = {"claim_calibration_samples": {**samples, key: value}}
        _raises_value_error(lambda bad=bad: net._fixed_claim_samples(bad, args, "static"))
    # A live chain that was not eligible/scheduled contributes no attempt, without invalidating other chains.
    net._append_claim_samples(samples, {"dyn": {"rtt_ns": 25}}, ["dynamic", "live"])
    assert "invalid_outcomes" not in samples
    baseline = {"claim_calibration_samples": samples}
    assert net._fixed_claim_samples(baseline, args, "dynamic") == [25]
    assert net._fixed_claim_samples(baseline, args, "live") == []


if __name__ == "__main__":
    from _runner import run

    run(globals())
