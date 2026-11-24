"""CPU checks that convenience setup stays independent of scored calibration."""

import json
import math
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from memory_accounting.verifier.claim_setup import automatic_claim_setup
from memory_accounting.verifier.timing_claim import estimate_baseline, params_from_baseline, plan


def _args(**overrides):
    return SimpleNamespace(
        **(
            {
                "claim_cap_us": None,
                "claim_window": None,
                "claim_F": 0.2,
                "claim_C": 0.99,
                "claim_alpha": None,
                "claim_sigma_margin": 0.1,
                "claim_M_factor": 1.0,
            }
            | overrides
        )
    )


def test_automatic_defaults_are_checked_without_resizing_from_baseline():
    args = _args()
    setup = automatic_claim_setup([99000, 100000, 101000] * 10, args, 8, 10, 400)
    assert setup["cap_us"] == 180 and setup["window"] > 0 and "reason" not in setup
    frozen = dict(setup)
    for raw, supported in (([99, 100, 101] * 134, True), ([0, 180] * 200, False)):
        base = estimate_baseline(raw, 8, 10, c=setup["cap_us"])
        p = params_from_baseline(base, args.claim_F, args.claim_C, 8, 10, sigma_margin=args.claim_sigma_margin)
        assert plan(p, setup["window"])["supported"] == supported
        assert setup == frozen


def test_manual_settings_and_partial_overrides_are_respected():
    assert automatic_claim_setup([], _args(claim_cap_us=100, claim_window=64), 8, 10, 400)["window"] == 64
    manual_n = automatic_claim_setup([100000], _args(claim_window=123), 8, 10, 400)
    assert manual_n["cap_us"] == 180 and manual_n["window"] == 123
    manual_c = automatic_claim_setup([99000, 100000, 101000], _args(claim_cap_us=200), 8, 10, 400)
    assert manual_c["cap_us"] == 200 and manual_c["window"] > 0


def test_missing_pilot_attempts_affect_planning_but_not_cap_estimation():
    args = _args(claim_F=0.5)
    received = [100000] * 99
    good = automatic_claim_setup(received + [100000], args, 8, 10, 400)
    missing = automatic_claim_setup(received + [None], args, 8, 10, 400)
    assert missing["cap_us"] == good["cap_us"] == 180
    assert missing["window"] > good["window"]
    assert missing["pilot_count"] == 100


def test_no_pilot_or_impossible_target_does_not_fall_back_to_another_test():
    for pilot, args, b in (
        ([], _args(), 400),
        ([None, None], _args(), 400),
        ([100000], _args(), 400),
        ([100000, 100000], _args(), 1),
        ([100000, 100000], _args(claim_F=1e-6), 400),
        ([100000, 100000], _args(claim_F=None), 400),
    ):
        out = automatic_claim_setup(pilot, args, 8, 10, b)
        assert out["window"] is None and out["reason"]
    for bad in (-1, math.nan, -math.inf):
        try:
            automatic_claim_setup([100000, bad], _args(), 8, 10, 400)
        except ValueError:
            continue
        raise AssertionError("invalid pilot accepted")


def test_per_chain_setup_survives_baseline_collection_and_cache_reuse():
    from memory_accounting.verifier import net

    args = _args()
    pilot, saved = net._new_claim_samples(args), net._new_claim_samples(args)
    event = {"challenges": [{"rtt_ns": 100000}], "dyn": {"rtt_ns": 200000, "live": {"rtt_ns": 300000}}}
    for _ in range(5):
        net._append_claim_samples(pilot, event, net.CLAIM_SAMPLES)
    for chain in net.CLAIM_SAMPLES:
        net._freeze_claim_setup(saved, pilot, args, chain, 8, 10, 400)
    frozen = json.dumps(saved["setups"], sort_keys=True)
    for _ in range(400):
        net._append_claim_samples(saved, event, net.CLAIM_SAMPLES)
    assert json.dumps(saved["setups"], sort_keys=True) == frozen
    baseline = json.loads(json.dumps({"claim_calibration_samples": saved}, allow_nan=False))
    for chain, expected_cap in zip(net.CLAIM_SAMPLES, (180, 280, 380)):
        setup = net._fixed_claim_setup(baseline, args, chain)
        assert setup["cap_us"] == expected_cap and setup["pilot_count"] == 5
        assert len(net._fixed_claim_samples(baseline, args, chain)) == 400
    try:
        net._fixed_claim_setup(baseline, _args(claim_cap_us=999), "static")
    except ValueError:
        pass
    else:
        raise AssertionError("explicit override disagreed with saved cap but was accepted")


if __name__ == "__main__":
    from _runner import run

    run(globals())
