import os
import json
from tools.beta_calibrator import BetaCalibrator, calibrate_probability


def test_flat_beta_calibrator_preserves_empirical_prior():
    """Verify calibrate_probability returns true empirical prior on flat calibrator without artificial inflation."""
    flat_data = {
        "scaling_method": "beta_calibration",
        "a": 0.01,
        "b": 0.01,
        "c": -0.76,  # sigmoid(-0.76) ≈ 0.3186
        "fitting_sample_size": 10000,
        "is_fitted": True
    }

    # Raw score 0.65 (high raw confidence)
    cal_conf = calibrate_probability(0.65, flat_data)

    # Must NOT be artificially inflated to > 0.45 via heuristic linear map (e.g. 0.35 + (0.65-0.5)*0.8 = 0.47)
    assert cal_conf < 0.35, f"Calibrated confidence should reflect empirical prior (~0.32), got {cal_conf}"
    assert 0.20 <= cal_conf <= 0.35


def test_calibrator_identity_fallback_on_none():
    assert calibrate_probability(0.60, None) == 0.60
    assert calibrate_probability(0.60, {}) == 0.60


def test_live_calibrator_slots_no_linear_escape_hatch():
    """
    Verify that all live champion calibrator files produce pure fitted Beta calibration output
    and do not substitute hardcoded linear mapping expressions.
    """
    cal_files = [
        "calibrator_trending_15.json",
        "calibrator_ranging_15.json",
        "calibrator_trending_30.json",
        "calibrator_ranging_60.json",
        "calibrator_ranging_240.json",
        "calibrator_trending_240.json"
    ]

    for fname in cal_files:
        if not os.path.exists(fname):
            continue

        with open(fname, "r") as f:
            cal_data = json.load(f)

        bc = BetaCalibrator.from_dict(cal_data)
        raw_score = 0.85
        expected_fitted = bc.predict_proba(raw_score)
        actual = calibrate_probability(raw_score, cal_data)

        assert abs(actual - expected_fitted) < 1e-6, f"{fname} mismatch: actual {actual} != expected {expected_fitted}"
