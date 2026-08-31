import pytest
import numpy as np
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
    
    # Must NOT be artificially inflated to > 0.45 via heuristic linear map
    assert cal_conf < 0.35, f"Calibrated confidence should reflect empirical prior (~0.32), got {cal_conf}"
    assert 0.20 <= cal_conf <= 0.35

def test_calibrator_identity_fallback_on_none():
    assert calibrate_probability(0.60, None) == 0.60
    assert calibrate_probability(0.60, {}) == 0.60
