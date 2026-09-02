import pytest
import numpy as np
from tools.beta_calibrator import BetaCalibrator, is_calibrator_viable, calibrate_probability


def test_beta_calibrator_insufficient_samples_sets_is_fitted_false():
    """Verify that BetaCalibrator.fit() sets is_fitted=False when samples are insufficient (< 20)."""
    bc = BetaCalibrator()
    scores = [0.6] * 10
    labels = [1] * 10
    bc.fit(scores, labels)

    assert bc.is_fitted is False
    assert bc.a == 1.0
    assert bc.b == 1.0
    assert bc.c == 0.0

    d = bc.to_dict()
    assert d["is_fitted"] is False
    assert d["is_fallback"] is True
    assert is_calibrator_viable(d) is False


def test_beta_calibrator_single_class_sets_is_fitted_false():
    """Verify that BetaCalibrator.fit() sets is_fitted=False when only one class is present."""
    bc = BetaCalibrator()
    scores = np.linspace(0.4, 0.8, 30)
    labels = [1] * 30  # All class 1
    bc.fit(scores, labels)

    assert bc.is_fitted is False
    d = bc.to_dict()
    assert d["is_fitted"] is False
    assert is_calibrator_viable(d) is False


def test_beta_calibrator_from_dict_identity_fallback_detection():
    """Verify from_dict correctly detects uncalibrated identity fallback dicts."""
    legacy_unfitted_dict = {
        "scaling_method": "beta_calibration",
        "a": 1.0,
        "b": 1.0,
        "c": 0.0,
        "fitting_sample_size": 10
    }
    bc = BetaCalibrator.from_dict(legacy_unfitted_dict)
    assert bc.is_fitted is False
    assert bc.is_viable() is False
    assert is_calibrator_viable(legacy_unfitted_dict) is False


def test_beta_calibrator_viable_genuine_fit():
    """Verify genuinely fitted BetaCalibrator passes viability checks."""
    bc = BetaCalibrator()
    np.random.seed(42)
    scores = np.random.uniform(0.3, 0.9, 200)
    labels = (scores > 0.55).astype(int)
    bc.fit(scores, labels)

    assert bc.is_fitted is True
    assert bc.a > 0.02
    assert bc.b > 0.02
    assert bc.is_viable(min_required_p_star=0.40) is True

    d = bc.to_dict()
    assert d["is_fitted"] is True
    assert d["is_fallback"] is False
    assert is_calibrator_viable(d, min_required_p_star=0.40) is True
