import pytest
import numpy as np
from tools.beta_calibrator import BetaCalibrator, is_calibrator_viable, calibrate_probability


def test_boundary_hit_marked_as_fit_failure():
    """Verify that when fit produces near-zero slope (a, b <= 0.02), it is marked as fit failure (is_fitted=False)."""
    bc = BetaCalibrator()
    # Construct uninformative flat data (random noise independent of labels)
    np.random.seed(42)
    scores = np.random.uniform(0.50, 0.99, size=1000)
    labels = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])

    bc.fit(scores, labels)
    # If slope is flat, it should not claim to be successfully fitted
    if bc.a <= 0.02 or bc.b <= 0.02:
        assert bc.is_fitted is False
        assert bc.is_viable() is False


def test_clamped_calibrator_economic_viability():
    """Verify that clamped flat calibrator dictionaries fail economic viability checks against break-even p*."""
    # Example flat calibrator dict from production 15m trending where ceiling is ~0.32
    flat_calibrator_15m = {
        "scaling_method": "beta_calibration",
        "a": 0.01,
        "b": 0.01,
        "c": -0.69176,
        "fitting_sample_size": 10000,
        "is_fitted": True
    }

    # Break-even p* for 15m trending is ~0.4712
    p_star_15m = 0.4712

    # Should be flagged as non-viable because a, b <= 0.02 and max ceiling is ~0.34 < 0.4712
    assert is_calibrator_viable(flat_calibrator_15m, min_required_p_star=p_star_15m) is False
    assert is_calibrator_viable(flat_calibrator_15m) is False


def test_healthy_calibrator_economic_viability():
    """Verify that a healthy, steep calibrator passes viability checks."""
    healthy_calibrator = {
        "scaling_method": "beta_calibration",
        "a": 1.5,
        "b": 1.2,
        "c": 0.1,
        "fitting_sample_size": 5000,
        "is_fitted": True
    }

    p_star = 0.45
    bc = BetaCalibrator.from_dict(healthy_calibrator)
    assert bc.is_viable(min_required_p_star=p_star) is True
    assert bc.max_achievable_probability(0.99) > p_star
    assert is_calibrator_viable(healthy_calibrator, min_required_p_star=p_star) is True


def test_legacy_flat_knot_calibrator_viability():
    """Verify that legacy knot calibrators with ceiling below p* are marked unviable."""
    flat_knot_calibrator = {
        "X": [0.50, 0.70, 0.90, 0.99],
        "y": [0.31, 0.31, 0.31, 0.31]
    }
    assert is_calibrator_viable(flat_knot_calibrator, min_required_p_star=0.45) is False
