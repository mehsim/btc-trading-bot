import pytest
import numpy as np
from sklearn.isotonic import IsotonicRegression
from mlops_engine import calculate_brier_score, calculate_expected_calibration_error
from risk_limits import assert_risk_governance_invariants

def test_brier_score_and_ece_calculation():
    """
    F-10 Unit Test:
    Verifies Brier Score and Expected Calibration Error (ECE) calculations.
    """
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    # Perfect probabilities matching ground truth
    y_perfect = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    
    bs_perfect = calculate_brier_score(y_true, y_perfect)
    ece_perfect = calculate_expected_calibration_error(y_true, y_perfect)
    
    assert bs_perfect == 0.0, f"Perfect probabilities must yield Brier Score of 0.0, got {bs_perfect}"
    assert ece_perfect == 0.0, f"Perfect probabilities must yield ECE of 0.0, got {ece_perfect}"

    # Overconfident uncalibrated probabilities
    y_overconfident = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
    bs_over = calculate_brier_score(y_true, y_overconfident)
    ece_over = calculate_expected_calibration_error(y_true, y_overconfident)

    assert bs_over > 0.15, f"Overconfident probabilities should yield higher Brier Score, got {bs_over:.3f}"
    assert ece_over > 0.30, f"Overconfident probabilities should yield high ECE, got {ece_over:.3f}"

def test_isotonic_calibration_reduces_ece():
    """
    F-10 Unit Test:
    Asserts that Isotonic Regression calibration reduces Expected Calibration Error on uncalibrated model outputs.
    """
    np.random.seed(42)
    n = 200
    # Synthetic uncalibrated model outputs (overconfident predictions)
    raw_probs = np.random.uniform(0.60, 0.95, size=n)
    # Realized outcomes (actual win rate is only ~50%)
    labels = (np.random.rand(n) < 0.52).astype(int)

    ece_before = calculate_expected_calibration_error(labels, raw_probs)

    # Apply Isotonic Regression calibration
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrated_probs = ir.fit_transform(raw_probs, labels)

    ece_after = calculate_expected_calibration_error(labels, calibrated_probs)

    assert ece_after < ece_before, f"Isotonic calibration must reduce ECE (before: {ece_before:.3f}, after: {ece_after:.3f})"
    assert ece_after <= 0.15, f"Calibrated probabilities should achieve low ECE <= 0.15, got {ece_after:.3f}"

def test_risk_limits_governance_assertion():
    """
    F-09 Governance Test:
    Asserts that startup governance invariants pass on valid config and fail if breached.
    """
    # 1. Normal assertion pass
    assert assert_risk_governance_invariants() is True

    # 2. Breach leverage cap -> Must raise PermissionError
    class BreachedConfig:
        TIMEFRAME_MAX_LEVERAGE_CAPS = {"60": 50.0} # Hard cap is 5.0x
        MAX_WALLET_MARGIN_UTILIZATION_PCT = 0.90
        MAX_SYMBOL_EXPOSURE_PCT = 0.20

    with pytest.raises(PermissionError) as exc_info:
        assert_risk_governance_invariants(BreachedConfig)

    assert "Governance Violation" in str(exc_info.value)
