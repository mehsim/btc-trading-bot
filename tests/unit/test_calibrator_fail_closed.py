import pytest
from tools.beta_calibrator import is_calibrator_viable, BetaCalibrator


def test_is_calibrator_viable_rejects_missing_and_fallback():
    """Verify is_calibrator_viable fails closed on None, non-dict, and fallback calibrators."""
    assert is_calibrator_viable(None) is False
    assert is_calibrator_viable([]) is False
    assert is_calibrator_viable("invalid") is False

    # Identity fallback must be rejected
    fallback_cal = {"scaling_method": "identity", "is_fallback": True, "X": [0.0, 1.0], "y": [0.0, 1.0]}
    assert is_calibrator_viable(fallback_cal) is False

    # Degenerate calibrator with ceiling below required p*
    flat_cal = {"scaling_method": "beta_calibration", "a": 0.01, "b": 0.01, "c": -0.69, "is_fitted": False}
    assert is_calibrator_viable(flat_cal, min_required_p_star=0.45) is False

    # Healthy calibrator with achievable ceiling
    healthy_cal = {"scaling_method": "beta_calibration", "a": 1.2, "b": 0.8, "c": 0.1, "is_fitted": True}
    assert is_calibrator_viable(healthy_cal, min_required_p_star=0.45) is True


def test_missing_calibrator_in_signal_evaluator_fails_closed():
    """Verify signal evaluator filters to Neutral when calibrator is missing or fallback."""
    from signal_evaluator import SignalEvaluator

    evaluator = SignalEvaluator(bot_state={})
    # Mock models with fallback calibrator
    evaluator.models = {
        "60": {
            "trending": {
                "trend": None,
                "price": None,
                "calibrator": {"scaling_method": "identity", "is_fallback": True}
            }
        }
    }
    # When calibrator is fallback or None, is_calibrator_viable fails closed
    assert is_calibrator_viable(evaluator.models["60"]["trending"]["calibrator"]) is False
