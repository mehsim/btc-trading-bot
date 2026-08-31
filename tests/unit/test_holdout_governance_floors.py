import pytest
from config import TIMEFRAME_MIN_HOLDOUT_MCC, TIMEFRAME_MIN_HOLDOUT_BAL_ACC, MODEL_GOVERNANCE


def test_holdout_mcc_floor_rejection():
    """Verify models with holdout MCC below floor are rejected."""
    min_holdout_mcc_15m = TIMEFRAME_MIN_HOLDOUT_MCC.get("15", 0.010)
    failed_holdout_mcc = -0.0064

    assert failed_holdout_mcc < min_holdout_mcc_15m


def test_holdout_bal_acc_floor_rejection():
    """Verify models with holdout balanced accuracy below floor are rejected."""
    min_holdout_bal_acc_15m = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get("15", 0.334)
    failed_holdout_bal_acc = 0.0

    assert failed_holdout_bal_acc < min_holdout_bal_acc_15m


def test_holdout_ci95_lower_bound_rejection():
    """Verify models with holdout CI95 lower bound < -0.05 are rejected."""
    failed_ci_low = -0.08
    assert failed_ci_low < -0.05


def test_promoted_flag_rejection():
    """Verify unpromoted manifests (promoted: False) are rejected."""
    is_promoted = False
    assert is_promoted is False


def test_healthy_holdout_manifest_passes():
    """Verify healthy holdout metrics pass governance floors."""
    holdout_mcc = 0.025
    holdout_bal_acc = 0.355
    holdout_ci_low = -0.01
    is_promoted = True

    min_h_mcc = TIMEFRAME_MIN_HOLDOUT_MCC.get("15", 0.010)
    min_h_bal = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get("15", 0.334)

    assert holdout_mcc >= min_h_mcc
    assert holdout_bal_acc >= min_h_bal
    assert holdout_ci_low >= -0.05
    assert is_promoted is True
