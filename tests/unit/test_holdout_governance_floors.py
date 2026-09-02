import pytest
from model_governance import extract_metric, validate_manifest_governance_floors


def test_extract_metric_safe_from_falsy_zero():
    """Verify extract_metric returns 0.0 as valid float, not evaluating as falsy or falling through."""
    data = {"holdout_mcc": 0.0, "metrics": {"holdout_mcc": 0.05}}
    val = extract_metric(data, ["holdout_mcc"])
    assert val == 0.0
    assert val is not None

    nested_data = {"cv_metrics": {"holdout_mcc": 0.0}}
    val_nested = extract_metric(nested_data, ["holdout_mcc"], ["cv_metrics", "holdout_mcc"])
    assert val_nested == 0.0

    missing_data = {"other": 123}
    assert extract_metric(missing_data, ["holdout_mcc"]) is None


def test_holdout_mcc_zero_rejection():
    """Verify that a manifest with holdout_mcc = 0.0 is rejected by governance floor (0.0 < 0.02)."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.0,
        "holdout_balanced_accuracy": 0.38,
        "promoted": True
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is False
    assert "Holdout MCC" in reason


def test_holdout_mcc_missing_rejection():
    """Verify that a manifest missing holdout_mcc is rejected (fail closed)."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_balanced_accuracy": 0.38,
        "promoted": True
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is False
    assert "Holdout MCC" in reason


def test_holdout_bal_acc_subfloor_rejection():
    """Verify that a manifest with holdout_balanced_accuracy < floor is rejected."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.05,
        "holdout_balanced_accuracy": 0.30,  # Below floor 0.334 / 0.34
        "promoted": True
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is False
    assert "Holdout BalAcc" in reason


def test_holdout_ci95_low_rejection():
    """Verify that holdout CI95 lower bound < -0.05 triggers rejection."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.05,
        "holdout_balanced_accuracy": 0.38,
        "cv_metrics": {"holdout_mcc_ci95": [-0.08, 0.12]},
        "promoted": True
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is False
    assert "lower bound" in reason


def test_unpromoted_manifest_rejection():
    """Verify that manifest with promoted=False triggers rejection."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.05,
        "holdout_balanced_accuracy": 0.38,
        "promoted": False
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is False
    assert "promoted=False" in reason


def test_healthy_manifest_passes_governance():
    """Verify that a complete manifest with metrics meeting all floors passes."""
    manifest = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.045,
        "holdout_balanced_accuracy": 0.375,
        "cv_metrics": {"holdout_mcc_ci95": [0.01, 0.08]},
        "promoted": True
    }
    ok, reason = validate_manifest_governance_floors(manifest, "15")
    assert ok is True
    assert reason == ""
