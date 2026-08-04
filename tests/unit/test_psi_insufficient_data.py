import pytest
import numpy as np
import pandas as pd
from drift_detector import calculate_psi, PSIDriftDetector
from ensemble import _slice_model_input


def test_calculate_psi_insufficient_data_returns_none():
    """Verify calculate_psi returns None (not 0.0) when sample size < 20."""
    baseline = np.random.normal(0, 1, 15)  # 15 samples < 20
    target = np.random.normal(0, 1, 15)

    res = calculate_psi(baseline, target)
    assert res is None, f"Expected None for insufficient data (< 20 samples), got {res}"


def test_psi_drift_detector_insufficient_data_status():
    """Verify PSIDriftDetector returns INSUFFICIENT_DATA status on sparse inputs."""
    detector = PSIDriftDetector()
    baseline = np.random.normal(0, 1, 10)
    live = np.random.normal(0, 1, 10)

    is_drift, score, status = detector.check_feature_drift(baseline, live)
    assert is_drift is False
    assert score is None
    assert status == "INSUFFICIENT_DATA"


def test_slice_model_input_excess_positional_features_raises_runtime_error():
    """Verify H-04: Positional input with excess features raises RuntimeError instead of silently truncating."""
    class DummyModel:
        pass

    model = DummyModel()
    # Model expecting 10 positional features ('0'...'9')
    model.feature_names_in_ = [str(i) for i in range(10)]

    # Live DataFrame providing 15 features
    df = pd.DataFrame(np.random.randn(5, 15))

    with pytest.raises(RuntimeError) as exc_info:
        _slice_model_input(model, df)

    assert "Feature Shape Mismatch Error" in str(exc_info.value)
    assert "15" in str(exc_info.value)
    assert "10" in str(exc_info.value)
