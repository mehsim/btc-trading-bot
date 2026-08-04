import os
import numpy as np
import pandas as pd
import pytest
from core import generate_triple_barrier_labels
from ensemble import write_model_manifest
from drift_detector import CUSUMDriftDetector, PSIDriftDetector, evaluate_drift_and_trigger_playbook

def test_generate_triple_barrier_labels():
    dates = pd.date_range("2026-01-01", periods=50, freq="15min")
    df = pd.DataFrame({
        "timestamp": dates.astype(int) // 10**6,
        "close": np.linspace(100, 150, 50),
        "high": np.linspace(101, 152, 50),
        "low": np.linspace(99, 148, 50),
        "ATR_norm": np.full(50, 0.01)
    })
    labels = generate_triple_barrier_labels(df, interval="15")
    assert len(labels) == 50
    assert set(labels.unique()).issubset({0, 1, 2})

def test_write_model_manifest(tmp_path):
    prefix = str(tmp_path / "test_model")
    write_model_manifest(prefix, feature_names=["RSI", "MACD_diff"], metrics={"win_rate": 0.58})
    manifest_file = f"{prefix}_manifest.json"
    assert os.path.exists(manifest_file)

def test_evaluate_drift_and_trigger_playbook():
    cusum = CUSUMDriftDetector(threshold_H=3.0)
    cusum.reset()
    psi = PSIDriftDetector()
    
    # 1. Stable run
    outcomes = [{"success": 1, "pnl_usd": 10.0, "confidence": 0.80}]
    res = evaluate_drift_and_trigger_playbook(cusum, psi, outcomes)
    assert res["status"] == "STABLE"
    assert res["de_risk"] is False
    
    # 2. Trigger consecutive losses to force CUSUM drift
    loss_outcomes = [{"success": 0, "pnl_usd": -5.0, "confidence": 0.80} for _ in range(10)]
    res_drift = evaluate_drift_and_trigger_playbook(cusum, psi, loss_outcomes)
    assert res_drift["de_risk"] is True
    assert res_drift["pause_entries"] is True
    assert res_drift["trigger_retrain"] is True
