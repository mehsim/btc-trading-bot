"""
Unit tests for Institutional Core Engine v7 components:
- Regime-Specific Calibration
- Return Distribution Forecast Engine
- Trade Lifecycle Health Scoring Engine
- Execution Shortfall Analytics Engine
- PSI Multi-Drift Retrainer
"""

import pytest
import numpy as np
import pandas as pd
from online_calibration_engine import regime_specific_calibrator
from return_distribution_engine import return_distribution_engine
from trade_lifecycle_health import trade_lifecycle_health_engine
from execution_shortfall_analytics import execution_shortfall_analytics
from psi_drift_retrainer import psi_multi_drift_retrainer

def test_regime_specific_calibration():
    # Trending regime calibration
    calib_trending = regime_specific_calibrator.calibrate_probability(0.85, regime="Trending (GMM)")
    # Crisis regime calibration (higher temperature scaling)
    calib_crisis = regime_specific_calibrator.calibrate_probability(0.85, regime="Crisis Volatility")
    
    assert calib_trending > 0.50
    assert calib_crisis < calib_trending  # Crisis applies stronger temperature scaling


def test_return_distribution_engine():
    res = return_distribution_engine.predict_return_distribution(
        predicted_change_pct=0.015, atr_norm=0.01, calibrated_confidence=0.75
    )
    assert "expected_return_mu" in res
    assert "return_variance_sigma2" in res
    assert "cvar_95_tail_risk" in res
    assert "predicted_expected_utility" in res
    assert res["expected_return_mu"] > 0


def test_trade_lifecycle_health_scoring():
    # Healthy trade
    health_h, exit_h, reason_h = trade_lifecycle_health_engine.evaluate_trade_health(
        entry_price=60000.0, current_price=60300.0, direction="BUY",
        initial_adx=30.0, current_adx=28.0
    )
    assert health_h >= 80.0
    assert exit_h is False

    # Degraded trade (adverse price move + decaying ADX + regime shift)
    health_d, exit_d, reason_d = trade_lifecycle_health_engine.evaluate_trade_health(
        entry_price=60000.0, current_price=59300.0, direction="BUY",
        initial_adx=30.0, current_adx=15.0, regime_transition_prob=0.35, orderbook_imbalance=0.20
    )
    assert health_d < 45.0
    assert exit_d is True


def test_execution_shortfall_analytics():
    now = 1700000000.0
    res = execution_shortfall_analytics.record_execution_telemetry(
        symbol="BTCUSDT", direction="BUY", decision_timestamp=now,
        order_sent_timestamp=now + 0.05, fill_timestamp=now + 0.12,
        arrival_price=60000.0, fill_price=60010.0, requested_qty=1.0, filled_qty=1.0
    )
    assert res["latency_ms"] == 120.0
    assert res["implementation_shortfall_bps"] > 0
    assert res["execution_quality"] == "EXCELLENT"


def test_psi_multi_drift_retrainer():
    base = np.random.normal(0, 1, 100)
    curr = np.random.normal(0.8, 1.2, 100)
    psi = psi_multi_drift_retrainer.calculate_psi(base, curr)
    assert psi > 0

    mhi, retrain, reason = psi_multi_drift_retrainer.evaluate_model_health_index(
        psi_score=psi, ece_score=0.08, recent_win_rate=0.40, baseline_win_rate=0.55
    )
    assert mhi < 80.0
