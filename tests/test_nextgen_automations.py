"""
tests/test_nextgen_automations.py
----------------------------------
Unit test suite covering Next-Generation Automations:
- Optuna hyperparameter optimization worker
- Economic calendar pre-news event guard
- 24-hour Risk Parity portfolio rebalancer
- Monthly performance summary generator
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from auto_retrain_optuna import optuna_retrainer
from economic_calendar_guard import economic_calendar_guard
from auto_risk_parity_rebalancer import auto_risk_parity_rebalancer
from monthly_pdf_reporter import monthly_pdf_reporter


def test_optuna_hyperparameter_tuning():
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], size=100)
    params = optuna_retrainer.optimize_hyperparameters(X, y)
    assert "max_depth" in params
    assert "learning_rate" in params
    assert params["learning_rate"] > 0.0


def test_economic_calendar_guard_fomc():
    fomc_time = datetime(2026, 7, 29, 17, 50, tzinfo=timezone.utc)
    is_blackout, reason = economic_calendar_guard.check_news_blackout(fomc_time)
    assert is_blackout is True
    assert "FOMC" in reason


def test_auto_risk_parity_rebalancing():
    vols = {"BTCUSDT": 0.02, "ETHUSDT": 0.04, "SOLUSDT": 0.08}
    weights = auto_risk_parity_rebalancer.compute_daily_rebalance_weights(vols)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["BTCUSDT"] > weights["SOLUSDT"]


def test_monthly_pdf_reporter():
    history = [
        {"pnl_usd": 2.50, "symbol": "BTCUSDT"},
        {"pnl_usd": -1.00, "symbol": "ETHUSDT"},
        {"pnl_usd": 3.00, "symbol": "SOLUSDT"}
    ]
    report = monthly_pdf_reporter.generate_monthly_performance_summary(history)
    assert report["total_trades"] == 3
    assert report["win_rate_pct"] == pytest.approx(66.67, rel=1e-2)
    assert report["total_pnl_usd"] == pytest.approx(4.50)
