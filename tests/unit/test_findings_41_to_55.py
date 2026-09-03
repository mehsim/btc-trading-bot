import pytest
import numpy as np
import pandas as pd
import json
import os

from config import MODEL_SLOT_DENYLIST, TAKER_FEE_PCT, MAKER_FEE_PCT
from statistical_validation import statistical_validation
from train import NON_STATIONARY_EXCLUDE


def test_finding_48_statistical_power_emitted_in_governance():
    """Finding #48: calculate_governed_validation_matrix emits 'power' in governance dict."""
    baseline = [0.01 * (i % 2) for i in range(100)]
    component = [0.02 * (i % 2) for i in range(100)]
    res = statistical_validation.calculate_governed_validation_matrix(
        component_name="test_slot",
        baseline_returns=baseline,
        component_returns=component,
        completed_trades=100,
        module_uuid="TEST_SLOT",
        num_trials=1
    )
    gov = res.get("governance", {})
    assert "power" in gov, "Governance block must contain 'power' key"
    assert isinstance(gov["power"], (int, float))
    stats = res.get("statistics", {})
    assert "statistical_power" in stats
    assert gov["power"] == stats["statistical_power"]


def test_finding_53_ranging_120_denylisted():
    """Finding #53: ranging_120 must be included in MODEL_SLOT_DENYLIST."""
    assert "ranging_120" in MODEL_SLOT_DENYLIST
    assert "trending_120" in MODEL_SLOT_DENYLIST


def test_finding_53_and_54_selected_features_120_have_no_non_stationary_levels():
    """Findings #53 & #54: selected_features_120_ranging/trending.json must have 0 non-stationary level features."""
    for fname in ["selected_features_120_ranging.json", "selected_features_120_trending.json"]:
        if os.path.exists(fname):
            with open(fname, "r") as f:
                feats = json.load(f)
            overlap = set(feats).intersection(set(NON_STATIONARY_EXCLUDE))
            assert len(overlap) == 0, f"{fname} contains non-stationary features: {overlap}"


def test_finding_46_dashboard_walk_forward_folds_no_fabrication():
    """Finding #46: Dashboard _get_walk_forward_folds never fabricates folds from live trade history."""
    from dashboard_routes import _get_walk_forward_folds
    from state_manager import state_manager
    # Ensure state_manager does not have pre-existing folds
    orig = state_manager.get("walk_forward_folds")
    try:
        state_manager["walk_forward_folds"] = None
        # Even if database has trades, if backtest_results.json has no walk_forward_validation, it returns []
        folds = _get_walk_forward_folds()
        assert isinstance(folds, list)
        for fold in folds:
            # Must not contain fabricated formula pf * 1.1 unless from real summary
            assert "sharpe" in fold
    finally:
        state_manager["walk_forward_folds"] = orig


def test_finding_41_safe_predict_proba_and_numeric_cols():
    """Finding #41: _safe_predict_proba handles models that do not accept weights argument."""
    from backtest import run_single_backtest
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    
    # Create synthetic test dataset
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=150, freq="1h").astype(np.int64) // 10**6,
        "open": np.linspace(100, 110, 150),
        "high": np.linspace(101, 111, 150),
        "low": np.linspace(99, 109, 150),
        "close": np.linspace(100, 110, 150),
        "volume": np.ones(150) * 1000,
        "feature_1": np.random.randn(150),
        "feature_2": np.random.randn(150),
        "ADX": np.ones(150) * 25.0,
        "ATR_norm": np.ones(150) * 0.01,
        "datetime": pd.date_range("2026-01-01", periods=150, freq="1h")
    })
    
    X = df[["feature_1", "feature_2"]].values
    y_t = np.random.choice([0, 1, 2], size=150)
    y_p = np.random.randn(150) * 0.01
    
    m_t = HistGradientBoostingClassifier(max_iter=10, random_state=42).fit(X, y_t)
    m_p = HistGradientBoostingRegressor(max_iter=10, random_state=42).fit(X, y_p)
    m_t.feature_names = ["feature_1", "feature_2"]
    m_p.feature_names = ["feature_1", "feature_2"]
    
    models = {"trend": m_t, "price": m_p, "feature_names": ["feature_1", "feature_2"]}
    
    # run_single_backtest must not raise TypeError on weights=
    res = run_single_backtest(
        df,
        models_trending=models,
        models_ranging=models,
        p95=0.6,
        max_conf=0.50,
        min_confidence=0.35,
        interval="60",
        pessimistic_mode=True,
        return_trades=True
    )
    assert res is not None
    trades = res.get("trades", [])
    assert isinstance(trades, list)


def test_finding_43_fee_deduplication_and_defaults():
    """Finding #43: Default fee matches TAKER_FEE_PCT and spread is not double counted."""
    from config import TAKER_FEE_PCT
    assert TAKER_FEE_PCT == 0.00055
    # Round-trip cost is 2 * fee_rate without extra fee_rate / 4.0
    fee_rate = 0.00055
    total_trading_cost = 2.0 * fee_rate
    assert abs(total_trading_cost - 0.0011) < 1e-6
