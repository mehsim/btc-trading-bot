import pytest
import numpy as np
import pandas as pd
import time
import json
import os

from ensemble import resolve_direction, EnsembleClassifier
from config import TIMEFRAME_MIN_HOLDOUT_MCC, TIMEFRAME_MIN_HOLDOUT_BAL_ACC
from exit_policy_engine import ExitPolicyEngine
from confluence_engine import check_pre_trade_confluence


def test_finding_58_resolve_direction_requires_directional_dominance():
    """Finding #58: resolve_direction must not resolve directional when Neutral dominates."""
    # 70% Neutral, 16% Bullish, 14% Bearish
    probs = np.array([0.14, 0.70, 0.16])
    label, conf = resolve_direction(probs, min_dir_mass=0.15)
    assert label == "Neutral", f"Expected 'Neutral' but got {label}"
    assert conf == 0.70

    # 40% Bullish, 35% Bearish, 25% Neutral -> dir_total = 0.75 >= 0.25 (Neutral), Bullish wins
    probs2 = np.array([0.35, 0.25, 0.40])
    label2, conf2 = resolve_direction(probs2, min_dir_mass=0.15)
    assert label2 == "Bullish"
    assert conf2 == pytest.approx(0.40 / 0.75, abs=1e-3)


def test_finding_57_holdout_floors_above_statistical_noise():
    """Finding #57: 15m and intraday holdout floors must be >= 0.025 MCC and >= 0.345 balanced accuracy."""
    assert TIMEFRAME_MIN_HOLDOUT_MCC["15"] >= 0.025
    assert TIMEFRAME_MIN_HOLDOUT_BAL_ACC["15"] >= 0.345


def test_finding_59_and_60_optuna_search_space_branching_and_trials():
    """Findings #59 & #60: Multi-hour trending regimes reach deep hyperparameter space and run proper trial count."""
    import train
    from sklearn.datasets import make_classification, make_regression

    X_tr, y_tr = make_classification(n_samples=60, n_features=10, n_informative=5, n_classes=3, random_state=42)
    X_val, y_val = make_classification(n_samples=20, n_features=10, n_informative=5, n_classes=3, random_state=43)
    X_tr_df = pd.DataFrame(X_tr)
    X_val_df = pd.DataFrame(X_val)
    y_tr_s = pd.Series(y_tr)
    y_val_s = pd.Series(y_val)

    init_trials = train._EXECUTED_OPTUNA_TRIALS
    # Run 60m trending
    best_params = train.optimize_xgb_classifier(X_tr_df, y_tr_s, X_val_df, y_val_s, sample_weights=np.ones(len(y_tr_s)), regime="trending", interval=60)
    assert train._EXECUTED_OPTUNA_TRIALS == init_trials + 8
    # max_depth for trending 60m should be within search space [5, 8]
    assert 5 <= best_params["max_depth"] <= 8


def test_finding_62_synthetic_bar_forward_fill_cap():
    """Finding #62: Synthetic bar forward-fill exceeding 3-bar gap sets gap_exceeded=True."""
    from data import get_history
    from unittest import mock

    now_ms = int(time.time() * 1000)
    # 20 bars ending near now_ms, with an 8-bar gap in between
    timestamps = [now_ms - (30 - i) * 900000 for i in range(10)]
    timestamps += [now_ms - (12 - i) * 900000 for i in range(10)]
    raw_rows = [[ts, 50000.0, 50100.0, 49900.0, 50050.0, 10.0, 500500.0] for ts in timestamps]

    cursor_mock = mock.MagicMock()
    cursor_mock.fetchall.return_value = raw_rows
    conn_mock = mock.MagicMock()
    conn_mock.cursor.return_value = cursor_mock

    with mock.patch("data.safe_get_sqlite_conn", return_value=conn_mock):
        res = get_history(symbol="BTCUSDT", interval="15", limit=20)
        assert res.attrs.get("gap_exceeded") is True
        assert res.attrs.get("is_discontinuous") is True


def test_finding_67_inline_atr_computation_on_raw_ohlcv():
    """Finding #67: Stress covariance in trade_calculators computes inline ATR_norm on raw get_history OHLCV."""
    from trade_calculators import calculate_covariance_multiplier
    from unittest import mock

    dates = pd.date_range("2026-01-01", periods=30, freq="1h")
    raw_df = pd.DataFrame({
        "timestamp": [int(d.timestamp() * 1000) for d in dates],
        "open": np.linspace(50000, 51000, 30),
        "high": np.linspace(50000, 51000, 30) + 100.0,
        "low": np.linspace(50000, 51000, 30) - 100.0,
        "close": np.linspace(50000, 51000, 30),
        "volume": [100.0] * 30
    })

    with mock.patch("data.get_history", return_value=raw_df):
        mult, avg_corr = calculate_covariance_multiplier(
            new_symbol="ETHUSDT",
            new_direction="Bullish",
            bot_state={"active_trade_15m": [{"symbol": "BTCUSDT", "direction": "Bullish", "position_size_usd": 100.0}]}
        )
        assert isinstance(mult, float)
        assert mult > 0.0


def test_finding_69_confluence_htf_trend_gate_intraday():
    """Finding #69: In trending regime on 15m, if both 1h and 4h trend oppose, trend_gates_passed must fail."""
    # Build dummy results where 1h and 4h both fail
    # When ml_trend is Bullish, but 1h and 4h are Bearish
    dates_4h = pd.date_range("2026-01-01", periods=30, freq="4h")
    # Downtrending close prices: EMA9 < EMA21 -> Bearish
    df_4h = pd.DataFrame({
        "close": np.linspace(60000, 40000, 30),
        "high": np.linspace(60100, 40100, 30),
        "low": np.linspace(59900, 39900, 30)
    })
    dates_1h = pd.date_range("2026-01-01", periods=30, freq="1h")
    df_1h = pd.DataFrame({
        "close": np.linspace(60000, 40000, 30),
        "high": np.linspace(60100, 40100, 30),
        "low": np.linspace(59900, 39900, 30)
    })

    # Call check_pre_trade_confluence with ml_trend = "Bullish"
    approved, details, score_pct = check_pre_trade_confluence(
        current_price=50000.0,
        df_1h=df_1h,
        ml_trend="Bullish",
        news_sentiment=0.0,
        expected_pct_change=0.02,
        interval="15",
        symbol="BTCUSDT",
        htf_cache={("BTCUSDT", "240"): df_4h},
        calibrated_confidence=0.85,
        dynamic_conf_threshold=0.50,
        current_regime="Trending (GMM)"
    )
    # Both 1h and 4h oppose Bullish -> trend_gates_passed must be False -> approved must be False
    assert approved is False
    assert "Trend Pass: False" in details.get("_Score_Summary", {}).get("detail", "")


def test_finding_70_exit_policy_regime_key_resolution():
    """Finding #70: ExitPolicyEngine._resolve_regime_key resolves compound live regime strings properly."""
    engine = ExitPolicyEngine()
    # Live GMM string with high ADX -> STRONG_TREND
    key1 = engine._resolve_regime_key("Trending (GMM)", adx_val=32.0)
    assert key1 == "STRONG_TREND"

    # Live GMM string with moderate ADX -> MODERATE_TREND
    key2 = engine._resolve_regime_key("Trending (GMM)", adx_val=22.0)
    assert key2 == "MODERATE_TREND"

    # Ranging string -> RANGING
    key3 = engine._resolve_regime_key("Ranging (GMM)", adx_val=15.0)
    assert key3 == "RANGING"
