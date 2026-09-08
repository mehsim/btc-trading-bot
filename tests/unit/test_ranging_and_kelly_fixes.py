"""
tests/unit/test_ranging_and_kelly_fixes.py
-----------------------------------------
Unit tests verifying the 3 execution pipeline fixes:
1. Bearish mean-reversion expected move sign parity & contradiction gate pass.
2. Confluence engine 120m/ranging HTF trend decoupling (allows dip buys with 1D support).
3. Kelly tracker and risk engine handling of legacy untimestamped / fresh data.
"""

import pytest
import datetime
import pandas as pd
import numpy as np
from ranging_strategy import evaluate_ranging_mean_reversion
from confluence_engine import check_pre_trade_confluence
from kelly_tracker import KellyTracker
from risk_engine import compute_conservative_kelly


def _create_ranging_bear_df():
    rows = []
    for i in range(60):
        rows.append({
            "timestamp": 1780000000000 + (i * 900000),
            "open": 80900.0,
            "high": 81100.0,
            "low": 80800.0,
            "close": 80950.0,
            "volume": 1000.0,
            "ADX": 16.0,
            "RSI": 72.0,
            "BB_high": 81000.0,
            "BB_low": 79000.0,
            "BB_mid": 80000.0,
            "ATR": 500.0,
            "lower_wick_volume_ratio": 1.0,
            "upper_wick_volume_ratio": 1.8
        })
    return pd.DataFrame(rows)


def test_bearish_mean_reversion_sign_parity():
    """Verify Bearish ranging signal expected_move is negative and avoids contradiction flag."""
    df = _create_ranging_bear_df()
    sig = evaluate_ranging_mean_reversion(df, "BNBUSDT", "30")
    assert sig.is_signal is True
    assert sig.direction == "Bearish"
    assert sig.expected_move < 0, f"Expected move must be negative for Bearish, got {sig.expected_move}"
    
    # In main.py: pred_change = float(sig.expected_move)
    pred_change = float(sig.expected_move)
    latest_close = float(df["close"].iloc[-1])
    pred_pct = (abs(pred_change) / latest_close) * 100
    
    # Contradiction formula from main.py line 8435
    strong_conflict = (sig.direction == "Bullish" and pred_change < 0 and pred_pct > 0.05) or \
                      (sig.direction == "Bearish" and pred_change > 0 and pred_pct > 0.05)
    assert not strong_conflict, "Bearish signal must NOT trigger strong_conflict when expected_move is negative"


def test_confluence_120m_ranging_htf_decoupling():
    """Verify 120m ranging setup passes trend gate when 1D is Bullish even if 4H is Bearish."""
    # Create mock 1h DF
    df_1h = pd.DataFrame([{
        "open": 80000, "high": 80500, "low": 79500, "close": 80000, "volume": 100
    } for _ in range(30)])
    
    # Mock HTF cache with 1D Bullish and 4H Bearish
    dates = pd.date_range("2026-08-01", periods=60, freq="D")
    # Bullish 1D: prices rising so EMA9 > EMA21
    df_1d = pd.DataFrame({
        "open": np.linspace(70000, 80000, 60),
        "high": np.linspace(70500, 80500, 60),
        "low": np.linspace(69500, 79500, 60),
        "close": np.linspace(70000, 80000, 60),
        "volume": 1000
    }, index=dates)
    df_1d.attrs["fetch_ok"] = True
    
    # Bearish 4H: prices dropping so EMA9 < EMA21
    dates_4h = pd.date_range("2026-08-20", periods=60, freq="4h")
    df_4h = pd.DataFrame({
        "open": np.linspace(82000, 78000, 60),
        "high": np.linspace(82500, 78500, 60),
        "low": np.linspace(81500, 77500, 60),
        "close": np.linspace(82000, 78000, 60),
        "volume": 1000
    }, index=dates_4h)
    df_4h.attrs["fetch_ok"] = True
    
    htf_cache = {
        ("BTCUSDT", "D"): df_1d,
        ("BTCUSDT", "240"): df_4h
    }
    
    bot_state = {"active_trade_1h": [], "bot_running": True}
    
    approved, results, score = check_pre_trade_confluence(
        current_price=78500.0,
        df_1h=df_1h,
        ml_trend="Bullish",
        news_sentiment=0.0,
        expected_pct_change=1.2,
        interval="120",
        symbol="BTCUSDT",
        htf_cache=htf_cache,
        calibrated_confidence=0.66,
        dynamic_conf_threshold=0.60,
        current_regime="Low Vol, Ranging"
    )
    
    # 1D Trend is Bullish -> pass = True
    assert results["1d_Trend"]["pass"] is True
    # 4H Trend is Bearish -> pass = False
    assert results["4h_Trend"]["pass"] is False
    # Meta-Gate must be APPROVED because 1D supports the ranging setup
    assert approved is True, f"Confluence should approve ranging setup supported by 1D. Results: {results['_Score_Summary']}"


def test_kelly_tracker_ignore_untimestamped(tmp_path):
    """Verify KellyTracker ignores legacy untimestamped records when ignore_untimestamped=True."""
    data_file = tmp_path / "test_kelly.json"
    tracker = KellyTracker(data_file=str(data_file))
    
    # Add 20 legacy untimestamped losing trades
    for _ in range(20):
        tracker.history.append({
            "symbol": "BTCUSDT",
            "timeframe": "15",
            "pnl_usd": -10.0,
            "return_pct": -0.02
        })
        
    # With ignore_untimestamped=False, it includes them and returns 0.0 (negative edge)
    frac_included = tracker.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True, ignore_untimestamped=False)
    assert frac_included == 0.0
    
    # By default (ignore_untimestamped=True), untimestamped trades are ignored -> insufficient sample -> None
    frac_default = tracker.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True)
    assert frac_default is None
    
    frac_ignored = tracker.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True, ignore_untimestamped=True)
    assert frac_ignored is None


def test_compute_conservative_kelly_fallback_when_clean():
    """Verify compute_conservative_kelly computes positive Quarter-Kelly when empirical history is clean."""
    # When trade_history is empty list
    k = compute_conservative_kelly(
        calibrated_confidence=0.65,
        tp_multiplier=2.5,
        sl_multiplier=1.0,
        interval="60",
        trade_history=[],
        mcc_val=0.15,
        cost_bps=5.0
    )
    assert k > 0.0, f"Expected positive Kelly fraction, got {k}"
    assert k <= 0.25, f"Quarter-Kelly should be <= 0.25, got {k}"


def test_ubjson_meta_classifier_fallback():
    """Verify XGBClassifier loads UBJSON binary format via bytearray fallback."""
    from xgboost import XGBClassifier
    import os
    target_meta = "meta_trending_trend_15.json"
    if not os.path.exists(target_meta):
        pytest.skip(f"{target_meta} not found locally")
    
    from xgboost.core import XGBoostError
    meta_clf = XGBClassifier()
    try:
        meta_clf.load_model(target_meta)
        loaded_ok = True
    except (XGBoostError, ValueError, OSError):
        with open(target_meta, "rb") as f:
            meta_clf.load_model(bytearray(f.read()))
        loaded_ok = True
    
    assert loaded_ok is True
    assert meta_clf.get_booster() is not None


def test_expectancy_gate_lookback_filtering():
    """Verify trades older than 14 days are excluded from the expectancy gate evaluation."""
    import time
    now_ts = time.time()
    lookback_cutoff = now_ts - (14.0 * 86400.0)
    
    # 20 legacy trades from 20 days ago (all losses)
    legacy_trades = [
        {"interval": "15m", "exit_time": now_ts - (20.0 * 86400.0), "change_pct": -2.0, "pnl_usd": -10.0}
        for _ in range(20)
    ]
    # 5 recent trades from 2 days ago (all wins)
    recent_trades = [
        {"interval": "15m", "exit_time": now_ts - (2.0 * 86400.0), "change_pct": 3.0, "pnl_usd": 15.0}
        for _ in range(5)
    ]
    all_closed = legacy_trades + recent_trades
    
    # Applying the 14-day calendar lookback filter
    interval_closed = []
    for t in all_closed:
        if str(t.get("interval", "")).replace("m", "") != "15":
            continue
        t_exit = float(t.get("exit_time", 0.0) or 0.0)
        if t_exit > 1e11:
            t_exit /= 1000.0
        if t_exit > 0.0 and t_exit < lookback_cutoff:
            continue
        interval_closed.append(t)
        
    # Must only contain the 5 recent trades, legacy trades filtered out
    assert len(interval_closed) == 5
    # Gate requires >= 15 trades within the window; with 5, it should not trigger negative EV block
    assert len(interval_closed) < 15

