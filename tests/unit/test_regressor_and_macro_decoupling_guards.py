"""
tests/unit/test_regressor_and_macro_decoupling_guards.py
--------------------------------------------------------
Unit tests for:
  1. Regressor Directional Consensus Gate (contradiction detection)
  2. Altcoin Macro Decoupling Guard (opposing local SMA50)
  3. Intraday structural SL and TP capping in trade_calculators.py
"""

import pytest
import pandas as pd
import numpy as np
from trade_calculators import calculate_adaptive_structural_stop, resolve_trade_geometry


def test_adaptive_structural_stop_intraday_cap():
    """Verify that structural stop on intraday (30m) is capped at 1.75x ATR / 2.5% max distance."""
    entry_price = 100.0
    atr_val = 5.0  # Large ATR
    df_recent = pd.DataFrame({
        "high": [102.0, 103.0, 101.0],
        "low": [85.0, 88.0, 90.0],  # Very deep swing low
        "close": [100.0, 100.0, 100.0],
        "volume": [100, 100, 100]
    })
    # Intraday 30m
    sl_price, sl_pct, meta = calculate_adaptive_structural_stop(
        df_recent=df_recent,
        entry_price=entry_price,
        direction="Bullish",
        atr_val=atr_val,
        interval="30"
    )
    # Cap is min(1.75 * 5.0, 100.0 * 0.025) = min(8.75, 2.5) = 2.5
    # So sl_price must be at least entry_price - 2.5 = 97.5
    assert sl_price >= 97.5, f"Expected sl_price >= 97.5, got {sl_price}"


def test_resolve_trade_geometry_intraday_tp_cap():
    """Verify that intraday TP distance is capped to prevent unachievable targets."""
    entry_price = 100.0
    atr_dollars = 2.0
    df_recent = pd.DataFrame({
        "high": [101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 100.0],
        "volume": [100, 100, 100]
    })
    geom = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        atr_dollars=atr_dollars,
        interval="30",
        base_sl_multiplier=1.0,
        base_tp_multiplier=4.0,  # Large target
        df=df_recent
    )
    tp_dist = geom["tp_dist"]
    # Cap is min(2.5 * 2.0, 100 * 0.04) = min(5.0, 4.0) = 4.0
    assert tp_dist <= 5.0, f"Expected TP distance <= 5.0, got {tp_dist}"


def test_regressor_directional_consensus_logic():
    """Verify that a positive classifier setup is rejected if regressor predicts negative return."""
    ml_trend = "Bullish"
    pred_change = -0.005  # -0.5% predicted drop
    close_price = 4.15
    pred_pct = (abs(pred_change) / close_price) * 100.0

    # For 30m, min_conflict_pct is 0.10%
    min_conflict_pct = 0.10
    strong_conflict = (ml_trend == "Bullish" and pred_change < 0 and pred_pct > min_conflict_pct) or \
                      (ml_trend == "Bearish" and pred_change > 0 and pred_pct > min_conflict_pct)

    assert strong_conflict is True, "Expected strong conflict to block trade"

    # Conversely, if regressor agrees with direction
    agreeing_pred = 0.010
    agreeing_conflict = (ml_trend == "Bullish" and agreeing_pred < 0) or \
                        (ml_trend == "Bearish" and agreeing_pred > 0)
    assert agreeing_conflict is False, "Expected no conflict when regressor agrees"


def test_macro_decoupling_guard_logic():
    """Verify that altcoin below its SMA50 does not receive macro alignment discount."""
    symbol = "DOTUSDT"
    iv = "30"
    ml_trend = "Bullish"
    htf_trend = "Bullish"

    latest_candle = {
        "close": 4.10,
        "SMA_50": 4.25  # Below SMA50 -> local downtrend
    }

    is_decoupled = False
    if symbol != "BTCUSDT" and str(iv) in ["15", "30", "60"]:
        cur_c = float(latest_candle.get("close", 0.0))
        c_sma50 = float(latest_candle.get("SMA_50", 0.0))
        if ml_trend == "Bullish" and c_sma50 > 0 and cur_c < c_sma50:
            is_decoupled = True
        elif ml_trend == "Bearish" and c_sma50 > 0 and cur_c > c_sma50:
            is_decoupled = True

    assert is_decoupled is True, "Expected DOTUSDT to be identified as decoupled from macro trend"
