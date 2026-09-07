"""
tests/unit/test_ranging_strategy.py
-----------------------------------
Unit tests for the Option B Specialized Mean-Reversion Ranging Strategy.
"""

import pytest
import numpy as np
import pandas as pd
from ranging_strategy import evaluate_ranging_mean_reversion, RangingSignal


def _create_mock_df(adx=18.0, rsi=50.0, close=80000.0, high=80500.0, low=79500.0, open_p=80000.0,
                     bb_high=81000.0, bb_low=79000.0, bb_mid=80000.0, lower_wick_ratio=1.0, upper_wick_ratio=1.0, n_bars=60):
    rows = []
    for i in range(n_bars):
        rows.append({
            "timestamp": 1780000000000 + (i * 900000),
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
            "ADX": adx,
            "RSI": rsi,
            "BB_high": bb_high,
            "BB_low": bb_low,
            "BB_mid": bb_mid,
            "ATR": 500.0,
            "lower_wick_volume_ratio": lower_wick_ratio,
            "upper_wick_volume_ratio": upper_wick_ratio
        })
    return pd.DataFrame(rows)


def test_trending_market_refusal():
    """Verify ranging strategy rejects trading when ADX >= 25."""
    df = _create_mock_df(adx=32.0, rsi=25.0, close=78900.0, low=78800.0, bb_low=79000.0)
    sig = evaluate_ranging_mean_reversion(df, "BTCUSDT", "15")
    assert sig.is_signal is False
    assert sig.direction == "Neutral"
    assert "Trending" in sig.reason


def test_inside_mid_range_neutral():
    """Verify price inside mid-range safely returns Neutral (no boundary exhaustion)."""
    df = _create_mock_df(adx=18.0, rsi=50.0, close=80000.0, bb_low=79000.0, bb_high=81000.0)
    sig = evaluate_ranging_mean_reversion(df, "BTCUSDT", "15")
    assert sig.is_signal is False
    assert sig.direction == "Neutral"
    assert "Inside Bollinger Mid-Range" in sig.reason


def test_bullish_lower_band_exhaustion_fade():
    """Verify touching lower band with RSI <= 36 and rejection wick triggers Bullish Maker setup."""
    df = _create_mock_df(
        adx=17.5,
        rsi=28.0,
        close=79050.0,
        open_p=79100.0,
        high=79200.0,
        low=78950.0,     # pierced lower band (79000)
        bb_low=79000.0,
        bb_mid=80000.0,
        bb_high=81000.0,
        lower_wick_ratio=1.6
    )
    sig = evaluate_ranging_mean_reversion(df, "BTCUSDT", "15")
    assert sig.is_signal is True
    assert sig.direction == "Bullish"
    assert sig.confidence >= 0.60
    assert sig.order_type == "POST_ONLY_LIMIT"
    assert sig.entry_price <= 79050.0
    assert sig.take_profit == 80000.0  # target at 20-SMA mid-band
    assert sig.stop_loss < sig.entry_price


def test_bearish_upper_band_exhaustion_fade():
    """Verify touching upper band with RSI >= 64 and rejection wick triggers Bearish Maker setup."""
    df = _create_mock_df(
        adx=16.0,
        rsi=72.0,
        close=80950.0,
        open_p=80900.0,
        high=81100.0,    # pierced upper band (81000)
        low=80800.0,
        bb_low=79000.0,
        bb_mid=80000.0,
        bb_high=81000.0,
        upper_wick_ratio=1.8
    )
    sig = evaluate_ranging_mean_reversion(df, "BTCUSDT", "15")
    assert sig.is_signal is True
    assert sig.direction == "Bearish"
    assert sig.confidence >= 0.60
    assert sig.order_type == "POST_ONLY_LIMIT"
    assert sig.entry_price >= 80950.0
    assert sig.take_profit == 80000.0  # target at 20-SMA mid-band
    assert sig.stop_loss > sig.entry_price


def test_empty_or_short_df():
    """Verify empty or insufficient DataFrame gracefully returns non-signal."""
    sig = evaluate_ranging_mean_reversion(pd.DataFrame(), "BTCUSDT", "15")
    assert sig.is_signal is False
    assert sig.direction == "Neutral"


def test_signal_evaluator_ranging_dispatch():
    """Verify SignalEvaluator invokes ranging engine and updates bot_state correctly."""
    from signal_evaluator import SignalEvaluator
    import signal_evaluator
    import core
    import data
    bot_state = {}
    evaluator = SignalEvaluator(bot_state)

    orig_get_history = signal_evaluator.get_history
    orig_add_features = core.add_features
    orig_merge = data.merge_derivatives_sentiment_features
    import features
    orig_feat_add = features.add_features

    # 1. Bullish boundary exhaustion
    df_bull = _create_mock_df(
        adx=17.5,
        rsi=28.0,
        close=79050.0,
        open_p=79100.0,
        high=79200.0,
        low=78950.0,
        bb_low=79000.0,
        bb_mid=80000.0,
        bb_high=81000.0,
        lower_wick_ratio=1.6,
        n_bars=60
    )
    df_bull.attrs["fetch_ok"] = True
    df_bull["close_btc"] = df_bull["close"]

    try:
        signal_evaluator.get_history = lambda symbol, interval, limit, **kw: df_bull.copy()
        core.add_features = lambda df, **kw: df
        features.add_features = lambda df, **kw: df
        data.merge_derivatives_sentiment_features = lambda df, **kw: df

        evaluator.evaluate_interval("BTCUSDT", "15")
        pred_bull = bot_state.get("latest_prediction_bg_BTCUSDT_15m")
        assert pred_bull is not None
        assert pred_bull["direction"] == "Bullish"
        assert pred_bull["signal_source"] == "MEAN_REVERSION_BB"
        assert pred_bull["order_type"] == "POST_ONLY_LIMIT"
        assert pred_bull["confidence"] >= 0.60
        assert pred_bull["take_profit"] == 80000.0

        # 2. Inside mid-range chop
        df_mid = _create_mock_df(adx=18.0, rsi=50.0, close=80000.0, bb_low=79000.0, bb_high=81000.0, n_bars=60)
        df_mid.attrs["fetch_ok"] = True
        df_mid["close_btc"] = df_mid["close"]
        signal_evaluator.get_history = lambda symbol, interval, limit, **kw: df_mid.copy()

        evaluator.evaluate_interval("BTCUSDT", "15")
        pred_mid = bot_state.get("latest_prediction_bg_BTCUSDT_15m")
        assert pred_mid is not None
        assert pred_mid["direction"] == "Neutral"
        assert pred_mid["signal_source"] == "MEAN_REVERSION_BB"
        assert "Inside Bollinger Mid-Range" in pred_mid["setup_type"]
    finally:
        signal_evaluator.get_history = orig_get_history
        core.add_features = orig_add_features
        features.add_features = orig_feat_add
        data.merge_derivatives_sentiment_features = orig_merge


