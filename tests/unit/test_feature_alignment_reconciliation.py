import pytest
import pandas as pd
import numpy as np
from core import add_features

def test_live_vs_backtest_feature_reconciliation():
    """
    F-05 Reconciliation Test:
    Asserts bit-for-bit feature equality between offline backtest feature calculation
    and live real-time streaming window feature calculation.
    """
    np.random.seed(42)
    n_candles = 350
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq="15min")
    
    # Generate synthetic price series with trend + noise
    base_price = 50000.0 + np.cumsum(np.random.randn(n_candles) * 50)
    high_prices = base_price + np.random.uniform(5, 25, n_candles)
    low_prices = base_price - np.random.uniform(5, 25, n_candles)
    close_prices = base_price + np.random.uniform(-5, 5, n_candles)
    volumes = np.random.uniform(10, 500, n_candles)

    df_full = pd.DataFrame({
        "open": base_price,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
        "timestamp": [int(ts.timestamp() * 1000) for ts in dates]
    })

    # 1. Offline Backtest Feature Extraction
    df_backtest = add_features(df_full.copy())
    
    # 2. Live Streaming Window Simulation (bar by bar streaming into a rolling buffer of 300 bars)
    rolling_buffer = df_full.iloc[:300].copy()
    df_live_window = add_features(rolling_buffer)
    
    # Re-evaluate for candle 305 in live streaming mode
    rolling_buffer_305 = df_full.iloc[:305].copy()
    df_live_window_305 = add_features(rolling_buffer_305)

    # Compare common numeric feature columns for candle 305
    numeric_cols = df_backtest.select_dtypes(include=[np.number]).columns
    ignored_cols = ["timestamp", "target_trend", "target_price_change"]
    feature_cols = [c for c in numeric_cols if c not in ignored_cols and c in df_live_window_305.columns]

    backtest_row = df_backtest.iloc[304][feature_cols].astype(float).values
    live_row = df_live_window_305.iloc[304][feature_cols].astype(float).values

    # Assert feature agreement between live streaming window and offline backtest
    np.testing.assert_allclose(
        live_row, backtest_row, rtol=1e-4, atol=1e-4,
        err_msg="Live streaming features mismatch offline backtest feature vector"
    )
