import pytest
import pandas as pd
import numpy as np
from confluence_engine import check_pre_trade_confluence

def create_mock_4h_df(trend="Bearish"):
    """Generates synthetic 4H DataFrame with controlled EMA trend and RSI."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq="4h")
    
    if trend == "Bearish":
        close_prices = np.linspace(100, 50, 50)
    elif trend == "Bullish":
        close_prices = np.linspace(50, 100, 50)
    else:
        close_prices = np.full(50, 75.0)

    df = pd.DataFrame({
        "open": close_prices,
        "high": close_prices + 1.0,
        "low": close_prices - 1.0,
        "close": close_prices,
        "volume": 1000.0
    }, index=dates)
    return df

def test_4h_trend_hard_gate_rejects_adverse_trend():
    """Asserts that check_pre_trade_confluence rejects a Bullish signal if 4H trend is Bearish."""
    df_1h = pd.DataFrame({
        "close": [50000.0] * 30,
        "open": [49950.0] * 30,
        "high": [50100.0] * 30,
        "low": [49900.0] * 30,
        "volume": [100.0] * 30,
        "RSI": [50.0] * 30
    })
    
    htf_cache = {}
    mock_4h_bearish = create_mock_4h_df(trend="Bearish")
    htf_cache[("BTCUSDT", "240")] = (mock_4h_bearish, 9999999999) # valid cache
    
    # Send a Bullish ML signal into a Bearish 4H market
    all_pass, results, score_pct = check_pre_trade_confluence(
        current_price=50000.0,
        df_1h=df_1h,
        ml_trend="Bullish",
        news_sentiment="Neutral",
        expected_pct_change=0.02,
        interval="60",
        symbol="BTCUSDT",
        htf_cache=htf_cache
    )

    assert not all_pass, "Expected confluence check to REJECT Bullish signal when 4H trend is Bearish"
    assert not results["4h_Trend"]["pass"], "4h_Trend check must fail on adverse data"

def test_4h_rsi_hard_gate_rejects_overbought():
    """Asserts that check_pre_trade_confluence rejects a Bullish signal if 4H RSI >= 75."""
    df_1h = pd.DataFrame({
        "close": [50000.0] * 30,
        "open": [49950.0] * 30,
        "high": [50100.0] * 30,
        "low": [49900.0] * 30,
        "volume": [100.0] * 30,
        "RSI": [50.0] * 30
    })

    htf_cache = {}
    mock_4h_bullish = create_mock_4h_df(trend="Bullish")
    # Force high RSI by giving a steep parabolic end
    mock_4h_bullish.loc[mock_4h_bullish.index[-15:], "close"] = np.linspace(100, 300, 15)
    htf_cache[("BTCUSDT", "240")] = (mock_4h_bullish, 9999999999)

    all_pass, results, score_pct = check_pre_trade_confluence(
        current_price=50000.0,
        df_1h=df_1h,
        ml_trend="Bullish",
        news_sentiment="Neutral",
        expected_pct_change=0.02,
        interval="60",
        symbol="BTCUSDT",
        htf_cache=htf_cache
    )

    # Check detail string contains 4h RSI and that overbought RSI causes rejection
    detail_str = results["4h_RSI"]["detail"]
    assert "4h RSI is" in detail_str
    # Extract the numerical float value safely
    tokens = detail_str.split()
    for tok in tokens:
        try:
            val = float(tok)
            if val >= 75.0:
                assert not results["4h_RSI"]["pass"]
                assert not all_pass
                break
        except ValueError:
            continue
