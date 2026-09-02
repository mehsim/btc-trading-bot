import pytest
from production_regime_engine import ProductionRegimeEngine

def test_adx_hysteresis_transitions():
    engine = ProductionRegimeEngine(adx_high=26.0, adx_low=22.0)
    sym = "BTCUSDT"
    iv = "60"

    # 1. Initial state is RANGING
    assert engine.update_regime(sym, iv, adx_value=20.0, volatility_ratio=1.0) == "RANGING"

    # 2. ADX rises to 25.0 (below 26.0 threshold) -> Remains RANGING
    assert engine.update_regime(sym, iv, adx_value=25.0, volatility_ratio=1.3) == "RANGING"

    # 3. ADX exceeds 26.0 with high vol -> Transitions to TRENDING
    assert engine.update_regime(sym, iv, adx_value=27.0, volatility_ratio=1.3) == "TRENDING"

    # 4. ADX drops to 24.0 (between 22.0 and 26.0) -> Hysteresis holds TRENDING state!
    assert engine.update_regime(sym, iv, adx_value=24.0, volatility_ratio=1.0) == "TRENDING"

    # 5. ADX drops below 22.0 -> Transitions back to RANGING
    assert engine.update_regime(sym, iv, adx_value=21.0, volatility_ratio=0.7) == "RANGING"

def test_regime_rsi_guards():
    engine = ProductionRegimeEngine()

    # RANGING Regime Checks
    # LONG with RSI > 70 should be BLOCKED (Overbought in range)
    res1 = engine.evaluate_confluence("Bullish", "RANGING", rsi=72.0)
    assert res1["execute"] is False
    assert "Overbought" in res1["reason"]

    # SHORT with RSI < 30 should be BLOCKED (Oversold in range)
    res2 = engine.evaluate_confluence("Bearish", "RANGING", rsi=28.0)
    assert res2["execute"] is False
    assert "Oversold" in res2["reason"]

    # Valid LONG in RANGING (RSI = 50) -> Approved
    res3 = engine.evaluate_confluence("Bullish", "RANGING", rsi=50.0)
    assert res3["execute"] is True

    # TRENDING Regime Checks
    # LONG with RSI < 25 should be BLOCKED (Extreme counter-trend exhaustion)
    res4 = engine.evaluate_confluence("Bullish", "TRENDING", rsi=22.0)
    assert res4["execute"] is False

    # Macro guard active -> Blocked
    res5 = engine.evaluate_confluence("Bullish", "TRENDING", rsi=55.0, macro_guard_active=True)
    assert res5["execute"] is False

def test_classify_market_regime_string_normalization():
    engine = ProductionRegimeEngine()

    # "High Vol, Ranging" maps to RANGING (not unconditionally blocked as CHOPPY)
    res_ranging = engine.evaluate_confluence("Bullish", "High Vol, Ranging", rsi=50.0)
    assert res_ranging["execute"] is True

    # "Choppy" maps to CHOPPY -> Blocked unconditionally
    res_chop = engine.evaluate_confluence("Bullish", "Choppy", rsi=50.0)
    assert res_chop["execute"] is False
    assert "CHOPPY" in res_chop["reason"]

    # "Low Vol, Ranging" with RSI > 70 -> Blocked (Overbought in Range)
    res_ob = engine.evaluate_confluence("Bullish", "Low Vol, Ranging", rsi=75.0)
    assert res_ob["execute"] is False
    assert "Overbought" in res_ob["reason"]

    # "Low Vol, Ranging" with RSI < 30 on Bearish -> Blocked (Oversold in Range)
    res_os = engine.evaluate_confluence("Bearish", "Low Vol, Ranging", rsi=25.0)
    assert res_os["execute"] is False
    assert "Oversold" in res_os["reason"]

    # "High Vol, Trending" with RSI < 25 on Bullish -> Blocked (Exhaustion)
    res_exh = engine.evaluate_confluence("Bullish", "High Vol, Trending", rsi=20.0)
    assert res_exh["execute"] is False
    assert "Exhaustion" in res_exh["reason"]

    # "Low Vol, Trending" with normal RSI (50.0) -> Approved
    res_ok = engine.evaluate_confluence("Bullish", "Low Vol, Trending", rsi=50.0)
    assert res_ok["execute"] is True

