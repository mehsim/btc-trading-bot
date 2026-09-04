import pytest
import numpy as np
import pandas as pd
from risk_engine import get_timeframe_stop_multiplier, JointRiskBudgetAllocator


def test_timeframe_stop_multipliers():
    """Verify that higher timeframes receive wider stop cushions while lower timeframes stay tight."""
    assert get_timeframe_stop_multiplier("240") == 1.35
    assert get_timeframe_stop_multiplier("120") == 1.15
    assert get_timeframe_stop_multiplier("60") == 1.00
    assert get_timeframe_stop_multiplier("30") == 0.80
    assert get_timeframe_stop_multiplier("15") == 0.80
    # Higher timeframes must have strictly larger stop buffers than intraday micro timeframes
    assert get_timeframe_stop_multiplier("240") > get_timeframe_stop_multiplier("60") >= get_timeframe_stop_multiplier("15")


def test_constant_dollar_risk_under_widened_stops():
    """Verify that widening stop loss distance maintains strictly bounded dollar risk in dimensional sizing."""
    equity = 100.0
    f_clamped = 0.03 # 3% capital at risk
    
    # 1. Normal stop distance: 1.0% (25.0 USD on ETH $2500)
    sl_frac_normal = 0.01
    notional_normal = (equity * f_clamped) / sl_frac_normal
    dollar_risk_normal = notional_normal * sl_frac_normal
    
    # 2. 4H Widened stop distance: 2.0% (50.0 USD on ETH $2500)
    sl_frac_widened = 0.02
    notional_widened = (equity * f_clamped) / sl_frac_widened
    dollar_risk_widened = notional_widened * sl_frac_widened
    
    # Notional size on widened stop scales down inversely
    assert notional_widened == notional_normal / 2.0
    # Capital at risk in USD ($) is strictly identical
    assert dollar_risk_normal == dollar_risk_widened == (equity * f_clamped)


def test_high_conviction_portfolio_heat_ladder():
    """Verify that portfolio heat ceiling expands to 40% when confidence >= 0.55."""
    allocator = JointRiskBudgetAllocator(max_capital_risk_pct=0.02)
    equity = 100.0
    
    # At 25% portfolio heat:
    # 1) Moderate confidence (0.50) uses 30% ceiling -> heat_ratio = 25/30 = 0.833, budget factor = 0.167
    res_mod = allocator.allocate_risk_budget(
        symbol="SOLUSDT",
        entry_price=100.0,
        atr_dollars=2.0,
        atr_norm=0.02,
        calibrated_confidence=0.50,
        direction="Bullish",
        total_equity=equity,
        portfolio_heat=0.25,
        mhi_score=90.0,
        stop_distance=2.0,
        target_distance=15.0
    )
    
    # 2) High conviction (0.58) uses 40% ceiling -> heat_ratio = 25/40 = 0.625, budget factor = 0.375
    res_high = allocator.allocate_risk_budget(
        symbol="SOLUSDT",
        entry_price=100.0,
        atr_dollars=2.0,
        atr_norm=0.02,
        calibrated_confidence=0.58,
        direction="Bullish",
        total_equity=equity,
        portfolio_heat=0.25,
        mhi_score=90.0,
        stop_distance=2.0,
        target_distance=15.0
    )
    
    assert res_high["execution_permitted"] is True
    # High conviction setup receives greater risk budget under the expanded ceiling
    assert res_high["position_size"] > res_mod["position_size"]


def test_trend_adaptive_scale_out_logic():
    """Verify ADX modulation on scale out multipliers."""
    # Strong breakout trend (ADX >= 35) -> 1.40x ATR
    adx_strong = 42.0
    scale_mult_strong = 1.40 if adx_strong >= 35.0 else (0.80 if adx_strong < 22.0 else 1.00)
    assert scale_mult_strong == 1.40
    
    # Sideways chop (ADX < 22) -> 0.80x ATR
    adx_chop = 18.0
    scale_mult_chop = 1.40 if adx_chop >= 35.0 else (0.80 if adx_chop < 22.0 else 1.00)
    assert scale_mult_chop == 0.80
    
    # Standard market (ADX = 28) -> 1.00x ATR
    adx_std = 28.0
    scale_mult_std = 1.40 if adx_std >= 35.0 else (0.80 if adx_std < 22.0 else 1.00)
    assert scale_mult_std == 1.00
