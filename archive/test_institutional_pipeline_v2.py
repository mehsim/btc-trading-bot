import pytest
import pandas as pd
from production_regime_engine import ProductionRegimeEngine
from trade_calculators import (
    calculate_uncertainty_tp_scaling,
    calculate_dynamic_structural_buffer,
    get_regime_adaptive_min_rr,
    calculate_3stage_partial_tp_targets
)
from confluence_engine import calculate_exit_quality_score, evaluate_expectancy_gate
import config

def test_regime_hysteresis_v2():
    engine = ProductionRegimeEngine(strong_enter=32.0, strong_exit=28.0, mod_low=22.0)
    
    # 1. RANGING state (ADX = 20)
    reg = engine.update_regime(symbol="BTCUSDT", interval="15", adx_value=20.0)
    assert reg == "RANGING"

    # 2. MODERATE_TREND state (ADX = 25)
    reg = engine.update_regime(symbol="BTCUSDT", interval="15", adx_value=25.0)
    assert reg == "MODERATE_TREND"

    # 3. Enter STRONG_TREND (ADX = 33 > 32)
    reg = engine.update_regime(symbol="BTCUSDT", interval="15", adx_value=33.0)
    assert reg == "STRONG_TREND"

    # 4. Stay STRONG_TREND during minor dip (ADX = 29 > 28 exit)
    reg = engine.update_regime(symbol="BTCUSDT", interval="15", adx_value=29.0)
    assert reg == "STRONG_TREND"

    # 5. Exit STRONG_TREND when ADX drops below 28 (ADX = 27)
    reg = engine.update_regime(symbol="BTCUSDT", interval="15", adx_value=27.0)
    assert reg == "MODERATE_TREND"

def test_uncertainty_tp_scaling():
    # 95% confidence -> 90% expected move scaling
    tp_95 = calculate_uncertainty_tp_scaling(0.95, 0.02)
    assert pytest.approx(tp_95, 0.0001) == 0.018

    # 75% confidence -> 75% expected move scaling
    tp_75 = calculate_uncertainty_tp_scaling(0.75, 0.02)
    assert pytest.approx(tp_75, 0.0001) == 0.015

def test_dynamic_structural_buffer():
    # Price $1000, ATR $10, Spread 0.05%
    buffer = calculate_dynamic_structural_buffer(atr_dollars=10.0, current_price=1000.0, spread_pct=0.0005)
    # 0.25 * 10 = 2.50 vs 1000 * 0.0015 = 1.50 -> 2.50
    assert buffer == 2.50

def test_regime_adaptive_min_rr():
    assert get_regime_adaptive_min_rr("STRONG_TREND") == 2.2
    assert get_regime_adaptive_min_rr("MODERATE_TREND") == 1.8
    assert get_regime_adaptive_min_rr("RANGING") == 1.5

def test_3stage_partial_tp_targets():
    targets = calculate_3stage_partial_tp_targets(entry_price=100.0, direction="Bullish", sl_price=95.0, tp_final_price=110.0)
    assert targets["tp1"]["price"] == 104.0
    assert targets["tp1"]["size_pct"] == 0.30
    assert targets["tp2"]["price"] == 107.0
    assert targets["runner"]["price"] == 110.0
    assert targets["runner"]["size_pct"] == 0.40

def test_exit_quality_score_and_expectancy_gate():
    eqs = calculate_exit_quality_score(
        structure_pass=True,
        liquidity_pass=True,
        expected_move_pct=0.02,
        spread_pct=0.0003,
        funding_rate=0.0001,
        atr_norm=0.005,
        regime="STRONG_TREND"
    )
    assert eqs >= config.MIN_EXIT_QUALITY_SCORE

    ev_pass, ev_val = evaluate_expectancy_gate(historical_win_rate=55.0, avg_win_pct=2.0, avg_loss_pct=1.0)
    assert ev_pass is True
    assert ev_val > 0.0
