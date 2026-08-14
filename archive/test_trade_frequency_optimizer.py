"""
tests/test_trade_frequency_optimizer.py
----------------------------------------
Unit tests covering TradeFrequencyOptimizer:
- Dynamic R:R Take-Profit adjustment to meet 2.0:1 floor
- Regime-adaptive confidence threshold scaling
- Expanded 25-asset symbol universe
"""

import pytest
from trade_frequency_optimizer import trade_frequency_optimizer, EXPANDED_SYMBOL_UNIVERSE


def test_expanded_universe_size():
    assert len(EXPANDED_SYMBOL_UNIVERSE) >= 20
    assert "BTCUSDT" in EXPANDED_SYMBOL_UNIVERSE
    assert "SUIUSDT" in EXPANDED_SYMBOL_UNIVERSE


def test_tp_target_rr_optimization():
    opt_tp, new_rr, adjusted = trade_frequency_optimizer.optimize_tp_target_for_rr(
        entry_price=100.0,
        stop_price=98.0,  # 2.0 stop dist
        atr_dollars=1.0,
        direction="Bullish",
        min_rr_required=2.0
    )
    assert opt_tp == 104.0  # 4.0 tp dist / 2.0 stop dist = 2.0 R:R
    assert new_rr == 2.0
    assert adjusted is True


def test_regime_adaptive_confidence_threshold():
    # Strong trend ADX >= 25 scales threshold down to 0.55
    thresh_trend = trade_frequency_optimizer.calculate_regime_adaptive_confidence_threshold(adx_val=28.0)
    assert thresh_trend == 0.55

    # Ranging chop ADX <= 15 tightens threshold to 0.75
    thresh_chop = trade_frequency_optimizer.calculate_regime_adaptive_confidence_threshold(adx_val=12.0)
    assert thresh_chop == 0.75
