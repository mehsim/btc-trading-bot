"""
tests/test_architecture_profit_booster.py
-------------------------------------------
Unit test suite covering ArchitectureProfitBooster:
- MFE-based partial profit target calculations
- GARCH volatility-adaptive leverage scaling
- Watchdog memory supervisor health evaluations
"""

import pytest
from architecture_profit_booster import architecture_profit_booster


def test_partial_profit_targets():
    targets = architecture_profit_booster.calculate_partial_profit_targets(
        entry_price=100.0, atr_dollars=2.0, direction="Bullish"
    )
    assert targets["tp_partial_50"] == 102.0
    assert targets["be_trigger"] == 102.4


def test_volatility_adaptive_leverage():
    # Low volatility boosts leverage (e.g. 10x -> 12.5x on BTC)
    lev_low_vol = architecture_profit_booster.compute_volatility_adaptive_leverage(
        base_leverage=10.0, garch_vol_forecast=0.010, symbol="BTCUSDT"
    )
    assert lev_low_vol == 12.5

    # High volatility reduces leverage (e.g. 10x -> 5.0x on Altcoins)
    lev_high_vol = architecture_profit_booster.compute_volatility_adaptive_leverage(
        base_leverage=10.0, garch_vol_forecast=0.040, symbol="ADAUSDT"
    )
    assert lev_high_vol <= 5.0  # Capped at Altcoin max 5.0x


def test_watchdog_memory_health():
    res_normal = architecture_profit_booster.evaluate_watchdog_health(current_memory_mb=450.0, thread_count=15)
    assert res_normal["status"] == "NORMAL"
    assert res_normal["action_required"] == "NONE"

    res_mem = architecture_profit_booster.evaluate_watchdog_health(current_memory_mb=820.0, thread_count=25)
    assert res_mem["status"] == "CRITICAL_MEMORY"
    assert res_mem["action_required"] == "FREE_EXPIRED_CACHES"
