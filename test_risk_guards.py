"""
test_risk_guards.py
--------------------
Unit tests verifying that risk guards, SL/TP geometry rules, and structure checks
permit valid trade setups while rejecting invalid ones.
"""

import pytest
from trade_calculators import assert_valid_geometry, validate_trade_structure

def test_assert_valid_geometry_permissive_bullish():
    # Valid Bullish: SL < Entry < TP
    assert assert_valid_geometry("Bullish", entry=68000.0, sl=67000.0, tp=70000.0, symbol="BTCUSDT") is True
    assert assert_valid_geometry("Long", entry=100.0, sl=95.0, tp=110.0, symbol="SOLUSDT") is True
    assert assert_valid_geometry("BUY", entry=1.0, sl=0.9, tp=1.2, symbol="ADAUSDT") is True

def test_assert_valid_geometry_permissive_bearish():
    # Valid Bearish: SL > Entry > TP
    assert assert_valid_geometry("Bearish", entry=68000.0, sl=69000.0, tp=66000.0, symbol="BTCUSDT") is True
    assert assert_valid_geometry("Short", entry=100.0, sl=105.0, tp=90.0, symbol="SOLUSDT") is True
    assert assert_valid_geometry("SELL", entry=1.0, sl=1.1, tp=0.8, symbol="ADAUSDT") is True

def test_assert_valid_geometry_rejects_invalid():
    # Invalid Bullish: SL > Entry
    with pytest.raises(ValueError):
        assert_valid_geometry("Bullish", entry=68000.0, sl=69000.0, tp=70000.0, symbol="BTCUSDT")

    # Invalid Bearish: SL < Entry
    with pytest.raises(ValueError):
        assert_valid_geometry("Bearish", entry=68000.0, sl=67000.0, tp=66000.0, symbol="BTCUSDT")

def test_validate_trade_structure_permissive():
    is_valid, adjusted_struct, struct_log = validate_trade_structure(
        entry_price=68000.0,
        stop_price=67000.0,
        tp_price=70000.0,
        atr_dollars=500.0,
        leverage=5.0,
        interval="15",
        symbol="BTCUSDT",
        direction="Bullish"
    )
    assert is_valid is True
    assert adjusted_struct["stop_price"] < 68000.0
    assert adjusted_struct["tp_price"] > 68000.0

if __name__ == "__main__":
    test_assert_valid_geometry_permissive_bullish()
    test_assert_valid_geometry_permissive_bearish()
    test_assert_valid_geometry_rejects_invalid()
    test_validate_trade_structure_permissive()
    print("✅ All risk guard permissive and blocking unit tests PASSED cleanly.")
