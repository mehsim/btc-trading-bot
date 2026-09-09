"""
tests/unit/test_empirical_outcomes_type_safety.py
-------------------------------------------------
Unit test suite verifying type safety for empirical outcome win rate calculations
in JointRiskBudgetAllocator. Ensures resilience against legacy string-serialized
PnL values, non-numeric entries, and mixed boolean types.
"""

import pandas as pd
import pytest
from risk_engine import JointRiskBudgetAllocator


def test_allocate_risk_budget_with_string_pnl_usd():
    """Verify allocate_risk_budget does not raise TypeError when df_completed has string pnl_usd."""
    allocator = JointRiskBudgetAllocator()
    df_with_strings = pd.DataFrame([
        {"pnl_usd": "0.07238063", "success": "True", "interval": "30"},
        {"pnl_usd": "0.13073715", "success": "True", "interval": "30"},
        {"pnl_usd": "-0.82", "success": "False", "interval": "30"},
        {"pnl_usd": 0.50, "success": 1, "interval": "30"},
        {"pnl_usd": -0.20, "success": 0, "interval": "30"},
        {"pnl_usd": 0.10, "success": 1, "interval": "30"},
        {"pnl_usd": -0.05, "success": 0, "interval": "30"},
        {"pnl_usd": 0.30, "success": 1, "interval": "30"},
        {"pnl_usd": -0.15, "success": 0, "interval": "30"},
        {"pnl_usd": 0.25, "success": 1, "interval": "30"},
    ])

    res = allocator.allocate_risk_budget(
        symbol="DOTUSDT",
        entry_price=0.82,
        atr_dollars=0.02,
        atr_norm=0.024,
        calibrated_confidence=0.55,
        direction="Bullish",
        total_equity=100.0,
        df_completed=df_with_strings,
        stop_distance=0.015,
        target_distance=0.035,
        interval="30"
    )

    assert isinstance(res, dict)
    assert "position_size" in res
    assert res["position_size"] >= 0.0


def test_allocate_risk_budget_with_string_success():
    """Verify allocate_risk_budget handles df_completed with string success without pnl_usd."""
    allocator = JointRiskBudgetAllocator()
    df_with_str_success = pd.DataFrame([
        {"success": "True", "interval": "240"},
        {"success": "False", "interval": "240"},
        {"success": "True", "interval": "240"},
        {"success": "False", "interval": "240"},
        {"success": "True", "interval": "240"},
        {"success": "False", "interval": "240"},
        {"success": "True", "interval": "240"},
        {"success": "False", "interval": "240"},
        {"success": "True", "interval": "240"},
        {"success": "True", "interval": "240"},
    ])

    res = allocator.allocate_risk_budget(
        symbol="BTCUSDT",
        entry_price=60000.0,
        atr_dollars=500.0,
        atr_norm=0.01,
        calibrated_confidence=0.58,
        direction="Bullish",
        total_equity=1000.0,
        df_completed=df_with_str_success,
        stop_distance=600.0,
        target_distance=1200.0,
        interval="240"
    )

    assert isinstance(res, dict)
    assert "position_size" in res


def test_allocate_risk_budget_with_trade_history_strings_and_none():
    """Verify fallback trade_history list handles strings, empty strings, and None."""
    allocator = JointRiskBudgetAllocator()
    th = [
        {"pnl_usd": "10.5", "return_pct": "1.2", "success": "True", "interval": "240"},
        {"pnl_usd": "-5.2", "return_pct": "-0.6", "success": "False", "interval": "240"},
        {"pnl_usd": None, "return_pct": None, "success": None, "interval": "240"},
        {"pnl_usd": "", "return_pct": "", "success": "win", "interval": "240"},
        {"pnl_usd": 12.0, "return_pct": 1.5, "success": True, "interval": "240"},
        {"pnl_usd": -3.0, "return_pct": -0.4, "success": False, "interval": "240"},
        {"pnl_usd": 8.0, "return_pct": 0.9, "success": 1, "interval": "240"},
        {"pnl_usd": -2.0, "return_pct": -0.2, "success": 0, "interval": "240"},
        {"pnl_usd": 15.0, "return_pct": 1.8, "success": True, "interval": "240"},
        {"pnl_usd": -1.0, "return_pct": -0.1, "success": False, "interval": "240"},
    ]

    res = allocator.allocate_risk_budget(
        symbol="XRPUSDT",
        entry_price=0.55,
        atr_dollars=0.01,
        atr_norm=0.018,
        calibrated_confidence=0.55,
        direction="Bullish",
        total_equity=500.0,
        trade_history=th,
        stop_distance=0.01,
        target_distance=0.025,
        interval="240"
    )

    assert isinstance(res, dict)
    assert "position_size" in res
