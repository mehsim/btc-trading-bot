"""
Tests for Institutional Joint Risk Budget Allocator, Context Feature Vector, Empirical Utility Estimator, and Execution Validator.
"""

import pytest
import numpy as np
import pandas as pd
from context_feature_vector import context_feature_vector_engine, FeatureMetadata
from utility_distribution_estimator import empirical_utility_estimator
from execution_validator import execution_validator
from risk_engine import joint_risk_budget_allocator

def test_context_feature_vector_freshness_and_decay():
    ctx_data = {
        "calibrated_confidence": 0.85,
        "atr_norm": 0.012,
        "adx": 32.0,
        "session": "london",
        "regime": "Trending (GMM)",
        "portfolio_heat": 0.05,
        "mhi": 88.0
    }
    features = context_feature_vector_engine.build_context_vector(ctx_data)
    assert "confidence" in features
    assert features["confidence"].value == 0.85
    assert features["confidence"].effective_weight > 0.80

    multipliers = context_feature_vector_engine.evaluate_context_multipliers(features)
    assert "target_expansion" in multipliers
    assert "stop_expansion" in multipliers
    assert multipliers["target_expansion"] >= 1.0


def test_stop_loss_not_squeezed_by_high_confidence():
    """Verify invariant: Stop Loss distance is NOT artificially squeezed when confidence increases in high volatility."""
    df_dummy = pd.DataFrame({
        "open": [100.0] * 50,
        "high": [102.0] * 50,
        "low": [98.0] * 50,
        "close": [101.0] * 50,
        "ATR": [2.0] * 50,
        "ATR_norm": [0.02] * 50
    })

    # Allocations for low vs high confidence
    alloc_normal_conf = joint_risk_budget_allocator.allocate_risk_budget(
        symbol="BTCUSDT", entry_price=60000.0, atr_dollars=500.0, atr_norm=0.015,
        calibrated_confidence=0.60, direction="Bullish", total_equity=1000.0, df_completed=df_dummy
    )

    alloc_high_conf = joint_risk_budget_allocator.allocate_risk_budget(
        symbol="BTCUSDT", entry_price=60000.0, atr_dollars=500.0, atr_norm=0.015,
        calibrated_confidence=0.95, direction="Bullish", total_equity=1000.0, df_completed=df_dummy
    )

    # Invariant check: Stop distance in high confidence trade must NOT be smaller than base structural stop
    assert alloc_high_conf["stop_distance"] >= alloc_normal_conf["stop_distance"] * 0.99
    # Capital sizing & Kelly fraction should scale UP with higher confidence
    assert alloc_high_conf["position_size"] > alloc_normal_conf["position_size"]


def test_portfolio_heat_upstream_budget_reduction():
    """Verify upstream portfolio heat reduces available risk budget and position size."""
    alloc_no_heat = joint_risk_budget_allocator.allocate_risk_budget(
        symbol="BTCUSDT", entry_price=60000.0, atr_dollars=500.0, atr_norm=0.01,
        calibrated_confidence=0.75, direction="Bullish", total_equity=1000.0, portfolio_heat=0.00
    )

    alloc_high_heat = joint_risk_budget_allocator.allocate_risk_budget(
        symbol="BTCUSDT", entry_price=60000.0, atr_dollars=500.0, atr_norm=0.01,
        calibrated_confidence=0.75, direction="Bullish", total_equity=1000.0, portfolio_heat=0.15
    )

    assert alloc_high_heat["position_size"] < alloc_no_heat["position_size"]
    assert alloc_high_heat["capital_at_risk"] < alloc_no_heat["capital_at_risk"]


def test_mhi_governance_kelly_capping():
    """Verify MHI score governs max Kelly fraction (CRITICAL halts trading)."""
    assert joint_risk_budget_allocator.get_mhi_max_kelly(85.0) == 0.25
    assert joint_risk_budget_allocator.get_mhi_max_kelly(70.0) == 0.20
    assert joint_risk_budget_allocator.get_mhi_max_kelly(55.0) == 0.10
    assert joint_risk_budget_allocator.get_mhi_max_kelly(45.0) == 0.00

    alloc_critical = joint_risk_budget_allocator.allocate_risk_budget(
        symbol="BTCUSDT", entry_price=60000.0, atr_dollars=500.0, atr_norm=0.01,
        calibrated_confidence=0.90, direction="Bullish", total_equity=1000.0, mhi_score=40.0
    )
    assert alloc_critical["execution_permitted"] is False
    assert alloc_critical["position_size"] == 0.0


def test_empirical_bootstrap_utility_estimator():
    u_est = empirical_utility_estimator.estimate_utility_distribution(
        predicted_win_rate=0.60, target_distance=1000.0, stop_distance=500.0
    )
    assert "expected_utility_mean" in u_est
    assert "expected_utility_std" in u_est
    assert "p_utility_positive" in u_est
    assert "cvar_95" in u_est
    assert u_est["p_utility_positive"] > 0.50


def test_pre_exchange_execution_validator():
    # 1. Valid Long Order
    is_valid, msg = execution_validator.validate_order(
        symbol="BTCUSDT", direction="BUY", entry_price=60000.0, stop_loss_price=59000.0,
        take_profit_price=62000.0, position_size_usd=100.0, live_price=60050.0
    )
    assert is_valid is True
    assert msg == "VALID"

    # 2. Invalid Stop Loss Side
    is_valid_sl, msg_sl = execution_validator.validate_order(
        symbol="BTCUSDT", direction="BUY", entry_price=60000.0, stop_loss_price=61000.0,
        take_profit_price=62000.0, position_size_usd=100.0, live_price=60050.0
    )
    assert is_valid_sl is False
    assert "strictly below" in msg_sl

    # 3. Low R:R Ratio
    is_valid_rr, msg_rr = execution_validator.validate_order(
        symbol="BTCUSDT", direction="BUY", entry_price=60000.0, stop_loss_price=59000.0,
        take_profit_price=60500.0, position_size_usd=100.0, live_price=60050.0
    )
    assert is_valid_rr is False
    assert "minimum threshold" in msg_rr
