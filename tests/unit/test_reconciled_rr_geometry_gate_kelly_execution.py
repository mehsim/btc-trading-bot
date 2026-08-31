import pytest
import numpy as np
from trade_calculators import get_realized_rr_haircut, UnifiedTargetGenerator
from risk_engine import compute_conservative_kelly


def test_gate_kelly_execution_rr_geometry_reconciliation():
    """Verify that the economic gate p*, Kelly payoff b_ratio, and executed targets
    are derived from the exact same resolved order geometry and realized R:R haircut."""
    entry_price = 50000.0
    atr_dollars = 500.0  # 1% ATR
    atr_norm = 0.01

    # Order resolution
    raw_cfg_sl_m = 0.5098
    raw_cfg_tp_m = 0.9782

    # Floors and structural stops applied
    min_target_floor = 1.50
    resolved_tp_m = max(raw_cfg_tp_m, min_target_floor)  # 1.50
    structural_sl_m = 1.25
    resolved_sl_m = structural_sl_m  # 1.25

    # Executed prices
    take_profit_price = entry_price + (resolved_tp_m * atr_dollars)
    stop_loss_price = entry_price - (resolved_sl_m * atr_dollars)

    # Haircut & Effective targets
    nominal_rr = resolved_tp_m / resolved_sl_m  # 1.50 / 1.25 = 1.20
    haircut = get_realized_rr_haircut(interval="15", regime="trending", nominal_rr=nominal_rr)
    effective_tp_m = resolved_tp_m * haircut

    # 1. Gate Break-Even p*
    p_star = resolved_sl_m / (effective_tp_m + resolved_sl_m)
    expected_b = effective_tp_m / resolved_sl_m
    assert np.isclose(p_star, 1.0 / (expected_b + 1.0))

    # 2. Kelly Payoff b_ratio
    # If confidence is exactly p*, Kelly must return 0.0 (no edge)
    kelly_at_p_star = compute_conservative_kelly(
        calibrated_confidence=p_star,
        tp_multiplier=effective_tp_m,
        sl_multiplier=resolved_sl_m,
        interval="15"
    )
    assert kelly_at_p_star == 0.0

    # If confidence is below p*, Kelly must return 0.0
    kelly_below_p_star = compute_conservative_kelly(
        calibrated_confidence=p_star - 0.05,
        tp_multiplier=effective_tp_m,
        sl_multiplier=resolved_sl_m,
        interval="15"
    )
    assert kelly_below_p_star == 0.0

    # If confidence is above p*, Kelly must return positive fraction
    kelly_above_p_star = compute_conservative_kelly(
        calibrated_confidence=p_star + 0.10,
        tp_multiplier=effective_tp_m,
        sl_multiplier=resolved_sl_m,
        interval="15"
    )
    assert kelly_above_p_star > 0.0

    # 3. Executed prices consistency
    actual_tp_dist = abs(take_profit_price - entry_price) / atr_dollars
    actual_sl_dist = abs(stop_loss_price - entry_price) / atr_dollars
    assert actual_tp_dist == resolved_tp_m
    assert actual_sl_dist == resolved_sl_m
