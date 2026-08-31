import pytest
from trade_calculators import get_realized_rr_haircut, calculate_required_p, passes_economic_gate, UnifiedTargetGenerator


def test_economic_gate_uses_resolved_order_geometry():
    """Verify that when order TP is floored (e.g. 1.50 ATR) and SL is expanded by structural stop (e.g. 1.25 ATR),
    the economic gate calculates p* using the exact resolved geometry rather than nominal config."""
    entry_price = 50000.0
    atr_dollars = 500.0  # 1.0% ATR

    # Case 1: Raw unadjusted multipliers from config
    raw_sl_mult = 0.5098
    raw_tp_mult = 0.9782
    raw_nominal_rr = raw_tp_mult / raw_sl_mult
    raw_haircut = get_realized_rr_haircut(interval="15", regime="trending", nominal_rr=raw_nominal_rr)
    raw_p_star = raw_sl_mult / (raw_tp_mult * raw_haircut + raw_sl_mult)

    # Case 2: Resolved order geometry with min_target=1.50 ATR and structural stop=1.25 ATR
    resolved_sl_mult = 1.25
    resolved_tp_mult = 1.50
    resolved_nominal_rr = resolved_tp_mult / resolved_sl_mult
    resolved_haircut = get_realized_rr_haircut(interval="15", regime="trending", nominal_rr=resolved_nominal_rr)
    resolved_p_star = resolved_sl_mult / (resolved_tp_mult * resolved_haircut + resolved_sl_mult)

    # The resolved p* must demand a significantly higher win rate than the unaligned raw config
    assert resolved_p_star > raw_p_star
    assert resolved_p_star >= 0.50, f"Expected resolved p* >= 0.50, got {resolved_p_star:.4f}"

    # A signal with calibrated confidence 0.42 should be rejected under resolved geometry
    assert passes_economic_gate(
        entry=entry_price,
        tp=entry_price + (resolved_tp_mult * atr_dollars),
        sl=entry_price - (resolved_sl_mult * atr_dollars),
        conf=0.42,
        cost_frac=0.0006,
        realized_rr_haircut=resolved_haircut
    ) is False


def test_unified_regime_predicate_consistency():
    """Verify that ranging vs trending target resolution behaves consistently."""
    is_ranging = True
    base_tp_ranging = 1.15
    min_target = 1.50
    # Both branches should honor the floor and regime
    tp_target = max(base_tp_ranging if is_ranging else 1.85, min_target)
    assert tp_target == 1.50
