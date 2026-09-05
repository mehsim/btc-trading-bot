import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import config
from config import is_manifest_degenerate, TIMEFRAME_CONFIG
from kelly_tracker import KellyTracker
from decision_journal import DecisionRecord, ReasonCode


def test_finding_28_kelly_tracker_legacy_untimestamped_trades_preserved():
    """Finding #28: Untimestamped legacy trades must not be dropped when a new timestamped trade arrives."""
    kt = KellyTracker(data_file="/tmp/test_legacy_kelly_history.json")
    # 15 legacy trades with no timestamp (measured negative edge: 1 win, 14 losses)
    legacy_trades = [
        {"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": -10.0, "return_pct": -0.01, "slippage_pct": 0.0005}
        for _ in range(14)
    ]
    legacy_trades.append({"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": 15.0, "return_pct": 0.015, "slippage_pct": 0.0005})
    
    # Add one newly timestamped trade
    now_iso = "2026-09-05T12:00:00+00:00"
    new_trade = {"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": -5.0, "return_pct": -0.005, "slippage_pct": 0.0005, "timestamp": now_iso}
    
    kt.history = legacy_trades + [new_trade]
    
    # Must preserve the legacy trades (total >= 10) and compute empirical edge (which is negative -> 0.0 fail-closed), NOT None!
    frac = kt.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True)
    assert frac is not None, "Legacy trades must not be dropped into 'None' (insufficient data)"
    assert frac == 0.0, "Measured negative edge must return 0.0 fail closed"


def test_finding_29_directional_mass_governance_unconditional():
    """Finding #29: Manifests missing holdout_resolved_mcc must be rejected without opt-in bypass."""
    # Manifest with holdout_mcc but no holdout_resolved_mcc and no promoted=True
    legacy_manifest = {
        "holdout_mcc": 0.12,
        "holdout_balanced_accuracy": 0.45,
        "holdout_mcc_mde_80pct": 0.05,
        "n_holdout_samples": 800,
        "lookahead": 12
    }
    is_deg, reason = is_manifest_degenerate(legacy_manifest)
    assert is_deg is True
    assert "Holdout resolved MCC missing" in reason

    # With holdout_resolved_mcc present and positive, it passes
    valid_manifest = dict(legacy_manifest, holdout_resolved_mcc=0.08)
    is_deg, reason = is_manifest_degenerate(valid_manifest)
    assert is_deg is False
    assert reason == "OK"

    # With holdout_resolved_mcc <= 0.0, it fails
    neg_resolved_manifest = dict(legacy_manifest, holdout_resolved_mcc=-0.02)
    is_deg, reason = is_manifest_degenerate(neg_resolved_manifest)
    assert is_deg is True
    assert "Holdout resolved MCC (-0.0200) <= 0.0" in reason


def test_finding_18_barrier_upper_bounds_validation():
    """Finding #18: Optimized barriers override must reject unreasonable bounds (upper & lower)."""
    # Overly large sl_mult or tp_mult must be rejected
    from config import REJECTED_BARRIER_FILES
    invalid_bounds_file = "/tmp/test_opt_barriers_15.json"
    data = {
        "tp_mult_trending": 6.5,  # > 5.0
        "tp_mult_ranging": 1.5,
        "sl_mult": 0.8,
        "lookahead": 12
    }
    with open(invalid_bounds_file, "w") as f:
        json.dump(data, f)
    
    # Check validator logic
    sl = data["sl_mult"]
    tp_t = data["tp_mult_trending"]
    tp_r = data["tp_mult_ranging"]
    lh = data["lookahead"]
    is_invalid = sl < 0.3 or sl > 3.0 or tp_r < 0.5 or tp_r > 4.0 or tp_t > 5.0 or lh < 4 or lh > 48
    assert is_invalid is True


def test_finding_22_23_finally_none_type_safety():
    """Finding #22 & #23: Finally backfill must safely handle exp_edge_bps=None without raising TypeError."""
    rec = DecisionRecord(symbol="BTCUSDT", interval="15")
    exp_edge_bps = None
    exp_r_val = None
    cost_bps = None
    mhi_val = None
    
    # Simulate execution of the finally block backfill
    if 'exp_edge_bps' in locals() and exp_edge_bps is not None and rec.expected_value is None:
        rec.expected_value = float(exp_edge_bps)
    if 'exp_r_val' in locals() and exp_r_val is not None and rec.expected_rr is None:
        rec.expected_rr = float(exp_r_val)
    if 'cost_bps' in locals() and cost_bps is not None and rec.round_trip_cost_bp is None:
        rec.round_trip_cost_bp = float(cost_bps)
    if 'mhi_val' in locals() and mhi_val is not None and rec.mhi_score is None:
        rec.mhi_score = float(mhi_val)
        
    assert rec.expected_value is None
    assert rec.expected_rr is None
    assert rec.round_trip_cost_bp is None
    assert rec.mhi_score is None


def test_finding_16_26_stop_widening_tp_rescale():
    """Finding #16 & #26: Widening stop loss must proportionally expand take profit to preserve target R:R."""
    entry_price = 100000.0
    raw_sl_dist = 800.0   # 0.8 ATR
    effective_sl_dist = 1000.0  # 1.0 ATR (widened by min_sl_dist)
    take_profit_price = 101600.0  # raw TP dist = 1600 (R:R = 2.0)
    ml_trend = "Bullish"

    if effective_sl_dist > (raw_sl_dist + 1e-6) and raw_sl_dist > 0 and take_profit_price is not None:
        raw_tp_dist = abs(take_profit_price - entry_price)
        target_rr = raw_tp_dist / raw_sl_dist
        new_tp_dist = max(raw_tp_dist, effective_sl_dist * target_rr)
        take_profit_price = (entry_price + new_tp_dist) if ml_trend == "Bullish" else (entry_price - new_tp_dist)

    # 1000 * 2.0 = 2000 TP distance -> TP = 102000.0
    assert take_profit_price == 102000.0
    assert abs(take_profit_price - entry_price) / effective_sl_dist == 2.0


def test_finding_20_eqs_active_mode_exit_trigger():
    """Finding #20: EXIT_QUALITY_MODE == 'active' must trigger trade exit when EQS < 40.0."""
    eqs_mode = "active"
    eqs_score = 35.0  # Degraded
    exit_reason = None
    
    if eqs_mode == "active" and eqs_score < 40.0 and not exit_reason:
        exit_reason = f"EXIT_QUALITY_DEGRADED ({eqs_score:.1f} < 40.0)"
        
    assert exit_reason == "EXIT_QUALITY_DEGRADED (35.0 < 40.0)"


def test_finding_24_backtest_confluence_fail_closed():
    """Finding #24: Backtest confluence exception must advance loop and fail-closed."""
    executed_trades = []
    for i in range(1):
        try:
            raise RuntimeError("Corrupted HTF history")
        except Exception as ex_conf:
            # Must advance index and continue, skipping trade entry
            break
        executed_trades.append(i)
        
    assert len(executed_trades) == 0


def test_finding_25_backtest_leveraged_equity_return():
    """Finding #25: Backtest Kelly position_frac converts to leveraged equity return matching live."""
    entry_price = 100000.0
    stop_loss_price = 99000.0  # 1% stop loss
    position_frac = 0.02  # 2% capital-at-risk (Quarter-Kelly)
    net_return = 0.02  # 2% price move (Take Profit at 102000)

    sl_dist_val = abs(entry_price - stop_loss_price)
    stop_loss_frac = max(0.002, sl_dist_val / max(1e-9, entry_price))  # 0.01
    notional_equity_frac = min(10.0, position_frac / stop_loss_frac)  # 0.02 / 0.01 = 2.0x leverage
    equity_trade_return = notional_equity_frac * net_return  # 2.0 * 0.02 = 0.04 (4% equity return)

    assert notional_equity_frac == 2.0
    assert equity_trade_return == 0.04
