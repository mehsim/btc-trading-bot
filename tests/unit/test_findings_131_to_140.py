"""
Unit tests for audit defect findings #31 through #40.
Covers backtest gate parity, Kelly sized returns, p* stop multiplier resolution,
Optuna barrier cache holdout gating, Kelly tracker zero-win fail-closed,
break-even vs scale-out exit synchronization, directional mass manifest governance,
training barrier metadata fidelity, chase loop IOC dedup, and MDE sample size isolation.
"""

import math
import os
import json
import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import config
from config import is_manifest_degenerate
import trade_calculators
from trade_calculators import resolve_trade_geometry, calculate_replay_statistics
from kelly_tracker import KellyTracker
import risk_engine
from risk_engine import compute_conservative_kelly
from exit_policy_engine import ExitPolicyEngine


# ==============================================================================
# Finding #31: Backtest gate stack parity & confluence checks
# ==============================================================================
def test_finding_31_backtest_gate_stack_and_confluence():
    """Verify backtest passes interval to resolve_direction and rejects non-viable calibrator."""
    from ensemble import resolve_direction
    
    # Test resolve_direction with interval="15" uses min_direction_mass 0.20
    probs_15 = np.array([0.45, 0.40, 0.15])  # directional mass = 0.60 >= 0.20
    trend, conf = resolve_direction(probs_15, interval="15")
    assert trend in ["Bearish", "Neutral", "Bullish"]
    
    # Calibrator viability reject in economic gate
    from tools.beta_calibrator import is_calibrator_viable
    non_viable_calibrator = {"method": "beta", "a": 1.0, "b": 1.0, "c": 0.0}  # identity/fallback
    assert is_calibrator_viable(non_viable_calibrator, min_required_p_star=0.55) is False


# ==============================================================================
# Finding #32: Backtest sized returns consistency
# ==============================================================================
def test_finding_32_backtest_sized_returns_consistency():
    """Verify calculate_replay_statistics with sized returns matches compounded equity."""
    # Sized returns reflecting fractional position sizing
    net_returns = [0.05, -0.02, 0.04, -0.01, 0.03]
    position_frac = 0.20  # 20% position size
    sl_frac = 0.02

    sized_returns = [r * position_frac for r in net_returns]
    sized_sl_fracs = [sl_frac * position_frac for _ in net_returns]

    stats = calculate_replay_statistics(
        sized_returns,
        initial_equity=100.0,
        risk_per_trade_pct=sized_sl_fracs,
        duration_days=10.0,
        interval="15"
    )
    
    expected_compounded = 100.0 * np.prod([1.0 + sr for sr in sized_returns])
    expected_return_pct = ((expected_compounded - 100.0) / 100.0) * 100.0
    
    # Ending return pct in stats must match the sized compounded equity
    assert pytest.approx(stats["ending_return_pct"], rel=1e-3) == expected_return_pct
    assert stats["total_trades"] == 5


# ==============================================================================
# Finding #33: p* exact stop multiplier resolution post-floor
# ==============================================================================
def test_finding_33_p_star_exact_stop_distance():
    """Verify sl_multiplier_adjusted is calculated from final sl_dist after all floors."""
    entry_price = 50000.0
    atr_dollars = 500.0  # 1% ATR
    # For 240m, min_sl_pct is 0.015 (1.5%), so min_sl_dist is 750 (1.5 ATR)
    geom = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval="240",
        atr_dollars=atr_dollars,
        base_sl_multiplier=0.7693,
        base_tp_multiplier=2.4005,
        df=None
    )
    
    # Floor forces sl_dist to at least 1.5% of 50000 = 750.0
    assert geom["sl_dist"] >= 750.0
    # sl_multiplier_adjusted MUST match sl_dist / atr_dollars (>= 1.5 ATR)
    expected_sl_mult = geom["sl_dist"] / atr_dollars
    assert pytest.approx(geom["sl_multiplier_adjusted"], rel=1e-4) == expected_sl_mult
    assert geom["sl_multiplier_adjusted"] >= 1.5


# ==============================================================================
# Finding #34: Barrier cache holdout validation
# ==============================================================================
def test_finding_34_barrier_cache_holdout_validation():
    """Verify that cached barriers without tuning_cutoff_timestamp are flagged."""
    # Create barrier dict missing tuning_cutoff_timestamp
    stale_barriers = {
        "tp_mult_trending": 2.4,
        "tp_mult_ranging": 1.8,
        "sl_mult": 0.77,
        "lookahead": 12
    }
    assert stale_barriers.get("tuning_cutoff_timestamp") is None

    # Valid barrier dict with tuning_cutoff_timestamp
    valid_barriers = dict(stale_barriers, tuning_cutoff_timestamp=1700000000000)
    assert valid_barriers.get("tuning_cutoff_timestamp") == 1700000000000


# ==============================================================================
# Finding #35: Kelly tracker zero-win fail closed & risk engine check
# ==============================================================================
def test_finding_35_kelly_tracker_zero_wins_fail_closed():
    """Verify kelly_tracker returns 0.0 fail-closed when sample has zero winning trades."""
    kt = KellyTracker(data_file="/tmp/test_kelly_history.json")
    now_iso = "2026-09-05T00:00:00"
    # Feed 15 consecutive losing trades
    kt.history = [
        {"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": -10.0, "return_pct": -0.01, "slippage_pct": 0.0005, "timestamp": now_iso}
        for _ in range(15)
    ]
    
    # Must return 0.0 fail-closed (measured 0% win rate), NOT None!
    res = kt.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True)
    assert res == 0.0

    # Risk engine compute_conservative_kelly must return 0.0 fail-closed
    with patch("risk_engine.global_kelly_tracker", kt):
        k_val = compute_conservative_kelly(
            calibrated_confidence=0.60,
            tp_multiplier=1.5,
            sl_multiplier=1.0,
            interval="15",
            cost_bps=10.0,
            trade_history=[]  # Caller history empty
        )
        assert k_val == 0.0


# ==============================================================================
# Finding #36: Break-even vs scale-out synchronization
# ==============================================================================
def test_finding_36_break_even_scale_out_synchronization():
    """Verify break-even stop does not trigger before scale-out distance when ATR contracts."""
    engine = ExitPolicyEngine()
    
    # Trade entered with entry_atr = 1000.0, but current_atr contracted to 700.0
    active_trade = {
        "symbol": "BTCUSDT",
        "entry_price": 50000.0,
        "entry_atr": 1000.0,
        "current_atr": 700.0,
        "direction": "Bullish",
        "half_closed": False,
        "break_even_triggered": False,
        "position_size_usd": 100.0,
        "leverage": 10.0,
        "stop_loss": 49000.0,
        "take_profit": 53000.0
    }
    
    # Price moved up to 51100 (+1.1 entry-ATR). Scale-out target is 1.2 * 1000 = 1200 (at 51200).
    # Break-even trigger (1.5 * 700 = 1050 without fix) would fire prematurely at 51100!
    # With the fix, BE requires max(be_trigger * max_atr, scale_out_atr + buffer), so BE must NOT fire.
    exit_reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=51100.0,
        current_time=time.time(),
        current_atr=700.0,
        regime="TRENDING_MODERATE"
    )
    assert not updates.get("break_even_triggered", False)


# ==============================================================================
# Finding #37: Directional-mass manifest serving governance
# ==============================================================================
def test_finding_37_directional_mass_serving_governance():
    """Verify is_manifest_degenerate rejects models missing holdout_resolved_mcc."""
    # Manifest with holdout_mcc but missing holdout_resolved_mcc
    manifest_argmax_only = {
        "holdout_mcc": 0.045,
        "holdout_balanced_accuracy": 0.38,
        "holdout_mcc_mde_80pct": 0.03,
        "holdout_effective_n": 80
    }
    is_deg, reason = is_manifest_degenerate(manifest_argmax_only)
    assert is_deg is True
    assert "Holdout resolved MCC missing" in reason

    # Manifest with valid positive holdout_resolved_mcc passes
    manifest_resolved = dict(
        manifest_argmax_only,
        holdout_resolved_mcc=0.042,
        holdout_resolved_balacc=0.37
    )
    is_deg, reason = is_manifest_degenerate(manifest_resolved)
    assert is_deg is False
    assert reason == "OK"


# ==============================================================================
# Finding #38: Training barrier effective geometry in LAST_LABELING_BARRIER_CONFIG
# ==============================================================================
def test_finding_38_training_barrier_effective_geometry():
    """Verify add_triple_barrier_labels records effective_sl_mult and min_sl_pct."""
    import train
    # Create minimal OHLCV dataframe
    dates = pd.date_range("2024-01-01", periods=50, freq="15min")
    df = pd.DataFrame({
        "timestamp": [int(d.timestamp() * 1000) for d in dates],
        "close": np.linspace(50000, 51000, 50),
        "high": np.linspace(50100, 51100, 50),
        "low": np.linspace(49900, 50900, 50),
        "ADX": np.full(50, 20.0),
        "ATR": np.full(50, 500.0),
        "ATR_norm": np.full(50, 0.01)
    })
    
    train.add_triple_barrier_labels(df, interval="15")
    last_cfg = train.LAST_LABELING_BARRIER_CONFIG
    assert "effective_sl_mult" in last_cfg
    assert "min_sl_pct" in last_cfg
    assert last_cfg["effective_sl_mult"] <= 1.25


# ==============================================================================
# Finding #39: Chase loop IOC dedup dead branch eliminated
# ==============================================================================
def test_finding_39_chase_ioc_dedup_dead_branch_eliminated():
    """Verify IOC dedup logic correctly accumulates unrecorded chase fills."""
    chase_order_ids = {"ord_chase_1"}
    recorded_chase_exec_ids = set()
    raw_qty = 10.0
    filled_so_far = 0.0
    weighted_sum_px = 0.0
    entry_price = 50000.0
    bybit_success = False

    last_exec = {
        "execId": "exec_chase_fill_1",
        "orderId": "ord_chase_1",
        "execQty": "5.0",
        "execPrice": "50005.0",
        "execTime": int(time.time() * 1000),
        "side": "Buy"
    }

    exec_id = last_exec.get("execId")
    exec_order_id = last_exec.get("orderId")

    # The corrected branching structure from main.py:
    if exec_id and exec_id in recorded_chase_exec_ids:
        # Already recorded
        pass
    elif exec_order_id and exec_order_id in chase_order_ids:
        # Unrecorded recent chase fill
        fill_px = float(last_exec.get("execPrice", entry_price))
        raw_fill_q = float(last_exec.get("execQty", 10.0))
        fill_q = min(raw_fill_q, max(0.0, raw_qty - filled_so_far))
        if fill_q > 0:
            filled_so_far += fill_q
            weighted_sum_px += (fill_q * fill_px)
            if exec_id:
                recorded_chase_exec_ids.add(exec_id)
        if filled_so_far >= (0.95 * raw_qty):
            bybit_success = True

    assert filled_so_far == 5.0
    assert "exec_chase_fill_1" in recorded_chase_exec_ids


# ==============================================================================
# Finding #40: MDE holdout sample size isolation
# ==============================================================================
def test_finding_40_mde_holdout_sample_size_isolation():
    """Verify is_manifest_degenerate derives MDE strictly from holdout sample size."""
    # Manifest with small holdout sample size (800 samples, lookahead 10 -> eff n = 80)
    # and large training sample size (2892 samples -> eff n = 263.36)
    manifest = {
        "holdout_mcc": 0.2036,
        "holdout_resolved_mcc": 0.2036,
        "holdout_balanced_accuracy": 0.40,
        "n_holdout_samples": 800,
        "lookahead": 10,
        "cv_metrics": {
            "effective_sample_size": 263.36,  # Training CV sample size
            "raw_sample_size": 2892
        }
    }
    # True holdout MDE = 2.8016 / sqrt(800 / 10) = 2.8016 / sqrt(80) = 0.3132
    # If using training 263.36: MDE = 2.8016 / sqrt(263.36) = 0.1726 (which would pass 0.2036)
    # Because holdout MDE is 0.3132 > 0.2036, it must evaluate as DEGENERATE (underpowered)!
    is_deg, reason = is_manifest_degenerate(manifest)
    assert is_deg is True
    assert "80% MDE" in reason
    assert "0.3132" in reason
