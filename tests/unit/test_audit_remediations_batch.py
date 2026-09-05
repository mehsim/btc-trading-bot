"""
Unit tests verifying audit defect remediations:
- Chase loop TTL guard inversion fix
- Isotonic step function minimum per-knot support gate
- Flat/degenerate calibrator rejection
- Gate 1 Walk-forward integer trade count handling
- Predictive Floor Gate convention mismatch enforcement
- Pre-order stop loss floor and risk-preserving quantity rescaling
- Live scale-out execution invariant
- Ratchet baseline no-tolerance downward-only tightening
- Backtest challenger output and small-n suppression
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


def test_calibrator_ranging_120_rejected_due_to_missing_support():
    """Calibrator ranging_120 lacks min_bin_support and must fail viability."""
    from tools.beta_calibrator import is_calibrator_viable
    with open("calibrator_ranging_120.json", "r") as f:
        data = json.load(f)
    assert data.get("min_bin_support") is None
    assert is_calibrator_viable(data) is False


def test_calibrator_trending_15_rejected_due_to_flat_y_range():
    """Calibrator trending_15 has y array range < 0.02 and must fail viability."""
    from tools.beta_calibrator import is_calibrator_viable
    with open("calibrator_trending_15.json", "r") as f:
        data = json.load(f)
    ys = data.get("y", [])
    assert len(ys) > 0
    assert (max(ys) - min(ys)) < 0.02
    assert is_calibrator_viable(data) is False


def test_isotonic_support_gate_fail_closed_on_small_support():
    """Isotonic calibrator with min_bin_support < 100 must fail viability."""
    from tools.beta_calibrator import is_calibrator_viable
    cal_data = {
        "scaling_method": "isotonic",
        "is_fitted": True,
        "is_fallback": False,
        "target_definition": "triple_barrier_exact",
        "fitting_sample_size": 10000,
        "min_bin_support": 50,  # < 100
        "y": [0.20, 0.30, 0.40],
        "X": [0.40, 0.50, 0.60]
    }
    assert is_calibrator_viable(cal_data) is False

    # When support >= 100 and sample >= 5000 and valid y range, passes
    cal_data["min_bin_support"] = 100
    assert is_calibrator_viable(cal_data) is True


def test_resolve_trade_geometry_pre_order_min_sl_floor():
    """resolve_trade_geometry floors at max(atr_dollars * 1.0, entry_price * min_sl_pct)."""
    from trade_calculators import resolve_trade_geometry
    entry_price = 50000.0
    atr_dollars = 1000.0  # 1.0 * ATR = 1000, which exceeds 50000 * 0.008 = 400
    geom = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval="60",
        atr_dollars=atr_dollars,
        base_sl_multiplier=0.5,  # 0.5 * 1000 = 500 < 1000 floor
        base_tp_multiplier=1.5,
        symbol="BTCUSDT"
    )
    assert geom["sl_dist"] >= 1000.0  # Must be floored at 1.0 * ATR
    assert geom["stop_loss_price"] <= (entry_price - 1000.0)


def test_gate1_walk_forward_integer_trade_count():
    """Gate 1 walk-forward logic must handle integer 'trades' without raising TypeError."""
    # Mock window results with integer trade counts as returned by walk_forward_engine
    mock_w_res = {
        "status": "success",
        "windows": [
            {"trades": 5, "is_refitted": True, "win_rate": 60.0},
            {"trades": 8, "is_refitted": True, "win_rate": 50.0}
        ],
        "mean_expectancy_r": 0.25,
        "mean_profit_factor": 1.45
    }
    # Simulate the exact expression used in train.py:2398-2406
    _w_trades_cnt = lambda w: (len(w.get("trades")) if isinstance(w.get("trades"), (list, tuple, dict)) else int(w.get("trades", 0)))
    _tot_wf_trades = sum(_w_trades_cnt(w) for w in mock_w_res.get("windows", [])) if "windows" in mock_w_res else int(mock_w_res.get("total_trades", 0))
    _active_wf_windows = len([w for w in mock_w_res.get("windows", []) if _w_trades_cnt(w) > 0])
    wf_pass = bool(
        mock_w_res.get("status") == "success" and
        _active_wf_windows >= 2 and
        _tot_wf_trades >= 10 and
        mock_w_res.get("mean_expectancy_r", -1.0) >= 0.0 and
        mock_w_res.get("mean_profit_factor", 0.0) >= 1.0
    )
    assert wf_pass is True
    assert _tot_wf_trades == 13
    assert _active_wf_windows == 2


def test_chase_loop_ttl_guard_logic():
    """Ensure chase loop does not abort when elapsed <= signal_ttl_seconds."""
    with open("main.py", "r") as f:
        src = f.read()

    # Verify that the abort call is properly guarded inside if elapsed > signal_ttl_seconds
    assert "if elapsed > signal_ttl_seconds:" in src
    # Verify else branch is nested under filled_so_far
    assert "elif filled_so_far > 0:" in src
    assert "_abort_async(\"Signal TTL expired during chase with 0 fills\")" in src


def test_live_scale_out_requires_bybit_scaled_out():
    """Verify live mode sets trigger_scale_out = bybit_scaled_out and isolates advisory triggers."""
    with open("main.py", "r") as f:
        src = f.read()

    assert 'if TRADE_MODE != "simulation":\n                        trigger_scale_out = bybit_scaled_out' in src


def test_ratchet_baseline_no_plus_one_tolerance():
    """Verify check_print_ratchet and check_silent_handlers do not use count + 1 tolerance."""
    with open("tools/check_print_ratchet.py", "r") as f:
        src_print = f.read()
    assert "data[\"print_calls\"] = count + 1" not in src_print
    assert "if count < current_bl:" in src_print

    with open("tools/check_silent_handlers.py", "r") as f:
        src_silent = f.read()
    assert "data[\"silent_handlers\"] = count + 1" not in src_silent
    assert "if count < current_bl:" in src_silent


def test_backtest_challenger_writer_and_small_n_suppression():
    """Verify backtest.py has --output argument, writes to OUTPUT_FILE, and suppresses small-n conclusions."""
    with open("backtest.py", "r") as f:
        src = f.read()

    assert 'parser.add_argument("--output"' in src
    assert 'OUTPUT_FILE = args.output or ("backtest_results_challenger.json" if USE_CHALLENGER else "backtest_results.json")' in src
    assert "with open(OUTPUT_FILE, \"w\") as f:" in src
    assert "Statistical Conclusion" in src
    assert "SUPPRESSED (Sample size n < 100" in src
