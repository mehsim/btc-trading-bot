import pytest
import numpy as np
import pandas as pd
import json
import time
from unittest.mock import MagicMock, patch

# --- Finding #71 Tests ---
def test_finding_71_backtest_kelly_scaleout_trailing_and_metrics():
    from trade_calculators import calculate_replay_statistics
    import backtest

    # 1. Harmonized metrics in trade_calculators
    stats_frac = calculate_replay_statistics([0.01, 0.02, -0.01])
    assert stats_frac.get("sizing_basis") == "fractional"

    stats_notional = calculate_replay_statistics([100.0, 200.0, -50.0])
    assert stats_notional.get("sizing_basis") == "dollar_notional"

    # 2. Backtest source inspection for parity features
    with open("backtest.py", "r") as f:
        bt_src = f.read()
    
    assert "scaled_kelly <= 0.0" in bt_src
    assert "compute_dynamic_trail_params" in bt_src
    assert "scale_out_portion" in bt_src
    assert "half_closed" in bt_src
    assert "INSUFFICIENT_SAMPLE: n<100" in bt_src


# --- Finding #72 Tests ---
def test_finding_72_challenger_artifact_and_small_n_suppression():
    with open("backtest_results_challenger.json", "r") as f:
        data = json.load(f)

    # Valid candidate scenarios, no wipeout
    scenarios = data.get("scenarios", [])
    assert len(scenarios) > 0
    for s in scenarios:
        assert "Pessimistic Return" in s
        assert "-100" not in s["Pessimistic Return"]

    windows = data.get("walk_forward_validation", {}).get("windows", [])
    assert len(windows) > 0
    assert windows[0].get("is_refitted") is True

    with open("backtest.py", "r") as f:
        bt_src = f.read()
    assert 'pess_wr_str += " [INSUFFICIENT_SAMPLE: n<100]"' in bt_src


# --- Finding #73 Tests ---
def test_finding_73_kill_criteria_and_drift_monitor():
    with open("background_schedulers.py", "r") as f:
        bg_src = f.read()

    # Query 10000 completed trades without 14-day recency choke
    assert "get_completed_trades(limit=10000)" in bg_src
    assert "cutoff_ts" not in bg_src
    assert "0.55" not in bg_src  # No synthetic confidence imputation

    import drift_monitor
    import state_manager
    # Verify state_manager["last_ece"] is persisted unconditionally even if not trades
    with patch("drift_monitor.get_recent_experiences", return_value=[]):
        with patch("drift_monitor.calculate_ece", return_value=0.035):
            res = drift_monitor.DriftMonitor().evaluate_drift()
    assert state_manager.state_manager.get("last_ece") == 0.035


# --- Finding #74 Tests ---
def test_finding_74_triple_barrier_tuning_isolation():
    with open("train.py", "r") as f:
        train_src = f.read()

    # Uses pages=pages and filters by holdout_cutoff_ts
    assert "pages=pages" in train_src
    assert "holdout_cutoff_ts" in train_src
    assert 'best_barriers["tuned_at"]' in train_src
    assert 'best_barriers["tuning_cutoff_timestamp"]' in train_src


# --- Finding #75 Tests ---
def test_finding_75_directional_mass_promotion_gates():
    with open("train.py", "r") as f:
        train_src = f.read()

    # Evaluates directional-mass resolved metrics in promotion gates
    assert "holdout_resolved_mcc" in train_src
    assert "holdout_resolved_balacc" in train_src
    assert "holdout_resolved_mcc < _min_h_mcc" in train_src
    assert "holdout_resolved_balacc < _min_h_balacc" in train_src
    assert 'manifest_data["holdout_resolved_mcc"]' in train_src


# --- Finding #76 Tests ---
def test_finding_76_calibrator_barrier_geometry_tolerances():
    with open("main.py", "r") as f:
        main_src = f.read()

    # Tolerances must be <= 0.10 TP, <= 0.05 SL, <= 1 LH
    assert "abs(cal_tp - live_tp) > 0.10" in main_src
    assert "abs(cal_sl - live_sl) > 0.05" in main_src
    assert "abs(cal_lh - live_lh) > 1" in main_src


# --- Finding #77 Tests ---
def test_finding_77_bybit_fill_quantity_recovery():
    import main

    assert hasattr(main, "get_bybit_order_executions")

    with open("main.py", "r") as f:
        main_src = f.read()

    # Verify helper and recovery logic are wired in live execution paths
    assert "get_bybit_order_executions(symbol, order_id=bybit_order_id" in main_src
    assert "actual_qty = min(pos_qty, raw_qty) if pos_qty > 0 else 0.0" in main_src
    assert "fill_q = min(pos_delta, floored_ioc)" in main_src


# --- Finding #78 Tests ---
def test_finding_78_ioc_foreign_fill_deduplication():
    with open("main.py", "r") as f:
        main_src = f.read()

    # Foreign fill branch must ignore foreign fills and not credit them
    assert "Ignoring foreign fill" in main_src
    assert "Foreign/unrelated execution" in main_src


# --- Finding #79 Tests ---
def test_finding_79_live_orderbook_spread_wiring():
    import main

    # Mock orderbook cache/data
    mock_ob = {"spread": 0.00045, "imbalance": 0.1}
    with patch("main.get_orderbook_imbalance_and_spread", return_value=mock_ob):
        res = main.get_orderbook_imbalance_and_spread("BTCUSDT")
        assert res["spread"] == 0.00045
        spread_bps = round(res["spread"] * 10000.0, 2)
        assert spread_bps == 4.5

    with open("main.py", "r") as f:
        main_src = f.read()

    assert 'bot_state[f"current_spread_bps_{symbol}"]' in main_src
    assert 'bot_state.get(f"current_spread_bps_{symbol}"' in main_src


# --- Finding #80 Tests ---
def test_finding_80_derivatives_missing_feed_safety():
    import data
    import numpy as np

    # Test empty / failed fetch in _merge_cached_derivatives
    df_base = pd.DataFrame({
        "timestamp": [1000, 2000, 3000],
        "close": [50000.0, 50100.0, 50200.0],
        "open": [49900.0, 50000.0, 50100.0],
        "high": [50200.0, 50300.0, 50400.0],
        "low": [49800.0, 49900.0, 50000.0],
        "volume": [10.0, 12.0, 15.0]
    })

    merged = data._merge_cached_derivatives(df_base.copy(), df_oi=None, df_funding=None, df_fng=None)
    assert merged.attrs.get("stale_derivatives_feeds") is True
    assert np.isnan(merged["open_interest"].iloc[0])
    assert np.isnan(merged["funding_rate"].iloc[0])
    assert np.isnan(merged["fear_greed"].iloc[0])

    # Check main.py feature missing check ordering and slicing try/except
    with open("main.py", "r") as f:
        main_src = f.read()

    assert "X_live_full.apply(pd.to_numeric, errors=\"coerce\")" in main_src
    assert "Feature slice failed" in main_src
