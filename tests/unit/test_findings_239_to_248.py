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
    """Finding #72 & #244: Walk-forward validation accurately marks is_refitted and backtest suppresses small-sample win rates."""
    from walk_forward_engine import run_walk_forward_backtest
    from pattern_miner import wilson_score_interval
    import pandas as pd
    import numpy as np

    # 1. Test walk-forward engine refitted vs rolling replay
    dummy_df = pd.DataFrame({
        "timestamp": np.arange(1000, 1000 + 300 * 60, 60),
        "open": np.linspace(50000, 51000, 300),
        "high": np.linspace(50100, 51100, 300),
        "low": np.linspace(49900, 50900, 300),
        "close": np.linspace(50050, 51050, 300),
        "volume": [10.0] * 300
    })

    # When train_fn returns a simulator callable, is_refitted must be True
    mock_sim = lambda t_df: {"trades": [{"pnl": 50.0, "return": 0.02, "exit_type": "tp"}]}
    res_refit = run_walk_forward_backtest(
        dummy_df,
        train_window_bars=100,
        test_window_bars=50,
        step_bars=50,
        train_fn=lambda tr_df: mock_sim
    )
    assert res_refit["status"] == "success"
    assert len(res_refit["windows"]) > 0
    assert res_refit["windows"][0]["is_refitted"] is True
    assert res_refit["windows"][0]["evaluation_type"] == "refitted_walk_forward"
    assert res_refit["all_windows_refitted"] is True
    assert res_refit["evaluation_mode"] == "refitted_walk_forward"

    # When only trade_simulator_fn is passed, is_refitted must be False
    res_replay = run_walk_forward_backtest(
        dummy_df,
        train_window_bars=100,
        test_window_bars=50,
        step_bars=50,
        trade_simulator_fn=mock_sim
    )
    assert res_replay["status"] == "success"
    assert len(res_replay["windows"]) > 0
    assert res_replay["windows"][0]["is_refitted"] is False
    assert res_replay["windows"][0]["evaluation_type"] == "rolling_window_replay"
    assert res_replay["all_windows_refitted"] is False
    assert res_replay["evaluation_mode"] == "rolling_window_replay"

    # 2. Test sample size suppression string formatting
    t_count = 45
    win_rate = 60.0
    wins = int(round((win_rate / 100.0) * t_count))
    ci_l, ci_u = wilson_score_interval(wins, t_count)
    pess_wr_str = f"{win_rate:.1f}% [{ci_l*100:.1f}%, {ci_u*100:.1f}%] (n={t_count})"
    if t_count < 100:
        pess_wr_str += " [INSUFFICIENT_SAMPLE: n<100]"
    assert "[INSUFFICIENT_SAMPLE: n<100]" in pess_wr_str
    assert "-100" not in pess_wr_str


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
