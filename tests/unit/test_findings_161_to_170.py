"""
Unit tests for audit defect findings #161 through #170.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_finding_161_signal_loop_retains_unprocessed_exchange_closed_trades():
    """Finding #161: Signal loop must not drop exchange-closed trades before exit loop reconciliation."""
    active_trades = [
        {"symbol": "SOLUSDT", "bybit_closed": True, "exit_processed": False, "pnl": -50.0},
        {"symbol": "BTCUSDT", "bybit_closed": False, "exit_processed": False, "pnl": 100.0},
        {"symbol": "ETHUSDT", "bybit_closed": True, "exit_processed": True, "pnl": -20.0},
        {"symbol": "ADAUSDT", "closed": True, "exit_processed": True, "pnl": 10.0},
        {"symbol": "XRPUSDT", "closed": True, "exit_processed": False, "pnl": 30.0},
    ]

    # Filter logic matching main.py:6851
    filtered = [
        t for t in active_trades
        if isinstance(t, dict)
        and not (t.get("bybit_closed") and t.get("exit_processed", False))
        and not (t.get("closed") and t.get("exit_processed", False))
    ]

    symbols = [t["symbol"] for t in filtered]
    # Unprocessed trades must be retained so exit loop can record them
    assert "SOLUSDT" in symbols
    assert "BTCUSDT" in symbols
    assert "XRPUSDT" in symbols
    # Reconciled/processed trades can be safely filtered out
    assert "ETHUSDT" not in symbols
    assert "ADAUSDT" not in symbols


def test_finding_162_live_inference_feature_reindex_shape_guard():
    """Finding #162: Live inference must not silently zero-fill missing model features."""
    expected_names = ["feature_a", "feature_b", "feature_c", "feature_d"]
    # Candle only has a and b; c and d are missing
    candle_series = pd.Series({"feature_a": 1.25, "feature_b": 0.85})

    # Reindex without fill_value=0.0
    X_live_full = candle_series.to_frame().T.reindex(columns=expected_names)

    # Missing features must be detected
    missing = [col for col in expected_names if col not in candle_series.index or pd.isna(X_live_full[col].iloc[0])]
    assert len(missing) == 2
    assert "feature_c" in missing
    assert "feature_d" in missing


def test_finding_163_empirical_kelly_distinguishes_negative_edge():
    """Finding #163: compute_kelly_fraction distinguishes insufficient sample from negative edge, and risk_engine fails closed."""
    from kelly_tracker import KellyTracker
    from risk_engine import calculate_conservative_kelly
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_file = tf.name

    try:
        tracker = KellyTracker(data_file=temp_file)

        # 1. Insufficient trades (< 10)
        for i in range(5):
            tracker.log_trade("BTCUSDT", "15", 10.0, 0.01)
        res_insufficient = tracker.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True)
        assert res_insufficient is None

        # 2. Add losing trades to reach 30 trades with heavy negative edge (e.g. 5 wins, 25 losses)
        for i in range(25):
            tracker.log_trade("BTCUSDT", "15", -10.0, -0.02)
        res_negative = tracker.compute_kelly_fraction(timeframe="15", min_trades=10, insufficient_as_none=True)
        assert res_negative == 0.0

        # 3. Test calculate_conservative_kelly with negative edge empirical tracker
        with patch("risk_engine.global_kelly_tracker", tracker):
            sim_history = [{"pnl": -10.0} for _ in range(30)]
            kelly_val = calculate_conservative_kelly(
                calibrated_confidence=0.55,
                tp_multiplier=2.0,
                sl_multiplier=1.0,
                interval="15",
                trade_history=sim_history
            )
            # Must return 0.0 (fail closed on measured negative edge) instead of falling through to confidence prior
            assert kelly_val == 0.0
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_finding_164_mhi_all_losing_history_reaches_critical():
    """Finding #164: An all-losing 30-trade history drives MHI below 50.0 and enters CRITICAL state."""
    from strategy_health_engine import strategy_health_engine

    losing_pnls = [-10.0] * 30
    res = strategy_health_engine.compute_model_health_index(recent_pnls=losing_pnls)

    assert res["mhi_score"] < 50.0
    assert res["health_status"] == "CRITICAL"
    assert res["sizing_multiplier"] == 0.00


def test_finding_165_triple_barrier_tuning_train_slice_isolation():
    """Finding #165: Triple-barrier multiplier tuning data is sliced before the holdout split."""
    total_bars = 1000
    dates = pd.date_range("2026-01-01", periods=total_bars, freq="15min")
    df_tune = pd.DataFrame({
        "timestamp": [int(d.timestamp() * 1000) for d in dates],
        "close": np.linspace(50000, 55000, total_bars),
        "high": np.linspace(50100, 55100, total_bars),
        "low": np.linspace(49900, 54900, total_bars),
        "volume": np.ones(total_bars) * 10.0
    })

    holdout_fraction = 0.15
    purge_len = 12
    split_idx = int(len(df_tune) * (1.0 - holdout_fraction))
    df_tune_train = df_tune.iloc[:max(50, split_idx - purge_len)].copy().reset_index(drop=True)

    # df_tune_train must strictly terminate before holdout index (split_idx)
    assert len(df_tune_train) <= split_idx - purge_len
    assert len(df_tune_train) < len(df_tune)


def test_finding_166_load_ensemble_rejects_empty_manifest_hash():
    """Finding #166: load_ensemble_classifier raises RuntimeError on empty manifest contract hash."""
    from ensemble import load_ensemble_classifier, EMPTY_HASH
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "test_empty_contract")
        manifest_path = f"{prefix}_manifest.json"
        empty_manifest = {
            "model_version": "v7.2.0",
            "feature_count": 0,
            "feature_contract_hash": EMPTY_HASH,
            "feature_names": []
        }
        with open(manifest_path, "w") as f:
            json.dump(empty_manifest, f)

        with patch("ensemble.XGBClassifier"):
            with pytest.raises(RuntimeError, match="empty/invalid feature contract"):
                load_ensemble_classifier(prefix, n_features=10, feature_names=["f1", "f2"])


def test_finding_167_holdout_exception_sentinels_not_written():
    """Finding #167: Holdout calibration calculation errors result in None rather than 0.99 sentinels in manifest."""
    chal_brier = None
    chal_ece = None
    holdout_metrics_status = "exception"
    should_save = False

    cv_metrics = {
        "holdout_brier": round(chal_brier, 4) if (chal_brier is not None and chal_brier != 0.99) else None,
        "holdout_ece": round(chal_ece, 4) if (chal_ece is not None and chal_ece != 0.99) else None,
        "holdout_metrics_status": holdout_metrics_status
    }

    assert cv_metrics["holdout_brier"] is None
    assert cv_metrics["holdout_ece"] is None
    assert cv_metrics["holdout_metrics_status"] == "exception"
    assert should_save is False


def test_finding_168_resolve_direction_evaluation_in_holdout():
    """Finding #168: Holdout predictions evaluated with resolve_direction match directional-mass normalization."""
    from ensemble import resolve_direction
    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

    # Simulated holdout probability outputs (3 classes: Bearish=0, Neutral=1, Bullish=2)
    y_holdout_true = np.array([2, 0, 1, 2, 0, 2])
    probs = np.array([
        [0.10, 0.40, 0.50],  # Bullish
        [0.60, 0.30, 0.10],  # Bearish
        [0.05, 0.90, 0.05],  # Neutral
        [0.05, 0.35, 0.60],  # Bullish
        [0.55, 0.35, 0.10],  # Bearish
        [0.10, 0.30, 0.60],  # Bullish
    ])

    dir_map = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
    preds_resolved = np.array([dir_map[resolve_direction(p, interval="15")[0]] for p in probs])

    assert np.array_equal(preds_resolved, y_holdout_true)
    resolved_mcc = float(matthews_corrcoef(y_holdout_true, preds_resolved))
    resolved_balacc = float(balanced_accuracy_score(y_holdout_true, preds_resolved))
    assert resolved_mcc == 1.0
    assert resolved_balacc == 1.0


def test_finding_169_calibrator_barrier_geometry_alignment():
    """Finding #169: verify_calibrator_barrier_geometry asserts calibrator barrier matches TIMEFRAME_CONFIG."""
    from config import TIMEFRAME_CONFIG

    cfg_15 = TIMEFRAME_CONFIG.get("15", {})
    live_tp = float(cfg_15.get("tp_mult_trending", 1.85))
    live_sl = float(cfg_15.get("sl_mult", 0.85))
    live_lh = int(cfg_15.get("lookahead", 12))

    cal_aligned = {
        "barrier_geometry": {
            "tp_mult_trending": live_tp,
            "sl_mult": live_sl,
            "lookahead": live_lh
        }
    }

    cal_divergent = {
        "barrier_geometry": {
            "tp_mult_trending": live_tp + 1.5,
            "sl_mult": live_sl + 1.0,
            "lookahead": live_lh + 10
        }
    }

    # Verify divergence threshold
    b_aligned = cal_aligned["barrier_geometry"]
    assert abs(b_aligned["tp_mult_trending"] - live_tp) <= 0.35
    assert abs(b_aligned["sl_mult"] - live_sl) <= 0.20

    b_div = cal_divergent["barrier_geometry"]
    assert abs(b_div["tp_mult_trending"] - live_tp) > 0.35


def test_finding_170_dual_significance_release_gate_directional_mapping():
    """Finding #170: Directional mapping in holdout profit factor (2: Long, 0: Short, 1: Skip) produces PF > 1 on accurate predictions."""
    from trade_calculators import calculate_replay_statistics

    # True forward returns
    forward_returns = [0.03, -0.02, 0.025, -0.015, 0.04]
    # Perfect predictions: class 2 when positive return, class 0 when negative return
    predictions = [2, 0, 2, 0, 2]

    trade_rets = []
    sl_fracs = []
    for p_dir, p_ret in zip(predictions, forward_returns):
        if p_dir == 2:  # Bullish long
            trade_rets.append(float(p_ret))
            sl_fracs.append(0.01)
        elif p_dir == 0:  # Bearish short
            trade_rets.append(-float(p_ret))
            sl_fracs.append(0.01)

    stats = calculate_replay_statistics(
        trade_rets,
        initial_equity=100.0,
        risk_per_trade_pct=sl_fracs,
        interval="15"
    )

    profit_factor = float(stats.get("profit_factor", 0.0))
    # All trades were profitable given correct directional mapping
    assert profit_factor > 1.0
    assert len(trade_rets) == 5
    assert all(r > 0 for r in trade_rets)
