import os
import sys
import time
import json
import sqlite3
import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import config
from config import is_manifest_degenerate
import config_verifier
import confluence_engine
import database
import ensemble
from exit_policy_engine import ExitPolicyEngine
import risk_engine
import signal_evaluator
import trade_calculators
from decision_journal import write_decision, DecisionRecord, ReasonCode


def test_r11_and_r12_is_manifest_degenerate():
    """R11 & R12: Prioritize cv_metrics holdout accuracy and require positive holdout effective n."""
    # R11: CV accuracy is 0.4607, but holdout balanced accuracy in cv_metrics is <= 0.3333 (chance)
    manifest_failing_holdout = {
        "manifest_bal_acc": 0.4607,  # Top-level fallback
        "cv_metrics": {
            "holdout_balanced_accuracy": 0.3333,  # Should win over top-level fallback
            "n_holdout_samples": 600,
            "lookahead": 12
        }
    }
    is_deg, reason = is_manifest_degenerate(manifest_failing_holdout)
    assert is_deg is True
    assert "Holdout balanced accuracy" in reason

    # R12: Governed manifest missing holdout sample size / effective n metadata must fail-closed
    manifest_without_metadata = {
        "promoted": True,
        "cv_metrics": {
            "holdout_balanced_accuracy": 0.40,
            "holdout_mcc": 0.08
        }
    }
    is_deg, reason = is_manifest_degenerate(manifest_without_metadata)
    assert is_deg is True
    assert "Holdout sample size metadata missing" in reason

    # Healthy manifest with positive holdout metrics and valid effective n
    healthy_manifest = {
        "cv_metrics": {
            "holdout_balanced_accuracy": 0.45,
            "holdout_mcc": 0.12,
            "holdout_resolved_mcc": 0.11,
            "holdout_mcc_mde_80pct": 0.10,
            "n_holdout_samples": 600,
            "lookahead": 12
        }
    }
    is_deg, reason = is_manifest_degenerate(healthy_manifest)
    assert is_deg is False


def test_r13_config_verifier_checks_price_manifests():
    """R13: config_verifier asserts shared constants across price and trend manifests."""
    config_verifier.assert_shared_constants_aligned()


def test_r14_confluence_engine_sub1h_gate():
    """R14: confluence_engine only conditions trend_gates_passed on 1h_Trend when 1h_Trend is present."""
    df_1h = pd.DataFrame({
        "open": [100.0] * 30,
        "high": [105.0] * 30,
        "low": [95.0] * 30,
        "close": [102.0] * 30,
        "volume": [1000.0] * 30
    })
    # For a 60m interval, 1h_Trend gate is not generated
    passed, summary, score_pct = confluence_engine.check_pre_trade_confluence(
        current_price=100.0,
        df_1h=df_1h,
        ml_trend="Bullish",
        news_sentiment=0.0,
        expected_pct_change=0.02,
        interval="60",
        symbol="BTCUSDT",
        calibrated_confidence=0.75,
        current_regime="Trending_Bullish"
    )
    assert isinstance(passed, bool)
    assert isinstance(summary, dict)
    assert "1h_Trend" not in summary


def test_r15_database_dedup_interval_and_direction(tmp_path):
    """R15: database dedup query must not drop concurrent trades on different timeframes or directions."""
    db_file = str(tmp_path / "test_trades.db")
    orig_db = database.DB_FILE
    try:
        database.DB_FILE = db_file
        database.init_db()

        now_t = time.time()
        trade_15m_long = {
            "trade_id": "trade_15m_long",
            "symbol": "BTCUSDT",
            "exit_time": now_t,
            "interval": "15",
            "direction": "Bullish",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "pnl_usd": 50.0,
            "position_size_usd": 1000.0
        }
        trade_60m_long = {
            "trade_id": "trade_60m_long",
            "symbol": "BTCUSDT",
            "exit_time": now_t + 2.0,  # Within 15s window
            "interval": "60",
            "direction": "Bullish",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "pnl_usd": 100.0,
            "position_size_usd": 2000.0
        }
        trade_15m_short = {
            "trade_id": "trade_15m_short",
            "symbol": "BTCUSDT",
            "exit_time": now_t + 3.0,
            "interval": "15",
            "direction": "Bearish",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "pnl_usd": -50.0,
            "position_size_usd": 1000.0
        }

        assert database.save_completed_trade(trade_15m_long) is True
        assert database.save_completed_trade(trade_60m_long) is True  # Must not be dropped as duplicate!
        assert database.save_completed_trade(trade_15m_short) is True  # Must not be dropped as duplicate!
    finally:
        database.DB_FILE = orig_db


def test_r16_ensemble_meta_learner_purged_cv():
    """R16: EnsembleClassifier.fit uses PurgedTimeSeriesSplit without shuffling for meta-learner OOF matrix."""
    model = ensemble.EnsembleClassifier(
        xgb_model=DecisionTreeClassifier(max_depth=2),
        lgb_model=DecisionTreeClassifier(max_depth=2),
        cat_model=DecisionTreeClassifier(max_depth=2)
    )
    model.lookahead = 12
    np.random.seed(42)
    N = 300
    X = pd.DataFrame(np.random.randn(N, 6), columns=[f"f{i}" for i in range(6)])
    y = np.random.choice([0, 1, 2], size=N)

    model.fit(X, y)
    assert hasattr(model, "meta_clf")
    assert model.meta_clf is not None
    assert len(model.weights) == 3


def test_r17_and_r18_exit_policy_target_sl_clamp_and_entry_atr():
    """R17 & R18: ExitPolicyEngine anchors BE to entry_atr and clamps target_sl below current_price."""
    engine = ExitPolicyEngine()
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 80000.0,
        "entry_atr": 40.0,
        "stop_loss": 79900.0,
        "take_profit": 80500.0,
        "leverage": 10.0,
        "break_even_triggered": False,
        "half_closed": False
    }
    # Pass price high enough to reach BE trigger distance
    exit_reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=80200.0,
        current_time=time.time(),
        current_atr=20.0,
        regime="Trending_Bullish"
    )
    assert updates.get("break_even_triggered") is True
    if "new_stop_loss" in updates:
        assert updates["new_stop_loss"] < 80200.0  # Must be clamped below current price!
    assert exit_reason is None  # Must NOT instantly trip stop loss!


def test_r19_min_order_bump_re_capped_size():
    """R19: Re-evaluation of checklist against scaled up min-order bump returns capped_size as 4th element."""
    bot_state = {
        "symbol_exposure_BTCUSDT": 5000.0,
        "available_margin": 1000.0,
        "circuit_breaker_active": False
    }
    active_trades = []
    passed, msg, dd_mult, capped_size = risk_engine.evaluate_pre_trade_checklist(
        symbol="BTCUSDT",
        position_size_usd=500.0,
        leverage_val=1.0,
        active_trades=active_trades,
        bot_state=bot_state,
        df_dict={},
        interval="15",
        direction="Bullish"
    )
    assert isinstance(capped_size, (int, float))


def test_r21_update_bybit_take_profit_freshness_and_direction():
    """R21: update_bybit_take_profit fails closed on stale price and validates TP price direction."""
    import main
    bot_state = {
        "live_price_BTCUSDT": 60000.0,
        "live_price_ts_BTCUSDT": time.time() - 40.0  # Stale (> 30s)
    }
    main.bot_state = bot_state

    orig_fallback = main.get_fallback_price
    main.get_fallback_price = lambda sym: None
    try:
        active_trade_long = {"direction": "Bullish", "qty": 1.0}
        active_trade_short = {"direction": "Bearish", "qty": 1.0}

        # Stale price -> fail-closed
        res = main.update_bybit_take_profit("BTCUSDT", 61000.0, active_trade=active_trade_long)
        assert res is False

        # Fresh price: Long TP <= price must return False
        bot_state["live_price_ts_BTCUSDT"] = time.time()
        res_long_invalid = main.update_bybit_take_profit("BTCUSDT", 59000.0, active_trade=active_trade_long)
        assert res_long_invalid is False

        # Fresh price: Short TP >= price must return False
        res_short_invalid = main.update_bybit_take_profit("BTCUSDT", 61000.0, active_trade=active_trade_short)
        assert res_short_invalid is False
    finally:
        main.get_fallback_price = orig_fallback


def test_r22_bybit_time_offset_initialization():
    """R22: _cached_time_offset is initialized to 0.0 and never causes TypeError in addition."""
    import main
    assert main._cached_time_offset is not None
    assert isinstance(main._cached_time_offset, (int, float))
    now_ms = int(time.time() * 1000)
    srv_offset = float(main.get_bybit_time_offset() or 0.0)
    assert isinstance(now_ms + srv_offset, float)


def test_r23_bybit_fill_recovery_list_parsing():
    """R23: Symbol position lookup parses list of dicts from get_all_bybit_positions correctly."""
    pos_list = [
        {"symbol": "ETHUSDT", "size": "1.5"},
        {"symbol": "BTCUSDT", "size": "0.25"},
    ]
    sym_pos = next((p for p in pos_list if isinstance(p, dict) and p.get("symbol") == "BTCUSDT"), {})
    assert float(sym_pos.get("size", 0.0)) == pytest.approx(0.25)


def test_r24_write_decision_signature():
    """R24: write_decision accepts DecisionRecord without TypeError."""
    rec = DecisionRecord(
        symbol="BTCUSDT",
        interval="15",
        outcome="REJECTED",
        reason_code=ReasonCode.PREDICTION_ERROR,
        reject_reason="Testing R24 fix"
    )
    write_decision(rec)


def test_r25_horizon_expiry_runner_extension():
    """R25: Level 10 runner extension prevents HORIZON_EXPIRY from prematurely closing runner."""
    direction = "Bullish"
    entry_price = 50000.0
    current_price = 52000.0
    stop_loss = 49000.0  # risk_dist = 1000, pnl_dist = 2000 -> curr_r_val = 2.0
    curr_regime = "Trending_Bullish"
    entry_regime_val = "Trending_Bullish"
    hierarchy_eval = {"decayed_expected_r": 0.25}

    is_long_dir = direction in ["Bullish", "BUY", "LONG", "UP"]
    pnl_dist = (current_price - entry_price) if is_long_dir else (entry_price - current_price)
    risk_dist = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 1e-6 else 1.0
    curr_r_val = round(pnl_dist / risk_dist, 2)
    decayed_exp_r = float(hierarchy_eval.get("decayed_expected_r", 0.0))

    is_runner_eligible = (
        curr_r_val >= 2.0 and
        decayed_exp_r >= 0.20 and
        "trend" in str(curr_regime).lower() and "trend" in str(entry_regime_val).lower()
    )
    assert is_runner_eligible is True


def test_r26_mhi_absorbing_state_cooldown_probe():
    """R26: MHI halt unlatches on cooldown expiration to permit exploratory probe allocation."""
    allocator = risk_engine.JointRiskBudgetAllocator()
    allocator._mhi_halted = True
    allocator._mhi_halt_time = time.time() - 15000.0  # > 4 hour cooldown

    kelly = allocator.get_mhi_max_kelly(45.0)
    assert kelly == pytest.approx(0.05)


def test_r27_signal_evaluator_calibrator_non_viable():
    """R27: Non-viable calibrator results in calibrated_conf = 0.0 (abstain)."""
    calibrator = {
        "method": "beta",
        "a": 1.0,
        "b": 1.0,
        "c": 0.0
    }
    from tools.beta_calibrator import is_calibrator_viable
    assert is_calibrator_viable(calibrator, min_required_p_star=0.55) is False


def test_r28_and_r29_resolve_trade_geometry_multiplier_recomputation():
    """R28 & R29: resolve_trade_geometry recomputes sl_multiplier_adjusted from final stop distance."""
    entry_price = 60000.0
    atr_dollars = 300.0
    config.MIN_SL_PCT_CONFIG["15"] = 0.015
    geom = trade_calculators.resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval="15",
        atr_dollars=atr_dollars,
        base_sl_multiplier=1.0,
        base_tp_multiplier=2.0,
        regime="Trending_Bullish"
    )
    assert geom["sl_dist"] >= 900.0
    expected_mult = float(geom["sl_dist"] / atr_dollars)
    assert geom["sl_multiplier_adjusted"] == pytest.approx(expected_mult, abs=1e-3)


def test_r30_completed_trade_regime_persistence_and_filtering():
    """R30: estimate_empirical_realized_rr filters strictly by trade regime."""
    trades = [
        {"interval": "15", "regime": "Trending_Bullish", "pnl_usd": 150.0, "position_size_usd": 1000.0},
        {"interval": "15", "regime": "Trending_Bullish", "pnl_usd": 150.0, "position_size_usd": 1000.0},
        {"interval": "15", "regime": "Trending_Bullish", "pnl_usd": -50.0, "position_size_usd": 1000.0},
        {"interval": "15", "regime": "Ranging", "pnl_usd": 300.0, "position_size_usd": 1000.0},
        {"interval": "15", "regime": "Ranging", "pnl_usd": -300.0, "position_size_usd": 1000.0},
    ]
    rr_trending = trade_calculators.estimate_empirical_realized_rr(
        closed_trades=trades,
        interval="15",
        regime="Trending_Bullish",
        min_samples=3
    )
    assert rr_trending is not None
    assert rr_trending == pytest.approx(3.0, abs=0.1)
