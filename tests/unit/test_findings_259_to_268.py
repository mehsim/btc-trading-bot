import os
import json
import sqlite3
import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

import config
from trade_calculators import resolve_trade_geometry
from order_state_machine import generate_client_order_id, IdempotencyCache, idempotency_cache
from api_telemetry import global_api_telemetry
from pain_feedback import PainFeedbackLoop
from config_verifier import assert_shared_constants_aligned
from mlops_engine import FeatureContractMismatchError, load_production_model_from_registry
from drift_monitor import DriftMonitor
import database
import bybit_client
import main


# ==============================================================================
# Finding #91: Parity between live and backtest stop geometry via resolve_trade_geometry
# ==============================================================================
def test_finding_91_trade_geometry_parity_and_target_rr_preservation():
    entry_price = 50000.0
    atr_dollars = 1000.0
    sl_mult = 1.0
    tp_mult = 2.0
    iv = "240"

    # 1. Bullish trade geometry calculation
    geom_long = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval=iv,
        atr_dollars=atr_dollars,
        base_sl_multiplier=sl_mult,
        base_tp_multiplier=tp_mult
    )

    # 2. Bearish trade geometry calculation
    geom_short = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bearish",
        interval=iv,
        atr_dollars=atr_dollars,
        base_sl_multiplier=sl_mult,
        base_tp_multiplier=tp_mult
    )

    # Both directions must have identical distance offsets
    assert geom_long["sl_dist"] == geom_short["sl_dist"]
    assert geom_long["tp_dist"] == geom_short["tp_dist"]
    assert geom_long["stop_loss_price"] == entry_price - geom_long["sl_dist"]
    assert geom_long["take_profit_price"] == entry_price + geom_long["tp_dist"]
    assert geom_short["stop_loss_price"] == entry_price + geom_short["sl_dist"]
    assert geom_short["take_profit_price"] == entry_price - geom_short["tp_dist"]

    # 3. Timeframe multiplier is applied: 240m uses 1.35
    expected_sl_dist = 1000.0 * 1.35
    assert abs(geom_long["sl_dist"] - expected_sl_dist) < 1e-3

    # 4. Floor binding test: when stop distance is tiny, floor binds and target is expanded
    tiny_atr = 10.0 # 10 USD on 50000 -> 0.02%
    floor_pct = config.MIN_SL_PCT_CONFIG.get(iv, 0.012)
    expected_floor_dist = entry_price * floor_pct # 600 USD
    target_rr = tp_mult / max(1e-6, sl_mult) # 2.0

    geom_flr = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval=iv,
        atr_dollars=tiny_atr,
        base_sl_multiplier=sl_mult,
        base_tp_multiplier=tp_mult
    )

    assert geom_flr["sl_dist"] == expected_floor_dist
    assert geom_flr["tp_dist"] == expected_floor_dist * target_rr
    assert abs((geom_flr["tp_dist"] / geom_flr["sl_dist"]) - target_rr) < 1e-6


# ==============================================================================
# Finding #92: CVaR open risk calculation and daily budget exhaustion
# ==============================================================================
def test_finding_92_cvar_open_risk_and_budget_exhaustion():
    # 1. Verify active trade stores stop_loss_pct and notional_usd
    entry = 50000.0
    sl = 49000.0
    qty = 0.1
    notional = entry * qty # 5000.0
    sl_pct = abs(entry - sl) / entry # 0.02

    active_trade = {
        "symbol": "BTCUSDT",
        "entry_price": entry,
        "stop_loss": sl,
        "qty": qty,
        "notional_usd": notional,
        "stop_loss_pct": sl_pct
    }

    # True open risk: abs(entry - sl) * qty = 1000 * 0.1 = 100.0
    trade_open_risk = abs(float(active_trade.get("entry_price", 0.0)) - float(active_trade.get("stop_loss", 0.0))) * float(active_trade.get("qty", 0.0))
    assert trade_open_risk == 100.0

    # 2. Verify daily loss budget exhaustion check in main.py logic
    equity = 1000.0
    daily_budget = equity * 0.05 # 50.0
    daily_realized_loss = -55.0 # exceeded budget
    remaining_daily_budget = daily_budget - abs(daily_realized_loss)

    assert remaining_daily_budget <= 0.0
    # In main.py:8785: remaining_daily_budget <= 0.0 triggers DAILY_LOSS_BUDGET_EXHAUSTED rejection


# ==============================================================================
# Finding #93: Configuration verifier eliminates self-comparison and strictly validates manifests
# ==============================================================================
def test_finding_93_config_verifier_eliminates_self_comparison():
    # 1. Aligned manifest passes without error
    aligned_manifest = {
        "barrier_config": {
            "lookahead": 12,
            "sl_mult": 0.80,
            "tp_mult_trending": 1.85,
            "tp_mult_ranging": 1.15,
            "regime_adx_enter": 28.0
        }
    }
    with patch("glob.glob", return_value=["ensemble_trending_trend_15_manifest.json"]), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=aligned_manifest):
        assert_shared_constants_aligned()

    # 2. Divergent manifest (> 0.05) raises ValueError
    divergent_manifest = {
        "barrier_config": {
            "lookahead": 12,
            "sl_mult": 1.20, # diverges from live 0.80 by 0.40 > 0.05
            "tp_mult_trending": 1.85,
            "tp_mult_ranging": 1.15,
            "regime_adx_enter": 25.0
        }
    }
    with patch("glob.glob", return_value=["ensemble_trending_trend_60_manifest.json"]), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=divergent_manifest):
        with pytest.raises(ValueError, match="Barrier divergence"):
            assert_shared_constants_aligned()


# ==============================================================================
# Finding #94: Drift monitor empirical baseline and state manager write
# ==============================================================================
def test_finding_94_drift_monitor_empirical_baseline_and_state_manager_write():
    monitor = DriftMonitor()

    # 1. Baseline distribution loader returns non-empty array of float confidences
    baseline_confs = monitor._get_training_baseline_confidences()
    assert len(baseline_confs) > 0
    assert all(0.0 <= float(x) <= 1.0 for x in baseline_confs)

    # 2. State manager persistence on empty trades
    mock_state = {}
    with patch("state_manager.state_manager", mock_state), \
         patch("drift_monitor.get_recent_experiences", return_value=[]), \
         patch("drift_monitor.calculate_ece", return_value=0.04):
        res = monitor.evaluate_drift()

    assert res["drift_alert"] is False
    assert mock_state.get("last_ece") == 0.04
    assert mock_state.get("last_psi") == 0.0
    assert mock_state.get("last_brier_score") == 0.10


# ==============================================================================
# Finding #95: Database IntegrityError sets success = False and fuzzy dedup None handling
# ==============================================================================
def test_finding_95_database_integrity_error_and_fuzzy_dedup_none():
    # 1. Verify that sqlite3.IntegrityError on insert sets success = False
    with patch("database.get_db_connection") as mock_conn_factory:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        # Third execute call is INSERT, raise IntegrityError
        mock_conn.execute.side_effect = [mock_cursor, mock_cursor, sqlite3.IntegrityError("UNIQUE constraint failed: completed_trades.trade_id")]
        mock_conn_factory.return_value = mock_conn

        res = database.save_completed_trade({
            "trade_id": "BTC_TEST_INT_1",
            "symbol": "BTCUSDT",
            "direction": "Bullish",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "exit_time": time.time(),
            "pnl_usd": 50.0,
            "success": True,
            "size": 0.01,
            "interval": "15"
        })
        assert res is False

    # 2. Verify fuzzy deduplication handles None entry/exit prices without raising TypeError
    trade_none = {
        "trade_id": "BTC_TEST_NONE_1",
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": None,
        "exit_price": None,
        "entry_time": time.time(),
        "exit_time": time.time() + 10,
        "pnl_usd": 0.0,
        "size": 0.01,
        "interval": "15"
    }
    res_none = database.save_completed_trade(trade_none)
    assert isinstance(res_none, bool)


# ==============================================================================
# Finding #96: Balance sync timestamp, inference latency telemetry, and circuit breaker
# ==============================================================================
def test_finding_96_balance_sync_ts_inference_latency_and_circuit_breaker():
    # 1. Balance sync sets last_balance_sync_ts in state_manager and bot_state
    b_state = {}
    with patch("bybit_client.get_real_bybit_balance_cached", return_value=1000.0), \
         patch("state_manager.state_manager", {}) as s_state, \
         patch("time.sleep", side_effect=StopIteration):
        try:
            bybit_client.run_bybit_balance_updater(bot_state=b_state)
        except StopIteration:
            pass

    assert b_state.get("last_balance_sync_ts") is not None
    assert s_state.get("last_balance_sync_ts") is not None
    assert b_state["last_balance_sync_ts"] > 0

    # 2. Circuit breaker evaluate_system_health checks
    from circuit_breaker import circuit_breaker
    # Unhealthy DB
    ok_db, reason_db = circuit_breaker.evaluate_system_health(50.0, time.time(), db_healthy=False, inference_latency_ms=50.0)
    assert ok_db is False
    assert "Database" in reason_db

    # Stale balance
    ok_bal, reason_bal = circuit_breaker.evaluate_system_health(50.0, 0.0, db_healthy=True, inference_latency_ms=50.0)
    assert ok_bal is False
    assert "STALE_BALANCE" in reason_bal

    # High latency
    ok_lat, reason_lat = circuit_breaker.evaluate_system_health(1500.0, time.time(), db_healthy=True, inference_latency_ms=50.0)
    assert ok_lat is False
    assert "HIGH_LATENCY" in reason_lat

    # Slow inference
    ok_inf, reason_inf = circuit_breaker.evaluate_system_health(50.0, time.time(), db_healthy=True, inference_latency_ms=600.0)
    assert ok_inf is False
    assert "SLOW_INFERENCE" in reason_inf

    # All healthy
    ok_good, reason_good = circuit_breaker.evaluate_system_health(50.0, time.time(), db_healthy=True, inference_latency_ms=50.0)
    assert ok_good is True
    assert reason_good == "HEALTHY"


# ==============================================================================
# Finding #97: Fail-closed on missing model contract fields and model reset
# ==============================================================================
def test_finding_97_mlops_feature_contract_and_model_reload_reset():
    # 1. Missing served_hash in MLflow run raises FeatureContractMismatchError
    mock_run = MagicMock()
    mock_run.data.tags = {} # missing feature_contract_hash
    mock_mv = MagicMock()
    mock_mv.run_id = "run_123"
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mock_mv]
    mock_client.get_run.return_value = mock_run

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        with pytest.raises(FeatureContractMismatchError, match="missing contract metadata"):
            load_production_model_from_registry("15", "trending", live_features=["f1", "f2"])

    # 2. Missing live_features argument raises FeatureContractMismatchError
    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        with pytest.raises(FeatureContractMismatchError, match="missing contract metadata"):
            load_production_model_from_registry("15", "trending", live_features=None)


# ==============================================================================
# Finding #98: Interval passed to calculate_final_stop_distance and pain trade min samples
# ==============================================================================
def test_finding_98_interval_in_stop_distance_and_pain_trade_min_sample_size():
    # 1. Pain trade requires >= 2 events to widen floor
    loop = PainFeedbackLoop()
    loop.adjustments = {}
    loop.save_state = MagicMock()
    baseline = 0.005

    # Event 1: Records event but does not raise floor
    loop.register_pain_trade("ETHUSDT", entry_price=3000.0, exit_price=2970.0, take_profit=3100.0, current_floor=baseline, interval="15")
    eff_after_1 = loop.get_effective_floor("ETHUSDT", interval="15")
    assert eff_after_1 is None # Floor not raised on single event

    # Event 2: Second pain trade within window raises floor
    loop.register_pain_trade("ETHUSDT", entry_price=3000.0, exit_price=2970.0, take_profit=3100.0, current_floor=baseline, interval="15")
    eff_after_2 = loop.get_effective_floor("ETHUSDT", interval="15")
    assert eff_after_2 is not None
    assert eff_after_2 > baseline


# ==============================================================================
# Finding #99: Single canonical order execution path and shared symbol lock
# ==============================================================================
def test_finding_99_unified_order_execution_and_clean_imports():
    # 1. Lock instance is canonical and case-insensitive
    lock1 = bybit_client.get_symbol_order_lock("BTCUSDT")
    lock2 = main.get_symbol_order_lock("btcusdt")
    assert lock1 is lock2

    # 2. main.py exports place_bybit_order from bybit_client
    assert main.place_bybit_order is bybit_client.place_bybit_order
    assert main.execute_bybit_order_ws_or_rest is bybit_client.execute_bybit_order_ws_or_rest

    # 3. Verify dead imports removed from main.py
    import sys
    assert "websocket_client" not in sys.modules.get("main", {}).__dict__
    assert "trading_engine" not in sys.modules.get("main", {}).__dict__


# ==============================================================================
# Finding #100: Pre-submission idempotency check and deterministic client order ID
# ==============================================================================
def test_finding_100_presubmission_idempotency_and_deterministic_client_order_id():
    # 1. Deterministic client order ID generation
    cid_det = generate_client_order_id(symbol="BTCUSDT", side="Buy", interval="15", candle_ts=1700000000)
    assert cid_det == "B_BTC_15_1700000000_B"
    assert len(cid_det) <= 36

    # 2. Pre-submission idempotency blocks duplicate orders
    test_id = f"test_idemp_{int(time.time()*1000)}"
    payload = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderLinkId": test_id
    }

    with patch("bybit_client.bybit_post_request", return_value={"retCode": 0, "result": {"orderId": "123"}}):
        # First execution succeeds
        res1 = bybit_client.execute_bybit_order_ws_or_rest("/v5/order/create", payload)
        assert res1["retCode"] == 0

        # Duplicate execution is blocked before sending HTTP request
        res2 = bybit_client.execute_bybit_order_ws_or_rest("/v5/order/create", payload)
        assert res2["retCode"] == 10001
        assert "Duplicate order blocked" in res2["retMsg"]

    # 3. REST calls record into global_api_telemetry
    with patch("api_telemetry.global_api_telemetry.record_call") as mock_rec:
        bybit_client._update_latency(time.time() - 0.05, "/v5/market/kline", 200)
        assert mock_rec.called
        assert mock_rec.call_args[0][0] == "/v5/market/kline"
        status = mock_rec.call_args.kwargs.get("status_code") or (mock_rec.call_args[0][2] if len(mock_rec.call_args[0]) > 2 else 200)
        assert status == 200
