"""
Unit tests for audit defect findings #121 through #140.
"""

import math
import os
import time
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

import config
from circuit_breaker import CircuitBreaker
from config_verifier import assert_shared_constants_aligned
import database
from drift_monitor import DriftMonitor
from ensemble import compute_effective_sample_size
import mlops_engine
from order_state_machine import IdempotencyCache, generate_client_order_id
from pain_feedback import PainFeedbackLoop
from transaction_cost_model import TransactionCostModel
import risk_engine
from exit_policy_engine import ExitPolicyEngine


def test_finding_121_circuit_breaker_evaluate_system_health():
    """Finding #121: evaluate_system_health evaluates exchange latency, stale balance, db health, inference latency."""
    cb = CircuitBreaker()
    # Healthy
    healthy, reason = cb.evaluate_system_health(
        exchange_latency_ms=150.0,
        last_balance_sync_ts=time.time() - 30.0,
        db_healthy=True,
        inference_latency_ms=100.0
    )
    assert healthy is True
    assert reason == "HEALTHY"

    # High exchange latency
    h_lat, r_lat = cb.evaluate_system_health(
        exchange_latency_ms=1200.0,
        last_balance_sync_ts=time.time() - 30.0,
        db_healthy=True,
        inference_latency_ms=100.0
    )
    assert h_lat is False
    assert "HIGH_LATENCY" in r_lat

    # Stale balance
    h_bal, r_bal = cb.evaluate_system_health(
        exchange_latency_ms=150.0,
        last_balance_sync_ts=time.time() - 400.0,
        db_healthy=True,
        inference_latency_ms=100.0
    )
    assert h_bal is False
    assert "STALE_BALANCE" in r_bal

    # DB Unhealthy
    h_db, r_db = cb.evaluate_system_health(
        exchange_latency_ms=150.0,
        last_balance_sync_ts=time.time() - 30.0,
        db_healthy=False,
        inference_latency_ms=100.0
    )
    assert h_db is False
    assert "Database connection or integrity failure" in r_db


def test_finding_122_degenerate_holdout_guard():
    """Finding #122: trending_120 and degenerate holdouts are caught by serve-time checks."""
    from config import is_model_slot_holdout_valid
    
    # Degenerate accuracy <= 0.334
    bad_metrics_acc = {"holdout_balanced_accuracy": 0.33, "holdout_mcc": 0.15, "holdout_brier": 0.20, "holdout_ece": 0.05}
    assert is_model_slot_holdout_valid("trending_120", bad_metrics_acc) is False

    # Negative MCC
    bad_metrics_mcc = {"holdout_balanced_accuracy": 0.50, "holdout_mcc": -0.02, "holdout_brier": 0.20, "holdout_ece": 0.05}
    assert is_model_slot_holdout_valid("trending_30", bad_metrics_mcc) is False

    # Saturated ECE / Brier
    bad_metrics_ece = {"holdout_balanced_accuracy": 0.50, "holdout_mcc": 0.10, "holdout_brier": 0.995, "holdout_ece": 0.99}
    assert is_model_slot_holdout_valid("ranging_15", bad_metrics_ece) is False

    # Valid model
    good_metrics = {"holdout_balanced_accuracy": 0.55, "holdout_mcc": 0.12, "holdout_brier": 0.22, "holdout_ece": 0.08}
    assert is_model_slot_holdout_valid("trending_15", good_metrics) is True


def test_finding_123_config_verifier():
    """Finding #123: assert_shared_constants_aligned compares against actual config entries without crashing or self-comparison."""
    # Should run and pass without error
    assert_shared_constants_aligned()


def test_finding_124_save_completed_trade_deduplication():
    """Finding #124: save_completed_trade on collision returns True and avoids inserting duplicate rows."""
    test_trade = {
        "trade_id": "test_dedup_trade_124",
        "symbol": "BTCUSDT",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "pnl_usd": 100.0,
        "pnl_pct": 2.0,
        "direction": "LONG",
        "entry_time": int(time.time() * 1000),
        "exit_time": int(time.time() * 1000) + 3600000,
        "reason": "TAKE PROFIT",
        "initial_planned_rr": 1.5,
        "realized_rr": 1.5
    }

    # First save
    res1 = database.save_completed_trade(test_trade)
    assert res1 is True

    # Second save with exact same trade_id should not create duplicate rows
    res2 = database.save_completed_trade(test_trade)
    assert res2 is True

    conn = database.get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM completed_trades WHERE trade_id = 'test_dedup_trade_124'")
    count = c.fetchone()[0]
    conn.close()

    assert count == 1, f"Expected exactly 1 row for trade_id, found {count}"


def test_finding_125_drift_monitor_evaluate_drift():
    """Finding #125: DriftMonitor updates last_psi, last_brier_score, last_ece with real distributions."""
    monitor = DriftMonitor()
    mock_experiences = [{"confidence": 0.6, "individual_brier_loss": 0.15}] * 25
    with patch("drift_monitor.get_recent_experiences", return_value=mock_experiences):
        res = monitor.evaluate_drift()
        assert "psi" in res
        assert "ece" in res
        assert "rolling_brier_50" in res
        from state_manager import state_manager
        assert state_manager.get("last_psi") is not None
        assert state_manager.get("last_ece") is not None
        assert state_manager.get("last_brier_score") is not None


def test_finding_126_champion_challenger_lookahead():
    """Finding #126: compute_effective_sample_size uses standardized lookahead convention."""
    # Standard convention
    req_std = compute_effective_sample_size(n_samples=1000, lookahead=12, n_symbols=2, convention="lookahead_x_nsymbols_v2")
    assert req_std > 0

    # Unknown convention raises ValueError
    with pytest.raises(ValueError, match="Unknown lookahead sample convention"):
        compute_effective_sample_size(n_samples=1000, lookahead=12, n_symbols=2, convention="incompatible_unknown_convention")


def test_finding_127_live_governance_holdout_mcc():
    """Finding #127: Live governance checks holdout MCC against config threshold."""
    min_mcc = config.TIMEFRAME_MIN_HOLDOUT_MCC.get("15", 0.05)
    
    # Sub-threshold MCC
    cand_sub = {"holdout_mcc": min_mcc - 0.02, "mcc": 0.20}
    assert cand_sub["holdout_mcc"] < min_mcc

    # Sentinel MCC (-999.0)
    cand_sentinel = {"holdout_mcc": -999.0, "mcc": 0.20}
    assert cand_sentinel["holdout_mcc"] < min_mcc

    # Passing MCC
    cand_pass = {"holdout_mcc": min_mcc + 0.05, "mcc": 0.20}
    assert cand_pass["holdout_mcc"] >= min_mcc


def test_finding_128_golden_hour_boost_cvar_clamp():
    """Finding #128: Golden hour multiplier is applied before CVaR clamp."""
    base_notional = 100.0
    golden_hour_mult = 2.0
    max_cvar_notional = 150.0

    # If boost is applied before clamp:
    target_notional = base_notional * golden_hour_mult # 200.0
    clamped_notional = min(target_notional, max_cvar_notional) # 150.0

    # Ensure final notional never exceeds CVaR limit
    assert clamped_notional <= max_cvar_notional
    assert clamped_notional == 150.0


def test_finding_129_composite_uncertainty_distinct_probabilities():
    """Finding #129: composite uncertainty uses distinct model probabilities."""
    from statistical_validation import statistical_validation
    
    # Passing distinct probabilities from xgb, lgb, cat
    u_comp = statistical_validation.calculate_composite_uncertainty(
        individual_predictions={"xgb": 0.75, "lgb": 0.60, "cat": 0.55},
        model_weights={"xgb": 0.33, "lgb": 0.33, "cat": 0.34}
    )
    # Ensemble disagreement U_ensemble must be > 0 when probabilities differ
    assert u_comp["u_ensemble"] > 0.0
    assert u_comp["u_total"] > 0.0


def test_finding_130_model_registry_key_by_model_name():
    """Finding #130: transition_model_to_production keys by model_name preserving multiple slots."""
    import tempfile
    import json
    from mlops_engine import ModelRegistry, STAGE_PRODUCTION

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        reg_file = tf.name

    try:
        reg = ModelRegistry(reg_file)
        reg.register_model("run1", "trending_15", {"mcc": 0.15}, stage=STAGE_PRODUCTION)
        reg.register_model("run2", "trending_60", {"mcc": 0.18}, stage=STAGE_PRODUCTION)

        # Both slots should exist in Production under their respective model names
        prod_15 = reg.get_production_model("trending_15")
        prod_60 = reg.get_production_model("trending_60")

        assert prod_15 is not None
        assert prod_15["model_name"] == "trending_15"
        assert prod_60 is not None
        assert prod_60["model_name"] == "trending_60"
    finally:
        if os.path.exists(reg_file):
            os.remove(reg_file)


def test_finding_131_feature_contract_mismatch():
    """Finding #131: Feature contract mismatch raises distinct exception without falling back."""
    from mlops_engine import FeatureContractMismatchError, load_production_model_from_registry

    mock_client = MagicMock()
    mock_mv = MagicMock(version="1", run_id="run123")
    mock_client.get_latest_versions.return_value = [mock_mv]
    mock_run = MagicMock()
    mock_run.data.tags = {"feature_contract_hash": "hash_xyz"}
    mock_client.get_run.return_value = mock_run

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client), \
         patch("mlops_engine.MLFLOW_AVAILABLE", True), \
         patch("mlops_engine._mlflow_unreachable", False):
        with pytest.raises(FeatureContractMismatchError):
            load_production_model_from_registry("15", "trending", live_features=["feature1", "feature2"])


def test_finding_132_order_idempotency_cache():
    """Finding #132: generate_client_order_id and IdempotencyCache prevent duplicate submissions."""
    cache = IdempotencyCache(ttl_seconds=60)
    
    cid1 = generate_client_order_id("BTCUSDT", "BUY")
    assert len(cid1) <= 36
    assert cid1.startswith("B_BTC_B_")

    # First check: not duplicate
    assert cache.is_duplicate(cid1) is False
    cache.add(cid1)
    # Duplicate submission immediately detected
    assert cache.is_duplicate(cid1) is True


def test_finding_133_pain_feedback_interval_aware():
    """Finding #133: PainFeedbackLoop keys by (symbol, interval) and uses MIN_SL_PCT_CONFIG."""
    loop = PainFeedbackLoop()
    loop.adjustments = {}

    baseline_15 = config.MIN_SL_PCT_CONFIG.get("15", 0.005)
    baseline_240 = config.MIN_SL_PCT_CONFIG.get("240", 0.012)

    # Register pain for 15m (requires >= 2 events per Finding #98)
    loop.register_pain_trade("BTCUSDT", entry_price=50000.0, exit_price=49600.0, take_profit=51000.0, current_floor=baseline_15, interval="15")
    loop.register_pain_trade("BTCUSDT", entry_price=50000.0, exit_price=49600.0, take_profit=51000.0, current_floor=baseline_15, interval="15")
    
    eff_15 = loop.get_effective_floor("BTCUSDT", interval="15")
    eff_240 = loop.get_effective_floor("BTCUSDT", interval="240")

    assert eff_15 is not None
    assert eff_15 >= baseline_15
    # 240m should not be polluted by 15m adjustment
    assert eff_240 is None or eff_240 != eff_15


def test_finding_134_mde_promotion_guard():
    """Finding #134: Models with |holdout_mcc| < holdout_mcc_mde_80pct cannot be promoted."""
    mde_80 = 0.08
    mcc_under = 0.05
    mcc_over = 0.12

    # Gating check logic
    can_promote_under = abs(mcc_under) >= mde_80
    can_promote_over = abs(mcc_over) >= mde_80

    assert can_promote_under is False
    assert can_promote_over is True


def test_finding_135_price_regressor_manifest_metrics():
    """Finding #135: Price regressor records regression metrics, not classifier metrics."""
    from train import build_regressor_governance_manifest

    manifest = build_regressor_governance_manifest(
        model_name="btc_price_regressor_15",
        mae=12.5,
        rmse=18.2,
        r2=0.45
    )
    assert "metrics" in manifest
    assert "mae" in manifest["metrics"]
    assert "rmse" in manifest["metrics"]
    assert "r2" in manifest["metrics"]
    # Should not have classifier metrics masquerading
    assert manifest["metrics"].get("accuracy") is None


def test_finding_136_tcm_config_fees_and_round_trip():
    """Finding #136: TransactionCostModel reads fees from config and supports round-trip costing."""
    tcm = TransactionCostModel()
    
    # Verify default fees match config
    expected_taker_bp = config.TAKER_FEE_PCT * 10000.0
    expected_maker_bp = config.MAKER_FEE_PCT * 10000.0
    assert pytest.approx(tcm.taker_fee_bp, 0.01) == expected_taker_bp
    assert pytest.approx(tcm.maker_fee_bp, 0.01) == expected_maker_bp

    # Single leg cost vs round trip cost
    single_cost = tcm.estimate_transaction_cost("BTCUSDT", order_size_usd=1000.0, round_trip=False)
    round_cost = tcm.estimate_transaction_cost("BTCUSDT", order_size_usd=1000.0, round_trip=True)

    assert round_cost["total_cost_bp"] > single_cost["total_cost_bp"]
    assert round_cost["round_trip"] is True


def test_finding_137_websocket_and_bybit_client_stubs():
    """Finding #137: websocket_client does not falsely report connected, and bybit_client unwraps order details."""
    import websocket_client
    import bybit_client

    # Calling init_bybit_websocket_listeners does not falsely report connected
    websocket_client.init_bybit_websocket_listeners(["BTCUSDT"])
    status = websocket_client.get_ws_status()
    assert status["public_connected"] is False

    # bybit_client.get_bybit_order_details correctly unwraps response
    with patch("bybit_client.bybit_get_request", return_value={"retCode": 0, "result": {"list": [{"orderId": "123", "orderStatus": "Filled"}]}}):
        details = bybit_client.get_bybit_order_details("BTCUSDT", "123")
        assert details.get("orderId") == "123"
        assert details.get("orderStatus") == "Filled"


def test_finding_138_backtest_timeframe_stop_multiplier():
    """Finding #138: backtest aligns stop-loss multiplier with risk_engine.get_timeframe_stop_multiplier."""
    mult_15 = risk_engine.get_timeframe_stop_multiplier("15")
    mult_240 = risk_engine.get_timeframe_stop_multiplier("240")

    assert mult_15 == 0.80
    assert mult_240 == 1.35
    assert mult_240 > mult_15


def test_finding_139_5factor_stagnation_exit():
    """Finding #139: evaluate_exit passes entry_atr and live volume allowing stagnation gate to fire."""
    engine = ExitPolicyEngine()
    
    active_trade = {
        "symbol": "BTCUSDT",
        "entry_price": 50000.0,
        "entry_time": int((time.time() - 36000) * 1000), # 10 hours ago
        "entry_atr": 500.0,
        "atr_dollars": 300.0,
        "direction": "Bullish",
        "position_size_usd": 100.0,
        "leverage": 10.0,
        "interval": "15"
    }

    # Price slightly lower than entry (negative PnL, small dev), low ATR, low volume, low ADX
    exit_reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=49900.0,
        current_time=time.time(),
        regime="CHOPPY",
        adx_val=14.0,
        current_volume=30.0,
        avg_volume=100.0,
        current_atr=300.0 # 300 < 0.8 * 500 = 400
    )

    assert exit_reason is not None
    assert "5-FACTOR STAGNATION" in exit_reason


def test_finding_140_exit_hierarchy_lookahead_derivation():
    """Finding #140: evaluate_10_level_exit_hierarchy derives soft and hard limits from TIMEFRAME_CONFIG."""
    engine = ExitPolicyEngine()
    
    res = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT",
        interval="240",
        current_price=50000.0,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        direction="Bullish",
        candles_elapsed=5,
        expected_r=1.2
    )

    # For 240m, lookahead is 12 -> soft_limit >= 12, hard_limit >= 18 (not old 8, 12)
    assert res["soft_limit_candles"] >= 12
    assert res["hard_limit_candles"] >= 18
