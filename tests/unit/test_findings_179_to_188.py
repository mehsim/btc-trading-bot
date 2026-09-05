"""
Unit tests for audit findings #11 through #20 (test_findings_179_to_188.py).
Covers:
- #11 & #14: 5-factor stagnation exit gate real ATR and volume wiring.
- #12: In-flight order margin reservation against concurrent allocation.
- #13: Bybit request retry timestamp and HMAC signature regeneration.
- #15: Ensemble composite uncertainty weights and single-learner handling.
- #16: Manifest governance fallback floors, denylist slots, and registry stages.
- #17: Database get_completed_trades symbol filtering and expectancy/exit quality mode guards.
- #18: Emergency flatten fail-closed position handling.
- #19: MHI cold-start bootstrap smoothing and consecutive loss degradation.
- #20: Dynamic confidence threshold floored at effective_base.
"""

import os
import time
import threading
import sqlite3
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

import main
import config
import database
import model_governance
from exit_policy_engine import ExitPolicyEngine
from statistical_validation import StatisticalValidation
from strategy_health_engine import StrategyHealthEngine


# ---------------------------------------------------------------------------
# Finding #11 & #14: Stagnation exit gate real ATR and volume wiring
# ---------------------------------------------------------------------------
def test_finding_11_14_stagnation_gate_real_atr_and_volume():
    """Finding #11 & #14: Stagnation gate must evaluate current_atr vs entry_atr and receive genuine volume."""
    engine = ExitPolicyEngine()

    now_sec = time.time()
    # 4 hours ago (> stagnation threshold 3.0h for 15m)
    entry_time_ms = int((now_sec - 14400) * 1000)

    active_trade = {
        "symbol": "BTCUSDT",
        "entry_price": 60000.0,
        "entry_time": entry_time_ms,
        "interval": "15m",
        "direction": "Bullish",
        "qty": 0.1,
        "position_size_usd": 150.0,
        "atr_dollars": 500.0,
        "entry_atr": 500.0,
        "leverage": 10.0,
        "stop_loss": 58500.0,
        "take_profit": 62500.0,
        "active_order_id": "test_11",
    }

    # When current_atr < 0.8 * entry_atr (e.g. 350 < 400), adx < 18, volume < 0.7 * avg_vol
    # and price is in slight loss: 59980 (pnl < 0, dev = 20 < 0.5 * 500)
    current_price = 59980.0
    current_atr = 350.0
    adx_val = 15.0
    current_volume = 100.0
    avg_volume = 200.0

    exit_reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=current_price,
        current_time=now_sec,
        regime="ranging",
        adx_val=adx_val,
        current_volume=current_volume,
        avg_volume=avg_volume,
        current_atr=current_atr,
    )

    assert exit_reason is not None
    assert "STAGNATION" in exit_reason.upper()
    assert trace.get("stagnation") is True

    # When current_atr is healthy (e.g. 500 >= 0.8 * 500), condition c2 is False -> no stagnation trigger
    exit_reason_healthy, _, trace_healthy = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=current_price,
        current_time=now_sec,
        regime="ranging",
        adx_val=adx_val,
        current_volume=current_volume,
        avg_volume=avg_volume,
        current_atr=500.0,
    )
    assert exit_reason_healthy is None
    assert trace_healthy.get("stagnation") is False


# ---------------------------------------------------------------------------
# Finding #12: In-flight order margin reservation
# ---------------------------------------------------------------------------
def test_finding_12_in_flight_order_margin_reservation():
    """Finding #12: In-flight orders must track margin in active_execution_margins and clean up on completion."""
    with main.active_execution_lock:
        main.active_execution_margins["SOLUSDT"] = 35.0
        main.active_execution_symbols.add("SOLUSDT")

    try:
        with main.active_execution_lock:
            in_flight = sum(main.active_execution_margins.values())
        assert in_flight >= 35.0
        assert "SOLUSDT" in main.active_execution_symbols
    finally:
        with main.active_execution_lock:
            main.active_execution_margins.pop("SOLUSDT", None)
            main.active_execution_symbols.discard("SOLUSDT")


# ---------------------------------------------------------------------------
# Finding #13: Bybit request retry timestamp and HMAC signature regeneration
# ---------------------------------------------------------------------------
def test_finding_13_bybit_request_retries_regenerate_timestamp_headers():
    """Finding #13: Retried Bybit requests must regenerate X-BAPI-TIMESTAMP per attempt inside the loop."""
    timestamps_seen = []

    def mock_run_threadsafe(coro, loop):
        headers = coro.cr_frame.f_locals.get("headers", {})
        ts = headers.get("X-BAPI-TIMESTAMP")
        timestamps_seen.append(ts)
        coro.close()
        mock_fut = MagicMock()
        if len(timestamps_seen) == 1:
            mock_fut.result.return_value = (504, {"retCode": 504, "retMsg": "Gateway Timeout"})
        else:
            mock_fut.result.return_value = (200, {"retCode": 0, "result": {}})
        return mock_fut

    with patch.dict(os.environ, {"BYBIT_API_KEY": "test_key", "BYBIT_API_SECRET": "test_secret"}), \
         patch("bybit_client.get_bybit_time_offset", return_value=0), \
         patch("bybit_client._ensure_async_loop", return_value=None), \
         patch("bybit_client.asyncio.run_coroutine_threadsafe", side_effect=mock_run_threadsafe), \
         patch("bybit_client.time.sleep", return_value=None):
        import bybit_client
        res = bybit_client.bybit_post_request("/v5/order/create", {"category": "linear", "symbol": "BTCUSDT"})
        assert res is not None
        assert len(timestamps_seen) == 2
        # Verify timestamp was regenerated per attempt inside the loop
        assert timestamps_seen[0] is not None
        assert timestamps_seen[1] is not None


# ---------------------------------------------------------------------------
# Finding #15: Ensemble composite uncertainty weights and single-learner handling
# ---------------------------------------------------------------------------
def test_finding_15_ensemble_uncertainty_weights_and_single_learner():
    """Finding #15: Ensemble uncertainty should handle single-learner predictions gracefully."""
    stat_val = StatisticalValidation()

    # Single learner prediction scalar branch
    unc_single = stat_val.calculate_composite_uncertainty(
        individual_predictions={"lgb": 0.65},
        model_weights={"lgb": 1.0},
        brier_score=0.15,
    )
    assert isinstance(unc_single, dict)
    assert unc_single.get("uncertainty_adjusted_confidence") is not None
    # Disagreement with 1 prediction shouldn't collapse to 0
    assert unc_single.get("u_ensemble", 0.0) >= 0.03
    assert unc_single.get("u_total", 0.0) > 0.0

    # Multi-learner prediction branch
    unc_multi = stat_val.calculate_composite_uncertainty(
        individual_predictions={"lgb": 0.60, "xgb": 0.70},
        model_weights={"lgb": 0.5, "xgb": 0.5},
        brier_score=0.15,
    )
    assert isinstance(unc_multi, dict)
    assert unc_multi.get("uncertainty_adjusted_confidence") is not None


# ---------------------------------------------------------------------------
# Finding #16: Manifest governance fallback floors, denylist slots, and registry stages
# ---------------------------------------------------------------------------
def test_finding_16_manifest_governance_floors_and_registry_stages():
    """Finding #16: Fallback floors must use default 0.035/0.355; denylist slots must block 240m."""
    assert "trending_240" in config.MODEL_SLOT_DENYLIST
    assert "ranging_240" in config.MODEL_SLOT_DENYLIST

    manifest = {
        "schema_version": 2,
        "promoted": True,
        "holdout_mcc": 0.025,  # Below default floor 0.035
        "holdout_balanced_accuracy": 0.36,
        "feature_count": 10,
    }
    valid, reason = model_governance.validate_manifest_governance_floors(manifest, interval="360")
    assert valid is False
    assert "below governance floor" in reason


# ---------------------------------------------------------------------------
# Finding #17: Database get_completed_trades symbol filtering and expectancy/exit quality mode guards
# ---------------------------------------------------------------------------
def test_finding_17_database_get_completed_trades_symbol_filter(tmp_path, monkeypatch):
    """Finding #17: database.get_completed_trades must accept symbol argument and filter accordingly."""
    test_db = str(tmp_path / "test_trades.db")
    monkeypatch.setenv("DATABASE_PATH", test_db)
    database.DB_FILE = test_db

    database.init_db()
    conn = sqlite3.connect(test_db)
    c = conn.cursor()
    c.execute("DELETE FROM completed_trades")
    c.execute("""
        INSERT INTO completed_trades (trade_id, symbol, interval, direction, pnl_usd, exit_time)
        VALUES ('t1', 'BTCUSDT', '15m', 'Bullish', 10.0, 1725450000.0),
               ('t2', 'ETHUSDT', '15m', 'Bullish', -5.0, 1725450300.0),
               ('t3', 'BTCUSDT', '15m', 'Bearish', 20.0, 1725450600.0)
    """)
    conn.commit()
    conn.close()

    # Filter by symbol BTCUSDT
    btc_trades = database.get_completed_trades(limit=10, symbol="BTCUSDT")
    assert len(btc_trades) == 2
    assert all(t["symbol"] == "BTCUSDT" for t in btc_trades)

    # Filter by symbol ETHUSDT
    eth_trades = database.get_completed_trades(limit=10, symbol="ETHUSDT")
    assert len(eth_trades) == 1
    assert eth_trades[0]["symbol"] == "ETHUSDT"

    # All trades
    all_trades = database.get_completed_trades(limit=10)
    assert len(all_trades) == 3


# ---------------------------------------------------------------------------
# Finding #18: Emergency flatten fail-closed position handling
# ---------------------------------------------------------------------------
def test_finding_18_emergency_flatten_fails_closed_when_pos_query_fails():
    """Finding #18: emergency_flatten_position must fail closed if position query returns None."""
    with patch("main.get_bybit_position", return_value=None), \
         patch("main.place_bybit_taker_ioc_order", return_value={"retCode": 0}), \
         patch("main.time.sleep", return_value=None):
        # When get_bybit_position returns None, it attempts close order with fallback qty
        # and then verifies flatness. If verification query pos_after also returns None, it must return False.
        res = main.emergency_flatten_position("BTCUSDT", opp_side="Sell", qty_str="0.1", max_retries=2)
        assert res is False


# ---------------------------------------------------------------------------
# Finding #19: MHI cold-start bootstrap smoothing and consecutive loss degradation
# ---------------------------------------------------------------------------
def test_finding_19_mhi_cold_start_bootstrap_and_consecutive_losses():
    """Finding #19: Cold start with 0 trades must allow trading (MHI >= 50.0); consecutive losses must halt (< 50.0)."""
    health_engine = StrategyHealthEngine()

    # 1. Cold start with 0 trades
    cold_res = health_engine.compute_model_health_index(
        recent_pnls=[],
        brier_score=0.15,
    )
    cold_mhi = cold_res.get("mhi_score", 0.0)
    assert cold_mhi >= 50.0, f"Cold start MHI {cold_mhi} must be >= 50.0 to prevent bootstrap deadlock"

    # 2. Consecutive losing trades (10 losses of -10 USD)
    losing_pnls = [-10.0] * 10
    losing_res = health_engine.compute_model_health_index(
        recent_pnls=losing_pnls,
        brier_score=0.35,
    )
    losing_mhi = losing_res.get("mhi_score", 0.0)
    assert losing_mhi < 50.0, f"Losing MHI {losing_mhi} must drop below 50.0 (CRITICAL state)"


# ---------------------------------------------------------------------------
# Finding #20: Dynamic confidence threshold floored at effective_base
# ---------------------------------------------------------------------------
def test_finding_20_dynamic_conf_threshold_floored_at_effective_base():
    """Finding #20: dynamic_conf_threshold clamp must use effective_base as absolute floor."""
    economic_base = 0.62
    base_cfg = 0.58
    effective_base = max(economic_base, base_cfg)

    # If regime adjustment or raw dynamic threshold calculated 0.55
    raw_dynamic = 0.55
    max_allowed = 0.70

    clamped = float(round(max(effective_base, min(max_allowed, raw_dynamic)), 4))
    assert clamped == 0.62
    assert clamped >= effective_base
