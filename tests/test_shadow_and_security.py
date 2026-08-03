"""
Unit tests for Shadow Trading Engine and Security Operations Engine.
"""

import pytest
import os
import json
from shadow_trading_engine import shadow_trading_engine
from security_operations import security_operations_engine

def test_shadow_trading_engine_evaluation():
    record = shadow_trading_engine.evaluate_shadow_signal(
        candidate_model_id="xgboost_v19_candidate",
        symbol="BTCUSDT",
        direction="BUY",
        predicted_change_pct=0.018,
        calibrated_confidence=0.82
    )
    assert record["candidate_model_id"] == "xgboost_v19_candidate"
    assert record["execution_status"] == "SHADOW_SIMULATION"
    assert record["real_orders_placed"] is False


def test_shadow_promotion_gate_evaluation():
    c_history = [{"realized_pnl": 10.0} for _ in range(50)] + [{"realized_pnl": -5.0} for _ in range(20)]
    s_history = [{"realized_pnl": 12.0} for _ in range(80)] + [{"realized_pnl": -4.0} for _ in range(30)]

    res = shadow_trading_engine.evaluate_promotion_readiness(
        candidate_model_id="xgboost_v19_candidate",
        champion_trade_history=c_history,
        shadow_trade_history=s_history
    )
    assert "promotion_approved" in res
    assert "canary_stage" in res
    assert res["shadow_sample_size"] == 110


def test_canary_progressive_rollout_stages():
    stage, alloc, reasons = shadow_trading_engine.evaluate_canary_rollout_stage(
        current_stage="SHADOW", shadow_trades_count=120, is_statistically_approved=True
    )
    assert stage == "CANARY_5"
    assert alloc == 0.05

    # High slippage execution quality gate rejection
    stage_high_slip, alloc_slip, reasons_slip = shadow_trading_engine.evaluate_canary_rollout_stage(
        current_stage="CANARY_5", shadow_trades_count=250, is_statistically_approved=True, mean_slippage_bps=20.0
    )
    assert stage_high_slip == "SHADOW"
    assert alloc_slip == 0.00
    assert "Execution Quality Gate Failed" in reasons_slip[0]


def test_dependency_security_scan():
    audit_res = security_operations_engine.scan_dependency_vulnerabilities()
    assert audit_res["status"] == "PASS"
    assert "findings" in audit_res


def test_security_audit_logging():
    entry_hash = security_operations_engine.log_security_event(
        event_type="SECRET_ROTATION_AUDIT",
        details={"status": "SUCCESS", "rotated_key": "BYBIT_API_KEY"}
    )
    assert len(entry_hash) == 64  # SHA-256 length
    assert os.path.exists("security_audit.log")
