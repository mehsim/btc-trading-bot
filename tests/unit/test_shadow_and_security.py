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
        current_stage="SHADOW", shadow_trades_count=120, is_statistically_approved=True, regimes_encountered=["TRENDING", "RANGING"]
    )
    assert stage == "CANARY_5"
    assert alloc == 0.05

    # Adaptive Slippage Calculation test
    ad_slip = shadow_trading_engine.calculate_adaptive_slippage_limit(atr_norm=0.0045, spread_pct=0.0004, ob_depth_usd=80000.0)
    assert 8.0 <= ad_slip <= 35.0

    # High CVaR tail risk rejection test
    stage_cvar, alloc_cvar, reasons_cvar = shadow_trading_engine.evaluate_canary_rollout_stage(
        current_stage="CANARY_5", shadow_trades_count=250, is_statistically_approved=True, candidate_cvar_99=0.15, champion_cvar_99=0.06
    )
    assert stage_cvar == "SHADOW"
    assert alloc_cvar == 0.00
    assert "CVaR Tail Risk Gate Failed" in reasons_cvar[0]


def test_dependency_security_scan():
    audit_res = security_operations_engine.scan_dependency_vulnerabilities()
    assert audit_res["status"] == "PASS"
    assert "findings" in audit_res
    assert "blocked_deployment" in audit_res


def test_secret_rotation_and_versioning():
    rot_res = security_operations_engine.rotate_secret_with_versioning("BYBIT_API_SECRET", "sz8921hd9ahsdj1298ahs1")
    assert rot_res["rotation_status"] == "SUCCESS"
    assert "version_id" in rot_res


def test_security_audit_logging():
    entry_hash = security_operations_engine.log_security_event(
        event_type="SECRET_ROTATION_AUDIT",
        details={"status": "SUCCESS", "rotated_key": "BYBIT_API_KEY"}
    )
    assert len(entry_hash) == 64  # SHA-256 length
    assert os.path.exists("security_audit.log")
