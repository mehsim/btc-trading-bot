"""
tests/test_medium_severity_fixes.py
------------------------------------
Unit test suite covering Medium Severity Audit Remediations (MEDIUM-1 to MEDIUM-5):
- Config YAML Schema Loader & Validation (MEDIUM-2)
- Champion/Challenger Model Drift Rollback (MEDIUM-3)
- Immutable HMAC Audit Trail Logging (MEDIUM-5)
"""

import os
import pytest
import json
from config_validator import load_and_validate_config
from champion_challenger_framework import champion_challenger_framework
from audit_trail_logger import audit_logger


def test_config_loader_validation():
    cfg = load_and_validate_config("config.yaml")
    assert "trading" in cfg
    assert "risk" in cfg
    assert cfg["trading"]["min_position_size_usd"] > 0


def test_champion_challenger_rollback():
    # Severe drift (p-value 0.08 > threshold 0.05) triggers rollback
    active_model, reason = champion_challenger_framework.evaluate_model_health(
        drift_score=0.08, challenger_accuracy=0.85, champion_accuracy=0.75
    )
    assert active_model == "v2.4_prod"
    assert "Rollback" in reason


def test_immutable_audit_trail(tmp_path):
    log_file = os.path.join(tmp_path, "test_audit.jsonl")
    from audit_trail_logger import ImmutableAuditTrailLogger
    test_audit = ImmutableAuditTrailLogger(log_file=log_file)

    entry = test_audit.record_trade_audit_event("ENTRY", {"symbol": "BTCUSDT", "price": 100.0})
    assert "hmac_signature" in entry
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0
