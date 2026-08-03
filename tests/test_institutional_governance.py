"""
Unit tests for Institutional Security Scope Classification, Operational Analytics Metadata, and Market Data Quality.
"""

import pytest
import time
from security_operations import security_operations_engine
from market_data_quality import market_data_quality_monitor

def test_scope_classified_dependency_scan():
    res = security_operations_engine.scan_dependency_vulnerabilities()
    assert res["status"] == "PASS"
    for item in res["findings"]:
        assert "scope" in item
        assert item["scope"] in ("CRITICAL_TRADING", "OPTIONAL_RUNTIME", "DEV_ONLY")


def test_operational_analytics_measurement_metadata():
    ops = security_operations_engine.compute_operational_analytics()
    assert "measurement_metadata" in ops
    meta = ops["measurement_metadata"]
    assert meta["environment_scope"] == "AWS_SINGAPORE_PRODUCTION"
    assert meta["data_source"] == "SECURITY_AUDIT_LOG_STREAM"


def test_market_data_quality_monitor_and_4tier_health():
    now_ms = time.time() * 1000.0
    
    # GREEN tier test
    green_res = market_data_quality_monitor.evaluate_feed_health(
        last_candle_timestamp=time.time() - 10.0,
        server_time_ms=now_ms,
        client_time_ms=now_ms - 20.0,
        seq_num=101
    )
    assert green_res["health_tier"] == "GREEN"
    assert green_res["trading_allowed"] is True

    # Explainable confidence decay test
    decay_audit = market_data_quality_monitor.apply_explainable_confidence_decay(
        raw_confidence=0.82, feed_health=green_res
    )
    assert decay_audit["raw_confidence"] == 0.82
    assert decay_audit["decayed_confidence"] == 0.82
    assert "GREEN" in decay_audit["decay_reasons"][0]

    # RED tier test (Severe stale data)
    red_res = market_data_quality_monitor.evaluate_feed_health(
        last_candle_timestamp=time.time() - 500.0,
        server_time_ms=now_ms,
        client_time_ms=now_ms - 20.0,
        seq_num=102
    )
    assert red_res["health_tier"] == "RED"
    assert red_res["trading_allowed"] is False
