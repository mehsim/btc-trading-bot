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


def test_market_data_quality_monitor():
    now_ms = time.time() * 1000.0
    res = market_data_quality_monitor.evaluate_feed_health(
        last_candle_timestamp=time.time() - 10.0,
        server_time_ms=now_ms,
        client_time_ms=now_ms - 20.0,
        seq_num=101
    )
    assert res["feed_status"] == "HEALTHY"
    assert res["sequence_gaps"] == 0
    assert res["duplicate_messages"] == 0
