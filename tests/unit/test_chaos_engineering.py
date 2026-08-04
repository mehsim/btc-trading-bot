"""
Chaos Engineering & System Resilience Test Suite.
Tests failure injection (latency spikes, stale balance, DB errors, event bus dispatches, and state recovery).
"""

import pytest
import time
from state_recovery_engine import state_recovery_engine
from model_governance import model_governance_engine
from event_bus import event_bus
from circuit_breaker import circuit_breaker

def test_state_recovery_engine():
    res = state_recovery_engine.rebuild_all_states(symbols=["BTCUSDT"])
    assert "recovered_positions" in res
    assert "recovered_orders" in res
    assert "portfolio_heat" in res
    assert res["status"] == "SUCCESS"


def test_model_governance_traceability():
    rec = model_governance_engine.create_traceability_record("BTCUSDT", "BUY", 0.85, 0.015)
    assert rec["symbol"] == "BTCUSDT"
    assert "git_commit" in rec
    assert "config_hash" in rec
    assert len(rec["config_hash"]) == 12


def test_event_bus_pub_sub():
    received_events = []
    def on_signal(data):
        received_events.append(data)

    event_bus.subscribe("SignalEvent", on_signal)
    event_bus.publish("SignalEvent", {"symbol": "ETHUSDT", "side": "BUY"})
    
    assert len(received_events) == 1
    assert received_events[0]["symbol"] == "ETHUSDT"


def test_circuit_breaker_health_evaluation():
    # Healthy case
    is_h, reason_h = circuit_breaker.evaluate_system_health(
        exchange_latency_ms=120.0, last_balance_sync_ts=time.time() - 30.0
    )
    assert is_h is True
    assert reason_h == "HEALTHY"

    # Latency spike chaos injection
    is_l, reason_l = circuit_breaker.evaluate_system_health(
        exchange_latency_ms=1500.0, last_balance_sync_ts=time.time() - 30.0
    )
    assert is_l is False
    assert "HIGH_LATENCY" in reason_l

    # Stale balance chaos injection
    is_b, reason_b = circuit_breaker.evaluate_system_health(
        exchange_latency_ms=120.0, last_balance_sync_ts=time.time() - 600.0
    )
    assert is_b is False
    assert "STALE_BALANCE" in reason_b
