"""
tests/test_auto_orchestrator.py
--------------------------------
Unit tests covering AutoSystemOrchestrator self-healing health checks
and automated maker/taker order routing optimization.
"""

import pytest
from auto_system_orchestrator import auto_orchestrator


def test_self_healing_health_check():
    res_healthy = auto_orchestrator.run_self_healing_health_check(ws_connected=True, rest_latency_ms=120.0)
    assert res_healthy["status"] == "HEALTHY"
    assert len(res_healthy["actions_taken"]) == 0

    res_degraded = auto_orchestrator.run_self_healing_health_check(ws_connected=False, rest_latency_ms=650.0)
    assert res_degraded["status"] == "DEGRADED"
    assert len(res_degraded["actions_taken"]) == 2


def test_optimal_order_type_routing():
    order_type_maker = auto_orchestrator.determine_optimal_order_type(spread_usd=0.1, atr_usd=5.0, ob_imbalance=0.7)
    assert order_type_maker == "POST_ONLY_LIMIT"

    order_type_twap = auto_orchestrator.determine_optimal_order_type(spread_usd=1.0, atr_usd=5.0, ob_imbalance=0.1)
    assert order_type_twap == "TWAP_MARKET"
