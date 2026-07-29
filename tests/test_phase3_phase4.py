"""
tests/test_phase3_phase4.py
---------------------------
Unit test suite covering Phase 3 (Async Ingestion, Model Serving, TWAP OMS)
and Phase 4 (Order Flow Imbalance OFI, Risk Parity Allocator) modules.
"""

import pytest
import numpy as np
from services.data_ingestion import async_data_service
from services.model_serving import model_serving_engine
from order_flow_analyzer import order_flow_analyzer
from risk_parity_allocator import risk_parity_allocator
from twap_execution_engine import twap_engine


def test_order_flow_imbalance_calculation():
    bids = [[100.0, 5.0], [99.9, 3.0]]
    asks = [[100.1, 1.0], [100.2, 1.0]]
    ofi_score = order_flow_analyzer.compute_ofi_delta(bids, asks)
    assert ofi_score > 0.0  # Buyer pressure > Seller pressure
    assert -1.0 <= ofi_score <= 1.0


def test_risk_parity_weights():
    vols = {
        "BTCUSDT": 0.02, # Low volatility
        "ETHUSDT": 0.03,
        "SOLUSDT": 0.06  # High volatility
    }
    weights = risk_parity_allocator.compute_risk_parity_weights(vols)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["BTCUSDT"] > weights["SOLUSDT"]  # Low vol asset gets higher risk parity weight


def test_twap_execution_slicing(monkeypatch):
    executed_slices = []
    def dummy_place_order(symbol, side, qty, sl=None, tp=None, reduce_only=False, order_type="Market", post_only=False):
        executed_slices.append(qty)
        return {"retCode": 0, "result": {"orderId": "test_twap_123"}}

    monkeypatch.setattr("twap_execution_engine.place_bybit_order", dummy_place_order)
    resp = twap_engine.execute_twap_order("BTCUSDT", "Buy", total_qty=1.0, num_slices=2)
    assert resp.get("retCode") == 0
    assert len(executed_slices) == 2
    assert sum(executed_slices) == pytest.approx(1.0)
