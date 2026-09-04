import time
import pytest
from unittest.mock import patch, MagicMock
from main import execute_bybit_order_ws_or_rest, get_symbol_order_lock

def test_order_execution_routes_direct_to_rest():
    """Verify execute_bybit_order_ws_or_rest calls bybit_post_request directly with zero WS timeout latency."""
    mock_payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.01"}
    
    with patch("main.bybit_post_request", return_value={"retCode": 0, "result": {"orderId": "test_123"}}) as mock_post:
        start_time = time.time()
        resp = execute_bybit_order_ws_or_rest("/v5/order/create", mock_payload)
        elapsed = time.time() - start_time
        
        # Verify it completed in milliseconds (not 2.0s WS timeout)
        assert elapsed < 0.1
        assert mock_post.called
        assert resp["retCode"] == 0
        # Verify orderLinkId was injected
        assert "orderLinkId" in mock_payload
        assert mock_payload["orderLinkId"].startswith("B_BTC_") or mock_payload["orderLinkId"].startswith("cl_BTCUSDT_")

def test_per_symbol_order_locks_are_isolated():
    """Verify that different symbols receive distinct lock instances."""
    lock_btc = get_symbol_order_lock("BTCUSDT")
    lock_eth = get_symbol_order_lock("ETHUSDT")
    lock_btc2 = get_symbol_order_lock("btcusdt")
    
    assert lock_btc is not lock_eth
    assert lock_btc is lock_btc2
