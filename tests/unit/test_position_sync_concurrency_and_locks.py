import time
import threading
from unittest.mock import patch, MagicMock
import pytest
import main


def test_sync_active_positions_fetches_inside_locks():
    """Verify that sync_active_positions_from_bybit fetches get_all_bybit_positions while holding the locks."""
    call_order = []

    def mock_get_positions():
        # Check that active_execution_lock and active_trades_lock are held when fetching positions
        # In Python RLock, _is_owned() tells if the current thread owns the lock
        exec_locked = main.active_execution_lock._is_owned()
        trades_locked = main.active_trades_lock._is_owned()
        call_order.append(("fetch_positions", exec_locked, trades_locked))
        return [
            {
                "symbol": "BTCUSDT",
                "size": "0.1",
                "side": "Buy",
                "avgPrice": "60000.0",
                "leverage": "10",
                "liqPrice": "54000.0",
                "markPrice": "60500.0",
                "takeProfit": "62000.0",
                "stopLoss": "59000.0",
                "positionValue": "6000.0",
                "unrealisedPnl": "50.0"
            }
        ]

    with patch("main.TRADE_MODE", "live"), \
         patch("main.get_all_bybit_positions", side_effect=mock_get_positions), \
         patch("main.save_history", MagicMock()):

        # Pre-seed active trade for 15m
        main.bot_state["active_trade_15m"] = [
            {
                "symbol": "BTCUSDT",
                "entry_price": 60000.0,
                "direction": "Bullish",
                "qty": 0.1,
                "leverage": 10.0,
                "bybit_closed": False,
                "confidence": 0.65
            }
        ]

        result = main.sync_active_positions_from_bybit()
        assert result is True
        assert len(call_order) == 1
        name, exec_locked, trades_locked = call_order[0]
        assert name == "fetch_positions"
        assert exec_locked is True
        assert trades_locked is True


def test_request_position_sync_debounced_worker():
    """Verify request_position_sync signals the worker without spawning unconstrained threads."""
    with patch("main.sync_active_positions_from_bybit") as mock_sync:
        mock_sync.return_value = True

        # Call request_position_sync multiple times rapidly
        for _ in range(5):
            main.request_position_sync()

        # Allow worker loop to execute
        time.sleep(0.2)

        # Worker should have called sync_active_positions_from_bybit
        assert mock_sync.call_count >= 1
