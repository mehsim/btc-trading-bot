import pytest
import time
import pandas as pd
from pain_feedback import PainFeedbackLoop

def test_pain_feedback_loop_registration():
    """Verify PainFeedbackLoop registers pain trades and returns effective stop floor."""
    loop = PainFeedbackLoop()
    
    # Register pain trade
    loop.register_pain_trade("BTCUSDT", entry_price=50000.0, exit_price=49500.0, take_profit=51000.0, current_floor=0.008)
    updated_floor = loop.get_effective_floor("BTCUSDT")
    assert updated_floor is not None and updated_floor > 0.008

def test_verify_pending_pain_trades():
    """Verify pending pain trades scanner processes items without errors."""
    loop = PainFeedbackLoop()
    
    class MockDB:
        @staticmethod
        def get_pending_pain_checks():
            return [
                {
                    "check_id": 1,
                    "symbol": "BTCUSDT",
                    "direction": "LONG",
                    "entry_price": 50000.0,
                    "exit_price": 49500.0,
                    "take_profit": 51000.0,
                    "exit_time": time.time() - 90000 # > 24 hours ago
                }
            ]
        @staticmethod
        def delete_pending_pain_check(check_id):
            pass

    def mock_fetch(symbol, interval, limit):
        return pd.DataFrame({
            "timestamp": [int((time.time() - 1800) * 1000)],
            "high": [52000.0],
            "low": [49000.0],
            "close": [51500.0]
        })

    loop.verify_pending_pain_trades(database_module=MockDB, fetch_kline_func=mock_fetch)
