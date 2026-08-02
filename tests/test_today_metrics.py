import sys
import unittest
import time
from unittest.mock import MagicMock, patch

# Mock redis module if not installed in local python environment
if "redis" not in sys.modules:
    mock_redis_mod = MagicMock()
    sys.modules["redis"] = mock_redis_mod

class TestTodayMetrics(unittest.TestCase):
    def setUp(self):
        try:
            from dashboard_routes import clear_endpoint_cache
            clear_endpoint_cache()
        except Exception:
            pass

    def test_today_metrics_with_mixed_trades(self):
        """Verify quick_status today metrics with 1 win and 1 loss today."""
        now = time.time()
        
        # Today trades: 2 trades (1 win: +10 USD, 1 loss: -5 USD) => today_win_rate = 50%, today_pf = 2.0
        today_trade_1 = {"exit_time": now - 3600, "pnl_usd": 10.0, "position_size_usd": 50.0, "leverage": 10.0}
        today_trade_2 = {"exit_time": now - 1800, "pnl_usd": -5.0, "position_size_usd": 50.0, "leverage": 10.0}
        
        # Older historical trades (10 losses: -10 USD each)
        old_trades = [
            {"exit_time": now - (86400 * 5 + i * 3600), "pnl_usd": -10.0, "position_size_usd": 50.0, "leverage": 10.0}
            for i in range(10)
        ]
        
        history = [today_trade_1, today_trade_2] + old_trades
        
        with patch("state_manager.state_manager.get") as mock_get:
            def side_effect(key, default=None):
                if key == "trade_history":
                    return history
                return default
            mock_get.side_effect = side_effect
            
            from dashboard_routes import api_institutional_summary
            from flask import Flask
            app = Flask(__name__)
            with app.app_context():
                res = api_institutional_summary()
                data = res.get_json()
                
                qs = data.get("quick_status", {})
                self.assertEqual(qs.get("today_trades_count"), 2)
                self.assertEqual(qs.get("today_win_rate_pct"), 50.0)
                self.assertEqual(qs.get("today_pf"), 2.0)
                self.assertEqual(qs.get("today_pnl_usd"), 5.0)

    def test_today_metrics_user_scenario(self):
        """Verify user screenshot scenario: 2 trades today, 1 win (+0.50) and 1 loss (-0.84)."""
        now = time.time()
        
        today_trade_1 = {"exit_time": now - 3600, "pnl_usd": 0.50, "position_size_usd": 30.0, "leverage": 2.0}
        today_trade_2 = {"exit_time": now - 1800, "pnl_usd": -0.84, "position_size_usd": 31.85, "leverage": 2.0}
        
        old_trades = [
            {"exit_time": now - (86400 * 3 + i * 3600), "pnl_usd": 1.0, "position_size_usd": 20.0, "leverage": 1.0}
            for i in range(80)
        ]
        
        history = [today_trade_1, today_trade_2] + old_trades
        
        with patch("state_manager.state_manager.get") as mock_get:
            def side_effect(key, default=None):
                if key == "trade_history":
                    return history
                return default
            mock_get.side_effect = side_effect
            
            from dashboard_routes import api_institutional_summary
            from flask import Flask
            app = Flask(__name__)
            with app.app_context():
                res = api_institutional_summary()
                data = res.get_json()
                
                qs = data.get("quick_status", {})
                self.assertEqual(qs.get("today_trades_count"), 2)
                self.assertEqual(qs.get("today_pnl_usd"), -0.34)
                self.assertEqual(qs.get("today_win_rate_pct"), 50.0)
                self.assertEqual(qs.get("today_pf"), 0.60)  # 0.50 / 0.84 = 0.595 -> 0.60

if __name__ == "__main__":
    unittest.main()
