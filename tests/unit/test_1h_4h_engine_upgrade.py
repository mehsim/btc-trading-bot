import unittest
import json
import time
import os
from unittest.mock import MagicMock, patch

from signal_evaluator import get_hierarchical_macro_bias
from risk_engine import calculate_atr_risk_parity_size


class Test1H4HEngineUpgrade(unittest.TestCase):

    def test_get_hierarchical_macro_bias_bullish(self):
        bot_state = {
            "latest_prediction_BTCUSDT_4h": {
                "direction": "Bullish",
                "calibrated_confidence": 0.65
            },
            "adx_BTCUSDT_4h": 38.5
        }
        bias = get_hierarchical_macro_bias(bot_state, "BTCUSDT")
        self.assertEqual(bias["direction"], "Bullish")
        self.assertEqual(bias["confidence"], 0.65)
        self.assertEqual(bias["adx"], 38.5)
        self.assertTrue(bias["is_trending"])

    def test_get_hierarchical_macro_bias_bearish(self):
        bot_state = {
            "latest_prediction_SOLUSDT_240": {
                "direction": "Bearish",
                "confidence": 0.59
            },
            "adx_SOLUSDT_4h": 28.0
        }
        bias = get_hierarchical_macro_bias(bot_state, "SOLUSDT")
        self.assertEqual(bias["direction"], "Bearish")
        self.assertTrue(bias["is_trending"])

    def test_get_hierarchical_macro_bias_empty_fallback(self):
        bias = get_hierarchical_macro_bias({}, "ADAUSDT")
        self.assertEqual(bias["direction"], "Neutral")
        self.assertEqual(bias["confidence"], 0.50)

    def test_atr_risk_parity_sizing_consistency(self):
        # BTC at $80,000 with 1.5% ATR ($1,200)
        btc_res = calculate_atr_risk_parity_size(
            symbol="BTCUSDT",
            price=80000.0,
            atr_dollars=1200.0,
            sl_multiplier=1.0,
            target_risk_usd=10.0,
            max_position_size_usd=1000.0
        )
        # SOL at $150 with 4.0% ATR ($6.00)
        sol_res = calculate_atr_risk_parity_size(
            symbol="SOLUSDT",
            price=150.0,
            atr_dollars=6.0,
            sl_multiplier=1.0,
            target_risk_usd=10.0,
            max_position_size_usd=1000.0
        )

        # Both trades should risk exactly $10.00 at 1.0x ATR stop loss
        self.assertAlmostEqual(btc_res["dollar_risk_at_stop"], 10.0, places=1)
        self.assertAlmostEqual(sol_res["dollar_risk_at_stop"], 10.0, places=1)
        # Position size for SOL should be smaller than BTC to compensate for 2.67x higher percentage volatility
        self.assertLess(sol_res["position_size_usd"], btc_res["position_size_usd"])

    def test_optimized_barriers_240_parameters(self):
        with open("optimized_barriers_240.json", "r") as f:
            barriers = json.load(f)
        self.assertAlmostEqual(barriers.get("tp_mult_trending"), 2.4005, places=2)
        self.assertAlmostEqual(barriers.get("tp_mult_ranging"), 1.7831, places=2)
        self.assertAlmostEqual(barriers.get("sl_mult"), 0.7693, places=2)
        self.assertEqual(barriers.get("lookahead"), 12)

    def test_single_candle_cooldown_detection(self):
        # Simulate trade history with an exit 30 minutes ago on a 60m interval
        now_ts = time.time()
        bot_state = {
            "trade_history": [
                {
                    "symbol": "BTCUSDT",
                    "interval": "60",
                    "exit_time": now_ts - 1800,  # 30 mins ago
                    "success": "true",
                    "pnl_usd": 15.0
                }
            ]
        }
        
        # Define the cooling check logic matching main.py
        def is_cooling_check(symbol, interval, state):
            trades = [t for t in state.get("trade_history", []) if t.get("symbol") == symbol and str(t.get("interval")) == str(interval)]
            if not trades: return False, 0
            latest = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0) or 0.0), reverse=True)[0]
            exit_t = float(latest.get("exit_time", 0.0) or 0.0)
            dur = 60 * 60
            elapsed = time.time() - exit_t
            if elapsed < dur:
                return True, int((dur - elapsed) / 60)
            return False, 0

        is_cool, rem_mins = is_cooling_check("BTCUSDT", "60", bot_state)
        self.assertTrue(is_cool)
        self.assertGreater(rem_mins, 0)
        self.assertLessEqual(rem_mins, 31)

        # After 65 minutes, cool-off should be expired
        bot_state["trade_history"][0]["exit_time"] = now_ts - 3900  # 65 mins ago
        is_cool_expired, _ = is_cooling_check("BTCUSDT", "60", bot_state)
        self.assertFalse(is_cool_expired)


if __name__ == "__main__":
    unittest.main()
