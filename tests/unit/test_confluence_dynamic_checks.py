import unittest
import numpy as np
import pandas as pd
from signal_evaluator import SignalEvaluator

class TestConfluenceDynamicChecks(unittest.TestCase):
    def test_dynamic_confluence_checks_no_fake_pass(self):
        """Verify F-14: Confluence checks use real inputs and emit pass=None when streams are uncomputed."""
        bot_state = {}
        evaluator = SignalEvaluator(bot_state)

        df = pd.DataFrame({
            "close": [100.0] * 30,
            "volume": [1000.0] * 30,
            "RSI": [50.0] * 30,
            "EMA_9": [105.0] * 30,
            "EMA_21": [100.0] * 30,
            "EMA_200": [90.0] * 30,
            "BB_high": [110.0] * 30,
            "BB_low": [90.0] * 30,
            "ATR_norm": [0.01] * 30,
            "ADX": [25.0] * 30,
            "return_5m": [0.001] * 30,
            "volume_ratio": [1.2] * 30
        })

        evaluator.update_confluence_results("15m", df, "BTCUSDT")
        conf_res = bot_state.get("confluence_results_15m", {}).get("checks", {})

        self.assertIsNotNone(conf_res)
        # Check computed real values
        self.assertTrue(conf_res["1d_Trend"]["pass"])
        self.assertTrue(conf_res["BB_Edge_Guard"]["pass"])
        self.assertTrue(conf_res["Volatility_Guard"]["pass"])

        # Check uncomputed / unavailable streams explicitly return pass=None and 'Not Evaluated'
        self.assertIsNone(conf_res["Orderbook_Imbalance"]["pass"])
        self.assertIn("Not Evaluated", conf_res["Orderbook_Imbalance"]["detail"])

        self.assertIsNone(conf_res["News_Sentiment"]["pass"])
        self.assertIn("Not Evaluated", conf_res["News_Sentiment"]["detail"])

        self.assertIsNone(conf_res["Open_Interest_Delta"]["pass"])
        self.assertIn("Not Evaluated", conf_res["Open_Interest_Delta"]["detail"])

if __name__ == "__main__":
    unittest.main()
