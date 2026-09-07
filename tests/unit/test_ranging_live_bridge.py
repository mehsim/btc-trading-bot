import unittest
import numpy as np
import pandas as pd
from ranging_strategy import evaluate_ranging_mean_reversion, RangingSignal


class TestRangingLiveBridge(unittest.TestCase):
    """Verifies that Option B ranging signals cleanly bridge into execution geometry."""

    def test_ranging_bridge_signal_generation_bullish(self):
        """Verify bullish BB fade emits actionable execution payload."""
        np.random.seed(42)
        closes = [100.0] * 30
        highs = [101.0] * 30
        lows = [99.0] * 30
        opens = [100.0] * 30
        
        # Penultimate bars create standard range
        df = pd.DataFrame({
            "timestamp": [1700000000.0 + i * 900 for i in range(30)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * 30,
            "RSI": [50.0] * 29 + [30.0],
            "ATR": [1.0] * 30,
            "ADX": [15.0] * 30,
            "lower_wick_volume_ratio": [1.0] * 29 + [1.8]
        })
        # Force last bar to pierce lower band
        df.loc[29, "low"] = 97.0
        df.loc[29, "close"] = 98.0
        df.loc[29, "open"] = 98.0
        df.loc[29, "high"] = 98.5

        sig = evaluate_ranging_mean_reversion(df, symbol="BTCUSDT", interval="15")
        self.assertTrue(sig.is_signal)
        self.assertEqual(sig.direction, "Bullish")
        self.assertGreaterEqual(sig.confidence, 0.60)
        self.assertLessEqual(sig.confidence, 0.72)
        self.assertGreater(sig.take_profit, sig.entry_price)
        self.assertLess(sig.stop_loss, sig.entry_price)
        self.assertEqual(sig.order_type, "POST_ONLY_LIMIT")

    def test_ranging_bridge_abstain_inside_bands(self):
        """Verify ranging engine explicitly abstains when price is inside mid-range."""
        df = pd.DataFrame({
            "timestamp": [1700000000.0 + i * 900 for i in range(30)],
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.0] * 30,
            "volume": [1000.0] * 30,
            "RSI": [50.0] * 30,
            "ATR": [1.0] * 30,
            "ADX": [15.0] * 30
        })
        sig = evaluate_ranging_mean_reversion(df, symbol="BTCUSDT", interval="15")
        self.assertFalse(sig.is_signal)
        self.assertIn("Inside Bollinger Mid-Range", sig.reason)


if __name__ == "__main__":
    unittest.main()
