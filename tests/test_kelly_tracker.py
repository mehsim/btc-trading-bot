import unittest
import os
import json
from kelly_tracker import KellyTracker

class TestKellyTracker(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_kelly_history.json"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        self.tracker = KellyTracker(data_file=self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_insufficient_trades_returns_zero(self):
        """Verify F-08: Insufficient trade history returns 0.0 fraction."""
        for i in range(5):
            self.tracker.log_trade("BTCUSDT", "15m", 10.0, 0.02)
        kf = self.tracker.compute_kelly_fraction(timeframe="15m", min_trades=30)
        self.assertEqual(kf, 0.0)

    def test_losing_history_returns_zero(self):
        """Verify F-08: Negative edge (losing trade history) returns 0.0 fraction."""
        # Log 35 losing trades (30% win rate, 1.0 R-ratio -> negative Kelly)
        for i in range(10):
            self.tracker.log_trade("BTCUSDT", "15m", 10.0, 0.01)   # 10 wins (+1%)
        for i in range(25):
            self.tracker.log_trade("BTCUSDT", "15m", -10.0, -0.01) # 25 losses (-1%)

        kf = self.tracker.compute_kelly_fraction(timeframe="15m", min_trades=30)
        self.assertEqual(kf, 0.0)

    def test_winning_history_returns_positive_fraction(self):
        """Verify positive edge produces positive Kelly fraction capped at max_kelly_cap."""
        # Log 35 winning trades (70% win rate, 1.5 R-ratio -> strong positive Kelly)
        for i in range(25):
            self.tracker.log_trade("BTCUSDT", "15m", 15.0, 0.015)  # 25 wins (+1.5%)
        for i in range(10):
            self.tracker.log_trade("BTCUSDT", "15m", -10.0, -0.01) # 10 losses (-1.0%)

        kf = self.tracker.compute_kelly_fraction(timeframe="15m", min_trades=30, max_kelly_cap=0.25)
        self.assertGreater(kf, 0.0)
        self.assertLessEqual(kf, 0.25)

if __name__ == "__main__":
    unittest.main()
