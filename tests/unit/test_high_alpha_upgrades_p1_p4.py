import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from bybit_client import get_orderbook_imbalance, calculate_optimal_maker_price
from portfolio_risk import portfolio_risk_engine
from risk_engine import calculate_anti_martingale_risk_multiplier
from features import add_features


class TestHighAlphaUpgradesP1P4(unittest.TestCase):

    def test_calculate_optimal_maker_price_buy_side(self):
        # Case 1: Strong buying wall (OBI = +0.60) -> Place directly at best_bid
        obi_bullish = {
            "best_bid": 100.0,
            "best_ask": 100.05,
            "obi": 0.60,
            "status": "OK"
        }
        p_buy = calculate_optimal_maker_price("BTCUSDT", "Buy", obi_data=obi_bullish)
        self.assertEqual(p_buy, 100.0)

        # Case 2: Heavy selling pressure (OBI = -0.50) -> Place 1 tick deeper for safety
        obi_bearish = {
            "best_bid": 100.0,
            "best_ask": 100.05,
            "obi": -0.50,
            "status": "OK"
        }
        p_buy_deep = calculate_optimal_maker_price("BTCUSDT", "Buy", obi_data=obi_bearish)
        self.assertLess(p_buy_deep, 100.0)

    def test_calculate_optimal_maker_price_sell_side(self):
        # Sell side with heavy asks (OBI = -0.60) -> Place at best_ask
        obi_bearish = {
            "best_bid": 100.0,
            "best_ask": 100.05,
            "obi": -0.60,
            "status": "OK"
        }
        p_sell = calculate_optimal_maker_price("BTCUSDT", "Sell", obi_data=obi_bearish)
        self.assertEqual(p_sell, 100.05)

    def test_check_correlated_cluster_exposure(self):
        # Existing open positions: 2 Layer-1 Longs (SOL + AVAX)
        open_positions = [
            {"symbol": "SOLUSDT", "direction": "Bullish"},
            {"symbol": "AVAXUSDT", "direction": "Bullish"}
        ]

        # Attempting a 3rd Layer-1 Long (ADA) -> MUST BE REJECTED
        approved, reason = portfolio_risk_engine.check_correlated_cluster_exposure(
            candidate_symbol="ADAUSDT",
            candidate_direction="Bullish",
            open_positions=open_positions,
            max_same_cluster_count=2
        )
        self.assertFalse(approved)
        self.assertIn("CLUSTER_EXPOSURE_LIMIT", reason)

        # Attempting a Non-L1 Long (BTC) -> MUST BE APPROVED
        approved_btc, _ = portfolio_risk_engine.check_correlated_cluster_exposure(
            candidate_symbol="BTCUSDT",
            candidate_direction="Bullish",
            open_positions=open_positions,
            max_same_cluster_count=2
        )
        self.assertTrue(approved_btc)

        # Attempting an L1 Short (ADA Short) while others are Long -> MUST BE APPROVED (Hedging/Uncorrelated direction)
        approved_short, _ = portfolio_risk_engine.check_correlated_cluster_exposure(
            candidate_symbol="ADAUSDT",
            candidate_direction="Bearish",
            open_positions=open_positions,
            max_same_cluster_count=2
        )
        self.assertTrue(approved_short)

    def test_anti_martingale_risk_scaling(self):
        # Case 1: Hot streak at high-watermark (< 1.5% DD, 4 wins in last 5)
        recent_wins = [{"success": "true", "pnl_usd": 10.0}] * 4 + [{"success": "false", "pnl_usd": -5.0}]
        hot_res = calculate_anti_martingale_risk_multiplier(
            current_equity=110.0,
            peak_equity=110.0,
            recent_trades=recent_wins
        )
        self.assertGreaterEqual(hot_res["multiplier"], 1.25)
        self.assertEqual(hot_res["regime"], "HOT_STREAK_COMPOUNDING")

        # Case 2: Drawdown (> 5% off ATH) -> Cut risk to 0.50x
        dd_res = calculate_anti_martingale_risk_multiplier(
            current_equity=94.0,
            peak_equity=100.0,
            recent_trades=[]
        )
        self.assertEqual(dd_res["multiplier"], 0.50)
        self.assertEqual(dd_res["regime"], "SEVERE_DRAWDOWN_DEFENSE")

        # Case 3: Standard normal trading (1.0% DD)
        std_res = calculate_anti_martingale_risk_multiplier(
            current_equity=99.0,
            peak_equity=100.0,
            recent_trades=[]
        )
        self.assertEqual(std_res["multiplier"], 1.00)
        self.assertEqual(std_res["regime"], "STANDARD_RISK")

    def test_cross_asset_lead_lag_indicators(self):
        # Build synthetic DataFrame with close and close_btc
        n = 250
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "timestamp": [int(d.timestamp() * 1000) for d in dates],
            "open": np.linspace(100, 150, n),
            "high": np.linspace(101, 152, n),
            "low": np.linspace(99, 148, n),
            "close": np.linspace(100, 150, n),
            "volume": np.full(n, 1000.0),
            "close_btc": np.linspace(50000, 75000, n),
            "symbol": ["SOLUSDT"] * n
        })

        df_feats = add_features(df, symbol="SOLUSDT", interval="15")
        self.assertIn("btc_surge_5m", df_feats.columns)
        self.assertIn("btc_surge_15m", df_feats.columns)
        self.assertFalse(df_feats["btc_surge_5m"].isna().any())


if __name__ == "__main__":
    unittest.main()
