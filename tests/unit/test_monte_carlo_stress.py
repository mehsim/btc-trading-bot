import unittest
import numpy as np
import pandas as pd
from portfolio_risk import portfolio_risk_engine, PortfolioRiskEngine

class TestMonteCarloStressEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PortfolioRiskEngine()
        # Create dummy returns dataframe with correlated returns
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=100, freq="15min")
        btc_returns = np.random.normal(0.0001, 0.01, 100)
        eth_returns = 1.2 * btc_returns + np.random.normal(0, 0.005, 100)
        sol_returns = 1.5 * btc_returns + np.random.normal(0, 0.008, 100)
        
        self.returns_df = pd.DataFrame({
            "BTCUSDT": btc_returns,
            "ETHUSDT": eth_returns,
            "SOLUSDT": sol_returns
        }, index=dates)

        self.total_equity = 1000.0

    def test_run_monte_carlo_stress_test_empty(self):
        res = self.engine.run_monte_carlo_stress_test(
            open_positions=[],
            returns_df=self.returns_df,
            total_equity=self.total_equity,
            num_simulations=1000,
            shock_pct=-0.30
        )
        self.assertEqual(res["projected_stress_loss_usd"], 0.0)
        self.assertEqual(res["projected_stress_loss_pct"], 0.0)
        self.assertTrue(res["is_within_budget"])

    def test_run_monte_carlo_stress_test_single_position(self):
        positions = [
            {"symbol": "BTCUSDT", "position_size_usd": 100.0, "leverage": 2.0, "direction": "Bullish"}
        ]
        res = self.engine.run_monte_carlo_stress_test(
            open_positions=positions,
            returns_df=self.returns_df,
            total_equity=self.total_equity,
            num_simulations=2000,
            shock_pct=-0.30
        )
        # Exposure = $200. On a -30% shock, expected loss ~ $60 (6.0% of $1000 equity)
        self.assertGreater(res["projected_stress_loss_usd"], 40.0)
        self.assertLess(res["projected_stress_loss_usd"], 80.0)
        self.assertTrue(res["is_within_budget"])

    def test_candidate_stress_budget_approval(self):
        open_positions = [
            {"symbol": "BTCUSDT", "position_size_usd": 50.0, "leverage": 2.0, "direction": "Bullish"}
        ]
        # Moderate candidate size -> should be approved
        approved, scale_factor, loss_pct, summary = self.engine.check_candidate_stress_budget(
            candidate_symbol="ETHUSDT",
            candidate_size_usd=50.0,
            candidate_lev=2.0,
            candidate_direction="Bullish",
            open_positions=open_positions,
            returns_df=self.returns_df,
            total_equity=self.total_equity,
            max_stress_loss_pct=0.25,
            shock_pct=-0.30
        )
        self.assertTrue(approved)
        self.assertEqual(scale_factor, 1.0)
        self.assertLessEqual(loss_pct, 0.25)

    def test_candidate_stress_budget_downscaling_and_rejection(self):
        # Massive open position ($500 size @ 2x = $1000 leveraged exposure)
        open_positions = [
            {"symbol": "BTCUSDT", "position_size_usd": 400.0, "leverage": 2.0, "direction": "Bullish"}
        ]
        # Adding another large position ($400 @ 2x) will exceed 25% max stress loss budget ($250)
        approved, scale_factor, loss_pct, summary = self.engine.check_candidate_stress_budget(
            candidate_symbol="SOLUSDT",
            candidate_size_usd=400.0,
            candidate_lev=2.0,
            candidate_direction="Bullish",
            open_positions=open_positions,
            returns_df=self.returns_df,
            total_equity=self.total_equity,
            max_stress_loss_pct=0.25,
            shock_pct=-0.30
        )
        # Must recommend downscaling (scale_factor < 1.0)
        self.assertLess(scale_factor, 1.0)

    def test_short_position_produces_non_zero_stress_loss(self):
        """Verify F-06: Short positions produce non-zero stress loss under bilateral shock simulation."""
        short_positions = [
            {"symbol": "BTCUSDT", "position_size_usd": 100.0, "leverage": 10.0, "direction": "Bearish"}
        ]
        res = self.engine.run_monte_carlo_stress_test(
            open_positions=short_positions,
            returns_df=self.returns_df,
            total_equity=self.total_equity,
            num_simulations=2000,
            shock_pct=-0.30
        )
        # Exposure = $1000 short. On a +30% rally shock, expected loss ~ $300 (30% of $1000 equity)
        self.assertGreater(res["projected_stress_loss_usd"], 200.0)
        self.assertGreater(res["projected_stress_loss_pct"], 0.20)

    def test_deterministic_reproducibility_and_tail_loss_gating(self):
        """Verify F-07: Monte Carlo stress simulation is 100% deterministic with seed and outputs Stress CVaR 99.9%."""
        positions = [
            {"symbol": "BTCUSDT", "position_size_usd": 150.0, "leverage": 2.0, "direction": "Bullish"}
        ]
        
        # 1. Verify exact reproducibility when seed is specified
        res1 = self.engine.run_monte_carlo_stress_test(positions, self.returns_df, self.total_equity, num_simulations=2000, seed=42)
        res2 = self.engine.run_monte_carlo_stress_test(positions, self.returns_df, self.total_equity, num_simulations=2000, seed=42)
        self.assertEqual(res1["projected_stress_loss_usd"], res2["projected_stress_loss_usd"])
        self.assertEqual(res1["stress_cvar_999_usd"], res2["stress_cvar_999_usd"])

        # 2. Verify Stress CVaR 99.9% tail loss is reported and strictly >= mean projected loss
        self.assertGreaterEqual(res1["stress_cvar_999_pct"], res1["projected_stress_loss_pct"])

if __name__ == "__main__":
    unittest.main()
