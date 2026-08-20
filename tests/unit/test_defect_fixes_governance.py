import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

class TestDefectFixesGovernance(unittest.TestCase):
    def test_governance_leverage_hard_caps(self):
        from risk_limits import HARD_TIMEFRAME_MAX_LEVERAGE_CAPS, assert_risk_governance_invariants
        import config

        self.assertIn("15", HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        self.assertIn("60", HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)
        self.assertIn("240", HARD_TIMEFRAME_MAX_LEVERAGE_CAPS)

        self.assertEqual(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS["15"], 10.0)
        self.assertEqual(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS["60"], 5.0)
        self.assertEqual(HARD_TIMEFRAME_MAX_LEVERAGE_CAPS["240"], 3.0)

        self.assertTrue(assert_risk_governance_invariants(config))

    def test_sync_active_positions_syntax_and_structure(self):
        import main
        self.assertTrue(callable(main.sync_active_positions_from_bybit))

    def test_confluence_macro_guard_and_df_1h(self):
        from confluence_engine import check_pre_trade_confluence

        df_mock_1h = pd.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "open": [50000.0, 50100.0, 50200.0],
            "high": [50500.0, 50600.0, 50700.0],
            "low": [49900.0, 50000.0, 50100.0],
            "close": [50200.0, 50300.0, 50400.0],
            "volume": [100.0, 120.0, 110.0],
            "RSI": [55.0, 56.0, 57.0],
            "hours_to_news": [0.2, 0.2, 0.2]
        })

        all_pass, results, score = check_pre_trade_confluence(
            current_price=50400.0,
            df_1h=df_mock_1h,
            ml_trend="Bullish",
            news_sentiment="BLACKOUT",
            expected_pct_change=0.5,
            interval="15",
            symbol="BTCUSDT",
            current_regime="Trending"
        )
        # Macro blackout should fail the Regime_RSI_Guard
        self.assertIn("Regime_RSI_Guard", results)
        self.assertFalse(results["Regime_RSI_Guard"]["pass"])

    def test_shadow_exit_evaluations(self):
        from exit_policy_engine import exit_policy_engine

        mock_trade = {
            "trade_id": "test_trade_001",
            "symbol": "BTCUSDT",
            "direction": "Bullish",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "position_size_usd": 20.0,
            "leverage": 5.0,
            "atr_dollars": 500.0,
            "entry_time": 1000.0,
            "qty": 0.002
        }

        exit_policy_engine.shadow_configs["shadow_test_v1"] = {
            "parameters": {
                "TRENDING": {"stagnation_exit_hours": 2.0, "scale_out_pct": 0.3}
            }
        }

        exit_policy_engine.evaluate_shadow_simulations(
            active_trade=mock_trade,
            current_price=50500.0,
            current_time=2000.0,
            regime="TRENDING",
            adx_val=28.0,
            current_volume=100.0,
            avg_volume=120.0,
            swing_price=49800.0
        )

        shadow_evals = exit_policy_engine.get_shadow_evaluations()
        self.assertIsInstance(shadow_evals, dict)
        self.assertGreater(len(shadow_evals), 0)
        first_eval = shadow_evals.get("shadow_test_v1")
        self.assertIsNotNone(first_eval)
        self.assertIn("shadow_sl", first_eval)
        self.assertIn("active_trade_id", first_eval)
        self.assertEqual(first_eval["active_trade_id"], "test_trade_001")

    def test_live_vs_replay_checksum_divergence(self):
        from statistical_validation import StatisticalValidation
        sv = StatisticalValidation()

        feat_live = {"rsi": 55.0, "atr_norm": 0.015, "cvd": 120.5}
        feat_rep_exact = {"rsi": 55.0, "atr_norm": 0.015, "cvd": 120.5}
        feat_rep_diff = {"rsi": 62.0, "atr_norm": 0.015, "cvd": 120.5}

        res_match = sv.compute_live_vs_replay_checksum(feat_live, feat_rep_exact, live_prediction="Bullish", replay_prediction="Bullish")
        self.assertTrue(res_match["deterministic_match"])
        self.assertEqual(res_match["feature_divergence_max"], 0.0)

        res_diff = sv.compute_live_vs_replay_checksum(feat_live, feat_rep_diff, live_prediction="Bullish", replay_prediction="Neutral")
        self.assertFalse(res_diff["deterministic_match"])
        self.assertGreater(res_diff["feature_divergence_max"], 0.0)
        self.assertFalse(res_diff["prediction_match"])

    def test_mcc_bootstrap_ci_support(self):
        from statistical_validation import StatisticalValidation
        sv = StatisticalValidation()

        # Perfectly correlated
        y_true = np.array([0, 1] * 30)
        y_pred = np.array([0, 1] * 30)
        mean_mcc, low_mcc, high_mcc = sv.compute_mcc_bootstrap_ci(y_true, y_pred, num_samples=100, block_len=2)
        self.assertAlmostEqual(mean_mcc, 1.0, places=2)
        self.assertAlmostEqual(high_mcc, 1.0, places=2)

        # Inversely correlated
        y_inv = np.array([1, 0] * 30)
        mean_inv, low_inv, high_inv = sv.compute_mcc_bootstrap_ci(y_true, y_inv, num_samples=100, block_len=2)
        self.assertAlmostEqual(mean_inv, -1.0, places=2)
        self.assertAlmostEqual(high_inv, -1.0, places=2)

    def test_holdout_governance_floors_harmonization(self):
        from config import MODEL_GOVERNANCE, TIMEFRAME_MIN_HOLDOUT_MCC, TIMEFRAME_MIN_HOLDOUT_BAL_ACC

        for tf in ["15", "30", "60", "120", "240"]:
            mcc_fl = TIMEFRAME_MIN_HOLDOUT_MCC.get(str(tf), MODEL_GOVERNANCE.get("min_holdout_mcc", 0.02))
            bal_fl = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get(str(tf), MODEL_GOVERNANCE.get("min_holdout_balanced_accuracy", 0.34))
            self.assertGreater(mcc_fl, 0.0)
            self.assertGreater(bal_fl, 0.33)

if __name__ == "__main__":
    unittest.main()
