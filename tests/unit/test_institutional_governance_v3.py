import unittest
import os
import json
import numpy as np
import pandas as pd
from mlops_engine import evaluate_champion_challenger_promotion, calculate_psi_per_feature
from ensemble import write_model_manifest
from config import MODEL_GOVERNANCE, SUPPORTED_MANIFEST_SCHEMA_VERSION
from risk_engine import joint_risk_budget_allocator

class TestInstitutionalGovernanceV3(unittest.TestCase):
    def setUp(self):
        self.prefix = "test_gov_v3_model"

    def tearDown(self):
        for ext in ["_xgb.json", "_weights.json", "_manifest.json"]:
            path = f"{self.prefix}{ext}"
            if os.path.exists(path):
                os.remove(path)

    def test_champion_challenger_promotion_gate(self):
        """Verify Item 1 & 2: Promotion requires Sharpe delta > 0.10, ECE <= 0.08, and Brier <= 0.22."""
        champ_metrics = {"sharpe": 1.20, "ece": 0.05, "brier": 0.18}
        
        # 1. Failing ECE challenger
        chal_high_ece = {"sharpe": 1.50, "ece": 0.09, "brier": 0.18}
        promoted, reason, metrics = evaluate_champion_challenger_promotion(champ_metrics, chal_high_ece)
        self.assertFalse(promoted)
        self.assertIn("ECE", reason)

        # 2. Failing Brier challenger
        chal_high_brier = {"sharpe": 1.50, "ece": 0.04, "brier": 0.25}
        promoted, reason, metrics = evaluate_champion_challenger_promotion(champ_metrics, chal_high_brier)
        self.assertFalse(promoted)
        self.assertIn("Brier", reason)

        # 3. Insufficient Sharpe delta
        chal_low_sharpe = {"sharpe": 1.25, "ece": 0.04, "brier": 0.18}
        promoted, reason, metrics = evaluate_champion_challenger_promotion(champ_metrics, chal_low_sharpe)
        self.assertFalse(promoted)
        self.assertIn("Sharpe delta", reason)

        # 4. Winning challenger
        chal_winner = {"sharpe": 1.45, "ece": 0.04, "brier": 0.18, "balanced_accuracy": 0.58}
        promoted, reason, metrics = evaluate_champion_challenger_promotion(champ_metrics, chal_winner)
        self.assertTrue(promoted)
        self.assertIn("balanced_accuracy", metrics)
        self.assertEqual(metrics["ece"], 0.04)

    def test_manifest_schema_v3_writing_and_lineage(self):
        """Verify Item 1, 9 & Lineage: Manifest schema version 3, VIF values, and lineage ancestry."""
        feats = ["RSI", "EMA_9", "MACD_diff"]
        vifs = {"RSI": 1.2, "EMA_9": 2.1, "MACD_diff": 1.1}
        write_model_manifest(
            self.prefix,
            feature_names=feats,
            vif_values=vifs,
            parent_model_hash="parent_sha_123",
            promotion_reason="Passed quality gate Sharpe delta +0.25"
        )

        manifest_file = f"{self.prefix}_manifest.json"
        self.assertTrue(os.path.exists(manifest_file))

        with open(manifest_file, "r") as f:
            m = json.load(f)

        self.assertEqual(m.get("manifest_schema_version"), SUPPORTED_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(m.get("parent_model_hash"), "parent_sha_123")
        self.assertEqual(m.get("vif_values"), vifs)
        self.assertIn("feature_pipeline_hash", m)
        self.assertEqual(m.get("governance_policy"), MODEL_GOVERNANCE)

    def test_per_feature_psi_calculation(self):
        """Verify Item 7: Per-feature PSI returns status dictionary and handles sparse windows."""
        np.random.seed(42)
        df_base = pd.DataFrame({"RSI": np.random.normal(50, 5, 50), "ATR": np.random.normal(10, 1, 50)})
        df_curr = pd.DataFrame({"RSI": np.random.normal(50, 5, 50), "ATR": np.random.normal(20, 1, 50)})

        res = calculate_psi_per_feature(df_base, df_curr, ["RSI", "ATR"])
        self.assertIn("RSI", res)
        self.assertIn("ATR", res)
        self.assertEqual(res["RSI"]["n_baseline"], 50)

        # Test sparse window (< 20)
        df_sparse = pd.DataFrame({"RSI": [50.0] * 10, "ATR": [10.0] * 10})
        res_sparse = calculate_psi_per_feature(df_base, df_sparse, ["RSI", "ATR"])
        self.assertEqual(res_sparse["RSI"]["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(res_sparse["RSI"]["psi"])

    def test_risk_gate_results_struct_logging(self):
        """Verify Item 5: Risk engine returns nested risk_gate_results object."""
        alloc = joint_risk_budget_allocator.allocate_risk_budget(
            symbol="BTCUSDT",
            entry_price=50000.0,
            atr_dollars=500.0,
            atr_norm=0.01,
            calibrated_confidence=0.75,
            direction="Bullish",
            total_equity=10000.0,
            portfolio_heat=0.05,
            mhi_score=90.0,
            top_book_depth_usd=250000.0
        )
        self.assertIn("risk_gate_results", alloc)
        gates = alloc["risk_gate_results"]
        self.assertIn("VaR", gates)
        self.assertIn("Stress", gates)
        self.assertIn("Heat", gates)
        self.assertIn("Kelly", gates)
        self.assertTrue(gates["VaR"]["pass"])


if __name__ == "__main__":
    unittest.main()
