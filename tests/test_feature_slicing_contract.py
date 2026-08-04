import unittest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ensemble import _slice_model_input, EnsembleClassifier

class TestFeatureSlicingContract(unittest.TestCase):
    def test_slice_by_exact_feature_names(self):
        """Verify F-12: _slice_model_input selects columns by exact name and order."""
        class MockNamedModel:
            feature_names_ = ["RSI", "EMA_9", "MACD_diff"]

        model = MockNamedModel()
        # Input DataFrame with extra columns and different column order
        df = pd.DataFrame({
            "MACD_diff": [0.5],
            "UNNEEDED_FEAT": [999.0],
            "RSI": [65.0],
            "EMA_9": [105.0]
        })

        sliced = _slice_model_input(model, df)
        self.assertEqual(list(sliced.columns), ["RSI", "EMA_9", "MACD_diff"])
        self.assertEqual(sliced["RSI"].iloc[0], 65.0)

    def test_missing_feature_raises_runtime_error(self):
        """Verify F-12: Missing required feature raises RuntimeError (Fail-Closed)."""
        class MockNamedModel:
            feature_names_ = ["RSI", "EMA_9", "MACD_diff"]

        model = MockNamedModel()
        df = pd.DataFrame({
            "RSI": [65.0],
            "EMA_9": [105.0]
        })

        with self.assertRaises(RuntimeError) as cm:
            _slice_model_input(model, df)
        self.assertIn("missing 1 required model features", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
