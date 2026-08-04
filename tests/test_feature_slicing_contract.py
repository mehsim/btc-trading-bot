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

    def test_positional_feature_mismatch_raises_runtime_error(self):
        """Verify F-12: Positional model feature count deficit raises RuntimeError (Fail-Closed)."""
        class MockPositionalModel:
            feature_names_ = [f"Column_{i}" for i in range(46)]

        model = MockPositionalModel()
        # Input DataFrame with only 35 features
        df = pd.DataFrame(np.ones((1, 35)))

        with self.assertRaises(RuntimeError) as cm:
            _slice_model_input(model, df)
        self.assertIn("Input vector feature count (35) does not match model expected features (46)", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
