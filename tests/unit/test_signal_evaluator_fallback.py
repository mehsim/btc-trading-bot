import unittest
import numpy as np
import pandas as pd
from signal_evaluator import SignalEvaluator

class TestSignalEvaluatorFallback(unittest.TestCase):
    def setUp(self):
        self.bot_state = {}
        self.evaluator = SignalEvaluator(self.bot_state)

    def test_fallback_signal_provenance_and_confidence_cap(self):
        """Verify F-10: ML model failure emits signal_source='RULE_BASED_FALLBACK', is_fallback=True, and caps confidence at 0.55."""
        # Setup mock bot_state with dummy models that will raise Exception on predict_proba
        class FailingModel:
            def predict_proba(self, X):
                raise RuntimeError("Simulated ML model failure")

        self.evaluator.models_by_interval = {
            "15": {
                "trending": {"trend": FailingModel(), "price": FailingModel()},
                "ranging": {"trend": FailingModel(), "price": FailingModel()}
            }
        }

        # Create dummy candle dataframe with >215 rows
        np.random.seed(42)
        base_ts = 1700000000.0
        df = pd.DataFrame({
            "timestamp": [float(base_ts + i * 900) for i in range(260)],
            "open": [100.0 + i * 0.1 for i in range(260)],
            "high": [102.0 + i * 0.1 for i in range(260)],
            "low": [98.0 + i * 0.1 for i in range(260)],
            "close": [100.0 + i * 0.1 for i in range(260)],
            "volume": [1000.0 for _ in range(260)],
            "RSI": [60.0 for _ in range(260)],
            "EMA_9": [105.0 for _ in range(260)],
            "EMA_21": [100.0 for _ in range(260)],
            "ADX": [25.0 for _ in range(260)]
        })

        # Mock get_history to return dummy df
        import signal_evaluator
        orig_get_history = signal_evaluator.get_history
        signal_evaluator.get_history = lambda symbol, interval, limit: df

        try:
            self.evaluator.evaluate_interval("BTCUSDT", "15")
            pred = self.bot_state.get("latest_prediction_15m") or self.bot_state.get("latest_prediction_15")
            self.assertIsNotNone(pred)
            self.assertEqual(pred.get("signal_source"), "RULE_BASED_FALLBACK")
            self.assertTrue(pred.get("is_fallback"))
            self.assertLessEqual(pred.get("confidence"), 0.55)
            self.assertLessEqual(pred.get("calibrated_confidence"), 0.55)
        finally:
            signal_evaluator.get_history = orig_get_history

    def test_ml_ensemble_healthy_provenance_and_calibrator_telemetry(self):
        """Verify C-01: Healthy ML model inference emits signal_source='ML_ENSEMBLE', is_fallback=False, and includes calibrator telemetry."""
        class HealthyClassifier:
            def predict_proba(self, X):
                return np.array([[0.1, 0.2, 0.7]])

        class HealthyRegressor:
            def predict(self, X):
                return np.array([0.01])

        calibrator_dict = {"X": [0.5, 0.7, 0.9], "y": [0.55, 0.72, 0.93], "version": "v2.1", "ece": 0.025}
        self.evaluator.models_by_interval = {
            "15": {
                "trending": {
                    "trend": HealthyClassifier(),
                    "price": HealthyRegressor(),
                    "calibrator": calibrator_dict
                },
                "ranging": {
                    "trend": HealthyClassifier(),
                    "price": HealthyRegressor(),
                    "calibrator": calibrator_dict
                }
            }
        }

        base_ts = 1700000000.0
        df = pd.DataFrame({
            "timestamp": [float(base_ts + i * 900) for i in range(260)],
            "open": [100.0 for _ in range(260)],
            "high": [102.0 for _ in range(260)],
            "low": [98.0 for _ in range(260)],
            "close": [100.0 for _ in range(260)],
            "volume": [1000.0 for _ in range(260)],
            "RSI": [60.0 for _ in range(260)],
            "EMA_9": [105.0 for _ in range(260)],
            "EMA_21": [100.0 for _ in range(260)],
            "ADX": [30.0 for _ in range(260)]
        })

        import signal_evaluator
        orig_get_history = signal_evaluator.get_history
        signal_evaluator.get_history = lambda symbol, interval, limit: df

        try:
            self.evaluator.evaluate_interval("BTCUSDT", "15")
            pred = self.bot_state.get("latest_prediction_15m") or self.bot_state.get("latest_prediction_15")
            self.assertIsNotNone(pred)
            self.assertEqual(pred.get("signal_source"), "ML_ENSEMBLE")
            self.assertFalse(pred.get("is_fallback"))
            self.assertEqual(pred.get("calibrator_version"), "v2.1")
            self.assertEqual(pred.get("calibrator_ece"), 0.025)
            self.assertGreater(pred.get("calibrated_confidence"), 0.70)
        finally:
            signal_evaluator.get_history = orig_get_history

if __name__ == "__main__":
    unittest.main()
