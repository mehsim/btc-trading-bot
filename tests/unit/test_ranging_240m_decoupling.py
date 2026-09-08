import unittest
import numpy as np
import pandas as pd
from signal_evaluator import SignalEvaluator


class TestRanging240mDecoupling(unittest.TestCase):
    """
    Verifies that the 240m timeframe executes independently on its dedicated ML ensemble
    when in a ranging regime (ADX < 25), bypassing the Option B Bollinger Bands bridge,
    while other timeframes (e.g. 15m) continue to use Option B.
    """

    def setUp(self):
        self.bot_state = {}
        self.evaluator = SignalEvaluator(self.bot_state)

    def _create_ranging_df(self, n_bars=260):
        np.random.seed(42)
        base_ts = 1700000000.0
        return pd.DataFrame({
            "timestamp": [float(base_ts + i * 14400) for i in range(n_bars)],
            "open": [100.0 + (i % 5) * 0.1 for i in range(n_bars)],
            "high": [102.0 + (i % 5) * 0.1 for i in range(n_bars)],
            "low": [98.0 + (i % 5) * 0.1 for i in range(n_bars)],
            "close": [100.0 + (i % 5) * 0.1 for i in range(n_bars)],
            "volume": [1000.0 for _ in range(n_bars)],
            "RSI": [50.0 for _ in range(n_bars)],
            "ATR": [1.5 for _ in range(n_bars)],
            "ADX": [18.0 for _ in range(n_bars)],  # ADX < 25 => Ranging regime
            "EMA_9": [100.0 for _ in range(n_bars)],
            "EMA_21": [100.0 for _ in range(n_bars)]
        })

    def test_240m_ranging_bypasses_bridge(self):
        """Verify 240m in ranging mode bypasses Option B MEAN_REVERSION_BB."""
        df = self._create_ranging_df(260)
        import signal_evaluator
        orig_get_history = signal_evaluator.get_history
        signal_evaluator.get_history = lambda symbol, interval, limit, **kwargs: df

        try:
            self.evaluator.evaluate_interval("BTCUSDT", "240")
            pred = (
                self.bot_state.get("latest_prediction_bg_BTCUSDT_4h")
                or self.bot_state.get("latest_prediction_bg_4h")
                or self.bot_state.get("evaluator_prediction_4h")
            )
            self.assertIsNotNone(pred, "Expected a prediction for 240m")
            # Must NOT be MEAN_REVERSION_BB because 240m is decoupled!
            self.assertNotEqual(
                pred.get("signal_source"),
                "MEAN_REVERSION_BB",
                f"240m ranging should not route to Option B bridge, got {pred.get('signal_source')}"
            )
            # Must be either ML_ENSEMBLE or GOVERNANCE_ABSTAIN / GOVERNANCE_DENIED
            self.assertIn(
                pred.get("signal_source"),
                ["ML_ENSEMBLE", "GOVERNANCE_ABSTAIN", "GOVERNANCE_DENIED"],
                f"Unexpected signal_source: {pred.get('signal_source')}"
            )
        finally:
            signal_evaluator.get_history = orig_get_history

    def test_15m_ranging_retains_bridge(self):
        """Verify 15m in ranging mode falls through to ML ensemble when in mid-range,
        and routes to Option B MEAN_REVERSION_BB when band extreme is pierced."""
        df = self._create_ranging_df(260)
        import signal_evaluator
        orig_get_history = signal_evaluator.get_history
        signal_evaluator.get_history = lambda symbol, interval, limit, **kwargs: df

        try:
            # 1. Mid-range fallthrough to ML Ensemble
            self.evaluator.evaluate_interval("BTCUSDT", "15")
            pred = (
                self.bot_state.get("latest_prediction_bg_BTCUSDT_15m")
                or self.bot_state.get("latest_prediction_bg_15m")
                or self.bot_state.get("evaluator_prediction_15m")
            )
            self.assertIsNotNone(pred, "Expected a prediction for 15m")
            self.assertEqual(
                pred.get("signal_source"),
                "ML_ENSEMBLE",
                f"15m ranging inside mid-range should fall through to ML ensemble, got {pred.get('signal_source')}"
            )

            # 2. Extreme band touch routes to MEAN_REVERSION_BB
            df_extreme = self._create_ranging_df(260)
            df_extreme.loc[df_extreme.index[-1], "low"] = 95.0
            df_extreme.loc[df_extreme.index[-1], "close"] = 96.0
            df_extreme.loc[df_extreme.index[-1], "open"] = 96.0
            df_extreme.loc[df_extreme.index[-1], "high"] = 97.0
            df_extreme.loc[df_extreme.index[-1], "RSI"] = 30.0
            df_extreme["lower_wick_volume_ratio"] = 1.0
            df_extreme.loc[df_extreme.index[-1], "lower_wick_volume_ratio"] = 1.8

            signal_evaluator.get_history = lambda symbol, interval, limit, **kwargs: df_extreme
            self.evaluator.evaluate_interval("BTCUSDT", "15")
            pred_ext = (
                self.bot_state.get("latest_prediction_bg_BTCUSDT_15m")
                or self.bot_state.get("latest_prediction_bg_15m")
                or self.bot_state.get("evaluator_prediction_15m")
            )
            self.assertIsNotNone(pred_ext, "Expected an extreme prediction for 15m")
            self.assertEqual(
                pred_ext.get("signal_source"),
                "MEAN_REVERSION_BB",
                f"15m ranging at band extreme should route to Option B bridge, got {pred_ext.get('signal_source')}"
            )
        finally:
            signal_evaluator.get_history = orig_get_history

    def test_240m_ranging_updates_existing_prediction_in_history(self):
        """Verify updating an existing prediction entry in prediction_history executes without error and updates fields."""
        df = self._create_ranging_df(260)
        import signal_evaluator
        orig_get_history = signal_evaluator.get_history
        signal_evaluator.get_history = lambda symbol, interval, limit, **kwargs: df

        last_row_ts = int(df.iloc[-1]["timestamp"] * 1000)
        # Pre-seed prediction_history with a stale bridge entry
        stale_entry = {
            "prediction_id": f"BTCUSDT_240_{last_row_ts}",
            "symbol": "BTCUSDT",
            "timestamp": 1700000000.0,
            "candle_timestamp": last_row_ts,
            "interval": "240",
            "direction": "Neutral",
            "status": "Abstain (Inside Bollinger Mid-Range)",
            "signal_source": "MEAN_REVERSION_BB",
            "predicted_change": 0.0,
            "model_version": "v1.0_ranging_bb"
        }
        self.bot_state["prediction_history"] = [stale_entry]

        try:
            self.evaluator.evaluate_interval("BTCUSDT", "240")
            updated = self.bot_state["prediction_history"][0]
            # Must have updated status away from stale bridge text
            self.assertNotEqual(updated["status"], "Abstain (Inside Bollinger Mid-Range)")
            self.assertEqual(updated["signal_source"], "ML_ENSEMBLE")
            self.assertIsNotNone(updated.get("model_version"))
        finally:
            signal_evaluator.get_history = orig_get_history


if __name__ == "__main__":
    unittest.main()
