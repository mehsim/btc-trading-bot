"""
test_learning_engine.py
------------------------
Unit and integration tests for Phase 1 Continuous Learning Engine.
Verifies experience DB storage, decision snapshot, failure attribution, calibration tracking,
counterfactual evaluation, trade similarity search, and learning engine event queue.
"""

import time
import os
import unittest

from experience_db import save_trade_experience, get_trade_experience
from decision_snapshot import build_decision_snapshot
from feature_health_monitor import feature_health_monitor
from calibration_tracker import record_trade_outcome, calculate_ece, get_calibration_summary
from failure_attribution_engine import failure_attribution_engine
from decision_outcome_replay import decision_outcome_replay
from pattern_miner import pattern_miner
from knowledge_base import save_rule, get_active_rules
from risk_multiplier import risk_multiplier_engine
from regime_memory import record_regime_trade, get_regime_summary
from counterfactual_engine import counterfactual_engine
from learning_scorer import calculate_learning_score
from drift_monitor import drift_monitor
from trade_similarity_search import trade_similarity_search

class TestContinuousLearningEngine(unittest.TestCase):

    def test_01_experience_db_save_and_fetch(self):
        trade_id = f"TEST_TRADE_{int(time.time())}"
        snap = build_decision_snapshot(symbol="BTCUSDT", direction="LONG", confidence=0.88)
        
        record = {
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "timestamp": time.time(),
            "confidence": 0.88,
            "decision_snapshot": snap,
            "adx": 32.5,
            "atr_pct": 0.012,
            "entry_price": 65000.0,
            "exit_price": 64000.0,
            "pnl_usd": -15.0,
            "realized_r": -1.0,
            "trade_outcome": "LOSS"
        }
        
        saved = save_trade_experience(record)
        self.assertTrue(saved)
        
        fetched = get_trade_experience(trade_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["symbol"], "BTCUSDT")
        self.assertEqual(fetched["confidence"], 0.88)

    def test_02_feature_health_monitor(self):
        good_record = {"adx": 25.0, "atr_pct": 0.01, "rsi": 55.0, "funding_rate": 0.0001, "oi_z_score": 0.5, "entry_price": 65000.0, "confidence": 0.75}
        is_healthy, issues = feature_health_monitor.inspect_record(good_record)
        self.assertTrue(is_healthy)
        self.assertEqual(len(issues), 0)

        bad_record = {"adx": float("nan"), "confidence": 1.5}
        is_healthy, issues = feature_health_monitor.inspect_record(bad_record)
        self.assertFalse(is_healthy)
        self.assertTrue(len(issues) >= 1)

    def test_03_calibration_tracker(self):
        record_trade_outcome(confidence=0.85, is_win=True)
        record_trade_outcome(confidence=0.85, is_win=False)
        
        summary = get_calibration_summary()
        self.assertTrue(len(summary) >= 5)
        ece = calculate_ece()
        self.assertIsInstance(ece, float)

    def test_04_failure_attribution_engine(self):
        loss_record = {
            "pnl_usd": -25.0,
            "ltf_conflict": 1,
            "atr_percentile": 82.0,
            "execution_latency_ms": 1500.0,
            "funding_rate": 0.0004
        }
        attribution = failure_attribution_engine.diagnose_loss(loss_record)
        self.assertIn("ltf_reversal", attribution)
        self.assertIn("high_volatility", attribution)
        
        # Verify percentages sum to 100%
        total_pct = sum(v["pct"] for v in attribution.values())
        self.assertEqual(total_pct, 100)

    def test_05_decision_outcome_replay(self):
        record = {"confidence": 0.90, "pnl_usd": -10.0, "signal_direction": "LONG", "ltf_conflict": 1, "atr_pct": 0.025}
        replay = decision_outcome_replay.replay_trade(record)
        self.assertEqual(replay["actual_outcome"], "LOSS")
        self.assertEqual(replay["prediction_error"], 0.90)

    def test_06_counterfactual_engine(self):
        record = {"realized_r": -1.0, "pnl_usd": -20.0, "exit_reason": "STOP LOSS", "mae_pct": 0.012, "atr_pct": 0.010}
        scenarios = counterfactual_engine.evaluate_scenarios(record)
        self.assertEqual(len(scenarios), 5)
        scenario_names = [s["scenario"] for s in scenarios]
        self.assertIn("Actual", scenario_names)
        self.assertIn("ATR 1.5x", scenario_names)
        self.assertIn("No Trade", scenario_names)

    def test_07_risk_multiplier(self):
        ctx = {"market_regime": "TRENDING", "ltf_conflict": 0, "confidence": 0.85, "atr_percentile": 50.0}
        mult = risk_multiplier_engine.get_risk_multiplier(ctx)
        self.assertGreaterEqual(mult, 0.50)
        self.assertLessEqual(mult, 1.00)

    def test_08_trade_similarity_search(self):
        ctx = {"adx": 30.0, "atr_pct": 0.015, "confidence": 0.80, "funding_rate": 0.0001, "oi_z_score": 0.5, "ltf_conflict": 0}
        res = trade_similarity_search.find_similar_trades(ctx, top_n=5)
        self.assertIn("win_rate", res)
        self.assertIn("avg_r", res)

    def test_09_audit_logger_and_schema_validator(self):
        from audit_logger import log_learning_action, get_recent_audit_logs
        from schema_validator import schema_validator
        
        log_learning_action("TEST_ACTION", "test_component", trade_id="TEST_999", details={"key": "val"})
        logs = get_recent_audit_logs(limit=5)
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["action_type"], "TEST_ACTION")
        
        is_valid, errors = schema_validator.validate_record({"trade_id": "T1", "symbol": "BTCUSDT", "confidence": 0.85})
        self.assertTrue(is_valid)

    def test_10_feature_availability_and_db_snapshot(self):
        from feature_availability import record_feature_sample, get_feature_availability_report
        from db_snapshot_manager import create_db_snapshot, list_db_snapshots
        from regime_transition_analyzer import record_transition_trade, get_transition_analysis
        
        record_feature_sample("adx", is_available=True)
        report = get_feature_availability_report()
        self.assertTrue(len(report) > 0)
        
        snap_path = create_db_snapshot(snapshot_tag="unittest")
        self.assertTrue(os.path.exists(snap_path))
        
        record_transition_trade("TRENDING", "HIGH_VOL", is_win=False, realized_r=-1.0)
        transitions = get_transition_analysis()
        self.assertTrue(len(transitions) > 0)

if __name__ == "__main__":
    unittest.main()

