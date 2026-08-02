import sys
import os
import unittest
import time

from trade_calculators import (
    calculate_probabilistic_utility_bootstrap,
    UnifiedTargetGenerator
)
from exit_policy_engine import (
    PortfolioUtilityOptimizer,
    generate_continuous_policy_vector,
    log_checksummed_exit_decision
)

class TestInstitutionalTPPipeline(unittest.TestCase):
    def test_probabilistic_utility_bootstrap(self):
        res = calculate_probabilistic_utility_bootstrap(
            symbol="BTCUSDT",
            entry_price=60000.0,
            candidate_tp=63000.0,
            candidate_sl=59000.0,
            direction="Bullish",
            win_prob=0.60,
            loss_prob=0.40,
            num_resamples=500
        )
        self.assertIn("mean_utility", res)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertGreaterEqual(res["ci_upper"], res["ci_lower"])

    def test_unified_target_generator(self):
        policy_vector = {
            "target_multiplier": 1.8,
            "partial_split": [0.20, 0.30, 0.50]
        }
        bootstrap_ci = {"ci_lower": 1.5, "ci_upper": 2.5}
        
        targets = UnifiedTargetGenerator.compute_targets(
            policy_vector=policy_vector,
            bootstrap_ci=bootstrap_ci,
            entry_price=60000.0,
            direction="Bullish",
            atr_dollars=1000.0,
            symbol="BTCUSDT"
        )
        
        self.assertGreater(targets["take_profit_price"], 60000.0)
        self.assertEqual(targets["stage_1_tp"]["size_pct"], 0.20)
        self.assertEqual(targets["stage_2_tp"]["size_pct"], 0.30)
        self.assertEqual(targets["stage_3_runner"]["size_pct"], 0.50)

    def test_extension_hysteresis(self):
        policy_cfg = {
            "hysteresis": {
                "min_delta_r": 0.20,
                "min_candles_elapsed": 3,
                "min_utility_gain_pct": 0.15
            }
        }
        
        now = time.time()
        # Test valid hysteresis
        valid, msg = UnifiedTargetGenerator.validate_extension_hysteresis(
            current_tp=62000.0,
            candidate_tp=63000.0,
            entry_price=60000.0,
            direction="Bullish",
            atr_dollars=1000.0,
            policy_cfg=policy_cfg,
            last_modified_time=now - (5 * 900),
            current_time=now,
            candle_interval_sec=900.0,
            utility_gain_pct=0.20
        )
        self.assertTrue(valid)

        # Test invalid delta R step
        invalid, msg = UnifiedTargetGenerator.validate_extension_hysteresis(
            current_tp=62000.0,
            candidate_tp=62100.0,
            entry_price=60000.0,
            direction="Bullish",
            atr_dollars=1000.0,
            policy_cfg=policy_cfg,
            last_modified_time=now - (5 * 900),
            current_time=now,
            candle_interval_sec=900.0,
            utility_gain_pct=0.20
        )
        self.assertFalse(invalid)

    def test_portfolio_utility_optimizer(self):
        active_trades = {
            "trade_1": {"entry_price": 60000.0, "current_price": 63000.0, "direction": "Bullish", "expected_r": 2.0},
            "trade_2": {"entry_price": 3000.0, "current_price": 2900.0, "direction": "Bullish", "expected_r": 0.5, "candles_elapsed": 20}
        }
        
        opt = PortfolioUtilityOptimizer.optimize_portfolio_capital(active_trades)
        self.assertIn("close_trades", opt)
        self.assertIn("trade_2", opt["close_trades"])

    def test_continuous_policy_vector(self):
        pv = generate_continuous_policy_vector("STRONG_TREND", adx_val=35.0)
        self.assertIn("target_multiplier", pv)
        self.assertIn("partial_split", pv)

    def test_exit_decision_dataset_logging(self):
        test_file = "test_exit_dataset_scratch.jsonl"
        if os.path.exists(test_file):
            os.remove(test_file)
            
        record = {"trade_id": "test_1", "action": "EXTEND_TP", "pnl": 10.0}
        log_checksummed_exit_decision(record, filepath=test_file)
        
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("schema_version", lines[0])
        
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_stop_state_machine(self):
        from order_state_machine import StopState, StopStateMachine
        
        # Forward transition should be valid
        self.assertTrue(StopStateMachine.can_transition(StopState.INITIAL, StopState.BREAK_EVEN))
        self.assertTrue(StopStateMachine.can_transition(StopState.BREAK_EVEN, StopState.PROFIT_LOCK))
        
        # Backward transition should be invalid
        self.assertFalse(StopStateMachine.can_transition(StopState.PROFIT_LOCK, StopState.TRAILING))

        # Monotonic price update validation for Longs
        valid_long, msg = StopStateMachine.validate_monotonic_stop_update(
            direction="Bullish",
            current_sl=100.0,
            proposed_sl=101.5,
            current_state_str="INITIAL",
            target_state_str="BREAK_EVEN"
        )
        self.assertTrue(valid_long)

        # Monotonic violation for Longs (regressing SL)
        invalid_long, msg = StopStateMachine.validate_monotonic_stop_update(
            direction="Bullish",
            current_sl=100.0,
            proposed_sl=99.5,
            current_state_str="INITIAL",
            target_state_str="BREAK_EVEN"
        )
        self.assertFalse(invalid_long)

if __name__ == "__main__":
    unittest.main()
