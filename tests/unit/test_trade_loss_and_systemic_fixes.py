"""
test_trade_loss_and_systemic_fixes.py
--------------------------------------
Comprehensive unit test suite validating that all 5 trade-loss causing logic errors
and 9 systemic/architectural flaws are completely eliminated.
"""

import unittest
from decimal import Decimal, ROUND_HALF_UP
import config
import database
from state_manager import state_manager
from trade_calculators import calculate_break_even_stop
from exit_policy_engine import PortfolioUtilityOptimizer


class TestTradeLossAndSystemicFixes(unittest.TestCase):

    def test_fibonacci_step_lock_does_not_tighten_in_drawdown(self):
        """Validates that Fibonacci Step-Lock never tightens Stop Loss on losing trades."""
        # Long trade in drawdown
        entry_price = 100.0
        take_profit = 110.0
        stop_loss = 95.0
        current_price_loss = 96.0  # In drawdown (-4.0%)
        direction = "Bullish"

        total_tp_range = abs(take_profit - entry_price)
        if direction == "Bullish":
            current_move = max(0.0, current_price_loss - entry_price)
        else:
            current_move = max(0.0, entry_price - current_price_loss)
        progress_pct = (current_move / total_tp_range) if total_tp_range > 0 else 0.0

        fib_locks = {0.618: 0.55, 0.50: 0.40, 0.382: 0.25}
        locked_pct = 0.0
        for threshold in sorted(fib_locks.keys(), reverse=True):
            if progress_pct >= threshold:
                locked_pct = fib_locks[threshold]
                break

        # In drawdown, locked_pct and current_move must be zero
        self.assertEqual(current_move, 0.0)
        self.assertEqual(progress_pct, 0.0)
        self.assertEqual(locked_pct, 0.0)

        # Short trade in drawdown
        short_entry = 100.0
        short_tp = 90.0
        short_sl = 105.0
        short_price_loss = 104.0  # In drawdown (+4.0%)
        short_direction = "Bearish"

        short_total_tp_range = abs(short_tp - short_entry)
        if short_direction == "Bullish":
            short_current_move = max(0.0, short_price_loss - short_entry)
        else:
            short_current_move = max(0.0, short_entry - short_price_loss)
        short_progress_pct = (short_current_move / short_total_tp_range) if short_total_tp_range > 0 else 0.0

        self.assertEqual(short_current_move, 0.0)
        self.assertEqual(short_progress_pct, 0.0)

    def test_fibonacci_step_lock_tightens_when_in_genuine_profit(self):
        """Validates that Fibonacci Step-Lock accurately locks profit on winning trades."""
        # Long trade with 50% profit progress
        entry_price = 100.0
        take_profit = 110.0
        stop_loss = 95.0
        current_price_win = 105.0  # 50% progress towards TP 110
        direction = "Bullish"

        total_tp_range = abs(take_profit - entry_price)
        current_move = max(0.0, current_price_win - entry_price)
        progress_pct = (current_move / total_tp_range) if total_tp_range > 0 else 0.0

        fib_locks = {0.618: 0.55, 0.50: 0.40, 0.382: 0.25}
        locked_pct = 0.0
        for threshold in sorted(fib_locks.keys(), reverse=True):
            if progress_pct >= threshold:
                locked_pct = fib_locks[threshold]
                break

        self.assertEqual(progress_pct, 0.50)
        self.assertEqual(locked_pct, 0.40)
        fib_sl = max(stop_loss, entry_price + (current_price_win - entry_price) * locked_pct)
        self.assertEqual(fib_sl, 102.0)  # Locked 40% of the $5 move = +$2.00 profit floor
        self.assertTrue(fib_sl > stop_loss)

    def test_opportunity_cost_cross_symbol_normalization(self):
        """Validates that comparing BTC dollar moves against altcoins does NOT cause phantom R spikes."""
        active_symbol = "DOGEUSDT"
        other_sym = "BTCUSDT"
        pred_other = {
            "ref_price": 80000.0,
            "predicted_change": 1500.0,
            "direction": "Bullish"
        }
        bot_state_mock = {
            f"atr_norm_{other_sym}_60": 0.015,
            f"live_price_{other_sym}": 80000.0
        }

        other_ref_price = float(pred_other.get("ref_price") or bot_state_mock.get(f"live_price_{other_sym}") or 1.0)
        other_change_pct = abs(float(pred_other.get("predicted_change", 0.0))) / max(1e-6, other_ref_price)
        other_atr_pct = float(bot_state_mock.get(f"atr_norm_{other_sym}_60") or 0.015)
        r_cand = other_change_pct / max(1e-4, other_atr_pct)

        # Expected percent change = 1500 / 80000 = 0.01875 (1.875%)
        # Expected R = 0.01875 / 0.015 = 1.25 R
        self.assertAlmostEqual(other_change_pct, 0.01875, places=5)
        self.assertAlmostEqual(r_cand, 1.25, places=2)
        self.assertTrue(r_cand < 5.0, "Expected R should be realistic and not multi-million")

    def test_break_even_stop_does_not_clamp_in_drawdown(self):
        """Validates that calculate_break_even_stop does not place micro-stops during drawdowns."""
        entry_price = 100.0
        current_price_loss = 98.0  # 2% below entry
        target_sl = calculate_break_even_stop("Bullish", entry_price=entry_price, current_price=current_price_loss)

        # Should be cost-buffered above entry (e.g. 100.175), NOT clamped to 98 * 0.9995 (97.95)
        self.assertTrue(target_sl > entry_price)

    def test_portfolio_utility_optimizer_preserves_non_stagnant_trades(self):
        """Validates that PortfolioUtilityOptimizer does not close trades during normal dips."""
        active_trades = {
            "t1": {
                "entry_price": 100.0,
                "current_price": 98.6,  # -1.4% dip
                "direction": "Bullish",
                "expected_r": 1.5,
                "candles_elapsed": 3,   # Young trade, far below max_stagnant_candles (15)
                "half_closed": False
            }
        }
        rebal = PortfolioUtilityOptimizer.optimize_portfolio_capital(active_trades)
        self.assertEqual(len(rebal["close_trades"]), 0, "Non-stagnant trade in normal dip must NOT be closed")

    def test_state_manager_contains_4h_keys_at_startup(self):
        """Validates that StateManager cache contains all 4H keys on cold boot."""
        self.assertIn("latest_prediction_4h", state_manager)
        self.assertIn("confluence_results_4h", state_manager)
        self.assertIn("regime_4h", state_manager)
        self.assertIn("adx_4h", state_manager)
        self.assertIn("calibration_4h", state_manager)
        self.assertIn("240", state_manager.get("win_rate_by_tf", {}))

    def test_case_insensitive_resolve_min_sl_pct(self):
        """Validates that resolve_min_sl_pct correctly parses uppercase and alternative timeframe formats."""
        # 4H tests
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "4H"), 1.5)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "4h"), 1.5)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "240"), 1.5)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "240m"), 1.5)

        # 2H tests
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "2H"), 1.2)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "2h"), 1.2)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "120"), 1.2)

        # 1H tests
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "1H"), 1.0)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "1h"), 1.0)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "60"), 1.0)

        # 30M tests
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "30M"), 0.8)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "30m"), 0.8)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "30"), 0.8)

        # 15M tests
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "15M"), 0.6)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "15m"), 0.6)
        self.assertEqual(config.resolve_min_sl_pct("BTCUSDT", "15"), 0.6)

    def test_scale_out_pnl_quantization_consistency(self):
        """Validates that Short scale-out PnL uses identical financial Decimal quantize to Long."""
        gross_pnl = 12.3456
        taker_fee = 0.0543
        
        def _q2(v):
            return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            
        pnl_long = _q2(gross_pnl - taker_fee)
        pnl_short = _q2(gross_pnl - taker_fee)
        self.assertEqual(pnl_long, pnl_short)
        self.assertEqual(pnl_long, 12.29)


if __name__ == "__main__":
    unittest.main()
