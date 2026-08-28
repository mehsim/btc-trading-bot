import unittest
from trade_calculators import calculate_break_even_stop


class TestTimeframeScaledExitFix(unittest.TestCase):

    def test_break_even_stop_15m_scalp(self):
        # 15m Scalp: Entry $2500, Price $2510, ATR $10.0
        # 15m uses tight cushion (0.15x ATR or cost_buffer)
        sl_15m = calculate_break_even_stop(
            direction="Bullish",
            entry_price=2500.0,
            current_price=2510.0,
            atr_dollars=10.0,
            interval="15"
        )
        # Should be strictly above entry, covering fees (~2504.38)
        self.assertGreater(sl_15m, 2500.0)
        self.assertLess(sl_15m, 2510.0)

    def test_break_even_stop_240m_swing(self):
        # 240m Swing: Entry $2500, Price $2540, ATR $45.0
        # 240m uses 0.50x ATR cushion ($22.50) from current price $2540 -> $2517.50
        sl_240m = calculate_break_even_stop(
            direction="Bullish",
            entry_price=2500.0,
            current_price=2540.0,
            atr_dollars=45.0,
            interval="240"
        )
        # Must give breathing room below $2540 (at around $2517.50) while staying above entry $2500
        self.assertGreaterEqual(sl_240m, 2504.0)
        self.assertLessEqual(sl_240m, 2525.0)

    def test_break_even_stop_bearish_240m(self):
        # 240m Short: Entry $2500, Price $2460, ATR $45.0
        # 240m uses 0.50x ATR cushion ($22.50) above current price $2460 -> $2482.50
        sl_bearish = calculate_break_even_stop(
            direction="Bearish",
            entry_price=2500.0,
            current_price=2460.0,
            atr_dollars=45.0,
            interval="240"
        )
        # Must stay below entry $2500 while giving breathing room above $2460
        self.assertLessEqual(sl_bearish, 2495.0)
        self.assertGreaterEqual(sl_bearish, 2480.0)


if __name__ == "__main__":
    unittest.main()
