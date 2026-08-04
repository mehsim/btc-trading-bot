import unittest
import numpy as np
from risk_engine import joint_risk_budget_allocator

class TestKellyMHIGovernanceCap(unittest.TestCase):
    def test_kelly_mhi_ceiling_never_exceeded_by_boost_multipliers(self):
        """Verify F-15: Property assertion that effective_kelly never exceeds max_kelly_frac for any size_boost multiplier."""
        mhi_caps = [0.25, 0.20, 0.10, 0.0]
        size_boosts = [0.5, 1.0, 1.25, 1.5, 2.0, 5.0]
        budget_factors = [0.1, 0.5, 1.0, 1.5]
        confidences = [0.55, 0.70, 0.85, 0.99]

        for mhi in [90.0, 75.0, 55.0, 30.0]:
            expected_mhi_cap = joint_risk_budget_allocator.get_mhi_max_kelly(mhi)
            for boost in size_boosts:
                for conf in confidences:
                    res = joint_risk_budget_allocator.allocate_risk_budget(
                        symbol="BTCUSDT",
                        entry_price=50000.0,
                        atr_dollars=500.0,
                        atr_norm=0.01,
                        calibrated_confidence=conf,
                        direction="Bullish",
                        total_equity=10000.0,
                        portfolio_heat=0.0,
                        mhi_score=mhi,
                        context_multipliers={"size_multiplier": boost}
                    )
                    eff_kelly = res["kelly_fraction"]
                    self.assertLessEqual(
                        eff_kelly,
                        expected_mhi_cap + 1e-6,
                        f"effective_kelly ({eff_kelly}) exceeded MHI cap ({expected_mhi_cap}) with size_boost={boost}, conf={conf}"
                    )

if __name__ == "__main__":
    unittest.main()
