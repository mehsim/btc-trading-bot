"""
tests/test_risk_and_execution_upgrades.py
-----------------------------------------
Unit test suite covering Risk Management & Execution Quality Upgrades:
- Refined Half-Kelly & CVaR tail risk position sizing
- Volatility Targeting at 10% annualized target
- Exponentially weighted correlation heatmap penalties
- Implementation Shortfall & Orderbook Imbalance Limit Pricing
"""

import pytest
import numpy as np
import pandas as pd
from refined_kelly_cvar import refined_kelly_cvar
from volatility_targeting import volatility_targeting_engine
from rolling_correlation_heat import rolling_correlation_engine
from implementation_shortfall import implementation_shortfall_tracker


def test_refined_kelly_cvar_sizing():
    returns_s = pd.Series(np.random.normal(-0.01, 0.02, 100))
    size_usd, reason = refined_kelly_cvar.calculate_refined_position_size(
        win_rate=0.60,
        win_loss_ratio=1.5,
        equity=100.0,
        returns_series=returns_s
    )
    assert size_usd > 0.0
    assert size_usd <= 20.0  # Respects 20% max drawdown limit ($20 on $100 equity)


def test_volatility_targeting_scaling():
    # Low daily vol returns (0.1% daily vol vs 0.52% target)
    returns_low = pd.Series([0.001, -0.001] * 10)
    scalar_low = volatility_targeting_engine.calculate_volatility_scalar(returns_low)
    assert scalar_low > 1.0  # Scales up in quiet markets

    # High daily vol returns (2.0% daily vol vs 0.52% target)
    returns_high = pd.Series([0.020, -0.020] * 10)
    scalar_high = volatility_targeting_engine.calculate_volatility_scalar(returns_high)
    assert scalar_high < 1.0 # Scales down in high vol markets


test_returns_df = pd.DataFrame({
    "BTCUSDT": np.linspace(0.01, 0.05, 50),
    "ETHUSDT": np.linspace(0.01, 0.05, 50) # Perfect positive correlation
})

def test_rolling_correlation_penalty():
    penalty = rolling_correlation_engine.evaluate_correlation_penalty("BTCUSDT", ["ETHUSDT"], test_returns_df)
    assert penalty < 1.0  # High correlation triggers sizing penalty


def test_implementation_shortfall_bp():
    bp = implementation_shortfall_tracker.calculate_shortfall_bp(100.0, 100.05, "Buy")
    assert bp == pytest.approx(5.0)  # 5 bps adverse slippage


def test_orderbook_imbalance_limit_pricing():
    price = implementation_shortfall_tracker.compute_ob_imbalance_limit_price(100.0, 100.2, 0.5, "Buy")
    assert 100.0 < price < 100.2
