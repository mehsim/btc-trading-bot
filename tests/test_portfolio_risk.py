import pytest
import pandas as pd
import numpy as np
from portfolio_risk import portfolio_risk_engine

def test_calculate_parametric_var():
    """Verify 99% Parametric VaR calculation."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    returns_df = pd.DataFrame({
        "BTCUSDT": np.random.normal(0.001, 0.02, 50),
        "ETHUSDT": np.random.normal(0.001, 0.03, 50)
    }, index=dates)
    
    positions = [
        {"symbol": "BTCUSDT", "position_size_usd": 100.0},
        {"symbol": "ETHUSDT", "position_size_usd": 50.0}
    ]
    
    var_usd, var_pct, is_ok = portfolio_risk_engine.calculate_parametric_var(positions, returns_df, total_equity=1000.0)
    assert var_usd >= 0.0
    assert var_pct >= 0.0
    assert isinstance(is_ok, bool)

def test_pca_factor_loadings():
    """Verify PCA eigenvalue factor loading calculation."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    returns_df = pd.DataFrame({
        "BTCUSDT": np.random.normal(0.001, 0.02, 50),
        "ETHUSDT": np.random.normal(0.001, 0.03, 50),
        "SOLUSDT": np.random.normal(0.001, 0.04, 50)
    }, index=dates)
    
    res = portfolio_risk_engine.calculate_pca_factor_loadings(returns_df)
    assert "pc1_explained_variance" in res
    assert 0.0 <= res["pc1_explained_variance"] <= 1.0
