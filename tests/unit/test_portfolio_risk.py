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

def test_calculate_mcr():
    """Verify MCR calculation with candidate position."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    returns_df = pd.DataFrame({
        "BTCUSDT": np.random.normal(0.001, 0.02, 50),
        "ETHUSDT": np.random.normal(0.001, 0.03, 50)
    }, index=dates)
    
    positions = [
        {"symbol": "ETHUSDT", "position_size_usd": 50.0}
    ]
    
    mcr, is_approved = portfolio_risk_engine.calculate_mcr(
        candidate_symbol="BTCUSDT",
        candidate_size_usd=100.0,
        open_positions=positions,
        returns_df=returns_df
    )
    assert isinstance(mcr, float)
    assert isinstance(is_approved, bool)

def test_parametric_var_fail_closed_and_prior():
    """Verify F-05: Parametric VaR uses conservative prior and fails closed when returns_df is None or empty."""
    positions = [{"symbol": "BTCUSDT", "position_size_usd": 1000.0}]
    
    # 1. Test missing returns_df -> applies conservative daily std prior (3%) instead of returning (0, 0, True)
    var_usd, var_pct, is_ok = portfolio_risk_engine.calculate_parametric_var(positions, None, total_equity=1000.0)
    assert var_usd > 0.0
    assert var_pct > 0.0
    assert is_ok is True  # 1.64485 * 0.03 * 1000 = ~49.35 USD (4.935% <= 5% cap)

    # 2. Test large position exceeding 5% VaR equity cap under conservative prior -> Fail-Closed rejection
    large_positions = [{"symbol": "BTCUSDT", "position_size_usd": 2000.0}]
    var_usd_large, var_pct_large, is_ok_large = portfolio_risk_engine.calculate_parametric_var(large_positions, None, total_equity=1000.0)
    assert var_pct_large > 0.05
    assert is_ok_large is False  # Rejected because VaR > 5% equity cap


