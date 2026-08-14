"""
tests/test_high_alpha_features.py
---------------------------------
Unit test suite covering Section D High-Alpha Feature Engineering Upgrades:
- Funding Rate Momentum
- Sentiment Velocity
- Cross-Asset Lead-Lag Dynamics
- Spot-Perp Basis Proxy
- Options Skew & Gamma Exposure (GEX) Proxy
- Liquidation Cluster Proximity
"""

import pytest
import numpy as np
import pandas as pd
from high_alpha_feature_engine import high_alpha_feature_engine


def test_high_alpha_feature_extraction():
    df = pd.DataFrame({
        "open": np.linspace(100, 105, 20),
        "high": np.linspace(101, 106, 20),
        "low": np.linspace(99, 104, 20),
        "close": np.linspace(100.5, 105.5, 20),
        "volume": [100.0] * 20,
        "funding_rate": [0.0001 * i for i in range(20)],
        "RSI": [55.0] * 20,
        "ATR": [1.5] * 20
    })

    res_df = high_alpha_feature_engine.compute_all_high_alpha_features(df)
    
    assert "funding_rate_momentum" in res_df.columns
    assert "sentiment_velocity" in res_df.columns
    assert "lead_lag_velocity" in res_df.columns
    assert "spot_perp_basis_proxy" in res_df.columns
    assert "options_skew_proxy" in res_df.columns
    assert "liq_cluster_proximity" in res_df.columns
    assert "ofi_volume_proxy" in res_df.columns
    assert "onchain_whale_proxy" in res_df.columns
    assert "gamma_exposure_proxy" in res_df.columns
    assert len(res_df) == 20
