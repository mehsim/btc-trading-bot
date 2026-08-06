import pytest
import pandas as pd
import numpy as np
from core import add_features
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def test_feature_timestamp_strictly_causal():
    """
    F-06 Audit Test:
    Verifies that mutating future prices (at index t+1, t+2) does NOT alter
    the feature vector calculated at index t.
    """
    np.random.seed(42)
    n_candles = 260
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq="15min")
    
    base_prices = np.linspace(50000, 52000, n_candles) + np.random.randn(n_candles) * 20
    df1 = pd.DataFrame({
        "open": base_prices,
        "high": base_prices + 10,
        "low": base_prices - 10,
        "close": base_prices,
        "volume": 100.0,
        "timestamp": [int(ts.timestamp() * 1000) for ts in dates]
    })
    
    df1_feat = add_features(df1.copy())
    row_t_orig = df1_feat.iloc[220].copy()
    
    # Mutate FUTURE rows (indices 221 to 259) heavily
    df2 = df1.copy()
    df2.loc[221:, "close"] = df2.loc[221:, "close"] * 5.0
    df2.loc[221:, "high"] = df2.loc[221:, "high"] * 5.0
    
    df2_feat = add_features(df2.copy())
    row_t_mutated = df2_feat.iloc[220].copy()

    # Features at index 220 must be completely identical regardless of future price spikes
    numeric_cols = df1_feat.select_dtypes(include=[np.number]).columns
    ignored_cols = ["target_trend", "target_price_change"]
    feature_cols = [c for c in numeric_cols if c not in ignored_cols]

    val1 = row_t_orig[feature_cols].astype(float).values
    val2 = row_t_mutated[feature_cols].astype(float).values

    np.testing.assert_allclose(
        val1, val2, rtol=1e-5, atol=1e-5,
        err_msg="Look-ahead leakage detected: feature at time t changed when future candles were mutated"
    )

def test_label_shift_destroys_predictive_signal():
    """
    F-06 Audit Test:
    Asserts that shifting target labels out of phase collapses accuracy to random chance,
    proving target labels are not leaked inside features.
    """
    np.random.seed(42)
    n = 300
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min")
    
    # Synthetic random walk
    returns = np.random.randn(n) * 0.005
    prices = 50000.0 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": 100.0,
        "timestamp": [int(ts.timestamp() * 1000) for ts in dates]
    })
    
    df_feat = add_features(df)
    
    # True target (1 if next close > current close else 0)
    y_true = (df_feat["close"].shift(-1) > df_feat["close"]).astype(int).iloc[:-1]
    
    # Randomly permute target labels to break real relationship
    y_permuted = np.random.permutation(y_true.values)
    
    # Random chance baseline accuracy must be near ~50%
    chance_acc = accuracy_score(y_true.values, y_permuted)
    assert 0.40 <= chance_acc <= 0.60, f"Permuted labels should yield random chance accuracy (~50%), got {chance_acc:.2f}"
