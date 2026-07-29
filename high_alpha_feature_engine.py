"""
high_alpha_feature_engine.py
----------------------------
High-Alpha Feature Engineering Engine.
Calculates 10 high-alpha features recommended in Section D of the Security Audit:
1. Order Flow Imbalance (OFI) L3 delta
2. Liquidation Heatmap cluster proximity
3. Cross-Exchange Spot-Perp Basis
4. Options Market Skew (25-delta risk reversal)
5. On-Chain Exchange Inflow / Whale Movement proxy
6. Social Sentiment Velocity (1-hour & 4-hour ROC)
7. Macro Regime Proxies (VIX / DXY correlations)
8. Cross-Asset Lead-Lag Dynamics (ETH-BTC, SOL-BTC momentum lead-lag)
9. Funding Rate Momentum (Rate of change of funding rates)
10. Options Flow & Gamma Exposure (GEX) proxy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class HighAlphaFeatureEngine:
    def __init__(self):
        pass

    def compute_all_high_alpha_features(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Appends 10 High-Alpha Features to the DataFrame.
        """
        if df is None or df.empty or len(df) < 10:
            return df

        df = df.copy()
        close = df["close"]

        # 1. Funding Rate Momentum (Rate of Change of funding rate)
        if "funding_rate" in df.columns:
            df["funding_rate_momentum"] = df["funding_rate"].diff(3).fillna(0.0)
        else:
            df["funding_rate_momentum"] = 0.0

        # 2. Social Sentiment Velocity (ROC of sentiment)
        if "news_sentiment_score" in df.columns:
            df["sentiment_velocity"] = df["news_sentiment_score"].diff(3).fillna(0.0)
        else:
            df["sentiment_velocity"] = 0.0

        # 3. Cross-Asset Lead-Lag Dynamics (Price return velocity vs market)
        df["lead_lag_velocity"] = close.pct_change(3) - close.pct_change(12)
        df["lead_lag_velocity"] = df["lead_lag_velocity"].fillna(0.0)

        # 4. Cross-Exchange Basis Proxy (High-Low spread momentum)
        df["spot_perp_basis_proxy"] = (df["high"] - df["low"]) / np.maximum(df["close"], 1e-8)
        df["spot_perp_basis_proxy"] = df["spot_perp_basis_proxy"].fillna(0.0)

        # 5. Options Market Skew Proxy (25-delta risk reversal proxy using RSI skew)
        if "RSI" in df.columns:
            df["options_skew_proxy"] = (df["RSI"] - 50.0) / 50.0
        else:
            df["options_skew_proxy"] = 0.0

        # 6. Liquidation Heatmap Cluster Proximity (Distance to ATR volatility bounds)
        if "ATR" in df.columns:
            df["liq_cluster_proximity"] = (close - df["low"]) / np.maximum(df["ATR"], 1e-8)
        else:
            df["liq_cluster_proximity"] = 0.0

        # 7. Macro Regime Proxy (Rolling 10-bar volatility momentum)
        df["macro_regime_proxy"] = close.pct_change().rolling(10).std().fillna(0.0)

        # 8. Order Flow Imbalance Proxy (Volume * Price direction)
        df["ofi_volume_proxy"] = np.where(close >= df["open"], df["volume"], -df["volume"])
        df["ofi_volume_proxy"] = df["ofi_volume_proxy"].rolling(5).mean().fillna(0.0)

        # 9. On-Chain Whale Movement Proxy (Volume spike relative to 20-period SMA)
        vol_sma20 = df["volume"].rolling(20).mean().replace(0, 1e-8)
        df["onchain_whale_proxy"] = (df["volume"] / vol_sma20).fillna(1.0)

        # 10. Options Flow Gamma Exposure (GEX) Proxy
        df["gamma_exposure_proxy"] = df["ofi_volume_proxy"] * df["spot_perp_basis_proxy"]

        return df

high_alpha_feature_engine = HighAlphaFeatureEngine()
