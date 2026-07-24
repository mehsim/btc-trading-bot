import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

class VolatilityClusterer:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_assignments: Dict[str, int] = {}
        self.tier_floors: Dict[int, float] = {0: 0.0050, 1: 0.0080, 2: 0.0120}

    def compute_parkinson_volatility(self, df: pd.DataFrame, window: int = 30) -> float:
        """
        Computes 30-day Parkinson Volatility:
        sigma_P = sqrt( (1 / (4 * ln(2) * N)) * sum( ln(H_i / L_i)^2 ) )
        """
        if df is None or df.empty or len(df) < window or "high" not in df.columns or "low" not in df.columns:
            return 0.02  # Default fallback

        sub_df = df.tail(window).copy()
        high_low_ratio = np.log(sub_df["high"] / np.maximum(1e-8, sub_df["low"]))
        sum_sq = float(np.sum(high_low_ratio ** 2))
        const_factor = 1.0 / (4.0 * np.log(2.0) * len(sub_df))
        parkinson_vol = float(np.sqrt(max(1e-8, const_factor * sum_sq)))

        return parkinson_vol

    def update_clusters(self, symbol_df_dict: Dict[str, pd.DataFrame]):
        """
        Runs K-Means clustering on Parkinson volatility & volume to auto-classify symbols into tiers.
        """
        features = []
        valid_symbols = []

        for symbol, df in symbol_df_dict.items():
            if isinstance(df, pd.DataFrame) and len(df) >= 30:
                p_vol = self.compute_parkinson_volatility(df)
                avg_vol = float(df["volume"].tail(30).mean()) if "volume" in df.columns else 1000.0
                features.append([p_vol, np.log1p(avg_vol)])
                valid_symbols.append(symbol)

        if len(valid_symbols) < self.n_clusters:
            return

        X = np.array(features)
        labels = self.kmeans.fit_predict(X)

        # Sort cluster IDs by average Parkinson volatility so Cluster 0 = Lowest Vol (Majors), Cluster 2 = Highest Vol (Alts)
        cluster_vols = {i: float(np.mean(X[labels == i, 0])) for i in range(self.n_clusters)}
        sorted_clusters = sorted(cluster_vols.keys(), key=lambda k: cluster_vols[k])
        rank_map = {old_id: new_rank for new_rank, old_id in enumerate(sorted_clusters)}

        for sym, old_label in zip(valid_symbols, labels):
            self.cluster_assignments[sym] = rank_map[old_label]

    def get_symbol_break_even_floor(self, symbol: str) -> float:
        """
        Returns dynamic Break-Even percentage floor based on Parkinson Volatility cluster.
        Tier 0 (Majors): 0.50%
        Tier 1 (Mid-caps): 0.80%
        Tier 2 (Alts): 1.20%
        """
        cluster_tier = self.cluster_assignments.get(symbol, 1 if symbol in ["SOLUSDT", "AVAXUSDT"] else (0 if symbol in ["BTCUSDT", "ETHUSDT"] else 2))
        return self.tier_floors.get(cluster_tier, 0.0080)

volatility_clusterer = VolatilityClusterer()
