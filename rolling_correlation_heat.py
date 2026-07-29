"""
rolling_correlation_heat.py
---------------------------
Rolling 30-Day Exponential-Weighted Portfolio Correlation Heatmap Engine.
Penalizes concentration dynamically when new trade candidate correlates highly (>0.70)
with open positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class RollingCorrelationEngine:
    def __init__(self, correlation_threshold: float = 0.70):
        self.correlation_threshold = correlation_threshold

    def calculate_exponential_correlation(self, returns_df: pd.DataFrame, sym_a: str, sym_b: str, halflife_days: int = 10) -> float:
        """
        Calculates exponentially weighted correlation between two assets over trailing 30 days.
        """
        if returns_df is None or sym_a not in returns_df.columns or sym_b not in returns_df.columns:
            return 0.0

        clean_df = returns_df[[sym_a, sym_b]].dropna()
        if len(clean_df) < 14:
            return 0.0

        ewma_cov = clean_df.ewm(halflife=halflife_days).cov().iloc[-2:, -2:]
        std_a = np.sqrt(ewma_cov.iloc[0, 0])
        std_b = np.sqrt(ewma_cov.iloc[1, 1])
        
        if std_a * std_b <= 0:
            return 0.0

        corr_val = float(ewma_cov.iloc[0, 1] / (std_a * std_b))
        return float(np.clip(corr_val, -1.0, 1.0))

    def evaluate_correlation_penalty(self, candidate_symbol: str, active_symbols: List[str], returns_df: pd.DataFrame) -> float:
        """
        Returns penalty multiplier (1.0 = No Penalty, 0.5 = 50% Size Reduction if high correlation detected).
        """
        if not active_symbols or returns_df is None:
            return 1.0

        max_corr = 0.0
        for active_sym in active_symbols:
            if active_sym != candidate_symbol:
                corr = self.calculate_exponential_correlation(returns_df, candidate_symbol, active_sym)
                max_corr = max(max_corr, corr)

        if max_corr >= 0.85:
            return 0.4  # Severe correlation penalty
        elif max_corr >= self.correlation_threshold:
            return 0.65 # Moderate correlation penalty
        return 1.0

rolling_correlation_engine = RollingCorrelationEngine()
