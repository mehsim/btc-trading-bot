"""
volatility_targeting.py
-----------------------
Volatility Targeting Engine.
Target 10% annualized volatility and scale position sizes dynamically
to maintain consistent portfolio volatility across low and high vol market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class VolatilityTargetingEngine:
    def __init__(self, target_annual_vol: float = 0.10):
        self.target_annual_vol = target_annual_vol
        self.target_daily_vol = target_annual_vol / np.sqrt(365.0)  # ~0.52% daily vol target

    def calculate_volatility_scalar(self, returns_series: pd.Series) -> float:
        """
        Calculates position scaling factor based on current realized daily volatility:
        scalar = target_daily_vol / current_realized_daily_vol
        """
        if returns_series is None or len(returns_series) < 14:
            return 1.0

        clean_returns = returns_series.dropna().values
        current_daily_vol = float(np.std(clean_returns[-20:]))
        if current_daily_vol <= 0:
            return 1.0

        vol_scalar = self.target_daily_vol / current_daily_vol
        # Clamp scalar between 0.3x (heavy vol reduction) and 1.5x (boost in quiet trends)
        return float(np.clip(vol_scalar, 0.3, 1.5))

volatility_targeting_engine = VolatilityTargetingEngine()
