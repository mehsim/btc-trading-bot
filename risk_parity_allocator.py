"""
risk_parity_allocator.py
------------------------
Risk Parity Asset Allocation Engine.
Calculates optimal portfolio weights inversely proportional to 30-day trailing asset volatility
across trading universe (BTC, ETH, SOL, AVAX, ADA, DOT, LTC).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LTCUSDT"]

class RiskParityAllocator:
    def __init__(self, target_portfolio_vol: float = 0.15):
        self.target_portfolio_vol = target_portfolio_vol

    def compute_risk_parity_weights(self, symbol_volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        Computes Risk Parity allocation weights for active trading symbols.
        Weight_i = (1 / vol_i) / sum(1 / vol_j)
        """
        if not symbol_volatilities:
            return {s: 1.0 / len(DEFAULT_SYMBOLS) for s in DEFAULT_SYMBOLS}

        inv_vols = {}
        for sym, vol in symbol_volatilities.items():
            safe_vol = max(0.005, float(vol))
            inv_vols[sym] = 1.0 / safe_vol

        sum_inv_vol = sum(inv_vols.values())
        if sum_inv_vol <= 0:
            return {s: 1.0 / len(symbol_volatilities) for s in symbol_volatilities}

        weights = {sym: inv_vol / sum_inv_vol for sym, inv_vol in inv_vols.items()}
        return weights

risk_parity_allocator = RiskParityAllocator()
