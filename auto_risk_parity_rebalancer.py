"""
auto_risk_parity_rebalancer.py
------------------------------
24-Hour Risk Parity Rebalancing Worker.
Re-calculates 30-day trailing asset volatilities across BTC, ETH, SOL, AVAX, ADA, DOT, and LTC
daily at 00:00 UTC, adjusting portfolio position sizing weights dynamically.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LTCUSDT"]

class AutoRiskParityRebalancer:
    def __init__(self, target_portfolio_vol: float = 0.10):
        self.target_portfolio_vol = target_portfolio_vol

    def compute_daily_rebalance_weights(self, symbol_vols: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates Risk Parity allocation weights inversely proportional to 30-day trailing volatility.
        w_i = (1 / vol_i) / sum(1 / vol_j)
        """
        if not symbol_vols:
            return {s: 1.0 / len(DEFAULT_SYMBOLS) for s in DEFAULT_SYMBOLS}

        inv_vols = {s: 1.0 / max(0.005, v) for s, v in symbol_vols.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol <= 0:
            return {s: 1.0 / len(symbol_vols) for s in symbol_vols}

        weights = {s: float(inv_v / total_inv_vol) for s, inv_v in inv_vols.items()}
        return weights

auto_risk_parity_rebalancer = AutoRiskParityRebalancer()
