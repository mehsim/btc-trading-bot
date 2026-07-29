"""
implementation_shortfall.py
---------------------------
Implementation Shortfall & Execution Quality Tracking Engine.
Tracks benchmark arrival price vs executed price to quantify slippage costs
and provides orderbook imbalance-based limit order pricing.
"""

import time
import numpy as np
from typing import Dict, Any

class ImplementationShortfallTracker:
    def __init__(self):
        self.execution_logs = []

    def calculate_shortfall_bp(self, arrival_price: float, execution_price: float, side: str) -> float:
        """
        Calculates Implementation Shortfall in basis points (1 bp = 0.01%).
        Positive shortfalls indicate adverse slippage.
        """
        if arrival_price <= 0 or execution_price <= 0:
            return 0.0

        if side.capitalize() == "Buy":
            diff_pct = (execution_price - arrival_price) / arrival_price
        else:
            diff_pct = (arrival_price - execution_price) / arrival_price

        shortfall_bp = float(diff_pct * 10000.0)
        return shortfall_bp

    def compute_ob_imbalance_limit_price(self, best_bid: float, best_ask: float, ob_imbalance: float, side: str) -> float:
        """
        Calculates optimal limit order price based on Level-2 orderbook imbalance (-1.0 to +1.0).
        """
        spread = max(0.0, best_ask - best_bid)
        if spread <= 0:
            return best_bid if side.capitalize() == "Buy" else best_ask

        # Offset limit price dynamically towards mid-price based on imbalance
        if side.capitalize() == "Buy":
            offset = spread * max(0.1, min(0.9, (ob_imbalance + 1.0) / 2.0))
            return best_bid + offset
        else:
            offset = spread * max(0.1, min(0.9, (1.0 - ob_imbalance) / 2.0))
            return best_ask - offset

implementation_shortfall_tracker = ImplementationShortfallTracker()
