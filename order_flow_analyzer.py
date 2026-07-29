"""
order_flow_analyzer.py
----------------------
Order Flow Imbalance (OFI) & Level-3 orderbook delta analysis module.
Calculates net aggressive buyer vs seller pressure from orderbook bid/ask deltas.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class OrderFlowAnalyzer:
    def __init__(self, depth_levels: int = 5):
        self.depth_levels = depth_levels

    def compute_ofi_delta(self, orderbook_bids: list, orderbook_asks: list, prev_bids: list = None, prev_asks: list = None) -> float:
        """
        Computes Order Flow Imbalance (OFI) score normalized between -1.0 (Heavy Seller Pressure)
        and +1.0 (Heavy Buyer Pressure).
        """
        if not orderbook_bids or not orderbook_asks:
            return 0.0

        bid_vol = sum(float(b[1]) for b in orderbook_bids[:self.depth_levels])
        ask_vol = sum(float(a[1]) for a in orderbook_asks[:self.depth_levels])
        total_vol = bid_vol + ask_vol
        
        if total_vol <= 0:
            return 0.0

        ofi_score = (bid_vol - ask_vol) / total_vol
        return float(np.clip(ofi_score, -1.0, 1.0))

order_flow_analyzer = OrderFlowAnalyzer()
