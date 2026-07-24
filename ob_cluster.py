import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class OrderBookClusterEngine:
    def __init__(self, wall_multiplier: float = 3.0):
        self.wall_multiplier = wall_multiplier  # Level depth > 3x average depth is a wall

    def find_liquidity_walls(self, order_book: Dict) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Parses order book dictionary ({'bids': [[price, qty]...], 'asks': [[price, qty]...]})
        Returns: (bid_walls, ask_walls)
        """
        if not order_book or not isinstance(order_book, dict):
            return [], []

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        bid_walls = []
        if bids:
            bid_sizes = [float(b[1]) for b in bids if len(b) >= 2]
            if bid_sizes:
                avg_bid_size = float(np.mean(bid_sizes))
                bid_walls = [(float(b[0]), float(b[1])) for b in bids if len(b) >= 2 and float(b[1]) >= self.wall_multiplier * avg_bid_size]

        ask_walls = []
        if asks:
            ask_sizes = [float(a[1]) for a in asks if len(a) >= 2]
            if ask_sizes:
                avg_ask_size = float(np.mean(ask_sizes))
                ask_walls = [(float(a[0]), float(a[1])) for a in asks if len(a) >= 2 and float(a[1]) >= self.wall_multiplier * avg_ask_size]

        return bid_walls, ask_walls

    def get_optimal_scale_out_price(self, direction: str, entry_price: float, default_target_price: float, order_book: Dict) -> float:
        """
        Rule 8: Finds nearest liquidity wall between entry price and default 1.0x ATR target.
        """
        bid_walls, ask_walls = self.find_liquidity_walls(order_book)

        if direction == "Bullish":
            # For longs: find nearest ask wall above entry price and below default target
            candidate_walls = [p for p, q in ask_walls if entry_price < p <= default_target_price]
            if candidate_walls:
                return float(min(candidate_walls))
        else:
            # For shorts: find nearest bid wall below entry price and above default target
            candidate_walls = [p for p, q in bid_walls if default_target_price <= p < entry_price]
            if candidate_walls:
                return float(max(candidate_walls))

        return default_target_price

ob_cluster_engine = OrderBookClusterEngine()
