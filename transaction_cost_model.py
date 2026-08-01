"""
transaction_cost_model.py
----------------------------
Institutional Transaction Cost Model (TCM) & Market Impact Estimator.
Estimates total transaction cost including taker/maker exchange fees, bid-ask spread,
and orderbook market impact based on order size relative to 24-hour volume.
"""

import numpy as np
from typing import Dict, Any

class TransactionCostModel:
    def __init__(self, default_taker_fee_bp: float = 6.0, default_maker_fee_bp: float = 1.0):
        self.taker_fee_bp = default_taker_fee_bp
        self.maker_fee_bp = default_maker_fee_bp

    def estimate_transaction_cost(
        self,
        symbol: str = "BTCUSDT",
        order_size_usd: float = 1000.0,
        volume_24h_usd: float = 50000000.0,
        bid_ask_spread_bp: float = 1.5,
        is_maker: bool = False
    ) -> Dict[str, Any]:
        """
        Estimates total trade cost in basis points and USD.
        Formula: Slippage_bps = Base_Spread_bps / 2 + Gamma * sqrt(Order_Size / Volume_24h) * 10000
        """
        fee_bp = self.maker_fee_bp if is_maker else self.taker_fee_bp
        half_spread_bp = float(bid_ask_spread_bp) / 2.0

        # Market impact scaling factor gamma
        gamma = 0.50
        liquidity_ratio = max(1e-8, float(order_size_usd) / max(1.0, float(volume_24h_usd)))
        market_impact_bp = gamma * np.sqrt(liquidity_ratio) * 10000.0

        total_cost_bp = round(fee_bp + half_spread_bp + market_impact_bp, 2)
        total_cost_usd = round((total_cost_bp / 10000.0) * float(order_size_usd), 4)

        # Threshold check: cost > 25 bps is flagged as high cost
        is_acceptable = total_cost_bp <= 25.0

        return {
            "symbol": symbol,
            "order_size_usd": order_size_usd,
            "fee_bp": fee_bp,
            "half_spread_bp": half_spread_bp,
            "market_impact_bp": round(market_impact_bp, 2),
            "total_cost_bp": total_cost_bp,
            "total_cost_usd": total_cost_usd,
            "is_acceptable": is_acceptable
        }

transaction_cost_model = TransactionCostModel()
