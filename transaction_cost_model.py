"""
transaction_cost_model.py
----------------------------
Institutional TCM with Almgren-Chriss volatility-adjusted market impact:
  Slippage_bps = Half_Spread + gamma * sigma * sqrt(Order_Size / Volume_24h) * 10000
"""

import numpy as np
from typing import Dict, Any

class TransactionCostModel:
    def __init__(self, default_taker_fee_bp: float = 6.0, default_maker_fee_bp: float = 1.0,
                 gamma: float = 0.42, max_acceptable_cost_bp: float = 25.0):
        self.taker_fee_bp = default_taker_fee_bp
        self.maker_fee_bp = default_maker_fee_bp
        self.gamma = gamma
        self.max_acceptable_cost_bp = max_acceptable_cost_bp

    def estimate_transaction_cost(
        self,
        symbol: str = "BTCUSDT",
        order_size_usd: float = 1000.0,
        volume_24h_usd: float = 50_000_000.0,
        bid_ask_spread_bp: float = 1.5,
        garch_sigma: float = 0.015,  # Almgren-Chriss: live realized vol from garch_monitor
        is_maker: bool = False,
        max_acceptable_cost_bp: float = None,
        orderbook_depth_usd: float = None
    ) -> Dict[str, Any]:
        """
        Almgren-Chriss extended market impact against instantaneous orderbook depth:
        Slippage_bps = Half_Spread + gamma * sigma * sqrt(Order_Size / Orderbook_Depth) * 100
        """
        fee_bp = self.maker_fee_bp if is_maker else self.taker_fee_bp
        half_spread_bp = float(bid_ask_spread_bp) / 2.0

        depth_usd = float(orderbook_depth_usd) if orderbook_depth_usd is not None and orderbook_depth_usd > 0 else max(10_000.0, float(volume_24h_usd) / 200.0)
        liquidity_ratio = max(1e-8, float(order_size_usd) / depth_usd)
        # Almgren-Chriss: volatility-adjusted impact relative to orderbook depth (fraction → bps requires ×10000)
        sigma = max(0.001, float(garch_sigma))
        market_impact_bp = self.gamma * sigma * np.sqrt(liquidity_ratio) * 10000.0

        total_cost_bp = round(fee_bp + half_spread_bp + market_impact_bp, 2)
        total_cost_usd = round((total_cost_bp / 10000.0) * float(order_size_usd), 4)
        cutoff = max_acceptable_cost_bp if max_acceptable_cost_bp is not None else self.max_acceptable_cost_bp
        is_acceptable = total_cost_bp <= cutoff

        return {
            "symbol": symbol,
            "order_size_usd": order_size_usd,
            "garch_sigma_input": round(sigma, 4),
            "fee_bp": fee_bp,
            "half_spread_bp": half_spread_bp,
            "market_impact_bp": round(market_impact_bp, 4),
            "total_cost_bp": total_cost_bp,
            "total_cost_usd": total_cost_usd,
            "is_acceptable": is_acceptable,
            "model": "Almgren-Chriss"
        }

transaction_cost_model = TransactionCostModel(gamma=0.42)

