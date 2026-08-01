"""
execution_simulator.py
-----------------------
Pre-trade Execution Simulator. Before sending a live order, simulates:
- Queue position and estimated wait time
- Partial fill probability
- Network latency estimate
- Market impact and slippage

Improves backtest fidelity and informs execution strategy selection.
"""

import numpy as np
from typing import Dict, Any

class ExecutionSimulator:

    def simulate(
        self,
        order_size_usd: float = 1000.0,
        orderbook_depth_usd: float = 150_000.0,
        bid_ask_spread_bp: float = 1.5,
        garch_sigma: float = 0.015,
        avg_fill_latency_ms: float = 45.0,
        is_maker: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates execution quality before placing a live order.

        Parameters:
        - order_size_usd: size of the order in USD
        - orderbook_depth_usd: total liquidity at top 5 levels
        - bid_ask_spread_bp: current bid-ask spread in basis points
        - garch_sigma: live GARCH realized volatility (from garch_monitor)
        - avg_fill_latency_ms: average network RTT to exchange in ms
        - is_maker: True if limit order, False if market/taker
        """
        # Queue position estimate: fraction of depth consumed
        depth_ratio = float(order_size_usd) / max(1.0, float(orderbook_depth_usd))
        queue_position_pct = round(min(depth_ratio * 100.0, 100.0), 2)

        # Partial fill risk: high if order > 15% of visible depth
        partial_fill_risk = "HIGH" if depth_ratio > 0.15 else "MEDIUM" if depth_ratio > 0.05 else "LOW"
        partial_fill_prob = round(min(depth_ratio * 3.0, 1.0), 4)

        # Slippage estimate (Almgren-Chriss-style, matches TCM formula)
        gamma = 0.42
        sigma = max(0.001, float(garch_sigma))
        slippage_bp = gamma * sigma * np.sqrt(depth_ratio) * 10000.0
        half_spread = float(bid_ask_spread_bp) / 2.0
        total_cost_bp = round(half_spread + slippage_bp, 3)

        # Latency estimate with jitter
        jitter_ms = avg_fill_latency_ms * 0.2 * (0.5 + depth_ratio)
        expected_latency_ms = round(avg_fill_latency_ms + jitter_ms, 1)

        # Expected fill price adjustment (in bps from mid)
        fill_price_offset_bp = total_cost_bp * (1 if not is_maker else -0.3)

        execution_grade = (
            "EXCELLENT" if total_cost_bp < 5 else
            "GOOD" if total_cost_bp < 10 else
            "FAIR" if total_cost_bp < 20 else
            "POOR"
        )

        return {
            "order_size_usd": order_size_usd,
            "queue_position_pct": queue_position_pct,
            "partial_fill_risk": partial_fill_risk,
            "partial_fill_probability": partial_fill_prob,
            "expected_slippage_bp": round(slippage_bp, 3),
            "expected_total_cost_bp": total_cost_bp,
            "fill_price_offset_bp": round(fill_price_offset_bp, 3),
            "expected_fill_latency_ms": expected_latency_ms,
            "execution_grade": execution_grade,
            "recommendation": "PROCEED" if execution_grade in ("EXCELLENT", "GOOD") else "CAUTION" if execution_grade == "FAIR" else "ABORT"
        }


execution_simulator = ExecutionSimulator()
