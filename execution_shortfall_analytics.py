"""
Execution Quality & Implementation Shortfall Analytics Engine.
Tracks decision arrival price, VWAP, execution latency, and implementation shortfall per order.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional

class ExecutionShortfallAnalytics:
    def __init__(self):
        self.execution_logs: List[Dict[str, Any]] = []

    def record_execution_telemetry(
        self,
        symbol: str,
        direction: str,
        decision_timestamp: float,
        order_sent_timestamp: float,
        fill_timestamp: float,
        arrival_price: float,
        fill_price: float,
        requested_qty: float,
        filled_qty: float,
        top_book_depth_usd: float = 50000.0
    ) -> Dict[str, Any]:
        latency_ms = max(0.0, (fill_timestamp - decision_timestamp) * 1000.0)
        
        # Implementation Shortfall in basis points (bps)
        dir_upper = direction.upper()
        if dir_upper in ("BUY", "LONG", "BULLISH"):
            shortfall_pct = (fill_price - arrival_price) / arrival_price if arrival_price > 0 else 0.0
        else:
            shortfall_pct = (arrival_price - fill_price) / arrival_price if arrival_price > 0 else 0.0

        shortfall_bps = round(shortfall_pct * 10000.0, 2)
        fill_ratio = round(filled_qty / requested_qty, 4) if requested_qty > 0 else 1.0
        market_impact_bps = round((requested_qty * fill_price / top_book_depth_usd) * 10000.0, 2) if top_book_depth_usd > 0 else 0.0

        # Dynamic Execution Quality Thresholds based on rolling history
        recent_bps = [log["implementation_shortfall_bps"] for log in self.execution_logs[-50:]] if self.execution_logs else []
        avg_bps = float(np.mean(recent_bps)) if recent_bps else 5.0
        excellent_thresh = max(3.0, min(10.0, avg_bps))
        acceptable_thresh = max(10.0, min(25.0, avg_bps * 2.5))

        execution_quality = "EXCELLENT" if shortfall_bps <= excellent_thresh else ("ACCEPTABLE" if shortfall_bps <= acceptable_thresh else "POOR")

        telemetry = {
            "symbol": symbol,
            "direction": direction,
            "decision_timestamp": decision_timestamp,
            "fill_timestamp": fill_timestamp,
            "latency_ms": round(latency_ms, 2),
            "arrival_price": arrival_price,
            "fill_price": fill_price,
            "implementation_shortfall_bps": shortfall_bps,
            "fill_ratio": fill_ratio,
            "market_impact_bps": market_impact_bps,
            "execution_quality": execution_quality
        }

        self.execution_logs.append(telemetry)
        if len(self.execution_logs) > 500:
            self.execution_logs.pop(0)

        return telemetry

execution_shortfall_analytics = ExecutionShortfallAnalytics()
