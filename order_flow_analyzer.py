"""
order_flow_analyzer.py
----------------------
Order Flow Imbalance (OFI), VPIN, Kyle's Lambda, and cancel-to-trade ratio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class OrderFlowAnalyzer:
    def __init__(self, depth_levels: int = 5):
        self.depth_levels = depth_levels

    def compute_ofi_delta(self, orderbook_bids: list, orderbook_asks: list,
                          prev_bids: list = None, prev_asks: list = None) -> float:
        """OFI: normalized [-1.0 sell pressure, +1.0 buy pressure]."""
        if not orderbook_bids or not orderbook_asks:
            return 0.0
        bid_vol = sum(float(b[1]) for b in orderbook_bids[:self.depth_levels])
        ask_vol = sum(float(a[1]) for a in orderbook_asks[:self.depth_levels])
        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return 0.0
        return float(np.clip((bid_vol - ask_vol) / total_vol, -1.0, 1.0))

    def compute_vpin(self, bucket_volumes: List[float], buy_volumes: List[float]) -> Dict[str, Any]:
        """
        VPIN: Volume-Synchronized Probability of Informed Trading.
        VPIN = |V_buy - V_sell| / V_total per bucket, averaged.
        High VPIN (>0.5) signals informed trading activity.
        """
        if not bucket_volumes or not buy_volumes or len(bucket_volumes) != len(buy_volumes):
            return {"vpin": 0.0, "signal": "INSUFFICIENT_DATA"}
        vpins = []
        for v_total, v_buy in zip(bucket_volumes, buy_volumes):
            v_sell = max(0.0, float(v_total) - float(v_buy))
            v_total_f = max(1e-8, float(v_total))
            vpins.append(abs(float(v_buy) - v_sell) / v_total_f)
        vpin = float(np.mean(vpins))
        signal = "HIGH_INFORMED_FLOW" if vpin > 0.5 else "NORMAL_FLOW" if vpin > 0.25 else "LOW_ACTIVITY"
        return {"vpin": round(vpin, 4), "signal": signal, "buckets": len(vpins)}

    def compute_kyle_lambda(self, price_changes: List[float], signed_order_flow: List[float]) -> Dict[str, Any]:
        """
        Kyle's Lambda: price impact per unit of signed order flow.
        lambda = cov(dP, dFlow) / var(dFlow)
        Higher lambda = lower liquidity, each trade moves price more.
        """
        if len(price_changes) < 3 or len(signed_order_flow) < 3:
            return {"kyle_lambda": 0.0, "liquidity": "UNKNOWN"}
        dP = np.array(price_changes, dtype=float)
        dF = np.array(signed_order_flow, dtype=float)
        var_flow = np.var(dF)
        if var_flow < 1e-12:
            return {"kyle_lambda": 0.0, "liquidity": "STABLE"}
        lam = float(np.cov(dP, dF)[0, 1] / var_flow)
        liquidity = "LOW" if lam > 0.01 else "MODERATE" if lam > 0.001 else "HIGH"
        return {"kyle_lambda": round(lam, 6), "liquidity": liquidity}

    def cancel_to_trade_ratio(self, cancels: int, executions: int) -> Dict[str, Any]:
        """
        Cancel-to-Trade Ratio: high ratio signals spoofing or extreme indecision.
        > 10x is typically flagged as suspicious.
        """
        if executions <= 0:
            return {"ctr": 0.0, "signal": "NO_TRADES"}
        ctr = float(cancels) / float(executions)
        signal = "SUSPICIOUS_SPOOFING" if ctr > 10.0 else "ELEVATED" if ctr > 5.0 else "NORMAL"
        return {"ctr": round(ctr, 2), "cancels": cancels, "executions": executions, "signal": signal}


order_flow_analyzer = OrderFlowAnalyzer()

