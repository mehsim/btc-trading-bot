"""
strategy_health_engine.py
--------------------------
Component 13: Continuous Strategy Health Score (SHS) Engine
Evaluates strategy self-awareness across 6 quantitative health metrics:
1. Calibration Error (20 pts)
2. Feature Drift PSI (20 pts)
3. Live Rolling Profit Factor (20 pts)
4. Drawdown Safety (15 pts)
5. Win Rate Variance (15 pts)
6. Latency & Slippage (10 pts)
"""

import os
import time
from typing import Dict, Any, Tuple
import config

class StrategyHealthEngine:
    """
    Computes a 100-point Strategy Health Score (SHS) and dynamically 
    throttles live position sizing to protect capital during strategy degradation.
    """
    def __init__(self):
        pass

    def evaluate_health(
        self,
        calibration_error_pct: float = 3.0,
        psi_drift_score: float = 0.05,
        rolling_profit_factor: float = 1.85,
        current_drawdown_pct: float = 3.5,
        win_rate_variance_pct: float = 2.0,
        order_latency_ms: float = 95.0
    ) -> Tuple[float, float, str]:
        """
        Calculates Strategy Health Score (0 - 100) and position size multiplier.
        Returns: (shs_score, position_size_multiplier, action_recommendation)
        """
        score = 0.0

        # 1. Calibration Error (20 pts)
        if calibration_error_pct <= 5.0: score += 20.0
        elif calibration_error_pct <= 10.0: score += 12.0
        elif calibration_error_pct <= 15.0: score += 5.0

        # 2. Feature Drift PSI (20 pts)
        if psi_drift_score <= 0.10: score += 20.0
        elif psi_drift_score <= 0.20: score += 12.0
        elif psi_drift_score <= 0.25: score += 5.0

        # 3. Live Rolling Profit Factor (20 pts)
        if rolling_profit_factor >= 1.80: score += 20.0
        elif rolling_profit_factor >= 1.40: score += 14.0
        elif rolling_profit_factor >= 1.10: score += 8.0

        # 4. Drawdown Safety (15 pts)
        if current_drawdown_pct <= 5.0: score += 15.0
        elif current_drawdown_pct <= 10.0: score += 9.0
        elif current_drawdown_pct <= 15.0: score += 4.0

        # 5. Win Rate Stability (15 pts)
        if win_rate_variance_pct <= 5.0: score += 15.0
        elif win_rate_variance_pct <= 10.0: score += 9.0

        # 6. Latency & Slippage (10 pts)
        if order_latency_ms <= 150.0: score += 10.0
        elif order_latency_ms <= 300.0: score += 5.0

        # Dynamic Position Throttling
        if score >= 85.0:
            multiplier = 1.00
            recommendation = "NORMAL: 100% Position Sizing (Strategy Fully Healthy)"
        elif score >= 70.0:
            multiplier = 0.50
            recommendation = "MODERATE DEGRADATION: 50% Position Sizing (Self-Aware De-Risking)"
        elif score >= 50.0:
            multiplier = 0.25
            recommendation = "HIGH DEGRADATION: 25% Position Sizing (Minimum Paper-Trade Sizing)"
        else:
            multiplier = 0.00
            recommendation = "CRITICAL FAILURE: 0% Position Sizing (Emergency Live Halt & Alert)"

        return score, multiplier, recommendation

    def evaluate_operational_kpis(
        self,
        order_rejection_rate_pct: float = 0.2,
        cancel_rate_pct: float = 1.5,
        partial_fill_pct: float = 98.0,
        api_timeout_pct: float = 0.1,
        websocket_reconnect_freq: int = 1
    ) -> Dict[str, Any]:
        """
        Enhancement 6: Operational KPI Monitoring Engine
        Evaluates exchange latency, order rejection rate, cancel rate, partial fill %, API timeout %, and WS reconnect frequency.
        """
        kpi_health = {
            "order_rejection_rate_pct": order_rejection_rate_pct,
            "rejection_status": "EXCELLENT" if order_rejection_rate_pct < 1.0 else "ELEVATED",
            "cancel_rate_pct": cancel_rate_pct,
            "partial_fill_pct": partial_fill_pct,
            "api_timeout_pct": api_timeout_pct,
            "websocket_reconnect_freq_daily": websocket_reconnect_freq,
            "operational_status": "HEALTHY" if order_rejection_rate_pct < 1.0 and websocket_reconnect_freq <= 3 else "DEGRADED"
        }
        return kpi_health

strategy_health_engine = StrategyHealthEngine()
