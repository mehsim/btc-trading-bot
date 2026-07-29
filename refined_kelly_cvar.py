"""
refined_kelly_cvar.py
---------------------
Refined Half-Kelly & CVaR (Expected Shortfall) Tail-Risk Integration Engine.
Calculates position size using Half-Kelly with a max 20% drawdown constraint
and enforces strict CVaR tail-risk limits: position_size = min(kelly_size, cvar_limit, max_drawdown_limit).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class RefinedKellyCVaR:
    def __init__(self, max_drawdown_target: float = 0.20, cvar_confidence: float = 0.95):
        self.max_drawdown_target = max_drawdown_target
        self.cvar_confidence = cvar_confidence

    def calculate_cvar_limit(self, returns_series: pd.Series, equity: float) -> float:
        """
        Calculates 95% CVaR (Expected Shortfall) tail risk dollar limit.
        """
        if returns_series is None or len(returns_series) < 20:
            return equity * 0.10  # Fallback 10% equity limit

        clean_returns = returns_series.dropna().values
        var_threshold = np.percentile(clean_returns, (1 - self.cvar_confidence) * 100)
        tail_returns = clean_returns[clean_returns <= var_threshold]
        
        cvar_pct = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_threshold
        cvar_limit_usd = abs(cvar_pct) * equity * 2.0
        return max(equity * 0.05, min(cvar_limit_usd, equity * 0.20))

    def calculate_refined_position_size(
        self,
        win_rate: float,
        win_loss_ratio: float,
        equity: float,
        returns_series: pd.Series = None
    ) -> Tuple[float, str]:
        """
        Half-Kelly with 20% max drawdown target & CVaR constraint integration:
        position_size = min(half_kelly_size, cvar_limit, max_drawdown_limit)
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return equity * 0.05, "Default 5% (No history)"

        # Full Kelly fraction: f* = p - (1-p)/b
        b = max(0.5, win_loss_ratio)
        p = min(0.95, max(0.05, win_rate))
        full_kelly = p - ((1 - p) / b)
        
        # Half-Kelly for capital preservation
        half_kelly = max(0.02, full_kelly * 0.50)
        kelly_size_usd = equity * half_kelly

        max_dd_limit_usd = equity * self.max_drawdown_target
        cvar_limit_usd = self.calculate_cvar_limit(returns_series, equity)

        final_size_usd = min(kelly_size_usd, cvar_limit_usd, max_dd_limit_usd)
        reason = f"Half-Kelly (${kelly_size_usd:.2f}) constrained by CVaR (${cvar_limit_usd:.2f}) & MaxDD (${max_dd_limit_usd:.2f})"
        return final_size_usd, reason

refined_kelly_cvar = RefinedKellyCVaR()
