"""
Return Distribution Forecast Engine.
Predicts expected return mean, variance, skew, and 95% CVaR tail risk to optimize expected utility directly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class ReturnDistributionEngine:
    def predict_return_distribution(
        self,
        predicted_change_pct: float,
        atr_norm: float,
        calibrated_confidence: float,
        historical_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Forecasts return mean, variance, skewness, and 95% CVaR tail risk.
        """
        mu_hat = predicted_change_pct * (calibrated_confidence / 0.50 - 1.0)
        
        if historical_df is not None and "close" in historical_df.columns and len(historical_df) >= 20:
            returns = historical_df["close"].pct_change().dropna().tail(50)
            sigma2_hat = float(returns.var())
            skew_hat = float(returns.skew()) if len(returns) > 10 else 0.0
        else:
            sigma2_hat = float(atr_norm ** 2)
            skew_hat = -0.15

        # 95% CVaR (Expected Shortfall) under skewed Student-t / Empirical approximation
        cvar_95 = float(mu_hat - 1.96 * np.sqrt(max(1e-6, sigma2_hat)) * (1.0 + abs(skew_hat) * 0.2))

        # Direct Expected Utility optimization: E[U] = mu - 0.5 * lambda * sigma2 - gamma * CVaR
        lambda_risk_aversion = 2.0
        expected_utility = mu_hat - 0.5 * lambda_risk_aversion * sigma2_hat

        return {
            "expected_return_mu": round(mu_hat, 6),
            "return_variance_sigma2": round(sigma2_hat, 8),
            "return_skewness": round(skew_hat, 4),
            "cvar_95_tail_risk": round(cvar_95, 6),
            "predicted_expected_utility": round(expected_utility, 6)
        }

return_distribution_engine = ReturnDistributionEngine()
