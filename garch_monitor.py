import numpy as np
import pandas as pd
from typing import Tuple

class GARCHVolatilityMonitor:
    def __init__(self, base_circuit_breaker_pct: float = 7.0):
        self.base_circuit_breaker_pct = base_circuit_breaker_pct
        self.long_term_avg_vol = 0.02  # 2% daily avg volatility baseline

    def estimate_garch_volatility(self, returns_series: pd.Series) -> float:
        """
        Fits simple GARCH(1,1) volatility model:
        sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
        Returns forecasted daily volatility.
        """
        clean_returns = returns_series.dropna().values
        if len(clean_returns) < 30:
            return self.long_term_avg_vol

        # GARCH(1,1) standard parameters: omega = 1e-5, alpha = 0.08, beta = 0.90
        omega = 1e-5
        alpha = 0.08
        beta = 0.90

        sigma2 = float(np.var(clean_returns))
        for r in clean_returns[-30:]:
            sigma2 = omega + alpha * (r ** 2) + beta * sigma2

        forecast_vol = float(np.sqrt(max(1e-6, sigma2)))
        return forecast_vol

    def get_dynamic_circuit_breaker_pct(self, btc_returns: pd.Series) -> float:
        """
        Rule 3: Dynamically adjusts daily drawdown halt threshold based on GARCH forecasted volatility:
        Circuit Breaker % = 7.0% * (sigma_forecast / sigma_avg)
        Calm markets: ~5.0% halt threshold. Crisis markets: ~12.0% halt threshold.
        """
        if btc_returns is None or len(btc_returns) < 30:
            return self.base_circuit_breaker_pct

        forecast_vol = self.estimate_garch_volatility(btc_returns)
        long_term_vol = float(np.std(btc_returns.values)) if len(btc_returns) > 0 else self.long_term_avg_vol
        long_term_vol = max(0.005, long_term_vol)

        vol_ratio = forecast_vol / long_term_vol
        dynamic_breaker = self.base_circuit_breaker_pct * vol_ratio

        # Clamp between 4.0% (strict protective limit in chop) and 12.0% (wide limit in volatile trends)
        return float(np.clip(dynamic_breaker, 4.0, 12.0))

garch_vol_monitor = GARCHVolatilityMonitor()
