import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class ProductionRegimeEngine:
    """
    Production-grade Market Regime Detector with ADX Hysteresis 
    and Volatility-Adjusted Confluence Safeguards.
    """
    def __init__(self, adx_high: float = 26.0, adx_low: float = 22.0):
        self.adx_high = adx_high
        self.adx_low = adx_low
        self._regime_state: Dict[str, str] = {}

    def update_regime(self, symbol: str = "DEFAULT", interval: str = "60", adx_value: float = 20.0, volatility_ratio: float = 1.0) -> str:
        """
        Updates market regime using ADX hysteresis to prevent signal flickering.
        - Transition to TRENDING when ADX > 26.0 and volatility_ratio > 1.2
        - Transition back to RANGING only when ADX < 22.0 or volatility_ratio < 0.8
        """
        key = f"{symbol}_{interval}"
        current = self._regime_state.get(key, "RANGING")

        if current == "RANGING":
            if float(adx_value) > self.adx_high and float(volatility_ratio) > 1.2:
                current = "TRENDING"
        elif current == "TRENDING":
            if float(adx_value) < self.adx_low or float(volatility_ratio) < 0.8:
                current = "RANGING"

        self._regime_state[key] = current
        return current

    def evaluate_confluence(
        self, 
        ml_direction: str, 
        regime: str, 
        rsi: float, 
        macro_guard_active: bool = False
    ) -> Dict[str, Any]:
        """
        Validates ML signal against macro guards and indicator confluence.
        ml_direction: "Bullish", "Bearish", "Neutral"
        """
        if macro_guard_active:
            return {"execute": False, "reason": "Blocked by Economic Event Guard"}

        if ml_direction == "Bullish":
            if regime == "RANGING" and rsi > 65.0:
                return {"execute": False, "reason": "Overbought in Ranging Regime (RSI > 65)"}
            if regime == "TRENDING" and rsi < 40.0:
                return {"execute": False, "reason": "Counter-trend momentum mismatch (RSI < 40)"}
                
        elif ml_direction == "Bearish":
            if regime == "RANGING" and rsi < 35.0:
                return {"execute": False, "reason": "Oversold in Ranging Regime (RSI < 35)"}
            if regime == "TRENDING" and rsi > 60.0:
                return {"execute": False, "reason": "Counter-trend momentum mismatch (RSI > 60)"}

        return {"execute": True, "reason": "Signal Confluence Confirmed"}

production_regime_engine = ProductionRegimeEngine()
