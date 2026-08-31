import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class ProductionRegimeEngine:
    """
    Production-grade Market Regime Detector with v2 4-State ADX Hysteresis (32/28)
    and Volatility-Adjusted Confluence Safeguards.
    """
    def __init__(self, strong_enter: float = 32.0, strong_exit: float = 28.0, mod_low: float = 22.0, adx_high: Optional[float] = None, adx_low: Optional[float] = None):
        self.legacy_mode = adx_high is not None
        self.strong_enter = adx_high if adx_high is not None else strong_enter
        self.strong_exit = adx_low if adx_low is not None else strong_exit
        self.mod_low = mod_low if mod_low is not None else 22.0
        self._regime_state: Dict[str, str] = {}

    def update_regime(
        self, 
        symbol: str = "DEFAULT", 
        interval: str = "60", 
        adx_value: float = 20.0, 
        volatility_ratio: float = 1.0,
        choppiness: float = 50.0,
        bb_width_pct: float = 50.0,
        volume_ratio_20d: float = 1.0
    ) -> str:
        """
        Updates market regime using ADX hysteresis to prevent signal flickering.
        """
        key = f"{symbol}_{interval}"
        current = self._regime_state.get(key, "RANGING")

        adx = float(adx_value)
        vol = float(volatility_ratio)
        chop = float(choppiness)

        if self.legacy_mode:
            if current == "RANGING":
                if adx > self.strong_enter and vol > 1.2:
                    current = "TRENDING"
            elif current in ["TRENDING", "STRONG_TREND"]:
                if adx < self.strong_exit or vol < 0.8:
                    current = "RANGING"
            self._regime_state[key] = current
            return current

        # Check CHOPPY hard state first
        if adx < 20.0 and chop > 60.0 and bb_width_pct < 15.0 and volume_ratio_20d < 1.0:
            current = "CHOPPY"
        elif current == "STRONG_TREND":
            if adx < self.strong_exit:
                current = "MODERATE_TREND" if adx >= self.mod_low else "RANGING"
        elif current in ["RANGING", "MODERATE_TREND", "CHOPPY"]:
            if adx >= self.strong_enter:
                current = "STRONG_TREND"
            elif adx >= self.mod_low:
                current = "MODERATE_TREND"
            else:
                current = "RANGING"

        self._regime_state[key] = current
        return current

    def _normalize_regime(self, regime: str) -> str:
        """
        Normalizes any regime string representation to canonical tokens:
        'STRONG_TREND', 'MODERATE_TREND', 'TRENDING', 'RANGING', 'CHOPPY'
        """
        if not regime or not isinstance(regime, str):
            return "RANGING"
        r_up = regime.upper().strip()
        
        # Exact canonical match
        if r_up in ["STRONG_TREND", "MODERATE_TREND", "TRENDING", "RANGING", "CHOPPY"]:
            return r_up
            
        # classify_market_regime mappings ("High Vol, Ranging" -> CHOPPY, "Low Vol, Ranging" -> RANGING, etc.)
        if "HIGH VOL" in r_up and "RANGING" in r_up:
            return "CHOPPY"
        if "RANGING" in r_up:
            return "RANGING"
        if "HIGH VOL" in r_up and "TREND" in r_up:
            return "STRONG_TREND"
        if "LOW VOL" in r_up and "TREND" in r_up:
            return "MODERATE_TREND"
        if "TREND" in r_up:
            return "TRENDING"
        if "CHOP" in r_up:
            return "CHOPPY"
            
        try:
            from logger import log_event
            log_event("WARNING", f"[Regime Normalization Warning] Unrecognized regime token '{regime}'. Defaulting to RANGING fail-safe.")
        except Exception:
            pass
        return "RANGING"

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

        norm_regime = self._normalize_regime(regime)

        if norm_regime == "CHOPPY":
            return {"execute": False, "reason": "Blocked: CHOPPY Market Regime (No Edge)"}

        if ml_direction == "Bullish":
            if norm_regime == "RANGING" and rsi > 70.0:
                return {"execute": False, "reason": "Overbought in Ranging Regime (RSI > 70)"}
            if norm_regime in ["TRENDING", "STRONG_TREND", "MODERATE_TREND"] and rsi < 25.0:
                return {"execute": False, "reason": "Extreme Counter-Trend Exhaustion (RSI < 25)"}
                
        elif ml_direction == "Bearish":
            if norm_regime == "RANGING" and rsi < 30.0:
                return {"execute": False, "reason": "Oversold in Ranging Regime (RSI > 30)"}
            if norm_regime in ["TRENDING", "STRONG_TREND", "MODERATE_TREND"] and rsi > 75.0:
                return {"execute": False, "reason": "Extreme Counter-Trend Exhaustion (RSI > 75)"}

        return {"execute": True, "reason": "Signal Confluence Confirmed"}

production_regime_engine = ProductionRegimeEngine()

