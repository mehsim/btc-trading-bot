import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class SentimentDecayEngine:
    def __init__(self, default_half_life_periods: float = 14.0):
        self.default_half_life = default_half_life_periods
        self.calibrated_half_life = default_half_life_periods

    def fit_exponential_decay(self, time_lags: List[float], price_responses: List[float]) -> float:
        """
        Fits exponential decay model: return(t) = A * exp(-lambda * t)
        Half-life = ln(2) / lambda
        """
        if len(time_lags) < 10 or len(price_responses) < 10:
            return self.default_half_life

        try:
            lags = np.array(time_lags)
            responses = np.abs(np.array(price_responses))
            valid = responses > 0
            if not np.any(valid):
                return self.default_half_life

            log_resp = np.log(responses[valid])
            clean_lags = lags[valid]

            # Fit linear slope log(y) = log(A) - lambda * t
            poly = np.polyfit(clean_lags, log_resp, 1)
            decay_lambda = -poly[0]

            if decay_lambda <= 0:
                return self.default_half_life

            half_life = float(np.log(2.0) / decay_lambda)
            self.calibrated_half_life = float(np.clip(half_life, 3.0, 48.0))
            return self.calibrated_half_life
        except Exception:
            return self.default_half_life

    def get_decay_factor(self) -> float:
        """
        Rule 20: Returns dynamic decay multiplier = exp(-1 / half_life_periods)
        Replaces fixed 0.95 decay multiplier.
        """
        return float(np.exp(-1.0 / self.calibrated_half_life))

sentiment_decay_engine = SentimentDecayEngine()
