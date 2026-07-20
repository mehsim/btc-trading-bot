import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def apply_bayesian_updating(primary_prob: float, prior_prob: float = 0.50, likelihood_ratio: float = 1.25) -> float:
    """Applies Bayesian prior-to-posterior likelihood ratio updates to avoid overconfidence."""
    prior_odds = prior_prob / (1.0 - prior_prob + 1e-8)
    model_odds = primary_prob / (1.0 - primary_prob + 1e-8)
    posterior_odds = prior_odds * (model_odds ** 0.5) * likelihood_ratio
    posterior_prob = posterior_odds / (1.0 + posterior_odds)
    return float(np.clip(posterior_prob, 0.05, 0.95))

class MetaModelGatekeeper:
    def __init__(self, confidence_threshold: float = 0.70):
        self.conf_threshold = confidence_threshold

    def is_meta_trade_approved(self, primary_trend: str, primary_conf: float, market_regime: str, recent_win_rate: float) -> tuple:
        # Gatekeeper "Hold" check: if prediction confidence is low or recent win rate is poor, hold trade
        if primary_conf < 0.55:
            return False, "REJECTED: Meta-Model Gatekeeper predicts 'No Trade' (Low confidence signal)"
        if recent_win_rate < 40.0:
            return False, f"REJECTED: Meta-Model Gatekeeper (Recent win rate {recent_win_rate:.1f}% < 40%)"
        return True, "APPROVED: Meta-Model gatekeeper confirmed profitability"

class OnlineIsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit_on_recent_trades(self, predictions: list, outcomes: list):
        if len(predictions) >= 50 and len(predictions) == len(outcomes):
            self.iso.fit(predictions, outcomes)
            self.is_fitted = True

    def calibrate(self, raw_prob: float) -> float:
        if self.is_fitted:
            return float(self.iso.transform([raw_prob])[0])
        return raw_prob

meta_gatekeeper = MetaModelGatekeeper()
online_calibrator = OnlineIsotonicCalibrator()
