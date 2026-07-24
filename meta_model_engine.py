import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def apply_bayesian_updating(primary_prob: float, prior_prob: float = 0.50, likelihood_ratio: float = 1.25) -> float:
    """Applies Bayesian prior-to-posterior likelihood ratio updates to avoid overconfidence."""
    prior_odds = prior_prob / (1.0 - prior_prob + 1e-8)
    model_odds = primary_prob / (1.0 - primary_prob + 1e-8)
    posterior_odds = prior_odds * (model_odds ** 0.5) * likelihood_ratio
    posterior_prob = posterior_odds / (1.0 + posterior_odds)
    return float(np.clip(posterior_prob, 0.05, 0.95))

class SupervisedMetaClassifier:
    """Rule 26: Supervised Meta-Classifier to reject trades predicting < 0.45 profit probability."""
    def __init__(self, min_profit_prob: float = 0.45):
        self.min_profit_prob = min_profit_prob
        self.model = LogisticRegression(class_weight="balanced")
        self.is_fitted = False

    def train_meta_classifier(self, feature_matrix: np.ndarray, outcomes: np.ndarray):
        if len(feature_matrix) >= 50 and len(feature_matrix) == len(outcomes):
            try:
                self.model.fit(feature_matrix, outcomes)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False

    def predict_profit_probability(self, features: np.ndarray) -> float:
        if not self.is_fitted:
            return 0.50
        try:
            probs = self.model.predict_proba(features.reshape(1, -1))[0]
            return float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception:
            return 0.50

class MetaModelGatekeeper:
    def __init__(self, confidence_threshold: float = 0.70):
        self.conf_threshold = confidence_threshold
        self.meta_classifier = SupervisedMetaClassifier(min_profit_prob=0.45)

    def is_meta_trade_approved(self, primary_trend: str, primary_conf: float, market_regime: str, recent_win_rate: float, feature_vector: np.ndarray = None) -> tuple:
        if primary_conf < 0.55:
            return False, "REJECTED: Meta-Model Gatekeeper predicts 'No Trade' (Low confidence signal)"
        if recent_win_rate < 40.0:
            return False, f"REJECTED: Meta-Model Gatekeeper (Recent win rate {recent_win_rate:.1f}% < 40%)"

        # Rule 26 check
        if feature_vector is not None and self.meta_classifier.is_fitted:
            profit_prob = self.meta_classifier.predict_profit_probability(feature_vector)
            if profit_prob < 0.45:
                return False, f"REJECTED: Supervised Meta-Classifier predicted profit probability {profit_prob:.2f} < 0.45"

        return True, "APPROVED: Meta-Model gatekeeper confirmed profitability"

class OnlineIsotonicCalibrator:
    """Rule 25: Platt Scaling (Logistic Regression) & Isotonic Probability Calibration."""
    def __init__(self):
        self.platt = LogisticRegression()
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit_on_recent_trades(self, predictions: list, outcomes: list):
        if len(predictions) >= 50 and len(predictions) == len(outcomes):
            try:
                X = np.array(predictions).reshape(-1, 1)
                y = np.array(outcomes)
                self.platt.fit(X, y)
                self.iso.fit(predictions, outcomes)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False

    def calibrate(self, raw_prob: float) -> float:
        if self.is_fitted:
            try:
                platt_prob = float(self.platt.predict_proba([[raw_prob]])[0][1])
                iso_prob = float(self.iso.transform([raw_prob])[0])
                return float(0.5 * platt_prob + 0.5 * iso_prob)
            except Exception:
                return raw_prob
        return raw_prob

meta_gatekeeper = MetaModelGatekeeper()
online_calibrator = OnlineIsotonicCalibrator()
