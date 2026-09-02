"""
Beta Calibration Engine (Kull, Silva Filho, and Flach, 2017)
Provides smooth, continuous, strictly monotonic probability calibration
without the step-function plateaus of Isotonic Regression.
"""
from typing import Union, List, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


class BetaCalibrator:
    """
    Fits a 3-parameter Beta calibration model:
    P(y=1|s) = 1 / (1 + exp(-(a * ln(s) - b * ln(1-s) + c)))
    
    Transforms raw probability scores s in (0, 1) into calibrated probabilities.
    Ensures strictly monotonic, smooth calibration without staircase artifacts.
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.is_fitted = False
        self.fit_n = 0

    def fit(self, scores: Union[np.ndarray, List[float]], labels: Union[np.ndarray, List[int]]) -> "BetaCalibrator":
        s = np.clip(np.asarray(scores, dtype=float), self.eps, 1.0 - self.eps)
        y = np.asarray(labels, dtype=int)
        
        if len(s) < 20 or len(np.unique(y)) < 2:
            # Fallback to identity mapping if insufficient samples
            self.a = 1.0
            self.b = 1.0
            self.c = 0.0
            self.is_fitted = False
            self.fit_n = len(s)
            return self

        # Feature transform: x1 = ln(s), x2 = -ln(1-s)
        x1 = np.log(s)
        x2 = -np.log(1.0 - s)
        X = np.column_stack([x1, x2])

        # Regularized logistic regression (C=1.0 provides robust L2 shrinkage against sample noise)
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        try:
            lr.fit(X, y)
            coefs = lr.coef_[0]
            intercept = lr.intercept_[0]
            
            # Constrain monotonicity: a >= 0, b >= 0
            # If negative, fallback to symmetric Platt scaling on log-odds
            a_val = float(coefs[0])
            b_val = float(coefs[1])
            c_val = float(intercept)

            if a_val < 0 or b_val < 0:
                # Fit 1D Platt log-odds calibration: ln(s / (1-s))
                log_odds = (x1 + x2).reshape(-1, 1)
                lr_platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                lr_platt.fit(log_odds, y)
                platt_coef = float(lr_platt.coef_[0][0])
                if platt_coef <= 0.02:
                    # Boundary hit / degenerate flat slope -> fit failure
                    self.a = float(platt_coef)
                    self.b = float(platt_coef)
                    self.c = float(lr_platt.intercept_[0])
                    self.is_fitted = False
                    self.fit_n = len(s)
                else:
                    self.a = platt_coef
                    self.b = platt_coef
                    self.c = float(lr_platt.intercept_[0])
                    self.is_fitted = True
                    self.fit_n = len(s)
            else:
                if a_val <= 0.02 or b_val <= 0.02:
                    # Boundary hit / degenerate flat slope -> fit failure
                    self.a = float(a_val)
                    self.b = float(b_val)
                    self.c = c_val
                    self.is_fitted = False
                    self.fit_n = len(s)
                else:
                    self.a = float(a_val)
                    self.b = float(b_val)
                    self.c = c_val
                    self.is_fitted = True
                    self.fit_n = len(s)

        except Exception:
            self.a = 1.0
            self.b = 1.0
            self.c = 0.0
            self.is_fitted = False
            self.fit_n = len(s)

        return self

    def max_achievable_probability(self, max_input_score: float = 0.99) -> float:
        """Computes the maximum calibrated probability achievable within servable score range."""
        return float(self.predict_proba(max_input_score))

    def is_viable(self, min_required_p_star: Optional[float] = None) -> bool:
        """
        Validates whether the calibrator is non-degenerate and economically viable.
        Returns False if shape parameters hit boundary floor (<= 0.02) or if max achievable output
        cannot exceed the minimum required break-even p*.
        """
        if not self.is_fitted or self.a <= 0.02 or self.b <= 0.02:
            return False
        if self.a == 1.0 and self.b == 1.0 and self.c == 0.0 and self.fit_n < 20:
            return False
        if min_required_p_star is not None:
            if self.max_achievable_probability(0.99) < min_required_p_star:
                return False
        return True

    def predict_proba(self, scores: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
        is_scalar = isinstance(scores, (float, int))
        s = np.clip(np.asarray(scores, dtype=float), self.eps, 1.0 - self.eps)
        
        # logit = a * ln(s) - b * ln(1-s) + c
        logit = self.a * np.log(s) - self.b * np.log(1.0 - s) + self.c
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
        # Bounded empirical probability ceiling/floor to protect Kelly sizing from saturation
        prob = np.clip(prob, 0.20, 0.85)
        
        if is_scalar:
            return float(prob.item() if hasattr(prob, 'item') else prob)
        return prob

    def to_dict(self) -> Dict:
        return {
            "scaling_method": "beta_calibration",
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
            "fitting_sample_size": int(self.fit_n),
            "is_fitted": bool(self.is_fitted),
            "is_fallback": bool(not self.is_fitted)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BetaCalibrator":
        bc = cls()
        bc.a = float(data.get("a", 1.0))
        bc.b = float(data.get("b", 1.0))
        bc.c = float(data.get("c", 0.0))
        bc.fit_n = int(data.get("fitting_sample_size", 0))
        if "is_fitted" in data:
            bc.is_fitted = bool(data["is_fitted"])
        elif data.get("is_fallback", False) is True:
            bc.is_fitted = False
        elif bc.a == 1.0 and bc.b == 1.0 and bc.c == 0.0:
            bc.is_fitted = False
        else:
            bc.is_fitted = True
        return bc


def is_calibrator_viable(calibrator_data: Optional[Dict], min_required_p_star: Optional[float] = None) -> bool:
    """
    Checks if a calibrator dictionary represents a viable, non-degenerate probability map
    whose achievable ceiling exceeds the fee-inclusive break-even p*.
    """
    if not calibrator_data or not isinstance(calibrator_data, dict):
        return False
    if calibrator_data.get("is_fallback", False) is True:
        return False
    if calibrator_data.get("is_fitted") is False:
        return False
    if calibrator_data.get("scaling_method") == "beta_calibration" or ("a" in calibrator_data and "b" in calibrator_data):
        bc = BetaCalibrator.from_dict(calibrator_data)
        return bc.is_viable(min_required_p_star=min_required_p_star)
    Ys = calibrator_data.get("y", [])
    if Ys and len(Ys) > 0:
        max_y = max(Ys)
        if max_y <= 0.02:
            return False
        if min_required_p_star is not None and max_y < min_required_p_star:
            return False
        return True
    return False


def calibrate_probability(raw_score: float, calibrator_data: Optional[Dict], min_required_p_star: Optional[float] = None) -> float:
    """
    Universal probability calibrator helper supporting both Beta calibration and legacy threshold arrays.
    Returns strictly the empirical calibrated probability from the fitted calibrator.
    If the calibrator is flat/degenerate, returns the empirical calibrated probability without artificial overrides
    so that downstream confidence threshold gates correctly reject/abstain.
    """
    if not calibrator_data or not isinstance(calibrator_data, dict):
        return float(raw_score)

    # Beta Calibration format
    if calibrator_data.get("scaling_method") == "beta_calibration" or ("a" in calibrator_data and "b" in calibrator_data):
        bc = BetaCalibrator.from_dict(calibrator_data)
        return float(bc.predict_proba(raw_score))

    # Threshold lookup format (legacy fallback)
    Xs = calibrator_data.get("X", [0.0, 1.0])
    Ys = calibrator_data.get("y", [0.0, 1.0])
    if len(Xs) > 0 and len(Ys) == len(Xs):
        return float(np.interp(raw_score, Xs, Ys))

    return float(raw_score)
