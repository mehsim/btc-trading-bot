import numpy as np
import threading
from typing import Dict, List, Tuple

from collections import deque

class CUSUMDriftDetector:
    def __init__(self, threshold_H: float = 5.0, allowance_K: float = 0.15, target_error_mu: float = 0.35):
        self.threshold_H = threshold_H  # Decision threshold
        self.allowance_K = allowance_K  # Slack allowance factor
        self.target_error_mu = target_error_mu  # Baseline expected error rate (35% expected loss rate)
        self.lock = threading.Lock()
        self.S_high: float = 0.0
        self.S_low: float = 0.0
        self.error_stream = deque(maxlen=200)

    def update(self, actual_outcome: int, predicted_confidence: float) -> Tuple[bool, float, float]:
        """
        Rule 24: CUSUM Drift Detection:
        actual_outcome: 1 if profitable, 0 if loss
        e_t = 1 - actual_outcome (error stream)
        S_t = max(0, S_{t-1} + e_t - target_error_mu - K)
        Returns: (is_drift_detected, S_high_val, error_rate)
        """
        with self.lock:
            # Error = 1.0 if loss, 0.0 if win
            error_val = 1.0 - float(actual_outcome)
            self.error_stream.append(error_val)
            recent_error_rate = float(np.mean(self.error_stream)) if self.error_stream else self.target_error_mu

            
            # Upper CUSUM accumulator for detecting degradation in accuracy against target baseline
            self.S_high = max(0.0, self.S_high + (error_val - self.target_error_mu - self.allowance_K))
            
            is_drift = self.S_high >= self.threshold_H
            return is_drift, float(self.S_high), recent_error_rate


    def reset(self):
        with self.lock:
            self.S_high = 0.0
            self.S_low = 0.0

def calculate_psi(baseline_data: np.ndarray, target_data: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline training distribution
    and recent live distribution.
    PSI < 0.10: Stable, 0.10 <= PSI < 0.25: Moderate Drift, PSI >= 0.25: Severe Drift
    """
    if len(baseline_data) < 20 or len(target_data) < 20:
        return 0.0
        
    b_arr = np.asarray(baseline_data, dtype=float)
    t_arr = np.asarray(target_data, dtype=float)
    
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(b_arr, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    b_counts, _ = np.histogram(b_arr, bins=bins)
    t_counts, _ = np.histogram(t_arr, bins=bins)
    
    b_pct = b_counts / float(len(b_arr))
    t_pct = t_counts / float(len(t_arr))
    
    # Avoid zero division with small epsilon
    eps = 1e-4
    b_pct = np.maximum(b_pct, eps)
    t_pct = np.maximum(t_pct, eps)
    
    psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
    return float(psi_val)

class PSIDriftDetector:
    def __init__(self, warning_psi: float = 0.10, severe_psi: float = 0.25):
        self.warning_psi = warning_psi
        self.severe_psi = severe_psi

    def check_feature_drift(self, baseline_feature: np.ndarray, live_feature: np.ndarray) -> Tuple[bool, float, str]:
        psi_score = calculate_psi(baseline_feature, live_feature)
        if psi_score >= self.severe_psi:
            return True, psi_score, "SEVERE_DRIFT"
        elif psi_score >= self.warning_psi:
            return False, psi_score, "MODERATE_DRIFT"
        return False, psi_score, "STABLE"

cusum_drift_detector = CUSUMDriftDetector()
psi_drift_detector = PSIDriftDetector()
