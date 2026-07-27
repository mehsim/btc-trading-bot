import numpy as np
import threading
from typing import Dict, List, Tuple

class CUSUMDriftDetector:
    def __init__(self, threshold_H: float = 5.0, allowance_K: float = 0.15, target_error_mu: float = 0.35):
        self.threshold_H = threshold_H  # Decision threshold
        self.allowance_K = allowance_K  # Slack allowance factor
        self.target_error_mu = target_error_mu  # Baseline expected error rate (35% expected loss rate)
        self.lock = threading.Lock()
        self.S_high: float = 0.0
        self.S_low: float = 0.0
        self.error_stream: List[float] = []

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
            if len(self.error_stream) > 200:
                self.error_stream.pop(0)

            recent_error_rate = float(np.mean(self.error_stream)) if self.error_stream else self.target_error_mu
            
            # Upper CUSUM accumulator for detecting degradation in accuracy against target baseline
            self.S_high = max(0.0, self.S_high + (error_val - self.target_error_mu - self.allowance_K))
            
            is_drift = self.S_high >= self.threshold_H
            return is_drift, float(self.S_high), recent_error_rate


    def reset(self):
        with self.lock:
            self.S_high = 0.0
            self.S_low = 0.0

cusum_drift_detector = CUSUMDriftDetector()
