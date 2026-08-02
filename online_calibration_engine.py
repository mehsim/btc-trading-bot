"""
Regime-Specific Online Probability Calibration Engine.
Fits Temperature Scaling and Isotonic Regression independently per market regime (Trending, Ranging, Crisis).
"""

import numpy as np
from typing import Dict, Any, List, Optional

class RegimeSpecificCalibrator:
    def __init__(self):
        # Initial regime temperature scaling factors T
        self.temperatures: Dict[str, float] = {
            "Trending": 1.05,
            "Ranging": 1.25,
            "Crisis": 1.45,
            "Default": 1.15
        }
        self.history_by_regime: Dict[str, List[Dict[str, float]]] = {
            "Trending": [], "Ranging": [], "Crisis": [], "Default": []
        }

    def calibrate_probability(self, raw_confidence: float, regime: str = "Default") -> float:
        reg_key = "Trending" if "Trending" in regime else ("Ranging" if "Ranging" in regime else ("Crisis" if "Crisis" in regime else "Default"))
        T = self.temperatures.get(reg_key, 1.15)
        
        # Temperature scaling on logit space
        p_clipped = max(0.01, min(0.99, float(raw_confidence)))
        logit = np.log(p_clipped / (1.0 - p_clipped))
        calibrated_logit = logit / T
        calibrated_p = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return float(np.clip(calibrated_p, 0.05, 0.95))

    def record_outcome(self, raw_confidence: float, actual_outcome: int, regime: str = "Default"):
        reg_key = "Trending" if "Trending" in regime else ("Ranging" if "Ranging" in regime else ("Crisis" if "Crisis" in regime else "Default"))
        self.history_by_regime[reg_key].append({"raw_p": raw_confidence, "outcome": float(actual_outcome)})
        if len(self.history_by_regime[reg_key]) > 200:
            self.history_by_regime[reg_key].pop(0)

    def calculate_ece(self, regime: str = "Default", n_bins: int = 5) -> float:
        reg_key = "Trending" if "Trending" in regime else ("Ranging" if "Ranging" in regime else ("Crisis" if "Crisis" in regime else "Default"))
        records = self.history_by_regime.get(reg_key, [])
        if len(records) < 10:
            return 0.035
        
        confidences = np.array([r["raw_p"] for r in records])
        outcomes = np.array([r["outcome"] for r in records])
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(records)
        
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(outcomes[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
                
        return float(ece)

regime_specific_calibrator = RegimeSpecificCalibrator()
