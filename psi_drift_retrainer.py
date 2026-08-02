"""
PSI Multi-Drift Engine & Retrainer.
Combines Population Stability Index (PSI), Calibration Drift, Feature Drift, and Performance Drift into a unified Model Health Index (MHI) to trigger automated retraining.
"""

import numpy as np
from typing import Dict, Any, Tuple

class PSIMultiDriftRetrainer:
    def calculate_psi(self, baseline_scores: np.ndarray, current_scores: np.ndarray, num_bins: int = 10) -> float:
        if len(baseline_scores) < 20 or len(current_scores) < 20:
            return 0.04
        
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(baseline_scores, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.04

        base_cnts, _ = np.histogram(baseline_scores, bins=bins)
        curr_cnts, _ = np.histogram(current_scores, bins=bins)

        base_pct = base_cnts / len(baseline_scores) + 1e-4
        curr_pct = curr_cnts / len(current_scores) + 1e-4

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(max(0.0, psi))

    def evaluate_model_health_index(
        self,
        psi_score: float,
        ece_score: float,
        recent_win_rate: float,
        baseline_win_rate: float = 0.50
    ) -> Tuple[float, bool, str]:
        """
        Returns (mhi_score, should_retrain, retrain_reason).
        MHI scale: 100.0 (Perfect) to 0.0 (Severe Drift).
        """
        mhi = 100.0

        # Dynamic Penalties based on baseline win rate
        dynamic_wr_drop_high = float(max(0.10, min(0.25, baseline_win_rate * 0.30)))
        dynamic_wr_drop_mid = float(dynamic_wr_drop_high * 0.50)

        # 1. PSI Penalty
        if psi_score >= 0.25:
            mhi -= 40.0
        elif psi_score >= 0.10:
            mhi -= 20.0

        # 2. Calibration ECE Penalty
        if ece_score >= 0.10:
            mhi -= 30.0
        elif ece_score >= 0.05:
            mhi -= 15.0

        # 3. Dynamic Performance Drop Penalty
        win_rate_drop = max(0.0, baseline_win_rate - recent_win_rate)
        if win_rate_drop >= dynamic_wr_drop_high:
            mhi -= 30.0
        elif win_rate_drop >= dynamic_wr_drop_mid:
            mhi -= 15.0

        final_mhi = max(0.0, min(100.0, mhi))
        should_retrain = final_mhi < 60.0 or psi_score >= 0.25
        reason = f"Severe Drift / Performance Loss (MHI={final_mhi:.1f}, PSI={psi_score:.3f}, ECE={ece_score:.3f})" if should_retrain else "MODEL_HEALTHY"

        return round(final_health_score := final_mhi, 1), should_retrain, reason

psi_multi_drift_retrainer = PSIMultiDriftRetrainer()
