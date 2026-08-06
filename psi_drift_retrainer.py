"""
PSI Multi-Drift Engine & Retrainer.
Combines Population Stability Index (PSI), Calibration Drift, Feature Drift, and Performance Drift into a unified Model Health Index (MHI) to trigger automated retraining.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

class PSIMultiDriftRetrainer:
    def calculate_psi(self, baseline_scores: np.ndarray, current_scores: np.ndarray, num_bins: int = 10) -> Optional[float]:
        if baseline_scores is None or current_scores is None or len(baseline_scores) < 20 or len(current_scores) < 20:
            print(f"[PSI Multi-Drift Warning] Insufficient sample size for PSI: baseline_n={len(baseline_scores) if baseline_scores is not None else 0}, current_n={len(current_scores) if current_scores is not None else 0}")
            return None
        
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(baseline_scores, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        base_cnts, _ = np.histogram(baseline_scores, bins=bins)
        curr_cnts, _ = np.histogram(current_scores, bins=bins)

        base_pct = base_cnts / float(len(baseline_scores)) + 1e-4
        curr_pct = curr_cnts / float(len(current_scores)) + 1e-4

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(max(0.0, psi))

    def evaluate_model_health_index(
        self,
        psi_score: Optional[float],
        ece_score: float,
        recent_win_rate: float,
        baseline_win_rate: float = 0.50
    ) -> Tuple[float, bool, str]:
        """
        Returns (mhi_score, should_retrain, retrain_reason).
        C-1/H-1: Continuous proportional penalties with per-signal trip conditions derived from config.py.
        """
        import config
        policy = getattr(config, "MHI_POLICY", {})
        retrain_thresh = policy.get("retrain_threshold", 60.0)
        severe_psi = policy.get("severe_psi", 0.25)
        max_ece = policy.get("max_ece", getattr(config, "MODEL_GOVERNANCE", {}).get("max_ece", 0.08))
        wr_factor = policy.get("wr_drop_factor", 0.30)
        wr_min = policy.get("wr_drop_min", 0.10)
        wr_max = policy.get("wr_drop_max", 0.25)

        dynamic_wr_drop_high = float(max(wr_min, min(wr_max, baseline_win_rate * wr_factor)))
        win_rate_drop = max(0.0, baseline_win_rate - recent_win_rate)

        # Continuous proportional penalties
        psi_val = psi_score if psi_score is not None else 0.0
        psi_pen = policy.get("max_psi_penalty", 40.0) * min(1.0, max(0.0, psi_val) / severe_psi) if psi_score is not None else 0.0
        ece_pen = policy.get("max_ece_penalty", 30.0) * min(1.0, max(0.0, ece_score) / max_ece)
        wr_pen = policy.get("max_wr_penalty", 30.0) * min(1.0, win_rate_drop / max(1e-6, dynamic_wr_drop_high))

        mhi = max(0.0, min(100.0, 100.0 - psi_pen - ece_pen - wr_pen))

        # Per-signal trip conditions prevent single severe degradation from being outvoted
        trip_psi = bool(psi_score is not None and psi_score >= severe_psi)
        trip_ece = bool(ece_score > max_ece)
        trip_wr = bool(win_rate_drop >= dynamic_wr_drop_high)

        should_retrain = (mhi < retrain_thresh) or trip_psi or trip_ece or trip_wr

        reasons = []
        if mhi < retrain_thresh:
            reasons.append(f"MHI={mhi:.1f}<{retrain_thresh}")
        if trip_psi:
            reasons.append(f"Severe PSI={psi_val:.3f}>={severe_psi}")
        if trip_ece:
            reasons.append(f"ECE={ece_score:.3f}>{max_ece}")
        if trip_wr:
            reasons.append(f"WR_Drop={win_rate_drop:.3f}>={dynamic_wr_drop_high:.3f}")

        reason = f"Retrain Triggered: {', '.join(reasons)}" if should_retrain else "MODEL_HEALTHY"
        return round(mhi, 1), should_retrain, reason


psi_multi_drift_retrainer = PSIMultiDriftRetrainer()

