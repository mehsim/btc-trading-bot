import numpy as np
import threading
from typing import Dict, List, Tuple, Optional

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
        self.processed_trade_ids = set()
        self._restore_state()

    def _restore_state(self):
        try:
            import json
            from database import get_setting
            val_s_high = get_setting("cusum_s_high")
            if val_s_high is not None:
                self.S_high = float(val_s_high)
            val_ids = get_setting("cusum_processed_trade_ids")
            if val_ids is not None:
                parsed_ids = json.loads(val_ids)
                if isinstance(parsed_ids, list):
                    self.processed_trade_ids = set(parsed_ids)
        except Exception as e:
            print(f"[CUSUM Drift Detector Warning] Failed to restore CUSUM state from database: {e}")

    def _persist_state(self):
        try:
            import json
            from database import set_setting
            set_setting("cusum_s_high", str(self.S_high))
            # Ring-buffer: keep only the most recent 1000 trade IDs to prevent unbounded growth
            capped_ids = list(self.processed_trade_ids)[-1000:]
            set_setting("cusum_processed_trade_ids", json.dumps(capped_ids))
        except Exception as e:
            print(f"[CUSUM Drift Detector Warning] Failed to persist CUSUM state to database: {e}")


    def update(self, actual_outcome: int, predicted_confidence: float = 0.70, trade_id: str = None) -> Tuple[bool, float, float]:
        """
        Rule 24: CUSUM Drift Detection with Idempotency & Confidence Weighting:
        actual_outcome: 1 if profitable, 0 if loss
        trade_id: Unique identifier to ensure idempotency across evaluation loops
        """
        with self.lock:
            if trade_id and trade_id in self.processed_trade_ids:
                # Already ingested — return current state without re-accumulating
                is_drift = self.S_high >= self.threshold_H
                recent_error_rate = float(np.mean(self.error_stream)) if self.error_stream else self.target_error_mu
                return is_drift, float(self.S_high), recent_error_rate

            if trade_id:
                self.processed_trade_ids.add(trade_id)

            # Error = 1.0 if loss, 0.0 if win
            error_val = 1.0 - float(actual_outcome)
            self.error_stream.append(error_val)
            recent_error_rate = float(np.mean(self.error_stream)) if self.error_stream else self.target_error_mu

            # Target error expected: dynamic based on predicted model confidence if available
            target_mu = (1.0 - float(predicted_confidence)) if (predicted_confidence and 0.0 < predicted_confidence < 1.0) else self.target_error_mu
            
            # Upper CUSUM accumulator for detecting degradation in accuracy against target baseline
            self.S_high = max(0.0, self.S_high + (error_val - target_mu - self.allowance_K))
            self._persist_state()
            
            is_drift = self.S_high >= self.threshold_H
            return is_drift, float(self.S_high), recent_error_rate

    def reset(self):
        with self.lock:
            self.S_high = 0.0
            self.S_low = 0.0
            self.processed_trade_ids.clear()
            self._persist_state()

def calculate_psi(baseline_data: np.ndarray, target_data: np.ndarray, num_bins: int = 10) -> Optional[float]:
    """
    Calculates Population Stability Index (PSI) between baseline training distribution
    and recent live distribution.
    Returns None when sample size is sparse (< 20 observations) to avoid false stability signaling.
    PSI < 0.10: Stable, 0.10 <= PSI < 0.25: Moderate Drift, PSI >= 0.25: Severe Drift
    """
    if baseline_data is None or target_data is None or len(baseline_data) < 20 or len(target_data) < 20:
        b_n = len(baseline_data) if baseline_data is not None else 0
        t_n = len(target_data) if target_data is not None else 0
        print(f"[PSI Telemetry Warning] Insufficient sample size for PSI calculation (baseline_n={b_n}, target_n={t_n}, min_required=20). Returning None (INSUFFICIENT_DATA).")
        return None
        
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

    def check_feature_drift(self, baseline_feature: np.ndarray, live_feature: np.ndarray) -> Tuple[bool, Optional[float], str]:
        psi_score = calculate_psi(baseline_feature, live_feature)
        if psi_score is None:
            return False, None, "INSUFFICIENT_DATA"
        if psi_score >= self.severe_psi:
            return True, psi_score, "SEVERE_DRIFT"
        elif psi_score >= self.warning_psi:
            return False, psi_score, "MODERATE_DRIFT"
        return False, psi_score, "STABLE"



def evaluate_drift_and_trigger_playbook(cusum_detector: CUSUMDriftDetector, psi_detector: PSIDriftDetector, recent_outcomes: list, live_features: np.ndarray = None, baseline_features: np.ndarray = None) -> dict:
    """
    Automated 5-Step Drift Playbook:
    1. De-risk: Reduce position cap to 50% on moderate drift (PSI >= 0.10)
    2. Pause: Pause new entries on severe drift or CUSUM threshold hit (PSI >= 0.25 or CUSUM >= 5.0)
    3. Retrain: Signal background model retraining script
    4. Shadow: Keep new challenger in shadow evaluation until outperforming champion
    5. Re-arm: Promote challenger and restore full risk limits
    """
    playbook_action = {
        "status": "STABLE",
        "de_risk": False,
        "pause_entries": False,
        "trigger_retrain": False,
        "cusum_score": 0.0,
        "psi_score": 0.0,
        "details": "All drift metrics within safe bounds"
    }

    if recent_outcomes:
        for out in recent_outcomes:
            outcome_val = 1 if out.get("success") == 1 or (out.get("pnl_usd") or 0.0) > 0 else 0
            t_id = out.get("trade_id") or out.get("id")
            is_drift, s_high, err_rate = cusum_detector.update(outcome_val, out.get("confidence", 0.70), trade_id=t_id)
            playbook_action["cusum_score"] = s_high
            if is_drift:
                playbook_action["status"] = "SEVERE_DRIFT_CUSUM"
                playbook_action["de_risk"] = True
                playbook_action["pause_entries"] = True
                playbook_action["trigger_retrain"] = True
                playbook_action["details"] = f"CUSUM score ({s_high:.2f}) exceeded threshold H ({cusum_detector.threshold_H})"
                return playbook_action

    if live_features is not None and baseline_features is not None:
        is_severe, psi_score, status_str = psi_detector.check_feature_drift(baseline_features, live_features)
        playbook_action["psi_score"] = psi_score
        if status_str == "SEVERE_DRIFT":
            playbook_action["status"] = "SEVERE_DRIFT_PSI"
            playbook_action["de_risk"] = True
            playbook_action["pause_entries"] = True
            playbook_action["trigger_retrain"] = True
            playbook_action["details"] = f"PSI score ({psi_score:.3f}) exceeded severe threshold ({psi_detector.severe_psi})"
        elif status_str == "MODERATE_DRIFT":
            playbook_action["status"] = "MODERATE_DRIFT_PSI"
            playbook_action["de_risk"] = True
            playbook_action["details"] = f"PSI score ({psi_score:.3f}) exceeded warning threshold ({psi_detector.warning_psi})"

    return playbook_action

cusum_drift_detector = CUSUMDriftDetector()
psi_drift_detector = PSIDriftDetector()
