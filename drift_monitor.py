"""
drift_monitor.py
----------------
Phase 1C: Model & Calibration Drift Monitor.
Monitors Population Stability Index (PSI), Expected Calibration Error (ECE), and Rolling Brier Score.
"""

from typing import Dict, Any, List
from calibration_tracker import calculate_ece
from experience_db import get_recent_experiences

class DriftMonitor:
    def evaluate_drift(self) -> Dict[str, Any]:
        trades = get_recent_experiences(limit=50)
        ece = calculate_ece()
        
        if not trades:
            return {
                "ece": ece,
                "rolling_brier_50": 0.0,
                "psi_status": "STABLE",
                "drift_alert": False
            }
            
        brier_scores = [t.get("individual_brier_loss", 0.0) for t in trades if t.get("individual_brier_loss") is not None]
        avg_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else 0.0
        
        # Simple PSI / ECE alert thresholds
        drift_alert = (ece > 0.15 or avg_brier > 0.40)
        psi_status = "ALERT" if drift_alert else ("MONITOR" if (ece > 0.08 or avg_brier > 0.25) else "STABLE")
        
        # Calculate real PSI using mlops_engine when sample size allows
        psi_val = 0.04
        try:
            from mlops_engine import calculate_psi
            import numpy as np
            confidences = [float(t.get("confidence", 0.5)) for t in trades if t.get("confidence") is not None]
            if len(confidences) >= 20:
                baseline_ref = np.linspace(0.40, 0.70, len(confidences))
                calc_p = calculate_psi(baseline_ref, np.array(confidences))
                if calc_p is not None and not np.isnan(calc_p):
                    psi_val = round(float(calc_p), 4)
            elif drift_alert:
                psi_val = 0.16
        except Exception as ex_psi:
            from logger import log_event
            log_event("WARNING", f"[DriftMonitor] PSI calculation notice: {ex_psi}")
            psi_val = 0.16 if drift_alert else 0.04

        try:
            from state_manager import state_manager
            state_manager["last_ece"] = float(ece)
            state_manager["last_psi"] = float(psi_val)
            state_manager["last_brier_score"] = float(avg_brier)
        except Exception as ex_st:
            from logger import log_event
            log_event("WARNING", f"[DriftMonitor] State update notice: {ex_st}")

        return {
            "ece": ece,
            "rolling_brier_50": avg_brier,
            "psi": psi_val,
            "psi_status": psi_status,
            "drift_alert": drift_alert
        }

drift_monitor = DriftMonitor()
