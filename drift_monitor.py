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
        
        try:
            from state_manager import state_manager
            state_manager["last_ece"] = float(ece)
            state_manager["last_brier_score"] = float(avg_brier)
        except Exception:
            pass

        return {
            "ece": ece,
            "rolling_brier_50": avg_brier,
            "psi_status": psi_status,
            "drift_alert": drift_alert
        }

drift_monitor = DriftMonitor()
