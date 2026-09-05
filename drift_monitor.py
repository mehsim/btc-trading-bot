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
    def _get_training_baseline_confidences(self, n_samples: int = 50) -> Any:
        import os
        import json
        import numpy as np

        baseline_file = "training_baseline_distribution.json"
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, "r") as f:
                    data = json.load(f)
                samples = data.get("baseline_samples", [])
                if len(samples) >= 20:
                    return np.array(samples, dtype=float)
            except Exception as ex:
                from logger import log_event
                log_event("WARNING", f"[DriftMonitor] Error loading training baseline: {ex}")

        # Finding R67 & #53: Fallback to empirical trade experience confidences from database with deterministic ORDER BY
        try:
            import sqlite3
            import database
            with sqlite3.connect(database.get_db_path()) as conn:
                c = conn.cursor()
                c.execute("SELECT confidence FROM completed_trades WHERE confidence > 0.1 ORDER BY timestamp DESC LIMIT 200")
                rows = [float(r[0]) for r in c.fetchall()]
                if len(rows) >= 20:
                    return np.array(rows, dtype=float)
        except Exception:
            pass

        # Fail-closed: Return None rather than a fabricated distribution
        return None

    def evaluate_drift(self) -> Dict[str, Any]:
        trades = get_recent_experiences(limit=50)
        ece = calculate_ece()
        
        if not trades:
            try:
                from state_manager import state_manager
                state_manager["last_ece"] = float(ece)
                state_manager["last_psi"] = 0.0
                state_manager["last_brier_score"] = 0.10
            except Exception as ex:
                from logger import log_event
                log_event("WARNING", f"Could not record baseline metrics to state_manager: {ex}")
            return {
                "ece": ece,
                "rolling_brier_50": 0.10,
                "psi": 0.0,
                "psi_status": "STABLE",
                "drift_alert": False
            }
            
        brier_scores = [t.get("individual_brier_loss", 0.0) for t in trades if t.get("individual_brier_loss") is not None]
        avg_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else 0.10
        
        # Simple PSI / ECE alert thresholds
        drift_alert = (ece > 0.15 or avg_brier > 0.40)
        psi_status = "ALERT" if drift_alert else ("MONITOR" if (ece > 0.08 or avg_brier > 0.25) else "STABLE")
        
        # Calculate real PSI using mlops_engine when sample size allows
        psi_val = 0.0
        try:
            from mlops_engine import calculate_psi
            import numpy as np
            confidences = [float(t.get("confidence", 0.5)) for t in trades if t.get("confidence") is not None]
            if len(confidences) >= 20:
                conf_arr = np.array(confidences, dtype=float)
                # Finding #154: Guard against zero variance / constant confidence values
                if np.std(conf_arr) < 1e-6:
                    from logger import log_event
                    log_event("WARNING", "[DriftMonitor] Zero variance / constant confidence values detected across recent trades. Defaulting PSI to 0.0.")
                    psi_val = 0.0
                else:
                    baseline_ref = self._get_training_baseline_confidences(len(conf_arr))
                    if baseline_ref is not None and len(baseline_ref) >= 20:
                        calc_p = calculate_psi(baseline_ref, conf_arr)
                        if calc_p is not None and not np.isnan(calc_p) and not np.isinf(calc_p):
                            psi_val = round(float(calc_p), 4)
                        else:
                            psi_val = 0.0
                    else:
                        psi_val = 0.0
            elif drift_alert:
                psi_val = 0.16
            else:
                psi_val = 0.0
        except Exception as ex_psi:
            from logger import log_event
            log_event("WARNING", f"[DriftMonitor] PSI calculation notice: {ex_psi}")
            psi_val = 0.16 if drift_alert else 0.0

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
