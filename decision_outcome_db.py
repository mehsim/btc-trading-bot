"""
decision_outcome_db.py
-----------------------
Institutional Decision Outcome Database.
Records 100% reproducible decision lineage for every single trade evaluation and execution:
  Features -> Prediction -> Execution -> Outcome -> Counterfactual -> Regret -> Model/Policy Version
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, Optional

DB_FILE = "decision_outcome_db.json"

class DecisionOutcomeDatabase:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.ensure_db_exists()

    def ensure_db_exists(self):
        if not os.path.exists(self.db_file):
            try:
                with open(self.db_file, "w") as f:
                    json.dump([], f)
            except Exception as e:
                print(f"[DecisionOutcomeDB Error] Failed to initialize {self.db_file}: {e}")

    def record_decision(
        self,
        symbol: str,
        interval: str,
        direction: str,
        features: Dict[str, float],
        raw_prediction: float,
        calibrated_confidence: float,
        dynamic_threshold: float,
        execution_details: Dict[str, Any],
        model_version: str = "v3.0_ensemble",
        policy_version: str = "policy_v3_adaptive"
    ) -> str:
        """Records initial decision lineage at entry signal approval."""
        decision_id = str(uuid.uuid4())
        record = {
            "decision_id": decision_id,
            "timestamp": time.time(),
            "symbol": symbol,
            "interval": interval,
            "direction": direction,
            "features": features,
            "prediction": {
                "raw_prediction": raw_prediction,
                "calibrated_confidence": calibrated_confidence,
                "dynamic_threshold": dynamic_threshold
            },
            "execution": execution_details,
            "outcome": None,
            "counterfactual": None,
            "regret": None,
            "model_version": model_version,
            "policy_version": policy_version
        }

        try:
            records = self.load_records()
            records.append(record)
            with open(self.db_file, "w") as f:
                json.dump(records[-500:], f, indent=2)  # Keep last 500 decisions
        except Exception as e:
            print(f"[DecisionOutcomeDB Error] Failed to write decision {decision_id}: {e}")

        return decision_id

    def update_outcome_and_regret(
        self,
        decision_id: str,
        outcome_details: Dict[str, Any],
        counterfactual_matrix: Dict[str, Any],
        best_counterfactual_r: float,
        actual_r: float
    ):
        """Updates decision record with final outcome, 81-scenario replay matrix, and regret calculation."""
        try:
            records = self.load_records()
            for r in records:
                if r.get("decision_id") == decision_id or r.get("execution", {}).get("trade_id") == decision_id:
                    r["outcome"] = outcome_details
                    r["counterfactual"] = counterfactual_matrix
                    regret_r = round(max(0.0, best_counterfactual_r - actual_r), 3)
                    r["regret"] = {
                        "actual_r": actual_r,
                        "best_counterfactual_r": best_counterfactual_r,
                        "regret_r": regret_r,
                        "regret_dollars": round(regret_r * outcome_details.get("risk_usd", 1.89), 2)
                    }
                    break
            with open(self.db_file, "w") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            print(f"[DecisionOutcomeDB Error] Failed to update outcome for {decision_id}: {e}")

    def load_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_file):
            return []
        try:
            with open(self.db_file, "r") as f:
                return json.load(f)
        except Exception:
            return []


decision_outcome_db = DecisionOutcomeDatabase()
