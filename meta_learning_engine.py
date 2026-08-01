"""
meta_learning_engine.py
------------------------
Regret Analysis Engine & Automated Meta-Learning Policy Auto-Tuner.
Calculates Regret = Best Counterfactual R - Actual R.
Aggregates policy regret and auto-tunes policy thresholds via Shadow Mode promotion when regret breaches 0.35R threshold.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional

class MetaLearningEngine:
    def __init__(self, history_file: str = "decision_outcome_db.json"):
        self.history_file = history_file

    def compute_regret_attribution(self, window: int = 30) -> Dict[str, Any]:
        """
        Calculates average regret R-multiples left on the table across recent trades.
        Regret = Best Counterfactual R - Actual R
        """
        if not os.path.exists(self.history_file):
            return {"total_trades_analyzed": 0, "avg_regret_r": 0.0, "total_dollar_regret": 0.0, "policy_recommendations": []}

        try:
            with open(self.history_file, "r") as f:
                records = json.load(f)
            
            valid_records = [r for r in records if r.get("regret") is not None]
            recent = valid_records[-window:] if len(valid_records) >= window else valid_records

            if not recent:
                return {"total_trades_analyzed": 0, "avg_regret_r": 0.0, "total_dollar_regret": 0.0, "policy_recommendations": []}

            regret_r_list = [r["regret"]["regret_r"] for r in recent]
            dollar_regret_list = [r["regret"]["regret_dollars"] for r in recent]

            avg_regret_r = round(sum(regret_r_list) / max(1, len(regret_r_list)), 3)
            total_dollar_regret = round(sum(dollar_regret_list), 2)

            recommendations = []
            if avg_regret_r > 0.35:
                recommendations.append(f"HIGH REGRET ALERT: Average regret is +{avg_regret_r:.2f}R. Triggering automated policy auto-tuning in Shadow Mode.")
            else:
                recommendations.append(f"Optimal policy alignment: Average regret (+{avg_regret_r:.2f}R) is below 0.35R threshold.")

            return {
                "total_trades_analyzed": len(recent),
                "avg_regret_r": avg_regret_r,
                "total_dollar_regret": total_dollar_regret,
                "max_single_trade_regret_r": round(max(regret_r_list, default=0.0), 3),
                "policy_recommendations": recommendations
            }
        except Exception as e:
            return {"error": str(e), "avg_regret_r": 0.0, "total_dollar_regret": 0.0, "policy_recommendations": []}

    def auto_tune_policy_shadow_mode(self) -> Dict[str, Any]:
        """Auto-tunes policy parameters based on regret analysis and evaluates candidate in Shadow Mode."""
        attribution = self.compute_regret_attribution()
        avg_regret = attribution.get("avg_regret_r", 0.0)

        shadow_config = {
            "version": "policy_v3.1_autotuned",
            "timestamp": time.time(),
            "avg_regret_trigger_r": avg_regret,
            "tuned_sl_mult": 1.15 if avg_regret > 0.35 else 1.25,
            "tuned_base_threshold": 0.70 if avg_regret > 0.35 else 0.68,
            "shadow_mode_active": True
        }

        try:
            with open("shadow_autotuned_policy.json", "w") as f:
                json.dump(shadow_config, f, indent=2)
        except Exception as e:
            print(f"[MetaLearningEngine Error] Failed to write shadow autotuned policy: {e}")

        return shadow_config


meta_learning_engine = MetaLearningEngine()
