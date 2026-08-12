"""
decision_outcome_replay.py
--------------------------
Phase 1B: Decision Outcome Replay Engine.
Compares model beliefs at entry vs market reality at exit to quantify Prediction Error,
Reason Error, and Feature Error.
"""

from typing import Dict, Any
from trade_calculators import safe_float

class DecisionOutcomeReplay:
    def replay_trade(self, record: Dict[str, Any]) -> Dict[str, Any]:
        conf = safe_float(record.get("confidence", 0.5), 0.5)
        pnl = safe_float(record.get("pnl_usd", 0.0))
        direction = record.get("signal_direction", "LONG")
        outcome = "WIN" if pnl >= 0 else "LOSS"
        
        # 1. Prediction Error: Difference between predicted probability and actual binary outcome (1.0 or 0.0)
        actual_binary = 1.0 if outcome == "WIN" else 0.0
        prediction_error = round(abs(conf - actual_binary), 4)
        
        # 2. Reason Error: Was HTF trend / LTF alignment prediction validated?
        ltf_conflict = record.get("ltf_conflict", 0)
        reason_error = "HTF/LTF Conflict" if (outcome == "LOSS" and ltf_conflict) else "None"
        
        # 3. Feature Error: Was volatility higher than expected?
        atr_pct = safe_float(record.get("atr_pct", 0.01), 0.01)
        feature_error = "Vol Spike (>2%)" if atr_pct > 0.02 else "Nominal"
        
        return {
            "predicted_confidence": conf,
            "actual_outcome": outcome,
            "prediction_error": prediction_error,
            "reason_error": reason_error,
            "feature_error": feature_error,
            "summary": f"Belief ({conf*100:.0f}% {direction}) vs Reality ({outcome}): PredErr={prediction_error:.2f}"
        }

decision_outcome_replay = DecisionOutcomeReplay()
