"""
counterfactual_engine.py
------------------------
Phase 1C: Simplified Counterfactual Engine.
Evaluates 5 core alternative scenarios for every closed trade using actual trade parameters:
1. Actual (Baseline)
2. ATR 1.5x (Wider Stop)
3. ATR 2.0x (Very Wide Stop)
4. RR 2.5 (Tighter Target)
5. No Trade (Avoided Loss / Opportunity Cost)
"""

from typing import Dict, Any, List
from trade_calculators import safe_float

class SimplifiedCounterfactualEngine:
    def evaluate_scenarios(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        actual_r = safe_float(record.get("realized_r", 0.0))
        pnl = safe_float(record.get("pnl_usd", 0.0))
        exit_reason = str(record.get("exit_reason", "MANUAL"))
        mae = safe_float(record.get("mae_pct", 0.0))
        atr_pct = safe_float(record.get("atr_pct", 0.01), 0.01)
        
        # 1. Actual
        scenarios = [{
            "scenario": "Actual",
            "realized_r": actual_r,
            "diff_vs_actual_r": 0.0,
            "note": "Executed strategy baseline"
        }]
        
        # 2. ATR 1.5x Stop (Wider Stop)
        # If exit was STOP loss and MAE was between 1.0 ATR and 1.5 ATR, wider stop would have survived
        if "STOP" in exit_reason.upper() and mae > 0 and (mae / atr_pct) <= 1.5:
            scenarios.append({
                "scenario": "ATR 1.5x",
                "realized_r": 0.5,  # Estimated recovery R
                "diff_vs_actual_r": round(0.5 - actual_r, 4),
                "note": "Wider stop would have survived local noise"
            })
        else:
            scenarios.append({
                "scenario": "ATR 1.5x",
                "realized_r": actual_r,
                "diff_vs_actual_r": 0.0,
                "note": "No outcome change under 1.5x ATR stop"
            })
            
        # 3. ATR 2.0x Stop (Very Wide Stop)
        if "STOP" in exit_reason.upper() and mae > 0 and (mae / atr_pct) <= 2.0:
            scenarios.append({
                "scenario": "ATR 2.0x",
                "realized_r": 0.8,
                "diff_vs_actual_r": round(0.8 - actual_r, 4),
                "note": "2.0x ATR stop avoided premature stop-out"
            })
        else:
            scenarios.append({
                "scenario": "ATR 2.0x",
                "realized_r": actual_r,
                "diff_vs_actual_r": 0.0,
                "note": "No outcome change under 2.0x ATR stop"
            })
            
        # 4. RR 2.5 (Tighter Target)
        scenarios.append({
            "scenario": "RR 2.5",
            "realized_r": 2.5 if actual_r > 1.0 else actual_r,
            "diff_vs_actual_r": round((2.5 - actual_r) if actual_r > 1.0 else 0.0, 4),
            "note": "Target adjusted to 2.5R"
        })
        
        # 5. No Trade (Avoided Loss)
        scenarios.append({
            "scenario": "No Trade",
            "realized_r": 0.0,
            "diff_vs_actual_r": round(0.0 - actual_r, 4),
            "note": "Opportunity cost / avoided loss zero baseline"
        })
        
        return scenarios

counterfactual_engine = SimplifiedCounterfactualEngine()
