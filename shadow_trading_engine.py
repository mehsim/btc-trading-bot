"""
Champion-Shadow Trading Engine & Policy Promotion Evaluator.
Runs candidate ML models in Shadow Mode (sizing_mult = 0.0, zero exchange orders) in parallel with production Champion models.
Evaluates 30-day / 100-trade shadow performance before promoting candidates to production.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from champion_challenger_framework import champion_challenger_framework

SHADOW_HISTORY_FILE = "shadow_trading_history.json"

class ShadowTradingEngine:
    def __init__(self, filename: str = SHADOW_HISTORY_FILE):
        self.filename = filename
        self.shadow_logs: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.shadow_logs[-1000:], f, indent=2)
        except Exception as e:
            print(f"[ShadowEngine Error] Failed to save shadow history: {e}")

    def evaluate_shadow_signal(
        self,
        candidate_model_id: str,
        symbol: str,
        direction: str,
        predicted_change_pct: float,
        calibrated_confidence: float,
        champion_model_id: str = "production_v7"
    ) -> Dict[str, Any]:
        """
        Evaluates a shadow candidate signal without placing real exchange orders.
        """
        shadow_record = {
            "timestamp": time.time(),
            "candidate_model_id": candidate_model_id,
            "champion_model_id": champion_model_id,
            "symbol": symbol,
            "direction": direction,
            "predicted_change_pct": round(predicted_change_pct, 6),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "execution_status": "SHADOW_SIMULATION",
            "real_orders_placed": False,
            "simulated_entry_price": 0.0,
            "simulated_outcome_pnl": 0.0
        }
        self.shadow_logs.append(shadow_record)
        self._save()
        return shadow_record

    def evaluate_promotion_readiness(
        self,
        candidate_model_id: str,
        champion_trade_history: List[Dict[str, Any]],
        shadow_trade_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates statistical promotion gates using champion_challenger_framework.
        """
        if len(shadow_trade_history) < 10:
            return {
                "candidate_model_id": candidate_model_id,
                "promotion_approved": False,
                "reason": f"Insufficient shadow trade history ({len(shadow_trade_history)} < 100 required)"
            }

        c_pnl = [t.get("realized_pnl", 0.0) for t in champion_trade_history]
        s_pnl = [t.get("realized_pnl", 0.0) for t in shadow_trade_history]

        c_wins = sum(1 for p in c_pnl if p > 0)
        s_wins = sum(1 for p in s_pnl if p > 0)

        c_wr = c_wins / max(1, len(c_pnl))
        s_wr = s_wins / max(1, len(s_pnl))

        c_pf = sum(p for p in c_pnl if p > 0) / max(1e-4, abs(sum(p for p in c_pnl if p < 0)))
        s_pf = sum(p for p in s_pnl if p > 0) / max(1e-4, abs(sum(p for p in s_pnl if p < 0)))

        eval_res = champion_challenger_framework.evaluate_bayesian_dual_governance_gate(
            shadow_trades_count=len(shadow_trade_history),
            champion_wins=c_wins,
            challenger_wins=s_wins
        )

        is_approved = bool(eval_res.get("approved_for_promotion", False))

        return {
            "candidate_model_id": candidate_model_id,
            "promotion_approved": is_approved,
            "gate_metrics": eval_res,
            "shadow_sample_size": len(shadow_trade_history),
            "champion_pf": round(c_pf, 2),
            "shadow_pf": round(s_pf, 2)
        }

shadow_trading_engine = ShadowTradingEngine()
