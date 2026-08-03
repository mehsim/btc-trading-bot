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

    def calculate_adaptive_slippage_limit(
        self,
        atr_norm: float = 0.0040,
        spread_pct: float = 0.0003,
        ob_depth_usd: float = 100000.0
    ) -> float:
        """
        Adaptive Slippage Limit: f(ATR, Spread, Depth)
        Max Allowed Slippage (bps) scales dynamically with market microstructure.
        """
        base_bps = (spread_pct * 10000.0) * 1.5
        atr_mult = 1.0 + max(0.0, (atr_norm - 0.0030) * 100.0)
        depth_mult = max(0.8, min(2.0, 100000.0 / max(1.0, ob_depth_usd)))
        return float(max(8.0, min(35.0, base_bps * atr_mult * depth_mult)))

    def evaluate_canary_rollout_stage(
        self,
        current_stage: str,
        shadow_trades_count: int,
        is_statistically_approved: bool,
        mean_slippage_bps: float = 5.0,
        candidate_var_99: float = 0.05,
        champion_var_99: float = 0.05,
        candidate_cvar_99: float = 0.07,
        champion_cvar_99: float = 0.07,
        regimes_encountered: Optional[List[str]] = None,
        atr_norm: float = 0.0040,
        spread_pct: float = 0.0003,
        ob_depth_usd: float = 100000.0
    ) -> Tuple[str, float, List[str]]:
        """
        Progressive Canary Rollout Pipeline:
        SHADOW (0%) -> CANARY_5 (5%) -> CANARY_20 (20%) -> CANARY_50 (50%) -> CHAMPION (100%)
        Requires multi-regime exposure (TRENDING & RANGING) + Execution Quality + Tail Risk (VaR & CVaR).
        """
        reasons = []
        
        # 1. Adaptive Execution Quality Gate: f(ATR, Spread, Depth)
        adaptive_slippage_limit = self.calculate_adaptive_slippage_limit(atr_norm, spread_pct, ob_depth_usd)
        if mean_slippage_bps > adaptive_slippage_limit:
            reasons.append(f"Execution Quality Gate Failed: Slippage ({mean_slippage_bps:.1f}bps > adaptive limit {adaptive_slippage_limit:.1f}bps)")
            return "SHADOW", 0.00, reasons

        # 2. Tail Risk Budget Gate: Evaluate VaR (99%) and Expected Shortfall CVaR (99%)
        dynamic_var_tolerance = 1.05 + max(0.0, 0.02 * (1.0 - shadow_trades_count / 500.0))
        dynamic_var_limit = champion_var_99 * dynamic_var_tolerance
        dynamic_cvar_limit = champion_cvar_99 * dynamic_var_tolerance

        if candidate_var_99 > dynamic_var_limit:
            reasons.append(f"VaR Risk Gate Failed: Candidate VaR ({candidate_var_99*100:.2f}%) exceeds limit ({dynamic_var_limit*100:.2f}%)")
            return "SHADOW", 0.00, reasons

        if candidate_cvar_99 > dynamic_cvar_limit:
            reasons.append(f"CVaR Tail Risk Gate Failed: Candidate CVaR ({candidate_cvar_99*100:.2f}%) exceeds limit ({dynamic_cvar_limit*100:.2f}%)")
            return "SHADOW", 0.00, reasons

        # 3. Multi-Regime Exposure Criterion
        unique_regimes = set(regimes_encountered or ["TRENDING", "RANGING"])
        if len(unique_regimes) < 2 and shadow_trades_count < 300:
            reasons.append(f"Multi-Regime Exposure Gate Pending: Experienced regimes {list(unique_regimes)} (Requires both TRENDING & RANGING)")
            return "SHADOW", 0.00, reasons

        if not is_statistically_approved:
            reasons.append("Statistical promotion gates not yet satisfied")
            return "SHADOW", 0.00, reasons

        # Progressive Allocation Scale (Dynamic trade count thresholds scaling with volatility)
        vol_scaler = max(0.5, min(2.0, 0.0040 / max(0.0010, atr_norm)))
        thresh_champ = int(500 * vol_scaler)
        thresh_50 = int(300 * vol_scaler)
        thresh_20 = int(200 * vol_scaler)
        thresh_5 = int(100 * vol_scaler)

        if shadow_trades_count >= thresh_champ and current_stage == "CANARY_50":
            reasons.append("Promoted to full CHAMPION production status (100% allocation)")
            return "CHAMPION", 1.00, reasons
        elif shadow_trades_count >= thresh_50 and current_stage in ("CANARY_20", "CANARY_50"):
            reasons.append("Advanced to CANARY_50 stage (50% allocation)")
            return "CANARY_50", 0.50, reasons
        elif shadow_trades_count >= thresh_20 and current_stage in ("CANARY_5", "CANARY_20"):
            reasons.append("Advanced to CANARY_20 stage (20% allocation)")
            return "CANARY_20", 0.20, reasons
        elif shadow_trades_count >= thresh_5:
            reasons.append("Advanced to CANARY_5 stage (5% allocation)")
            return "CANARY_5", 0.05, reasons
        else:
            reasons.append("Retained in SHADOW evaluation stage (0% allocation)")
            return "SHADOW", 0.00, reasons

    def evaluate_promotion_readiness(
        self,
        candidate_model_id: str,
        champion_trade_history: List[Dict[str, Any]],
        shadow_trade_history: List[Dict[str, Any]],
        current_stage: str = "SHADOW",
        mean_slippage_bps: float = 4.0,
        candidate_var_99: float = 0.04,
        champion_var_99: float = 0.04
    ) -> Dict[str, Any]:
        """
        Evaluates 5 statistical promotion gates + Execution Quality + VaR Gate + Progressive Canary Stage.
        """
        if len(shadow_trade_history) < 10:
            return {
                "candidate_model_id": candidate_model_id,
                "promotion_approved": False,
                "canary_stage": "SHADOW",
                "allocation_pct": 0.0,
                "reason": f"Insufficient shadow trade history ({len(shadow_trade_history)} < 100 required)"
            }

        c_pnl = [t.get("realized_pnl", 0.0) for t in champion_trade_history]
        s_pnl = [t.get("realized_pnl", 0.0) for t in shadow_trade_history]

        c_wins = sum(1 for p in c_pnl if p > 0)
        s_wins = sum(1 for p in s_pnl if p > 0)

        c_pf = sum(p for p in c_pnl if p > 0) / max(1e-4, abs(sum(p for p in c_pnl if p < 0)))
        s_pf = sum(p for p in s_pnl if p > 0) / max(1e-4, abs(sum(p for p in s_pnl if p < 0)))

        eval_res = champion_challenger_framework.evaluate_bayesian_dual_governance_gate(
            shadow_trades_count=len(shadow_trade_history),
            champion_wins=c_wins,
            challenger_wins=s_wins
        )

        is_approved = bool(eval_res.get("approved_for_promotion", False))

        next_stage, alloc_pct, canary_reasons = self.evaluate_canary_rollout_stage(
            current_stage=current_stage,
            shadow_trades_count=len(shadow_trade_history),
            is_statistically_approved=is_approved,
            mean_slippage_bps=mean_slippage_bps,
            candidate_var_99=candidate_var_99,
            champion_var_99=champion_var_99
        )

        return {
            "candidate_model_id": candidate_model_id,
            "promotion_approved": is_approved and next_stage != "SHADOW",
            "canary_stage": next_stage,
            "allocation_pct": alloc_pct,
            "canary_reasons": canary_reasons,
            "gate_metrics": eval_res,
            "shadow_sample_size": len(shadow_trade_history),
            "champion_pf": round(c_pf, 2),
            "shadow_pf": round(s_pf, 2)
        }

shadow_trading_engine = ShadowTradingEngine()
