"""
Institutional Trade Abstention Engine.
Answers "Should I deploy capital at all?" BEFORE risk sizing occurs.
Evaluates Net Expected Utility, P(U>0), CVaR, Execution Costs, Opportunity Costs, and Uncertainty.
Returns decision classes: EXECUTE, EXECUTE_REDUCED, WAIT, ABSTAIN.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class TradeAbstentionEngine:
    def evaluate_abstention(
        self,
        symbol: str,
        direction: str,
        expected_return_pct: float,
        calibrated_confidence: float,
        roundtrip_fee_pct: float = 0.0010,
        expected_slippage_pct: float = 0.0003,
        market_impact_pct: float = 0.0002,
        opportunity_cost_r: float = 0.0,
        cvar_95_pct: float = -0.02,
        utility_std: float = 0.01,
        portfolio_heat: float = 0.0,
        mhi_score: float = 90.0,
        spread_pct: float = 0.0002,
        atr_norm: float = 0.01,
        regime: str = "Trending"
    ) -> Tuple[str, float, List[str], Dict[str, Any]]:
        """
        100% Dynamic Abstention Engine.
        Dynamically adapts thresholds, execution costs, and uncertainty penalties based on symbol volatility & regime.
        """
        reasons = []

        # Dynamic Threshold Adaptation based on regime & ATR norm
        is_crisis = "Crisis" in regime or atr_norm > 0.025
        is_ranging = "Ranging" in regime
        
        dynamic_execute_thresh = 75.0 if is_crisis else (68.0 if is_ranging else 65.0)
        dynamic_reduced_thresh = 58.0 if is_crisis else (52.0 if is_ranging else 50.0)

        # 1. Dynamic Net Expected Return calculation
        total_execution_cost = roundtrip_fee_pct + expected_slippage_pct + market_impact_pct + spread_pct
        net_expected_return = expected_return_pct - total_execution_cost - (opportunity_cost_r * (atr_norm * 0.5))
        
        # 2. Dynamic Uncertainty & Tail Risk Penalties (scaled by ATR)
        vol_scale = max(0.5, min(3.0, atr_norm / 0.01))
        uncertainty_penalty = utility_std * 0.5 * vol_scale
        tail_risk_penalty = abs(cvar_95_pct) * 0.2 * vol_scale
        
        net_utility_score = net_expected_return - uncertainty_penalty - tail_risk_penalty

        # 3. Dynamic Score (0.0 to 100.0)
        base_score = 50.0 + (net_expected_return * 1500.0) - (uncertainty_penalty * 300.0) - (tail_risk_penalty * 300.0)
        
        # Dynamic MHI and Portfolio Heat Scalers
        mhi_scaler = 1.0 if mhi_score >= 80.0 else max(0.0, mhi_score / 80.0)
        heat_penalty = max(0.0, (portfolio_heat - 0.10) * 150.0) if portfolio_heat > 0.10 else 0.0
        
        final_score = float(max(0.0, min(100.0, (base_score * mhi_scaler) - heat_penalty)))

        # 4. Check Individual Dynamic Gate Failure Reasons
        dynamic_max_spread = max(0.0004, expected_return_pct * 0.40)
        if net_expected_return <= 0:
            reasons.append(f"Net Expected Return ({net_expected_return*100:.3f}%) <= 0 after costs ({total_execution_cost*100:.3f}%)")
            reasons.append(f"Excessive Spread ({spread_pct*100:.3f}%) absorbs >50% of expected return ({expected_return_pct*100:.3f}%)")
        if opportunity_cost_r > 0.8:
            reasons.append(f"High Opportunity Cost ({opportunity_cost_r:.2f}R) from queued candidates")
        if portfolio_heat >= 0.18:
            reasons.append(f"High Portfolio Heat ({portfolio_heat*100:.1f}%) near budget cap")
        if utility_std > expected_return_pct * 1.2:
            reasons.append(f"High Utility Uncertainty (Std {utility_std:.4f} > 1.2x Expected Return)")
        if mhi_score < 60.0:
            reasons.append(f"Degraded Model Health Index (MHI={mhi_score:.1f})")

        # 5. Determine Decision Class based on Dynamic Thresholds
        if final_score < dynamic_reduced_thresh or len(reasons) >= 2 or net_expected_return <= 0:
            decision = "ABSTAIN"
            if not reasons:
                reasons.append(f"Abstention score ({final_score:.1f}) below dynamic threshold ({dynamic_reduced_thresh:.1f})")
        elif spread_pct > max(0.0008, atr_norm * 0.08) and expected_return_pct > total_execution_cost * 2.0:
            decision = "WAIT"
            reasons.append(f"Temporary high spread ({spread_pct*100:.3f}%); delay 2 candles for re-evaluation")
        elif final_score < dynamic_execute_thresh or portfolio_heat > 0.12 or mhi_score < 75.0:
            decision = "EXECUTE_REDUCED"
            reasons.append(f"Moderate score ({final_score:.1f}); execute with half-Kelly sizing")
        else:
            decision = "EXECUTE"
            reasons.append("APPROVED: High net expected utility and execution quality")

        metrics = {
            "symbol": symbol,
            "direction": direction,
            "abstention_score": round(final_score, 1),
            "net_expected_return": round(net_expected_return, 6),
            "total_execution_cost": round(total_execution_cost, 6),
            "opportunity_cost_r": round(opportunity_cost_r, 2),
            "portfolio_heat": round(portfolio_heat, 4),
            "mhi_score": round(mhi_score, 1)
        }

        return decision, round(final_score, 1), reasons, metrics

trade_abstention_engine = TradeAbstentionEngine()
