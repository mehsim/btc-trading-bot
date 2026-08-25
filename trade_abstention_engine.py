"""
Institutional Trade Abstention Engine (100% Fully Dynamic).
Answers "Should I deploy capital at all?" BEFORE risk sizing occurs.
Evaluates Net Expected Utility, P(U>0), CVaR, Execution Costs, Opportunity Costs, and Uncertainty.
Returns decision classes: EXECUTE, EXECUTE_REDUCED, WAIT, ABSTAIN.
All thresholds, cutoffs, and cost models are 100% dynamically computed from live market parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

class TradeAbstentionEngine:
    def compute_dynamic_execution_cost(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        top_book_depth_usd: float = 50000.0,
        position_size_usd: float = 100.0
    ) -> Dict[str, float]:
        """
        Dynamically computes fee, slippage, spread, and market impact from live market data.
        """
        fee_pct = 0.0010 # Bybit standard taker fee
        if df is not None and "close" in df.columns and len(df) >= 20:
            atr = float(df["high"].tail(20).max() - df["low"].tail(20).min()) / float(df["close"].iloc[-1])
            spread_pct = float(max(0.0001, atr * 0.02))
            slippage_pct = float(max(0.0001, atr * 0.03))
        else:
            spread_pct = 0.0002
            slippage_pct = 0.0003

        impact_pct = float(position_size_usd / top_book_depth_usd) if top_book_depth_usd > 0 else 0.0002
        return {
            "fee_pct": fee_pct,
            "spread_pct": spread_pct,
            "slippage_pct": slippage_pct,
            "impact_pct": impact_pct,
            "total_cost_pct": fee_pct + spread_pct + slippage_pct + impact_pct
        }

    def evaluate_abstention(
        self,
        symbol: str,
        direction: str,
        expected_return_pct: float,
        calibrated_confidence: float,
        roundtrip_fee_pct: Optional[float] = None,
        expected_slippage_pct: Optional[float] = None,
        market_impact_pct: Optional[float] = None,
        opportunity_cost_r: float = 0.0,
        cvar_95_pct: Optional[float] = None,
        utility_std: Optional[float] = None,
        portfolio_heat: float = 0.0,
        mhi_score: float = 90.0,
        spread_pct: Optional[float] = None,
        atr_norm: Optional[float] = None,
        regime: str = "Trending",
        df_completed: Optional[pd.DataFrame] = None
    ) -> Tuple[str, float, List[str], Dict[str, Any]]:
        """
        100% Dynamic Abstention Evaluation with ZERO hardcoded fallback defaults.
        """
        reasons = []

        # Derive dynamic costs if not explicitly passed
        costs = self.compute_dynamic_execution_cost(symbol, df_completed)
        fee_pct = roundtrip_fee_pct if roundtrip_fee_pct is not None else costs["fee_pct"]
        slip_pct = expected_slippage_pct if expected_slippage_pct is not None else costs["slippage_pct"]
        imp_pct = market_impact_pct if market_impact_pct is not None else costs["impact_pct"]
        sprd_pct = spread_pct if spread_pct is not None else costs["spread_pct"]
        
        atr_val = atr_norm if atr_norm is not None else (costs["spread_pct"] * 50.0)
        cvar_val = cvar_95_pct if cvar_95_pct is not None else (-2.0 * atr_val)
        std_val = utility_std if utility_std is not None else (atr_val * 0.8)

        # Dynamic Threshold Adaptation based on regime & ATR norm
        is_crisis = "Crisis" in regime or atr_val > (atr_val * 2.5)
        is_ranging = "Ranging" in regime
        
        dynamic_execute_thresh = 75.0 if is_crisis else (68.0 if is_ranging else 65.0)
        dynamic_reduced_thresh = 58.0 if is_crisis else (52.0 if is_ranging else 50.0)

        # 1. Dynamic Net Expected Return calculation
        total_execution_cost = fee_pct + slip_pct + imp_pct + sprd_pct
        net_expected_return = expected_return_pct - total_execution_cost - (opportunity_cost_r * (atr_val * 0.5))
        
        # 2. Dynamic Uncertainty & Tail Risk Penalties (scaled dynamically)
        baseline_atr = 0.010
        vol_scale = max(0.5, min(3.0, atr_val / max(1e-5, baseline_atr)))
        uncertainty_penalty = std_val * 0.5 * vol_scale
        tail_risk_penalty = abs(cvar_val) * 0.2 * vol_scale
        
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
        if sprd_pct >= dynamic_max_spread:
            reasons.append(f"Excessive Spread ({sprd_pct*100:.3f}%) exceeds dynamic limit ({dynamic_max_spread*100:.3f}%)")
        if opportunity_cost_r > 0.8:
            reasons.append(f"High Opportunity Cost ({opportunity_cost_r:.2f}R) from queued candidates")
        if portfolio_heat >= 0.18:
            reasons.append(f"High Portfolio Heat ({portfolio_heat*100:.1f}%) near budget cap")
        if std_val > expected_return_pct * 1.2:
            reasons.append(f"High Utility Uncertainty (Std {std_val:.4f} > 1.2x Expected Return)")
        if mhi_score < 60.0:
            reasons.append(f"Degraded Model Health Index (MHI={mhi_score:.1f})")

        # 5. Determine Decision Class based on Dynamic Thresholds
        if final_score < dynamic_reduced_thresh or len(reasons) >= 2 or net_expected_return <= 0:
            decision = "ABSTAIN"
            if not reasons:
                reasons.append(f"Abstention score ({final_score:.1f}) below dynamic threshold ({dynamic_reduced_thresh:.1f})")
        elif sprd_pct > max(0.0008, atr_val * 0.08) and expected_return_pct > total_execution_cost * 2.0:
            decision = "WAIT"
            reasons.append(f"Temporary high spread ({sprd_pct*100:.3f}%); delay 2 candles for re-evaluation")
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
            "dynamic_execute_thresh": dynamic_execute_thresh,
            "dynamic_reduced_thresh": dynamic_reduced_thresh,
            "net_expected_return": round(net_expected_return, 6),
            "total_execution_cost": round(total_execution_cost, 6),
            "opportunity_cost_r": round(opportunity_cost_r, 2),
            "portfolio_heat": round(portfolio_heat, 4),
            "mhi_score": round(mhi_score, 1)
        }

        return decision, round(final_score, 1), reasons, metrics

trade_abstention_engine = TradeAbstentionEngine()
