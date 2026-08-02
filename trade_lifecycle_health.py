"""
Trade Lifecycle Health Scoring Engine.
Computes a continuous Trade Health Score (100% -> 0%) evaluating momentum decay, ADX decay, regime transition, orderbook deterioration, and OI divergence to trigger early exits.
"""

from typing import Dict, Any, Tuple

class TradeLifecycleHealthEngine:
    def evaluate_trade_health(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        initial_adx: float,
        current_adx: float,
        regime_transition_prob: float = 0.05,
        orderbook_imbalance: float = 0.50,
        oi_delta_pct: float = 0.0,
        atr_norm: float = 0.01
    ) -> Tuple[float, bool, str]:
        """
        Returns (health_score_pct, should_exit_early, reason).
        Health score scale: 100.0 (Perfect) to 0.0 (Degraded/Exit).
        Adverse move thresholds scale dynamically with symbol atr_norm.
        """
        health = 100.0

        # 1. Price Momentum / Adverse Move Check (Dynamic ATR scaling)
        dir_upper = direction.upper()
        if dir_upper in ("BUY", "LONG", "BULLISH"):
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        dynamic_adverse_major = -1.0 * max(0.005, atr_norm * 1.0)
        dynamic_adverse_minor = -1.0 * max(0.002, atr_norm * 0.5)

        if pnl_pct < dynamic_adverse_major:
            health -= 25.0
        elif pnl_pct < dynamic_adverse_minor:
            health -= 15.0

        # 2. ADX Trend Decay Check
        if initial_adx > 25.0 and current_adx < 18.0:
            health -= 20.0
        elif current_adx < initial_adx - 10.0:
            health -= 10.0

        # 3. Regime Transition Risk Check
        if regime_transition_prob > 0.25:
            health -= 25.0
        elif regime_transition_prob > 0.15:
            health -= 10.0

        # 4. Orderbook Deterioration Check
        if dir_upper in ("BUY", "LONG", "BULLISH") and orderbook_imbalance < 0.35:
            health -= 15.0
        elif dir_upper in ("SELL", "SHORT", "BEARISH") and orderbook_imbalance > 0.65:
            health -= 15.0

        # 5. Open Interest Divergence Check
        if oi_delta_pct < -0.03:
            health -= 10.0

        final_health = max(0.0, min(100.0, health))
        should_exit = final_health < 45.0
        reason = f"Trade Health Degraded ({final_health:.1f}%)" if should_exit else "HEALTHY"

        return round(final_health, 1), should_exit, reason

trade_lifecycle_health_engine = TradeLifecycleHealthEngine()
