"""
Pre-Exchange Execution Validator.
Final invariant check prior to order submission to Bybit.
Ensures structural, sizing, and market safety invariants are maintained.
"""

from typing import Dict, Any, Tuple, Optional

class ExecutionValidator:
    def __init__(self, min_rr_ratio: float = 1.20, max_market_impact_pct: float = 0.02):
        self.min_rr_ratio = min_rr_ratio
        self.max_market_impact_pct = max_market_impact_pct

    def validate_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        position_size_usd: float,
        live_price: float,
        top_book_depth_usd: float = 50000.0,
        portfolio_heat: float = 0.0,
        max_portfolio_heat: float = 0.20
    ) -> Tuple[bool, str]:
        """
        Validates order invariants. Returns (is_valid, reason).
        """
        if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
            return False, f"Invalid prices: Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}"

        if position_size_usd <= 0:
            return False, f"Position size must be > 0 (got {position_size_usd})"

        # 1. Stop Loss Invariant Check
        dir_upper = direction.upper()
        if dir_upper in ("BUY", "LONG", "BULLISH"):
            if stop_loss_price >= entry_price:
                return False, f"Long Stop Loss ({stop_loss_price}) must be strictly below Entry price ({entry_price})"
            if take_profit_price <= entry_price:
                return False, f"Long Take Profit ({take_profit_price}) must be strictly above Entry price ({entry_price})"
        elif dir_upper in ("SELL", "SHORT", "BEARISH"):
            if stop_loss_price <= entry_price:
                return False, f"Short Stop Loss ({stop_loss_price}) must be strictly above Entry price ({entry_price})"
            if take_profit_price >= entry_price:
                return False, f"Short Take Profit ({take_profit_price}) must be strictly below Entry price ({entry_price})"
        else:
            return False, f"Unknown direction: {direction}"

        # 2. Minimum R:R Ratio Invariant
        stop_dist = abs(entry_price - stop_loss_price)
        target_dist = abs(take_profit_price - entry_price)
        rr_ratio = target_dist / stop_dist if stop_dist > 0 else 0.0
        if rr_ratio < self.min_rr_ratio - 1e-4:
            return False, f"R:R Ratio ({rr_ratio:.2f}) below required minimum threshold ({self.min_rr_ratio:.2f})"

        # 3. Market Impact Invariant Check
        if top_book_depth_usd > 0:
            estimated_impact = position_size_usd / top_book_depth_usd
            if estimated_impact > self.max_market_impact_pct:
                return False, f"Estimated Market Impact ({estimated_impact*100:.3f}%) exceeds safety limit ({self.max_market_impact_pct*100:.3f}%)"

        # 4. Immediate Trigger Invariant Check (Limit Order / Stop Order Safety)
        if live_price > 0:
            if dir_upper in ("BUY", "LONG", "BULLISH") and live_price <= stop_loss_price:
                return False, f"Live price ({live_price}) is already at or below Long Stop Loss ({stop_loss_price})"
            if dir_upper in ("SELL", "SHORT", "BEARISH") and live_price >= stop_loss_price:
                return False, f"Live price ({live_price}) is already at or above Short Stop Loss ({stop_loss_price})"

        # 5. Portfolio Heat Invariant Check
        if portfolio_heat >= max_portfolio_heat:
            return False, f"Current Portfolio Heat ({portfolio_heat*100:.1f}%) reaches max allowed budget ({max_portfolio_heat*100:.1f}%)"

        return True, "VALID"

execution_validator = ExecutionValidator()
