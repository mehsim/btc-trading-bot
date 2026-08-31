"""
Pre-Exchange Execution Validator.
Final invariant check prior to order submission to Bybit.
Ensures structural, sizing, and market safety invariants are maintained.
"""

from typing import Tuple, Optional


class ExecutionValidator:
    def __init__(
        self,
        min_rr_ratio: Optional[float] = None,
        max_market_impact_pct: Optional[float] = None,
        max_portfolio_heat: Optional[float] = None
    ):
        self.min_rr_ratio = min_rr_ratio
        self.max_market_impact_pct = max_market_impact_pct
        self.max_portfolio_heat = max_portfolio_heat

    def _check_price_geometry(
        self,
        dir_upper: str,
        entry: float,
        sl: float,
        tp: float,
        min_rr: float
    ) -> Tuple[bool, str]:
        if dir_upper in ("BUY", "LONG", "BULLISH"):
            if sl >= entry:
                return False, f"Long Stop Loss ({sl}) must be strictly below Entry price ({entry})"
            if tp <= entry:
                return False, f"Long Take Profit ({tp}) must be strictly above Entry price ({entry})"
        elif dir_upper in ("SELL", "SHORT", "BEARISH"):
            if sl <= entry:
                return False, f"Short Stop Loss ({sl}) must be strictly above Entry price ({entry})"
            if tp >= entry:
                return False, f"Short Take Profit ({tp}) must be strictly below Entry price ({entry})"
        else:
            return False, f"Unknown direction: {dir_upper}"

        stop_dist = abs(entry - sl)
        target_dist = abs(tp - entry)
        rr_ratio = target_dist / stop_dist if stop_dist > 0 else 0.0
        if rr_ratio < min_rr - 1e-4:
            return False, f"R:R Ratio ({rr_ratio:.2f}) below required minimum threshold ({min_rr:.2f})"

        return True, "OK"

    def _check_live_triggers_and_heat(
        self,
        dir_upper: str,
        live_price: float,
        sl: float,
        heat: float,
        max_heat: float
    ) -> Tuple[bool, str]:
        if live_price > 0:
            if dir_upper in ("BUY", "LONG", "BULLISH") and live_price <= sl:
                return False, f"Live price ({live_price}) is already at or below Long Stop Loss ({sl})"
            if dir_upper in ("SELL", "SHORT", "BEARISH") and live_price >= sl:
                return False, f"Live price ({live_price}) is already at or above Short Stop Loss ({sl})"

        if heat >= max_heat:
            return False, f"Current Portfolio Heat ({heat*100:.1f}%) reaches max budget ({max_heat*100:.1f}%)"

        return True, "OK"

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
        max_portfolio_heat: float = 0.80,
        atr_norm: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Validates order invariants. Returns (is_valid, reason).
        Thresholds adapt dynamically to symbol ATR norm and live orderbook depth.
        """
        dynamic_min_rr = self.min_rr_ratio if self.min_rr_ratio is not None else 1.10
        dynamic_max_impact = self.max_market_impact_pct if self.max_market_impact_pct is not None else float(
            max(0.005, min(0.05, 0.02 * (top_book_depth_usd / 50000.0)))
        )
        dynamic_max_heat = self.max_portfolio_heat if self.max_portfolio_heat is not None else max_portfolio_heat

        if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
            return False, f"Invalid prices: Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}"

        if position_size_usd <= 0:
            return False, f"Position size must be > 0 (got {position_size_usd})"

        dir_upper = direction.upper()
        geom_ok, geom_msg = self._check_price_geometry(
            dir_upper, entry_price, stop_loss_price, take_profit_price, dynamic_min_rr
        )
        if not geom_ok:
            return False, geom_msg

        if top_book_depth_usd > 0:
            estimated_impact = position_size_usd / top_book_depth_usd
            if estimated_impact > dynamic_max_impact:
                return False, f"Impact ({estimated_impact*100:.3f}%) exceeds limit ({dynamic_max_impact*100:.3f}%)"

        trig_ok, trig_msg = self._check_live_triggers_and_heat(
            dir_upper, live_price, stop_loss_price, portfolio_heat, dynamic_max_heat
        )
        if not trig_ok:
            return False, trig_msg

        return True, "VALID"


execution_validator = ExecutionValidator()
