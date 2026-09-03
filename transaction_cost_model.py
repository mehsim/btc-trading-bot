"""
transaction_cost_model.py — CANONICAL TCM (authoritative, migrate callers here)
----------------------------
Institutional TCM with Almgren-Chriss volatility-adjusted market impact:
  impact_bp = gamma * sigma * sqrt(Order_Size / Orderbook_Depth) * 10000

Uses live realized volatility (GARCH sigma) and orderbook depth rather than ADV,
making it more responsive to market microstructure than the simplified version in
trade_calculators.TransactionCostModel (which should be retired).

ASSUMPTION: gamma=0.42 is an empirical starting point, not fitted to execution
telemetry. Calibrate against compare_modeled_vs_realized_slippage() output.
GAMMA_SET_DATE = "2026-08-06"  # date assumption was recorded, not calibration date

Currently unused — all live callers import from trade_calculators. Migrate them here.
"""

import numpy as np
from typing import Dict, Any, Optional
import config

class TransactionCostModel:
    def __init__(self, default_taker_fee_bp: Optional[float] = None, default_maker_fee_bp: Optional[float] = None,
                 gamma: float = 0.42, max_acceptable_cost_bp: float = 25.0):
        # Read institutional fee schedule directly from config (Finding #136)
        self.taker_fee_bp = default_taker_fee_bp if default_taker_fee_bp is not None else float(getattr(config, "TAKER_FEE_PCT", 0.00055)) * 10000.0
        self.maker_fee_bp = default_maker_fee_bp if default_maker_fee_bp is not None else float(getattr(config, "MAKER_FEE_PCT", 0.00020)) * 10000.0
        self.gamma = gamma
        self.max_acceptable_cost_bp = max_acceptable_cost_bp

    def estimate_transaction_cost(
        self,
        symbol: str = "BTCUSDT",
        order_size_usd: float = 1000.0,
        volume_24h_usd: float = 50_000_000.0,
        bid_ask_spread_bp: float = 1.5,
        garch_sigma: float = 0.015,  # Almgren-Chriss: live realized vol from garch_monitor
        is_maker: bool = False,
        round_trip: bool = False,
        max_acceptable_cost_bp: Optional[float] = None,
        orderbook_depth_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Almgren-Chriss extended market impact against instantaneous orderbook depth:
        Slippage_bps = Half_Spread + gamma * sigma * sqrt(Order_Size / Orderbook_Depth) * 100
        """
        if round_trip:
            # Entry leg (maker or taker) + Exit leg (typically taker for stop-loss/market exit)
            fee_bp = (self.maker_fee_bp if is_maker else self.taker_fee_bp) + self.taker_fee_bp
            spread_cost_bp = float(bid_ask_spread_bp) # Full spread across round trip
        else:
            fee_bp = self.maker_fee_bp if is_maker else self.taker_fee_bp
            spread_cost_bp = float(bid_ask_spread_bp) / 2.0

        depth_usd = float(orderbook_depth_usd) if orderbook_depth_usd is not None and orderbook_depth_usd > 0 else max(10_000.0, float(volume_24h_usd) / 200.0)
        liquidity_ratio = max(1e-8, float(order_size_usd) / depth_usd)
        # Almgren-Chriss: volatility-adjusted impact relative to orderbook depth (fraction → bps requires ×10000)
        sigma = max(0.001, float(garch_sigma))
        market_impact_bp = self.gamma * sigma * np.sqrt(liquidity_ratio) * 10000.0
        if round_trip:
            market_impact_bp *= 1.5  # 2 legs with reduced impact on exit

        total_cost_bp = round(fee_bp + spread_cost_bp + market_impact_bp, 2)
        total_cost_usd = round((total_cost_bp / 10000.0) * float(order_size_usd), 4)
        cutoff = max_acceptable_cost_bp if max_acceptable_cost_bp is not None else self.max_acceptable_cost_bp
        is_acceptable = total_cost_bp <= cutoff

        return {
            "symbol": symbol,
            "order_size_usd": order_size_usd,
            "garch_sigma_input": round(sigma, 4),
            "fee_bp": fee_bp,
            "half_spread_bp": round(spread_cost_bp, 2),
            "market_impact_bp": round(market_impact_bp, 4),
            "total_cost_bp": total_cost_bp,
            "total_cost_usd": total_cost_usd,
            "is_acceptable": is_acceptable,
            "round_trip": round_trip,
            "model": "Almgren-Chriss"
        }

    def calculate_slippage_ratio(
        self,
        trade_notional: float,
        market_depth_usd: float,
        max_slippage_pct: Optional[float] = None,
        epsilon: float = 1e-6
    ) -> float:
        """
        Finding #156: Protected Division in Slippage Ratio Calculation.
        Computes slippage_ratio = trade_notional / market_depth_usd safely.
        When market_depth_usd is 0.0 or negative, defaults to configured max slippage
        or clamps denominator to epsilon.
        """
        configured_max = getattr(config, "MAX_SLIPPAGE_PCT", 0.005)
        max_slip = float(max_slippage_pct) if max_slippage_pct is not None else float(configured_max)
        if market_depth_usd is None or float(market_depth_usd) <= 0.0:
            return max_slip

        denom = max(epsilon, float(market_depth_usd))
        slippage_ratio = float(trade_notional) / denom
        return min(slippage_ratio, max_slip)


def calculate_slippage_ratio(
    trade_notional: float,
    market_depth_usd: float,
    max_slippage_pct: Optional[float] = None,
    epsilon: float = 1e-6
) -> float:
    """Module-level helper for protected slippage ratio computation."""
    return transaction_cost_model.calculate_slippage_ratio(
        trade_notional=trade_notional,
        market_depth_usd=market_depth_usd,
        max_slippage_pct=max_slippage_pct,
        epsilon=epsilon
    )


transaction_cost_model = TransactionCostModel(gamma=0.42)

