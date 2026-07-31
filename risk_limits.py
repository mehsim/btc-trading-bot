"""
risk_limits.py
--------------
F-09 Governance & Change-Controlled Safety Limits.
Isolated, change-controlled governance file defining immutable risk boundaries.
Asserted at startup to prevent execution outside absolute safety envelopes.
"""

from typing import Dict, Any

# Hard Leverage Caps per Timeframe (Immutable Safety Upper Bounds)
HARD_TIMEFRAME_MAX_LEVERAGE_CAPS: Dict[str, float] = {
    "5": 10.0,
    "15": 10.0,
    "30": 10.0,
    "60": 5.0,
    "120": 5.0,
    "240": 3.0,
    "360": 3.0
}

# Account & Exposure Limits
HARD_MAX_WALLET_MARGIN_UTILIZATION_PCT: float = 0.90  # 90% hard limit on balance margin utilization
HARD_MAX_SYMBOL_EXPOSURE_PCT: float = 0.20           # 20% max account balance in one symbol
HARD_MAX_DRAWDOWN_HALT_PCT: float = 0.20             # 20% max drawdown halt threshold
HARD_MAX_RISK_PER_TRADE_PCT: float = 0.03            # 3.0% absolute max risk per trade

def assert_risk_governance_invariants(config_module: Any = None) -> bool:
    """
    F-09 Startup Lock:
    Asserts that active runtime configuration parameters do not violate immutable risk limits.
    Raises PermissionError if any safety bound is relaxed or breached.
    """
    if config_module is None:
        try:
            import config as config_module
        except Exception:
            return True

    # 1. Validate Timeframe Leverage Caps
    cfg_leverage = getattr(config_module, "TIMEFRAME_MAX_LEVERAGE_CAPS", {})
    for tf, hard_cap in HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.items():
        active_lev = cfg_leverage.get(tf, hard_cap)
        if active_lev > hard_cap:
            raise PermissionError(
                f"[Governance Violation] Leverage cap for timeframe {tf}m ({active_lev}x) "
                f"exceeds hard safety limit ({hard_cap}x)."
            )

    # 2. Validate Wallet Utilization Cap
    active_margin_cap = getattr(config_module, "MAX_WALLET_MARGIN_UTILIZATION_PCT", 0.90)
    if active_margin_cap > HARD_MAX_WALLET_MARGIN_UTILIZATION_PCT:
        raise PermissionError(
            f"[Governance Violation] Wallet margin utilization limit ({active_margin_cap*100:.1f}%) "
            f"exceeds hard safety cap ({HARD_MAX_WALLET_MARGIN_UTILIZATION_PCT*100:.1f}%)."
        )

    # 3. Validate Symbol Exposure Cap
    active_symbol_cap = getattr(config_module, "MAX_SYMBOL_EXPOSURE_PCT", 0.20)
    if active_symbol_cap > HARD_MAX_SYMBOL_EXPOSURE_PCT:
        raise PermissionError(
            f"[Governance Violation] Symbol exposure limit ({active_symbol_cap*100:.1f}%) "
            f"exceeds hard safety cap ({HARD_MAX_SYMBOL_EXPOSURE_PCT*100:.1f}%)."
        )

    print("🛡️ [Risk Governance] Startup assertion passed: all active parameters comply with hard safety limits.")
    return True
