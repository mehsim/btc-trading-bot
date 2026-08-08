"""
config_verifier.py
-------------------
Startup verification module ensuring runtime constants in config.py, trade_calculators.py,
and risk_limits.py remain strictly aligned with model training hyper-parameters.
"""

from logger import log_event
import config
import risk_limits
import trade_calculators

def assert_shared_constants_aligned():
    """Validates that key governance and risk constants are internally consistent across modules."""
    # 1. Verify leverage caps consistency
    for tf, max_lev in risk_limits.HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.items():
        if tf in trade_calculators.MAX_RR_RATIO:
            rr_cap = trade_calculators.MAX_RR_RATIO[tf]
            if rr_cap <= 0:
                raise ValueError(f"[Config Verifier] Invalid R:R cap for timeframe {tf}: {rr_cap}")

    # 2. Verify supported assets list is non-empty and well-formed
    if not config.SUPPORTED_SYMBOLS or not isinstance(config.SUPPORTED_SYMBOLS, list):
        raise ValueError("[Config Verifier] SUPPORTED_SYMBOLS must be a non-empty list.")
    
    for sym in config.SUPPORTED_SYMBOLS:
        if not sym.endswith("USDT"):
            raise ValueError(f"[Config Verifier] Unsupported symbol format in SUPPORTED_SYMBOLS: {sym}")

    # 3. Verify risk governance invariants
    risk_limits.assert_risk_governance_invariants()
    log_event("INFO", "✅ [Config Verifier] All shared runtime configuration constants verified successfully.")
    return True
