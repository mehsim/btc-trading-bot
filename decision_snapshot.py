"""
decision_snapshot.py
--------------------
Phase 1A: Decision Snapshot Builder.
Captures model beliefs, reason codes, indicators, and market context at trade execution time.
"""

import time
from typing import Dict, Any, List, Optional

def build_decision_snapshot(
    symbol: str,
    direction: str,
    confidence: float,
    signal_score: float = 0.90,
    market_regime: str = "TRENDING",
    adx: float = 25.0,
    atr_pct: float = 1.0,
    oi_z: float = 0.0,
    funding: float = 0.0,
    htf: str = "BEARISH",
    ltf: str = "BULLISH",
    reason_codes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Constructs a standardized decision snapshot dictionary.
    """
    if reason_codes is None:
        reason_codes = ["HTF_ALIGNMENT", "ADX_STRONG"]
        
    return {
        "timestamp": time.time(),
        "symbol": symbol,
        "direction": direction,
        "confidence": round(float(confidence), 4),
        "signal_score": round(float(signal_score), 4),
        "market_regime": market_regime,
        "adx": round(float(adx), 2),
        "atr_pct": round(float(atr_pct), 4),
        "oi_z": round(float(oi_z), 2),
        "funding": round(float(funding), 6),
        "htf": htf,
        "ltf": ltf,
        "reason_codes": reason_codes
    }
