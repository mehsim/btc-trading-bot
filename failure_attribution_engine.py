"""
failure_attribution_engine.py
-------------------------------
Phase 1B: Failure Attribution Engine.
Calculates weighted percentage attribution across 5 core failure factors for losing trades:
1. LTF Reversal (15M counter-trend / momentum)
2. High Volatility (ATR percentile / BB expansion)
3. Poor Entry Timing (distance from signal to execution)
4. High Funding (adverse funding rate drain)
5. Random Noise (residual variance)
"""

from typing import Dict, Any
from trade_calculators import safe_float

class FailureAttributionEngine:
    def diagnose_loss(self, record: Dict[str, Any]) -> Dict[str, Any]:
        pnl = safe_float(record.get("pnl_usd", 0.0))
        if pnl >= 0:
            return {}  # No loss to diagnose
            
        ltf_conflict = bool(record.get("ltf_conflict", False))
        atr_pct = safe_float(record.get("atr_pct", 1.0), 1.0)
        atr_percentile = safe_float(record.get("atr_percentile", 50.0), 50.0)
        funding = safe_float(record.get("funding_rate", 0.0))
        latency = safe_float(record.get("execution_latency_ms", 0.0))
        
        scores = {}
        
        # 1. LTF Reversal Score
        if ltf_conflict:
            scores["ltf_reversal"] = 45.0
        else:
            scores["ltf_reversal"] = 5.0
            
        # 2. High Volatility Score
        if atr_percentile >= 75.0 or atr_pct > 0.02:
            scores["high_volatility"] = 30.0
        elif atr_percentile >= 50.0:
            scores["high_volatility"] = 15.0
        else:
            scores["high_volatility"] = 5.0
            
        # 3. Poor Entry Timing
        if latency > 1000:  # >1s latency
            scores["poor_entry_timing"] = 15.0
        else:
            scores["poor_entry_timing"] = 5.0
            
        # 4. High Funding
        if abs(funding) > 0.0003:  # >0.03%
            scores["high_funding"] = 10.0
        else:
            scores["high_funding"] = 2.0
            
        # 5. Residual Random Noise
        scores["random_noise"] = 10.0
        
        total_score = sum(scores.values())
        
        # Normalize to percentages summing to 100%
        attribution = {}
        for factor, raw_score in scores.items():
            pct = round((raw_score / total_score) * 100.0)
            evidence = self._get_evidence_text(factor, record)
            attribution[factor] = {"pct": pct, "evidence": evidence}
            
        return attribution

    def _get_evidence_text(self, factor: str, record: Dict[str, Any]) -> str:
        if factor == "ltf_reversal":
            return "15M timeframe counter-trend alignment" if record.get("ltf_conflict") else "15M timeframe aligned"
        elif factor == "high_volatility":
            return f"ATR percentile at {record.get('atr_percentile', 50):.1f}th"
        elif factor == "poor_entry_timing":
            return f"Execution latency {record.get('execution_latency_ms', 0):.0f}ms"
        elif factor == "high_funding":
            return f"Funding rate {record.get('funding_rate', 0)*100:.4f}%"
        else:
            return "Unexplained market noise"

failure_attribution_engine = FailureAttributionEngine()
