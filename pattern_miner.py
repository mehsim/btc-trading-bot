"""
pattern_miner.py
----------------
Phase 1B: Stage 1 Rule-Based Pattern Miner.
Discovers trade clusters based on deterministic setup keys and computes win rates,
sample sizes, Wilson 95% confidence intervals, and average R.
"""

import math
from typing import Dict, Any, List, Tuple
from experience_db import get_recent_experiences
from trade_calculators import safe_float

def wilson_score_interval(wins: int, n: int, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score 95% confidence interval for a binomial proportion.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96  # 95% confidence
    phat = wins / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))

class RuleBasedPatternMiner:
    def __init__(self, min_sample_size: int = 10):
        self.min_sample_size = min_sample_size

    def generate_cluster_key(self, record: Dict[str, Any]) -> str:
        regime = str(record.get("market_regime", "TRENDING"))
        ltf_conf = "CONFLICT" if record.get("ltf_conflict") else "ALIGNED"
        conf = safe_float(record.get("confidence", 0.5), 0.5)
        conf_bucket = "HIGH" if conf >= 0.80 else ("MED" if conf >= 0.65 else "LOW")
        atr_p = safe_float(record.get("atr_percentile", 50.0), 50.0)
        vol = "HIGH_VOL" if atr_p >= 75.0 else "NORM_VOL"
        
        return f"{regime}|{ltf_conf}|{conf_bucket}|{vol}"

    def mine_patterns(self, limit: int = 500, decay_factor: float = 0.98) -> List[Dict[str, Any]]:
        trades = get_recent_experiences(limit=limit)
        clusters = {}
        
        # trades are ordered DESC (idx 0 is newest trade)
        for idx, t in enumerate(trades):
            weight = decay_factor ** idx
            key = self.generate_cluster_key(t)
            if key not in clusters:
                clusters[key] = {"wins": 0, "total": 0, "sum_r": 0.0, "weighted_wins": 0.0, "weighted_total": 0.0, "trades": []}
                
            clusters[key]["total"] += 1
            clusters[key]["weighted_total"] += weight
            is_win = (t.get("trade_outcome") == "WIN" or safe_float(t.get("pnl_usd", 0.0)) > 0)
            if is_win:
                clusters[key]["wins"] += 1
                clusters[key]["weighted_wins"] += weight
            clusters[key]["sum_r"] += safe_float(t.get("realized_r", 0.0)) * weight
            clusters[key]["trades"].append(t.get("trade_id"))
            
        results = []
        for key, data in clusters.items():
            n = data["total"]
            w = data["wins"]
            w_tot = data["weighted_total"]
            w_win = data["weighted_wins"]
            
            # Recency-weighted win rate
            win_rate = round(w_win / w_tot, 4) if w_tot > 0 else 0.0
            avg_r = round(data["sum_r"] / w_tot, 4) if w_tot > 0 else 0.0
            ci_lower, ci_upper = wilson_score_interval(w, n)
            
            results.append({
                "cluster_key": key,
                "sample_size": n,
                "wins": w,
                "win_rate": win_rate,
                "avg_r": avg_r,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "trade_ids": data["trades"],
                "is_significant": n >= self.min_sample_size
            })
            
        results.sort(key=lambda x: x["sample_size"], reverse=True)
        return results


pattern_miner = RuleBasedPatternMiner()
