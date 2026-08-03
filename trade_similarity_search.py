"""
trade_similarity_search.py
--------------------------
Phase 1C: Historical Trade Similarity Search.
Uses normalized Euclidean / Cosine feature distance over historical trade experiences
to return the top N most similar past trades for any candidate signal.
"""

import math
from typing import Dict, Any, List
from experience_db import get_recent_experiences

class TradeSimilaritySearch:
    def __init__(self, feature_weights: Dict[str, float] = None):
        if feature_weights is None:
            self.feature_weights = {
                "adx": 1.5,
                "atr_pct": 1.2,
                "confidence": 2.0,
                "funding_rate": 1.0,
                "oi_z_score": 1.0,
                "ltf_conflict": 2.5
            }
        else:
            self.feature_weights = feature_weights

    def compute_distance(self, vec1: Dict[str, Any], vec2: Dict[str, Any]) -> float:
        dist_sq = 0.0
        for feat, weight in self.feature_weights.items():
            v1 = float(vec1.get(feat, 0.0) or 0.0)
            v2 = float(vec2.get(feat, 0.0) or 0.0)
            dist_sq += weight * ((v1 - v2) ** 2)
        return math.sqrt(dist_sq)

    def find_similar_trades(self, signal_context: Dict[str, Any], top_n: int = 10, limit: int = 200) -> Dict[str, Any]:
        past_trades = get_recent_experiences(limit=limit)
        if not past_trades:
            return {
                "similar_trades_count": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "top_similar_ids": []
            }
            
        scored_trades = []
        for t in past_trades:
            d = self.compute_distance(signal_context, t)
            scored_trades.append((d, t))
            
        scored_trades.sort(key=lambda x: x[0])
        top_matches = [t for _, t in scored_trades[:top_n]]
        
        wins = sum(1 for t in top_matches if (t.get("trade_outcome") == "WIN" or t.get("pnl_usd", 0) > 0))
        sum_r = sum(t.get("realized_r", 0.0) for t in top_matches)
        
        n = len(top_matches)
        wr = round(wins / n, 4) if n > 0 else 0.0
        avg_r = round(sum_r / n, 4) if n > 0 else 0.0
        
        return {
            "similar_trades_count": n,
            "win_rate": wr,
            "avg_r": avg_r,
            "top_similar_ids": [t.get("trade_id") for t in top_matches]
        }

trade_similarity_search = TradeSimilaritySearch()
