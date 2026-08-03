"""
risk_multiplier.py
------------------
Phase 1B: Conservative Risk Multiplier.
Calculates a position-size multiplier strictly in the range [0.50, 1.00].
ONLY shrinks position size on low-win-rate or low-confidence setups (N >= 30).
Never inflates position size beyond 1.0.
"""

from typing import Dict, Any
from pattern_miner import pattern_miner

class ConservativeRiskMultiplier:
    def __init__(self, min_sample_size: int = 30):
        self.min_sample_size = min_sample_size

    def get_risk_multiplier(self, signal_context: Dict[str, Any]) -> float:
        """
        Returns position size multiplier in [0.50, 1.00].
        """
        cluster_key = pattern_miner.generate_cluster_key(signal_context)
        patterns = pattern_miner.mine_patterns(limit=500)
        
        matching_cluster = None
        for p in patterns:
            if p["cluster_key"] == cluster_key:
                matching_cluster = p
                break
                
        if not matching_cluster or matching_cluster["sample_size"] < self.min_sample_size:
            return 1.0  # Insufficient data to adjust — baseline
            
        win_rate = matching_cluster["win_rate"]
        
        # If win rate is below 50% baseline on N >= 30 trades, shrink position proportional to deficit
        if win_rate < 0.50:
            deficit = 0.50 - win_rate
            multiplier = max(0.50, 1.0 - (deficit * 2.0))
            return round(multiplier, 2)
            
        return 1.0  # Never inflate above 1.0

risk_multiplier_engine = ConservativeRiskMultiplier()
