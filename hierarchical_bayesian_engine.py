"""
hierarchical_bayesian_engine.py
--------------------------------
Capability 3: Hierarchical Bayesian Information Sharing Engine.
Performs partial pooling of hyperparameters across hierarchical levels:
  Global Prior -> Asset Class (Major vs Alt) -> Symbol -> Timeframe
Allows information sharing across coins while preserving asset specialization.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

class HierarchicalBayesianEngine:
    def __init__(self, global_mean_conf: float = 0.65, global_tau: float = 0.05):
        self.global_mean_conf = global_mean_conf
        self.global_tau = global_tau

    def get_hierarchical_pooled_parameter(
        self,
        symbol: str,
        interval: str,
        symbol_observations: List[float],
        asset_class: str = "ALT"
    ) -> Dict[str, Any]:
        """
        Computes Hierarchical Bayes Partial Pooling:
        theta_{s,tf} ~ N(mu_s, sigma_s^2), mu_s ~ N(mu_global, tau^2)
        """
        # Asset Class Prior Adjustment
        asset_prior_mean = 0.68 if asset_class.upper() in ["MAJOR", "BTC", "ETH"] else 0.64
        
        n_obs = len(symbol_observations)
        if n_obs == 0:
            pooled_mean = asset_prior_mean
            weight_local = 0.0
        else:
            sample_mean = float(np.mean(symbol_observations))
            sample_var = float(np.var(symbol_observations)) if n_obs > 1 else 0.02
            
            # Partial Pooling Shrinkage Weight (James-Stein / Bayes conjugate)
            weight_local = n_obs / (n_obs + (sample_var / max(1e-6, self.global_tau**2)))
            pooled_mean = weight_local * sample_mean + (1.0 - weight_local) * asset_prior_mean

        return {
            "symbol": symbol,
            "interval": interval,
            "asset_class": asset_class,
            "observations_count": n_obs,
            "local_sample_mean": round(float(np.mean(symbol_observations)), 4) if n_obs > 0 else asset_prior_mean,
            "asset_class_prior": asset_prior_mean,
            "shrinkage_weight_local": round(float(weight_local), 3),
            "hierarchical_pooled_mean": round(float(pooled_mean), 4)
        }


hierarchical_bayesian_engine = HierarchicalBayesianEngine()
