"""
Empirical Bootstrap Utility Distribution Estimator.
Estimates non-Gaussian Expected Utility distribution, P(Utility > 0), and 95% CVaR (Expected Shortfall) using Empirical Bootstrap Resampling.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class EmpiricalUtilityEstimator:
    def __init__(self, n_bootstrap_samples: int = 1000, random_seed: int = 42):
        self.n_bootstrap_samples = n_bootstrap_samples
        self.rng = np.random.RandomState(random_seed)

    def estimate_utility_distribution(
        self,
        predicted_win_rate: float,
        target_distance: float,
        stop_distance: float,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        roundtrip_fee_pct: float = 0.0010
    ) -> Dict[str, float]:
        """
        Computes non-parametric empirical bootstrap distribution of trade outcomes.
        Returns:
            - expected_utility_mean: Mean Expected Utility in R-multiples or USD
            - expected_utility_std: Standard deviation of Expected Utility
            - p_utility_positive: Probability that Utility > 0
            - cvar_95: 95% Conditional Value at Risk (Expected Shortfall)
            - median_utility: 50th percentile Utility
        """
        # Baseline historical returns or synthetic kernel if sample size is small
        r_outcomes = []
        if trade_history and len(trade_history) >= 10:
            for t in trade_history:
                pnl = t.get("realized_pnl", 0.0)
                risk = t.get("risk_amount", 1.0)
                if risk > 0:
                    r_outcomes.append(pnl / risk)

        if len(r_outcomes) < 10:
            # Construct synthetic empirical returns anchored on predicted win rate & R:R
            win_r = target_distance / stop_distance if stop_distance > 0 else 1.5
            loss_r = -1.0 - (roundtrip_fee_pct * 2.0)
            win_r_net = win_r - (roundtrip_fee_pct * 2.0)

            # Build empirical distribution with fat tails
            n_wins = int(predicted_win_rate * 100)
            n_losses = 100 - n_wins
            r_outcomes = [win_r_net * (1.0 + self.rng.normal(0, 0.15)) for _ in range(n_wins)] + \
                         [loss_r * (1.0 + abs(self.rng.normal(0, 0.20))) for _ in range(n_losses)]

        r_array = np.array(r_outcomes, dtype=float)
        
        # Empirical Bootstrap Resampling
        bootstrap_means = []
        sample_size = len(r_array)
        for _ in range(self.n_bootstrap_samples):
            boot_sample = self.rng.choice(r_array, size=sample_size, replace=True)
            bootstrap_means.append(np.mean(boot_sample))

        boot_means_arr = np.array(bootstrap_means)
        
        mean_u = float(np.mean(boot_means_arr))
        std_u = float(np.std(boot_means_arr))
        p_pos = float(np.mean(boot_means_arr > 0))
        median_u = float(np.median(boot_means_arr))

        # 95% Conditional Value at Risk (CVaR / Expected Shortfall)
        var_95_threshold = np.percentile(boot_means_arr, 5)
        tail_losses = boot_means_arr[boot_means_arr <= var_95_threshold]
        cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else float(var_95_threshold)

        return {
            "expected_utility_mean": round(mean_u, 4),
            "expected_utility_std": round(std_u, 4),
            "p_utility_positive": round(p_pos, 4),
            "cvar_95": round(cvar_95, 4),
            "median_utility": round(median_u, 4)
        }

empirical_utility_estimator = EmpiricalUtilityEstimator()
