"""
probabilistic_policy_selector.py
----------------------------------
Probabilistic Risk-Adjusted Utility Policy Selector.
Evaluates candidate policies {P1, P2, P3, P4} under risk-adjusted utility:
  Utility(P_k) = Expected R(P_k) - 0.5 * Variance(P_k)
Returns policy candidate with maximum risk-adjusted utility.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

class ProbabilisticPolicySelector:
    def __init__(self, risk_aversion_gamma: float = 0.5):
        self.risk_aversion_gamma = risk_aversion_gamma

    def select_best_policy(
        self,
        symbol: str,
        interval: str,
        regime: str,
        expected_r_base: float = 1.4,
        uncertainty: float = 0.04
    ) -> Dict[str, Any]:
        """
        Evaluates 4 policy candidates under Utility = E[R] - gamma * Var[R].
        """
        # Candidate 1: Structural Wide Policy
        exp_r_1 = expected_r_base * 1.15
        var_1 = 0.35 + uncertainty * 2.0
        util_1 = exp_r_1 - (self.risk_aversion_gamma * var_1)

        # Candidate 2: Tight Scalp Policy
        exp_r_2 = expected_r_base * 0.90
        var_2 = 0.15 + uncertainty * 1.0
        util_2 = exp_r_2 - (self.risk_aversion_gamma * var_2)

        # Candidate 3: Time Decay Adaptive Policy
        exp_r_3 = expected_r_base * 1.05
        var_3 = 0.22 + uncertainty * 1.2
        util_3 = exp_r_3 - (self.risk_aversion_gamma * var_3)

        # Candidate 4: Conservative Low-Leverage Policy
        exp_r_4 = expected_r_base * 0.95
        var_4 = 0.10 + uncertainty * 0.5
        util_4 = exp_r_4 - (self.risk_aversion_gamma * var_4)

        candidates = [
            {"policy_id": "Structural_Wide", "expected_r": round(exp_r_1, 3), "variance": round(var_1, 3), "utility": round(util_1, 3)},
            {"policy_id": "Tight_Scalp", "expected_r": round(exp_r_2, 3), "variance": round(var_2, 3), "utility": round(util_2, 3)},
            {"policy_id": "Time_Decay_Adaptive", "expected_r": round(exp_r_3, 3), "variance": round(var_3, 3), "utility": round(util_3, 3)},
            {"policy_id": "Conservative_LowLev", "expected_r": round(exp_r_4, 3), "variance": round(var_4, 3), "utility": round(util_4, 3)}
        ]

        best_candidate = max(candidates, key=lambda c: c["utility"])

        return {
            "selected_policy": best_candidate["policy_id"],
            "max_utility": best_candidate["utility"],
            "expected_r": best_candidate["expected_r"],
            "variance": best_candidate["variance"],
            "candidate_evaluations": candidates
        }


probabilistic_policy_selector = ProbabilisticPolicySelector()
