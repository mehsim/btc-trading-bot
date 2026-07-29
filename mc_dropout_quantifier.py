"""
mc_dropout_quantifier.py
------------------------
Monte Carlo (MC) Dropout Uncertainty Quantifier.
Estimates model epistemic uncertainty (model doubt) by evaluating stochastic variance
across stochastic ensemble prediction passes to complement conformal prediction.
"""

import numpy as np
from typing import Tuple, Dict, Any

class MCDropoutQuantifier:
    def __init__(self, n_samples: int = 10):
        self.n_samples = n_samples

    def quantify_uncertainty(self, probs_list: list) -> Tuple[float, float, bool]:
        """
        Calculates mean prediction probability and epistemic uncertainty (variance).
        Returns: (mean_conf, variance_uncertainty, is_uncertain)
        """
        if not probs_list:
            return 0.5, 0.0, False

        probs_arr = np.array(probs_list)
        mean_probs = np.mean(probs_arr, axis=0)
        var_probs = float(np.var(probs_arr, axis=0).max())
        
        winning_class = int(np.argmax(mean_probs))
        mean_conf = float(mean_probs[winning_class])
        
        # High epistemic uncertainty if variance across ensemble predictions > 0.04
        is_uncertain = (var_probs > 0.04)
        return mean_conf, var_probs, is_uncertain

mc_dropout_quantifier = MCDropoutQuantifier()
