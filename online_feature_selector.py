"""
online_feature_selector.py
--------------------------
Online Dynamic Feature Selection Engine.
Ranks features dynamically using Mean Decrease Impurity (MDI) feature importances
and filters top alpha features to prevent overfitting on stale indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

class OnlineFeatureSelector:
    def __init__(self, top_n_features: int = 35):
        self.top_n_features = top_n_features

    def select_top_features(self, feature_names: List[str], importances: np.ndarray) -> List[str]:
        """
        Ranks and filters the top N highest alpha features based on model feature importances.
        """
        if importances is None or len(importances) != len(feature_names):
            return feature_names[:self.top_n_features]

        feature_scores = list(zip(feature_names, importances))
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        top_selected = [f[0] for f in sorted_features[:self.top_n_features]]
        return top_selected

online_feature_selector = OnlineFeatureSelector()
