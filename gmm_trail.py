import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, List

class GMMTrailingEngine:
    def __init__(self):
        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.is_fitted = False
        self.trending_component_idx = 1

    def fit_adx_distribution(self, adx_series: List[float]):
        """Fits 2-component Gaussian Mixture Model on historical ADX values."""
        clean_adx = [a for a in adx_series if not np.isnan(a) and a > 0]
        if len(clean_adx) < 50:
            return

        X = np.array(clean_adx).reshape(-1, 1)
        self.gmm.fit(X)
        self.is_fitted = True

        # Assign component with higher mean ADX as the 'Trending' component
        means = self.gmm.means_.flatten()
        self.trending_component_idx = int(np.argmax(means))

    def calculate_gmm_trailing_multiplier(self, current_adx: float) -> float:
        """
        Rule 6: Continuously interpolates trailing multiplier between 0.90x and 1.50x
        multiplier = 0.90 + (1.50 - 0.90) * p_trending
        """
        if not self.is_fitted or current_adx is None or np.isnan(current_adx) or current_adx <= 0:
            # Fallback step rules if GMM is not fitted yet
            if current_adx is not None and current_adx >= 25.0:
                return 1.50
            elif current_adx is not None and current_adx < 18.0:
                return 0.90
            return 1.25



        probs = self.gmm.predict_proba([[current_adx]])[0]
        p_trend = float(probs[self.trending_component_idx])

        # Smooth interpolation between 0.90x and 1.50x
        multiplier = 0.90 + (1.50 - 0.90) * p_trend
        return float(np.clip(multiplier, 0.90, 1.50))

gmm_trailing_engine = GMMTrailingEngine()
