"""
shap_explainer.py
------------------
Cross-Model SHAP (SHapley Additive exPlanations) Explainability Engine.
Computes TreeExplainer SHAP values for XGBoost/LightGBM/CatBoost and
extracts top feature contributions per prediction for debugging and audit.
"""

import numpy as np
from typing import Dict, Any, List, Optional

class SHAPExplainer:

    def __init__(self):
        self._shap_available = False
        try:
            import shap  # noqa
            self._shap_available = True
        except ImportError:
            print("[SHAP] shap package not installed. Install with: pip install shap")

    def explain_prediction(
        self,
        model,
        feature_values: Dict[str, float],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Computes SHAP values for a single prediction using TreeExplainer.
        Returns top_n feature contributions sorted by absolute impact.

        Args:
            model: Trained XGBoost / LightGBM / CatBoost model
            feature_values: dict of {feature_name: value}
            top_n: number of top features to return

        Returns:
            Dict with top feature contributions, base value, and prediction
        """
        if not self._shap_available:
            return {"status": "shap_not_installed", "top_features": []}

        import shap
        import pandas as pd

        try:
            features = list(feature_values.keys())
            X = pd.DataFrame([feature_values])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Handle multi-output (CatBoost/LightGBM may return list)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # class 1 (positive)
            else:
                shap_vals = shap_values[0]

            base_value = float(explainer.expected_value) if not isinstance(
                explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[1])

            contributions = sorted(
                [{"feature": f, "shap_value": round(float(v), 5),
                  "feature_value": round(float(feature_values[f]), 4),
                  "direction": "↑ BULLISH" if v > 0 else "↓ BEARISH"}
                 for f, v in zip(features, shap_vals)],
                key=lambda x: abs(x["shap_value"]), reverse=True
            )

            return {
                "status": "ok",
                "base_value": round(base_value, 4),
                "prediction": round(base_value + sum(shap_vals), 4),
                "top_features": contributions[:top_n],
                "total_features_analyzed": len(features)
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "top_features": []}

    def explain_ensemble(
        self,
        models: Dict[str, Any],
        feature_values: Dict[str, float],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Runs SHAP on each model in the ensemble and aggregates feature importance.
        Returns consensus top features across XGBoost, LightGBM, CatBoost.
        """
        if not self._shap_available:
            return {"status": "shap_not_installed"}

        all_contributions: Dict[str, List[float]] = {}
        per_model_results = {}

        for model_name, model in models.items():
            result = self.explain_prediction(model, feature_values, top_n=len(feature_values))
            per_model_results[model_name] = result
            if result["status"] == "ok":
                for feat in result["top_features"]:
                    fname = feat["feature"]
                    if fname not in all_contributions:
                        all_contributions[fname] = []
                    all_contributions[fname].append(feat["shap_value"])

        # Average SHAP across models
        consensus = sorted(
            [{"feature": f, "mean_shap": round(float(np.mean(vals)), 5),
              "agreement": "STRONG" if np.std(vals) < 0.02 else "MODERATE" if np.std(vals) < 0.05 else "WEAK"}
             for f, vals in all_contributions.items()],
            key=lambda x: abs(x["mean_shap"]), reverse=True
        )

        return {
            "status": "ok",
            "per_model": per_model_results,
            "consensus_top_features": consensus[:top_n]
        }


shap_explainer = SHAPExplainer()
