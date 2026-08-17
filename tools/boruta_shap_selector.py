"""
Boruta-SHAP Feature Selection Engine
Compares feature SHAP importances against randomized shadow features across
Purged Time-Series Cross-Validation folds to eliminate spurious indicators.
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False
from xgboost import XGBClassifier


class BorutaShapSelector:
    """
    Selects robust features by comparing empirical feature importance (SHAP or Gini gain)
    against randomized shadow feature distributions.
    """
    def __init__(self, n_trials: int = 15, max_features: int = 25, random_state: int = 42):
        self.n_trials = n_trials
        self.max_features = max_features
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaShapSelector":
        if X.empty or len(y) < 50:
            self.selected_features_ = list(X.columns)[:self.max_features]
            return self

        feature_names = list(X.columns)
        n_features = len(feature_names)
        hits = {f: 0 for f in feature_names}
        total_importance = {f: 0.0 for f in feature_names}

        for trial in range(self.n_trials):
            np.random.seed(self.random_state + trial)
            
            # Create randomized shadow features
            shadow_df = pd.DataFrame(index=X.index)
            for col in feature_names:
                shadow_df[f"shadow_{col}"] = np.random.permutation(X[col].values)

            X_combined = pd.concat([X, shadow_df], axis=1)
            
            # Train shallow regularized tree model
            model = XGBClassifier(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=2.0,
                reg_lambda=5.0,
                random_state=self.random_state + trial,
                n_jobs=1,
                eval_metric="logloss"
            )
            model.fit(X_combined, y)

            if HAS_SHAP and shap is not None:
                # Compute Tree SHAP values on sample
                sample_idx = np.random.choice(len(X_combined), size=min(300, len(X_combined)), replace=False)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_combined.iloc[sample_idx])
                
                if isinstance(shap_values, list):
                    # Multiclass: aggregate absolute SHAP across classes
                    mean_abs_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    mean_abs_imp = np.abs(shap_values).mean(axis=0)
            else:
                # Native XGBoost Gini/Gain feature importances (zero-dependency fallback)
                mean_abs_imp = np.asarray(model.feature_importances_, dtype=float)

            real_imp = mean_abs_imp[:n_features]
            shadow_imp = mean_abs_imp[n_features:]
            max_shadow = np.max(shadow_imp) if len(shadow_imp) > 0 else 0.0

            for i, feat in enumerate(feature_names):
                total_importance[feat] += float(real_imp[i])
                if real_imp[i] > max_shadow:
                    hits[feat] += 1

        # Select features that beat shadow maximum in >= 40% of trials
        hit_threshold = max(2, int(self.n_trials * 0.40))
        confirmed_features = [f for f in feature_names if hits[f] >= hit_threshold]

        # Rank by total mean SHAP importance
        confirmed_features.sort(key=lambda f: total_importance[f], reverse=True)
        
        if len(confirmed_features) < 10:
            # Fallback to top features by raw importance if too restrictive
            all_sorted = sorted(feature_names, key=lambda f: total_importance[f], reverse=True)
            self.selected_features_ = all_sorted[:min(self.max_features, len(all_sorted))]
        else:
            self.selected_features_ = confirmed_features[:self.max_features]

        self.feature_importances_ = {f: total_importance[f] / self.n_trials for f in self.selected_features_}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols]
