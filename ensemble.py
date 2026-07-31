import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from xgboost import XGBClassifier, XGBRegressor

def _slice_model_input(model, X):
    """
    Slices input feature matrix X to match model's expected number of input features (n_features_in_)
    if X has extra columns/features.
    """
    if model is None or X is None:
        return X

    n_expected = None
    if hasattr(model, "booster_") and hasattr(model.booster_, "num_feature"):
        try:
            n_expected = model.booster_.num_feature()
        except Exception:
            pass
    if n_expected is None and hasattr(model, "feature_names_") and model.feature_names_:
        try:
            n_expected = len(model.feature_names_)
        except Exception:
            pass
    if n_expected is None and hasattr(model, "get_num_features"):
        try:
            n_expected = model.get_num_features()
        except Exception:
            pass
    if n_expected is None:
        n_expected = getattr(model, "n_features_in_", None)
    if n_expected is None:
        n_expected = getattr(model, "_n_features_in", None)
    if n_expected is None:
        n_expected = getattr(model, "_n_features", None)
    if n_expected is None:
        n_expected = getattr(model, "n_features_", None)

    if n_expected is None:
        return X

    try:
        if isinstance(X, pd.DataFrame):
            if hasattr(model, "feature_names_") and model.feature_names_:
                valid_cols = [c for c in model.feature_names_ if c in X.columns]
                if len(valid_cols) == n_expected:
                    return X[valid_cols]
            if X.shape[1] > n_expected:
                return X.iloc[:, :n_expected]
            return X
        elif isinstance(X, pd.Series):
            if len(X) > n_expected:
                return X.iloc[:n_expected]
            return X
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 2:
                if X_arr.shape[1] > n_expected:
                    return X_arr[:, :n_expected]
                elif X_arr.shape[1] < n_expected:
                    pad_len = n_expected - X_arr.shape[1]
                    return np.pad(X_arr, ((0, 0), (0, pad_len)), mode="constant")
                return X_arr
            elif X_arr.ndim == 1:
                if X_arr.shape[0] > n_expected:
                    return X_arr[:n_expected]
                elif X_arr.shape[0] < n_expected:
                    pad_len = n_expected - X_arr.shape[0]
                    return np.pad(X_arr, (0, pad_len), mode="constant")
                return X_arr
            return X
    except Exception:
        return X

class PurgedEmbargoTimeSeriesSplit:
    """
    Implements Purged and Embargoed Time-Series Cross-Validation.
    Prevents overlapping lookahead window leakage between train and validation sets.
    """
    def __init__(self, n_splits=5, lookahead=6, embargo_pct=0.01):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        block_size = n_samples // (self.n_splits + 1)
        
        for i in range(1, self.n_splits + 1):
            val_start = i * block_size
            val_end = min(n_samples, (i + 1) * block_size)
            
            # Purge lookahead samples before the validation block
            train_1_end = max(0, val_start - self.lookahead)
            train_1_indices = indices[:train_1_end]
            
            # Embargo samples after the validation block to allow post-event drift to cool down
            embargo_offset = int(n_samples * self.embargo_pct)
            train_2_start = min(n_samples, val_end + self.lookahead + embargo_offset)
            train_2_indices = indices[train_2_start:]
            
            train_indices = np.concatenate([train_1_indices, train_2_indices])
            val_indices = indices[val_start:val_end]
            
            if len(train_indices) > 50 and len(val_indices) > 50:
                yield train_indices, val_indices

class EnsembleClassifier:
    """
    Blends XGBoost, LightGBM, and CatBoost classifiers using a Stacking Meta-Classifier
    (Logistic Regression) with validation-calibrated coefficients and fallbacks.
    """
    def __init__(self, xgb_model, lgb_model=None, cat_model=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        self.meta_coef_ = None
        self.meta_intercept_ = None

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, X_train=None, y_train=None, sample_weight_train=None):
        X_arr = np.asarray(X, dtype=float)
        
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        self.meta_coef_ = None
        self.meta_intercept_ = None
        
        # Fit stacking meta-classifier on validation out-of-fold predictions
        if X_val is not None and y_val is not None and self.lgb_model is not None and self.cat_model is not None:
            try:
                from sklearn.metrics import accuracy_score
                from sklearn.linear_model import LogisticRegression
                
                # Fit base models on training fold ONLY to avoid data leakage
                X_tr = X_train if X_train is not None else X
                y_tr = y_train if y_train is not None else y
                w_tr = sample_weight_train if sample_weight_train is not None else sample_weight
                
                X_tr_arr = np.asarray(X_tr, dtype=float)
                self.xgb_model.fit(X_tr_arr, y_tr, sample_weight=w_tr)
                self.lgb_model.fit(X_tr_arr, y_tr, sample_weight=w_tr)
                self.cat_model.fit(X_tr_arr, y_tr, sample_weight=w_tr)
                
                X_v_arr = np.asarray(X_val, dtype=float)
                xgb_acc = accuracy_score(y_val, self.xgb_model.predict(X_v_arr))
                lgb_acc = accuracy_score(y_val, self.lgb_model.predict(X_v_arr))
                cat_acc = accuracy_score(y_val, self.cat_model.predict(X_v_arr))
                raw_weights = [max(0.01, xgb_acc), max(0.01, lgb_acc), max(0.01, cat_acc)]
                sum_w = max(1e-9, sum(raw_weights))
                self.weights = [w / sum_w for w in raw_weights]
                
                # Extract validation probabilities from each base model
                p_xgb = self.xgb_model.predict_proba(X_v_arr)
                p_lgb = self.lgb_model.predict_proba(X_v_arr)
                p_cat = self.cat_model.predict_proba(X_v_arr)
                
                # Stack features for meta-learner (multi-class probabilities stacked column-wise)
                # Extract out-of-fold predictions
                xgb_val_prob = self.xgb_model.predict_proba(_slice_model_input(self.xgb_model, X_val))
                lgb_val_prob = self.lgb_model.predict_proba(_slice_model_input(self.lgb_model, X_val))
                cat_val_prob = self.cat_model.predict_proba(_slice_model_input(self.cat_model, X_val))
                
                acc_xgb = accuracy_score(y_val, np.argmax(xgb_val_prob, axis=1))
                acc_lgb = accuracy_score(y_val, np.argmax(lgb_val_prob, axis=1))
                acc_cat = accuracy_score(y_val, np.argmax(cat_val_prob, axis=1))
                
                raw_w = np.array([acc_xgb, acc_lgb, acc_cat], dtype=float)
                raw_w = np.maximum(0.01, raw_w - 0.33)
                self.weights = (raw_w / np.sum(raw_w)).tolist()
                
                # Create Meta-Feature Matrix [N, 9] (3 classes x 3 models)
                X_meta = np.column_stack([xgb_val_prob, lgb_val_prob, cat_val_prob])
                
                # Fit L2 Regularized Logistic Regression Stacking Meta-Classifier
                meta_clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
                meta_clf.fit(X_meta, y_val)
                self.meta_coef_ = meta_clf.coef_.tolist()
                self.meta_intercept_ = meta_clf.intercept_.tolist()
                self.meta_clf = meta_clf
                print(f"[Ensemble Stacking] Classifier Meta-Learner calibrated successfully with accuracy weights: {self.weights}")
            except Exception as e:
                self.meta_coef_ = None
                self.meta_intercept_ = None
                self.meta_clf = None
                print(f"[Ensemble Stacking Warning] Stacking calibration failed, using weighted average fallback: {e}")
                
        # Now refit base models on the ENTIRE dataset for live trading
        self.xgb_model.fit(_slice_model_input(self.xgb_model, X_arr), y, sample_weight=sample_weight)
        if self.lgb_model is not None:
            try:
                self.lgb_model.fit(_slice_model_input(self.lgb_model, X_arr), y, sample_weight=sample_weight)
            except Exception:
                pass
        if self.cat_model is not None:
            try:
                self.cat_model.fit(_slice_model_input(self.cat_model, X_arr), y, sample_weight=sample_weight)
            except Exception:
                pass
        return self

    def predict_proba(self, X, weights=None):
        xgb_prob = self.xgb_model.predict_proba(_slice_model_input(self.xgb_model, X))

        lgb_prob = None
        if self.lgb_model is not None:
            try:
                lgb_prob = self.lgb_model.predict_proba(_slice_model_input(self.lgb_model, X))
            except Exception:
                lgb_prob = None

        cat_prob = None
        if self.cat_model is not None:
            try:
                cat_prob = self.cat_model.predict_proba(_slice_model_input(self.cat_model, X))
            except Exception:
                cat_prob = None

        if lgb_prob is None or cat_prob is None or getattr(self, "meta_coef_", None) is None:
            w_to_use = weights if weights is not None else getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            w = np.array(w_to_use, dtype=float)
            w = w / np.sum(w)
            if lgb_prob is None and cat_prob is None:
                return xgb_prob
            elif lgb_prob is not None and cat_prob is None:
                return (xgb_prob * 0.5 + lgb_prob * 0.5)
            elif lgb_prob is None and cat_prob is not None:
                return (xgb_prob * 0.5 + cat_prob * 0.5)
            return (xgb_prob * w[0] + lgb_prob * w[1] + cat_prob * w[2])
            
        # Stack probabilities for prediction
        X_meta = np.column_stack([xgb_prob, lgb_prob, cat_prob])
        
        if getattr(self, "meta_clf", None) is not None:
            try:
                return self.meta_clf.predict_proba(X_meta)
            except Exception:
                pass

        from sklearn.linear_model import LogisticRegression
        meta_clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
        coef = np.array(self.meta_coef_)
        intercept = np.array(self.meta_intercept_)
        
        meta_clf.n_features_in_ = X_meta.shape[1]
        if coef.ndim == 1 or coef.shape[0] == 1:
            meta_clf.coef_ = coef.reshape(1, -1)
            meta_clf.intercept_ = intercept.reshape(-1)
            meta_clf.classes_ = np.array([0, 2])
            try:
                binary_probs = meta_clf.predict_proba(X_meta)
                three_probs = np.zeros((binary_probs.shape[0], 3))
                three_probs[:, 0] = binary_probs[:, 0]
                three_probs[:, 2] = binary_probs[:, 1]
                three_probs[:, 1] = np.maximum(0.0, 1.0 - (three_probs[:, 0] + three_probs[:, 2]))
                row_sums = three_probs.sum(axis=1, keepdims=True)
                three_probs = three_probs / np.maximum(1e-9, row_sums)
                return three_probs

            except Exception:
                w = np.array(self.weights, dtype=float)
                w = w / max(1e-9, float(np.sum(w)))
                return (xgb_prob * w[0] + lgb_prob * w[1] + cat_prob * w[2])
        else:
            meta_clf.coef_ = coef
            meta_clf.intercept_ = intercept
            meta_clf.classes_ = np.array([0, 1, 2])
            return meta_clf.predict_proba(X_meta)

    def predict(self, X, weights=None):
        probs = self.predict_proba(X, weights=weights)
        return np.argmax(probs, axis=1)

    def predict_with_uncertainty(self, X, weights=None, uncertainty_threshold=0.18):
        """
        Calculates ensemble prediction probabilities alongside Conformal Uncertainty.
        Evaluates base model prediction variance and top-class margin.
        Returns: (probs, uncertainty_score, is_uncertain)
        """
        X_arr = np.asarray(X, dtype=float)
        probs = self.predict_proba(X_arr, weights=weights)
        
        # Calculate base model predictions to measure ensemble disagreement
        xgb_p = self.xgb_model.predict_proba(_slice_model_input(self.xgb_model, X_arr))
        if self.lgb_model is not None and self.cat_model is not None:
            lgb_p = self.lgb_model.predict_proba(_slice_model_input(self.lgb_model, X_arr))
            cat_p = self.cat_model.predict_proba(_slice_model_input(self.cat_model, X_arr))
            disagreement = np.std([xgb_p, lgb_p, cat_p], axis=0).mean()
        else:
            disagreement = 0.0
            
        # Top-class margin: margin between highest and second highest probability per row
        sorted_p = np.sort(probs, axis=1)
        margins = (sorted_p[:, -1] - sorted_p[:, -2]) if sorted_p.shape[1] >= 2 else np.ones(len(probs))
        margin = float(np.nan_to_num(margins[0] if len(margins) == 1 else margins.mean(), nan=0.10, posinf=1.0, neginf=0.0))
        
        # Conformal uncertainty score
        uncertainty_score = float(np.nan_to_num(disagreement * 0.7 + max(0.0, 0.25 - margin) * 0.3, nan=0.0, posinf=1.0, neginf=0.0))
        is_uncertain = (disagreement > uncertainty_threshold) or (margin < 0.08)
        
        return probs, uncertainty_score, is_uncertain

class EnsembleRegressor:
    """
    Blends XGBoost, LightGBM, and CatBoost regressors using a Stacking Meta-Regressor (Ridge).
    """
    def __init__(self, xgb_model, lgb_model=None, cat_model=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        self.meta_coef_ = None
        self.meta_intercept_ = None

    def fit(self, X, y, X_val=None, y_val=None, X_train=None, y_train=None):
        X_arr = np.asarray(X, dtype=float)
        
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        self.meta_coef_ = None
        self.meta_intercept_ = None
        
        if X_val is not None and y_val is not None and self.lgb_model is not None and self.cat_model is not None:
            try:
                from sklearn.metrics import mean_absolute_error
                from sklearn.linear_model import Ridge
                
                # Fit base models on training fold ONLY to avoid data leakage
                X_tr = X_train if X_train is not None else X
                y_tr = y_train if y_train is not None else y
                
                X_tr_arr = np.asarray(X_tr, dtype=float)
                self.xgb_model.fit(X_tr_arr, y_tr)
                self.lgb_model.fit(X_tr_arr, y_tr)
                self.cat_model.fit(X_tr_arr, y_tr)
                
                X_v_arr = np.asarray(X_val, dtype=float)
                xgb_mae = mean_absolute_error(y_val, self.xgb_model.predict(X_v_arr))
                lgb_mae = mean_absolute_error(y_val, self.lgb_model.predict(X_v_arr))
                cat_mae = mean_absolute_error(y_val, self.cat_model.predict(X_v_arr))
                raw_weights = [1.0 / max(1e-6, xgb_mae), 1.0 / max(1e-6, lgb_mae), 1.0 / max(1e-6, cat_mae)]
                sum_w = max(1e-9, sum(raw_weights))
                self.weights = [w / sum_w for w in raw_weights]
                
                # Extract predictions for stacking
                p_xgb = self.xgb_model.predict(X_v_arr)
                p_lgb = self.lgb_model.predict(X_v_arr)
                p_cat = self.cat_model.predict(X_v_arr)
                
                X_meta = np.column_stack([p_xgb, p_lgb, p_cat])
                
                # Apply exponential time-decay sample weighting (newer validation samples weighted higher)
                N_val = len(y_val)
                val_decay_weights = np.exp(-0.02 * np.arange(N_val)[::-1])
                val_decay_weights = val_decay_weights / np.sum(val_decay_weights) * N_val
                
                meta_reg = Ridge(alpha=1.0, random_state=42)
                meta_reg.fit(X_meta, y_val, sample_weight=val_decay_weights)
                self.meta_coef_ = meta_reg.coef_.tolist()
                self.meta_intercept_ = float(meta_reg.intercept_)
                print(f"[Ensemble Stacking] Regressor Meta-Learner calibrated successfully with time-decay weights.")
            except Exception as e:
                print(f"[Ensemble Stacking Warning] Stacking calibration failed, using weighted average fallback: {e}")
                
        # Now refit base models on the ENTIRE dataset for live trading
        self.xgb_model.fit(X_arr, y)
        if self.lgb_model is not None:
            self.lgb_model.fit(X_arr, y)
        if self.cat_model is not None:
            self.cat_model.fit(X_arr, y)
        return self

    def predict(self, X, weights=None):
        xgb_pred = self.xgb_model.predict(_slice_model_input(self.xgb_model, X))

        lgb_pred = None
        if self.lgb_model is not None:
            try:
                lgb_pred = self.lgb_model.predict(_slice_model_input(self.lgb_model, X))
            except Exception:
                lgb_pred = None

        cat_pred = None
        if self.cat_model is not None:
            try:
                cat_pred = self.cat_model.predict(_slice_model_input(self.cat_model, X))
            except Exception:
                cat_pred = None

        if lgb_pred is None or cat_pred is None or getattr(self, "meta_coef_", None) is None:
            w_to_use = weights if weights is not None else getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            w = np.array(w_to_use, dtype=float)
            w = w / np.sum(w)
            if lgb_pred is None and cat_pred is None:
                return xgb_pred
            elif lgb_pred is not None and cat_pred is None:
                return (xgb_pred * 0.5 + lgb_pred * 0.5)
            elif lgb_pred is None and cat_pred is not None:
                return (xgb_pred * 0.5 + cat_pred * 0.5)
            return (xgb_pred * w[0] + lgb_pred * w[1] + cat_pred * w[2])
            
        # Apply Ridge meta-model via linear matrix multiplication
        X_meta = np.column_stack([xgb_pred, lgb_pred, cat_pred])
        coef = np.array(self.meta_coef_, dtype=float)
        intercept = float(self.meta_intercept_) if self.meta_intercept_ is not None else 0.0
        return np.dot(X_meta, coef) + intercept

# ==========================================
# NATIVE SAVING/LOADING (TEXT/JSON ONLY)
# ==========================================

def save_ensemble_classifier(model, prefix):
    import json
    model.xgb_model.save_model(f"{prefix}_xgb.json")
    if hasattr(model.lgb_model, "_Booster") and model.lgb_model._Booster is not None:
        model.lgb_model._Booster.save_model(f"{prefix}_lgb.txt")
    elif hasattr(model.lgb_model, "booster_"):
        model.lgb_model.booster_.save_model(f"{prefix}_lgb.txt")
    elif hasattr(model.lgb_model, "save_model"):
        model.lgb_model.save_model(f"{prefix}_lgb.txt")
    model.cat_model.save_model(f"{prefix}_cat.json", format="json")
    
    meta_data = {
        "weights": getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0]),
        "meta_coef": getattr(model, "meta_coef_", None),
        "meta_intercept": getattr(model, "meta_intercept_", None)
    }
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(meta_data, f)

def load_ensemble_classifier(prefix, n_features=54):
    xgb = XGBClassifier()
    xgb.load_model(f"{prefix}_xgb.json")
    
    clf = EnsembleClassifier(xgb, None, None)
    
    import json
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r") as f:
                meta_data = json.load(f)
                clf.weights = meta_data.get("weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
                clf.meta_coef_ = meta_data.get("meta_coef")
                clf.meta_intercept_ = meta_data.get("meta_intercept")
        except Exception:
            clf.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            
    if os.environ.get("SPACE_ID"):
        return clf
        
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    lgb_clf = LGBMClassifier(objective="multiclass", num_class=3)
    booster_obj = lgb.Booster(model_file=f"{prefix}_lgb.txt")
    lgb_clf._Booster = booster_obj
    if hasattr(lgb_clf, "booster_"):
        try:
            lgb_clf.booster_ = booster_obj
        except Exception:
            pass
    lgb_clf.fitted_ = True
    lgb_clf._n_classes = 3
    lgb_clf._classes = np.array([0, 1, 2])
    lgb_clf._n_features = n_features
    lgb_clf._n_features_in = n_features
    lgb_clf.n_features_in_ = n_features
    
    cat = CatBoostClassifier()
    cat.load_model(f"{prefix}_cat.json", format="json")
    
    clf.lgb_model = lgb_clf
    clf.cat_model = cat
    return clf

import hashlib, subprocess, datetime

def write_model_manifest(prefix: str, feature_names: list = None, metrics: dict = None):
    try:
        git_sha = "unknown"
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            pass

        feat_str = ",".join(feature_names or [])
        feat_hash = hashlib.sha256(feat_str.encode("utf-8")).hexdigest()

        manifest = {
            "prefix": prefix,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": git_sha,
            "feature_count": len(feature_names or []),
            "feature_contract_hash": feat_hash,
            "metrics": metrics or {}
        }
        with open(f"{prefix}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"[Model Governance Warning] Could not write manifest for {prefix}: {e}")

def save_ensemble_regressor(model, prefix, feature_names=None):
    import json
    model.xgb_model.save_model(f"{prefix}_xgb.json")
    if hasattr(model.lgb_model, "_Booster") and model.lgb_model._Booster is not None:
        model.lgb_model._Booster.save_model(f"{prefix}_lgb.txt")
    elif hasattr(model.lgb_model, "booster_"):
        model.lgb_model.booster_.save_model(f"{prefix}_lgb.txt")
    elif hasattr(model.lgb_model, "save_model"):
        model.lgb_model.save_model(f"{prefix}_lgb.txt")
    model.cat_model.save_model(f"{prefix}_cat.json", format="json")
    
    meta_data = {
        "weights": getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0]),
        "meta_coef": getattr(model, "meta_coef_", None),
        "meta_intercept": getattr(model, "meta_intercept_", None)
    }
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(meta_data, f)
    write_model_manifest(prefix, feature_names=feature_names)

def load_ensemble_regressor(prefix, n_features=54):
    xgb = XGBRegressor()
    xgb.load_model(f"{prefix}_xgb.json")
    
    reg = EnsembleRegressor(xgb, None, None)
    
    import json
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r") as f:
                meta_data = json.load(f)
                reg.weights = meta_data.get("weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
                reg.meta_coef_ = meta_data.get("meta_coef")
                reg.meta_intercept_ = meta_data.get("meta_intercept")
        except Exception:
            reg.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            
    if os.environ.get("SPACE_ID"):
        return reg
        
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    lgb_reg = LGBMRegressor()
    lgb_reg._Booster = lgb.Booster(model_file=f"{prefix}_lgb.txt")
    lgb_reg.fitted_ = True
    lgb_reg._n_features = n_features
    lgb_reg._n_features_in = n_features
    lgb_reg.n_features_in_ = n_features
    
    cat = CatBoostRegressor()
    cat.load_model(f"{prefix}_cat.json", format="json")
    
    reg.lgb_model = lgb_reg
    reg.cat_model = cat
    return reg

