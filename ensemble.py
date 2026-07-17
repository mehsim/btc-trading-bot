import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from xgboost import XGBClassifier, XGBRegressor

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
                sum_w = sum(raw_weights)
                self.weights = [w / sum_w for w in raw_weights]
                
                # Extract validation probabilities from each base model
                p_xgb = self.xgb_model.predict_proba(X_v_arr)
                p_lgb = self.lgb_model.predict_proba(X_v_arr)
                p_cat = self.cat_model.predict_proba(X_v_arr)
                
                # Stack features for meta-learner (multi-class probabilities stacked column-wise)
                X_meta = np.column_stack([p_xgb, p_lgb, p_cat])
                
                # Apply exponential time-decay sample weighting (newer validation samples weighted higher)
                N_val = len(y_val)
                val_decay_weights = np.exp(-0.02 * np.arange(N_val)[::-1])
                val_decay_weights = val_decay_weights / np.sum(val_decay_weights) * N_val
                
                meta_clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
                meta_clf.fit(X_meta, y_val, sample_weight=val_decay_weights)
                self.meta_coef_ = meta_clf.coef_.tolist()
                self.meta_intercept_ = meta_clf.intercept_.tolist()
                print(f"[Ensemble Stacking] Classifier Meta-Learner calibrated successfully with time-decay weights (leak-free).")
            except Exception as e:
                print(f"[Ensemble Stacking Warning] Stacking calibration failed, using weighted average fallback: {e}")
                
        # Now refit base models on the ENTIRE dataset for live trading
        self.xgb_model.fit(X_arr, y, sample_weight=sample_weight)
        if self.lgb_model is not None:
            self.lgb_model.fit(X_arr, y, sample_weight=sample_weight)
        if self.cat_model is not None:
            self.cat_model.fit(X_arr, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X, weights=None):
        X_arr = np.asarray(X, dtype=float)
        xgb_prob = self.xgb_model.predict_proba(X_arr)
        if self.lgb_model is None or self.cat_model is None or self.meta_coef_ is None:
            # Fallback to performance-weighted blending if base models or meta-coefficients are missing
            w_to_use = weights if weights is not None else getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            w = np.array(w_to_use, dtype=float)
            w = w / np.sum(w)
            if self.lgb_model is None or self.cat_model is None:
                return xgb_prob
            lgb_prob = self.lgb_model.predict_proba(X_arr)
            cat_prob = self.cat_model.predict_proba(X_arr)
            return (xgb_prob * w[0] + lgb_prob * w[1] + cat_prob * w[2])
            
        lgb_prob = self.lgb_model.predict_proba(X_arr)
        cat_prob = self.cat_model.predict_proba(X_arr)
        
        # Stack probabilities for prediction
        X_meta = np.column_stack([xgb_prob, lgb_prob, cat_prob])
        
        from sklearn.linear_model import LogisticRegression
        meta_clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
        coef = np.array(self.meta_coef_)
        intercept = np.array(self.meta_intercept_)
        
        if coef.ndim == 1 or coef.shape[0] == 1:
            meta_clf.coef_ = coef.reshape(1, -1)
            meta_clf.intercept_ = intercept.reshape(-1)
            meta_clf.classes_ = np.array([0, 2])
            try:
                binary_probs = meta_clf.predict_proba(X_meta)
                three_probs = np.zeros((binary_probs.shape[0], 3))
                three_probs[:, 0] = binary_probs[:, 0]
                three_probs[:, 2] = binary_probs[:, 1]
                return three_probs
            except Exception:
                w = np.array(self.weights, dtype=float)
                w = w / np.sum(w)
                return (xgb_prob * w[0] + lgb_prob * w[1] + cat_prob * w[2])
        else:
            meta_clf.coef_ = coef
            meta_clf.intercept_ = intercept
            meta_clf.classes_ = np.array([0, 1, 2])
            return meta_clf.predict_proba(X_meta)

    def predict(self, X, weights=None):
        probs = self.predict_proba(X, weights=weights)
        return np.argmax(probs, axis=1)

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
                sum_w = sum(raw_weights)
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
        X_arr = np.asarray(X, dtype=float)
        xgb_pred = self.xgb_model.predict(X_arr)
        if self.lgb_model is None or self.cat_model is None or self.meta_coef_ is None:
            # Fallback to performance-weighted average if any base models or meta-coefficients are missing
            w_to_use = weights if weights is not None else getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            w = np.array(w_to_use, dtype=float)
            w = w / np.sum(w)
            if self.lgb_model is None or self.cat_model is None:
                return xgb_pred
            lgb_pred = self.lgb_model.predict(X_arr)
            cat_pred = self.cat_model.predict(X_arr)
            return (xgb_pred * w[0] + lgb_pred * w[1] + cat_pred * w[2])
            
        lgb_pred = self.lgb_model.predict(X_arr)
        cat_pred = self.cat_model.predict(X_arr)
        
        # Apply Ridge meta-model
        X_meta = np.column_stack([xgb_pred, lgb_pred, cat_pred])
        
        from sklearn.linear_model import Ridge
        meta_reg = Ridge(alpha=1.0, random_state=42)
        meta_reg.coef_ = np.array(self.meta_coef_)
        meta_reg.intercept_ = float(self.meta_intercept_)
        return meta_reg.predict(X_meta)

# ==========================================
# NATIVE SAVING/LOADING (TEXT/JSON ONLY)
# ==========================================

def save_ensemble_classifier(model, prefix):
    import json
    model.xgb_model.save_model(f"{prefix}_xgb.json")
    model.lgb_model.booster_.save_model(f"{prefix}_lgb.txt")
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
    lgb_clf._Booster = lgb.Booster(model_file=f"{prefix}_lgb.txt")
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

def save_ensemble_regressor(model, prefix):
    import json
    model.xgb_model.save_model(f"{prefix}_xgb.json")
    model.lgb_model.booster_.save_model(f"{prefix}_lgb.txt")
    model.cat_model.save_model(f"{prefix}_cat.json", format="json")
    
    meta_data = {
        "weights": getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0]),
        "meta_coef": getattr(model, "meta_coef_", None),
        "meta_intercept": getattr(model, "meta_intercept_", None)
    }
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(meta_data, f)

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

