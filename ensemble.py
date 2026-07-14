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
    Blends XGBoost, LightGBM, and CatBoost classifiers using performance-weighted probability voting.
    """
    def __init__(self, xgb_model, lgb_model=None, cat_model=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        X_arr = np.asarray(X, dtype=float)
        self.xgb_model.fit(X_arr, y, sample_weight=sample_weight)
        if self.lgb_model is not None:
            self.lgb_model.fit(X_arr, y, sample_weight=sample_weight)
        if self.cat_model is not None:
            self.cat_model.fit(X_arr, y, sample_weight=sample_weight)
            
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        if X_val is not None and y_val is not None and self.lgb_model is not None and self.cat_model is not None:
            try:
                from sklearn.metrics import accuracy_score
                X_v_arr = np.asarray(X_val, dtype=float)
                xgb_acc = accuracy_score(y_val, self.xgb_model.predict(X_v_arr))
                lgb_acc = accuracy_score(y_val, self.lgb_model.predict(X_v_arr))
                cat_acc = accuracy_score(y_val, self.cat_model.predict(X_v_arr))
                
                raw_weights = [max(0.01, xgb_acc), max(0.01, lgb_acc), max(0.01, cat_acc)]
                sum_w = sum(raw_weights)
                self.weights = [w / sum_w for w in raw_weights]
                print(f"[Ensemble Weighting] Classifier Weights calibrated: XGB={self.weights[0]:.3f}, LGB={self.weights[1]:.3f}, CAT={self.weights[2]:.3f}")
            except Exception as e:
                print(f"[Ensemble Weighting Warning] Failed to calibrate weights: {e}")
        return self

    def predict_proba(self, X, weights=None):
        X_arr = np.asarray(X, dtype=float)
        xgb_prob = self.xgb_model.predict_proba(X_arr)
        if self.lgb_model is None or self.cat_model is None:
            return xgb_prob
        lgb_prob = self.lgb_model.predict_proba(X_arr)
        cat_prob = self.cat_model.predict_proba(X_arr)
        
        w_to_use = weights
        if w_to_use is None:
            w_to_use = getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            
        w = np.array(w_to_use, dtype=float)
        w = w / np.sum(w)
        return (xgb_prob * w[0] + lgb_prob * w[1] + cat_prob * w[2])

    def predict(self, X, weights=None):
        probs = self.predict_proba(X, weights=weights)
        return np.argmax(probs, axis=1)

class EnsembleRegressor:
    """
    Blends XGBoost, LightGBM, and CatBoost regressors using performance-weighted averaging.
    """
    def __init__(self, xgb_model, lgb_model=None, cat_model=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]

    def fit(self, X, y, X_val=None, y_val=None):
        X_arr = np.asarray(X, dtype=float)
        self.xgb_model.fit(X_arr, y)
        if self.lgb_model is not None:
            self.lgb_model.fit(X_arr, y)
        if self.cat_model is not None:
            self.cat_model.fit(X_arr, y)
            
        self.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        if X_val is not None and y_val is not None and self.lgb_model is not None and self.cat_model is not None:
            try:
                from sklearn.metrics import mean_absolute_error
                X_v_arr = np.asarray(X_val, dtype=float)
                xgb_mae = mean_absolute_error(y_val, self.xgb_model.predict(X_v_arr))
                lgb_mae = mean_absolute_error(y_val, self.lgb_model.predict(X_v_arr))
                cat_mae = mean_absolute_error(y_val, self.cat_model.predict(X_v_arr))
                
                raw_weights = [1.0 / max(1e-6, xgb_mae), 1.0 / max(1e-6, lgb_mae), 1.0 / max(1e-6, cat_mae)]
                sum_w = sum(raw_weights)
                self.weights = [w / sum_w for w in raw_weights]
                print(f"[Ensemble Weighting] Regressor Weights calibrated: XGB={self.weights[0]:.3f}, LGB={self.weights[1]:.3f}, CAT={self.weights[2]:.3f}")
            except Exception as e:
                print(f"[Ensemble Weighting Warning] Failed to calibrate weights: {e}")
        return self

    def predict(self, X, weights=None):
        X_arr = np.asarray(X, dtype=float)
        xgb_pred = self.xgb_model.predict(X_arr)
        if self.lgb_model is None or self.cat_model is None:
            return xgb_pred
        lgb_pred = self.lgb_model.predict(X_arr)
        cat_pred = self.cat_model.predict(X_arr)
        
        w_to_use = weights
        if w_to_use is None:
            w_to_use = getattr(self, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
            
        w = np.array(w_to_use, dtype=float)
        w = w / np.sum(w)
        return (xgb_pred * w[0] + lgb_pred * w[1] + cat_pred * w[2])

# ==========================================
# NATIVE SAVING/LOADING (TEXT/JSON ONLY)
# ==========================================

def save_ensemble_classifier(model, prefix):
    import json
    model.xgb_model.save_model(f"{prefix}_xgb.json")
    model.lgb_model.booster_.save_model(f"{prefix}_lgb.txt")
    model.cat_model.save_model(f"{prefix}_cat.json", format="json")
    
    weights = getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(weights, f)

def load_ensemble_classifier(prefix, n_features=54):
    xgb = XGBClassifier()
    xgb.load_model(f"{prefix}_xgb.json")
    
    clf = EnsembleClassifier(xgb, None, None)
    
    import json
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            clf.weights = json.load(f)
            
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
    
    weights = getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(weights, f)

def load_ensemble_regressor(prefix, n_features=54):
    xgb = XGBRegressor()
    xgb.load_model(f"{prefix}_xgb.json")
    
    reg = EnsembleRegressor(xgb, None, None)
    
    import json
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            reg.weights = json.load(f)
            
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

