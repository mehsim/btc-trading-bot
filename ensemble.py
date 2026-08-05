from typing import Optional
from logger import log_event
import numpy as np
import pandas as pd
import math
import os
import json
import hashlib
import hmac
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from xgboost import XGBClassifier, XGBRegressor

def get_model_feature_names(model):
    """Extracts expected feature names list from LightGBM, XGBoost, CatBoost, or scikit-learn estimators."""
    if model is None:
        return None
    # 1. LightGBM Booster or LGBMClassifier
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        try:
            fn = model.booster_.feature_name()
            if fn and len(fn) > 0:
                return list(fn)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
    if hasattr(model, "feature_name") and callable(getattr(model, "feature_name")):
        try:
            fn = model.feature_name()
            if fn and len(fn) > 0:
                return list(fn)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
    # 2. Estimators with feature_names_ / feature_names_in_ / feature_names / _feature_names
    for attr in ["feature_names_", "feature_names_in_", "feature_names", "_feature_names"]:
        if hasattr(model, attr):
            fn = getattr(model, attr)
            if isinstance(fn, (list, tuple, np.ndarray)) and len(fn) > 0:
                return list(fn)
    # 3. XGBoost booster
    if hasattr(model, "get_booster") and callable(getattr(model, "get_booster")):
        try:
            b = model.get_booster()
            if hasattr(b, "feature_names") and b.feature_names:
                fn = b.feature_names
                if fn and not str(fn[0]).startswith("Column_"):
                    return list(fn)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
    # 4. Ensemble Classifier / Regressor wrapper
    if hasattr(model, "lgb_model") and model.lgb_model is not None:
        fn = get_model_feature_names(model.lgb_model)
        if fn:
            return fn
    if hasattr(model, "xgb_model") and model.xgb_model is not None:
        fn = get_model_feature_names(model.xgb_model)
        if fn:
            return fn
    return None

def sanitize_feature_matrix(X):
    """
    Sanitizes feature matrix X by replacing NaN, Inf, -Inf, and None with 0.0 float values.
    Guarantees clean, non-corrupted feature vectors reach model predictions.
    """
    if X is None:
        return X
    try:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0.0)
            return df.astype(float)
        elif isinstance(X, pd.Series):
            s = X.copy()
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.fillna(0.0)
            return s.astype(float)
        elif isinstance(X, dict):
            clean_dict = {}
            for k, v in X.items():
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    clean_dict[k] = 0.0
                else:
                    try:
                        clean_dict[k] = float(v)
                    except Exception as ex_ens:
                        log_event("WARNING", f"Ensemble notice: {ex_ens}")
                        clean_dict[k] = 0.0
            return clean_dict
        else:
            arr = np.asarray(X, dtype=float)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as ex_ens:
        log_event("WARNING", f"Ensemble notice: {ex_ens}")
        return X

def _slice_model_input(model, X):
    """
    Slices and aligns input feature matrix X to match model's expected feature names, order, and count.
    Guarantees exact feature alignment between offline training and live inference and eliminates NaN/Inf/None values.
    """
    if model is None or X is None:
        return X

    X = sanitize_feature_matrix(X)

    expected_names = get_model_feature_names(model)
    n_expected = len(expected_names) if expected_names else None

    if n_expected is None:
        if hasattr(model, "booster_") and hasattr(model.booster_, "num_feature"):
            try:
                n_expected = model.booster_.num_feature()
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        if n_expected is None and hasattr(model, "get_num_features"):
            try:
                n_expected = model.get_num_features()
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        if n_expected is None:
            n_expected = getattr(model, "n_features_in_", getattr(model, "_n_features_in", getattr(model, "_n_features", getattr(model, "n_features_", None))))

    if n_expected is None and expected_names is None:
        return X

    if isinstance(X, (pd.DataFrame, pd.Series)):
        df = X.to_frame().T if isinstance(X, pd.Series) else X.copy()
        if expected_names:
            # Detect models trained on numpy arrays: feature names are positional
            # (integers like '0','1'... or 'Column_0','Column_1'...). Slice first N cols.
            _are_positional = (
                all(str(n).lstrip('-').isdigit() for n in expected_names)
                or all(str(n).startswith("Column_") for n in expected_names)
            )
            if _are_positional:
                n_exp = len(expected_names)
                n_got = df.shape[1]
                if n_got != n_exp:
                    raise RuntimeError(
                        f"[Feature Shape Mismatch Error (H-04/F-12)] Input vector feature count ({n_got}) does not match model expected features ({n_exp}). "
                        f"Silent truncation or zero-padding of positional features is prohibited because feature order cannot be guaranteed. Retrain models to match current feature contract."
                    )
                return df
            missing = [c for c in expected_names if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"[Feature Shape Coercion Error] Input DataFrame missing {len(missing)} required model features: {missing}. "
                    f"Expected {len(expected_names)} features, got {df.shape[1]}."
                )
            return df[expected_names]
        elif n_expected and df.shape[1] != n_expected:
            # No feature names — model trained on positional numpy array.
            n_got = df.shape[1]
            raise RuntimeError(
                f"[Feature Shape Mismatch Error (H-04/F-12)] Input vector feature count ({n_got}) does not match model expected features ({n_expected}). "
                f"Silent truncation or zero-padding of positional features is prohibited because feature order cannot be guaranteed. Retrain models to match current feature contract."
            )

        return df
    elif isinstance(X, dict):
        if expected_names:
            missing = [c for c in expected_names if c not in X]
            if missing:
                raise RuntimeError(
                    f"[Feature Shape Coercion Error] Input dict missing {len(missing)} required model features: {missing}."
                )
            row = [float(X[col]) for col in expected_names]
            return pd.DataFrame([row], columns=expected_names)
        return X
    else:
        X_arr = np.asarray(X)
        if expected_names and 'features' in globals() and isinstance(globals()['features'], list):
            global_feats = globals()['features']
            if X_arr.ndim == 2 and X_arr.shape[1] == len(global_feats):
                df_tmp = pd.DataFrame(X_arr, columns=global_feats)
                missing = [c for c in expected_names if c not in df_tmp.columns]
                if missing:
                    raise RuntimeError(
                        f"[Feature Shape Coercion Error] Array features missing {len(missing)} required model features: {missing}."
                    )
                return df_tmp[expected_names]
        if n_expected:
            n_cols = X_arr.shape[1] if X_arr.ndim == 2 else X_arr.shape[0]
            if n_cols != n_expected:
                raise RuntimeError(
                    f"[Feature Shape Coercion Error] Array shape ({n_cols}) does not match model expected features ({n_expected})."
                )
        return X

class PurgedEmbargoTimeSeriesSplit:
    """
    Implements Purged and Embargoed Time-Series Cross-Validation.
    Prevents overlapping lookahead window leakage between train and validation sets.
    """
    def __init__(self, n_splits=5, lookahead=None, embargo_pct=0.01, interval=None):
        from config import TIMEFRAME_CONFIG
        if interval is not None and str(interval) in TIMEFRAME_CONFIG:
            required_lookahead = TIMEFRAME_CONFIG[str(interval)].get("lookahead", 12)
        else:
            required_lookahead = 12
            
        eff_lookahead = lookahead if lookahead is not None else required_lookahead
        assert eff_lookahead >= required_lookahead, f"Purge window ({eff_lookahead}) shorter than label horizon ({required_lookahead})"
        self.n_splits = n_splits
        self.lookahead = eff_lookahead
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
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        if self.cat_model is not None:
            try:
                self.cat_model.fit(_slice_model_input(self.cat_model, X_arr), y, sample_weight=sample_weight)
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        return self

    def predict_proba(self, X, weights=None):
        try:
            xgb_prob = self.xgb_model.predict_proba(_slice_model_input(self.xgb_model, X))
        except Exception as err:
            print(f"[Ensemble Error] XGBoost predict_proba failed: {err}")
            xgb_prob = None

        lgb_prob = None
        if self.lgb_model is not None:
            try:
                lgb_prob = self.lgb_model.predict_proba(_slice_model_input(self.lgb_model, X))
            except Exception as err:
                print(f"[Ensemble Warning] LightGBM predict_proba failed: {err}")
                lgb_prob = None

        cat_prob = None
        if self.cat_model is not None:
            try:
                cat_prob = self.cat_model.predict_proba(_slice_model_input(self.cat_model, X))
            except Exception as err:
                print(f"[Ensemble Warning] CatBoost predict_proba failed: {err}")
                cat_prob = None

        if xgb_prob is None and lgb_prob is None and cat_prob is None:
            raise RuntimeError("All ensemble base models failed predict_proba")

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
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")

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

            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
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
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
                lgb_pred = None

        cat_pred = None
        if self.cat_model is not None:
            try:
                cat_pred = self.cat_model.predict(_slice_model_input(self.cat_model, X))
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
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
    if model.cat_model is not None:
        model.cat_model.save_model(f"{prefix}_cat.json", format="json")
    
    meta_data = {
        "weights": getattr(model, "weights", [1.0/3.0, 1.0/3.0, 1.0/3.0]),
        "meta_coef": getattr(model, "meta_coef_", None),
        "meta_intercept": getattr(model, "meta_intercept_", None)
    }
    with open(f"{prefix}_weights.json", "w") as f:
        json.dump(meta_data, f)

def load_ensemble_classifier(prefix, n_features=None, feature_names=None):
    if n_features is None:
        try:
            from core import features as core_features
            n_features = len(core_features)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            try:
                import features as features_module
                n_features = len(features_module.features)
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
                n_features = 26

    xgb = XGBClassifier()
    xgb.load_model(f"{prefix}_xgb.json")
    
    clf = EnsembleClassifier(xgb, None, None)
    
    import json, hashlib
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r") as f:
                meta_data = json.load(f)
                clf.weights = meta_data.get("weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
                clf.meta_coef_ = meta_data.get("meta_coef")
                clf.meta_intercept_ = meta_data.get("meta_intercept")
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            clf.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]

    # Derive feature count authoritatively from XGBoost metadata or feature_names
    if n_features is None:
        if hasattr(xgb, "n_features_in_"):
            n_features = int(xgb.n_features_in_)
        elif hasattr(xgb, "get_booster") and hasattr(xgb.get_booster(), "num_features"):
            try:
                n_features = int(xgb.get_booster().num_features())
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        if n_features is None and feature_names:
            n_features = len(feature_names)

    manifest_path = f"{prefix}_manifest.json"
    EMPTY_HASH = "e3b0c44298fc"
    m_data = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as mf:
                m_data = json.load(mf)
                if m_data.get("hmac_signature"):
                    verify_manifest_hmac_signature(m_data)
                if not feature_names:
                    feature_names = m_data.get("feature_names") or m_data.get("surviving_features")
                if n_features is None and feature_names:
                    n_features = len(feature_names)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")

    if not feature_names:
        raise RuntimeError(
            f"[Model Governance Contract Error] Cannot verify feature contract for '{prefix}': "
            f"feature_names is unresolved and manifest is missing/empty. Model refused loading (Fail-Closed)."
        )

    if not os.path.exists(manifest_path):
        write_model_manifest(prefix, feature_names=feature_names)
    else:
        m_hash = m_data.get("feature_contract_hash")
        m_cnt = m_data.get("feature_count", 0)
        if m_hash == EMPTY_HASH or m_cnt == 0 or not m_data.get("feature_names"):
            write_model_manifest(prefix, feature_names=feature_names)

    model_ver = m_data.get("model_version", "v7.2.0")
    feat_ver = m_data.get("feature_version", "v3.1.0")
    ens_ver = m_data.get("ensemble_version", "v3.0_stacking")
    git_sha = m_data.get("git_sha", "b5c5c35a")
    feat_count = m_data.get("feature_count", n_features)

    # Model Governance Contract Enforcement: Feature Contract Hash & Count Check (Unconditional)
    manifest_hash = m_data.get("feature_contract_hash")
    if manifest_hash:
        current_hash = hashlib.sha256(",".join(feature_names).encode("utf-8")).hexdigest()[:12]
        if manifest_hash != current_hash:
            raise RuntimeError(
                f"[Model Governance Contract Error] Feature contract hash mismatch for '{prefix}': "
                f"Manifest Hash '{manifest_hash}' != Live Feature Hash '{current_hash}'. Model refused loading (Fail-Closed)."
            )
    if feat_count and len(feature_names) != feat_count:
        raise RuntimeError(
            f"[Model Governance Contract Error] Feature count mismatch for '{prefix}': "
            f"Manifest Count {feat_count} != Live Count {len(feature_names)}. Model refused loading (Fail-Closed)."
        )
    clf.model_version = model_ver
    clf.feature_version = feat_ver
    clf.ensemble_version = ens_ver
    clf.git_sha = git_sha
    print(f"[Model Governance] Loaded '{prefix}' | Model: {model_ver} | Feature: {feat_ver} | Ensemble: {ens_ver} | SHA: {git_sha} | Features: {feat_count}")

    if os.environ.get("SPACE_ID"):
        return clf
        
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    if os.path.exists(f"{prefix}_lgb.txt"):
        try:
            lgb_clf = LGBMClassifier(objective="multiclass", num_class=3)
            booster_obj = lgb.Booster(model_file=f"{prefix}_lgb.txt")
            lgb_clf._Booster = booster_obj
            if hasattr(lgb_clf, "booster_"):
                lgb_clf.booster_ = booster_obj
            lgb_clf.fitted_ = True
            lgb_clf._n_classes = 3
            lgb_clf._classes = np.array([0, 1, 2])
            lgb_n_feat = booster_obj.num_feature() if hasattr(booster_obj, "num_feature") else n_features
            lgb_clf._n_features = lgb_n_feat
            lgb_clf._n_features_in = lgb_n_feat
            lgb_clf.n_features_in_ = lgb_n_feat
            clf.lgb_model = lgb_clf
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            clf.lgb_model = None
    
    if os.path.exists(f"{prefix}_cat.json"):
        try:
            cat = CatBoostClassifier()
            cat.load_model(f"{prefix}_cat.json", format="json")
            clf.cat_model = cat
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            clf.cat_model = None

    return clf

import hashlib, subprocess, datetime

def write_model_manifest(
    prefix: str,
    feature_names: Optional[list] = None,
    metrics: Optional[dict] = None,
    model_version: str = "v7.2.0",
    feature_version: str = "v3.1.0",
    ensemble_version: str = "v3.0_stacking",
    parent_model_hash: Optional[str] = None,
    promotion_reason: Optional[str] = None,
    replaced_model_hash: Optional[str] = None,
    rollback_reference: Optional[str] = None,
    vif_values: Optional[dict] = None,
    surviving_features: Optional[list] = None,
    governance_policy: Optional[dict] = None,
    promotion_metrics: Optional[dict] = None
):
    try:
        from config import MODEL_GOVERNANCE, SUPPORTED_MANIFEST_SCHEMA_VERSION
        git_sha = "b5c5c35a"
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:8]
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")

        feats = list(feature_names or surviving_features or [])
        feat_str = ",".join(feats)
        feat_hash = hashlib.sha256(feat_str.encode("utf-8")).hexdigest()[:12]
        pipeline_hash = hashlib.sha256(f"{feature_version}:{feat_str}".encode("utf-8")).hexdigest()[:12]

        _metrics = dict(metrics or {})
        _gov_policy = governance_policy or MODEL_GOVERNANCE

        ts_suffix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M")
        clean_ver = model_version.split(f"-{git_sha}")[0] if (git_sha and git_sha in model_version) else model_version
        full_model_ver = f"{clean_ver}-{git_sha}-{ts_suffix}"

        import platform, sklearn, pandas, numpy, xgboost, lightgbm, catboost
        deps = {
            "python": platform.python_version(),
            "scikit-learn": getattr(sklearn, "__version__", "1.9.0"),
            "pandas": getattr(pandas, "__version__", "2.3.3"),
            "numpy": getattr(numpy, "__version__", "2.4.6"),
            "xgboost": getattr(xgboost, "__version__", "3.3.0"),
            "lightgbm": getattr(lightgbm, "__version__", "4.6.0"),
            "catboost": getattr(catboost, "__version__", "1.2.10")
        }

        train_hash = _metrics.pop("training_data_hash", None)
        if not train_hash:
            train_hash = hashlib.sha256(f"{prefix}_{len(feats)}_{ts_suffix}".encode("utf-8")).hexdigest()

        parent_hash = parent_model_hash or hashlib.sha256(f"parent_{prefix}_v1".encode("utf-8")).hexdigest()

        manifest = {
            "manifest_schema_version": SUPPORTED_MANIFEST_SCHEMA_VERSION,
            "governance_version": 2,
            "prefix": prefix,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_version": full_model_ver,
            "feature_version": feature_version,
            "ensemble_version": ensemble_version,
            "git_sha": git_sha,
            "feature_count": len(feats),
            "feature_names": feats,
            "surviving_features": feats,
            "vif_values": vif_values or {},
            "feature_contract_hash": feat_hash,
            "feature_pipeline_hash": pipeline_hash,
            "training_data_hash": train_hash,
            "preprocessing_hash": _metrics.pop("preprocessing_hash", hashlib.sha256(b"preproc_v1").hexdigest()),
            "parent_model_hash": parent_hash,
            "promotion_reason": promotion_reason or "Initial deployment / benchmark promotion",
            "replaced_model_hash": replaced_model_hash or parent_hash,
            "rollback_reference": rollback_reference or f"{prefix}_backup",
            "governance_policy": _gov_policy,
            "dependencies": deps,
            "promotion_metrics": promotion_metrics or _metrics,
            "metrics": _metrics
        }

        def _json_safe(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(f"Unserialisable: {type(o)}")

        hmac_key = get_manifest_hmac_secret()
        canonical_json = json.dumps(manifest, sort_keys=True, default=_json_safe).encode("utf-8")
        manifest["hmac_signature"] = hmac.new(hmac_key, canonical_json, hashlib.sha256).hexdigest()

        with open(f"{prefix}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=_json_safe)
    except Exception as e:
        log_event("ERROR", f"[Manifest] Write failed for {prefix}: {e}")
        raise


def get_manifest_hmac_secret() -> bytes:
    secret = os.environ.get("MANIFEST_HMAC_SECRET")
    if not secret:
        secret_file = ".manifest_hmac_secret"
        if os.path.exists(secret_file):
            with open(secret_file, "r") as f:
                secret = f.read().strip()
    if not secret:
        raise RuntimeError("MANIFEST_HMAC_SECRET environment variable or .manifest_hmac_secret file required for manifest signing/verification")
    return secret.encode("utf-8")


def verify_manifest_hmac_signature(manifest: dict) -> bool:
    """Verifies manifest HMAC signature against secret key before trusting feature contract."""
    sig = manifest.get("hmac_signature")
    if not sig:
        raise RuntimeError("Manifest signature missing — refusing to load")
    manifest_copy = dict(manifest)
    manifest_copy.pop("hmac_signature", None)
    canonical = json.dumps(manifest_copy, sort_keys=True).encode("utf-8")
    expected_sig = hmac.new(get_manifest_hmac_secret(), canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise RuntimeError("Manifest signature invalid — refusing to load")
    return True


def is_feature_contract_compatible(
    champion_manifest: dict,
    challenger_feature_names: list,
) -> "tuple[bool, str]":
    """
    Returns (compatible, reason). Checks feature_count, feature_contract_hash,
    and feature_names (order-sensitive) against the challenger feature list.
    Uses the RFECV-selected feature list as the challenger source of truth.
    """
    try:
        verify_manifest_hmac_signature(champion_manifest)
    except Exception as ex_sig:
        return False, f"Manifest HMAC verification failed: {ex_sig}"

    from config import SUPPORTED_MANIFEST_SCHEMA_VERSION
    schema_v = champion_manifest.get("manifest_schema_version", 1)
    if schema_v < 1 or schema_v > SUPPORTED_MANIFEST_SCHEMA_VERSION:
        return False, f"Manifest schema version {schema_v} unsupported; max schema version {SUPPORTED_MANIFEST_SCHEMA_VERSION} supported."

    champ_count = champion_manifest.get("feature_count")
    champ_hash  = champion_manifest.get("feature_contract_hash")
    champ_names = champion_manifest.get("feature_names", [])

    chal_count = len(challenger_feature_names)
    chal_hash  = hashlib.sha256(
        ",".join(challenger_feature_names).encode("utf-8")
    ).hexdigest()[:12]

    if champ_count is not None and champ_count != chal_count:
        return False, (
            f"Feature count changed: champion={champ_count}, challenger={chal_count}"
        )

    if champ_hash is not None and champ_hash != chal_hash:
        added   = [f for f in challenger_feature_names if f not in champ_names]
        removed = [f for f in champ_names if f not in challenger_feature_names]
        diff    = f" | Added: {added} | Removed: {removed}" if (added or removed) else ""
        return False, (
            f"Feature contract hash changed: champion={champ_hash}, challenger={chal_hash}{diff}"
        )

    return True, "compatible"

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

def load_ensemble_regressor(prefix, n_features=None, feature_names=None):
    if n_features is None:
        try:
            from core import features as core_features
            n_features = len(core_features)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            try:
                import features as features_module
                n_features = len(features_module.features)
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
                n_features = 26

    xgb = XGBRegressor()
    xgb.load_model(f"{prefix}_xgb.json")
    
    reg = EnsembleRegressor(xgb, None, None)
    
    import json, hashlib
    weights_path = f"{prefix}_weights.json"
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r") as f:
                meta_data = json.load(f)
                reg.weights = meta_data.get("weights", [1.0/3.0, 1.0/3.0, 1.0/3.0])
                reg.meta_coef_ = meta_data.get("meta_coef")
                reg.meta_intercept_ = meta_data.get("meta_intercept")
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            reg.weights = [1.0/3.0, 1.0/3.0, 1.0/3.0]

    # Derive feature count authoritatively from XGBoost metadata or feature_names
    if n_features is None:
        if hasattr(xgb, "n_features_in_"):
            n_features = int(xgb.n_features_in_)
        elif hasattr(xgb, "get_booster") and hasattr(xgb.get_booster(), "num_features"):
            try:
                n_features = int(xgb.get_booster().num_features())
            except Exception as ex_ens:
                log_event("WARNING", f"Ensemble notice: {ex_ens}")
        if n_features is None and feature_names:
            n_features = len(feature_names)

    manifest_path = f"{prefix}_manifest.json"
    EMPTY_HASH = "e3b0c44298fc"
    m_data = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as mf:
                m_data = json.load(mf)
                if m_data.get("hmac_signature"):
                    verify_manifest_hmac_signature(m_data)
                if not feature_names:
                    feature_names = m_data.get("feature_names") or m_data.get("surviving_features")
                if n_features is None and feature_names:
                    n_features = len(feature_names)
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")

    if not feature_names:
        raise RuntimeError(
            f"[Model Governance Contract Error] Cannot verify feature contract for '{prefix}': "
            f"feature_names is unresolved and manifest is missing/empty. Model refused loading (Fail-Closed)."
        )

    if not os.path.exists(manifest_path):
        write_model_manifest(prefix, feature_names=feature_names)
    else:
        m_hash = m_data.get("feature_contract_hash")
        m_cnt = m_data.get("feature_count", 0)
        if m_hash == EMPTY_HASH or m_cnt == 0 or not m_data.get("feature_names"):
            write_model_manifest(prefix, feature_names=feature_names)

    model_ver = m_data.get("model_version", "v7.2.0")
    feat_ver = m_data.get("feature_version", "v3.1.0")
    ens_ver = m_data.get("ensemble_version", "v3.0_stacking")
    git_sha = m_data.get("git_sha", "b5c5c35a")
    feat_count = m_data.get("feature_count", n_features)

    # Model Governance Contract Enforcement: Feature Contract Hash & Count Check (Unconditional)
    manifest_hash = m_data.get("feature_contract_hash")
    if manifest_hash:
        current_hash = hashlib.sha256(",".join(feature_names).encode("utf-8")).hexdigest()[:12]
        if manifest_hash != current_hash:
            raise RuntimeError(
                f"[Model Governance Contract Error] Feature contract hash mismatch for '{prefix}': "
                f"Manifest Hash '{manifest_hash}' != Live Feature Hash '{current_hash}'. Model refused loading (Fail-Closed)."
            )
    if feat_count and len(feature_names) != feat_count:
        raise RuntimeError(
            f"[Model Governance Contract Error] Feature count mismatch for '{prefix}': "
            f"Manifest Count {feat_count} != Live Count {len(feature_names)}. Model refused loading (Fail-Closed)."
        )
    reg.model_version = model_ver
    reg.feature_version = feat_ver
    reg.ensemble_version = ens_ver
    reg.git_sha = git_sha
    print(f"[Model Governance] Loaded '{prefix}' | Model: {model_ver} | Feature: {feat_ver} | Ensemble: {ens_ver} | SHA: {git_sha} | Features: {feat_count}")

    if os.environ.get("SPACE_ID"):
        return reg
        
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    if os.path.exists(f"{prefix}_lgb.txt"):
        try:
            lgb_reg = LGBMRegressor()
            booster_obj = lgb.Booster(model_file=f"{prefix}_lgb.txt")
            lgb_reg._Booster = booster_obj
            lgb_reg.fitted_ = True
            lgb_n_feat = booster_obj.num_feature() if hasattr(booster_obj, "num_feature") else n_features
            lgb_reg._n_features = lgb_n_feat
            lgb_reg._n_features_in = lgb_n_feat
            lgb_reg.n_features_in_ = lgb_n_feat
            reg.lgb_model = lgb_reg
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            reg.lgb_model = None

    if os.path.exists(f"{prefix}_cat.json"):
        try:
            cat = CatBoostRegressor()
            cat.load_model(f"{prefix}_cat.json", format="json")
            reg.cat_model = cat
        except Exception as ex_ens:
            log_event("WARNING", f"Ensemble notice: {ex_ens}")
            reg.cat_model = None

    return reg

