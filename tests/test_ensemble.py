import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ensemble import EnsembleClassifier, EnsembleRegressor, _slice_model_input

def test_slice_model_input():
    """Verify _slice_model_input correctly trims features based on n_features_in_."""
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    
    xgb = XGBClassifier(n_estimators=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Pass input with extra features (15 columns instead of 10)
    X_extra_np = np.random.randn(10, 15)
    sliced_np = _slice_model_input(xgb, X_extra_np)
    assert sliced_np.shape[1] == 10

    # Pass DataFrame with extra features
    X_extra_df = pd.DataFrame(X_extra_np)
    sliced_df = _slice_model_input(xgb, X_extra_df)
    assert sliced_df.shape[1] == 10

def test_ensemble_classifier_predict_proba():
    """Verify EnsembleClassifier predict_proba works with _slice_model_input."""
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    
    xgb = XGBClassifier(n_estimators=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    ensemble = EnsembleClassifier(xgb)
    
    # Predict with matching features
    probs = ensemble.predict_proba(X_train)
    assert probs.shape == (100, 3)

    # Predict with extra features
    X_extra = np.random.randn(100, 15)
    probs_extra = ensemble.predict_proba(X_extra)
    assert probs_extra.shape == (100, 3)

def test_meta_classifier_slicing():
    """Verify meta-classifier prediction works when input array has extra or full feature counts."""
    X_train = np.random.randn(100, 70)
    y_train = np.random.randint(0, 2, 100)
    
    meta_clf = XGBClassifier(n_estimators=5, random_state=42)
    meta_clf.fit(X_train, y_train)
    
    # Simulate full features input vs trimmed
    X_full = np.random.randn(1, 70)
    X_input = _slice_model_input(meta_clf, X_full)
    pred = meta_clf.predict(X_input)
    assert len(pred) == 1

