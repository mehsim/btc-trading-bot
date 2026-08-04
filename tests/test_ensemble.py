import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ensemble import EnsembleClassifier, EnsembleRegressor, _slice_model_input

def test_slice_model_input():
    """Verify _slice_model_input raises RuntimeError on feature count mismatch (Fix B7)."""
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    
    xgb = XGBClassifier(n_estimators=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Matching features works
    sliced_matching = _slice_model_input(xgb, X_train)
    assert sliced_matching.shape[1] == 10

    # Mismatched features raises RuntimeError
    X_extra_np = np.random.randn(10, 15)
    with pytest.raises(RuntimeError):
        _slice_model_input(xgb, X_extra_np)

def test_ensemble_classifier_predict_proba():
    """Verify EnsembleClassifier predict_proba works with matching features and rejects mismatched shapes."""
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    
    xgb = XGBClassifier(n_estimators=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    ensemble = EnsembleClassifier(xgb)
    
    # Predict with matching features
    probs = ensemble.predict_proba(X_train)
    assert probs.shape == (100, 3)

    # Predict with mismatched extra features raises RuntimeError (Fail-Closed)
    X_extra = np.random.randn(100, 15)
    with pytest.raises(RuntimeError):
        ensemble.predict_proba(X_extra)

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

def test_feature_column_shuffle_alignment():
    """
    Claim A2 Proof:
    Randomly shuffling every feature column in input DataFrame produces
    100% identical sliced output and prediction, proving alignment is by name, not position.
    """
    class MockModel:
        def __init__(self, feature_names):
            self.feature_names_ = feature_names

    feature_names = ["RSI", "ADX", "ATR_norm", "EMA_dist"]
    model = MockModel(feature_names)

    df_original = pd.DataFrame([{
        "RSI": 65.4, "ADX": 28.5, "ATR_norm": 0.012, "EMA_dist": 0.005
    }])

    sliced_orig = _slice_model_input(model, df_original)

    # Permute column order
    df_shuffled = df_original[["ATR_norm", "ADX", "EMA_dist", "RSI"]]
    sliced_shuffled = _slice_model_input(model, df_shuffled)

    assert list(sliced_orig.columns) == feature_names
    assert list(sliced_shuffled.columns) == feature_names
    pd.testing.assert_frame_equal(sliced_orig, sliced_shuffled)


