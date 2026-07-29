"""
tests/test_ml_upgrades.py
-------------------------
Unit tests covering Machine Learning Profitability & Accuracy Upgrades:
- Dynamic Ensemble Weighting via Bayesian Dirichlet optimization
- HMM Regime Detection
- Purged K-Fold Cross Validation with Embargo
- Online Feature Selection
- MC Dropout Uncertainty Quantification
"""

import pytest
import numpy as np
import pandas as pd
from dynamic_ensemble_optimizer import dynamic_ensemble_optimizer
from hmm_regime_detector import hmm_regime_detector
from purged_kfold_cv import purged_cv
from online_feature_selector import online_feature_selector
from mc_dropout_quantifier import mc_dropout_quantifier


def test_dynamic_ensemble_weights():
    perfs = {"xgb": 0.80, "lgb": 0.60, "cat": 0.50}
    weights = dynamic_ensemble_optimizer.optimize_weights(perfs)
    assert len(weights) == 3
    assert weights[0] > weights[2]  # XGBoost gets higher weight due to superior score
    assert sum(weights) == pytest.approx(1.0)


def test_hmm_regime_detection():
    df = pd.DataFrame({
        "close": np.linspace(100, 110, 30),
        "ADX": [25.0] * 30,
        "ATR_norm": [0.01] * 30
    })
    res = hmm_regime_detector.detect_regime(df)
    assert "regime" in res
    assert res["state_id"] in [0, 1, 2, 3]


def test_purged_kfold_split():
    df = pd.DataFrame({"feat": np.arange(100)})
    splits = list(purged_cv.split(df))
    assert len(splits) == 5
    train_idx, test_idx = splits[0]
    assert len(train_idx) > 0
    assert len(test_idx) > 0


def test_online_feature_selection():
    feats = [f"f_{i}" for i in range(10)]
    imps = np.array([0.01, 0.50, 0.20, 0.05, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05])
    selected = online_feature_selector.select_top_features(feats, imps)
    assert selected[0] == "f_1"  # Highest importance selected first


def test_mc_dropout_uncertainty():
    probs = [
        [0.80, 0.10, 0.10],
        [0.78, 0.12, 0.10],
        [0.82, 0.08, 0.10]
    ]
    mean_conf, var_unc, is_unc = mc_dropout_quantifier.quantify_uncertainty(probs)
    assert mean_conf > 0.75
    assert is_unc is False
