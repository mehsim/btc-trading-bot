"""
tests/test_background_ml_tester.py
-----------------------------------
Unit test suite covering BackgroundMLTester:
- Shadow model Challenger vs Champion paper evaluation
- Adversarial noise injection stress testing
- Feature importance predictive decay auditing
"""

import pytest
import numpy as np
from background_ml_tester import background_ml_tester


def test_shadow_paper_evaluation():
    champ_probs = np.array([0.60, 0.40, 0.70, 0.30])
    chall_probs = np.array([0.65, 0.35, 0.80, 0.20])
    actual_labels = np.array([1, 0, 1, 0])

    res = background_ml_tester.run_shadow_paper_evaluation(champ_probs, chall_probs, actual_labels)
    assert "champion_accuracy" in res
    assert "challenger_accuracy" in res
    assert res["champion_accuracy"] == 1.0
    assert res["challenger_accuracy"] == 1.0


def test_adversarial_stress_test():
    X = np.random.randn(50, 10)
    res = background_ml_tester.run_adversarial_stress_test(X, noise_std=0.01)
    assert res["status"] == "PASS"
    assert res["is_stable"] is True


def test_feature_importance_decay_audit():
    names = ["funding_mom", "rsi", "macd", "decayed_feat"]
    weights = np.array([0.45, 0.35, 0.19, 0.005])

    decayed = background_ml_tester.audit_feature_importance_decay(names, weights)
    assert "decayed_feat" in decayed
    assert "funding_mom" not in decayed


def test_send_telegram_report_formatting():
    shadow_res = {"champion_accuracy": 0.85, "challenger_accuracy": 0.88, "promoted": True}
    stress_res = {"is_stable": True, "mean_prediction_shift": 0.02}
    decayed = []
    # Mock send_telegram_alert call to return True
    sent = background_ml_tester.send_telegram_report(shadow_res, stress_res, decayed)
    assert isinstance(sent, bool)

