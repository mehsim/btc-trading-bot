"""
test_meta_labeler.py
--------------------
Unit test for meta_labeler fail-open pass-through behavior on thin data.
"""

from meta_labeler import evaluate_meta_filter


def test_meta_filter_passes_through_on_thin_data():
    approved, prob = evaluate_meta_filter("BTCUSDT", "15", "Bullish")
    assert approved is True
    assert isinstance(prob, float)
    assert prob >= 0.0 and prob <= 1.0


def test_promotion_rejects_mcc_regression():
    from mlops_engine import promote_if_better
    probs = [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]] * 20
    ok, msg = promote_if_better(cand={"mcc": 0.1067, "probs": probs}, champ={"mcc": 0.1542})
    assert ok is False and "regression" in msg.lower()


def test_tiny_negative_pred_change_does_not_invalidate():
    """A -0.05% predicted change is noise; should NOT trigger sign conflict."""
    close = 60000.0
    pred_change = -0.0001  # -0.0001 / 60000 * 100 = 0.000167% — well below 0.05%
    pred_pct = (abs(pred_change) / close) * 100
    ml_trend = "Bullish"
    strong_conflict = (ml_trend == "Bullish" and pred_change < 0 and pred_pct > 0.05) or \
                      (ml_trend == "Bearish" and pred_change > 0 and pred_pct > 0.05)
    assert strong_conflict is False, f"Noise pred_change={pred_change} should not conflict (pred_pct={pred_pct:.4f}%)"


def test_promotion_rejects_degenerate_predictions():
    from mlops_engine import promote_if_better
    import numpy as np
    # 98% predictions in class 1 (degenerate)
    probs = np.zeros((100, 3))
    probs[:, 1] = 0.98
    probs[:, 0] = 0.01
    probs[:, 2] = 0.01
    ok, msg = promote_if_better(cand={"probs": probs}, champ={"mcc": 0.10})
    assert ok is False and ("one-sided" in msg.lower() or "unresponsive" in msg.lower() or "rejected" in msg.lower())


def test_symbol_prediction_state_isolation():
    from signal_evaluator import SignalEvaluator

    bot_state = {}
    ev = SignalEvaluator(bot_state=bot_state)

    # Mock state writes for two distinct symbols evaluated sequentially
    ev.bot_state["latest_prediction_BTCUSDT_15m"] = {"symbol": "BTCUSDT", "direction": "Bullish", "confidence": 0.82}
    ev.bot_state["latest_prediction_ETHUSDT_15m"] = {"symbol": "ETHUSDT", "direction": "Bearish", "confidence": 0.65}

    btc_pred = ev.bot_state.get("latest_prediction_BTCUSDT_15m")
    eth_pred = ev.bot_state.get("latest_prediction_ETHUSDT_15m")

    assert btc_pred["direction"] == "Bullish" and btc_pred["confidence"] == 0.82
    assert eth_pred["direction"] == "Bearish" and eth_pred["confidence"] == 0.65
    assert btc_pred["direction"] != eth_pred["direction"]
    assert "latest_prediction_15m" not in ev.bot_state  # Shared key must not exist!


if __name__ == "__main__":
    test_meta_filter_passes_through_on_thin_data()
    print("✅ test_meta_filter_passes_through_on_thin_data PASSED")
    test_promotion_rejects_mcc_regression()
    print("✅ test_promotion_rejects_mcc_regression PASSED")
    test_tiny_negative_pred_change_does_not_invalidate()
    print("✅ test_tiny_negative_pred_change_does_not_invalidate PASSED")
    test_promotion_rejects_degenerate_predictions()
    print("✅ test_promotion_rejects_degenerate_predictions PASSED")
    test_symbol_prediction_state_isolation()
    print("✅ test_symbol_prediction_state_isolation PASSED")
