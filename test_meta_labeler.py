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
    ok, msg = promote_if_better(cand={"mcc": 0.1067}, champ={"mcc": 0.1542})
    assert ok is False and "regression" in msg.lower()


if __name__ == "__main__":
    test_meta_filter_passes_through_on_thin_data()
    print("✅ test_meta_filter_passes_through_on_thin_data PASSED")
    test_promotion_rejects_mcc_regression()
    print("✅ test_promotion_rejects_mcc_regression PASSED")
