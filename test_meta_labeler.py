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


if __name__ == "__main__":
    test_meta_filter_passes_through_on_thin_data()
    print("✅ test_meta_filter_passes_through_on_thin_data PASSED")
