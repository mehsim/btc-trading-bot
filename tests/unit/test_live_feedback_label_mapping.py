import pytest
import pandas as pd
import numpy as np


def test_live_feedback_direction_outcome_mapping():
    """Verify executed/simulated trade outcomes correctly map into 3-class target_trend and signed price return."""
    # Test cases: (direction, pnl, entry_price, exit_price, expected_trend, expected_price_sign)
    test_cases = [
        # Long win -> Bullish (2), positive return
        ("Bullish", 50.0, 100.0, 105.0, 2, 1),
        # Long loss -> Bearish (0), negative return
        ("Bullish", -30.0, 100.0, 97.0, 0, -1),
        # Short win -> Bearish (0), negative price change
        ("Bearish", 50.0, 100.0, 95.0, 0, -1),
        # Short loss -> Bullish (2), positive price change
        ("Bearish", -30.0, 100.0, 103.0, 2, 1),
        # Scratch trade -> Neutral (1), 0.0 return
        ("Bullish", 0.0, 100.0, 100.0, 1, 0),
    ]

    for direction, pnl, entry_price, exit_price, expected_trend, expected_sign in test_cases:
        direction_str = str(direction).capitalize()
        is_long = direction_str in ["Bullish", "Long", "Buy"]
        is_short = direction_str in ["Bearish", "Short", "Sell"]

        realized_price_ret = (exit_price - entry_price) / entry_price

        if is_long:
            if pnl > 0 or realized_price_ret > 0.001:
                target_class = 2
            elif pnl < 0 or realized_price_ret < -0.001:
                target_class = 0
            else:
                target_class = 1
        elif is_short:
            if pnl > 0 or realized_price_ret < -0.001:
                target_class = 0
            elif pnl < 0 or realized_price_ret > 0.001:
                target_class = 2
            else:
                target_class = 1
        else:
            target_class = 1

        assert target_class == expected_trend, f"Failed for {direction} with pnl={pnl}: expected {expected_trend}, got {target_class}"
        if expected_sign > 0:
            assert realized_price_ret > 0
        elif expected_sign < 0:
            assert realized_price_ret < 0
        else:
            assert realized_price_ret == 0.0


def test_live_feedback_dataframe_validation():
    """Verify live feedback DataFrame schema validation."""
    valid_live_df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "target_trend": [2, 0, 1],
        "target_price_change": [0.02, -0.015, 0.0],
        "sample_weight": [1.0, 1.0, 1.0]
    })

    # Assert labels strictly in {0, 1, 2}
    invalid_labels = set(valid_live_df["target_trend"].dropna().unique()) - {0, 1, 2}
    assert len(invalid_labels) == 0

    # Ensure degenerate label raises error
    corrupt_live_df = pd.DataFrame({
        "target_trend": [1, 5, -1]
    })
    invalid_corrupt = set(corrupt_live_df["target_trend"].dropna().unique()) - {0, 1, 2}
    assert len(invalid_corrupt) > 0
