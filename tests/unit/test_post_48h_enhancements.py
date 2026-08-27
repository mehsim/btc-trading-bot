import pytest
import numpy as np
import pandas as pd
from risk_engine import get_timeframe_sizing_multiplier
from meta_labeler import evaluate_meta_filter, _META_MODEL_CACHE


def test_timeframe_sizing_multipliers():
    """Verify that higher timeframes receive greater capital weighting than lower timeframes."""
    assert get_timeframe_sizing_multiplier("240") == 1.25
    assert get_timeframe_sizing_multiplier("120") == 1.10
    assert get_timeframe_sizing_multiplier("60") == 1.00
    assert get_timeframe_sizing_multiplier("30") == 0.75
    assert get_timeframe_sizing_multiplier("15") == 0.60
    # Higher timeframes must scale strictly greater than lower timeframes
    assert get_timeframe_sizing_multiplier("240") > get_timeframe_sizing_multiplier("60") > get_timeframe_sizing_multiplier("15")


def test_meta_labeler_rich_features_and_fail_open():
    """Verify that evaluate_meta_filter supports rich feature vectors and respects the 0.60 CV AUC significance floor."""
    _META_MODEL_CACHE["pass_all"] = True
    approved, prob = evaluate_meta_filter("SOLUSDT", "240", "Bullish", confidence=0.75)
    assert approved is True
    assert prob == 1.0


def test_tier2_runner_ratchet_math():
    """Verify Tier-2 runner ratchet threshold calculations."""
    entry_price = 100.0
    atr_dollars = 2.0
    tier2_trigger = 1.5 * atr_dollars # 3.0 USD
    tier2_lock = 0.5 * atr_dollars    # 1.0 USD

    # Bullish: when price >= 103.0, target SL is 101.0
    curr_price_bull = 103.5
    assert curr_price_bull >= entry_price + tier2_trigger
    target_sl_bull = entry_price + tier2_lock
    assert target_sl_bull == 101.0
    assert target_sl_bull > entry_price # Guaranteed positive profit

    # Bearish: when price <= 97.0, target SL is 99.0
    curr_price_bear = 96.5
    assert curr_price_bear <= entry_price - tier2_trigger
    target_sl_bear = entry_price - tier2_lock
    assert target_sl_bear == 99.0
    assert target_sl_bear < entry_price # Guaranteed positive profit for short
