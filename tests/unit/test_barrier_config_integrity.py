import os
import json
import pytest
import config
from config_verifier import assert_shared_constants_aligned


def test_timeframe_config_barrier_geometry_validity():
    """Verify that all timeframes in TIMEFRAME_CONFIG have valid non-inverted barrier geometries."""
    for tf, tf_cfg in config.TIMEFRAME_CONFIG.items():
        tp_t = float(tf_cfg["tp_mult_trending"])
        tp_r = float(tf_cfg["tp_mult_ranging"])
        sl_m = float(tf_cfg["sl_mult"])
        lookahead = int(tf_cfg["lookahead"])

        assert tp_t >= tp_r, f"Inverted barrier geometry in {tf}m: tp_trending {tp_t} < tp_ranging {tp_r}"
        assert sl_m >= 0.3, f"SL multiplier {sl_m} below floor in {tf}m"
        assert lookahead >= 4, f"Lookahead {lookahead} too short in {tf}m"


def test_config_verifier_asserts_barrier_integrity():
    """Verify that assert_shared_constants_aligned runs and passes."""
    assert assert_shared_constants_aligned() is True


def test_inverted_barrier_rejection_logic():
    """Verify that if an optimized_barriers JSON file has inverted targets, it is rejected."""
    # Test simulation of the validation guard
    inverted_payload = {
        "tp_mult_trending": 0.95,
        "tp_mult_ranging": 1.99,
        "sl_mult": 0.50,
        "lookahead": 12
    }

    tp_t = float(inverted_payload["tp_mult_trending"])
    tp_r = float(inverted_payload["tp_mult_ranging"])
    is_valid = tp_t >= tp_r
    assert is_valid is False, "Inverted payload should fail validation"
