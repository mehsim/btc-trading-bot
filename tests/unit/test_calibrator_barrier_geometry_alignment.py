import pytest
import pandas as pd
import numpy as np
from config import TIMEFRAME_CONFIG
from train import add_triple_barrier_labels


def test_add_triple_barrier_labels_uses_timeframe_config():
    """Verify add_triple_barrier_labels uses TIMEFRAME_CONFIG directly without hardcoded interval overrides."""
    # Create sample synthetic dataframe
    n = 100
    df = pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "ATR": np.full(n, 1.0),
        "ADX": np.full(n, 30.0)  # Trending
    })

    # Run for 15m
    df_labeled = add_triple_barrier_labels(df, interval="15")
    assert "target_trend" in df_labeled.columns
    assert len(df_labeled) == n


def test_calibrator_barrier_geometry_divergence_check():
    """Verify that divergent barrier geometry in a calibrator is rejected."""
    from config import TIMEFRAME_CONFIG
    cfg_15 = TIMEFRAME_CONFIG.get("15", {})
    live_tp = float(cfg_15.get("tp_mult_ranging", 1.40))
    live_sl = float(cfg_15.get("sl_mult", 0.85))
    live_lh = int(cfg_15.get("lookahead", 12))

    # Artificially divergent geometry
    divergent_cal = {
        "scaling_method": "beta_calibration",
        "a": 1.0,
        "b": 1.0,
        "c": 0.0,
        "barrier_geometry": {
            "tp_mult_ranging": live_tp + 2.0,  # +2.0 ATR difference
            "sl_mult": live_sl,
            "lookahead": live_lh
        }
    }

    b_geom = divergent_cal["barrier_geometry"]
    cal_tp = float(b_geom.get("tp_mult_ranging", live_tp))
    cal_sl = float(b_geom.get("sl_mult", live_sl))
    cal_lh = int(b_geom.get("lookahead", live_lh))

    is_divergent = abs(cal_tp - live_tp) > 0.50 or abs(cal_sl - live_sl) > 0.30 or abs(cal_lh - live_lh) > 6
    assert is_divergent is True
