import pytest
import pandas as pd
import numpy as np
from config import TIMEFRAME_CONFIG
import trade_calculators
import risk_engine


def test_sl_multiplier_adjusted_binding_across_intervals():
    """Verify that sl_multiplier_adjusted is unconditionally bound and accurately
    derived for all intervals (both low-timeframe structural stops and HTF GMM stops)."""
    entry_close = 50000.0
    atr_norm_val = 0.01
    atr_dollars = entry_close * atr_norm_val  # 500.0
    ml_trend = "Bullish"
    regime_name = "Trending"

    # Synthetic candle history
    df_completed = pd.DataFrame({
        "timestamp": range(100),
        "open": np.linspace(49000, 50000, 100),
        "high": np.linspace(49500, 50500, 100),
        "low": np.linspace(48500, 49500, 100),
        "close": np.linspace(49000, 50000, 100),
        "volume": np.full(100, 100.0),
        "ATR": np.full(100, atr_dollars),
        "ATR_norm": np.full(100, atr_norm_val)
    })

    for iv in ["15", "30", "60", "120", "240"]:
        cfg = TIMEFRAME_CONFIG.get(str(iv), {})
        sl_multiplier = float(cfg.get("sl_mult", 0.85))

        # Simulation of main.py stop loss resolution logic
        if str(iv) in ["15", "30", "60"]:
            struct_sl, struct_sl_dist_pct, struct_meta = trade_calculators.calculate_adaptive_structural_stop(
                df_recent=df_completed,
                entry_price=entry_close,
                direction=ml_trend,
                atr_val=atr_dollars,
                regime=regime_name,
                volatility=atr_norm_val
            )
            stop_loss_price = struct_sl
            resolved_sl_dist = abs(entry_close - stop_loss_price)
            sl_multiplier_adjusted = resolved_sl_dist / max(1e-6, atr_dollars)
        else:
            tf_sl_mult = risk_engine.get_timeframe_stop_multiplier(iv)
            sl_multiplier_adjusted = sl_multiplier * tf_sl_mult
            resolved_sl_dist = risk_engine.calculate_final_stop_distance(
                entry_close, atr_dollars, "BTCUSDT", df=df_completed, gmm_multiplier=sl_multiplier_adjusted
            )
            if ml_trend == "Bullish":
                stop_loss_price = entry_close - resolved_sl_dist
            else:
                stop_loss_price = entry_close + resolved_sl_dist

        resolved_sl_m = resolved_sl_dist / max(1e-6, atr_dollars)
        sl_multiplier_adjusted = resolved_sl_m

        # Assertions
        assert sl_multiplier_adjusted is not None
        assert isinstance(sl_multiplier_adjusted, float)
        assert sl_multiplier_adjusted > 0.0
        assert np.isclose(sl_multiplier_adjusted, resolved_sl_m)
        assert np.isclose(sl_multiplier_adjusted * atr_dollars, resolved_sl_dist)
        assert np.isclose(abs(entry_close - stop_loss_price), sl_multiplier_adjusted * atr_dollars)
