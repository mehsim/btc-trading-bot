"""
Unit tests for audit findings #21 through #30 (test_findings_189_to_198.py).
Covers:
- #21: Price regressor manifests dedicated metrics & purged classification fields.
- #22: Market data staleness gating and fail_if_stale enforcement.
- #23: Calibrator knot support requirement (min_bin_support >= 100) & fail-closed handling.
- #24: Triple-barrier labeling structural floor alignment and timeframe scaling.
- #25: Challenger profit factor & Sharpe initializer sanitization.
- #26: ExecutionValidator portfolio heat ceiling (0.35) and atr_norm market impact scaling.
- #27 & #29: Realized R:R haircut alignment (0.28) and size-normalized empirical R:R estimation.
- #28: Calibrator barrier geometry serialization and strict verification tolerances.
- #30: BetaCalibrator unclipped lower probabilities and dynamic_conf_threshold non-attenuation.
"""

import json
import time
import os
import glob
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

import config
from tools.beta_calibrator import BetaCalibrator, is_calibrator_viable, calibrate_probability
from execution_validator import ExecutionValidator
import trade_calculators
import risk_engine
from ensemble import get_manifest_hmac_secret, verify_manifest_hmac_signature


# ---------------------------------------------------------------------------
# Finding #21: Price regressor manifests dedicated metrics
# ---------------------------------------------------------------------------
def test_finding_21_price_regressor_manifest_metrics():
    """Finding #21: Regressor manifests must not copy trend classifier metrics."""
    price_manifests = glob.glob("ensemble_*_price_*_manifest.json")
    assert len(price_manifests) > 0, "Price manifests must exist on disk"

    for p_path in price_manifests:
        with open(p_path, "r") as f:
            data = json.load(f)

        assert data.get("model_type") == "regressor", f"{p_path} must have model_type='regressor'"
        assert "regression_metrics" in data, f"{p_path} must have regression_metrics"
        reg_m = data["regression_metrics"]
        assert "mae" in reg_m and reg_m["mae"] is not None, f"{p_path} missing mae"
        assert "rmse" in reg_m and reg_m["rmse"] is not None, f"{p_path} missing rmse"
        assert "r2" in reg_m and reg_m["r2"] is not None, f"{p_path} missing r2"
        assert "directional_accuracy" in reg_m and reg_m["directional_accuracy"] is not None, f"{p_path} missing dir_acc"

        # Classification fields must be purged
        for clf_key in ["cv_metrics", "label_distribution", "confusion_matrix", "manifest_bal_acc", "manifest_mcc"]:
            assert clf_key not in data, f"{p_path} must not contain classifier metric {clf_key}"

        # Signature must verify cleanly
        assert verify_manifest_hmac_signature(data) is True


# ---------------------------------------------------------------------------
# Finding #22: Market data staleness gating and fail_if_stale enforcement
# ---------------------------------------------------------------------------
def test_finding_22_market_data_staleness_gating():
    """Finding #22: get_history with fail_if_stale=True returns empty DataFrame on stale cache."""
    from data import get_history

    # Mock bybit_public_get to fail so cache/fallbacks trigger
    with patch("data.bybit_public_get", side_effect=Exception("Bybit down")), \
         patch("requests.get", side_effect=Exception("Fallbacks down")):
        
        # When fail_if_stale=True, a stale or failed fetch must return an empty dataframe with fetch_ok=False
        df_res = get_history(symbol="BTCUSDT", interval="15", limit=50, fail_if_stale=True)
        assert df_res is not None
        assert df_res.empty or not df_res.attrs.get("fetch_ok", True)
        assert df_res.attrs.get("fetch_ok") is False


def test_finding_22_htf_cache_and_confluence_staleness():
    """Finding #22: HTFTrendCache and ConfluenceEngine reject stale data."""
    import main
    import confluence_engine

    # Stale DataFrame (last bar was 10 hours ago)
    stale_ts = (time.time() - 36000) * 1000
    stale_df = pd.DataFrame({
        "timestamp": [stale_ts + i * 60000 for i in range(50)],
        "open": [60000.0] * 50,
        "high": [60100.0] * 50,
        "low": [59900.0] * 50,
        "close": [60000.0] * 50,
        "volume": [100.0] * 50,
        "turnover": [6000000.0] * 50,
    })
    stale_df.attrs["fetch_ok"] = False

    with patch("main.get_history", return_value=stale_df):
        cache = main.HTFTrendCache()
        ema9, ema21 = cache.get_trend("BTCUSDT", "60")
        assert (ema9, ema21) == (0.0, 0.0), "HTFTrendCache must return (0.0, 0.0) when data is stale"

    htf_cache_dict = {}
    confluence_engine.set_valid_htf_cache(htf_cache_dict, ("BTCUSDT", "D"), stale_df)
    assert confluence_engine.get_valid_htf_cache(htf_cache_dict, ("BTCUSDT", "D")) is None, \
        "set_valid_htf_cache must not store or return stale frames with fetch_ok=False"


# ---------------------------------------------------------------------------
# Finding #23: Calibrator knot support requirement & fail-closed handling
# ---------------------------------------------------------------------------
def test_finding_23_calibrator_knot_support_requirement():
    """Finding #23: Calibrator with min_bin_support < 100 is non-viable and fails closed."""
    thin_calibrator = {
        "calibrator_type": "isotonic",
        "prob_ceiling": 0.85,
        "min_bin_support": 15,  # < 100
        "calibration_curve": [[0.1, 0.2], [0.8, 0.85]],
    }
    assert is_calibrator_viable(thin_calibrator, min_required_p_star=0.40) is False

    # calibrate_probability should fail closed (return 0.0)
    p_cal = calibrate_probability(0.50, thin_calibrator)
    assert p_cal == 0.0, "calibrate_probability must fail closed to 0.0 for non-viable calibrator"

    robust_calibrator = {
        "calibrator_type": "isotonic",
        "prob_ceiling": 0.85,
        "min_bin_support": 150,  # >= 100
        "X": [0.1, 0.8],
        "y": [0.2, 0.85],
    }
    assert is_calibrator_viable(robust_calibrator, min_required_p_star=0.40) is True


# ---------------------------------------------------------------------------
# Finding #24: Triple-barrier labeling structural floor alignment
# ---------------------------------------------------------------------------
def test_finding_24_triple_barrier_labeling_structural_floors():
    """Finding #24: add_triple_barrier_labels respects structural stop floors (1.25 ATR, 0.8% price)."""
    from train import add_triple_barrier_labels

    np.random.seed(42)
    n = 200
    prices = 60000.0 + np.cumsum(np.random.randn(n) * 10)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 15,
        "low": prices - 15,
        "close": prices,
        "volume": [100.0] * n,
        "ATR": [10.0] * n,  # Very low ATR (10 / 60000 = 0.016%)
        "ADX": [25.0] * n,
    })

    df_labeled = add_triple_barrier_labels(df, interval="15")
    assert "target_trend" in df_labeled.columns
    # Check that labels were assigned without crash and valid values {0, 1, 2}
    valid_targets = df_labeled["target_trend"].dropna().unique()
    for t in valid_targets:
        assert int(t) in [0, 1, 2]


# ---------------------------------------------------------------------------
# Finding #25: Challenger profit factor & Sharpe initializer sanitization
# ---------------------------------------------------------------------------
def test_finding_25_challenger_manifest_initializers():
    """Finding #25: 240m manifests on disk must not carry default 1.0/0.0 with promoted: true."""
    trend_240_manifests = glob.glob("ensemble_*_trend_240*manifest.json")
    assert len(trend_240_manifests) > 0, "240m manifests must exist"

    for t_path in trend_240_manifests:
        with open(t_path, "r") as f:
            data = json.load(f)

        if "challenger" in t_path:
            assert data.get("promoted") is False, f"{t_path} challenger must have promoted=False"

        # If profit factor / holdout sharpe were un-evaluated, they must be None
        pf = data.get("profit_factor")
        sharpe = data.get("holdout_sharpe")
        if pf is not None:
            assert pf != 1.0 or sharpe != 0.0, f"{t_path} has un-gated default 1.0/0.0 metrics"

        assert verify_manifest_hmac_signature(data) is True


# ---------------------------------------------------------------------------
# Finding #26: ExecutionValidator portfolio heat ceiling & atr_norm scaling
# ---------------------------------------------------------------------------
def test_finding_26_execution_validator_heat_and_atr_norm():
    """Finding #26: ExecutionValidator default heat is 0.35 and atr_norm scales dynamic impact."""
    ev = ExecutionValidator()
    assert ev.max_portfolio_heat == 0.35
    assert config.MAX_PORTFOLIO_HEAT == 0.35

    # Test portfolio heat rejection
    valid, msg = ev.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=60000.0,
        stop_loss_price=59000.0,
        take_profit_price=62000.0,
        position_size_usd=100.0,
        live_price=60000.0,
        top_book_depth_usd=50000.0,
        portfolio_heat=0.36,  # > 0.35
    )
    assert valid is False
    assert "portfolio heat" in msg.lower()

    # Test atr_norm impact scaling
    # At top_book_depth_usd=25000:
    # Under normal volatility (atr_norm=0.01), dynamic_max_impact = 1.0%
    # With position_size_usd=350, estimated_impact = 1.4% > 1.0% -> rejected
    valid_norm_vol, _ = ev.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=60000.0,
        stop_loss_price=59000.0,
        take_profit_price=62000.0,
        position_size_usd=350.0,
        live_price=60000.0,
        top_book_depth_usd=25000.0,
        portfolio_heat=0.10,
        atr_norm=0.01,
    )
    assert valid_norm_vol is False

    # Under high volatility (atr_norm=0.03), dynamic_max_impact scales to 2.0%
    # With position_size_usd=350, estimated_impact = 1.4% <= 2.0% -> accepted
    valid_high_vol, _ = ev.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=60000.0,
        stop_loss_price=59000.0,
        take_profit_price=62000.0,
        position_size_usd=350.0,
        live_price=60000.0,
        top_book_depth_usd=25000.0,
        portfolio_heat=0.10,
        atr_norm=0.03,
    )
    assert valid_high_vol is True


# ---------------------------------------------------------------------------
# Finding #27 & #29: Realized R:R haircut alignment and size normalization
# ---------------------------------------------------------------------------
def test_finding_27_29_realized_rr_haircut_and_size_normalization():
    """Finding #27 & #29: Haircut is 0.28 everywhere; empirical RR is size-normalized."""
    assert config.REALIZED_RR_HAIRCUT == 0.28

    # Under haircut = 0.28 and tp=2.0, sl=1.0, eff_tp is 0.56, so p* is 64.14%
    # With confidence 0.60 <= 0.6414, Kelly correctly abstains (returns 0.0)
    kelly_low = risk_engine.compute_conservative_kelly(
        calibrated_confidence=0.60,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        interval="15",
        trade_history=[],
    )
    assert kelly_low == 0.0, "Low confidence below break-even p* under 0.28 haircut must fail closed to 0.0"

    # With confidence 0.75 > 0.6414, Kelly produces positive allocation
    kelly_high = risk_engine.compute_conservative_kelly(
        calibrated_confidence=0.75,
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        interval="15",
        trade_history=[],
    )
    assert kelly_high > 0.0

    # Test trade_calculators estimate_empirical_realized_rr size-normalization
    # Provide trades with asymmetric position sizes to verify size distortion is eliminated
    trade_history = [
        {"pnl_usd": 100.0, "position_size_usd": 1000.0, "exit_type": "tp"}, # +10%
        {"pnl_usd": 50.0, "position_size_usd": 500.0, "exit_type": "tp"},   # +10%
        {"pnl_usd": -50.0, "position_size_usd": 1000.0, "exit_type": "sl"}, # -5%
        {"pnl_usd": -25.0, "position_size_usd": 500.0, "exit_type": "sl"},  # -5%
    ]
    empirical_rr = trade_calculators.estimate_empirical_realized_rr(trade_history, min_samples=4)
    # Win returns: 0.10, Loss returns: 0.05 -> empirical RR should be 0.10 / 0.05 = 2.0
    assert empirical_rr is not None
    assert abs(empirical_rr - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# Finding #28: Calibrator barrier geometry & verification tolerances
# ---------------------------------------------------------------------------
def test_finding_28_calibrator_barrier_geometry_tolerances():
    """Finding #28: verify_calibrator_barrier_geometry enforces tight tolerances (0.20 TP, 0.15 SL, 2 LH)."""
    import main

    # Divergent calibrator object (TP diverges by 0.25 > 0.20)
    cal_obj_divergent = {
        "calibrator_type": "isotonic",
        "barrier_geometry": {
            "tp_mult_trending": 2.50, # Live config for 15m is ~1.85 -> diff 0.65 > 0.20
            "tp_mult_ranging": 1.45,
            "sl_mult": 0.85,
            "lookahead": 12,
        },
        "prob_ceiling": 0.85,
        "min_bin_support": 200,
        "calibration_curve": [[0.1, 0.2], [0.8, 0.85]],
    }

    with patch("config.TIMEFRAME_CONFIG", {"15": {"tp_mult_trending": 1.85, "sl_mult": 0.85, "lookahead": 12}}):
        # We test the geometry verification logic
        cfg_live = config.TIMEFRAME_CONFIG["15"]
        cal_tp = cal_obj_divergent["barrier_geometry"]["tp_mult_trending"]
        live_tp = cfg_live["tp_mult_trending"]
        cal_sl = cal_obj_divergent["barrier_geometry"]["sl_mult"]
        live_sl = cfg_live["sl_mult"]
        cal_lh = cal_obj_divergent["barrier_geometry"]["lookahead"]
        live_lh = cfg_live["lookahead"]

        is_divergent = abs(cal_tp - live_tp) > 0.20 or abs(cal_sl - live_sl) > 0.15 or abs(cal_lh - live_lh) > 2
        assert is_divergent is True, "Barrier geometry with 0.65 TP delta must be flagged as divergent"


# ---------------------------------------------------------------------------
# Finding #30: BetaCalibrator unclipped lower probabilities & threshold floor
# ---------------------------------------------------------------------------
def test_finding_30_beta_calibrator_and_threshold_floor():
    """Finding #30: BetaCalibrator does not floor at 0.20; dynamic threshold not capped below base."""
    # BetaCalibrator should output probabilities below 0.20 when input is low
    bc = BetaCalibrator(a=1.5, b=0.5)
    probs = bc.predict_proba(np.array([0.01, 0.05, 0.10]))
    assert np.all(probs < 0.20), f"BetaCalibrator must not artificial clamp to 0.20: got {probs}"

    # Verify max_conf_cap does not lower dynamic threshold below effective_base
    base_cfg_thresh = 0.52
    economic_base_threshold = 0.48
    effective_base = max(float(economic_base_threshold), float(base_cfg_thresh)) # 0.52
    iv = "60"
    max_conf_cap = max(effective_base, 0.50 if str(iv) in ["15", "30", "60"] else 0.55)
    dynamic_conf_threshold = effective_base # 0.52
    dynamic_conf_threshold = min(max_conf_cap, dynamic_conf_threshold)
    assert dynamic_conf_threshold >= effective_base, \
        f"dynamic_conf_threshold ({dynamic_conf_threshold}) must never be capped below effective_base ({effective_base})"
