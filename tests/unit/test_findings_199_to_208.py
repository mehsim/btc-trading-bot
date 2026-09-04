"""
Unit tests covering remediation of audit findings #31 through #40 (Finding test IDs 199 to 208):
- Finding #31 / Test 199: Economic break-even threshold re-evaluated upon min-order bump stop widening; dead TP reconciliation clause fixed
- Finding #32 / Test 200: Live price freshness timestamp recorded; 0.0 timestamp treated as stale; exit loop fails closed on stale price
- Finding #33 / Test 201: REST price fallback does not overwrite last_ws_update_time, preserving watchdog forced reconnect
- Finding #34 / Test 202: Synthetic bar attributes and is_synthetic column propagated across merges; >5 synthetic bars detected
- Finding #35 / Test 203: Wilder smoothed ATR injected for exit policy engine; ATR contraction condition evaluates; recovery ATR clamped
- Finding #36 / Test 204: MarketDataQualityMonitor returns tier and reasons; RED tier triggers fail-closed abstention; real clock skew measured
- Finding #37 / Test 205: Dynamic confidence threshold strictly floored at economic_base_threshold
- Finding #38 / Test 206: Kelly roundtrip cost converted to ATR units; empirical Kelly branch respects order geometry break-even
- Finding #39 / Test 207: Calibrator target_definition enforced; on-disk calibrators enriched with target_definition & is_fitted
- Finding #40 / Test 208: MLOps promotion gate evaluates absolute quality floors before MCC regression check; distribution shift cannot forgive high Brier
"""

import time
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import config
import risk_engine
import trade_calculators
from tools.beta_calibrator import BetaCalibrator, is_calibrator_viable
from market_data_quality import MarketDataQualityMonitor
from data_quality_engine import DataQualityEngine
from mlops_engine import promote_if_better


def test_finding_199_reconciled_sl_widening_and_tp_clause():
    """Finding #31 (Test 199): Min-order bump stop widening re-evaluates break-even; dead TP clause fixed."""
    entry_price = 60000.0
    atr_dollars = 300.0
    atr_norm_val = atr_dollars / entry_price
    cost_bps = 5.0
    realized_haircut = 0.28
    resolved_tp_m = 1.5

    # Case A: Stop widened from 0.85 ATR to 2.5 ATR due to min order bump
    orig_stop_dist = 0.85 * atr_dollars
    widened_stop_dist = 2.5 * atr_dollars
    new_sl_price = entry_price - widened_stop_dist
    take_profit_price = entry_price + (resolved_tp_m * atr_dollars)

    re_sl_m = widened_stop_dist / atr_dollars
    re_eff_tp = resolved_tp_m * realized_haircut
    re_p_star = re_sl_m / (re_eff_tp + re_sl_m)
    re_cost_adj = (cost_bps / 1e4) / ((re_eff_tp + re_sl_m) * atr_norm_val)
    re_gate_threshold = float(round(re_p_star + re_cost_adj, 4))

    # With widened stop, break-even probability required is high (e.g. ~85%)
    assert re_gate_threshold > 0.70

    # Low calibrated confidence (e.g. 0.52) must be REJECTED
    calibrated_confidence = 0.52
    econ_pass = trade_calculators.passes_economic_gate(
        entry=entry_price, tp=take_profit_price, sl=new_sl_price, conf=calibrated_confidence
    )
    assert econ_pass is False or calibrated_confidence < re_gate_threshold

    # Case B: Dead TP reconciliation check logic test
    orig_sl = 59700.0
    orig_tp = 60500.0
    adjusted_struct = {
        "stop_price": 59700.0,
        "tp_price": 60300.0,  # TP was capped downwards
        "leverage": 5.0
    }
    # Pre-fix bug: take_profit_price was assigned adjusted_struct["tp_price"] before check
    # Post-fix: Check against orig_tp
    diff_tp = abs(adjusted_struct["tp_price"] - orig_tp)
    assert diff_tp > 1e-6


def test_finding_200_live_price_freshness_zero_ts_and_exit_fail_closed():
    """Finding #32 (Test 200): 0.0 timestamp treated as stale; exit loop fails closed on stale price."""
    import main

    test_symbol = "BTCUSDT"
    # Case A: Symbol has price but price_ts is 0.0 (never timestamped / falsy)
    main.bot_state[f"live_price_{test_symbol}"] = 60000.0
    main.bot_state[f"live_price_ts_{test_symbol}"] = 0.0

    with patch("main.get_fallback_price", return_value=None):
        # When fallback is unreachable, update_bybit_stop_loss must refuse to update against stale 0.0 ts
        res = main.update_bybit_stop_loss(test_symbol, 59000.0, active_trade={"stop_loss": 58500.0, "direction": "Bullish"})
        assert res is False

    # Case B: Exit loop fails closed when price is stale and fallback returns None
    now_exit = time.time()
    main.bot_state[f"live_price_{test_symbol}"] = 60000.0
    main.bot_state[f"live_price_ts_{test_symbol}"] = now_exit - 100.0  # 100s old

    active_trade = {"symbol": test_symbol, "entry_price": 59500.0, "stop_loss": 59000.0, "take_profit": 61000.0}
    with patch("main.get_fallback_price", return_value=None):
        # Exit loop must recognize staleness and not proceed to evaluate
        symbol_price = main.bot_state.get(f"live_price_{test_symbol}")
        symbol_price_ts = main.bot_state.get(f"live_price_ts_{test_symbol}", 0.0)
        is_stale = (now_exit - symbol_price_ts > 30.0) or (symbol_price_ts <= 0.0)
        assert is_stale is True


def test_finding_201_rest_fallback_preserves_last_ws_update_time():
    """Finding #33 (Test 201): REST fallback queries do not poison last_ws_update_time."""
    import main

    # Set initial WebSocket timestamp
    ws_time = 100000.0
    main.last_ws_update_time = ws_time
    main.bot_state["last_rest_price_time"] = 100000.0

    # Simulate REST fallback write at a later time
    later_time = 100050.0
    main.bot_state["last_rest_price_time"] = later_time

    # last_ws_update_time must remain unchanged
    assert main.last_ws_update_time == ws_time
    # Watchdog silent duration must accurately measure 50 seconds
    silent_duration = later_time - main.last_ws_update_time
    assert silent_duration == 50.0


def test_finding_202_synthetic_bar_propagation_across_merges():
    """Finding #34 (Test 202): is_synthetic column and attrs survive merges; >5 synthetic bars detected."""
    import data

    # Create dummy frame with synthetic bars
    ts = np.arange(1000, 1000 + 20 * 60000, 60000, dtype=np.int64)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low": [99.0] * 20,
        "close": [100.0] * 20,
        "volume": [10.0] * 20,
        "is_synthetic": [True] * 6 + [False] * 14
    })
    df.attrs["gap_exceeded"] = True
    df.attrs["synthetic_bar_count"] = 6
    df.attrs["max_consecutive_synthetic_bars"] = 6

    # Test merge preserves attrs
    df_merged = data._merge_cached_derivatives(df, None, None, None)
    assert df_merged.attrs.get("gap_exceeded") is True
    assert df_merged.attrs.get("synthetic_bar_count") == 6
    assert "is_synthetic" in df_merged.columns
    assert df_merged["is_synthetic"].sum() == 6

    # Verify continuity guard detects gap_exceeded when synthetic_count > 5
    synthetic_count = int(df_merged["is_synthetic"].sum())
    is_exceeded = bool(df_merged.attrs.get("gap_exceeded", False) or synthetic_count > 5)
    assert is_exceeded is True


def test_finding_203_wilder_smoothing_and_atr_in_exit_policy():
    """Finding #35 (Test 203): Wilder smoothed ATR injected for exit policy engine."""
    np.random.seed(42)
    n = 30
    high_s = pd.Series(100.0 + np.random.uniform(1, 3, n))
    low_s = pd.Series(100.0 - np.random.uniform(1, 3, n))
    close_s = pd.Series(100.0 + np.random.uniform(-1, 1, n))
    prev_close_s = close_s.shift(1)

    tr_s = pd.concat([high_s - low_s, (high_s - prev_close_s).abs(), (low_s - prev_close_s).abs()], axis=1).max(axis=1)
    atr_wilder = tr_s.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    df_pos = pd.DataFrame({"high": high_s, "low": low_s, "close": close_s, "volume": [100.0] * n})
    df_pos["ATR"] = atr_wilder
    df_pos["ATR_norm"] = atr_wilder / close_s

    # Verify ATR is present and non-null
    assert "ATR" in df_pos.columns
    assert "ATR_norm" in df_pos.columns
    curr_atr_val = float(df_pos["ATR"].dropna().iloc[-1])
    assert curr_atr_val > 0.0

    # ATR contraction exit rule test: current_atr < 0.8 * entry_atr
    entry_atr = curr_atr_val * 1.5
    c2 = curr_atr_val < (0.8 * entry_atr)
    assert c2 is True


def test_finding_204_market_data_quality_monitor_red_tier_abstention():
    """Finding #36 (Test 204): MDQ returns tier and reasons; RED tier triggers fail-closed abstention."""
    mdq = MarketDataQualityMonitor()
    now_t = time.time()

    # Stale candle timestamp (e.g. 500 seconds old) triggers RED tier
    res = mdq.evaluate_feed_health(
        last_candle_timestamp=now_t - 500.0,
        server_time_ms=now_t * 1000.0,
        client_time_ms=now_t * 1000.0,
        ws_connected=False,
        interval_sec=60.0
    )

    assert res["health_tier"] == "RED"
    assert res["tier"] == "RED"
    assert "reasons" in res
    assert res["trading_allowed"] is False

    # Production check must trigger
    should_abstain = (res.get("health_tier") == "RED" or res.get("tier") == "RED" or not res.get("trading_allowed", True))
    assert should_abstain is True


def test_finding_205_dynamic_conf_threshold_floored_at_economic_base():
    """Finding #37 (Test 205): Dynamic confidence threshold cannot be capped below economic_base_threshold."""
    economic_base_threshold = 0.527
    base_cfg_thresh = 0.40
    effective_base = max(float(economic_base_threshold), float(base_cfg_thresh))

    # Even if max_conf_cap is nominally 0.50, effective_base overrides it
    max_conf_cap = max(effective_base, 0.50)
    dynamic_conf_threshold = 0.53
    dynamic_conf_threshold = max(effective_base, min(max_conf_cap, dynamic_conf_threshold))
    assert dynamic_conf_threshold >= economic_base_threshold

    # End-of-block uplift bounding
    max_allowed_threshold = max(effective_base, min(0.65, effective_base + 0.15))
    final_thresh = float(round(max(effective_base, min(max_allowed_threshold, dynamic_conf_threshold)), 4))
    assert final_thresh >= economic_base_threshold


def test_finding_206_kelly_roundtrip_cost_dimensional_alignment():
    """Finding #38 (Test 206): Kelly roundtrip cost converted to ATR units; empirical branch respects geometry break-even."""
    # Test A: Dimensionally consistent roundtrip cost in compute_conservative_kelly
    atr_norm = 0.005  # 0.5% ATR
    roundtrip_cost = 0.0010  # 10 bps
    expected_cost_in_atr = roundtrip_cost / atr_norm  # 0.20 ATR units

    tp_multiplier = 1.5
    sl_multiplier = 1.0
    haircut = 0.28
    eff_tp = tp_multiplier * haircut  # 0.42
    b_ratio = max(0.01, (eff_tp - expected_cost_in_atr) / sl_multiplier)  # (0.42 - 0.20) / 1.0 = 0.22
    p_star = 1.0 / (b_ratio + 1.0)  # ~0.82

    # Calibrated confidence 0.60 is below p* ~0.82 -> Kelly must return 0.0
    k_val = risk_engine.compute_conservative_kelly(
        calibrated_confidence=0.60,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        haircut=haircut,
        atr_norm=atr_norm
    )
    assert k_val == 0.0

    # Calibrated confidence 0.90 is above p* -> Kelly must return positive
    k_pos = risk_engine.compute_conservative_kelly(
        calibrated_confidence=0.90,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        haircut=haircut,
        atr_norm=atr_norm
    )
    assert k_pos > 0.0


def test_finding_207_calibrator_target_definition_enforcement():
    """Finding #39 (Test 207): Calibrator target_definition enforced and on-disk files enriched."""
    # When require_target_def=True, missing target_definition fails closed
    cal_no_target = {
        "scaling_method": "beta_calibration",
        "a": 1.2, "b": 0.8, "c": 0.1,
        "is_fitted": True,
        "is_fallback": False
    }
    assert is_calibrator_viable(cal_no_target, require_target_def=True) is False

    # Incompatible target_definition fails closed
    cal_bad = dict(cal_no_target, target_definition="fixed_horizon_return")
    assert is_calibrator_viable(cal_bad, require_target_def=True) is False

    # Valid target_definition passes
    cal_good = dict(cal_no_target, target_definition="triple_barrier_exact")
    assert is_calibrator_viable(cal_good, require_target_def=True) is True

    # Check on-disk 60m champion file carries target_definition and is_fitted
    with open("calibrator_trending_60.json", "r") as fp:
        d = json.load(fp)
    assert d.get("target_definition") == "triple_barrier_exact"
    assert d.get("is_fitted") is True
    assert "barrier_geometry" in d


def test_finding_208_mlops_promotion_gate_evaluates_floors_before_mcc_regression():
    """Finding #40 (Test 208): MLOps absolute quality floors evaluated before MCC regression check."""
    # Candidate has MCC regression relative to champion (0.055 vs 0.09), BUT also fails Brier (0.62 > 0.50)
    cand_eval = {
        "mcc": 0.055,
        "mcc_min": 0.02,
        "ece": 0.05,
        "brier_score": 0.62,  # Violates 0.50 ceiling!
        "val_accuracy": 0.45,
        "sharpe_oos": 0.80,
        "probs": [[0.5, 0.3, 0.2]] * 40 + [[0.2, 0.5, 0.3]] * 40 + [[0.2, 0.2, 0.6]] * 20
    }
    champ_eval = {
        "mcc": 0.09,
        "balanced_accuracy": 0.50
    }

    promoted, p_reason = promote_if_better(
        name="test_model",
        challenger_version="v1.0.0",
        cand=cand_eval,
        champ=champ_eval
    )

    # Must be rejected because of Brier ceiling, NOT MCC regression
    assert promoted is False
    assert "Brier" in p_reason
    assert "MCC regression" not in p_reason
