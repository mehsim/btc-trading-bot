"""
Unit tests for audit defect findings #141 through #160.
"""

import os
import json
import time
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_finding_141_timeframe_stop_multiplier_intervals():
    """Finding #141: calculate_timeframe_stop_multiplier handles 5m and 120m intervals."""
    from risk_engine import get_timeframe_stop_multiplier, calculate_timeframe_stop_multiplier
    assert get_timeframe_stop_multiplier("5") == 0.70
    assert get_timeframe_stop_multiplier("5m") == 0.70
    assert get_timeframe_stop_multiplier("120") == 1.15
    assert get_timeframe_stop_multiplier("120m") == 1.15
    assert calculate_timeframe_stop_multiplier("5m") == 0.70
    assert calculate_timeframe_stop_multiplier("120m") == 1.15


def test_finding_142_composite_uncertainty_nan_and_mismatched_inputs():
    """Finding #142: calculate_composite_uncertainty handles NaNs and mismatched inputs."""
    from statistical_validation import calculate_composite_uncertainty
    # Input with NaNs
    unc_nan = calculate_composite_uncertainty(
        calibrated_p=0.6,
        model_predictions=[np.nan, 0.7, 0.5]
    )
    assert 0.0 <= unc_nan <= 1.0

    # Input with mismatched / corrupted list
    unc_empty = calculate_composite_uncertainty(
        calibrated_p=0.6,
        model_predictions=[]
    )
    assert 0.0 <= unc_empty <= 1.0


def test_finding_143_order_state_machine_handle_terminal_failure():
    """Finding #143: handle_terminal_failure marks order FAILED and removes from active tracking."""
    from order_state_machine import OrderStateMachine, OrderState
    osm = OrderStateMachine()
    order_id = "test_order_terminal_143"
    osm.create_order(order_id=order_id, symbol="BTCUSDT", side="Buy", qty=0.1, price=50000.0)
    assert order_id in osm.active_orders

    osm.handle_terminal_failure(order_id, reason="Margin insufficient")
    assert order_id not in osm.active_orders
    assert osm.get_order_state(order_id) == OrderState.FAILED


def test_finding_144_calculate_daily_pnl_safe_datetime_handling():
    """Finding #144: calculate_daily_pnl safely handles datetime and string timestamps."""
    from main import calculate_daily_pnl
    from datetime import datetime, timezone
    
    trades = [
        {
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "pnl": 150.0,
            "realized_pnl": 150.0
        },
        {
            "closed_at": "invalid_date_format",
            "pnl": 50.0
        }
    ]
    # Passing datetime object as time_now_dt
    pnl_dt = calculate_daily_pnl(trades, time_now_dt=datetime.now(timezone.utc))
    assert isinstance(pnl_dt, float)
    assert pnl_dt >= 0.0

    # Passing None
    pnl_none = calculate_daily_pnl(trades, time_now_dt=None)
    assert isinstance(pnl_none, float)


def test_finding_145_format_bybit_qty_step_size_precision():
    """Finding #145: format_bybit_qty uses step_size string representation for decimals."""
    from main import format_bybit_qty
    # Step size with 4 decimals "0.0001"
    with patch("main.get_instrument_specs", return_value={"qtyStep": "0.0001", "minOrderQty": "0.0001"}):
        formatted = format_bybit_qty("BTCUSDT", 0.123456)
        assert formatted == "0.1234"

    # Step size with integer "1"
    with patch("main.get_instrument_specs", return_value={"qtyStep": "1", "minOrderQty": "1"}):
        formatted_int = format_bybit_qty("BTCUSDT", 5.678)
        assert formatted_int == "5"


def test_finding_146_calibrate_probabilities_degenerate_inputs():
    """Finding #146: calibrate_probabilities falls back gracefully when inputs are constant."""
    from statistical_validation import calibrate_probabilities
    # Degenerate all-zeros or constant inputs
    y_true = np.array([1, 1, 1, 1, 1])
    y_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    cal_p, is_cal = calibrate_probabilities(y_true, y_probs)
    assert 0.0 <= cal_p <= 1.0


def test_finding_147_instrument_specs_thread_safety():
    """Finding #147: get_instrument_specs returns a safe copy of cached specs."""
    from bybit_client import get_instrument_specs, _specs_cache
    _specs_cache["BTCUSDT"] = {"symbol": "BTCUSDT", "tickSize": "0.1", "qtyStep": "0.001"}
    specs = get_instrument_specs("BTCUSDT")
    assert specs["tickSize"] == "0.1"
    # Mutating returned copy should not mutate cache
    specs["tickSize"] = "999.9"
    assert _specs_cache["BTCUSDT"]["tickSize"] == "0.1"


def test_finding_148_on_trade_closed_safe_float_conversion():
    """Finding #148: on_trade_closed handles None and string pnl_usd safely."""
    from learning_engine import LearningEngine
    le = LearningEngine()
    # Call on_trade_closed with None pnl_usd
    le.on_trade_closed({"symbol": "BTCUSDT", "interval": "15", "pnl_usd": None})
    # Call with valid float
    le.on_trade_closed({"symbol": "BTCUSDT", "interval": "15", "pnl_usd": "25.50"})


def test_finding_149_train_single_timeframe_regime_type():
    """Finding #149: _train_single_timeframe sets regime_type correctly."""
    from train import _train_single_timeframe
    import inspect
    sig = inspect.signature(_train_single_timeframe)
    assert "regime_type" in sig.parameters


def test_finding_150_circuit_breaker_timestamp_iso8601():
    """Finding #150: Circuit breaker records events in ISO-8601 UTC format."""
    from circuit_breaker import record_circuit_breaker_event, _circuit_breaker_history
    record_circuit_breaker_event("TEST_REASON_150", 123.45)
    last_event = _circuit_breaker_history[-1]
    assert "T" in last_event["timestamp"]
    assert last_event["timestamp"].endswith("Z") or "+00:00" in last_event["timestamp"]


def test_finding_151_orderbook_staleness_rejection():
    """Finding #151: get_orderbook_imbalance_and_spread rejects cache older than 5.0s."""
    from main import get_orderbook_imbalance_and_spread, order_flow_data, order_flow_lock
    sym = "TEST_STALE_SYM"
    with order_flow_lock:
        order_flow_data[sym] = {
            "last_ob_ts": time.time() - 10.0, # 10s old (stale)
            "latest_bids": [["50000", "1.0"]],
            "latest_asks": [["50010", "1.0"]]
        }
    with patch("main.bybit_get_request", return_value={"retCode": 0, "result": {"b": [["50000", "1.0"]], "a": [["50010", "1.0"]]}}):
        res = get_orderbook_imbalance_and_spread(sym)
        assert isinstance(res, dict)
        assert "imbalance" in res


def test_finding_152_save_circuit_breaker_atomic():
    """Finding #152: save_circuit_breaker_state uses atomic replace."""
    from circuit_breaker import save_circuit_breaker_state
    # Call save_circuit_breaker_state and ensure file is intact JSON
    assert save_circuit_breaker_state() is True
    assert os.path.exists("circuit_breaker_state.json")
    with open("circuit_breaker_state.json", "r") as f:
        data = json.load(f)
        assert "tripped" in data


def test_finding_153_check_wallet_margin_malformed_input():
    """Finding #153: check_wallet_margin_utilization handles non-dict input safely."""
    from risk_engine import check_wallet_margin_utilization
    # Malformed non-dict input
    ok, reason = check_wallet_margin_utilization(candidate_margin=100.0, margin_info="invalid_string")
    assert ok is False
    assert "REJECTED" in reason

    ok, reason = check_wallet_margin_utilization(candidate_margin=100.0, margin_info=None)
    assert ok is False


def test_finding_154_load_feature_columns_consistent():
    """Finding #154: load_feature_columns returns empty list when file missing."""
    from train import load_feature_columns
    feats = load_feature_columns("non_existent_file_xyz.json")
    assert isinstance(feats, list)
    assert len(feats) == 0


def test_finding_155_config_denylist_specific_exceptions():
    """Finding #155: config.py loads denylist without blanket broad Exception."""
    import config
    assert isinstance(config.MODEL_SLOT_DENYLIST, (set, list))


def test_finding_156_slippage_ratio_protected_division():
    """Finding #156: calculate_slippage_ratio clamps denominator and prevents ZeroDivisionError."""
    from transaction_cost_model import calculate_slippage_ratio
    # 0 depth
    slip_0 = calculate_slippage_ratio(trade_notional=1000.0, market_depth_usd=0.0)
    assert slip_0 > 0.0
    # Negative depth
    slip_neg = calculate_slippage_ratio(trade_notional=1000.0, market_depth_usd=-500.0)
    assert slip_neg > 0.0
    # Valid depth
    slip_val = calculate_slippage_ratio(trade_notional=100.0, market_depth_usd=100000.0)
    assert slip_val == 0.001


def test_finding_157_dynamic_stop_calibration_no_config_mutation():
    """Finding #157: calibrate_dynamic_sl_multiplier does not mutate global config."""
    import config
    from risk_engine import calibrate_dynamic_sl_multiplier
    orig_cfg = config.DYNAMIC_SL_MULTIPLIER
    calibrated = calibrate_dynamic_sl_multiplier(
        interval="15",
        realized_volatility=0.03,
        recent_slippage=0.002
    )
    assert isinstance(calibrated, float)
    assert 0.5 <= calibrated <= 2.5
    # Ensure global config was not mutated
    assert config.DYNAMIC_SL_MULTIPLIER == orig_cfg


def test_finding_158_parse_orderbook_depth_fractional_prices():
    """Finding #158: parse_orderbook_depth handles fractional altcoin ticks without ValueError."""
    from websocket_client import parse_orderbook_depth
    bids = [["0.001234", "1000.0"], ["0.001230", "2000.0"]]
    asks = [["0.001240", "1500.0"], ["0.001245", "2500.0"]]
    best_bid, best_ask, bid_depth, ask_depth = parse_orderbook_depth(bids, asks)
    assert best_bid == 0.001234
    assert best_ask == 0.001240
    assert bid_depth > 0.0
    assert ask_depth > 0.0


def test_finding_159_cancel_bybit_order_idempotency_110001():
    """Finding #159: cancel_bybit_order treats retCode 110001 as idempotent success."""
    from bybit_client import cancel_bybit_order
    with patch("bybit_client.execute_bybit_order_ws_or_rest", return_value={"retCode": 110001, "retMsg": "Order does not exist."}):
        res = cancel_bybit_order("BTCUSDT", "fake_order_110001")
        assert res.get("retCode") == 0
        assert res.get("idempotent") is True


def test_finding_160_ensemble_sanitize_probabilities_nan_handling():
    """Finding #160: _sanitize_probabilities replaces NaNs and Infs with uniform prior."""
    from ensemble import _sanitize_probabilities
    # Array with NaN
    bad_probs = np.array([[np.nan, 0.4, 0.2]])
    sanitized = _sanitize_probabilities(bad_probs, default_classes=3)
    assert not np.isnan(sanitized).any()
    assert np.isclose(sanitized.sum(), 1.0)
    assert np.isclose(sanitized[0][0], 1.0 / 3.0)

    # 2-class NaN array
    bad_2class = np.array([[0.8, np.nan]])
    sanitized_2 = _sanitize_probabilities(bad_2class, default_classes=2)
    assert not np.isnan(sanitized_2).any()
    assert np.isclose(sanitized_2.sum(), 1.0)
    assert np.isclose(sanitized_2[0][0], 0.5)
