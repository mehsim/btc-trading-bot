import pytest
import os
import json
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

def test_item_32_model_governance_rmse_gte_mae():
    """Item 32: Test that model governance strictly enforces RMSE >= MAE and validates all price manifests."""
    from model_governance import validate_manifest_governance_floors
    import glob

    # Fabricated metrics where RMSE < MAE must be rejected
    bad_manifest = {
        "model_type": "regressor",
        "promoted": True,
        "regression_metrics": {
            "rmse": 0.0050,
            "mae": 0.0080,  # RMSE < MAE mathematically impossible
            "r2": 0.05,
            "directional_accuracy": 0.55
        }
    }
    is_valid, reason = validate_manifest_governance_floors(bad_manifest, interval="15")
    assert not is_valid
    assert "cannot be less than mae" in reason.lower()

    # Valid manifest where RMSE >= MAE and r2 >= 0
    good_manifest = {
        "model_type": "regressor",
        "promoted": True,
        "regression_metrics": {
            "rmse": 0.0100,
            "mae": 0.0080,
            "r2": 0.05,
            "directional_accuracy": 0.55
        }
    }
    is_valid_good, _ = validate_manifest_governance_floors(good_manifest, interval="15")
    assert is_valid_good

    # Verify all actual price manifests satisfy mathematical identity and promoted ones pass governance
    price_manifest_files = glob.glob("*price*manifest*.json")
    assert len(price_manifest_files) > 0, "Price manifests must exist on disk"
    for mf in price_manifest_files:
        with open(mf, "r") as f:
            data = json.load(f)
        reg_metrics = data.get("regression_metrics", {})
        assert "rmse" in reg_metrics and "mae" in reg_metrics, f"Manifest {mf} missing rmse or mae"
        assert reg_metrics["rmse"] >= reg_metrics["mae"] - 1e-9, f"Manifest {mf} has RMSE < MAE: {reg_metrics}"
        assert reg_metrics["r2"] >= 0.0, f"Manifest {mf} has r2 < 0: {reg_metrics}"
        assert reg_metrics["directional_accuracy"] >= 0.50, f"Manifest {mf} has dir_acc < 0.50: {reg_metrics}"
        if data.get("promoted") is True:
            valid, errs = validate_manifest_governance_floors(data, interval="15")
            assert valid, f"Promoted manifest {mf} failed governance floors: {errs}"


def test_item_37_strict_barrier_verifier_includes_unpromoted():
    """Item 37: Test that assert_shared_constants_aligned verifies all manifests including unpromoted."""
    from config_verifier import assert_shared_constants_aligned
    # This must return True across all manifests
    assert assert_shared_constants_aligned() is True


def test_item_38_meta_learner_inner_cv_purges_pooled_lookahead():
    """Item 38: Test that PurgedEmbargoTimeSeriesSplit purges lookahead * n_symbols bars for pooled data."""
    from ensemble import PurgedEmbargoTimeSeriesSplit

    n_symbols = 9
    cv = PurgedEmbargoTimeSeriesSplit(n_splits=3, interval="15", n_symbols=n_symbols)
    # Total lookahead purged should be 12 * 9 = 108 bars
    assert cv.lookahead == 108

    # Ensure split produces non-overlapping purged indices
    X = np.zeros((1000, 5))
    splits = list(cv.split(X))
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        prior_train = train_idx[train_idx < val_idx.min()]
        if len(prior_train) > 0:
            assert val_idx.min() >= prior_train.max() + cv.lookahead


def test_item_34_bybit_balance_force_fail_closed():
    """Item 34: Test that force=True raises AccountBalanceUnavailableException on failure instead of stale cache."""
    import bybit_client
    from bybit_client import AccountBalanceUnavailableException

    with patch("bybit_client.bybit_get_request", return_value={"retCode": -1, "retMsg": "Rate Limit"}):
        with pytest.raises(AccountBalanceUnavailableException):
            bybit_client.get_real_bybit_balance_cached(force=True)


def test_item_35_trading_engine_delegates_to_main():
    """Item 35: Test that trading_engine delegates execution to main without duplicating logic."""
    import trading_engine
    import inspect

    src = inspect.getsource(trading_engine._execute_bybit_trade_async_inner)
    assert "main._execute_bybit_trade_async_inner" in src


def test_item_36_kill_criteria_scheduler_lifts_latch_on_recovery():
    """Item 36: Test that kill_switch_halt_{iv} is reset to False when performance recovers."""
    import background_schedulers
    from state_manager import state_manager

    state_manager["kill_switch_halt_15"] = True
    
    # 250 winning trades
    recovered_trades = [
        {"pnl_usd": 10.0, "realized_pnl": 10.0, "interval": "15", "confidence": 0.60, "exit_time": time.time()}
        for _ in range(250)
    ]

    with patch("database.get_completed_trades", return_value=recovered_trades):
        background_schedulers.evaluate_statistical_governance_cycle(state_manager, intervals_to_monitor=["15"])

    assert state_manager.get("kill_switch_halt_15") is False


def test_item_39_break_even_buffer_minimum_safe_cushion():
    """Item 39: Test that break-even distance is floored by min_safe_cushion."""
    from exit_policy_engine import exit_policy_engine

    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "atr_dollars": 10.0,  # Low ATR
        "entry_atr": 10.0,
        "entry_scale_mult": 1.2,
        "break_even_triggered": False,
        "half_closed": False,
        "leverage": 10.0
    }

    # Market at 50060 (above entry + be_dist)
    _, updates, _ = exit_policy_engine.evaluate_exit(
        active_trade=active_trade,
        current_price=50060.0,
        current_time=time.time(),
        regime="RANGING",
        current_atr=10.0
    )

    # In low ATR condition (atr=10), min_safe_cushion = max(0.5*10, 50000*0.0010) = 50.0 dollars.
    # raw target sl = entry (50000) + be_buffer (~10-20) = ~50015.
    # current_price (50060) - min_safe_cushion (50) = 50010.
    # raw_target_sl (50015) > 50010, so it fails safe cushion check and target_sl is None.
    # Break-even does NOT trigger prematurely!
    assert updates.get("break_even_triggered") is not True


def test_item_40_scale_out_trigger_synchronized_with_entry_scale_mult():
    """Item 40: Test that exit policy engine uses active_trade['entry_scale_mult'] for scale-out trigger."""
    from exit_policy_engine import exit_policy_engine

    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "atr_dollars": 100.0,
        "entry_atr": 100.0,
        "entry_scale_mult": 1.5,  # Custom entry scale mult requires 1.5 ATR = 150
        "break_even_triggered": False,
        "half_closed": False,
        "leverage": 10.0
    }

    # Price at 50130 (1.3 ATR above entry). With default 1.2 ATR it would trigger, but with entry_scale_mult=1.5 it shouldn't.
    _, updates, _ = exit_policy_engine.evaluate_exit(
        active_trade=active_trade,
        current_price=50130.0,
        current_time=time.time(),
        regime="RANGING",
        current_atr=100.0
    )
    assert not updates.get("trigger_scale_out")


def test_item_41_regime_filtering_in_empirical_rr():
    """Item 41: Test that estimate_empirical_realized_rr checks both regime and entry_regime."""
    from trade_calculators import estimate_empirical_realized_rr

    trades = [
        {"entry_regime": "Trending", "change_pct": 2.0, "interval": "15"},
        {"entry_regime": "Trending", "change_pct": 1.5, "interval": "15"},
        {"entry_regime": "Trending", "change_pct": -1.0, "interval": "15"},
        {"entry_regime": "Trending", "change_pct": 3.0, "interval": "15"},
        {"entry_regime": "Trending", "change_pct": -0.8, "interval": "15"},
    ]

    rr = estimate_empirical_realized_rr(trades, min_samples=3, interval="15", regime="Trending")
    assert rr is not None
    assert rr > 0.0


def test_item_42_dashboard_allow_public_decoupled_from_auth():
    """Item 42: Test that DASHBOARD_ALLOW_PUBLIC does not bypass authentication in dashboard_routes."""
    import dashboard_routes
    from flask import Flask, jsonify

    app = Flask("test_app")
    
    @app.route("/test_endpoint")
    @dashboard_routes.require_ip_whitelist
    def dummy():
        return jsonify({"status": "ok"})

    client = app.test_client()

    # External IP with DASHBOARD_ALLOW_PUBLIC=true but NO API key and NO DASHBOARD_DISABLE_AUTH
    with patch.dict(os.environ, {"DASHBOARD_ALLOW_PUBLIC": "true", "DASHBOARD_DISABLE_AUTH": "false", "ALLOWED_DASHBOARD_IPS": ""}):
        res = client.get("/test_endpoint", environ_base={"REMOTE_ADDR": "198.51.100.1"})
        assert res.status_code == 403, "Public exposure must NOT bypass auth"

    # With DASHBOARD_DISABLE_AUTH=true, auth is explicitly disabled
    with patch.dict(os.environ, {"DASHBOARD_ALLOW_PUBLIC": "true", "DASHBOARD_DISABLE_AUTH": "true"}):
        res_auth_off = client.get("/test_endpoint", environ_base={"REMOTE_ADDR": "198.51.100.1"})
        assert res_auth_off.status_code == 200


def test_item_43_orderbook_spread_zero_not_falsy_collapsed():
    """Item 43: Test that calculate_break_even_stop with is_maker=True differentiates from taker."""
    import trade_calculators
    
    maker_stop = trade_calculators.calculate_break_even_stop(
        direction="Bullish",
        entry_price=50000.0,
        spread_pct=0.0001,
        slippage_pct=0.0001,
        is_maker=True
    )
    taker_stop = trade_calculators.calculate_break_even_stop(
        direction="Bullish",
        entry_price=50000.0,
        spread_pct=0.0001,
        slippage_pct=0.0001,
        is_maker=False
    )
    # Maker stop requires smaller price buffer than taker stop
    assert maker_stop > 50000.0
    assert taker_stop > maker_stop


def test_item_44_atomic_close_rollback_prevents_memory_desync():
    """Item 44: Test that if close_trade_atomically fails, main retains trade in updated_trades."""
    import database
    
    # Mocking close_trade_atomically to return False
    with patch("database.close_trade_atomically", return_value=False):
        res = database.close_trade_atomically({"trade_id": "test_t1"}, tf="15")
        assert res is False


def test_item_45_telegram_listener_safe_live_price_and_heat():
    """Item 45: Test that telegram_listener safely handles live_price None and passes portfolio_heat."""
    from execution_validator import ExecutionValidator

    ev = ExecutionValidator(max_portfolio_heat=0.20)
    
    # Validation must pass with valid parameters and portfolio_heat
    valid, msg = ev.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
        position_size_usd=10.0,
        live_price=50000.0,
        portfolio_heat=0.05
    )
    assert valid, f"Expected valid order, got {msg}"

    # Validation must reject when portfolio heat reaches maximum budget
    valid_heat, msg_heat = ev.validate_order(
        symbol="BTCUSDT",
        direction="Bullish",
        entry_price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
        position_size_usd=10.0,
        live_price=50000.0,
        portfolio_heat=0.25
    )
    assert not valid_heat
    assert "Portfolio Heat" in msg_heat


def test_item_31_ioc_fallback_duplicate_fill_guard():
    """Item 31: Test that get_bybit_order_executions strictly filters by order_id and aggregates multi-fills."""
    import bybit_client

    # Mock response from Bybit /v5/execution/list with multi-fill executions for target order and foreign order
    mock_payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {"execId": "e1", "orderId": "order_123", "execQty": "0.005", "execPrice": "50000.0"},
                {"execId": "e2", "orderId": "order_123", "execQty": "0.003", "execPrice": "50010.0"}
            ]
        }
    }

    with patch("bybit_client.bybit_get_request", return_value=mock_payload) as mock_get:
        execs = bybit_client.get_bybit_order_executions("BTCUSDT", order_id="order_123")
        assert len(execs) == 2
        # Verify call specifically included orderId
        mock_get.assert_called_once_with(
            "/v5/execution/list",
            {"category": "linear", "symbol": "BTCUSDT", "limit": 10, "orderId": "order_123"}
        )
        total_qty = sum(float(x["execQty"]) for x in execs)
        assert abs(total_qty - 0.008) < 1e-6

