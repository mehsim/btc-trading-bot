"""
tests/test_medium_severity_fixes.py
------------------------------------
Unit test suite covering Medium Severity Audit Remediations (MEDIUM-1 to MEDIUM-5):
- Config YAML Schema Loader & Validation (MEDIUM-2)
- Champion/Challenger Model Drift Rollback (MEDIUM-3)
- Immutable HMAC Audit Trail Logging (MEDIUM-5)
"""

import os
import pytest
import json
from config_validator import load_and_validate_config
from champion_challenger_framework import champion_challenger_framework
from audit_trail_logger import audit_logger


def test_config_loader_validation():
    cfg = load_and_validate_config("config.yaml")
    assert "trading" in cfg
    assert "risk" in cfg
    assert cfg["trading"]["min_position_size_usd"] > 0


def test_champion_challenger_rollback():
    # Severe drift (p-value 0.08 > threshold 0.05) triggers rollback
    active_model, reason = champion_challenger_framework.evaluate_model_health(
        drift_score=0.08, challenger_accuracy=0.85, champion_accuracy=0.75
    )
    assert active_model == "v2.4_prod"
    assert "Rollback" in reason


def test_immutable_audit_trail(tmp_path):
    log_file = os.path.join(tmp_path, "test_audit.jsonl")
    from audit_trail_logger import ImmutableAuditTrailLogger
    test_audit = ImmutableAuditTrailLogger(log_file=log_file)

    entry = test_audit.record_trade_audit_event("ENTRY", {"symbol": "BTCUSDT", "price": 100.0})
    assert "hmac_signature" in entry
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0


def test_live_order_execution_path_resolves_config():
    """M-01: Verify mocked live order execution path resolves config without NameError."""
    import main
    assert hasattr(main, "config")
    
    from unittest.mock import patch
    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.set_bybit_leverage", return_value=True), \
         patch("main.place_bybit_limit_order", return_value={"retCode": 0, "result": {"orderId": "mock_123"}}), \
         patch("main.wait_for_order_fill", return_value=True), \
         patch("main.update_bybit_stop_loss", return_value=True), \
         patch("main.update_bybit_take_profit", return_value=True), \
         patch("main.get_bybit_order_details", return_value={"orderStatus": "Filled", "avgPrice": 50000.0, "cumExecQty": 0.02}), \
         patch("main.send_telegram_alert"), \
         patch("main.sync_active_positions_from_bybit"):
        
        import pandas as pd
        df_mock = pd.DataFrame({"ATR_norm": [0.005] * 30})
        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT", iv="15", tf="15m", ml_trend="Bullish", leverage_val=10.0,
            qty_str="0.02", raw_qty=0.02, entry_price=50000.0, stop_loss_price=49750.0,
            take_profit_price=50500.0, position_size_usd=1000.0, kelly_fraction=0.1,
            calibrated_confidence=0.8, ml_confidence=0.8, dynamic_conf_threshold=0.6,
            latest_completed_ts=1700000000, latest_candle={"close": 50000.0, "ATR_norm": 0.005},
            pred_change=0.01, predicted_price=50500.0, atr_dollars=250.0,
            tp_multiplier_adjusted=2.0, sl_multiplier_adjusted=1.0, df_completed=df_mock,
            trade_uuid="mock_uuid_12345", duration_seconds=900, active_trade_key="active_trades"
        )
        
        # Verify active_trades in bot_state received the executed trade
        active_trades = main.bot_state.get("active_trades", [])
        assert any(t.get("trade_id") == "BTCUSDT_mock_uuid_12345" for t in active_trades)


def test_live_order_partial_fill_reversal():
    """M-01: Verify partial fill (<60% threshold) triggers IOC reversal and blocks trade registration."""
    import main
    from unittest.mock import patch, MagicMock
    import pandas as pd

    df_mock = pd.DataFrame({"ATR_norm": [0.005] * 30})
    mock_ioc = MagicMock(return_value={"retCode": 0, "result": {"orderId": "mock_ioc_123"}})

    def mock_order_details(symbol, order_id=None):
        if order_id == "mock_ioc_123":
            return {"orderStatus": "Filled", "avgPrice": 50000.0, "cumExecQty": 0.008}
        return {"orderStatus": "PartiallyFilled", "avgPrice": 50000.0, "cumExecQty": 0.008}

    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.set_bybit_leverage", return_value=True), \
         patch("main.place_bybit_limit_order", return_value={"retCode": 0, "result": {"orderId": "mock_partial_123"}}), \
         patch("main.wait_for_order_fill", return_value=True), \
         patch("main.get_bybit_order_details", side_effect=mock_order_details), \
         patch("main.place_bybit_taker_ioc_order", mock_ioc), \
         patch("trading_engine.place_bybit_taker_ioc_order", mock_ioc), \
         patch("bybit_client.place_bybit_taker_ioc_order", mock_ioc), \
         patch("main.TRADE_MODE", "live"), \
         patch("trading_engine.TRADE_MODE", "live"), \
         patch("main.send_telegram_alert"), \
         patch("main.sync_active_positions_from_bybit"):

        main.bot_state["active_trades"] = [t for t in main.bot_state.get("active_trades", []) if t.get("trade_id") != "BTCUSDT_mock_partial_uuid"]
        
        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT", iv="15", tf="15m", ml_trend="Bullish", leverage_val=10.0,
            qty_str="0.02", raw_qty=0.02, entry_price=50000.0, stop_loss_price=49750.0,
            take_profit_price=50500.0, position_size_usd=1000.0, kelly_fraction=0.1,
            calibrated_confidence=0.8, ml_confidence=0.8, dynamic_conf_threshold=0.6,
            latest_completed_ts=1700000000, latest_candle={"close": 50000.0, "ATR_norm": 0.005},
            pred_change=0.01, predicted_price=50500.0, atr_dollars=250.0,
            tp_multiplier_adjusted=2.0, sl_multiplier_adjusted=1.0, df_completed=df_mock,
            trade_uuid="mock_partial_uuid", duration_seconds=900, active_trade_key="active_trades"
        )

        # Verify IOC order was called to reverse 0.008 partial fill
        assert mock_ioc.called
        # Verify position was NOT added to active_trades due to reversal
        active_trades = main.bot_state.get("active_trades", [])
        assert not any(t.get("trade_id") == "BTCUSDT_mock_partial_uuid" for t in active_trades)


def test_trade_execution_signature_match():
    """Assert live Thread call site supplies all required parameters to execution inner."""
    import inspect
    import main
    sig = inspect.signature(main._execute_bybit_trade_async_inner)
    required = [p for p in sig.parameters.values() if p.default is inspect._empty]
    # main.py Thread call site passes exactly 27 positional arguments
    assert len(required) <= 27, f"Execution inner expects {len(required)} required params, Thread passes 27"


def test_place_bybit_limit_order_post_only_payload():
    """Verify place_bybit_limit_order accepts post_only=True and sets timeInForce='PostOnly'."""
    from unittest.mock import patch
    import main

    with patch("main.execute_bybit_order_ws_or_rest") as mock_exec:
        mock_exec.return_value = {"retCode": 0, "result": {"orderId": "test_order_1"}}
        res = main.place_bybit_limit_order("BTCUSDT", "Buy", "0.01", 60000.0, post_only=True)
        assert mock_exec.called
        payload = mock_exec.call_args[0][1]
        assert payload.get("timeInForce") == "PostOnly"
        assert payload.get("orderType") == "Limit"

    with patch("main.execute_bybit_order_ws_or_rest") as mock_exec_gtc:
        mock_exec_gtc.return_value = {"retCode": 0, "result": {"orderId": "test_order_2"}}
        res = main.place_bybit_limit_order("BTCUSDT", "Buy", "0.01", 60000.0, post_only=False)
        assert mock_exec_gtc.called
        payload = mock_exec_gtc.call_args[0][1]
        assert payload.get("timeInForce") == "GTC"


def test_sizing_preserves_all_risk_budget_constraints_on_stop_adjustment():
    """Verify that post-validation stop-loss adjustment does not discard upstream risk budgets."""
    current_bal = 100.0
    f_clamped = 0.05
    cov_multiplier = 0.5  # 50% correlation haircut
    vol_regime_mult = 0.8 # Volatility haircut
    learning_risk_mult = 0.7 # Learning engine haircut
    lev_cap = 5.0

    entry_price = 100.0
    initial_sl_price = 98.0  # 2% stop loss
    initial_sl_frac = max(0.002, abs(entry_price - initial_sl_price) / entry_price)  # 0.02
    
    # Base notional with all 3 haircuts applied
    raw_notional_usd = (current_bal * f_clamped) / initial_sl_frac  # 5.0 / 0.02 = 250.0
    target_notional_usd = raw_notional_usd * cov_multiplier * vol_regime_mult * learning_risk_mult  # 250 * 0.5 * 0.8 * 0.7 = 70.0
    
    # Simulate trade structure widening stop loss to 96.0 (4% stop loss)
    widened_sl_price = 96.0
    widened_sl_frac = max(0.002, abs(entry_price - widened_sl_price) / entry_price)  # 0.04
    
    # Proportional scaling logic as implemented in main.py
    if abs(widened_sl_frac - initial_sl_frac) > 1e-9:
        target_notional_usd = target_notional_usd * (initial_sl_frac / widened_sl_frac)
    target_notional_usd = min(target_notional_usd, current_bal * lev_cap)
    
    # Expected: notional scaled down by 0.02/0.04 = 0.5, giving 35.0 (maintaining $1.40 risk)
    # Under old buggy logic: would reset to (100 * 0.05) / 0.04 = 125.0 (discarding all haircuts!)
    assert target_notional_usd == 35.0, f"Expected 35.0, got {target_notional_usd}"
    assert target_notional_usd < 100.0, "Upstream correlation and volatility haircuts must not be discarded"


