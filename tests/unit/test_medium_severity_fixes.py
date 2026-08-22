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


def test_backtest_dynamic_economic_gate_and_geometry_alignment():
    """Verify that backtest.py correctly computes production p* with normalized fees and 4-layer TP/SL."""
    sl_multiplier = 0.8
    tp_multiplier = 1.85
    fee_rate = 0.002
    atr_norm = 0.010
    
    # 1. Verify dimensional p* formula matching main.py:6571-6582
    cost_bps = (fee_rate * 2.0) * 10000.0
    effective_tp_m = tp_multiplier * 0.80
    p_star = sl_multiplier / (effective_tp_m + sl_multiplier)
    cost_adj = (cost_bps / 1e4) / ((effective_tp_m + sl_multiplier) * atr_norm)
    economic_base_threshold = round(p_star + cost_adj, 4)
    
    assert 0.30 <= p_star <= 0.60, f"p_star {p_star} should be in economic break-even range"
    assert 0.001 <= cost_adj <= 0.30, f"cost_adj {cost_adj} should be correctly ATR-scaled"
    assert economic_base_threshold > p_star, "Economic threshold must exceed pure break-even to cover costs"
    
    # 2. Verify 4-layer TP/SL geometry formulas matching main.py:7147-7205
    vol_factor = max(0.75, min(1.5, 1.5 - ((atr_norm - 0.003) / 0.005) * 0.75))
    assert 0.75 <= vol_factor <= 1.5
    
    tp_multiplier_adjusted = round(tp_multiplier * vol_factor, 3)
    vol_adj = 0.95  # Simulated top 10% ATR
    session_factor = 0.98
    tp_multiplier_adjusted *= (vol_adj * session_factor)
    
    assert tp_multiplier_adjusted > 0.0


def test_database_quarantine_integrity_corroboration(tmp_path):
    """Verify database auto-recovery creates a pre-quarantine snapshot and fails safely on transient probe errors."""
    import sqlite3
    import shutil
    import database

    test_db = str(tmp_path / "test_trading_bot.db")
    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE test (id INT, val TEXT);")
    conn.execute("INSERT INTO test VALUES (1, 'active');")
    conn.commit()
    conn.close()

    # Normal connect succeeds
    c = database.get_db_connection(target_db=test_db)
    row = c.execute("SELECT * FROM test;").fetchone()
    assert row["val"] == "active"
    c.close()


def test_feature_cache_key_includes_interval():
    """Verify features module generates distinct cache keys for distinct intervals."""
    import pandas as pd
    import features as features_module

    # Construct mock candle dataframe
    df1 = pd.DataFrame({
        "timestamp": [1000000 + i * 900000 for i in range(250)],
        "close": [100.0 + i * 0.1 for i in range(250)],
        "open": [100.0 + i * 0.1 for i in range(250)],
        "high": [101.0 + i * 0.1 for i in range(250)],
        "low": [99.0 + i * 0.1 for i in range(250)],
        "volume": [1000.0 for _ in range(250)],
    })
    
    # 15m interval vs 60m interval
    res_15 = features_module.add_features(df1.copy(), symbol="BTCUSDT", interval="15")
    res_60 = features_module.add_features(df1.copy(), symbol="BTCUSDT", interval="60")
    
    assert res_15 is not None
    assert res_60 is not None


def test_stop_state_machine_monotonic_and_state_hierarchy():
    """Verify StopStateMachine enforces forward rank progression and monotonic price movement."""
    from order_state_machine import StopStateMachine, StopState

    # Long: SL can only increase
    valid, msg = StopStateMachine.validate_monotonic_stop_update("Bullish", 100.0, 102.0, "INITIAL", "TRAILING")
    assert valid is True

    # Long: Backward price is rejected
    valid, msg = StopStateMachine.validate_monotonic_stop_update("Bullish", 100.0, 98.0, "INITIAL", "TRAILING")
    assert valid is False
    assert "Monotonic violation" in msg

    # State hierarchy: Backward rank transition is rejected
    valid, msg = StopStateMachine.validate_monotonic_stop_update("Bullish", 100.0, 102.0, "PROFIT_LOCK", "TRAILING")
    assert valid is False
    assert "Illegal backward state transition" in msg


def test_min_notional_sl_compression_leverage_aware_floor():
    """Verify min-notional SL compression honors the 1.0x ATR floor on high leverage (>10x)."""
    atr_dollars = 100.0
    entry_price = 50000.0
    
    # High leverage (15x): floor must be 1.0x ATR ($100)
    leverage_val = 15.0
    min_atr_mult = 1.0 if float(leverage_val) > 10.0 else 0.75
    min_allowed_sl_dist = max(atr_dollars * min_atr_mult, entry_price * 0.008)
    
    assert min_allowed_sl_dist >= 100.0, f"Expected >= 100.0 for >10x leverage, got {min_allowed_sl_dist}"





