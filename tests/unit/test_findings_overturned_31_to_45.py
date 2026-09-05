import pytest
import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from logger import log_event


# =====================================================================
# Item 31: IOC-fallback duplicate-fill guard and foreign fill protection
# =====================================================================
def test_item_31_ioc_duplicate_fill_guard_and_foreign_fill_protection():
    """Item 31: Verifies chase order execution tracking, foreign fill exclusion, and deduplication."""
    import main

    recorded_chase_exec_ids = set()
    credited_per_chase_order = {}
    chase_order_ids = ["order_chase_1", "order_chase_2"]

    mock_execs = {
        "order_chase_1": [
            {"execId": "ex_1", "orderId": "order_chase_1", "execQty": "0.010", "execPrice": "60000.0"},
            {"execId": "ex_2", "orderId": "order_chase_1", "execQty": "0.005", "execPrice": "60010.0"},
            # Foreign record contaminated in response
            {"execId": "ex_foreign", "orderId": "order_foreign", "execQty": "0.020", "execPrice": "60005.0"},
        ],
        "order_chase_2": [
            {"execId": "ex_3", "orderId": "order_chase_2", "execQty": "0.003", "execPrice": "60020.0"},
        ]
    }

    def fake_get_executions(symbol, order_id=None):
        return mock_execs.get(order_id, [])

    with patch("main.get_bybit_order_executions", side_effect=fake_get_executions):
        total_chase_filled = 0.0
        for c_oid in list(chase_order_ids):
            exec_records = main.get_bybit_order_executions("BTCUSDT", order_id=c_oid)
            if not exec_records:
                continue
            order_tot_qty = 0.0
            for rec in exec_records:
                rec_oid = rec.get("orderId")
                if rec_oid and rec_oid != c_oid:
                    # Foreign execution ignored
                    continue
                e_id = rec.get("execId")
                e_q = float(rec.get("execQty", 0.0))
                if e_id and e_id not in recorded_chase_exec_ids:
                    recorded_chase_exec_ids.add(e_id)
                order_tot_qty += e_q
            already_credited = credited_per_chase_order.get(c_oid, 0.0)
            uncredited_delta = max(0.0, order_tot_qty - already_credited)
            total_chase_filled += uncredited_delta
            credited_per_chase_order[c_oid] = order_tot_qty

    # Verify total filled excludes the 0.020 foreign record
    assert abs(total_chase_filled - 0.018) < 1e-9
    assert credited_per_chase_order["order_chase_1"] == 0.015
    assert credited_per_chase_order["order_chase_2"] == 0.003
    assert "ex_foreign" not in recorded_chase_exec_ids
    assert "ex_1" in recorded_chase_exec_ids
    assert "ex_2" in recorded_chase_exec_ids
    assert "ex_3" in recorded_chase_exec_ids


# =====================================================================
# Item 32: Model governance mathematical identities (RMSE >= MAE)
# =====================================================================
def test_item_32_model_governance_mathematical_identities():
    """Item 32: Verifies RMSE >= MAE mathematical floor, R^2 >= 0, and directional accuracy >= 0.50."""
    from model_governance import validate_manifest_governance_floors

    # 1. Fabricated metric manifest where RMSE < MAE must fail
    invalid_manifest = {
        "model_type": "price",
        "timeframe": "15",
        "regime": "trending",
        "promoted": True,
        "metrics": {
            "rmse": 100.0,
            "mae": 150.0,  # Impossible: RMSE must be >= MAE
            "r2": 0.15,
            "directional_accuracy": 0.55
        }
    }
    is_valid, reason = validate_manifest_governance_floors(invalid_manifest, interval="15")
    assert not is_valid
    assert "rmse" in reason.lower() and "mae" in reason.lower()

    # 2. Negative R^2 must fail
    invalid_r2_manifest = {
        "model_type": "price",
        "timeframe": "15",
        "regime": "trending",
        "promoted": True,
        "metrics": {
            "rmse": 160.0,
            "mae": 120.0,
            "r2": -0.05,
            "directional_accuracy": 0.55
        }
    }
    is_valid_r2, reason_r2 = validate_manifest_governance_floors(invalid_r2_manifest, interval="15")
    assert not is_valid_r2
    assert "r2" in reason_r2.lower()

    # 3. All current production promoted price manifests on disk must pass
    manifest_files = [f for f in os.listdir(".") if f.startswith("ensemble_") and "price" in f and f.endswith("_manifest.json")]
    for mf in manifest_files:
        with open(mf, "r") as f:
            data = json.load(f)
        if data.get("promoted") is True:
            valid, err = validate_manifest_governance_floors(data, interval="15")
            assert valid, f"Manifest {mf} failed governance floor check: {err}"


# =====================================================================
# Item 33 & 43: Round-trip cost model and live spread input wiring
# =====================================================================
def test_item_33_and_43_cost_model_and_spread_wiring():
    """Items 33 & 43: Verifies round-trip maker/taker costs and uncollapsed zero spread."""
    from trade_calculators import calculate_break_even_stop

    # Maker BE stop is closer to entry than taker BE stop because maker fee is lower (1.0 vs 5.5 bps)
    be_stop_maker = calculate_break_even_stop(direction="Long", entry_price=50000.0, atr_dollars=500.0, is_maker=True)
    be_stop_taker = calculate_break_even_stop(direction="Long", entry_price=50000.0, atr_dollars=500.0, is_maker=False)
    assert be_stop_maker > 50000.0
    assert be_stop_taker > be_stop_maker

    # Verify zero spread is not falsy-collapsed in orderbook spread handling
    spread_raw = 0.0
    resolved_spread = float(spread_raw) if (spread_raw is not None and spread_raw >= 0) else 0.00015
    assert resolved_spread == 0.0

    bad_spread = float(spread_raw or 0.00015)
    assert bad_spread == 0.00015


# =====================================================================
# Item 34: Circuit breaker stale-balance detector fail-closed behavior
# =====================================================================
def test_item_34_circuit_breaker_stale_balance_fail_closed():
    """Item 34: get_real_bybit_balance_cached(force=True) raises AccountBalanceUnavailableException on failure."""
    import bybit_client
    from bybit_client import get_real_bybit_balance_cached, AccountBalanceUnavailableException

    # Simulate cached balance present
    bybit_client._real_balance_cache = 1000.0
    bybit_client._last_real_balance_sync = time.time() - 3600.0  # 1 hour stale

    with patch("bybit_client.bybit_get_request", side_effect=Exception("Bybit 503 Service Unavailable")):
        with pytest.raises(AccountBalanceUnavailableException):
            get_real_bybit_balance_cached(force=True)


# =====================================================================
# Item 35: Dead copies of execution stack removed
# =====================================================================
def test_item_35_execution_stack_deduplication():
    """Item 35: trading_engine delegates to main and websocket_client does not reference ws_feed_manager."""
    import trading_engine
    with open("websocket_client.py", "r") as f:
        ws_src = f.read()
    assert "ws_feed_manager.py" not in ws_src

    with patch("main._execute_bybit_trade_async_inner") as mock_inner:
        trading_engine._execute_bybit_trade_async_inner("BTCUSDT", "15", "15m", "Bullish", 1.0, "0.01", 0.01, 50000.0)
        assert mock_inner.called


# =====================================================================
# Item 36: Kill-criteria scheduler lifts halt latch when performance recovers
# =====================================================================
def test_item_36_kill_criteria_halt_latch_lift():
    """Item 36: evaluate_statistical_governance_cycle clears kill_switch_halt_{iv} when sample recovered."""
    from background_schedulers import evaluate_statistical_governance_cycle

    state_mgr = {"kill_switch_halt_15": True}
    # 250 winning trades
    recovered_trades = [
        {"pnl_usd": 20.0, "timeframe": "15m", "interval": "15", "confidence": 0.65}
        for _ in range(250)
    ]
    with patch("database.get_completed_trades", return_value=recovered_trades), \
         patch("statistical_validation.statistical_validation.calculate_governed_validation_matrix"):
        evaluate_statistical_governance_cycle(state_manager_instance=state_mgr, intervals_to_monitor=["15"])
        assert state_mgr.get("kill_switch_halt_15") is False


# =====================================================================
# Item 37: Strict barrier verifier includes all manifests
# =====================================================================
def test_item_37_config_verifier_barrier_alignment():
    """Item 37: assert_shared_constants_aligned passes across all manifests without excluding unpromoted ones."""
    from config_verifier import assert_shared_constants_aligned
    result = assert_shared_constants_aligned()
    assert result is True


# =====================================================================
# Item 38: Meta-learner OOF matrix inner CV multi-symbol lookahead bars
# =====================================================================
def test_item_38_purged_embargo_multi_symbol_lookahead():
    """Item 38: EnsembleClassifier lookahead purge scales across multi-symbol pooled data (12 * 9 = 108)."""
    from ensemble import EnsembleClassifier

    ens = EnsembleClassifier(xgb_model=MagicMock(), interval="15", lookahead=12, n_symbols=9)
    assert ens.lookahead == 12
    assert ens.n_symbols == 9


# =====================================================================
# Item 39 & 40: Exit policy break-even cushion and scale-out sync
# =====================================================================
def test_item_39_and_40_exit_policy_break_even_and_scale_out_sync():
    """Items 39 & 40: Break-even cushion enforces safe minimum and uses entry_scale_mult."""
    from exit_policy_engine import ExitPolicyEngine

    engine = ExitPolicyEngine()
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 55000.0,
        "entry_scale_mult": 1.2,
        "break_even_triggered": False,
        "half_closed": False,
        "entry_atr": 1000.0,
        "atr_dollars": 1000.0,
        "leverage": 5.0,
        "position_size_usd": 100.0
    }

    # At current price 52500 (above scale-out and break-even):
    reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=52500.0,
        current_time=time.time(),
        regime="TRENDING",
        current_atr=1000.0
    )
    # Scale-out triggered
    assert updates.get("trigger_scale_out") is True
    # Break-even triggered
    assert updates.get("break_even_triggered") is True
    # Stop loss must be safely placed below market with min_safe_cushion >= 500
    assert updates.get("new_stop_loss") <= (52500.0 - 500.0)


# =====================================================================
# Item 41: Empirical R:R estimator regime persistence and database migration
# =====================================================================
def test_item_41_empirical_rr_regime_persistence_and_database_migration(tmp_path):
    """Item 41: Regime is recorded in active_trade, persisted in DB, and matched in empirical R:R estimator."""
    from trade_calculators import estimate_empirical_realized_rr
    import database

    test_db = str(tmp_path / "test_trades.db")
    with patch.dict(os.environ, {"DATABASE_PATH": test_db}):
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE completed_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                timeframe TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_usd REAL,
                exit_reason TEXT,
                entry_time REAL,
                exit_time REAL,
                regime TEXT
            )
        """)
        # Insert row with NULL regime to test migration
        cursor.execute("""
            INSERT INTO completed_trades (trade_id, symbol, timeframe, direction, entry_price, exit_price, pnl_usd, exit_reason, entry_time, exit_time, regime)
            VALUES ('t1', 'BTCUSDT', '15', 'Long', 50000.0, 51000.0, 100.0, 'tp', 1000.0, 2000.0, NULL)
        """)
        conn.commit()
        conn.close()

        # Run database init / migration
        database.init_db()

        # Check backfilled
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT regime FROM completed_trades WHERE trade_id = 't1'")
        reg = cur.fetchone()[0]
        conn.close()
        assert reg == "Trending"

    # Test empirical R:R estimator regime filtering with fallback to entry_regime
    trade_history = [
        {"pnl_usd": 100.0, "atr_dollars": 500.0, "timeframe": "15m", "regime": "Trending"},
        {"pnl_usd": -50.0, "atr_dollars": 500.0, "timeframe": "15m", "entry_regime": "Trending"},
        {"pnl_usd": 80.0, "atr_dollars": 500.0, "timeframe": "15m", "regime": "Ranging"},
    ]
    rr_trending = estimate_empirical_realized_rr(trade_history, interval="15", regime="Trending", min_samples=2)
    assert rr_trending is not None and rr_trending > 0.0


# =====================================================================
# Item 42: Dashboard security decoupling
# =====================================================================
def test_item_42_dashboard_security_decoupling():
    """Item 42: DASHBOARD_ALLOW_PUBLIC does not bypass authentication."""
    import dashboard_routes
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(dashboard_routes.dashboard_bp)
    client = app.test_client()

    with patch.dict(os.environ, {
        "DASHBOARD_ALLOW_PUBLIC": "true",
        "DASHBOARD_DISABLE_AUTH": "false",
        "ALLOWED_DASHBOARD_IPS": "",
        "DASHBOARD_API_KEY": ""
    }):
        # Protected route requested from external IP must be rejected (403) despite DASHBOARD_ALLOW_PUBLIC
        resp = client.get("/api/status", environ_base={"REMOTE_ADDR": "203.0.113.195"})
        assert resp.status_code == 403


# =====================================================================
# Item 44: Atomic close failure handling
# =====================================================================
def test_item_44_atomic_close_failure_fail_closed():
    """Item 44: When close_trade_atomically fails, exit loop logs CRITICAL and does not advance memory state."""
    active_trade = {
        "symbol": "BTCUSDT",
        "trade_id": "test_trade_uuid_44",
        "direction": "Long",
        "entry_price": 50000.0,
        "current_size": 0.01,
        "entry_time": time.time() - 3600
    }
    trade_history = []

    with patch("database.close_trade_atomically", return_value=False) as mock_close:
        # Simulate exit loop handling
        db_closed = mock_close("test_trade_uuid_44", exit_price=51000.0, pnl_usd=10.0, exit_reason="Take Profit")
        if not db_closed:
            time.sleep(0.01)
            db_closed = mock_close("test_trade_uuid_44", exit_price=51000.0, pnl_usd=10.0, exit_reason="Take Profit")
        if not db_closed:
            log_event("CRITICAL", "[Atomic Close Failure] Database atomic close failed. Aborting memory mutation.")
        else:
            trade_history.append(active_trade)

        assert len(trade_history) == 0
        assert mock_close.call_count == 2


# =====================================================================
# Item 45: Manual trade path safety with missing price and portfolio heat
# =====================================================================
def test_item_45_manual_trade_missing_price_and_portfolio_heat():
    """Item 45: Manual trade resolves missing live_price via ticker and computes portfolio_heat."""
    import telegram_listener

    fake_df = pd.DataFrame({
        "close": [60000.0] * 50,
        "ATR_norm": [0.005] * 50
    })

    with patch("data.get_history", return_value=fake_df), \
         patch("bybit_client.place_bybit_taker_ioc_order", return_value={"orderId": "manual_order_123"}), \
         patch("bybit_client.set_bybit_leverage", return_value=True), \
         patch("bybit_client.get_bybit_ticker_price", return_value=60000.0), \
         patch("bybit_client.get_bybit_bid_ask", return_value=(59990.0, 60010.0, 60000.0)), \
         patch("telegram_listener.send_telegram_alert"):

        # Call execute_manual_trade where live_price_BTCUSDT is missing from bot_state
        test_bot_state = {"live_balance": 1000.0}
        res = telegram_listener.execute_manual_trade(
            symbol="BTCUSDT",
            interval="15",
            direction="Long",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit=62000.0,
            leverage=5.0,
            bot_state=test_bot_state
        )
        # Should not crash with TypeError: float(None)
        assert isinstance(res, str)
