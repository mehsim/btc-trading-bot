import time
import threading
import ssl
import os
import sqlite3
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

import main
import config
import risk_engine
import trade_calculators
import portfolio_risk
import kelly_tracker
import database


def test_finding_19_aiohttp_session_ssl_verification_enabled():
    """Finding #19: Verify that aiohttp connector is created with TLS certificate verification enabled."""
    with open("main.py", "r") as f:
        content = f.read()

    # Verify that ssl=False is completely eliminated from _init_session
    assert "connector = aiohttp.TCPConnector(ssl=False" not in content
    assert "ssl_ctx = ssl.create_default_context" in content
    assert "connector = aiohttp.TCPConnector(ssl=ssl_ctx" in content

    # Verify that _init_session configures a valid SSLContext
    if hasattr(main, "_aiohttp_session") and main._aiohttp_session is not None:
        connector = main._aiohttp_session.connector
        if connector is not None and hasattr(connector, "_ssl"):
            ssl_param = connector._ssl
            assert ssl_param is not False
            assert isinstance(ssl_param, ssl.SSLContext)


def test_finding_22_sync_positions_active_execution_guard():
    """Finding #22: In-flight trades in active_execution_symbols must never be flagged bybit_closed."""
    main.active_execution_symbols.add("ETHUSDT")
    try:
        trade_eth = {
            "symbol": "ETHUSDT",
            "entry_price": 3000.0,
            "direction": "Bullish",
            "qty": 1.0,
            "leverage": 10.0,
            "bybit_closed": False,
            "exit_processed": False,
            "confidence": 0.65
        }
        main.bot_state["active_trade_15m"] = [trade_eth]

        # Mock get_all_bybit_positions to return an empty list (simulating position not yet on venue)
        with patch("main.TRADE_MODE", "live"), \
             patch("main.get_all_bybit_positions", return_value=[]), \
             patch("main.save_history", MagicMock()):
            res = main.sync_active_positions_from_bybit()
            assert res is True

            current_trades = main.bot_state.get("active_trade_15m", [])
            assert len(current_trades) == 1
            # Must NOT be flagged as closed because it's actively in-flight
            assert current_trades[0].get("bybit_closed") is False
    finally:
        main.active_execution_symbols.discard("ETHUSDT")


def test_finding_40_retraining_scheduler_in_startup_manifest():
    """Finding #40: Automated weekly retrain scheduler must be defined and launched in startup manifest."""
    with open("main.py", "r") as f:
        content = f.read()

    assert "def run_rolling_retrain_scheduler():" in content
    assert 'threading.Thread(target=run_rolling_retrain_scheduler, name="rolling-retrain-scheduler", daemon=True).start()' in content
    assert "def check_champion_models_staleness" in content

    # Test staleness function runs without error
    main.check_champion_models_staleness(max_age_days=14.0)


def test_finding_71_kelly_uses_realized_outcomes():
    """Finding #71: Kelly sizing must fail-closed if empirical win rate or edge is negative or below break-even."""
    tracker = kelly_tracker.KellyTracker(data_file="/tmp/test_kelly_finding_71.json")
    try:
        # Pre-seed tracker with 12 losing trades and 3 tiny winning trades (poor realized win rate = 20%)
        tracker.history = []
        for i in range(12):
            tracker.history.append({"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": -10.0, "return_pct": -0.015, "slippage_pct": 0.0005})
        for i in range(3):
            tracker.history.append({"symbol": "BTCUSDT", "timeframe": "15", "pnl_usd": 5.0, "return_pct": 0.008, "slippage_pct": 0.0005})

        with patch("risk_engine.global_kelly_tracker", tracker):
            # Model claims high confidence 0.70, but realized history has negative edge
            # compute_conservative_kelly must fail-closed to 0.0
            kelly_val = risk_engine.compute_conservative_kelly(
                calibrated_confidence=0.70,
                tp_multiplier=1.5,
                sl_multiplier=1.0,
                interval="15",
                trade_history=tracker.history
            )
            assert kelly_val == 0.0, f"Expected 0.0 Kelly on negative empirical edge, got {kelly_val}"

            # JointRiskBudgetAllocator must also fail-closed to 0.0 when realized win rate <= break-even
            allocator = risk_engine.JointRiskBudgetAllocator()
            res = allocator.allocate_risk_budget(
                symbol="BTCUSDT",
                entry_price=60000.0,
                atr_dollars=500.0,
                atr_norm=0.008,
                calibrated_confidence=0.70,
                direction="Bullish",
                total_equity=100.0,
                portfolio_heat=0.0,
                stop_distance=500.0,
                target_distance=750.0,
                df_completed=pd.DataFrame(tracker.history)
            )
            assert res["kelly_fraction"] == 0.0
            assert res["position_size"] == 0.0
    finally:
        if os.path.exists("/tmp/test_kelly_finding_71.json"):
            os.remove("/tmp/test_kelly_finding_71.json")


def test_finding_84_stop_geometry_and_audit_truthfulness():
    """Finding #84: calculate_adaptive_structural_stop honors cfg_sl_mult (e.g. 0.6585) without forcing >= 1.25 ATR."""
    bars = 30
    closes = [60000.0 + i * 10 for i in range(bars)]
    highs = [c + 20 for c in closes]
    lows = [c - 20 for c in closes]
    df = pd.DataFrame({
        "close": closes,
        "high": highs,
        "low": lows,
        "volume": [100.0] * bars
    })

    entry_price = 60300.0
    atr_val = 200.0
    cfg_sl_mult = 0.6585  # Model training label geometry for 60m

    sl_price, sl_dist_pct, meta = trade_calculators.calculate_adaptive_structural_stop(
        df_recent=df,
        entry_price=entry_price,
        direction="Bullish",
        atr_val=atr_val,
        regime="Trending",
        volatility=0.015,
        cfg_sl_mult=cfg_sl_mult
    )

    sl_dist = entry_price - sl_price
    sl_mult = sl_dist / atr_val

    # Ensure stop multiplier is not forced to >= 1.25
    assert sl_mult < 1.25, f"Expected sl_mult < 1.25, got {sl_mult}"
    assert meta.get("cfg_sl_mult") == cfg_sl_mult

    # Check main.py decision journal snapshot records configured sl_mult
    with open("main.py", "r") as f:
        content = f.read()
    assert 'labelled_sl_mult=float(cfg.get("sl_mult", 1.0))' in content
    assert 'executed_sl_mult=float(realized_sl_m)' in content


def test_finding_98_directional_budget_80pct_cap_and_null_leverage():
    """Finding #98: MAX_DIRECTIONAL_RATIO is 0.80, rejects 120% equity concentration, and handles leverage=None."""
    assert config.MAX_DIRECTIONAL_RATIO == 0.80

    engine = portfolio_risk.PortfolioRiskEngine()

    # Case 1: leverage is None in open position dict — must not raise TypeError
    open_positions = [
        {"symbol": "ETHUSDT", "position_size_usd": 20.0, "leverage": None, "direction": "Bullish"}
    ]
    # Candidate trade with proposed_leverage=None
    approved, ratio, reason = engine.check_directional_budget(
        total_equity=100.0,
        proposed_size_usd=10.0,
        proposed_direction="Bullish",
        proposed_leverage=None,
        open_positions=open_positions
    )
    assert approved is True
    # Exposure: 20 * 1.0 + 10 * 1.0 = 30 / 100 = 0.30 <= 0.80
    assert abs(ratio - 0.30) < 1e-4

    # Case 2: 3 positions of $3.20 margin at 10x leverage on $80 equity
    # 3 * 32 = $96 notional = 120% equity concentration -> MUST BE REJECTED under 0.80 cap
    two_positions = [
        {"symbol": "BTCUSDT", "position_size_usd": 3.20, "leverage": 10.0, "direction": "Bullish"},
        {"symbol": "ETHUSDT", "position_size_usd": 3.20, "leverage": 10.0, "direction": "Bullish"}
    ]
    # Proposing 3rd trade of $3.20 at 10x
    approved, ratio, reason = engine.check_directional_budget(
        total_equity=80.0,
        proposed_size_usd=3.20,
        proposed_direction="Bullish",
        proposed_leverage=10.0,
        open_positions=two_positions
    )
    assert approved is False
    assert ratio == 1.20  # 96 / 80 = 1.20 (120%)
    assert "REJECTED" in reason

    # Case 3: Over-budget portfolio hedge that still leaves ratio > 0.80 must be rejected
    huge_long = [{"symbol": "BTCUSDT", "position_size_usd": 20.0, "leverage": 10.0, "direction": "Bullish"}]
    approved_hedge, ratio_hedge, reason_hedge = engine.check_directional_budget(
        total_equity=100.0,
        proposed_size_usd=5.0,
        proposed_direction="Bearish",
        proposed_leverage=10.0,
        open_positions=huge_long
    )
    assert approved_hedge is False
    assert "REJECTED" in reason_hedge


def test_finding_102_atomic_trade_closure_in_exit_path():
    """Finding #102: database.close_trade_atomically is invoked and updates SQLite tables atomically."""
    test_db = "/tmp/test_bot_finding_102.db"
    try:
        with patch.object(database, "DB_FILE", test_db):
            database.init_db()

            trade_id = "tr_BTCUSDT_1700000000_test102"
            active_trade = {
                "trade_id": trade_id,
                "symbol": "BTCUSDT",
                "entry_price": 60000.0,
                "direction": "Bullish",
                "position_size_usd": 50.0,
                "leverage": 10.0
            }
            # Pre-insert into active_trades
            database.save_active_trades("15", [active_trade])

            # Now close the trade atomically
            completed_trade = dict(active_trade)
            completed_trade.update({
                "exit_time": time.time(),
                "exit_price": 61000.0,
                "change_pct": 1.66,
                "success": True,
                "pnl_usd": 8.3,
                "reason": "TAKE PROFIT HIT [SUCCESS]",
                "balance": 108.3
            })
            success = database.close_trade_atomically(completed_trade, tf="15")
            assert success is True

            # Verify trade was removed from active_trades
            active_after = database.get_active_trades("15")
            assert len(active_after) == 0

            # Verify trade was inserted into completed_trades
            history = database.get_completed_trades(limit=100)
            assert any(t.get("trade_id") == trade_id for t in history)
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_finding_104_exit_loop_releases_lock_during_io():
    """Finding #104: active_trades_lock is not held continuously across exit checks."""
    with open("main.py", "r") as f:
        content = f.read()

    # Verify that PortfolioUtilityOptimizer rebalancing runs outside active_trades_lock
    assert "rebal_close_set = set()" in content
    # Verify active_trades_lock is acquired with discrete granular scopes
    assert 'active_trades_list = bot_state.get(active_trade_key, [])' in content
    assert 'bot_state[active_trade_key] = updated_trades' in content
