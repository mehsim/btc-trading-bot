import os
import json
import sqlite3
import pytest
from unittest.mock import patch
import database
import main
from state_manager import StateManager


def test_sqlite_is_single_source_of_truth_on_startup(tmp_path, monkeypatch):
    """Verify that load_history() preserves SQLite active_trades across restarts and doesn't overwrite from JSON."""
    test_db = str(tmp_path / "test_trading_bot.db")
    test_json = str(tmp_path / "dashboard_history.json")
    monkeypatch.setenv("DATABASE_PATH", test_db)

    # Write a stale JSON file where all active_trade_* arrays are empty []
    stale_json_data = {
        "simulated_balance": 100.0,
        "trade_history": [],
        "prediction_history": [],
        "active_trade_15m": [],
        "active_trade_30m": [],
        "active_trade_1h": [],
        "active_trade_2h": [],
        "active_trade_4h": [],
        "active_trade_6h": [],
        "fresh_reset_v3": True
    }
    with open(test_json, "w") as f:
        json.dump(stale_json_data, f)

    database.init_db()

    # Insert an active 15m trade directly into SQLite (simulating an existing open position)
    live_trade = {
        "trade_id": "BTCUSDT_test_15m_1001",
        "symbol": "BTCUSDT",
        "interval": "15",
        "timeframe": "15m",
        "entry_price": 60000.0,
        "position_size_usd": 50.0,
        "half_closed": True,
        "scaled_out_pnl": 0.90,
        "stop_loss": 60000.0,
        "take_profit": 61500.0
    }
    database.save_active_trades("15m", [live_trade])

    # Verify it exists in SQLite
    db_trades = database.get_active_trades("15m")
    assert len(db_trades) == 1
    assert db_trades[0]["trade_id"] == "BTCUSDT_test_15m_1001"
    assert db_trades[0]["half_closed"] is True
    assert db_trades[0]["scaled_out_pnl"] == 0.90

    # Initialize StateManager (simulating bot reboot)
    sm = StateManager()
    assert len(sm["active_trade_15m"]) == 1
    assert sm["active_trade_15m"][0]["trade_id"] == "BTCUSDT_test_15m_1001"

    # Call load_history() with bot_state pointing to sm and HISTORY_FILE pointing to test_json
    with patch("main.bot_state", sm), patch("main.HISTORY_FILE", test_json):
        main.load_history()

    # Verify active trades were NOT wiped by load_history()
    assert len(sm["active_trade_15m"]) == 1
    assert sm["active_trade_15m"][0]["trade_id"] == "BTCUSDT_test_15m_1001"
    assert sm["active_trade_15m"][0]["half_closed"] is True
    assert sm["active_trade_15m"][0]["scaled_out_pnl"] == 0.90

    # Verify SQLite still has the active trade row
    db_trades_after = database.get_active_trades("15m")
    assert len(db_trades_after) == 1
    assert db_trades_after[0]["trade_id"] == "BTCUSDT_test_15m_1001"


def test_save_active_trades_upsert_and_reconciliation(tmp_path, monkeypatch):
    """Verify database.save_active_trades performs upsert and reconciles closed trades properly."""
    test_db = str(tmp_path / "test_upsert_bot.db")
    monkeypatch.setenv("DATABASE_PATH", test_db)

    database.init_db()

    trade_1 = {"trade_id": "T1", "symbol": "BTCUSDT", "entry_price": 60000.0, "pnl": 1.0}
    trade_2 = {"trade_id": "T2", "symbol": "ETHUSDT", "entry_price": 3000.0, "pnl": 2.0}

    # Save both trades
    assert database.save_active_trades("15m", [trade_1, trade_2]) is True
    trades = database.get_active_trades("15m")
    assert len(trades) == 2
    trade_ids = {t["trade_id"] for t in trades}
    assert trade_ids == {"T1", "T2"}

    # Update trade_1 and close trade_2 (pass only trade_1 with modified pnl)
    trade_1_updated = {"trade_id": "T1", "symbol": "BTCUSDT", "entry_price": 60000.0, "pnl": 5.5}
    assert database.save_active_trades("15m", [trade_1_updated]) is True

    trades_updated = database.get_active_trades("15m")
    assert len(trades_updated) == 1
    assert trades_updated[0]["trade_id"] == "T1"
    assert trades_updated[0]["pnl"] == 5.5

    # Close all trades for 15m
    assert database.save_active_trades("15m", []) is True
    assert len(database.get_active_trades("15m")) == 0
