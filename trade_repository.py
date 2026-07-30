"""
trade_repository.py
--------------------
Abstract Data Access Layer (Repository Pattern) for completed trades and analytics.
Decouples application logic from database engines (SQLite / DuckDB / PostgreSQL / ClickHouse).
"""

from abc import ABC, abstractmethod
import sqlite3
import json
from typing import List, Dict, Any, Optional

class TradeRepository(ABC):
    """
    Abstract interface defining data access contracts for trading records.
    """
    @abstractmethod
    def save_completed_trade(self, trade_data: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_completed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_trades_by_setup(self, symbol: str, interval: str, regime: str) -> List[Dict[str, Any]]:
        pass


class SQLiteTradeRepository(TradeRepository):
    """
    SQLite implementation of the TradeRepository interface.
    """
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def save_completed_trade(self, trade_data: Dict[str, Any]) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completed_trades (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    interval TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_usd REAL,
                    change_pct REAL,
                    confidence REAL,
                    reason TEXT,
                    exit_time REAL,
                    raw_data TEXT
                );
            """)
            cursor.execute("""
                INSERT INTO completed_trades (symbol, interval, direction, entry_price, exit_price, pnl_usd, change_pct, confidence, reason, exit_time, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                trade_data.get("symbol"),
                trade_data.get("interval"),
                trade_data.get("direction"),
                trade_data.get("entry_price", 0.0),
                trade_data.get("exit_price", 0.0),
                trade_data.get("pnl_usd", 0.0),
                trade_data.get("change_pct", 0.0),
                trade_data.get("confidence", 0.75),
                trade_data.get("reason", "SUCCESS"),
                trade_data.get("exit_time", 0.0),
                json.dumps(trade_data)
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving trade in SQLiteTradeRepository: {e}")
            return False

    def get_completed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(completed_trades);")
            cols = [c[1] for c in cursor.fetchall()]
            cursor.execute("SELECT * FROM completed_trades ORDER BY rowid DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            conn.close()

            results = []
            for r in rows:
                d = dict(zip(cols, r))
                results.append(d)
            return results
        except Exception as e:
            print(f"Error reading completed trades: {e}")
            return []

    def get_trades_by_setup(self, symbol: str, interval: str, regime: str) -> List[Dict[str, Any]]:
        all_trades = self.get_completed_trades(limit=200)
        filtered = []
        for t in all_trades:
            if t.get("symbol") == symbol and str(t.get("interval")) == str(interval):
                filtered.append(t)
        return filtered

default_trade_repository = SQLiteTradeRepository()
