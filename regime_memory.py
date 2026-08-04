"""
regime_memory.py
----------------
Phase 1B: Regime Memory & Dashboard Tracker.
Tracks performance, Win Rate, Expectancy, Profit Factor, and Duration grouped by Regime ID and Regime Type.
"""

import sqlite3
import os
import time
import threading
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
from database import db_lock

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_regime_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_memory (
                    regime_id TEXT PRIMARY KEY,
                    regime_type TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    n_trades INTEGER DEFAULT 0,
                    n_wins INTEGER DEFAULT 0,
                    total_pnl_usd REAL DEFAULT 0.0,
                    total_realized_r REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    expectancy REAL DEFAULT 0.0
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[regime_memory Error] Init failed: {e}")
        finally:
            conn.close()

def record_regime_trade(regime_type: str, pnl_usd: float, realized_r: float, regime_id: str = None) -> str:
    now = time.time()
    if not regime_id:
        regime_id = f"{regime_type}_{time.strftime('%Y_%m_%d')}"
        
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regime_memory WHERE regime_id = ?;", (regime_id,))
            row = cursor.fetchone()
            if not row:
                cursor.execute("""
                    INSERT INTO regime_memory (regime_id, regime_type, started_at, n_trades, n_wins, total_pnl_usd, total_realized_r, win_rate, expectancy)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?);
                """, (regime_id, regime_type, now, 1 if pnl_usd > 0 else 0, pnl_usd, realized_r, 1.0 if pnl_usd > 0 else 0.0, realized_r))
            else:
                n_trades = row["n_trades"] + 1
                n_wins = row["n_wins"] + (1 if pnl_usd > 0 else 0)
                tot_pnl = row["total_pnl_usd"] + pnl_usd
                tot_r = row["total_realized_r"] + realized_r
                wr = n_wins / n_trades
                exp = tot_r / n_trades
                cursor.execute("""
                    UPDATE regime_memory
                    SET n_trades = ?, n_wins = ?, total_pnl_usd = ?, total_realized_r = ?, win_rate = ?, expectancy = ?
                    WHERE regime_id = ?;
                """, (n_trades, n_wins, tot_pnl, tot_r, wr, exp, regime_id))
            conn.commit()
            return regime_id
        except Exception as e:
            print(f"[regime_memory Error] Record failed: {e}")
            return regime_id
        finally:
            conn.close()

def get_regime_summary() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regime_memory ORDER BY started_at DESC;")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[regime_memory Error] Summary failed: {e}")
            return []
        finally:
            conn.close()

init_regime_db()
