"""
regime_transition_analyzer.py
------------------------------
Phase 1 Extension: Regime Transition Analyzer.
Captures transition events between market regimes (e.g. Trending -> High Vol, Range -> Breakout)
and measures performance degradation specifically during regime transition windows.
"""

import sqlite3
import os
import time
import threading
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_transition_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_transitions (
                    transition_key TEXT PRIMARY KEY,
                    from_regime TEXT NOT NULL,
                    to_regime TEXT NOT NULL,
                    n_trades INTEGER DEFAULT 0,
                    n_wins INTEGER DEFAULT 0,
                    total_realized_r REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    avg_r REAL DEFAULT 0.0,
                    last_updated REAL DEFAULT 0.0
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[regime_transition Error] Init failed: {e}")
        finally:
            conn.close()

def record_transition_trade(from_regime: str, to_regime: str, is_win: bool, realized_r: float):
    if not from_regime or not to_regime:
        return
        
    transition_key = f"{from_regime}->{to_regime}"
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT n_trades, n_wins, total_realized_r FROM regime_transitions WHERE transition_key = ?;", (transition_key,))
            row = cursor.fetchone()
            if not row:
                n_trades = 1
                n_wins = 1 if is_win else 0
                tot_r = realized_r
            else:
                n_trades = row["n_trades"] + 1
                n_wins = row["n_wins"] + (1 if is_win else 0)
                tot_r = row["total_realized_r"] + realized_r
                
            wr = round(n_wins / n_trades, 4)
            avg_r = round(tot_r / n_trades, 4)
            now = time.time()
            
            cursor.execute("""
                INSERT OR REPLACE INTO regime_transitions (transition_key, from_regime, to_regime, n_trades, n_wins, total_realized_r, win_rate, avg_r, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (transition_key, from_regime, to_regime, n_trades, n_wins, tot_r, wr, avg_r, now))
            conn.commit()
        except Exception as e:
            print(f"[regime_transition Error] Record failed: {e}")
        finally:
            conn.close()

def get_transition_analysis() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regime_transitions ORDER BY n_trades DESC;")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[regime_transition Error] Fetch failed: {e}")
            return []
        finally:
            conn.close()

init_transition_db()
