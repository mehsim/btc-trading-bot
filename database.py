import math
import sqlite3
import os
import json
import uuid
import threading
from decimal import Decimal, ROUND_HALF_UP

import time
from typing import Dict, List, Tuple, Optional, Any, Union

def round_monetary(val: Any, decimals: int = 4) -> float:
    if val is None or val == "MT" or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    try:
        d = Decimal(str(val))
        fmt = "0." + "0" * decimals if decimals > 0 else "0"
        return float(d.quantize(Decimal(fmt), rounding=ROUND_HALF_UP))
    except Exception:
        try:
            return round(float(val), decimals)
        except Exception:
            return 0.0

DB_FILE = "/data/trading_bot.db" if os.path.exists("/data") and os.access("/data", os.W_OK) else "trading_bot.db"
db_lock = threading.Lock()


def get_db_connection():
    for attempt in range(3):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=60.0)
            conn.execute("PRAGMA busy_timeout = 60000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.row_factory = sqlite3.Row
            res = conn.execute("PRAGMA quick_check;").fetchone()
            if res and res[0] != "ok":
                raise sqlite3.DatabaseError(f"Corrupt DB disk image: {res[0]}")
            return conn
        except sqlite3.DatabaseError as e:
            print(f"[Database Auto-Recovery] Detected corruption/lock ({e}). Rebuilding DB file {DB_FILE}...")
            try:
                if 'conn' in locals() and conn:
                    conn.close()
            except Exception:
                pass
            for ext in ["", "-wal", "-shm"]:
                target = f"{DB_FILE}{ext}"
                if os.path.exists(target):
                    try:
                        os.remove(target)
                    except Exception:
                        pass
            time.sleep(0.5)
    conn = sqlite3.connect(DB_FILE, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = get_db_connection()
        try:
            try:
                check_res = conn.execute("PRAGMA integrity_check;").fetchone()
                if check_res and check_res[0] == "ok":
                    print("[Database] SQLite integrity check passed.")
                else:
                    print(f"[Database Warning] SQLite integrity check failed: {check_res}")
            except Exception as e:
                print(f"[Database Error] SQLite integrity check failed with exception: {e}")
            cursor = conn.cursor()
            
            # 1. Create Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER,
                    symbol TEXT,
                    interval TEXT,
                    direction TEXT,
                    confidence REAL,
                    raw_data TEXT
                );
            """)
            
            # 2. Create Completed Trades Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completed_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    exit_time REAL,
                    interval TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    change_pct REAL,
                    success INTEGER,
                    reason TEXT,
                    position_size_usd REAL,
                    original_size REAL,
                    pnl_usd REAL,
                    balance REAL,
                    leverage REAL,
                    confidence REAL,
                    take_profit REAL,
                    stop_loss REAL,
                    atr_dollars REAL,
                    fill_pct REAL,
                    mae REAL,
                    mfe REAL,
                    duration_seconds REAL,
                    entry_heat REAL,
                    entry_drawdown_pct REAL,
                    entry_correlation REAL,
                    model_version TEXT,
                    model_sha TEXT,
                    feature_set_version TEXT,
                    raw_data TEXT
                );
            """)
            
            # Migration check for model lineage columns
            cols = [row[1] for row in cursor.execute("PRAGMA table_info(completed_trades);").fetchall()]
            for col_name in ["model_version", "model_sha", "feature_set_version"]:
                if col_name not in cols:
                    try:
                        cursor.execute(f"ALTER TABLE completed_trades ADD COLUMN {col_name} TEXT;")
                    except Exception:
                        pass
            
            # 3. Create Active Trades Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_trades (
                    trade_id TEXT PRIMARY KEY,
                    tf TEXT,
                    symbol TEXT,
                    raw_data TEXT
                );
            """)
            
            # 4. Create Settings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            
            # 6. Create Derivatives Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS derivatives_cache (
                    timestamp INTEGER,
                    symbol TEXT,
                    open_interest REAL,
                    funding_rate REAL,
                    PRIMARY KEY (timestamp, symbol)
                );
            """)
            
            # 7. Create Sentiment Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    timestamp INTEGER PRIMARY KEY,
                    fear_and_greed_val INTEGER
                );
            """)
            
            # Enable Incremental Vacuum for storage optimization
            cursor.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            
            # High-Performance Query Indexes (Fix B9 & F-18)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_completed_trades_symbol_ts ON completed_trades(symbol, exit_time);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_completed_trades_exit_time ON completed_trades(exit_time DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_trades_symbol ON active_trades(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_derivatives_symbol ON derivatives_cache(symbol);")
            
            # 8. Create Pending Pain Checks Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_pain_checks (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    take_profit REAL,
                    stop_loss REAL,
                    exit_time REAL,
                    direction TEXT,
                    reason TEXT
                );
            """)
            
            conn.commit()
            
            # 5. Check if migration from JSON is needed
            cursor.execute("SELECT COUNT(*) FROM completed_trades;")
            count = cursor.fetchone()[0]
            
            legacy_file = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"
            
            if count == 0 and os.path.exists(legacy_file):
                print("[Database] Migrating legacy JSON history into SQLite database...")
                try:
                    with open(legacy_file, "r") as f:
                        data = json.load(f)
                        
                        # Migrate completed trades
                        for t in data.get("trade_history", []):
                            t_id = t.get("trade_id") or f"{t.get('symbol')}_{int(t.get('exit_time', 0))}"
                            cursor.execute("""
                                INSERT OR IGNORE INTO completed_trades (
                                    trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                                    change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                                    balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """, (
                                t_id, t.get("symbol"), t.get("exit_time"), t.get("interval", "60"), t.get("direction"),
                                t.get("entry_price"), t.get("exit_price"), t.get("change_pct"), 1 if t.get("success") else 0,
                                t.get("reason"), t.get("position_size_usd"), t.get("original_size"), t.get("pnl_usd"),
                                t.get("balance"), t.get("leverage"), t.get("confidence"), t.get("take_profit"),
                                t.get("stop_loss"), t.get("atr_dollars"), t.get("fill_pct"), json.dumps(t)
                            ))
                            
                        # Migrate prediction history
                        for p in data.get("prediction_history", []):
                            p_id = p.get("prediction_id") or f"{p.get('symbol')}_{int(p.get('timestamp', 0))}"
                            cursor.execute("""
                                INSERT OR IGNORE INTO predictions (id, timestamp, symbol, interval, direction, confidence, raw_data)
                                VALUES (?, ?, ?, ?, ?, ?, ?);
                            """, (
                                p_id, p.get("timestamp"), p.get("symbol"), p.get("interval", "60"),
                                p.get("direction"), p.get("confidence"), json.dumps(p)
                            ))
                            
                        # Migrate settings
                        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", ("simulated_balance", str(data.get("simulated_balance", 80.0))))
                        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", ("bot_running", str(data.get("bot_running", True))))
                        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", ("fresh_reset_v3", str(data.get("fresh_reset_v3", False))))
                        
                        conn.commit()
                        print("[Database] Legacy JSON migration completed successfully.")
                except Exception as migrate_err:
                    print(f"[Database Error] Exception during JSON migration: {migrate_err}")
        finally:
            conn.close()



def save_prediction(pred) -> bool:
    p_id = pred.get("prediction_id") or f"{pred.get('symbol')}_{int(pred.get('timestamp', 0))}"
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO predictions (id, timestamp, symbol, interval, direction, confidence, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                p_id, pred.get("timestamp"), pred.get("symbol"), pred.get("interval", "60"),
                pred.get("direction"), pred.get("confidence"), json.dumps(pred)
            ))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save prediction: {e} | Payload: {json.dumps(pred)}")
            return False
        finally:
            conn.close()

def get_prediction_history(limit=500):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_data FROM predictions ORDER BY timestamp DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            res = []
            for row in rows:
                if row[0]:
                    try:
                        res.append(json.loads(row[0]))
                    except Exception:
                        pass
            return res[::-1]  # Return in chronological order
        except Exception as e:
            print(f"[Database Error] Failed to fetch prediction history: {e}")
            return []
        finally:
            conn.close()

def save_pending_pain_check(trade) -> bool:
    t_id = trade.get("trade_id") or f"{trade.get('symbol')}_{int(trade.get('exit_time', 0))}"
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO pending_pain_checks (
                    trade_id, symbol, entry_price, exit_price, take_profit, stop_loss, exit_time, direction, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                t_id, trade.get("symbol"), trade.get("entry_price"), trade.get("exit_price"),
                trade.get("take_profit"), trade.get("stop_loss"), trade.get("exit_time"),
                trade.get("direction"), trade.get("reason")
            ))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save pending pain check: {e} | Payload: {json.dumps(trade)}")
            return False
        finally:
            conn.close()

def get_pending_pain_checks():
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT trade_id, symbol, entry_price, exit_price, take_profit, stop_loss, exit_time, direction, reason FROM pending_pain_checks;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[Database Error] Failed to fetch pending pain checks: {e}")
            return []
        finally:
            conn.close()

def delete_pending_pain_check(trade_id) -> bool:
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM pending_pain_checks WHERE trade_id = ?;", (trade_id,))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to delete pending pain check {trade_id}: {e}")
            return False
        finally:
            conn.close()

def save_completed_trade(trade) -> bool:
    trade_uuid = str(uuid.uuid4())
    exit_ts = int(trade.get("exit_time", time.time()))
    t_id = trade.get("trade_id")
    if not t_id or not isinstance(t_id, str) or t_id.endswith(f"_{exit_ts}"):
        t_id = f"tr_{trade.get('symbol')}_{exit_ts}_{trade_uuid}"
        trade["trade_id"] = t_id

    success = False
    with db_lock:
        conn = get_db_connection()
        try:
            try:
                conn.execute("""
                    INSERT INTO completed_trades (
                        trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                        change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                        balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, trade.get("symbol"), trade.get("exit_time"), trade.get("interval", "60"), trade.get("direction"),
                    round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                    round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                    trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4), round_monetary(trade.get("original_size"), 4),
                    round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                    round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                    round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                    round_monetary(trade.get("fill_pct"), 2), json.dumps(trade)
                ))
                conn.commit()
                success = True
            except sqlite3.IntegrityError as ie:
                new_id = f"tr_{trade.get('symbol')}_{exit_ts}_{uuid.uuid4()}"
                trade["trade_id"] = new_id
                print(f"[Database Warning] Duplicate trade_id '{t_id}' detected. Retrying with fresh UUID '{new_id}': {ie}")
                conn.execute("""
                    INSERT INTO completed_trades (
                        trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                        change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                        balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    new_id, trade.get("symbol"), trade.get("exit_time"), trade.get("interval", "60"), trade.get("direction"),
                    round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                    round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                    trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4), round_monetary(trade.get("original_size"), 4),
                    round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                    round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                    round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                    round_monetary(trade.get("fill_pct"), 2), json.dumps(trade)
                ))
                conn.commit()
                success = True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] CRITICAL: Failed to save completed trade: {e} | Payload: {json.dumps(trade)}")
            success = False
        finally:
            conn.close()

    if success:
        reason_str = str(trade.get("reason", "")).upper()
        if "STOP LOSS" in reason_str or "BREAK-EVEN" in reason_str:
            save_pending_pain_check(trade)

        # Hook Phase 1 Continuous Learning Engine (non-blocking event dispatch)
        try:
            from learning_engine import continuous_learning_engine
            continuous_learning_engine.on_trade_closed(trade)
        except Exception as le:
            print(f"[LearningEngine Hook] Non-critical dispatch error: {le}")

    return success

def close_trade_atomically(trade: dict, tf: str = "60") -> bool:
    """
    Executes a 100% atomic transaction for trade closure:
    1. Inserts record into completed_trades
    2. Deletes trade from active_trades table for timeframe tf and symbol
    3. Inserts into pending_pain_checks if STOP LOSS or BREAK-EVEN
    If ANY step fails or if process crashes mid-way, SQLite AUTOMATICALLY ROLLS BACK EVERYTHING.
    """
    trade_uuid = str(uuid.uuid4())
    exit_ts = int(trade.get("exit_time", time.time()))
    t_id = trade.get("trade_id")
    if not t_id or not isinstance(t_id, str) or t_id.endswith(f"_{exit_ts}"):
        t_id = f"tr_{trade.get('symbol')}_{exit_ts}_{trade_uuid}"
        trade["trade_id"] = t_id

    symbol = trade.get("symbol")
    reason_str = str(trade.get("reason", "")).upper()
    
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("BEGIN TRANSACTION;")
            # Step 1: Insert into completed_trades (with IntegrityError fallback handling)
            try:
                conn.execute("""
                    INSERT INTO completed_trades (
                        trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                        change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                        balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, symbol, trade.get("exit_time"), str(tf), trade.get("direction"),
                    round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                    round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                    trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4), round_monetary(trade.get("original_size"), 4),
                    round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                    round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                    round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                    round_monetary(trade.get("fill_pct"), 2), json.dumps(trade)
                ))
            except sqlite3.IntegrityError:
                t_id = f"tr_{symbol}_{exit_ts}_{uuid.uuid4()}"
                trade["trade_id"] = t_id
                conn.execute("""
                    INSERT INTO completed_trades (
                        trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                        change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                        balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, symbol, trade.get("exit_time"), str(tf), trade.get("direction"),
                    round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                    round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                    trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4), round_monetary(trade.get("original_size"), 4),
                    round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                    round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                    round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                    round_monetary(trade.get("fill_pct"), 2), json.dumps(trade)
                ))
            
            # Step 2: Delete from active_trades
            conn.execute("DELETE FROM active_trades WHERE trade_id = ? OR (tf = ? AND symbol = ?);", (t_id, str(tf), symbol))
            
            # Step 3: Pending pain check if applicable
            if "STOP LOSS" in reason_str or "BREAK-EVEN" in reason_str:
                conn.execute("""
                    INSERT OR REPLACE INTO pending_pain_checks (
                        trade_id, symbol, exit_time, exit_price, direction, reason, initial_floor, highest_post_exit_price, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, symbol, trade.get("exit_time"), round_monetary(trade.get("exit_price"), 4),
                    trade.get("direction"), trade.get("reason"), 0.008, round_monetary(trade.get("exit_price"), 4), json.dumps(trade)
                ))
            
            conn.commit()
            print(f"[Database Atomic Success] Closed trade {t_id} for {symbol} ({tf}m) atomically.")
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database Error] Transaction rolled back for trade {t_id}: {e}")
            return False
        finally:
            conn.close()

    # Hook Phase 1 Continuous Learning Engine (non-blocking event dispatch)
    try:
        from learning_engine import continuous_learning_engine
        continuous_learning_engine.on_trade_closed(trade)
    except Exception as le:
        print(f"[LearningEngine Hook] Non-critical dispatch error: {le}")

    return True



def get_trade_history(limit=500):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_data FROM completed_trades ORDER BY exit_time DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            res = []
            for row in rows:
                if row[0]:
                    try:
                        res.append(json.loads(row[0]))
                    except Exception:
                        pass
            return res[::-1]  # Return in chronological order
        except Exception as e:
            print(f"[Database Error] Failed to fetch trade history: {e}")
            return []
        finally:
            conn.close()

def get_active_trades(tf):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_data FROM active_trades WHERE tf = ?;", (tf,))
            rows = cursor.fetchall()
            return [json.loads(row[0]) for row in rows]
        except Exception as e:
            print(f"[Database Error] Failed to fetch active trades for {tf}: {e}")
            return []
        finally:
            conn.close()

def save_active_trades(tf, trades) -> bool:
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM active_trades WHERE tf = ?;", (tf,))
            for t in trades:
                t_id = t.get("trade_id") or f"{t.get('symbol')}_{int(t.get('entry_time', 0))}"
                conn.execute("""
                    INSERT OR REPLACE INTO active_trades (trade_id, tf, symbol, raw_data)
                    VALUES (?, ?, ?, ?);
                """, (t_id, tf, t.get("symbol"), json.dumps(t)))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save active trades for {tf}: {e} | Payload: {json.dumps(trades)}")
            return False
        finally:
            conn.close()

def get_setting(key, default=None):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?;", (key,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return default
        except Exception as e:
            print(f"[Database Error] Failed to fetch setting {key}: {e}")
            return default
        finally:
            conn.close()

def set_setting(key, value) -> bool:
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", (key, str(value)))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save setting {key}: {e}")
            return False
        finally:
            conn.close()

def save_cached_derivatives(symbol, records) -> bool:
    """
    records is a list of dicts: [{'timestamp': int_ms, 'open_interest': float, 'funding_rate': float}]
    """
    with db_lock:
        conn = get_db_connection()
        try:
            for r in records:
                conn.execute("""
                    INSERT OR REPLACE INTO derivatives_cache (timestamp, symbol, open_interest, funding_rate)
                    VALUES (?, ?, ?, ?);
                """, (r['timestamp'], symbol, r.get('open_interest'), r.get('funding_rate')))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save cached derivatives for {symbol}: {e}")
            return False
        finally:
            conn.close()

def get_cached_derivatives(symbol, since_ts):
    """
    Returns list of dicts containing cached data since since_ts (timestamp in ms)
    """
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, open_interest, funding_rate FROM derivatives_cache
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC;
            """, (symbol, since_ts))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[Database Error] Failed to fetch cached derivatives for {symbol}: {e}")
            return []
        finally:
            conn.close()

def save_cached_sentiment(records) -> bool:
    """
    records is a list of dicts: [{'timestamp': int_ms, 'fear_and_greed_val': int}]
    """
    with db_lock:
        conn = get_db_connection()
        try:
            for r in records:
                conn.execute("""
                    INSERT OR REPLACE INTO sentiment_cache (timestamp, fear_and_greed_val)
                    VALUES (?, ?);
                """, (r['timestamp'], r.get('fear_and_greed_val')))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to save cached sentiment: {e}")
            return False
        finally:
            conn.close()

def get_cached_sentiment(since_ts):
    """
    Returns list of dicts since since_ts (timestamp in ms)
    """
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, fear_and_greed_val FROM sentiment_cache
                WHERE timestamp >= ?
                ORDER BY timestamp ASC;
            """, (since_ts,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[Database Error] Failed to fetch cached sentiment: {e}")
            return []

def purge_old_order_flow_logs(retention_days: int = 60) -> int:
    """
    Purges historical order flow ticks older than retention_days (default 60 days) to optimize disk storage.
    Executes PRAGMA incremental_vacuum to reclaim disk space.
    """
    cutoff_ts = time.time() - (retention_days * 86400)
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM historical_order_flow WHERE timestamp < ?", (cutoff_ts,))
            deleted_rows = cursor.rowcount
            conn.commit()
            cursor.execute("PRAGMA incremental_vacuum;")
            if deleted_rows > 0:
                print(f"🧹 [Storage Cleanup] Purged {deleted_rows} order flow ticks older than {retention_days} days. Vacuum complete.")
            return deleted_rows
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to purge old order flow logs: {e}")
            return 0
        finally:
            conn.close()

def archive_legacy_recovery_trades() -> int:
    """
    Archives legacy pre-upgrade trades marked with 'OFFLINE' or 'RECOVERY'
    into completed_trades_archive table so tearsheet reflects clean live performance.
    """
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completed_trades_archive AS 
                SELECT * FROM completed_trades WHERE 1=0;
            """)
            cursor.execute("""
                INSERT INTO completed_trades_archive 
                SELECT * FROM completed_trades WHERE reason LIKE '%OFFLINE%' OR reason LIKE '%RECOVERY%';
            """)
            cursor.execute("""
                DELETE FROM completed_trades WHERE reason LIKE '%OFFLINE%' OR reason LIKE '%RECOVERY%';
            """)
            archived_count = cursor.rowcount
            conn.commit()
            cursor.execute("PRAGMA incremental_vacuum;")
            if archived_count > 0:
                print(f"📦 [Database Archive] Moved {archived_count} legacy offline recovery trades into completed_trades_archive.")
            return archived_count
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed to archive legacy trades: {e}")
            return 0
        finally:
            conn.close()


def clear_all_trade_history() -> bool:
    """
    Completely clears completed_trades and predictions tables from SQLite,
    resetting all-time cumulative PnL to $0.00.
    """
    success = False
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM completed_trades;")
            cursor.execute("DELETE FROM predictions;")
            conn.commit()
            cursor.execute("PRAGMA incremental_vacuum;")
            print("🧹 [Database Clear] Successfully wiped all completed_trades and predictions records.")
            success = True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"[Database ERROR] Failed clearing trade history: {e}")
            success = False
        finally:
            conn.close()

    for fname in ["dashboard_history.json", "kelly_trade_history.json", "pain_feedback_state.json"]:
        if os.path.exists(fname):
            try:
                with open(fname, "w") as f:
                    json.dump([], f) if fname != "pain_feedback_state.json" else json.dump({}, f)
            except Exception:
                pass
    return success


def backup_database(dest_path=None) -> bool:
    """
    Creates an automated online snapshot of trading_bot.db for RPO < 1 min disaster recovery.
    Uses SQLite's native online backup API to avoid blocking concurrent database access.
    """
    if dest_path is None:
        db_dir = os.path.dirname(DB_FILE) if os.path.dirname(DB_FILE) else "."
        dest_path = os.path.join(db_dir, "trading_bot_backup.db")
    with db_lock:
        try:
            source_conn = sqlite3.connect(DB_FILE, timeout=10)
            backup_conn = sqlite3.connect(dest_path, timeout=10)
            with backup_conn:
                source_conn.backup(backup_conn)
            backup_conn.close()
            source_conn.close()
            return True
        except Exception as e:
            print(f"[Database Backup Error] Failed automated backup: {e}")
            return False
