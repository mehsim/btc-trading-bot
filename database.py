import sqlite3
import os
import json
import threading

DB_FILE = "/data/trading_bot.db" if os.path.exists("/data") and os.access("/data", os.W_OK) else "trading_bot.db"
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = get_db_connection()
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
                raw_data TEXT
            );
        """)
        
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
                
        conn.close()

def save_prediction(pred):
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
        except Exception as e:
            print(f"[Database Error] Failed to save prediction: {e}")
        finally:
            conn.close()

def get_prediction_history(limit=500):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_data FROM predictions ORDER BY timestamp DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [json.loads(row[0]) for row in rows][::-1]  # Return in chronological order
        except Exception as e:
            print(f"[Database Error] Failed to fetch prediction history: {e}")
            return []
        finally:
            conn.close()

def save_completed_trade(trade):
    t_id = trade.get("trade_id") or f"{trade.get('symbol')}_{int(trade.get('exit_time', 0))}"
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO completed_trades (
                    trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                    change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                    balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                t_id, trade.get("symbol"), trade.get("exit_time"), trade.get("interval", "60"), trade.get("direction"),
                trade.get("entry_price"), trade.get("exit_price"), trade.get("change_pct"), 1 if trade.get("success") else 0,
                trade.get("reason"), trade.get("position_size_usd"), trade.get("original_size"), trade.get("pnl_usd"),
                trade.get("balance"), trade.get("leverage"), trade.get("confidence"), trade.get("take_profit"),
                trade.get("stop_loss"), trade.get("atr_dollars"), trade.get("fill_pct"), json.dumps(trade)
            ))
            conn.commit()
        except Exception as e:
            print(f"[Database Error] Failed to save completed trade: {e}")
        finally:
            conn.close()

def get_trade_history(limit=500):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_data FROM completed_trades ORDER BY exit_time DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [json.loads(row[0]) for row in rows][::-1]  # Return in chronological order
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

def save_active_trades(tf, trades):
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
        except Exception as e:
            print(f"[Database Error] Failed to save active trades for {tf}: {e}")
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

def set_setting(key, value):
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", (key, str(value)))
            conn.commit()
        except Exception as e:
            print(f"[Database Error] Failed to save setting {key}: {e}")
        finally:
            conn.close()

def save_cached_derivatives(symbol, records):
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
        except Exception as e:
            print(f"[Database Error] Failed to save cached derivatives for {symbol}: {e}")
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

def save_cached_sentiment(records):
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
        except Exception as e:
            print(f"[Database Error] Failed to save cached sentiment: {e}")
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
        finally:
            conn.close()
