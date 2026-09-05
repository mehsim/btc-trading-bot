from logger import log_event
import math
import sqlite3
import os
import shutil
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
    except Exception as ex_database:
        log_event("WARNING", f"database notice: {ex_database}")
        try:
            return round(float(val), decimals)
        except Exception as ex_database:
            log_event("WARNING", f"database notice: {ex_database}")
            return 0.0

def get_db_path() -> str:
    env_path = os.getenv("DATABASE_PATH")
    if env_path:
        return env_path
    return "/data/trading_bot.db" if os.path.exists("/data") and os.access("/data", os.W_OK) else "trading_bot.db"

DB_FILE = get_db_path()
db_lock = threading.RLock()


_in_quarantine_recovery = False

def get_db_connection(target_db=None):
    if target_db is None:
        target_db = get_db_path()
    global _in_quarantine_recovery
    for attempt in range(5):
        conn = None
        try:
            conn = sqlite3.connect(target_db, timeout=60.0)
            conn.execute("PRAGMA busy_timeout = 60000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError as op_err:
            # Handle transient database lock / contention with exponential backoff
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            if "locked" in str(op_err).lower() or "busy" in str(op_err).lower():
                time.sleep(0.2 * (2 ** attempt))
                continue
            time.sleep(0.5)
        except sqlite3.DatabaseError as e:
            err_msg = str(e).lower()
            # Only quarantine file if error indicates genuine file malformation/corruption
            if ("malformed" in err_msg or "file is not a database" in err_msg) and not _in_quarantine_recovery:
                _in_quarantine_recovery = True
                is_corrupted = False
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

                # Corroborate corruption with a read-only integrity check and proper resource cleanup
                chk_conn = None
                try:
                    if os.path.exists(target_db):
                        chk_conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True, timeout=60.0)
                        chk_conn.execute("PRAGMA busy_timeout = 60000;")
                        res = chk_conn.execute("PRAGMA integrity_check;").fetchone()
                        if res and "ok" not in str(res[0]).lower():
                            is_corrupted = True
                        else:
                            is_corrupted = False
                    else:
                        is_corrupted = False
                except Exception as chk_err:
                    # Fail-safe: Transient lock/access error in read-only probe does NOT prove corruption
                    log_event("WARNING", f"[Database Auto-Recovery] Integrity probe error ({chk_err}); skipping premature quarantine.")
                    is_corrupted = False
                finally:
                    if chk_conn is not None:
                        try:
                            chk_conn.close()
                        except Exception:
                            pass

                if is_corrupted:
                    log_event("CRITICAL", f"[Database Auto-Recovery] Verified genuine database corruption ({e}). Creating backup snapshot & quarantining DB file {target_db}...")
                    timestamp = int(time.time())
                    
                    # Pre-quarantine snapshot for disaster recovery
                    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(target_db)) or ".", "db_snapshots", f"pre_quarantine_{timestamp}")
                    try:
                        os.makedirs(snapshot_dir, exist_ok=True)
                        for ext in ["", "-wal", "-shm"]:
                            src = f"{target_db}{ext}"
                            if os.path.exists(src):
                                shutil.copy2(src, os.path.join(snapshot_dir, f"{os.path.basename(target_db)}{ext}"))
                        log_event("INFO", f"[Database Auto-Recovery] Pre-quarantine snapshot saved to {snapshot_dir}")
                    except Exception as snap_err:
                        log_event("WARNING", f"[Database Auto-Recovery] Warning: Pre-quarantine snapshot failed: {snap_err}")

                    renamed_main = False
                    for ext in ["", "-wal", "-shm"]:
                        target = f"{target_db}{ext}"
                        if os.path.exists(target):
                            try:
                                os.rename(target, f"{target}.corrupt.{timestamp}")
                                if ext == "":
                                    renamed_main = True
                            except Exception as ren_err:
                                log_event("WARNING", f"[Database Auto-Recovery] Could not rename {target}: {ren_err}")
                    if renamed_main:
                        try:
                            init_db()
                        except Exception as init_err:
                            log_event("WARNING", f"[Database Auto-Recovery] init_db after quarantine notice: {init_err}")
                else:
                    log_event("INFO", f"[Database Transient Warning] Integrity check passed or non-corrupt; avoiding premature quarantine.")
                _in_quarantine_recovery = False
                time.sleep(0.5)
            else:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                time.sleep(0.5 * (attempt + 1))
    conn = sqlite3.connect(target_db, timeout=60.0)
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
            
            # Migration check for model lineage & venue & intended size columns
            cols = [row[1] for row in cursor.execute("PRAGMA table_info(completed_trades);").fetchall()]
            for col_name, col_type in [
                ("model_version", "TEXT"), ("model_sha", "TEXT"), ("feature_set_version", "TEXT"),
                ("venue_closed_pnl", "REAL"), ("venue_qty", "REAL"), ("venue_entry_value", "REAL"),
                ("intended_size_usd", "REAL"), ("initial_stop_loss", "REAL"), ("initial_take_profit", "REAL"),
                ("initial_rr", "REAL"), ("kelly_size_usd", "REAL"), ("clamped_size_usd", "REAL"),
                ("final_size_usd", "REAL"), ("mae", "REAL"), ("mfe", "REAL"), ("pnl_source", "TEXT"),
                ("regime", "TEXT")
            ]:
                if col_name not in cols:
                    try:
                        cursor.execute(f"ALTER TABLE completed_trades ADD COLUMN {col_name} {col_type};")
                    except Exception as ex_database:
                        log_event("WARNING", f"database notice: {ex_database}")
            
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
                    reason TEXT,
                    interval TEXT
                );
            """)
            try:
                cursor.execute("PRAGMA table_info(pending_pain_checks);")
                _ppc_cols = [c[1] for c in cursor.fetchall()]
                if "interval" not in _ppc_cols:
                    cursor.execute("ALTER TABLE pending_pain_checks ADD COLUMN interval TEXT;")
            except Exception as _mig_err:
                log_event("WARNING", f"[Database Migration] pending_pain_checks interval column: {_mig_err}")
            
            conn.commit()
            try:
                from decision_journal import init_decision_journal_db
                init_decision_journal_db()
            except Exception as ex_dj:
                log_event("WARNING", f"decision_journal init notice: {ex_dj}")
            
            # 5. Check if migration from JSON is needed
            cursor.execute("SELECT COUNT(*) FROM completed_trades;")
            count = cursor.fetchone()[0]
            
            legacy_file = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"
            csv_journal = "/data/trade_journal.csv" if os.path.exists("/data") and os.access("/data", os.W_OK) else "trade_journal.csv"
            
            if count == 0:
                if os.path.exists(legacy_file):
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

                if os.path.exists(csv_journal):
                    print("[Database] Migrating trade_journal.csv into SQLite completed_trades table...")
                    try:
                        import csv
                        from datetime import datetime, timezone
                        with open(csv_journal, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            inserted_cnt = 0
                            for row in reader:
                                ts_str = row.get("timestamp", "")
                                try:
                                    exit_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                                except Exception:
                                    exit_time = time.time()
                                t_id = f"journal_{row.get('symbol', 'BTCUSDT')}_{int(exit_time)}_{inserted_cnt}"
                                success_val = 1 if str(row.get("success", "")).lower() in ["true", "1", "yes"] else 0
                                def _sf(v, default=0.0):
                                    try:
                                        return float(v)
                                    except Exception:
                                        return default

                                cursor.execute("""
                                    INSERT OR IGNORE INTO completed_trades (
                                        trade_id, symbol, exit_time, interval, direction, entry_price, exit_price,
                                        change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                                        balance, leverage, confidence, raw_data
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """, (
                                    t_id, row.get("symbol"), exit_time, str(row.get("interval", "60")), row.get("direction"),
                                    _sf(row.get("entry_price")), _sf(row.get("exit_price")),
                                    _sf(row.get("change_pct")), success_val, row.get("reason"),
                                    20.0, 20.0, _sf(row.get("pnl_usd")),
                                    _sf(row.get("balance")), _sf(row.get("leverage"), 1.0),
                                    _sf(row.get("confidence"), 0.50), json.dumps(row)
                                ))
                                inserted_cnt += 1
                            conn.commit()
                            print(f"[Database] Successfully migrated {inserted_cnt} trades from trade_journal.csv into SQLite completed_trades table.")
                    except Exception as csv_err:
                        print(f"[Database Error] Exception during trade_journal.csv migration: {csv_err}")
        finally:
            conn.close()


def get_completed_trades(limit: int = 100, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent completed trades from SQLite database, deduplicated by symbol, exit_time, entry & exit price."""
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM completed_trades 
                    WHERE symbol = ? AND rowid IN (
                        SELECT MIN(rowid) 
                        FROM completed_trades 
                        WHERE symbol = ?
                        GROUP BY symbol, round(exit_time / 60), round(entry_price, 4), round(exit_price, 4)
                    )
                    ORDER BY exit_time DESC LIMIT ?;
                """, (symbol, symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM completed_trades 
                    WHERE rowid IN (
                        SELECT MIN(rowid) 
                        FROM completed_trades 
                        GROUP BY symbol, round(exit_time / 60), round(entry_price, 4), round(exit_price, 4)
                    )
                    ORDER BY exit_time DESC LIMIT ?;
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as ex:
            log_event("WARNING", f"database get_completed_trades failed: {ex}")
            return []
        finally:
            conn.close()


def save_prediction(pred) -> bool:
    c_ts = pred.get("candle_timestamp") or pred.get("timestamp", 0)
    p_id = pred.get("prediction_id") or f"{pred.get('symbol')}_{pred.get('interval', '15')}_{int(c_ts)}"
    pred["prediction_id"] = p_id
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
                    except Exception as ex_database:
                        log_event("WARNING", f"database notice: {ex_database}")
            return res[::-1]  # Return in chronological order
        except Exception as e:
            print(f"[Database Error] Failed to fetch prediction history: {e}")
            return []
        finally:
            conn.close()

def save_pending_pain_check(trade) -> bool:
    t_id = trade.get("trade_id") or f"{trade.get('symbol')}_{int(trade.get('exit_time', 0))}"
    iv_val = str(trade.get("interval") or trade.get("tf") or "15").replace("m", "")
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO pending_pain_checks (
                    trade_id, symbol, entry_price, exit_price, take_profit, stop_loss, exit_time, direction, reason, interval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                t_id, trade.get("symbol"), trade.get("entry_price"), trade.get("exit_price"),
                trade.get("take_profit"), trade.get("stop_loss"), trade.get("exit_time"),
                trade.get("direction"), trade.get("reason"), iv_val
            ))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
            print(f"[Database ERROR] Failed to save pending pain check: {e} | Payload: {json.dumps(trade)}")
            return False
        finally:
            conn.close()

def get_pending_pain_checks():
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT trade_id, symbol, entry_price, exit_price, take_profit, stop_loss, exit_time, direction, reason, interval FROM pending_pain_checks;")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            entry_p = round_monetary(trade.get("entry_price"), 4)
            exit_p = round_monetary(trade.get("exit_price"), 4)
            sym = trade.get("symbol")
            
            # Deduplication Guard: Check if matching trade already exists in DB
            c1 = conn.execute("SELECT trade_id, venue_closed_pnl FROM completed_trades WHERE trade_id = ?;", (t_id,))
            existing_row = c1.fetchone()
            if existing_row:
                if trade.get("venue_closed_pnl") is not None and existing_row[1] is None:
                    conn.execute("""
                        UPDATE completed_trades SET venue_closed_pnl = ?, venue_qty = ?, venue_entry_value = ? WHERE trade_id = ?;
                    """, (round_monetary(trade.get("venue_closed_pnl"), 4), round_monetary(trade.get("venue_qty"), 4), round_monetary(trade.get("venue_entry_value"), 4), t_id))
                    conn.commit()
                conn.close()
                return True

            t_interval = str(trade.get("interval", "60")).replace("m", "").replace("h", "")
            t_dir = str(trade.get("direction", "")).upper()
            if exit_ts and sym:
                if entry_p is not None and exit_p is not None:
                    c2 = conn.execute("""
                        SELECT trade_id FROM completed_trades 
                        WHERE symbol = ?
                          AND replace(replace(interval, 'm', ''), 'h', '') = ?
                          AND UPPER(direction) = ?
                          AND abs(exit_time - ?) < 15.0 
                          AND abs(entry_price - ?) / max(1.0, entry_price) < 0.0005 
                          AND abs(exit_price - ?) / max(1.0, exit_price) < 0.0005;
                    """, (sym, t_interval, t_dir, exit_ts, entry_p, exit_p))
                    if c2.fetchone():
                        conn.close()
                        return True
                else:
                    c2 = conn.execute("""
                        SELECT trade_id FROM completed_trades 
                        WHERE symbol = ?
                          AND replace(replace(interval, 'm', ''), 'h', '') = ?
                          AND UPPER(direction) = ?
                          AND abs(exit_time - ?) < 5.0;
                    """, (sym, t_interval, t_dir, exit_ts))
                    if c2.fetchone():
                        conn.close()
                        return True

            try:
                cur = conn.execute("""
                    INSERT INTO completed_trades (
                        trade_id, symbol, exit_time, interval, direction, regime, entry_price, exit_price,
                        change_pct, success, reason, position_size_usd, intended_size_usd, original_size, pnl_usd,
                        balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct,
                        venue_closed_pnl, venue_qty, venue_entry_value, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, sym, trade.get("exit_time"), trade.get("interval", "60"), trade.get("direction"),
                    str(trade.get("regime", "")),
                    entry_p, exit_p,
                    round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                    trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4),
                    round_monetary(trade.get("intended_size_usd", trade.get("position_size_usd")), 4),
                    round_monetary(trade.get("original_size"), 4),
                    round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                    round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                    round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                    round_monetary(trade.get("fill_pct"), 2),
                    round_monetary(trade.get("venue_closed_pnl"), 4) if trade.get("venue_closed_pnl") is not None else None,
                    round_monetary(trade.get("venue_qty"), 4) if trade.get("venue_qty") is not None else None,
                    round_monetary(trade.get("venue_entry_value"), 4) if trade.get("venue_entry_value") is not None else None,
                    json.dumps(trade)
                ))
                conn.commit()
                success = bool(cur.rowcount == 1)
            except sqlite3.IntegrityError as ie:
                log_event("WARNING", f"[Database Deduplication] IntegrityError inserting trade {t_id} (non-insert handled): {ie}")
                success = False
        except Exception as e:
            try:
                conn.rollback()
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            # Step 1: Idempotent Insert/Update into completed_trades
            conn.execute("""
                INSERT INTO completed_trades (
                    trade_id, symbol, exit_time, interval, direction, regime, entry_price, exit_price,
                    change_pct, success, reason, position_size_usd, original_size, pnl_usd,
                    balance, leverage, confidence, take_profit, stop_loss, atr_dollars, fill_pct, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    exit_time=excluded.exit_time,
                    exit_price=excluded.exit_price,
                    change_pct=excluded.change_pct,
                    success=excluded.success,
                    reason=excluded.reason,
                    regime=excluded.regime,
                    position_size_usd=excluded.position_size_usd,
                    original_size=excluded.original_size,
                    pnl_usd=excluded.pnl_usd,
                    balance=excluded.balance,
                    raw_data=excluded.raw_data;
            """, (
                t_id, symbol, trade.get("exit_time"), str(tf), trade.get("direction"),
                str(trade.get("regime", "")),
                round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                round_monetary(trade.get("change_pct"), 4), 1 if trade.get("success") else 0,
                trade.get("reason"), round_monetary(trade.get("position_size_usd"), 4), round_monetary(trade.get("original_size"), 4),
                round_monetary(trade.get("pnl_usd"), 4), round_monetary(trade.get("balance"), 4), round_monetary(trade.get("leverage"), 2),
                round_monetary(trade.get("confidence"), 4), round_monetary(trade.get("take_profit"), 4),
                round_monetary(trade.get("stop_loss"), 4), round_monetary(trade.get("atr_dollars"), 4),
                round_monetary(trade.get("fill_pct"), 2), json.dumps(trade)
            ))

            # Step 1b: Set initial stop/TP/RR and size breakdown columns if available
            try:
                conn.execute("""
                    UPDATE completed_trades SET
                        initial_stop_loss = ?,
                        initial_take_profit = ?,
                        initial_rr = ?,
                        kelly_size_usd = ?,
                        clamped_size_usd = ?,
                        final_size_usd = ?,
                        pnl_source = ?
                    WHERE trade_id = ?;
                """, (
                    round_monetary(trade.get("initial_stop_loss"), 4),
                    round_monetary(trade.get("initial_take_profit"), 4),
                    round_monetary(trade.get("initial_rr"), 4),
                    round_monetary(trade.get("kelly_size_usd"), 4),
                    round_monetary(trade.get("clamped_size_usd"), 4),
                    round_monetary(trade.get("final_size_usd"), 4),
                    str(trade.get("pnl_source", "ESTIMATED")),
                    t_id
                ))
            except Exception as ex_init_cols:
                log_event("WARNING", f"database notice updating initial/size columns: {ex_init_cols}")
            
            # Step 2: Delete from active_trades
            clean_tf = str(tf).replace("m", "")
            conn.execute("DELETE FROM active_trades WHERE trade_id = ? OR (symbol = ? AND replace(tf, 'm', '') = ?);", (t_id, symbol, clean_tf))
            
            # Step 3: Pending pain check if applicable
            if "STOP LOSS" in reason_str or "BREAK-EVEN" in reason_str:
                conn.execute("""
                    INSERT OR REPLACE INTO pending_pain_checks (
                        trade_id, symbol, entry_price, exit_price, take_profit, stop_loss, exit_time, direction, reason, interval
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    t_id, symbol, round_monetary(trade.get("entry_price"), 4), round_monetary(trade.get("exit_price"), 4),
                    round_monetary(trade.get("take_profit"), 4), round_monetary(trade.get("stop_loss"), 4),
                    trade.get("exit_time"), trade.get("direction"), trade.get("reason"),
                    clean_tf
                ))
            
            conn.commit()
            print(f"[Database Atomic Success] Closed trade {t_id} for {symbol} ({tf}m) atomically.")
        except Exception as e:
            try:
                conn.rollback()
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
                    except Exception as ex_database:
                        log_event("WARNING", f"database notice: {ex_database}")
            return res[::-1]  # Return in chronological order
        except Exception as e:
            print(f"[Database Error] Failed to fetch trade history: {e}")
            return []
        finally:
            conn.close()

def get_active_trades(tf=None):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if tf is not None:
                cursor.execute("SELECT raw_data FROM active_trades WHERE tf = ?;", (tf,))
            else:
                cursor.execute("SELECT raw_data FROM active_trades;")
            rows = cursor.fetchall()
            return [json.loads(row[0]) for row in rows]
        except Exception as e:
            print(f"[Database Error] Failed to fetch active trades: {e}")
            return []
        finally:
            conn.close()

def save_active_trades(tf, trades) -> bool:
    with db_lock:
        conn = get_db_connection()
        try:
            if not isinstance(trades, list):
                trades = [trades] if isinstance(trades, dict) else []

            current_ids = set()
            for t in trades:
                if not isinstance(t, dict):
                    continue
                t_id = t.get("trade_id") or f"{t.get('symbol')}_{int(t.get('entry_time', 0))}"
                current_ids.add(t_id)
                conn.execute("""
                    INSERT OR REPLACE INTO active_trades (trade_id, tf, symbol, raw_data)
                    VALUES (?, ?, ?, ?);
                """, (t_id, tf, t.get("symbol"), json.dumps(t)))

            # Reconcile: delete only trades that are for this tf but not in current_ids
            if current_ids:
                placeholders = ",".join("?" for _ in current_ids)
                conn.execute(f"DELETE FROM active_trades WHERE tf = ? AND trade_id NOT IN ({placeholders});", (tf, *current_ids))  # nosec B608
            else:
                log_event("INFO", f"[Database] save_active_trades for {tf} received empty list. Clearing active trades for {tf}.")
                conn.execute("DELETE FROM active_trades WHERE tf = ?;", (tf,))
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
        finally:
            conn.close()

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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
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
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
            print(f"[Database ERROR] Failed clearing trade history: {e}")
            success = False
        finally:
            conn.close()

    for fname in ["dashboard_history.json", "kelly_trade_history.json", "pain_feedback_state.json"]:
        if os.path.exists(fname):
            try:
                with open(fname, "w") as f:
                    json.dump([], f) if fname != "pain_feedback_state.json" else json.dump({}, f)
            except Exception as ex_database:
                log_event("WARNING", f"database notice: {ex_database}")
    return success


def backup_database(dest_path=None) -> bool:
    """
    Creates an automated online snapshot of trading_bot.db for RPO < 1 min disaster recovery.
    Uses SQLite's native online backup API to avoid blocking concurrent database access.
    """
    target_db = get_db_path()
    if dest_path is None:
        db_dir = os.path.dirname(target_db) if os.path.dirname(target_db) else "."
        dest_path = os.path.join(db_dir, "trading_bot_backup.db")
    with db_lock:
        try:
            source_conn = sqlite3.connect(target_db, timeout=10)
            backup_conn = sqlite3.connect(dest_path, timeout=10)
            with backup_conn:
                source_conn.backup(backup_conn)
            backup_conn.close()
            source_conn.close()
            return True
        except Exception as e:
            print(f"[Database Backup Error] Failed automated backup: {e}")
            return False
