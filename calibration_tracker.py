"""
calibration_tracker.py
----------------------
Phase 1A: Confidence Calibration Tracker.
Measures model confidence reliability by grouping predictions into 10% confidence buckets
('50-60%', '60-70%', '70-80%', '80-90%', '90-100%') and calculating Expected Calibration Error (ECE).
"""

import sqlite3
import os
import threading
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
from database import db_lock

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_calibration_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_buckets (
                    bucket TEXT PRIMARY KEY,
                    n_trades INTEGER DEFAULT 0,
                    n_wins INTEGER DEFAULT 0,
                    sum_confidence REAL DEFAULT 0.0,
                    predicted_mean REAL DEFAULT 0.0,
                    actual_win_rate REAL DEFAULT 0.0,
                    calibration_error REAL DEFAULT 0.0,
                    last_updated REAL DEFAULT 0.0
                );
            """)
            # Initialize default buckets
            default_buckets = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
            for b in default_buckets:
                conn.execute("INSERT OR IGNORE INTO calibration_buckets (bucket) VALUES (?);", (b,))
            conn.commit()
        except Exception as e:
            print(f"[calibration_tracker Error] Init failed: {e}")
        finally:
            conn.close()

def get_bucket_label(confidence: float) -> str:
    c = max(0.50, min(1.00, float(confidence)))
    if c < 0.60:
        return "50-60%"
    elif c < 0.70:
        return "60-70%"
    elif c < 0.80:
        return "70-80%"
    elif c < 0.90:
        return "80-90%"
    else:
        return "90-100%"

def record_trade_outcome(confidence: float, is_win: bool):
    bucket = get_bucket_label(confidence)
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT n_trades, n_wins, sum_confidence FROM calibration_buckets WHERE bucket = ?;", (bucket,))
            row = cursor.fetchone()
            if row:
                n_trades = row["n_trades"] + 1
                n_wins = row["n_wins"] + (1 if is_win else 0)
                sum_conf = row["sum_confidence"] + confidence
                pred_mean = sum_conf / n_trades
                actual_wr = n_wins / n_trades
                calib_err = pred_mean - actual_wr
                import time
                cursor.execute("""
                    UPDATE calibration_buckets
                    SET n_trades = ?, n_wins = ?, sum_confidence = ?, predicted_mean = ?,
                        actual_win_rate = ?, calibration_error = ?, last_updated = ?
                    WHERE bucket = ?;
                """, (n_trades, n_wins, sum_conf, pred_mean, actual_wr, calib_err, time.time(), bucket))
                conn.commit()
        except Exception as e:
            print(f"[calibration_tracker Error] Record failed: {e}")
        finally:
            conn.close()

def get_calibration_summary() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM calibration_buckets ORDER BY bucket ASC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[calibration_tracker Error] Summary failed: {e}")
            return []
        finally:
            conn.close()

def calculate_ece() -> float:
    summary = get_calibration_summary()
    total_trades = sum(b["n_trades"] for b in summary)
    if total_trades == 0:
        return 0.0
    weighted_error = sum(b["n_trades"] * abs(b["calibration_error"]) for b in summary)
    return round(weighted_error / total_trades, 4)

init_calibration_db()
