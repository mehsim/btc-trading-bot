"""
audit_logger.py
----------------
Phase 1 Extension: Immutable Learning Audit Log.
Append-only audit log recording every learning action (trade close, attribution,
calibration update, rule evaluation, risk multiplier change, report generation).
"""

import sqlite3
import os
import time
import threading
from typing import Dict, Any, List

from database import db_lock, get_db_connection

def init_audit_log_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    trade_id TEXT,
                    action_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    details_json TEXT,
                    learning_version TEXT DEFAULT 'v1.0.0'
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[audit_logger Error] Init failed: {e}")
        finally:
            conn.close()

def log_learning_action(action_type: str, component: str, trade_id: str = None, details: Dict[str, Any] = None, learning_version: str = "v1.0.0"):
    import json
    with db_lock:
        conn = get_db_connection()
        try:
            now = time.time()
            details_json = json.dumps(details) if isinstance(details, dict) else str(details or {})
            conn.execute("""
                INSERT INTO learning_audit_log (timestamp, trade_id, action_type, component, details_json, learning_version)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (now, trade_id, action_type, component, details_json, learning_version))
            conn.commit()
        except Exception as e:
            print(f"[audit_logger Error] Log action failed: {e}")
        finally:
            conn.close()

def get_recent_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learning_audit_log ORDER BY log_id DESC LIMIT ?;", (limit,))
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[audit_logger Error] Fetch failed: {e}")
            return []
        finally:
            conn.close()

init_audit_log_db()
