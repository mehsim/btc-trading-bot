"""
shap_store.py
-------------
Phase 1C: Rolling Feature Importance (SHAP) History.
Stores SHAP contributions per trade and tracks rolling feature importance rank shifts.
"""

import sqlite3
import json
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

def init_shap_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shap_history (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    top_feature_1 TEXT,
                    val_1 REAL,
                    top_feature_2 TEXT,
                    val_2 REAL,
                    top_feature_3 TEXT,
                    val_3 REAL,
                    raw_shap_json TEXT
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[shap_store Error] Init failed: {e}")
        finally:
            conn.close()

def save_shap_record(trade_id: str, symbol: str, shap_dict: Dict[str, float]):
    if not shap_dict:
        return
        
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    f1, v1 = sorted_feats[0] if len(sorted_feats) > 0 else ("N/A", 0.0)
    f2, v2 = sorted_feats[1] if len(sorted_feats) > 1 else ("N/A", 0.0)
    f3, v3 = sorted_feats[2] if len(sorted_feats) > 2 else ("N/A", 0.0)
    
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO shap_history (trade_id, symbol, timestamp, top_feature_1, val_1, top_feature_2, val_2, top_feature_3, val_3, raw_shap_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (trade_id, symbol, time.time(), f1, v1, f2, v2, f3, v3, json.dumps(shap_dict)))
            conn.commit()
        except Exception as e:
            print(f"[shap_store Error] Save failed: {e}")
        finally:
            conn.close()

def get_recent_shap_history(limit=50) -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM shap_history ORDER BY timestamp DESC LIMIT ?;", (limit,))
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[shap_store Error] Fetch failed: {e}")
            return []
        finally:
            conn.close()

init_shap_db()
