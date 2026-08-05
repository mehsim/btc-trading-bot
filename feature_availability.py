"""
feature_availability.py
-----------------------
Phase 1 Extension: Feature Availability & Quality Metrics Tracker.
Tracks availability % of features over time to detect unreliable API feeds or indicator bugs.
"""

import sqlite3
import os
import time
import threading
from typing import Dict, Any, List

from database import db_lock, get_db_connection

def init_availability_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_availability_stats (
                    feature TEXT PRIMARY KEY,
                    total_samples INTEGER DEFAULT 0,
                    available_samples INTEGER DEFAULT 0,
                    availability_pct REAL DEFAULT 100.0,
                    last_updated REAL DEFAULT 0.0
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[feature_availability Error] Init failed: {e}")
        finally:
            conn.close()

def record_feature_sample(feature: str, is_available: bool):
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT total_samples, available_samples FROM feature_availability_stats WHERE feature = ?;", (feature,))
            row = cursor.fetchone()
            if not row:
                tot = 1
                avail = 1 if is_available else 0
            else:
                tot = row["total_samples"] + 1
                avail = row["available_samples"] + (1 if is_available else 0)
                
            pct = round((avail / tot) * 100.0, 2)
            now = time.time()
            cursor.execute("""
                INSERT OR REPLACE INTO feature_availability_stats (feature, total_samples, available_samples, availability_pct, last_updated)
                VALUES (?, ?, ?, ?, ?);
            """, (feature, tot, avail, pct, now))
            conn.commit()
        except Exception as e:
            print(f"[feature_availability Error] Record failed: {e}")
        finally:
            conn.close()

def get_feature_availability_report() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feature_availability_stats ORDER BY availability_pct ASC;")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[feature_availability Error] Report failed: {e}")
            return []
        finally:
            conn.close()

init_availability_db()
