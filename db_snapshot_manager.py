from typing import Optional
"""
db_snapshot_manager.py
----------------------
Phase 1 Extension: Learning Database Snapshot Manager.
Creates periodic snapshots of trading_bot.db for historical research preservation and rollback capability.
"""

import os
import shutil
import time
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "db_snapshots")

def create_db_snapshot(snapshot_tag: Optional[str] = None) -> str:
    """
    Safely creates a snapshot copy of trading_bot.db using SQLite backup API.
    """
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        
    tag = snapshot_tag or time.strftime("%Y%m%d_%H%M%S")
    dest_path = os.path.join(SNAPSHOT_DIR, f"trading_bot_snapshot_{tag}.db")
    
    try:
        source_conn = sqlite3.connect(DB_PATH)
        dest_conn = sqlite3.connect(dest_path)
        with dest_conn:
            source_conn.backup(dest_conn)
        source_conn.close()
        dest_conn.close()
        print(f"[db_snapshot_manager] Created snapshot: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"[db_snapshot_manager Error] Failed to create snapshot: {e}")
        return ""

def list_db_snapshots() -> list:
    if not os.path.exists(SNAPSHOT_DIR):
        return []
    files = [os.path.join(SNAPSHOT_DIR, f) for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".db")]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files
