"""
knowledge_base.py
-----------------
Phase 1B: Institutional Knowledge Base.
Persists validated statistical rules, evidence scores, Wilson CIs, and counter-evidence.
"""

import sqlite3
import json
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

def init_knowledge_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_rules (
                    rule_id TEXT PRIMARY KEY,
                    cluster_key TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_r REAL NOT NULL,
                    ci_lower REAL NOT NULL,
                    ci_upper REAL NOT NULL,
                    evidence_score REAL NOT NULL,
                    counter_evidence_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'VALIDATED',
                    recommendation TEXT,
                    trade_ids_json TEXT,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                );
            """)
            
            # Migration check for trade_ids_json
            cols = [row[1] for row in conn.execute("PRAGMA table_info(knowledge_rules);").fetchall()]
            if "trade_ids_json" not in cols:
                try:
                    conn.execute("ALTER TABLE knowledge_rules ADD COLUMN trade_ids_json TEXT;")
                except Exception:
                    pass
                    
            # Recommendation #11 Indexes for High-Frequency Rules Queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_status ON knowledge_rules(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_cluster ON knowledge_rules(cluster_key);")
            
            conn.commit()
        except Exception as e:
            print(f"[knowledge_base Error] Init failed: {e}")
        finally:
            conn.close()

def save_rule(rule_dict: Dict[str, Any]) -> bool:
    with db_lock:
        conn = get_db_connection()
        try:
            now = time.time()
            trade_ids = rule_dict.get("trade_ids", [])
            trade_ids_json = json.dumps(trade_ids) if isinstance(trade_ids, list) else str(trade_ids or "[]")
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_rules (
                    rule_id, cluster_key, sample_size, win_rate, avg_r,
                    ci_lower, ci_upper, evidence_score, counter_evidence_count,
                    status, recommendation, trade_ids_json, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                rule_dict.get("rule_id"), rule_dict.get("cluster_key"),
                rule_dict.get("sample_size", 0), rule_dict.get("win_rate", 0.0),
                rule_dict.get("avg_r", 0.0), rule_dict.get("ci_lower", 0.0),
                rule_dict.get("ci_upper", 0.0), rule_dict.get("evidence_score", 50.0),
                rule_dict.get("counter_evidence_count", 0), rule_dict.get("status", "VALIDATED"),
                rule_dict.get("recommendation", ""), trade_ids_json, rule_dict.get("created_at", now), now
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"[knowledge_base Error] Save failed: {e}")
            return False
        finally:
            conn.close()

def get_active_rules() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_rules WHERE status = 'VALIDATED' ORDER BY sample_size DESC;")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[knowledge_base Error] Fetch failed: {e}")
            return []
        finally:
            conn.close()

def get_rule_lifecycle_status(sample_size: int) -> str:
    """
    Rule Lifecycle Stage:
    - sample_size < 20: DRAFT
    - 20 <= sample_size < 50: HYPOTHESIS
    - 50 <= sample_size < 75: VALIDATED
    - sample_size >= 75: ACTIVE
    """
    if sample_size >= 75:
        return "ACTIVE"
    elif sample_size >= 50:
        return "VALIDATED"
    elif sample_size >= 20:
        return "HYPOTHESIS"
    return "DRAFT"

init_knowledge_db()
