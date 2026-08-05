"""
learning_scorer.py & research_queue.py
---------------------------------------
Phase 1C: Learning Scorer & Research Queue.
Assigns a 0-100 Learning Score to every completed trade based on individual Brier loss,
counterfactual disagreement, and cluster rarity. Routes score >= 80 to research queue.
"""

import sqlite3
import os
import time
import threading
from typing import Dict, Any, List

from database import db_lock, get_db_connection

def init_research_queue_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_queue (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    learning_score REAL NOT NULL,
                    brier_loss REAL NOT NULL,
                    trade_outcome TEXT NOT NULL,
                    reason TEXT,
                    status TEXT DEFAULT 'PENDING_REVIEW'
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[research_queue Error] Init failed: {e}")
        finally:
            conn.close()

def calculate_learning_score(record: Dict[str, Any], cf_scenarios: List[Dict[str, Any]] = None) -> float:
    brier = record.get("individual_brier_loss", 0.0)
    conf = record.get("confidence", 0.5)
    pnl = record.get("pnl_usd", 0.0)
    ltf_conflict = record.get("ltf_conflict", 0)
    
    score = 20.0  # Base score
    
    # High Brier loss penalty/score boost (model was very wrong)
    if brier > 0.5:
        score += 30.0
    elif brier > 0.25:
        score += 15.0
        
    # High confidence loss (overconfidence)
    if pnl < 0 and conf >= 0.80:
        score += 25.0
        
    # LTF conflict presence
    if ltf_conflict:
        score += 15.0
        
    # Counterfactual disagreement
    if cf_scenarios:
        best_diff = max(s.get("diff_vs_actual_r", 0.0) for s in cf_scenarios)
        if best_diff > 0.5:
            score += 10.0
            
    final_score = round(min(100.0, score), 1)
    
    # Queue for research if score >= 80
    if final_score >= 80.0:
        enqueue_research_trade(
            trade_id=record.get("trade_id", "UNKNOWN"),
            symbol=record.get("symbol", "BTCUSDT"),
            learning_score=final_score,
            brier_loss=brier,
            outcome="LOSS" if pnl < 0 else "WIN",
            reason=f"High learning value ({final_score:.0f}/100) — Brier={brier:.2f}, Conf={conf*100:.0f}%"
        )
        
    return final_score

def enqueue_research_trade(trade_id: str, symbol: str, learning_score: float, brier_loss: float, outcome: str, reason: str):
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO research_queue (trade_id, symbol, timestamp, learning_score, brier_loss, trade_outcome, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (trade_id, symbol, time.time(), learning_score, brier_loss, outcome, reason))
            conn.commit()
        except Exception as e:
            print(f"[research_queue Error] Enqueue failed: {e}")
        finally:
            conn.close()

def get_pending_research_queue() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM research_queue WHERE status = 'PENDING_REVIEW' ORDER BY learning_score DESC;")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"[research_queue Error] Fetch failed: {e}")
            return []
        finally:
            conn.close()

init_research_queue_db()
