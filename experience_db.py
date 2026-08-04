"""
experience_db.py
----------------
Phase 1A: Core Trade Experience Database.
Persists 45 high-impact curated features, full decision snapshots, and complete version lineage
for every completed trade. Thread-safe SQLite implementation.
"""

import sqlite3
import json
import os
import time
import threading

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_experience_db():
    with db_lock:
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_experience (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    feature_pipeline_version TEXT,
                    model_version TEXT,
                    ensemble_version TEXT,
                    normalizer_version TEXT,
                    learning_engine_version TEXT DEFAULT 'v1.0.0',

                    
                    -- Decision Snapshot & Reason Codes
                    decision_snapshot_json TEXT,
                    
                    -- Classification & Regime
                    market_regime TEXT,
                    regime_id TEXT,
                    signal_direction TEXT,
                    confidence REAL,
                    tf_4h TEXT,
                    tf_1h TEXT,
                    tf_15m TEXT,
                    ltf_conflict INTEGER,
                    
                    -- Market Indicators
                    adx REAL,
                    adx_slope REAL,
                    atr_pct REAL,
                    atr_percentile REAL,
                    ema_distance_pct REAL,
                    vwap_distance_pct REAL,
                    rsi REAL,
                    volume_percentile REAL,
                    volume_expansion_pct REAL,
                    funding_rate REAL,
                    oi_z_score REAL,
                    
                    -- Trade Parameters & Execution
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    leverage REAL,
                    position_size_usd REAL,
                    slippage_bps REAL,
                    execution_latency_ms REAL,
                    entry_type TEXT,
                    exit_reason TEXT,
                    
                    -- Performance & Risk Outcomes
                    realized_r REAL,
                    pnl_usd REAL,
                    mae_pct REAL,
                    mfe_pct REAL,
                    time_in_trade_min REAL,
                    individual_brier_loss REAL,
                    trade_outcome TEXT,
                    
                    -- Diagnosis & Scoring (Phases 1B / 1C)
                    failure_attribution_json TEXT,
                    decision_replay_json TEXT,
                    learning_score REAL,
                    research_priority INTEGER DEFAULT 0
                );
            """)
            
            # Recommendation #11 Indexes for High-Frequency Queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_symbol_ts ON trade_experience(symbol, timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_regime ON trade_experience(market_regime);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_score ON trade_experience(learning_score);")
            
            conn.commit()
        except Exception as e:
            print(f"[experience_db Error] Failed to initialize table: {e}")
        finally:
            conn.close()

def save_trade_experience(record: dict) -> bool:
    """
    Persists a completed trade experience record.
    Safe to call concurrently.
    """
    t_id = record.get("trade_id")
    if not t_id:
        print("[experience_db Warning] Missing trade_id in record")
        return False
        
    with db_lock:
        conn = get_db_connection()
        try:
            decision_snap = record.get("decision_snapshot")
            decision_json = json.dumps(decision_snap) if isinstance(decision_snap, dict) else record.get("decision_snapshot_json", "{}")
            
            failure_attrib = record.get("failure_attribution")
            failure_json = json.dumps(failure_attrib) if isinstance(failure_attrib, dict) else record.get("failure_attribution_json", "{}")
            
            replay_data = record.get("decision_replay")
            replay_json = json.dumps(replay_data) if isinstance(replay_data, dict) else record.get("decision_replay_json", "{}")

            conn.execute("""
                INSERT OR REPLACE INTO trade_experience (
                    trade_id, symbol, timestamp, feature_pipeline_version, model_version,
                    ensemble_version, normalizer_version, decision_snapshot_json,
                    market_regime, regime_id, signal_direction, confidence, tf_4h, tf_1h, tf_15m,
                    ltf_conflict, adx, adx_slope, atr_pct, atr_percentile, ema_distance_pct,
                    vwap_distance_pct, rsi, volume_percentile, volume_expansion_pct, funding_rate,
                    oi_z_score, entry_price, exit_price, stop_loss, take_profit, leverage,
                    position_size_usd, slippage_bps, execution_latency_ms, entry_type, exit_reason,
                    realized_r, pnl_usd, mae_pct, mfe_pct, time_in_trade_min, individual_brier_loss,
                    trade_outcome, failure_attribution_json, decision_replay_json, learning_score,
                    research_priority
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
            """, (
                t_id, record.get("symbol"), record.get("timestamp", time.time()),
                record.get("feature_pipeline_version", "v1.0"), record.get("model_version", "v2.4_prod"),
                record.get("ensemble_version", "v1.0"), record.get("normalizer_version", "v1.0"),
                decision_json, record.get("market_regime", "TRENDING"), record.get("regime_id"),
                record.get("signal_direction"), record.get("confidence", 0.0),
                record.get("tf_4h"), record.get("tf_1h"), record.get("tf_15m"),
                1 if record.get("ltf_conflict") else 0, record.get("adx", 0.0), record.get("adx_slope", 0.0),
                record.get("atr_pct", 0.0), record.get("atr_percentile", 50.0), record.get("ema_distance_pct", 0.0),
                record.get("vwap_distance_pct", 0.0), record.get("rsi", 50.0), record.get("volume_percentile", 50.0),
                record.get("volume_expansion_pct", 0.0), record.get("funding_rate", 0.0), record.get("oi_z_score", 0.0),
                record.get("entry_price", 0.0), record.get("exit_price", 0.0), record.get("stop_loss", 0.0),
                record.get("take_profit", 0.0), record.get("leverage", 1.0), record.get("position_size_usd", 0.0),
                record.get("slippage_bps", 0.0), record.get("execution_latency_ms", 0.0), record.get("entry_type", "TREND"),
                record.get("exit_reason", "MANUAL"), record.get("realized_r", 0.0), record.get("pnl_usd", 0.0),
                record.get("mae_pct", 0.0), record.get("mfe_pct", 0.0), record.get("time_in_trade_min", 0.0),
                record.get("individual_brier_loss", 0.0), record.get("trade_outcome", "LOSS" if record.get("pnl_usd", 0) < 0 else "WIN"),
                failure_json, replay_json, record.get("learning_score", 0.0), record.get("research_priority", 0)
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"[experience_db Error] Failed to save trade experience {t_id}: {e}")
            return False
        finally:
            conn.close()

def get_trade_experience(trade_id: str) -> dict:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_experience WHERE trade_id = ?;", (trade_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                for json_col in ["decision_snapshot_json", "failure_attribution_json", "decision_replay_json"]:
                    if d.get(json_col):
                        try:
                            d[json_col.replace("_json", "")] = json.loads(d[json_col])
                        except Exception:
                            pass
                return d
            return None
        except Exception as e:
            print(f"[experience_db Error] Failed to fetch trade experience {trade_id}: {e}")
            return None
        finally:
            conn.close()

def get_recent_experiences(limit=100) -> list:
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_experience ORDER BY timestamp DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            res = []
            for row in rows:
                d = dict(row)
                for json_col in ["decision_snapshot_json", "failure_attribution_json", "decision_replay_json"]:
                    if d.get(json_col):
                        try:
                            d[json_col.replace("_json", "")] = json.loads(d[json_col])
                        except Exception:
                            pass
                res.append(d)
            return res
        except Exception as e:
            print(f"[experience_db Error] Failed to fetch recent experiences: {e}")
            return []
        finally:
            conn.close()

# Initialize table upon module import
init_experience_db()
