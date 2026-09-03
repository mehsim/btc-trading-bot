"""
decision_journal.py
-------------------
Append-only per-decision journal table and DecisionRecord dataclass.
Tracks every trading decision (executed, rejected, errored) with gate verdicts,
model provenance, market context, and input payloads.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import sqlite3
from database import db_lock, get_db_connection

get_connection = get_db_connection
_db_initialized = False


class ReasonCode:
    TCM_NET_EDGE_NEGATIVE = "TCM_NET_EDGE_NEGATIVE"
    RR_BELOW_FLOOR = "RR_BELOW_FLOOR"
    EXPECTANCY_NEGATIVE = "EXPECTANCY_NEGATIVE"
    MACRO_OPPOSITION = "MACRO_OPPOSITION"
    FLASH_CRASH_ACTIVE = "FLASH_CRASH_ACTIVE"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    SPREAD_WIDENING = "SPREAD_WIDENING"
    KELLY_EDGE_NON_POSITIVE = "KELLY_EDGE_NON_POSITIVE"
    RISK_CHECKLIST_BLOCKED = "RISK_CHECKLIST_BLOCKED"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    CONFIDENCE_BELOW_DYNAMIC_THRESHOLD = "CONFIDENCE_BELOW_DYNAMIC_THRESHOLD"
    CIRCUIT_BREAKER_ACTIVE = "CIRCUIT_BREAKER_ACTIVE"
    BOT_STOPPED = "BOT_STOPPED"
    MARGIN_GUARD_EXCEEDED = "MARGIN_GUARD_EXCEEDED"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    EXECUTION_VALIDATION_FAILED = "EXECUTION_VALIDATION_FAILED"
    MAX_CONCURRENT_POSITIONS = "MAX_CONCURRENT_POSITIONS"
    COOL_OFF_ACTIVE = "COOL_OFF_ACTIVE"
    FUNDING_BLOCK = "FUNDING_BLOCK"
    HIGH_CONFORMAL_UNCERTAINTY = "HIGH_CONFORMAL_UNCERTAINTY"
    DIRECTIONAL_MASS_FAILED = "DIRECTIONAL_MASS_FAILED"
    CALIBRATOR_NON_VIABLE = "CALIBRATOR_NON_VIABLE"
    ADX_BELOW_FLOOR = "ADX_BELOW_FLOOR"
    CLUSTER_LIMIT = "CLUSTER_LIMIT"
    ALREADY_ACTIVE = "ALREADY_ACTIVE"
    NEWS_BLOCK = "NEWS_BLOCK"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"
    UNCERTAINTY_HIGH = "UNCERTAINTY_HIGH"
    VOLUME_COMPRESSION = "VOLUME_COMPRESSION"
    ATR_SPIKE = "ATR_SPIKE"
    MANUAL_BYPASS = "MANUAL_BYPASS"


@dataclass
class DecisionRecord:
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)
    candle_timestamp: Optional[int] = None
    symbol: str = ""
    interval: str = ""

    # signal provenance
    signal_source: str = "UNSET"
    is_fallback: int = 0
    direction: Optional[str] = None
    raw_confidence: Optional[float] = None
    calibrated_conf: Optional[float] = None
    calibrator_version: Optional[str] = None
    calibrator_ece: Optional[float] = None
    directional_mass_passed: Optional[int] = None
    is_calibrated: int = 0
    is_floor_scaled: int = 0

    # model identity
    model_version: Optional[str] = None
    feature_hash: Optional[str] = None
    manifest_schema: Optional[int] = None
    git_sha: Optional[str] = None

    # market context
    regime: Optional[str] = None
    adx: Optional[float] = None
    atr_norm: Optional[float] = None
    liquidity_score: Optional[float] = None
    spread_bp: Optional[float] = None

    # economics and sizing intent
    expected_value: Optional[float] = None
    expected_rr: Optional[float] = None
    round_trip_cost_bp: Optional[float] = None

    # gate verdicts: NULL means NOT EVALUATED (default None)
    gate_var_pct: Optional[float] = None
    gate_var_pass: Optional[int] = None
    gate_stress_pct: Optional[float] = None
    gate_stress_pass: Optional[int] = None
    gate_stress_cvar: Optional[float] = None
    gate_heat_pct: Optional[float] = None
    gate_heat_pass: Optional[int] = None
    gate_corr: Optional[float] = None
    gate_corr_pass: Optional[int] = None
    gate_cost_bp: Optional[float] = None
    gate_cost_pass: Optional[int] = None

    kelly_raw: Optional[float] = None
    kelly_effective: Optional[float] = None
    mhi_score: Optional[float] = None
    mhi_cap: Optional[float] = None

    # outcome
    outcome: str = "ERROR"  # pessimistic default; overwritten on EXECUTED / REJECTED
    reject_reason: Optional[str] = None
    reason_code: Optional[str] = None
    position_size_usd: Optional[float] = None
    leverage: Optional[float] = None
    trade_id: Optional[str] = None
    order_payload_json: Optional[str] = None
    venue_response_json: Optional[str] = None

    # reproducibility
    simulation_seed: Optional[int] = None
    schema_version: int = 1

    # Internal private dict for snapshotting arbitrary inputs
    _inputs: Dict[str, Any] = field(default_factory=dict)

    def gate(self, name: str, value: Optional[float] = None, passed: Optional[bool] = None):
        """Stamps gate verdict onto record. Default value & passed are None (NULL)."""
        if name == "cost":
            col_name = "gate_cost_bp"
        elif name == "corr":
            col_name = "gate_corr"
        else:
            col_name = f"gate_{name}_pct"

        if hasattr(self, col_name):
            setattr(self, col_name, value)
        pass_col = f"gate_{name}_pass"
        if hasattr(self, pass_col):
            setattr(self, pass_col, None if passed is None else int(passed))
        return self

    def snapshot(self, **kw):
        """Snapshots inputs into internal dict for inputs_json serialization."""
        self._inputs.update(kw)
        return self


_last_initialized_db = None

def init_decision_journal_db(force: bool = False):
    global _last_initialized_db
    import database
    curr_db = database.get_db_path()
    if _last_initialized_db == curr_db and not force:
        return
    with db_lock:
        if _last_initialized_db == curr_db and not force:
            return
        conn = get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_journal (
                    decision_id        TEXT PRIMARY KEY,
                    ts                 REAL NOT NULL,
                    candle_timestamp   INTEGER,
                    symbol             TEXT NOT NULL,
                    interval           TEXT NOT NULL,

                    signal_source      TEXT NOT NULL,
                    is_fallback        INTEGER NOT NULL,
                    direction          TEXT,
                    raw_confidence     REAL,
                    calibrated_conf    REAL,
                    calibrator_version TEXT,
                    calibrator_ece     REAL,

                    model_version      TEXT,
                    feature_hash       TEXT,
                    manifest_schema    INTEGER,
                    git_sha            TEXT,

                    regime             TEXT,
                    adx                REAL,
                    atr_norm           REAL,
                    liquidity_score    REAL,
                    spread_bp          REAL,

                    expected_value     REAL,
                    expected_rr        REAL,
                    round_trip_cost_bp REAL,

                    gate_var_pct       REAL, gate_var_pass       INTEGER,
                    gate_stress_pct    REAL, gate_stress_pass    INTEGER,
                    gate_stress_cvar   REAL,
                    gate_heat_pct      REAL, gate_heat_pass      INTEGER,
                    gate_corr          REAL, gate_corr_pass      INTEGER,
                    gate_cost_bp       REAL, gate_cost_pass      INTEGER,

                    kelly_raw          REAL, kelly_effective     REAL,
                    mhi_score          REAL, mhi_cap             REAL,

                    outcome            TEXT NOT NULL,
                    reject_reason      TEXT,
                    reason_code        TEXT,
                    position_size_usd  REAL,
                    leverage           REAL,
                    trade_id           TEXT,
                    order_payload_json TEXT,
                    venue_response_json TEXT,

                    simulation_seed    INTEGER,
                    inputs_json        TEXT,
                    schema_version     INTEGER NOT NULL DEFAULT 1
                );
            """)

            for col_n, col_t in [
                ("candle_timestamp", "INTEGER"),
                ("directional_mass_passed", "INTEGER"),
                ("is_calibrated", "INTEGER"),
                ("is_floor_scaled", "INTEGER"),
                ("order_payload_json", "TEXT"),
                ("venue_response_json", "TEXT"),
                ("reason_code", "TEXT"),
                ("expected_value", "REAL"),
                ("expected_rr", "REAL"),
                ("round_trip_cost_bp", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE decision_journal ADD COLUMN {col_n} {col_t};")
                except sqlite3.OperationalError:
                    pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_dj_ts        ON decision_journal(ts DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dj_sym_ts    ON decision_journal(symbol, ts DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dj_outcome   ON decision_journal(outcome, ts DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dj_source    ON decision_journal(signal_source, ts DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dj_trade     ON decision_journal(trade_id);")

            conn.commit()
            _last_initialized_db = curr_db
        except Exception as e:
            print(f"[DecisionJournal ERROR] DB Init failed: {e}")
        finally:
            conn.close()


def write_decision(rec: DecisionRecord) -> bool:
    """
    Appends a DecisionRecord to decision_journal table in SQLite.
    Never raises an exception or blocks execution; logs errors gracefully.
    """
    try:
        init_decision_journal_db()
        cols_dict = {k: v for k, v in asdict(rec).items() if not k.startswith("_")}
        cols_dict["inputs_json"] = json.dumps(rec._inputs, default=str)

        col_names = list(cols_dict.keys())
        placeholders = ",".join(["?"] * len(col_names))
        sql = f"INSERT INTO decision_journal ({','.join(col_names)}) VALUES ({placeholders})"  # nosec B608

        with db_lock:
            conn = get_db_connection()
            try:
                conn.execute(sql, tuple(cols_dict.values()))
                conn.commit()
                return True
            except Exception as e:
                conn.rollback()
                print(f"[DecisionJournal ERROR] Failed to write decision {rec.decision_id}: {e}")
                return False
            finally:
                conn.close()
    except Exception as ex:
        print(f"[DecisionJournal ERROR] General failure in write_decision: {ex}")
        return False


def purge_old_decision_journal_inputs(days: int = 90) -> int:
    """Sets inputs_json=NULL for decision_journal rows older than specified days to manage storage."""
    try:
        init_decision_journal_db()
        cutoff_ts = time.time() - (days * 86400.0)
        with db_lock:
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE decision_journal SET inputs_json = NULL WHERE ts < ? AND inputs_json IS NOT NULL;", (cutoff_ts,))
                count = cursor.rowcount
                conn.commit()
                return count
            except Exception as e:
                conn.rollback()
                print(f"[DecisionJournal ERROR] Retention purge failed: {e}")
                return 0
            finally:
                conn.close()
    except Exception as ex:
        print(f"[DecisionJournal ERROR] General failure in purge: {ex}")
        return 0
