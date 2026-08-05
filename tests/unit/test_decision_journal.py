"""
test_decision_journal.py
-------------------------
Unit test suite for decision_journal.py: asserts schema invariants, gate NULL defaults,
database writing, retention purging, and non-blocking safety.
"""

import os
import sqlite3
import time
import unittest
from decision_journal import DecisionRecord, write_decision, purge_old_decision_journal_inputs, init_decision_journal_db
from database import DB_FILE, db_lock, get_db_connection


class TestDecisionJournal(unittest.TestCase):

    def setUp(self):
        with db_lock:
            conn = get_db_connection()
            try:
                conn.execute("DROP TABLE IF EXISTS decision_journal;")
                conn.commit()
            finally:
                conn.close()
        init_decision_journal_db()

    def test_gate_defaults_are_none_null(self):
        """Invariant Check: All gate columns MUST default to None (NULL), never 0 or 1."""
        rec = DecisionRecord(symbol="BTCUSDT", interval="15")
        self.assertIsNone(rec.gate_var_pct)
        self.assertIsNone(rec.gate_var_pass)
        self.assertIsNone(rec.gate_stress_pct)
        self.assertIsNone(rec.gate_stress_pass)
        self.assertIsNone(rec.gate_stress_cvar)
        self.assertIsNone(rec.gate_heat_pct)
        self.assertIsNone(rec.gate_heat_pass)
        self.assertIsNone(rec.gate_corr)
        self.assertIsNone(rec.gate_corr_pass)
        self.assertIsNone(rec.gate_cost_bp)
        self.assertIsNone(rec.gate_cost_pass)
        self.assertIsNone(rec.kelly_raw)
        self.assertIsNone(rec.kelly_effective)
        self.assertIsNone(rec.mhi_score)
        self.assertIsNone(rec.mhi_cap)

    def test_gate_stamping_and_snapshot(self):
        """Verify gate stamping methods update corresponding fields."""
        rec = DecisionRecord(symbol="BTCUSDT", interval="15")
        rec.gate("var", value=4.5, passed=True)
        rec.gate("stress", value=18.2, passed=True)
        rec.gate_stress_cvar = 22.1
        rec.gate("heat", value=110.0, passed=True)
        rec.gate("corr", value=0.42, passed=True)
        rec.snapshot(test_key="test_value")

        self.assertEqual(rec.gate_var_pct, 4.5)
        self.assertEqual(rec.gate_var_pass, 1)
        self.assertEqual(rec.gate_stress_pct, 18.2)
        self.assertEqual(rec.gate_stress_pass, 1)
        self.assertEqual(rec.gate_stress_cvar, 22.1)
        self.assertEqual(rec.gate_heat_pct, 110.0)
        self.assertEqual(rec.gate_heat_pass, 1)
        self.assertEqual(rec.gate_corr, 0.42)
        self.assertEqual(rec.gate_corr_pass, 1)
        self.assertEqual(rec._inputs["test_key"], "test_value")

    def test_write_decision_to_database(self):
        """Verify write_decision persists records to decision_journal table in SQLite."""
        rec = DecisionRecord(symbol="ETHUSDT", interval="60")
        rec.signal_source = "ML_ENSEMBLE"
        rec.is_fallback = 0
        rec.direction = "Bullish"
        rec.raw_confidence = 0.82
        rec.calibrated_conf = 0.85
        rec.outcome = "EXECUTED"
        rec.position_size_usd = 250.0
        rec.leverage = 5.0
        rec.trade_id = "ETHUSDT_test_123"
        rec.gate("var", value=3.2, passed=True)

        success = write_decision(rec)
        self.assertTrue(success)

        with db_lock:
            conn = get_db_connection()
            try:
                row = conn.execute("SELECT * FROM decision_journal WHERE decision_id = ?;", (rec.decision_id,)).fetchone()
                self.assertIsNotNone(row)
                row_dict = dict(row)
                self.assertEqual(row_dict["symbol"], "ETHUSDT")
                self.assertEqual(row_dict["interval"], "60")
                self.assertEqual(row_dict["signal_source"], "ML_ENSEMBLE")
                self.assertEqual(row_dict["is_fallback"], 0)
                self.assertEqual(row_dict["outcome"], "EXECUTED")
                self.assertEqual(row_dict["gate_var_pct"], 3.2)
                self.assertEqual(row_dict["gate_var_pass"], 1)
                self.assertIsNone(row_dict["gate_stress_pct"])  # Un-evaluated gate is NULL
                self.assertIsNone(row_dict["gate_stress_pass"]) # Un-evaluated gate is NULL
            finally:
                conn.close()

    def test_purge_old_inputs_retention(self):
        """Verify purge_old_decision_journal_inputs clears inputs_json for old rows while retaining gate columns."""
        rec_old = DecisionRecord(symbol="SOLUSDT", interval="15")
        rec_old.ts = time.time() - (100 * 86400.0)  # 100 days old
        rec_old.outcome = "REJECTED"
        rec_old.reject_reason = "REJECTED: Portfolio heat limit exceeded"
        rec_old.snapshot(heavy_payload="A" * 1000)
        rec_old.gate("heat", value=310.0, passed=False)

        write_decision(rec_old)

        purged_count = purge_old_decision_journal_inputs(days=90)
        self.assertGreaterEqual(purged_count, 1)

        with db_lock:
            conn = get_db_connection()
            try:
                row = conn.execute("SELECT inputs_json, gate_heat_pct, gate_heat_pass, outcome FROM decision_journal WHERE decision_id = ?;", (rec_old.decision_id,)).fetchone()
                self.assertIsNotNone(row)
                self.assertIsNone(row["inputs_json"])
                self.assertEqual(row["gate_heat_pct"], 310.0)
                self.assertEqual(row["gate_heat_pass"], 0)
                self.assertEqual(row["outcome"], "REJECTED")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
