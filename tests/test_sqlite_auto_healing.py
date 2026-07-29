"""
tests/test_sqlite_auto_healing.py
----------------------------------
Rigorously tests safe_get_sqlite_conn under:
1. High-concurrency multithreaded read/write operations (busy_timeout verification)
2. Database disk image corruption auto-recovery (detecting malformed DB, purging, rebuilding)
"""

import pytest
import os
import time
import threading
import sqlite3
from data import safe_get_sqlite_conn


def test_sqlite_multithreaded_concurrency(tmp_path):
    db_file = str(tmp_path / "test_concurrency.db")

    # Initialize table
    conn = safe_get_sqlite_conn(db_file)
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, val TEXT);")
    conn.commit()
    conn.close()

    errors = []

    def worker_thread(thread_id):
        try:
            for i in range(10):
                c = safe_get_sqlite_conn(db_file)
                c.execute("INSERT INTO test_data (val) VALUES (?);", (f"thread_{thread_id}_item_{i}",))
                c.commit()
                c.close()
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0

    conn_final = safe_get_sqlite_conn(db_file)
    count = conn_final.execute("SELECT COUNT(*) FROM test_data;").fetchone()[0]
    conn_final.close()
    assert count == 50


def test_sqlite_corruption_auto_recovery(tmp_path):
    db_file = str(tmp_path / "test_corrupt.db")

    # Create a valid database
    conn = safe_get_sqlite_conn(db_file)
    conn.execute("CREATE TABLE sample (id INT);")
    conn.commit()
    conn.close()

    # Intentionally corrupt the database file with garbage bytes
    with open(db_file, "wb") as f:
        f.write(b"CORRUPT_INVALID_HEADER_GARBAGE_BYTES_1234567890")

    # safe_get_sqlite_conn should detect corruption, remove corrupt file, and return fresh conn
    recovered_conn = safe_get_sqlite_conn(db_file)
    recovered_conn.execute("CREATE TABLE sample (id INT);")
    recovered_conn.execute("INSERT INTO sample VALUES (1);")
    recovered_conn.commit()
    res = recovered_conn.execute("SELECT COUNT(*) FROM sample;").fetchone()[0]
    recovered_conn.close()

    assert res == 1
