#!/usr/bin/env python3
"""
tools/measure_calibration.py
------------------------------
R-4: Measure Calibration Before Sharpe Ratio
Run weekly from ~50 trades onward to verify whether cross-validated edge generalizes to live execution.
Buckets calibrated_confidence (0.5, 0.6, 0.7...) and compares against realised win rate.
"""

import os
import sys
import sqlite3
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

QUERY_CALIBRATION = """
SELECT 
    interval,
    ROUND(calibrated_confidence, 1) AS bucket,
    COUNT(*) AS n,
    AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS realised_win_rate,
    AVG(pnl_pct) AS avg_pnl_pct
FROM completed_trades
WHERE interval IN ('15', '60', '240')
GROUP BY interval, bucket
ORDER BY interval, bucket;
"""

def measure_calibration(db_path: str = "trading_bot.db"):
    if not os.path.exists(db_path):
        print(f"[R-4 Calibration Audit] Database {db_path} not found. No closed trades to measure.")
        return

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(QUERY_CALIBRATION, conn)
    except Exception as ex:
        print(f"[R-4 Calibration Audit Notice] Could not query completed_trades: {ex}")
        conn.close()
        return
    conn.close()

    if df.empty:
        print("[R-4 Calibration Audit] No completed trades available in target intervals.")
        return

    print("=" * 80)
    print("R-4 CALIBRATION VS REALISED WIN RATE AUDIT")
    print("=" * 80)
    print(f"{'Interval':<10} | {'Bucket':<10} | {'Trades (N)':<12} | {'Realised WR':<15} | {'Avg PnL %':<12}")
    print("-" * 80)

    for _, row in df.iterrows():
        iv = str(row["interval"])
        bkt = float(row["bucket"])
        n = int(row["n"])
        wr = float(row["realised_win_rate"] or 0.0)
        pnl = float(row["avg_pnl_pct"] or 0.0)
        print(f"{iv:<10} | {bkt:<10.1f} | {n:<12} | {wr:<15.2%} | {pnl:<12.4f}%")

    print("-" * 80)

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "trading_bot.db")
    measure_calibration(db)
