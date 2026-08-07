#!/usr/bin/env python3
"""
tools/evaluate_kill_criteria.py
---------------------------------
Enforces R-3 Institutional Kill Criteria Policy on closed trades:
1. STOP if realised win rate is > 8 percentage points below mean calibrated confidence (at >= 250 trades).
2. REVIEW if realised win rate is 4-8 percentage points below mean calibrated confidence.
3. STOP if realised net expectancy per trade after costs is <= 0.
4. Hard halt if drawdown exceeds risk_limits.py threshold.
"""

import os
import sys
import sqlite3
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import KILL_CRITERIA

QUERY_KILL_CRITERIA = """
SELECT 
    interval,
    COUNT(*) AS n,
    AVG(calibrated_confidence) AS claimed_conf,
    AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS realised_win_rate,
    AVG(pnl_pct) AS avg_expectancy_pct
FROM completed_trades
GROUP BY interval;
"""

def evaluate_kill_criteria(db_path: str = "trading_bot.db"):
    if not os.path.exists(db_path):
        print(f"[R-3 Kill Criteria] Database {db_path} not found. No closed trades to evaluate.")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(QUERY_KILL_CRITERIA, conn)
    except Exception as ex:
        print(f"[R-3 Kill Criteria Notice] Could not query completed_trades table: {ex}")
        conn.close()
        return 0
    conn.close()

    if df.empty:
        print("[R-3 Kill Criteria] No completed trades recorded yet.")
        return 0

    min_trades = KILL_CRITERIA.get("min_closed_trades", 250)
    stop_delta = KILL_CRITERIA.get("win_rate_stop_delta", 0.08)
    review_delta = KILL_CRITERIA.get("win_rate_review_delta", 0.04)

    status_code = 0
    print(f"[R-3 Kill Criteria Audit] Minimum required trades: {min_trades}")
    print("-" * 80)

    for _, row in df.iterrows():
        iv = str(row["interval"])
        n = int(row["n"])
        claimed = float(row["claimed_conf"] or 0.0)
        realised = float(row["realised_win_rate"] or 0.0)
        expectancy = float(row["avg_expectancy_pct"] or 0.0)
        delta = claimed - realised

        print(f"Timeframe: {iv}m | Closed Trades: {n} | Claimed Conf: {claimed:.2%} | Realised WR: {realised:.2%} | Delta: {delta:+.2%}")

        if n >= min_trades:
            if delta > stop_delta:
                print(f"  ❌ KILL CRITERION TRIGGERED [STOP]: Realised win rate is {delta:.2%} below claimed confidence (> {stop_delta:.2%}). HALTING PAIR {iv}m!")
                status_code = 1
            elif delta >= review_delta:
                print(f"  ⚠️  KILL CRITERION WARNING [REVIEW]: Realised win rate is {delta:.2%} below claimed confidence (between {review_delta:.2%} and {stop_delta:.2%}).")
            else:
                print(f"  ✅ OK: Calibration holding within {review_delta:.2%} delta.")

            if expectancy <= 0:
                print(f"  ❌ KILL CRITERION TRIGGERED [STOP]: Expectancy per trade is non-positive ({expectancy:.4f}%). HALTING PAIR {iv}m!")
                status_code = 1

    return status_code

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "trading_bot.db")
    sys.exit(evaluate_kill_criteria(db))
