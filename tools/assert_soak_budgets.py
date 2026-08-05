"""
tools/assert_soak_budgets.py
----------------------------
Performance Budget Asserts for Soak Replay:
Verifies that decision latencies, memory growth, queue depth, and database lock count stay within ratcheted performance budgets.
"""

import argparse
import json
import sys


def assert_soak_budgets(report_path: str = "soak.json") -> bool:
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Budget Failure] Could not open soak report file {report_path}: {e}")
        sys.exit(1)

    failures = []

    p99_lat = data.get("latencies_ms", {}).get("p99", 999.0)
    db_p99_lat = data.get("db_write_latencies_ms", {}).get("p99", 999.0)
    rss_growth = data.get("memory_mb", {}).get("rss_growth_pct", 999.0)
    queue_full = data.get("queue_full_count", 99)
    db_locked = data.get("database_locked_errors", 99)

    # Budget Rules
    if p99_lat > 100.0:
        failures.append(f"p99 decision latency ({p99_lat}ms) exceeded budget (100.0ms)")

    if db_p99_lat > 50.0:
        failures.append(f"p99 DB write latency ({db_p99_lat}ms) exceeded budget (50.0ms)")

    if rss_growth > 15.0:
        failures.append(f"RSS memory growth ({rss_growth}%) exceeded budget (15.0%)")

    if queue_full > 0:
        failures.append(f"Learning queue depth overflow: {queue_full} queue.Full exceptions occurred")

    if db_locked > 0:
        failures.append(f"Database locked errors occurred: {db_locked} OperationalErrors")

    if failures:
        print("\n❌ PERFORMANCE BUDGET VIOLATIONS:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\n✅ ALL PERFORMANCE BUDGETS PASSED (p99={p99_lat}ms, db_p99={db_p99_lat}ms, RSS growth={rss_growth}%)")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Budget Asserts for Soak Replay")
    parser.add_argument("report", type=str, nargs="?", default="soak.json", help="Path to soak.json report")
    args = parser.parse_args()

    assert_soak_budgets(args.report)
