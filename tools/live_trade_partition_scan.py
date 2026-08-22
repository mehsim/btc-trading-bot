#!/usr/bin/env python3
"""
Conditional Partition Scan on Completed Live Trades (Hypothesis Generation Only)
Evaluates 231 clean live trades from trading_bot.db across pre-registered partitions
and reports Bonferroni-adjusted p-values with strict underpowered warnings.
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

def run_partition_scan(db_path: str = "trading_bot.db"):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT symbol, interval, direction, confidence, leverage, entry_heat,
           entry_correlation, entry_drawdown_pct, mae, mfe, duration_seconds,
           success, change_pct, pnl_usd 
    FROM completed_trades 
    WHERE reason IS NOT NULL AND reason NOT LIKE '%RECOVERY SCAN%';
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    total_n = len(df)
    if total_n == 0:
        print("No completed trades found.")
        return

    # Base baseline statistics
    df["win"] = df["success"].astype(int)
    base_wins = df["win"].sum()
    base_win_rate = base_wins / total_n
    base_mean_pnl = df["pnl_usd"].mean()
    base_mean_ret = df["change_pct"].mean()

    # Pre-register candidate partitions across populated columns
    # Partitions MUST be defined before evaluating outcomes
    candidate_partitions = []

    # 1. Symbol partitions (9 symbols)
    for sym in sorted(df["symbol"].unique()):
        candidate_partitions.append(("symbol", sym, df["symbol"] == sym))

    # 2. Interval partitions
    for iv in sorted(df["interval"].unique()):
        candidate_partitions.append(("interval", str(iv), df["interval"] == iv))

    # 3. Direction partitions
    for direction in ["Bullish", "Bearish"]:
        candidate_partitions.append(("direction", direction, df["direction"] == direction))

    # 4. Confidence partitions (median split)
    med_conf = df["confidence"].median()
    candidate_partitions.append(("confidence", f"high (>{med_conf:.2f})", df["confidence"] > med_conf))
    candidate_partitions.append(("confidence", f"low (<={med_conf:.2f})", df["confidence"] <= med_conf))

    # 5. Leverage partitions
    med_lev = df["leverage"].median()
    candidate_partitions.append(("leverage", f"high (>{med_lev:.0f}x)", df["leverage"] > med_lev))
    candidate_partitions.append(("leverage", f"low (<={med_lev:.0f}x)", df["leverage"] <= med_lev))

    # Total pre-registered tests
    num_tests = len(candidate_partitions)

    records = []
    for category, label, mask in candidate_partitions:
        sub = df[mask]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        k_sub = int(sub["win"].sum())
        win_rate_sub = k_sub / n_sub
        mean_pnl_sub = sub["pnl_usd"].mean()
        mean_ret_sub = sub["change_pct"].mean()

        # Two-sided binomial test vs base win rate
        binom_res = stats.binomtest(k_sub, n_sub, p=base_win_rate, alternative="two-sided")
        raw_p = binom_res.pvalue
        adj_p = min(1.0, raw_p * num_tests)  # Bonferroni adjustment

        records.append({
            "Category": category,
            "Partition": label,
            "N": n_sub,
            "Wins": k_sub,
            "WinRate": f"{win_rate_sub * 100:.1f}%",
            "WinRate_diff": f"{(win_rate_sub - base_win_rate) * 100:+.1f}%",
            "Mean_PnL_$": f"{mean_pnl_sub:+.2f}",
            "Mean_Ret_%": f"{mean_ret_sub:+.2f}%",
            "p_value_raw": raw_p,
            "p_value_bonferroni": adj_p,
            "Significant_5pct": "YES" if adj_p < 0.05 else "NO"
        })

    res_df = pd.DataFrame(records)

    print("=" * 105)
    print("CONDITIONAL PARTITION SCAN ON LIVE TRADES (HYPOTHESIS GENERATION ONLY)")
    print(f"Population: N={total_n} trades | Base Win Rate: {base_win_rate*100:.2f}% | Base Mean PnL: ${base_mean_pnl:+.2f} | Base Mean Ret: {base_mean_ret:+.2f}%")
    print(f"Pre-Registered Partitions: {num_tests} tests (Bonferroni Multiplier M={num_tests})")
    print("=" * 105)
    print(res_df.to_string(index=False, formatters={
        "p_value_raw": lambda x: f"{x:.4f}",
        "p_value_bonferroni": lambda x: f"{x:.4f}"
    }))
    print("=" * 105)
    print("\nCRITICAL STATISTICAL NOTICE:")
    print("1. All partitions are underpowered (N=231 total across 9 symbols; minimum required sample size for 80% power")
    print("   at alpha=0.05 is N >= 1,171 per arm for 60m and N >= 285 per arm for 15m).")
    print("2. The outputs above represent exploratory HYPOTHESES ONLY, not causal relationships.")
    print("3. DO NOT hardcode or enforce these partitions as live gating rules.")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    run_partition_scan()
