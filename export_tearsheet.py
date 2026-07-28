import sqlite3
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

def export_performance_tearsheet(db_path: str = "trading_bot.db") -> dict:
    """
    Independent Performance Auditor.
    Reads raw trade execution logs directly from SQLite database without importing strategy modules.
    Derives unbiased performance statistics: Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, and Profit Factor.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return {}

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM completed_trades ORDER BY exit_time ASC", conn)
    except Exception as e:
        print(f"Error reading completed_trades table: {e}")
        conn.close()
        return {}
    conn.close()

    if df.empty:
        print("No completed trade records found in database.")
        return {}

    # Parse numeric fields safely
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce").fillna(0.0)
    df["position_size_usd"] = pd.to_numeric(df["position_size_usd"], errors="coerce").fillna(0.0)
    
    total_trades = len(df)
    wins = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] < 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0
    
    total_pnl = df["pnl_usd"].sum()
    gross_profit = wins["pnl_usd"].sum()
    gross_loss = abs(losses["pnl_usd"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
    
    # Cumulative PnL and Drawdown Curve
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    df["peak"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["cum_pnl"] - df["peak"]
    max_drawdown_usd = abs(float(df["drawdown"].min())) if not df["drawdown"].empty else 0.0
    
    initial_balance = float(df["balance"].iloc[0]) if "balance" in df.columns and len(df) > 0 and pd.notnull(df["balance"].iloc[0]) else 80.0
    peak_balance = max(initial_balance, initial_balance + df["peak"].max()) if not df.empty else initial_balance
    max_drawdown_pct = (max_drawdown_usd / peak_balance * 100.0) if peak_balance > 0 else 0.0
    
    # Risk-Adjusted Ratios (Sharpe, Sortino, Calmar)
    pnl_series = df["pnl_usd"].values
    mean_pnl = np.mean(pnl_series)
    std_pnl = np.std(pnl_series, ddof=1) if len(pnl_series) > 1 else 1e-6
    
    # Sharpe Ratio (assumes ~252 trading sessions per year)
    sharpe_ratio = float((mean_pnl / (std_pnl + 1e-8)) * np.sqrt(252)) if std_pnl > 0 else 0.0
    
    # Sortino Ratio (downside risk only)
    downside_returns = pnl_series[pnl_series < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 1e-6
    sortino_ratio = float((mean_pnl / (downside_std + 1e-8)) * np.sqrt(252)) if downside_std > 0 else 0.0
    
    # Calmar Ratio
    calmar_ratio = float(total_pnl / max_drawdown_usd) if max_drawdown_usd > 0 else 0.0

    tearsheet = {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "total_pnl_usd": round(float(total_pnl), 2),
        "gross_profit_usd": round(float(gross_profit), 2),
        "gross_loss_usd": round(float(gross_loss), 2),
        "profit_factor": round(float(profit_factor), 2),
        "max_drawdown_usd": round(float(max_drawdown_usd), 2),
        "max_drawdown_pct": round(float(max_drawdown_pct), 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2)
    }
    
    print("\n==================================================")
    print("📊 INDEPENDENT PERFORMANCE AUDITOR TEARSHEET")
    print("==================================================")
    print(f"• Total Trades Executed : {total_trades}")
    print(f"• Win Rate              : {win_rate:.2f}% ({win_count} W / {loss_count} L)")
    print(f"• Net Realized PnL      : ${total_pnl:+.2f}")
    print(f"• Profit Factor         : {profit_factor:.2f}")
    print(f"• Max Peak Drawdown     : -${max_drawdown_usd:.2f} (-{max_drawdown_pct:.2f}%)")
    print(f"• Sharpe Ratio (Ann.)   : {sharpe_ratio:.2f}")
    print(f"• Sortino Ratio (Ann.)  : {sortino_ratio:.2f}")
    print(f"• Calmar Ratio          : {calmar_ratio:.2f}")
    print("==================================================\n")
    
    return tearsheet

if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else "trading_bot.db"
    export_performance_tearsheet(db_file)
