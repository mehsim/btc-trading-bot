import numpy as np
import pandas as pd
import json
from datetime import datetime

def run_monte_carlo_bootstrap(returns: list, num_simulations: int = 10000) -> dict:
    if not returns:
        return {"p5": 0.0, "p50": 0.0, "p95": 0.0, "p5_mdd": 0.0}
    
    ret_arr = np.array(returns)
    n_trades = len(ret_arr)
    sim_returns = []
    sim_mdds = []
    
    for _ in range(num_simulations):
        sampled = np.random.choice(ret_arr, size=n_trades, replace=True)
        equity_curve = np.cumprod(1.0 + sampled / 100.0)
        final_ret = (equity_curve[-1] - 1.0) * 100.0
        
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak * 100.0
        max_dd = np.max(dd)
        
        sim_returns.append(final_ret)
        sim_mdds.append(max_dd)
        
    return {
        "p5_return": float(np.percentile(sim_returns, 5)),
        "p50_return": float(np.percentile(sim_returns, 50)),
        "p95_return": float(np.percentile(sim_returns, 95)),
        "p5_max_drawdown": float(np.percentile(sim_mdds, 95)), # 95th percentile worst drawdown
        "p50_max_drawdown": float(np.percentile(sim_mdds, 50))
    }

def calculate_equity_attribution(equity_returns: list, benchmark_returns: list) -> dict:
    if not equity_returns or not benchmark_returns or len(equity_returns) != len(benchmark_returns):
        return {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0}
    
    eq = np.array(equity_returns)
    bench = np.array(benchmark_returns)
    
    cov = np.cov(eq, bench)[0, 1]
    var_b = np.var(bench) + 1e-8
    beta = cov / var_b
    alpha = np.mean(eq) - beta * np.mean(bench)
    
    return {
        "alpha": float(alpha * 100.0),
        "beta": float(beta),
        "gamma": float(np.std(eq) / (np.std(bench) + 1e-8)),
        "delta": 0.0003  # Execution quality baseline (0.03%)
    }

def export_trade_journal(trades: list, filename: str = "trade_journal.json"):
    journal_entries = []
    for t in trades:
        entry = {
            "trade_id": t.get("trade_id"),
            "symbol": t.get("symbol"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "direction": t.get("direction"),
            "pnl_usd": t.get("pnl_usd"),
            "gross_pnl": t.get("gross_pnl", t.get("pnl_usd")),
            "fees_paid": t.get("fees_paid", 0.0),
            "mae": t.get("mae", 0.0),
            "mfe": t.get("mfe", 0.0),
            "duration_seconds": t.get("duration_seconds", 0.0),
            "reason": t.get("reason", "Exit")
        }
        journal_entries.append(entry)
        
    with open(filename, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "trades": journal_entries}, f, indent=2)
    print(f"[Trade Journal] Exported {len(journal_entries)} trade entries to {filename}")
