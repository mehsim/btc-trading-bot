import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from mlops_engine import model_registry, calculate_psi, generate_model_card, STAGE_PRODUCTION, STAGE_STAGING

class AutomatedRetrainTrigger:
    def __init__(self):
        self.last_retrain_time = time.time()
        self.last_data_count = 0

    def check_triggers(self, current_data_len: int, live_sharpe: float = 1.0, current_psi: float = 0.0) -> tuple:
        now = time.time()
        
        # 1. Schedule Trigger: 7 days elapsed
        if now - self.last_retrain_time >= 7 * 86400:
            return True, "Schedule Trigger (7 Days Elapsed)"
            
        # 2. Performance Trigger: Sharpe ratio < 0.5
        if live_sharpe < 0.5:
            return True, f"Performance Trigger (Live Sharpe {live_sharpe:.2f} < 0.50)"
            
        # 3. Drift Trigger: PSI > 0.20
        if current_psi > 0.20:
            return True, f"Drift Trigger (PSI {current_psi:.3f} > 0.20)"
            
        # 4. Data Accumulation Trigger: 30+ days of new data (approx 720 1h candles)
        if self.last_data_count > 0 and (current_data_len - self.last_data_count) >= 720:
            return True, "Data Trigger (30+ Days of New Data Accumulated)"
            
        return False, "No Retrain Triggered"

def evaluate_holdout_test_protocol(trades: list, min_trades: int = 100) -> tuple:
    """Holds out unseen test dataset and checks statistical significance (min 100 trades)."""
    n_trades = len(trades)
    if n_trades < min_trades:
        return False, f"FAILED: Only {n_trades} trades in test set (min {min_trades} required for statistical significance)"
        
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    win_rate = (len(wins) / n_trades) * 100.0
    
    gross_profits = sum(t.get("pnl_usd", 0) for t in wins)
    gross_losses = abs(sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) < 0))
    pf = gross_profits / (gross_losses + 1e-8)
    
    if win_rate >= 52.0 and pf >= 1.25:
        return True, f"PASSED: Holdout test passed (Win Rate: {win_rate:.1f}%, Profit Factor: {pf:.2f}, Trades: {n_trades})"
    else:
        return False, f"FAILED: Holdout test metrics insufficient (Win Rate: {win_rate:.1f}%, Profit Factor: {pf:.2f})"

retrain_trigger = AutomatedRetrainTrigger()
