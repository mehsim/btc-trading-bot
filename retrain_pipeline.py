import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from mlops_engine import model_registry, calculate_psi, generate_model_card, STAGE_PRODUCTION, STAGE_STAGING, MLFLOW_AVAILABLE

if MLFLOW_AVAILABLE:
    import mlflow

RETRAIN_INTERVAL_DAYS = {
    "15": 3,   # Retrain 15m every 3 days
    "30": 5,   # Retrain 30m every 5 days
    "60": 7,   # Retrain 60m every 7 days
    "120": 7   # Retrain 120m every 7 days
}

class AutomatedRetrainTrigger:
    def __init__(self):
        self.last_retrain_time = time.time()
        self.last_data_count = 0

    def check_triggers_by_interval(self, interval: str, current_data_len: int, live_sharpe: float = 1.0, current_psi: float = 0.0) -> tuple:
        now = time.time()
        max_days = RETRAIN_INTERVAL_DAYS.get(str(interval), 7)
        if now - self.last_retrain_time >= max_days * 86400:
            return True, f"Schedule Trigger ({max_days} Days Elapsed for {interval}m)"
        return self.check_triggers(current_data_len, live_sharpe, current_psi)

    def check_triggers(self, current_data_len: int, live_sharpe: float = 1.0, current_psi: float = 0.0, is_cusum_drift: bool = False) -> tuple:
        """Rule 27: On-demand event-driven retraining triggered by PSI > 0.25 or CUSUM drift."""
        now = time.time()
        triggered = False
        reason = "No Retrain Triggered"
        
        # 1. Rule 27: Event-driven CUSUM Drift Trigger
        if is_cusum_drift:
            triggered, reason = True, "Event-Driven Trigger (CUSUM Concept Drift Fired)"

        # 2. Rule 27: Event-driven PSI Trigger (PSI > 0.25)
        elif current_psi > 0.25:
            triggered, reason = True, f"Event-Driven Trigger (PSI Distribution Shift {current_psi:.3f} > 0.25)"
            
        # 3. Performance Trigger: Sharpe ratio < 0.5
        elif live_sharpe < 0.5:
            triggered, reason = True, f"Performance Trigger (Live Sharpe {live_sharpe:.2f} < 0.50)"
            
        # 4. Fallback Schedule Trigger: 7 days elapsed (max 7-day fallback guard)
        elif now - self.last_retrain_time >= 7 * 86400:
            triggered, reason = True, "Fallback Schedule Trigger (7 Days Elapsed)"
            
        # 5. Data Accumulation Trigger: 30+ days of new data (approx 720 1h candles)
        elif self.last_data_count > 0 and (current_data_len - self.last_data_count) >= 720:
            triggered, reason = True, "Data Trigger (30+ Days of New Data Accumulated)"

        if MLFLOW_AVAILABLE and triggered:
            try:
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.set_tag("retrain_triggered", True)
                    mlflow.set_tag("retrain_reason", reason)
                    mlflow.log_metric("live_sharpe", live_sharpe)
                    mlflow.log_metric("data_psi", current_psi)
            except Exception:
                pass
            
        return triggered, reason

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
    
    passed = (win_rate >= 52.0 and pf >= 1.25)
    detail = f"PASSED: Holdout test passed (Win Rate: {win_rate:.1f}%, Profit Factor: {pf:.2f}, Trades: {n_trades})" if passed else f"FAILED: Holdout test metrics insufficient (Win Rate: {win_rate:.1f}%, Profit Factor: {pf:.2f})"

    if MLFLOW_AVAILABLE:
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.log_metric("holdout_win_rate", win_rate)
                mlflow.log_metric("holdout_profit_factor", pf)
                mlflow.set_tag("holdout_passed", passed)
        except Exception:
            pass

    return passed, detail

retrain_trigger = AutomatedRetrainTrigger()

def select_shap_features(model, X: pd.DataFrame, feature_names: list, top_n: int = 30) -> list:
    """Ranks and selects features using SHAP values (or tree feature importances fallback)."""
    if len(feature_names) <= top_n:
        return feature_names
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranked_indices = np.argsort(mean_abs_shap)[::-1]
        selected = [feature_names[i] for i in ranked_indices[:top_n]]
        print(f"[SHAP Pipeline] Successfully selected top {len(selected)} features using SHAP TreeExplainer.")

        if MLFLOW_AVAILABLE:
            try:
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.log_param("selected_shap_feature_count", len(selected))
            except Exception:
                pass

        return selected
    except Exception as e:
        print(f"[SHAP Pipeline Info] SHAP evaluation fallback to Tree Feature Importances: {e}")
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                ranked_indices = np.argsort(importances)[::-1]
                selected = [feature_names[i] for i in ranked_indices[:top_n]]
                print(f"[SHAP Pipeline Fallback] Selected top {len(selected)} features using Model Feature Importances.")
                return selected
        except Exception:
            pass
        return feature_names[:top_n]
