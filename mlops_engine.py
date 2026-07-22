import os
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime

# MLflow Integration with Fallback
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

def init_mlflow(experiment_name: str = "BTC_Trading_Bot"):
    """Initializes MLflow tracking URI and experiment."""
    if not MLFLOW_AVAILABLE:
        return False
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"[MLflow] Initialized experiment '{experiment_name}' with tracking URI '{tracking_uri}'")
        return True
    except Exception as e:
        print(f"[MLflow Warning] Failed to initialize MLflow: {e}")
        return False

# Model Registry Stages
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

class ModelRegistry:
    def __init__(self, registry_file: str = "model_registry.json"):
        self.registry_file = registry_file
        self.models = self._load()
        init_mlflow()

    def _load(self) -> dict:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"Production": None, "Staging": None, "Archived": []}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(self, run_id: str, model_name: str, metrics: dict, stage: str = STAGE_STAGING):
        record = {
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics,
            "stage": stage,
            "timestamp": datetime.now().isoformat()
        }
        if stage == STAGE_PRODUCTION:
            if self.models["Production"]:
                self.models["Archived"].append(self.models["Production"])
            self.models["Production"] = record
        elif stage == STAGE_STAGING:
            self.models["Staging"] = record
        else:
            self.models["Archived"].append(record)
        self._save()
        print(f"[Model Registry] Registered {model_name} (Run ID: {run_id}) under stage '{stage}'")

        # Sync to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                active_run = mlflow.active_run()
                if active_run:
                    for metric_key, metric_val in metrics.items():
                        if isinstance(metric_val, (int, float)):
                            mlflow.log_metric(metric_key, float(metric_val))
                    mlflow.set_tag("model_stage", stage)
                    mlflow.set_tag("model_name", model_name)
                    print(f"[MLflow Registry Sync] Synced model '{model_name}' metrics and stage '{stage}' to run {active_run.info.run_id}")
            except Exception as ml_err:
                print(f"[MLflow Registry Warning] Could not sync model to MLflow: {ml_err}")

def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between baseline and target distributions."""
    if len(baseline) == 0 or len(target) == 0:
        return 0.0
    
    quantiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(baseline, quantiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5
    
    base_counts, _ = np.histogram(baseline, bins=buckets)
    target_counts, _ = np.histogram(target, bins=buckets)
    
    base_pct = base_counts / len(baseline)
    target_pct = target_counts / len(target)
    
    # Avoid zero division
    base_pct = np.where(base_pct == 0, 0.0001, base_pct)
    target_pct = np.where(target_pct == 0, 0.0001, target_pct)
    
    psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
    return float(psi_val)

def explain_prediction_top_features(candle_series: pd.Series, feature_names: list, top_k: int = 3) -> list:
    """Extracts top feature drivers for model explainability."""
    try:
        vals = candle_series[feature_names].abs()
        top_series = vals.sort_values(ascending=False).head(top_k)
        return [(feat, float(candle_series[feat])) for feat in top_series.index]
    except Exception:
        return [(feat, 0.0) for feat in feature_names[:top_k]]

def generate_model_card(model_name: str, run_id: str, metrics: dict, feature_names: list) -> str:
    card_dir = "model_cards"
    os.makedirs(card_dir, exist_ok=True)
    card_path = os.path.join(card_dir, f"{model_name}_{run_id}.json")
    
    card_data = {
        "model_name": model_name,
        "run_id": run_id,
        "date_created": datetime.now().isoformat(),
        "intended_use": "Autonomous BTC Algorithmic Trading",
        "performance_metrics": metrics,
        "features_used": feature_names,
        "author": "Deepmind Advanced Agentic Coding - Antigravity AI"
    }
    with open(card_path, "w") as f:
        json.dump(card_data, f, indent=2)
    print(f"[Model Card] Created model card at {card_path}")

    # Log model card artifact to MLflow if active
    if MLFLOW_AVAILABLE:
        try:
            if mlflow.active_run():
                mlflow.log_artifact(card_path, artifact_path="model_cards")
                print(f"[MLflow Artifact] Logged model card {card_path} to MLflow")
        except Exception as ml_err:
            print(f"[MLflow Artifact Warning] Could not log model card artifact: {ml_err}")

    return card_path

model_registry = ModelRegistry()

from collections import defaultdict

class IntervalPerformanceTracker:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "predictions": [],
            "accuracy": 0.0
        })

    def log_prediction(self, interval: str, prediction: str, confidence: float, actual_outcome: str):
        interval_key = str(interval)
        self.metrics[interval_key]["predictions"].append({
            "prediction": prediction,
            "confidence": confidence,
            "actual": actual_outcome,
            "timestamp": datetime.utcnow().isoformat()
        })

    def calculate_interval_accuracy(self, interval: str, window: int = 100) -> float:
        preds = self.metrics[str(interval)]["predictions"][-window:]
        if len(preds) < 10:
            return None
        correct = sum(1 for p in preds if p["prediction"] == p["actual"])
        acc = round(correct / len(preds), 4)
        self.metrics[str(interval)]["accuracy"] = acc
        return acc

    def should_retrain_interval(self, interval: str) -> bool:
        accuracy = self.calculate_interval_accuracy(interval)
        if accuracy is not None and accuracy < 0.50:
            return True
        return False

class TradeOutcomeAnalyzer:
    def __init__(self):
        self.trade_db = []
        self._lock = threading.Lock()

    def log_trade_outcome(self, interval: str, direction: str, confidence: float, ci_val: float, session: str, pnl_pct: float, exit_reason: str):
        with self._lock:
            self.trade_db.append({
                "timestamp": datetime.utcnow().isoformat(),
                "interval": str(interval),
                "direction": direction,
                "confidence": confidence,
                "choppiness_index": ci_val,
                "session": session,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason
            })
            if len(self.trade_db) > 200:
                self.trade_db = self.trade_db[-200:]

    def analyze_and_auto_tune(self, interval: str, n_trades: int = 50) -> dict:
        with self._lock:
            recent = [t for t in self.trade_db if t["interval"] == str(interval)][-n_trades:]
        if len(recent) < 10:
            return {}

        winners = [t for t in recent if t["pnl_pct"] > 0]
        losers = [t for t in recent if t["pnl_pct"] <= 0]
        
        adjustments = {}
        if winners and losers:
            avg_ci_win = float(np.mean([t["choppiness_index"] for t in winners]))
            avg_ci_loss = float(np.mean([t["choppiness_index"] for t in losers]))
            if avg_ci_loss > avg_ci_win + 5.0:
                adjustments["ci_adjustment"] = +3.0
                print(f"[Self-Learning Auto-Tune] Raised CI threshold for {interval}m by +3.0 (Loss CI: {avg_ci_loss:.1f} > Win CI: {avg_ci_win:.1f})")

        session_rates = {}
        for sess in ["asian", "london", "ny"]:
            sess_trades = [t for t in recent if t["session"] == sess]
            if len(sess_trades) >= 5:
                wr = sum(1 for t in sess_trades if t["pnl_pct"] > 0) / len(sess_trades)
                session_rates[sess] = wr

        if len(session_rates) >= 2:
            best_sess = max(session_rates, key=session_rates.get)
            worst_sess = min(session_rates, key=session_rates.get)
            if session_rates[best_sess] - session_rates[worst_sess] > 0.15:
                adjustments["session_bias"] = {best_sess: -0.03, worst_sess: +0.05}
                print(f"[Self-Learning Auto-Tune] Session Bias for {interval}m: Boosted {best_sess} (-3% required), Penalized {worst_sess} (+5% required)")

        return adjustments

trade_outcome_analyzer = TradeOutcomeAnalyzer()
global_interval_tracker = IntervalPerformanceTracker()
