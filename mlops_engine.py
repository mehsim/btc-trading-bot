import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Model Registry Stages
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

class ModelRegistry:
    def __init__(self, registry_file: str = "model_registry.json"):
        self.registry_file = registry_file
        self.models = self._load()

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
    return card_path

model_registry = ModelRegistry()
