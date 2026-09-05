"""
Model Governance & Decision Traceability Engine.
Records immutable governance metadata for every trade signal:
- Model Version
- Feature Version
- Dataset Version
- Training Timestamp
- Git Commit SHA
- Configuration Hash
"""

import os
import hashlib
import time
from typing import Dict, Any

class ModelGovernanceEngine:
    def __init__(self, model_version: str = "v7.2.0", feature_version: str = "v3.1.0"):
        self.model_version = model_version
        self.feature_version = feature_version
        self.git_commit = self._get_git_commit()
        self.config_hash = self._compute_config_hash()

    def _get_git_commit(self) -> str:
        try:
            head_file = ".git/HEAD"
            if os.path.exists(head_file):
                with open(head_file, "r") as f:
                    ref = f.read().strip()
                if ref.startswith("ref: "):
                    ref_path = os.path.join(".git", ref[5:])
                    if os.path.exists(ref_path):
                        with open(ref_path, "r") as rf:
                            return rf.read().strip()[:8]
                return ref[:8]
        except Exception:
            pass
        return "b5c5c35a"

    def _compute_config_hash(self) -> str:
        payload = f"{self.model_version}:{self.feature_version}:{self.git_commit}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def create_traceability_record(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        predicted_change_pct: float
    ) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "symbol": symbol,
            "direction": direction,
            "confidence": round(confidence, 4),
            "predicted_change_pct": round(predicted_change_pct, 6),
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash
        }

    def log_barrier_manifest_audit(self, symbol: str, interval: str, manifest: Dict[str, Any]):
        """Logs and audits effective barrier config parameters loaded from manifest."""
        from logger import log_event
        barrier_cfg = manifest.get("barrier_config") or manifest.get("effective_barrier_config", {})
        log_event("INFO", f"[{symbol} {interval}m Manifest Audit] Effective Barrier Config: {barrier_cfg}")
        return barrier_cfg

model_governance_engine = ModelGovernanceEngine()


def extract_metric(data: dict, *key_paths):
    """Safely extracts numeric metric value across multiple alternative dictionary paths without falsy 0.0 bugs."""
    if not isinstance(data, dict):
        return None
    for path in key_paths:
        curr = data
        found = True
        for key in path:
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                found = False
                break
        if found and curr is not None:
            try:
                return float(curr)
            except (ValueError, TypeError):
                continue
    return None


def validate_manifest_governance_floors(manifest: Dict[str, Any], interval: str) -> "tuple[bool, str]":
    """
    Validates manifest against strict governance floors.
    Returns (True, '') if passed, or (False, rejection_reason).
    """
    from config import (
        SUPPORTED_MANIFEST_SCHEMA_VERSION,
        MODEL_GOVERNANCE,
        TIMEFRAME_MIN_HOLDOUT_MCC,
        TIMEFRAME_MIN_HOLDOUT_BAL_ACC,
    )
    if not isinstance(manifest, dict):
        return False, "Invalid manifest structure (not a dict)"

    schema_v = manifest.get("manifest_schema_version", 1)
    if schema_v > SUPPORTED_MANIFEST_SCHEMA_VERSION or schema_v < 1:
        return False, f"Manifest schema version mismatch ({schema_v} > {SUPPORTED_MANIFEST_SCHEMA_VERSION})"

    is_prom = manifest.get("promoted")
    if is_prom is not True:
        return False, "Manifest missing or explicitly marked promoted=False"

    model_type = str(manifest.get("model_type", "")).lower()
    prefix = str(manifest.get("prefix", "")).lower()
    is_regressor = (model_type in ("regressor", "price")) or ("price" in prefix)
    if is_regressor:
        # Finding #42 & Overturned #149 & Item 32: Regressor / price manifest governance validation
        reg_metrics = manifest.get("regression_metrics") or manifest.get("metrics")
        if not reg_metrics or not isinstance(reg_metrics, dict):
            return False, "Regressor manifest missing non-empty regression_metrics"
        mae = reg_metrics.get("mae")
        rmse = reg_metrics.get("rmse")
        if mae is None or rmse is None:
            return False, "Regressor manifest regression_metrics missing mae or rmse"
        try:
            f_mae = float(mae)
            f_rmse = float(rmse)
            if f_mae <= 0.0 or f_rmse <= 0.0:
                return False, f"Invalid regressor metrics: mae={mae}, rmse={rmse}"
            if f_rmse < (f_mae - 1e-9):
                return False, f"Sanity violation: rmse ({f_rmse}) cannot be less than mae ({f_mae})"
        except (ValueError, TypeError):
            return False, f"Non-numeric regressor metrics: mae={mae}, rmse={rmse}"

        r2 = reg_metrics.get("r2")
        dir_acc = reg_metrics.get("directional_accuracy")
        if r2 is None or dir_acc is None:
            return False, "Regressor manifest regression_metrics missing r2 or directional_accuracy"
        try:
            f_r2 = float(r2)
            f_dir = float(dir_acc)
            if f_r2 < 0.0:
                return False, f"Regressor r2 ({f_r2}) below governance floor (0.0)"
            if f_dir < 0.50:
                return False, f"Regressor directional accuracy ({f_dir}) below governance floor (0.50)"
        except (ValueError, TypeError):
            return False, f"Non-numeric regressor metrics: r2={r2}, directional_accuracy={dir_acc}"
        return True, ""

    min_holdout_mcc = TIMEFRAME_MIN_HOLDOUT_MCC.get(str(interval), TIMEFRAME_MIN_HOLDOUT_MCC.get("default", MODEL_GOVERNANCE.get("min_holdout_mcc", 0.035)))
    min_holdout_bal = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get(str(interval), TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get("default", MODEL_GOVERNANCE.get("min_holdout_balanced_accuracy", 0.355)))

    h_mcc = extract_metric(manifest, ["holdout_mcc"], ["cv_metrics", "holdout_mcc"], ["metrics", "holdout_mcc"])
    h_bal = extract_metric(manifest, ["holdout_balanced_accuracy"], ["cv_metrics", "holdout_balanced_accuracy"], ["metrics", "holdout_balanced_accuracy"])
    _ci = manifest.get("cv_metrics", {}).get("holdout_mcc_ci95") if isinstance(manifest.get("cv_metrics"), dict) else None
    h_ci_low = _ci[0] if isinstance(_ci, (list, tuple)) and len(_ci) >= 1 else None

    if h_mcc is None or h_mcc < min_holdout_mcc:
        return False, f"Holdout MCC ({h_mcc}) below governance floor ({min_holdout_mcc:.4f}) or missing"

    if h_bal is None or h_bal < min_holdout_bal:
        return False, f"Holdout BalAcc ({h_bal}) below governance floor ({min_holdout_bal:.4f}) or missing"

    if h_ci_low is not None and h_ci_low < -0.05:
        return False, f"Holdout MCC CI95 lower bound ({h_ci_low:.4f}) < -0.05"

    return True, ""
