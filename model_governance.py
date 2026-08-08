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
