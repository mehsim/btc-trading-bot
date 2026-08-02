"""
Abstention Replay Dataset & Counterfactual Logger.
Logs all skipped, delayed, or scaled signals into abstention_history.json for counterfactual replay analysis.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional

ABSTENTION_FILE = "abstention_history.json"

class AbstentionDatasetLogger:
    def __init__(self, filename: str = ABSTENTION_FILE):
        self.filename = filename
        self.history: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.history[-1000:], f, indent=2)
        except Exception as e:
            print(f"[AbstentionLogger Error] Failed to save history: {e}")

    def log_abstention_event(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        decision: str,
        score: float,
        reasons: List[str],
        metrics: Dict[str, Any]
    ):
        event = {
            "timestamp": time.time(),
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "decision": decision,
            "score": score,
            "reasons": reasons,
            "metrics": metrics,
            "actual_future_return": None # Filled later by counterfactual evaluator
        }
        self.history.append(event)
        self._save()

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"total_events": 0, "abstain_count": 0, "wait_count": 0, "execute_reduced_count": 0}

        total = len(self.history)
        abstains = sum(1 for e in self.history if e["decision"] == "ABSTAIN")
        waits = sum(1 for e in self.history if e["decision"] == "WAIT")
        reduced = sum(1 for e in self.history if e["decision"] == "EXECUTE_REDUCED")

        return {
            "total_events": total,
            "abstain_count": abstains,
            "wait_count": waits,
            "execute_reduced_count": reduced,
            "abstain_rate_pct": round((abstains / total) * 100.0, 1)
        }

abstention_dataset_logger = AbstentionDatasetLogger()
