import os
import json
import numpy as np
import threading
from typing import Dict, List, Tuple

KELLY_DATA_FILE = "kelly_trade_history.json"

def _normalize_tf(tf_str: str) -> str:
    s = str(tf_str or "").lower().strip()
    mapping = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "60": "60", "120": "120", "15": "15", "30": "30"}
    return mapping.get(s, s)

class KellyTracker:
    def __init__(self, data_file: str = KELLY_DATA_FILE):
        self.data_file = data_file
        self.lock = threading.Lock()
        self.history: List[Dict] = []
        self._load_history()

    def _load_history(self):
        with self.lock:
            if os.path.exists(self.data_file):
                try:
                    with open(self.data_file, "r") as f:
                        self.history = json.load(f)
                except Exception as e:
                    print(f"[KellyTracker Warning] Failed to load trade history: {e}")
                    self.history = []

    def save_history(self):
        with self.lock:
            try:
                with open(self.data_file, "w") as f:
                    json.dump(self.history[-1000:], f, indent=2)
            except Exception as e:
                print(f"[KellyTracker Error] Failed to save trade history: {e}")

    def log_trade(self, symbol: str, timeframe: str, pnl_usd: float, return_pct: float):
        """Logs a completed trade outcome."""
        with self.lock:
            self.history.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "pnl_usd": float(pnl_usd),
                "return_pct": float(return_pct)
            })
            try:
                with open(self.data_file, "w") as f:
                    json.dump(self.history[-1000:], f, indent=2)
            except Exception as e:
                print(f"[KellyTracker Error] Failed to save trade history: {e}")

    def compute_kelly_fraction(self, timeframe: str = None, min_trades: int = 10, max_kelly_cap: float = 0.25) -> float:
        """
        Computes dynamic Quarter-Kelly fraction per timeframe:
        Kelly = (W * R - (1 - W)) / R
        Where W = Win Rate, R = Avg Win / Avg Loss ratio.
        Capped at max_kelly_cap (default 0.25 / Quarter-Kelly).
        """
        with self.lock:
            if timeframe:
                norm_target = _normalize_tf(timeframe)
                filtered = [t for t in self.history if _normalize_tf(t.get("timeframe")) == norm_target]
            else:
                filtered = self.history

            if len(filtered) < min_trades:
                # Default safety fallback if not enough trade history accumulated
                return 0.10


            returns = [t["return_pct"] for t in filtered[-100:]]
            wins = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]

            if not wins or not losses:
                return 0.10

            w_rate = len(wins) / len(returns)
            avg_win = float(np.mean(wins))
            avg_loss = float(np.mean(losses)) if np.mean(losses) > 0 else 1.0

            r_ratio = avg_win / max(1e-9, avg_loss)
            if r_ratio <= 0:
                return 0.05

            full_kelly = (w_rate * r_ratio - (1.0 - w_rate)) / max(1e-9, r_ratio)
            quarter_kelly = full_kelly * 0.25

            # Clamp between 2% minimum and max_kelly_cap (25%)
            return float(np.clip(quarter_kelly, 0.02, max_kelly_cap))

global_kelly_tracker = KellyTracker()
