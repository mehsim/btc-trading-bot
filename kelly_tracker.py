import os
import json
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional

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

    def compute_kelly_fraction(self, timeframe: Optional[str] = None, min_trades: int = 30, min_losses: int = 3, max_kelly_cap: float = 0.25) -> float:
        """
        Computes dynamic Quarter-Kelly fraction per timeframe:
        Kelly = (W * R - (1 - W)) / R
        Where W = Win Rate, R = Avg Win / Avg Loss ratio.
        Requires at least min_trades (30) and min_losses (3) to prevent zero-loss distortion.
        Capped at max_kelly_cap (default 0.25 / Quarter-Kelly).
        Returns 0.0 on zero or negative edge (proven losing strategy) or insufficient data.
        """
        with self.lock:
            if timeframe:
                norm_target = _normalize_tf(timeframe)
                filtered = [t for t in self.history if _normalize_tf(t.get("timeframe")) == norm_target]
            else:
                filtered = self.history

            from config import MIN_KELLY_SAMPLE_SIZE
            effective_min_trades = max(min_trades, MIN_KELLY_SAMPLE_SIZE)
            if len(filtered) < effective_min_trades:
                return 0.0

            returns = [t["return_pct"] for t in filtered[-100:]]
            slippages = [abs(t.get("slippage_pct", 0.0005)) for t in filtered[-100:]]
            wins = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]

            # Require minimum sample of both wins and losses to prevent distortion
            if len(wins) < 1 or len(losses) < min_losses:
                return 0.0

            n = len(returns)
            p_hat = len(wins) / float(n)
            
            # Wilson Score 95% confidence interval lower bound (z = 1.96)
            z = 1.96
            denom = 1.0 + (z**2 / n)
            center = p_hat + (z**2 / (2.0 * n))
            spread = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2 / (4.0 * n**2)))
            w_rate = max(0.0, (center - spread) / denom)

            avg_win = float(np.mean(wins))
            avg_slippage = float(np.mean(slippages)) if slippages else 0.0005
            raw_avg_loss = float(np.mean(losses))
            avg_loss = max(0.001, raw_avg_loss)

            # 95% Bootstrap lower bound on payoff ratio R for conservative estimation
            n_boot = 200
            boot_r = []
            rng = np.random.RandomState(42)
            for _ in range(n_boot):
                b_wins = rng.choice(wins, size=len(wins), replace=True)
                b_loss = rng.choice(losses, size=len(losses), replace=True)
                b_win_m = max(0.0001, float(np.mean(b_wins)) - avg_slippage)
                b_loss_m = max(0.001, float(np.mean(b_loss)) + avg_slippage)
                boot_r.append(b_win_m / b_loss_m)
            r_ratio_lower = float(np.percentile(boot_r, 5)) if boot_r else 1.0

            # Realized execution payoff ratio using bootstrap 95% lower bound
            net_win = max(0.0001, avg_win - avg_slippage)
            net_loss = avg_loss + avg_slippage
            point_r = net_win / max(1e-9, net_loss)
            r_ratio = max(0.1, min(point_r, r_ratio_lower))
            if r_ratio <= 0:
                return 0.0

            full_kelly = (w_rate * r_ratio - (1.0 - w_rate)) / max(1e-9, r_ratio)

            # Negative or zero edge -> Zero position allocation
            if full_kelly <= 0:
                return 0.0

            quarter_kelly = full_kelly * 0.25

            # Clamp between 0.0 (no negative edge allocation) and max_kelly_cap (25%)
            return float(np.clip(quarter_kelly, 0.0, max_kelly_cap))

global_kelly_tracker = KellyTracker()
