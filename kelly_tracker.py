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

    def log_trade(self, symbol: str, timeframe: str, pnl_usd: float, return_pct: float,
                  slippage_pct: float = 0.0005, timestamp: Optional[str] = None):
        """Logs a completed trade outcome with timestamp for calendar-time windowing."""
        import datetime as _dt
        with self.lock:
            self.history.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "pnl_usd": float(pnl_usd),
                "return_pct": float(return_pct),
                "slippage_pct": float(slippage_pct),
                "timestamp": timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat(),
            })
            try:
                with open(self.data_file, "w") as f:
                    json.dump(self.history[-1000:], f, indent=2)
            except Exception as e:
                print(f"[KellyTracker Error] Failed to save trade history: {e}")

    def get_realized_win_rate(self, timeframe: Optional[str] = None, min_trades: int = 10) -> Optional[float]:
        """Returns empirical win rate for timeframe if >= min_trades exist, else None."""
        with self.lock:
            if timeframe:
                norm_target = _normalize_tf(timeframe)
                filtered = [t for t in self.history if _normalize_tf(t.get("timeframe")) == norm_target]
            else:
                filtered = self.history
            if len(filtered) < min_trades:
                return None
            returns = [float(t.get("return_pct", 0.0) or (t.get("pnl_usd", 0.0) > 0)) for t in filtered]
            wins = [r for r in returns if r > 0]
            return float(len(wins) / len(returns))

    def compute_kelly_fraction(self, timeframe: Optional[str] = None, min_trades: int = 30, min_losses: int = 3, max_kelly_cap: float = 0.25, insufficient_as_none: bool = False) -> Optional[float]:
        """
        Computes dynamic Quarter-Kelly fraction per timeframe.
        H-1: Rolling window is calendar-time based (lookback_days) with a per-timeframe
             max-trade cap, so slow timeframes see regime-appropriate history.
        H-3: Minimum trade gate scales per timeframe so sizing isn't blocked for months
             on slow intervals; falls back to MIN_KELLY_SAMPLE_SIZE floor.
        Finding #163: When insufficient_as_none=True, returns None for insufficient sample size
             so risk_engine can distinguish 'no data' (fallback to prior) from a measured
             negative empirical edge (return 0.0 -> fail closed).
        """
        import datetime as _dt
        with self.lock:
            if timeframe:
                norm_target = _normalize_tf(timeframe)
                filtered = [t for t in self.history if _normalize_tf(t.get("timeframe")) == norm_target]
            else:
                filtered = self.history

            # H-1 & H-3: per-timeframe window and minimum gate from config
            from config import MIN_KELLY_SAMPLE_SIZE, KELLY_WINDOW_CONFIG, KELLY_WINDOW_DEFAULT
            tf_key = _normalize_tf(timeframe) if timeframe else None
            win_cfg = KELLY_WINDOW_CONFIG.get(tf_key, KELLY_WINDOW_DEFAULT) if tf_key else KELLY_WINDOW_DEFAULT
            max_trades = win_cfg["max_trades"]
            lookback_days = win_cfg["lookback_days"]

            # H-3: per-timeframe minimum; if min_trades explicitly passed, honor it
            if min_trades is not None and min_trades != 30:
                effective_min_trades = min_trades
            else:
                tf_min_map = {"15": 30, "30": 25, "60": 20, "120": 15, "240": 10, "360": 8}
                tf_min = tf_min_map.get(tf_key, 30) if tf_key else 30
                effective_min_trades = max(tf_min, MIN_KELLY_SAMPLE_SIZE)

            # H-1: filter to calendar lookback window first, then cap at max_trades
            cutoff_dt = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=lookback_days)
            cutoff_iso = cutoff_dt.isoformat()
            windowed = [t for t in filtered if t.get("timestamp", "2000-01-01") >= cutoff_iso]
            # If timestamp is missing on old records, fall back to tail of filtered list
            if not windowed:
                windowed = filtered
            windowed = windowed[-max_trades:]

            if len(windowed) < effective_min_trades:
                return None if insufficient_as_none else 0.0

            returns = [t["return_pct"] for t in windowed]
            slippages = [abs(t.get("slippage_pct", 0.0005)) for t in windowed]
            wins = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]

            # Finding #35: If sample window has 0 wins, that is a measured 0% win rate (negative edge) -> Fail-closed 0.0!
            if len(wins) < 1:
                return 0.0
            if len(losses) < min_losses:
                return None if insufficient_as_none else 0.0

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
            import config
            n_boot = getattr(config, "KELLY_BOOTSTRAP_SAMPLES", 1000)
            boot_r = []
            rng = np.random.RandomState(42)
            slippage_pct_scaled = avg_slippage * 100.0  # Scale fraction (e.g. 0.0005) to percentage points (0.05%)
            for _ in range(n_boot):
                b_wins = rng.choice(wins, size=len(wins), replace=True)
                b_loss = rng.choice(losses, size=len(losses), replace=True)
                b_win_m = max(0.0001, float(np.mean(b_wins)) - slippage_pct_scaled)
                b_loss_m = max(0.001, float(np.mean(b_loss)) + slippage_pct_scaled)
                boot_r.append(b_win_m / b_loss_m)
            r_ratio_lower = float(np.percentile(boot_r, 5)) if boot_r else 1.0

            # Realized execution payoff ratio using bootstrap 95% lower bound
            net_win = max(0.0001, avg_win - slippage_pct_scaled)
            net_loss = avg_loss + slippage_pct_scaled
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
