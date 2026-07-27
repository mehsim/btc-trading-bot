import numpy as np
import threading
from typing import Dict, List, Tuple

def _normalize_tf(tf_str: str) -> str:
    s = str(tf_str or "").lower().strip()
    mapping = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "60": "60", "120": "120", "15": "15", "30": "30"}
    return mapping.get(s, s)

class TimeDecayCalibrator:
    def __init__(self):
        self.lock = threading.Lock()
        self.median_durations_hours: Dict[str, float] = {
            "15": 2.3,
            "30": 4.5,
            "60": 8.0,
            "120": 16.0
        }

    def update_durations_from_history(self, trade_history: List[Dict]):
        """Calculates median trade duration per timeframe from historical trade logs."""
        with self.lock:
            for tf in ["15", "30", "60", "120"]:
                tf_trades = [t for t in trade_history if _normalize_tf(t.get("timeframe") or t.get("interval")) == tf]
                durations = []
                for t in tf_trades:
                    entry = float(t.get("entry_time") or 0)
                    exit = float(t.get("exit_time") or 0)
                    if entry > 1e11:
                        entry /= 1000.0
                    if exit > 1e11:
                        exit /= 1000.0
                    if entry > 0 and exit > entry:
                        durations.append((exit - entry) / 3600.0)

                if len(durations) >= 10:
                    self.median_durations_hours[tf] = float(np.median(durations))

    def get_decay_start_and_rate(self, timeframe: str) -> Tuple[float, float]:
        """
        Rule 2:
        Decay start = median_duration * 0.5
        Decay rate = 0.05 * (4.0 / median_duration)
        Returns: (decay_start_hours, decay_rate_per_2h)
        """
        norm_tf = _normalize_tf(timeframe)
        with self.lock:
            med_dur = self.median_durations_hours.get(norm_tf, 4.0)
            start_hours = med_dur * 0.5
            decay_rate = 0.05 * (4.0 / max(1.0, med_dur))
            return start_hours, decay_rate


decay_calibrator = TimeDecayCalibrator()
