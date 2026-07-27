import numpy as np
import threading
from typing import Dict, List, Tuple

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
                tf_trades = [t for t in trade_history if str(t.get("timeframe", t.get("interval"))) == tf]
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
        with self.lock:
            med_dur = self.median_durations_hours.get(str(timeframe), 4.0)
            start_hours = med_dur * 0.5
            decay_rate = 0.05 * (4.0 / max(1.0, med_dur))
            return start_hours, decay_rate

decay_calibrator = TimeDecayCalibrator()
