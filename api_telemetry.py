import time
import numpy as np
import threading
from typing import Dict, List, Tuple

class APITelemetryTracker:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.latencies_sec: List[float] = []
        self.call_logs: List[Dict] = []

    def record_call(self, endpoint: str, latency_sec: float, status_code: int = 200, error_type: str = None):
        """Records an API call execution time and latency."""
        with self.lock:
            self.latencies_sec.append(latency_sec)
            if len(self.latencies_sec) > self.capacity:
                self.latencies_sec.pop(0)

            self.call_logs.append({
                "timestamp": time.time(),
                "endpoint": endpoint,
                "latency_ms": round(latency_sec * 1000.0, 2),
                "status": status_code,
                "error": error_type
            })
            if len(self.call_logs) > 200:
                self.call_logs.pop(0)

    def get_median_latency_sec(self) -> float:
        with self.lock:
            if not self.latencies_sec:
                return 0.250
            return float(np.median(self.latencies_sec))

    def get_telemetry_params(self) -> Tuple[float, float, float]:
        """
        Returns dynamic backoff parameters:
        (base_delay_sec, max_delay_sec, idempotency_ttl_sec)
        """
        with self.lock:
            if not self.latencies_sec:
                return 0.5, 30.0, 60.0

            median_lat = float(np.median(self.latencies_sec))
            base_delay = max(0.10, median_lat * 2.0)
            max_delay = min(60.0, max(5.0, median_lat * 10.0))
            idempotency_ttl = max(30.0, median_lat * 5.0)

            return base_delay, max_delay, idempotency_ttl

global_api_telemetry = APITelemetryTracker()
