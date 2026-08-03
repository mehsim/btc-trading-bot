"""
System Circuit Breaker & Health Monitor.
Monitors exchange latency, balance freshness, DB integrity, and model inference latency.
Auto-disables new trades if health thresholds are breached.
"""

import time
from typing import Dict, Any, Tuple

class CircuitBreaker:
    def __init__(self, max_latency_ms: float = 1000.0, max_balance_age_sec: float = 300.0):
        self.max_latency_ms = max_latency_ms
        self.max_balance_age_sec = max_balance_age_sec
        self.is_circuit_active: bool = False
        self.active_reason: str = ""

    def evaluate_system_health(
        self,
        exchange_latency_ms: float,
        last_balance_sync_ts: float,
        db_healthy: bool = True,
        inference_latency_ms: float = 50.0
    ) -> Tuple[bool, str]:
        """
        Returns (is_healthy, reason).
        """
        now = time.time()
        balance_age = now - last_balance_sync_ts if last_balance_sync_ts > 0 else 999.0

        if not db_healthy:
            self.is_circuit_active = True
            self.active_reason = "CRITICAL: Database connection or integrity failure"
            return False, self.active_reason

        if exchange_latency_ms > self.max_latency_ms:
            self.is_circuit_active = True
            self.active_reason = f"HIGH_LATENCY: Exchange REST latency ({exchange_latency_ms:.1f}ms > {self.max_latency_ms}ms)"
            return False, self.active_reason

        if balance_age > self.max_balance_age_sec:
            self.is_circuit_active = True
            self.active_reason = f"STALE_BALANCE: Balance cache age ({balance_age:.1f}s > {self.max_balance_age_sec}s)"
            return False, self.active_reason

        if inference_latency_ms > 500.0:
            self.is_circuit_active = True
            self.active_reason = f"SLOW_INFERENCE: ML inference latency ({inference_latency_ms:.1f}ms > 500ms)"
            return False, self.active_reason

        self.is_circuit_active = False
        self.active_reason = "HEALTHY"
        return True, "HEALTHY"

circuit_breaker = CircuitBreaker()
