"""
data_quality_engine.py
-----------------------
Component 2: Rule-Based Data Quality Engine & System Health Separator.
Classifies operational severity levels (Critical, High, Medium, Low) and decouples
Data Quality issues from Infrastructure System Health failures.
"""

from typing import Dict, Any, List

class DataQualityEngine:
    """
    Evaluates market data integrity before inference and assigns rule-based severities.
    """
    def evaluate_data_quality(
        self,
        missing_candles_count: int = 0,
        corrupt_timestamps_count: int = 0,
        timestamp_gap_seconds: float = 0.0,
        stale_feed_seconds: float = 0.0,
        zero_price_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Returns: Dict with severity level, action recommendation, and detailed checks.
        """
        if corrupt_timestamps_count > 0 or zero_price_detected:
            severity = "CRITICAL"
            action = "STOP_TRADING_IMMEDIATELY"
            detail = "Corrupt timestamps or zero prices detected in market feed."
        elif missing_candles_count >= 5 or timestamp_gap_seconds >= 900.0 or stale_feed_seconds >= 300.0:
            severity = "HIGH"
            action = "DISABLE_NEW_ENTRIES"
            detail = f"Data gaps or stale market feed ({stale_feed_seconds}s stale)."
        elif missing_candles_count > 0 or timestamp_gap_seconds >= 300.0:
            severity = "MEDIUM"
            action = "ALERT_AND_CAUTION"
            detail = "Minor candle gaps detected. Continue trading with reduced risk."
        else:
            severity = "LOW"
            action = "LOG_AND_MONITOR"
            detail = "Market data integrity verified."

        return {
            "severity": severity,
            "action": action,
            "detail": detail
        }


class SystemHealthEngine:
    """
    Evaluates Infrastructure System Health independently from Market Data Quality.
    """
    def evaluate_system_health(
        self,
        exchange_connected: bool = True,
        db_connected: bool = True,
        api_latency_ms: float = 85.0,
        clock_drift_ms: float = 12.0
    ) -> Dict[str, Any]:
        if not exchange_connected or not db_connected:
            severity = "CRITICAL"
            action = "INFRASTRUCTURE_FAILOVER"
            detail = "Exchange or Database connection lost."
        elif api_latency_ms >= 1000.0 or clock_drift_ms >= 5000.0:
            severity = "HIGH"
            action = "PAUSE_EXECUTION"
            detail = f"Excessive latency ({api_latency_ms}ms) or clock drift ({clock_drift_ms}ms)."
        else:
            severity = "LOW"
            action = "HEALTHY"
            detail = "System infrastructure operational."

        return {
            "severity": severity,
            "action": action,
            "detail": detail
        }

data_quality_engine = DataQualityEngine()
system_health_engine = SystemHealthEngine()
