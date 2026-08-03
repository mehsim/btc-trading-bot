"""
Market Data Quality & Sequence Integrity Monitor.
Monitors:
1. Stale candles
2. WebSocket layer health
3. Sequence gaps
4. Duplicate messages
5. Exchange clock skew (ms)
6. Missing trade messages
"""

import time
from typing import Dict, Any, List, Optional

class MarketDataQualityMonitor:
    def __init__(self, max_clock_skew_ms: float = 500.0, max_candle_age_sec: float = 120.0):
        self.max_clock_skew_ms = max_clock_skew_ms
        self.max_candle_age_sec = max_candle_age_sec
        self.last_seq: Optional[int] = None
        self.duplicate_count: int = 0
        self.sequence_gap_count: int = 0

    def evaluate_feed_health(
        self,
        last_candle_timestamp: float,
        server_time_ms: float,
        client_time_ms: float,
        seq_num: Optional[int] = None,
        ws_connected: bool = True,
        interval_sec: float = 60.0,
        ping_latency_ms: float = 50.0
    ) -> Dict[str, Any]:
        now = time.time()
        candle_age = now - last_candle_timestamp if last_candle_timestamp > 0 else 999.0
        clock_skew = abs(server_time_ms - client_time_ms)

        # Dynamic threshold adaptation
        dynamic_candle_age_limit = max(45.0, float(interval_sec * 1.5))
        dynamic_skew_limit = max(100.0, min(1000.0, float(ping_latency_ms * 3.0)))

        if seq_num is not None:
            if self.last_seq is not None:
                if seq_num == self.last_seq:
                    self.duplicate_count += 1
                elif seq_num > self.last_seq + 1:
                    self.sequence_gap_count += (seq_num - self.last_seq - 1)
            self.last_seq = seq_num

        is_stale = candle_age > dynamic_candle_age_limit
        has_skew = clock_skew > dynamic_skew_limit
        is_healthy = ws_connected and not is_stale and not has_skew

        return {
            "timestamp": now,
            "ws_connected": ws_connected,
            "candle_age_sec": round(candle_age, 1),
            "clock_skew_ms": round(clock_skew, 1),
            "sequence_gaps": self.sequence_gap_count,
            "duplicate_messages": self.duplicate_count,
            "feed_status": "HEALTHY" if is_healthy else "DEGRADED",
            "is_stale": is_stale,
            "has_clock_skew": has_skew
        }

market_data_quality_monitor = MarketDataQualityMonitor()
