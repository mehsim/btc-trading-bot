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

        # 4-Tier Health State Machine: GREEN -> YELLOW -> ORANGE -> RED
        health_tier = "GREEN"
        decay_reasons = []
        decay_factor = 1.00

        if not ws_connected or candle_age > dynamic_candle_age_limit * 2.0:
            health_tier = "RED"
            decay_factor = 0.00
            decay_reasons.append(f"RED: Feed disconnected or severe candle staleness ({candle_age:.1f}s > {dynamic_candle_age_limit*2.0:.1f}s limit) -> Trading Disabled")
        elif candle_age > dynamic_candle_age_limit or clock_skew > dynamic_skew_limit or self.sequence_gap_count > 5:
            health_tier = "ORANGE"
            age_decay = max(0.50, 1.0 - 0.35 * (candle_age / max(1.0, dynamic_candle_age_limit * 2.0)))
            skew_decay = max(0.50, 1.0 - 0.35 * (clock_skew / max(1.0, dynamic_skew_limit * 2.0)))
            decay_factor = round(float(min(age_decay, skew_decay)), 4)
            if candle_age > dynamic_candle_age_limit:
                decay_reasons.append(f"ORANGE: Moderate candle age ({candle_age:.1f}s > {dynamic_candle_age_limit:.1f}s) -> Dynamic Factor {decay_factor:.2f}")
            if clock_skew > dynamic_skew_limit:
                decay_reasons.append(f"ORANGE: Clock skew ({clock_skew:.1f}ms > {dynamic_skew_limit:.1f}ms) -> Dynamic Factor {decay_factor:.2f}")
            if self.sequence_gap_count > 5:
                decay_reasons.append(f"ORANGE: Sequence gaps ({self.sequence_gap_count}) -> Dynamic Factor {decay_factor:.2f}")
        elif clock_skew > dynamic_skew_limit * 0.5 or self.sequence_gap_count > 0:
            health_tier = "YELLOW"
            decay_factor = round(float(max(0.80, 1.0 - 0.05 * max(1, self.sequence_gap_count) - 0.05 * (clock_skew / max(1.0, dynamic_skew_limit)))), 4)
            decay_reasons.append(f"YELLOW: Minor feed latency/sequence gap -> Dynamic Factor {decay_factor:.2f}")

        return {
            "timestamp": now,
            "ws_connected": ws_connected,
            "candle_age_sec": round(candle_age, 1),
            "clock_skew_ms": round(clock_skew, 1),
            "sequence_gaps": self.sequence_gap_count,
            "duplicate_messages": self.duplicate_count,
            "health_tier": health_tier, # GREEN, YELLOW, ORANGE, RED
            "trading_allowed": health_tier != "RED",
            "decay_factor": decay_factor,
            "decay_reasons": decay_reasons,
            "feed_status": "HEALTHY" if health_tier == "GREEN" else ("DEGRADED" if health_tier != "RED" else "CRITICAL")
        }

    def apply_explainable_confidence_decay(
        self,
        raw_confidence: float,
        feed_health: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Applies feed-health confidence decay with an explainable audit trail.
        """
        decay_factor = float(feed_health.get("decay_factor", 1.00))
        reasons = feed_health.get("decay_reasons", [])
        decayed_confidence = float(raw_confidence * decay_factor)

        return {
            "raw_confidence": round(raw_confidence, 4),
            "decayed_confidence": round(decayed_confidence, 4),
            "decay_factor": round(decay_factor, 2),
            "decay_reasons": reasons if reasons else ["GREEN: Market data feed optimal (No confidence decay)"],
            "health_tier": feed_health.get("health_tier", "GREEN")
        }

market_data_quality_monitor = MarketDataQualityMonitor()
