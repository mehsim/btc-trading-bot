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
        self.current_tier: str = "GREEN"
        self.consecutive_healthy_count: int = 0
        self.health_history: List[float] = []
        self.transition_log: List[Dict[str, Any]] = []

    def _determine_raw_tier(
        self,
        ws_connected: bool,
        candle_age: float,
        clock_skew: float,
        dynamic_candle_age_limit: float,
        dynamic_skew_limit: float
    ) -> str:
        if not ws_connected or candle_age > dynamic_candle_age_limit * 2.0:
            return "RED"
        elif candle_age > dynamic_candle_age_limit or clock_skew > dynamic_skew_limit or self.sequence_gap_count > 5:
            return "ORANGE"
        elif clock_skew > dynamic_skew_limit * 0.5 or self.sequence_gap_count > 0:
            return "YELLOW"
        return "GREEN"

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

        dynamic_candle_age_limit = max(45.0, float(interval_sec * 1.5))
        dynamic_skew_limit = max(100.0, min(1000.0, float(ping_latency_ms * 3.0)))

        if seq_num is not None:
            if self.last_seq is not None:
                if seq_num == self.last_seq:
                    self.duplicate_count += 1
                elif seq_num > self.last_seq + 1:
                    self.sequence_gap_count += (seq_num - self.last_seq - 1)
            self.last_seq = seq_num

        raw_tier = self._determine_raw_tier(ws_connected, candle_age, clock_skew, dynamic_candle_age_limit, dynamic_skew_limit)
        
        # Hysteresis State Machine logic (Downgrades immediately; requires 3 consecutive healthy checks to upgrade)
        prev_tier = self.current_tier
        tier_hierarchy = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}

        if tier_hierarchy[raw_tier] > tier_hierarchy[self.current_tier]:
            # Immediate downgrade on failure
            self.current_tier = raw_tier
            self.consecutive_healthy_count = 0
        elif tier_hierarchy[raw_tier] < tier_hierarchy[self.current_tier]:
            # Delayed upgrade using stability buffer (Hysteresis)
            self.consecutive_healthy_count += 1
            dynamic_hysteresis_buffer = 3
            if self.consecutive_healthy_count >= dynamic_hysteresis_buffer:
                self.current_tier = raw_tier
                self.consecutive_healthy_count = 0
        else:
            self.consecutive_healthy_count = max(0, self.consecutive_healthy_count + 1)

        # Log state transition if tier changed
        if prev_tier != self.current_tier:
            trans_entry = {
                "timestamp": now,
                "previous_tier": prev_tier,
                "new_tier": self.current_tier,
                "reason": f"Feed transition from {prev_tier} to {self.current_tier} (Age={candle_age:.1f}s, Skew={clock_skew:.1f}ms)"
            }
            self.transition_log.append(trans_entry)

        # Feed Quality Trend calculation (Rolling health score)
        current_health_score = max(0.0, 100.0 - (candle_age / max(1.0, dynamic_candle_age_limit)) * 20.0 - (clock_skew / max(1.0, dynamic_skew_limit)) * 20.0)
        self.health_history.append(current_health_score)
        if len(self.health_history) > 10:
            self.health_history.pop(0)

        feed_trend = "STABLE"
        if len(self.health_history) >= 3:
            recent_diff = self.health_history[-1] - self.health_history[-3]
            if recent_diff > 2.0:
                feed_trend = "IMPROVING"
            elif recent_diff < -2.0:
                feed_trend = "DETERIORATING"

        # Calculate dynamic decay factor
        decay_factor = 1.00
        if self.current_tier == "RED":
            decay_factor = 0.00
        elif self.current_tier == "ORANGE":
            decay_factor = round(float(max(0.50, 1.0 - 0.35 * (candle_age / (dynamic_candle_age_limit * 2.0)))), 4)
        elif self.current_tier == "YELLOW":
            decay_factor = round(float(max(0.80, 1.0 - 0.05 * max(1, self.sequence_gap_count) - 0.05 * (clock_skew / dynamic_skew_limit))), 4)

        return {
            "timestamp": now,
            "ws_connected": ws_connected,
            "candle_age_sec": round(candle_age, 1),
            "clock_skew_ms": round(clock_skew, 1),
            "dynamic_candle_age_limit": round(dynamic_candle_age_limit, 1),
            "dynamic_skew_limit": round(dynamic_skew_limit, 1),
            "sequence_gaps": self.sequence_gap_count,
            "duplicate_messages": self.duplicate_count,
            "health_tier": self.current_tier,
            "raw_tier": raw_tier,
            "feed_trend": feed_trend, # IMPROVING, STABLE, DETERIORATING
            "trading_allowed": self.current_tier != "RED",
            "decay_factor": decay_factor,
            "transition_count": len(self.transition_log),
            "feed_status": "HEALTHY" if self.current_tier == "GREEN" else ("DEGRADED" if self.current_tier != "RED" else "CRITICAL")
        }

    def apply_explainable_confidence_decay(
        self,
        raw_confidence: float,
        feed_health: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Applies feed-health confidence decay with Additive Confidence Attribution Decomposition.
        Decomposes total deduction into age, skew, and sequence gap percentages.
        """
        decay_factor = float(feed_health.get("decay_factor", 1.00))
        health_tier = feed_health.get("health_tier", "GREEN")
        
        candle_age = float(feed_health.get("candle_age_sec", 0.0))
        clock_skew = float(feed_health.get("clock_skew_ms", 0.0))
        seq_gaps = int(feed_health.get("sequence_gaps", 0))
        dynamic_candle_age_limit = float(feed_health.get("dynamic_candle_age_limit", 45.0))
        dynamic_skew_limit = float(feed_health.get("dynamic_skew_limit", 150.0))

        # 100% Dynamic Additive Confidence Attribution Breakdown
        age_deduction_pct = round(max(0.0, min(25.0, (candle_age / max(1.0, dynamic_candle_age_limit)) * 10.0)), 2)
        skew_deduction_pct = round(max(0.0, min(25.0, (clock_skew / max(1.0, dynamic_skew_limit)) * 10.0)), 2)
        gap_deduction_pct = round(max(0.0, min(20.0, float(seq_gaps * 2.5))), 2)
        total_deduction_pct = age_deduction_pct + skew_deduction_pct + gap_deduction_pct

        final_confidence = max(0.0, raw_confidence * (1.0 - total_deduction_pct / 100.0) * decay_factor)

        attribution_breakdown = {
            "raw_confidence_pct": round(raw_confidence * 100.0, 1),
            "candle_age_deduction_pct": -age_deduction_pct,
            "clock_skew_deduction_pct": -skew_deduction_pct,
            "sequence_gap_deduction_pct": -gap_deduction_pct,
            "total_deduction_pct": -total_deduction_pct,
            "final_confidence_pct": round(final_confidence * 100.0, 1)
        }

        return {
            "raw_confidence": round(raw_confidence, 4),
            "decayed_confidence": round(final_confidence, 4),
            "decay_factor": round(decay_factor, 2),
            "attribution_breakdown": attribution_breakdown,
            "health_tier": health_tier,
            "feed_trend": feed_health.get("feed_trend", "STABLE")
        }

market_data_quality_monitor = MarketDataQualityMonitor()
