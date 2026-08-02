"""
Context Feature Vector Engine with Feature Reliability & Freshness Decay.
Encapsulates market context into a continuous vector representation with reliability weighting and staleness decay.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class FeatureMetadata:
    def __init__(self, name: str, value: float, reliability_score: float = 1.0, timestamp: Optional[float] = None, source: str = "internal", max_staleness_sec: float = 300.0):
        self.name = name
        self.value = float(value)
        self.reliability_score = float(max(0.0, min(1.0, reliability_score)))
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.source = source
        self.max_staleness_sec = max_staleness_sec

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    @property
    def freshness_factor(self) -> float:
        """Exponential decay based on age relative to max staleness."""
        if self.max_staleness_sec <= 0:
            return 1.0
        if self.age_seconds >= self.max_staleness_sec:
            return 0.0
        return float(np.exp(-2.0 * (self.age_seconds / self.max_staleness_sec)))

    @property
    def effective_weight(self) -> float:
        """Combined reliability score and freshness factor."""
        return float(self.reliability_score * self.freshness_factor)

    def get_effective_value(self, default_value: float = 0.0) -> float:
        """Returns value weighted towards default if reliability/freshness is low."""
        w = self.effective_weight
        return float(w * self.value + (1.0 - w) * default_value)


class ContextFeatureVector:
    def __init__(self):
        self.feature_reliability_db: Dict[str, float] = {
            "confidence": 0.95,
            "atr_norm": 0.98,
            "adx": 0.95,
            "session": 0.99,
            "regime": 0.92,
            "portfolio_heat": 0.99,
            "mhi": 0.90,
            "liquidity": 0.85,
            "spread": 0.90,
            "funding_rate": 0.80,
            "open_interest_delta": 0.75
        }

    def update_feature_reliability(self, feature_name: str, new_reliability: float):
        self.feature_reliability_db[feature_name] = max(0.10, min(1.0, float(new_reliability)))

    def build_context_vector(self, context_dict: Dict[str, Any]) -> Dict[str, FeatureMetadata]:
        """
        Transforms a dictionary of raw context variables into a dictionary of FeatureMetadata objects.
        """
        now = time.time()
        features = {}

        # 1. Confidence
        conf_val = float(context_dict.get("calibrated_confidence", context_dict.get("confidence", 0.50)))
        conf_ts = float(context_dict.get("confidence_timestamp", now))
        features["confidence"] = FeatureMetadata(
            "confidence", conf_val, 
            reliability_score=self.feature_reliability_db.get("confidence", 0.95),
            timestamp=conf_ts, source="ml_calibrator", max_staleness_sec=120.0
        )

        # 2. ATR Norm
        atr_val = float(context_dict.get("atr_norm", 0.01))
        features["atr_norm"] = FeatureMetadata(
            "atr_norm", atr_val,
            reliability_score=self.feature_reliability_db.get("atr_norm", 0.98),
            timestamp=now, source="indicators", max_staleness_sec=300.0
        )

        # 3. ADX
        adx_val = float(context_dict.get("adx", 20.0))
        features["adx"] = FeatureMetadata(
            "adx", adx_val,
            reliability_score=self.feature_reliability_db.get("adx", 0.95),
            timestamp=now, source="indicators", max_staleness_sec=300.0
        )

        # 4. Session Index (Asian=0, London=1, NY=2, Overlap=3)
        session_str = str(context_dict.get("session", "asian")).lower()
        sess_idx = 0.0
        if "london" in session_str:
            sess_idx = 1.0
        elif "ny" in session_str or "new_york" in session_str:
            sess_idx = 2.0
        elif "overlap" in session_str:
            sess_idx = 3.0
        features["session"] = FeatureMetadata(
            "session", sess_idx,
            reliability_score=self.feature_reliability_db.get("session", 0.99),
            timestamp=now, source="time_engine", max_staleness_sec=3600.0
        )

        # 5. Regime Index (Trending=1.0, Ranging=0.0)
        regime_str = str(context_dict.get("regime", "Ranging"))
        reg_idx = 1.0 if "Trending" in regime_str else 0.0
        features["regime"] = FeatureMetadata(
            "regime", reg_idx,
            reliability_score=self.feature_reliability_db.get("regime", 0.92),
            timestamp=now, source="gmm_classifier", max_staleness_sec=600.0
        )

        # 6. Portfolio Heat
        heat_val = float(context_dict.get("portfolio_heat", 0.0))
        features["portfolio_heat"] = FeatureMetadata(
            "portfolio_heat", heat_val,
            reliability_score=self.feature_reliability_db.get("portfolio_heat", 0.99),
            timestamp=now, source="risk_engine", max_staleness_sec=60.0
        )

        # 7. Model Health Index (MHI) (0.0 to 100.0)
        mhi_val = float(context_dict.get("mhi", 90.0))
        features["mhi"] = FeatureMetadata(
            "mhi", mhi_val,
            reliability_score=self.feature_reliability_db.get("mhi", 0.90),
            timestamp=now, source="health_dashboard", max_staleness_sec=600.0
        )

        # 8. Orderbook Top Depth Liquidity (USD)
        liq_val = float(context_dict.get("top_book_depth_usd", 50000.0))
        liq_ts = float(context_dict.get("liquidity_timestamp", now))
        features["liquidity"] = FeatureMetadata(
            "liquidity", liq_val,
            reliability_score=self.feature_reliability_db.get("liquidity", 0.85),
            timestamp=liq_ts, source="bybit_l2", max_staleness_sec=30.0
        )

        # 9. Spread (fraction)
        spread_val = float(context_dict.get("spread_pct", 0.0001))
        features["spread"] = FeatureMetadata(
            "spread", spread_val,
            reliability_score=self.feature_reliability_db.get("spread", 0.90),
            timestamp=now, source="bybit_l2", max_staleness_sec=30.0
        )

        # 10. Open Interest Delta
        oi_val = float(context_dict.get("oi_delta_pct", 0.0))
        features["open_interest_delta"] = FeatureMetadata(
            "open_interest_delta", oi_val,
            reliability_score=self.feature_reliability_db.get("open_interest_delta", 0.75),
            timestamp=now, source="bybit_rest", max_staleness_sec=300.0
        )

        return features

    def evaluate_context_multipliers(self, features: Dict[str, FeatureMetadata]) -> Dict[str, float]:
        """
        Evaluates continuous, non-additive context policy multipliers based on feature vector values.
        """
        conf_eff = features["confidence"].get_effective_value(0.50)
        atr_eff = features["atr_norm"].get_effective_value(0.01)
        adx_eff = features["adx"].get_effective_value(20.0)
        sess_eff = features["session"].get_effective_value(0.0)
        reg_eff = features["regime"].get_effective_value(0.0)

        # Base target expansion
        target_expansion = 1.0
        stop_expansion = 1.0
        size_multiplier = 1.0

        # Continuous non-additive interaction terms
        if reg_eff > 0.5 and adx_eff > 25.0:
            target_expansion += 0.15 * (adx_eff / 40.0)
        
        # High confidence in low volatility session
        if conf_eff > 0.70 and atr_eff < 0.015 and sess_eff in (1.0, 3.0):
            target_expansion *= 1.10
            size_multiplier *= 1.08
        elif conf_eff > 0.75 and atr_eff > 0.025:
            # High confidence + high volatility: slightly expand stop to avoid noise, expand TP target
            stop_expansion *= 1.08
            target_expansion *= 1.12
            size_multiplier *= 0.90  # Reduce size in high vol

        return {
            "target_expansion": float(np.clip(target_expansion, 0.80, 1.60)),
            "stop_expansion": float(np.clip(stop_expansion, 0.90, 1.40)),
            "size_multiplier": float(np.clip(size_multiplier, 0.50, 1.30))
        }

context_feature_vector_engine = ContextFeatureVector()
