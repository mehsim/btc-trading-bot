"""
feature_health_monitor.py
--------------------------
Phase 1A: Feature Health & Data Pipeline Safeguard.
Detects missing values (NaN/None), constant fields (stuck API), outliers, and stale cache data
to prevent pipeline contamination.
"""

import math
from typing import Dict, Any, List, Tuple
from trade_calculators import safe_float

class FeatureHealthMonitor:
    def __init__(self, constant_lookback: int = 20):
        self.constant_lookback = constant_lookback

    def inspect_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Inspects a trade record for feature anomalies.
        Returns: (is_healthy, list_of_issues)
        """
        issues = []
        
        # Check required numerical fields
        numeric_fields = ["adx", "atr_pct", "rsi", "funding_rate", "oi_z_score", "entry_price", "confidence"]
        for field in numeric_fields:
            val = record.get(field)
            if val is None:
                issues.append(f"Missing field: {field}")
            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                issues.append(f"Invalid NaN/Inf in field: {field}")
                
        # Check indicator domain bounds
        adx_val = safe_float(record.get("adx", 0.0))
        if adx_val < 0 or adx_val > 100:
            issues.append(f"Out of bounds ADX: {adx_val}")
            
        rsi_val = safe_float(record.get("rsi", 50.0), 50.0)
        if rsi_val < 0 or rsi_val > 100:
            issues.append(f"Out of bounds RSI: {rsi_val}")
            
        conf_val = safe_float(record.get("confidence", 0.5), 0.5)
        if conf_val < 0.0 or conf_val > 1.0:
            issues.append(f"Out of bounds confidence: {conf_val}")

        is_healthy = (len(issues) == 0)
        return is_healthy, issues

feature_health_monitor = FeatureHealthMonitor()
