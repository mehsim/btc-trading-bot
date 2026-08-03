"""
schema_validator.py
-------------------
Phase 1 Extension: Feature Schema Validator.
Validates trade experience records before storage to prevent silent database corruption.
Checks missing fields, datatypes, value ranges, and version compatibility.
"""

import math
from typing import Dict, Any, Tuple, List

EXPECTED_SCHEMA = {
    "trade_id": (str, False),
    "symbol": (str, False),
    "confidence": (float, True),
    "adx": (float, True),
    "atr_pct": (float, True),
    "rsi": (float, True),
    "entry_price": (float, True),
    "exit_price": (float, True),
    "pnl_usd": (float, True),
    "realized_r": (float, True),
    "learning_engine_version": (str, True)
}


VALUE_RANGES = {
    "confidence": (0.0, 1.0),
    "adx": (0.0, 100.0),
    "rsi": (0.0, 100.0),
    "atr_pct": (0.0, 0.50),
    "atr_percentile": (0.0, 100.0),
    "slippage_bps": (-500.0, 500.0)
}

class SchemaValidator:
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # 1. Required fields check
        for field, (expected_type, optional) in EXPECTED_SCHEMA.items():
            if field not in record or record[field] is None:
                if not optional:
                    errors.append(f"Missing required field: {field}")
                continue
                
            val = record[field]
            if expected_type == float and isinstance(val, int):
                val = float(val)
                
            if not isinstance(val, expected_type):
                errors.append(f"Type mismatch for '{field}': expected {expected_type.__name__}, got {type(val).__name__}")
                continue
                
            if expected_type == float and (math.isnan(val) or math.isinf(val)):
                errors.append(f"Invalid NaN/Inf float for '{field}'")
                
        # 2. Value range check
        for field, (min_val, max_val) in VALUE_RANGES.items():
            if field in record and record[field] is not None:
                try:
                    v = float(record[field])
                    if v < min_val or v > max_val:
                        errors.append(f"Range breach for '{field}': {v} not in [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    pass

        is_valid = (len(errors) == 0)
        return is_valid, errors

schema_validator = SchemaValidator()
