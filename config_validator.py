"""
config_validator.py
-------------------
Configuration Loader & Schema Validator (MEDIUM-2 Remediation).
Loads config.yaml, validates schema bounds, and replaces hardcoded magic numbers.
"""

import os
import yaml
from typing import Dict, Any

DEFAULT_CONFIG = {
    "trading": {
        "symbol_universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "candle_check_window_mins": 15,
        "min_position_size_usd": 5.0,
        "max_position_size_usd": 25.0
    },
    "risk": {
        "max_safe_leverage_btc": 20.0,
        "max_safe_leverage_eth_sol": 15.0,
        "max_safe_leverage_altcoins": 5.0,
        "max_drawdown_halt_pct": 0.20,
        "target_annualized_volatility": 0.10,
        "cvar_confidence_level": 0.95
    },
    "funding_arbitrage": {
        "funding_arb_threshold": 0.001,
        "funding_arb_size_usd": 20.0
    },
    "execution": {
        "twap_slice_threshold_usd": 500.0,
        "twap_default_slices": 4,
        "twap_slice_interval_seconds": 3.0
    },
    "model": {
        "conformal_confidence_req": 0.61,
        "drift_ks_threshold": 0.05,
        "pattern_overlay_boost_pct": 0.04
    }
}

def load_and_validate_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        
        # Schema validation
        trading_cfg = cfg.get("trading", {})
        assert trading_cfg.get("min_position_size_usd", 5.0) > 0
        assert trading_cfg.get("max_position_size_usd", 25.0) >= trading_cfg.get("min_position_size_usd", 5.0)

        risk_cfg = cfg.get("risk", {})
        assert 0.05 <= risk_cfg.get("max_drawdown_halt_pct", 0.20) <= 0.50

        # Merge with defaults for any missing keys
        merged = DEFAULT_CONFIG.copy()
        for section, values in cfg.items():
            if isinstance(values, dict) and section in merged:
                merged[section].update(values)
            else:
                merged[section] = values
        return merged
    except Exception as e:
        print(f"[Config Loader Warning] Failed loading {config_path}, fallback to defaults: {e}")
        return DEFAULT_CONFIG

bot_config = load_and_validate_config()

def get_config_val(section: str, key: str, default: Any = None) -> Any:
    """Helper accessor to get a config value by section and key."""
    return bot_config.get(section, {}).get(key, default)
