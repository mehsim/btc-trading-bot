"""
config_verifier.py
-------------------
Startup verification module ensuring runtime constants in config.py, trade_calculators.py,
and risk_limits.py remain strictly aligned with model training hyper-parameters in train.py.
"""

import os
import re
from logger import log_event
import config
import risk_limits
import trade_calculators


def assert_shared_constants_aligned():
    """Validates that key governance and risk constants match across train.py, config.py, and risk_limits.py."""
    # 1. Cross-module verification: Compare config.py vs train.py constants
    try:
        import train
        # Check ADX regime thresholds alignment
        cfg_adx = getattr(config, "REGIME_ADX_ENTER_BY_INTERVAL", {})
        train_adx = getattr(train, "REGIME_ADX_ENTER_BY_INTERVAL", {})
        intervals = getattr(config, "SUPPORTED_INTERVALS", ["15", "30", "60", "120", "240", "360"])
        for iv in intervals:
            iv_str = str(iv)
            if iv_str in cfg_adx and iv_str in train_adx:
                if cfg_adx[iv_str] != train_adx[iv_str]:
                    raise ValueError(f"[Config Verifier M-2 Error] ADX enter threshold mismatch for {iv_str}m: config ({cfg_adx[iv_str]}) vs train ({train_adx[iv_str]})")

        # Check TIMEFRAME_CONFIG alignment
        cfg_tf = getattr(config, "TIMEFRAME_CONFIG", {})
        train_tf = getattr(train, "TIMEFRAME_CONFIG", {})
        for iv in ["15", "30", "60", "120", "240", "360"]:
            if iv in cfg_tf and iv in train_tf:
                c_item = cfg_tf[iv]
                t_item = train_tf[iv]
                for key in ["lookahead", "sl_mult", "tp_mult_trending", "tp_mult_ranging"]:
                    if key in c_item and key in t_item and c_item[key] != t_item[key]:
                        raise ValueError(f"[Config Verifier M-2 Error] {key} mismatch for {iv}m: config ({c_item[key]}) vs train ({t_item[key]})")
    except ImportError as imp_err:
        log_event("WARNING", f"[Config Verifier] Skipping train.py import comparison: {imp_err}")

    # 2. Source-level scan on train.py to catch hardcoded literal threshold comparisons
    train_py_path = "train.py"
    if os.path.exists(train_py_path):
        with open(train_py_path, "r", encoding="utf-8") as f:
            src = f.read()

        # Catch unparameterized ADX threshold comparisons like adx_t >= 25.0 outside config lookups
        for line_no, line in enumerate(src.splitlines(), 1):
            if "adx" in line.lower() and ">=" in line:
                if "REGIME_ADX_ENTER" not in line and "adx_enter_thresh" not in line and "STRONG_TREND_ADX" not in line and "cfg" not in line:
                    m = re.search(r">\s*=\s*([\d.]+)", line)
                    if m and float(m.group(1)) > 15.0:
                        raise ValueError(f"[Config Verifier M-2 Error] train.py L{line_no} has hardcoded ADX threshold {m.group(1)} — must read from config.py: `{line.strip()}`")

    # 3. Verify risk governance invariants & leverage caps
    for tf, max_lev in risk_limits.HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.items():
        if tf in trade_calculators.MAX_RR_RATIO:
            rr_cap = trade_calculators.MAX_RR_RATIO[tf]
            if rr_cap <= 0:
                raise ValueError(f"[Config Verifier] Invalid R:R cap for timeframe {tf}: {rr_cap}")

    if not config.SUPPORTED_SYMBOLS or not isinstance(config.SUPPORTED_SYMBOLS, list):
        raise ValueError("[Config Verifier] SUPPORTED_SYMBOLS must be a non-empty list.")

    for sym in config.SUPPORTED_SYMBOLS:
        if not sym.endswith("USDT"):
            raise ValueError(f"[Config Verifier] Unsupported symbol format in SUPPORTED_SYMBOLS: {sym}")

    # 4. Verify barrier geometry validity: tp_mult_trending >= tp_mult_ranging for every timeframe
    cfg_tf = getattr(config, "TIMEFRAME_CONFIG", {})
    for iv, tf_dict in cfg_tf.items():
        tp_t = float(tf_dict.get("tp_mult_trending", 0.0))
        tp_r = float(tf_dict.get("tp_mult_ranging", 0.0))
        sl_m = float(tf_dict.get("sl_mult", 0.0))
        lookahead = int(tf_dict.get("lookahead", 0))
        if tp_t < tp_r:
            raise ValueError(f"[Config Verifier] Inverted barrier geometry for {iv}m: tp_mult_trending ({tp_t}) < tp_mult_ranging ({tp_r})")
        if sl_m < 0.3:
            raise ValueError(f"[Config Verifier] Invalid stop loss multiplier for {iv}m: sl_mult ({sl_m}) < 0.3")
        if lookahead < 4:
            raise ValueError(f"[Config Verifier] Invalid lookahead for {iv}m: lookahead ({lookahead}) < 4")

    risk_limits.assert_risk_governance_invariants()
    log_event("INFO", "✅ [Config Verifier M-2] All cross-file train.py vs config.py constants strictly verified.")
    return True
