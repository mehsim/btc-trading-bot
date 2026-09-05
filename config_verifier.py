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
    # 1. Cross-module verification: Compare config.py vs on-disk model manifests
    try:
        import glob
        import json
        cfg_tf = getattr(config, "TIMEFRAME_CONFIG", {})
        cfg_adx = getattr(config, "REGIME_ADX_ENTER_BY_INTERVAL", {})
        manifest_paths = glob.glob("ensemble_*_manifest.json")
        for m_path in manifest_paths:
            # Check champion manifests (skip challenger / backup)
            if "_challenger" in m_path or "_backup" in m_path:
                continue
            try:
                with open(m_path, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                b_cfg = m_data.get("barrier_config")
                if b_cfg and isinstance(b_cfg, dict):
                    base_parts = os.path.basename(m_path).replace("_manifest.json", "").split("_")
                    if len(base_parts) >= 4:
                        iv_key = base_parts[-1]
                        # Only check trend models which govern directional entry barrier geometry
                        if "trend" in base_parts and iv_key in cfg_tf:
                            live_c = cfg_tf[iv_key]
                            # Finding #161 (Finding #93): Strictly verify all barrier parameters
                            for b_key in ["lookahead", "sl_mult", "tp_mult_trending", "tp_mult_ranging"]:
                                if b_key in b_cfg and b_key in live_c:
                                    diff = abs(float(b_cfg[b_key]) - float(live_c[b_key]))
                                    if diff > 0.05:
                                        raise ValueError(
                                            f"[Config Verifier M-2 Error] Barrier divergence in {m_path} for {b_key}: "
                                            f"manifest ({b_cfg[b_key]}) vs live config ({live_c[b_key]}) > 0.05 tolerance"
                                        )
                            # ADX verification: manifest recorded ADX threshold vs live config
                            if "regime_adx_enter" in b_cfg and iv_key in cfg_adx:
                                adx_diff = abs(float(b_cfg["regime_adx_enter"]) - float(cfg_adx[iv_key]))
                                if adx_diff > 0.05:
                                    raise ValueError(
                                        f"[Config Verifier M-2 Error] ADX enter threshold divergence in {m_path}: "
                                        f"manifest ({b_cfg['regime_adx_enter']}) vs live config ({cfg_adx[iv_key]})"
                                    )
            except ValueError:
                raise
            except Exception as ex_m:
                log_event("WARNING", f"[Config Verifier] Skipping manifest check for {m_path}: {ex_m}")
    except ValueError:
        raise
    except Exception as ex_gen:
        log_event("WARNING", f"[Config Verifier] Manifest comparison notice: {ex_gen}")

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

    # 5. Verify MIN_SL_PCT_CONFIG has valid floors for all supported intervals (Finding #53)
    min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
    for req_iv in ["15", "30", "60", "120", "240", "360"]:
        if req_iv not in min_sl_cfg:
            raise ValueError(f"[Config Verifier Finding #53] Missing timeframe floor for {req_iv}m in MIN_SL_PCT_CONFIG")
        val = float(min_sl_cfg[req_iv])
        if val <= 0.0 or val > 0.05:
            raise ValueError(f"[Config Verifier Finding #53] Invalid MIN_SL_PCT_CONFIG value for {req_iv}m: {val}")

    risk_limits.assert_risk_governance_invariants()
    log_event("INFO", "✅ [Config Verifier M-2] All cross-file train.py vs config.py constants strictly verified.")
    return True


def assert_manifest_live_parity(manifest_path: str, live_config: dict, tolerance: float = 0.05):
    """
    Finding #3: Directly validates that an on-disk manifest barrier_config strictly aligns
    with live timeframe configuration parameters within tolerance.
    """
    import json
    with open(manifest_path, "r", encoding="utf-8") as mf:
        m_data = json.load(mf)
    b_cfg = m_data.get("barrier_config", m_data)
    for k in ["lookahead", "sl_mult", "tp_mult_trending", "tp_mult_ranging", "regime_adx_enter"]:
        if k in b_cfg and k in live_config:
            diff = abs(float(b_cfg[k]) - float(live_config[k]))
            if diff > tolerance:
                raise ValueError(
                    f"[Config Verifier Error] Manifest {manifest_path} {k} ({b_cfg[k]}) "
                    f"diverges from live config ({live_config[k]}) by {diff:.4f} > {tolerance}"
                )
