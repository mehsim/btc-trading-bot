"""
signal_evaluator.py
-------------------
Background worker evaluating real-time market data across timeframes, classifying regimes (Trending vs Ranging),
calculating ensemble predictions, and updating state_manager for the live dashboard and trade execution.
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any

from logger import log_event
from data import get_history, merge_derivatives_sentiment_features
from core import add_features, calibrate_confidence, features
from ensemble import load_ensemble_classifier, load_ensemble_regressor, _slice_model_input
from config import (
    ENABLE_REGIME_HYSTERESIS, STRONG_TREND_ADX_ENTER, STRONG_TREND_ADX_EXIT,
    REGIME_ADX_ENTER_BY_INTERVAL, REGIME_ADX_EXIT_BY_INTERVAL
)

from collections import OrderedDict


TF_MAP = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h"}


def get_hierarchical_macro_bias(bot_state: dict, symbol: str) -> dict:
    """
    Resolves the dominant 4-Hour (240m) macro trend direction, strength, and regime posture.
    """
    from collections.abc import Mapping
    if not isinstance(bot_state, (dict, Mapping)) and not hasattr(bot_state, "get"):
        return {"direction": "Neutral", "confidence": 0.50, "adx": 25.0, "is_trending": False, "setup_type": "Neutral"}
    
    macro_pred = (
        bot_state.get(f"latest_prediction_{symbol}_4h") or
        bot_state.get(f"latest_prediction_{symbol}_240") or
        bot_state.get(f"latest_prediction_240m_{symbol}") or
        bot_state.get("latest_prediction_4h") or
        bot_state.get("latest_prediction_240m")
    )
    macro_dir = "Neutral"
    macro_conf = 0.50
    if isinstance(macro_pred, dict):
        macro_dir = str(macro_pred.get("direction", "Neutral"))
        macro_conf = float(macro_pred.get("calibrated_confidence") or macro_pred.get("confidence") or 0.50)
        
    macro_adx = float(bot_state.get(f"adx_{symbol}_4h") or bot_state.get("adx_4h") or 25.0)
    is_trending = macro_adx >= 25.0
    
    return {
        "direction": macro_dir,
        "confidence": macro_conf,
        "adx": macro_adx,
        "is_trending": is_trending,
        "setup_type": f"4H_{macro_dir}"
    }


class SignalEvaluator:
    def __init__(self, bot_state):
        self.bot_state = bot_state
        self.state_lock = threading.Lock()
        self.models_by_interval = OrderedDict()
        self.max_cache_size = 12

    def get_models(self, interval: str, is_trending: bool):
        regime_key = "trending" if is_trending else "ranging"
        cache_key = f"{interval}_{regime_key}"
        with self.state_lock:
            if cache_key in self.models_by_interval:
                self.models_by_interval.move_to_end(cache_key)
                return self.models_by_interval[cache_key]
            if interval in self.models_by_interval and isinstance(self.models_by_interval[interval], dict) and regime_key in self.models_by_interval[interval]:
                return self.models_by_interval[interval][regime_key]

        import json, os, gc

        manifest_path = f"ensemble_{regime_key}_trend_{interval}_manifest.json"
        feat_list = None
        model_ver_str = None
        git_sha_str = None

        manifest_mcc = None
        manifest_mcc_min = None
        manifest_bal_acc = None

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as mf:
                    m_data = json.load(mf)
                    m_ver = m_data.get("model_version", "v7.2.0")
                    git_sha_str = m_data.get("git_sha", "unknown")
                    model_ver_str = f"btc_{interval}m_{regime_key}_clf:{m_ver}"

                    cv_m = m_data.get("cv_metrics", {})
                    m_m = m_data.get("metrics", {})
                    manifest_mcc = cv_m.get("mcc") if isinstance(cv_m.get("mcc"), (int, float)) else (m_m.get("mcc") if isinstance(m_m.get("mcc"), (int, float)) else (cv_m.get("mcc", {}).get("mean") if isinstance(cv_m.get("mcc"), dict) else None))
                    manifest_mcc_min = cv_m.get("mcc", {}).get("min") if isinstance(cv_m.get("mcc"), dict) else m_m.get("mcc_min")
                    manifest_bal_acc = cv_m.get("balanced_accuracy") if isinstance(cv_m.get("balanced_accuracy"), (int, float)) else (m_m.get("balanced_accuracy") if isinstance(m_m.get("balanced_accuracy"), (int, float)) else (cv_m.get("balanced_accuracy", {}).get("mean") if isinstance(cv_m.get("balanced_accuracy"), dict) else None))
                    holdout_mcc = m_data.get("holdout_mcc") or cv_m.get("holdout_mcc") or m_m.get("holdout_mcc")
                    holdout_bal_acc = m_data.get("holdout_balanced_accuracy") or cv_m.get("holdout_balanced_accuracy")
                    _ci = cv_m.get("holdout_mcc_ci95")
                    holdout_ci95_low = _ci[0] if isinstance(_ci, (list, tuple)) and len(_ci) >= 1 else None
                    is_promoted = m_data.get("promoted", True)
            except Exception as ex_m:
                log_event("WARNING", f"[SignalEvaluator Warning] Failed reading manifest {manifest_path}: {ex_m}")

        # Independent feature source of truth
        for fname in [f"selected_features_{interval}_{regime_key}.json", f"selected_features_{interval}.json"]:
            if os.path.exists(fname):
                try:
                    with open(fname) as f:
                        feat_list = json.load(f)
                        if feat_list:
                            break
                except (ValueError, TypeError, KeyError, OSError) as ex_f:
                    log_event("WARNING", f"[SignalEvaluator Info] Feature file notice: {ex_f}")

        if not feat_list:
            log_event("ERROR", f"[SignalEvaluator ERROR] No feature contract list found for {interval}m ({regime_key}). Failing closed.")
            return None

        if not model_ver_str:
            model_ver_str = f"btc_{interval}m_{regime_key}_clf:v7.2.0"

        # Load Calibrator Artefact (C-04)
        calibrator_data = None
        cal_ver_str = "v1.0_default"
        cal_ece_val = 0.035
        cal_path = f"calibrator_{regime_key}_{interval}.json"
        if os.path.exists(cal_path):
            try:
                with open(cal_path, "r") as cf:
                    calibrator_data = json.load(cf)
                    cal_ver_str = calibrator_data.get("version", f"v1.0_isotonic_{interval}m")
                    cal_ece_val = float(calibrator_data.get("ece", 0.035))
            except Exception as ex_cal:
                log_event("CRITICAL", f"[SignalEvaluator CRITICAL] Failed reading calibrator {cal_path}: {ex_cal}. Setting slot to fallback.")
                calibrator_data = None
        else:
            log_event("CRITICAL", f"[SignalEvaluator CRITICAL] Calibrator {cal_path} not found on disk. Setting slot to fallback.")
            calibrator_data = None

        if calibrator_data is None:
            calibrator_data = {"scaling_method": "identity", "is_fallback": True, "version": "v0.0_missing", "ece": 0.080}

        try:
            m_trend = load_ensemble_classifier(f"ensemble_{regime_key}_trend_{interval}", n_features=len(feat_list), feature_names=feat_list)
            m_price = load_ensemble_regressor(f"ensemble_{regime_key}_price_{interval}", n_features=len(feat_list), feature_names=feat_list)
            if m_trend is None or m_price is None:
                log_event("ERROR", f"[SignalEvaluator ERROR] Ensemble loading returned None for {interval}m ({regime_key})")
                return None
                
            res = {
                "trend": m_trend,
                "price": m_price,
                "selected_features": feat_list,
                "model_version": model_ver_str,
                "git_sha": git_sha_str,
                "calibrator": calibrator_data,
                "calibrator_version": cal_ver_str,
                "calibrator_ece": cal_ece_val,
                "manifest_mcc": manifest_mcc,
                "manifest_mcc_min": manifest_mcc_min,
                "manifest_bal_acc": manifest_bal_acc,
                "holdout_mcc": holdout_mcc,
                "holdout_bal_acc": holdout_bal_acc,
                "holdout_ci95_low": holdout_ci95_low,
                "is_promoted": is_promoted
            }
            with self.state_lock:
                while len(self.models_by_interval) >= self.max_cache_size:
                    self.models_by_interval.popitem(last=False)
                    gc.collect()
                self.models_by_interval[cache_key] = res
            return res
        except Exception as e:
            log_event("ERROR", f"[SignalEvaluator ERROR] Lazy model loading for {interval}m ({regime_key}): {e}")
            return None

    def evaluate_interval(self, symbol, interval):
        tf_key = TF_MAP.get(interval, f"{interval}m")
        try:
            df = get_history(symbol=symbol, interval=interval, limit=350)
            if df is None or len(df) < 50:
                return

            if symbol.upper() != "BTCUSDT" and "close_btc" not in df.columns:
                try:
                    df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=350)
                    if df_btc is not None and "close" in df_btc.columns and "timestamp" in df_btc.columns:
                        btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                        df = pd.merge(df, btc_sub, on="timestamp", how="inner")
                except Exception as ex_btc:
                    log_event("WARNING", f"[SignalEvaluator] Failed merging BTC history for {symbol}: {ex_btc}")
            if "close_btc" not in df.columns:
                df["close_btc"] = df["close"]

            df = merge_derivatives_sentiment_features(df, symbol=symbol, interval=interval)
            df = add_features(df)
            
            last_row = df.iloc[-1]
            adx_val = float(last_row.get("ADX", 20.0)) if "ADX" in last_row and not np.isnan(last_row["ADX"]) else 20.0
            
            prev_regime = self.bot_state.get(f"regime_{symbol}_{tf_key}", "")
            was_trending = "Trending" in prev_regime
            adx_enter = REGIME_ADX_ENTER_BY_INTERVAL.get(str(interval), STRONG_TREND_ADX_ENTER)
            adx_exit = REGIME_ADX_EXIT_BY_INTERVAL.get(str(interval), STRONG_TREND_ADX_EXIT)
            if ENABLE_REGIME_HYSTERESIS:
                if was_trending:
                    is_trending = adx_val >= adx_exit
                else:
                    is_trending = adx_val >= adx_enter
            else:
                is_trending = adx_val >= 20.0

            regime_str = f"Trending (ADX {adx_val:.1f})" if is_trending else f"Ranging (ADX {adx_val:.1f})"
            
            with self.state_lock:
                self.bot_state[f"regime_{symbol}_{tf_key}"] = regime_str
                self.bot_state[f"adx_{symbol}_{tf_key}"] = adx_val
                if symbol == "BTCUSDT" or symbol == self.bot_state.get("active_symbol", "BTCUSDT"):
                    self.bot_state[f"regime_{tf_key}"] = regime_str
                    self.bot_state[f"adx_{tf_key}"] = adx_val

            # Check governance slot denylist before model evaluation or rule fallback
            from ensemble import is_model_slot_denied
            _regime_key = "trending" if is_trending else "ranging"
            if is_model_slot_denied(f"{_regime_key}_trend_{interval}") or is_model_slot_denied(f"{_regime_key}_price_{interval}"):
                log_event("INFO", f"[SignalEvaluator Denylist] {symbol} {interval}m ({_regime_key}) is denied by governance policy — skipping safely.")
                denied_entry = {
                    "symbol": str(symbol),
                    "direction": "Neutral",
                    "confidence": 0.0,
                    "calibrated_confidence": 0.0,
                    "predicted_change": 0.0,
                    "signal_source": "GOVERNANCE_DENIED",
                    "timestamp": time.time()
                }
                with self.state_lock:
                    self.bot_state[f"latest_prediction_{symbol}_{tf_key}"] = denied_entry
                    if symbol == "BTCUSDT" or symbol == self.bot_state.get("active_symbol", "BTCUSDT"):
                        self.bot_state[f"latest_prediction_{tf_key}"] = denied_entry
                return

            # Lazy model evaluation
            model_eval_success = False
            models = self.get_models(interval, is_trending)
            if models is not None:
                
                # C-1 Predictive Floor Check: Refuse trading if model sits at statistical chance
                from config import MODEL_GOVERNANCE, TIMEFRAME_MIN_MCC, TIMEFRAME_MIN_BAL_ACC
                min_mcc_floor = TIMEFRAME_MIN_MCC.get(str(interval), MODEL_GOVERNANCE.get("min_mcc", 0.05))
                min_bal_acc_floor = TIMEFRAME_MIN_BAL_ACC.get(str(interval), MODEL_GOVERNANCE.get("min_balanced_accuracy", 0.36))

                mcc_val = models.get("manifest_mcc")
                mcc_min_val = models.get("manifest_mcc_min")
                bal_acc_val = models.get("manifest_bal_acc")

                if mcc_val is None:
                    man_path = f"ensemble_{_regime_key}_trend_{interval}_manifest.json"
                    if os.path.exists(man_path):
                        try:
                            import json
                            with open(man_path, "r") as mf:
                                _mdata = json.load(mf)
                                mcc_val = _mdata.get("manifest_mcc")
                                mcc_min_val = _mdata.get("manifest_mcc_min")
                                bal_acc_val = _mdata.get("manifest_bal_acc")
                        except Exception:
                            pass

                # Helper to write neutral abstention state
                def _record_abstain(reason_msg):
                    neut_pred = {
                        "symbol": str(symbol),
                        "direction": "Neutral",
                        "confidence": 0.0,
                        "calibrated_confidence": 0.0,
                        "predicted_change": 0.0,
                        "signal_source": "GOVERNANCE_ABSTAIN",
                        "status": f"Abstain ({reason_msg})",
                        "timestamp": time.time()
                    }
                    with self.state_lock:
                        self.bot_state[f"latest_prediction_{symbol}_{tf_key}"] = neut_pred
                        if symbol == "BTCUSDT" or symbol == self.bot_state.get("active_symbol", "BTCUSDT"):
                            self.bot_state[f"latest_prediction_{tf_key}"] = neut_pred

                # Fail-closed: unmeasured models (no cv_metrics) must not trade
                if mcc_val is None:
                    log_event("WARNING", f"[SignalEvaluator Gate] {interval}m ({_regime_key}): no CV metrics recorded — quality unverified. ABSTAIN.")
                    _record_abstain("Model predictive content unverified (no cv_metrics)")
                    return "Neutral", 0.0, "Model predictive content unverified (no cv_metrics)"

                if mcc_val < min_mcc_floor:
                    log_event("WARNING", f"[SignalEvaluator Gate] {interval}m ({_regime_key}): MCC ({mcc_val:.4f}) below predictive floor ({min_mcc_floor}). ABSTAIN.")
                    _record_abstain(f"MCC {mcc_val:.4f} < {min_mcc_floor}")
                    return "Neutral", 0.0, f"Model predictive content below governance floor (MCC {mcc_val:.4f} < {min_mcc_floor})"

                if mcc_min_val is not None and mcc_min_val < -0.05:
                    log_event("WARNING", f"[SignalEvaluator Gate] {interval}m ({_regime_key}): severely anti-correlated on CV fold (min MCC {mcc_min_val:.4f} < -0.05). ABSTAIN.")
                    _record_abstain(f"min fold MCC {mcc_min_val:.4f} < -0.05")
                    return "Neutral", 0.0, f"Model severely anti-correlated on CV fold (min fold MCC {mcc_min_val:.4f} < -0.05)"

                if bal_acc_val is not None and bal_acc_val < min_bal_acc_floor:
                    log_event("WARNING", f"[SignalEvaluator Gate] {interval}m ({_regime_key}): BalAcc ({bal_acc_val:.4f}) below floor ({min_bal_acc_floor}). ABSTAIN.")
                    _record_abstain(f"BalAcc {bal_acc_val:.4f} < {min_bal_acc_floor}")
                    return "Neutral", 0.0, f"Model balanced accuracy below governance floor ({bal_acc_val:.4f} < {min_bal_acc_floor})"

                try:
                    _feat_list = models.get("selected_features") or features

                    # Ensure df has no duplicate columns
                    if df.columns.duplicated().any():
                        df = df.loc[:, ~df.columns.duplicated()].copy()

                    # Ensure all required features are present in df
                    for f_col in _feat_list:
                        if f_col not in df.columns:
                            df[f_col] = 0.0

                    row_X = df[_feat_list].iloc[[-1]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    row_X_sliced = _slice_model_input(models["trend"], row_X)

                    probs = models["trend"].predict_proba(row_X_sliced)[0]
                    pred_pct = float(models["price"].predict(row_X_sliced)[0])
                    
                    if len(probs) >= 3:
                        prob_bearish = float(probs[0])
                        prob_neutral = float(probs[1])
                        prob_bullish = float(probs[2])
                    elif len(probs) == 2:
                        prob_bearish = float(probs[0])
                        prob_neutral = 0.0
                        prob_bullish = float(probs[1])
                    else:
                        prob_bearish = float(probs[0]) if float(probs[0]) < 0.5 else 0.0
                        prob_neutral = 0.0
                        prob_bullish = float(probs[0]) if float(probs[0]) >= 0.5 else 0.0

                    from config import TIMEFRAME_CONFIG, MIN_EVAL_THRESHOLD_FLOOR
                    from trade_calculators import transaction_cost_model, UnifiedTargetGenerator
                    cfg = TIMEFRAME_CONFIG.get(str(interval), {})
                    tp_m = cfg.get("tp_mult_trending", 1.85)
                    sl_m = cfg.get("sl_mult", 0.8)
                    close_price = float(df["close"].iloc[-1]) if ("close" in df.columns and len(df) > 0) else 100.0
                    atr_val = float(df["ATR"].iloc[-1]) if ("ATR" in df.columns and len(df) > 0 and pd.notna(df["ATR"].iloc[-1])) else float(close_price * 0.01)
                    actual_tp_m = UnifiedTargetGenerator.resolve_tp_multiplier(interval, close_price, atr_val, tp_m)

                    # H-4: compute round-trip cost from TCM using per-symbol ADV from df.
                    # df loaded at line 125; tail(96) × 15m bars ≈ 24h volume.
                    _bars_per_day = max(1, round(1440 / max(1, int(interval))))
                    _adv_se = float(df["volume"].tail(_bars_per_day).sum() * df["close"].iloc[-1]) if ("volume" in df.columns and "close" in df.columns and len(df) >= _bars_per_day) else 50_000_000.0
                    _order_se = float(self.bot_state.get("position_size_usd", 1000.0)) if hasattr(self, "bot_state") and isinstance(self.bot_state, dict) else 1000.0
                    _tcm = transaction_cost_model.estimate_transaction_cost(
                        order_size_usd=_order_se,
                        volume_24h_usd=_adv_se,
                        is_maker=True,
                    )
                    cost_bps = _tcm["total_cost_bps"] * 2.0  # round-trip
                    atr_norm = (atr_val / close_price) if close_price > 0 else 0.01
                    p_star = sl_m / (actual_tp_m + sl_m)
                    cost_adj = (cost_bps / 1e4) / max(1e-6, (actual_tp_m + sl_m) * max(1e-4, atr_norm))
                    
                    # Information-Theoretic Regime Entropy Hurdle:
                    # In quiet sideways markets (ADX < 25), demand higher statistical signal-to-noise ratio (up to +0.15).
                    # In strong trending markets (ADX >= 30), scale smoothly down to economic p_star baseline.
                    adx_cur = float(df["ADX"].iloc[-1]) if ("ADX" in df.columns and len(df) > 0 and pd.notna(df["ADX"].iloc[-1])) else 20.0
                    entropy_penalty = 0.15 * max(0.0, 1.0 - (adx_cur / 30.0))
                    prior_hurdle = 0.333 + entropy_penalty
                    
                    # Low-Timeframe Chop Gate: on 15m/30m, quiet market chop (ADX < 22.0) requires +5% higher conviction to avoid fee churn
                    ltf_chop_penalty = 0.05 if (str(interval) in ["15", "30"] and adx_cur < 22.0) else 0.0
                    
                    eval_threshold = round(min(0.55, max(MIN_EVAL_THRESHOLD_FLOOR, p_star + cost_adj, prior_hurdle) + ltf_chop_penalty), 4)

                    from ensemble import resolve_direction
                    _top_trend, _top_conf = resolve_direction(probs)
                    raw_conf = _top_conf
                    if _top_trend in ["Bullish", "Bearish"] and _top_conf >= eval_threshold:
                        # Governance Holdout & CV Metric Validation
                        from config import TIMEFRAME_MIN_HOLDOUT_MCC, TIMEFRAME_MIN_HOLDOUT_BAL_ACC, MODEL_GOVERNANCE
                        min_h_mcc = TIMEFRAME_MIN_HOLDOUT_MCC.get(str(interval), MODEL_GOVERNANCE.get("min_holdout_mcc", 0.02))
                        min_h_bal = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get(str(interval), MODEL_GOVERNANCE.get("min_holdout_balanced_accuracy", 0.34))
                        m_hmcc = models.get("holdout_mcc")
                        m_hbal = models.get("holdout_bal_acc")
                        m_hci = models.get("holdout_ci95_low")
                        m_prom = models.get("is_promoted")

                        if m_hmcc is not None and m_hmcc < min_h_mcc:
                            log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m Holdout MCC ({m_hmcc:.4f}) < floor ({min_h_mcc:.4f}). Neutral.")
                            direction = "Neutral"
                        elif m_hbal is not None and m_hbal < min_h_bal:
                            log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m Holdout BalAcc ({m_hbal:.4f}) < floor ({min_h_bal:.4f}). Neutral.")
                            direction = "Neutral"
                        elif m_hci is not None and m_hci < -0.05:
                            log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m Holdout CI95 lower bound ({m_hci:.4f}) < -0.05. Neutral.")
                            direction = "Neutral"
                        elif m_prom is False:
                            log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m manifest promoted=False. Neutral.")
                            direction = "Neutral"
                        else:
                            direction = _top_trend
                    else:
                        direction = "Neutral"

                    setup_type = "Standard"
                    macro_bias = get_hierarchical_macro_bias(getattr(self, "bot_state", {}), symbol)
                    
                    if direction in ["Bullish", "Bearish"]:
                        # Multi-Timeframe Hierarchical Trend Lock (15m/30m/1h/2h align with 4h macro trend)
                        if str(interval) in ["15", "30", "60", "120"]:
                            macro_dir = macro_bias.get("direction", "Neutral")
                            rsi_val = float(df["RSI"].iloc[-1]) if ("RSI" in df.columns and len(df) > 0 and pd.notna(df["RSI"].iloc[-1])) else 50.0
                            
                            if macro_dir == "Bullish":
                                if direction == "Bearish":
                                    # Hard Directional Lock: Suppress counter-trend short entries into 4H bull trend
                                    log_event("INFO", f"[Hierarchical 4H Lock] {symbol} {interval}m Bearish suppressed under 4H Bullish macro trend (ADX {macro_bias.get('adx', 0):.1f}). Waiting for dip support.")
                                    direction = "Neutral"
                                    setup_type = "Waiting for Dip Support"
                                elif direction == "Bullish":
                                    setup_type = "Pullback Long" if rsi_val <= 55.0 else "Breakout Long"
                            elif macro_dir == "Bearish":
                                if direction == "Bullish":
                                    # Hard Directional Lock: Suppress counter-trend long entries into 4H bear trend
                                    log_event("INFO", f"[Hierarchical 4H Lock] {symbol} {interval}m Bullish suppressed under 4H Bearish macro trend (ADX {macro_bias.get('adx', 0):.1f}). Waiting for resistance rejection.")
                                    direction = "Neutral"
                                    setup_type = "Waiting for Resistance Rejection"
                                elif direction == "Bearish":
                                    setup_type = "Relief Bounce Short" if rsi_val >= 45.0 else "Breakdown Short"

                    if direction in ["Bullish", "Bearish"]:
                        from meta_labeler import evaluate_meta_filter
                        meta_approved, meta_prob = evaluate_meta_filter(symbol, interval, direction, confidence=raw_conf)
                        if not meta_approved:
                            log_event("INFO", f"[MetaLabeler Filtered] {symbol} {interval}m {direction} signal rejected by second-stage meta-classifier (meta_prob={meta_prob*100:.1f}%). Filtered to Neutral.")
                            direction = "Neutral"

                    calibrator = models.get("calibrator")
                    if calibrator is not None and isinstance(calibrator, dict) and direction in ["Bullish", "Bearish"]:
                        from tools.beta_calibrator import calibrate_probability, is_calibrator_viable
                        p_star_req = float(round(p_star + cost_adj, 4))
                        if not is_calibrator_viable(calibrator, min_required_p_star=p_star_req):
                            log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m calibrator unviable or fallback. Filtering to Neutral (Fail-Closed).")
                            direction = "Neutral"
                            calibrated_conf = 0.50
                        else:
                            calibrated_conf = float(calibrate_probability(raw_conf, calibrator, min_required_p_star=p_star_req))
                    elif direction in ["Bullish", "Bearish"]:
                        log_event("WARNING", f"[Signal Evaluator] {symbol} {interval}m missing calibrator. Filtering to Neutral (Fail-Closed).")
                        direction = "Neutral"
                        calibrated_conf = 0.50
                    else:
                        calibrated_conf = float(raw_conf)

                    cal_ver = models.get("calibrator_version") or (calibrator.get("version", "v1.0") if isinstance(calibrator, dict) else "v1.0_default")
                    cal_ece = float(models.get("calibrator_ece") or (calibrator.get("ece", 0.035) if isinstance(calibrator, dict) else 0.035))
                    served_version = models.get("model_version")

                    pred_entry = {
                        "symbol": str(symbol),
                        "direction": str(direction),
                        "confidence": float(raw_conf),
                        "calibrated_confidence": float(calibrated_conf),
                        "predicted_change": float(pred_pct * float(last_row["close"])),
                        "signal_source": "ML_ENSEMBLE",
                        "model_version": served_version,
                        "calibrator_version": cal_ver,
                        "calibrator_ece": cal_ece,
                        "manifest_mcc": mcc_val,
                        "is_fallback": False,
                        "setup_type": setup_type,
                        "macro_4h_direction": str(macro_bias.get("direction", "Neutral")),
                        "timestamp": time.time()
                    }
                    with self.state_lock:
                        self.bot_state[f"latest_prediction_{symbol}_{tf_key}"] = pred_entry
                        if symbol == "BTCUSDT" or symbol == self.bot_state.get("active_symbol", "BTCUSDT"):
                            self.bot_state[f"latest_prediction_{tf_key}"] = pred_entry
                        
                        history = self.bot_state.get("prediction_history", [])
                        if isinstance(history, list):
                            c_ts = int(time.time() * 1000)
                            if "timestamp" in last_row:
                                try:
                                    raw_t = last_row["timestamp"]
                                    if isinstance(raw_t, (int, float, np.integer, np.floating)):
                                        v = float(raw_t)
                                        c_ts = int(v * 1000) if v < 1e11 else int(v)
                                    else:
                                        c_ts = int(pd.to_datetime(raw_t).timestamp() * 1000)
                                except (ValueError, TypeError, KeyError, AttributeError):
                                    pass
                            
                            existing_p = next((p for p in history if isinstance(p, dict) and p.get("candle_timestamp") == c_ts and str(p.get("interval")) == str(interval) and str(p.get("symbol")) == str(symbol)), None)
                            if existing_p is not None:
                                existing_p["timestamp"] = float(time.time())
                                existing_p["direction"] = str(direction)
                                existing_p["calibrated_confidence"] = float(calibrated_conf)
                                existing_p["raw_confidence"] = float(raw_conf)
                                existing_p["dynamic_threshold"] = float(eval_threshold)
                            else:
                                new_pred = {
                                    "prediction_id": f"{symbol}_{interval}_{int(c_ts)}",
                                    "symbol": str(symbol),
                                    "timestamp": float(time.time()),
                                    "candle_timestamp": c_ts,
                                    "interval": str(interval),
                                    "direction": str(direction),
                                    "ref_price": float(last_row["close"]),
                                    "predicted_change": float(pred_pct * float(last_row["close"])),
                                    "predicted_price": float(last_row["close"]) * (1.0 + pred_pct),
                                    "status": f"Skipped (Neutral)" if direction == "Neutral" else "Pending Risk Evaluation",
                                    "calibrated_confidence": float(calibrated_conf),
                                    "raw_confidence": float(raw_conf),
                                    "dynamic_threshold": float(eval_threshold),
                                    "evaluation": {"evaluated": False, "exit_price": None, "change": None, "change_pct": None, "success": None}
                                }
                                history.append(new_pred)
                                try:
                                    from database import save_prediction as db_save_pred
                                    db_save_pred(new_pred)
                                except Exception as ex_save:
                                    log_event("WARNING", f"[SignalEvaluator] Failed saving prediction to DB: {ex_save}")
                            
                            try:
                                from decision_journal import DecisionRecord, write_decision
                                rec = DecisionRecord(
                                    ts=float(time.time()),
                                    candle_timestamp=c_ts,
                                    symbol=str(symbol),
                                    interval=str(interval),
                                    signal_source="ML_ENSEMBLE",
                                    direction=str(direction),
                                    raw_confidence=float(raw_conf),
                                    calibrated_conf=float(calibrated_conf),
                                    regime=regime_str,
                                    adx=adx_val,
                                    outcome="SKIPPED" if direction == "Neutral" else "EVALUATED",
                                    reject_reason=f"Skipped (Neutral)" if direction == "Neutral" else "Pending Risk Evaluation"
                                )
                                rec.snapshot(
                                    predicted_change=float(pred_pct * float(last_row["close"])),
                                    dynamic_threshold=float(eval_threshold)
                                )
                                write_decision(rec)
                            except Exception as ex_dj:
                                log_event("WARNING", f"[SignalEvaluator] Decision journal write notice: {ex_dj}")
                            if len(history) > 200:
                                self.bot_state["prediction_history"] = history[-200:]
                    model_eval_success = True
                except (NameError, AttributeError) as prog_err:
                    import traceback
                    log_event("CRITICAL", f"[SignalEvaluator CRITICAL PROGRAMMING ERROR] {prog_err}\n{traceback.format_exc()}")
                    raise prog_err
                except RuntimeError as contract_err:
                    import traceback
                    log_event("WARNING", f"[SignalEvaluator Contract Warning] Feature contract mismatch for {symbol} {interval}m: {contract_err}. Falling back to rule-based evaluation.")
                    model_eval_success = False
                except Exception as ex_m:
                    import traceback
                    log_event("WARNING", f"[SignalEvaluator Warning] ML Ensemble inference failed for {symbol} {interval}m: {ex_m}. Falling back to rule-based evaluation.")
                    model_eval_success = False

            if not model_eval_success:
                # Technical rule-based signal fallback (Logged and tagged explicitly as fallback)
                print(f"[Signal WARNING] ML Model unavailable for {symbol} {interval}m. Engaging rule-based fallback with capped confidence.")
                rsi = float(last_row.get("RSI", 50.0)) if "RSI" in last_row and not np.isnan(last_row["RSI"]) else 50.0
                ema9 = float(last_row.get("EMA_9", last_row["close"]))
                ema21 = float(last_row.get("EMA_21", last_row["close"]))
                close_p = float(last_row["close"])
                
                if ema9 >= ema21:
                    direction = "Bullish"
                    conf = min(0.55, 0.50 + max(0.0, (rsi - 45.0) / 200.0))
                    change_val = close_p * 0.003
                else:
                    direction = "Bearish"
                    conf = min(0.55, 0.50 + max(0.0, (55.0 - rsi) / 200.0))
                    change_val = -close_p * 0.003

                with self.state_lock:
                    fallback_dict = {
                        "symbol": str(symbol),
                        "direction": str(direction),
                        "confidence": float(conf),
                        "calibrated_confidence": float(conf),
                        "predicted_change": float(change_val),
                        "signal_source": "RULE_BASED_FALLBACK",
                        "calibrator_version": "v0.0_fallback",
                        "calibrator_ece": 0.080,
                        "is_fallback": True
                    }
                    self.bot_state[f"latest_prediction_{symbol}_{tf_key}"] = fallback_dict
                    if symbol == "BTCUSDT" or symbol == self.bot_state.get("active_symbol", "BTCUSDT"):
                        self.bot_state[f"latest_prediction_{tf_key}"] = fallback_dict
                    history = self.bot_state.get("prediction_history", [])
                    if isinstance(history, list):
                        c_ts = int(time.time() * 1000)
                        exists = any(p.get("candle_timestamp") == c_ts and p.get("interval") == str(interval) and p.get("symbol") == str(symbol) for p in history if isinstance(p, dict))
                        if not exists:
                            new_pred = {
                                "symbol": str(symbol),
                                "timestamp": float(time.time()),
                                "candle_timestamp": c_ts,
                                "interval": str(interval),
                                "direction": str(direction),
                                "ref_price": float(close_p),
                                "predicted_change": float(change_val),
                                "predicted_price": float(close_p + change_val),
                                "status": f"Skipped (Neutral)" if direction == "Neutral" else f"Fallback ({direction})",
                                "calibrated_confidence": float(conf),
                                "raw_confidence": float(conf),
                                "dynamic_threshold": None,
                                "evaluation": {"evaluated": False, "exit_price": None, "change": None, "change_pct": None, "success": None}
                            }
                            history.append(new_pred)
                            try:
                                from database import save_prediction as db_save_pred
                                db_save_pred(new_pred)
                            except Exception as ex_save:
                                log_event("WARNING", f"[SignalEvaluator] Failed saving fallback prediction to DB: {ex_save}")
                            try:
                                from decision_journal import DecisionRecord, write_decision
                                rec = DecisionRecord(
                                    ts=float(time.time()),
                                    candle_timestamp=c_ts,
                                    symbol=str(symbol),
                                    interval=str(interval),
                                    signal_source="RULE_BASED_FALLBACK",
                                    direction=str(direction),
                                    raw_confidence=float(conf),
                                    calibrated_conf=float(conf),
                                    regime=regime_str,
                                    adx=adx_val,
                                    outcome="SKIPPED" if direction == "Neutral" else "FALLBACK",
                                    reject_reason=f"Skipped (Neutral)" if direction == "Neutral" else f"Fallback ({direction})"
                                ).snapshot(
                                    predicted_change=float(change_val)
                                )
                                write_decision(rec)
                            except Exception as ex_dj:
                                log_event("WARNING", f"[SignalEvaluator] Decision journal fallback write notice: {ex_dj}")
                            if len(history) > 200:
                                self.bot_state["prediction_history"] = history[-200:]

            # Update Confluence Results for UI
            self.update_confluence_results(tf_key, df, symbol)
            with self.state_lock:
                sig_src = self.bot_state.get(f"latest_prediction_{symbol}_{tf_key}", {}).get("signal_source", "UNKNOWN")
            print(f"[SignalEvaluator] Evaluated {interval}m ({symbol}): Regime={regime_str}, Direction={direction}, Source={sig_src}, ADX={adx_val:.1f}")

        except Exception as e:
            import traceback
            print(f"[SignalEvaluator Error] Exception evaluating {interval}m ({symbol}): {e}\n{traceback.format_exc()}")

    def update_confluence_results(self, tf_key, df, symbol):
        if df is None or len(df) == 0:
            return
        last_row = df.iloc[-1]
        close = float(last_row["close"])
        ema9 = float(last_row.get("EMA_9", close))
        ema21 = float(last_row.get("EMA_21", close))
        ema200 = float(last_row.get("EMA_200", close))
        rsi = float(last_row.get("RSI", 50.0))
        vol = float(last_row.get("volume", 0.0))
        avg_vol = float(df["volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else vol
        vol_ratio = float(last_row.get("volume_ratio", vol / (avg_vol + 1e-8)))
        bb_high = float(last_row.get("BB_high", close * 1.02))
        bb_low = float(last_row.get("BB_low", close * 0.98))
        ret_5m = float(last_row.get("return_5m", 0.0))
        atr_norm = float(last_row.get("ATR_norm", 0.01))
        adx_val = float(last_row.get("ADX", 20.0))
        
        # Pred info & external data lookup
        with self.state_lock:
            pred_info = dict(self.bot_state.get(f"latest_prediction_{symbol}_{tf_key}", {}))
            ofi = self.bot_state.get("latest_ofi")
            fg_raw = last_row.get("fear_greed")
            fg_val = float(fg_raw) if (fg_raw is not None and not pd.isna(pd.to_numeric(fg_raw, errors="coerce"))) else None
            sentiment = self.bot_state.get("latest_sentiment_score") or fg_val
            pred_15 = self.bot_state.get(f"latest_prediction_{symbol}_15m", {}).get("direction")
            pred_1h = self.bot_state.get(f"latest_prediction_{symbol}_1h", {}).get("direction")

        pred_change = abs(float(pred_info.get("predicted_change", 0.0)))
        fee_hurdle = close * 0.0012

        # 1d / Macro Trend
        pass_1d = bool(close >= ema200)
        detail_1d = f"Price ({close:.1f}) >= EMA200 ({ema200:.1f})" if pass_1d else f"Price ({close:.1f}) < EMA200 ({ema200:.1f})"

        # BB Guard
        pass_bb = bool(bb_low <= close <= bb_high)
        detail_bb = f"Price {close:.1f} inside BB [{bb_low:.1f}, {bb_high:.1f}]"

        # Counter Momentum
        pass_cm = bool(abs(ret_5m) <= 0.02)
        detail_cm = f"Recent return {ret_5m*100:+.2f}% within safe momentum limits"

        # Volatility Guard
        pass_vol_g = bool(atr_norm <= 0.025)
        detail_vol_g = f"Normalized ATR {atr_norm*100:.2f}% (max 2.50%)"

        # ADX Regime
        pass_adx = bool(adx_val >= 15.0)
        detail_adx = f"ADX {adx_val:.1f} confirms active regime (min 15.0)"

        # Fee Coverage
        if pred_change > 0:
            pass_fee = bool(pred_change >= fee_hurdle)
            detail_fee = f"Expected move ${pred_change:.2f} vs Fee Hurdle ${fee_hurdle:.2f}"
        else:
            pass_fee = None
            detail_fee = "Not Evaluated (No Model Target)"

        # Orderbook Imbalance (L2 Stream)
        if ofi is not None:
            pass_ofi = bool(ofi >= 0.0)
            detail_ofi = f"L2 Orderbook OFI {ofi:+.3f}"
        else:
            pass_ofi = None
            detail_ofi = "Not Evaluated (No L2 Data Stream)"

        # News Sentiment
        if sentiment is not None:
            pass_sent = bool(float(sentiment) >= 30.0)
            detail_sent = f"Sentiment Index {float(sentiment):.1f} (min 30.0)"
        else:
            pass_sent = None
            detail_sent = "Not Evaluated (No Sentiment Stream)"

        # Expected Change / Conf
        raw_conf = float(pred_info.get("confidence", 0.0))
        if raw_conf > 0:
            pass_exp = bool(raw_conf >= 0.52)
            detail_exp = f"Model confidence {raw_conf*100:.1f}% >= 52.0%"
        else:
            pass_exp = None
            detail_exp = "Not Evaluated (No Model Target)"

        # Timeframe Alignment
        if pred_15 and pred_1h:
            pass_align = bool(pred_15 == pred_1h)
            detail_align = f"15m ({pred_15}) aligned with 1h ({pred_1h})"
        else:
            pass_align = None
            detail_align = "Not Evaluated (Multi-TF Pending)"

        # Open Interest Delta
        oi_change = float(last_row.get("oi_change_1h", 0.0)) if "oi_change_1h" in last_row else None
        if oi_change is not None and not np.isnan(oi_change) and oi_change != 0.0:
            pass_oi = bool(oi_change >= 0.0)
            detail_oi = f"1h Open Interest Delta {oi_change:+.2f}%"
        else:
            pass_oi = None
            detail_oi = "Not Evaluated (No Live OI Feed)"

        with self.state_lock:
            confl_dict = {
                "checks": {
                    "1d_Trend": {"pass": pass_1d, "detail": detail_1d},
                    "4h_Trend": {"pass": bool(ema9 >= ema21), "detail": f"EMA9 ({ema9:.2f}) vs EMA21 ({ema21:.2f})"},
                    "4h_RSI": {"pass": bool(rsi >= 30.0 and rsi <= 70.0), "detail": f"4h RSI {rsi:.1f} in safe neutral band [30, 70]"},
                    "1h_RSI": {"pass": bool(rsi >= 25.0 and rsi <= 75.0), "detail": f"1h RSI {rsi:.1f} in safe neutral band [25, 75]"},
                    "Volume_Participation": {"pass": bool(vol_ratio >= 0.8), "detail": f"Volume ratio {vol_ratio:.2f}x vs 20-avg (min 0.8x)"},
                    "BB_Edge_Guard": {"pass": pass_bb, "detail": detail_bb},
                    "Counter_Momentum": {"pass": pass_cm, "detail": detail_cm},
                    "Volatility_Guard": {"pass": pass_vol_g, "detail": detail_vol_g},
                    "ADX_Regime": {"pass": pass_adx, "detail": detail_adx},
                    "Fee_Coverage": {"pass": pass_fee, "detail": detail_fee},
                    "Orderbook_Imbalance": {"pass": pass_ofi, "detail": detail_ofi},
                    "News_Sentiment": {"pass": pass_sent, "detail": detail_sent},
                    "Expected_Change": {"pass": pass_exp, "detail": detail_exp},
                    "Timeframe_Alignment": {"pass": pass_align, "detail": detail_align},
                    "Open_Interest_Delta": {"pass": pass_oi, "detail": detail_oi}
                }
            }
            self.bot_state[f"confluence_results_{symbol}_{tf_key}"] = confl_dict
            self.bot_state[f"confluence_results_{tf_key}"] = confl_dict


def verify_manifest_health():
    """Startup assertion verifying presence of all 10 required feature-contract manifests."""
    import os
    missing = []
    for rg in ["trending", "ranging"]:
        for iv in ["15", "30", "60", "120", "240"]:
            mf = f"ensemble_{rg}_trend_{iv}_manifest.json"
            if not os.path.exists(mf):
                missing.append(mf)
    if missing:
        log_event("WARNING", f"[SignalEvaluator Startup Warning] Missing {len(missing)}/10 manifests: {missing}")
    else:
        log_event("INFO", "[SignalEvaluator Startup] All 10 feature-contract manifests present and verified.")

def run_signal_evaluator_loop(bot_state):
    log_event("INFO", "[SignalEvaluator] Background market evaluation worker thread started.")
    verify_manifest_health()
    evaluator = SignalEvaluator(bot_state)
    import gc, ctypes
    while True:
        try:
            from config import SUPPORTED_SYMBOLS
            for sym in SUPPORTED_SYMBOLS:
                for iv in ["15", "30", "60", "120", "240"]:
                    try:
                        evaluator.evaluate_interval(symbol=sym, interval=iv)
                    except RuntimeError as run_err:
                        log_event("CRITICAL", f"[SignalEvaluator Governance Alert] {sym} {iv}m failed contract verification: {run_err}")
                    except Exception as iv_err:
                        log_event("ERROR", f"[SignalEvaluator Error] {sym} {iv}m evaluation error: {iv_err}")
                    time.sleep(1)
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except (OSError, AttributeError):
                pass
        except Exception as e:
            log_event("ERROR", f"[SignalEvaluator Loop Error] {e}")
        time.sleep(60)
