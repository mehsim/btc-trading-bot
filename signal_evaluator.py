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
from config import ENABLE_REGIME_HYSTERESIS, STRONG_TREND_ADX_ENTER, STRONG_TREND_ADX_EXIT

TF_MAP = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h"}

class SignalEvaluator:
    def __init__(self, bot_state):
        self.bot_state = bot_state
        self.state_lock = threading.Lock()
        self.models_by_interval = {}

    def get_models(self, interval: str, is_trending: bool):
        regime_key = "trending" if is_trending else "ranging"
        cache_key = f"{interval}_{regime_key}"
        if cache_key in self.models_by_interval:
            return self.models_by_interval[cache_key]
        if interval in self.models_by_interval and isinstance(self.models_by_interval[interval], dict) and regime_key in self.models_by_interval[interval]:
            return self.models_by_interval[interval][regime_key]

        import json, os, gc

        manifest_path = f"ensemble_{regime_key}_trend_{interval}_manifest.json"
        feat_list = None
        model_ver_str = None
        git_sha_str = None

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as mf:
                    m_data = json.load(mf)
                    feat_list = m_data.get("feature_names") or m_data.get("surviving_features")
                    m_ver = m_data.get("model_version", "v7.2.0")
                    git_sha_str = m_data.get("git_sha", "unknown")
                    model_ver_str = f"btc_{interval}m_{regime_key}_clf:{m_ver}"
            except Exception as ex_m:
                log_event("WARNING", f"[SignalEvaluator Warning] Failed reading manifest {manifest_path}: {ex_m}")

        if not feat_list:
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
                log_event("WARNING", f"[SignalEvaluator Warning] Failed reading calibrator {cal_path}: {ex_cal}")

        if calibrator_data is None:
            calibrator_data = {"X": [0.0, 1.0], "y": [0.0, 1.0], "version": cal_ver_str, "ece": cal_ece_val}

        try:
            m_trend = load_ensemble_classifier(f"ensemble_{regime_key}_trend_{interval}", n_features=len(feat_list), feature_names=feat_list)
            m_price = load_ensemble_regressor(f"ensemble_{regime_key}_price_{interval}", n_features=len(feat_list), feature_names=feat_list)
            if m_trend is None or m_price is None:
                log_event("ERROR", f"[SignalEvaluator ERROR] Ensemble loading returned None for {interval}m ({regime_key})")
                return None
                
            # Keep at most 2 active regime models in memory
            if len(self.models_by_interval) >= 2:
                old_k = list(self.models_by_interval.keys())[0]
                del self.models_by_interval[old_k]
                gc.collect()

            res = {
                "trend": m_trend,
                "price": m_price,
                "selected_features": feat_list,
                "model_version": model_ver_str,
                "git_sha": git_sha_str,
                "calibrator": calibrator_data,
                "calibrator_version": cal_ver_str,
                "calibrator_ece": cal_ece_val
            }
            self.models_by_interval[cache_key] = res
            return res
        except Exception as e:
            log_event("ERROR", f"[SignalEvaluator ERROR] Lazy model loading for {interval}m ({regime_key}): {e}")
            return None

    def evaluate_interval(self, symbol="BTCUSDT", interval="15"):
        tf_key = TF_MAP.get(interval, f"{interval}m")
        try:
            df = get_history(symbol=symbol, interval=interval, limit=200)
            if df is None or len(df) < 50:
                return

            df = merge_derivatives_sentiment_features(df, symbol=symbol, interval=interval)
            df = add_features(df)
            
            last_row = df.iloc[-1]
            adx_val = float(last_row.get("ADX", 20.0)) if "ADX" in last_row and not np.isnan(last_row["ADX"]) else 20.0
            
            prev_regime = self.bot_state.get(f"regime_{tf_key}", "")
            was_trending = "Trending" in prev_regime
            if ENABLE_REGIME_HYSTERESIS:
                if was_trending:
                    is_trending = adx_val >= STRONG_TREND_ADX_EXIT
                else:
                    is_trending = adx_val >= STRONG_TREND_ADX_ENTER
            else:
                is_trending = adx_val >= 20.0

            regime_str = f"Trending (ADX {adx_val:.1f})" if is_trending else f"Ranging (ADX {adx_val:.1f})"
            
            with self.state_lock:
                self.bot_state[f"regime_{tf_key}"] = regime_str
                self.bot_state[f"adx_{tf_key}"] = adx_val

            # Lazy model evaluation
            model_eval_success = False
            models = self.get_models(interval, is_trending)
            if models is not None:
                try:
                    _regime_key = "trending" if is_trending else "ranging"
                    _feat_list = models.get("selected_features") or features

                    # Ensure all required features are present in df
                    for f_col in _feat_list:
                        if f_col not in df.columns:
                            df[f_col] = 0.0

                    row_X = df[_feat_list].iloc[[-1]]
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

                    dir_total = prob_bearish + prob_bullish
                    if str(interval) in ["15", "30"] and dir_total >= 0.15:
                        norm_bear = prob_bearish / max(1e-9, dir_total)
                        norm_bull = prob_bullish / max(1e-9, dir_total)
                        if norm_bull >= 0.52:
                            direction = "Bullish"
                            raw_conf = min(0.95, max(0.55, norm_bull * (1.0 - prob_neutral * 0.2)))
                        elif norm_bear >= 0.52:
                            direction = "Bearish"
                            raw_conf = min(0.95, max(0.55, norm_bear * (1.0 - prob_neutral * 0.2)))
                        else:
                            direction = "Neutral"
                            raw_conf = max(prob_bullish, prob_bearish, prob_neutral)
                    else:
                        if prob_bullish > max(prob_bearish, prob_neutral) and prob_bullish >= 0.50:
                            direction = "Bullish"
                            raw_conf = prob_bullish
                        elif prob_bearish > max(prob_bullish, prob_neutral) and prob_bearish >= 0.50:
                            direction = "Bearish"
                            raw_conf = prob_bearish
                        else:
                            direction = "Neutral"
                            raw_conf = max(prob_bullish, prob_bearish, prob_neutral)

                    calibrator = models.get("calibrator")
                    if calibrator is not None and isinstance(calibrator, dict) and "X" in calibrator and "y" in calibrator and direction in ["Bullish", "Bearish"]:
                        calibrated_conf = float(np.interp(raw_conf, calibrator["X"], calibrator["y"]))
                    else:
                        calibrated_conf = float(raw_conf)

                    cal_ver = models.get("calibrator_version") or (calibrator.get("version", "v1.0") if isinstance(calibrator, dict) else "v1.0_default")
                    cal_ece = float(models.get("calibrator_ece") or (calibrator.get("ece", 0.035) if isinstance(calibrator, dict) else 0.035))
                    served_version = models.get("model_version") or f"btc_{interval}m_{_regime_key}_clf:v1.0"

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
                        "is_fallback": False,
                        "timestamp": time.time()
                    }
                    with self.state_lock:
                        self.bot_state[f"latest_prediction_{tf_key}"] = pred_entry
                        
                        # Append to prediction_history for dashboard and telemetry
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
                            exists = any(p.get("candle_timestamp") == c_ts and p.get("interval") == str(interval) and p.get("symbol") == str(symbol) for p in history if isinstance(p, dict))
                            if not exists:
                                history.append({
                                    "symbol": str(symbol),
                                    "timestamp": float(time.time()),
                                    "candle_timestamp": c_ts,
                                    "interval": str(interval),
                                    "direction": str(direction),
                                    "ref_price": float(last_row["close"]),
                                    "predicted_change": float(pred_pct * float(last_row["close"])),
                                    "predicted_price": float(last_row["close"]) * (1.0 + pred_pct),
                                    "status": f"Evaluated ({direction})",
                                    "calibrated_confidence": float(calibrated_conf),
                                    "raw_confidence": float(raw_conf),
                                    "dynamic_threshold": 0.52,
                                    "evaluation": {"evaluated": False, "exit_price": None, "change": None, "change_pct": None, "success": None}
                                })
                                if len(history) > 200:
                                    self.bot_state["prediction_history"] = history[-200:]
                    model_eval_success = True
                except (NameError, AttributeError) as prog_err:
                    import traceback
                    print(f"[SignalEvaluator CRITICAL PROGRAMMING ERROR] {prog_err}\n{traceback.format_exc()}")
                    raise prog_err
                except Exception as ex_m:
                    import traceback
                    print(f"[SignalEvaluator ERROR] ML Ensemble inference failed for {symbol} {interval}m: {ex_m}\n{traceback.format_exc()}")

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
                    self.bot_state[f"latest_prediction_{tf_key}"] = {
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
                    history = self.bot_state.get("prediction_history", [])
                    if isinstance(history, list):
                        c_ts = int(time.time() * 1000)
                        exists = any(p.get("candle_timestamp") == c_ts and p.get("interval") == str(interval) and p.get("symbol") == str(symbol) for p in history if isinstance(p, dict))
                        if not exists:
                            history.append({
                                "symbol": str(symbol),
                                "timestamp": float(time.time()),
                                "candle_timestamp": c_ts,
                                "interval": str(interval),
                                "direction": str(direction),
                                "ref_price": float(close_p),
                                "predicted_change": float(change_val),
                                "predicted_price": float(close_p + change_val),
                                "status": f"Fallback ({direction})",
                                "calibrated_confidence": float(conf),
                                "raw_confidence": float(conf),
                                "dynamic_threshold": 0.50,
                                "evaluation": {"evaluated": False, "exit_price": None, "change": None, "change_pct": None, "success": None}
                            })
                            if len(history) > 200:
                                self.bot_state["prediction_history"] = history[-200:]

            # Update Confluence Results for UI
            self.update_confluence_results(tf_key, df, symbol)
            with self.state_lock:
                sig_src = self.bot_state.get(f"latest_prediction_{tf_key}", {}).get("signal_source", "UNKNOWN")
            print(f"[SignalEvaluator] Evaluated {interval}m: Regime={regime_str}, Direction={direction}, Source={sig_src}, ADX={adx_val:.1f}")

        except Exception as e:
            import traceback
            print(f"[SignalEvaluator Error] Exception evaluating {interval}m: {e}\n{traceback.format_exc()}")

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
            pred_info = dict(self.bot_state.get(f"latest_prediction_{tf_key}", {}))
            ofi = self.bot_state.get("latest_ofi")
            sentiment = self.bot_state.get("latest_sentiment_score") or (last_row["fear_greed"] if "fear_greed" in last_row and not np.isnan(last_row["fear_greed"]) else None)
            pred_15 = self.bot_state.get("latest_prediction_15m", {}).get("direction")
            pred_1h = self.bot_state.get("latest_prediction_1h", {}).get("direction")

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
            self.bot_state[f"confluence_results_{tf_key}"] = {
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


def run_signal_evaluator_loop(bot_state):
    print("[SignalEvaluator] Background market evaluation worker thread started.")
    evaluator = SignalEvaluator(bot_state)
    import gc, ctypes
    while True:
        try:
            for iv in ["15", "30", "60", "120", "240"]:
                evaluator.evaluate_interval(symbol="BTCUSDT", interval=iv)
                time.sleep(1)
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except (OSError, AttributeError):
                pass
        except Exception as e:
            print(f"[SignalEvaluator Loop Error] {e}")
        time.sleep(60)
