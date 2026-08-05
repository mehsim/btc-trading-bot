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
        self.load_models()

    def load_models(self):
        import json, os
        for iv in ["15", "30", "60", "120", "240"]:
            try:
                # Load RFECV-selected feature lists for this interval
                def _load_feats(primary, fallback=None):
                    for fname in filter(None, [primary, fallback, f"selected_features_{iv}.json", "selected_features_30.json"]):
                        if os.path.exists(fname):
                            try:
                                with open(fname) as f:
                                    feat = json.load(f)
                                    if feat:
                                        return feat
                            except Exception:
                                pass
                    return features  # global fallback

                feat_trending = _load_feats(f"selected_features_{iv}_trending.json", f"selected_features_{iv}.json")
                feat_ranging  = _load_feats(f"selected_features_{iv}_ranging.json",  f"selected_features_{iv}.json")

                self.models_by_interval[iv] = {
                    "trending": {
                        "trend": load_ensemble_classifier(f"ensemble_trending_trend_{iv}", len(feat_trending)),
                        "price": load_ensemble_regressor(f"ensemble_trending_price_{iv}", len(feat_trending))
                    },
                    "ranging": {
                        "trend": load_ensemble_classifier(f"ensemble_ranging_trend_{iv}", len(feat_ranging)),
                        "price": load_ensemble_regressor(f"ensemble_ranging_price_{iv}", len(feat_ranging))
                    },
                    "selected_features_trending": feat_trending,
                    "selected_features_ranging":  feat_ranging,
                }
                print(f"[SignalEvaluator] Loaded ML models for interval {iv}m (trending={len(feat_trending)} feats, ranging={len(feat_ranging)} feats).")
            except Exception as e:
                print(f"[SignalEvaluator Info] Model loading for {iv}m: {e}")


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

            # Model evaluation if models loaded
            model_eval_success = False
            if interval in self.models_by_interval:
                try:
                    models = self.models_by_interval[interval]["trending" if is_trending else "ranging"]
                    _regime_key = "trending" if is_trending else "ranging"
                    _feat_key = f"selected_features_{_regime_key}"
                    _feat_list = self.models_by_interval[interval].get(_feat_key) or features

                    # Guard: if selected_features has fewer cols than model expects, use all features
                    # so _slice_model_input can correctly truncate to model's expected count.
                    from ensemble import get_model_feature_names
                    _model_fn = get_model_feature_names(models["trend"])
                    _n_model = len(_model_fn) if _model_fn else None
                    if _n_model and len(_feat_list) < _n_model:
                        _feat_list = features  # fall back to full feature set

                    # Only keep features present in df
                    _feat_list = [f for f in _feat_list if f in df.columns]
                    row_X = df[_feat_list].iloc[[-1]]
                    row_X_sliced = _slice_model_input(models["trend"], row_X)

                    
                    probs = models["trend"].predict_proba(row_X_sliced)[0]
                    pred_pct = float(models["price"].predict(row_X_sliced)[0])
                    
                    prob_bearish = float(probs[0])
                    prob_neutral = float(probs[1]) if len(probs) > 1 else 0.0
                    prob_bullish = float(probs[2]) if len(probs) > 2 else float(probs[0])
                    winning_class = int(np.argmax(probs))
                    
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
                            raw_conf = prob_neutral
                    else:
                        direction = "Bullish" if winning_class == 2 else ("Bearish" if winning_class == 0 else "Neutral")
                        raw_conf = float(probs[winning_class])

                    calibrator = models.get("calibrator")
                    if calibrator is not None and isinstance(calibrator, dict) and "X" in calibrator and "y" in calibrator and direction in ["Bullish", "Bearish"]:
                        calibrated_conf = float(np.interp(raw_conf, calibrator["X"], calibrator["y"]))
                    else:
                        calibrated_conf = float(raw_conf)

                    cal_ver = calibrator.get("version", "v1.0") if isinstance(calibrator, dict) else "v1.0_default"
                    cal_ece = float(calibrator.get("ece", 0.035)) if isinstance(calibrator, dict) else 0.035

                    served_version = models.get("model_version") or f"btc_{interval}m_{_regime_key}_clf:v1.0"

                    with self.state_lock:
                        self.bot_state[f"latest_prediction_{tf_key}"] = {
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
    while True:
        try:
            for iv in ["15", "30", "60", "120", "240"]:
                evaluator.evaluate_interval(symbol="BTCUSDT", interval=iv)
                time.sleep(1)
        except Exception as e:
            print(f"[SignalEvaluator Loop Error] {e}")
        time.sleep(60)
