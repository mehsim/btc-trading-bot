import os
import json
import numpy as np
import pandas as pd
import features as features_module
from config import TIMEFRAME_CONFIG

SYMBOL = "BTCUSDT"
INTERVAL = "60"

features = [
    "RSI", "MACD_diff", "MFI", "ATR_norm",
    "close_to_EMA9", "close_to_EMA21", "close_to_EMA50", "close_to_EMA200", "EMA9_to_EMA21", 
    "BB_pct", "BB_width", "return_5m", "volatility_10m", "volume_ratio",
    "high_low_ratio", "open_close_ratio", "RSI_diff", "MACD_diff_diff", "ROC_5", "ROC_10",
    "ADX", "ADX_pos", "ADX_neg", "close_to_VWAP",
    "btc_return_5m", "btc_return_5m_lag1", "btc_return_5m_lag2", "btc_return_5m_lag3",
    "RSI_24", "ROC_24", "volatility_24h",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "RSI_z", "ADX_z", "close_to_Kalman",
    "stoch_k", "stoch_d", "williams_r", "cci", "bb_zscore"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

def get_core_features_for_interval(interval: str = "60") -> list:
    """
    Finding #153: Returns the deterministic list of features for the specified timeframe interval.
    Inspects on-disk manifests or selected_features_{interval}.json first,
    falling back to core.features to ensure all intervals in TIMEFRAME_CONFIG
    (1, 5, 15, 30, 60, 120, 240, 360, D) return valid, consistent feature lists without throwing KeyError.
    """
    iv_str = str(interval).replace("m", "").replace("M", "")
    manifest_filename = f"ensemble_trending_trend_{iv_str}_manifest.json"
    selected_features_filename = f"selected_features_{iv_str}.json"
    
    if os.path.exists(manifest_filename):
        try:
            with open(manifest_filename, "r") as f:
                m_data = json.load(f)
                feat_list = m_data.get("feature_names") or m_data.get("features")
                if feat_list and isinstance(feat_list, list) and len(feat_list) >= 10:
                    return list(feat_list)
        except (IOError, OSError, json.JSONDecodeError):
            pass

    if os.path.exists(selected_features_filename):
        try:
            with open(selected_features_filename, "r") as f:
                feat_list = json.load(f)
                if feat_list and isinstance(feat_list, list) and len(feat_list) >= 10:
                    return list(feat_list)
        except (IOError, OSError, json.JSONDecodeError):
            pass
            
    return list(features)

def add_features(df, fetch_calendar_callback=None, symbol=None, interval=None):
    return features_module.add_features(df, fetch_calendar_callback=fetch_calendar_callback, symbol=symbol, interval=interval)

def calibrate_confidence(raw_conf, eps=1e-3):
    """
    Preserves true calibrated probability output from ensemble classifier
    clipped away from 0.0 and 1.0 saturation boundary (EPS = 1e-3).
    """
    return float(np.clip(raw_conf, eps, 1.0 - eps))

def calculate_historical_thresholds(model_trend, interval):
    if model_trend is None:
        return 0.55, 0.75
    print(f"Fetching historical data to calibrate confidence percentiles (last 5,000 candles for {SYMBOL} + BTCUSDT on {interval}m interval)...")
    try:
        from data import get_history, merge_derivatives_sentiment_features
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=5)
        df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=5)
        
        if df_target is not None and len(df_target) > 0 and df_btc is not None and len(df_btc) > 0:
            df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
            df = pd.merge(df_target, df_btc_sub, on="timestamp", how="left")
            df["close_btc"] = df["close_btc"].ffill().bfill().fillna(df["close"])

            if len(df) > 0:
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=interval)
                df = add_features(df, symbol=SYMBOL, interval=interval)
                
                selected_features_list = None
                manifest_filename = f"ensemble_trending_trend_{interval}_manifest.json"
                selected_features_filename = f"selected_features_{interval}.json"
                
                if os.path.exists(manifest_filename):
                    try:
                        with open(manifest_filename, "r") as f:
                            m_data = json.load(f)
                            selected_features_list = m_data.get("feature_names") or m_data.get("features")
                    except Exception as ex_manifest:
                        from logger import log_event
                        log_event("WARNING", f"Manifest load notice: {ex_manifest}")
                
                if not selected_features_list and os.path.exists(selected_features_filename):
                    try:
                        with open(selected_features_filename, "r") as f:
                            selected_features_list = json.load(f)
                    except Exception as ex_feat_file:
                        from logger import log_event
                        log_event("WARNING", f"Selected features file load notice: {ex_feat_file}")
                
                model_feats = getattr(model_trend, "feature_names", None)
                if model_feats:
                    feat_cols = [col for col in model_feats if col in df.columns]
                elif selected_features_list:
                    feat_cols = [col for col in selected_features_list if col in df.columns]
                else:
                    feat_cols = [c for c in features if c in df.columns]
                            
                from ensemble import _slice_model_input
                X_hist = _slice_model_input(model_trend, df[feat_cols])
                probs = model_trend.predict_proba(X_hist)
                confidences = np.max(probs, axis=1)
                
                p95 = float(np.percentile(confidences, 95))
                max_conf = float(np.max(confidences))
                mean_conf = float(np.mean(confidences))
                
                print(f"Confidence Calibration Done for {interval}m:")
                print(f"  - Historical Mean: {mean_conf*100:.2f}%")
                print(f"  - 95th Percentile Threshold (Maps to 80%): {p95*100:.2f}%")
                print(f"  - Maximum Confidence (Maps to 100%): {max_conf*100:.2f}%")
                return p95, max_conf
    except Exception as e:
        from logger import log_event
        log_event("WARNING", f"[Calibration Fallback Engaged] Error calculating calibration for {interval}m: {e}. Defaulting to baseline thresholds (0.55, 0.75).")
    
    return 0.55, 0.75


def generate_triple_barrier_labels(df: pd.DataFrame, interval: str = "60") -> pd.Series:
    """
    Generates Triple-Barrier target labels for ML training based on exact SL/TP barrier hits.
    - Label 2 (Bullish Win): TP hit before SL for Long
    - Label 0 (Bearish Win): TP hit before SL for Short
    - Label 1 (Neutral): Neither barrier hit or SL hit before TP
    """
    cfg = TIMEFRAME_CONFIG.get(str(interval), TIMEFRAME_CONFIG["60"])
    lookahead = cfg["lookahead"]
    sl_mult = cfg["sl_mult"]
    tp_mult = cfg["tp_mult_trending"]

    n = len(df)
    prices = df["close"].values
    highs = df["high"].values if "high" in df.columns else prices
    lows = df["low"].values if "low" in df.columns else prices
    atrs = (df["ATR_norm"] * df["close"]).values if "ATR_norm" in df.columns else (df["close"] * 0.015).values
    
    labels = np.ones(n, dtype=int)
    outcomes = np.full(n, "timed_out", dtype=object)
    ambiguous_count = 0

    for i in range(n - lookahead):
        p_entry = prices[i]
        atr_d = atrs[i]
        
        long_tp = p_entry + tp_mult * atr_d
        long_sl = p_entry - sl_mult * atr_d
        
        short_tp = p_entry - tp_mult * atr_d
        short_sl = p_entry + sl_mult * atr_d
        
        long_won = False
        short_won = False
        long_stopped = False
        short_stopped = False

        for k in range(1, lookahead + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]
            if (h_k >= long_tp and l_k <= long_sl) or (l_k <= short_tp and h_k >= short_sl):
                ambiguous_count += 1

            if not long_won and not long_stopped:
                if l_k <= long_sl:
                    long_stopped = True
                    break
                if h_k >= long_tp:
                    long_won = True
                    break

        for k in range(1, lookahead + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]

            if not short_won and not short_stopped:
                if h_k >= short_sl:
                    short_stopped = True
                    break
                if l_k <= short_tp:
                    short_won = True
                    break

        if long_won and not short_won:
            labels[i] = 2
            outcomes[i] = "long_win"
        elif short_won and not long_won:
            labels[i] = 0
            outcomes[i] = "short_win"
        elif long_stopped or short_stopped:
            labels[i] = 1
            outcomes[i] = "stopped"
        else:
            labels[i] = 1
            outcomes[i] = "timed_out"

    ambiguous_bar_pct = round((ambiguous_count / max(1, n * lookahead)) * 100.0, 4)
    s = pd.Series(labels, index=df.index)
    s.attrs["ambiguous_bar_pct"] = ambiguous_bar_pct
    s.attrs["neutral_pct"] = round(float(np.mean(labels == 1)) * 100.0, 2)
    valid_n = max(1, n - lookahead)
    s.attrs["outcome_breakdown"] = {
        "long_win_pct": round(float(np.sum(outcomes == "long_win") / valid_n) * 100.0, 2),
        "short_win_pct": round(float(np.sum(outcomes == "short_win") / valid_n) * 100.0, 2),
        "timed_out_pct": round(float(np.sum(outcomes == "timed_out") / valid_n) * 100.0, 2),
        "stopped_out_pct": round(float(np.sum(outcomes == "stopped") / valid_n) * 100.0, 2)
    }
    return s


def compute_sample_uniqueness(t1: pd.Series, close_idx: pd.Index) -> pd.Series:
    """
    Computes average sample uniqueness per observation to weight overlapping labels (AFML Ch. 4).
    Prevents overstating effective sample size.
    """
    count = pd.Series(0.0, index=close_idx)
    for start, end in t1.items():
        if pd.isna(end):
            continue
        try:
            count.loc[start:end] += 1.0
        except Exception:
            pass

    uniqueness = {}
    for start, end in t1.items():
        if pd.isna(end):
            uniqueness[start] = 1.0
        else:
            try:
                c_slice = count.loc[start:end]
                uniqueness[start] = float((1.0 / np.maximum(1.0, c_slice)).mean())
            except Exception:
                uniqueness[start] = 1.0
    return pd.Series(uniqueness, index=t1.index).reindex(close_idx).fillna(1.0)


def compute_balanced_uniqueness_weights(
    y: pd.Series,
    uniqueness: pd.Series | np.ndarray | None = None,
    decay: pd.Series | np.ndarray | None = None
) -> np.ndarray:
    """
    Composes sample uniqueness, recency decay, and per-class balance weighting (AFML Ch. 4).
    Re-normalizes weights per class so that total weight per class is equal,
    preventing decay/uniqueness multiplication from destroying class equality (H-1).
    Asserts max(class_totals) / min(class_totals) < 1.05.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    base = np.ones(n, dtype=float)
    if uniqueness is not None:
        base *= np.asarray(uniqueness, dtype=float)
    if decay is not None:
        base *= np.asarray(decay, dtype=float)

    w = base.copy()
    classes = np.unique(y_arr)
    n_classes = len(classes)
    for c in classes:
        mask = (y_arr == c)
        c_sum = base[mask].sum()
        if c_sum > 0:
            w[mask] *= (n / (n_classes * c_sum))

    class_totals = [w[y_arr == c].sum() for c in classes]
    if len(class_totals) > 1 and min(class_totals) > 0:
        ratio = max(class_totals) / min(class_totals)
        assert ratio < 1.05, f"Class balance assertion broken: totals={class_totals}, ratio={ratio:.4f}"
    return w

