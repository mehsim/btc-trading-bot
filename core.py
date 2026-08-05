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
    "RSI_z", "ADX_z", "close_to_Kalman"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

def add_features(df, fetch_calendar_callback=None):
    return features_module.add_features(df, fetch_calendar_callback=fetch_calendar_callback)

def calibrate_confidence(raw_conf, p95=0.55, max_conf=0.75):
    """
    Preserves true calibrated probability output from ensemble classifier
    without ad-hoc piecewise linear stretching (Fix B12).
    """
    return float(np.clip(raw_conf, 0.0, 1.0))

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
                df = add_features(df)
                
                selected_features_list = None
                selected_features_filename = f"selected_features_{interval}.json"
                if os.path.exists(selected_features_filename):
                    with open(selected_features_filename, "r") as f:
                        selected_features_list = json.load(f)
                            
                from ensemble import _slice_model_input
                X_hist = _slice_model_input(model_trend, df[features])
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
        print(f"Error calculating calibration for {interval}m: {e}. Using defaults.")
    
    return 0.55, 0.75


def generate_triple_barrier_labels(df: pd.DataFrame, interval: str = "60") -> pd.Series:
    """
    Generates Triple-Barrier target labels for ML training based on exact SL/TP barrier hits.
    - Label 2 (Bullish Win): TP hit before SL for Long
    - Label 0 (Bearish Win): TP hit before SL for Short
    - Label 1 (Neutral): Neither barrier hit or SL hit before TP
    """
    cfg = TIMEFRAME_CONFIG.get(str(interval), {"lookahead": 10, "sl_mult": 1.5, "tp_mult_ranging": 1.5, "tp_mult_trending": 2.5})
    lookahead = cfg.get("lookahead", 10)
    sl_mult = cfg.get("sl_mult", 1.5)
    tp_mult = cfg.get("tp_mult_trending", 2.5)

    labels = np.full(len(df), 1, dtype=int)
    prices = df["close"].values
    highs = df["high"].values if "high" in df.columns else prices
    lows = df["low"].values if "low" in df.columns else prices
    atrs = (df["ATR_norm"] * df["close"]).values if "ATR_norm" in df.columns else (df["close"] * 0.015).values

    n = len(df)
    for i in range(n - lookahead):
        p_entry = prices[i]
        atr_d = atrs[i]
        
        long_tp = p_entry + tp_mult * atr_d
        long_sl = p_entry - sl_mult * atr_d
        
        short_tp = p_entry - tp_mult * atr_d
        short_sl = p_entry + sl_mult * atr_d
        
        long_won = False
        short_won = False

        ambiguous_count = 0
        for k in range(1, lookahead + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]
            if (h_k >= long_tp and l_k <= long_sl) or (l_k <= short_tp and h_k >= short_sl):
                ambiguous_count += 1

            if not long_won:
                if l_k <= long_sl:
                    break
                if h_k >= long_tp:
                    long_won = True
                    break

        for k in range(1, lookahead + 1):
            h_k = highs[i + k]
            l_k = lows[i + k]

            if not short_won:
                if h_k >= short_sl:
                    break
                if l_k <= short_tp:
                    short_won = True
                    break

        if long_won and not short_won:
            labels[i] = 2
        elif short_won and not long_won:
            labels[i] = 0
        else:
            labels[i] = 1

    ambiguous_bar_pct = round((ambiguous_count / max(1, n * lookahead)) * 100.0, 4)
    s = pd.Series(labels, index=df.index)
    s.attrs["ambiguous_bar_pct"] = ambiguous_bar_pct
    s.attrs["neutral_pct"] = round(float(np.mean(labels == 1)) * 100.0, 2)
    s.attrs["outcome_breakdown"] = {
        "long_win_pct": round(float(np.mean(labels == 2)) * 100.0, 2),
        "short_win_pct": round(float(np.mean(labels == 0)) * 100.0, 2),
        "neutral_or_stopped_pct": round(float(np.mean(labels == 1)) * 100.0, 2)
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
