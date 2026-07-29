import os
import json
import numpy as np
import pandas as pd
import features as features_module

SYMBOL = "BTCUSDT"
INTERVAL = "60"

TIMEFRAME_CONFIG = {
    "15": {   # 15M Timeframe
        "lookahead": 12,
        "sl_mult": 0.75,
        "tp_mult_ranging": 1.2,
        "tp_mult_trending": 1.3
    },
    "30": {   # 30M Timeframe
        "lookahead": 12,
        "sl_mult": 0.80,
        "tp_mult_ranging": 1.3,
        "tp_mult_trending": 1.5
    },
    "60": {
        "lookahead": 10,
        "sl_mult": 1.5,
        "tp_mult_ranging": 1.5,
        "tp_mult_trending": 2.5
    },
    "120": {
        "lookahead": 12,
        "sl_mult": 1.5,
        "tp_mult_ranging": 1.4,
        "tp_mult_trending": 2.2
    },
    "240": {
        "lookahead": 12,
        "sl_mult": 1.8,
        "tp_mult_ranging": 1.3,
        "tp_mult_trending": 2.0
    },
    "360": {
        "lookahead": 16,
        "sl_mult": 2.0,
        "tp_mult_ranging": 1.2,
        "tp_mult_trending": 1.8
    }
}

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

def calibrate_confidence(raw_conf, p95, max_conf):
    if max_conf <= p95:
        max_conf = p95 + 0.01
    if p95 <= 0.33:
        p95 = 0.34
        
    if raw_conf < p95:
        calibrated = 50.0 + (raw_conf - 0.33) / (p95 - 0.33) * 30.0
    else:
        calibrated = 80.0 + (raw_conf - p95) / (max_conf - p95) * 20.0
        
    return min(100.0, max(50.0, calibrated)) / 100.0

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
                X_hist = _slice_model_input(model_trend, df[features].values)
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
