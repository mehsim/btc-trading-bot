import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = 15
N_WINDOWS = 15

print("Loading dataset...")
dfs = []
for s in SUPPORTED_SYMBOLS:
    df_s = get_history(symbol=s, interval=INTERVAL, limit=1000, pages=3)
    if df_s is not None and len(df_s) > 100:
        df_s["symbol"] = s
        df_s["close_btc"] = df_s["close"]
        df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=INTERVAL)
        df_s = add_features(df_s)
        dfs.append(df_s)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

model_ranging = load_ensemble_classifier("ensemble_ranging_trend_15")
model_trending = load_ensemble_classifier("ensemble_trending_trend_15")

with open("selected_features_15_ranging.json") as f:
    feats_ranging = json.load(f)
with open("selected_features_15_trending.json") as f:
    feats_trending = json.load(f)

probs_ranging = model_ranging.predict_proba(df_all[feats_ranging].values)
probs_trending = model_trending.predict_proba(df_all[feats_trending].values)
adxs = df_all["ADX"].values
probs_all = np.where(adxs[:, None] >= 25.0, probs_trending, probs_ranging)

total_bars = len(df_all)
window_size = total_bars // N_WINDOWS

w = 1  # Window 2 (0-indexed w=1)
w_start = w * window_size
w_end = (w + 1) * window_size

closes = df_all["close"].values
highs = df_all["high"].values
lows = df_all["low"].values
atr_norms = df_all["ATR_norm"].values
symbols = df_all["symbol"].values
timestamps = df_all["timestamp"].values

trades = []
active_until_ts = {}

for i in range(w_start, w_end - 12):
    sym = symbols[i]
    ts = timestamps[i]
    
    p_bear = probs_all[i, 0]
    p_bull = probs_all[i, 2]
    dir_total = p_bear + p_bull
    if dir_total < 0.15:
        continue
    
    norm_bear = p_bear / dir_total
    norm_bull = p_bull / dir_total
    
    if norm_bull >= 0.2862:
        is_bull = True
    elif norm_bear >= 0.2862:
        is_bull = False
    else:
        continue
    
    if ts < active_until_ts.get(sym, 0):
        continue
        
    p_entry = closes[i]
    atr = atr_norms[i] * p_entry
    if atr <= 0:
        continue
    
    upper_tp = p_entry + 2.5 * atr
    lower_sl = p_entry - 1.0 * atr
    lower_tp = p_entry - 2.5 * atr
    upper_sl = p_entry + 1.0 * atr
    
    for step in range(1, 13):
        h = highs[i + step]
        l = lows[i + step]
        exit_ts = timestamps[i + step]
        if is_bull:
            if h >= upper_tp or l <= lower_sl:
                active_until_ts[sym] = exit_ts
                trades.append({"entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "symbol": sym, "direction": "BULL", "bars_held": step})
                break
        else:
            if l <= lower_tp or h >= upper_sl:
                active_until_ts[sym] = exit_ts
                trades.append({"entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "symbol": sym, "direction": "BEAR", "bars_held": step})
                break

print("=== WINDOW 2: FIRST 30 TRADES ===")
for t in trades[:30]:
    print(f"{t['entry_ts']} | {t['symbol']:9} | {t['direction']:5} | exit {t['exit_ts']} | {t['bars_held']} bars")
