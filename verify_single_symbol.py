import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

symbol = "BTCUSDT"
df_s = get_history(symbol=symbol, interval=15, limit=1000, pages=3)
df_s["symbol"] = symbol
df_s["close_btc"] = df_s["close"]
df_s = merge_derivatives_sentiment_features(df_s, symbol=symbol, interval=15)
df_s = add_features(df_s)

model_ranging = load_ensemble_classifier("ensemble_ranging_trend_15")
model_trending = load_ensemble_classifier("ensemble_trending_trend_15")

with open("selected_features_15_ranging.json") as f:
    feats_ranging = json.load(f)
with open("selected_features_15_trending.json") as f:
    feats_trending = json.load(f)

p_rang = model_ranging.predict_proba(df_s[feats_ranging].values)
p_trend = model_trending.predict_proba(df_s[feats_trending].values)
adxs = df_s["ADX"].values
df_s["p_bear"] = np.where(adxs >= 25.0, p_trend[:, 0], p_rang[:, 0])
df_s["p_bull"] = np.where(adxs >= 25.0, p_trend[:, 2], p_rang[:, 2])

trades = []
active_until_ts = 0

for i in range(len(df_s) - 12):
    ts = df_s.loc[i, "timestamp"]
    if ts < active_until_ts:
        continue
        
    p_bear = df_s.loc[i, "p_bear"]
    p_bull = df_s.loc[i, "p_bull"]
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
    
    p_entry = df_s.loc[i, "close"]
    atr = df_s.loc[i, "ATR_norm"] * p_entry
    if atr <= 0:
        continue
    
    upper_tp = p_entry + 2.5 * atr
    lower_sl = p_entry - 1.0 * atr
    lower_tp = p_entry - 2.5 * atr
    upper_sl = p_entry + 1.0 * atr
    
    for step in range(1, 13):
        h = df_s.loc[i + step, "high"]
        l = df_s.loc[i + step, "low"]
        exit_ts = df_s.loc[i + step, "timestamp"]
        
        if is_bull:
            if h >= upper_tp or l <= lower_sl:
                active_until_ts = exit_ts
                trades.append({"symbol": symbol, "entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "bars_held": step})
                break
        else:
            if l <= lower_tp or h >= upper_sl:
                active_until_ts = exit_ts
                trades.append({"symbol": symbol, "entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "bars_held": step})
                break

print("=== BTCUSDT SINGLE SYMBOL CONSECUTIVE TRADES VERIFICATION ===")
for t in trades[:15]:
    print(f"{t['symbol']} | Entry: {t['entry_ts']} -> Exit: {t['exit_ts']} | Bars: {t['bars_held']:2d}")
