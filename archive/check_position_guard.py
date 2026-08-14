import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

df = get_history(symbol="BTCUSDT", interval=15, limit=1000, pages=3)
df["symbol"] = "BTCUSDT"
df["close_btc"] = df["close"]
df = merge_derivatives_sentiment_features(df, symbol="BTCUSDT", interval=15)
df = add_features(df)

model_ranging = load_ensemble_classifier("ensemble_ranging_trend_15")
model_trending = load_ensemble_classifier("ensemble_trending_trend_15")

with open("selected_features_15_ranging.json") as f:
    feats_ranging = json.load(f)
with open("selected_features_15_trending.json") as f:
    feats_trending = json.load(f)

probs_ranging = model_ranging.predict_proba(df[feats_ranging].values)
probs_trending = model_trending.predict_proba(df[feats_trending].values)
adxs = df["ADX"].values

probs_all = np.where(adxs[:, None] >= 25.0, probs_trending, probs_ranging)

closes = df["close"].values
highs = df["high"].values
lows = df["low"].values
atr_norms = df["ATR_norm"].values
timestamps = df["timestamp"].values

def simulate_btc(use_timestamp_guard=True):
    trades = []
    active_until_ts = 0
    eval_thresh = 0.2862
    
    for i in range(len(df) - 12):
        ts = timestamps[i]
        if use_timestamp_guard and ts < active_until_ts:
            continue
            
        p_bear = probs_all[i, 0]
        p_bull = probs_all[i, 2]
        dir_total = p_bear + p_bull
        if dir_total < 0.15:
            continue
        
        norm_bear = p_bear / dir_total
        norm_bull = p_bull / dir_total
        
        if norm_bull >= eval_thresh:
            is_bull = True
        elif norm_bear >= eval_thresh:
            is_bull = False
        else:
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
                    active_until_ts = exit_ts
                    trades.append({"entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "p_entry": p_entry, "bars": step})
                    break
            else:
                if l <= lower_tp or h >= upper_sl:
                    active_until_ts = exit_ts
                    trades.append({"entry_ts": pd.to_datetime(ts, unit="ms"), "exit_ts": pd.to_datetime(exit_ts, unit="ms"), "p_entry": p_entry, "bars": step})
                    break
    return trades

trades_unguarded = simulate_btc(use_timestamp_guard=False)
trades_guarded = simulate_btc(use_timestamp_guard=True)

print(f"BTCUSDT Total Trades UNGUARDED : {len(trades_unguarded)}")
print(f"BTCUSDT Total Trades GUARDED   : {len(trades_guarded)}")

print("\n=== BTCUSDT GUARDED TRADES (2026-07-13 04:00 to 09:00 UTC) ===")
for t in trades_guarded:
    ts_s = str(t["entry_ts"])
    if "2026-07-13" in ts_s and "04:" <= ts_s[11:14] <= "09:":
        print(f"  Entry: {t['entry_ts']} -> Exit: {t['exit_ts']} (Held {t['bars']} bars) | Price: ${t['p_entry']:.2f}")
