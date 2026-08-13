import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = 15
FEE_RATE = 0.0008  # 0.08% fee per leg (0.16% round-trip)
TP_MULT = 2.5
SL_MULT = 1.0
N_WINDOWS = 15
MIN_EVAL_THRESHOLD_FLOOR = 0.25

p_star = SL_MULT / (TP_MULT + SL_MULT)  # 1.0 / 3.5 = 0.2857
cost_bps = 16.0
eval_threshold = round(min(0.52, max(MIN_EVAL_THRESHOLD_FLOOR, p_star + (cost_bps / 1e4) / (TP_MULT + SL_MULT))), 4)

print(f"=== FULL 15-WINDOW WALK-FORWARD: TAKEN VS SKIPPED SIGNALS COUNT ===")

print("Loading 15m candle history across 9 symbols...")
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
total_bars = len(df_all)

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

window_size = total_bars // N_WINDOWS
window_metrics = []

closes = df_all["close"].values
highs = df_all["high"].values
lows = df_all["low"].values
atr_norms = df_all["ATR_norm"].values
symbols = df_all["symbol"].values
timestamps = df_all["timestamp"].values

print("\n--- PER-WINDOW TAKEN VS SKIPPED SIGNALS BREAKDOWN ---")

for w in range(N_WINDOWS):
    w_start = w * window_size
    w_end = (w + 1) * window_size if w < N_WINDOWS - 1 else total_bars
    
    trades = []
    skipped = 0
    active_until_ts = {}  # Tracks per-symbol position exit timestamp (ms)
    
    for i in range(w_start, w_end - 12):
        sym = symbols[i]
        ts = timestamps[i]
        
        # Check model directional signal first
        p_bear = probs_all[i, 0]
        p_bull = probs_all[i, 2]
        dir_total = p_bear + p_bull
        
        if dir_total < 0.15:
            continue
            
        norm_bear = p_bear / max(1e-9, dir_total)
        norm_bull = p_bull / max(1e-9, dir_total)
        
        if norm_bull >= eval_threshold:
            is_bull = True
        elif norm_bear >= eval_threshold:
            is_bull = False
        else:
            continue
            
        # ACCURATE LIVE CONSTRAINT: Count and skip if symbol position is active until future exit timestamp
        if ts < active_until_ts.get(sym, 0):
            skipped += 1
            continue
            
        p_entry = closes[i]
        atr = atr_norms[i] * p_entry
        if atr <= 0:
            continue
        
        upper_tp = p_entry + TP_MULT * atr
        lower_sl = p_entry - SL_MULT * atr
        lower_tp = p_entry - TP_MULT * atr
        upper_sl = p_entry + SL_MULT * atr
        
        pnl = None
        outcome = None
        for step in range(1, 13):
            h = highs[i + step]
            l = lows[i + step]
            exit_ts = timestamps[i + step]
            if is_bull:
                if h >= upper_tp:
                    pnl = (upper_tp - p_entry) / p_entry - 2 * FEE_RATE
                    outcome = "TP"
                    active_until_ts[sym] = exit_ts
                    break
                elif l <= lower_sl:
                    pnl = (lower_sl - p_entry) / p_entry - 2 * FEE_RATE
                    outcome = "SL"
                    active_until_ts[sym] = exit_ts
                    break
            else:
                if l <= lower_tp:
                    pnl = (p_entry - lower_tp) / p_entry - 2 * FEE_RATE
                    outcome = "TP"
                    active_until_ts[sym] = exit_ts
                    break
                elif h >= upper_sl:
                    pnl = (p_entry - upper_sl) / p_entry - 2 * FEE_RATE
                    outcome = "SL"
                    active_until_ts[sym] = exit_ts
                    break
        if pnl is not None:
            trades.append({"pnl": pnl, "win": pnl > 0, "outcome": outcome})
            
    n_t = len(trades)
    win_rate = (sum(t["win"] for t in trades) / n_t * 100) if n_t > 0 else 0.0
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
    pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else 0.0
    trade_pnls = [t["pnl"] * 100 for t in trades] if n_t > 0 else [0.0]
    mean_ret = np.mean(trade_pnls) if n_t > 0 else 0.0
    med_ret = float(np.median(trade_pnls)) if n_t > 0 else 0.0
    
    print(f"window {w+1:02d}: {n_t} taken, {skipped} skipped | WinRate={win_rate:.2f}% PF={pf:.3f}")
    
    window_metrics.append({
        "window": w + 1,
        "trades": n_t,
        "skipped": skipped,
        "win_rate": round(float(win_rate), 2),
        "profit_factor": round(float(pf), 3),
        "mean_return_pct": round(float(mean_ret), 3),
        "median_return_pct": round(float(med_ret), 3),
        "is_profitable": bool(pf > 1.0)
    })

total_trades = sum(w["trades"] for w in window_metrics)
total_skipped = sum(w["skipped"] for w in window_metrics)
mean_wr = np.mean([w["win_rate"] for w in window_metrics if w["trades"] > 0])
mean_pf = np.mean([w["profit_factor"] for w in window_metrics if w["trades"] > 0])
overall_med_ret = np.median([w["median_return_pct"] for w in window_metrics if w["trades"] > 0])
profitable_windows = sum(1 for w in window_metrics if w["is_profitable"])

print(f"\n===================================================================================")
print(f"   SUMMARY REPORT WITH TAKEN VS SKIPPED SIGNALS")
print(f"===================================================================================")
print(f"Total Out-Of-Sample Windows : {N_WINDOWS}")
print(f"Total Signals Generated     : {total_trades + total_skipped}")
print(f"Total Trades Taken          : {total_trades}")
print(f"Total Signals Skipped       : {total_skipped} ({total_skipped/(total_trades+total_skipped):.2%})")
print(f"Mean Out-Of-Sample Win Rate : {mean_wr:.2f}% (Required Break-Even: 41.6%)")
print(f"Mean Profit Factor          : {mean_pf:.3f}")
print(f"Median Trade Return         : {overall_med_ret:+.3f}%")
print(f"Profitable Windows Count    : {profitable_windows} / {N_WINDOWS}")
print(f"===================================================================================")
