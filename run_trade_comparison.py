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
FEE_RATE = 0.0008

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

closes = df_all["close"].values
highs = df_all["high"].values
lows = df_all["low"].values
atr_norms = df_all["ATR_norm"].values
timestamps = df_all["timestamp"].values
symbols = df_all["symbol"].values

def simulate_run(tp_mult, sl_mult, use_min_floor=False):
    trades = []
    eval_thresh = 0.2862 if use_min_floor else 0.35
    for i in range(len(df_all) - 12):
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
            
        upper_tp = p_entry + tp_mult * atr
        lower_sl = p_entry - sl_mult * atr
        lower_tp = p_entry - tp_mult * atr
        upper_sl = p_entry + sl_mult * atr
        
        for step in range(1, 13):
            h = highs[i + step]
            l = lows[i + step]
            if is_bull:
                if h >= upper_tp:
                    pnl = (upper_tp - p_entry) / p_entry - 2 * FEE_RATE
                    trades.append({"bar": i, "ts": pd.to_datetime(timestamps[i], unit="ms"), "symbol": symbols[i], "dir": "BULL", "entry": p_entry, "tp": upper_tp, "sl": lower_sl, "pnl": pnl, "outcome": "TP", "bars_held": step})
                    break
                elif l <= lower_sl:
                    pnl = (lower_sl - p_entry) / p_entry - 2 * FEE_RATE
                    trades.append({"bar": i, "ts": pd.to_datetime(timestamps[i], unit="ms"), "symbol": symbols[i], "dir": "BULL", "entry": p_entry, "tp": upper_tp, "sl": lower_sl, "pnl": pnl, "outcome": "SL", "bars_held": step})
                    break
            else:
                if l <= lower_tp:
                    pnl = (p_entry - lower_tp) / p_entry - 2 * FEE_RATE
                    trades.append({"bar": i, "ts": pd.to_datetime(timestamps[i], unit="ms"), "symbol": symbols[i], "dir": "BEAR", "entry": p_entry, "tp": lower_tp, "sl": upper_sl, "pnl": pnl, "outcome": "TP", "bars_held": step})
                    break
                elif h >= upper_sl:
                    pnl = (p_entry - upper_sl) / p_entry - 2 * FEE_RATE
                    trades.append({"bar": i, "ts": pd.to_datetime(timestamps[i], unit="ms"), "symbol": symbols[i], "dir": "BEAR", "entry": p_entry, "tp": lower_tp, "sl": upper_sl, "pnl": pnl, "outcome": "SL", "bars_held": step})
                    break
    return trades

trades_run1 = simulate_run(0.98, 0.78, False)
trades_run2 = simulate_run(2.50, 1.00, True)

print("=" * 88)
print("FIRST 20 TRADES SIDE-BY-SIDE COMPARISON: RUN 1 (0.98/0.78) VS RUN 2 (2.50/1.00 MIN_FLOOR)")
print("=" * 88)
print(f"{'#':<2} | {'Timestamp':<16} | {'Symbol':<7} | {'Dir':<4} | {'Entry Price':<11} | {'Run 1 Outcome (0.98/0.78)':<24} | {'Run 2 Outcome (2.50/1.00)':<24}")
print("-" * 88)

n_compare = min(20, len(trades_run1), len(trades_run2))
for k in range(n_compare):
    t1 = trades_run1[k]
    t2 = trades_run2[k]
    ts_str = str(t1['ts'])[:16]
    r1_str = f"{t1['outcome']} ({t1['pnl']*100:+.2f}%) [{t1['bars_held']}b]"
    r2_str = f"{t2['outcome']} ({t2['pnl']*100:+.2f}%) [{t2['bars_held']}b]"
    print(f"{k+1:02d} | {ts_str} | {t1['symbol']:<7} | {t1['dir']:<4} | ${t1['entry']:<10.2f} | {r1_str:<24} | {r2_str:<24}")
