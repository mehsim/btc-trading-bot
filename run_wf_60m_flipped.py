import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from main import load_model_weights, models_by_interval
from ensemble import get_model_feature_names, _slice_model_input

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = 60
FEE_RATE = 0.0004  # 0.04% per leg (0.08% round-trip)
TP_MULT = 2.50
SL_MULT = 1.50
LOOKAHEAD = 12
N_WINDOWS = 15
eval_threshold = 0.52

print(f"=== 60M FLIPPED DIRECTION (BULLISH <-> BEARISH) WALK-FORWARD BACKTEST (TP={TP_MULT}, SL={SL_MULT}, CONF={eval_threshold}) ===")

# Load active 60m models via main loader
load_model_weights("60")
load_model_weights(60)
models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}

model_trending = models_60.get("trending", {}).get("trend") or models_60.get("ranging", {}).get("trend")
model_ranging = models_60.get("ranging", {}).get("trend") or model_trending

feat_trending = models_60.get("selected_features_trending") or models_60.get("selected_features_ranging") or models_60.get("selected_features")
feat_ranging = models_60.get("selected_features_ranging") or feat_trending

symbol_dfs = {}
print("Loading 60m candle history across 9 symbols...")
for s in SUPPORTED_SYMBOLS:
    df_s = get_history(symbol=s, interval=INTERVAL, limit=1000, pages=3)
    if df_s is not None and len(df_s) > 100:
        df_s["symbol"] = s
        df_s["close_btc"] = df_s["close"]
        df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=INTERVAL)
        df_s = add_features(df_s)
        
        # Prepare trending feature matrix
        _exp_names_t = get_model_feature_names(model_trending) if model_trending else None
        if _exp_names_t and not all(str(n).startswith("Column_") for n in _exp_names_t):
            X_full_t = df_s.reindex(columns=_exp_names_t, fill_value=0.0)
        elif feat_trending:
            _avail_t = [f for f in feat_trending if f in df_s.columns]
            X_full_t = df_s[_avail_t]
        else:
            from core import features as master_features
            _avail_t = [f for f in master_features if f in df_s.columns]
            X_full_t = df_s[_avail_t]

        # Prepare ranging feature matrix
        _exp_names_r = get_model_feature_names(model_ranging) if model_ranging else None
        if _exp_names_r and not all(str(n).startswith("Column_") for n in _exp_names_r):
            X_full_r = df_s.reindex(columns=_exp_names_r, fill_value=0.0)
        elif feat_ranging:
            _avail_r = [f for f in feat_ranging if f in df_s.columns]
            X_full_r = df_s[_avail_r]
        else:
            from core import features as master_features
            _avail_r = [f for f in master_features if f in df_s.columns]
            X_full_r = df_s[_avail_r]

        # Handle positional feature shape match if model expects 22 features
        if hasattr(model_trending, "n_features_in_") and X_full_t.shape[1] != model_trending.n_features_in_:
            X_full_t = X_full_t.iloc[:, :model_trending.n_features_in_]
        if hasattr(model_ranging, "n_features_in_") and X_full_r.shape[1] != model_ranging.n_features_in_:
            X_full_r = X_full_r.iloc[:, :model_ranging.n_features_in_]

        X_trend = _slice_model_input(model_trending, X_full_t)
        X_rang = _slice_model_input(model_ranging, X_full_r)

        p_trend = model_trending.predict_proba(X_trend)
        p_rang = model_ranging.predict_proba(X_rang)

        adxs = df_s["ADX"].values
        df_s["p_bear"] = np.where(adxs >= 25.0, p_trend[:, 0], p_rang[:, 0])
        df_s["p_neut"] = np.where(adxs >= 25.0, p_trend[:, 1], p_rang[:, 1])
        df_s["p_bull"] = np.where(adxs >= 25.0, p_trend[:, 2], p_rang[:, 2])
        
        symbol_dfs[s] = df_s

# Combine into single timeline sorted by timestamp
all_rows = []
for s, df_s in symbol_dfs.items():
    for idx, row in df_s.iterrows():
        all_rows.append({
            "symbol": s,
            "sym_idx": idx,
            "timestamp": row["timestamp"],
            "p_bear": row["p_bear"],
            "p_neut": row["p_neut"],
            "p_bull": row["p_bull"],
            "close": row["close"],
            "ATR_norm": row["ATR_norm"]
        })

df_all = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
total_bars = len(df_all)
window_size = total_bars // N_WINDOWS
window_metrics = []

print("\n--- 60M FLIPPED PER-WINDOW PERFORMANCE BREAKDOWN ---")

for w in range(N_WINDOWS):
    w_start = w * window_size
    w_end = (w + 1) * window_size if w < N_WINDOWS - 1 else total_bars
    
    trades = []
    skipped = 0
    active_until_ts = {}
    
    for i in range(w_start, w_end):
        sym = df_all.loc[i, "symbol"]
        ts = df_all.loc[i, "timestamp"]
        
        p_bear = df_all.loc[i, "p_bear"]
        p_neut = df_all.loc[i, "p_neut"]
        p_bull = df_all.loc[i, "p_bull"]
        
        # FLIPPED DIRECTION ASSIGNMENT:
        if p_bull >= eval_threshold and p_bull > max(p_bear, p_neut):
            is_bull = False  # FLIPPED!
        elif p_bear >= eval_threshold and p_bear > max(p_bull, p_neut):
            is_bull = True   # FLIPPED!
        else:
            continue
            
        # LIVE GUARD: Skip if symbol position is active until future exit timestamp
        if ts < active_until_ts.get(sym, 0):
            skipped += 1
            continue
            
        # Perform exit evaluation on THIS SYMBOL'S OWN 60m candle series
        df_s = symbol_dfs[sym]
        s_idx = df_all.loc[i, "sym_idx"]
        
        if s_idx + LOOKAHEAD >= len(df_s):
            continue
            
        p_entry = df_all.loc[i, "close"]
        atr = df_all.loc[i, "ATR_norm"] * p_entry
        if atr <= 0:
            continue
            
        upper_tp = p_entry + TP_MULT * atr
        lower_sl = p_entry - SL_MULT * atr
        lower_tp = p_entry - TP_MULT * atr
        upper_sl = p_entry + SL_MULT * atr
        
        pnl = None
        outcome = None
        for step in range(1, LOOKAHEAD + 1):
            h = df_s.loc[s_idx + step, "high"]
            l = df_s.loc[s_idx + step, "low"]
            exit_ts = df_s.loc[s_idx + step, "timestamp"]
            
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
            trades.append({
                "entry_ts": pd.to_datetime(ts, unit="ms"),
                "exit_ts": pd.to_datetime(active_until_ts[sym], unit="ms"),
                "symbol": sym,
                "direction": "BULL" if is_bull else "BEAR",
                "bars_held": step,
                "pnl": pnl,
                "win": pnl > 0
            })

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
print(f"   60M FLIPPED DIRECTION WALK-FORWARD RESULTS SUMMARY")
print(f"===================================================================================")
print(f"Total Out-Of-Sample Windows : {N_WINDOWS}")
print(f"Total Signals Generated     : {total_trades + total_skipped}")
print(f"Total Trades Taken          : {total_trades}")
print(f"Total Signals Skipped       : {total_skipped} ({total_skipped/max(1, total_trades+total_skipped):.2%})")
print(f"Mean Out-Of-Sample Win Rate : {mean_wr:.2f}% (Required Break-Even: 29.1%)")
print(f"Mean Profit Factor          : {mean_pf:.3f}")
print(f"Median Trade Return         : {overall_med_ret:+.3f}%")
print(f"Profitable Windows Count    : {profitable_windows} / {N_WINDOWS}")
print(f"===================================================================================")
