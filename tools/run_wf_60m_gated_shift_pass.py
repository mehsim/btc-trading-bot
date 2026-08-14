import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from ta.trend import ADXIndicator
from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from main import load_model_weights, models_by_interval
from ensemble import get_model_feature_names, _slice_model_input

# Step 1 & 5: Out-of-Sample Shifted Validation Pass (ADX_4H >= 22.0, limit=1500, pages=4)
ADX_4H_MIN = 22.0

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = 60
FEE_RATE = 0.0004  # 0.04% per leg (0.08% round-trip)
TP_MULT = 2.50
SL_MULT = 1.50
LOOKAHEAD = 12
N_WINDOWS = 15
eval_threshold = 0.52

print(f"=== 60M 4H-ADX GATED OUT-OF-SAMPLE SHIFT VALIDATION (PAGES=4 / LIMIT=1500) ===")
print(f"Locked ADX_4H_MIN Gate Threshold : {ADX_4H_MIN}")
print(f"TP Multiplier / SL Multiplier     : {TP_MULT} / {SL_MULT}")
print(f"Confidence Threshold             : {eval_threshold}\n")

# Load active 60m models
load_model_weights("60")
load_model_weights(60)
models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}

model_trending = models_60.get("trending", {}).get("trend") or models_60.get("ranging", {}).get("trend")
model_ranging = models_60.get("ranging", {}).get("trend") or model_trending

feat_trending = models_60.get("selected_features_trending") or models_60.get("selected_features_ranging") or models_60.get("selected_features")
feat_ranging = models_60.get("selected_features_ranging") or feat_trending

symbol_dfs = {}
print("Loading 60m shifted candle history (pages=4, limit=1500) across 9 symbols...")
for s in SUPPORTED_SYMBOLS:
    df_s = get_history(symbol=s, interval=INTERVAL, limit=1500, pages=4)
    if df_s is not None and len(df_s) > 100:
        df_s["symbol"] = s
        df_s["close_btc"] = df_s["close"]
        df_s = merge_derivatives_sentiment_features(df_s, symbol=s, interval=INTERVAL)
        df_s = add_features(df_s)
        
        # Calculate 4H ADX via resampling & forward fill
        df_dt = df_s.copy()
        df_dt["dt"] = pd.to_datetime(df_dt["timestamp"], unit="ms", utc=True)
        df_dt.set_index("dt", inplace=True)
        
        df_4h = df_dt.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        if len(df_4h) > 15:
            adx_4h_series = ADXIndicator(df_4h["high"], df_4h["low"], df_4h["close"], window=14).adx()
            df_4h["adx_4h"] = adx_4h_series
            df_dt = df_dt.join(df_4h[["adx_4h"]], how="left")
            df_dt["adx_4h"] = df_dt["adx_4h"].ffill().bfill().fillna(0.0)
            df_s["adx_4h"] = df_dt["adx_4h"].values
        else:
            df_s["adx_4h"] = df_s["ADX"]
        
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
            "adx_4h": row["adx_4h"],
            "close": row["close"],
            "ATR_norm": row["ATR_norm"]
        })

df_all = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
total_bars = len(df_all)
window_size = total_bars // N_WINDOWS

def run_simulation(is_flipped_mode=False):
    window_metrics = []
    total_regime_skipped = 0
    total_pos_skipped = 0
    total_trades_taken = 0
    
    for w in range(N_WINDOWS):
        w_start = w * window_size
        w_end = (w + 1) * window_size if w < N_WINDOWS - 1 else total_bars
        
        trades = []
        skipped_regime = 0
        skipped_pos = 0
        active_until_ts = {}
        
        for i in range(w_start, w_end):
            sym = df_all.loc[i, "symbol"]
            ts = df_all.loc[i, "timestamp"]
            
            p_bear = df_all.loc[i, "p_bear"]
            p_neut = df_all.loc[i, "p_neut"]
            p_bull = df_all.loc[i, "p_bull"]
            adx_4h_val = df_all.loc[i, "adx_4h"]
            
            # 1. Base Confidence Check
            if is_flipped_mode:
                if p_bull >= eval_threshold and p_bull > max(p_bear, p_neut):
                    is_bull = False
                elif p_bear >= eval_threshold and p_bear > max(p_bull, p_neut):
                    is_bull = True
                else:
                    continue
            else:
                if p_bull >= eval_threshold and p_bull > max(p_bear, p_neut):
                    is_bull = True
                elif p_bear >= eval_threshold and p_bear > max(p_bull, p_neut):
                    is_bull = False
                else:
                    continue

            # 2. ADX 4H Regime Gate Check
            if adx_4h_val < ADX_4H_MIN:
                skipped_regime += 1
                continue
                
            # 3. Position Overlap Guard
            if ts < active_until_ts.get(sym, 0):
                skipped_pos += 1
                continue
                
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
            for step in range(1, LOOKAHEAD + 1):
                h = df_s.loc[s_idx + step, "high"]
                l = df_s.loc[s_idx + step, "low"]
                exit_ts = df_s.loc[s_idx + step, "timestamp"]
                
                if is_bull:
                    if h >= upper_tp:
                        pnl = (upper_tp - p_entry) / p_entry - 2 * FEE_RATE
                        active_until_ts[sym] = exit_ts
                        break
                    elif l <= lower_sl:
                        pnl = (lower_sl - p_entry) / p_entry - 2 * FEE_RATE
                        active_until_ts[sym] = exit_ts
                        break
                else:
                    if l <= lower_tp:
                        pnl = (p_entry - lower_tp) / p_entry - 2 * FEE_RATE
                        active_until_ts[sym] = exit_ts
                        break
                    elif h >= upper_sl:
                        pnl = (p_entry - upper_sl) / p_entry - 2 * FEE_RATE
                        active_until_ts[sym] = exit_ts
                        break
            
            if pnl is None:
                p_exit = df_s.loc[s_idx + LOOKAHEAD, "close"]
                active_until_ts[sym] = df_s.loc[s_idx + LOOKAHEAD, "timestamp"]
                pnl = ((p_exit - p_entry) / p_entry if is_bull else (p_entry - p_exit) / p_entry) - 2 * FEE_RATE

            trades.append(pnl)
            
        n_taken = len(trades)
        wins = sum(1 for r in trades if r > 0)
        wr = (wins / n_taken * 100) if n_taken > 0 else 0.0
        
        gross_win = sum(r for r in trades if r > 0)
        gross_loss = abs(sum(r for r in trades if r < 0))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        
        window_metrics.append({
            "window": w + 1,
            "taken": n_taken,
            "skipped_regime": skipped_regime,
            "skipped_pos": skipped_pos,
            "win_rate": wr,
            "pf": pf,
            "net_pnl": sum(trades) if n_taken > 0 else 0.0
        })
        total_regime_skipped += skipped_regime
        total_pos_skipped += skipped_pos
        total_trades_taken += n_taken
        
    return pd.DataFrame(window_metrics), total_regime_skipped, total_pos_skipped, total_trades_taken

df_normal, reg_skip_n, pos_skip_n, total_n = run_simulation(is_flipped_mode=False)
df_flipped, reg_skip_f, pos_skip_f, total_f = run_simulation(is_flipped_mode=True)

print("=" * 105)
print(f"   60M SHIFTED CANDLE RANGE (PAGES=4) WALK-FORWARD SUMMARY (ADX_4H >= {ADX_4H_MIN})")
print("=" * 105)
print(f"{'Window':<8} | {'Normal Taken':<12} | {'Regime Skip':<12} | {'Normal WR%':<12} | {'Normal PF':<10} | {'Flipped Taken':<12} | {'Flipped WR%':<12} | {'Flipped PF':<10}")
print("-" * 105)

for i in range(N_WINDOWS):
    rn = df_normal.iloc[i]
    rf = df_flipped.iloc[i]
    w_num = int(rn['window'])
    print(f"W{w_num:02d}      | {int(rn['taken']):<12} | {int(rn['skipped_regime']):<12} | {rn['win_rate']:>6.2f}%      | {rn['pf']:>8.3f}   | {int(rf['taken']):<12} | {rf['win_rate']:>6.2f}%      | {rf['pf']:>8.3f}")

print("=" * 105)

mean_wr_n = df_normal["win_rate"].replace(0.0, np.nan).dropna().mean() if len(df_normal[df_normal['taken'] > 0]) > 0 else 0.0
mean_pf_n = df_normal[df_normal['taken'] > 0]["pf"].mean() if len(df_normal[df_normal['taken'] > 0]) > 0 else 0.0

mean_wr_f = df_flipped["win_rate"].replace(0.0, np.nan).dropna().mean() if len(df_flipped[df_flipped['taken'] > 0]) > 0 else 0.0
mean_pf_f = df_flipped[df_flipped['taken'] > 0]["pf"].mean() if len(df_flipped[df_flipped['taken'] > 0]) > 0 else 0.0

print(f"\nSHIFTED AGGREGATED STATS (GATED NORMAL):")
print(f"  • Total Trades Taken           : {total_n}")
print(f"  • Total Regime Skipped (ADX<22): {reg_skip_n}")
print(f"  • Mean Window Win Rate        : {mean_wr_n:.2f}%")
print(f"  • Mean Window Profit Factor   : {mean_pf_n:.3f}")

print(f"\nSHIFTED AGGREGATED STATS (GATED FLIPPED):")
print(f"  • Total Trades Taken           : {total_f}")
print(f"  • Total Regime Skipped (ADX<22): {reg_skip_f}")
print(f"  • Mean Window Win Rate        : {mean_wr_f:.2f}%")
print(f"  • Mean Window Profit Factor   : {mean_pf_f:.3f}")
