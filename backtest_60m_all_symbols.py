import os
import sys
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import load_ensemble_classifier, load_ensemble_regressor, resolve_direction, _slice_model_input
from tools.beta_calibrator import calibrate_probability
import config

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = "60"
FEE_RATE = 0.0006  # 0.06% taker fee per leg (0.12% roundtrip)
SLIPPAGE = 0.0003  # 0.03% slippage per leg

print("=" * 70)
print(f"60-MINUTE RANGING & TRENDING BACKTEST — ALL {len(SUPPORTED_SYMBOLS)} SYMBOLS")
print("=" * 70)

from main import load_model_weights, models_by_interval

print("Loading 60m models via runtime loader...")
load_model_weights("60")
models_60 = models_by_interval.get("60", {})

model_t = models_60.get("trending", {}).get("trend")
model_r = models_60.get("ranging", {}).get("trend") or model_t

feat_t = models_60.get("selected_features_trending")
feat_r = models_60.get("selected_features_ranging") or feat_t

print(f"Loaded Trending Model: {type(model_t)} (Features: {len(feat_t) if feat_t else 'N/A'})")
print(f"Loaded Ranging Model: {type(model_r)} (Features: {len(feat_r) if feat_r else 'N/A'})")

# Load calibrators
cal_r = None
for c_name in ["calibrator_ranging_60_challenger.json", "calibrator_ranging_60.json"]:
    if os.path.exists(c_name):
        with open(c_name) as f:
            cal_r = json.load(f)
            break

cal_t = None
for c_name in ["calibrator_trending_60_challenger.json", "calibrator_trending_60.json"]:
    if os.path.exists(c_name):
        with open(c_name) as f:
            cal_t = json.load(f)
            break

# Barrier & execution configuration
cfg_60 = config.TIMEFRAME_CONFIG.get("60", {})
TP_RANGING = float(cfg_60.get("tp_mult_ranging", 1.40))
TP_TRENDING = float(cfg_60.get("tp_mult_trending", 2.00))
SL_MULT = float(cfg_60.get("sl_mult", 1.15))
LOOKAHEAD = int(cfg_60.get("lookahead", 16))
CONF_THRESH = float(cfg_60.get("base_confidence_threshold", 0.40))

print(f"\nExecution Settings: TP_Ranging={TP_RANGING:.2f}x, TP_Trending={TP_TRENDING:.2f}x, SL={SL_MULT:.2f}x, Lookahead={LOOKAHEAD}h, ConfThresh={CONF_THRESH*100:.1f}%\n")

all_trades = []
per_symbol_stats = {}

for sym in SUPPORTED_SYMBOLS:
    print(f"Processing {sym} (12,000 hourly candles)...")
    df = get_history(symbol=sym, interval=INTERVAL, limit=1000, pages=12)
    if df is None or len(df) < 200:
        print(f"  Insufficient data for {sym}, skipping.")
        continue
    
    df["symbol"] = sym
    df["close_btc"] = df["close"]
    df = merge_derivatives_sentiment_features(df, symbol=sym, interval=INTERVAL)
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)
    
    sym_trades = []
    i = 50
    while i < len(df) - LOOKAHEAD:
        row = df.iloc[i]
        adx_val = float(row.get("ADX", 20.0))
        is_trending = adx_val >= 24.0
        
        active_model = model_t if is_trending else model_r
        active_features = feat_t if is_trending else feat_r
        active_calibrator = cal_t if is_trending else cal_r
        tp_mult = TP_TRENDING if is_trending else TP_RANGING
        regime_label = "Trending" if is_trending else "Ranging"
        
        # Prepare feature vector
        avail_cols = [f for f in active_features if f in df.columns]
        if len(avail_cols) < len(active_features):
            for missing_f in active_features:
                if missing_f not in df.columns:
                    df[missing_f] = 0.0
        
        X_vec = df[active_features].iloc[[i]]
        
        try:
            # Predict directly with numpy values or dataframe matching feature count
            probs = active_model.predict_proba(X_vec.values)[0]
            ml_trend, ml_conf = resolve_direction(probs)
        except Exception as ex:
            try:
                probs = active_model.predict_proba(X_vec)[0]
                ml_trend, ml_conf = resolve_direction(probs)
            except Exception:
                i += 1
                continue
            
        if ml_trend not in ["Bullish", "Bearish"]:
            i += 1
            continue
            
        cal_conf = calibrate_probability(ml_conf, active_calibrator) if active_calibrator else ml_conf
        
        if cal_conf < CONF_THRESH:
            i += 1
            continue
            
        # Simulate trade execution with pessimistic fills
        entry_price = float(df.iloc[i + 1]["open"]) * (1.0 + SLIPPAGE if ml_trend == "Bullish" else 1.0 - SLIPPAGE)
        atr_val = float(row.get("ATR", row["close"] * row.get("ATR_norm", 0.015)))
        if atr_val <= 0:
            atr_val = entry_price * 0.015
            
        sl_dist = atr_val * SL_MULT
        tp_dist = atr_val * tp_mult
        
        if ml_trend == "Bullish":
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist
            
        exit_price = None
        exit_reason = "Time"
        bars_held = 0
        
        for bar in range(1, LOOKAHEAD + 1):
            curr_bar = df.iloc[i + bar]
            h = float(curr_bar["high"])
            l = float(curr_bar["low"])
            
            if ml_trend == "Bullish":
                if h >= tp_price and l <= sl_price:
                    exit_price = sl_price * (1.0 - SLIPPAGE)
                    exit_reason = "SL (Wick)"
                    bars_held = bar
                    break
                elif h >= tp_price:
                    exit_price = tp_price * (1.0 - SLIPPAGE)
                    exit_reason = "TP"
                    bars_held = bar
                    break
                elif l <= sl_price:
                    exit_price = sl_price * (1.0 - SLIPPAGE)
                    exit_reason = "SL"
                    bars_held = bar
                    break
            else: # Bearish
                if l <= tp_price and h >= sl_price:
                    exit_price = sl_price * (1.0 + SLIPPAGE)
                    exit_reason = "SL (Wick)"
                    bars_held = bar
                    break
                elif l <= tp_price:
                    exit_price = tp_price * (1.0 + SLIPPAGE)
                    exit_reason = "TP"
                    bars_held = bar
                    break
                elif h >= sl_price:
                    exit_price = sl_price * (1.0 + SLIPPAGE)
                    exit_reason = "SL"
                    bars_held = bar
                    break
                    
        if exit_price is None:
            close_bar = df.iloc[i + LOOKAHEAD]
            exit_price = float(close_bar["close"]) * (1.0 - SLIPPAGE if ml_trend == "Bullish" else 1.0 + SLIPPAGE)
            exit_reason = "Time Exit"
            bars_held = LOOKAHEAD
            
        if ml_trend == "Bullish":
            gross_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price
            
        net_pnl_pct = gross_pnl_pct - (FEE_RATE * 2.0)
        
        trade_record = {
            "symbol": sym,
            "entry_time": str(df.iloc[i]["timestamp"]),
            "direction": ml_trend,
            "regime": regime_label,
            "cal_conf": cal_conf,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "net_pnl_pct": net_pnl_pct,
            "is_win": net_pnl_pct > 0
        }
        
        sym_trades.append(trade_record)
        all_trades.append(trade_record)
        
        # Advance index to avoid overlapping bar entry
        i += max(1, bars_held)
        
    # Symbol Summary
    if len(sym_trades) > 0:
        df_sym_t = pd.DataFrame(sym_trades)
        wins = df_sym_t[df_sym_t["net_pnl_pct"] > 0]["net_pnl_pct"]
        losses = df_sym_t[df_sym_t["net_pnl_pct"] <= 0]["net_pnl_pct"].abs()
        gross_profit = wins.sum()
        gross_loss = losses.sum()
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
        wr = float((df_sym_t["is_win"]).mean() * 100.0)
        tot_return = float(df_sym_t["net_pnl_pct"].sum() * 100.0)
        ranging_trades = len(df_sym_t[df_sym_t["regime"] == "Ranging"])
        trending_trades = len(df_sym_t[df_sym_t["regime"] == "Trending"])
        
        per_symbol_stats[sym] = {
            "trades": len(sym_trades),
            "ranging": ranging_trades,
            "trending": trending_trades,
            "win_rate": wr,
            "profit_factor": pf,
            "net_return": tot_return
        }
        print(f"  -> {sym}: {len(sym_trades)} trades (Ranging: {ranging_trades}, Trending: {trending_trades}) | WR: {wr:.1f}% | PF: {pf:.2f} | Net: {tot_return:+.2f}%")

print("\n" + "=" * 70)
print("PORTFOLIO-WIDE 60-MINUTE BACKTEST PERFORMANCE SUMMARY")
print("=" * 70)

if len(all_trades) == 0:
    print("No trades generated across any symbol.")
    sys.exit(0)

df_all = pd.DataFrame(all_trades)
tot_trades = len(df_all)
wins = df_all[df_all["net_pnl_pct"] > 0]["net_pnl_pct"]
losses = df_all[df_all["net_pnl_pct"] <= 0]["net_pnl_pct"].abs()
gross_profit = wins.sum()
gross_loss = losses.sum()
overall_pf = (gross_profit / gross_loss) if gross_loss > 0 else 99.9
overall_wr = float(df_all["is_win"].mean() * 100.0)
total_net_return = float(df_all["net_pnl_pct"].sum() * 100.0)
avg_trade = float(df_all["net_pnl_pct"].mean() * 100.0)

# Ranging Breakdown
df_ranging = df_all[df_all["regime"] == "Ranging"]
r_trades = len(df_ranging)
r_wins = df_ranging[df_ranging["net_pnl_pct"] > 0]["net_pnl_pct"].sum()
r_loss = df_ranging[df_ranging["net_pnl_pct"] <= 0]["net_pnl_pct"].abs().sum()
r_pf = (r_wins / r_loss) if r_loss > 0 else 0.0
r_wr = float(df_ranging["is_win"].mean() * 100.0) if r_trades > 0 else 0.0
r_ret = float(df_ranging["net_pnl_pct"].sum() * 100.0) if r_trades > 0 else 0.0

# Trending Breakdown
df_trending = df_all[df_all["regime"] == "Trending"]
t_trades = len(df_trending)
t_wins = df_trending[df_trending["net_pnl_pct"] > 0]["net_pnl_pct"].sum()
t_loss = df_trending[df_trending["net_pnl_pct"] <= 0]["net_pnl_pct"].abs().sum()
t_pf = (t_wins / t_loss) if t_loss > 0 else 0.0
t_wr = float(df_trending["is_win"].mean() * 100.0) if t_trades > 0 else 0.0
t_ret = float(df_trending["net_pnl_pct"].sum() * 100.0) if t_trades > 0 else 0.0

# Max Drawdown
cum_returns = (1.0 + df_all["net_pnl_pct"]).cumprod()
peak = cum_returns.cummax()
drawdown = (cum_returns - peak) / peak
max_dd = float(drawdown.min() * 100.0)

# Sharpe Ratio (annualized for hourly trades assuming ~1500 active bars/yr)
trade_std = float(df_all["net_pnl_pct"].std())
sharpe = (float(df_all["net_pnl_pct"].mean()) / trade_std * np.sqrt(1500)) if trade_std > 0 else 0.0

print(f"Total Trades Executed:   {tot_trades}")
print(f"Overall Win Rate:        {overall_wr:.2f}%")
print(f"Overall Profit Factor:   {overall_pf:.2f}")
print(f"Cumulative Net Return:   {total_net_return:+.2f}%")
print(f"Average Return / Trade:  {avg_trade:+.2f}%")
print(f"Max Portfolio Drawdown:  {max_dd:.2f}%")
print(f"Estimated Sharpe Ratio:  {sharpe:.2f}")
print("-" * 70)
print(f"RANGING REGIME PERFORMANCE  ({r_trades} trades):")
print(f"  - Win Rate:      {r_wr:.2f}%")
print(f"  - Profit Factor: {r_pf:.2f}")
print(f"  - Net Return:    {r_ret:+.2f}%")
print("-" * 70)
print(f"TRENDING REGIME PERFORMANCE ({t_trades} trades):")
print(f"  - Win Rate:      {t_wr:.2f}%")
print(f"  - Profit Factor: {t_pf:.2f}")
print(f"  - Net Return:    {t_ret:+.2f}%")
print("=" * 70)
