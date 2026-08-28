import os
import sys
import numpy as np
import pandas as pd
import json
import warnings
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

from data import get_history, merge_derivatives_sentiment_features
from core import add_features
from ensemble import EnsembleClassifier, resolve_direction
from tools.beta_calibrator import calibrate_probability

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]
INTERVAL = "60"
FEE_RATE = 0.0006  # 0.06% per leg (0.12% round-trip)
SLIPPAGE = 0.0003  # 0.03% slippage per leg

# Exact 48-Feature Stationary Contract from Task-1276 Retrain
FEATURES_48 = [
    'close_to_Kalman_lag2', 'BB_pct_lag2', 'btc_return_5m_lag3', 'RSI', 'volatility_gk_lag1', 
    'BB_width', 'ATR_norm', 'ADX_pos', 'lead_lag_diff_4h_lag1', 'lead_lag_diff_1h_lag1', 
    'lead_lag_diff_4h', 'volatility_gk_lag2', 'oi_change_4h_lag2', 'close_to_EMA50', 
    'btc_return_5m_lag2', 'ADX', 'lead_lag_diff_4h_lag2', 'btc_return_5m_lag1', 'ROC_5', 
    'close_to_EMA200', 'MACD_diff_diff', 'btc_rsi_lag1', 'MACD_diff_lag1', 'volatility_10m', 
    'ADX_z', 'fear_greed_lag1', 'volume_ratio', 'ADX_neg', 'btc_rsi', 'lead_lag_diff_1h', 
    'MACD_diff', 'ROC_24', 'volatility_24h', 'BB_pct', 'RSI_diff', 'close_to_VWAP', 
    'fear_greed_lag2', 'day_of_week_cos', 'RSI_24', 'volatility_gk', 'btc_rsi_lag2', 
    'fear_greed', 'EMA9_to_EMA21', 'day_of_week_sin', 'hour_sin', 'RSI_z', 'ROC_10', 'close_to_Kalman'
]

print("=" * 75)
print(f"60-MINUTE CHALLENGER BACKTEST (48-FEATURE STATIONARY SELECTION) — ALL 9 SYMBOLS")
print("=" * 75)

# Load 48-Feature XGBoost, LightGBM, CatBoost directly
print(f"Loading 48-Feature Ranging Challenger Ensemble Models...")

# 1. XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model("ensemble_ranging_trend_60_challenger_xgb.json")

# 2. LightGBM
lgb_model = lgb.Booster(model_file="ensemble_ranging_trend_60_challenger_lgb.txt")

# 3. CatBoost
cat_model = CatBoostClassifier()
cat_model.load_model("ensemble_ranging_trend_60_challenger_cat.json", format="json")

# 4. Meta Weights
with open("ensemble_ranging_trend_60_challenger_weights.json") as f:
    weights_data = json.load(f)
    weights = weights_data.get("weights", [0.263, 0.263, 0.474])

# 5. Calibrator
with open("calibrator_ranging_60_challenger.json") as f:
    cal_data = json.load(f)

print(f"  -> Successfully loaded 48-feature XGB, LGBM, CatBoost (Stacking weights: {weights})")

# Execution Parameters (Proven 16h geometry)
TP_MULT = 1.40
SL_MULT = 1.15
LOOKAHEAD = 16
CONF_THRESH = 0.52  # High-conviction threshold (filters out low-conviction signals)

print(f"Execution Parameters: TP={TP_MULT:.2f}x ATR, SL={SL_MULT:.2f}x ATR, Lookahead={LOOKAHEAD}h, Min Calibrated Conf={CONF_THRESH*100:.1f}%\n")

all_trades = []
per_sym_results = {}

for sym in SUPPORTED_SYMBOLS:
    print(f"Evaluating {sym} over 12,000 hourly candles...")
    df = get_history(symbol=sym, interval=INTERVAL, limit=1000, pages=12)
    if df is None or len(df) < 200:
        continue
        
    df["symbol"] = sym
    df["close_btc"] = df["close"]
    df = merge_derivatives_sentiment_features(df, symbol=sym, interval=INTERVAL)
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)
    
    # Ensure all 48 features exist
    for col in FEATURES_48:
        if col not in df.columns:
            df[col] = 0.0
            
    X_mat = df[FEATURES_48].values
    
    # Generate predictions using 48-feature ensemble
    # XGBoost
    dmat = xgb.DMatrix(X_mat)
    p_xgb = xgb_model.predict(dmat)
    
    # LightGBM
    p_lgb = lgb_model.predict(X_mat)
    
    # CatBoost
    p_cat = cat_model.predict_proba(X_mat)
    
    # Stacking ensemble blend
    p_ens = (weights[0] * p_xgb) + (weights[1] * p_lgb) + (weights[2] * p_cat)
    
    sym_trades = []
    i = 50
    while i < len(df) - LOOKAHEAD:
        row = df.iloc[i]
        adx_val = float(row.get("ADX", 20.0))
        
        # Only evaluate Ranging regime (ADX < 24.0)
        if adx_val >= 24.0:
            i += 1
            continue
            
        probs = p_ens[i]
        ml_trend, ml_conf = resolve_direction(probs)
        
        if ml_trend not in ["Bullish", "Bearish"]:
            i += 1
            continue
            
        cal_conf = calibrate_probability(ml_conf, cal_data)
        if cal_conf < CONF_THRESH:
            i += 1
            continue
            
        # Entry price with slippage
        entry_price = float(df.iloc[i + 1]["open"]) * (1.0 + SLIPPAGE if ml_trend == "Bullish" else 1.0 - SLIPPAGE)
        atr_val = float(row.get("ATR", row["close"] * row.get("ATR_norm", 0.015)))
        if atr_val <= 0:
            atr_val = entry_price * 0.015
            
        sl_dist = atr_val * SL_MULT
        tp_dist = atr_val * TP_MULT
        
        if ml_trend == "Bullish":
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist
            
        exit_price = None
        exit_reason = "Time Exit"
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
        
        i += max(1, bars_held)
        
    if len(sym_trades) > 0:
        df_st = pd.DataFrame(sym_trades)
        wins = df_st[df_st["net_pnl_pct"] > 0]["net_pnl_pct"].sum()
        loss = df_st[df_st["net_pnl_pct"] <= 0]["net_pnl_pct"].abs().sum()
        pf = (wins / loss) if loss > 0 else 99.9
        wr = float(df_st["is_win"].mean() * 100.0)
        tot_ret = float(df_st["net_pnl_pct"].sum() * 100.0)
        
        per_sym_results[sym] = {"trades": len(sym_trades), "wr": wr, "pf": pf, "net": tot_ret}
        print(f"  -> {sym}: {len(sym_trades)} Ranging Trades | Win Rate: {wr:.1f}% | Profit Factor: {pf:.2f} | Net Return: {tot_ret:+.2f}%")

print("\n" + "=" * 75)
print("PORTFOLIO 48-FEATURE RANGING CHALLENGER BACKTEST SUMMARY")
print("=" * 75)

if len(all_trades) > 0:
    df_res = pd.DataFrame(all_trades)
    tot_trades = len(df_res)
    wins = df_res[df_res["net_pnl_pct"] > 0]["net_pnl_pct"].sum()
    loss = df_res[df_res["net_pnl_pct"] <= 0]["net_pnl_pct"].abs().sum()
    pf = (wins / loss) if loss > 0 else 99.9
    wr = float(df_res["is_win"].mean() * 100.0)
    net_ret = float(df_res["net_pnl_pct"].sum() * 100.0)
    avg_trade = float(df_res["net_pnl_pct"].mean() * 100.0)
    
    tp_exits = len(df_res[df_res["exit_reason"] == "TP"])
    sl_exits = len(df_res[df_res["exit_reason"].str.startswith("SL")])
    time_exits = len(df_res[df_res["exit_reason"] == "Time Exit"])
    
    print(f"Total 60m Ranging Trades:  {tot_trades}")
    print(f"Overall Win Rate:          {wr:.2f}%")
    print(f"Overall Profit Factor:     {pf:.2f}")
    print(f"Portfolio Net Return:      {net_ret:+.2f}%")
    print(f"Average Return / Trade:    {avg_trade:+.2f}%")
    print(f"Take-Profit Exits (TP):    {tp_exits} ({tp_exits/tot_trades*100:.1f}%)")
    print(f"Stop-Loss Exits (SL):      {sl_exits} ({sl_exits/tot_trades*100:.1f}%)")
    print(f"Time Horizon Exits (16h):  {time_exits} ({time_exits/tot_trades*100:.1f}%)")
    print("=" * 75)
else:
    print("No trades generated.")
