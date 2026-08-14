import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input
from main import load_model_weights, models_by_interval

def run_60m_grid_backtest():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT"]
    iv = "60"
    round_trip_fee_pct = 0.08 / 100.0  # 0.08% fee & slippage (8 bps)
    
    print("=" * 70, flush=True)
    print(" 🚀 1-HOUR (60M) MODEL COMPREHENSIVE BACKTEST (SWEEPING TP & SL RANGES)", flush=True)
    print("=" * 70, flush=True)

    print("1. Fetching historical candle datasets across 8 crypto pairs...", flush=True)
    
    # Fetch BTC first for BTC-relative features
    df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=1000, pages=5)
    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})

    all_dfs = []
    for sym in symbols:
        df_sym = get_history(symbol=sym, interval=iv, limit=1000, pages=5)
        if df_sym is None or df_sym.empty or len(df_sym) <= 100: continue

        if sym != "BTCUSDT":
            df_sym = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
            df_sym["close_btc"] = df_sym["close_btc"].ffill().bfill().fillna(df_sym["close"])
        else:
            df_sym["close_btc"] = df_sym["close"]

        df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=iv)
        df_sym = add_features(df_sym)
        df_sym["symbol"] = sym
        all_dfs.append(df_sym)
        print(f"   • Loaded {sym}: {len(df_sym)} candles", flush=True)

    from ensemble import load_ensemble_models, get_selected_features
    m_trend, _ = load_ensemble_models("trending", "60")
    m_ranging, _ = load_ensemble_models("ranging", "60")
    
    if m_trend is None and m_ranging is None:
        print("❌ Error: 60m model weights not found!", flush=True)
        return

    feat_trending = get_selected_features("trending", "60")
    feat_ranging = get_selected_features("ranging", "60")

    # Generate predictions for each asset
    dataset_preds = []
    for df in all_dfs:
        sym = df["symbol"].iloc[0]
        
        # Evaluate trending model predictions
        _exp_names = get_model_feature_names(m_trend) if m_trend else None
        if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
            X_full = df.reindex(columns=_exp_names, fill_value=0.0)
        elif feat_trending:
            _avail = [f for f in feat_trending if f in df.columns]
            X_full = df[_avail]
        else:
            from core import features as master_features
            _avail = [f for f in master_features if f in df.columns]
            X_full = df[_avail]

        X_input = _slice_model_input(m_trend, X_full)
        probs_trending = m_trend.predict_proba(X_input)

        # Evaluate ranging model predictions
        _exp_names_r = get_model_feature_names(m_ranging) if m_ranging else None
        if _exp_names_r and not all(str(n).startswith("Column_") for n in _exp_names_r):
            X_full_r = df.reindex(columns=_exp_names_r, fill_value=0.0)
        elif feat_ranging:
            _avail_r = [f for f in feat_ranging if f in df.columns]
            X_full_r = df[_avail_r]
        else:
            from core import features as master_features
            _avail_r = [f for f in master_features if f in df.columns]
            X_full_r = df[_avail_r]

        X_input_r = _slice_model_input(m_ranging, X_full_r)
        probs_ranging = m_ranging.predict_proba(X_input_r)

        dataset_preds.append((df, probs_trending, probs_ranging))

    print("\n2. Sweeping TP / SL Multiplier Grid & Evaluating System Expectancy...", flush=True)
    
    tp_grid = [1.5, 1.8, 2.0, 2.2, 2.5]
    sl_grid = [0.75, 1.0, 1.25, 1.5]
    conf_thresholds = [0.45, 0.50, 0.52]
    lookahead = 12

    results_table = []

    for conf_thresh in conf_thresholds:
        for tp_mult in tp_grid:
            for sl_mult in sl_grid:
                rr_planned = tp_mult / sl_mult
                if rr_planned < 1.20:
                    continue  # Filter un-economical geometries

                all_trades = []

                for df, probs_all in dataset_preds:
                    highs = df["high"].values
                    lows = df["low"].values
                    closes = df["close"].values
                    atrs = df["ATR"].values if "ATR" in df.columns else closes * 0.01

                    for i in range(50, len(df) - lookahead):
                        probs = probs_all[i]
                        p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs)>=3 else (float(probs[0]), 0.0, float(probs[1]))
                        
                        direction = "Bullish" if p_bull >= max(p_bear, p_neut) else ("Bearish" if p_bear >= max(p_bull, p_neut) else "Neutral")
                        conf = max(p_bull, p_bear)

                        if direction == "Neutral" or conf < conf_thresh:
                            continue

                        entry_price = closes[i]
                        atr_val = atrs[i] if atrs[i] > 0 and not np.isnan(atrs[i]) else entry_price * 0.01

                        sl_dist = atr_val * sl_mult
                        tp_dist = atr_val * tp_mult

                        sl_pct = sl_dist / entry_price
                        fee_r_penalty = round_trip_fee_pct / max(1e-6, sl_pct)

                        if direction == "Bullish":
                            tp_price = entry_price + tp_dist
                            sl_price = entry_price - sl_dist
                        else:
                            tp_price = entry_price - tp_dist
                            sl_price = entry_price + sl_dist

                        trade_outcome = None
                        r_mult_raw = 0.0

                        # Strict Intra-Bar Resolution
                        for k in range(i+1, i+1+lookahead):
                            f_h, f_l = highs[k], lows[k]
                            if direction == "Bullish":
                                if f_l <= sl_price: trade_outcome = "SL"; r_mult_raw = -1.0; break
                                elif f_h >= tp_price: trade_outcome = "TP"; r_mult_raw = tp_mult / sl_mult; break
                            else:
                                if f_h >= sl_price: trade_outcome = "SL"; r_mult_raw = -1.0; break
                                elif f_l <= tp_price: trade_outcome = "TP"; r_mult_raw = tp_mult / sl_mult; break

                        if trade_outcome is None:
                            exit_price = closes[i+lookahead]
                            diff = (exit_price - entry_price) if direction == "Bullish" else (entry_price - exit_price)
                            r_mult_raw = diff / max(1e-9, sl_dist)

                        r_mult_net = r_mult_raw - fee_r_penalty
                        all_trades.append(r_mult_net)

                if not all_trades:
                    continue

                n_trades = len(all_trades)
                win_count = sum(1 for r in all_trades if r > 0)
                win_rate = (win_count / n_trades) * 100.0
                total_r = sum(all_trades)
                expectancy_r = total_r / n_trades

                wins = [r for r in all_trades if r > 0]
                losses = [abs(r) for r in all_trades if r <= 0]
                gross_win = sum(wins)
                gross_loss = sum(losses)
                pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0

                results_table.append({
                    "conf_thresh": conf_thresh,
                    "tp_mult": tp_mult,
                    "sl_mult": sl_mult,
                    "rr_planned": rr_planned,
                    "n_trades": n_trades,
                    "win_rate": win_rate,
                    "expectancy_r": expectancy_r,
                    "total_r": total_r,
                    "profit_factor": pf
                })

    # Sort results by Profit Factor DESC
    results_table.sort(key=lambda x: x["profit_factor"], reverse=True)

    print("\n" + "=" * 80, flush=True)
    print(" 🏆 TOP 1-HOUR (60M) BACKTEST PARAMETER SETUPS (RANKED BY PROFIT FACTOR)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Conf Thresh':<12}{'TP Mult':<10}{'SL Mult':<10}{'Planned RR':<12}{'Trades':<10}{'Win Rate':<12}{'PF':<10}{'Expectancy (R)':<15}{'Total Net R':<12}", flush=True)
    print("-" * 95, flush=True)

    for res in results_table[:15]:
        print(f"{res['conf_thresh']*100:.0f}%{'':<8}{res['tp_mult']:<10.2f}{res['sl_mult']:<10.2f}{res['rr_planned']:<12.2f}{res['n_trades']:<10}{res['win_rate']:<12.2f}%{res['profit_factor']:<10.2f}+{res['expectancy_r']:<14.3f}R+{res['total_r']:<12.1f}R", flush=True)

    best = results_table[0] if results_table else None
    if best:
        print("=" * 95, flush=True)
        print(f"🎯 OPTIMAL PARAMETER SETTING FOR 1H MODEL:")
        print(f"   • Confidence Threshold : {best['conf_thresh']*100:.0f}%")
        print(f"   • Take-Profit (TP) Mult: {best['tp_mult']:.2f}x ATR")
        print(f"   • Stop-Loss (SL) Mult  : {best['sl_mult']:.2f}x ATR")
        print(f"   • Planned Risk-Reward  : {best['rr_planned']:.2f} : 1")
        print(f"   • Profit Factor (PF)   : {best['profit_factor']:.2f}")
        print(f"   • Win Rate             : {best['win_rate']:.2f}% ({best['n_trades']} trades)")
        print(f"   • Expectancy per Trade : +{best['expectancy_r']:.3f} R")
        print("=" * 95, flush=True)

if __name__ == "__main__":
    run_60m_grid_backtest()
