import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
import config
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input
from main import load_model_weights

def run_production_aligned_walk_forward():
    print("=" * 85, flush=True)
    print("PRODUCTION-ALIGNED 15-WINDOW WALK-FORWARD OOS SWEEP (SL_MULT = 1.00 & 1.25)", flush=True)
    print("Rules: MIN_FLOOR Enabled (1.00%), Post-Floor Economic Gate, ADX Enter=28.0, Fee=0.08%", flush=True)
    print("=" * 85, flush=True)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT"]
    iv = "60"
    pages = 10
    round_trip_fee_pct = 0.08 / 100.0

    df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=1000, pages=pages)
    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})

    all_dfs = []
    total_candles = 0

    for sym in symbols:
        df_sym = get_history(symbol=sym, interval=iv, limit=1000, pages=pages)
        if df_sym is None or df_sym.empty: continue
        if sym != "BTCUSDT":
            df_sym = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
            df_sym["close_btc"] = df_sym["close_btc"].ffill().bfill().fillna(df_sym["close"])
        else:
            df_sym["close_btc"] = df_sym["close"]

        df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=iv)
        df_sym = add_features(df_sym)
        df_sym["symbol"] = sym
        all_dfs.append(df_sym)
        total_candles += len(df_sym)

    load_model_weights("60")
    load_model_weights(60)
    from main import models_by_interval
    models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}

    sl_candidates = [1.00, 1.25]
    n_windows = 15
    lookahead = 10

    for sl_mult in sl_candidates:
        print(f"\n" + "-" * 85, flush=True)
        print(f"EVALUATING SL_MULT = {sl_mult:.2f} ATR (WITH MIN_FLOOR & ECONOMIC GATE)", flush=True)
        print("-" * 85, flush=True)

        window_results = []

        for w in range(n_windows):
            start_pct = w / (n_windows + 1)
            end_pct = (w + 1) / (n_windows + 1)
            
            trades_w = []
            for df in all_dfs:
                sym = df["symbol"].iloc[0]
                n_rows = len(df)
                i_start = int(n_rows * start_pct)
                i_end = int(n_rows * end_pct)
                df_sub = df.iloc[i_start:i_end].copy().reset_index(drop=True)
                if len(df_sub) <= 50: continue

                m_trend = models_60.get("trending", {}).get("trend") or models_60.get("ranging", {}).get("trend")
                feat_list = models_60.get("selected_features_trending") or models_60.get("selected_features")
                if m_trend is None: continue

                _exp_names = get_model_feature_names(m_trend)
                if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
                    X_full = df_sub.reindex(columns=_exp_names, fill_value=0.0)
                elif feat_list:
                    _avail = [f for f in feat_list if f in df_sub.columns]
                    X_full = df_sub[_avail]
                else:
                    from core import features as master_features
                    _avail = [f for f in master_features if f in df_sub.columns]
                    X_full = df_sub[_avail]

                X_input = _slice_model_input(m_trend, X_full)
                probs_all = m_trend.predict_proba(X_input)

                highs = df_sub["high"].values
                lows = df_sub["low"].values
                closes = df_sub["close"].values
                atrs = df_sub["ATR"].values if "ATR" in df_sub.columns else closes * 0.01
                adxs = df_sub["ADX"].values if "ADX" in df_sub.columns else np.zeros(len(df_sub))

                for i in range(50, len(df_sub) - lookahead):
                    probs = probs_all[i]
                    p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs)>=3 else (float(probs[0]), 0.0, float(probs[1]))
                    direction = "Bullish" if p_bull >= max(p_bear, p_neut) else ("Bearish" if p_bear >= max(p_bull, p_neut) else "Neutral")
                    calibrated_conf = max(p_bull, p_bear)

                    if direction == "Neutral": continue

                    entry_price = closes[i]
                    atr_val = atrs[i] if atrs[i] > 0 and not np.isnan(atrs[i]) else entry_price * 0.01
                    atr_dollars = atr_val

                    # 3. Align ADX threshold
                    adx_enter_map = getattr(config, "REGIME_ADX_ENTER_BY_INTERVAL", {})
                    adx_enter = adx_enter_map.get(str(iv), 28.0)
                    adx_val = adxs[i]
                    tp_multiplier = 1.5567 if adx_val >= adx_enter else 2.2600

                    # 1. Add MIN_FLOOR to backtest
                    raw_sl_dist = sl_mult * atr_dollars
                    min_sl_pct = config.resolve_min_sl_pct(sym, iv)
                    min_sl_dist = entry_price * (min_sl_pct / 100.0)
                    sl_dist = max(raw_sl_dist, min_sl_dist)

                    if direction == "Bullish":
                        tp_price = entry_price + tp_multiplier * atr_dollars
                        sl_price = entry_price - sl_dist
                    else:
                        tp_price = entry_price - tp_multiplier * atr_dollars
                        sl_price = entry_price + sl_dist

                    # 2. Add post-floor economic gate
                    final_rr = abs(tp_price - entry_price) / max(1e-9, sl_dist)
                    required_p = 1.0 / (1.0 + final_rr) + 0.0016
                    if calibrated_conf < required_p:
                        continue  # Abort trade (mirrors production)

                    fee_r_penalty = round_trip_fee_pct / max(1e-6, sl_dist / entry_price)

                    trade_outcome = None
                    r_raw = 0.0

                    # Strict SL-First Intra-Bar Resolution
                    for k in range(i+1, i+1+lookahead):
                        f_h, f_l = highs[k], lows[k]
                        if direction == "Bullish":
                            if f_l <= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                            elif f_h >= tp_price: trade_outcome = "TP"; r_raw = final_rr; break
                        else:
                            if f_h >= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                            elif f_l <= tp_price: trade_outcome = "TP"; r_raw = final_rr; break

                    if trade_outcome is None:
                        exit_price = closes[i+lookahead]
                        diff = (exit_price - entry_price) if direction == "Bullish" else (entry_price - exit_price)
                        r_raw = diff / max(1e-9, sl_dist)

                    r_net = r_raw - fee_r_penalty
                    trades_w.append(r_net)

            n_w = len(trades_w)
            if n_w > 0:
                r_w_arr = np.array(trades_w)
                wins_w = r_w_arr[r_w_arr > 0]
                losses_w = r_w_arr[r_w_arr <= 0]
                wr_w = (len(wins_w) / n_w) * 100.0
                pf_w = np.sum(wins_w) / max(1e-9, np.abs(np.sum(losses_w)))
                exp_w = np.mean(r_w_arr)

                cap = 100.0
                eq = [cap]
                for r in r_w_arr:
                    cap *= (1.0 + 0.01 * r)
                    eq.append(cap)
                eq = np.array(eq)
                peak = np.maximum.accumulate(eq)
                mdd_w = float(np.abs(np.min((eq - peak) / peak))) * 100.0
                ret_w = float((cap - 100.0) / 100.0) * 100.0

                window_results.append({
                    "window": w + 1,
                    "trades": n_w,
                    "win_rate": wr_w,
                    "pf": pf_w,
                    "expectancy": exp_w,
                    "return_pct": ret_w,
                    "mdd_pct": mdd_w
                })

                print(f"  Window {w+1:02d}/15 | Trades: {n_w:3d} | WinRate: {wr_w:5.1f}% | NET PF: {pf_w:5.2f} | Expectancy: {exp_w:+6.3f}R | Return: {ret_w:+7.2f}% | MDD: -{mdd_w:4.1f}%", flush=True)

        if window_results:
            w_df = pd.DataFrame(window_results)
            prof_count = (w_df["return_pct"] > 0).sum()
            mean_wr = w_df["win_rate"].mean()
            mean_pf = w_df["pf"].mean()
            worst_mdd = w_df["mdd_pct"].max()
            mean_ret = w_df["return_pct"].mean()

            print(f"\nSUMMARY FOR SL = {sl_mult:.2f} ATR (PRODUCTION-ALIGNED):", flush=True)
            print(f"  • Profitable Windows : {prof_count}/15 ({(prof_count/n_windows)*100:.1f}%)", flush=True)
            print(f"  • Mean OOS Win Rate  : {mean_wr:.2f}%", flush=True)
            print(f"  • Mean OOS Net PF    : {mean_pf:.3f}", flush=True)
            print(f"  • Mean OOS Return    : {mean_ret:+.2f}%", flush=True)
            print(f"  • Worst OOS Drawdown : -{worst_mdd:.2f}%", flush=True)

if __name__ == "__main__":
    run_production_aligned_walk_forward()
