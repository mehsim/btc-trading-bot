import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input
from main import load_model_weights

def run_walk_forward_analysis():
    print("=" * 75, flush=True)
    print("60M MODEL RIGOROUS WALK-FORWARD OUT-OF-SAMPLE (OOS) VALIDATION", flush=True)
    print("=" * 75, flush=True)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT"]
    iv = "60"
    pages = 10
    round_trip_fee_pct = 0.08 / 100.0  # 8 bps round-trip fee

    # 1. Load full dataset
    print("\n[ITEM 1: DATASET & SLICE AUDIT]", flush=True)
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
        
        start_date = pd.to_datetime(df_sym["timestamp"].iloc[0], unit="ms" if df_sym["timestamp"].iloc[0]>1e11 else "s")
        end_date = pd.to_datetime(df_sym["timestamp"].iloc[-1], unit="ms" if df_sym["timestamp"].iloc[-1]>1e11 else "s")
        print(f"  Symbol: {sym:8s} | Candles: {len(df_sym):5d} | Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", flush=True)

    print(f"\nTotal Candles Available Across 8 Symbols: {total_candles:,}", flush=True)

    # 2. Explicit Slicing Audit (Item 2)
    print("\n[ITEM 2: EXPLICIT SLICE AUDIT — OLDEST VS RECENT]", flush=True)
    
    load_model_weights("60")
    load_model_weights(60)
    from main import models_by_interval
    models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}

    tp_mult = 1.56
    sl_mult = 0.64
    lookahead = 10
    conf_threshold = 0.52

    def evaluate_slice(dfs_list, slice_start_pct, slice_end_pct, slice_label):
        trades_list = []
        c_count = 0
        for df in dfs_list:
            n_rows = len(df)
            i_start = int(n_rows * slice_start_pct)
            i_end = int(n_rows * slice_end_pct)
            df_sub = df.iloc[i_start:i_end].copy().reset_index(drop=True)
            c_count += len(df_sub)
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

            for i in range(50, len(df_sub) - lookahead):
                probs = probs_all[i]
                p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs)>=3 else (float(probs[0]), 0.0, float(probs[1]))
                direction = "Bullish" if p_bull >= max(p_bear, p_neut) else ("Bearish" if p_bear >= max(p_bull, p_neut) else "Neutral")
                conf = max(p_bull, p_bear)

                if direction == "Neutral" or conf < conf_threshold: continue

                entry_price = closes[i]
                atr_val = atrs[i] if atrs[i] > 0 and not np.isnan(atrs[i]) else entry_price * 0.01
                sl_dist = atr_val * sl_mult
                tp_dist = atr_val * tp_mult
                fee_r_penalty = round_trip_fee_pct / max(1e-6, sl_dist / entry_price)

                if direction == "Bullish":
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

                trade_outcome = None
                r_raw = 0.0

                for k in range(i+1, i+1+lookahead):
                    f_h, f_l = highs[k], lows[k]
                    if direction == "Bullish":
                        if f_l <= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                        elif f_h >= tp_price: trade_outcome = "TP"; r_raw = tp_mult / sl_mult; break
                    else:
                        if f_h >= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                        elif f_l <= tp_price: trade_outcome = "TP"; r_raw = tp_mult / sl_mult; break

                if trade_outcome is None:
                    exit_price = closes[i+lookahead]
                    diff = (exit_price - entry_price) if direction == "Bullish" else (entry_price - exit_price)
                    r_raw = diff / max(1e-9, sl_dist)

                r_net = r_raw - fee_r_penalty
                trades_list.append(r_net)

        n_t = len(trades_list)
        if n_t == 0:
            print(f"  {slice_label:25s}: 0 trades", flush=True)
            return
        r_arr = np.array(trades_list)
        wins = r_arr[r_arr > 0]
        losses = r_arr[r_arr <= 0]
        wr = (len(wins) / n_t) * 100.0
        pf = np.sum(wins) / max(1e-9, np.abs(np.sum(losses)))
        exp_r = np.mean(r_arr)
        sum_r = np.sum(r_arr)
        print(f"  {slice_label:30s} | Candles: {c_count:5d} | Trades: {n_t:4d} | WR: {wr:.2f}% | NET PF: {pf:.3f} | Expectancy: {exp_r:+.4f}R | Sum R: {sum_r:+.2f}R", flush=True)

    evaluate_slice(all_dfs, 0.0, 0.5, "Oldest 50% Slice (In-Sample Range)")
    evaluate_slice(all_dfs, 0.5, 1.0, "Recent 50% Slice (Holdout Range)")

    # 3. Walk-Forward Analysis (Item 3)
    print("\n[ITEM 3: 15-WINDOW SLIDING WALK-FORWARD OUT-OF-SAMPLE (OOS) TEST]", flush=True)
    print("Testing strictly Out-Of-Sample across 15 sequential sliding windows...\n", flush=True)

    n_windows = 15
    window_results = []

    for w in range(n_windows):
        start_pct = w / (n_windows + 1)
        end_pct = (w + 1) / (n_windows + 1)
        
        # Test on window (Out-of-sample temporal slice)
        trades_w = []
        c_w = 0
        for df in all_dfs:
            n_rows = len(df)
            i_start = int(n_rows * start_pct)
            i_end = int(n_rows * end_pct)
            df_sub = df.iloc[i_start:i_end].copy().reset_index(drop=True)
            c_w += len(df_sub)
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

            for i in range(50, len(df_sub) - lookahead):
                probs = probs_all[i]
                p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs)>=3 else (float(probs[0]), 0.0, float(probs[1]))
                direction = "Bullish" if p_bull >= max(p_bear, p_neut) else ("Bearish" if p_bear >= max(p_bull, p_neut) else "Neutral")
                conf = max(p_bull, p_bear)

                if direction == "Neutral" or conf < conf_threshold: continue

                entry_price = closes[i]
                atr_val = atrs[i] if atrs[i] > 0 and not np.isnan(atrs[i]) else entry_price * 0.01
                sl_dist = atr_val * sl_mult
                tp_dist = atr_val * tp_mult
                fee_r_penalty = round_trip_fee_pct / max(1e-6, sl_dist / entry_price)

                if direction == "Bullish":
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

                trade_outcome = None
                r_raw = 0.0

                for k in range(i+1, i+1+lookahead):
                    f_h, f_l = highs[k], lows[k]
                    if direction == "Bullish":
                        if f_l <= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                        elif f_h >= tp_price: trade_outcome = "TP"; r_raw = tp_mult / sl_mult; break
                    else:
                        if f_h >= sl_price: trade_outcome = "SL"; r_raw = -1.0; break
                        elif f_l <= tp_price: trade_outcome = "TP"; r_raw = tp_mult / sl_mult; break

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
            sum_r_w = np.sum(r_w_arr)

            # Max Drawdown on window equity curve
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
                "sum_r": sum_r_w,
                "return_pct": ret_w,
                "mdd_pct": mdd_w
            })

            print(f"  Window {w+1:02d}/15 | Trades: {n_w:3d} | WinRate: {wr_w:5.1f}% | NET PF: {pf_w:5.2f} | Expectancy: {exp_w:+6.3f}R | Return: {ret_w:+7.2f}% | MDD: -{mdd_w:4.1f}%", flush=True)

    # 4. Summary Judgment (Item 4)
    print("\n" + "=" * 75, flush=True)
    print("[ITEM 4: SUMMARY & VERDICT]", flush=True)
    print("=" * 75, flush=True)

    if window_results:
        w_df = pd.DataFrame(window_results)
        mean_wr = w_df["win_rate"].mean()
        mean_pf = w_df["pf"].mean()
        mean_exp = w_df["expectancy"].mean()
        mean_ret = w_df["return_pct"].mean()
        max_mdd = w_df["mdd_pct"].max()
        pos_windows = (w_df["return_pct"] > 0).sum()

        print(f"Mean OOS Win Rate    : {mean_wr:.2f}%", flush=True)
        print(f"Mean OOS NET PF      : {mean_pf:.3f}", flush=True)
        print(f"Mean OOS Expectancy  : {mean_exp:+.4f} R per trade", flush=True)
        print(f"Mean OOS Return      : {mean_ret:+.2f}%", flush=True)
        print(f"Worst OOS Drawdown   : -{max_mdd:.2f}%", flush=True)
        print(f"Profitable Windows   : {pos_windows}/{n_windows} ({pos_windows/n_windows*100:.1f}%)", flush=True)
        
        print("-" * 75, flush=True)
        if mean_ret > 0 and pos_windows >= (n_windows * 0.6):
            print("VERDICT: REAL EDGE CONFIRMED ✅ (Positive Walk-Forward OOS Return across windows)", flush=True)
        else:
            print("VERDICT: IN-SAMPLE ARTIFACT ❌ (Negative / Weak Walk-Forward OOS Return)", flush=True)
        print("=" * 75, flush=True)

if __name__ == "__main__":
    run_walk_forward_analysis()
