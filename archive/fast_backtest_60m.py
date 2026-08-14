import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input
from main import load_model_weights

def run_fast_backtest(start_page=5, end_page=10, window_name="First 24,000 Candles (Oldest Window)"):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT"]
    iv = "60"
    round_trip_fee_pct = 0.08 / 100.0  # 0.08% round-trip fee & slippage (8 bps)

    print(f"1. Fetching historical candle dataset ({window_name})...", flush=True)
    
    # Load candles for target window
    all_dfs = []
    total_candles = 0

    # Fetch BTC first for BTC-relative features
    df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=1000, pages=end_page)
    # Slice target pages window
    df_btc = df_btc.iloc[max(0, (start_page-1)*1000) : min(len(df_btc), end_page*1000)]
    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})

    for sym in symbols:
        df_sym = get_history(symbol=sym, interval=iv, limit=1000, pages=end_page)
        if df_sym is None or df_sym.empty: continue
        df_sym = df_sym.iloc[max(0, (start_page-1)*1000) : min(len(df_sym), end_page*1000)]
        if len(df_sym) <= 50: continue

        total_candles += len(df_sym)
        if sym != "BTCUSDT":
            df_sym = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
            df_sym["close_btc"] = df_sym["close_btc"].ffill().bfill().fillna(df_sym["close"])
        else:
            df_sym["close_btc"] = df_sym["close"]

        df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=iv)
        df_sym = add_features(df_sym)
        df_sym["symbol"] = sym
        all_dfs.append(df_sym)

    load_model_weights("60")
    load_model_weights(60)
    from main import models_by_interval
    models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}
    
    tp_mult = 1.56
    sl_mult = 0.64
    lookahead = 10
    conf_threshold = 0.52

    trades = []

    for df in all_dfs:
        sym = df["symbol"].iloc[0]
        if len(df) <= 50: continue

        m_trend = models_60.get("trending", {}).get("trend") or models_60.get("ranging", {}).get("trend")
        feat_list = models_60.get("selected_features_trending") or models_60.get("selected_features_ranging") or models_60.get("selected_features")
        
        if m_trend is None: continue

        _exp_names = get_model_feature_names(m_trend)

        if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
            X_full = df.reindex(columns=_exp_names, fill_value=0.0)
        elif feat_list:
            _avail = [f for f in feat_list if f in df.columns]
            X_full = df[_avail]
        else:
            from core import features as master_features
            _avail = [f for f in master_features if f in df.columns]
            X_full = df[_avail]

        X_input = _slice_model_input(m_trend, X_full)
        probs_all = m_trend.predict_proba(X_input)

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df["ATR"].values if "ATR" in df.columns else closes * 0.01

        for i in range(50, len(df) - lookahead):
            probs = probs_all[i]
            p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2]) if len(probs)>=3 else (float(probs[0]), 0.0, float(probs[1]))
            
            direction = "Bullish" if p_bull >= max(p_bear, p_neut) else ("Bearish" if p_bear >= max(p_bull, p_neut) else "Neutral")
            conf = max(p_bull, p_bear)

            if direction == "Neutral" or conf < conf_threshold:
                continue

            entry_price = closes[i]
            atr_val = atrs[i] if atrs[i] > 0 and not np.isnan(atrs[i]) else entry_price * 0.01

            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult

            # Fee penalty in R units
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

            # Strict SL-First Intra-Bar Resolution
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

            # Net R after 0.08% round-trip fee & slippage deduction
            r_mult_net = r_mult_raw - fee_r_penalty

            trades.append({
                "timestamp": df["timestamp"].iloc[i],
                "r_raw": r_mult_raw,
                "r_net": r_mult_net,
                "fee_r": fee_r_penalty
            })

    n_trades = len(trades)
    if n_trades == 0:
        print(f"[{window_name}] No trades generated!", flush=True)
        return

    trades.sort(key=lambda t: t["timestamp"])
    r_net_arr = np.array([t["r_net"] for t in trades])
    r_raw_arr = np.array([t["r_raw"] for t in trades])

    wins = r_net_arr[r_net_arr > 0]
    losses = r_net_arr[r_net_arr <= 0]

    win_rate = (len(wins) / max(1, n_trades)) * 100.0
    gross_gain = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor_net = gross_gain / max(1e-9, gross_loss)
    
    gross_gain_raw = float(np.sum(r_raw_arr[r_raw_arr > 0])) if len(r_raw_arr[r_raw_arr > 0]) > 0 else 0.0
    gross_loss_raw = float(np.abs(np.sum(r_raw_arr[r_raw_arr <= 0]))) if len(r_raw_arr[r_raw_arr <= 0]) > 0 else 0.0
    profit_factor_raw = gross_gain_raw / max(1e-9, gross_loss_raw)

    expectancy_net = float(np.mean(r_net_arr))
    sum_r_net = float(np.sum(r_net_arr))

    # Additive Risk Return (Fixed $ risk per trade)
    total_return_additive_pct = sum_r_net * 1.0  # 1% risk per 1.0 R

    # Compounding Risk Return
    capital = 100.0
    equity_curve = [capital]
    for r in r_net_arr:
        capital *= (1.0 + 0.01 * r)
        equity_curve.append(capital)

    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_drawdown_pct = float(np.abs(np.min(drawdowns))) * 100.0
    total_return_compounded_pct = float((capital - 100.0) / 100.0) * 100.0

    print("\n" + "=" * 70, flush=True)
    print(f"BACKTEST RESULTS ({window_name.upper()})", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Candles evaluated : {total_candles:,}", flush=True)
    print(f"Total Trade Count ($n$) : {n_trades} trades", flush=True)
    print(f"Win Rate (Net)          : {win_rate:.2f}% ({len(wins)}/{n_trades})", flush=True)
    print(f"Raw Profit Factor       : {profit_factor_raw:.3f} (No Fees)", flush=True)
    print(f"NET Profit Factor (PF)  : {profit_factor_net:.3f} (After 0.08% Fees & Slippage)", flush=True)
    print(f"Net Expectancy ($R$)    : {expectancy_net:+.4f} R / trade", flush=True)
    print(f"Sum of R-multiples      : {sum_r_net:+.2f} R", flush=True)
    print(f"Additive Return (1% risk): {total_return_additive_pct:+.2f}%", flush=True)
    print(f"Compounded Return (1%)  : {total_return_compounded_pct:+.2f}%", flush=True)
    print(f"Max Drawdown (MDD)      : -{max_drawdown_pct:.2f}%", flush=True)
    print(f"Fees & Slippage Deducted: 0.08% Round-Trip (8 bps) per trade", flush=True)
    print(f"Barrier Order           : Strict Pessimistic Intra-Bar (SL evaluated BEFORE TP)", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    name = sys.argv[3] if len(sys.argv) > 3 else "First 24,000 Candles (Oldest Window)"
    run_fast_backtest(start_page=start, end_page=end, window_name=name)
