import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input
from main import load_model_weights, classify_market_regime

def run_60m_backtest():
    print("=" * 70, flush=True)
    print("RUNNING HISTORICAL BACKTEST FOR NEW 60M MODEL (55,917 CANDLES)", flush=True)
    print("Parameters: TP=1.56x ATR, SL=0.64x ATR, Lookahead=10 bars, No Exit Management", flush=True)
    print("=" * 70, flush=True)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT"]
    iv = "60"
    pages = 10

    print("\n1. Fetching historical candle data across top 8 symbols...", flush=True)
    all_dfs = []
    
    df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=1000, pages=pages)
    if df_btc is None or df_btc.empty:
        print("Error fetching BTCUSDT 60m data!", flush=True)
        return

    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})

    total_raw_candles = 0
    for sym in symbols:
        df_sym = get_history(symbol=sym, interval=iv, limit=1000, pages=pages)
        if df_sym is None or df_sym.empty:
            continue
        total_raw_candles += len(df_sym)
        if sym != "BTCUSDT":
            df_sym = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
            df_sym["close_btc"] = df_sym["close_btc"].ffill().bfill().fillna(df_sym["close"])
        else:
            df_sym["close_btc"] = df_sym["close"]

        df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=iv)
        df_sym = add_features(df_sym)
        df_sym["symbol"] = sym
        all_dfs.append(df_sym)
        print(f"Loaded {sym}: {len(df_sym)} candles", flush=True)

    print(f"Loaded {len(all_dfs)} symbols | Total raw candles: {total_raw_candles}", flush=True)

    print("\n2. Loading 60m ensemble models...", flush=True)
    load_model_weights(60)
    from main import models_by_interval
    models_60 = models_by_interval.get(60, {})
    if not models_60:
        print("Failed to load 60m model weights!", flush=True)
        return

    tp_mult = 1.56
    sl_mult = 0.64
    lookahead = 10
    conf_threshold = 0.52

    trades = []

    print("\n3. Evaluating model signals and simulating trades...", flush=True)
    for df in all_dfs:
        sym = df["symbol"].iloc[0]
        if len(df) <= 50:
            continue

        for i in range(50, len(df) - lookahead):
            df_slice = df.iloc[:i+1]
            latest_candle = df_slice.iloc[-1]
            entry_price = float(latest_candle["close"])
            atr_val = float(latest_candle.get("ATR", entry_price * 0.01))
            if atr_val <= 0 or np.isnan(atr_val):
                atr_val = entry_price * 0.01

            regime = classify_market_regime(df_slice, interval=60)
            regime_key = regime.lower() if regime in ["Trending", "Ranging"] else "trending"
            
            m_trend = models_60.get(regime_key, {}).get("trend")
            m_price = models_60.get(regime_key, {}).get("price")
            feat_list = models_60.get(f"selected_features_{regime_key}") or models_60.get("selected_features")

            if m_trend is None or m_price is None:
                alt_key = "ranging" if regime_key == "trending" else "trending"
                m_trend = models_60.get(alt_key, {}).get("trend")
                m_price = models_60.get(alt_key, {}).get("price")
                feat_list = models_60.get(f"selected_features_{alt_key}") or models_60.get("selected_features")

            if m_trend is None or m_price is None:
                continue

            _exp_names = get_model_feature_names(m_trend)
            if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
                X_live_full = latest_candle.to_frame().T.reindex(columns=_exp_names, fill_value=0.0)
            elif feat_list:
                _avail = [f for f in feat_list if f in latest_candle.index]
                X_live_full = latest_candle[_avail].to_frame().T
            else:
                from core import features as master_features
                _avail = [f for f in master_features if f in latest_candle.index]
                X_live_full = latest_candle[_avail].to_frame().T

            X_live = _slice_model_input(m_trend, X_live_full)

            try:
                probs = m_trend.predict_proba(X_live)[0]
            except Exception:
                continue

            if len(probs) >= 3:
                p_bear, p_neut, p_bull = float(probs[0]), float(probs[1]), float(probs[2])
            else:
                p_bear, p_neut, p_bull = float(probs[0]), 0.0, float(probs[1])

            probs_dict = {"Bearish": p_bear, "Neutral": p_neut, "Bullish": p_bull}
            direction = max(probs_dict, key=probs_dict.get)
            conf = probs_dict[direction]

            if direction == "Neutral" or conf < conf_threshold:
                continue

            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult

            if direction == "Bullish":
                tp_price = entry_price + tp_dist
                sl_price = entry_price - sl_dist
            else:
                tp_price = entry_price - tp_dist
                sl_price = entry_price + sl_dist

            trade_outcome = None
            r_multiple = 0.0

            future_candles = df.iloc[i+1 : i+1+lookahead]
            for _, f_candle in future_candles.iterrows():
                f_high = float(f_candle["high"])
                f_low = float(f_candle["low"])

                if direction == "Bullish":
                    if f_low <= sl_price:
                        trade_outcome = "SL"
                        r_multiple = -1.0
                        break
                    elif f_high >= tp_price:
                        trade_outcome = "TP"
                        r_multiple = tp_mult / sl_mult
                        break
                else: # Bearish
                    if f_high >= sl_price:
                        trade_outcome = "SL"
                        r_multiple = -1.0
                        break
                    elif f_low <= tp_price:
                        trade_outcome = "TP"
                        r_multiple = tp_mult / sl_mult
                        break

            if trade_outcome is None:
                exit_price = float(future_candles.iloc[-1]["close"])
                if direction == "Bullish":
                    diff = exit_price - entry_price
                else:
                    diff = entry_price - exit_price
                r_multiple = diff / max(1e-9, sl_dist)
                trade_outcome = "EXP" if r_multiple > 0 else "EXSL"

            trades.append({
                "symbol": sym,
                "direction": direction,
                "confidence": conf,
                "outcome": trade_outcome,
                "r_multiple": r_multiple
            })

    print("\n" + "=" * 70, flush=True)
    print("BACKTEST RESULTS (NEW 60M MODEL)", flush=True)
    print("=" * 70, flush=True)

    n_trades = len(trades)
    if n_trades == 0:
        print("No trades generated during backtest!", flush=True)
        return

    r_array = np.array([t["r_multiple"] for t in trades])
    wins = r_array[r_array > 0]
    losses = r_array[r_array <= 0]

    win_count = len(wins)
    win_rate = (win_count / n_trades) * 100.0

    gross_gain = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

    profit_factor = gross_gain / max(1e-9, gross_loss)
    expectancy_r = float(np.mean(r_array))

    print(f"Candles Analyzed  : {total_raw_candles:,}", flush=True)
    print(f"Total Trade Count : {n_trades} trades", flush=True)
    print(f"Win Rate          : {win_rate:.2f}% ({win_count}/{n_trades})", flush=True)
    print(f"Profit Factor     : {profit_factor:.3f}", flush=True)
    print(f"Expectancy (R)    : {expectancy_r:+.4f} R per trade", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_60m_backtest()
