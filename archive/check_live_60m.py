import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history, merge_derivatives_sentiment_features
from features import add_features
from ensemble import get_model_feature_names, _slice_model_input, resolve_direction
from main import load_model_weights

def check_live_60m():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT"]
    iv = "60"

    print("Fetching live market data (300 bars) for 60m (1h) timeframe across symbols...", flush=True)

    df_btc = get_history(symbol="BTCUSDT", interval=iv, limit=300, pages=1)
    if df_btc is None or df_btc.empty:
        print("Failed to fetch BTCUSDT history!", flush=True)
        return

    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})

    load_model_weights("60")
    load_model_weights(60)
    from main import models_by_interval
    models_60 = models_by_interval.get("60") or models_by_interval.get(60) or {}

    results = []

    for sym in symbols:
        try:
            df_sym = get_history(symbol=sym, interval=iv, limit=300, pages=1)
            if df_sym is None or df_sym.empty:
                print(f"Skipping {sym}: No data", flush=True)
                continue

            if sym != "BTCUSDT":
                df_sym = pd.merge(df_sym, df_btc_sub, on="timestamp", how="left")
                df_sym["close_btc"] = df_sym["close_btc"].ffill().bfill().fillna(df_sym["close"])
            else:
                df_sym["close_btc"] = df_sym["close"]

            df_sym = merge_derivatives_sentiment_features(df_sym, symbol=sym, interval=iv)
            df_sym = add_features(df_sym)

            m_trend = models_60.get("trending", {}).get("trend") or models_60.get("ranging", {}).get("trend")
            feat_list = models_60.get("selected_features_trending") or models_60.get("selected_features")

            if m_trend is None:
                print(f"No 60m model loaded for {sym}", flush=True)
                continue

            _exp_names = get_model_feature_names(m_trend)
            if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
                X_full = df_sym.reindex(columns=_exp_names, fill_value=0.0)
            elif feat_list:
                _avail = [f for f in feat_list if f in df_sym.columns]
                X_full = df_sym[_avail]
            else:
                from core import features as master_features
                _avail = [f for f in master_features if f in df_sym.columns]
                X_full = df_sym[_avail]

            X_input = _slice_model_input(m_trend, X_full)
            probs = m_trend.predict_proba(X_input)[-1]  # Latest candle

            p_bear = float(probs[0])
            p_neut = float(probs[1]) if len(probs) >= 2 else 0.0
            p_bull = float(probs[2]) if len(probs) >= 3 else float(probs[1])

            direction, raw_conf = resolve_direction(probs)
            
            latest_price = df_sym["close"].iloc[-1]
            latest_ts = pd.to_datetime(df_sym["timestamp"].iloc[-1], unit="ms" if df_sym["timestamp"].iloc[-1]>1e11 else "s")

            results.append({
                "symbol": sym,
                "price": round(latest_price, 4),
                "direction": direction,
                "raw_conf": f"{raw_conf * 100.0:.2f}%",
                "p_bear": f"{p_bear * 100.0:.1f}%",
                "p_neut": f"{p_neut * 100.0:.1f}%",
                "p_bull": f"{p_bull * 100.0:.1f}%",
                "timestamp": latest_ts.strftime("%Y-%m-%d %H:%M UTC")
            })
        except Exception as e:
            print(f"Error processing {sym}: {e}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print(f"LIVE 60M (1H) MARKET PREDICTIONS AS OF {results[0]['timestamp'] if results else 'NOW'}", flush=True)
    print("=" * 85, flush=True)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False), flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    check_live_60m()
