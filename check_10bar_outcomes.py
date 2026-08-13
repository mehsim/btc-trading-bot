import os, sys, time
import pandas as pd
import numpy as np

sys.path.append("/Users/mehsimkhurshid/Downloads/btc-trading-bot")
from data import get_history

def check_outcomes():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT"]
    iv = "60"

    print("Fetching post-16:00 UTC price action (10 bars up to 02:00 UTC)...", flush=True)

    results = []

    for sym in symbols:
        try:
            df = get_history(symbol=sym, interval=iv, limit=100, pages=1)
            if df is None or df.empty:
                continue

            # Convert timestamps to datetime
            timestamps = pd.to_datetime(df["timestamp"], unit="ms" if df["timestamp"].iloc[0]>1e11 else "s")
            df["datetime"] = timestamps

            # Find row closest to 16:00 UTC
            # Get latest 10 candles
            df_recent = df.iloc[-10:].reset_index(drop=True)
            
            entry_price = df_recent["close"].iloc[0]
            entry_time = df_recent["datetime"].iloc[0].strftime("%H:%M UTC")
            
            max_high = df_recent["high"].max()
            min_low = df_recent["low"].min()
            exit_price = df_recent["close"].iloc[-1]
            exit_time = df_recent["datetime"].iloc[-1].strftime("%H:%M UTC")

            max_up_pct = ((max_high - entry_price) / entry_price) * 100.0
            max_down_pct = ((entry_price - min_low) / entry_price) * 100.0
            net_change_pct = ((exit_price - entry_price) / entry_price) * 100.0

            results.append({
                "symbol": sym,
                "entry_close": round(entry_price, 4),
                "max_high": round(max_high, 4),
                "min_low": round(min_low, 4),
                "exit_close": round(exit_price, 4),
                "max_up_pct": f"+{max_up_pct:.2f}%",
                "max_down_pct": f"-{max_down_pct:.2f}%",
                "net_change": f"{net_change_pct:+.2f}%",
            })
        except Exception as e:
            print(f"Error for {sym}: {e}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("PRICE ACTION OUTCOME: 16:00 UTC TO 02:00 UTC (10-BAR LOOKAHEAD)", flush=True)
    print("=" * 90, flush=True)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False), flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    check_outcomes()
