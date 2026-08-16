#!/usr/bin/env python3
"""
tools/flow_collector.py
-----------------------
Always-on real-time order flow and liquidation collector for Bybit V5 Linear Futures.
Collects:
  - publicTrade.BTCUSDT (Taker buys, Taker sells, Volume, Imbalance)
  - allLiquidation.BTCUSDT (Long liquidations, Short liquidations)
Aggregates into 1-minute bars and appends to local Parquet files and SQLite.
"""

import os
import sys
import time
import json
import ssl
import threading
import sqlite3
from datetime import datetime, timezone
import websocket
import pandas as pd
import numpy as np

SYMBOL = "BTCUSDT"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
PARQUET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "flow_parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)

# Lock for accumulating 1-minute bucket metrics
bucket_lock = threading.Lock()
current_minute_ts = int(time.time() // 60 * 60 * 1000)

current_bucket = {
    "taker_buy_vol": 0.0,
    "taker_sell_vol": 0.0,
    "taker_buy_count": 0,
    "taker_sell_count": 0,
    "liq_long_vol": 0.0,
    "liq_short_vol": 0.0,
    "liq_long_count": 0,
    "liq_short_count": 0,
    "trade_turnover_usd": 0.0,
    "trade_volume_total": 0.0,
}

def flush_bucket(ts_ms, data):
    """Flushes a completed 1-minute bar to daily Parquet file."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    date_str = dt.strftime("%Y%m%d")
    parquet_path = os.path.join(PARQUET_DIR, f"flow_{SYMBOL}_{date_str}.parquet")
    
    total_vol = data["taker_buy_vol"] + data["taker_sell_vol"]
    taker_imbalance = (data["taker_buy_vol"] - data["taker_sell_vol"]) / (total_vol + 1e-8)
    total_liq = data["liq_long_vol"] + data["liq_short_vol"]
    liq_imbalance = (data["liq_long_vol"] - data["liq_short_vol"]) / (total_liq + 1e-8)
    
    row = {
        "timestamp": ts_ms,
        "datetime": dt.isoformat(),
        "symbol": SYMBOL,
        "taker_buy_vol": float(data["taker_buy_vol"]),
        "taker_sell_vol": float(data["taker_sell_vol"]),
        "taker_buy_count": int(data["taker_buy_count"]),
        "taker_sell_count": int(data["taker_sell_count"]),
        "taker_imbalance": float(taker_imbalance),
        "liq_long_vol": float(data["liq_long_vol"]),
        "liq_short_vol": float(data["liq_short_vol"]),
        "liq_long_count": int(data["liq_long_count"]),
        "liq_short_count": int(data["liq_short_count"]),
        "liq_imbalance": float(liq_imbalance),
        "trade_turnover_usd": float(data["trade_turnover_usd"]),
    }
    
    df_new = pd.DataFrame([row])
    
    if os.path.exists(parquet_path):
        try:
            df_existing = pd.read_parquet(parquet_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            df_combined.to_parquet(parquet_path, index=False)
        except Exception as e:
            df_new.to_parquet(parquet_path, index=False)
    else:
        df_new.to_parquet(parquet_path, index=False)

def on_message(ws, message):
    global current_minute_ts, current_bucket
    try:
        data = json.loads(message)
        topic = data.get("topic", "")
        msg_data = data.get("data", [])
        
        now_ts = int(time.time() * 1000)
        now_minute = int(now_ts // 60000 * 60000)
        
        with bucket_lock:
            if now_minute > current_minute_ts:
                # Flush previous minute
                flush_bucket(current_minute_ts, current_bucket)
                current_minute_ts = now_minute
                current_bucket = {
                    "taker_buy_vol": 0.0,
                    "taker_sell_vol": 0.0,
                    "taker_buy_count": 0,
                    "taker_sell_count": 0,
                    "liq_long_vol": 0.0,
                    "liq_short_vol": 0.0,
                    "liq_long_count": 0,
                    "liq_short_count": 0,
                    "trade_turnover_usd": 0.0,
                    "trade_volume_total": 0.0,
                }
            
            if "publicTrade" in topic:
                for trade in msg_data:
                    side = trade.get("S", "")  # "Buy" or "Sell"
                    size = float(trade.get("v", 0.0))
                    price = float(trade.get("p", 0.0))
                    turnover = size * price
                    
                    if side == "Buy":
                        current_bucket["taker_buy_vol"] += size
                        current_bucket["taker_buy_count"] += 1
                    else:
                        current_bucket["taker_sell_vol"] += size
                        current_bucket["taker_sell_count"] += 1
                        
                    current_bucket["trade_turnover_usd"] += turnover
                    current_bucket["trade_volume_total"] += size
                    
            elif "allLiquidation" in topic:
                for liq in msg_data:
                    side = liq.get("S", "")  # "Buy" or "Sell"
                    size = float(liq.get("v", 0.0))
                    price = float(liq.get("p", 0.0))
                    usd_val = size * price
                    
                    # Bybit: Side Buy in liquidation means Short position liquidated; Side Sell means Long position liquidated
                    if side == "Sell":
                        current_bucket["liq_long_vol"] += usd_val
                        current_bucket["liq_long_count"] += 1
                    else:
                        current_bucket["liq_short_vol"] += usd_val
                        current_bucket["liq_short_count"] += 1
                        
    except Exception as ex:
        pass

def on_error(ws, error):
    print(f"[Flow Collector Error] {error}", file=sys.stderr)

def on_close(ws, close_status_code, close_msg):
    print(f"[Flow Collector Closed] code={close_status_code}, msg={close_msg}")

def on_open(ws):
    print(f"[Flow Collector] Connected to Bybit WS. Subscribing to publicTrade.{SYMBOL} and allLiquidation.{SYMBOL}...")
    sub_msg = {
        "op": "subscribe",
        "args": [
            f"publicTrade.{SYMBOL}",
            f"allLiquidation.{SYMBOL}"
        ]
    }
    ws.send(json.dumps(sub_msg))

def run_collector():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=20, ping_timeout=10)
        except Exception as e:
            print(f"[Flow Collector Exception] {e}. Reconnecting in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    print(f"Starting Flow Collector Daemon for {SYMBOL} -> {PARQUET_DIR}")
    run_collector()
