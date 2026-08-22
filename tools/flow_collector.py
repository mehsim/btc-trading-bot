#!/usr/bin/env python3
"""
tools/flow_collector.py
-----------------------
Always-on real-time order flow and liquidation collector for Bybit V5 Linear Futures.
Collects multi-symbol market flow across all supported universe symbols:
  - publicTrade.{SYMBOL} (Taker buys, Taker sells, Volume, Imbalance)
  - allLiquidation.{SYMBOL} (Long liquidations, Short liquidations)
Aggregates into 1-minute bars and appends to local Parquet files (flow_{SYMBOL}_{date}.parquet).
"""

import os
import sys
import time
import json
import ssl
import threading
from datetime import datetime, timezone
import websocket
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import SUPPORTED_SYMBOLS
except Exception:
    SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

WS_URL = "wss://stream.bybit.com/v5/public/linear"
PARQUET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "flow_parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)

# Lock for accumulating 1-minute bucket metrics
bucket_lock = threading.Lock()
current_minute_ts = int(time.time() // 60 * 60 * 1000)

def create_empty_bucket():
    return {
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

current_buckets = {sym: create_empty_bucket() for sym in SUPPORTED_SYMBOLS}

cumulative_cvd_map = {}

def bridge_to_sqlite(sym, ts_ms, data):
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kline_cache.db")
    try:
        global cumulative_cvd_map
        delta_vol = float(data["taker_buy_vol"] - data["taker_sell_vol"])
        cumulative_cvd_map[sym] = cumulative_cvd_map.get(sym, 0.0) + delta_vol
        cvd_val = cumulative_cvd_map[sym]
        ofi_val = delta_vol
        liq_long = float(data["liq_long_vol"])
        liq_short = float(data["liq_short_vol"])
        
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_order_flow (
                    symbol TEXT,
                    timestamp REAL,
                    cvd REAL,
                    ofi REAL,
                    ob_imbalance_L2 REAL,
                    ob_spread_L2 REAL,
                    liq_long_1h REAL,
                    liq_short_1h REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            conn.execute(
                "INSERT OR REPLACE INTO historical_order_flow (symbol, timestamp, cvd, ofi, ob_imbalance_L2, ob_spread_L2, liq_long_1h, liq_short_1h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sym, ts_ms, cvd_val, ofi_val, 0.0, 0.0, liq_long, liq_short)
            )
            conn.commit()
    except Exception as ex_db:
        print(f"[Flow Collector SQLite Bridge Error] {sym}: {ex_db}", file=sys.stderr)

def flush_symbol_bucket(sym, ts_ms, data):
    """Flushes a completed 1-minute bar for a specific symbol to its daily Parquet file and SQLite cache."""
    # Only write if there was at least some trade or liquidation activity or valid bucket
    total_vol = data["taker_buy_vol"] + data["taker_sell_vol"]
    total_liq = data["liq_long_vol"] + data["liq_short_vol"]
    if total_vol == 0.0 and total_liq == 0.0:
        return

    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    date_str = dt.strftime("%Y%m%d")
    parquet_path = os.path.join(PARQUET_DIR, f"flow_{sym}_{date_str}.parquet")
    
    taker_imbalance = (data["taker_buy_vol"] - data["taker_sell_vol"]) / (total_vol + 1e-8)
    liq_imbalance = (data["liq_long_vol"] - data["liq_short_vol"]) / (total_liq + 1e-8)
    
    row = {
        "timestamp": ts_ms,
        "datetime": dt.isoformat(),
        "symbol": sym,
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
        except Exception:
            df_new.to_parquet(parquet_path, index=False)
    else:
        df_new.to_parquet(parquet_path, index=False)

    bridge_to_sqlite(sym, ts_ms, data)

def flush_all_buckets(ts_ms):
    global current_buckets
    for sym, b_data in current_buckets.items():
        try:
            flush_symbol_bucket(sym, ts_ms, b_data)
        except Exception as e:
            print(f"[Flow Collector Flush Error] {sym}: {e}", file=sys.stderr)
    current_buckets = {s: create_empty_bucket() for s in SUPPORTED_SYMBOLS}

def on_message(ws, message):
    global current_minute_ts
    try:
        data = json.loads(message)
        topic = data.get("topic", "")
        msg_data = data.get("data", [])
        
        if not topic or not msg_data:
            return
            
        sym = topic.split(".")[-1]
        if sym not in current_buckets:
            return
            
        now_ts = int(time.time() * 1000)
        now_minute = int(now_ts // 60000 * 60000)
        
        with bucket_lock:
            if now_minute > current_minute_ts:
                # Flush previous minute for all symbols
                flush_all_buckets(current_minute_ts)
                current_minute_ts = now_minute
            
            bucket = current_buckets[sym]
            
            if "publicTrade" in topic:
                for trade in msg_data:
                    side = trade.get("S", "")  # "Buy" or "Sell"
                    size = float(trade.get("v", 0.0))
                    price = float(trade.get("p", 0.0))
                    turnover = size * price
                    
                    if side == "Buy":
                        bucket["taker_buy_vol"] += size
                        bucket["taker_buy_count"] += 1
                    else:
                        bucket["taker_sell_vol"] += size
                        bucket["taker_sell_count"] += 1
                        
                    bucket["trade_turnover_usd"] += turnover
                    bucket["trade_volume_total"] += size
                    
            elif "allLiquidation" in topic:
                for liq in msg_data:
                    side = liq.get("S", "")  # "Buy" or "Sell"
                    size = float(liq.get("v", 0.0))
                    price = float(liq.get("p", 0.0))
                    usd_val = size * price
                    
                    # Bybit: Side Buy in liquidation means Short position liquidated; Side Sell means Long position liquidated
                    if side == "Sell":
                        bucket["liq_long_vol"] += usd_val
                        bucket["liq_long_count"] += 1
                    else:
                        bucket["liq_short_vol"] += usd_val
                        bucket["liq_short_count"] += 1
                        
    except Exception:
        pass

def on_error(ws, error):
    print(f"[Flow Collector Error] {error}", file=sys.stderr)

def on_close(ws, close_status_code, close_msg):
    print(f"[Flow Collector Closed] code={close_status_code}, msg={close_msg}")

def on_open(ws):
    print(f"[Flow Collector] Connected to Bybit WS. Subscribing to {len(SUPPORTED_SYMBOLS)} symbols in batches...")
    topics = []
    for s in SUPPORTED_SYMBOLS:
        topics += [f"publicTrade.{s}", f"allLiquidation.{s}"]
        
    for i in range(0, len(topics), 10):
        batch = topics[i:i+10]
        sub_msg = {"op": "subscribe", "args": batch}
        ws.send(json.dumps(sub_msg))
        time.sleep(0.05)
    print(f"✅ Successfully subscribed to {len(topics)} topic streams across {len(SUPPORTED_SYMBOLS)} symbols.")

def handle_shutdown(signum=None, frame=None):
    print(f"\n[Flow Collector] Shutdown signal received ({signum}). Flushing in-flight minute buckets to disk...")
    with bucket_lock:
        try:
            flush_all_buckets(current_minute_ts)
            print("[Flow Collector] In-flight buckets successfully flushed to Parquet. Exiting cleanly.")
        except Exception as e:
            print(f"[Flow Collector] Error flushing bucket during shutdown: {e}", file=sys.stderr)
    if signum is not None:
        sys.exit(0)

import signal
import atexit
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
atexit.register(handle_shutdown)

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
    print(f"Starting Multi-Symbol Flow Collector Daemon for {len(SUPPORTED_SYMBOLS)} symbols -> {PARQUET_DIR}")
    run_collector()
