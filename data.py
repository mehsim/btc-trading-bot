from typing import Optional
from logger import log_event
import os
import time
import requests
import pandas as pd
import threading
import numpy as np

from dotenv import load_dotenv
load_dotenv()

raw_trade_mode = os.environ.get("TRADE_MODE", "simulation").lower()
TRADE_MODE = raw_trade_mode if raw_trade_mode in ["live", "testnet", "simulation"] else "simulation"
BYBIT_BASE_URL = "https://api-testnet.bybit.com" if TRADE_MODE == "testnet" else "https://api.bybit.com"


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kline_cache.db")
db_write_lock = threading.Lock()
_db_initialized = False
_db_init_lock = threading.Lock()

def safe_get_sqlite_conn(db_path=DB_PATH, timeout=60.0):
    import sqlite3
    for attempt in range(3):
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.execute("PRAGMA busy_timeout = 60000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.row_factory = sqlite3.Row
            res = conn.execute("PRAGMA quick_check(1);").fetchone()
            if res and res[0] != "ok":
                raise sqlite3.DatabaseError(f"Corrupt DB disk image: {res[0]}")
            return conn
        except sqlite3.DatabaseError as e:
            print(f"[SQLite Auto-Recovery] Detected database corruption ({e}). Rebuilding database file {db_path}...")
            try:
                if conn is not None:
                    conn.close()
            except Exception as ex_data:
                log_event("WARNING", f"data notice: {ex_data}")
            for ext in ["", "-wal", "-shm"]:
                target = f"{db_path}{ext}"
                if os.path.exists(target):
                    try:
                        os.remove(target)
                    except Exception as ex_data:
                        log_event("WARNING", f"data notice: {ex_data}")
            time.sleep(0.5)
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

def init_db():
    global _db_initialized
    if _db_initialized:
        return
    with _db_init_lock:
        if _db_initialized:
            return
        conn = safe_get_sqlite_conn(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kline_data (
                symbol TEXT,
                interval TEXT,
                timestamp REAL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                turnover REAL,
                UNIQUE(symbol, interval, timestamp) ON CONFLICT REPLACE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oi_data (
                symbol TEXT,
                interval TEXT,
                timestamp REAL,
                open_interest REAL,
                UNIQUE(symbol, interval, timestamp) ON CONFLICT REPLACE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_data (
                symbol TEXT,
                timestamp REAL,
                funding_rate REAL,
                UNIQUE(symbol, timestamp) ON CONFLICT REPLACE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fng_data (
                timestamp REAL PRIMARY KEY,
                fear_greed REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_order_flow (
                symbol TEXT,
                timestamp REAL,
                cvd REAL,
                ofi REAL,
                ob_imbalance_L2 REAL DEFAULT 0.0,
                ob_spread_L2 REAL DEFAULT 0.0,
                liq_long_1h REAL DEFAULT 0.0,
                liq_short_1h REAL DEFAULT 0.0,
                UNIQUE(symbol, timestamp) ON CONFLICT REPLACE
            )
        """)
        # High-Performance Indexes for Time-Series Queries (F-18 Fix)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_sym_iv_ts ON kline_data(symbol, interval, timestamp DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_oi_sym_iv_ts ON oi_data(symbol, interval, timestamp DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_funding_sym_ts ON funding_data(symbol, timestamp DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hof_sym_ts ON historical_order_flow(symbol, timestamp DESC);")
        
        # Migration for existing databases (C11: SQL injection pattern prevention)
        cursor = conn.execute("PRAGMA table_info(historical_order_flow)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        import re
        for col in ["ob_imbalance_L2", "ob_spread_L2", "liq_long_1h", "liq_short_1h"]:
            if col not in existing_cols and re.match(r'^[a-zA-Z0-9_]+$', col):
                try:
                    conn.execute(f"ALTER TABLE historical_order_flow ADD COLUMN {col} REAL DEFAULT 0.0")
                except Exception as ex_data:
                    log_event("WARNING", f"data notice: {ex_data}")
        conn.commit()
        conn.close()
        _db_initialized = True

def get_bybit_proxies():
    # If running on Hugging Face and no explicit BYBIT_PROXY is set, bypass internal HF proxy
    if os.environ.get("SPACE_ID") and not os.environ.get("BYBIT_PROXY"):
        return None

    proxy = (
        os.environ.get("BYBIT_PROXY") or
        os.environ.get("HTTPS_PROXY") or
        os.environ.get("HTTP_PROXY") or
        os.environ.get("https_proxy") or
        os.environ.get("http_proxy")
    )
    if proxy:
        if "://" not in proxy:
            proxy = "http://" + proxy
        return {
            "http": proxy,
            "https": proxy
        }
    return None

_session = None
_session_lock = threading.Lock()

def get_shared_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                proxies = get_bybit_proxies()
                if proxies:
                    _session.proxies.update(proxies)
    return _session

def bybit_public_get(url, params=None, headers=None, max_retries=3):
    session = get_shared_session()
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in [402, 403, 429, 502, 503, 504] and attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            raise e

def get_history(symbol="BTCUSDT", interval="15", limit=1000, pages=1):
    init_db()
    target_count = limit * pages
    
    # 1. Load cache if it exists
    df_cache = None
    try:
        conn = safe_get_sqlite_conn(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, open, high, low, close, volume, turnover FROM kline_data WHERE symbol=? AND interval=? ORDER BY timestamp ASC",
            (symbol, str(interval))
        )
        rows = cursor.fetchall()
        conn.close()
        if rows:
            df_cache = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    except Exception as e:
        print(f"[Cache Load Error] {e}. Rebuilding cache for {symbol} {interval}.")
        df_cache = None

    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    all_data = []
    
    if df_cache is not None and len(df_cache) > 0:
        # Cache exists. Fetch only new candles (timestamp > cache_max_ts)
        cache_max_ts = float(df_cache["timestamp"].max())
        
        # Step A: Fetch newer candles
        new_data = []
        current_end = None
        stop_fetching = False
        
        for page in range(pages):
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": str(interval),
                "limit": limit
            }
            if current_end is not None:
                params["end"] = current_end
                
            try:
                response = bybit_public_get(url, params=params, headers=headers)
                if response.status_code == 200:
                    res = response.json()
                    batch = res.get("result", {}).get("list", [])
                    if not batch:
                        break
                    
                    # Parse batch
                    for item in batch:
                        ts = float(item[0])
                        if ts < cache_max_ts:
                            stop_fetching = True
                            break
                        new_data.append(item)
                    
                    if stop_fetching:
                        break
                        
                    oldest_ts = int(float(batch[-1][0]))
                    current_end = oldest_ts - 1
                else:
                    print(f"Error fetching newer page {page + 1}: Received status code {response.status_code}")
                    break
            except Exception as e:
                print(f"Error fetching newer page {page + 1}: {e}")
                break
                
            if pages > 1:
                time.sleep(0.1)

        # Convert new_data list to DataFrame structure matching cache
        if new_data:
            df_new = pd.DataFrame(new_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df_new = df_new.astype(float)
            df_merged = pd.concat([df_new, df_cache], ignore_index=True)
        else:
            df_merged = df_cache.copy()
            
        df_merged = df_merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Step B: If the merged dataset does not have enough candles to satisfy requested target_count, fetch older candles
        if len(df_merged) < target_count:
            needed = target_count - len(df_merged)
            pages_needed = (needed // limit) + 1
            current_end = int(df_merged["timestamp"].min()) - 1
            older_data = []
            
            for page in range(pages_needed):
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": str(interval),
                    "limit": limit,
                    "end": current_end
                }
                try:
                    response = bybit_public_get(url, params=params, headers=headers)
                    if response.status_code == 200:
                        res = response.json()
                        batch = res.get("result", {}).get("list", [])
                        if not batch:
                            break
                        older_data.extend(batch)
                        oldest_ts = int(float(batch[-1][0]))
                        current_end = oldest_ts - 1
                    else:
                        break
                except Exception as e:
                    print(f"Error fetching older page: {e}")
                    break
                time.sleep(0.1)
                
            if older_data:
                df_older = pd.DataFrame(older_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df_older = df_older.astype(float)
                df_merged = pd.concat([df_merged, df_older], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                
        df_history = df_merged
        
    else:
        # Cache does not exist. Fetch full history from Bybit
        current_end = None
        for page in range(pages):
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": str(interval),
                "limit": limit
            }
            if current_end is not None:
                params["end"] = current_end
                
            try:
                response = bybit_public_get(url, params=params, headers=headers)
                if response.status_code == 200:
                    res = response.json()
                    if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                        batch = res["result"]["list"]
                        all_data.extend(batch)
                        oldest_ts = int(float(batch[-1][0]))
                        current_end = oldest_ts - 1
                    else:
                        break
                else:
                    print(f"Error fetching page {page + 1}: Received status code {response.status_code}")
                    break
            except Exception as e:
                print(f"Error fetching page {page + 1}: {e}")
                break
                
            if pages > 1:
                time.sleep(0.1)
                
        if all_data:
            df_history = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df_history = df_history.astype(float)
            df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df_history = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    # If Bybit fetches failed/returned nothing, try fallback endpoints (Binance/Kraken)
    if len(df_history) == 0:
        print(f"Bybit klines returned no data. Attempting Binance API fallback for {symbol}...")
        try:
            binance_interval = "1h"
            if str(interval) == "5":
                binance_interval = "5m"
            if str(interval) == "15":
                binance_interval = "15m"
            elif str(interval) == "30":
                binance_interval = "30m"
            elif str(interval) == "60":
                binance_interval = "1h"
            elif str(interval) == "120":
                binance_interval = "2h"
            elif str(interval) == "240":
                binance_interval = "4h"
            elif str(interval) == "360":
                binance_interval = "6h"
            elif str(interval).upper() == "D":
                binance_interval = "1d"

            binance_url = "https://api.binance.com/api/v3/klines"
            binance_params = {
                "symbol": symbol.upper(),
                "interval": binance_interval,
                "limit": min(limit * pages, 1000)
            }
            # Binance fallback does not use proxy to conserve metered proxy bandwidth
            resp = requests.get(binance_url, params=binance_params, headers=headers, timeout=10)
            if resp.status_code == 200:
                binance_data = resp.json()
                fallback_data = []
                for item in binance_data:
                    fallback_data.append([
                        float(item[0]), # timestamp
                        float(item[1]), # open
                        float(item[2]), # high
                        float(item[3]), # low
                        float(item[4]), # close
                        float(item[5]), # volume
                        float(item[7])  # turnover
                    ])
                df_history = pd.DataFrame(fallback_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df_history = df_history.astype(float)
                df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                print(f"Successfully loaded {len(df_history)} candles from Binance API fallback.")
            else:
                print(f"Binance fallback failed with HTTP {resp.status_code}")
        except Exception as ex:
            print(f"Error fetching Binance fallback klines: {ex}")

    if len(df_history) == 0:
        print(f"Bybit & Binance failed. Attempting Kraken API fallback for {symbol}...")
        try:
            kraken_interval = 60
            if str(interval) == "5":
                kraken_interval = 5
            elif str(interval) == "15":
                kraken_interval = 15
            elif str(interval) == "60":
                kraken_interval = 60
            elif str(interval) == "120":
                kraken_interval = 60
            elif str(interval) == "240":
                kraken_interval = 240
            elif str(interval) == "360":
                kraken_interval = 240
            elif str(interval).upper() == "D":
                kraken_interval = 1440

            # Explicit symbol map for Kraken (None = not supported, skip cleanly)
            KRAKEN_SYMBOL_MAP = {
                "BTCUSDT": "XBTUSDT",
                "ETHUSDT": "ETHUSDT",
                "SOLUSDT": "SOLUSDT",
                "XRPUSDT": "XRPUSDT",
                "LTCUSDT": "LTCUSDT",
                "DOGEUSDT": "XDGUSDT",
                "AVAXUSDT": "AVAXUSDT",
                "NEARUSDT": None,   # Not available on Kraken
                "BNBUSDT": None,    # Not available on Kraken
                "SUIUSDT": None,    # Not available on Kraken
                "APTUSDT": None,    # Not available on Kraken
                "DOTUSDT": "DOTUSDT",
                "ADAUSDT": "ADAUSDT",
            }
            symbol_upper = symbol.upper()
            kraken_pair = KRAKEN_SYMBOL_MAP.get(symbol_upper, symbol_upper)
            if kraken_pair is None:
                print(f"[Kraken] {symbol_upper} not supported on Kraken. Skipping fallback.")
                kraken_pair = None  # Will be caught below

            if kraken_pair is None:
                pass  # Already logged above, skip request
            else:
                kraken_url = "https://api.kraken.com/0/public/OHLC"
                kraken_params = {
                    "pair": kraken_pair,
                    "interval": kraken_interval
                }
                # Kraken fallback does not use proxy to conserve metered proxy bandwidth
                resp = requests.get(kraken_url, params=kraken_params, headers=headers, timeout=10)
                if resp.status_code == 200:
                    res = resp.json()
                    if "result" in res and len(res["result"]) > 0:
                        pair_key = [k for k in res["result"].keys() if k != "last"][0]
                        kraken_data = res["result"][pair_key]
                        candles_to_take = kraken_data[-limit:] if len(kraken_data) > limit else kraken_data
                        fallback_data = []
                        for item in candles_to_take:
                            fallback_data.append([
                                float(item[0]) * 1000,
                                float(item[1]),
                                float(item[2]),
                                float(item[3]),
                                float(item[4]),
                                float(item[6]),
                                float(item[6]) * float(item[4])
                            ])
                        df_history = pd.DataFrame(fallback_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                        df_history = df_history.astype(float)
                        df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                        print(f"Successfully loaded {len(df_history)} candles from Kraken API fallback.")
                    else:
                        print(f"Kraken returned empty results or error: {res.get('error')}")
                else:
                    print(f"Kraken fallback failed with HTTP {resp.status_code}")
        except Exception as ex:
            print(f"Error fetching Kraken fallback klines: {ex}")


    if len(df_history) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    # Save/update cache
    if len(df_history) > 0:
        if df_cache is not None:
            df_combined = pd.concat([df_history, df_cache], ignore_index=True)
        else:
            df_combined = df_history
            
        df_combined = df_combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Keep at most 30,000 candles to save disk space
        if len(df_combined) > 30000:
            df_combined = df_combined.iloc[-30000:]
            
        try:
            with db_write_lock:
                conn = safe_get_sqlite_conn(DB_PATH)
                conn.executemany(
                    "INSERT OR REPLACE INTO kline_data (symbol, interval, timestamp, open, high, low, close, volume, turnover) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(symbol, str(interval), float(row["timestamp"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), float(row["turnover"])) for _, row in df_combined.iterrows()]
                )
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"[Cache Write Error] {e}")

    # Validate and enforce candle continuity with capped synthetic bar synthesis (S-3)
    final_df = df_history.iloc[-target_count:].reset_index(drop=True)
    if len(final_df) > 1:
        try:
            str_iv = str(interval).strip().upper()
            if str_iv.isdigit():
                expected_step_ms = int(str_iv) * 60 * 1000
            elif str_iv in ("D", "1D", "DAY", "DAILY"):
                expected_step_ms = 86400 * 1000
            elif str_iv in ("W", "1W", "WEEK", "WEEKLY"):
                expected_step_ms = 7 * 86400 * 1000
            elif str_iv.endswith("M") and str_iv[:-1].isdigit():
                expected_step_ms = int(str_iv[:-1]) * 60 * 1000
            elif str_iv.endswith("H") and str_iv[:-1].isdigit():
                expected_step_ms = int(str_iv[:-1]) * 3600 * 1000
            else:
                expected_step_ms = 3600 * 1000
            diffs = final_df["timestamp"].diff().dropna()
            gaps = diffs[diffs > expected_step_ms * 1.5]
            if len(gaps) > 0:
                max_gap_bars = int(round(float(gaps.max()) / max(1, expected_step_ms)))
                recent_window = final_df.iloc[-100:] if len(final_df) >= 100 else final_df
                recent_diffs = recent_window["timestamp"].diff().dropna()
                recent_gaps = recent_diffs[recent_diffs > expected_step_ms * 1.5]
                recent_max_gap_bars = int(round(float(recent_gaps.max()) / max(1, expected_step_ms))) if len(recent_gaps) > 0 else 0

                if recent_max_gap_bars > 3:
                    log_event("WARNING", f"[{symbol} {interval}m Data Continuity] Unserviceable recent gap detected: {recent_max_gap_bars} consecutive bars in active evaluation window (limit: 3). Refusing flat bar synthesis to prevent feature distortion.")
                    final_df.attrs["is_discontinuous"] = True
                    final_df.attrs["gap_exceeded"] = True
                    final_df.attrs["synthetic_bar_count"] = 0
                    final_df.attrs["max_consecutive_synthetic_bars"] = recent_max_gap_bars
                else:
                    if max_gap_bars > 3:
                        log_event("INFO", f"[{symbol} {interval}m Data Continuity] Detected historical tail gap ({max_gap_bars} bars > 100 bars ago). Active evaluation window is contiguous ({recent_max_gap_bars} bars). Reindexing grid.")
                    else:
                        log_event("INFO", f"[{symbol} {interval}m Data Continuity] Detected {len(gaps)} minor gap(s) (max: {max_gap_bars} bars). Reindexing to contiguous grid (capped <= 3).")
                    min_ts = int(final_df["timestamp"].iloc[0])
                    max_ts = int(final_df["timestamp"].iloc[-1])
                    full_ts = np.arange(min_ts, max_ts + expected_step_ms, expected_step_ms, dtype=np.int64)
                    df_grid = pd.DataFrame({"timestamp": full_ts})
                    merged_df = pd.merge(df_grid, final_df, on="timestamp", how="left")
                    merged_df["is_synthetic"] = merged_df["close"].isna()
                    synthetic_count = int(merged_df["is_synthetic"].sum())
                    merged_df["close"] = merged_df["close"].ffill().bfill()
                    merged_df["open"] = merged_df["open"].fillna(merged_df["close"])
                    merged_df["high"] = merged_df["high"].fillna(merged_df["close"])
                    merged_df["low"] = merged_df["low"].fillna(merged_df["close"])
                    merged_df["volume"] = merged_df["volume"].fillna(0.0)
                    final_df = merged_df.iloc[-target_count:].reset_index(drop=True)
                    final_df.attrs["is_discontinuous"] = False
                    final_df.attrs["gap_exceeded"] = False
                    final_df.attrs["synthetic_bar_count"] = synthetic_count
                    final_df.attrs["max_consecutive_synthetic_bars"] = max_gap_bars
            else:
                final_df.attrs["is_discontinuous"] = False
                final_df.attrs["gap_exceeded"] = False
                final_df.attrs["synthetic_bar_count"] = 0
                final_df.attrs["max_consecutive_synthetic_bars"] = 0
        except Exception as ex_gap:
            log_event("WARNING", f"[{symbol} {interval}m Data Continuity] Continuity reindex notice: {ex_gap}")
            final_df.attrs["is_discontinuous"] = False
            final_df.attrs["gap_exceeded"] = False
            final_df.attrs["synthetic_bar_count"] = 0
            final_df.attrs["max_consecutive_synthetic_bars"] = 0

    return final_df

def get_bybit_oi_history(symbol="BTCUSDT", interval="15", start_ts_ms=None, end_ts_ms=None):
    # Load cache from CSV
    cache_dir = "kline_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    csv_file = os.path.join(cache_dir, f"oi_{symbol}_{interval}.csv")
    
    df_cache = None
    if os.path.exists(csv_file):
        try:
            df_cache = pd.read_csv(csv_file)
            df_cache["timestamp"] = df_cache["timestamp"].astype(float)
            df_cache["open_interest"] = df_cache["open_interest"].astype(float)
        except Exception as e:
            print(f"[OI CSV Load Error] {e}")
            df_cache = None

    url = f"{BYBIT_BASE_URL}/v5/market/open-interest"
    interval_time = "1h"
    if str(interval) == "5":
        interval_time = "5min"
    elif str(interval) == "15":
        interval_time = "15min"
    elif str(interval) in ["60", "120"]:
        interval_time = "1h"
    elif str(interval) in ["240", "360"]:
        interval_time = "4h"

    oi_data = []
    cursor = None
    
    cache_max_ts = float(df_cache["timestamp"].max()) if df_cache is not None and len(df_cache) > 0 else None
    stop_fetching = False
    max_pages = 2 if cache_max_ts is not None else 50
    
    for page in range(max_pages):
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": 200
        }
        if cursor:
            params["cursor"] = cursor
            
        try:
            resp = bybit_public_get(url, params=params)
            if resp.status_code != 200:
                break
            res = resp.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                batch = res["result"]["list"]
                
                # Check for overlap with cache
                for item in batch:
                    ts = float(item["timestamp"])
                    if cache_max_ts is not None and ts <= cache_max_ts:
                        stop_fetching = True
                        break
                    oi_data.append(item)
                    
                if stop_fetching:
                    break
                    
                cursor = res["result"].get("nextPageCursor")
                if not cursor:
                    break
            else:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching OI history: {e}")
            break

    # Parse and merge
    df_new = pd.DataFrame()
    if oi_data:
        df_new = pd.DataFrame(oi_data)
        df_new["timestamp"] = df_new["timestamp"].astype(float)
        df_new["open_interest"] = df_new["openInterest"].astype(float)
        df_new = df_new[["timestamp", "open_interest"]]
        
    if df_cache is not None:
        if not df_new.empty:
            df_history = pd.concat([df_cache, df_new], ignore_index=True)
        else:
            df_history = df_cache
    else:
        df_history = df_new

    if not df_history.empty:
        df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Save cache
        try:
            df_history.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"[OI CSV Write Error] {e}")

    # Slice for start/end if requested
    if start_ts_ms:
        df_history = df_history[df_history["timestamp"] >= float(start_ts_ms)]
    if end_ts_ms:
        df_history = df_history[df_history["timestamp"] <= float(end_ts_ms)]
        
    return df_history.reset_index(drop=True)

funding_locks = {}
funding_locks_guard = threading.Lock()

def get_bybit_funding_history(symbol="BTCUSDT", start_ts_ms=None, end_ts_ms=None):
    with funding_locks_guard:
        if symbol not in funding_locks:
            funding_locks[symbol] = threading.Lock()
        sym_lock = funding_locks[symbol]
    with sym_lock:
        return _get_bybit_funding_history_impl(symbol, start_ts_ms, end_ts_ms)

def _get_bybit_funding_history_impl(symbol="BTCUSDT", start_ts_ms=None, end_ts_ms=None):
    # Load cache from CSV
    cache_dir = "kline_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    csv_file = os.path.join(cache_dir, f"funding_{symbol}.csv")
    alt_csv_file = os.path.join(cache_dir, f"{symbol}_funding.csv")
    
    df_cache = None
    if os.path.exists(csv_file):
        try:
            df_cache = pd.read_csv(csv_file)
            df_cache["timestamp"] = df_cache["timestamp"].astype(float)
            df_cache["funding_rate"] = df_cache["funding_rate"].astype(float)
        except Exception:
            df_cache = None
    elif os.path.exists(alt_csv_file):
        try:
            df_cache = pd.read_csv(alt_csv_file)
            df_cache["timestamp"] = df_cache["timestamp"].astype(float)
            df_cache["funding_rate"] = df_cache["funding_rate"].astype(float)
        except Exception:
            df_cache = None

    url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
    funding_data = []
    
    cache_max_ts = float(df_cache["timestamp"].max()) if df_cache is not None and len(df_cache) > 0 else None
    cache_min_ts = float(df_cache["timestamp"].min()) if df_cache is not None and len(df_cache) > 0 else None
    current_end = int(end_ts_ms) if end_ts_ms else int(time.time() * 1000)
    stop_fetching = False
    max_pages = 50
    
    for page in range(max_pages):
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": 200,
            "endTime": current_end
        }
        try:
            resp = bybit_public_get(url, params=params)
            if resp.status_code != 200:
                break
            res = resp.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                batch = res["result"]["list"]
                
                # Check for overlap with cache
                for item in batch:
                    ts = float(item["fundingRateTimestamp"])
                    if cache_max_ts is not None and ts <= cache_max_ts:
                        stop_fetching = True
                        break
                    funding_data.append(item)
                    
                if stop_fetching:
                    break
                    
                oldest_ts = int(batch[-1]["fundingRateTimestamp"])
                if start_ts_ms and oldest_ts < int(start_ts_ms):
                    break
                
                current_end = oldest_ts - 1
            else:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching Funding history: {e}")
            break

    # Parse and merge
    df_new = pd.DataFrame()
    if funding_data:
        df_new = pd.DataFrame(funding_data)
        df_new["timestamp"] = df_new["fundingRateTimestamp"].astype(float)
        df_new["funding_rate"] = df_new["fundingRate"].astype(float)
        df_new = df_new[["timestamp", "funding_rate"]]
        
    if df_cache is not None:
        if not df_new.empty:
            df_history = pd.concat([df_cache, df_new], ignore_index=True)
        else:
            df_history = df_cache
    else:
        df_history = df_new

    if not df_history.empty:
        df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Save cache
        try:
            df_history.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"[Funding CSV Write Error] {e}")

    # Slice for start/end if requested
    if start_ts_ms:
        df_history = df_history[df_history["timestamp"] >= float(start_ts_ms)]
    if end_ts_ms:
        df_history = df_history[df_history["timestamp"] <= float(end_ts_ms)]
        
    return df_history.reset_index(drop=True)

fng_cache = None
fng_lock = threading.Lock()
fng_cache_time = 0.0

def get_fear_and_greed_history():
    global fng_cache, fng_cache_time
    if fng_cache is not None and (time.time() - fng_cache_time) < 14400:
        return fng_cache
    with fng_lock:
        # Cache for 4 hours (14400 seconds) in memory
        if fng_cache is not None and (time.time() - fng_cache_time) < 14400:
            return fng_cache
            
        # Try loading from local sqlite database first
        df_cache = None
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, fear_greed FROM fng_data ORDER BY timestamp ASC;")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                df_cache = pd.DataFrame(rows, columns=["timestamp", "fear_greed"])
        except Exception as e:
            print(f"[FnG Cache Load Error] {e}")
            df_cache = None

        # Fetch fresh data from API (limit=30 if cache is warm, limit=0/all if cold)
        limit_val = 0 if df_cache is None or len(df_cache) == 0 else 30
        url = f"https://api.alternative.me/fng/?limit={limit_val}&format=json"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                res = resp.json()
                if "data" in res and len(res["data"]) > 0:
                    fng_data = []
                    for item in res["data"]:
                        fng_data.append({
                            "timestamp": float(item["timestamp"]) * 1000,
                            "fear_greed": float(item["value"])
                        })
                    df_new = pd.DataFrame(fng_data)
                    
                    if df_cache is not None:
                        df_history = pd.concat([df_cache, df_new], ignore_index=True)
                    else:
                        df_history = df_new
                        
                    df_history = df_history.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    
                    # Write to database cache
                    try:
                        with db_write_lock:
                            conn = sqlite3.connect(DB_PATH, timeout=30.0)
                            conn.execute("PRAGMA journal_mode=WAL;")
                            conn.executemany(
                                "INSERT OR REPLACE INTO fng_data (timestamp, fear_greed) VALUES (?, ?);",
                                [(float(row["timestamp"]), float(row["fear_greed"])) for _, row in df_history.iterrows()]
                            )
                            conn.commit()
                            conn.close()
                    except Exception as e:
                        print(f"[FnG Cache Write Error] {e}")
                        
                    fng_cache = df_history
                    fng_cache_time = time.time()
                    return df_history
        except Exception as e:
            print(f"Error fetching Fear & Greed history: {e}")
            
        if df_cache is not None and not df_cache.empty:
            fng_cache = df_cache
            fng_cache_time = time.time()
            return df_cache
            
    if fng_cache is not None:
        return fng_cache

    return pd.DataFrame(columns=["timestamp", "fear_greed"])

def _merge_cached_derivatives(df, df_oi, df_funding, df_fng, df_btc=None, symbol="BTCUSDT"):
    """Merge pre-fetched OI, funding, F&G, and BTC DataFrames into df without making live REST calls.
    Mirrors the complete merge logic of merge_derivatives_sentiment_features for use in the parallel fetch pipeline."""
    if df.empty:
        df["open_interest"] = 0.0
        df["funding_rate"] = 0.0
        df["fear_greed"] = 50.0
        df["btc_close"] = 0.0
        df["btc_volume"] = 0.0
        df["btc_rsi"] = 50.0
        return df

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")

    if df_oi is not None and not df_oi.empty:
        df_oi = df_oi.copy()
        df_oi["timestamp"] = pd.to_numeric(df_oi["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_oi = df_oi.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_oi, on="timestamp", direction="backward")
    else:
        df["open_interest"] = 0.0

    if df_funding is not None and not df_funding.empty:
        df_funding = df_funding.copy()
        df_funding["timestamp"] = pd.to_numeric(df_funding["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_funding = df_funding.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_funding, on="timestamp", direction="backward")
    else:
        df["funding_rate"] = 0.0

    if df_fng is not None and not df_fng.empty:
        df_fng = df_fng.copy()
        df_fng["timestamp"] = pd.to_numeric(df_fng["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_fng, on="timestamp", direction="backward")
    else:
        df["fear_greed"] = 50.0

    # Calculate OI momentum and sanitize inf values (mirrors merge_derivatives_sentiment_features)
    df["oi_change_1h"] = df["open_interest"].pct_change(periods=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["oi_change_4h"] = df["open_interest"].pct_change(periods=4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["oi_change_pct"] = df["oi_change_1h"]
    df["oi_price_divergence"] = np.sign(df["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)) * np.sign(df["oi_change_pct"])
    df["funding_pctile"] = df["funding_rate"].rolling(720, min_periods=24).rank(pct=True).fillna(0.5)
    df["funding_roc"] = df["funding_rate"].diff().fillna(0.0)

    # Merge BTCUSDT Correlation Features
    if symbol != "BTCUSDT":
        if df_btc is not None and not df_btc.empty:
            df_btc_sub = df_btc.copy()
            df_btc_sub["timestamp"] = pd.to_numeric(df_btc_sub["timestamp"], errors="coerce").fillna(0).astype("int64")
            df_btc_sub = df_btc_sub.sort_values("timestamp").reset_index(drop=True)
            delta = df_btc_sub["close"].diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            df_btc_sub["btc_rsi"] = 100.0 - (100.0 / (1.0 + rs))
            df_btc_sub["btc_rsi"] = df_btc_sub["btc_rsi"].fillna(50.0)
            
            df_btc_features = df_btc_sub[["timestamp", "close", "volume", "btc_rsi"]].rename(
                columns={"close": "btc_close", "volume": "btc_volume"}
            )
            df = pd.merge_asof(df, df_btc_features, on="timestamp", direction="backward")
        else:
            df["btc_close"] = df["close"]
            df["btc_volume"] = df["volume"]
            df["btc_rsi"] = 50.0
    else:
        df["btc_close"] = df["close"]
        df["btc_volume"] = df["volume"]
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df["btc_rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["btc_rsi"] = df["btc_rsi"].fillna(50.0)

    # Clean up NaNs using forward-fill ONLY to eliminate look-ahead leakage
    df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
    df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    df["fear_greed"] = df["fear_greed"].ffill().fillna(50.0)
    df["oi_change_1h"] = df["oi_change_1h"].ffill().fillna(0.0)
    df["oi_change_4h"] = df["oi_change_4h"].ffill().fillna(0.0)
    df["oi_change_pct"] = df["oi_change_pct"].ffill().fillna(0.0)
    df["oi_price_divergence"] = df["oi_price_divergence"].ffill().fillna(0.0)
    df["funding_pctile"] = df["funding_pctile"].ffill().fillna(0.5)
    df["funding_roc"] = df["funding_roc"].ffill().fillna(0.0)
    df["btc_close"] = df["btc_close"].ffill().fillna(df["close"])
    df["btc_volume"] = df["btc_volume"].ffill().fillna(df["volume"])
    df["btc_rsi"] = df["btc_rsi"].ffill().fillna(50.0)
    
    return df


def merge_derivatives_sentiment_features(df, symbol, interval):
    if df.empty:
        df["open_interest"] = 0.0
        df["funding_rate"] = 0.0
        df["fear_greed"] = 50.0
        return df

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
    min_ts = int(df["timestamp"].min())
    max_ts = int(df["timestamp"].max())

    # Hermetic Fetch with Fallback
    try:
        df_oi = get_bybit_oi_history(symbol=symbol, interval=interval, start_ts_ms=min_ts, end_ts_ms=max_ts)
    except Exception as ex_data:
        log_event("WARNING", f"data notice: {ex_data}")
        df_oi = pd.DataFrame(columns=["timestamp", "open_interest"])

    try:
        df_funding = get_bybit_funding_history(symbol=symbol, start_ts_ms=min_ts, end_ts_ms=max_ts)
    except Exception as ex_data:
        log_event("WARNING", f"data notice: {ex_data}")
        df_funding = pd.DataFrame(columns=["timestamp", "funding_rate"])

    try:
        df_fng = get_fear_and_greed_history()
    except Exception as ex_data:
        log_event("WARNING", f"data notice: {ex_data}")
        df_fng = pd.DataFrame(columns=["timestamp", "fear_greed"])

    # Merge open interest
    if not df_oi.empty:
        df_oi["timestamp"] = pd.to_numeric(df_oi["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_oi = df_oi.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_oi, on="timestamp", direction="backward")
    else:
        df["open_interest"] = 0.0

    # Merge funding rate
    if not df_funding.empty:
        df_funding["timestamp"] = pd.to_numeric(df_funding["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_funding = df_funding.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_funding, on="timestamp", direction="backward")
    else:
        df["funding_rate"] = 0.0

    # Merge fear and greed
    if not df_fng.empty:
        df_fng["timestamp"] = pd.to_numeric(df_fng["timestamp"], errors="coerce").fillna(0).astype("int64")
        df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_fng, on="timestamp", direction="backward")
    else:
        df["fear_greed"] = 50.0
        
    # Calculate OI momentum and sanitize inf values
    df["oi_change_1h"] = df["open_interest"].pct_change(periods=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["oi_change_4h"] = df["open_interest"].pct_change(periods=4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["oi_change_pct"] = df["oi_change_1h"]
    df["oi_price_divergence"] = np.sign(df["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)) * np.sign(df["oi_change_pct"])
    df["funding_pctile"] = df["funding_rate"].rolling(720, min_periods=24).rank(pct=True).fillna(0.5)
    df["funding_roc"] = df["funding_rate"].diff().fillna(0.0)
    
    # Merge BTCUSDT Correlation Features
    if symbol != "BTCUSDT":
        try:
            df_btc = get_history(symbol="BTCUSDT", interval=str(interval), limit=len(df) + 100)
            if not df_btc.empty:
                df_btc["timestamp"] = pd.to_numeric(df_btc["timestamp"], errors="coerce").fillna(0).astype("int64")
                df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
                delta = df_btc["close"].diff()
                gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-9)
                df_btc["btc_rsi"] = 100.0 - (100.0 / (1.0 + rs))
                df_btc["btc_rsi"] = df_btc["btc_rsi"].fillna(50.0)
                
                df_btc_features = df_btc[["timestamp", "close", "volume", "btc_rsi"]].rename(
                    columns={"close": "btc_close", "volume": "btc_volume"}
                )
                df = pd.merge_asof(df, df_btc_features, on="timestamp", direction="backward")
            else:
                df["btc_close"] = df["close"]
                df["btc_volume"] = df["volume"]
                df["btc_rsi"] = 50.0
        except Exception as e:
            print(f"[BTC Correlation] Error merging BTC features: {e}")
            df["btc_close"] = df["close"]
            df["btc_volume"] = df["volume"]
            df["btc_rsi"] = 50.0
    else:
        df["btc_close"] = df["close"]
        df["btc_volume"] = df["volume"]
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df["btc_rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["btc_rsi"] = df["btc_rsi"].fillna(50.0)

    # Clean up NaNs using forward-fill ONLY to eliminate look-ahead leakage
    df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
    df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    df["fear_greed"] = df["fear_greed"].ffill().fillna(50.0)
    df["oi_change_1h"] = df["oi_change_1h"].ffill().fillna(0.0)
    df["oi_change_4h"] = df["oi_change_4h"].ffill().fillna(0.0)
    df["oi_change_pct"] = df["oi_change_pct"].ffill().fillna(0.0)
    df["oi_price_divergence"] = df["oi_price_divergence"].ffill().fillna(0.0)
    df["funding_pctile"] = df["funding_pctile"].ffill().fillna(0.5)
    df["funding_roc"] = df["funding_roc"].ffill().fillna(0.0)
    df["btc_close"] = df["btc_close"].ffill().fillna(df["close"])
    df["btc_volume"] = df["btc_volume"].ffill().fillna(df["volume"])
    df["btc_rsi"] = df["btc_rsi"].ffill().fillna(50.0)
    
    return df

def classify_market_regime(df_history: pd.DataFrame, interval: Optional[str] = None) -> str:
    """
    4-Regime Classification:
    1. High Vol, Trending (Breakout)
    2. Low Vol, Trending (Trend Following)
    3. Low Vol, Ranging (Mean Reversion)
    4. High Vol, Ranging (Chop - Avoid Trading)
    """
    if "ATR_norm" not in df_history.columns or "ADX" not in df_history.columns or len(df_history) < 20:
        return "Low Vol, Ranging"

    df_clean = df_history[["ATR_norm", "ADX"]].dropna()
    if len(df_clean) == 0:
        return "Low Vol, Ranging"

    from config import REGIME_ADX_ENTER_BY_INTERVAL, REGIME_ADX_EXIT_BY_INTERVAL, STRONG_TREND_ADX_ENTER, STRONG_TREND_ADX_EXIT
    _enter_thr = REGIME_ADX_ENTER_BY_INTERVAL.get(str(interval), STRONG_TREND_ADX_ENTER)
    _exit_thr = REGIME_ADX_EXIT_BY_INTERVAL.get(str(interval), STRONG_TREND_ADX_EXIT)

    adx_series = df_clean["ADX"].values
    is_trending_state = False
    for a in adx_series:
        if not is_trending_state and a >= _enter_thr:
            is_trending_state = True
        elif is_trending_state and a <= _exit_thr:
            is_trending_state = False

    latest_atr = df_clean["ATR_norm"].iloc[-1]
    atr_median = float(df_clean["ATR_norm"].iloc[:-1].median()) if len(df_clean) > 1 else float(latest_atr)

    is_high_vol = latest_atr > atr_median
    is_trending = is_trending_state

    if is_high_vol and is_trending:
        return "High Vol, Trending"
    elif not is_high_vol and is_trending:
        return "Low Vol, Trending"
    elif not is_high_vol and not is_trending:
        return "Low Vol, Ranging"
    else:
        return "High Vol, Ranging"


def get_orderbook_imbalance(symbol: str = "BTCUSDT") -> dict:
    """
    Fetches real-time L2 orderbook snapshot from Bybit (with Binance depth fallback),
    returning weighted order imbalance, normalized bid-ask spread, and total depth in USD.
    """
    if not symbol or not isinstance(symbol, str):
        symbol = "BTCUSDT"

    try:
        url = f"{BYBIT_BASE_URL}/v5/market/orderbook"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol.upper(), "limit": 25}, headers=headers, timeout=10)
        res = None
        if response.status_code == 200:
            res = response.json()
        else:
            # Fallback to Binance depth
            binance_url = "https://api.binance.com/api/v3/depth"
            resp = requests.get(binance_url, params={"symbol": symbol.upper(), "limit": 25}, headers=headers, timeout=10)
            if resp.status_code == 200:
                binance_data = resp.json()
                res = {
                    "result": {
                        "b": binance_data.get("bids", []),
                        "a": binance_data.get("asks", [])
                    }
                }

        if res and "result" in res and "b" in res["result"] and "a" in res["result"]:
            bids = res["result"]["b"]
            asks = res["result"]["a"]
            if not bids or not asks:
                return {"imbalance": 0.0, "spread": 0.0, "total_depth": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0

            bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
            ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
            total_depth_usd = bid_depth_usd + ask_depth_usd

            weighted_bid_size = sum(float(b[1]) / (abs(float(b[0]) - mid_price) + 1e-8) for b in bids)
            weighted_ask_size = sum(float(a[1]) / (abs(float(a[0]) - mid_price) + 1e-8) for a in asks)
            weighted_imbalance = (weighted_bid_size - weighted_ask_size) / (weighted_bid_size + weighted_ask_size + 1e-8)

            return {
                "imbalance": float(weighted_imbalance),
                "spread": float(spread),
                "total_depth": float(total_depth_usd),
                "bid_depth": float(bid_depth_usd),
                "ask_depth": float(ask_depth_usd)
            }
    except Exception as e:
        print(f"[Orderbook] Error fetching depth metrics for {symbol}: {e}")

    return {"imbalance": 0.0, "spread": 0.0, "total_depth": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}

