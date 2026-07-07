import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kline_cache")

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

def bybit_public_get(url, params=None, headers=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, proxies=get_bybit_proxies(), timeout=10)
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
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")
    
    target_count = limit * pages
    
    # 1. Load cache if it exists
    df_cache = None
    if os.path.exists(cache_file):
        try:
            df_cache = pd.read_csv(cache_file)
            df_cache["timestamp"] = df_cache["timestamp"].astype(float)
            df_cache = df_cache.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
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
                        if ts <= cache_max_ts:
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
            elif str(interval) == "15":
                binance_interval = "15m"
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

            kraken_pair = "XBTUSDT"
            symbol_upper = symbol.upper()
            if symbol_upper == "BTCUSDT":
                kraken_pair = "XBTUSDT"
            elif symbol_upper == "ETHUSDT":
                kraken_pair = "ETHUSDT"
            elif symbol_upper == "SOLUSDT":
                kraken_pair = "SOLUSDT"
            else:
                kraken_pair = symbol_upper

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
                            float(item[0]) * 1000, # timestamp in ms
                            float(item[1]),        # open
                            float(item[2]),        # high
                            float(item[3]),        # low
                            float(item[4]),        # close
                            float(item[6]),        # volume
                            float(item[6]) * float(item[4]) # turnover
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
            df_combined.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"[Cache Write Error] {e}")

    return df_history.iloc[-target_count:].reset_index(drop=True)

def get_bybit_oi_history(symbol="BTCUSDT", interval="15", start_ts_ms=None, end_ts_ms=None):
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
    for page in range(100):
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
                oi_data.extend(batch)
                
                oldest_ts = int(batch[-1]["timestamp"])
                if start_ts_ms and oldest_ts < int(start_ts_ms):
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
            
    if not oi_data:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
        
    df_oi = pd.DataFrame(oi_data)
    df_oi["timestamp"] = df_oi["timestamp"].astype(float)
    df_oi["open_interest"] = df_oi["openInterest"].astype(float)
    df_oi = df_oi[["timestamp", "open_interest"]].sort_values("timestamp").reset_index(drop=True)
    return df_oi

def get_bybit_funding_history(symbol="BTCUSDT", start_ts_ms=None, end_ts_ms=None):
    url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
    funding_data = []
    current_end = int(end_ts_ms) if end_ts_ms else int(time.time() * 1000)
    for page in range(50):
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
                funding_data.extend(batch)
                
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
            
    if not funding_data:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
        
    df_funding = pd.DataFrame(funding_data)
    df_funding["timestamp"] = df_funding["fundingRateTimestamp"].astype(float)
    df_funding["funding_rate"] = df_funding["fundingRate"].astype(float)
    df_funding = df_funding[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    return df_funding

fng_cache = None
fng_cache_time = 0.0

def get_fear_and_greed_history():
    global fng_cache, fng_cache_time
    # Cache for 4 hours (14400 seconds)
    if fng_cache is not None and (time.time() - fng_cache_time) < 14400:
        return fng_cache
        
    url = "https://api.alternative.me/fng/?limit=0&format=json"
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
                df_fng = pd.DataFrame(fng_data)
                df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
                fng_cache = df_fng
                fng_cache_time = time.time()
                return df_fng
    except Exception as e:
        print(f"Error fetching Fear & Greed history: {e}")
        if fng_cache is not None:
            return fng_cache
    return pd.DataFrame(columns=["timestamp", "fear_greed"])

def merge_derivatives_sentiment_features(df, symbol, interval):
    if df.empty:
        df["open_interest"] = 0.0
        df["funding_rate"] = 0.0
        df["fear_greed"] = 50.0
        return df

    min_ts = int(df["timestamp"].min())
    max_ts = int(df["timestamp"].max())
    
    # Fetch
    df_oi = get_bybit_oi_history(symbol=symbol, interval=interval, start_ts_ms=min_ts, end_ts_ms=max_ts)
    df_funding = get_bybit_funding_history(symbol=symbol, start_ts_ms=min_ts, end_ts_ms=max_ts)
    df_fng = get_fear_and_greed_history()
    
    # Merge open interest
    if not df_oi.empty:
        df_oi = df_oi.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_oi, on="timestamp", direction="backward")
    else:
        df["open_interest"] = 0.0
        
    # Merge funding rate
    if not df_funding.empty:
        df_funding = df_funding.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_funding, on="timestamp", direction="backward")
    else:
        df["funding_rate"] = 0.0
        
    # Merge fear and greed
    if not df_fng.empty:
        df_fng = df_fng.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_fng, on="timestamp", direction="backward")
    else:
        df["fear_greed"] = 50.0
        
    # Clean up NaNs
    df["open_interest"] = df["open_interest"].ffill().bfill().fillna(0.0)
    df["funding_rate"] = df["funding_rate"].ffill().bfill().fillna(0.0)
    df["fear_greed"] = df["fear_greed"].ffill().bfill().fillna(50.0)
    
    return df